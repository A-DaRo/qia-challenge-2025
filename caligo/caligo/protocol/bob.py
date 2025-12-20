"""Bob role program for Caligo Phase E."""

from __future__ import annotations

from typing import Any, Dict, Generator, Tuple

import numpy as np

from caligo.amplification import OTOutputFormatter
from caligo.connection.envelope import MessageType
from caligo.protocol.base import CaligoProgram, ProtocolParameters
from caligo.reconciliation import constants as recon_constants
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import ReconciliationOrchestrator, ReconciliationOrchestratorConfig
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.ldpc_decoder import BeliefPropagationDecoder
from caligo.reconciliation import constants as recon_constants_pkg
from caligo.types.exceptions import SecurityError, SynchronizationError
from caligo.sifting.commitment import SHA256Commitment
from caligo.types.keys import BobObliviousKey
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class BobProgram(CaligoProgram):
    """Bob's (receiver) program."""

    PEER = "Alice"
    ROLE = "bob"

    def __init__(self, params: ProtocolParameters, choice_bit: int = 0) -> None:
        super().__init__(params=params)
        if choice_bit not in (0, 1):
            raise ValueError("choice_bit must be 0 or 1")
        self._choice_bit = int(choice_bit)
        self._commitment = SHA256Commitment()

    @property
    def choice_bit(self) -> int:
        return self._choice_bit

    def _run_protocol(self, context) -> Generator[Any, None, Dict[str, Any]]:
        assert self._ordered_socket is not None

        bob_outcomes, bob_bases, nonce, commitment = yield from self._phase1_quantum_and_commit(
            context
        )

        # Send commitment (ordered).
        yield from self._ordered_socket.send(
            MessageType.DETECTION_COMMITMENT,
            {"commitment": commitment.hex()},
        )

        # Receive Alice bases.
        bases_msg = yield from self._ordered_socket.recv(MessageType.BASIS_REVEAL)
        _alice_bases = np.frombuffer(
            bytes.fromhex(str(bases_msg["bases"])), dtype=np.uint8
        )

        # Open commitment by sending nonce and data.
        yield from self._ordered_socket.send(
            MessageType.COMMITMENT_OPENING,
            {
                "nonce": nonce.hex(),
                "outcomes": bob_outcomes.astype(np.uint8).tobytes().hex(),
                "bases": bob_bases.astype(np.uint8).tobytes().hex(),
            },
        )

        # Optional Phase II test exchange.
        # For blind reconciliation, Alice will skip INDEX_LISTS/TEST_OUTCOMES.
        test_indices: np.ndarray
        if self.params.reconciliation.requires_qber_estimation:
            idx_msg = yield from self._ordered_socket.recv(MessageType.INDEX_LISTS)
            test_indices = np.array(idx_msg["test_indices"], dtype=np.int64)
            test_bits = bob_outcomes[test_indices].astype(np.uint8)

            yield from self._ordered_socket.send(
                MessageType.TEST_OUTCOMES,
                {"test_bits": test_bits.tobytes().hex()},
            )

        # Phase III: receive per-block syndrome(s), decode locally.
        # Determine matching indices from bases only.
        matching_mask = _alice_bases == bob_bases
        matching_indices = np.where(matching_mask)[0].astype(np.int64)

        # Partition by basis value at matching positions.
        matching_bases = _alice_bases[matching_indices]
        i0_mask = matching_bases == 0
        i1_mask = matching_bases == 1
        i0_all = matching_indices[i0_mask]
        i1_all = matching_indices[i1_mask]

        # For baseline, Alice removes test indices; mimic that removal locally.
        if self.params.reconciliation.requires_qber_estimation:
            # Remove test indices from matching_indices.
            test_set = set(test_indices.tolist())
            key_indices = np.array([i for i in matching_indices.tolist() if i not in test_set], dtype=np.int64)
        else:
            key_indices = matching_indices

        # Build key bitarray from Bob outcomes at key indices.
        bob_key_np = bob_outcomes[key_indices].astype(np.uint8)

        # Partitions (I0/I1) restricted to key indices.
        i0_indices = np.intersect1d(i0_all, key_indices)
        i1_indices = np.intersect1d(i1_all, key_indices)

        matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
        config = ReconciliationOrchestratorConfig(
            frame_size=int(self.params.reconciliation.frame_size),
            max_iterations=int(self.params.reconciliation.max_iterations),
            max_retries=int(self.params.reconciliation.max_blind_rounds),
        )
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            config=config,
            safety_cap=10**12,
        )

        hash_verifier = PolynomialHashVerifier(hash_bits=config.hash_bits)

        reconciled_parts: list[np.ndarray] = []
        verified_positions: list[int] = []  # Track which original bit positions are verified
        for block_id, start in enumerate(range(0, len(bob_key_np), config.frame_size)):
            msg0 = yield from self._ordered_socket.recv(MessageType.SYNDROME)
            kind = str(msg0.get("kind"))
            if int(msg0.get("block_id", -1)) != int(block_id):
                raise SecurityError("Reconciliation block_id mismatch")

            if kind == "baseline":
                # Effectively-public on-wire fields Bob consumes for baseline:
                # - payload_length, frame_size, rate
                # - puncture_pattern (hex), n_punctured
                # - syndrome (bytes), qber_channel/qber_prior
                # - hash_int, hash_seed
                payload_len = int(msg0["payload_length"])
                frame_size = int(msg0.get("frame_size", config.frame_size))
                syndrome = np.frombuffer(bytes.fromhex(str(msg0["syndrome"])), dtype=np.uint8)
                rate = float(msg0["rate"])
                puncture_pattern = np.frombuffer(bytes.fromhex(str(msg0["puncture_pattern"])), dtype=np.uint8)
                n_punctured = int(msg0["n_punctured"])
                # Use qber_channel for decoder LLR (accurate channel model)
                # Falls back to qber_prior for backward compatibility
                qber_channel = float(msg0.get(
                    "qber_channel",
                    msg0.get("qber_prior", self.params.nsm_params.qber_conditional)
                ))
                expected_hash = int(msg0["hash_int"])
                hash_seed = int(msg0["hash_seed"])

                # Cheap synchronization guards: these should never fail for
                # honest peers, but catch accidental coupling bugs early.
                if frame_size != int(config.frame_size):
                    raise SynchronizationError(
                        "Reconciliation frame_size mismatch: "
                        f"msg={frame_size} != expected={int(config.frame_size)}"
                    )
                if payload_len < 0 or payload_len > frame_size:
                    raise SynchronizationError("Reconciliation payload_length out of range")

                bob_payload = bob_key_np[start:start + payload_len]
                end = start + payload_len
                
                # Pad short blocks to frame_size (for last block) - matches Alice's padding
                if len(bob_payload) < frame_size:
                    padding_needed = frame_size - len(bob_payload)
                    bob_payload_padded = np.concatenate([bob_payload, np.zeros(padding_needed, dtype=np.uint8)])
                else:
                    bob_payload_padded = bob_payload
                
                # Use mother code (rate 0.5) with puncture pattern
                compiled_H = matrix_manager.get_compiled(0.5)
                if int(syndrome.shape[0]) != int(compiled_H.m):
                    raise SynchronizationError(
                        "Reconciliation syndrome length mismatch: "
                        f"len={int(syndrome.shape[0])} != expected={int(compiled_H.m)}"
                    )
                    
                # Build LLR using puncture pattern (replaces n_shortened/prng_seed)
                from caligo.reconciliation.ldpc_decoder import build_channel_llr
                llr = build_channel_llr(
                    bob_payload_padded,
                    qber_channel,
                    puncture_pattern.astype(bool)
                )
                
                # Decode with retry
                decoder = BeliefPropagationDecoder(max_iterations=config.max_iterations)
                decode = decoder.decode(llr, syndrome, H=compiled_H)

                corrected_payload = decode.corrected_bits[:payload_len]  # Extract only original payload
                verified = hash_verifier.verify(corrected_payload, expected_hash, seed=hash_seed)
                yield from self._ordered_socket.send(
                    MessageType.SYNDROME_RESPONSE,
                    {
                        "kind": "baseline",
                        "block_id": int(block_id),
                        "verified": bool(verified),
                        "converged": bool(decode.converged),
                        "syndrome_errors": int(decode.syndrome_errors),
                        "corrected_payload": corrected_payload.astype(np.uint8).tobytes().hex(),
                    },
                )
                if not verified:
                    # Log warning but continue with other blocks
                    from caligo.utils.logging import get_logger
                    _logger = get_logger(__name__)
                    _logger.warning(
                        f"Block {block_id} failed verification - excluding from final key"
                    )
                    continue  # Skip this block, try remaining blocks

                reconciled_parts.append(corrected_payload.astype(np.uint8))
                # Track the original bit positions that are now verified
                verified_positions.extend(range(start, end))

            elif kind == "blind":
                # Blind reconciliation: same as baseline but Bob doesn't know QBER a priori
                payload_len = int(msg0["payload_length"])
                frame_size = int(msg0.get("frame_size", config.frame_size))
                syndrome = np.frombuffer(bytes.fromhex(str(msg0["syndrome"])), dtype=np.uint8)
                rate = float(msg0["rate"])
                puncture_pattern = np.frombuffer(bytes.fromhex(str(msg0["puncture_pattern"])), dtype=np.uint8)
                n_punctured = int(msg0["n_punctured"])
                # Use qber_channel for decoder LLR
                qber_channel = float(msg0.get(
                    "qber_channel",
                    msg0.get("qber_prior", self.params.nsm_params.qber_conditional)
                ))
                expected_hash = int(msg0["hash_int"])
                hash_seed = int(msg0["hash_seed"])

                bob_payload = bob_key_np[start:start + payload_len]
                end = start + payload_len
                
                # Pad short blocks to frame_size (for last block) - matches Alice's padding
                if len(bob_payload) < frame_size:
                    padding_needed = frame_size - len(bob_payload)
                    bob_payload_padded = np.concatenate([bob_payload, np.zeros(padding_needed, dtype=np.uint8)])
                else:
                    bob_payload_padded = bob_payload
                
                # Use mother code (rate 0.5) with puncture pattern
                compiled_H = matrix_manager.get_compiled(0.5)
                
                # Build LLR using puncture pattern
                from caligo.reconciliation.ldpc_decoder import build_channel_llr
                llr = build_channel_llr(
                    bob_payload_padded,
                    qber_channel,
                    puncture_pattern.astype(bool)
                )
                
                # Decode
                decoder = BeliefPropagationDecoder(max_iterations=config.max_iterations)
                res = decoder.decode(llr, syndrome.astype(np.uint8, copy=False), H=compiled_H)
                corrected_payload = res.corrected_bits[:payload_len]  # Extract only original payload
                verified = hash_verifier.verify(corrected_payload, expected_hash, seed=hash_seed)

                yield from self._ordered_socket.send(
                    MessageType.SYNDROME_RESPONSE,
                    {
                        "kind": "blind",
                        "block_id": int(block_id),
                        "verified": bool(verified),
                        "converged": bool(res.converged),
                        "syndrome_errors": int(res.syndrome_errors),
                        "corrected_payload": corrected_payload.astype(np.uint8).tobytes().hex(),
                    },
                )
                if not verified:
                    # Log warning but continue with other blocks
                    from caligo.utils.logging import get_logger
                    _logger = get_logger(__name__)
                    _logger.warning(
                        f"Block {block_id} failed verification - excluding from final key"
                    )
                    continue  # Skip this block, try remaining blocks

                reconciled_parts.append(corrected_payload.astype(np.uint8))
                # Track the original bit positions that are now verified
                verified_positions.extend(range(start, end))

            else:
                raise SecurityError("Unsupported reconciliation kind")

        # Check if we have enough reconciled bits
        if len(reconciled_parts) == 0:
            raise SecurityError("All reconciliation blocks failed verification")
        
        reconciled_np = np.concatenate(reconciled_parts).astype(np.uint8) if reconciled_parts else np.array([], dtype=np.uint8)

        # Filter partition indices to only include verified positions.
        # The reconciled_np corresponds to verified_positions in the original key.
        verified_set = set(verified_positions)
        verified_key_indices = np.array([i for i in key_indices.tolist() if i in verified_set], dtype=np.int64)
        verified_i0 = np.array([i for i in i0_indices.tolist() if i in verified_set], dtype=np.int64)
        verified_i1 = np.array([i for i in i1_indices.tolist() if i in verified_set], dtype=np.int64)

        # Receive Toeplitz seeds.
        seed_msg = yield from self._ordered_socket.recv(MessageType.TOEPLITZ_SEED)
        key_length = int(seed_msg["key_length"])
        seed_0 = bytes.fromhex(str(seed_msg["seed_0"]))
        seed_1 = bytes.fromhex(str(seed_msg["seed_1"]))

        # Partition reconciled key using the same sifting mapping.
        from caligo.sifting.sifter import Sifter

        sifter = Sifter()
        reconciled_ba = bitarray_from_numpy(reconciled_np)
        key_i0_ba, key_i1_ba = sifter.extract_partition_keys(
            sifted_bits=reconciled_ba,
            i0_indices=verified_i0,
            i1_indices=verified_i1,
            matching_indices=verified_key_indices,
        )

        formatter = OTOutputFormatter(
            key_length=key_length,
            seed_0=seed_0,
            seed_1=seed_1,
        )

        bob_out = formatter.compute_bob_key(
            bob_key_i0=bitarray_to_numpy(key_i0_ba),
            bob_key_i1=bitarray_to_numpy(key_i1_ba),
            choice_bit=self._choice_bit,
        )

        bob_key = BobObliviousKey(
            sc=bitarray_from_numpy(bob_out.key_c),
            choice_bit=int(bob_out.choice_bit),
            key_length=int(bob_out.key_length),
            security_parameter=1e-10,
        )

        return {
            "role": self.ROLE,
            "aborted": False,
            "bob_key": bob_key,
            "choice_bit": int(self._choice_bit),
        }

    def _phase1_quantum_and_commit(
        self, context
    ) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray, bytes, bytes]]:
        """Receive and measure EPR pairs (Bob side), then commit."""

        if self.params.precomputed_epr is not None:
            n = int(self.params.num_pairs)
            outcomes = np.asarray(self.params.precomputed_epr.bob_outcomes, dtype=np.uint8)
            bases = np.asarray(self.params.precomputed_epr.bob_bases, dtype=np.uint8)
            if len(outcomes) != n or len(bases) != n:
                raise ValueError(
                    "precomputed_epr length mismatch: "
                    f"expected n={n}, got outcomes={len(outcomes)} bases={len(bases)}"
                )

            data_bytes = np.concatenate([outcomes, bases]).astype(np.uint8).tobytes()
            commit_res = self._commitment.commit(data_bytes)

            return (
                outcomes.astype(np.uint8),
                bases.astype(np.uint8),
                commit_res.nonce,
                commit_res.commitment,
            )

        from caligo.quantum import BasisSelector, MeasurementExecutor

        epr_socket = context.epr_sockets[self.PEER]
        basis_selector = BasisSelector()
        meas = MeasurementExecutor()

        n = int(self.params.num_pairs)
        bases = basis_selector.select_batch(n)
        outcomes = np.zeros(n, dtype=np.uint8)

        # Use the same pattern as SquidASM's QKD example:
        # receive and measure one EPR pair per round to preserve ordering.
        for round_id in range(n):
            q = epr_socket.recv_keep(1)[0]
            outcomes[round_id] = yield from meas.measure_qubit(
                qubit=q,
                basis=int(bases[round_id]),
                round_id=round_id,
                context=context,
            )

        data_bytes = np.concatenate([outcomes, bases]).astype(np.uint8).tobytes()
        commit_res = self._commitment.commit(data_bytes)

        return (
            outcomes.astype(np.uint8),
            bases.astype(np.uint8),
            commit_res.nonce,
            commit_res.commitment,
        )

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
from caligo.types.exceptions import SecurityError
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
                payload_len = int(msg0["payload_length"])
                syndrome = np.frombuffer(bytes.fromhex(str(msg0["syndrome"])), dtype=np.uint8)
                rate = float(msg0["rate"])
                n_shortened = int(msg0["n_shortened"])
                prng_seed = int(msg0["prng_seed"])
                # Use qber_channel for decoder LLR (accurate channel model)
                # Falls back to qber_prior for backward compatibility
                qber_channel = float(msg0.get(
                    "qber_channel",
                    msg0.get("qber_prior", self.params.nsm_params.qber_conditional)
                ))
                expected_hash = int(msg0["hash_int"])
                hash_seed = int(msg0["hash_seed"])

                bob_payload = bob_key_np[start:start + payload_len]
                end = start + payload_len
                compiled_H = matrix_manager.get_compiled(rate)
                decode = orchestrator._decode_with_retry(
                    bob_key=bob_payload,
                    syndrome=syndrome,
                    H=compiled_H,
                    n_shortened=n_shortened,
                    prng_seed=prng_seed,
                    qber_estimate=qber_channel,
                )

                corrected_payload = decode.corrected_bits[:payload_len]
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
                max_rounds = int(self.params.reconciliation.max_blind_rounds)

                payload_len = int(msg0["payload_length"])
                syndrome = np.frombuffer(bytes.fromhex(str(msg0["syndrome"])), dtype=np.uint8)
                rate = float(msg0["rate"])
                n_padding = int(msg0["n_padding"])
                # Use qber_channel for decoder LLR (falls back to qber_prior for compat)
                qber_channel = float(msg0.get(
                    "qber_channel",
                    msg0.get("qber_prior", self.params.nsm_params.qber_conditional)
                ))
                expected_hash = int(msg0["hash_int"])
                hash_seed = int(msg0["hash_seed"])

                # Known padding values filled progressively; unknown => None.
                known_padding: list[int | None] = [None] * n_padding

                initial_reveal = np.frombuffer(bytes.fromhex(str(msg0.get("reveal", ""))), dtype=np.uint8)
                for i in range(min(len(initial_reveal), n_padding)):
                    known_padding[i] = int(initial_reveal[i])

                bob_payload = bob_key_np[start:start + payload_len]
                end = start + payload_len
                compiled_H = matrix_manager.get_compiled(rate)
                decoder = BeliefPropagationDecoder(max_iterations=config.max_iterations)

                best_corrected: np.ndarray | None = None
                best_verified = False
                best_converged = False
                best_syndrome_errors = 0

                def _attempt_decode() -> None:
                    nonlocal best_corrected, best_verified, best_converged, best_syndrome_errors

                    q = float(np.clip(qber_channel, 1e-6, 0.5 - 1e-6))
                    channel_llr = float(np.log((1.0 - q) / q))

                    llr = np.zeros(payload_len + n_padding, dtype=np.float64)
                    llr[:payload_len] = channel_llr * (1 - 2 * bob_payload.astype(np.float64))

                    for j, val in enumerate(known_padding):
                        if val is None:
                            continue
                        llr[payload_len + j] = recon_constants_pkg.LDPC_LLR_SHORTENED * (1 - 2 * float(val))

                    res = decoder.decode(llr, syndrome.astype(np.uint8, copy=False), H=compiled_H)
                    corrected_payload = res.corrected_bits[:payload_len]
                    verified = hash_verifier.verify(corrected_payload, expected_hash, seed=hash_seed)
                    if verified and not best_verified:
                        best_corrected = corrected_payload.astype(np.uint8)
                        best_verified = True
                        best_converged = bool(res.converged)
                        best_syndrome_errors = int(res.syndrome_errors)

                # Attempt after round 0.
                _attempt_decode()

                # Consume remaining rounds.
                revealed_count = len(initial_reveal)
                for _round in range(1, max_rounds + 1):
                    msg_r = yield from self._ordered_socket.recv(MessageType.SYNDROME)
                    if str(msg_r.get("kind")) != "blind":
                        raise SecurityError("Unexpected reconciliation kind")
                    if int(msg_r.get("block_id", -1)) != int(block_id):
                        raise SecurityError("Reconciliation block_id mismatch")

                    chunk = np.frombuffer(bytes.fromhex(str(msg_r.get("reveal", ""))), dtype=np.uint8)
                    for i in range(len(chunk)):
                        idx = revealed_count + i
                        if idx >= n_padding:
                            break
                        known_padding[idx] = int(chunk[i])
                    revealed_count = min(n_padding, revealed_count + len(chunk))

                    if not best_verified:
                        _attempt_decode()

                if best_corrected is None:
                    # Fall back to Bob's current payload if never verified.
                    best_corrected = bob_payload.copy().astype(np.uint8)

                yield from self._ordered_socket.send(
                    MessageType.SYNDROME_RESPONSE,
                    {
                        "kind": "blind",
                        "block_id": int(block_id),
                        "verified": bool(best_verified),
                        "converged": bool(best_converged),
                        "syndrome_errors": int(best_syndrome_errors),
                        "corrected_payload": best_corrected.astype(np.uint8).tobytes().hex(),
                    },
                )
                if not best_verified:
                    # Log warning but continue with other blocks
                    from caligo.utils.logging import get_logger
                    _logger = get_logger(__name__)
                    _logger.warning(
                        f"Block {block_id} failed verification - excluding from final key"
                    )
                    continue  # Skip this block, try remaining blocks

                reconciled_parts.append(best_corrected.astype(np.uint8))
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

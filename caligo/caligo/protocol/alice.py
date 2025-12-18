"""Alice role program for Caligo Phase E."""

from __future__ import annotations

from typing import Any, Dict, Generator, Tuple

import numpy as np
from bitarray import bitarray

from caligo.amplification import (
    NSMEntropyCalculator,
    OTOutputFormatter,
    SecureKeyLengthCalculator,
    ToeplitzHasher,
)
from caligo.connection.envelope import MessageType
from caligo.protocol.base import CaligoProgram, ProtocolParameters
from caligo.reconciliation import constants as recon_constants
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import ReconciliationOrchestrator, ReconciliationOrchestratorConfig
from caligo.sifting.commitment import SHA256Commitment
from caligo.sifting.qber import QBEREstimator
from caligo.sifting.sifter import Sifter
from caligo.types.exceptions import EntropyDepletedError, SecurityError
from caligo.types.keys import AliceObliviousKey
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class AliceProgram(CaligoProgram):
    """Alice's (sender) program."""

    PEER = "Bob"
    ROLE = "alice"

    def __init__(self, params: ProtocolParameters) -> None:
        super().__init__(params=params)
        self._commitment = SHA256Commitment()
        self._sifter = Sifter()
        self._qber_estimator = QBEREstimator()

    def _run_protocol(self, context) -> Generator[Any, None, Dict[str, Any]]:
        assert self._ordered_socket is not None

        # ------------------------------------------------------------------
        # Phase I: Quantum (implemented minimally; results are exchanged via
        # classical messages for now).
        # ------------------------------------------------------------------
        alice_outcomes, alice_bases = yield from self._phase1_quantum(context)

        # ------------------------------------------------------------------
        # Phase II: Sifting + QBER using ordered messaging with timing barrier.
        # ------------------------------------------------------------------
        sifting_payload, timing_ok = yield from self._phase2_sifting(
            alice_outcomes=alice_outcomes,
            alice_bases=alice_bases,
        )

        # ------------------------------------------------------------------
        # Phase III: Reconciliation (centralized at Alice for now).
        # ------------------------------------------------------------------
        reconciled_bits, total_syndrome_bits = self._phase3_reconcile(
            alice_bits=sifting_payload["alice_bits"],
            bob_bits=sifting_payload["bob_bits"],
            qber_adjusted=float(sifting_payload["qber_adjusted"]),
        )

        # Send reconciled material + indices to Bob for his amplification.
        yield from self._ordered_socket.send(
            MessageType.SYNDROME_RESPONSE,
            {
                "reconciled": bitarray_to_numpy(reconciled_bits).tobytes().hex(),
                "matching_indices": sifting_payload["matching_indices"],
                "i0_indices": sifting_payload["i0_indices"],
                "i1_indices": sifting_payload["i1_indices"],
            },
        )

        # ------------------------------------------------------------------
        # Phase IV: Amplification (Alice computes S0,S1 and sends seeds).
        # ------------------------------------------------------------------
        alice_key, key_length, entropy_consumed, entropy_rate, seeds = self._phase4_amplify(
            reconciled_bits=reconciled_bits,
            total_syndrome_bits=total_syndrome_bits,
            matching_indices=np.array(sifting_payload["matching_indices"], dtype=np.int64),
            i0_indices=np.array(sifting_payload["i0_indices"], dtype=np.int64),
            i1_indices=np.array(sifting_payload["i1_indices"], dtype=np.int64),
        )

        yield from self._ordered_socket.send(
            MessageType.TOEPLITZ_SEED,
            {
                "key_length": int(key_length),
                "seed_0": seeds[0].hex(),
                "seed_1": seeds[1].hex(),
            },
        )

        return {
            "role": self.ROLE,
            "aborted": False,
            "alice_key": alice_key,
            "qber": float(sifting_payload["qber_adjusted"]),
            "key_length": int(key_length),
            "entropy_consumed": float(entropy_consumed),
            "entropy_rate": float(entropy_rate),
            "timing_compliant": bool(timing_ok),
            "total_rounds": int(self.params.num_pairs),
        }

    def _phase1_quantum(
        self, context
    ) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray]]:
        """Generate and measure EPR pairs (Alice side)."""

        from caligo.quantum import BasisSelector, EPRGenerator, MeasurementExecutor

        epr_socket = context.epr_sockets[self.PEER]
        basis_selector = BasisSelector()
        meas = MeasurementExecutor()
        epr_gen = EPRGenerator()

        n = int(self.params.num_pairs)
        bases = basis_selector.select_batch(n)
        outcomes = np.zeros(n, dtype=np.uint8)

        # NetQASM's EPRSocket.create_keep is limited by the number of qubits
        # available on the stack (max_qubits). Generate in batches.
        batch_size = max(1, int(self.params.num_qubits))
        for start in range(0, n, batch_size):
            count = min(batch_size, n - start)
            batch = yield from epr_gen.generate_batch(
                epr_socket=epr_socket, num_pairs=count, context=context
            )
            for j, q in enumerate(batch.qubit_refs):
                round_id = start + j
                outcomes[round_id] = yield from meas.measure_qubit(
                    qubit=q,
                    basis=int(bases[round_id]),
                    round_id=round_id,
                    context=context,
                )

        # Mark quantum phase completion as the reference for Δt.
        self._timing_barrier.mark_quantum_complete()

        return outcomes.astype(np.uint8), bases.astype(np.uint8)

    def _phase2_sifting(
        self, alice_outcomes: np.ndarray, alice_bases: np.ndarray
    ) -> Generator[Any, None, Tuple[Dict[str, Any], bool]]:
        assert self._ordered_socket is not None

        # 1) Receive Bob's commitment.
        commit_msg = yield from self._ordered_socket.recv(MessageType.DETECTION_COMMITMENT)
        commitment = bytes.fromhex(str(commit_msg["commitment"]))

        # 2) Enforce Δt before revealing bases.
        yield from self._timing_barrier.wait_delta_t()
        timing_ok = self._timing_barrier.timing_compliant

        # 3) Reveal Alice bases.
        yield from self._ordered_socket.send(
            MessageType.BASIS_REVEAL,
            {"bases": alice_bases.tobytes().hex()},
        )

        # 4) Receive Bob opening.
        opening_msg = yield from self._ordered_socket.recv(MessageType.COMMITMENT_OPENING)
        nonce = bytes.fromhex(str(opening_msg["nonce"]))
        bob_outcomes = np.frombuffer(
            bytes.fromhex(str(opening_msg["outcomes"])), dtype=np.uint8
        )
        bob_bases = np.frombuffer(
            bytes.fromhex(str(opening_msg["bases"])), dtype=np.uint8
        )

        # Verify commitment binds Bob's measurement record.
        data_bytes = np.concatenate([bob_outcomes, bob_bases]).astype(np.uint8).tobytes()
        self._commitment.verify(commitment=commitment, nonce=nonce, data=data_bytes)

        # Compute sifted keys.
        alice_sift, bob_sift = self._sifter.compute_sifted_key(
            alice_bases=alice_bases,
            alice_outcomes=alice_outcomes,
            bob_bases=bob_bases,
            bob_outcomes=bob_outcomes,
        )

        # Select test subset (original indices).
        test_pos, key_pos = self._sifter.select_test_subset(
            matching_indices=alice_sift.matching_indices,
            test_fraction=0.1,
            min_test_size=1,
        )
        test_indices = alice_sift.matching_indices[test_pos]
        key_indices = alice_sift.matching_indices[key_pos]

        yield from self._ordered_socket.send(
            MessageType.INDEX_LISTS,
            {"test_indices": test_indices.astype(int).tolist()},
        )

        test_outcomes_msg = yield from self._ordered_socket.recv(MessageType.TEST_OUTCOMES)
        bob_test_bits = np.frombuffer(
            bytes.fromhex(str(test_outcomes_msg["test_bits"])), dtype=np.uint8
        )
        alice_test_bits = alice_outcomes[test_indices]

        qber = self._qber_estimator.estimate(
            alice_test_bits=alice_test_bits,
            bob_test_bits=bob_test_bits,
            key_size=len(key_indices),
        )

        # Filter sifted keys down to key_indices only.
        # Build an index map from original index -> sifted position.
        idx_map = {orig: i for i, orig in enumerate(alice_sift.matching_indices)}
        key_positions = [idx_map[idx] for idx in key_indices]
        alice_key_bits = bitarray([alice_sift.sifted_bits[p] for p in key_positions])
        bob_key_bits = bitarray([bob_sift.sifted_bits[p] for p in key_positions])

        # Also filter I0/I1 partitions down to key indices.
        i0_indices = np.intersect1d(alice_sift.i0_indices, key_indices)
        i1_indices = np.intersect1d(alice_sift.i1_indices, key_indices)

        return (
            {
                "alice_bits": alice_key_bits,
                "bob_bits": bob_key_bits,
                "matching_indices": key_indices.astype(int).tolist(),
                "i0_indices": i0_indices.astype(int).tolist(),
                "i1_indices": i1_indices.astype(int).tolist(),
                "test_set_indices": test_indices.astype(int).tolist(),
                "qber_estimate": float(qber.observed_qber),
                "qber_adjusted": float(qber.adjusted_qber),
                "finite_size_penalty": float(qber.mu_penalty),
                "test_set_size": int(qber.num_test_bits),
            },
            timing_ok,
        )

    def _phase3_reconcile(
        self, alice_bits: bitarray, bob_bits: bitarray, qber_adjusted: float
    ) -> Tuple[bitarray, int]:
        alice_arr = bitarray_to_numpy(alice_bits)
        bob_arr = bitarray_to_numpy(bob_bits)

        matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            config=ReconciliationOrchestratorConfig(frame_size=4096, max_retries=2),
            safety_cap=10**12,
        )

        block_result = orchestrator.reconcile_block(
            alice_key=alice_arr,
            bob_key=bob_arr,
            qber_estimate=qber_adjusted,
            block_id=0,
        )

        if not block_result.verified:
            raise SecurityError("Reconciliation failed verification")

        reconciled = bitarray_from_numpy(block_result.corrected_payload)
        total_syndrome_bits = int(block_result.syndrome_length)
        return reconciled, total_syndrome_bits

    def _phase4_amplify(
        self,
        reconciled_bits: bitarray,
        total_syndrome_bits: int,
        matching_indices: np.ndarray,
        i0_indices: np.ndarray,
        i1_indices: np.ndarray,
    ) -> Tuple[AliceObliviousKey, int, float, float, Tuple[bytes, bytes]]:
        reconciled_arr = bitarray_to_numpy(reconciled_bits)

        entropy_calc = NSMEntropyCalculator(
            storage_noise_r=float(self.params.nsm_params.storage_noise_r)
        )
        key_length_calc = SecureKeyLengthCalculator(entropy_calculator=entropy_calc)
        detailed = key_length_calc.compute_detailed(
            reconciled_length=int(len(reconciled_arr)),
            syndrome_leakage=int(total_syndrome_bits),
        )

        if detailed.final_length <= 0:
            raise EntropyDepletedError("Key extraction not viable (Death Valley)")

        # Partition reconciled key.
        reconciled_ba = bitarray_from_numpy(reconciled_arr)
        key_i0_ba, key_i1_ba = self._sifter.extract_partition_keys(
            sifted_bits=reconciled_ba,
            i0_indices=i0_indices,
            i1_indices=i1_indices,
            matching_indices=matching_indices,
        )

        key_i0_np = bitarray_to_numpy(key_i0_ba)
        key_i1_np = bitarray_to_numpy(key_i1_ba)

        # Seeds for Toeplitz; must be shared with Bob.
        # Toeplitz needs (n + m - 1) random bits.
        seed_0_bytes = ToeplitzHasher.generate_seed(
            num_bits=int(len(key_i0_np) + detailed.final_length - 1)
        )
        seed_1_bytes = ToeplitzHasher.generate_seed(
            num_bits=int(len(key_i1_np) + detailed.final_length - 1)
        )

        formatter = OTOutputFormatter(
            key_length=int(detailed.final_length),
            seed_0=seed_0_bytes,
            seed_1=seed_1_bytes,
        )

        alice_out = formatter.compute_alice_keys(key_i0=key_i0_np, key_i1=key_i1_np)

        alice_key = AliceObliviousKey(
            s0=bitarray_from_numpy(alice_out.key_0),
            s1=bitarray_from_numpy(alice_out.key_1),
            key_length=int(detailed.final_length),
            security_parameter=1e-10,
            entropy_consumed=float(detailed.entropy_consumed),
        )

        entropy_rate = float(detailed.final_length) / float(len(reconciled_arr))

        return (
            alice_key,
            int(detailed.final_length),
            float(detailed.entropy_consumed),
            float(entropy_rate),
            (seed_0_bytes, seed_1_bytes),
        )

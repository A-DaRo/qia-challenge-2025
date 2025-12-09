"""Bob baseline E-HOK role using the extensible orchestration layer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.constants import TARGET_EPSILON_SEC
from ehok.core.data_structures import ObliviousKey
from ehok.protocols.base import EHOKRole
from ehok.quantum.runner import QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.bob")


class BobBaselineEHOK(EHOKRole):
    """Baseline Bob role with pluggable strategies."""

    PEER_NAME = "alice"
    ROLE = "bob"

    def __init__(self, config: ProtocolConfig | None = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    def _execute_remaining_phases(
        self, quantum_result: QuantumPhaseResult
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        outcomes_bob = quantum_result.outcomes
        bases_bob = quantum_result.bases

        decommitment_salt, _commitment = yield from self._phase2_send_commitment(
            outcomes_bob, bases_bob
        )
        I_0, I_1, key_set = yield from self._phase3_sifting_sampling(
            outcomes_bob, bases_bob, decommitment_salt
        )

        bob_key = outcomes_bob[key_set]
        qber = yield from self._phase4_reconciliation(bob_key)

        self._build_privacy_amplifier()
        oblivious_key = yield from self._phase5_privacy_amplification(
            bob_key, qber, I_1, len(outcomes_bob)
        )

        return self._result_success(
            oblivious_key=oblivious_key,
            qber=qber,
            raw_count=len(outcomes_bob),
            sifted_count=len(I_0),
            test_count=len(I_0) - len(key_set),
            final_count=oblivious_key.final_length,
            role=self.ROLE,
            measurement_records=quantum_result.measurement_records,
        )

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def _phase2_send_commitment(
        self, outcomes_bob: np.ndarray, bases_bob: np.ndarray
    ) -> Generator[EventExpression, None, Tuple[bytes, bytes]]:
        logger.info("=== PHASE 2: Commitment ===")
        data = np.concatenate([outcomes_bob, bases_bob])
        commitment, decommitment_salt = self.commitment_scheme.commit(data)
        self.context.csockets[self.PEER_NAME].send(commitment.hex())
        yield from self.context.connection.flush()
        return decommitment_salt, commitment

    def _phase3_sifting_sampling(
        self, outcomes_bob: np.ndarray, bases_bob: np.ndarray, decommitment_salt: bytes
    ) -> Generator[EventExpression, None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        logger.info("=== PHASE 3: Sifting & Sampling ===")

        bases_alice_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        bases_alice = np.frombuffer(bytes.fromhex(bases_alice_msg), dtype=np.uint8)

        I_0, I_1 = self.sifting_manager.identify_matching_bases(bases_alice, bases_bob)
        test_set, key_set = self.sampling_strategy.select_sets(
            I_0,
            fraction=self.config.security.test_set_fraction,
            min_size=self.config.security.min_test_set_size,
            seed=self.config.sampling_seed,
        )

        full_data = np.concatenate(
            [
                outcomes_bob,
                bases_bob,
                np.frombuffer(decommitment_salt, dtype=np.uint8),
            ]
        )
        bob_data_msg = full_data.tobytes().hex()
        self.context.csockets[self.PEER_NAME].send(bob_data_msg)
        return I_0, I_1, key_set

    def _phase4_reconciliation(
        self, bob_key: np.ndarray
    ) -> Generator[EventExpression, None, float]:
        logger.info("=== PHASE 4: Information Reconciliation ===")

        syndrome_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        syndrome = np.frombuffer(bytes.fromhex(syndrome_msg), dtype=np.uint8)

        parity_check = self._load_ldpc_matrix(len(bob_key))
        self._build_reconciliator(parity_check)

        if len(bob_key) > self.reconciliator.n:
            logger.warning(
                "Truncating key from %d to %d to match LDPC matrix",
                len(bob_key),
                self.reconciliator.n,
            )
            bob_key = bob_key[: self.reconciliator.n]

        bob_corrected = self.reconciliator.reconcile(bob_key, syndrome)
        errors = np.sum(bob_key != bob_corrected)
        qber = errors / len(bob_key) if len(bob_key) > 0 else 0.0

        bob_hash = hashlib.sha256(bob_corrected.tobytes()).hexdigest()
        self.context.csockets[self.PEER_NAME].send(bob_hash)

        bob_key[:] = bob_corrected
        logger.info("Reconciliation complete (errors corrected=%d)", errors)
        return qber

    def _phase5_privacy_amplification(
        self, bob_key: np.ndarray, qber: float, I_1: np.ndarray, total_length: int
    ) -> Generator[EventExpression, None, ObliviousKey]:
        logger.info("=== PHASE 5: Privacy Amplification ===")
        seed_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        seed = np.frombuffer(bytes.fromhex(seed_msg), dtype=np.uint8)

        final_key = self.privacy_amplifier.compress(bob_key, seed)
        final_length = len(final_key)

        fraction_unknown = len(I_1) / total_length if total_length else 0
        num_unknown = int(final_length * fraction_unknown)
        knowledge_mask = np.zeros(final_length, dtype=np.uint8)
        if num_unknown > 0:
            knowledge_mask[:num_unknown] = 1

        oblivious_key = ObliviousKey(
            key_value=final_key,
            knowledge_mask=knowledge_mask,
            security_param=TARGET_EPSILON_SEC,
            qber=qber,
            final_length=final_length,
        )
        return oblivious_key

    # ------------------------------------------------------------------
    def _load_ldpc_matrix(self, n: int) -> sp.spmatrix:
        ldpc_dir = Path(__file__).parent.parent / "configs" / "ldpc_matrices"
        available_sizes = [1000, 2000, 4500, 5000]
        closest = min(available_sizes, key=lambda x: abs(x - n))
        matrix_file = ldpc_dir / f"ldpc_{closest}_rate05.npz"
        if not matrix_file.exists():
            raise FileNotFoundError(
                f"LDPC matrix not found: {matrix_file}. "
                "Run ehok/configs/generate_ldpc.py to generate matrices."
            )

        H = sp.load_npz(matrix_file)
        if n < H.shape[1]:
            H = H[:, :n]
            m_new = int(n * self.config.reconciliation.code_rate)
            H = H[:m_new, :]
        return H


# Backwards compatibility aliases
BobEHOKProgram = BobBaselineEHOK

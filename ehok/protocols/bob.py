"""Bob baseline E-HOK role using the extensible orchestration layer."""

from __future__ import annotations

import json
from typing import Generator, Dict, Any, Tuple

import numpy as np
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.constants import TARGET_EPSILON_SEC
from ehok.core.data_structures import LDPCBlockResult, ObliviousKey
from ehok.core.exceptions import MatrixSynchronizationError
from ehok.protocols.base import EHOKRole
from ehok.implementations import factories
from ehok.quantum.runner import QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.bob")


class BobBaselineEHOK(EHOKRole):
    """Baseline Bob role with pluggable strategies."""

    PEER_NAME = "alice"
    ROLE = "bob"

    def __init__(self, config: ProtocolConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

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
        self._build_reconciliator()
        block_result = yield from self._phase4_reconciliation(bob_key)
        qber = (
            block_result.error_count / block_result.block_length
            if block_result.block_length
            else 0.0
        )

        self._build_privacy_amplifier()
        oblivious_key = yield from self._phase5_privacy_amplification(
            bob_key, block_result, I_1, len(outcomes_bob)
        )

        # If privacy amplification yields zero-length key, abort gracefully
        if oblivious_key.final_length == 0:
            abort_reason = "PRIVACY_AMPLIFICATION_NO_SECURE_KEY"
            return self._result_abort(
                abort_reason=abort_reason,
                qber=qber,
                raw_count=len(outcomes_bob),
                sifted_count=len(I_0),
                test_count=len(I_0) - len(key_set),
                role=self.ROLE,
                measurement_records=quantum_result.measurement_records,
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
            seed=self.config.sampling_seed
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
        # Store estimated QBER for reconciliator (will be updated after Alice measures)
        self.measured_qber = 0.05  # Conservative initial, will be refined via reconciliation feedback
        return I_0, I_1, key_set

    def _phase4_reconciliation(
        self, bob_key: np.ndarray
    ) -> Generator[EventExpression, None, LDPCBlockResult]:
        logger.info("=== PHASE 4: Information Reconciliation ===")

        checksum = self.reconciliator.matrix_manager.checksum  # type: ignore[attr-defined]
        self.context.csockets[self.PEER_NAME].send(checksum)
        remote_checksum = yield from self.context.csockets[self.PEER_NAME].recv()
        if remote_checksum != checksum:
            raise MatrixSynchronizationError(checksum, remote_checksum)

        # Use measured QBER from Phase 3 for rate selection
        qber_est = getattr(self, 'measured_qber', 0.05)
        rate = self.reconciliator.select_rate(qber_est)
        n_short = self.reconciliator.compute_shortening(rate, qber_est, len(bob_key))
        seed = int(self.config.sampling_seed or 0)
        syndrome = self.reconciliator.compute_syndrome_block(bob_key, rate, n_short, seed)
        bob_hash = self.reconciliator.hash_verifier.compute_hash(bob_key, seed)  # type: ignore[attr-defined]

        payload = {
            "rate": rate,
            "n_short": n_short,
            "seed": seed,
            "payload_len": len(bob_key),
            "syndrome": syndrome.tobytes().hex(),
            "hash": bob_hash.hex(),
        }
        self.context.csockets[self.PEER_NAME].send(json.dumps(payload))

        ack_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        ack = json.loads(ack_msg)
        verified = bool(ack.get("verified", False))
        error_count = int(ack.get("error_count", len(bob_key)))
        if not verified:
            raise RuntimeError("Reconciliation failed: hash mismatch")
        result = LDPCBlockResult(
            verified=True,
            error_count=error_count,
            block_length=len(bob_key),
            syndrome_length=len(syndrome),
            hash_bits=self.reconciliator.hash_verifier.hash_length_bits,  # type: ignore[attr-defined]
        )
        logger.info("Reconciliation complete (errors corrected=%d)", error_count)
        return result

    def _phase5_privacy_amplification(
        self, bob_key: np.ndarray, block_result: LDPCBlockResult, I_1: np.ndarray, total_length: int
    ) -> Generator[EventExpression, None, ObliviousKey]:
        logger.info("=== PHASE 5: Privacy Amplification ===")
        seed_msg = yield from self.context.csockets[self.PEER_NAME].recv()
        seed_bytes = bytes.fromhex(seed_msg)
        if len(seed_bytes) == 0:
            seed = np.zeros(0, dtype=np.uint8)
            final_key = np.zeros(0, dtype=np.uint8)
        else:
            seed = np.frombuffer(seed_bytes, dtype=np.uint8)
            final_key = self.privacy_amplifier.compress(bob_key, seed)
        final_length = len(final_key)

        fraction_unknown = len(I_1) / total_length if total_length else 0
        num_unknown = int(final_length * fraction_unknown)
        knowledge_mask = np.zeros(final_length, dtype=np.uint8)
        if num_unknown > 0:
            knowledge_mask[:num_unknown] = 1

        if block_result.block_length == 0:
            qber = 0.0 if block_result.verified else 0.5
        else:
            qber = block_result.error_count / block_result.block_length if block_result.verified else 0.5
        oblivious_key = ObliviousKey(
            key_value=final_key,
            knowledge_mask=knowledge_mask,
            security_param=TARGET_EPSILON_SEC,
            qber=qber,
            final_length=final_length,
        )
        return oblivious_key

    # ------------------------------------------------------------------
    def _build_reconciliator(self) -> None:
        if getattr(self, "reconciliator", None) is not None:
            return
        self.reconciliator = factories.build_reconciliator(self.config)
        # Update reconciliator with measured QBER if available
        if hasattr(self, "measured_qber"):
            self.reconciliator.current_qber_est = self.measured_qber

# Backwards compatibility aliases
BobEHOKProgram = BobBaselineEHOK

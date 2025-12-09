"""
LDPC reconciliation orchestrator implementing block-based interface.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from ehok.core import constants
from ehok.core.data_structures import LDPCBlockResult, LDPCReconciliationResult
from ehok.core.exceptions import ReconciliationFailedError
from ehok.interfaces.reconciliation import IReconciliator
from ehok.implementations.reconciliation.ldpc_bp_decoder import LDPCBeliefPropagation
from ehok.implementations.reconciliation.ldpc_matrix_manager import LDPCMatrixManager
from ehok.implementations.reconciliation.polynomial_hash import PolynomialHashVerifier
from ehok.implementations.reconciliation.qber_estimator import IntegratedQBEREstimator
from ehok.utils.logging import get_logger

logger = get_logger("reconciliation.ldpc_reconciliator")


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class LDPCReconciliator(IReconciliator):
    """
    Block-based LDPC reconciliator with rate adaptation and integrated QBER estimation.
    """

    def __init__(
        self,
        matrix_manager: LDPCMatrixManager,
        bp_decoder: LDPCBeliefPropagation | None = None,
        hash_verifier: PolynomialHashVerifier | None = None,
        qber_estimator: IntegratedQBEREstimator | None = None,
        initial_qber_est: float = 0.05,
    ) -> None:
        self.matrix_manager = matrix_manager
        self.bp_decoder = bp_decoder or LDPCBeliefPropagation()
        self.hash_verifier = hash_verifier or PolynomialHashVerifier()
        self.qber_estimator = qber_estimator or IntegratedQBEREstimator()
        self.current_qber_est = initial_qber_est
        self.rng = np.random.default_rng(constants.PEG_DEFAULT_SEED)

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------
    def select_rate(self, qber_est: float) -> float:
        rates = self.matrix_manager.rates
        entropy = _binary_entropy(qber_est)
        if entropy == 0.0:
            entropy = 1e-9
        for rate in rates:
            if (1 - rate) / entropy < constants.LDPC_F_CRIT:
                return float(rate)
        return float(rates[-1])

    def compute_shortening(self, rate: float, qber_est: float, target_payload: int) -> int:
        n = self.matrix_manager.frame_size
        entropy = _binary_entropy(qber_est)
        if entropy == 0.0:
            entropy = 1e-9
        n_s = int(math.floor(n - target_payload / (constants.LDPC_F_CRIT * entropy)))
        n_s = max(0, min(n_s, n - 1))
        # Ensure payload + padding equals frame size
        if target_payload + n_s < n:
            n_s = n - target_payload
        return n_s

    def reconcile_block(
        self,
        key_block: np.ndarray,
        syndrome: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int,
    ) -> Tuple[np.ndarray, bool, int]:
        n = self.matrix_manager.frame_size
        if key_block.dtype != np.uint8:
            raise ValueError("key_block must be uint8")
        if key_block.size + n_shortened != n:
            padding_needed = n - key_block.size - n_shortened
            if padding_needed < 0:
                raise ValueError("block length exceeds frame size")
            n_shortened += padding_needed
        padding = self._generate_padding(n_shortened, prng_seed)
        full_frame = np.concatenate([key_block, padding])
        H = self.matrix_manager.get_matrix(rate)

        # Compute error syndrome
        local_syndrome = (H @ full_frame) % 2
        target_syndrome = (syndrome ^ local_syndrome).astype(np.uint8)

        llr_error = self._build_error_llrs(key_block.size, n_shortened)
        error_vector, converged, _ = self.bp_decoder.decode(H, llr_error, target_syndrome)
        corrected_frame = full_frame ^ error_vector.astype(np.uint8)
        corrected_payload = corrected_frame[: key_block.size]
        error_count = int(np.sum(error_vector[: key_block.size]))
        if not converged:
            logger.warning("Decoder did not converge for block (errors=%s)", error_count)
        return corrected_payload, converged, error_count

    def compute_syndrome_block(
        self, key_block: np.ndarray, rate: float, n_shortened: int, prng_seed: int
    ) -> np.ndarray:
        n = self.matrix_manager.frame_size
        if key_block.dtype != np.uint8:
            raise ValueError("key_block must be uint8")
        if key_block.size + n_shortened != n:
            padding_needed = n - key_block.size - n_shortened
            if padding_needed < 0:
                raise ValueError("block length exceeds frame size")
            n_shortened += padding_needed
        padding = self._generate_padding(n_shortened, prng_seed)
        full_frame = np.concatenate([key_block, padding])
        H = self.matrix_manager.get_matrix(rate)
        return (H @ full_frame) % 2

    def verify_block(self, block_alice: np.ndarray, block_bob: np.ndarray) -> Tuple[bool, bytes]:
        seed = block_bob.size
        hash_bob = self.hash_verifier.compute_hash(block_bob, seed)
        hash_alice = self.hash_verifier.compute_hash(block_alice, seed)
        return hash_alice == hash_bob, hash_bob

    def estimate_leakage_block(self, syndrome_length: int, hash_bits: int = constants.LDPC_HASH_BITS) -> int:
        return int(syndrome_length + hash_bits)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _build_error_llrs(self, payload_len: int, n_shortened: int) -> np.ndarray:
        n = payload_len + n_shortened
        p = max(min(self.current_qber_est, 1 - 1e-6), 1e-6)
        llr_payload = math.log((1 - p) / p)
        llrs = np.full(n, llr_payload, dtype=float)
        if n_shortened > 0:
            llrs[payload_len:] = 100.0  # Effectively infinite certainty of zero error
        return llrs

    def _generate_padding(self, length: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=length, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Convenience aggregation helpers
    # ------------------------------------------------------------------
    def aggregate_results(
        self, block_results: list[LDPCBlockResult], corrected_payloads: list[np.ndarray]
    ) -> LDPCReconciliationResult:
        qber_est = self.qber_estimator.estimate(block_results)
        total_leakage = sum(r.syndrome_length + r.hash_bits for r in block_results)
        blocks_verified = sum(1 for r in block_results if r.verified)
        blocks_discarded = len(block_results) - blocks_verified
        corrected_key = np.concatenate(corrected_payloads) if corrected_payloads else np.array([], dtype=np.uint8)
        return LDPCReconciliationResult(
            corrected_key=corrected_key,
            qber_estimate=qber_est,
            total_leakage=total_leakage,
            blocks_processed=len(block_results),
            blocks_verified=blocks_verified,
            blocks_discarded=blocks_discarded,
        )

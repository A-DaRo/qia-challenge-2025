"""
LDPC reconciliation orchestrator implementing block-based interface.

This module provides the main LDPC reconciliation class with adaptive rate
selection, dynamic LLR computation, and syndrome-guided soft-decision decoding.
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
    """
    Compute binary entropy function.

    Parameters
    ----------
    p : float
        Probability value in [0, 1].

    Returns
    -------
    float
        Binary entropy h(p) = -p*log2(p) - (1-p)*log2(1-p).
        Returns 0.0 for boundary values p=0 or p=1.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class LDPCReconciliator(IReconciliator):
    """
    Block-based LDPC reconciliator with rate adaptation and integrated QBER estimation.

    This reconciliator implements the core error correction functionality for QKD
    post-processing using Low-Density Parity-Check (LDPC) codes. It features:

    - Dynamic LLR computation based on measured QBER
    - Adaptive BP iteration scaling for noisy channels
    - Syndrome-guided soft-decision decoding
    - Integrated QBER estimation and rate selection

    Attributes
    ----------
    matrix_manager : LDPCMatrixManager
        Manager for LDPC parity-check matrices.
    bp_decoder : LDPCBeliefPropagation
        Belief-propagation decoder instance.
    hash_verifier : PolynomialHashVerifier
        Hash verifier for block authentication.
    qber_estimator : IntegratedQBEREstimator
        QBER estimator from block reconciliation results.
    current_qber_est : float
        Current QBER estimate used for rate selection and LLR computation.
    rng : numpy.random.Generator
        Random number generator for padding generation.
    """

    def __init__(
        self,
        matrix_manager: LDPCMatrixManager,
        bp_decoder: LDPCBeliefPropagation | None = None,
        hash_verifier: PolynomialHashVerifier | None = None,
        qber_estimator: IntegratedQBEREstimator | None = None,
        initial_qber_est: float = 0.05,
    ) -> None:
        """
        Initialize LDPC reconciliator.

        Parameters
        ----------
        matrix_manager : LDPCMatrixManager
            Manager instance providing LDPC parity-check matrices.
        bp_decoder : LDPCBeliefPropagation or None, optional
            Belief-propagation decoder. If None, creates default instance.
        hash_verifier : PolynomialHashVerifier or None, optional
            Hash verifier for block verification. If None, creates default instance.
        qber_estimator : IntegratedQBEREstimator or None, optional
            QBER estimator. If None, creates default instance.
        initial_qber_est : float, optional
            Initial QBER estimate, by default 0.05 (5%).
        """
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
        """
        Select appropriate LDPC code rate based on QBER estimate.

        Parameters
        ----------
        qber_est : float
            Estimated QBER value in [0, 1].

        Returns
        -------
        float
            Selected code rate from available rates.

        Notes
        -----
        Uses the critical efficiency criterion:
            (1 - R) / h(QBER) < f_crit

        where f_crit = 1.22 is the reconciliation efficiency threshold and
        h(QBER) is the binary entropy function. Selects the highest rate
        satisfying this criterion to maximize throughput while ensuring
        sufficient redundancy for error correction.
        """
        rates = self.matrix_manager.rates
        entropy = _binary_entropy(qber_est)
        if entropy == 0.0:
            entropy = 1e-9
        for rate in rates:
            if (1 - rate) / entropy < constants.LDPC_F_CRIT:
                return float(rate)
        return float(rates[-1])

    def compute_shortening(self, rate: float, qber_est: float, target_payload: int) -> int:
        """
        Compute number of shortened bits for a given payload length.

        Parameters
        ----------
        rate : float
            Selected LDPC code rate.
        qber_est : float
            Estimated QBER value.
        target_payload : int
            Desired payload length in bits.

        Returns
        -------
        int
            Number of bits to shorten (pad with known values).

        Notes
        -----
        Shortening adapts the fixed-size LDPC frame to variable payload lengths
        by padding with known bits that are assigned high reliability (LLR → ∞).
        The shortened bits effectively reduce the code dimension without changing
        the parity-check matrix structure.
        """
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
        max_retries: int = 2,
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Reconcile a single block using LDPC belief-propagation decoding.

        Parameters
        ----------
        key_block : np.ndarray
            Bob's sifted key block (uint8 array).
        syndrome : np.ndarray
            Alice's syndrome for this block.
        rate : float
            Selected LDPC code rate.
        n_shortened : int
            Number of shortened (padding) bits.
        prng_seed : int
            Seed for deterministic padding generation.
        max_retries : int, optional
            Maximum retry attempts on decode failure, by default 2.

        Returns
        -------
        corrected_payload : np.ndarray
            Corrected key block after error correction.
        converged : bool
            True if BP decoder converged to valid codeword.
        error_count : int
            Number of bit errors detected in payload.

        Raises
        ------
        ValueError
            If key_block dtype is not uint8 or block length exceeds frame size.

        Notes
        -----
        This method implements the core reconciliation algorithm with retry:

        1. Pad key_block to frame size using PRNG-generated bits
        2. Compute local syndrome and derive error syndrome
        3. Build dynamic LLRs using measured QBER
        4. Apply syndrome-guided LLR refinement
        5. Run adaptive BP decoder with QBER-scaled iterations
        6. On failure, retry with increased iterations and LLR damping
        7. Return corrected payload (excluding padding)

        Retry Strategy:
        - Attempt 1: Standard LLR with measured QBER
        - Attempt 2: 1.5x iterations, 0.85 LLR damping
        - Attempt 3: 2x iterations, 0.70 LLR damping
        """
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
        
        # Diagnostic logging
        try:
            row_count, col_count = H.shape
        except Exception:
            row_count = col_count = None

        # Compute error syndrome
        local_syndrome = (H @ full_frame) % 2
        target_syndrome = (syndrome ^ local_syndrome).astype(np.uint8)

        # Compute base adaptive iteration count
        base_iterations = self.compute_adaptive_iterations(self.current_qber_est)

        # Retry loop with LLR damping
        for attempt in range(max_retries + 1):
            # Scale parameters based on attempt
            iter_scale = 1.0 + attempt * 0.5
            llr_damping = 1.0 - attempt * 0.15

            # Build LLRs with damping for retries
            llr_error = self._build_error_llrs(key_block.size, n_shortened, self.current_qber_est)
            
            # Apply LLR damping for retry attempts (reduces confidence)
            payload_len = key_block.size
            llr_error[:payload_len] *= llr_damping

            # Apply syndrome-guided LLR refinement
            llr_error = self.syndrome_guided_llr_init(
                full_frame, target_syndrome, H, llr_error
            )
            
            # Compute scaled iteration count
            scaled_iterations = int(base_iterations * iter_scale)
            
            decoder = LDPCBeliefPropagation(
                max_iterations=scaled_iterations,
                threshold=self.bp_decoder.threshold
            )
            
            error_vector, converged, iterations = decoder.decode(H, llr_error, target_syndrome)
            
            if converged:
                break
            
            if attempt < max_retries:
                logger.debug(
                    "Decode attempt %d failed (iter=%d), retrying with damping=%.2f",
                    attempt + 1, iterations, llr_damping - 0.15
                )

        # Compute error count and log details
        error_count = int(np.sum(error_vector[: key_block.size]))
        target_syndrome_weight = int(np.sum(target_syndrome))
        local_syndrome_weight = int(np.sum(local_syndrome))
        logger.debug(
            "Reconcile block: rate=%s, payload_len=%s, n_shortened=%s, H=%s x %s, "
            "local_syndrome=%s, target_syndrome=%s, error_count=%s, iter=%s, "
            "converged=%s, attempts=%s",
            rate,
            key_block.size,
            n_shortened,
            row_count,
            col_count,
            local_syndrome_weight,
            target_syndrome_weight,
            error_count,
            iterations,
            converged,
            attempt + 1,
        )
        
        corrected_frame = full_frame ^ error_vector.astype(np.uint8)
        corrected_payload = corrected_frame[: key_block.size]
        
        if not converged:
            logger.warning(
                "Decoder did not converge after %d attempts (errors=%s, iter=%s)",
                max_retries + 1, error_count, iterations
            )
        
        return corrected_payload, converged, error_count

    def compute_syndrome_block(
        self, key_block: np.ndarray, rate: float, n_shortened: int, prng_seed: int
    ) -> np.ndarray:
        """
        Compute syndrome for a key block.

        Parameters
        ----------
        key_block : np.ndarray
            Key block to compute syndrome for (uint8 array).
        rate : float
            LDPC code rate.
        n_shortened : int
            Number of shortened bits.
        prng_seed : int
            Seed for padding generation.

        Returns
        -------
        np.ndarray
            Syndrome vector (binary array).

        Raises
        ------
        ValueError
            If key_block dtype is not uint8 or block length exceeds frame size.

        Notes
        -----
        Computes syndrome s = H · x (mod 2) where H is the parity-check matrix
        and x is the padded frame (key_block + padding).
        """
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
        """
        Verify block equality using polynomial hash.

        Parameters
        ----------
        block_alice : np.ndarray
            Alice's block (uint8 array).
        block_bob : np.ndarray
            Bob's corrected block (uint8 array).

        Returns
        -------
        matches : bool
            True if hash values match.
        hash_value : bytes
            Hash value computed from Bob's block.

        Notes
        -----
        Uses ε-universal polynomial hash with collision probability 2^{-50}.
        Hash seed is derived from block length for deterministic verification.
        """
        seed = block_bob.size
        hash_bob = self.hash_verifier.compute_hash(block_bob, seed)
        hash_alice = self.hash_verifier.compute_hash(block_alice, seed)
        return hash_alice == hash_bob, hash_bob

    def estimate_leakage_block(
        self,
        syndrome_length: int,
        hash_bits: int = constants.LDPC_HASH_BITS,
        n_shortened: int = 0,
        frame_size: int = 0,
    ) -> int:
        """
        Estimate information leakage for a single block.

        Parameters
        ----------
        syndrome_length : int
            Number of syndrome bits exchanged.
        hash_bits : int, optional
            Number of hash verification bits, by default constants.LDPC_HASH_BITS.
        n_shortened : int, optional
            Number of shortened bits (for enhanced accounting).
        frame_size : int, optional
            Frame size (for enhanced accounting).

        Returns
        -------
        int
            Total leakage in bits.

        Notes
        -----
        Information leakage accounts for all classical communication during
        reconciliation that could provide information to an eavesdropper:

        1. Syndrome bits: syndrome_length
        2. Hash verification: hash_bits
        3. Shortening positions: log2(C(n, n_s)) ≈ n_s * log2(n/n_s) bits
           (Upper bound assuming positions are disclosed)
        4. Rate selection: log2(|rates|) bits (negligible, ~3-4 bits)

        For conservative security, we use the upper bound for shortening leakage.
        """
        base_leakage = syndrome_length + hash_bits

        # Enhanced leakage accounting for shortening positions
        shortening_leakage = 0.0
        if n_shortened > 0 and frame_size > 0:
            # Upper bound: assume shortening positions are fully disclosed
            # Information content: log2(C(n, n_s)) ≈ n_s * log2(n / n_s) for large n
            ratio = frame_size / max(1, n_shortened)
            shortening_leakage = n_shortened * math.log2(ratio) if ratio > 1 else 0

        # Rate selection leakage (minor)
        rate_leakage = math.log2(len(constants.LDPC_CODE_RATES))

        total = base_leakage + shortening_leakage + rate_leakage
        return int(math.ceil(total))

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _build_error_llrs(self, payload_len: int, n_shortened: int, measured_qber: float) -> np.ndarray:
        """
        Build initial LLRs for error estimation using measured QBER.

        Parameters
        ----------
        payload_len : int
            Number of payload bits in the block.
        n_shortened : int
            Number of shortened (padding) bits.
        measured_qber : float
            Measured QBER from parameter estimation phase.
            Must be in range (0, 0.5) exclusive.

        Returns
        -------
        np.ndarray
            Initial LLR values for each bit position.

        Notes
        -----
        LLR = log((1-p)/p) where p is the crossover probability.
        For QBER → 0, LLR → +∞ (high confidence in channel observation).
        For QBER → 0.5, LLR → 0 (no information from channel).
        Shortened bits are assigned very high LLR (100.0) to enforce zero error.
        """
        n = payload_len + n_shortened
        # Clamp QBER to valid range to avoid numerical issues
        p = np.clip(measured_qber, 1e-6, 0.5 - 1e-6)
        llr_payload = math.log((1 - p) / p)
        llrs = np.full(n, llr_payload, dtype=float)
        if n_shortened > 0:
            llrs[payload_len:] = 100.0  # Effectively infinite certainty of zero error
        return llrs

    def compute_adaptive_iterations(
        self, measured_qber: float, base_iterations: int = 60
    ) -> int:
        """
        Compute adaptive iteration count based on QBER.

        Parameters
        ----------
        measured_qber : float
            Measured QBER from parameter estimation.
        base_iterations : int, optional
            Base iteration count for low-QBER scenarios, by default 60.

        Returns
        -------
        int
            Recommended iteration count.

        Notes
        -----
        Scaling formula derived from empirical decoder performance:
        - QBER < 2%: base_iterations (60)
        - QBER 2-5%: linear scaling up to 2x
        - QBER 5-10%: linear scaling up to 5x
        - QBER > 10%: capped at 5x base_iterations

        References
        ----------
        Kiktenko et al., "Post-processing procedure for industrial QKD systems"
        (2016): "For QBER > 5%, iteration counts of 100-200 may be necessary."
        """
        if measured_qber < 0.02:
            return base_iterations
        elif measured_qber < 0.05:
            # Linear interpolation: 60 at 2%, 120 at 5%
            scale = 1 + (measured_qber - 0.02) / 0.03
            return int(base_iterations * scale)
        elif measured_qber < 0.10:
            # Linear interpolation: 120 at 5%, 300 at 10%
            scale = 2 + 3 * (measured_qber - 0.05) / 0.05
            return int(base_iterations * scale)
        else:
            return base_iterations * 5  # Maximum scaling

    def syndrome_guided_llr_init(
        self,
        bob_block: np.ndarray,
        error_syndrome: np.ndarray,
        parity_matrix: sp.csr_matrix,
        initial_llrs: np.ndarray,
    ) -> np.ndarray:
        """
        Initialize LLRs with syndrome-guided adjustments.

        Parameters
        ----------
        bob_block : np.ndarray
            Bob's received block.
        error_syndrome : np.ndarray
            Error syndrome vector (unsatisfied parity checks).
        parity_matrix : sp.csr_matrix
            Parity-check matrix.
        initial_llrs : np.ndarray
            Initial LLR array (includes payload LLRs and high-reliability padding markers).

        Returns
        -------
        np.ndarray
            Refined LLR values with syndrome-guided adjustments.

        Notes
        -----
        Bits participating in unsatisfied checks are assigned lower reliability.
        This soft-decision approach improves decoder performance near capacity by
        leveraging syndrome structure to refine initial channel estimates.

        The reliability factor is computed as:
            reliability_factor = 1 - (unsatisfied_count / max_degree)

        where unsatisfied_count is the number of unsatisfied checks a variable
        participates in, and max_degree is the maximum column weight in H.
        
        IMPORTANT: This method preserves the high-reliability markers (LLR=100.0)
        for padding bits, ensuring shortened bits remain fixed during decoding.
        """
        n_vars = len(bob_block)
        # Start with the carefully constructed initial LLRs (preserves padding markers)
        llrs = initial_llrs.copy()

        # Count unsatisfied checks per variable
        unsatisfied_count = np.zeros(n_vars)
        for check_idx, syndrome_bit in enumerate(error_syndrome):
            if syndrome_bit == 1:  # Unsatisfied check
                # Get participating variables for this check
                start, end = parity_matrix.indptr[check_idx], parity_matrix.indptr[check_idx + 1]
                participating_vars = parity_matrix.indices[start:end]
                unsatisfied_count[participating_vars] += 1

        # Reduce LLR for variables in many unsatisfied checks
        # CRITICAL: Only adjust payload bits, preserve padding markers at 100.0
        max_degree = float(parity_matrix.getnnz(axis=0).max())
        if max_degree > 0:
            reliability_factor = 1.0 - (unsatisfied_count / max_degree)
            # Apply adjustment only to payload region (avoid corrupting padding markers)
            payload_len = np.sum(initial_llrs < 50.0)  # Padding has LLR=100.0, payload has LLR~3-5
            llrs[:payload_len] = llrs[:payload_len] * reliability_factor[:payload_len]

        return llrs

    def _generate_padding(self, length: int, seed: int) -> np.ndarray:
        """
        Generate deterministic padding bits for frame shortening.

        Parameters
        ----------
        length : int
            Number of padding bits to generate.
        seed : int
            PRNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            Binary padding array (uint8) of specified length.

        Notes
        -----
        Uses NumPy's default_rng for cryptographically strong PRNG.
        Both Alice and Bob must use the same seed to generate identical padding.
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=length, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Convenience aggregation helpers
    # ------------------------------------------------------------------
    def aggregate_results(
        self, block_results: list[LDPCBlockResult], corrected_payloads: list[np.ndarray]
    ) -> LDPCReconciliationResult:
        """
        Aggregate block-level results into final reconciliation outcome.

        Parameters
        ----------
        block_results : list of LDPCBlockResult
            Individual block reconciliation results.
        corrected_payloads : list of np.ndarray
            Corrected payload blocks from verified blocks.

        Returns
        -------
        LDPCReconciliationResult
            Aggregated reconciliation result containing:
            - corrected_key: Concatenated verified blocks
            - qber_estimate: Updated QBER estimate
            - total_leakage: Sum of all syndrome and hash bits
            - blocks_processed: Total number of blocks
            - blocks_verified: Number of successfully verified blocks
            - blocks_discarded: Number of failed blocks

        Notes
        -----
        QBER estimate is computed from block error counts using the integrated
        estimator. Failed blocks contribute QBER = 0.5 (maximum uncertainty).
        """
        qber_est = self.qber_estimator.estimate(block_results)
        # Update the internal current QBER estimate for subsequent LLR computation
        self.current_qber_est = qber_est
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

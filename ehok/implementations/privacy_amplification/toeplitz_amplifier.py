"""
Toeplitz hashing for privacy amplification.

This module implements the privacy amplification phase using Toeplitz matrices.
It provides functionality to generate a random Toeplitz seed, compress the
reconciled key, and calculate the secure final key length based on the
leftover hash lemma.
"""

import numpy as np
from typing import Any
from ehok.interfaces.privacy_amplification import IPrivacyAmplifier
from ehok.core.constants import TARGET_EPSILON_SEC, PA_SECURITY_MARGIN
from ehok.utils.logging import get_logger

logger = get_logger("toeplitz_pa")


class ToeplitzAmplifier(IPrivacyAmplifier):
    """
    Toeplitz matrix privacy amplification implementation.
    
    This class implements the IPrivacyAmplifier interface using Toeplitz
    matrices for 2-universal hashing.
    """

    def generate_hash_seed(self, input_length: int, output_length: int) -> np.ndarray:
        """
        Generate random seed for Toeplitz matrix.

        Parameters
        ----------
        input_length : int
            Length of the input key (n).
        output_length : int
            Length of the output key (m).

        Returns
        -------
        seed : np.ndarray
            Random bitstring of length (m + n - 1).
        """
        # The seed length required for an m x n Toeplitz matrix is m + n - 1
        seed_length = output_length + input_length - 1
        
        # Generate random bits
        seed = np.random.randint(0, 2, size=seed_length, dtype=np.uint8)
        
        logger.debug(f"Generated Toeplitz seed: length={seed_length}")
        return seed

    def compress(self, key: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """
        Apply Toeplitz matrix multiplication.

        Mathematical Operation
        ----------------------
        final_key = T @ key mod 2

        Where T is constructed from the seed.
        This implementation uses a sliding window approach which effectively
        constructs a Hankel matrix (T[i, j] = seed[i + j]). This is equivalent
        to a Toeplitz matrix up to permutation and maintains the 2-universal
        property.

        Parameters
        ----------
        key : np.ndarray
            Reconciled key of length n.
        seed : np.ndarray
            Hash function seed of length m + n - 1.

        Returns
        -------
        final_key : np.ndarray
            Compressed key of length m.
        """
        n = len(key)
        # seed length = m + n - 1 => m = seed_length - n + 1
        m = len(seed) - n + 1

        # Negative m indicates seed too short (invalid)
        if m < 0:
            raise ValueError(
                f"Invalid seed length {len(seed)} for key length {n}. "
                f"Seed must have length m + n - 1 where m >= 0. "
                f"For n={n}, seed length must be >= {n - 1}."
            )

        # If output length m == 0 (no secure key extractable), return empty array
        if m == 0:
            return np.zeros(0, dtype=np.uint8)

        # Pre-allocate output array
        final_key = np.zeros(m, dtype=np.uint8)

        # Perform matrix multiplication using sliding window
        # This is O(m*n) but avoids constructing the full matrix in memory.
        # For typical QKD block sizes (e.g. 10k bits), this is acceptable.
        
        # Optimization: Cast key to int32 to avoid overflow in np.dot with uint8.
        # While uint8 overflow (mod 256) preserves parity (mod 2), explicit
        # accumulation in a larger type is safer and clearer.
        key_int = key.astype(np.int32)
        
        for i in range(m):
            # Row i corresponds to the window seed[i : i+n]
            row = seed[i : i + n]
            # Dot product over GF(2)
            final_key[i] = np.dot(row, key_int) % 2

        logger.info(f"Privacy amplification completed: {n} -> {m} bits")
        return final_key

    def compute_final_length(
        self,
        sifted_length: int,
        qber: float,
        leakage: float,
        epsilon: float = TARGET_EPSILON_SEC
    ) -> int:
        """
        Calculate secure final key length.

        Formula (from leftover hash lemma):
        m <= n * [1 - h(qber)] - leakage - 2*log2(1/epsilon) - margin

        Parameters
        ----------
        sifted_length : int
            Length of reconciled key (n).
        qber : float
            Measured Quantum Bit Error Rate.
        leakage : float
            Information leaked during reconciliation (bits).
        epsilon : float
            Target security parameter.

        Returns
        -------
        final_length : int
            Maximum secure output length (m).
        """
        # Calculate binary entropy h(qber)
        if qber <= 0 or qber >= 1:
            h_qber = 0.0
        else:
            h_qber = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

        # Min-entropy of the raw key given Eve's knowledge (asymptotic)
        # H_min approx n * (1 - h(QBER))
        min_entropy = sifted_length * (1 - h_qber)

        # Security parameter cost: 2 * log2(1/epsilon)
        # If epsilon is 1e-9, log2(1e9) approx 30. 2*30 = 60 bits.
        epsilon_cost = 2 * np.log2(1.0 / epsilon)

        # Calculate final length
        # m = H_min - leakage - security_cost - margin
        m_float = min_entropy - leakage - epsilon_cost - PA_SECURITY_MARGIN
        
        m = int(np.floor(m_float))

        # If m <= 0, no secure key can be extracted
        if m <= 0:
            logger.warning(
                f"Cannot extract secure key: calculated m={m_float:.2f}. Returning zero length."
            )
            m = 0
        
        logger.info(
            f"Final length calculation: "
            f"n={sifted_length}, QBER={qber*100:.2f}%, "
            f"h(QBER)={h_qber:.4f}, leakage={leakage:.1f}, "
            f"epsilon={epsilon:.1e}, cost={epsilon_cost:.1f}, "
            f"margin={PA_SECURITY_MARGIN} -> m={m}"
        )

        return m

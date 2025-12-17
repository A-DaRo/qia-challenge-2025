"""
Toeplitz hashing for privacy amplification.

This module implements the privacy amplification phase using Toeplitz matrices.
It provides functionality to generate a random Toeplitz seed, compress the
reconciled key, and calculate the secure final key length using NSM-compliant
finite-key bounds.

References
----------
[1] Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R. (2012).
    "Tight finite-key analysis for quantum cryptography."
    Nature Communications, 3, 634.
[2] Lupo et al. (2023): NSM Max Bound formula.
"""

import secrets
import numpy as np
from typing import Optional
from ehok.interfaces.privacy_amplification import IPrivacyAmplifier
from ehok.core.constants import (
    TARGET_EPSILON_SEC,
    TEST_SET_FRACTION,
    MIN_TEST_SET_SIZE,
)
from ehok.utils.logging import get_logger
from .finite_key import (
    FiniteKeyParams,
    compute_final_length_finite_key,
    compute_blind_reconciliation_leakage,
    compute_final_length_blind_mode,
    binary_entropy,
    DEFAULT_EPSILON_COR,
)

logger = get_logger("toeplitz_pa")


class ToeplitzAmplifier(IPrivacyAmplifier):
    """
    Toeplitz matrix privacy amplification implementation.

    This class implements the IPrivacyAmplifier interface using Toeplitz
    matrices for 2-universal hashing with rigorous NSM finite-key security bounds.

    The implementation uses the NSM-compliant finite-key formula with
    statistical fluctuation correction μ(ε).

    Attributes
    ----------
    epsilon_sec : float
        Target security parameter (trace distance from ideal key).
    epsilon_cor : float
        Correctness parameter (probability of key mismatch).
    use_fft : bool
        If True, use FFT-based O(n log n) compression for large keys.
    fft_threshold : int
        Key length above which FFT compression is used (if use_fft=True).
    """

    def __init__(
        self,
        epsilon_sec: float = TARGET_EPSILON_SEC,
        epsilon_cor: float = DEFAULT_EPSILON_COR,
        use_fft: bool = False,
        fft_threshold: int = 10000,
    ) -> None:
        """
        Initialize Toeplitz amplifier with NSM finite-key security parameters.

        Parameters
        ----------
        epsilon_sec : float
            Target security parameter (default: 1e-9).
        epsilon_cor : float
            Correctness parameter (default: 1e-15).
        use_fft : bool
            Whether to use FFT-based compression for large keys.
        fft_threshold : int
            Key length above which FFT is used (if use_fft=True).
        """
        self.epsilon_sec = epsilon_sec
        self.epsilon_cor = epsilon_cor
        self.use_fft = use_fft
        self.fft_threshold = fft_threshold

    def generate_hash_seed(self, input_length: int, output_length: int) -> np.ndarray:
        """
        Generate cryptographically secure random seed for Toeplitz matrix.

        Uses OS-level entropy via `secrets.token_bytes()` to ensure the seed
        is unpredictable to any adversary without access to the authenticated
        classical channel.

        Parameters
        ----------
        input_length : int
            Length of the input key (n).
        output_length : int
            Length of the output key (m).

        Returns
        -------
        seed : np.ndarray
            Random bitstring of length (m + n - 1) with cryptographic randomness.

        Notes
        -----
        For Toeplitz hashing, the seed defines a specific hash function from
        the 2-universal family. Security requires uniform randomness.
        """
        seed_length = output_length + input_length - 1

        if seed_length <= 0:
            return np.zeros(0, dtype=np.uint8)

        # Generate cryptographically random bytes
        byte_length = (seed_length + 7) // 8
        random_bytes = secrets.token_bytes(byte_length)

        # Convert to bit array
        seed = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))[:seed_length]

        logger.debug(f"Generated Toeplitz seed: length={seed_length} (cryptographic)")
        return seed.astype(np.uint8)

    def compress(self, key: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """
        Apply Toeplitz matrix multiplication.

        Automatically selects between direct O(mn) and FFT O(n log n) methods
        based on key size and configuration.

        Mathematical Operation
        ----------------------
        final_key = T @ key mod 2

        Where T is constructed from the seed.

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
        m = len(seed) - n + 1

        # Validate inputs
        if m < 0:
            raise ValueError(
                f"Invalid seed length {len(seed)} for key length {n}. "
                f"Seed must have length m + n - 1 where m >= 0."
            )

        if m == 0:
            return np.zeros(0, dtype=np.uint8)

        # Select compression method based on key size
        if self.use_fft and n >= self.fft_threshold:
            return self._compress_fft(key, seed)
        else:
            return self._compress_direct(key, seed)

    def _compress_direct(self, key: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """
        Direct O(mn) Toeplitz matrix-vector multiplication.

        Uses sliding window approach which avoids constructing full matrix.
        """
        n = len(key)
        m = len(seed) - n + 1

        if m <= 0:
            return np.zeros(0, dtype=np.uint8)

        final_key = np.zeros(m, dtype=np.uint8)
        key_int = key.astype(np.int32)

        for i in range(m):
            row = seed[i : i + n]
            final_key[i] = np.dot(row, key_int) % 2

        logger.info(f"Privacy amplification completed: {n} -> {m} bits (direct)")
        return final_key

    def _compress_fft(self, key: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """
        FFT-based O(n log n) Toeplitz matrix-vector multiplication.

        Toeplitz matrix-vector multiply T @ x can be embedded in a circulant
        matrix-vector multiply, which is computed via FFT convolution.

        For GF(2), we work over integers and take mod 2 at the end.

        Parameters
        ----------
        key : np.ndarray
            Input key of length n.
        seed : np.ndarray
            Toeplitz seed of length m + n - 1.

        Returns
        -------
        np.ndarray
            Compressed key of length m.

        Notes
        -----
        Complexity: O((m+n) * log(m+n)) vs O(m*n) for direct method.
        Useful for large keys (n > 10000).
        """
        from scipy.fft import fft, ifft

        n = len(key)
        m = len(seed) - n + 1

        if m <= 0:
            return np.zeros(0, dtype=np.uint8)

        # Size for circulant embedding (must be at least m + n - 1)
        fft_size = m + n - 1

        # Construct first column of circulant embedding
        # The Toeplitz matrix T has T[i,j] = seed[i + (n-1-j)] for our sliding window
        # We need to embed this in a circulant matrix
        first_col = np.zeros(fft_size, dtype=np.float64)

        # First column of circulant is [seed[n-1], seed[n], ..., seed[m+n-2], 0, ..., 0, seed[0], ..., seed[n-2]]
        # For Toeplitz T@x where T[i,j] = t_{i-j}, first col is [t_0, t_1, ..., t_{m-1}, 0, ..., 0, t_{-(n-1)}, ..., t_{-1}]
        # Our seed structure: seed[i:i+n] gives row i, so T[i,j] = seed[i+j]
        # This is a Hankel matrix, equivalent to Toeplitz up to row/column reversal

        # For Hankel: H[i,j] = h_{i+j}
        # Result: y = H @ x where y[i] = sum_j h_{i+j} * x[j] = sum_j seed[i+j] * key[j]
        # This is a convolution: y = conv(seed, key)[0:m]

        # Pad key for convolution
        key_padded = np.zeros(fft_size, dtype=np.float64)
        key_padded[:n] = key.astype(np.float64)

        # Pad seed
        seed_padded = np.zeros(fft_size, dtype=np.float64)
        seed_padded[:len(seed)] = seed.astype(np.float64)

        # FFT convolution
        key_fft = fft(key_padded)
        seed_fft = fft(seed_padded)
        result_fft = ifft(seed_fft * key_fft)

        # Extract first m elements (the valid convolution outputs)
        result = np.round(np.real(result_fft)[:m]).astype(np.int64)

        # Reduce mod 2
        final_key = (result % 2).astype(np.uint8)

        logger.info(f"Privacy amplification completed: {n} -> {m} bits (FFT)")
        return final_key

    def compute_final_length(
        self,
        sifted_length: int,
        qber: float,
        leakage: float,
        epsilon: Optional[float] = None,
        test_bits: Optional[int] = None,
    ) -> int:
        """
        Calculate secure final key length with automatic finite-key correction.

        This method uses the Tomamichel et al. finite-key formula with
        statistical fluctuation correction μ(ε), eliminating the need for
        arbitrary security margins or fixed output length workarounds.

        Parameters
        ----------
        sifted_length : int
            Length of the reconciled key (n).
        qber : float
            Measured QBER from error correction or test bits.
        leakage : float
            Total information leaked (syndrome + verification hash bits).
        epsilon : float, optional
            Target security parameter. Uses self.epsilon_sec if not provided.
        test_bits : int, optional
            Number of bits used for QBER estimation (k). If None, estimated
            from TEST_SET_FRACTION * original_sifted_length.

        Returns
        -------
        int
            Maximum secure output length. Always non-negative.

        Notes
        -----
        The finite-key correction term μ provides a rigorous, automatic buffer
        that scales appropriately with key size:
        - Small keys (n~100): μ ≈ 0.05-0.08, conservative bound
        - Large keys (n~10000): μ ≈ 0.01-0.02, tighter bound

        This eliminates the need for:
        - `security_margin` parameter (deprecated)
        - `fixed_output_length` workaround in tests
        """
        if epsilon is None:
            epsilon = self.epsilon_sec

        # Estimate test bits if not provided
        if test_bits is None:
            # Assume standard fraction was used
            # sifted_length = original * (1 - test_fraction)
            # So original = sifted_length / (1 - test_fraction)
            total_sifted_estimate = int(sifted_length / (1 - TEST_SET_FRACTION))
            test_bits = max(MIN_TEST_SET_SIZE, int(total_sifted_estimate * TEST_SET_FRACTION))

        # Handle edge cases
        if sifted_length <= 0 or test_bits <= 0:
            return 0

        try:
            params = FiniteKeyParams(
                n=sifted_length,
                k=test_bits,
                qber_measured=qber,
                leakage=leakage,
                epsilon_sec=epsilon,
                epsilon_cor=self.epsilon_cor,
            )
            result = compute_final_length_finite_key(params)
        except ValueError as e:
            logger.warning(f"Invalid parameters for finite-key calculation: {e}")
            return 0

        logger.info(
            f"Final length (finite-key): n={sifted_length}, QBER={qber*100:.2f}%, "
            f"k={test_bits}, leakage={leakage:.1f}, ε={epsilon:.1e} -> m={result}"
        )

        return result

    def compute_final_length_blind(
        self,
        reconciled_length: int,
        error_count: int,
        frame_size: int,
        n_shortened: int,
        successful_rate: float,
        hash_bits: int,
        test_bits: int = 0,
        failed_attempts: int = 0,
    ) -> int:
        """
        Calculate secure key length for blind reconciliation scenario.

        Use this method when QBER is not known a priori and is inferred
        from reconciliation error counts.

        Parameters
        ----------
        reconciled_length : int
            Final reconciled key length.
        error_count : int
            Errors corrected during reconciliation.
        frame_size : int
            LDPC frame size used.
        n_shortened : int
            Number of shortened (padding) bits.
        successful_rate : float
            Final successful code rate.
        hash_bits : int
            Hash verification bits.
        test_bits : int
            Pre-reconciliation test bits (0 if blind).
        failed_attempts : int
            Number of failed decode attempts.

        Returns
        -------
        int
            Maximum secure output length.
        """
        return compute_final_length_blind_mode(
            reconciled_length=reconciled_length,
            error_count=error_count,
            frame_size=frame_size,
            n_shortened=n_shortened,
            successful_rate=successful_rate,
            hash_bits=hash_bits,
            test_bits=test_bits,
            failed_attempts=failed_attempts,
            epsilon_sec=self.epsilon_sec,
            epsilon_cor=self.epsilon_cor,
        )

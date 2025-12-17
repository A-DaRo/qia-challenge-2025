"""
Toeplitz matrix hashing for privacy amplification.

This module implements 2-universal hashing using Toeplitz matrices,
providing information-theoretic security for key extraction.

References
----------
- Carter & Wegman (1979): Universal hash families
- Tomamichel et al. (2011): Leftover hash lemma
"""

from __future__ import annotations

import secrets
from typing import Optional

import numpy as np
from numpy.fft import fft, ifft

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError

logger = get_logger(__name__)


class ToeplitzHasher:
    """
    2-universal hashing using Toeplitz matrices.

    A Toeplitz matrix is fully specified by its first row and column,
    requiring only (n + m - 1) random bits for an m×n matrix.

    Parameters
    ----------
    input_length : int
        Length of input in bits (n).
    output_length : int
        Length of output in bits (m).
    seed : Optional[bytes]
        Random seed for matrix generation.
    use_fft : bool
        Use FFT for O(n log n) computation (default True for large inputs).

    Notes
    -----
    A Toeplitz matrix T has the property T[i,j] = T[i-1,j-1], i.e.,
    constant along diagonals. This enables:
    - Compact representation: (n + m - 1) bits
    - Fast multiplication via FFT in O(n log n)

    The 2-universal property ensures:
        ∀x≠x': Pr[h(x) = h(x')] ≤ 2^{-m}

    References
    ----------
    - Carter & Wegman (1979): "Universal Classes of Hash Functions"
    - Krawczyk (1994): LFSR-based Toeplitz hashing
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        seed: Optional[bytes] = None,
        use_fft: bool = True,
    ) -> None:
        """
        Initialize Toeplitz hasher.

        Parameters
        ----------
        input_length : int
            Input key length (n).
        output_length : int
            Output key length (m).
        seed : Optional[bytes]
            Random seed for reproducibility.
        use_fft : bool
            Use FFT acceleration.
        """
        if input_length <= 0:
            raise InvalidParameterError(
                f"input_length={input_length} must be positive"
            )
        if output_length <= 0:
            raise InvalidParameterError(
                f"output_length={output_length} must be positive"
            )
        if output_length > input_length:
            raise InvalidParameterError(
                f"output_length={output_length} cannot exceed "
                f"input_length={input_length}"
            )

        self._n = input_length
        self._m = output_length
        self._use_fft = use_fft

        # Generate random bits for Toeplitz matrix
        num_random_bits = input_length + output_length - 1
        if seed is not None:
            # Use seed deterministically
            np.random.seed(int.from_bytes(seed[:4], "big"))
            self._random_bits = np.random.randint(
                0, 2, size=num_random_bits, dtype=np.uint8
            )
        else:
            # Cryptographically secure random
            self._random_bits = self._generate_random_bits(num_random_bits)

        self._seed = seed

        logger.debug(
            f"ToeplitzHasher: {input_length} → {output_length} bits, "
            f"FFT={'enabled' if use_fft else 'disabled'}"
        )

    def _generate_random_bits(self, n: int) -> np.ndarray:
        """Generate n cryptographically random bits."""
        num_bytes = (n + 7) // 8
        random_bytes = secrets.token_bytes(num_bytes)
        bits = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))
        return bits[:n].astype(np.uint8)

    def hash(self, input_key: np.ndarray) -> np.ndarray:
        """
        Apply Toeplitz hash to input.

        Parameters
        ----------
        input_key : np.ndarray
            Input bits as array of 0/1, length n.

        Returns
        -------
        np.ndarray
            Output bits, length m.

        Raises
        ------
        InvalidParameterError
            If input length doesn't match.
        """
        if len(input_key) != self._n:
            raise InvalidParameterError(
                f"Input length {len(input_key)} != expected {self._n}"
            )

        if self._use_fft and self._n > 64:
            return self._hash_fft(input_key)
        else:
            return self._hash_direct(input_key)

    def _hash_direct(self, input_key: np.ndarray) -> np.ndarray:
        """
        Direct matrix-vector multiplication.

        O(n·m) complexity but exact computation.
        """
        output = np.zeros(self._m, dtype=np.uint8)

        for i in range(self._m):
            # Row i of Toeplitz matrix: random_bits[i:i+n]
            row = self._random_bits[i : i + self._n]
            # Dot product mod 2
            output[i] = np.dot(row, input_key) % 2

        return output

    def _hash_fft(self, input_key: np.ndarray) -> np.ndarray:
        """
        FFT-accelerated Toeplitz multiplication.

        O(n log n) complexity using circular convolution.

        Notes
        -----
        Toeplitz multiplication can be embedded in circulant matrix
        multiplication, which is computed via FFT.
        """
        # Pad to next power of 2 for efficiency
        fft_size = 1
        while fft_size < self._n + self._m - 1:
            fft_size *= 2

        # First column of embedding circulant matrix
        first_col = np.zeros(fft_size, dtype=np.float64)
        first_col[: self._n + self._m - 1] = self._random_bits[
            : self._n + self._m - 1
        ]

        # Pad input
        padded_input = np.zeros(fft_size, dtype=np.float64)
        padded_input[: self._n] = input_key

        # FFT-based convolution
        col_fft = fft(first_col)
        input_fft = fft(padded_input)
        result_fft = col_fft * input_fft
        result = np.real(ifft(result_fft))

        # Round and take mod 2
        output = (np.round(result[: self._m]) % 2).astype(np.uint8)

        return output

    @staticmethod
    def generate_seed(num_bits: int) -> bytes:
        """
        Generate a cryptographic seed for the hash function.

        Parameters
        ----------
        num_bits : int
            Number of random bits needed (n + m - 1).

        Returns
        -------
        bytes
            Seed bytes.
        """
        num_bytes = (num_bits + 7) // 8
        return secrets.token_bytes(num_bytes)

    @property
    def input_length(self) -> int:
        """Input key length."""
        return self._n

    @property
    def output_length(self) -> int:
        """Output key length."""
        return self._m

    @property
    def random_bits(self) -> np.ndarray:
        """Random bits defining the Toeplitz matrix (read-only copy)."""
        return self._random_bits.copy()

    def get_matrix(self) -> np.ndarray:
        """
        Construct explicit Toeplitz matrix (for debugging).

        Returns
        -------
        np.ndarray
            Full m×n Toeplitz matrix.

        Notes
        -----
        Only for small matrices - O(mn) memory.
        """
        T = np.zeros((self._m, self._n), dtype=np.uint8)
        for i in range(self._m):
            T[i, :] = self._random_bits[i : i + self._n]
        return T

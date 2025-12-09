"""
Polynomial hashing for LDPC block verification.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ehok.core import constants


class PolynomialHashVerifier:
    """
    ε-universal polynomial hash for block verification.
    """

    def __init__(self, hash_bits: int = constants.LDPC_HASH_BITS, prime: int = 2 ** 61 - 1) -> None:
        if hash_bits <= 0:
            raise ValueError("hash_bits must be positive")
        self.hash_bits = hash_bits
        self.prime = prime
        self.mod_mask = (1 << hash_bits) - 1

    def compute_hash(self, bits: np.ndarray, seed: int) -> bytes:
        """
        Compute polynomial hash of bit vector using seed as generator.

        Parameters
        ----------
        bits : np.ndarray
            Payload bits as uint8 array.
        seed : int
            Shared seed for polynomial base.

        Returns
        -------
        bytes
            Hash value encoded in little-endian form using minimum bytes.
        """

        if bits.dtype != np.uint8:
            raise ValueError("bits must be uint8")
        base = (seed % self.prime) + 1
        acc = 0
        for bit in bits.tolist():
            acc = (acc * base + int(bit)) % self.prime
        acc &= self.mod_mask
        byte_len = (self.hash_bits + 7) // 8
        return int(acc).to_bytes(byte_len, byteorder="little")

    def verify(self, hash_value: bytes, bits: np.ndarray, seed: int) -> bool:
        """Verify hash for provided bits and seed."""
        return hash_value == self.compute_hash(bits, seed)

    @property
    def hash_length_bits(self) -> int:
        """Return length of hash in bits."""
        return self.hash_bits

    def hash_and_seed(self, bits: np.ndarray, seed: int) -> Tuple[bytes, int]:
        """Convenience helper returning hash and seed as tuple."""
        return self.compute_hash(bits, seed), seed

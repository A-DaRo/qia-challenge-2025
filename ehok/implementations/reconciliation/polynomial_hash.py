"""
Polynomial hashing for LDPC block verification.

This module provides an ε-universal hash function based on polynomial evaluation
over a prime field for secure and efficient block verification.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ehok.core import constants


class PolynomialHashVerifier:
    """
    ε-universal polynomial hash for block verification.

    Implements polynomial evaluation hash with configurable output length
    for collision-resistant verification of reconciled blocks.

    Parameters
    ----------
    hash_bits : int, optional
        Output hash length in bits, by default constants.LDPC_HASH_BITS (50 bits).
    prime : int, optional
        Prime modulus for polynomial field, by default 2**61 - 1.

    Attributes
    ----------
    hash_bits : int
        Configured hash length.
    prime : int
        Prime modulus.
    mod_mask : int
        Bit mask for truncating hash to specified length.

    Raises
    ------
    ValueError
        If hash_bits is not positive.

    Notes
    -----
    The hash function computes:

    .. math::
        h(x_1, \ldots, x_n) = \left(\sum_{i=1}^n x_i \cdot g^{n-i}\right) \mod p

    where g is the generator (derived from seed) and p is the prime modulus.
    Collision probability is bounded by 2^{-hash_bits} for ε-universal hashing.
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
        """
        Verify hash for provided bits and seed.

        Parameters
        ----------
        hash_value : bytes
            Expected hash value.
        bits : np.ndarray
            Bit array to verify (uint8).
        seed : int
            Shared seed.

        Returns
        -------
        bool
            True if computed hash matches provided hash_value.
        """
        return hash_value == self.compute_hash(bits, seed)

    @property
    def hash_length_bits(self) -> int:
        """
        Return length of hash in bits.

        Returns
        -------
        int
            Hash output length in bits.
        """
        return self.hash_bits

    def hash_and_seed(self, bits: np.ndarray, seed: int) -> Tuple[bytes, int]:
        """
        Convenience helper returning hash and seed as tuple.

        Parameters
        ----------
        bits : np.ndarray
            Bit array (uint8).
        seed : int
            Hash seed.

        Returns
        -------
        tuple of (bytes, int)
            Computed hash value and the seed used.
        """
        return self.compute_hash(bits, seed), seed

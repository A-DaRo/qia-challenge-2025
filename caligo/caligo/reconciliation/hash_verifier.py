"""
Polynomial Hash Verifier for Block Verification.

Provides ε-universal polynomial hashing for collision-resistant
verification of reconciled LDPC blocks.

The hash verifier is used after BP decoding to confirm that
Alice and Bob have identical corrected keys without transmitting
the full key.

References
----------
- Kiktenko et al. (2016): PolyR hash for industrial QKD
- Carter & Wegman (1979): Universal hash families
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from caligo.reconciliation import constants


class PolynomialHashVerifier:
    """
    ε-universal polynomial hash for block verification.

    Computes h(x) = Σ(xᵢ · gⁿ⁻ⁱ) mod p, truncated to output_bits.

    Parameters
    ----------
    output_bits : int, optional
        Output hash length in bits.
    hash_bits : int, optional
        Alias for output_bits (backward compatibility).
    prime : int, optional
        Prime modulus for polynomial field.

    Attributes
    ----------
    output_bits : int
        Configured hash length.
    prime : int
        Prime modulus (2^61 - 1 by default).
    mod_mask : int
        Bit mask for truncating to output_bits.

    Notes
    -----
    Collision probability is bounded by 2^{-output_bits}.
    With output_bits=50, P(collision) < 10^{-15}.
    """

    def __init__(
        self,
        output_bits: int = None,
        hash_bits: int = None,  # Alias for backward compatibility
        prime: int = constants.LDPC_HASH_PRIME,
    ) -> None:
        # Support both output_bits and hash_bits
        bits = output_bits if output_bits is not None else (
            hash_bits if hash_bits is not None else constants.LDPC_HASH_BITS
        )
        if bits <= 0:
            raise ValueError("output_bits must be positive")
        self._output_bits = bits
        self.prime = prime
        self.mod_mask = (1 << bits) - 1
        # Seed for deterministic hashing (shared secret)
        self._default_seed = 0x5EED

    @property
    def output_bits(self) -> int:
        """Return configured output hash length in bits."""
        return self._output_bits

    @property
    def hash_bits(self) -> int:
        """Alias for output_bits (backward compatibility)."""
        return self._output_bits

    def compute_hash(
        self,
        bits: np.ndarray,
        seed: int = None,
    ) -> int:
        """
        Compute polynomial hash of bit vector.

        Parameters
        ----------
        bits : np.ndarray
            Bit array (int8 or uint8, values 0 or 1).
        seed : int, optional
            Shared seed for polynomial base. Defaults to internal seed.

        Returns
        -------
        int
            Hash value as integer.
        """
        if seed is None:
            seed = self._default_seed

        base = (seed % self.prime) + 1
        acc = 0
        for bit in bits.flat:
            acc = (acc * base + int(bit)) % self.prime

        return acc & self.mod_mask

    def compute_hash_bytes(self, bits: np.ndarray, seed: int = None) -> bytes:
        """
        Compute polynomial hash and return as bytes.

        Parameters
        ----------
        bits : np.ndarray
            Bit array (int8 or uint8, values 0 or 1).
        seed : int, optional
            Shared seed for polynomial base.

        Returns
        -------
        bytes
            Hash value in little-endian encoding.
        """
        hash_value = self.compute_hash(bits, seed)
        byte_len = (self._output_bits + 7) // 8
        return int(hash_value).to_bytes(byte_len, byteorder="little")

    def verify(
        self,
        bits: np.ndarray,
        expected_hash: int,
        seed: int = None,
    ) -> bool:
        """
        Verify hash matches expected value.

        Parameters
        ----------
        bits : np.ndarray
            Bit array to verify.
        expected_hash : int
            Expected hash value (as integer).
        seed : int, optional
            Shared seed.

        Returns
        -------
        bool
            True if hashes match.
        """
        computed = self.compute_hash(bits, seed)
        return computed == expected_hash

    def verify_bytes(
        self,
        bits: np.ndarray,
        expected_hash: bytes,
        seed: int = None,
    ) -> bool:
        """
        Verify hash matches expected bytes value.

        Parameters
        ----------
        bits : np.ndarray
            Bit array to verify.
        expected_hash : bytes
            Expected hash value.
        seed : int, optional
            Shared seed.

        Returns
        -------
        bool
            True if hashes match.
        """
        computed = self.compute_hash_bytes(bits, seed)
        return computed == expected_hash

    def hash_and_seed(self, bits: np.ndarray, seed: int) -> Tuple[bytes, int]:
        """
        Convenience method returning hash and seed tuple.

        Parameters
        ----------
        bits : np.ndarray
            Bit array to hash.
        seed : int
            Hash seed.

        Returns
        -------
        Tuple[bytes, int]
            (hash_value, seed) pair.
        """
        return self.compute_hash(bits, seed), seed

    @property
    def hash_length_bits(self) -> int:
        """Return configured hash length in bits."""
        return self.hash_bits

    @property
    def hash_length_bytes(self) -> int:
        """Return hash length in bytes (ceiling)."""
        return (self.hash_bits + 7) // 8

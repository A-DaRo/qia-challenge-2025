"""
Basis selection for BB84-style quantum measurements.

This module provides uniform random basis selection using
cryptographic random number generation (secrets module).

References
----------
- BB84 Protocol: Computational (Z) and Hadamard (X) bases
- Erven et al. (2014): "choosing a random bit b"
"""

from __future__ import annotations

import secrets
from typing import Tuple

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError

logger = get_logger(__name__)


# Basis constants
BASIS_Z = 0  # Computational basis (standard)
BASIS_X = 1  # Hadamard basis (superposition)


class BasisSelector:
    """
    Uniform random basis selection for quantum measurements.

    Provides cryptographically secure basis selection for both
    single measurements and batch operations.

    Parameters
    ----------
    seed : Optional[bytes]
        Seed for reproducible testing (NOT for production).
        When None, uses cryptographic random source.

    Notes
    -----
    Security Note: In production, always use seed=None to ensure
    cryptographic randomness. Seeded selection is only for testing.

    References
    ----------
    - BB84: Two mutually unbiased bases
    - Erven et al. (2014): "each measurement basis is chosen uniformly"
    """

    def __init__(self, seed: bytes | None = None) -> None:
        """
        Initialize basis selector.

        Parameters
        ----------
        seed : Optional[bytes]
            Optional seed for reproducible selection (testing only).
        """
        self._rng: np.random.Generator | None = None
        if seed is not None:
            # Convert bytes to integer seed for numpy
            seed_int = int.from_bytes(seed[:8], "big")
            self._rng = np.random.default_rng(seed_int)
            logger.warning(
                "BasisSelector initialized with seed - NOT for production use"
            )

    def select_single(self) -> int:
        """
        Select a single random basis.

        Returns
        -------
        int
            Basis choice: 0 (Z) or 1 (X).

        Notes
        -----
        Uses secrets.randbelow() for cryptographic security when
        no seed is provided.
        """
        if self._rng is not None:
            return int(self._rng.integers(0, 2))
        return secrets.randbelow(2)

    def select_batch(self, n: int) -> np.ndarray:
        """
        Select bases for a batch of measurements.

        Parameters
        ----------
        n : int
            Number of basis choices to generate.

        Returns
        -------
        np.ndarray
            Array of basis choices (0 or 1), shape (n,), dtype uint8.

        Raises
        ------
        InvalidParameterError
            If n <= 0.

        Notes
        -----
        For large batches, uses secrets.token_bytes() for efficiency
        while maintaining cryptographic security.
        """
        if n <= 0:
            raise InvalidParameterError(f"Batch size n={n} must be positive")

        if self._rng is not None:
            # Seeded mode for testing
            return self._rng.integers(0, 2, size=n, dtype=np.uint8)

        # Cryptographic mode: generate random bytes and extract bits
        num_bytes = (n + 7) // 8
        random_bytes = secrets.token_bytes(num_bytes)

        # Convert to numpy array of bits
        bits = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))
        return bits[:n].astype(np.uint8)

    def select_weighted(self, n: int, p_x: float = 0.5) -> np.ndarray:
        """
        Select bases with weighted probability.

        Parameters
        ----------
        n : int
            Number of basis choices.
        p_x : float
            Probability of selecting X basis (default 0.5).

        Returns
        -------
        np.ndarray
            Array of basis choices, shape (n,), dtype uint8.

        Raises
        ------
        InvalidParameterError
            If n <= 0 or p_x not in [0, 1].

        Notes
        -----
        Standard BB84 uses p_x = 0.5 (uniform). Some protocols may
        benefit from asymmetric basis selection.
        """
        if n <= 0:
            raise InvalidParameterError(f"Batch size n={n} must be positive")
        if not 0 <= p_x <= 1:
            raise InvalidParameterError(f"p_x={p_x} must be in [0, 1]")

        if self._rng is not None:
            # Seeded mode
            return (self._rng.random(n) < p_x).astype(np.uint8)

        # Cryptographic mode using secrets
        # Generate uniform random and compare to threshold
        random_vals = np.frombuffer(
            secrets.token_bytes(n * 8), dtype=np.float64
        )[:n]
        # Normalize to [0, 1]
        random_vals = (random_vals - random_vals.min()) / (
            random_vals.max() - random_vals.min() + 1e-10
        )
        return (random_vals < p_x).astype(np.uint8)


def basis_to_string(basis: int) -> str:
    """
    Convert basis value to human-readable string.

    Parameters
    ----------
    basis : int
        Basis value (0 or 1).

    Returns
    -------
    str
        "Z" for computational, "X" for Hadamard.
    """
    return "Z" if basis == BASIS_Z else "X"


def bases_match(alice_basis: int, bob_basis: int) -> bool:
    """
    Check if Alice and Bob selected matching bases.

    Parameters
    ----------
    alice_basis : int
        Alice's basis choice.
    bob_basis : int
        Bob's basis choice.

    Returns
    -------
    bool
        True if bases match.
    """
    return alice_basis == bob_basis


def compute_matching_mask(
    alice_bases: np.ndarray, bob_bases: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices where bases match.

    Parameters
    ----------
    alice_bases : np.ndarray
        Alice's basis choices.
    bob_bases : np.ndarray
        Bob's basis choices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (matching_indices, basis_at_match) - indices where bases match
        and the common basis value at those positions.

    Raises
    ------
    ValueError
        If array lengths differ.
    """
    if len(alice_bases) != len(bob_bases):
        raise ValueError(
            f"Basis array lengths differ: {len(alice_bases)} != {len(bob_bases)}"
        )

    matching_mask = alice_bases == bob_bases
    matching_indices = np.where(matching_mask)[0]
    basis_at_match = alice_bases[matching_indices]

    return matching_indices, basis_at_match

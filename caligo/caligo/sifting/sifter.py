"""
Key sifting by basis match and I₀/I₁ partitioning.

This module implements the core sifting logic: extracting key bits
where Alice and Bob used matching bases, and partitioning into
I₀ (basis 0 matches) and I₁ (basis 1 matches) for OT.

References
----------
- Schaffner et al. (2009): "Sifting" step definition
- Lemus et al. (2020): I₀/I₁ partition for OT
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from bitarray import bitarray

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError, ProtocolError
from caligo.quantum.basis import BASIS_X, BASIS_Z

logger = get_logger(__name__)


@dataclass
class SiftingResult:
    """
    Result of the sifting operation.

    Parameters
    ----------
    sifted_bits : bitarray
        Key bits from matching basis positions.
    matching_indices : np.ndarray
        Original indices where bases matched.
    i0_indices : np.ndarray
        Indices from I₀ partition (Z basis matches).
    i1_indices : np.ndarray
        Indices from I₁ partition (X basis matches).
    total_matches : int
        Total number of basis matches.
    i0_count : int
        Count of I₀ partition positions.
    i1_count : int
        Count of I₁ partition positions.
    """

    sifted_bits: bitarray
    matching_indices: np.ndarray
    i0_indices: np.ndarray
    i1_indices: np.ndarray
    total_matches: int
    i0_count: int
    i1_count: int


class Sifter:
    """
    Basis sifting and key partitioning for E-HOK protocol.

    Performs the sifting step:
    1. Identify positions where Alice and Bob used matching bases
    2. Extract measurement outcomes at those positions
    3. Partition into I₀ (Z basis) and I₁ (X basis) sets

    The I₀/I₁ partition is used for the 1-out-of-2 OT:
    - Key from I₀: S₀ (Alice's key 0, Bob gets if choice=0)
    - Key from I₁: S₁ (Alice's key 1, Bob gets if choice=1)

    Notes
    -----
    BB84 with uniform basis selection yields ~50% matching rate.
    The sifted key length is approximately n/2 raw pairs.

    References
    ----------
    - Schaffner et al. (2009) Section 4: "Sifting"
    - Lemus et al. (2020): OT structure from basis partition
    """

    def __init__(self) -> None:
        """Initialize sifter."""
        self._sift_count = 0

    def compute_sifted_key(
        self,
        alice_bases: np.ndarray,
        alice_outcomes: np.ndarray,
        bob_bases: np.ndarray,
        bob_outcomes: np.ndarray,
    ) -> Tuple[SiftingResult, SiftingResult]:
        """
        Compute sifted keys for both parties.

        Parameters
        ----------
        alice_bases : np.ndarray
            Alice's basis choices.
        alice_outcomes : np.ndarray
            Alice's measurement outcomes.
        bob_bases : np.ndarray
            Bob's basis choices.
        bob_outcomes : np.ndarray
            Bob's measurement outcomes.

        Returns
        -------
        Tuple[SiftingResult, SiftingResult]
            (alice_result, bob_result) - sifting results for both parties.

        Raises
        ------
        InvalidParameterError
            If array lengths don't match.
        """
        n = len(alice_bases)

        # Validate inputs
        if len(bob_bases) != n:
            raise InvalidParameterError(
                f"Basis array lengths differ: {n} vs {len(bob_bases)}"
            )
        if len(alice_outcomes) != n:
            raise InvalidParameterError(
                f"Alice outcomes length {len(alice_outcomes)} != {n}"
            )
        if len(bob_outcomes) != n:
            raise InvalidParameterError(
                f"Bob outcomes length {len(bob_outcomes)} != {n}"
            )

        # Find matching bases
        matching_mask = alice_bases == bob_bases
        matching_indices = np.where(matching_mask)[0]
        total_matches = len(matching_indices)

        # Partition by basis value at matching positions
        matching_bases = alice_bases[matching_indices]
        i0_mask = matching_bases == BASIS_Z
        i1_mask = matching_bases == BASIS_X

        i0_indices = matching_indices[i0_mask]
        i1_indices = matching_indices[i1_mask]

        # Extract outcomes
        alice_sifted = alice_outcomes[matching_indices]
        bob_sifted = bob_outcomes[matching_indices]

        # Convert to bitarray
        alice_bits = bitarray(alice_sifted.tolist())
        bob_bits = bitarray(bob_sifted.tolist())

        logger.debug(
            f"Sifting complete: {n} pairs → {total_matches} matches "
            f"(I₀={len(i0_indices)}, I₁={len(i1_indices)})"
        )

        alice_result = SiftingResult(
            sifted_bits=alice_bits,
            matching_indices=matching_indices,
            i0_indices=i0_indices,
            i1_indices=i1_indices,
            total_matches=total_matches,
            i0_count=len(i0_indices),
            i1_count=len(i1_indices),
        )

        bob_result = SiftingResult(
            sifted_bits=bob_bits,
            matching_indices=matching_indices,
            i0_indices=i0_indices,
            i1_indices=i1_indices,
            total_matches=total_matches,
            i0_count=len(i0_indices),
            i1_count=len(i1_indices),
        )

        self._sift_count += 1
        return alice_result, bob_result

    def extract_partition_keys(
        self,
        sifted_bits: bitarray,
        i0_indices: np.ndarray,
        i1_indices: np.ndarray,
        matching_indices: np.ndarray,
    ) -> Tuple[bitarray, bitarray]:
        """
        Extract keys from I₀ and I₁ partitions.

        Parameters
        ----------
        sifted_bits : bitarray
            Full sifted key bits.
        i0_indices : np.ndarray
            Original indices for I₀.
        i1_indices : np.ndarray
            Original indices for I₁.
        matching_indices : np.ndarray
            All matching indices.

        Returns
        -------
        Tuple[bitarray, bitarray]
            (key_0, key_1) - keys from I₀ and I₁ partitions.
        """
        # Create index mapping from original to sifted position
        idx_map = {orig: i for i, orig in enumerate(matching_indices)}

        # Extract I₀ bits
        i0_positions = [idx_map[idx] for idx in i0_indices if idx in idx_map]
        key_0 = bitarray([sifted_bits[p] for p in i0_positions])

        # Extract I₁ bits
        i1_positions = [idx_map[idx] for idx in i1_indices if idx in idx_map]
        key_1 = bitarray([sifted_bits[p] for p in i1_positions])

        return key_0, key_1

    def select_test_subset(
        self,
        matching_indices: np.ndarray,
        test_fraction: float = 0.1,
        min_test_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly select a subset for QBER testing.

        Parameters
        ----------
        matching_indices : np.ndarray
            All matching position indices.
        test_fraction : float
            Fraction to use for testing (default 10%).
        min_test_size : int
            Minimum test set size.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (test_indices, key_indices) - indices for testing and key bits.
        """
        n_matches = len(matching_indices)
        test_size = max(min_test_size, int(n_matches * test_fraction))
        test_size = min(test_size, n_matches // 2)  # Don't use more than half

        # Cryptographically secure random selection
        all_indices = np.arange(n_matches)
        # Use secrets for secure shuffling
        shuffled = list(all_indices)
        # Fisher-Yates shuffle with secrets.randbelow
        for i in range(len(shuffled) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        test_positions = np.array(sorted(shuffled[:test_size]))
        key_positions = np.array(sorted(shuffled[test_size:]))

        # Map back to original indices
        test_indices = matching_indices[test_positions]
        key_indices = matching_indices[key_positions]

        logger.debug(
            f"Test subset: {len(test_indices)} test, {len(key_indices)} key"
        )

        return test_indices, key_indices

    @staticmethod
    def expected_matches(n_pairs: int, basis_match_prob: float = 0.5) -> float:
        """
        Calculate expected number of basis matches.

        Parameters
        ----------
        n_pairs : int
            Total EPR pairs.
        basis_match_prob : float
            Probability of basis match (0.5 for uniform).

        Returns
        -------
        float
            Expected matches.
        """
        return n_pairs * basis_match_prob

    @staticmethod
    def compute_match_rate(
        alice_bases: np.ndarray, bob_bases: np.ndarray
    ) -> float:
        """
        Compute actual basis match rate.

        Parameters
        ----------
        alice_bases : np.ndarray
            Alice's bases.
        bob_bases : np.ndarray
            Bob's bases.

        Returns
        -------
        float
            Match rate in [0, 1].
        """
        if len(alice_bases) == 0:
            return 0.0
        return np.mean(alice_bases == bob_bases)

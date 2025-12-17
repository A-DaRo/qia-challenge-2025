"""
Unit tests for Sifter.
"""

import numpy as np
import pytest
from bitarray import bitarray

from caligo.sifting.sifter import Sifter, SiftingResult
from caligo.types.exceptions import InvalidParameterError


class TestSifter:
    """Tests for Sifter class."""

    def test_compute_sifted_key_all_match(self):
        """All bases match."""
        sifter = Sifter()
        
        alice_bases = np.array([0, 0, 1, 1], dtype=np.uint8)
        alice_outcomes = np.array([1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([0, 0, 1, 1], dtype=np.uint8)
        bob_outcomes = np.array([1, 0, 1, 0], dtype=np.uint8)
        
        alice_result, bob_result = sifter.compute_sifted_key(
            alice_bases, alice_outcomes, bob_bases, bob_outcomes
        )
        
        assert alice_result.total_matches == 4
        assert len(alice_result.sifted_bits) == 4

    def test_compute_sifted_key_none_match(self):
        """No bases match."""
        sifter = Sifter()
        
        alice_bases = np.array([0, 0, 0, 0], dtype=np.uint8)
        alice_outcomes = np.array([1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([1, 1, 1, 1], dtype=np.uint8)
        bob_outcomes = np.array([0, 1, 0, 1], dtype=np.uint8)
        
        alice_result, bob_result = sifter.compute_sifted_key(
            alice_bases, alice_outcomes, bob_bases, bob_outcomes
        )
        
        assert alice_result.total_matches == 0
        assert len(alice_result.sifted_bits) == 0

    def test_compute_sifted_key_partial_match(self):
        """Partial basis match."""
        sifter = Sifter()
        
        # Positions 0, 2, 3 match
        alice_bases = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        alice_outcomes = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        bob_bases = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        bob_outcomes = np.array([1, 1, 1, 0, 0], dtype=np.uint8)
        
        alice_result, bob_result = sifter.compute_sifted_key(
            alice_bases, alice_outcomes, bob_bases, bob_outcomes
        )
        
        assert alice_result.total_matches == 3
        assert np.array_equal(alice_result.matching_indices, [0, 2, 3])

    def test_i0_i1_partition(self):
        """I₀ and I₁ partitions are correct."""
        sifter = Sifter()
        
        # Bases: 0, 1, 0, 1 (all match)
        alice_bases = np.array([0, 1, 0, 1], dtype=np.uint8)
        alice_outcomes = np.array([1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([0, 1, 0, 1], dtype=np.uint8)
        bob_outcomes = np.array([1, 0, 1, 0], dtype=np.uint8)
        
        alice_result, _ = sifter.compute_sifted_key(
            alice_bases, alice_outcomes, bob_bases, bob_outcomes
        )
        
        # I₀ = positions with basis 0 = [0, 2]
        # I₁ = positions with basis 1 = [1, 3]
        assert np.array_equal(alice_result.i0_indices, [0, 2])
        assert np.array_equal(alice_result.i1_indices, [1, 3])
        assert alice_result.i0_count == 2
        assert alice_result.i1_count == 2

    def test_length_mismatch_raises(self):
        """Mismatched array lengths raise error."""
        sifter = Sifter()
        
        alice_bases = np.array([0, 1, 0], dtype=np.uint8)
        alice_outcomes = np.array([1, 0, 1], dtype=np.uint8)
        bob_bases = np.array([0, 1], dtype=np.uint8)  # Wrong length
        bob_outcomes = np.array([1, 0, 1], dtype=np.uint8)
        
        with pytest.raises(InvalidParameterError):
            sifter.compute_sifted_key(
                alice_bases, alice_outcomes, bob_bases, bob_outcomes
            )

    def test_extract_partition_keys(self):
        """Extract keys from I₀ and I₁ partitions."""
        sifter = Sifter()
        
        # Create sifted result manually
        sifted_bits = bitarray("1010")  # 4 bits
        matching_indices = np.array([0, 2, 5, 7])
        i0_indices = np.array([0, 5])  # Positions with basis 0
        i1_indices = np.array([2, 7])  # Positions with basis 1
        
        key_0, key_1 = sifter.extract_partition_keys(
            sifted_bits, i0_indices, i1_indices, matching_indices
        )
        
        # key_0 should have bits at positions 0, 2 in sifted (indices 0, 5)
        assert len(key_0) == 2
        assert key_0[0] == sifted_bits[0]  # '1'
        assert key_0[1] == sifted_bits[2]  # '1'
        
        # key_1 should have bits at positions 1, 3 in sifted (indices 2, 7)
        assert len(key_1) == 2

    def test_select_test_subset(self):
        """Random test subset selection."""
        sifter = Sifter()
        
        matching_indices = np.arange(1000)
        
        test_indices, key_indices = sifter.select_test_subset(
            matching_indices, test_fraction=0.1
        )
        
        # Should have ~10% test, ~90% key
        assert len(test_indices) >= 100
        assert len(key_indices) > 800
        
        # No overlap
        assert len(set(test_indices) & set(key_indices)) == 0
        
        # Union covers all
        assert len(test_indices) + len(key_indices) == 1000

    def test_select_test_subset_min_size(self):
        """Minimum test subset size."""
        sifter = Sifter()
        
        matching_indices = np.arange(500)
        
        test_indices, key_indices = sifter.select_test_subset(
            matching_indices, test_fraction=0.01, min_test_size=50
        )
        
        # Should use min_test_size when 1% is too small
        assert len(test_indices) >= 50

    def test_expected_matches(self):
        """Expected match calculation."""
        expected = Sifter.expected_matches(n_pairs=10000, basis_match_prob=0.5)
        assert expected == 5000.0

    def test_compute_match_rate(self):
        """Actual match rate computation."""
        alice = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
        bob = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
        
        rate = Sifter.compute_match_rate(alice, bob)
        
        # Matches at positions 0, 2, 4 = 3/5
        assert rate == pytest.approx(0.6)

    def test_compute_match_rate_empty(self):
        """Match rate for empty arrays."""
        rate = Sifter.compute_match_rate(
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8)
        )
        assert rate == 0.0


class TestSiftingResult:
    """Tests for SiftingResult dataclass."""

    def test_sifting_result_creation(self):
        """Create SiftingResult."""
        result = SiftingResult(
            sifted_bits=bitarray("1010"),
            matching_indices=np.array([0, 1, 2, 3]),
            i0_indices=np.array([0, 2]),
            i1_indices=np.array([1, 3]),
            total_matches=4,
            i0_count=2,
            i1_count=2,
        )
        
        assert len(result.sifted_bits) == 4
        assert result.total_matches == 4

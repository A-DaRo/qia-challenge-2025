"""
Unit tests for BasisSelector.

Tests the uniform random basis selection for BB84-style measurements.
"""

import numpy as np
import pytest

from caligo.quantum.basis import (
    BasisSelector,
    BASIS_X,
    BASIS_Z,
    basis_to_string,
    bases_match,
    compute_matching_mask,
)
from caligo.types.exceptions import InvalidParameterError


class TestBasisSelector:
    """Tests for BasisSelector class."""

    def test_select_single_returns_valid_basis(self):
        """Single selection returns 0 or 1."""
        selector = BasisSelector()
        for _ in range(100):
            basis = selector.select_single()
            assert basis in (0, 1)

    def test_select_batch_returns_correct_size(self):
        """Batch selection returns correct array size."""
        selector = BasisSelector()
        n = 1000
        bases = selector.select_batch(n)
        assert len(bases) == n
        assert bases.dtype == np.uint8

    def test_select_batch_returns_only_valid_values(self):
        """Batch contains only 0 and 1."""
        selector = BasisSelector()
        bases = selector.select_batch(1000)
        assert set(np.unique(bases)) <= {0, 1}

    def test_select_batch_approximately_uniform(self):
        """Large batch is approximately uniform (50% each)."""
        selector = BasisSelector()
        n = 10000
        bases = selector.select_batch(n)
        
        # Expect ~50% each with some tolerance
        x_fraction = np.mean(bases)
        assert 0.45 < x_fraction < 0.55, f"X basis fraction {x_fraction} not ~0.5"

    def test_select_batch_invalid_n_raises(self):
        """Negative or zero n raises error."""
        selector = BasisSelector()
        
        with pytest.raises(InvalidParameterError):
            selector.select_batch(0)
        
        with pytest.raises(InvalidParameterError):
            selector.select_batch(-10)

    def test_seeded_selector_reproducible(self):
        """Seeded selector produces reproducible results."""
        seed = b"test_seed_12345"
        
        selector1 = BasisSelector(seed=seed)
        selector2 = BasisSelector(seed=seed)
        
        bases1 = selector1.select_batch(100)
        bases2 = selector2.select_batch(100)
        
        assert np.array_equal(bases1, bases2)

    def test_different_seeds_different_results(self):
        """Different seeds produce different results."""
        selector1 = BasisSelector(seed=b"seed_one")
        selector2 = BasisSelector(seed=b"seed_two")
        
        bases1 = selector1.select_batch(100)
        bases2 = selector2.select_batch(100)
        
        # Should differ (with overwhelming probability)
        assert not np.array_equal(bases1, bases2)

    def test_select_weighted_uniform(self):
        """Weighted selection with p_x=0.5 is uniform."""
        selector = BasisSelector(seed=b"test_seed")
        bases = selector.select_weighted(10000, p_x=0.5)
        
        x_fraction = np.mean(bases)
        assert 0.45 < x_fraction < 0.55

    def test_select_weighted_biased(self):
        """Weighted selection with p_x=0.8 is biased."""
        selector = BasisSelector(seed=b"test_seed")
        bases = selector.select_weighted(10000, p_x=0.8)
        
        x_fraction = np.mean(bases)
        assert 0.70 < x_fraction < 0.90

    def test_select_weighted_invalid_p_raises(self):
        """Invalid probability raises error."""
        selector = BasisSelector()
        
        with pytest.raises(InvalidParameterError):
            selector.select_weighted(100, p_x=-0.1)
        
        with pytest.raises(InvalidParameterError):
            selector.select_weighted(100, p_x=1.5)


class TestBasisHelpers:
    """Tests for basis helper functions."""

    def test_basis_to_string(self):
        """Basis conversion to string."""
        assert basis_to_string(BASIS_Z) == "Z"
        assert basis_to_string(BASIS_X) == "X"
        assert basis_to_string(0) == "Z"
        assert basis_to_string(1) == "X"

    def test_bases_match_same(self):
        """Matching bases return True."""
        assert bases_match(0, 0) is True
        assert bases_match(1, 1) is True

    def test_bases_match_different(self):
        """Different bases return False."""
        assert bases_match(0, 1) is False
        assert bases_match(1, 0) is False

    def test_compute_matching_mask_all_match(self):
        """All matching bases."""
        alice = np.array([0, 0, 1, 1], dtype=np.uint8)
        bob = np.array([0, 0, 1, 1], dtype=np.uint8)
        
        indices, bases = compute_matching_mask(alice, bob)
        
        assert len(indices) == 4
        assert np.array_equal(indices, [0, 1, 2, 3])

    def test_compute_matching_mask_none_match(self):
        """No matching bases."""
        alice = np.array([0, 0, 0, 0], dtype=np.uint8)
        bob = np.array([1, 1, 1, 1], dtype=np.uint8)
        
        indices, bases = compute_matching_mask(alice, bob)
        
        assert len(indices) == 0

    def test_compute_matching_mask_partial(self):
        """Partial matching."""
        alice = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        
        indices, bases = compute_matching_mask(alice, bob)
        
        # Positions 0, 2, 3 match
        assert len(indices) == 3
        assert np.array_equal(indices, [0, 2, 3])
        assert np.array_equal(bases, [0, 0, 1])

    def test_compute_matching_mask_length_mismatch_raises(self):
        """Mismatched lengths raise error."""
        alice = np.array([0, 1, 0], dtype=np.uint8)
        bob = np.array([0, 1], dtype=np.uint8)
        
        with pytest.raises(ValueError):
            compute_matching_mask(alice, bob)


class TestBasisConstants:
    """Tests for basis constants."""

    def test_basis_constants_values(self):
        """Basis constants have expected values."""
        assert BASIS_Z == 0
        assert BASIS_X == 1

"""
Unit tests for Phase 3: Sifting & Sampling.

These tests verify the correctness of the SiftingManager class, including
basis matching, test set selection, and QBER estimation, as defined in
e-hok-baseline-tests.md Section 5.
"""

import pytest
import numpy as np
from ehok.core.sifting import SiftingManager
from ehok.core.exceptions import QBERTooHighError
from ehok.core.constants import QBER_THRESHOLD, TEST_SET_FRACTION


class TestSifting:
    """Test suite for SiftingManager."""

    def setup_method(self):
        """Setup for each test."""
        self.sifter = SiftingManager()

    def test_basis_matching(self):
        """
        Test Case 5.1.1: Matching Indices Identification.
        
        Verifies that matching and non-matching basis indices are correctly identified.
        """
        bases_alice = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        bases_bob =   np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)

        I_0, I_1 = self.sifter.identify_matching_bases(bases_alice, bases_bob)

        # Expected sets
        expected_I_0 = {0, 2, 3, 5, 6}
        expected_I_1 = {1, 4, 7}

        # Postconditions
        assert set(I_0) == expected_I_0
        assert set(I_1) == expected_I_1
        assert len(I_0) + len(I_1) == len(bases_alice)
        assert set(I_0).intersection(set(I_1)) == set()

    def test_qber_estimation_exact(self):
        """
        Test Case 5.2.1: Known Error Rate.
        
        Verifies that QBER estimation is exact for a known error pattern.
        """
        N = 1000
        outcomes_alice = np.random.randint(0, 2, size=N, dtype=np.uint8)
        outcomes_bob = outcomes_alice.copy()
        
        # Introduce exactly 50 errors (5%)
        error_indices = np.arange(50)  # First 50 indices
        outcomes_bob[error_indices] = 1 - outcomes_bob[error_indices]
        
        test_set = np.arange(N)  # Use all indices for test

        qber = self.sifter.estimate_qber(outcomes_alice, outcomes_bob, test_set)

        # Postcondition: qber = 50/1000 = 0.05
        assert abs(qber - 0.05) < 1e-10

    def test_qber_abort_threshold(self):
        """
        Test Case 5.2.2: QBER Abort Threshold.
        
        Verifies that QBERTooHighError is raised only when QBER > threshold.
        """
        # Threshold is typically 0.11
        threshold = QBER_THRESHOLD

        # Case 1: Below threshold (0.10) -> No exception
        self.sifter.check_qber_abort(0.10, threshold)

        # Case 2: At threshold (0.11) -> No exception
        self.sifter.check_qber_abort(0.11, threshold)

        # Case 3: Above threshold (0.12) -> Raise QBERTooHighError
        with pytest.raises(QBERTooHighError) as excinfo:
            self.sifter.check_qber_abort(0.12, threshold)
        assert excinfo.value.measured_qber == 0.12
        assert excinfo.value.threshold == threshold

        # Case 4: Significantly above (0.15) -> Raise QBERTooHighError
        with pytest.raises(QBERTooHighError):
            self.sifter.check_qber_abort(0.15, threshold)

    def test_test_set_selection(self):
        """
        Test Case 5.3.1: Fraction Verification and Determinism.
        
        Verifies that test set size is correct and selection is deterministic with seed.
        """
        I_0 = np.arange(5000)  # 5000 matching bases
        seed = 42
        fraction = TEST_SET_FRACTION  # Typically 0.1

        test_set, key_set = self.sifter.select_test_set(I_0, fraction=fraction, seed=seed)

        # Postconditions
        expected_test_size = int(5000 * fraction)  # 500
        expected_key_size = 5000 - expected_test_size  # 4500

        assert len(test_set) == expected_test_size
        assert len(key_set) == expected_key_size
        assert set(test_set).union(set(key_set)) == set(I_0)
        assert set(test_set).intersection(set(key_set)) == set()

        # Determinism Test
        test_set_2, key_set_2 = self.sifter.select_test_set(I_0, fraction=fraction, seed=seed)
        assert np.array_equal(test_set, test_set_2)
        assert np.array_equal(key_set, key_set_2)

        # Different seed should produce different result (with high probability)
        test_set_3, _ = self.sifter.select_test_set(I_0, fraction=fraction, seed=seed + 1)
        assert not np.array_equal(test_set, test_set_3)

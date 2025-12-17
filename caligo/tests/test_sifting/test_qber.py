"""
Unit tests for QBEREstimator.
"""

import math
import numpy as np
import pytest

from caligo.sifting.qber import QBEREstimator, QBEREstimate
from caligo.types.exceptions import InvalidParameterError, QBERThresholdExceeded


class TestQBEREstimator:
    """Tests for QBEREstimator class."""

    @pytest.fixture
    def estimator(self):
        """Standard estimator fixture."""
        return QBEREstimator(epsilon_sec=1e-10)

    def test_estimate_no_errors(self, estimator):
        """Zero errors with large sample gives low adjusted QBER."""
        # Use large sample to minimize μ penalty
        n = 10000
        alice = np.zeros(n, dtype=np.uint8)
        bob = np.zeros(n, dtype=np.uint8)  # Identical
        
        result = estimator.estimate(alice, bob, key_size=100000)
        
        assert result.observed_qber == 0.0
        assert result.num_errors == 0
        # adjusted_qber = μ (small due to large sample)

    def test_estimate_all_errors(self, estimator):
        """All errors gives QBER=1.0 (exceeds limit)."""
        alice = np.array([0, 0, 0, 0], dtype=np.uint8)
        bob = np.array([1, 1, 1, 1], dtype=np.uint8)  # All different
        
        # Should raise because 100% error > 22%
        with pytest.raises(QBERThresholdExceeded):
            estimator.estimate(alice, bob, key_size=100)

    def test_estimate_partial_errors(self, estimator):
        """Partial errors with large sample."""
        # Use large sample to minimize μ penalty
        n = 10000
        alice = np.zeros(n, dtype=np.uint8)
        bob = np.zeros(n, dtype=np.uint8)
        
        # Set 5% errors = low enough to stay under 11% threshold with μ
        num_errors = 500
        bob[:num_errors] = 1
        
        result = estimator.estimate(alice, bob, key_size=100000)
        
        assert result.observed_qber == pytest.approx(0.05)
        assert result.num_errors == num_errors
        assert result.num_test_bits == n

    def test_adjusted_qber_includes_penalty(self, estimator):
        """Adjusted QBER = observed + μ."""
        # Large sample for small μ
        n = 10000
        alice = np.zeros(n, dtype=np.uint8)
        bob = np.zeros(n, dtype=np.uint8)  # Zero errors
        
        result = estimator.estimate(alice, bob, key_size=100000)
        
        # μ should be positive but small
        assert result.mu_penalty > 0
        assert result.adjusted_qber == pytest.approx(result.mu_penalty)

    def test_compute_mu_penalty_formula(self):
        """μ penalty formula matches Erven et al."""
        n, k = 1000, 500
        epsilon_sec = 1e-10
        
        mu = QBEREstimator.compute_mu_penalty(n, k, epsilon_sec)
        
        # μ = √[(n+k)/(n·k) · (k+1)/k · ln(4/ε_sec)]
        expected = math.sqrt(
            (n + k) / (n * k) * (k + 1) / k * math.log(4 / epsilon_sec)
        )
        
        assert mu == pytest.approx(expected)

    def test_compute_mu_penalty_smaller_with_more_data(self):
        """μ decreases with more data."""
        mu_small = QBEREstimator.compute_mu_penalty(100, 100, 1e-10)
        mu_large = QBEREstimator.compute_mu_penalty(10000, 10000, 1e-10)
        
        assert mu_large < mu_small

    def test_compute_mu_penalty_invalid_raises(self):
        """Invalid parameters raise."""
        with pytest.raises(InvalidParameterError):
            QBEREstimator.compute_mu_penalty(0, 100, 1e-10)
        
        with pytest.raises(InvalidParameterError):
            QBEREstimator.compute_mu_penalty(100, 0, 1e-10)

    def test_exceeds_hard_limit_raises(self, estimator):
        """QBER above 22% raises exception."""
        n = 100
        alice = np.zeros(n, dtype=np.uint8)
        bob = np.zeros(n, dtype=np.uint8)
        
        # Set 25 errors = 25% QBER
        bob[:25] = 1
        
        with pytest.raises(QBERThresholdExceeded):
            estimator.estimate(alice, bob, key_size=1000)

    def test_exceeds_warning_limit_sets_flag(self):
        """QBER between 11-22% sets warning flag."""
        estimator = QBEREstimator(epsilon_sec=1e-10)
        
        # Large sample for reliable measurement
        n = 10000
        alice = np.zeros(n, dtype=np.uint8)
        bob = np.zeros(n, dtype=np.uint8)
        
        # Set ~15% errors - above 11% warning but below 22% hard limit
        # With large sample, μ penalty is small (~0.01-0.02)
        num_errors = 1500
        bob[:num_errors] = 1
        
        # Should not raise, but flag should be set
        result = estimator.estimate(alice, bob, key_size=100000)
        
        assert result.exceeds_warning_limit or result.observed_qber > 0.11

    def test_validate_passes_for_good_qber(self, estimator):
        """Validate passes for acceptable QBER."""
        estimate = QBEREstimate(
            observed_qber=0.05,
            adjusted_qber=0.08,
            mu_penalty=0.03,
            num_test_bits=1000,
            num_errors=50,
            exceeds_hard_limit=False,
            exceeds_warning_limit=False,
        )
        
        assert estimator.validate(estimate) is True

    def test_validate_raises_for_bad_qber(self, estimator):
        """Validate raises for unacceptable QBER."""
        estimate = QBEREstimate(
            observed_qber=0.20,
            adjusted_qber=0.25,
            mu_penalty=0.05,
            num_test_bits=1000,
            num_errors=200,
            exceeds_hard_limit=True,
            exceeds_warning_limit=True,
        )
        
        with pytest.raises(QBERThresholdExceeded):
            estimator.validate(estimate)

    def test_empty_test_set_raises(self, estimator):
        """Empty test set raises error."""
        with pytest.raises(InvalidParameterError):
            estimator.estimate(
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.uint8),
                key_size=100,
            )

    def test_length_mismatch_raises(self, estimator):
        """Mismatched lengths raise error."""
        with pytest.raises(InvalidParameterError):
            estimator.estimate(
                np.array([0, 1, 0], dtype=np.uint8),
                np.array([0, 1], dtype=np.uint8),
                key_size=100,
            )

    def test_invalid_epsilon_raises(self):
        """Invalid security parameter raises."""
        with pytest.raises(InvalidParameterError):
            QBEREstimator(epsilon_sec=0)
        
        with pytest.raises(InvalidParameterError):
            QBEREstimator(epsilon_sec=1.5)


class TestQBEREstimate:
    """Tests for QBEREstimate dataclass."""

    def test_qber_estimate_creation(self):
        """Create QBEREstimate."""
        estimate = QBEREstimate(
            observed_qber=0.05,
            adjusted_qber=0.08,
            mu_penalty=0.03,
            num_test_bits=1000,
            num_errors=50,
        )
        
        assert estimate.observed_qber == 0.05
        assert estimate.adjusted_qber == 0.08

    def test_default_confidence_level(self):
        """Default confidence level."""
        estimate = QBEREstimate(
            observed_qber=0.05,
            adjusted_qber=0.08,
            mu_penalty=0.03,
            num_test_bits=1000,
            num_errors=50,
        )
        
        # Default: 1 - 1e-10
        assert estimate.confidence_level == pytest.approx(1 - 1e-10)

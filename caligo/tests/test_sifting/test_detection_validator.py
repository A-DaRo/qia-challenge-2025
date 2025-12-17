"""
Unit tests for DetectionValidator.
"""

import math
import numpy as np
import pytest

from caligo.sifting.detection_validator import (
    DetectionValidator,
    ValidationResult,
    HoeffdingBound,
)
from caligo.types.exceptions import InvalidParameterError


class TestDetectionValidator:
    """Tests for DetectionValidator class."""

    @pytest.fixture
    def validator(self):
        """Standard validator fixture."""
        return DetectionValidator(
            expected_detection_rate=1.0,
            tolerance=0.05,
            confidence=0.999,
        )

    def test_validate_perfect_detection(self, validator):
        """Perfect detection passes."""
        result = validator.validate_statistics(
            num_detections=1000,
            num_attempted=1000,
        )
        
        assert result.is_valid is True
        assert result.detection_rate == 1.0

    def test_validate_within_tolerance(self, validator):
        """Detection within tolerance passes."""
        result = validator.validate_statistics(
            num_detections=960,
            num_attempted=1000,
        )
        
        # 96% detection with 5% tolerance around 100%
        assert result.detection_rate == 0.96
        assert result.is_valid is True

    def test_validate_outside_tolerance_fails(self, validator):
        """Detection outside tolerance fails."""
        result = validator.validate_statistics(
            num_detections=800,
            num_attempted=1000,
        )
        
        # 80% detection exceeds 5% tolerance
        assert result.detection_rate == 0.80
        assert result.is_valid is False

    def test_validate_with_basis_balance(self, validator):
        """Validate with basis balance check."""
        bases = np.array([0] * 500 + [1] * 500, dtype=np.uint8)
        
        result = validator.validate_statistics(
            num_detections=1000,
            num_attempted=1000,
            bases=bases,
        )
        
        assert result.basis_balance_valid == True
        assert result.basis_0_fraction == 0.5

    def test_validate_unbalanced_bases_fails(self, validator):
        """Unbalanced bases fail validation."""
        bases = np.array([0] * 900 + [1] * 100, dtype=np.uint8)  # 90% basis 0
        
        result = validator.validate_statistics(
            num_detections=1000,
            num_attempted=1000,
            bases=bases,
        )
        
        assert result.basis_0_fraction == 0.9
        # With confidence=0.999, this imbalance should be detected
        assert result.basis_balance_valid == False

    def test_validate_no_attempts_fails(self, validator):
        """Zero attempts fails."""
        result = validator.validate_statistics(
            num_detections=0,
            num_attempted=0,
        )
        
        assert result.is_valid is False
        assert "No detection attempts" in result.message

    def test_hoeffding_bound_calculation(self, validator):
        """Hoeffding bound is computed."""
        result = validator.validate_statistics(
            num_detections=950,
            num_attempted=1000,
        )
        
        assert result.hoeffding_result is not None
        assert result.hoeffding_result.observed_rate == 0.95
        assert result.hoeffding_result.sample_size == 1000

    def test_hoeffding_bound_passes_large_sample(self, validator):
        """Large sample passes Hoeffding bound."""
        result = validator.validate_statistics(
            num_detections=9900,
            num_attempted=10000,
        )
        
        # Small deviation with large sample should pass
        assert result.hoeffding_result.passes is True

    def test_invalid_expected_rate_raises(self):
        """Invalid expected rate raises."""
        with pytest.raises(InvalidParameterError):
            DetectionValidator(expected_detection_rate=0)
        
        with pytest.raises(InvalidParameterError):
            DetectionValidator(expected_detection_rate=1.5)

    def test_invalid_tolerance_raises(self):
        """Invalid tolerance raises."""
        with pytest.raises(InvalidParameterError):
            DetectionValidator(tolerance=0)
        
        with pytest.raises(InvalidParameterError):
            DetectionValidator(tolerance=1.0)

    def test_invalid_confidence_raises(self):
        """Invalid confidence raises."""
        with pytest.raises(InvalidParameterError):
            DetectionValidator(confidence=0)
        
        with pytest.raises(InvalidParameterError):
            DetectionValidator(confidence=1.0)

    def test_required_samples_for_tolerance(self):
        """Required samples calculation."""
        n = DetectionValidator.required_samples_for_tolerance(
            tolerance=0.01,
            confidence=0.999,
        )
        
        # For 1% tolerance, 99.9% confidence, need ~38k samples
        assert n > 10000
        
        # Looser tolerance needs fewer samples
        n_loose = DetectionValidator.required_samples_for_tolerance(
            tolerance=0.1,
            confidence=0.999,
        )
        assert n_loose < n

    def test_properties(self, validator):
        """Property accessors."""
        assert validator.expected_rate == 1.0
        assert validator.tolerance == 0.05
        assert validator.confidence == 0.999


class TestHoeffdingBound:
    """Tests for HoeffdingBound dataclass."""

    def test_hoeffding_bound_creation(self):
        """Create HoeffdingBound."""
        bound = HoeffdingBound(
            observed_rate=0.95,
            expected_rate=1.0,
            deviation=0.05,
            bound=0.01,
            sample_size=1000,
            passes=True,
        )
        
        assert bound.deviation == 0.05
        assert bound.passes is True

    def test_hoeffding_formula(self):
        """Verify Hoeffding bound formula."""
        # P(|X̄ - μ| ≥ t) ≤ 2·exp(-2n·t²)
        t = 0.1
        n = 100
        
        bound = 2 * math.exp(-2 * n * t ** 2)
        
        # Should be very small
        assert bound < 0.3


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Create ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            detection_rate=0.98,
            expected_rate=1.0,
            message="Validation passed",
        )
        
        assert result.is_valid is True
        assert result.detection_rate == 0.98

    def test_default_values(self):
        """Default values are set."""
        result = ValidationResult(
            is_valid=True,
            detection_rate=1.0,
            expected_rate=1.0,
        )
        
        assert result.basis_balance_valid is True
        assert result.basis_0_fraction == 0.5
        assert result.message == ""

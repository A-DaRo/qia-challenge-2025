"""
Unit tests for Sprint 2 Statistical Validation.

Tests DetectionValidator, finite-size penalty calculation, and QBER adjustment
per sprint_2_specification.md Section 3 and Section 5.2-5.3.
"""

import math
import pytest

from ehok.protocols.ordered_messaging import DetectionReport
from ehok.protocols.statistical_validation import (
    # Abort codes
    ABORT_CODE_DETECTION_ANOMALY,
    ABORT_CODE_QBER_HIGH,
    # Enums
    ValidationStatus,
    # Result dataclasses
    DetectionValidationResult,
    FiniteSizePenaltyResult,
    QBERAdjustmentResult,
    # Classes
    DetectionValidator,
    QBERAdjuster,
    # Functions
    compute_finite_size_penalty,
    adjust_qber,
)
from ehok.analysis.nsm_bounds import QBER_HARD_LIMIT, QBER_WARNING_THRESHOLD


# =============================================================================
# Detection Validator Tests (TASK-DETECT-VALID-001)
# =============================================================================


class TestDetectionValidator:
    """Tests for Hoeffding/Chernoff detection validation."""

    @pytest.fixture
    def validator(self) -> DetectionValidator:
        """Create validator with typical parameters."""
        return DetectionValidator(
            expected_detection_prob=0.015,  # 1.5% detection rate
            failure_probability=1e-10,
        )

    def test_init_valid_params(self) -> None:
        """Validator initializes with valid parameters."""
        validator = DetectionValidator(
            expected_detection_prob=0.5, failure_probability=0.01
        )
        assert validator.expected_detection_prob == 0.5
        assert validator.failure_probability == 0.01

    def test_init_invalid_detection_prob_low(self) -> None:
        """Detection prob <= 0 should raise."""
        with pytest.raises(ValueError, match="expected_detection_prob"):
            DetectionValidator(expected_detection_prob=0.0, failure_probability=0.01)

    def test_init_invalid_detection_prob_high(self) -> None:
        """Detection prob >= 1 should raise."""
        with pytest.raises(ValueError, match="expected_detection_prob"):
            DetectionValidator(expected_detection_prob=1.0, failure_probability=0.01)

    def test_init_invalid_failure_prob(self) -> None:
        """Failure prob out of (0,1) should raise."""
        with pytest.raises(ValueError, match="failure_probability"):
            DetectionValidator(expected_detection_prob=0.5, failure_probability=0.0)

    def test_compute_tolerance_formula(self) -> None:
        """Tolerance should match ζ = √(ln(2/ε)/(2M))."""
        validator = DetectionValidator(
            expected_detection_prob=0.5, failure_probability=1e-10
        )
        m = 10000

        expected_zeta = math.sqrt(math.log(2.0 / 1e-10) / (2.0 * m))
        computed_zeta = validator.compute_tolerance(m)

        assert computed_zeta == pytest.approx(expected_zeta, rel=1e-10)

    def test_compute_tolerance_invalid_rounds(self, validator: DetectionValidator) -> None:
        """Zero or negative rounds should raise."""
        with pytest.raises(ValueError, match="total_rounds"):
            validator.compute_tolerance(0)

    def test_acceptance_interval(self, validator: DetectionValidator) -> None:
        """Acceptance interval should be [(P-ζ)M, (P+ζ)M]."""
        m = 10000
        zeta = validator.compute_tolerance(m)
        p = validator.expected_detection_prob

        lower, upper = validator.compute_acceptance_interval(m)

        expected_lower = (p - zeta) * m
        expected_upper = (p + zeta) * m

        assert lower == pytest.approx(expected_lower, rel=1e-10)
        assert upper == pytest.approx(expected_upper, rel=1e-10)

    def test_validate_pass_inside_interval(self, validator: DetectionValidator) -> None:
        """Detection count inside interval should pass."""
        # With P=0.015 and M=10000, expected S ≈ 150
        # Create report with S inside the interval
        m = 10000
        lower, upper = validator.compute_acceptance_interval(m)
        s = int((lower + upper) / 2)  # Middle of interval

        report = DetectionReport(
            total_rounds=m,
            detected_indices=list(range(s)),
            missing_indices=list(range(s, m)),
        )

        result = validator.validate(report)

        assert result.status == ValidationStatus.PASSED
        assert result.abort_code is None
        assert result.observed_s == s
        assert result.total_rounds == m

    def test_validate_fail_below_interval(self, validator: DetectionValidator) -> None:
        """Detection count well below expected should fail."""
        # Use higher detection rate so lower bound > 0
        validator = DetectionValidator(
            expected_detection_prob=0.5, failure_probability=1e-10
        )
        m = 10000
        lower, _ = validator.compute_acceptance_interval(m)
        # Set s well below lower bound
        s = max(0, int(lower) - 1000)

        report = DetectionReport(
            total_rounds=m,
            detected_indices=list(range(s)),
            missing_indices=list(range(s, m)),
        )

        result = validator.validate(report)

        assert result.status == ValidationStatus.FAILED
        assert result.abort_code == ABORT_CODE_DETECTION_ANOMALY
        assert "FAILED" in result.message

    def test_validate_fail_above_interval(self, validator: DetectionValidator) -> None:
        """Detection count above interval should fail."""
        m = 10000
        _, upper = validator.compute_acceptance_interval(m)
        s = min(m, int(upper) + 100)  # Well above interval

        report = DetectionReport(
            total_rounds=m,
            detected_indices=list(range(s)),
            missing_indices=list(range(s, m)),
        )

        result = validator.validate(report)

        assert result.status == ValidationStatus.FAILED
        assert result.abort_code == ABORT_CODE_DETECTION_ANOMALY

    def test_validate_boundary_inside(self, validator: DetectionValidator) -> None:
        """Detection count inside acceptance interval should pass."""
        # Use higher detection rate for clearer bounds
        validator = DetectionValidator(
            expected_detection_prob=0.5, failure_probability=1e-10
        )
        m = 10000
        lower, upper = validator.compute_acceptance_interval(m)

        # Use a value clearly inside the interval
        s = int((lower + upper) / 2)
        report = DetectionReport(
            total_rounds=m,
            detected_indices=list(range(s)),
            missing_indices=list(range(s, m)),
        )

        result = validator.validate(report)
        assert result.status == ValidationStatus.PASSED

        result = validator.validate(report)
        assert result.status == ValidationStatus.PASSED

    def test_validate_diagnostic_message(self, validator: DetectionValidator) -> None:
        """Result should contain diagnostic information."""
        # Use higher detection prob for cleaner bounds
        validator = DetectionValidator(
            expected_detection_prob=0.5, failure_probability=1e-10
        )
        m = 10000
        lower, upper = validator.compute_acceptance_interval(m)
        s = int((lower + upper) / 2)

        report = DetectionReport(
            total_rounds=m,
            detected_indices=list(range(s)),
            missing_indices=list(range(s, m)),
        )

        result = validator.validate(report)

        # Check diagnostic fields
        assert result.observed_s == s
        assert result.total_rounds == m
        assert result.expected_p == validator.expected_detection_prob
        assert result.tolerance_zeta > 0
        # With P=0.5 and M=10000, lower bound will be positive
        assert result.lower_bound > 0
        assert result.upper_bound > result.lower_bound
        assert "S=" in result.message

    def test_validate_invalid_rounds(self, validator: DetectionValidator) -> None:
        """Zero total_rounds should return INVALID_INPUT."""
        # Can't create DetectionReport with total_rounds=0 if indices non-empty
        # But we can test the edge case
        result = validator.validate(
            DetectionReport(total_rounds=0, detected_indices=[], missing_indices=[])
        )
        assert result.status == ValidationStatus.INVALID_INPUT


# =============================================================================
# Finite-Size Penalty Tests (TASK-FINITE-SIZE-001)
# =============================================================================


class TestFiniteSizePenalty:
    """Tests for finite-size penalty μ calculation."""

    def test_formula_matches_scarani(self) -> None:
        """μ should match Scarani Eq. (2): √((n+k)/(nk)·(k+1)/k)·ln(4/ε)."""
        n = 90000
        k = 10000
        eps = 1e-10

        # Compute expected per formula
        term1 = (n + k) / (n * k)
        term2 = (k + 1) / k
        expected_mu = math.sqrt(term1 * term2) * math.log(4.0 / eps)

        result = compute_finite_size_penalty(n, k, eps)

        assert result.mu == pytest.approx(expected_mu, rel=1e-10)

    def test_result_contains_inputs(self) -> None:
        """Result should contain input parameters."""
        result = compute_finite_size_penalty(
            key_size_n=80000, test_size_k=20000, epsilon_sec=1e-9
        )

        assert result.test_size_k == 20000
        assert result.key_size_n == 80000
        assert result.epsilon_sec == 1e-9

    def test_mu_decreases_with_larger_test_set(self) -> None:
        """μ should decrease as test set grows (μ ∝ 1/√k)."""
        n = 90000
        eps = 1e-10

        mu_small_k = compute_finite_size_penalty(n, 1000, eps).mu
        mu_large_k = compute_finite_size_penalty(n, 10000, eps).mu

        assert mu_large_k < mu_small_k

    def test_mu_increases_with_tighter_security(self) -> None:
        """μ should increase with smaller ε_sec (μ ∝ ln(1/ε))."""
        n = 90000
        k = 10000

        mu_loose = compute_finite_size_penalty(n, k, 1e-6).mu
        mu_tight = compute_finite_size_penalty(n, k, 1e-12).mu

        assert mu_tight > mu_loose

    def test_invalid_n_raises(self) -> None:
        """n <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="key_size_n"):
            compute_finite_size_penalty(0, 10000, 1e-10)

    def test_invalid_k_raises(self) -> None:
        """k <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="test_size_k"):
            compute_finite_size_penalty(90000, 0, 1e-10)

    def test_invalid_epsilon_raises(self) -> None:
        """ε ∉ (0,1) should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon_sec"):
            compute_finite_size_penalty(90000, 10000, 0.0)
        with pytest.raises(ValueError, match="epsilon_sec"):
            compute_finite_size_penalty(90000, 10000, 1.0)

    def test_realistic_values(self) -> None:
        """Test with realistic QKD parameters."""
        # Typical parameters from Erven et al.
        n = 100000  # Key bits
        k = 10000  # Test bits
        eps = 1e-10  # Security parameter

        result = compute_finite_size_penalty(n, k, eps)

        # μ should be positive and well-defined
        assert result.mu > 0
        # The formula gives ~0.26 for these params, which is a significant penalty
        # This is expected for finite-size regime
        assert result.mu < 1.0  # Should be less than 100%


# =============================================================================
# QBER Adjustment Tests (TASK-QBER-ADJUST-001)
# =============================================================================


class TestQBERAdjustment:
    """Tests for QBER adjustment and threshold checks."""

    def test_adjustment_adds_penalty(self) -> None:
        """e_adj = e_obs + μ."""
        result = adjust_qber(observed_qber=0.05, penalty_mu=0.003)

        assert result.adjusted_qber == pytest.approx(0.053, rel=1e-10)
        assert result.observed_qber == 0.05
        assert result.penalty_mu == 0.003

    def test_pass_below_warning(self) -> None:
        """QBER below warning threshold should pass."""
        result = adjust_qber(observed_qber=0.05, penalty_mu=0.01)

        # 0.05 + 0.01 = 0.06 < 0.11
        assert result.status == ValidationStatus.PASSED
        assert result.abort_code is None

    def test_warning_between_thresholds(self) -> None:
        """QBER between 11% and 22% should warn."""
        result = adjust_qber(observed_qber=0.10, penalty_mu=0.05)

        # 0.10 + 0.05 = 0.15, which is > 0.11 but < 0.22
        assert result.status == ValidationStatus.WARNING
        assert result.abort_code is None
        assert "WARNING" in result.message

    def test_fail_above_hard_limit(self) -> None:
        """QBER above 22% should abort."""
        result = adjust_qber(observed_qber=0.18, penalty_mu=0.05)

        # 0.18 + 0.05 = 0.23 > 0.22
        assert result.status == ValidationStatus.FAILED
        assert result.abort_code == ABORT_CODE_QBER_HIGH
        assert "HARD LIMIT" in result.message

    def test_boundary_at_warning(self) -> None:
        """QBER exactly at 11% should warn (>)."""
        result = adjust_qber(observed_qber=0.10, penalty_mu=0.01)

        # 0.10 + 0.01 = 0.11, exactly at warning threshold
        # Per spec: > 0.11 triggers warning, so 0.11 exactly passes
        assert result.status == ValidationStatus.PASSED

    def test_boundary_above_warning(self) -> None:
        """QBER just above 11% should warn."""
        result = adjust_qber(observed_qber=0.10, penalty_mu=0.02)

        # 0.10 + 0.02 = 0.12 > 0.11
        assert result.status == ValidationStatus.WARNING

    def test_boundary_at_hard_limit(self) -> None:
        """QBER exactly at 22% should not abort (> not >=)."""
        result = adjust_qber(observed_qber=0.20, penalty_mu=0.02)

        # 0.20 + 0.02 = 0.22, exactly at hard limit
        # Per spec: strictly > triggers abort
        assert result.status == ValidationStatus.WARNING

    def test_boundary_above_hard_limit(self) -> None:
        """QBER just above 22% should abort."""
        result = adjust_qber(observed_qber=0.20, penalty_mu=0.03)

        # 0.20 + 0.03 = 0.23 > 0.22
        assert result.status == ValidationStatus.FAILED

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        result = adjust_qber(
            observed_qber=0.06,
            penalty_mu=0.01,
            warning_threshold=0.05,  # Custom lower warning
            hard_limit=0.10,  # Custom lower hard limit
        )

        # 0.06 + 0.01 = 0.07 > 0.05 (custom warning)
        assert result.status == ValidationStatus.WARNING
        assert result.warning_threshold == 0.05
        assert result.hard_limit == 0.10

    def test_invalid_qber_raises(self) -> None:
        """QBER outside [0, 0.5] should raise."""
        with pytest.raises(ValueError, match="observed_qber"):
            adjust_qber(observed_qber=-0.01, penalty_mu=0.01)
        with pytest.raises(ValueError, match="observed_qber"):
            adjust_qber(observed_qber=0.6, penalty_mu=0.01)

    def test_invalid_penalty_raises(self) -> None:
        """Negative penalty should raise."""
        with pytest.raises(ValueError, match="penalty_mu"):
            adjust_qber(observed_qber=0.05, penalty_mu=-0.01)

    def test_result_thresholds_included(self) -> None:
        """Result should include threshold values."""
        result = adjust_qber(observed_qber=0.05, penalty_mu=0.01)

        assert result.warning_threshold == QBER_WARNING_THRESHOLD
        assert result.hard_limit == QBER_HARD_LIMIT


class TestQBERAdjuster:
    """Tests for stateful QBERAdjuster class."""

    @pytest.fixture
    def adjuster(self) -> QBERAdjuster:
        """Create adjuster with typical parameters."""
        return QBERAdjuster(epsilon_sec=1e-10)

    def test_compute_adjusted_qber(self, adjuster: QBERAdjuster) -> None:
        """Adjuster should compute penalty and check thresholds."""
        result = adjuster.compute_adjusted_qber(
            observed_qber=0.05, key_size_n=90000, test_size_k=10000
        )

        assert result.observed_qber == 0.05
        assert result.penalty_mu > 0
        assert result.adjusted_qber == result.observed_qber + result.penalty_mu
        assert result.status in ValidationStatus

    def test_adjuster_uses_configured_thresholds(self) -> None:
        """Adjuster should use configured thresholds."""
        adjuster = QBERAdjuster(
            epsilon_sec=1e-10, warning_threshold=0.05, hard_limit=0.10
        )

        result = adjuster.compute_adjusted_qber(
            observed_qber=0.04, key_size_n=90000, test_size_k=10000
        )

        assert result.warning_threshold == 0.05
        assert result.hard_limit == 0.10

    def test_adjuster_properties(self) -> None:
        """Adjuster should expose configured properties."""
        adjuster = QBERAdjuster(
            epsilon_sec=1e-9, warning_threshold=0.08, hard_limit=0.15
        )

        assert adjuster.epsilon_sec == 1e-9
        assert adjuster.warning_threshold == 0.08
        assert adjuster.hard_limit == 0.15

    def test_invalid_epsilon_raises(self) -> None:
        """Invalid epsilon should raise."""
        with pytest.raises(ValueError, match="epsilon_sec"):
            QBERAdjuster(epsilon_sec=0.0)


# =============================================================================
# Integration Test: Detection Validation + QBER Adjustment
# =============================================================================


class TestStatisticalValidationIntegration:
    """Integration tests for the complete statistical validation pipeline."""

    def test_full_validation_pipeline_pass(self) -> None:
        """Full pipeline should pass with good parameters."""
        # Setup
        expected_detection = 0.5  # Higher detection rate
        total_rounds = 100000
        epsilon_sec = 1e-10

        # Create detection validator
        det_validator = DetectionValidator(
            expected_detection_prob=expected_detection,
            failure_probability=epsilon_sec,
        )

        # Simulate detection report within bounds
        lower, upper = det_validator.compute_acceptance_interval(total_rounds)
        detected = int((lower + upper) / 2)
        detection_report = DetectionReport(
            total_rounds=total_rounds,
            detected_indices=list(range(detected)),
            missing_indices=list(range(detected, total_rounds)),
        )

        # Validate detection
        det_result = det_validator.validate(detection_report)
        assert det_result.status == ValidationStatus.PASSED

        # Simulate QBER estimation with larger test set
        observed_qber = 0.03  # 3% observed errors
        test_size = 50000  # Large test set
        key_size = 50000  # Remaining key

        # Compute adjusted QBER
        adjuster = QBERAdjuster(epsilon_sec=epsilon_sec)
        qber_result = adjuster.compute_adjusted_qber(
            observed_qber=observed_qber, key_size_n=key_size, test_size_k=test_size
        )

        # With large test set, penalty should be small enough to pass
        assert qber_result.adjusted_qber < QBER_HARD_LIMIT
        assert qber_result.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def test_full_validation_pipeline_detection_anomaly(self) -> None:
        """Pipeline should abort on detection anomaly."""
        det_validator = DetectionValidator(
            expected_detection_prob=0.015, failure_probability=1e-10
        )

        # Create anomalous report (way too many detections)
        total_rounds = 100000
        detected = 50000  # 50% detection when expecting 1.5%

        detection_report = DetectionReport(
            total_rounds=total_rounds,
            detected_indices=list(range(detected)),
            missing_indices=list(range(detected, total_rounds)),
        )

        det_result = det_validator.validate(detection_report)
        assert det_result.status == ValidationStatus.FAILED
        assert det_result.abort_code == ABORT_CODE_DETECTION_ANOMALY

    def test_full_validation_pipeline_qber_abort(self) -> None:
        """Pipeline should abort on high QBER."""
        observed_qber = 0.20  # 20% errors
        penalty_mu = 0.05  # Large penalty

        result = adjust_qber(observed_qber=observed_qber, penalty_mu=penalty_mu)

        assert result.status == ValidationStatus.FAILED
        assert result.abort_code == ABORT_CODE_QBER_HIGH

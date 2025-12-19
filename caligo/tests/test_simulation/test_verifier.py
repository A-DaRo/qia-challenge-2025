"""
Unit tests for caligo.simulation.verifier module.

Tests NSM security condition verification, QBER validation, and timing compliance.
"""

from __future__ import annotations

import pytest

from caligo.simulation.verifier import (
    NSMVerificationResult,
    verify_nsm_security_condition,
    validate_qber_measurement,
    validate_timing_compliance,
    preflight_security_check,
    postflight_security_check,
)
from caligo.simulation.physical_model import NSMParameters
from caligo.types.exceptions import SecurityError, QBERThresholdExceeded


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def secure_nsm_params() -> NSMParameters:
    """NSM parameters that should pass security checks."""
    return NSMParameters(
        storage_noise_r=0.75,  # Q_storage = 0.125
        storage_rate_nu=0.002,  # C_N * ν << 0.5
        delta_t_ns=1_000_000,
        channel_fidelity=0.95,  # Q_channel ≈ 0.025
    )


@pytest.fixture
def marginal_nsm_params() -> NSMParameters:
    """NSM parameters near the security boundary."""
    return NSMParameters(
        storage_noise_r=0.80,  # Q_storage = 0.10
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.82,  # Q_channel ≈ 0.09
    )


@pytest.fixture
def insecure_nsm_params() -> NSMParameters:
    """NSM parameters that violate security condition."""
    return NSMParameters(
        storage_noise_r=0.90,  # Q_storage = 0.05
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.85,  # Q_channel ≈ 0.075 > 0.05
    )


# =============================================================================
# NSMVerificationResult Tests
# =============================================================================


class TestNSMVerificationResult:
    """Tests for NSMVerificationResult dataclass."""

    def test_result_creation(self) -> None:
        """Result should be creatable with all fields."""
        result = NSMVerificationResult(
            is_secure=True,
            measured_qber=0.03,
            storage_noise_bound=0.125,
            security_margin=0.095,
            storage_capacity_satisfied=True,
            below_conservative_threshold=True,
            below_hard_limit=True,
            warnings=[],
        )
        assert result.is_secure
        assert result.measured_qber == 0.03
        assert result.security_margin == 0.095

    def test_result_immutable(self) -> None:
        """Result should be frozen (immutable)."""
        result = NSMVerificationResult(
            is_secure=True,
            measured_qber=0.03,
            storage_noise_bound=0.125,
            security_margin=0.095,
            storage_capacity_satisfied=True,
            below_conservative_threshold=True,
            below_hard_limit=True,
            warnings=[],
        )
        with pytest.raises(AttributeError):
            result.is_secure = False  # type: ignore


# =============================================================================
# verify_nsm_security_condition Tests
# =============================================================================


class TestVerifyNSMSecurityCondition:
    """Tests for verify_nsm_security_condition function."""

    def test_secure_parameters_pass(self, secure_nsm_params: NSMParameters) -> None:
        """Secure parameters should return is_secure=True."""
        result = verify_nsm_security_condition(
            measured_qber=0.03,
            nsm_params=secure_nsm_params,
            strict=True,
        )
        assert result.is_secure
        assert result.below_conservative_threshold
        assert result.below_hard_limit
        assert result.storage_capacity_satisfied
        assert result.security_margin > 0

    def test_marginal_parameters_pass(self, marginal_nsm_params: NSMParameters) -> None:
        """Marginal parameters should pass with warnings."""
        result = verify_nsm_security_condition(
            measured_qber=0.09,
            nsm_params=marginal_nsm_params,
            strict=True,
        )
        assert result.is_secure
        assert result.security_margin > 0
        # Should have narrow margin warning
        assert any("narrow" in w.lower() for w in result.warnings)

    def test_insecure_parameters_fail_strict(
        self, insecure_nsm_params: NSMParameters
    ) -> None:
        """Insecure parameters should raise SecurityError in strict mode."""
        with pytest.raises(SecurityError, match="NSM security violated"):
            verify_nsm_security_condition(
                measured_qber=0.075,
                nsm_params=insecure_nsm_params,
                strict=True,
            )

    def test_insecure_parameters_return_false_lenient(
        self, insecure_nsm_params: NSMParameters
    ) -> None:
        """Insecure parameters should return is_secure=False in lenient mode."""
        result = verify_nsm_security_condition(
            measured_qber=0.075,
            nsm_params=insecure_nsm_params,
            strict=False,
        )
        assert not result.is_secure
        assert result.security_margin < 0

    def test_hard_limit_exceeded_raises(self, secure_nsm_params: NSMParameters) -> None:
        """QBER >= Q_storage should raise SecurityError first."""
        # The verifier checks NSM condition first, then hard limit
        # With secure_nsm_params: Q_storage = (1-0.75)/2 = 0.125
        # QBER=0.25 > Q_storage=0.125, so SecurityError is raised first
        with pytest.raises(SecurityError, match="NSM security violated"):
            verify_nsm_security_condition(
                measured_qber=0.25,
                nsm_params=secure_nsm_params,
                strict=True,
            )

    def test_conservative_threshold_warning(
        self, secure_nsm_params: NSMParameters
    ) -> None:
        """QBER >= 0.11 but < 0.22 should warn but not fail."""
        result = verify_nsm_security_condition(
            measured_qber=0.12,
            nsm_params=secure_nsm_params,
            strict=True,
        )
        assert result.is_secure  # Still secure (below Q_storage)
        assert not result.below_conservative_threshold
        assert any("conservative" in w.lower() for w in result.warnings)


# =============================================================================
# validate_qber_measurement Tests
# =============================================================================


class TestValidateQBERMeasurement:
    """Tests for validate_qber_measurement function."""

    def test_valid_measurement(self) -> None:
        """Measurement within tolerance should return True."""
        result = validate_qber_measurement(
            measured_qber=0.031,
            expected_qber=0.030,
            tolerance=0.01,
            strict=False,
        )
        assert result

    def test_invalid_measurement(self) -> None:
        """Measurement outside tolerance should return False."""
        result = validate_qber_measurement(
            measured_qber=0.05,
            expected_qber=0.030,
            tolerance=0.01,
            strict=False,
        )
        assert not result

    def test_strict_mode_raises(self) -> None:
        """Strict mode should raise ValueError on invalid measurement."""
        with pytest.raises(ValueError, match="validation failed"):
            validate_qber_measurement(
                measured_qber=0.05,
                expected_qber=0.030,
                tolerance=0.01,
                strict=True,
            )

    def test_exact_match(self) -> None:
        """Exact match should always pass."""
        result = validate_qber_measurement(
            measured_qber=0.03,
            expected_qber=0.03,
            tolerance=0.001,
            strict=True,
        )
        assert result


# =============================================================================
# validate_timing_compliance Tests
# =============================================================================


class TestValidateTimingCompliance:
    """Tests for validate_timing_compliance function."""

    def test_sufficient_wait(self) -> None:
        """Sufficient wait time should return True."""
        result = validate_timing_compliance(
            actual_wait_ns=1_000_000,
            required_wait_ns=1_000_000,
            tolerance_fraction=0.01,
            strict=True,
        )
        assert result

    def test_wait_with_tolerance(self) -> None:
        """Wait within tolerance should return True."""
        result = validate_timing_compliance(
            actual_wait_ns=995_000,  # 0.5% below
            required_wait_ns=1_000_000,
            tolerance_fraction=0.01,
            strict=True,
        )
        assert result

    def test_insufficient_wait_strict(self) -> None:
        """Insufficient wait should raise SecurityError in strict mode."""
        with pytest.raises(SecurityError, match="Timing constraint violated"):
            validate_timing_compliance(
                actual_wait_ns=900_000,  # 10% below
                required_wait_ns=1_000_000,
                tolerance_fraction=0.01,
                strict=True,
            )

    def test_insufficient_wait_lenient(self) -> None:
        """Insufficient wait should return False in lenient mode."""
        result = validate_timing_compliance(
            actual_wait_ns=900_000,
            required_wait_ns=1_000_000,
            tolerance_fraction=0.01,
            strict=False,
        )
        assert not result


# =============================================================================
# preflight_security_check Tests
# =============================================================================


class TestPreflightSecurityCheck:
    """Tests for preflight_security_check function."""

    def test_secure_config_passes(self, secure_nsm_params: NSMParameters) -> None:
        """Secure configuration should pass preflight check."""
        result = preflight_security_check(secure_nsm_params, strict=True)
        assert result.is_secure

    def test_insecure_config_fails(self, insecure_nsm_params: NSMParameters) -> None:
        """Insecure configuration should fail preflight check."""
        with pytest.raises(SecurityError):
            preflight_security_check(insecure_nsm_params, strict=True)

    def test_preflight_uses_expected_qber(
        self, secure_nsm_params: NSMParameters
    ) -> None:
        """Preflight should use theoretical QBER from parameters."""
        result = preflight_security_check(secure_nsm_params, strict=False)
        expected_qber = secure_nsm_params.qber_channel
        assert abs(result.measured_qber - expected_qber) < 0.001


# =============================================================================
# postflight_security_check Tests
# =============================================================================


class TestPostflightSecurityCheck:
    """Tests for postflight_security_check function."""

    def test_postflight_with_measured_qber(
        self, secure_nsm_params: NSMParameters
    ) -> None:
        """Postflight should use measured QBER."""
        result = postflight_security_check(
            measured_qber=0.04,
            nsm_params=secure_nsm_params,
            timing_barrier=None,
            strict=True,
        )
        assert result.is_secure
        assert result.measured_qber == 0.04

    def test_postflight_fails_on_high_qber(
        self, secure_nsm_params: NSMParameters
    ) -> None:
        """Postflight should fail if measured QBER exceeds bound."""
        with pytest.raises(SecurityError):
            postflight_security_check(
                measured_qber=0.15,  # > Q_storage = 0.125
                nsm_params=secure_nsm_params,
                timing_barrier=None,
                strict=True,
            )

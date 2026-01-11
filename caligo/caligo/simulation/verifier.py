"""
NSM security condition verification and QBER validation.

This module provides runtime verification that NSM security conditions
are satisfied during protocol execution. It implements the validation
framework specified in nsm_parameters_enforcement.md Section 7.

Security Conditions
-------------------
The fundamental NSM security condition is:

    Q_channel < Q_storage = (1 - r) / 2

Where:
- Q_channel: Measured channel QBER (from protocol execution)
- Q_storage: Storage noise bound (from NSM parameter r)
- r: Adversary storage preservation probability

Additionally:
- C_N * ν < 1/2: Storage capacity constraint
- Q_channel < 0.11: Schaffner conservative threshold
- Q_channel < 0.22: König hard limit

References
----------
- König et al. (2012): NSM definition, 22% hard limit
- Schaffner et al. (2009), Corollary 7: 11% optimal threshold
- nsm_parameters_enforcement.md Section 7: Validation specification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from caligo.simulation.physical_model import NSMParameters
from caligo.simulation.constants import QBER_CONSERVATIVE_LIMIT, QBER_HARD_LIMIT
from caligo.types.exceptions import SecurityError, QBERThresholdExceeded
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Verification Result Dataclass
# =============================================================================


@dataclass(frozen=True)
class NSMVerificationResult:
    """
    Result of NSM security condition verification.

    Attributes
    ----------
    is_secure : bool
        True if all security conditions are satisfied.
    measured_qber : float
        Empirically measured channel QBER.
    storage_noise_bound : float
        Maximum allowed QBER for security: (1 - r) / 2.
    security_margin : float
        Difference: storage_noise_bound - measured_qber.
        Positive means secure.
    storage_capacity_satisfied : bool
        True if C_N * ν < 1/2.
    below_conservative_threshold : bool
        True if measured_qber < 0.11.
    below_hard_limit : bool
        True if measured_qber < 0.22.
    warnings : list[str]
        List of warning messages for borderline conditions.
    """

    is_secure: bool
    measured_qber: float
    storage_noise_bound: float
    security_margin: float
    storage_capacity_satisfied: bool
    below_conservative_threshold: bool
    below_hard_limit: bool
    warnings: list


# =============================================================================
# NSM Security Condition Verification
# =============================================================================


def verify_nsm_security_condition(
    measured_qber: float,
    nsm_params: NSMParameters,
    strict: bool = True,
) -> NSMVerificationResult:
    """
    Verify that measured channel QBER satisfies NSM security condition.

    The fundamental NSM condition is:
        Q_channel < Q_storage = (1 - r) / 2

    Parameters
    ----------
    measured_qber : float
        Empirically measured QBER from protocol execution.
    nsm_params : NSMParameters
        Configured NSM parameters including storage_noise_r.
    strict : bool
        If True, raises SecurityError on violation. Default: True.

    Returns
    -------
    NSMVerificationResult
        Detailed verification result with all security metrics.

    Raises
    ------
    SecurityError
        If strict=True and Q_channel >= Q_storage.
    QBERThresholdExceeded
        If strict=True and Q_channel >= hard limit (0.22).

    Notes
    -----
    The "strictly less" condition from Schaffner et al. (2009) is critical:
    "Security can be obtained as long as the quantum bit-error rate of
    the channel does not exceed 11% and the noise on the channel is
    **strictly less** than the quantum storage noise."

    References
    ----------
    - Schaffner et al. (2009), Corollary 7
    - nsm_parameters_enforcement.md Section 7.1
    """
    # Calculate storage noise bound
    r = nsm_params.storage_noise_r
    storage_noise_bound = (1.0 - r) / 2.0
    security_margin = storage_noise_bound - measured_qber

    # Check storage capacity constraint
    storage_capacity_satisfied = nsm_params.storage_security_satisfied

    # Check thresholds
    below_conservative = measured_qber < QBER_CONSERVATIVE_LIMIT
    below_hard_limit = measured_qber < QBER_HARD_LIMIT

    # Collect warnings
    warnings = []

    # Primary security condition: Q_channel < Q_storage
    nsm_condition_satisfied = measured_qber < storage_noise_bound

    if not nsm_condition_satisfied:
        msg = (
            f"NSM security violated: Q_channel={measured_qber:.4f} >= "
            f"Q_storage={storage_noise_bound:.4f} (r={r:.3f})"
        )
        warnings.append(msg)
        if strict:
            raise SecurityError(msg)

    # Hard limit check
    if not below_hard_limit:
        msg = (
            f"QBER exceeds hard limit: {measured_qber:.4f} >= "
            f"{QBER_HARD_LIMIT:.2f} (security impossible)"
        )
        warnings.append(msg)
        if strict:
            raise QBERThresholdExceeded(msg)

    # Conservative threshold warning
    if not below_conservative:
        msg = (
            f"QBER exceeds conservative threshold: {measured_qber:.4f} >= "
            f"{QBER_CONSERVATIVE_LIMIT:.2f} (reduced security margin)"
        )
        warnings.append(msg)
        logger.warning(msg)

    # Storage capacity warning
    if not storage_capacity_satisfied:
        msg = (
            f"Storage capacity constraint violated: "
            f"C_N * ν = {nsm_params.storage_capacity * nsm_params.storage_rate_nu:.4f} >= 0.5"
        )
        warnings.append(msg)
        logger.warning(msg)

    # Security margin warning
    if 0 < security_margin < 0.02:
        msg = (
            f"Security margin is narrow: {security_margin:.4f}. "
            "Protocol may fail with parameter variations."
        )
        warnings.append(msg)
        logger.warning(msg)

    # Overall security assessment
    is_secure = (
        nsm_condition_satisfied
        and below_hard_limit
        and storage_capacity_satisfied
    )

    return NSMVerificationResult(
        is_secure=is_secure,
        measured_qber=measured_qber,
        storage_noise_bound=storage_noise_bound,
        security_margin=security_margin,
        storage_capacity_satisfied=storage_capacity_satisfied,
        below_conservative_threshold=below_conservative,
        below_hard_limit=below_hard_limit,
        warnings=warnings,
    )


# =============================================================================
# QBER Measurement Validation
# =============================================================================


def validate_qber_measurement(
    measured_qber: float,
    expected_qber: float,
    tolerance: float = 0.01,
    strict: bool = False,
) -> bool:
    """
    Validate that measured QBER matches theoretical prediction.

    Compares empirically measured QBER against the expected value
    calculated from NSMParameters or ChannelNoiseProfile.

    Parameters
    ----------
    measured_qber : float
        Empirically measured QBER from protocol execution.
    expected_qber : float
        Theoretical QBER from noise model calculations.
    tolerance : float
        Absolute tolerance for comparison. Default: 0.01 (1%).
    strict : bool
        If True, raises ValueError on deviation. Default: False.

    Returns
    -------
    bool
        True if |measured - expected| <= tolerance.

    Raises
    ------
    ValueError
        If strict=True and deviation exceeds tolerance.

    Notes
    -----
    QBER deviations can occur due to:
    1. Statistical fluctuations (small sample size)
    2. Model approximations (simplified noise model)
    3. Configuration errors (wrong parameters)

    A warning is logged for any deviation > tolerance/2.

    References
    ----------
    - nsm_parameters_enforcement.md Section 7.2
    """
    deviation = abs(measured_qber - expected_qber)
    is_valid = deviation <= tolerance

    if deviation > tolerance / 2:
        logger.warning(
            f"QBER deviation: measured={measured_qber:.4f}, "
            f"expected={expected_qber:.4f}, diff={deviation:.4f}"
        )

    if not is_valid:
        msg = (
            f"QBER measurement validation failed: "
            f"measured={measured_qber:.4f}, expected={expected_qber:.4f}, "
            f"deviation={deviation:.4f} > tolerance={tolerance:.4f}"
        )
        if strict:
            raise ValueError(msg)
        logger.error(msg)

    return is_valid


# =============================================================================
# Timing Enforcement Validation
# =============================================================================


def validate_timing_compliance(
    actual_wait_ns: float,
    required_wait_ns: float,
    tolerance_fraction: float = 0.01,
    strict: bool = True,
) -> bool:
    """
    Validate that the actual wait duration meets the Δt requirement.

    Parameters
    ----------
    actual_wait_ns : float
        Actual time waited in the simulation (nanoseconds).
    required_wait_ns : float
        Required Δt from NSM parameters (nanoseconds).
    tolerance_fraction : float
        Acceptable fraction below requirement. Default: 0.01 (1%).
    strict : bool
        If True, raises SecurityError on violation. Default: True.

    Returns
    -------
    bool
        True if actual_wait >= required_wait * (1 - tolerance_fraction).

    Raises
    ------
    SecurityError
        If strict=True and wait time is insufficient.

    Notes
    -----
    The Δt wait time is critical for NSM security: it allows any
    quantum information stored by a dishonest party to decohere
    before basis revelation.

    A small tolerance (1%) accounts for discrete-event simulation
    timing precision.

    References
    ----------
    - Erven et al. (2014): "Both parties now wait a time, Δt..."
    - nsm_parameters_enforcement.md Section 7.3
    """
    min_required = required_wait_ns * (1.0 - tolerance_fraction)
    is_compliant = actual_wait_ns >= min_required

    if not is_compliant:
        msg = (
            f"Timing constraint violated: waited {actual_wait_ns:.0f} ns, "
            f"required >= {min_required:.0f} ns (Δt = {required_wait_ns:.0f} ns)"
        )
        if strict:
            raise SecurityError(msg)
        logger.error(msg)

    return is_compliant


# =============================================================================
# Pre-Flight Security Check
# =============================================================================


def preflight_security_check(
    nsm_params: NSMParameters,
    strict: bool = True,
) -> NSMVerificationResult:
    """
    Perform pre-flight security check before protocol execution.

    Validates that NSM parameters are configured for secure operation
    by checking expected QBER against security thresholds.

    Parameters
    ----------
    nsm_params : NSMParameters
        Configured NSM parameters.
    strict : bool
        If True, raises on security violations. Default: True.

    Returns
    -------
    NSMVerificationResult
        Verification result using expected (not measured) QBER.

    Notes
    -----
    This check uses the theoretical QBER calculated from parameters,
    not an empirically measured value. It should be called before
    starting protocol execution to fail fast on misconfiguration.
    """
    expected_qber = nsm_params.qber_channel

    logger.debug(
        f"Pre-flight security check: expected QBER={expected_qber:.4f}, "
        f"r={nsm_params.storage_noise_r:.3f}, ν={nsm_params.storage_rate_nu:.4f}"
    )

    return verify_nsm_security_condition(
        measured_qber=expected_qber,
        nsm_params=nsm_params,
        strict=strict,
    )


# =============================================================================
# Post-Protocol Security Check
# =============================================================================


def postflight_security_check(
    measured_qber: float,
    nsm_params: NSMParameters,
    timing_barrier: Optional[object] = None,
    strict: bool = True,
) -> NSMVerificationResult:
    """
    Perform post-flight security check after protocol execution.

    Validates that actual measurements satisfy NSM security conditions.

    Parameters
    ----------
    measured_qber : float
        Empirically measured QBER from protocol.
    nsm_params : NSMParameters
        Configured NSM parameters.
    timing_barrier : TimingBarrier, optional
        TimingBarrier instance for timing compliance check.
    strict : bool
        If True, raises on security violations. Default: True.

    Returns
    -------
    NSMVerificationResult
        Verification result with measured QBER.
    """
    # Verify NSM condition
    result = verify_nsm_security_condition(
        measured_qber=measured_qber,
        nsm_params=nsm_params,
        strict=strict,
    )

    # Validate QBER against expected
    expected_qber = nsm_params.qber_channel
    validate_qber_measurement(
        measured_qber=measured_qber,
        expected_qber=expected_qber,
        tolerance=0.02,  # Allow 2% deviation for statistical noise
        strict=False,  # Warning only
    )

    # Check timing compliance if barrier provided
    if timing_barrier is not None:
        actual_wait = getattr(timing_barrier, "actual_wait_duration_ns", None)
        if actual_wait is not None:
            validate_timing_compliance(
                actual_wait_ns=actual_wait,
                required_wait_ns=nsm_params.delta_t_ns,
                strict=strict,
            )

    logger.debug(
        f"Post-flight security check: measured QBER={measured_qber:.4f}, "
        f"security_margin={result.security_margin:.4f}, secure={result.is_secure}"
    )

    return result

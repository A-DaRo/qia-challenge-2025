"""
Pre-flight security feasibility validation for E-HOK protocol.

This module validates whether the E-HOK protocol can produce a
secure oblivious key given the physical parameters. All checks
MUST pass before protocol execution proceeds.

Key Validations
---------------
1. **QBER threshold**: Must be below 22% hard limit (11% conservative)
2. **Storage capacity constraint**: C_N · ν < 1/2
3. **"Strictly less" condition**: h(P_error) < h_min(r)
4. **Batch size feasibility**: Expected key length > 0

References
----------
- König et al. (2012): Storage capacity constraint
- Schaffner et al. (2009): 11% threshold, Corollary 7
- Lupo et al. (2023): 22% absolute limit
- Erven et al. (2014): Experimental validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from caligo.security.bounds import (
    QBER_CONSERVATIVE_THRESHOLD,
    QBER_ABSOLUTE_THRESHOLD,
    R_TILDE,
    DEFAULT_EPSILON_SEC,
    max_bound_entropy,
    bounded_storage_entropy,
    compute_extractable_key_rate,
)
from caligo.types.exceptions import (
    InvalidParameterError,
    QBERThresholdExceeded,
    NSMViolationError,
    FeasibilityError,
)
from caligo.utils.math import binary_entropy

if TYPE_CHECKING:
    from caligo.simulation.physical_model import NSMParameters


# =============================================================================
# Result Types
# =============================================================================


class FeasibilityStatus(Enum):
    """Status of a feasibility check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True)
class FeasibilityResult:
    """
    Result of a single feasibility check.

    Parameters
    ----------
    status : FeasibilityStatus
        Whether the check passed, warned, or failed.
    check_name : str
        Name of the check performed.
    is_feasible : bool
        True if the check passed (security may still be possible).
    value : float
        The computed value being checked.
    threshold : float
        The threshold against which value was compared.
    margin : float
        Distance from threshold (positive = safe, negative = violated).
    message : str
        Human-readable diagnostic message.
    """

    status: FeasibilityStatus
    check_name: str
    is_feasible: bool
    value: float
    threshold: float
    margin: float
    message: str


@dataclass
class PreflightReport:
    """
    Comprehensive report from all pre-flight security checks.

    This report aggregates results from all feasibility checks and
    provides recommendations for protocol execution.

    Parameters
    ----------
    is_feasible : bool
        True if all checks passed (protocol may proceed).
    results : list[FeasibilityResult]
        Individual results from each check.
    expected_key_rate : float
        Expected extractable key rate per raw bit.
    min_batch_size : int
        Minimum recommended batch size for positive key length.
    security_margin : float
        Overall margin from security boundaries.
    warnings : list[str]
        Warning messages for near-threshold conditions.
    """

    is_feasible: bool
    results: list[FeasibilityResult] = field(default_factory=list)
    expected_key_rate: float = 0.0
    min_batch_size: int = 0
    security_margin: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return human-readable summary of preflight report."""
        lines = [
            "=" * 60,
            "PREFLIGHT SECURITY REPORT",
            "=" * 60,
            f"Feasible: {'YES' if self.is_feasible else 'NO'}",
            f"Expected Key Rate: {self.expected_key_rate:.4f} bits/qubit",
            f"Min Batch Size: {self.min_batch_size:,}",
            f"Security Margin: {self.security_margin:.4f}",
            "-" * 60,
            "Check Results:",
        ]
        for result in self.results:
            status_icon = {
                FeasibilityStatus.PASSED: "✓",
                FeasibilityStatus.WARNING: "⚠",
                FeasibilityStatus.FAILED: "✗",
            }[result.status]
            lines.append(f"  {status_icon} {result.check_name}: {result.message}")

        if self.warnings:
            lines.append("-" * 60)
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_expected_qber(
    channel_fidelity: float,
    detection_efficiency: float = 1.0,
    dark_count_prob: float = 0.0,
    intrinsic_error: float = 0.0,
) -> float:
    """
    Compute expected QBER from physical device parameters.

    The QBER arises from multiple sources:
    1. EPR pair infidelity: (1 - F)/2 contribution
    2. Detector dark counts: contribution proportional to dark count rate
    3. Intrinsic detection error: e_det direct contribution

    Parameters
    ----------
    channel_fidelity : float
        EPR pair fidelity F ∈ [0.5, 1].
    detection_efficiency : float
        Total detection efficiency η ∈ (0, 1]. Default: 1.0.
    dark_count_prob : float
        Dark count probability per detection window. Default: 0.0.
    intrinsic_error : float
        Intrinsic detector error rate e_det. Default: 0.0.

    Returns
    -------
    float
        Expected QBER ∈ [0, 0.5].

    Raises
    ------
    InvalidParameterError
        If parameters are out of valid range.

    References
    ----------
    - Erven et al. (2014), Table I

    Examples
    --------
    >>> compute_expected_qber(0.99)  # High fidelity
    0.005
    >>> compute_expected_qber(0.95, intrinsic_error=0.01)
    0.035
    """
    if not 0.5 <= channel_fidelity <= 1.0:
        raise InvalidParameterError(
            f"channel_fidelity={channel_fidelity} must be in [0.5, 1]"
        )
    if not 0 < detection_efficiency <= 1.0:
        raise InvalidParameterError(
            f"detection_efficiency={detection_efficiency} must be in (0, 1]"
        )
    if not 0 <= dark_count_prob <= 1.0:
        raise InvalidParameterError(
            f"dark_count_prob={dark_count_prob} must be in [0, 1]"
        )
    if not 0 <= intrinsic_error <= 0.5:
        raise InvalidParameterError(
            f"intrinsic_error={intrinsic_error} must be in [0, 0.5]"
        )

    # QBER from channel infidelity
    qber_infidelity = (1.0 - channel_fidelity) / 2.0

    # QBER from dark counts (simplified model)
    # Dark counts contribute errors when they cause false detections
    qber_dark = dark_count_prob * 0.5  # Random errors from dark counts

    # Total QBER
    qber_total = qber_infidelity + intrinsic_error + qber_dark

    # Clamp to valid range
    return min(0.5, qber_total)


def compute_storage_capacity(r: float) -> float:
    """
    Compute classical capacity of depolarizing storage channel.

    For depolarizing channel with parameter r:
        C_N = 1 - h((1+r)/2)

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].

    Returns
    -------
    float
        Channel capacity C_N ∈ [0, 1].

    Examples
    --------
    >>> compute_storage_capacity(1.0)  # Perfect storage
    1.0
    >>> compute_storage_capacity(0.0)  # Complete depolarization
    0.0
    """
    if r == 0 or r == 1:
        return r  # Edge cases: C(0) = 0, C(1) = 1

    p = (1.0 + r) / 2.0
    return 1.0 - binary_entropy(p)


# =============================================================================
# FeasibilityChecker Class
# =============================================================================


class FeasibilityChecker:
    """
    Pre-flight security feasibility validation.

    This class validates whether the E-HOK protocol can produce a
    secure oblivious key given the physical parameters. All checks
    MUST pass before protocol execution.

    Parameters
    ----------
    storage_noise_r : float
        Adversary's storage noise parameter ∈ [0, 1].
    storage_rate_nu : float
        Adversary's storage rate (fraction of qubits storable) ∈ [0, 1].
    expected_qber : float
        Expected quantum bit error rate from honest devices/channel.
    security_parameter : float
        Target security parameter ε_sec. Default: 1e-10.

    Attributes
    ----------
    storage_noise_r : float
        Storage noise parameter.
    storage_rate_nu : float
        Storage rate parameter.
    expected_qber : float
        Expected QBER.
    security_parameter : float
        Security parameter.

    References
    ----------
    - Schaffner et al. (2009), Corollary 7
    - Lupo et al. (2023), Section VI
    - König et al. (2012), Theorem I.1

    Examples
    --------
    >>> checker = FeasibilityChecker(
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ...     expected_qber=0.02,
    ... )
    >>> report = checker.run_preflight_checks()
    >>> report.is_feasible
    True
    """

    def __init__(
        self,
        storage_noise_r: float,
        storage_rate_nu: float,
        expected_qber: float,
        security_parameter: float = DEFAULT_EPSILON_SEC,
    ) -> None:
        """Initialize FeasibilityChecker with protocol parameters."""
        # Validate parameters
        if not 0 <= storage_noise_r <= 1:
            raise InvalidParameterError(
                f"storage_noise_r={storage_noise_r} must be in [0, 1]"
            )
        if not 0 <= storage_rate_nu <= 1:
            raise InvalidParameterError(
                f"storage_rate_nu={storage_rate_nu} must be in [0, 1]"
            )
        if not 0 <= expected_qber <= 0.5:
            raise InvalidParameterError(
                f"expected_qber={expected_qber} must be in [0, 0.5]"
            )
        if not 0 < security_parameter < 1:
            raise InvalidParameterError(
                f"security_parameter={security_parameter} must be in (0, 1)"
            )

        self.storage_noise_r = storage_noise_r
        self.storage_rate_nu = storage_rate_nu
        self.expected_qber = expected_qber
        self.security_parameter = security_parameter

    @classmethod
    def from_nsm_parameters(
        cls,
        params: "NSMParameters",
        security_parameter: float = DEFAULT_EPSILON_SEC,
    ) -> "FeasibilityChecker":
        """
        Create FeasibilityChecker from NSMParameters.

        Parameters
        ----------
        params : NSMParameters
            Physical parameters from simulation layer.
        security_parameter : float
            Target security parameter.

        Returns
        -------
        FeasibilityChecker
            Configured checker instance.
        """
        expected_qber = compute_expected_qber(
            channel_fidelity=params.channel_fidelity,
            detection_efficiency=params.detection_eff_eta,
            dark_count_prob=params.dark_count_prob,
            intrinsic_error=params.detector_error,
        )
        return cls(
            storage_noise_r=params.storage_noise_r,
            storage_rate_nu=params.storage_rate_nu,
            expected_qber=expected_qber,
            security_parameter=security_parameter,
        )

    def check_qber_threshold(
        self,
        qber: Optional[float] = None,
        raise_on_failure: bool = False,
    ) -> FeasibilityResult:
        """
        Check if QBER is below security thresholds.

        Parameters
        ----------
        qber : float, optional
            QBER to check. If None, uses expected_qber.
        raise_on_failure : bool
            If True, raise exception on failure.

        Returns
        -------
        FeasibilityResult
            Check result with status and diagnostics.

        Raises
        ------
        QBERThresholdExceeded
            If raise_on_failure=True and QBER exceeds absolute threshold.

        References
        ----------
        - Schaffner et al. (2009), Corollary 7: 11% conservative
        - Lupo et al. (2023), Section VI: 22% absolute
        """
        qber = qber if qber is not None else self.expected_qber

        if qber > QBER_ABSOLUTE_THRESHOLD:
            result = FeasibilityResult(
                status=FeasibilityStatus.FAILED,
                check_name="QBER Threshold",
                is_feasible=False,
                value=qber,
                threshold=QBER_ABSOLUTE_THRESHOLD,
                margin=QBER_ABSOLUTE_THRESHOLD - qber,
                message=f"QBER {qber:.2%} exceeds absolute limit {QBER_ABSOLUTE_THRESHOLD:.0%}",
            )
            if raise_on_failure:
                raise QBERThresholdExceeded(result.message)
            return result

        if qber > QBER_CONSERVATIVE_THRESHOLD:
            return FeasibilityResult(
                status=FeasibilityStatus.WARNING,
                check_name="QBER Threshold",
                is_feasible=True,
                value=qber,
                threshold=QBER_CONSERVATIVE_THRESHOLD,
                margin=QBER_CONSERVATIVE_THRESHOLD - qber,
                message=f"QBER {qber:.2%} exceeds conservative limit {QBER_CONSERVATIVE_THRESHOLD:.0%}",
            )

        return FeasibilityResult(
            status=FeasibilityStatus.PASSED,
            check_name="QBER Threshold",
            is_feasible=True,
            value=qber,
            threshold=QBER_CONSERVATIVE_THRESHOLD,
            margin=QBER_CONSERVATIVE_THRESHOLD - qber,
            message=f"QBER {qber:.2%} within safe limits",
        )

    def check_storage_capacity_constraint(
        self,
        raise_on_failure: bool = False,
    ) -> FeasibilityResult:
        """
        Verify the noisy storage capacity constraint.

        For security to hold, the classical capacity of the adversary's
        storage channel multiplied by storage rate must satisfy:

            C_N · ν < 1/2

        Parameters
        ----------
        raise_on_failure : bool
            If True, raise exception on failure.

        Returns
        -------
        FeasibilityResult
            Check result with status and diagnostics.

        Raises
        ------
        NSMViolationError
            If raise_on_failure=True and constraint is violated.

        References
        ----------
        - König et al. (2012), Corollary I.2
        """
        capacity = compute_storage_capacity(self.storage_noise_r)
        product = capacity * self.storage_rate_nu
        threshold = 0.5
        margin = threshold - product

        if product >= threshold:
            result = FeasibilityResult(
                status=FeasibilityStatus.FAILED,
                check_name="Storage Capacity",
                is_feasible=False,
                value=product,
                threshold=threshold,
                margin=margin,
                message=f"C_N·ν = {product:.4f} ≥ 0.5 violates NSM constraint",
            )
            if raise_on_failure:
                raise NSMViolationError(result.message)
            return result

        # Warning if close to threshold
        if margin < 0.1:
            return FeasibilityResult(
                status=FeasibilityStatus.WARNING,
                check_name="Storage Capacity",
                is_feasible=True,
                value=product,
                threshold=threshold,
                margin=margin,
                message=f"C_N·ν = {product:.4f} close to 0.5 limit (margin: {margin:.4f})",
            )

        return FeasibilityResult(
            status=FeasibilityStatus.PASSED,
            check_name="Storage Capacity",
            is_feasible=True,
            value=product,
            threshold=threshold,
            margin=margin,
            message=f"C_N·ν = {product:.4f} < 0.5 satisfied",
        )

    def check_strictly_less_condition(
        self,
        raise_on_failure: bool = False,
    ) -> FeasibilityResult:
        """
        Verify the fundamental NSM "strictly less" condition.

        Security requires that information leaked via error correction
        is strictly less than the min-entropy from storage decoherence:

            h(P_error) < h_min(r)

        For depolarizing storage (Schaffner Corollary 7):
        - If r ≥ r̃ ≈ 0.78: h((1+r)/2) > h(P_error)
        - If r < r̃:         1/2 > h(P_error) → P_error < 11%

        Parameters
        ----------
        raise_on_failure : bool
            If True, raise exception on failure.

        Returns
        -------
        FeasibilityResult
            Check result with status and diagnostics.

        Raises
        ------
        NSMViolationError
            If raise_on_failure=True and condition is violated.

        References
        ----------
        - Schaffner et al. (2009), Corollary 7
        - Lupo et al. (2023), Section VI
        """
        h_error = binary_entropy(self.expected_qber) if self.expected_qber > 0 else 0.0
        h_min = max_bound_entropy(self.storage_noise_r)
        margin = h_min - h_error

        if margin <= 0:
            result = FeasibilityResult(
                status=FeasibilityStatus.FAILED,
                check_name="Strictly Less",
                is_feasible=False,
                value=h_error,
                threshold=h_min,
                margin=margin,
                message=f"h(QBER)={h_error:.4f} ≥ h_min(r)={h_min:.4f} violates condition",
            )
            if raise_on_failure:
                raise NSMViolationError(result.message)
            return result

        # Warning if margin is small
        if margin < 0.05:
            return FeasibilityResult(
                status=FeasibilityStatus.WARNING,
                check_name="Strictly Less",
                is_feasible=True,
                value=h_error,
                threshold=h_min,
                margin=margin,
                message=f"h(QBER)={h_error:.4f} < h_min(r)={h_min:.4f} (tight margin: {margin:.4f})",
            )

        return FeasibilityResult(
            status=FeasibilityStatus.PASSED,
            check_name="Strictly Less",
            is_feasible=True,
            value=h_error,
            threshold=h_min,
            margin=margin,
            message=f"h(QBER)={h_error:.4f} < h_min(r)={h_min:.4f} satisfied",
        )

    def check_batch_size_feasibility(
        self,
        n_raw_bits: int,
        ec_efficiency: float = 1.16,
        raise_on_failure: bool = False,
    ) -> FeasibilityResult:
        """
        Check if batch size yields positive extractable key length.

        Computes the expected final key length using the E-HOK formula:

            ℓ = n · h_min(r,ν) - n · f · h(Q) - 2·log₂(1/ε_sec)

        Parameters
        ----------
        n_raw_bits : int
            Number of raw bits after sifting.
        ec_efficiency : float
            Error correction efficiency factor f. Default: 1.16.
        raise_on_failure : bool
            If True, raise exception on failure.

        Returns
        -------
        FeasibilityResult
            Check result with expected key length.

        Raises
        ------
        FeasibilityError
            If raise_on_failure=True and expected key length ≤ 0.

        References
        ----------
        - Erven et al. (2014), Eq. (8)
        - Lupo et al. (2023), Eq. (43)
        """
        import math

        key_rate = compute_extractable_key_rate(
            r=self.storage_noise_r,
            nu=self.storage_rate_nu,
            qber=self.expected_qber,
            ec_efficiency=ec_efficiency,
        )

        # Finite-key security cost
        security_cost = 2.0 * math.log2(1.0 / self.security_parameter)

        # Expected key length
        expected_length = n_raw_bits * key_rate - security_cost

        if expected_length <= 0:
            result = FeasibilityResult(
                status=FeasibilityStatus.FAILED,
                check_name="Batch Size",
                is_feasible=False,
                value=expected_length,
                threshold=0.0,
                margin=expected_length,
                message=f"Expected key length ℓ={expected_length:.0f} ≤ 0 (need more qubits)",
            )
            if raise_on_failure:
                raise FeasibilityError(result.message)
            return result

        # Warning if key rate is marginal
        if key_rate < 0.01:
            return FeasibilityResult(
                status=FeasibilityStatus.WARNING,
                check_name="Batch Size",
                is_feasible=True,
                value=expected_length,
                threshold=0.0,
                margin=expected_length,
                message=f"Expected key length ℓ≈{expected_length:.0f} (marginal rate: {key_rate:.4f})",
            )

        return FeasibilityResult(
            status=FeasibilityStatus.PASSED,
            check_name="Batch Size",
            is_feasible=True,
            value=expected_length,
            threshold=0.0,
            margin=expected_length,
            message=f"Expected key length ℓ≈{expected_length:.0f} (rate: {key_rate:.4f})",
        )

    def compute_min_batch_size(
        self,
        ec_efficiency: float = 1.16,
        target_key_length: int = 128,
    ) -> int:
        """
        Compute minimum batch size for target key length.

        Parameters
        ----------
        ec_efficiency : float
            Error correction efficiency factor.
        target_key_length : int
            Desired final key length in bits.

        Returns
        -------
        int
            Minimum required raw bits after sifting.
        """
        import math

        key_rate = compute_extractable_key_rate(
            r=self.storage_noise_r,
            nu=self.storage_rate_nu,
            qber=self.expected_qber,
            ec_efficiency=ec_efficiency,
        )

        if key_rate <= 0:
            return float("inf")  # type: ignore

        security_cost = 2.0 * math.log2(1.0 / self.security_parameter)
        min_n = math.ceil((target_key_length + security_cost) / key_rate)

        return max(1, min_n)

    def run_preflight_checks(
        self,
        n_raw_bits: Optional[int] = None,
        ec_efficiency: float = 1.16,
        raise_on_failure: bool = True,
    ) -> PreflightReport:
        """
        Execute all feasibility checks before protocol execution.

        This is the main entry point that validates whether the protocol
        can succeed given the provided configuration. All checks must pass
        for the protocol to proceed.

        Checks Performed (in order):
        1. QBER threshold (conservative 11%, hard limit 22%)
        2. Storage capacity constraint (C_N · ν < 1/2)
        3. "Strictly less" condition (h(P_error) < h_min(r))
        4. Batch size feasibility (expected ℓ > 0) [if n_raw_bits provided]

        Parameters
        ----------
        n_raw_bits : int, optional
            Number of raw bits for batch size check. If None, skipped.
        ec_efficiency : float
            Error correction efficiency factor.
        raise_on_failure : bool
            If True, raise exception on first failure.

        Returns
        -------
        PreflightReport
            Comprehensive report with all check results and recommendations.

        Raises
        ------
        QBERThresholdExceeded
            If QBER exceeds absolute threshold.
        NSMViolationError
            If NSM constraints are violated.
        FeasibilityError
            If batch size yields no key.
        """
        results: list[FeasibilityResult] = []
        warnings: list[str] = []
        is_feasible = True

        # Check 1: QBER threshold
        qber_result = self.check_qber_threshold(raise_on_failure=raise_on_failure)
        results.append(qber_result)
        if qber_result.status == FeasibilityStatus.FAILED:
            is_feasible = False
        elif qber_result.status == FeasibilityStatus.WARNING:
            warnings.append(qber_result.message)

        # Check 2: Storage capacity constraint
        capacity_result = self.check_storage_capacity_constraint(
            raise_on_failure=raise_on_failure
        )
        results.append(capacity_result)
        if capacity_result.status == FeasibilityStatus.FAILED:
            is_feasible = False
        elif capacity_result.status == FeasibilityStatus.WARNING:
            warnings.append(capacity_result.message)

        # Check 3: Strictly less condition
        strictly_result = self.check_strictly_less_condition(
            raise_on_failure=raise_on_failure
        )
        results.append(strictly_result)
        if strictly_result.status == FeasibilityStatus.FAILED:
            is_feasible = False
        elif strictly_result.status == FeasibilityStatus.WARNING:
            warnings.append(strictly_result.message)

        # Check 4: Batch size (if provided)
        if n_raw_bits is not None:
            batch_result = self.check_batch_size_feasibility(
                n_raw_bits=n_raw_bits,
                ec_efficiency=ec_efficiency,
                raise_on_failure=raise_on_failure,
            )
            results.append(batch_result)
            if batch_result.status == FeasibilityStatus.FAILED:
                is_feasible = False
            elif batch_result.status == FeasibilityStatus.WARNING:
                warnings.append(batch_result.message)

        # Compute expected key rate and min batch size
        key_rate = compute_extractable_key_rate(
            r=self.storage_noise_r,
            nu=self.storage_rate_nu,
            qber=self.expected_qber,
            ec_efficiency=ec_efficiency,
        )
        min_batch = self.compute_min_batch_size(ec_efficiency=ec_efficiency)

        # Compute overall security margin (minimum of all margins)
        security_margin = min(r.margin for r in results)

        return PreflightReport(
            is_feasible=is_feasible,
            results=results,
            expected_key_rate=key_rate,
            min_batch_size=min_batch if min_batch != float("inf") else 0,
            security_margin=security_margin,
            warnings=warnings,
        )

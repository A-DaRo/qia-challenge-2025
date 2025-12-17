"""
Statistical Validation for E-HOK Protocol.

This module implements statistical guardrails for Phase II of the E-HOK
protocol, including detection validation (Chernoff/Hoeffding bounds),
finite-size penalty calculation, and QBER adjustment with abort logic.

Components
----------
1. DetectionValidator (TASK-DETECT-VALID-001): Validates Bob's detection
   report against expected channel transmittance using Hoeffding bounds.

2. FiniteSizePenalty (TASK-FINITE-SIZE-001): Computes the statistical
   penalty μ bridging observed QBER to worst-case bounds.

3. QBERAdjuster (TASK-QBER-ADJUST-001): Applies penalty to observed QBER
   and enforces abort/warning thresholds.

Security Rationale
------------------
- Detection validation prevents post-selection attacks where Bob claims
  "missing" strategically on rounds where his storage failed.
- Finite-size penalty ensures composable security by accounting for
  statistical fluctuations in small samples.
- QBER thresholds (11% warning, 22% hard limit) derive from NSM
  information-theoretic bounds.

References
----------
- Schaffner et al. (2009): Chernoff validation, 11% conservative bound
- Erven et al. (2014): Penalty term μ, experimental parameters
- Lupo et al. (2023): 22% hard limit derivation
- sprint_2_specification.md Sections 3.1-3.3
- Tight Finite-Key Analysis (Scarani et al.): Eq. (2) for μ formula
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ehok.analysis.nsm_bounds import QBER_HARD_LIMIT, QBER_WARNING_THRESHOLD
from ehok.protocols.ordered_messaging import DetectionReport
from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Abort Codes (Phase II)
# =============================================================================

ABORT_CODE_DETECTION_ANOMALY = "ABORT-II-DET-001"
ABORT_CODE_QBER_HIGH = "ABORT-II-QBER-001"


# =============================================================================
# Validation Results
# =============================================================================


class ValidationStatus(Enum):
    """
    Result status for statistical validation checks.

    Attributes
    ----------
    PASSED : auto
        Validation passed, no issues.
    WARNING : auto
        Validation passed but with warning-level concern.
    FAILED : auto
        Validation failed, abort required.
    INVALID_INPUT : auto
        Input parameters are invalid.
    """

    PASSED = auto()
    WARNING = auto()
    FAILED = auto()
    INVALID_INPUT = auto()


@dataclass(frozen=True)
class DetectionValidationResult:
    """
    Result of detection report validation.

    Attributes
    ----------
    status : ValidationStatus
        Overall validation status.
    observed_s : int
        Number of detected rounds (S).
    total_rounds : int
        Total rounds (M).
    expected_p : float
        Expected detection probability.
    tolerance_zeta : float
        Computed Hoeffding tolerance ζ.
    lower_bound : float
        Lower acceptance bound.
    upper_bound : float
        Upper acceptance bound.
    message : str
        Human-readable diagnostic message.
    abort_code : Optional[str]
        Abort code if validation failed.
    """

    status: ValidationStatus
    observed_s: int
    total_rounds: int
    expected_p: float
    tolerance_zeta: float
    lower_bound: float
    upper_bound: float
    message: str
    abort_code: Optional[str] = None


@dataclass(frozen=True)
class FiniteSizePenaltyResult:
    """
    Result of finite-size penalty calculation.

    Attributes
    ----------
    mu : float
        The computed penalty value.
    test_size_k : int
        Test set size used.
    key_size_n : int
        Key size used.
    epsilon_sec : float
        Security parameter used.
    """

    mu: float
    test_size_k: int
    key_size_n: int
    epsilon_sec: float


@dataclass(frozen=True)
class QBERAdjustmentResult:
    """
    Result of QBER adjustment and threshold check.

    Attributes
    ----------
    status : ValidationStatus
        PASSED, WARNING (11-22%), or FAILED (>22%).
    observed_qber : float
        Raw observed QBER.
    penalty_mu : float
        Finite-size penalty applied.
    adjusted_qber : float
        observed_qber + penalty_mu.
    warning_threshold : float
        The 11% warning threshold.
    hard_limit : float
        The 22% hard abort limit.
    message : str
        Diagnostic message.
    abort_code : Optional[str]
        Abort code if threshold exceeded.
    """

    status: ValidationStatus
    observed_qber: float
    penalty_mu: float
    adjusted_qber: float
    warning_threshold: float
    hard_limit: float
    message: str
    abort_code: Optional[str] = None


# =============================================================================
# Detection Validator (TASK-DETECT-VALID-001)
# =============================================================================


class DetectionValidator:
    """
    Validates detection reports using Hoeffding/Chernoff bounds.

    This class implements the statistical test to detect post-selection
    attacks where Bob claims "missing rounds" strategically.

    The test validates:
        Pr[|S - P_expected·M| ≥ ζ·M] < ε

    with tolerance:
        ζ = √(ln(2/ε) / (2M))

    Attributes
    ----------
    expected_detection_prob : float
        Expected detection probability P_expected from channel calibration.
    failure_probability : float
        Failure probability budget ε for the test.

    References
    ----------
    - sprint_2_specification.md Section 3.1
    - Phase II analysis Section 2.2: Missing Rounds Constraint
    - Erven et al. (2014): Hoeffding inequality application

    Examples
    --------
    >>> validator = DetectionValidator(expected_detection_prob=0.5, failure_probability=1e-10)
    >>> validator.expected_detection_prob
    0.5
    >>> validator.failure_probability
    1e-10
    """

    def __init__(
        self, expected_detection_prob: float, failure_probability: float = 1e-10
    ) -> None:
        """
        Initialize detection validator.

        Parameters
        ----------
        expected_detection_prob : float
            Expected detection probability P_expected. Must be in (0, 1).
        failure_probability : float
            Failure probability budget ε for Hoeffding test.
            Must be in (0, 1). Default: 1e-10.

        Raises
        ------
        ValueError
            If parameters are out of valid ranges.
        """
        if not 0 < expected_detection_prob < 1:
            raise ValueError(
                f"expected_detection_prob must be in (0, 1), got {expected_detection_prob}"
            )
        if not 0 < failure_probability < 1:
            raise ValueError(
                f"failure_probability must be in (0, 1), got {failure_probability}"
            )

        self._expected_p = expected_detection_prob
        self._epsilon = failure_probability

        logger.debug(
            "DetectionValidator initialized: P_expected=%.6f, ε=%.2e",
            self._expected_p,
            self._epsilon,
        )

    @property
    def expected_detection_prob(self) -> float:
        """Get expected detection probability."""
        return self._expected_p

    @property
    def failure_probability(self) -> float:
        """Get failure probability budget."""
        return self._epsilon

    def compute_tolerance(self, total_rounds: int) -> float:
        """
        Compute Hoeffding tolerance ζ for given sample size.

        Formula: ζ = √(ln(2/ε) / (2M))

        Parameters
        ----------
        total_rounds : int
            Total number of rounds M.

        Returns
        -------
        float
            Tolerance value ζ.
        """
        if total_rounds <= 0:
            raise ValueError(f"total_rounds must be positive, got {total_rounds}")

        return math.sqrt(math.log(2.0 / self._epsilon) / (2.0 * total_rounds))

    def compute_acceptance_interval(
        self, total_rounds: int
    ) -> tuple[float, float]:
        """
        Compute acceptance interval for detection count.

        Parameters
        ----------
        total_rounds : int
            Total number of rounds M.

        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound) for accepted detection counts.
        """
        zeta = self.compute_tolerance(total_rounds)
        lower = (self._expected_p - zeta) * total_rounds
        upper = (self._expected_p + zeta) * total_rounds
        return (lower, upper)

    def validate(self, report: DetectionReport) -> DetectionValidationResult:
        """
        Validate a detection report against statistical bounds.

        Parameters
        ----------
        report : DetectionReport
            Bob's detection report to validate.

        Returns
        -------
        DetectionValidationResult
            Validation result with diagnostics.
        """
        m = report.total_rounds
        s = report.num_detected

        if m <= 0:
            return DetectionValidationResult(
                status=ValidationStatus.INVALID_INPUT,
                observed_s=s,
                total_rounds=m,
                expected_p=self._expected_p,
                tolerance_zeta=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                message=f"Invalid total_rounds: {m}",
                abort_code=ABORT_CODE_DETECTION_ANOMALY,
            )

        zeta = self.compute_tolerance(m)
        lower, upper = self.compute_acceptance_interval(m)

        # Check if S is within acceptance interval
        if lower <= s <= upper:
            status = ValidationStatus.PASSED
            message = (
                f"Detection validation PASSED: S={s} in [{lower:.1f}, {upper:.1f}] "
                f"(M={m}, P_exp={self._expected_p:.4f}, ζ={zeta:.6f})"
            )
            abort_code = None
            logger.info(
                "DETECTION_VALIDATION_PASSED S=%d M=%d interval=[%.1f, %.1f]",
                s,
                m,
                lower,
                upper,
            )
        else:
            status = ValidationStatus.FAILED
            message = (
                f"Detection validation FAILED: S={s} outside [{lower:.1f}, {upper:.1f}] "
                f"(M={m}, P_exp={self._expected_p:.4f}, ζ={zeta:.6f}). "
                f"Possible post-selection attack detected."
            )
            abort_code = ABORT_CODE_DETECTION_ANOMALY
            logger.warning(
                "DETECTION_VALIDATION_FAILED S=%d M=%d interval=[%.1f, %.1f] abort_code=%s",
                s,
                m,
                lower,
                upper,
                abort_code,
            )

        return DetectionValidationResult(
            status=status,
            observed_s=s,
            total_rounds=m,
            expected_p=self._expected_p,
            tolerance_zeta=zeta,
            lower_bound=lower,
            upper_bound=upper,
            message=message,
            abort_code=abort_code,
        )


# =============================================================================
# Finite-Size Penalty Calculator (TASK-FINITE-SIZE-001)
# =============================================================================


def compute_finite_size_penalty(
    key_size_n: int, test_size_k: int, epsilon_sec: float
) -> FiniteSizePenaltyResult:
    """
    Compute finite-size statistical penalty μ.

    This penalty bridges observed test-set QBER to conservative
    worst-case bounds for security calculations.

    Formula (Scarani et al. Eq. 2):
        μ = √((n+k)/(nk) · (k+1)/k) · ln(4/ε_sec)

    Parameters
    ----------
    key_size_n : int
        Remaining key size after test set removal.
    test_size_k : int
        Test set size.
    epsilon_sec : float
        Security parameter ε_sec ∈ (0, 1).

    Returns
    -------
    FiniteSizePenaltyResult
        Result containing computed μ and input parameters.

    Raises
    ------
    ValueError
        If parameters are invalid (n ≤ 0, k ≤ 0, ε_sec ∉ (0,1)).

    References
    ----------
    - sprint_2_specification.md Section 3.2
    - Tight Finite-Key Analysis Eq. (2)
    - Phase II analysis Section 2.3

    Examples
    --------
    >>> result = compute_finite_size_penalty(key_size_n=90000, test_size_k=10000, epsilon_sec=1e-10)
    >>> result.mu > 0
    True
    >>> result.key_size_n
    90000
    """
    # Validate inputs
    if key_size_n <= 0:
        raise ValueError(f"key_size_n must be positive, got {key_size_n}")
    if test_size_k <= 0:
        raise ValueError(f"test_size_k must be positive, got {test_size_k}")
    if not 0 < epsilon_sec < 1:
        raise ValueError(f"epsilon_sec must be in (0, 1), got {epsilon_sec}")

    # Compute penalty per Scarani formula
    # μ = √((n+k)/(nk) · (k+1)/k) · ln(4/ε_sec)
    n = key_size_n
    k = test_size_k

    term1 = (n + k) / (n * k)
    term2 = (k + 1) / k
    ln_term = math.log(4.0 / epsilon_sec)

    mu = math.sqrt(term1 * term2) * ln_term

    logger.debug(
        "FINITE_SIZE_PENALTY computed: μ=%.6f (n=%d, k=%d, ε_sec=%.2e)",
        mu,
        n,
        k,
        epsilon_sec,
    )

    return FiniteSizePenaltyResult(
        mu=mu, test_size_k=k, key_size_n=n, epsilon_sec=epsilon_sec
    )


# =============================================================================
# QBER Adjustment (TASK-QBER-ADJUST-001)
# =============================================================================


def adjust_qber(
    observed_qber: float,
    penalty_mu: float,
    warning_threshold: float = QBER_WARNING_THRESHOLD,
    hard_limit: float = QBER_HARD_LIMIT,
) -> QBERAdjustmentResult:
    """
    Adjust observed QBER with penalty and check thresholds.

    Computes e_adj = e_obs + μ and checks against:
    - Warning threshold (default 11%): Continue but log degraded regime
    - Hard limit (default 22%): Abort required

    Parameters
    ----------
    observed_qber : float
        Raw observed QBER from test set.
    penalty_mu : float
        Finite-size penalty to add.
    warning_threshold : float
        Warning threshold (default: 0.11 = 11%).
    hard_limit : float
        Hard abort limit (default: 0.22 = 22%).

    Returns
    -------
    QBERAdjustmentResult
        Result with adjusted QBER and status.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    - sprint_2_specification.md Section 3.3
    - Phase II analysis: Hard abort trigger

    Examples
    --------
    >>> result = adjust_qber(observed_qber=0.05, penalty_mu=0.003)
    >>> result.adjusted_qber
    0.053
    >>> result.status == ValidationStatus.PASSED
    True
    """
    # Validate inputs
    if not 0 <= observed_qber <= 0.5:
        raise ValueError(f"observed_qber must be in [0, 0.5], got {observed_qber}")
    if penalty_mu < 0:
        raise ValueError(f"penalty_mu must be non-negative, got {penalty_mu}")

    # Compute adjusted QBER
    adjusted_qber = observed_qber + penalty_mu

    # Determine status
    if adjusted_qber > hard_limit:
        status = ValidationStatus.FAILED
        abort_code = ABORT_CODE_QBER_HIGH
        message = (
            f"QBER HARD LIMIT EXCEEDED: e_adj={adjusted_qber:.4f} > {hard_limit:.2f}. "
            f"(e_obs={observed_qber:.4f}, μ={penalty_mu:.4f}). "
            f"Secure OT is impossible at this error rate."
        )
        logger.warning(
            "QBER_ABORT e_adj=%.4f > hard_limit=%.2f abort_code=%s",
            adjusted_qber,
            hard_limit,
            abort_code,
        )
    elif adjusted_qber > warning_threshold:
        status = ValidationStatus.WARNING
        abort_code = None
        message = (
            f"QBER WARNING: e_adj={adjusted_qber:.4f} > {warning_threshold:.2f} "
            f"but ≤ {hard_limit:.2f}. "
            f"(e_obs={observed_qber:.4f}, μ={penalty_mu:.4f}). "
            f"Operating in degraded regime with reduced key rate."
        )
        logger.warning(
            "QBER_WARNING e_adj=%.4f > warning=%.2f",
            adjusted_qber,
            warning_threshold,
        )
    else:
        status = ValidationStatus.PASSED
        abort_code = None
        message = (
            f"QBER within bounds: e_adj={adjusted_qber:.4f} ≤ {warning_threshold:.2f}. "
            f"(e_obs={observed_qber:.4f}, μ={penalty_mu:.4f})."
        )
        logger.info(
            "QBER_OK e_adj=%.4f <= warning=%.2f",
            adjusted_qber,
            warning_threshold,
        )

    return QBERAdjustmentResult(
        status=status,
        observed_qber=observed_qber,
        penalty_mu=penalty_mu,
        adjusted_qber=adjusted_qber,
        warning_threshold=warning_threshold,
        hard_limit=hard_limit,
        message=message,
        abort_code=abort_code,
    )


class QBERAdjuster:
    """
    Stateful QBER adjuster combining penalty calculation and threshold checks.

    This class combines finite-size penalty computation and QBER adjustment
    into a single interface for convenience.

    Attributes
    ----------
    epsilon_sec : float
        Security parameter for penalty calculation.
    warning_threshold : float
        Warning threshold (default 11%).
    hard_limit : float
        Hard abort limit (default 22%).

    Examples
    --------
    >>> adjuster = QBERAdjuster(epsilon_sec=1e-10)
    >>> adjuster.epsilon_sec
    1e-10
    >>> adjuster.warning_threshold
    0.11
    """

    def __init__(
        self,
        epsilon_sec: float,
        warning_threshold: float = QBER_WARNING_THRESHOLD,
        hard_limit: float = QBER_HARD_LIMIT,
    ) -> None:
        """
        Initialize QBER adjuster.

        Parameters
        ----------
        epsilon_sec : float
            Security parameter for finite-size penalty.
        warning_threshold : float
            Warning threshold (default: 0.11).
        hard_limit : float
            Hard abort limit (default: 0.22).
        """
        if not 0 < epsilon_sec < 1:
            raise ValueError(f"epsilon_sec must be in (0, 1), got {epsilon_sec}")

        self._epsilon_sec = epsilon_sec
        self._warning_threshold = warning_threshold
        self._hard_limit = hard_limit

    @property
    def epsilon_sec(self) -> float:
        """Get security parameter."""
        return self._epsilon_sec

    @property
    def warning_threshold(self) -> float:
        """Get warning threshold."""
        return self._warning_threshold

    @property
    def hard_limit(self) -> float:
        """Get hard limit."""
        return self._hard_limit

    def compute_adjusted_qber(
        self, observed_qber: float, key_size_n: int, test_size_k: int
    ) -> QBERAdjustmentResult:
        """
        Compute adjusted QBER with penalty and check thresholds.

        Parameters
        ----------
        observed_qber : float
            Raw observed QBER from test set.
        key_size_n : int
            Remaining key size.
        test_size_k : int
            Test set size.

        Returns
        -------
        QBERAdjustmentResult
            Result with adjusted QBER and validation status.
        """
        # Compute penalty
        penalty_result = compute_finite_size_penalty(
            key_size_n=key_size_n,
            test_size_k=test_size_k,
            epsilon_sec=self._epsilon_sec,
        )

        # Adjust and check thresholds
        return adjust_qber(
            observed_qber=observed_qber,
            penalty_mu=penalty_result.mu,
            warning_threshold=self._warning_threshold,
            hard_limit=self._hard_limit,
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Abort codes
    "ABORT_CODE_DETECTION_ANOMALY",
    "ABORT_CODE_QBER_HIGH",
    # Enums
    "ValidationStatus",
    # Result dataclasses
    "DetectionValidationResult",
    "FiniteSizePenaltyResult",
    "QBERAdjustmentResult",
    # Classes
    "DetectionValidator",
    "QBERAdjuster",
    # Functions
    "compute_finite_size_penalty",
    "adjust_qber",
]

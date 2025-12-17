"""
Statistical validation using Hoeffding bounds.

This module provides detection event validation and statistical
testing for quantum measurement consistency.

References
----------
- Hoeffding (1963): Probability Inequalities
- Erven et al. (2014): Detection rate validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError

logger = get_logger(__name__)


@dataclass
class HoeffdingBound:
    """
    Hoeffding bound calculation result.

    Parameters
    ----------
    observed_rate : float
        Observed detection/error rate.
    expected_rate : float
        Expected rate under null hypothesis.
    deviation : float
        Absolute deviation |observed - expected|.
    bound : float
        Hoeffding bound on deviation probability.
    sample_size : int
        Number of samples.
    passes : bool
        True if deviation is within acceptable bounds.
    """

    observed_rate: float
    expected_rate: float
    deviation: float
    bound: float
    sample_size: int
    passes: bool


@dataclass
class ValidationResult:
    """
    Result of detection validation.

    Parameters
    ----------
    is_valid : bool
        Overall validation result.
    detection_rate : float
        Observed detection rate.
    expected_rate : float
        Expected detection rate.
    hoeffding_result : Optional[HoeffdingBound]
        Hoeffding bound analysis if computed.
    basis_balance_valid : bool
        True if basis distribution is balanced.
    basis_0_fraction : float
        Fraction of Z basis measurements.
    message : str
        Human-readable validation message.
    """

    is_valid: bool
    detection_rate: float
    expected_rate: float
    hoeffding_result: Optional[HoeffdingBound] = None
    basis_balance_valid: bool = True
    basis_0_fraction: float = 0.5
    message: str = ""


class DetectionValidator:
    """
    Statistical validation of detection events.

    Uses Hoeffding bounds to validate that:
    1. Detection rate matches expected rate
    2. Basis distribution is approximately uniform
    3. No statistical anomalies suggest attacks

    Parameters
    ----------
    expected_detection_rate : float
        Expected detection efficiency (0-1).
    tolerance : float
        Acceptable deviation from expected rate.
    confidence : float
        Required statistical confidence.

    Notes
    -----
    Hoeffding's inequality:
        P(|X̄ - μ| ≥ t) ≤ 2·exp(-2n·t²)

    For confidence 1-δ, acceptable deviation is:
        t = √[ln(2/δ) / (2n)]

    References
    ----------
    - Hoeffding (1963): "Probability Inequalities for Sums"
    - Erven et al. (2014): Detection validation in experiment
    """

    def __init__(
        self,
        expected_detection_rate: float = 1.0,
        tolerance: float = 0.05,
        confidence: float = 0.999,
    ) -> None:
        """
        Initialize validator.

        Parameters
        ----------
        expected_detection_rate : float
            Expected detection probability.
        tolerance : float
            Maximum acceptable deviation.
        confidence : float
            Required statistical confidence.
        """
        if not 0 < expected_detection_rate <= 1:
            raise InvalidParameterError(
                f"expected_detection_rate={expected_detection_rate} must be in (0, 1]"
            )
        if not 0 < tolerance < 1:
            raise InvalidParameterError(
                f"tolerance={tolerance} must be in (0, 1)"
            )
        if not 0 < confidence < 1:
            raise InvalidParameterError(
                f"confidence={confidence} must be in (0, 1)"
            )

        self._expected_rate = expected_detection_rate
        self._tolerance = tolerance
        self._confidence = confidence

    def validate_statistics(
        self,
        num_detections: int,
        num_attempted: int,
        bases: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Validate detection statistics.

        Parameters
        ----------
        num_detections : int
            Number of successful detections.
        num_attempted : int
            Number of detection attempts.
        bases : Optional[np.ndarray]
            Basis choices (for balance check).

        Returns
        -------
        ValidationResult
            Complete validation result.
        """
        if num_attempted <= 0:
            return ValidationResult(
                is_valid=False,
                detection_rate=0.0,
                expected_rate=self._expected_rate,
                message="No detection attempts",
            )

        observed_rate = num_detections / num_attempted

        # Compute Hoeffding bound
        hoeffding = self._compute_hoeffding_bound(
            observed_rate=observed_rate,
            expected_rate=self._expected_rate,
            n=num_attempted,
        )

        # Check basis balance if provided
        basis_valid = True
        basis_0_frac = 0.5
        if bases is not None and len(bases) > 0:
            basis_0_frac = np.mean(bases == 0)
            # Check if close to 0.5 using Hoeffding
            basis_hoeffding = self._compute_hoeffding_bound(
                observed_rate=basis_0_frac,
                expected_rate=0.5,
                n=len(bases),
            )
            basis_valid = basis_hoeffding.passes

        # Overall validation
        detection_valid = (
            abs(observed_rate - self._expected_rate) <= self._tolerance
            and hoeffding.passes
        )

        is_valid = detection_valid and basis_valid

        # Generate message
        if is_valid:
            message = (
                f"Validation passed: detection={observed_rate:.4f}, "
                f"basis_0={basis_0_frac:.4f}"
            )
        else:
            issues = []
            if not detection_valid:
                issues.append(
                    f"detection rate {observed_rate:.4f} deviates from "
                    f"expected {self._expected_rate:.4f}"
                )
            if not basis_valid:
                issues.append(
                    f"basis imbalance: {basis_0_frac:.4f} vs expected 0.5"
                )
            message = "Validation failed: " + "; ".join(issues)

        logger.debug(message)

        return ValidationResult(
            is_valid=is_valid,
            detection_rate=observed_rate,
            expected_rate=self._expected_rate,
            hoeffding_result=hoeffding,
            basis_balance_valid=basis_valid,
            basis_0_fraction=basis_0_frac,
            message=message,
        )

    def _compute_hoeffding_bound(
        self,
        observed_rate: float,
        expected_rate: float,
        n: int,
    ) -> HoeffdingBound:
        """
        Compute Hoeffding concentration bound.

        Parameters
        ----------
        observed_rate : float
            Observed rate.
        expected_rate : float
            Expected rate.
        n : int
            Sample size.

        Returns
        -------
        HoeffdingBound
            Bound calculation result.
        """
        deviation = abs(observed_rate - expected_rate)

        # Hoeffding bound: P(|X̄ - μ| ≥ t) ≤ 2·exp(-2n·t²)
        if deviation > 0 and n > 0:
            bound = 2.0 * math.exp(-2.0 * n * deviation ** 2)
        else:
            bound = 1.0

        # For confidence, acceptable t satisfies: 2·exp(-2n·t²) ≤ 1 - confidence
        # Solving: t = √[ln(2/(1-confidence)) / (2n)]
        delta = 1.0 - self._confidence
        if n > 0 and delta > 0:
            acceptable_deviation = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        else:
            acceptable_deviation = float("inf")

        passes = deviation <= acceptable_deviation

        return HoeffdingBound(
            observed_rate=observed_rate,
            expected_rate=expected_rate,
            deviation=deviation,
            bound=bound,
            sample_size=n,
            passes=passes,
        )

    @staticmethod
    def required_samples_for_tolerance(
        tolerance: float,
        confidence: float = 0.999,
    ) -> int:
        """
        Calculate samples needed for given tolerance.

        Parameters
        ----------
        tolerance : float
            Maximum acceptable deviation.
        confidence : float
            Required confidence level.

        Returns
        -------
        int
            Minimum required sample size.

        Notes
        -----
        Inverts Hoeffding bound:
            n = ln(2/δ) / (2·t²)
        where δ = 1 - confidence, t = tolerance.
        """
        delta = 1.0 - confidence
        n = math.log(2.0 / delta) / (2.0 * tolerance ** 2)
        return int(math.ceil(n))

    @property
    def expected_rate(self) -> float:
        """Expected detection rate."""
        return self._expected_rate

    @property
    def tolerance(self) -> float:
        """Acceptable deviation tolerance."""
        return self._tolerance

    @property
    def confidence(self) -> float:
        """Required confidence level."""
        return self._confidence

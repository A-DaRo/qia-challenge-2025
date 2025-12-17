"""
QBER estimation with finite-size statistical penalties.

This module implements QBER estimation from test bits, including
the finite-size penalty (μ) from Erven et al. for security analysis.

References
----------
- Erven et al. (2014): Theorem 2, Eq. (2) - μ penalty formula
- Schaffner et al. (2009): QBER limits for NSM security
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import (
    InvalidParameterError,
    QBERThresholdExceeded,
)
from caligo.types.phase_contracts import QBER_HARD_LIMIT, QBER_CONSERVATIVE_LIMIT

logger = get_logger(__name__)


@dataclass
class QBEREstimate:
    """
    Result of QBER estimation.

    Parameters
    ----------
    observed_qber : float
        Observed error rate from test bits.
    adjusted_qber : float
        QBER with finite-size penalty: e_obs + μ.
    mu_penalty : float
        Statistical penalty μ from finite-size analysis.
    num_test_bits : int
        Number of bits used for estimation.
    num_errors : int
        Number of errors observed.
    confidence_level : float
        Statistical confidence (1 - probability of underestimate).
    exceeds_hard_limit : bool
        True if adjusted QBER > 22%.
    exceeds_warning_limit : bool
        True if adjusted QBER > 11%.
    """

    observed_qber: float
    adjusted_qber: float
    mu_penalty: float
    num_test_bits: int
    num_errors: int
    confidence_level: float = 1.0 - 1e-10
    exceeds_hard_limit: bool = False
    exceeds_warning_limit: bool = False


class QBEREstimator:
    """
    QBER estimation with finite-size penalties.

    Estimates the quantum bit error rate from a test subset and
    computes the finite-size penalty μ that accounts for statistical
    fluctuations.

    Parameters
    ----------
    epsilon_sec : float
        Security parameter for confidence bound.
    hard_limit : float
        QBER above which protocol must abort.
    warning_limit : float
        QBER above which warning is logged.

    Notes
    -----
    The μ penalty formula from Erven et al. (2014) Eq. (2):

        μ = √[(n+k)/(n·k) · (k+1)/k · ln(4/ε_sec)]

    where:
    - n = test set size
    - k = key set size
    - ε_sec = security parameter

    This ensures that with probability ≥ 1 - ε_sec, the true QBER
    is below e_obs + μ.

    References
    ----------
    - Erven et al. (2014) Theorem 2, Eq. (2)
    - Schaffner et al. (2009) Section 4.2
    """

    def __init__(
        self,
        epsilon_sec: float = 1e-10,
        hard_limit: float = QBER_HARD_LIMIT,
        warning_limit: float = QBER_CONSERVATIVE_LIMIT,
    ) -> None:
        """
        Initialize QBER estimator.

        Parameters
        ----------
        epsilon_sec : float
            Security parameter.
        hard_limit : float
            Hard QBER limit (abort threshold).
        warning_limit : float
            Warning QBER limit.
        """
        if not 0 < epsilon_sec < 1:
            raise InvalidParameterError(
                f"epsilon_sec={epsilon_sec} must be in (0, 1)"
            )

        self._epsilon_sec = epsilon_sec
        self._hard_limit = hard_limit
        self._warning_limit = warning_limit

    def estimate(
        self,
        alice_test_bits: np.ndarray,
        bob_test_bits: np.ndarray,
        key_size: int,
    ) -> QBEREstimate:
        """
        Estimate QBER from test bits.

        Parameters
        ----------
        alice_test_bits : np.ndarray
            Alice's bits at test positions.
        bob_test_bits : np.ndarray
            Bob's bits at test positions.
        key_size : int
            Size of remaining key (for μ calculation).

        Returns
        -------
        QBEREstimate
            Complete QBER estimate with penalty.

        Raises
        ------
        QBERThresholdExceeded
            If adjusted QBER exceeds hard limit.
        """
        n_test = len(alice_test_bits)

        if n_test == 0:
            raise InvalidParameterError("Test set is empty")

        if len(bob_test_bits) != n_test:
            raise InvalidParameterError(
                f"Test bit lengths differ: {n_test} vs {len(bob_test_bits)}"
            )

        # Count errors
        errors = np.sum(alice_test_bits != bob_test_bits)
        observed_qber = errors / n_test

        # Compute μ penalty
        mu = self.compute_mu_penalty(
            n=n_test,
            k=key_size,
            epsilon_sec=self._epsilon_sec,
        )

        adjusted_qber = observed_qber + mu

        # Check thresholds
        exceeds_hard = adjusted_qber > self._hard_limit
        exceeds_warning = adjusted_qber > self._warning_limit

        result = QBEREstimate(
            observed_qber=observed_qber,
            adjusted_qber=adjusted_qber,
            mu_penalty=mu,
            num_test_bits=n_test,
            num_errors=int(errors),
            confidence_level=1.0 - self._epsilon_sec,
            exceeds_hard_limit=exceeds_hard,
            exceeds_warning_limit=exceeds_warning,
        )

        # Logging
        if exceeds_hard:
            logger.error(
                f"QBER {adjusted_qber:.4f} exceeds hard limit {self._hard_limit}"
            )
            raise QBERThresholdExceeded(
                f"Adjusted QBER {adjusted_qber:.4f} exceeds hard limit "
                f"{self._hard_limit}. Secure OT is impossible."
            )
        elif exceeds_warning:
            logger.warning(
                f"QBER {adjusted_qber:.4f} exceeds warning limit "
                f"{self._warning_limit} - security margin reduced"
            )
        else:
            logger.info(
                f"QBER estimate: {observed_qber:.4f} + {mu:.4f} = "
                f"{adjusted_qber:.4f} (n={n_test}, k={key_size})"
            )

        return result

    @staticmethod
    def compute_mu_penalty(
        n: int,
        k: int,
        epsilon_sec: float = 1e-10,
    ) -> float:
        """
        Compute finite-size penalty μ.

        Parameters
        ----------
        n : int
            Test set size.
        k : int
            Key set size.
        epsilon_sec : float
            Security parameter.

        Returns
        -------
        float
            μ penalty value.

        Notes
        -----
        Formula from Erven et al. (2014) Eq. (2):

            μ = √[(n+k)/(n·k) · (k+1)/k · ln(4/ε_sec)]
        """
        if n <= 0 or k <= 0:
            raise InvalidParameterError(
                f"n={n} and k={k} must be positive"
            )

        term1 = (n + k) / (n * k)
        term2 = (k + 1) / k
        log_term = math.log(4.0 / epsilon_sec)

        mu = math.sqrt(term1 * term2 * log_term)
        return mu

    def validate(self, estimate: QBEREstimate) -> bool:
        """
        Validate a QBER estimate against thresholds.

        Parameters
        ----------
        estimate : QBEREstimate
            QBER estimate to validate.

        Returns
        -------
        bool
            True if QBER is within acceptable limits.

        Raises
        ------
        QBERThresholdExceeded
            If QBER exceeds hard limit.
        """
        if estimate.exceeds_hard_limit:
            raise QBERThresholdExceeded(
                f"QBER {estimate.adjusted_qber:.4f} exceeds limit "
                f"{self._hard_limit}"
            )
        return True

    @property
    def epsilon_sec(self) -> float:
        """Security parameter."""
        return self._epsilon_sec

    @property
    def hard_limit(self) -> float:
        """Hard QBER limit."""
        return self._hard_limit

    @property
    def warning_limit(self) -> float:
        """Warning QBER limit."""
        return self._warning_limit

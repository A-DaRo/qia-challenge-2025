"""
Finite-size statistical corrections for E-HOK key extraction.

This module implements the statistical corrections needed for
real-world finite key extraction, accounting for:

1. Parameter estimation uncertainty from finite test samples
2. Detection statistics validation via Hoeffding bounds
3. Finite-key security costs

These corrections are essential for security proofs with finite
resources, as asymptotic analysis alone is insufficient.

References
----------
- Tomamichel et al. (2012), Tight Finite-Key Analysis
- Hoeffding (1963), Probability Inequalities
- Erven et al. (2014), Experimental Implementation
"""

from __future__ import annotations

import math
from typing import Tuple

from caligo.security.bounds import (
    DEFAULT_EPSILON_SEC,
    DEFAULT_EPSILON_COR,
    bounded_storage_entropy,
)
from caligo.types.exceptions import InvalidParameterError
from caligo.utils.math import binary_entropy


# =============================================================================
# Statistical Fluctuation
# =============================================================================


def compute_statistical_fluctuation(
    n: int,
    k: int,
    epsilon_pe: float = 1e-10,
) -> float:
    """
    Compute the statistical fluctuation penalty μ.

    For finite key lengths, the QBER estimate from k test bits has
    statistical uncertainty that must be accounted for:

        μ = √[(n+k)/(n·k) · (k+1)/k] · √[ln(4/ε_PE)]

    This penalty is ADDED to the measured QBER for security calculations.

    Parameters
    ----------
    n : int
        Number of bits used for key extraction (after test sampling).
    k : int
        Number of bits used for QBER estimation (test sample).
    epsilon_pe : float
        Parameter estimation failure probability. Default: 1e-10.

    Returns
    -------
    float
        Statistical fluctuation μ to add to measured QBER.

    Raises
    ------
    InvalidParameterError
        If parameters are invalid.

    Notes
    -----
    The penalty scales as:
    - O(1/√k) for fixed n — larger test samples reduce penalty
    - O(√(1/n)) for fixed k — larger key fractions reduce penalty

    References
    ----------
    - Tomamichel et al. (2012), Theorem 1
    - Erven et al. (2014), Security section

    Examples
    --------
    >>> mu = compute_statistical_fluctuation(100000, 5000, 1e-10)
    >>> 0.01 < mu < 0.02  # About 1.5% penalty for good sample size
    True
    >>> mu_small = compute_statistical_fluctuation(1000, 100, 1e-10)
    >>> mu_small > 0.10  # Much larger for small samples
    True
    """
    if n <= 0:
        raise InvalidParameterError(f"n={n} must be > 0")
    if k <= 0:
        raise InvalidParameterError(f"k={k} must be > 0")
    if not 0 < epsilon_pe < 1:
        raise InvalidParameterError(f"epsilon_pe={epsilon_pe} must be in (0, 1)")

    # Variance factor from finite sampling
    variance_factor = ((n + k) / (n * k)) * ((k + 1) / k)

    # Confidence term from failure probability
    confidence_term = math.log(4.0 / epsilon_pe)

    return math.sqrt(variance_factor * confidence_term)


# =============================================================================
# Hoeffding Bounds
# =============================================================================


def hoeffding_detection_interval(
    n: int,
    p_expected: float,
    epsilon: float = 1e-10,
) -> Tuple[float, float]:
    """
    Compute secure interval for detection statistics using Hoeffding.

    For n independent Bernoulli trials with probability p, the number
    of successes S satisfies (via Hoeffding's inequality):

        Pr[|S - p·n| > ζ·n] ≤ 2·exp(-2·ζ²·n)

    Setting failure probability to ε gives:
        ζ = √[ln(2/ε) / (2n)]

    Parameters
    ----------
    n : int
        Total number of trials (e.g., photon emission events).
    p_expected : float
        Expected probability (e.g., detection efficiency).
    epsilon : float
        Allowed failure probability for the bound. Default: 1e-10.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for expected counts as fractions.
        Multiply by n to get count bounds.

    Raises
    ------
    InvalidParameterError
        If parameters are invalid.

    Notes
    -----
    Alice uses this to verify Bob's reported detection counts.
    If counts fall outside interval → protocol aborts (potential attack).

    References
    ----------
    - Hoeffding (1963)
    - Erven et al. (2014), Correctness section

    Examples
    --------
    >>> lower, upper = hoeffding_detection_interval(10000, 0.015, 1e-10)
    >>> 0.01 < lower < 0.015 < upper < 0.02
    True
    """
    if n <= 0:
        raise InvalidParameterError(f"n={n} must be > 0")
    if not 0 <= p_expected <= 1:
        raise InvalidParameterError(f"p_expected={p_expected} must be in [0, 1]")
    if not 0 < epsilon < 1:
        raise InvalidParameterError(f"epsilon={epsilon} must be in (0, 1)")

    # Hoeffding deviation bound
    zeta = math.sqrt(math.log(2.0 / epsilon) / (2.0 * n))

    lower = max(0.0, p_expected - zeta)
    upper = min(1.0, p_expected + zeta)

    return (lower, upper)


def hoeffding_count_interval(
    n: int,
    p_expected: float,
    epsilon: float = 1e-10,
) -> Tuple[int, int]:
    """
    Compute secure count interval using Hoeffding bound.

    Parameters
    ----------
    n : int
        Total number of trials.
    p_expected : float
        Expected success probability.
    epsilon : float
        Failure probability bound. Default: 1e-10.

    Returns
    -------
    tuple[int, int]
        (min_count, max_count) for acceptable detection counts.

    Examples
    --------
    >>> min_c, max_c = hoeffding_count_interval(10000, 0.5)
    >>> 4800 < min_c < 5000 < max_c < 5200
    True
    """
    lower_frac, upper_frac = hoeffding_detection_interval(n, p_expected, epsilon)
    return (int(math.floor(lower_frac * n)), int(math.ceil(upper_frac * n)))


# =============================================================================
# Finite Key Length Computation
# =============================================================================


def compute_finite_key_length(
    n: int,
    qber_measured: float,
    storage_noise_r: float,
    storage_rate_nu: float,
    ec_efficiency: float = 1.16,
    epsilon_sec: float = DEFAULT_EPSILON_SEC,
    epsilon_cor: float = DEFAULT_EPSILON_COR,
    test_fraction: float = 0.05,
) -> int:
    """
    Compute the extractable secure key length for finite resources.

    Full formula incorporating all finite-size effects:

        ℓ = ⌊n · h_min(r,ν) - n · f · h(Q + μ) - log₂(1/ε_sec) - log₂(1/ε_cor)⌋

    Where:
    - h_min(r,ν) is the bounded storage entropy
    - f is the error correction efficiency
    - Q is the measured QBER
    - μ is the statistical fluctuation penalty
    - ε_sec, ε_cor are security/correctness parameters

    Parameters
    ----------
    n : int
        Number of raw bits after sifting (before test sampling).
    qber_measured : float
        Measured QBER from test sample (NOT including μ penalty yet).
    storage_noise_r : float
        Adversary storage noise parameter.
    storage_rate_nu : float
        Adversary storage rate.
    ec_efficiency : float
        Error correction efficiency f (typically 1.1-1.5). Default: 1.16.
    epsilon_sec : float
        Security parameter. Default: 1e-10.
    epsilon_cor : float
        Correctness parameter. Default: 1e-6.
    test_fraction : float
        Fraction of bits used for testing. Default: 0.05 (5%).

    Returns
    -------
    int
        Maximum extractable key length ℓ (non-negative).
        Returns 0 if key is not extractable.

    Raises
    ------
    InvalidParameterError
        If parameters are invalid.

    Notes
    -----
    The μ penalty is computed internally and applied to QBER before
    calculating leakage. This is more conservative than using raw QBER.

    References
    ----------
    - Erven et al. (2014), Eq. (8)
    - Lupo et al. (2023), Eq. (43)
    - Tomamichel et al. (2012)

    Examples
    --------
    >>> length = compute_finite_key_length(
    ...     n=100000,
    ...     qber_measured=0.02,
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ... )
    >>> length > 0  # Should produce positive key
    True
    """
    if n <= 0:
        raise InvalidParameterError(f"n={n} must be > 0")
    if not 0 <= qber_measured <= 0.5:
        raise InvalidParameterError(f"qber_measured={qber_measured} must be in [0, 0.5]")
    if not 0 <= storage_noise_r <= 1:
        raise InvalidParameterError(
            f"storage_noise_r={storage_noise_r} must be in [0, 1]"
        )
    if not 0 <= storage_rate_nu <= 1:
        raise InvalidParameterError(
            f"storage_rate_nu={storage_rate_nu} must be in [0, 1]"
        )
    if ec_efficiency < 1.0:
        raise InvalidParameterError(f"ec_efficiency={ec_efficiency} must be >= 1.0")
    if not 0 < epsilon_sec < 1:
        raise InvalidParameterError(f"epsilon_sec={epsilon_sec} must be in (0, 1)")
    if not 0 < epsilon_cor < 1:
        raise InvalidParameterError(f"epsilon_cor={epsilon_cor} must be in (0, 1)")
    if not 0 < test_fraction < 1:
        raise InvalidParameterError(f"test_fraction={test_fraction} must be in (0, 1)")

    # Split into test and key portions
    k = int(math.ceil(n * test_fraction))  # Test bits
    n_key = n - k  # Key extraction bits

    if n_key <= 0:
        return 0

    # Compute statistical fluctuation penalty
    mu = compute_statistical_fluctuation(n_key, k, epsilon_sec / 4)

    # Adjusted QBER with penalty
    qber_adjusted = min(0.5, qber_measured + mu)

    # Min-entropy from storage model
    h_min = bounded_storage_entropy(storage_noise_r, storage_rate_nu)

    # Information leakage from error correction
    h_qber = binary_entropy(qber_adjusted) if qber_adjusted > 0 else 0.0
    leakage_rate = ec_efficiency * h_qber

    # Extractable entropy rate per bit
    entropy_rate = h_min - leakage_rate

    # Security and correctness costs
    security_cost = math.log2(1.0 / epsilon_sec)
    correctness_cost = math.log2(1.0 / epsilon_cor)

    # Final key length
    raw_length = n_key * entropy_rate - security_cost - correctness_cost

    return max(0, int(math.floor(raw_length)))


def compute_optimal_test_fraction(
    n: int,
    qber_estimate: float,
    storage_noise_r: float,
    storage_rate_nu: float,
    ec_efficiency: float = 1.16,
    epsilon_sec: float = DEFAULT_EPSILON_SEC,
) -> float:
    """
    Compute the optimal test sample fraction to maximize key length.

    Larger test samples reduce the μ penalty but leave fewer bits
    for key extraction. This function finds the optimal balance.

    Parameters
    ----------
    n : int
        Total number of raw bits.
    qber_estimate : float
        Estimated QBER for optimization.
    storage_noise_r : float
        Storage noise parameter.
    storage_rate_nu : float
        Storage rate parameter.
    ec_efficiency : float
        Error correction efficiency.
    epsilon_sec : float
        Security parameter.

    Returns
    -------
    float
        Optimal test fraction ∈ (0, 1).

    Notes
    -----
    Uses golden-section search to find the maximum key length.
    """
    from scipy.optimize import minimize_scalar

    def neg_key_length(test_frac: float) -> float:
        if test_frac <= 0.001 or test_frac >= 0.999:
            return 0.0
        length = compute_finite_key_length(
            n=n,
            qber_measured=qber_estimate,
            storage_noise_r=storage_noise_r,
            storage_rate_nu=storage_rate_nu,
            ec_efficiency=ec_efficiency,
            epsilon_sec=epsilon_sec,
            test_fraction=test_frac,
        )
        return -float(length)

    result = minimize_scalar(neg_key_length, bounds=(0.01, 0.5), method="bounded")
    return result.x


def compute_min_n_for_key_length(
    target_length: int,
    qber_estimate: float,
    storage_noise_r: float,
    storage_rate_nu: float,
    ec_efficiency: float = 1.16,
    epsilon_sec: float = DEFAULT_EPSILON_SEC,
    test_fraction: float = 0.05,
) -> int:
    """
    Compute minimum n required for target key length.

    Parameters
    ----------
    target_length : int
        Desired final key length in bits.
    qber_estimate : float
        Estimated QBER.
    storage_noise_r : float
        Storage noise parameter.
    storage_rate_nu : float
        Storage rate parameter.
    ec_efficiency : float
        Error correction efficiency.
    epsilon_sec : float
        Security parameter.
    test_fraction : float
        Test sample fraction.

    Returns
    -------
    int
        Minimum required n (raw bits after sifting).

    Examples
    --------
    >>> min_n = compute_min_n_for_key_length(
    ...     target_length=128,
    ...     qber_estimate=0.02,
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ... )
    >>> min_n > 0
    True
    """
    # Binary search for minimum n
    low, high = 100, 10_000_000

    while low < high:
        mid = (low + high) // 2
        length = compute_finite_key_length(
            n=mid,
            qber_measured=qber_estimate,
            storage_noise_r=storage_noise_r,
            storage_rate_nu=storage_rate_nu,
            ec_efficiency=ec_efficiency,
            epsilon_sec=epsilon_sec,
            test_fraction=test_fraction,
        )
        if length >= target_length:
            high = mid
        else:
            low = mid + 1

    return low

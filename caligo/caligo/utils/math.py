"""
Mathematical functions for E-HOK protocol calculations.

This module provides the mathematical functions used throughout
the protocol, particularly for entropy calculations and security bounds.

References
----------
- Schaffner et al. (2009): Binary entropy, channel capacity
- Erven et al. (2014): Finite-size penalty (μ)
- Lupo et al. (2020): Gamma function for NSM bounds
"""

from __future__ import annotations

import math

import numpy as np


def binary_entropy(p: float) -> float:
    """
    Compute the binary entropy function h(p).

    h(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Parameters
    ----------
    p : float
        Probability value in [0, 1].

    Returns
    -------
    float
        Binary entropy in bits.

    Raises
    ------
    ValueError
        If p is not in [0, 1].

    Notes
    -----
    Edge cases handled by convention:
    - h(0) = 0 (since lim_{x→0} x·log(x) = 0)
    - h(1) = 0
    - h(0.5) = 1 (maximum entropy)

    References
    ----------
    - Schaffner et al. (2009) Section 3.2: "binary-entropy function"

    Examples
    --------
    >>> binary_entropy(0.5)
    1.0
    >>> binary_entropy(0.0)
    0.0
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p={p} must be in [0, 1]")

    # Handle edge cases
    if p == 0 or p == 1:
        return 0.0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def channel_capacity(qber: float) -> float:
    """
    Compute binary symmetric channel capacity.

    C(QBER) = 1 - h(QBER)

    Parameters
    ----------
    qber : float
        Quantum bit error rate in [0, 0.5].

    Returns
    -------
    float
        Channel capacity in bits per symbol.

    Raises
    ------
    ValueError
        If qber is not in [0, 0.5].

    References
    ----------
    - Shannon's noisy channel coding theorem

    Examples
    --------
    >>> channel_capacity(0.0)
    1.0
    >>> channel_capacity(0.5)
    0.0
    """
    if not 0 <= qber <= 0.5:
        raise ValueError(f"qber={qber} must be in [0, 0.5]")

    return 1.0 - binary_entropy(qber)


def finite_size_penalty(n: int, k: int, epsilon_sec: float = 1e-10) -> float:
    """
    Compute finite-size statistical penalty μ.

    μ = √((n + k)/(n·k) · (k + 1)/k) · ln(4/ε_sec)

    This penalty accounts for the statistical uncertainty when
    estimating QBER from a finite test sample.

    Parameters
    ----------
    n : int
        Size of remaining key (after test set removal).
    k : int
        Size of test set.
    epsilon_sec : float
        Security parameter (default: 10^{-10}).

    Returns
    -------
    float
        Statistical penalty μ to add to observed QBER.

    Raises
    ------
    ValueError
        If preconditions are violated.

    Notes
    -----
    Pre-conditions:
    - n > 0
    - k > 0
    - epsilon_sec ∈ (0, 1)

    References
    ----------
    - Erven et al. (2014) Theorem 2, Eq. (2)
    - phase_II.md Section 2.B

    Examples
    --------
    >>> mu = finite_size_penalty(10000, 1000, 1e-10)
    >>> mu < 0.1  # Reasonable penalty for decent sample size
    True
    """
    if n <= 0:
        raise ValueError(f"n={n} must be > 0")
    if k <= 0:
        raise ValueError(f"k={k} must be > 0")
    if not (0 < epsilon_sec < 1):
        raise ValueError(f"epsilon_sec={epsilon_sec} must be in (0, 1)")

    # Compute the variance factor
    variance_factor = ((n + k) / (n * k)) * ((k + 1) / k)

    # Compute the confidence term
    confidence_term = math.log(4.0 / epsilon_sec)

    return math.sqrt(variance_factor * confidence_term)


def gamma_function(r: float) -> float:
    """
    Compute Γ(r) for NSM security bound with depolarizing storage.

    For depolarizing storage with noise parameter r:
    Γ(r) = 1 - log₂(1 + 3r²)

    This characterizes the adversary's storage quality in the
    Noisy Storage Model.

    Parameters
    ----------
    r : float
        Storage noise parameter in [0, 1].
        r = 0: Perfect storage (worst case for honest parties)
        r = 1: Complete depolarization (best case for security)

    Returns
    -------
    float
        Γ(r) value used in entropy bounds.

    Raises
    ------
    ValueError
        If r is not in [0, 1].

    Notes
    -----
    The depolarizing channel with parameter r maps:
    ρ → (1-r)ρ + r·I/2

    References
    ----------
    - Lupo et al. (2020) "Max Bound" derivation
    - phase_IV.md Section 2.A

    Examples
    --------
    >>> gamma_function(0.0)  # Perfect storage
    1.0
    >>> gamma_function(1.0)  # Full depolarization
    -1.0
    """
    if not 0 <= r <= 1:
        raise ValueError(f"r={r} must be in [0, 1]")

    return 1.0 - math.log2(1.0 + 3.0 * r * r)


def smooth_min_entropy_rate(qber: float, gamma: float) -> float:
    """
    Compute smooth min-entropy rate for NSM security.

    This function computes the per-bit entropy rate after
    accounting for QBER and adversary storage quality.

    h_rate = Γ(r) - h(qber)

    Parameters
    ----------
    qber : float
        Quantum bit error rate in [0, 0.5].
    gamma : float
        Γ(r) value from gamma_function.

    Returns
    -------
    float
        Smooth min-entropy rate per bit.

    Notes
    -----
    For secure OT, we require h_rate > 0, which bounds the
    maximum tolerable QBER for a given storage noise r.
    """
    return gamma - binary_entropy(qber)


def key_length_bound(
    n_sifted: int,
    qber: float,
    leakage_bits: int,
    epsilon_sec: float = 1e-10,
    gamma: float = 0.0,
) -> int:
    """
    Compute maximum secure key length from sifted key.

    ℓ ≤ n · h_rate - leakage - 2·log₂(1/ε_sec)

    Parameters
    ----------
    n_sifted : int
        Length of sifted key in bits.
    qber : float
        Adjusted QBER (including finite-size penalty).
    leakage_bits : int
        Total information leakage |Σ| from reconciliation.
    epsilon_sec : float
        Security parameter.
    gamma : float
        Γ(r) from storage noise. If 0, uses 1-h(qber) (QKD mode).

    Returns
    -------
    int
        Maximum secure key length (floored).

    References
    ----------
    - Lupo et al. (2023), Eq. (43)
    """
    if gamma == 0.0:
        # QKD-style bound
        h_rate = 1.0 - binary_entropy(qber)
    else:
        h_rate = gamma - binary_entropy(qber)

    security_cost = 2.0 * math.log2(1.0 / epsilon_sec)
    raw_length = n_sifted * h_rate - leakage_bits - security_cost

    return max(0, int(math.floor(raw_length)))

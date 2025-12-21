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
from typing import Any

import numpy as np

# Import LDPC constants for rate selection
from caligo.reconciliation.constants import LDPC_F_CRIT
from caligo.reconciliation.constants import RATE_MIN, RATE_MAX


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


def suggested_ldpc_rate_from_qber(
    qber: float,
    safety_margin: float = 0.0,
) -> float:
    """
    Suggest an LDPC code rate for a given QBER.

    Uses RATE_MIN and RATE_MAX from caligo.reconciliation.constants
    to provide a bounded continuous suggestion instead of relying on
    a discrete LDPC_CODE_RATES table.

    Selection rules (highest feasible rate in [RATE_MIN, RATE_MAX]):
    1) Leakage-efficiency cap:
       (1 - rate) / h(qber) < f_crit  =>  rate > 1 - f_crit * h(qber)
    2) Decodability (Shannon capacity for a BSC):
       rate <= 1 - h(qber)

    Parameters
    ----------
    qber : float
        Estimated quantum bit error rate in [0, 0.5].
    safety_margin : float
        Additional margin to add to QBER for conservative rate selection.

    Returns
    -------
    float
        Suggested LDPC code rate clipped to [RATE_MIN, RATE_MAX].

    Notes
    -----
    This implementation prefers the highest rate within the intersection
    of the decodability and leakage-efficiency bounds while respecting
    the configured RATE_MIN and RATE_MAX from constants.
    """

    if not 0.0 <= qber <= 0.5:
        raise ValueError(f"qber={qber} must be in [0, 0.5]")
    if safety_margin < 0.0:
        raise ValueError(f"safety_margin={safety_margin} must be >= 0")

    effective_qber = min(qber + safety_margin, 0.499)
    entropy = binary_entropy(effective_qber)

    # Very low QBER: use maximum allowed rate
    if entropy < 1e-10:
        return RATE_MAX

    # Decodability bound (capacity)
    max_rate = max(0.0, 1.0 - entropy)
    # Leakage-efficiency cap: rate > 1 - f_crit * h(e)
    min_rate = max(0.0, 1.0 - LDPC_F_CRIT * entropy)

    # Intersection with configured allowed rate range
    lower = max(RATE_MIN, min_rate)
    upper = min(RATE_MAX, max_rate)

    if upper >= lower:
        # Choose the highest feasible rate in the intersection
        return upper

    # If no intersection, prefer decodability (cap to RATE_MAX)
    preferred = min(max_rate, RATE_MAX)
    if preferred >= RATE_MIN:
        return preferred

    # As a last resort return the minimum configured rate
    return RATE_MIN


def compute_qber_erven(
    fidelity: float,
    detector_error: float,
    detection_efficiency: float,
    dark_count_prob: float,
) -> float:
    """
    Compute total QBER using Erven et al. (2014) formula.

    The total QBER combines three error sources:

    1. **Source errors**: From imperfect Bell state preparation
       Q_source = (1 - F) / 2

    2. **Detector errors**: Intrinsic measurement errors
       Q_det = e_det

    3. **Dark count errors**: False detections when no photon arrives
       Q_dark = (1 - η) × P_dark / 2

    Parameters
    ----------
    fidelity : float
        EPR source fidelity F ∈ (0.5, 1].
    detector_error : float
        Intrinsic detector error rate e_det ∈ [0, 0.5].
    detection_efficiency : float
        Combined detection efficiency η ∈ (0, 1].
    dark_count_prob : float
        Dark count probability P_dark ∈ [0, 1].

    Returns
    -------
    float
        Total QBER = Q_source + Q_det + Q_dark.

    References
    ----------
    - Erven et al. (2014) Eq. 8 and Table I
    """
    q_source = (1.0 - fidelity) / 2.0
    q_det = detector_error
    q_dark = (1.0 - detection_efficiency) * dark_count_prob / 2.0
    return q_source + q_det + q_dark


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


def blind_reconciliation_initial_config(
    qber: float,
    frame_size: int = 4096,
    mother_rate: float = 0.5,
) -> dict[str, Any]:
    """
    Compute initial configuration for blind reconciliation from QBER heuristic.

    Per Theoretical Report v2 §4 and Implementation Report v2 §2.4.4,
    blind reconciliation can use an optional NSM-gated QBER heuristic
    (from compute_qber_erven) to inform:
    1. Starting rate cap (modulation budget)
    2. Iteration budget (higher QBER → more iterations allowed)

    This function implements the NSM-gated variant that provides a
    permissive starting point while maintaining the blind protocol's
    advantage of not requiring explicit test-bit QBER estimation.

    Parameters
    ----------
    qber : float
        Heuristic QBER estimate from NSM channel parameters.
        Computed via compute_qber_erven() using physical model.
    frame_size : int, optional
        LDPC frame size (default: 4096).
    mother_rate : float, optional
        Mother code rate (default: 0.5).

    Returns
    -------
    dict
        Configuration dictionary with keys:
        - 'initial_shortening': Initial number of bits to shorten (s₁)
        - 'max_iterations': Recommended iteration budget (t)
        - 'modulation_delta': Modulation budget (δ)
        - 'heuristic_qber': Input QBER (for logging)

    Notes
    -----
    The NSM-gated variant uses a conservative mapping:
    - Low QBER (≤ 0.05): Minimal shortening (s₁=0), tight budget (t=3)
    - Medium QBER (0.05-0.10): Moderate shortening, standard budget (t=3)
    - High QBER (> 0.10): Aggressive shortening, extended budget (t=5)

    This maintains blind reconciliation's efficiency advantage while
    using available physical information to avoid unnecessary iterations.

    References
    ----------
    - Martinez-Mateo et al. (2012): Blind Reconciliation protocol
    - Erven et al. (2014): NSM experimental parameters
    - Implementation Report v2 §2.4.4: NSM-Gated Heuristic

    Examples
    --------
    >>> config = blind_reconciliation_initial_config(qber=0.03)
    >>> config['initial_shortening']
    0
    >>> config['max_iterations']
    3
    """
    # Conservative QBER-to-shortening mapping
    # Per Theoretical Report v2: Starting rate should be permissive
    # to avoid excessive early shortening
    if qber <= 0.05:
        # Low noise: Start with minimal shortening
        initial_shortening = 0
        max_iterations = 3
        delta = 0.10  # Standard modulation budget
    elif qber <= 0.10:
        # Medium noise: Moderate pre-shortening
        # Target starting rate ≈ 0.55
        target_rate = 0.55
        # R_eff = (R₀ - s/n) / (1 - s/n)
        # Solve for s: s = n × (R₀ - R_eff) / (1 - R_eff)
        shortening_fraction = (mother_rate - target_rate) / (1.0 - target_rate)
        initial_shortening = int(shortening_fraction * frame_size)
        max_iterations = 3
        delta = 0.15
    else:
        # High noise: Aggressive pre-shortening
        target_rate = 0.52
        shortening_fraction = (mother_rate - target_rate) / (1.0 - target_rate)
        initial_shortening = int(shortening_fraction * frame_size)
        max_iterations = 5  # Allow more iterations for difficult channel
        delta = 0.20

    return {
        "initial_shortening": initial_shortening,
        "max_iterations": max_iterations,
        "modulation_delta": delta,
        "heuristic_qber": qber,
    }

"""
NSM entropy bounds for E-HOK protocol security analysis.

This module implements the entropy calculations required for
privacy amplification key length determination in the Noisy
Storage Model.

The key insight is that security derives from the adversary's
quantum memory being noisy—storage decoherence limits the
information an adversary can extract, even if they store qubits
during the protocol.

Key Bounds
----------
- **Dupuis-König (collision entropy)**: Better for high noise (small r)
- **Lupo (virtual erasure)**: Better for low noise (large r)
- **Max Bound**: Optimal selection of the above for all regimes
- **Rational Adversary**: Accounts for immediate measurement strategy

References
----------
- König et al. (2012), IEEE Trans. Inf. Theory 58(3)
- Schaffner et al. (2009), QIC 9(11&12)
- Lupo et al. (2023), arXiv:2308.05098
- Dupuis et al. (2014), Entanglement Sampling and Applications
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.optimize import brentq, minimize_scalar

from caligo.types.exceptions import InvalidParameterError
from caligo.utils.math import binary_entropy


# =============================================================================
# Constants
# =============================================================================

QBER_CONSERVATIVE_THRESHOLD: float = 0.11
"""
Conservative QBER limit from Schaffner Corollary 7.

Below this threshold, security is assured for most storage parameters.
This is the recommended operating limit.
"""

QBER_ABSOLUTE_THRESHOLD: float = 0.22
"""
Absolute QBER limit from Lupo Section VI.

Above this threshold, secure OT is impossible regardless of
adversary storage parameters. This is the hard security limit.
"""

R_TILDE: float = 0.7798
"""
Storage noise threshold: 2·h⁻¹(1/2) - 1.

For r ≥ r̃, the adversary's storage is "good enough" that QBER
must be below 11% for security. For r < r̃, higher QBER may be
tolerated depending on storage noise.
"""

R_CROSSOVER: float = 0.25
"""
Crossover point where Dupuis-König and Lupo bounds are equal.

The collision entropy bound h_A = Γ[1 - log₂(1 + 3r²)] equals the
virtual erasure bound h_B = 1 - r at approximately r = 0.25, where
both bounds give h_min ≈ 0.75.

- For r < R_CROSSOVER: Dupuis-König bound is tighter
- For r > R_CROSSOVER: Lupo bound is tighter

Note: For r > ~0.58, the collision entropy rate becomes negative,
causing the DK bound to approach 0, making Lupo clearly dominant.
"""

DEFAULT_EPSILON_SEC: float = 1e-10
"""Standard security parameter for finite-key analysis."""

DEFAULT_EPSILON_COR: float = 1e-6
"""Standard correctness parameter for error reconciliation."""

# Erven experimental reference values
ERVEN_STORAGE_NOISE_R: float = 0.75
"""Assumed storage parameter from Erven et al. (2014)."""

ERVEN_STORAGE_RATE_NU: float = 0.002
"""Assumed storable fraction from Erven et al. (2014)."""


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_r(r: float, name: str = "r") -> None:
    """
    Validate storage noise parameter r ∈ [0, 1].

    Parameters
    ----------
    r : float
        Storage noise parameter to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].
    """
    if not 0 <= r <= 1:
        raise InvalidParameterError(f"{name}={r} must be in [0, 1]")


def _validate_nu(nu: float, name: str = "nu") -> None:
    """
    Validate storage rate parameter ν ∈ [0, 1].

    Parameters
    ----------
    nu : float
        Storage rate parameter to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    InvalidParameterError
        If nu is not in [0, 1].
    """
    if not 0 <= nu <= 1:
        raise InvalidParameterError(f"{name}={nu} must be in [0, 1]")


def _g_function(y: float) -> float:
    """
    Compute g(y) = h(y) + y - 1 for gamma function inversion.

    Parameters
    ----------
    y : float
        Input value in (0, 1).

    Returns
    -------
    float
        Value of g(y).

    Notes
    -----
    Properties of g:
    - g(0) = -1
    - g(1/2) = 1/2
    - g(1) = 0
    - Monotonically increasing on [0, 1]
    """
    if y <= 0 or y >= 1:
        return float("inf")
    return binary_entropy(y) + y - 1.0


# =============================================================================
# Core Entropy Bound Functions
# =============================================================================


def gamma_function(x: float) -> float:
    """
    Compute the Γ function for collision entropy regularization.

    The Γ function maps collision entropy rate to min-entropy rate,
    accounting for the relationship between Rényi entropies of
    different orders.

    Parameters
    ----------
    x : float
        Collision entropy rate h₂ ∈ [0, 1].

    Returns
    -------
    float
        Regularized min-entropy rate Γ(x).

    Raises
    ------
    InvalidParameterError
        If x is not in valid range.

    Notes
    -----
    Mathematical Definition:
        Γ(x) = x           if x ≥ 1/2
        Γ(x) = g⁻¹(x)      if x < 1/2

    where g(y) = h(y) + y - 1 and h is binary entropy.

    References
    ----------
    - Lupo et al. (2023), Eq. (24)-(25)
    - Dupuis et al. (2014), Theorem 1

    Examples
    --------
    >>> gamma_function(0.7)
    0.7
    >>> gamma_function(0.5)
    0.5
    >>> 0 < gamma_function(0.3) < 0.5
    True
    """
    if x >= 0.5:
        return x

    if x <= 0:
        # g⁻¹(0) is at g(y) = 0 → y = 1
        # But we return 0 for x <= 0 as limiting case
        return 0.0

    # Numerically invert g(y) = x using Brent's method
    # g is monotonic on (0, 1) with g(0) = -1 and g(0.5) = 0.5
    try:
        result = brentq(lambda y: _g_function(y) - x, 1e-10, 0.5 - 1e-10)
        return result
    except ValueError:
        # Fallback for edge cases
        return x


def collision_entropy_rate(r: float) -> float:
    """
    Compute collision entropy rate for depolarizing storage channel.

    For a depolarizing channel that preserves state with probability r
    and completely depolarizes with probability (1-r), the collision
    entropy rate is:

        h₂ = 1 - log₂(1 + 3r²)

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
        r = 0: Complete depolarization (best for security)
        r = 1: Perfect storage (worst for security)

    Returns
    -------
    float
        Collision entropy rate h₂.

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    Range:
    - h₂(0) = 1 - log₂(1) = 1
    - h₂(1) = 1 - log₂(4) = -1 (edge case, indicates no security)

    References
    ----------
    - Lupo et al. (2023), Eq. (27)

    Examples
    --------
    >>> collision_entropy_rate(0.0)
    1.0
    >>> abs(collision_entropy_rate(0.5) - 0.678) < 0.01
    True
    >>> collision_entropy_rate(1.0)
    -1.0
    """
    _validate_r(r, "r")
    return 1.0 - math.log2(1.0 + 3.0 * r * r)


def dupuis_konig_bound(r: float) -> float:
    """
    Compute min-entropy bound from collision entropy (Dupuis-König).

    This bound is derived from the collision entropy of the stored
    quantum state after depolarizing noise:

        h_A = Γ[1 - log₂(1 + 3r²)]

    This bound is tighter for HIGH noise (small r) storage.

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].

    Returns
    -------
    float
        Min-entropy rate h_A.

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    For r approaching 1 (perfect storage), this bound approaches
    negative values and becomes less useful. Use Lupo bound instead.

    References
    ----------
    - Dupuis et al. (2014), Theorem 1
    - Lupo et al. (2023), Eq. (28)-(29)

    Examples
    --------
    >>> abs(dupuis_konig_bound(0.0) - 1.0) < 0.001
    True
    >>> abs(dupuis_konig_bound(0.1) - 0.957) < 0.01
    True
    """
    _validate_r(r, "r")
    h2 = collision_entropy_rate(r)
    return gamma_function(h2)


def lupo_virtual_erasure_bound(r: float) -> float:
    """
    Compute min-entropy bound from virtual erasure argument (Lupo).

    This bound treats depolarized qubits as virtually erased, giving
    an adversary flag information about which qubits were corrupted:

        h_B = 1 - r

    This bound is tighter for LOW noise (large r) storage.

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].

    Returns
    -------
    float
        Min-entropy rate h_B ∈ [0, 1].

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    Physical intuition: if about (1-r) fraction of qubits are
    depolarized, an adversary knows nothing about those bits,
    contributing (1-r) bits of entropy per qubit on average.

    References
    ----------
    - Lupo et al. (2023), Eq. (34)-(35)

    Examples
    --------
    >>> lupo_virtual_erasure_bound(0.0)
    1.0
    >>> lupo_virtual_erasure_bound(0.75)
    0.25
    >>> lupo_virtual_erasure_bound(1.0)
    0.0
    """
    _validate_r(r, "r")
    return 1.0 - r


def max_bound_entropy(r: float) -> float:
    """
    Compute the optimal min-entropy bound by selecting the maximum.

    The "Max Bound" extracts strictly more key than either individual
    bound by selecting the optimal one for each noise regime:

        h_min = max{ Γ[1 - log₂(1 + 3r²)], 1 - r }

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].

    Returns
    -------
    float
        Optimal min-entropy rate h_min.

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    Crossover point: The two bounds are equal at r ≈ 0.82.
    - For r < 0.82: Dupuis-König (collision) bound is better
    - For r > 0.82: Lupo (virtual erasure) bound is better

    References
    ----------
    - Lupo et al. (2023), Eq. (36)

    Examples
    --------
    >>> abs(max_bound_entropy(0.1) - 0.957) < 0.01
    True
    >>> max_bound_entropy(0.75)
    0.25
    >>> max_bound_entropy(0.9)
    0.1
    """
    _validate_r(r, "r")
    h_dk = dupuis_konig_bound(r)
    h_lupo = lupo_virtual_erasure_bound(r)
    return max(h_dk, h_lupo)


def rational_adversary_bound(r: float) -> float:
    """
    Compute min-entropy assuming a rational adversary.

    A rational adversary will not store qubits if immediate measurement
    (without waiting for basis information) yields more information.
    Measuring in random BB84 basis yields 1/2 bit of min-entropy.

        h_min^rational = min{ 1/2, max{ Γ[1-log(1+3r²)], 1-r } }

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
        r = 1: Perfect storage (no noise)
        r = 0: Complete depolarization (maximum noise)

    Returns
    -------
    float
        Min-entropy rate for rational adversary ∈ [0, 0.5].

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    For r < 0.5: Storing is irrational; honest behavior (measure
    immediately) is the best strategy, giving h_min = 0.5.

    References
    ----------
    - Lupo et al. (2023), Eq. (37)-(38)

    Examples
    --------
    >>> rational_adversary_bound(0.0)
    0.5
    >>> rational_adversary_bound(0.3)
    0.5
    >>> rational_adversary_bound(0.75)
    0.25
    """
    _validate_r(r, "r")
    max_b = max_bound_entropy(r)
    return min(0.5, max_b)


def bounded_storage_entropy(r: float, nu: float) -> float:
    """
    Compute min-entropy for noisy AND bounded quantum storage.

    When the adversary can store at most a fraction ν of received
    qubits, the entropy bound is improved:

        h_min = (1-ν)/2 + ν·max{ Γ[1-log(1+3r²)], 1-r }

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    nu : float
        Storage rate ∈ [0, 1]. Fraction of qubits adversary can store.
        ν = 0: No storage capability
        ν = 1: Can store all qubits (pure noisy storage model)

    Returns
    -------
    float
        Min-entropy rate for bounded noisy storage.

    Raises
    ------
    InvalidParameterError
        If r or nu is not in [0, 1].

    Notes
    -----
    Physical interpretation:
    - (1-ν) fraction must be measured immediately → 1/2 bit each
    - ν fraction stored with noise r → max_bound_entropy(r) bits each

    References
    ----------
    - Lupo et al. (2023), Eq. (49)
    - König et al. (2012), Corollary I.2

    Examples
    --------
    >>> bounded_storage_entropy(0.75, 1.0)  # Full storage
    0.25
    >>> bounded_storage_entropy(0.75, 0.5)  # Half storage
    0.375
    >>> bounded_storage_entropy(0.75, 0.0)  # No storage
    0.5
    """
    _validate_r(r, "r")
    _validate_nu(nu, "nu")

    h_stored = max_bound_entropy(r)
    h_immediate = 0.5  # Random basis measurement

    return (1.0 - nu) * h_immediate + nu * h_stored


def strong_converse_exponent(r: float, rate: float) -> float:
    """
    Compute strong converse error exponent for depolarizing channel.

    The error exponent γ_r(R) quantifies how fast the success
    probability of decoding decays above channel capacity:

        P_succ^{N^⊗n}(nR) ≲ 2^{-n·γ_r(R)}

    For the depolarizing channel, we optimize over α > 1.

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    rate : float
        Attempted transmission rate R (bits per qubit).

    Returns
    -------
    float
        Error exponent γ_r(R) ≥ 0.
        Returns 0 if R ≤ capacity of the channel.

    Raises
    ------
    InvalidParameterError
        If r is not in [0, 1].

    Notes
    -----
    The formula involves optimization:
        γ_r(R) = 1 + max_{α>1} [(α-1)(R-1) - log[(1+r)^α + (1-r)^α]] / α

    This is computed numerically using scipy.optimize.

    References
    ----------
    - Lupo et al. (2023), Eq. (16)
    - König et al. (2012), strong converse theorem

    Examples
    --------
    >>> strong_converse_exponent(0.5, 0.0)  # Below capacity
    0.0
    >>> strong_converse_exponent(0.5, 1.0) > 0  # Above capacity
    True
    """
    _validate_r(r, "r")

    # Channel capacity for depolarizing channel
    # C = 1 - h((1+r)/2) but simplified: if r=1, C=1; if r=0, C=0
    # For simplicity, we use the Holevo capacity approximation
    if r == 0:
        capacity = 0.0
    elif r == 1:
        capacity = 1.0
    else:
        p = (1.0 + r) / 2.0
        capacity = 1.0 - binary_entropy(p)

    if rate <= capacity:
        return 0.0

    # Optimize over α > 1
    def neg_exponent(alpha: float) -> float:
        if alpha <= 1:
            return 0.0
        # (1+r)^α + (1-r)^α computation
        term1 = (1.0 + r) ** alpha
        term2 = (1.0 - r) ** alpha
        log_term = math.log2(term1 + term2) if (term1 + term2) > 0 else 0.0

        exponent = 1.0 + ((alpha - 1.0) * (rate - 1.0) - log_term) / alpha
        return -exponent  # Negative for minimization

    # Optimize α in (1, 100] range
    result = minimize_scalar(
        neg_exponent, bounds=(1.0001, 100.0), method="bounded"
    )

    return max(0.0, -result.fun)


# =============================================================================
# Extractable Key Length
# =============================================================================


def compute_extractable_key_rate(
    r: float,
    nu: float,
    qber: float,
    ec_efficiency: float = 1.16,
) -> float:
    """
    Compute the asymptotic extractable key rate per raw bit.

    Key rate formula (asymptotic):
        λ = h_min(r, ν) - f·h(Q)

    where:
    - h_min is the bounded storage entropy
    - f is the error correction efficiency (f ≥ 1)
    - h(Q) is the binary entropy of QBER

    Parameters
    ----------
    r : float
        Storage noise parameter ∈ [0, 1].
    nu : float
        Storage rate ∈ [0, 1].
    qber : float
        Quantum bit error rate ∈ [0, 0.5].
    ec_efficiency : float
        Error correction efficiency factor f ≥ 1. Default: 1.16.

    Returns
    -------
    float
        Extractable key rate per bit. Can be negative if infeasible.

    Raises
    ------
    InvalidParameterError
        If parameters are out of valid range.

    References
    ----------
    - Lupo et al. (2023), Eq. (43)
    - Erven et al. (2014), Eq. (8)

    Examples
    --------
    >>> rate = compute_extractable_key_rate(0.75, 0.002, 0.02)
    >>> rate > 0  # Should be positive for good parameters
    True
    """
    _validate_r(r, "r")
    _validate_nu(nu, "nu")

    if not 0 <= qber <= 0.5:
        raise InvalidParameterError(f"qber={qber} must be in [0, 0.5]")
    if ec_efficiency < 1.0:
        raise InvalidParameterError(
            f"ec_efficiency={ec_efficiency} must be >= 1.0"
        )

    h_min = bounded_storage_entropy(r, nu)
    h_qber = binary_entropy(qber) if qber > 0 else 0.0
    leakage_rate = ec_efficiency * h_qber

    return h_min - leakage_rate

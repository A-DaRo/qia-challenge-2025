"""
NSM (Noisy Storage Model) Bounds Calculator.

This module provides pure-Python, deterministic, simulator-independent
implementations of the NSM security bounds for the E-HOK protocol. All
entropy calculations use base-2 logarithms as per information-theoretic
convention.

Key Components
--------------
1. **gamma_function**: The Γ regularization function mapping collision entropy
   to min-entropy (Lupo et al. 2023, Eq. 24-25).

2. **collision_entropy_rate**: Computes h₂(σ) for depolarizing channel
   (Lupo et al. 2023, Eq. 27).

3. **max_bound_entropy_rate**: The "Max Bound" combining collision-based and
   virtual-erasure bounds (Lupo et al. 2023, Eq. 36).

4. **NSMBoundsCalculator**: Stateful calculator for key-length feasibility
   and security parameter validation.

Mathematical Foundation
-----------------------
The Max Bound selects the optimal min-entropy rate for depolarizing noise:

    h_min(r) ≥ max { Γ[1 - log₂(1 + 3r²)], 1 - r }

where r ∈ [0, 1] is the storage retention probability (r=0: complete noise,
r=1: perfect storage).

The Γ function regularizes collision entropy to min-entropy:

    Γ(x) = x                if x ≥ 1/2
    Γ(x) = g⁻¹(x)          if x < 1/2

where g(y) = -y log₂(y) - (1-y) log₂(1-y) + y - 1.

References
----------
- Lupo et al. (2023): "Error-tolerant oblivious transfer in the noisy-storage
  model", arXiv:2309.xxxxx, Eqs. (24)-(27), (36).
- König et al. (2012): "Unconditional Security from Noisy Quantum Storage",
  IEEE TIT, Section I-D.
- Phase IV Analysis: docs/implementation plan/phase_IV_analysis.md

IMPORTANT: This module must NOT import NetSquid, SquidASM, or any simulator
packages. All calculations must be pure-Python and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# QBER thresholds from phase_I_analysis.md and Lupo et al. (2023) Eq. (43)
QBER_WARNING_THRESHOLD = 0.11  # Conservative bound from Schaffner et al. (2009)
QBER_HARD_LIMIT = 0.22  # Hard limit from Lupo et al. (2023)

# Numerical tolerance for float comparisons
_EPSILON = 1e-15

# Newton-Raphson iteration limits for g⁻¹(x)
_MAX_ITERATIONS = 100
_CONVERGENCE_TOLERANCE = 1e-12

# Crossover point where bounds are equal (r ≈ 0.82)
_CROSSOVER_R_APPROXIMATE = 0.82


# =============================================================================
# Feasibility Result Enum
# =============================================================================


class FeasibilityResult(Enum):
    """
    Result status for NSM bounds feasibility check.

    Attributes
    ----------
    FEASIBLE
        All parameters valid, positive key length achievable.
    INFEASIBLE_QBER_TOO_HIGH
        Adjusted QBER exceeds the 22% hard limit.
    INFEASIBLE_INSUFFICIENT_ENTROPY
        Computed max key length ≤ 0 (Death Valley).
    INFEASIBLE_INVALID_PARAMETERS
        One or more input parameters outside valid range.
    """

    FEASIBLE = auto()
    INFEASIBLE_QBER_TOO_HIGH = auto()
    INFEASIBLE_INSUFFICIENT_ENTROPY = auto()
    INFEASIBLE_INVALID_PARAMETERS = auto()


# =============================================================================
# Core Mathematical Functions
# =============================================================================


def _binary_entropy(p: float) -> float:
    """
    Compute binary Shannon entropy h(p) = -p log₂(p) - (1-p) log₂(1-p).

    Parameters
    ----------
    p : float
        Probability value in [0, 1].

    Returns
    -------
    float
        Binary entropy in bits. Returns 0 for p ∈ {0, 1}.

    Notes
    -----
    Uses base-2 logarithm as per information-theoretic convention.
    Handles edge cases p=0 and p=1 to avoid log(0).
    """
    if p <= _EPSILON or p >= 1.0 - _EPSILON:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# Public alias for binary entropy function
binary_entropy = _binary_entropy


def _g_function(y: float) -> float:
    """
    Compute g(y) = -y log₂(y) - (1-y) log₂(1-y) + y - 1.

    This is the auxiliary function whose inverse defines Γ for x < 1/2.

    Parameters
    ----------
    y : float
        Input value in (0, 1).

    Returns
    -------
    float
        g(y) = h(y) + y - 1, where h is binary entropy.

    References
    ----------
    Lupo et al. (2023) Eq. (25).
    """
    return _binary_entropy(y) + y - 1.0


def _g_derivative(y: float) -> float:
    """
    Compute g'(y) = log₂((1-y)/y) + 1.

    Derivative of g(y) for Newton-Raphson iteration.

    Parameters
    ----------
    y : float
        Input value in (0, 1).

    Returns
    -------
    float
        Derivative dg/dy.
    """
    if y <= _EPSILON or y >= 1.0 - _EPSILON:
        # Limit behavior: derivative diverges at boundaries
        return float("inf") if y <= _EPSILON else float("-inf")
    return math.log2((1 - y) / y) + 1.0


def _g_inverse(x: float) -> float:
    """
    Compute g⁻¹(x) using Newton-Raphson iteration.

    Finds y such that g(y) = x for x < 1/2.

    Parameters
    ----------
    x : float
        Target value in (-1, 1/2).

    Returns
    -------
    float
        y ∈ (0, 1) such that g(y) ≈ x.

    Raises
    ------
    ValueError
        If Newton-Raphson fails to converge.

    Notes
    -----
    g(y) is monotonically increasing on (0, 1) with:
    - g(0+) → -1
    - g(1/2) = 1/2
    - g(1-) → 0

    Initial guess uses linear interpolation between known bounds.
    """
    # Handle boundary case
    if abs(x - 0.5) < _EPSILON:
        return 0.5

    # g(y) ranges from -1 to 0 as y goes from 0+ to 1-
    # For x < 0.5, we need to find y in (0, 0.5)
    # Initial guess: linear interpolation
    # g(0+) ≈ -1, g(0.5) = 0.5
    # Use y = 0.5 * (x + 1) / 1.5 as starting point, clamped
    y = max(0.01, min(0.49, 0.5 * (x + 1.0) / 1.5))

    for iteration in range(_MAX_ITERATIONS):
        g_y = _g_function(y)
        error = g_y - x

        if abs(error) < _CONVERGENCE_TOLERANCE:
            return y

        g_prime = _g_derivative(y)
        if abs(g_prime) < _EPSILON:
            # Derivative too small, use bisection step
            if error > 0:
                y = y * 0.5
            else:
                y = (y + 0.5) * 0.5
            continue

        # Newton-Raphson step with damping
        delta = error / g_prime
        y_new = y - delta

        # Clamp to valid range (0, 1)
        y_new = max(_EPSILON, min(1.0 - _EPSILON, y_new))

        # Check for oscillation and apply damping
        if abs(y_new - y) < _CONVERGENCE_TOLERANCE:
            return y_new

        y = y_new

    raise ValueError(
        f"g_inverse failed to converge for x={x} after {_MAX_ITERATIONS} iterations"
    )


def gamma_function(x: float) -> float:
    """
    Compute Γ(x), the regularization mapping collision entropy to min-entropy.

    The Γ function is defined piecewise:
        Γ(x) = x           if x ≥ 1/2
        Γ(x) = g⁻¹(x)      if x < 1/2

    This accounts for the relationship between Rényi entropies of different
    orders when converting collision entropy bounds to min-entropy bounds.

    Parameters
    ----------
    x : float
        Collision entropy rate input.

    Returns
    -------
    float
        Min-entropy rate Γ(x).

    References
    ----------
    Lupo et al. (2023) Eq. (24): Definition of Γ function.
    Dupuis et al. (2015): Entanglement sampling and applications.

    Examples
    --------
    >>> gamma_function(0.6)  # x ≥ 0.5, returns x
    0.6
    >>> gamma_function(0.3)  # x < 0.5, returns g⁻¹(0.3)
    # Returns value y where g(y) = 0.3
    """
    if x >= 0.5:
        return x
    return _g_inverse(x)


def collision_entropy_rate(storage_noise_r: float) -> float:
    """
    Compute collision entropy rate h₂(σ) for depolarizing storage channel.

    For a depolarizing channel with parameter r, the state after storage is:
        τ = r·Ψ + (1-r)·I/2 ⊗ I/2

    The collision entropy rate is:
        h₂(σ) = 1 - log₂(1 + 3r²)

    Parameters
    ----------
    storage_noise_r : float
        Depolarizing channel retention parameter in [0, 1].
        r = 0: Complete depolarization (maximally mixed output)
        r = 1: Perfect storage (no noise)

    Returns
    -------
    float
        Collision entropy rate in bits per qubit.

    References
    ----------
    Lupo et al. (2023) Eq. (27).

    Examples
    --------
    >>> collision_entropy_rate(0.0)  # Complete noise
    1.0
    >>> collision_entropy_rate(1.0)  # Perfect storage
    -1.0  # Note: 1 - log₂(4) = 1 - 2 = -1
    """
    return 1.0 - math.log2(1.0 + 3.0 * storage_noise_r**2)


def max_bound_entropy_rate(storage_noise_r: float) -> float:
    """
    Compute NSM "Max Bound" min-entropy rate for depolarizing storage.

    The Max Bound selects the optimal bound between:
    1. Dupuis-König bound: Γ[1 - log₂(1 + 3r²)] (better for high noise, low r)
    2. Virtual erasure bound: 1 - r (better for low noise, high r)

    The final bound is:
        h_min(r) ≥ max{ Γ[h₂(r)], 1 - r }

    Parameters
    ----------
    storage_noise_r : float
        Depolarizing channel retention parameter in [0, 1].
        Interpretation: Probability that storage retains quantum state.

    Returns
    -------
    float
        Min-entropy rate per sifted bit in [0, 1].

    Raises
    ------
    ValueError
        If storage_noise_r is outside [0, 1].

    References
    ----------
    Lupo et al. (2023) Eq. (36): The Max Bound definition.
    Phase IV Analysis Section 1.3: "The Max Bound Selection".

    Examples
    --------
    >>> max_bound_entropy_rate(0.1)  # High noise
    0.957...  # Dupuis-König bound dominates
    >>> max_bound_entropy_rate(0.9)  # Low noise
    0.100     # Virtual erasure bound dominates

    Notes
    -----
    The crossover occurs at r ≈ 0.82 where both bounds are approximately equal.
    For security, higher h_min means more extractable key bits per sifted bit.
    """
    if storage_noise_r < 0.0 or storage_noise_r > 1.0:
        raise ValueError(
            f"storage_noise_r must be in [0, 1], got {storage_noise_r}"
        )

    # Bound A: Dupuis-König collision entropy bound
    h2 = collision_entropy_rate(storage_noise_r)
    bound_a = gamma_function(h2)

    # Bound B: Virtual erasure bound
    bound_b = 1.0 - storage_noise_r

    # Select the maximum (tighter bound)
    return max(bound_a, bound_b)


def channel_capacity(storage_noise_r: float) -> float:
    """
    Compute classical capacity C_N of the depolarizing channel.

    The capacity determines the security condition C_N·ν < 1/2.

    Formula (König et al. 2012, derived from King 2003):
        C_N = 1 - h((1+r)/2)

    where h is binary Shannon entropy.

    Parameters
    ----------
    storage_noise_r : float
        Depolarizing channel retention parameter in [0, 1].

    Returns
    -------
    float
        Classical capacity in bits per channel use.

    References
    ----------
    - König et al. (2012) after Corollary I.2
    - King (2003): Classical capacity of depolarizing channel
    - Phase I Analysis Eq. (14)
    """
    if storage_noise_r < 0.0 or storage_noise_r > 1.0:
        raise ValueError(
            f"storage_noise_r must be in [0, 1], got {storage_noise_r}"
        )

    p = (1.0 + storage_noise_r) / 2.0
    return 1.0 - _binary_entropy(p)


# =============================================================================
# Input/Output Dataclasses
# =============================================================================


@dataclass(frozen=True)
class NSMBoundsInputs:
    """
    Input parameters for NSM bounds calculation.

    All parameters are validated in the calculator's compute method.

    Attributes
    ----------
    storage_noise_r : float
        Depolarizing channel retention probability in [0, 1].
        From literature: r in τ = r·Ψ + (1-r)·I/2⊗I/2 (Lupo Eq. 26).
    adjusted_qber : float
        Adjusted QBER = Q_measured + μ (statistical penalty).
        Must be in [0, 0.5] as a probability.
    total_leakage_bits : int
        Total publicly revealed bits (syndrome + hash verification).
        Non-negative integer; placeholder in Sprint 1, computed in Phase III.
    epsilon_sec : float
        Security parameter, trace distance from ideal.
        Must satisfy 0 < ε_sec < 1.
    n_sifted_bits : int
        Number of sifted bits available for privacy amplification.
        Typically |I_0| - k (test bits) from Phase II.

    References
    ----------
    - sprint_1_specification.md Section 2.3 (Input definitions)
    """

    storage_noise_r: float
    adjusted_qber: float
    total_leakage_bits: int
    epsilon_sec: float
    n_sifted_bits: int


@dataclass(frozen=True)
class NSMBoundsResult:
    """
    Output of NSM bounds calculation.

    Attributes
    ----------
    min_entropy_per_bit : float
        h_min(r) from the Max Bound calculation.
    max_secure_key_length_bits : int
        Maximum extractable key length given inputs.
        May be 0 or negative if infeasible.
    feasibility_status : FeasibilityResult
        Status code indicating feasibility or reason for infeasibility.
    recommended_min_n : int | None
        For INFEASIBLE_INSUFFICIENT_ENTROPY, the minimum n for positive key.
        None if feasible or for other infeasibility reasons.

    References
    ----------
    - sprint_1_specification.md Section 2.4 (Output definitions)
    """

    min_entropy_per_bit: float
    max_secure_key_length_bits: int
    feasibility_status: FeasibilityResult
    recommended_min_n: int | None = None


# =============================================================================
# NSM Bounds Calculator
# =============================================================================


class NSMBoundsCalculator:
    """
    Calculator for NSM security bounds and key-length feasibility.

    This class provides the core security calculations for the E-HOK protocol,
    implementing the Sprint 1 key-length upper bound formula:

        ℓ_max = ⌊n · h_min(r) - L - 2·log₂(1/ε_sec)⌋

    The calculator is stateless; each compute() call is independent.

    Methods
    -------
    compute(inputs: NSMBoundsInputs) -> NSMBoundsResult
        Compute bounds and feasibility for given inputs.

    References
    ----------
    - sprint_1_specification.md Section 2 (TASK-NSM-001)
    - Lupo et al. (2023) Eq. (36) (Max Bound)
    - Phase IV Analysis Section 3.1 (key-length formula)

    Examples
    --------
    >>> calculator = NSMBoundsCalculator()
    >>> inputs = NSMBoundsInputs(
    ...     storage_noise_r=0.5,
    ...     adjusted_qber=0.05,
    ...     total_leakage_bits=1000,
    ...     epsilon_sec=1e-9,
    ...     n_sifted_bits=100000
    ... )
    >>> result = calculator.compute(inputs)
    >>> result.feasibility_status
    FeasibilityResult.FEASIBLE
    """

    def compute(self, inputs: NSMBoundsInputs) -> NSMBoundsResult:
        """
        Compute NSM bounds and feasibility status.

        Parameters
        ----------
        inputs : NSMBoundsInputs
            Validated input parameters.

        Returns
        -------
        NSMBoundsResult
            Bounds calculation results with feasibility status.

        Notes
        -----
        Validation order follows sprint_1_specification.md Section 2.5:
        1. Parameter range validation → INFEASIBLE_INVALID_PARAMETERS
        2. QBER check → INFEASIBLE_QBER_TOO_HIGH
        3. Key length check → INFEASIBLE_INSUFFICIENT_ENTROPY
        """
        # Step 1: Validate parameter ranges
        validation_error = self._validate_inputs(inputs)
        if validation_error is not None:
            logger.warning(
                "Invalid NSM bounds input: %s",
                validation_error,
            )
            return NSMBoundsResult(
                min_entropy_per_bit=0.0,
                max_secure_key_length_bits=0,
                feasibility_status=FeasibilityResult.INFEASIBLE_INVALID_PARAMETERS,
                recommended_min_n=None,
            )

        # Step 2: Check QBER against hard limit
        if inputs.adjusted_qber > QBER_HARD_LIMIT:
            logger.warning(
                "QBER %.4f exceeds hard limit %.2f",
                inputs.adjusted_qber,
                QBER_HARD_LIMIT,
            )
            return NSMBoundsResult(
                min_entropy_per_bit=0.0,
                max_secure_key_length_bits=0,
                feasibility_status=FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH,
                recommended_min_n=None,
            )

        # Step 3: Compute min-entropy rate using Max Bound
        h_min = max_bound_entropy_rate(inputs.storage_noise_r)

        # Step 4: Compute security penalty term
        # Penalty = 2·log₂(1/ε_sec) = -2·log₂(ε_sec)
        security_penalty = -2.0 * math.log2(inputs.epsilon_sec)

        # Step 5: Compute maximum secure key length
        # ℓ_max = ⌊n · h_min(r) - L - 2·log₂(1/ε_sec)⌋
        raw_key_length = (
            inputs.n_sifted_bits * h_min
            - inputs.total_leakage_bits
            - security_penalty
        )
        max_key_length = int(math.floor(raw_key_length))

        # Step 6: Check for Death Valley (insufficient entropy)
        if max_key_length <= 0:
            # Compute recommended minimum n for positive key
            recommended_n = self._compute_minimum_n(
                h_min=h_min,
                total_leakage_bits=inputs.total_leakage_bits,
                security_penalty=security_penalty,
            )
            logger.warning(
                "Insufficient entropy: ℓ_max=%d. Recommended min n: %d",
                max_key_length,
                recommended_n,
            )
            return NSMBoundsResult(
                min_entropy_per_bit=h_min,
                max_secure_key_length_bits=max_key_length,
                feasibility_status=FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY,
                recommended_min_n=recommended_n,
            )

        # Log warning if QBER is in conservative zone
        if inputs.adjusted_qber > QBER_WARNING_THRESHOLD:
            logger.info(
                "QBER %.4f exceeds conservative threshold %.2f but below hard limit",
                inputs.adjusted_qber,
                QBER_WARNING_THRESHOLD,
            )

        logger.debug(
            "NSM bounds computed: h_min=%.6f, ℓ_max=%d, n=%d",
            h_min,
            max_key_length,
            inputs.n_sifted_bits,
        )

        return NSMBoundsResult(
            min_entropy_per_bit=h_min,
            max_secure_key_length_bits=max_key_length,
            feasibility_status=FeasibilityResult.FEASIBLE,
            recommended_min_n=None,
        )

    def _validate_inputs(self, inputs: NSMBoundsInputs) -> str | None:
        """
        Validate all input parameters.

        Returns
        -------
        str | None
            Error message if validation fails, None if valid.
        """
        # storage_noise_r in [0, 1]
        if inputs.storage_noise_r < 0.0 or inputs.storage_noise_r > 1.0:
            return f"storage_noise_r must be in [0, 1], got {inputs.storage_noise_r}"

        # adjusted_qber in [0, 0.5]
        if inputs.adjusted_qber < 0.0 or inputs.adjusted_qber > 0.5:
            return f"adjusted_qber must be in [0, 0.5], got {inputs.adjusted_qber}"

        # epsilon_sec in (0, 1)
        if inputs.epsilon_sec <= 0.0 or inputs.epsilon_sec >= 1.0:
            return f"epsilon_sec must be in (0, 1), got {inputs.epsilon_sec}"

        # total_leakage_bits >= 0
        if inputs.total_leakage_bits < 0:
            return f"total_leakage_bits must be >= 0, got {inputs.total_leakage_bits}"

        # n_sifted_bits >= 0
        if inputs.n_sifted_bits < 0:
            return f"n_sifted_bits must be >= 0, got {inputs.n_sifted_bits}"

        return None

    def _compute_minimum_n(
        self,
        h_min: float,
        total_leakage_bits: int,
        security_penalty: float,
    ) -> int:
        """
        Compute minimum n required for positive key length.

        Solves for n in: n · h_min - L - penalty > 0
        → n > (L + penalty) / h_min

        Returns
        -------
        int
            Minimum n (ceiling) for positive key, or a large value if h_min ≈ 0.
        """
        if h_min < _EPSILON:
            # Cannot achieve positive key with negligible entropy rate
            return 2**30  # Effectively infinite

        min_n_float = (total_leakage_bits + security_penalty + 1) / h_min
        return int(math.ceil(min_n_float))


# =============================================================================
# Module-level convenience function
# =============================================================================


def compute_nsm_key_length(
    n_sifted_bits: int,
    storage_noise_r: float,
    total_leakage_bits: int,
    epsilon_sec: float,
    adjusted_qber: float = 0.0,
) -> Tuple[int, FeasibilityResult]:
    """
    Convenience function to compute maximum secure key length.

    Parameters
    ----------
    n_sifted_bits : int
        Number of sifted bits available.
    storage_noise_r : float
        Storage retention parameter r ∈ [0, 1].
    total_leakage_bits : int
        Total information leakage in bits.
    epsilon_sec : float
        Security parameter ε ∈ (0, 1).
    adjusted_qber : float, optional
        Adjusted QBER (default: 0.0).

    Returns
    -------
    Tuple[int, FeasibilityResult]
        (max_key_length, feasibility_status)
    """
    calculator = NSMBoundsCalculator()
    inputs = NSMBoundsInputs(
        storage_noise_r=storage_noise_r,
        adjusted_qber=adjusted_qber,
        total_leakage_bits=total_leakage_bits,
        epsilon_sec=epsilon_sec,
        n_sifted_bits=n_sifted_bits,
    )
    result = calculator.compute(inputs)
    return result.max_secure_key_length_bits, result.feasibility_status

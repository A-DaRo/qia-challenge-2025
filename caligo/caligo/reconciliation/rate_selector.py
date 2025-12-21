"""
Adaptive Rate Selection for LDPC Reconciliation.

Selects optimal code rate based on QBER estimate using the efficiency
criterion from industrial QKD post-processing.

Also computes shortening parameters for adapting fixed-size LDPC
frames to variable payload lengths.

References
----------
- Martinez-Mateo et al. (2012): Rate-adaptive reconciliation
- Kiktenko et al. (2016): Rate selection criterion
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from caligo.reconciliation import constants


# =============================================================================
# Helper Functions
# =============================================================================


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

    Notes
    -----
    Edge cases handled by convention:
    - h(0) = 0 (since lim_{x→0} x·log(x) = 0)
    - h(1) = 0
    - h(0.5) = 1 (maximum entropy)

    This is a local implementation to avoid circular imports with utils.math.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p={p} must be in [0, 1]")

    # Handle edge cases
    if p == 0 or p == 1:
        return 0.0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class RateSelection:
    """
    Rate selection result with computed parameters.

    Attributes
    ----------
    rate : float
        Selected LDPC code rate.
    n_shortened : int
        Number of shortened bits.
    n_punctured : int
        Number of punctured bits (for blind mode).
    expected_efficiency : float
        Expected reconciliation efficiency f.
    syndrome_length : int
        Expected syndrome bits: m = n(1-R).
    """

    rate: float
    n_shortened: int
    n_punctured: int
    expected_efficiency: float
    syndrome_length: int


# =============================================================================
# Core Functions
# =============================================================================

def select_rate(
    qber_estimate: float,
    available_rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    f_crit: float = constants.LDPC_F_CRIT,
) -> float:
    """
    Select optimal effective rate using reconciliation efficiency model.

    This implements the rate selection formula from Elkouss et al. (2010) [1]
    and Martinez-Mateo et al. (2012) [2], as formalized in Theoretical Report
    v2 §3.2:

        R = 1 - f(p*) × h(p*)

    where h(·) is the binary entropy function and f(·) ≥ 1 is the target
    reconciliation efficiency.

    For reconciliation, rate selection balances:
    1. Error correction capability: lower rates correct more errors
    2. Leakage efficiency: higher rates leak less information

    The selected rate ensures that the reconciliation efficiency constraint
    is satisfied: (1-R)/h(QBER) ≈ f_crit, which is equivalent to
    R ≈ 1 - f_crit × h(QBER).

    Parameters
    ----------
    qber_estimate : float
        Estimated QBER from test set.
    available_rates : Tuple[float, ...]
        Available code rates (sorted ascending).
    f_crit : float
        Target reconciliation efficiency threshold.

    Returns
    -------
    float
        Selected effective code rate.

    Notes
    -----
    Rate selection follows the mathematical framework:

    1. **Theoretical Target:** Compute ideal rate R = 1 - f_crit × h(QBER)
    2. **Practical Bounds:** Clamp to achievable range via modulation parameter
    3. **Quantization:** Round to nearest available rate from pattern library

    The achievable rate range depends on the modulation parameter δ = (p+s)/n:
        R_min = (R_0 - δ)/(1 - δ) ≤ R_eff ≤ R_0/(1 - δ) = R_max

    For mother code R_0 = 0.5:
        - With δ = 0.1: R ∈ [0.444, 0.556]  (narrow adaptation)
        - With δ = 0.44: R ∈ [0.11, 0.89]   (wide adaptation)

    Edge Cases
    ----------
    - QBER ≤ 0: Returns highest available rate (minimal correction needed)
    - QBER ≥ 0.5: Returns lowest available rate (maximum correction needed)
    - h(QBER) = 0: Returns highest rate (perfect channel)

    References
    ----------
    [1] Elkouss et al. (2010), "Rate Compatible Protocol for Information
        Reconciliation: An Application to QKD"
    [2] Martinez-Mateo et al. (2012), "Blind Reconciliation"
    [6] Kiktenko et al. (2016), "Post-processing procedure for industrial
        quantum key distribution systems"

    Examples
    --------
    >>> select_rate(qber_estimate=0.05, f_crit=1.1)
    0.71  # R ≈ 1 - 1.1 × h(0.05) ≈ 1 - 1.1 × 0.286 ≈ 0.685
    """
    # Handle edge cases first
    if qber_estimate <= 0:
        # Perfect channel → highest rate
        return float(max(available_rates)) if available_rates else 0.9
    
    if qber_estimate >= 0.5:
        # Worst-case channel → lowest rate
        return float(min(available_rates)) if available_rates else 0.5
    
    # Compute binary entropy
    entropy = binary_entropy(qber_estimate)
    
    # Apply reconciliation efficiency model: R = 1 - f × h(QBER)
    # This is the theoretical target rate from [1], [6]
    target_rate = 1.0 - f_crit * entropy
    
    # Clamp to physically achievable range
    # For mother code R_0 = 0.5, theoretical bounds are:
    #   - Lower bound: approaching 0 with very large modulation
    #   - Upper bound: approaching 1 with very large puncturing
    # Practical implementation limits based on finite-length effects
    r_min = 0.1  # Highly conservative (very noisy channels)
    r_max = 0.9  # High-rate operation (low-noise channels)
    
    target_rate = max(r_min, min(target_rate, r_max))
    
    # Quantize to nearest available rate from pattern library
    if not available_rates:
        return target_rate
    
    available = sorted(available_rates)
    
    # Find closest available rate
    closest_rate = min(available, key=lambda r: abs(r - target_rate))
    
    return float(closest_rate)


def compute_shortening(
    rate: float,
    qber_estimate: float,
    payload_length: int,
    frame_size: int = constants.LDPC_FRAME_SIZE,
    f_crit: float = constants.LDPC_F_CRIT,
) -> int:
    """
    Compute number of shortened bits for target efficiency.

    Shortening fixes bit positions to known values, effectively
    reducing the code dimension while maintaining the frame size.

    Parameters
    ----------
    rate : float
        Selected LDPC code rate.
    qber_estimate : float
        Estimated QBER.
    payload_length : int
        Desired payload bits.
    frame_size : int
        LDPC frame size n.
    f_crit : float
        Target efficiency.

    Returns
    -------
    int
        Number of bits to shorten (non-negative).

    Notes
    -----
    In the current Caligo implementation, shortening is used solely to embed a
    variable-length payload into a fixed-length LDPC frame. That means the
    number of shortened bits is fully determined by geometry:

    - payload_length + n_shortened = frame_size

    Additional “extra shortening” to meet an efficiency criterion would require
    either puncturing or dropping payload bits, which is not implemented in the
    baseline flow.
    """
    _ = (rate, qber_estimate, f_crit)  # Reserved for future rate-compatible schemes.
    return max(0, int(frame_size) - int(payload_length))


def compute_puncturing(
    base_rate: float,
    target_rate: float,
    frame_size: int = constants.LDPC_FRAME_SIZE,
) -> int:
    """
    Compute punctured bits to achieve target rate from base rate.

    This implements the puncturing fraction calculation for rate-compatible
    LDPC codes, as described in Elkouss et al. (2010) [1] and formalized in
    Theoretical Report v2 §2.1.

    Puncturing increases effective rate by treating some bit positions as
    erasures (unknown to receiver). The effective rate with puncturing only
    (no shortening) is:

        R_eff = R_0 / (1 - π)

    where π = p/n is the puncturing fraction.

    Parameters
    ----------
    base_rate : float
        Base (mother) code rate R₀.
    target_rate : float
        Desired effective rate R_eff > R₀.
    frame_size : int
        LDPC frame size n.

    Returns
    -------
    int
        Number of bits to puncture (non-negative).

    Notes
    -----
    **Derivation:**
    From R_eff = R_0/(1 - π), solving for π:

        π = 1 - R_0/R_eff
        p = ⌊π · n⌋ = ⌊n · (1 - R_0/R_eff)⌋

    **Physical Constraints:**
    - If target_rate ≤ base_rate: no puncturing needed (p = 0)
    - Maximum puncturing: p < n (cannot puncture all bits)

    **Example:**
    For R_0 = 0.5, R_eff = 0.9, n = 4096:
        π = 1 - 0.5/0.9 ≈ 0.444
        p ≈ 1820 bits punctured

    This high puncturing fraction (44%) requires Hybrid Pattern Library with
    ACE-guided puncturing for Regime B (see Theoretical Report v2 §2.2.3).

    References
    ----------
    [1] Elkouss et al. (2010), "Rate Compatible Protocol for Information
        Reconciliation: An Application to QKD"
    [3] Elkouss et al. (2012), "Untainted Puncturing for Irregular
        Low-Density Parity-Check Codes"
    [4] Liu & de Lamare (2014), "Rate-Compatible LDPC Codes Based on
        Puncturing and Extension Techniques for Short Block Lengths"
    """
    if target_rate <= base_rate:
        return 0

    # Compute puncturing fraction: π = 1 - R_0/R_eff
    puncturing_fraction = 1.0 - base_rate / target_rate
    
    # Convert to bit count: p = ⌊π · n⌋
    n_punctured = int(math.floor(puncturing_fraction * frame_size))
    
    # Physical constraint: cannot puncture all bits
    n_punctured = max(0, min(n_punctured, frame_size - 1))
    
    return n_punctured


def select_rate_with_parameters(
    qber_estimate: float,
    payload_length: int,
    frame_size: int = constants.LDPC_FRAME_SIZE,
    available_rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    f_crit: float = constants.LDPC_F_CRIT,
) -> RateSelection:
    """
    Select rate and compute all rate-compatible parameters.

    This is the primary API for baseline reconciliation setup. It combines:
    1. Rate selection via efficiency model (select_rate)
    2. Shortening computation for geometric frame packing
    3. Puncturing computation for rate adaptation
    4. Efficiency and leakage calculations

    The function implements the complete parameter pipeline from Theoretical
    Report v2 §3.2, producing all values needed for syndrome transmission
    and decoder initialization.

    Parameters
    ----------
    qber_estimate : float
        Estimated QBER from parameter estimation phase.
    payload_length : int
        Number of correlated payload bits to reconcile.
    frame_size : int
        LDPC frame size n (typically 4096).
    available_rates : Tuple[float, ...]
        Available effective rates from pattern library.
    f_crit : float
        Target reconciliation efficiency threshold.

    Returns
    -------
    RateSelection
        Complete rate selection result containing:
        - rate: Selected effective rate R_eff
        - n_shortened: Shortening bits (for frame packing)
        - n_punctured: Puncturing bits (for rate adaptation)
        - expected_efficiency: Predicted efficiency f = leak/[n·h(QBER)]
        - syndrome_length: Leakage from syndrome |s| = (1-R_0)·n

    Notes
    -----
    **Leakage Accounting (Critical for NSM Security):**

    The syndrome length is computed from the **mother code rate R_0**, not
    the effective rate R_eff:

        |syndrome| = (1 - R_0) · n

    This is because the syndrome is computed using the fixed mother matrix H,
    regardless of puncturing/shortening applied for rate adaptation [1].

    For Caligo with R_0 = 0.5 and n = 4096:
        |syndrome| = (1 - 0.5) × 4096 = 2048 bits (constant)

    **Efficiency Calculation:**

    Expected efficiency is computed as:

        f = (syndrome_length) / (n · h(QBER))
          = (1 - R_0) / h(QBER)

    Note: This is the **syndrome-only** efficiency. Total leakage includes:
        - Syndrome: (1 - R_0)·n
        - Verification hash: typically 32-128 bits
        - Test bits (parameter estimation): t bits

    For blind reconciliation, add revealed bits from iterations.

    **Puncturing Strategy:**

    In the current baseline implementation, puncturing is determined by:
        n_punctured = compute_puncturing(R_0, rate, frame_size)

    This uses the optimistic formula (shortening = 0):
        R_eff = R_0 / (1 - π)

    For hybrid patterns supporting wide rate ranges (R ∈ [0.5, 0.9]), this
    requires aggressive puncturing at high rates (π ≈ 0.44 for R = 0.9).

    References
    ----------
    [1] Elkouss et al. (2010), "Rate Compatible Protocol"
    [6] Kiktenko et al. (2016), "Post-processing procedure for industrial
        quantum key distribution systems"

    Examples
    --------
    >>> result = select_rate_with_parameters(
    ...     qber_estimate=0.05,
    ...     payload_length=3800,
    ...     frame_size=4096,
    ...     f_crit=1.1
    ... )
    >>> result.rate
    0.71  # Effective rate for QBER 5%
    >>> result.syndrome_length
    2048  # Constant: (1 - 0.5) × 4096
    >>> result.expected_efficiency
    1.59  # (1 - 0.5) / h(0.05) ≈ 0.5 / 0.286
    """
    # Step 1: Select effective rate via efficiency model
    rate = select_rate(qber_estimate, available_rates, f_crit)
    
    # Step 2: Compute shortening (geometric constraint)
    n_shortened = compute_shortening(
        rate, qber_estimate, payload_length, frame_size, f_crit
    )
    
    # Step 3: Compute puncturing (rate adaptation)
    # Mother code rate is always 0.5 for Caligo
    mother_rate = constants.MOTHER_CODE_RATE
    n_punctured = compute_puncturing(mother_rate, rate, frame_size)

    # Step 4: Calculate reconciliation efficiency
    entropy = binary_entropy(qber_estimate)
    if entropy > 0:
        # Efficiency: f = (syndrome_length) / (payload · h(QBER))
        # For mother code: syndrome_length = (1 - R_0) · n
        efficiency = (1.0 - mother_rate) / entropy
    else:
        # Perfect channel (QBER = 0): efficiency is undefined, set to 1.0
        efficiency = 1.0

    # Step 5: Syndrome length (leakage from error correction)
    # CRITICAL: Use mother_rate, not effective rate
    # Syndrome is computed via H_mother, so length = (1 - R_0) · n
    syndrome_length = int(frame_size * (1.0 - mother_rate))

    return RateSelection(
        rate=rate,
        n_shortened=n_shortened,
        n_punctured=n_punctured,
        expected_efficiency=efficiency,
        syndrome_length=syndrome_length,
    )

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


def binary_entropy(p: float) -> float:
    """
    Compute binary entropy function h(p).

    Parameters
    ----------
    p : float
        Probability in [0, 1].

    Returns
    -------
    float
        Binary entropy: h(p) = -p·log₂(p) - (1-p)·log₂(1-p).
        Returns 0.0 for boundary values p=0 or p=1.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def select_rate(
    qber_estimate: float,
    available_rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    f_crit: float = constants.LDPC_F_CRIT,
) -> float:
    """
    Select optimal rate balancing error correction and leakage.

    For reconciliation, rate selection must balance:
    1. Error correction capability: lower rates correct more errors
    2. Leakage efficiency: higher rates leak less information

    Parameters
    ----------
    qber_estimate : float
        Estimated QBER from test set.
    available_rates : Tuple[float, ...]
        Available code rates (sorted ascending).
    f_crit : float
        Critical efficiency threshold.

    Returns
    -------
    float
        Selected code rate.

    Notes
    -----
    Rate selection uses QBER-based thresholds with efficiency constraint:
    
    1. Start from QBER-based threshold (error correction capability)
    2. Ensure efficiency criterion is satisfied: (1-R)/h(QBER) < f_crit
    
    The efficiency criterion requires R > 1 - f_crit * h(QBER).
    We return the highest available rate satisfying this constraint.
    """
    entropy = binary_entropy(qber_estimate)
    
    # For perfect or near-perfect channels, use highest rate
    if entropy <= 0.0 or qber_estimate <= 0.001:
        return float(available_rates[-1])
    
    # Compute minimum rate from efficiency criterion
    # (1-R)/h(QBER) < f_crit  =>  R > 1 - f_crit * h(QBER)
    r_min_efficiency = 1.0 - f_crit * entropy
    
    # QBER-based rate thresholds from spec table (error correction guidance)
    # Format: (qber_threshold, rate) - select rate if qber < threshold
    qber_thresholds = [
        (0.015, 0.90),  # 0.0 - 1.5%
        (0.030, 0.80),  # 1.5 - 3.0%
        (0.045, 0.70),  # 3.0 - 4.5%
        (0.060, 0.60),  # 4.5 - 6.0%
        (0.080, 0.55),  # 6.0 - 8.0%
        (0.110, 0.50),  # 8.0 - 11%
    ]
    
    # Find rate from QBER table
    qber_rate = available_rates[0]  # Default lowest
    for threshold, rate in qber_thresholds:
        if qber_estimate < threshold:
            qber_rate = rate
            break
    
    # Take the maximum of QBER-based rate and efficiency-required rate
    # This ensures both error correction capability and efficiency criterion
    target_rate = max(qber_rate, r_min_efficiency)
    
    # Find lowest available rate >= target_rate
    # (we want to satisfy the constraint but not overshoot)
    for rate in available_rates:
        if rate >= target_rate:
            return float(rate)
    
    # If no rate satisfies, return highest available
    return float(available_rates[-1])


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
    Shortened bits have infinite LLR confidence, ensuring no
    errors in those positions during decoding.
    
    Basic constraint: payload + shortened = frame_size
    """
    # Basic shortening: fill frame with padding
    basic_shortened = max(0, frame_size - payload_length)
    
    entropy = binary_entropy(qber_estimate)
    if entropy <= 0.0:
        # Perfect channel: use basic shortening
        return basic_shortened

    # From Martinez-Mateo: additional shortening for efficiency target
    # n_s = n - m / (f_crit · h(QBER))
    # But we must at least have basic_shortened to fill the frame
    n_s_extra = int(math.floor(frame_size - payload_length / (f_crit * entropy)))
    n_s_extra = max(0, n_s_extra)

    # Take the maximum of basic (required) and extra (for efficiency)
    n_shortened = max(basic_shortened, n_s_extra)
    
    # Cap at frame_size - 1 (need at least 1 payload bit)
    n_shortened = min(n_shortened, frame_size - 1)

    return n_shortened


def compute_puncturing(
    base_rate: float,
    target_rate: float,
    frame_size: int = constants.LDPC_FRAME_SIZE,
) -> int:
    """
    Compute punctured bits to achieve target rate from base rate.

    Puncturing increases effective rate by treating some positions
    as unknown (random bits not transmitted).

    Parameters
    ----------
    base_rate : float
        Base LDPC code rate R₀.
    target_rate : float
        Desired effective rate R > R₀.
    frame_size : int
        LDPC frame size.

    Returns
    -------
    int
        Number of bits to puncture.

    Notes
    -----
    Effective rate with puncturing: R' = R₀ / (1 - p/n)
    Solving for p: p = n · (1 - R₀/R')
    """
    if target_rate <= base_rate:
        return 0

    # p = n · (1 - R₀/R')
    n_p = int(math.floor(frame_size * (1 - base_rate / target_rate)))
    return max(0, min(n_p, frame_size - 1))


def select_rate_with_parameters(
    qber_estimate: float,
    payload_length: int,
    frame_size: int = constants.LDPC_FRAME_SIZE,
    available_rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    f_crit: float = constants.LDPC_F_CRIT,
) -> RateSelection:
    """
    Select rate and compute all shortening parameters.

    Convenience function combining rate selection with parameter
    computation for a complete reconciliation setup.

    Parameters
    ----------
    qber_estimate : float
        Estimated QBER.
    payload_length : int
        Desired payload bits per block.
    frame_size : int
        LDPC frame size.
    available_rates : Tuple[float, ...]
        Available code rates.
    f_crit : float
        Efficiency threshold.

    Returns
    -------
    RateSelection
        Complete rate selection with parameters.
    """
    rate = select_rate(qber_estimate, available_rates, f_crit)
    n_shortened = compute_shortening(rate, qber_estimate, payload_length, frame_size, f_crit)

    # Compute efficiency
    entropy = binary_entropy(qber_estimate)
    if entropy > 0:
        efficiency = (1 - rate) / entropy
    else:
        efficiency = 1.0

    # Syndrome length for selected rate
    syndrome_length = int(frame_size * (1 - rate))

    return RateSelection(
        rate=rate,
        n_shortened=n_shortened,
        n_punctured=0,  # Baseline mode: no puncturing
        expected_efficiency=efficiency,
        syndrome_length=syndrome_length,
    )

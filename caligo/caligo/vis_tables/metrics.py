"""
Metric computation utilities for exploration results.

This module provides functions to compute derived metrics from exploration
samples, bridging the exploration module with the simulation physical models.

The metrics here enable:
- QBER estimation from sample parameters
- Theoretical bounds computation
- Distance-efficiency mappings
- DataFrame conversion for analysis
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caligo.utils.math import (
    binary_entropy,
    compute_qber_erven,
    finite_size_penalty,
)
from caligo.simulation.constants import (
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
)


# =============================================================================
# Constants for Metric Computation
# =============================================================================

# Fiber attenuation (dB/km) for distance estimation
FIBER_ATTENUATION_DB_PER_KM: float = 0.2

# Reference detection efficiency at zero distance
REFERENCE_ETA_ZERO_DISTANCE: float = 0.8


# =============================================================================
# QBER Computation
# =============================================================================


def compute_qber_from_sample(
    channel_fidelity: float,
    detector_error: float,
    detection_efficiency: float,
    dark_count_prob: float,
) -> float:
    """
    Compute total QBER from sample parameters using Erven model.

    Parameters
    ----------
    channel_fidelity : float
        Channel fidelity F ∈ (0.5, 1].
    detector_error : float
        Detector error rate e_det ∈ [0, 0.5].
    detection_efficiency : float
        Detection efficiency η ∈ (0, 1].
    dark_count_prob : float
        Dark count probability P_dark ∈ [0, 1].

    Returns
    -------
    float
        Estimated QBER.

    Notes
    -----
    Uses the Erven et al. (2014) QBER formula:
        Q = (1-F)/2 + e_det + (1-η)·P_dark/2
    """
    return compute_qber_erven(
        fidelity=channel_fidelity,
        detector_error=detector_error,
        detection_efficiency=detection_efficiency,
        dark_count_prob=dark_count_prob,
    )


def compute_qber_from_array(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute QBER for an array of samples.

    Parameters
    ----------
    X : NDArray[np.floating]
        Sample array of shape (n_samples, 9) in transformed space.
        - Index 3: channel_fidelity (linear)
        - Index 4: log10(detection_efficiency)
        - Index 5: detector_error (linear)
        - Index 6: log10(dark_count_prob)

    Returns
    -------
    NDArray[np.floating]
        QBER values for each sample.
    """
    n_samples = X.shape[0]
    qber = np.zeros(n_samples)
    
    for i in range(n_samples):
        qber[i] = compute_qber_erven(
            fidelity=X[i, 3],
            detector_error=X[i, 5],
            detection_efficiency=10 ** X[i, 4],
            dark_count_prob=10 ** X[i, 6],
        )
    
    return qber


# =============================================================================
# Theoretical Bounds
# =============================================================================


def compute_theoretical_min_n(
    qber: float,
    security_epsilon: float = 1e-10,
) -> Optional[int]:
    """
    Compute minimum block size N for secure key extraction.

    Parameters
    ----------
    qber : float
        Quantum bit error rate.
    security_epsilon : float
        Security parameter.

    Returns
    -------
    Optional[int]
        Minimum N, or None if QBER exceeds threshold.

    Notes
    -----
    Based on asymptotic keyrate formula:
        r = 1 - 2·h(Q)
    
    For positive keyrate: Q < 0.11 (11% threshold)
    
    The minimum N accounts for finite-size effects via the penalty μ.
    """
    if qber >= QBER_CONSERVATIVE_LIMIT:
        return None
    
    if qber <= 0:
        return 100  # Minimal block size
    
    # Asymptotic keyrate per bit
    h_q = binary_entropy(qber)
    r_asymp = max(0, 1 - 2 * h_q)
    
    if r_asymp <= 0:
        return None
    
    # Estimate minimum N: need enough samples such that
    # finite-size penalty doesn't kill the rate
    # Heuristic: N ≈ (log(4/ε) / r)^2
    log_term = math.log(4 / security_epsilon)
    min_n = int((log_term / r_asymp) ** 2)
    
    # Ensure reasonable bounds
    return max(100, min(min_n, 10_000_000))


def compute_theoretical_efficiency(
    qber: float,
    block_size: int,
    security_epsilon: float = 1e-10,
    reconciliation_efficiency: float = 1.0,
) -> float:
    """
    Compute theoretical net efficiency from parameters.

    Parameters
    ----------
    qber : float
        Quantum bit error rate.
    block_size : int
        Number of raw key bits N.
    security_epsilon : float
        Security parameter.
    reconciliation_efficiency : float
        Reconciliation efficiency factor.

    Returns
    -------
    float
        Theoretical efficiency ∈ [0, 1], or 0 if insecure.
    """
    if qber >= QBER_HARD_LIMIT:
        return 0.0
    
    if block_size <= 0:
        return 0.0
    
    # Binary entropy of QBER
    h_q = binary_entropy(min(qber, 0.5))
    
    # Asymptotic keyrate
    r_asymp = 1 - 2 * h_q
    
    if r_asymp <= 0:
        return 0.0
    
    # Finite-size penalty (simplified)
    # Use fraction of block for testing
    k = max(1, int(block_size * 0.1))
    n = block_size - k
    
    if n <= 0:
        return 0.0
    
    try:
        mu = finite_size_penalty(n, k, security_epsilon)
    except ValueError:
        return 0.0
    
    # Effective QBER with penalty
    q_eff = qber + mu
    if q_eff >= 0.5:
        return 0.0
    
    # Effective keyrate
    h_q_eff = binary_entropy(q_eff)
    r_eff = max(0, 1 - 2 * h_q_eff)
    
    # Apply reconciliation efficiency
    r_final = r_eff * reconciliation_efficiency
    
    return max(0.0, min(1.0, r_final))


# =============================================================================
# Distance Estimation
# =============================================================================


def detection_efficiency_to_distance(
    eta: float,
    eta_0: float = REFERENCE_ETA_ZERO_DISTANCE,
    attenuation_db_km: float = FIBER_ATTENUATION_DB_PER_KM,
) -> float:
    """
    Estimate fiber distance from detection efficiency.

    Parameters
    ----------
    eta : float
        Total detection efficiency.
    eta_0 : float
        Detection efficiency at zero distance.
    attenuation_db_km : float
        Fiber attenuation in dB/km.

    Returns
    -------
    float
        Estimated distance in km.

    Notes
    -----
    Assumes exponential attenuation:
        η = η_0 × 10^(-α·L/10)
    
    Solving for L:
        L = -10·log10(η/η_0) / α
    """
    if eta <= 0 or eta > eta_0:
        return 0.0
    
    # η = η_0 × 10^(-αL/10)
    # log10(η/η_0) = -αL/10
    # L = -10·log10(η/η_0) / α
    
    ratio = eta / eta_0
    if ratio <= 0:
        return float("inf")
    
    distance = -10 * math.log10(ratio) / attenuation_db_km
    return max(0.0, distance)


def distance_to_detection_efficiency(
    distance_km: float,
    eta_0: float = REFERENCE_ETA_ZERO_DISTANCE,
    attenuation_db_km: float = FIBER_ATTENUATION_DB_PER_KM,
) -> float:
    """
    Compute detection efficiency from fiber distance.

    Parameters
    ----------
    distance_km : float
        Fiber distance in kilometers.
    eta_0 : float
        Detection efficiency at zero distance.
    attenuation_db_km : float
        Fiber attenuation in dB/km.

    Returns
    -------
    float
        Detection efficiency η.
    """
    if distance_km <= 0:
        return eta_0
    
    # η = η_0 × 10^(-αL/10)
    loss_db = attenuation_db_km * distance_km
    eta = eta_0 * (10 ** (-loss_db / 10))
    
    return max(0.0, min(1.0, eta))


def compute_max_distance_km(
    storage_noise_r: float,
    dark_count_prob: float,
    channel_fidelity: float = 0.99,
    detector_error: float = 0.01,
    target_efficiency: float = 0.0,
) -> float:
    """
    Compute maximum distance for positive key rate.

    Parameters
    ----------
    storage_noise_r : float
        Storage noise parameter r.
    dark_count_prob : float
        Dark count probability.
    channel_fidelity : float
        Channel fidelity.
    detector_error : float
        Detector error rate.
    target_efficiency : float
        Target minimum efficiency (0 = any positive).

    Returns
    -------
    float
        Maximum distance in km, or 0 if always insecure.
    """
    # Binary search for maximum distance
    low, high = 0.0, 500.0  # km
    
    while high - low > 1.0:
        mid = (low + high) / 2
        eta = distance_to_detection_efficiency(mid)
        qber = compute_qber_erven(
            fidelity=channel_fidelity,
            detector_error=detector_error,
            detection_efficiency=eta,
            dark_count_prob=dark_count_prob,
        )
        
        # Check if QBER is within threshold
        if qber < QBER_CONSERVATIVE_LIMIT:
            low = mid
        else:
            high = mid
    
    return low


# =============================================================================
# DataFrame Conversion
# =============================================================================


def samples_to_dataframe(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    compute_derived: bool = True,
) -> pd.DataFrame:
    """
    Convert sample arrays to a pandas DataFrame.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features of shape (n_samples, 9).
    y : NDArray[np.floating]
        Target values (efficiency) of shape (n_samples,).
    compute_derived : bool
        Whether to compute derived metrics (QBER, distance).

    Returns
    -------
    pd.DataFrame
        DataFrame with named columns and derived metrics.
    """
    n_samples = X.shape[0]
    
    # Build base DataFrame
    df = pd.DataFrame({
        "storage_noise_r": X[:, 0],
        "storage_rate_nu": 10 ** X[:, 1],  # Inverse log transform
        "wait_time_ns": 10 ** X[:, 2],
        "channel_fidelity": X[:, 3],
        "detection_efficiency": 10 ** X[:, 4],
        "detector_error": X[:, 5],
        "dark_count_prob": 10 ** X[:, 6],
        "num_pairs": (10 ** X[:, 7]).astype(int),
        "strategy": np.where(X[:, 8] < 0.5, "baseline", "blind"),
        "net_efficiency": y,
    })
    
    if compute_derived:
        # Compute QBER
        df["qber"] = compute_qber_from_array(X)
        
        # Compute estimated distance
        df["distance_km"] = df["detection_efficiency"].apply(
            detection_efficiency_to_distance
        )
        
        # Flag success/failure
        df["is_success"] = df["net_efficiency"] > 0
        
        # Compute theoretical efficiency
        df["theoretical_eff"] = df.apply(
            lambda row: compute_theoretical_efficiency(
                qber=row["qber"],
                block_size=int(row["num_pairs"]),
            ),
            axis=1,
        )
        
        # Efficiency gap
        df["eff_gap"] = df["theoretical_eff"] - df["net_efficiency"]
    
    return df


def aggregate_by_qber_bins(
    df: pd.DataFrame,
    qber_bins: List[float],
) -> pd.DataFrame:
    """
    Aggregate results by QBER bins.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from samples_to_dataframe.
    qber_bins : List[float]
        QBER bin edges.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per QBER bin.
    """
    df = df.copy()
    df["qber_bin"] = pd.cut(
        df["qber"],
        bins=qber_bins,
        labels=[f"{q1:.0%}-{q2:.0%}" for q1, q2 in zip(qber_bins[:-1], qber_bins[1:])],
    )
    
    agg = df.groupby("qber_bin", observed=True).agg({
        "net_efficiency": ["mean", "std", "min", "max"],
        "num_pairs": ["min", "median", "max"],
        "is_success": ["sum", "mean"],
        "distance_km": ["mean", "max"],
    })
    
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    return agg.reset_index()


def aggregate_by_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate results by strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per strategy.
    """
    return df.groupby("strategy").agg({
        "net_efficiency": ["mean", "std", "min", "max", "count"],
        "qber": ["mean", "std"],
        "is_success": ["sum", "mean"],
    }).reset_index()

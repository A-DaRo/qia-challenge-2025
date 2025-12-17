"""
NSM-correct privacy amplification for E-HOK protocol.

This module implements the privacy amplification phase using the Noisy Storage
Model (NSM) security bounds instead of standard QKD bounds. The key insight is
that NSM provides entropy bounds based on the adversary's storage capabilities,
not channel eavesdropping.

Key Formula (Phase IV)
----------------------
    ℓ ≤ n · h_min(r) - |Σ| - 2·log₂(1/ε_sec) - Δ_finite

Where:
    - n: reconciled key length (bits)
    - h_min(r): min-entropy rate from NSM Max Bound
    - |Σ|: syndrome leakage from Phase III (wiretap cost)
    - ε_sec: security parameter
    - Δ_finite: finite-key statistical penalty

The Max Bound selects: h_min(r) ≥ max { Γ[1 - log₂(1 + 3r²)], 1 - r }

References
----------
[1] Lupo et al. (2023) Eq. (36): Max Bound definition.
[2] Dupuis et al. (2015): Entanglement sampling and min-entropy.
[3] König et al. (2012): NSM security framework.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ehok.analysis.nsm_bounds import (
    max_bound_entropy_rate,
    gamma_function,
    collision_entropy_rate,
    NSMBoundsCalculator,
    NSMBoundsInputs,
    FeasibilityResult,
)
from ehok.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NSMPrivacyAmplificationParams:
    """
    Parameters for NSM-correct privacy amplification.

    Attributes
    ----------
    reconciled_key_length : int
        Length of error-corrected key (n bits).
    storage_noise_r : float
        Adversary's storage retention parameter r ∈ [0, 1].
        r = 0: Complete depolarization (maximum security).
        r = 1: Perfect storage (minimum security, protocol unsafe).
    syndrome_leakage_bits : int
        Total syndrome bits revealed in Phase III (|Σ|).
    hash_leakage_bits : int
        Verification hash bits revealed in Phase III.
    epsilon_sec : float
        Target security parameter (trace distance from ideal).
    adjusted_qber : float
        QBER with statistical penalty (for QBER abort check only).

    Invariants
    ----------
    - reconciled_key_length > 0
    - storage_noise_r ∈ [0, 1]
    - epsilon_sec ∈ (0, 1)
    - syndrome_leakage_bits >= 0
    """

    reconciled_key_length: int
    storage_noise_r: float
    syndrome_leakage_bits: int
    hash_leakage_bits: int
    epsilon_sec: float
    adjusted_qber: float = 0.0

    def __post_init__(self) -> None:
        """Validate NSM PA parameters."""
        if self.reconciled_key_length < 0:
            raise ValueError(
                f"reconciled_key_length must be non-negative, got {self.reconciled_key_length}"
            )
        if not 0.0 <= self.storage_noise_r <= 1.0:
            raise ValueError(
                f"storage_noise_r must be in [0, 1], got {self.storage_noise_r}"
            )
        if not 0.0 < self.epsilon_sec < 1.0:
            raise ValueError(
                f"epsilon_sec must be in (0, 1), got {self.epsilon_sec}"
            )
        if self.syndrome_leakage_bits < 0:
            raise ValueError(
                f"syndrome_leakage_bits must be non-negative, got {self.syndrome_leakage_bits}"
            )
        if self.hash_leakage_bits < 0:
            raise ValueError(
                f"hash_leakage_bits must be non-negative, got {self.hash_leakage_bits}"
            )

    @property
    def total_leakage(self) -> int:
        """Total wiretap cost: syndrome + hash bits."""
        return self.syndrome_leakage_bits + self.hash_leakage_bits


@dataclass(frozen=True)
class NSMPrivacyAmplificationResult:
    """
    Result of NSM privacy amplification calculation.

    Attributes
    ----------
    secure_key_length : int
        Maximum extractable key length (≥ 0).
    min_entropy_rate : float
        h_min(r) from Max Bound calculation.
    extractable_entropy : float
        Total min-entropy: n · h_min(r).
    entropy_consumed : float
        Entropy consumed: |Σ| + 2·log₂(1/ε).
    feasibility : FeasibilityResult
        Feasibility status code.
    entropy_bound_used : str
        Which bound dominated: "dupuis_konig" or "virtual_erasure".
    """

    secure_key_length: int
    min_entropy_rate: float
    extractable_entropy: float
    entropy_consumed: float
    feasibility: FeasibilityResult
    entropy_bound_used: str


def compute_nsm_key_length(
    params: NSMPrivacyAmplificationParams,
) -> NSMPrivacyAmplificationResult:
    """
    Compute NSM-correct maximum secure key length for Phase IV.

    This function implements the key length formula from the NSM security
    proof, using the Max Bound for min-entropy calculation.

    Formula
    -------
        ℓ = ⌊n · h_min(r) - |Σ| - 2·log₂(1/ε_sec)⌋

    Where h_min(r) = max { Γ[1 - log₂(1 + 3r²)], 1 - r }

    Parameters
    ----------
    params : NSMPrivacyAmplificationParams
        All parameters needed for NSM key length calculation.

    Returns
    -------
    NSMPrivacyAmplificationResult
        Complete result with key length and entropy accounting.

    Notes
    -----
    The Death Valley guard is implemented here: if the computed key length
    is ≤ 0, we return INFEASIBLE_INSUFFICIENT_ENTROPY status rather than
    proceeding with empty key generation.

    References
    ----------
    - Lupo et al. (2023) Eq. (36): Max Bound formula.
    - sprint_3_specification.md Section 3.6: Death Valley guard.
    """
    n = params.reconciled_key_length

    # Handle edge case: empty reconciled key
    if n == 0:
        return NSMPrivacyAmplificationResult(
            secure_key_length=0,
            min_entropy_rate=0.0,
            extractable_entropy=0.0,
            entropy_consumed=0.0,
            feasibility=FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY,
            entropy_bound_used="none",
        )

    # Step 1: Compute min-entropy rate using Max Bound
    r = params.storage_noise_r
    h_min = max_bound_entropy_rate(r)

    # Determine which bound dominated
    h2_rate = collision_entropy_rate(r)
    dupuis_konig = gamma_function(h2_rate)
    virtual_erasure = 1.0 - r

    if dupuis_konig >= virtual_erasure:
        entropy_bound_used = "dupuis_konig"
    else:
        entropy_bound_used = "virtual_erasure"

    # Step 2: Compute extractable entropy
    extractable_entropy = n * h_min

    # Step 3: Compute security penalty
    import math
    security_penalty = 2.0 * math.log2(1.0 / params.epsilon_sec)

    # Step 4: Compute total entropy consumed
    entropy_consumed = params.total_leakage + security_penalty

    # Step 5: Compute maximum secure key length
    raw_length = extractable_entropy - entropy_consumed
    secure_key_length = max(0, int(math.floor(raw_length)))

    # Step 6: Determine feasibility (Death Valley guard)
    if secure_key_length <= 0:
        feasibility = FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        logger.warning(
            "Death Valley: extractable=%.1f, consumed=%.1f, key_length=%d",
            extractable_entropy,
            entropy_consumed,
            secure_key_length,
        )
    else:
        feasibility = FeasibilityResult.FEASIBLE

    logger.debug(
        "NSM PA: n=%d, r=%.3f, h_min=%.4f, leakage=%d, ε=%.2e → ℓ=%d [%s]",
        n,
        r,
        h_min,
        params.total_leakage,
        params.epsilon_sec,
        secure_key_length,
        entropy_bound_used,
    )

    return NSMPrivacyAmplificationResult(
        secure_key_length=secure_key_length,
        min_entropy_rate=h_min,
        extractable_entropy=extractable_entropy,
        entropy_consumed=entropy_consumed,
        feasibility=feasibility,
        entropy_bound_used=entropy_bound_used,
    )


def compute_minimum_n_for_positive_key(
    storage_noise_r: float,
    total_leakage_bits: int,
    epsilon_sec: float,
) -> int:
    """
    Compute minimum reconciled key length for positive key output.

    Useful for protocol design: given expected leakage and security target,
    what is the minimum key size to avoid Death Valley?

    Formula
    -------
    Solves for n in: n · h_min(r) - L - 2·log₂(1/ε) > 0
    → n > (L + 2·log₂(1/ε)) / h_min(r)

    Parameters
    ----------
    storage_noise_r : float
        Adversary's storage retention parameter r ∈ [0, 1].
    total_leakage_bits : int
        Expected total leakage from Phase III.
    epsilon_sec : float
        Target security parameter.

    Returns
    -------
    int
        Minimum n required for positive key length.
    """
    import math

    h_min = max_bound_entropy_rate(storage_noise_r)

    if h_min < 1e-10:
        # Near-perfect storage: cannot achieve positive key
        return 2**30

    security_penalty = 2.0 * math.log2(1.0 / epsilon_sec)
    min_n = (total_leakage_bits + security_penalty + 1) / h_min

    return int(math.ceil(min_n))


def validate_nsm_feasibility(
    reconciled_key_length: int,
    storage_noise_r: float,
    total_leakage_bits: int,
    epsilon_sec: float,
    adjusted_qber: float = 0.0,
) -> Tuple[bool, FeasibilityResult, int]:
    """
    Quick feasibility check for NSM privacy amplification.

    Use this before Phase IV to determine if protocol should continue
    or abort early (Death Valley guard).

    Parameters
    ----------
    reconciled_key_length : int
        Length of reconciled key.
    storage_noise_r : float
        Adversary's storage retention parameter.
    total_leakage_bits : int
        Total leakage from reconciliation.
    epsilon_sec : float
        Target security parameter.
    adjusted_qber : float, optional
        Adjusted QBER for additional checks (default: 0.0).

    Returns
    -------
    Tuple[bool, FeasibilityResult, int]
        (is_feasible, status, expected_key_length)
    """
    from ehok.analysis.nsm_bounds import QBER_HARD_LIMIT

    # Check QBER first
    if adjusted_qber > QBER_HARD_LIMIT:
        return False, FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH, 0

    params = NSMPrivacyAmplificationParams(
        reconciled_key_length=reconciled_key_length,
        storage_noise_r=storage_noise_r,
        syndrome_leakage_bits=total_leakage_bits,
        hash_leakage_bits=0,  # Already included in total_leakage_bits
        epsilon_sec=epsilon_sec,
        adjusted_qber=adjusted_qber,
    )

    result = compute_nsm_key_length(params)

    is_feasible = result.feasibility == FeasibilityResult.FEASIBLE
    return is_feasible, result.feasibility, result.secure_key_length

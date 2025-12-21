"""
Phase III Reconciliation Constants.

This module centralizes all constants for LDPC-based reconciliation,
enabling consistent configuration across encoder, decoder, and orchestrator.

Architecture
------------
The Hybrid Rate-Compatible Architecture uses a single R_0=0.5 mother code
with dynamic rate adaptation via puncturing and shortening. Rates are
generated dynamically from RATE_MIN to RATE_MAX with step RATE_STEP.

References
----------
- Martinez-Mateo et al. (2012): Blind reconciliation parameters
- Kiktenko et al. (2016): Industrial LDPC parameters
- Erven et al. (2014): E-HOK experimental implementation
- Elkouss et al. (2010): Rate-compatible LDPC for QKD
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# =============================================================================
# LDPC Code Parameters
# =============================================================================

LDPC_FRAME_SIZE: int = 4096

# =============================================================================
# Hybrid Architecture Constants
# =============================================================================

MOTHER_CODE_RATE: float = 0.5
"""Base rate of the single mother code used for all derived rates."""

RATE_STEP: float = 0.01
"""Granularity of rate adaptation."""

UNTAINTED_SATURATION_RATE: float = 0.6426
"""
Threshold rate where untainted puncturing saturates.
Rates <= this use Untainted Puncturing (Regime A).
Rates > this use ACE-Guided Puncturing (Regime B).
"""

RATE_MIN: float = 0.50
RATE_MAX: float = 0.95

# =============================================================================
# Dynamic Rate Generation
# =============================================================================


def get_available_rates() -> Tuple[float, ...]:
    """
    Generate available rates for rate-compatible reconciliation.

    Dynamically computes rates from RATE_MIN to RATE_MAX (inclusive)
    with step size RATE_STEP. This replaces the legacy static LDPC_CODE_RATES.

    Returns
    -------
    Tuple[float, ...]
        Sorted tuple of available effective rates.

    Notes
    -----
    The hybrid architecture uses a single R_0=0.5 mother code with
    puncturing/shortening for rate adaptation. Available rates are
    determined by the Hybrid Pattern Library, not by multiple pre-generated
    LDPC matrices.

    Rate coverage: R_eff ∈ [RATE_MIN, RATE_MAX] with Δ R = RATE_STEP
    For default parameters: R ∈ [0.50, 0.95] with Δ R = 0.01
    """
    rates = np.arange(RATE_MIN, RATE_MAX + RATE_STEP / 2, RATE_STEP)
    return tuple(float(round(r, 2)) for r in rates)


# Compatibility aliases for code that expects static rate tuples
LDPC_CODE_RATES: Tuple[float, ...] = get_available_rates()
"""Available LDPC code rates (dynamically generated from RATE_MIN to RATE_MAX)."""

LDPC_DEFAULT_RATE: float = MOTHER_CODE_RATE
"""Default LDPC code rate (equals mother code rate R_0 = 0.5)."""

# =============================================================================
# Numba Configuration
# =============================================================================

NUMBA_CACHE_ENABLED: bool = True
NUMBA_PARALLEL_ENABLED: bool = False
NUMBA_FASTMATH_ENABLED: bool = True

# =============================================================================
# Decoder Parameters
# =============================================================================

LDPC_MAX_ITERATIONS: int = 60
"""Maximum BP iterations per decode attempt."""

LDPC_BP_THRESHOLD: float = 1e-6
"""Message stability threshold for early stopping."""

LDPC_MAX_RETRIES: int = 2
"""Maximum retry attempts with LLR damping on decode failure."""

LDPC_LLR_SHORTENED: float = 100.0
"""LLR value for shortened bits (effectively infinite confidence)."""

# =============================================================================
# Determinism / Synchronization
# =============================================================================

SEED_OFFSET: int = 12345
"""Deterministic per-block seed offset.

Padding generation must be deterministic and synchronized between Alice and Bob.
The protocol derives per-block seeds as:

    seed = block_id + SEED_OFFSET

This keeps legacy behavior stable while removing magic numbers.
"""

# =============================================================================
# Verification Parameters
# =============================================================================

LDPC_HASH_BITS: int = 50
"""
Polynomial hash output length in bits.

Collision probability bounded by 2^{-50} ≈ 10^{-15}.
"""

LDPC_HASH_PRIME: int = 2**61 - 1
"""Mersenne prime for polynomial hash modular arithmetic."""

# =============================================================================
# Efficiency Targets
# =============================================================================

LDPC_F_CRIT: float = 1.22
"""
Critical reconciliation efficiency threshold.

Rate selection criterion: (1-R) / h(QBER) < f_crit
Industry standard value from Kiktenko et al. (2016).
"""

# =============================================================================
# Blind Reconciliation Parameters
# =============================================================================

BLIND_MAX_ITERATIONS: int = 3
"""
Default maximum iterations for blind reconciliation.

Martinez-Mateo et al. recommend t=3 for good efficiency/latency balance.
"""

BLIND_MODULATION_FRACTION: float = 0.10
"""
Default modulation fraction δ = d/n for blind reconciliation.

Determines rate coverage: R ∈ [R0-δ, R0/(1-δ)].
Higher δ covers wider QBER range but reduces baseline efficiency.
"""

# Alias for test compatibility
BLIND_DELTA_MODULATION: float = BLIND_MODULATION_FRACTION
"""Alias for BLIND_MODULATION_FRACTION (test compatibility)."""

# =============================================================================
# Matrix Generation (Offline Tools)
# =============================================================================

PEG_MAX_TREE_DEPTH: int = 20
"""Maximum BFS depth for girth optimization in PEG construction."""

PEG_DEFAULT_SEED: int = 42
"""Default seed for deterministic matrix generation."""

# =============================================================================
# File Patterns and Paths
# =============================================================================

LDPC_MATRIX_FILE_PATTERN: str = "ldpc_{frame_size}_rate{rate:.2f}.npz"
"""Filename pattern for stored LDPC matrices."""

LDPC_MATRICES_DIR: Path = Path(__file__).parent.parent / "configs" / "ldpc_matrices"
"""Default directory for pre-generated LDPC matrices."""

# Alias for backward compatibility with tests
LDPC_MATRICES_PATH: Path = LDPC_MATRICES_DIR
"""Alias for LDPC_MATRICES_DIR (backward compatibility)."""

LDPC_DEGREE_DISTRIBUTIONS_PATH: Path = (
    Path(__file__).parent.parent / "configs" / "ldpc_degree_distributions.yaml"
)
"""Path to degree distributions YAML for PEG generation."""

# =============================================================================
# Security Thresholds
# =============================================================================

QBER_RECONCILIATION_LIMIT: float = 0.11
"""
Maximum QBER for viable reconciliation.

Beyond ~11%, syndrome cost exceeds available entropy.
"""

QBER_ABORT_MARGIN: float = 0.02
"""Safety margin below QBER limit for abort decision."""

# =============================================================================
# Leakage Accounting
# =============================================================================

RATE_SELECTION_LEAKAGE_BITS: int = 4
"""
Conservative estimate for rate selection leakage.

log2(|LDPC_CODE_RATES|) ≈ 3.2 bits, rounded up.
"""

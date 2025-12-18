"""
Phase III Reconciliation Constants.

This module centralizes all constants for LDPC-based reconciliation,
enabling consistent configuration across encoder, decoder, and orchestrator.

References
----------
- Martinez-Mateo et al. (2012): Blind reconciliation parameters
- Kiktenko et al. (2016): Industrial LDPC parameters
- Erven et al. (2014): E-HOK experimental implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

# =============================================================================
# LDPC Code Parameters
# =============================================================================

LDPC_FRAME_SIZE: int = 4096
"""
Fixed LDPC frame size n in bits.

All parity-check matrices are constructed for this frame size.
Rate adaptation uses puncturing/shortening, not frame resizing.
"""

LDPC_CODE_RATES: Tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)
"""
Discrete LDPC code rates supported by the matrix pool.

Sorted ascending. Rate selection chooses highest rate satisfying
the efficiency criterion (1-R)/h(QBER) < f_crit.
"""

LDPC_DEFAULT_RATE: float = 0.50
"""Default rate used when QBER is unknown or very high."""

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

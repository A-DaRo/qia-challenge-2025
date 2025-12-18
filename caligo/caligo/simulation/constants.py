"""
Simulation constants from literature.

This module centralizes all physical and security constants
used across the simulation modules, preventing duplication.

References
----------
- König et al. (2012): NSM definition, storage capacity constraint
- Erven et al. (2014): Experimental parameters (Table I)
- Schaffner et al. (2009): 11% QBER threshold
"""

from __future__ import annotations


# =============================================================================
# Time Unit Constants (NetSquid compatibility)
# =============================================================================

NANOSECOND: float = 1.0
MICROSECOND: float = 1e3
MILLISECOND: float = 1e6
SECOND: float = 1e9

# Typical timing values
TYPICAL_DELTA_T_NS: float = 1_000_000  # 1 ms (Δt for NSM)
TYPICAL_CYCLE_TIME_NS: float = 10_000  # 10 μs (EPR generation)
TYPICAL_T1_NS: float = 10_000_000  # 10 ms (T1 relaxation)
TYPICAL_T2_NS: float = 1_000_000  # 1 ms (T2 dephasing)


# =============================================================================
# Security Thresholds
# =============================================================================

QBER_HARD_LIMIT: float = 0.22  # König et al. (2012) - absolute maximum
QBER_CONSERVATIVE_LIMIT: float = 0.11  # Schaffner et al. (2009) - practical limit


# =============================================================================
# Erven et al. (2014) Table I Experimental Values
# =============================================================================

ERVEN_MU: float = 3.145e-5  # Mean photon pair number per coherence time
ERVEN_ETA: float = 0.0150  # Total transmittance
ERVEN_E_DET: float = 0.0093  # Intrinsic detector error rate
ERVEN_P_DARK: float = 1.50e-8  # Dark count probability
ERVEN_R: float = 0.75  # Storage noise parameter
ERVEN_NU: float = 0.002  # Storage rate

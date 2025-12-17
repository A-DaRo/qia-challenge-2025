"""
Security analysis layer for E-HOK protocol.

This package implements the security validation and entropy bounds
calculations for the Noisy Storage Model (NSM). It determines whether
secure oblivious transfer is feasible given physical parameters.

Modules
-------
bounds
    NSM entropy bounds (Max Bound, Dupuis-König, Lupo, etc.).
feasibility
    Pre-flight security validation before protocol execution.
finite_key
    Finite-size statistical corrections for real-world key extraction.

Public API
----------
Entropy Bounds:
    gamma_function, collision_entropy_rate
    dupuis_konig_bound, lupo_virtual_erasure_bound
    max_bound_entropy, rational_adversary_bound, bounded_storage_entropy
    strong_converse_exponent

Feasibility Checking:
    FeasibilityChecker, FeasibilityResult, PreflightReport
    compute_expected_qber

Finite-Key Corrections:
    compute_statistical_fluctuation, hoeffding_detection_interval
    compute_finite_key_length

Constants:
    QBER_CONSERVATIVE_THRESHOLD, QBER_ABSOLUTE_THRESHOLD
    R_TILDE, R_CROSSOVER

References
----------
- König et al. (2012): NSM definition, storage capacity constraint
- Schaffner et al. (2009): 11% QBER threshold, Corollary 7
- Lupo et al. (2023): Max Bound, virtual erasure, 22% limit
- Erven et al. (2014): Experimental parameters, finite-key formula
"""

from caligo.security.bounds import (
    # Constants
    QBER_CONSERVATIVE_THRESHOLD,
    QBER_ABSOLUTE_THRESHOLD,
    R_TILDE,
    R_CROSSOVER,
    DEFAULT_EPSILON_SEC,
    DEFAULT_EPSILON_COR,
    # Core functions
    gamma_function,
    collision_entropy_rate,
    dupuis_konig_bound,
    lupo_virtual_erasure_bound,
    max_bound_entropy,
    rational_adversary_bound,
    bounded_storage_entropy,
    strong_converse_exponent,
)

from caligo.security.feasibility import (
    FeasibilityChecker,
    FeasibilityResult,
    PreflightReport,
    compute_expected_qber,
)

from caligo.security.finite_key import (
    compute_statistical_fluctuation,
    hoeffding_detection_interval,
    compute_finite_key_length,
)

__all__ = [
    # Constants
    "QBER_CONSERVATIVE_THRESHOLD",
    "QBER_ABSOLUTE_THRESHOLD",
    "R_TILDE",
    "R_CROSSOVER",
    "DEFAULT_EPSILON_SEC",
    "DEFAULT_EPSILON_COR",
    # Bounds
    "gamma_function",
    "collision_entropy_rate",
    "dupuis_konig_bound",
    "lupo_virtual_erasure_bound",
    "max_bound_entropy",
    "rational_adversary_bound",
    "bounded_storage_entropy",
    "strong_converse_exponent",
    # Feasibility
    "FeasibilityChecker",
    "FeasibilityResult",
    "PreflightReport",
    "compute_expected_qber",
    # Finite-key
    "compute_statistical_fluctuation",
    "hoeffding_detection_interval",
    "compute_finite_key_length",
]

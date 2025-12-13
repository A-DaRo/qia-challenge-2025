"""
Privacy amplification implementations.

This subpackage contains concrete implementations of the IPrivacyAmplifier
interface, including Toeplitz hashing for 2-universal hashing with rigorous
finite-key security bounds.

Sprint 3 adds NSM-correct privacy amplification using the Max Bound formula.
"""

from .toeplitz_amplifier import ToeplitzAmplifier
from .finite_key import (
    FiniteKeyParams,
    compute_final_length_finite_key,
    compute_blind_reconciliation_leakage,
    compute_final_length_blind_mode,
    binary_entropy,
    compute_statistical_fluctuation,
    estimate_qber_from_reconciliation,
    DEFAULT_EPSILON_SEC,
    DEFAULT_EPSILON_COR,
)
from .nsm_privacy_amplifier import (
    NSMPrivacyAmplificationParams,
    NSMPrivacyAmplificationResult,
    compute_nsm_key_length,
    compute_minimum_n_for_positive_key,
    validate_nsm_feasibility,
)

__all__ = [
    # Toeplitz amplifier
    "ToeplitzAmplifier",
    # QKD finite-key (legacy)
    "FiniteKeyParams",
    "compute_final_length_finite_key",
    "compute_blind_reconciliation_leakage",
    "compute_final_length_blind_mode",
    "binary_entropy",
    "compute_statistical_fluctuation",
    "estimate_qber_from_reconciliation",
    "DEFAULT_EPSILON_SEC",
    "DEFAULT_EPSILON_COR",
    # NSM privacy amplification (Sprint 3)
    "NSMPrivacyAmplificationParams",
    "NSMPrivacyAmplificationResult",
    "compute_nsm_key_length",
    "compute_minimum_n_for_positive_key",
    "validate_nsm_feasibility",
]

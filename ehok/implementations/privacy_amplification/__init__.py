"""
Privacy amplification implementations.

This subpackage contains concrete implementations of the IPrivacyAmplifier
interface, including Toeplitz hashing for 2-universal hashing with rigorous
finite-key security bounds.
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

__all__ = [
    "ToeplitzAmplifier",
    "FiniteKeyParams",
    "compute_final_length_finite_key",
    "compute_blind_reconciliation_leakage",
    "compute_final_length_blind_mode",
    "binary_entropy",
    "compute_statistical_fluctuation",
    "estimate_qber_from_reconciliation",
    "DEFAULT_EPSILON_SEC",
    "DEFAULT_EPSILON_COR",
]

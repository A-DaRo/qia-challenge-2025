"""
Abstract interfaces for E-HOK protocol components.

This subpackage defines the abstract base classes that enable modular design
and "hot-swapping" of different implementations for commitment schemes,
reconciliation algorithms, and privacy amplification methods.
"""

from ehok.interfaces.commitment import ICommitmentScheme
from ehok.interfaces.reconciliation import IReconciliator
from ehok.interfaces.privacy_amplification import IPrivacyAmplifier

__all__ = [
    "ICommitmentScheme",
    "IReconciliator",
    "IPrivacyAmplifier",
]

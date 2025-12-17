"""
Sifting package for E-HOK protocol Phase II.

This package implements the sifting operations: commitment-based
basis revelation, key sifting by basis match, and QBER estimation
with finite-size penalties.

Public API
----------
SHA256Commitment
    Cryptographic commitment scheme using SHA-256.
Sifter
    Basis sifting and I₀/I₁ key partitioning.
QBEREstimator
    QBER estimation with finite-size penalties.
DetectionValidator
    Statistical validation using Hoeffding bounds.

References
----------
- Erven et al. (2014): QBER estimation and μ penalty
- Schaffner et al. (2009): Sifting protocol definition
"""

from caligo.sifting.commitment import SHA256Commitment, CommitmentResult
from caligo.sifting.detection_validator import (
    DetectionValidator,
    ValidationResult,
    HoeffdingBound,
)
from caligo.sifting.qber import QBEREstimator, QBEREstimate
from caligo.sifting.sifter import Sifter, SiftingResult

__all__ = [
    # Commitment
    "SHA256Commitment",
    "CommitmentResult",
    # Sifting
    "Sifter",
    "SiftingResult",
    # QBER
    "QBEREstimator",
    "QBEREstimate",
    # Validation
    "DetectionValidator",
    "ValidationResult",
    "HoeffdingBound",
]

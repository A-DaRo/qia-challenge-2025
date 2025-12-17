"""
Privacy amplification package for E-HOK protocol Phase IV.

This package implements privacy amplification using Toeplitz hashing
to extract secure OT keys from reconciled key material.

Public API
----------
NSMEntropyCalculator
    Min-entropy calculation for NSM security.
SecureKeyLengthCalculator
    Secure key length from Lupo formula.
ToeplitzHasher
    Universal hash using Toeplitz matrices.
OTOutputFormatter
    Formats final OT outputs (S₀, S₁, Sᴄ).

References
----------
- Lupo et al. (2023): Key length formula Eq. (43)
- Carter & Wegman (1979): Universal hash families
- Tomamichel et al. (2011): Leftover hash lemma
"""

from caligo.amplification.entropy import NSMEntropyCalculator
from caligo.amplification.formatter import OTOutputFormatter, AliceOTOutput, BobOTOutput
from caligo.amplification.key_length import SecureKeyLengthCalculator, KeyLengthResult
from caligo.amplification.toeplitz import ToeplitzHasher

__all__ = [
    # Entropy
    "NSMEntropyCalculator",
    # Key length
    "SecureKeyLengthCalculator",
    "KeyLengthResult",
    # Hashing
    "ToeplitzHasher",
    # Formatting
    "OTOutputFormatter",
    "AliceOTOutput",
    "BobOTOutput",
]

"""
Noise estimator interface for mapping observed QBER to leakage estimates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class INoiseEstimator(ABC):
    """Estimate channel noise parameters from observed statistics."""

    @abstractmethod
    def estimate_leakage(self, qber: float, sifted_length: int) -> float:
        """Return a leakage estimate (in bits) for a given QBER."""
        raise NotImplementedError

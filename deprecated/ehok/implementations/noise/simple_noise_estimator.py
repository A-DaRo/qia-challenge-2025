"""Baseline noise estimator mapping QBER to leakage estimate."""

from __future__ import annotations

import numpy as np

from ehok.interfaces.noise_estimator import INoiseEstimator


class SimpleNoiseEstimator(INoiseEstimator):
    """Binary-entropy-based leakage approximation."""

    def estimate_leakage(self, qber: float, sifted_length: int) -> float:
        if sifted_length < 0:
            raise ValueError("sifted_length must be non-negative")
        if qber < 0 or qber > 1:
            raise ValueError("qber must be in [0,1]")

        if qber in (0.0, 1.0):
            return 0.0

        h = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        return float(sifted_length * h)

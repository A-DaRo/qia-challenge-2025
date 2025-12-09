"""
Integrated QBER estimator leveraging LDPC block results.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

import numpy as np

from ehok.core import constants
from ehok.core.data_structures import LDPCBlockResult


class IntegratedQBEREstimator:
    """
    Compute rolling QBER estimates from LDPC reconciliation outcomes.
    """

    def __init__(self, window_size: int = constants.LDPC_QBER_WINDOW_SIZE) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._window: Deque[LDPCBlockResult] = deque(maxlen=window_size)

    def estimate(self, block_results: Iterable[LDPCBlockResult]) -> float:
        """
        Estimate QBER from a collection of block results.

        Parameters
        ----------
        block_results : Iterable[LDPCBlockResult]
            Results over which to compute the estimate.

        Returns
        -------
        float
            Estimated QBER in [0, 1].
        """

        results: List[LDPCBlockResult] = list(block_results)
        if not results:
            return 0.5

        total = 0.0
        for res in results:
            if res.verified:
                if res.block_length == 0:
                    continue
                total += res.error_count / res.block_length
            else:
                total += 0.5
        return float(total / len(results))

    def update_rolling(self, new_result: LDPCBlockResult) -> float:
        """
        Add a new result to the rolling window and return updated estimate.
        """

        self._window.append(new_result)
        return self.estimate(self._window)

    @property
    def window(self) -> List[LDPCBlockResult]:
        """Copy of current window contents for inspection/testing."""
        return list(self._window)

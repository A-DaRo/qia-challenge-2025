"""
Sampling strategy interface for test/key set selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ISamplingStrategy(ABC):
    """Strategy for selecting test and key subsets from sifted indices."""

    @abstractmethod
    def select_sets(
        self,
        matched_indices: np.ndarray,
        fraction: float,
        min_size: int,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the test set and remaining key set."""
        raise NotImplementedError

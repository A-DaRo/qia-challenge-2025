"""Baseline random sampling strategy.

Wraps the legacy sifting logic in a strategy object so it can be replaced by
cut-and-choose or block-based samplers without touching protocol code.
"""

from __future__ import annotations

import numpy as np

from ehok.interfaces.sampling_strategy import ISamplingStrategy
from ehok.utils.logging import get_logger

logger = get_logger("implementations.sampling.random")


class RandomSamplingStrategy(ISamplingStrategy):
    """Uniform random sampling with optional deterministic seed."""

    def select_sets(
        self,
        matched_indices: np.ndarray,
        fraction: float,
        min_size: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(matched_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        if seed is None:
            seed = int(np.sum(matched_indices) % (2**31))

        rng = np.random.default_rng(seed)

        test_size_fraction = int(len(matched_indices) * fraction)
        test_size = max(1, min(len(matched_indices), max(test_size_fraction, min_size)))

        test_set = rng.choice(matched_indices, size=test_size, replace=False)
        test_set.sort()
        key_set = np.setdiff1d(matched_indices, test_set)

        logger.debug(
            "Sampling: |matched|=%d, |test|=%d, |key|=%d, seed=%s",
            len(matched_indices),
            len(test_set),
            len(key_set),
            seed,
        )
        return test_set, key_set

"""Security validation tests: i.i.d. preservation under parallel generation.

These tests do not attempt to re-prove security; they validate that the
implementation preserves basic i.i.d.-compatible statistics:
- basis distribution is approximately uniform
- basis choices are approximately independent over time
- outcomes show no obvious batch-boundary artifacts

Notes
-----
We use tolerant statistical checks to avoid flaky tests.
"""

from __future__ import annotations

import math

import pytest

from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
from caligo.quantum.parallel import ParallelEPRConfig


def _fraction_ones(bits: list[int]) -> float:
    if len(bits) == 0:
        return 0.0
    return float(sum(bits)) / float(len(bits))


@pytest.mark.security
class TestParallelIIDPreservation:
    """Validate parallel generation maintains i.i.d.-compatible behavior."""

    def test_basis_balance(self) -> None:
        """Bases should be close to uniform over many pairs."""
        n = 20_000
        cfg = CaligoConfig(
            num_epr_pairs=n,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=10_000,
                shuffle_results=True,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(cfg)
        strategy = factory.create_strategy()

        try:
            _, alice_bases, _, bob_bases = strategy.generate(n)
        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

        # With n=20k, 0.45..0.55 is extremely safe.
        assert 0.45 < _fraction_ones(alice_bases) < 0.55
        assert 0.45 < _fraction_ones(bob_bases) < 0.55

    def test_basis_approx_independence(self) -> None:
        """Consecutive bases should not show strong correlation."""
        n = 20_000
        cfg = CaligoConfig(
            num_epr_pairs=n,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=10_000,
                shuffle_results=True,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(cfg)
        strategy = factory.create_strategy()

        try:
            _, bases, _, _ = strategy.generate(n)
        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

        # Estimate correlation between successive bits.
        # For i.i.d. Bernoulli(0.5), correlation should be near 0.
        x = bases[:-1]
        y = bases[1:]
        mean_x = _fraction_ones(x)
        mean_y = _fraction_ones(y)

        # Compute covariance and normalize.
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
        var_x = mean_x * (1.0 - mean_x)
        var_y = mean_y * (1.0 - mean_y)

        corr = 0.0
        if var_x > 0 and var_y > 0:
            corr = cov / math.sqrt(var_x * var_y)

        # Very tolerant bound; should virtually never fail unless there's a bug.
        assert abs(corr) < 0.05

    def test_batch_boundary_mixing_outcomes(self) -> None:
        """Outcomes near batch boundaries should not have obvious bias."""
        n = 8_000
        batch = 2_000  # 4 batches
        cfg = CaligoConfig(
            num_epr_pairs=n,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=batch,
                shuffle_results=False,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(cfg)
        strategy = factory.create_strategy()

        try:
            alice_outcomes, _, _, _ = strategy.generate(n)
        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

        # Compare per-batch means to the global mean. If batching introduced
        # a systematic artifact, some segments would be noticeably biased.
        global_mean = _fraction_ones(alice_outcomes)
        segment_means = [
            _fraction_ones(alice_outcomes[i : i + batch])
            for i in range(0, n, batch)
        ]

        # Wide tolerance to avoid flakiness while still catching systematic bias.
        for seg_mean in segment_means:
            assert abs(seg_mean - global_mean) < 0.10

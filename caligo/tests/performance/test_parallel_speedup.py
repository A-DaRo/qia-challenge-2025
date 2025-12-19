"""Performance tests for parallel EPR generation.

These tests are intentionally lightweight and marked so they can be
selectively run in CI. They validate that the parallel execution path
works and that overhead remains bounded for moderate workloads.

Notes
-----
We avoid strict speedup assertions because wall-clock timings are noisy
on shared runners. Instead, we check that parallel execution is not
catastrophically slower than sequential for a moderate workload.
"""

from __future__ import annotations

import time

import pytest

from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
from caligo.quantum.parallel import ParallelEPRConfig


def _run_strategy(config: CaligoConfig, num_pairs: int) -> float:
    factory = EPRGenerationFactory(config)
    strategy = factory.create_strategy()

    start = time.perf_counter()
    try:
        _ = strategy.generate(num_pairs)
    finally:
        if isinstance(strategy, ParallelEPRStrategy):
            strategy.shutdown()
    return time.perf_counter() - start


@pytest.mark.performance
class TestParallelSpeedup:
    """Measure overhead bounds for parallel generation."""

    @pytest.mark.parametrize("num_pairs", [5_000, 10_000])
    def test_parallel_not_catastrophically_slower(self, num_pairs: int) -> None:
        """Parallel execution should not be wildly slower than sequential.

        The goal is to ensure the parallel codepath is viable and doesn't
        regress into pathological overhead.
        """

        # Keep the noise non-zero to avoid trivial all-correlated patterns.
        network_cfg = {"noise": 0.05}

        seq_cfg = CaligoConfig(
            num_epr_pairs=num_pairs,
            parallel_config=ParallelEPRConfig(enabled=False),
            network_config=network_cfg,
        )
        par_cfg = CaligoConfig(
            num_epr_pairs=num_pairs,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                # Use only 2 workers to keep process spawn overhead low
                # and reduce CI resource usage.
                num_workers=2,
                # Make ~2 batches to ensure the pool is exercised.
                pairs_per_batch=max(1, num_pairs // 2),
                shuffle_results=True,
            ),
            network_config=network_cfg,
        )

        t_seq = _run_strategy(seq_cfg, num_pairs)
        t_par = _run_strategy(par_cfg, num_pairs)

        # Be very lenient: parallel may be slower for moderate N due to
        # process start-up, but it should remain within a sane bound.
        # (If this fails, something is seriously off.)
        assert t_par < max(10.0, 5.0 * t_seq)

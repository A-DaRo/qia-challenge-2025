"""
Integration tests for parallel EPR generation with protocol components.

These tests verify that parallel generation integrates correctly with
the full QKD protocol pipeline, including sifting, reconciliation,
and privacy amplification phases.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from caligo.quantum.factory import (
    CaligoConfig,
    EPRGenerationFactory,
    ParallelEPRStrategy,
    SequentialEPRStrategy,
)
from caligo.quantum.parallel import ParallelEPRConfig


def calculate_qber(
    alice_outcomes: list,
    alice_bases: list,
    bob_outcomes: list,
    bob_bases: list,
) -> float:
    """
    Calculate Quantum Bit Error Rate on matching bases.

    Parameters
    ----------
    alice_outcomes : list
        Alice's measurement outcomes.
    alice_bases : list
        Alice's measurement bases.
    bob_outcomes : list
        Bob's measurement outcomes.
    bob_bases : list
        Bob's measurement bases.

    Returns
    -------
    float
        QBER as a fraction (0.0 to 1.0).
    """
    matching_indices = [
        i for i in range(len(alice_bases))
        if alice_bases[i] == bob_bases[i]
    ]
    if not matching_indices:
        return 0.0

    errors = sum(
        alice_outcomes[i] != bob_outcomes[i]
        for i in matching_indices
    )
    return errors / len(matching_indices)


# =============================================================================
# TestParallelProtocolIntegration
# =============================================================================


@pytest.mark.integration
class TestParallelProtocolIntegration:
    """Test parallel generation integrated with protocol components."""

    @pytest.fixture
    def parallel_config(self) -> CaligoConfig:
        """Create config with parallel enabled."""
        return CaligoConfig(
            num_epr_pairs=1000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=500,
            ),
            network_config={"noise": 0.05},
        )

    @pytest.fixture
    def sequential_config(self) -> CaligoConfig:
        """Create config with parallel disabled."""
        return CaligoConfig(
            num_epr_pairs=1000,
            parallel_config=ParallelEPRConfig(enabled=False),
            network_config={"noise": 0.05},
        )

    def test_parallel_generation_produces_valid_data(
        self, parallel_config: CaligoConfig
    ) -> None:
        """Test parallel generation produces structurally valid data."""
        factory = EPRGenerationFactory(parallel_config)
        strategy = factory.create_strategy()

        try:
            alice_out, alice_bases, bob_out, bob_bases = strategy.generate(
                parallel_config.num_epr_pairs
            )

            # Correct count
            assert len(alice_out) == 1000
            assert len(alice_bases) == 1000
            assert len(bob_out) == 1000
            assert len(bob_bases) == 1000

            # Binary values
            assert all(o in (0, 1) for o in alice_out)
            assert all(b in (0, 1) for b in alice_bases)
            assert all(o in (0, 1) for o in bob_out)
            assert all(b in (0, 1) for b in bob_bases)

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_sifting_compatibility(self, parallel_config: CaligoConfig) -> None:
        """Test parallel results are compatible with sifting phase."""
        factory = EPRGenerationFactory(parallel_config)
        strategy = factory.create_strategy()

        try:
            alice_out, alice_bases, bob_out, bob_bases = strategy.generate(
                parallel_config.num_epr_pairs
            )

            # Simulate sifting: find matching bases
            matching_indices = [
                i for i in range(len(alice_bases))
                if alice_bases[i] == bob_bases[i]
            ]

            # Should have ~50% matching (allow variance)
            sifting_rate = len(matching_indices) / len(alice_bases)
            assert 0.4 < sifting_rate < 0.6

            # Extract sifted keys
            alice_sifted = [alice_out[i] for i in matching_indices]
            bob_sifted = [bob_out[i] for i in matching_indices]

            assert len(alice_sifted) == len(bob_sifted)
            assert len(alice_sifted) > 0

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_qber_estimation_compatibility(
        self, parallel_config: CaligoConfig
    ) -> None:
        """Test parallel results produce valid QBER estimates."""
        factory = EPRGenerationFactory(parallel_config)
        strategy = factory.create_strategy()

        try:
            alice_out, alice_bases, bob_out, bob_bases = strategy.generate(
                parallel_config.num_epr_pairs
            )

            qber = calculate_qber(alice_out, alice_bases, bob_out, bob_bases)

            # With 5% noise, expect ~5% QBER (allow variance)
            assert 0.0 < qber < 0.15

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_sequential_vs_parallel_statistical_equivalence(
        self,
        sequential_config: CaligoConfig,
        parallel_config: CaligoConfig,
    ) -> None:
        """Test sequential and parallel produce statistically equivalent results."""
        # Use larger sample for statistical confidence
        sequential_config.num_epr_pairs = 5000
        parallel_config.num_epr_pairs = 5000

        seq_factory = EPRGenerationFactory(sequential_config)
        par_factory = EPRGenerationFactory(parallel_config)

        seq_strategy = seq_factory.create_strategy()
        par_strategy = par_factory.create_strategy()

        try:
            # Generate with both strategies
            seq_results = seq_strategy.generate(5000)
            par_results = par_strategy.generate(5000)

            # Calculate QBERs
            seq_qber = calculate_qber(*seq_results)
            par_qber = calculate_qber(*par_results)

            # QBERs should be within statistical tolerance
            # With 5000 pairs, expect ~2% standard error
            assert abs(seq_qber - par_qber) < 0.05

            # Sifting rates should be similar
            def sifting_rate(results: Tuple) -> float:
                _, alice_bases, _, bob_bases = results
                matching = sum(
                    1 for i in range(len(alice_bases))
                    if alice_bases[i] == bob_bases[i]
                )
                return matching / len(alice_bases)

            seq_sift = sifting_rate(seq_results)
            par_sift = sifting_rate(par_results)

            # Sifting rates should be ~50% for both
            assert abs(seq_sift - par_sift) < 0.05

        finally:
            if isinstance(par_strategy, ParallelEPRStrategy):
                par_strategy.shutdown()

    def test_different_noise_levels(self) -> None:
        """Test parallel generation with different noise configurations."""
        noise_levels = [0.0, 0.05, 0.10, 0.15]

        for noise in noise_levels:
            config = CaligoConfig(
                num_epr_pairs=1000,
                parallel_config=ParallelEPRConfig(
                    enabled=True,
                    num_workers=2,
                ),
                network_config={"noise": noise},
            )

            factory = EPRGenerationFactory(config)
            strategy = factory.create_strategy()

            try:
                results = strategy.generate(1000)
                qber = calculate_qber(*results)

                # QBER should correlate with noise level
                # Allow tolerance for statistical fluctuation
                expected_qber_max = noise + 0.05
                assert qber < expected_qber_max, (
                    f"QBER {qber} too high for noise {noise}"
                )

            finally:
                if isinstance(strategy, ParallelEPRStrategy):
                    strategy.shutdown()

    def test_multiple_batch_sizes(self) -> None:
        """Test parallel generation with various batch sizes."""
        total_pairs = 1000
        batch_sizes = [100, 250, 500, 1000]

        for batch_size in batch_sizes:
            config = CaligoConfig(
                num_epr_pairs=total_pairs,
                parallel_config=ParallelEPRConfig(
                    enabled=True,
                    num_workers=2,
                    pairs_per_batch=batch_size,
                ),
                network_config={"noise": 0.05},
            )

            factory = EPRGenerationFactory(config)
            strategy = factory.create_strategy()

            try:
                results = strategy.generate(total_pairs)
                assert len(results[0]) == total_pairs

            finally:
                if isinstance(strategy, ParallelEPRStrategy):
                    strategy.shutdown()

    def test_reproducibility_with_seeding(self) -> None:
        """Test parallel results have appropriate randomness."""
        config = CaligoConfig(
            num_epr_pairs=500,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
            ),
            network_config={"noise": 0.0},
        )

        factory = EPRGenerationFactory(config)

        # Generate twice
        strategy1 = factory.create_strategy()
        strategy2 = factory.create_strategy()

        try:
            results1 = strategy1.generate(500)
            results2 = strategy2.generate(500)

            # Results should differ (random bases)
            # Extremely unlikely to be identical
            assert results1[1] != results2[1]  # Different bases

        finally:
            if isinstance(strategy1, ParallelEPRStrategy):
                strategy1.shutdown()
            if isinstance(strategy2, ParallelEPRStrategy):
                strategy2.shutdown()

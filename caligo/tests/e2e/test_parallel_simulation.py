"""
End-to-end tests for parallel EPR generation simulation.

These tests run full parallel generation scenarios with realistic
parameters, verifying the complete pipeline from configuration
through result aggregation.
"""

from __future__ import annotations

import pytest

from caligo.quantum.factory import (
    CaligoConfig,
    EPRGenerationFactory,
    ParallelEPRStrategy,
)
from caligo.quantum.parallel import ParallelEPRConfig


def calculate_qber(
    alice_outcomes: list,
    alice_bases: list,
    bob_outcomes: list,
    bob_bases: list,
) -> float:
    """Calculate Quantum Bit Error Rate on matching bases."""
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
# TestParallelSimulationE2E
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
class TestParallelSimulationE2E:
    """End-to-end tests for parallel simulation."""

    def test_10k_pairs_parallel_4_workers(self) -> None:
        """Generate 10k EPR pairs using 4 workers."""
        config = CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=2500,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(total_pairs=10000)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            # Validate result count
            assert len(alice_outcomes) == 10000
            assert len(alice_bases) == 10000
            assert len(bob_outcomes) == 10000
            assert len(bob_bases) == 10000

            # Both bases should be used
            assert set(alice_bases) == {0, 1}
            assert set(bob_bases) == {0, 1}

            # Outcomes should be roughly 50% ones
            alice_ratio = sum(alice_outcomes) / len(alice_outcomes)
            assert 0.4 < alice_ratio < 0.6

            # QBER should be reasonable
            qber = calculate_qber(
                alice_outcomes, alice_bases, bob_outcomes, bob_bases
            )
            assert 0.0 < qber < 0.15

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_50k_pairs_parallel_8_workers(self) -> None:
        """Generate 50k EPR pairs using 8 workers."""
        config = CaligoConfig(
            num_epr_pairs=50000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=8,
                pairs_per_batch=6250,  # 50000 / 8
            ),
            network_config={"noise": 0.03},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(total_pairs=50000)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            assert len(alice_outcomes) == 50000

            qber = calculate_qber(
                alice_outcomes, alice_bases, bob_outcomes, bob_bases
            )
            # With 3% noise, expect ~3% QBER
            assert 0.0 < qber < 0.10

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_realistic_noise_model(self) -> None:
        """Test parallel generation with realistic channel noise."""
        # Simulate 10 km fiber with 0.2 dB/km loss
        # Approximate this as ~5% depolarizing noise
        config = CaligoConfig(
            num_epr_pairs=5000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=2500,
            ),
            network_config={
                "noise": 0.05,
                "distance_km": 10,
                "fiber_loss_db_per_km": 0.2,
            },
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(5000)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            # Calculate empirical QBER
            qber = calculate_qber(
                alice_outcomes, alice_bases, bob_outcomes, bob_bases
            )

            # Realistic QBER range for practical QKD
            assert 0.01 < qber < 0.15

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_ideal_channel(self) -> None:
        """Test parallel generation with ideal (noiseless) channel."""
        config = CaligoConfig(
            num_epr_pairs=5000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
            ),
            network_config={"noise": 0.0},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(5000)

            qber = calculate_qber(*results)

            # Ideal channel should have zero QBER
            assert qber == 0.0

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_high_noise_channel(self) -> None:
        """Test parallel generation with high noise (near threshold)."""
        # 11% is the security threshold for BB84
        config = CaligoConfig(
            num_epr_pairs=5000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
            ),
            network_config={"noise": 0.10},  # 10% depolarizing noise
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(5000)

            qber = calculate_qber(*results)

            # Should have ~10% QBER
            assert 0.05 < qber < 0.20

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_uneven_batch_distribution(self) -> None:
        """Test with pair count not evenly divisible by batch size."""
        # 10000 pairs with 3000 per batch = 4 batches (3000+3000+3000+1000)
        config = CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=3000,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(10000)

            # Should still have exactly 10000 pairs
            assert len(results[0]) == 10000

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_single_worker_large_batch(self) -> None:
        """Test single worker with large batch (sequential-like)."""
        config = CaligoConfig(
            num_epr_pairs=5000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=1,
                pairs_per_batch=5000,
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(5000)
            assert len(results[0]) == 5000

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_many_small_batches(self) -> None:
        """Test many small batches (high parallelism)."""
        config = CaligoConfig(
            num_epr_pairs=1000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=50,  # 20 batches
            ),
            network_config={"noise": 0.05},
        )

        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(1000)
            assert len(results[0]) == 1000

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

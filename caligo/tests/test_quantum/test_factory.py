"""
Unit tests for EPR generation factory pattern.

Tests cover:
- EPRGenerationFactory strategy creation
- SequentialEPRStrategy generation
- ParallelEPRStrategy generation
- Strategy protocol compliance
- CaligoConfig dataclass
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import Mock, patch

import pytest

from caligo.quantum.factory import (
    EPRGenerationFactory,
    EPRGenerationStrategy,
    SequentialEPRStrategy,
    ParallelEPRStrategy,
    CaligoConfig,
)
from caligo.quantum.parallel import ParallelEPRConfig


# =============================================================================
# TestCaligoConfig
# =============================================================================


class TestCaligoConfig:
    """Test CaligoConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CaligoConfig(num_epr_pairs=1000)

        assert config.num_epr_pairs == 1000
        assert config.parallel_config.enabled is False
        assert config.network_config == {}
        assert config.security_epsilon == 1e-10

    def test_custom_parallel_config(self) -> None:
        """Test custom parallel configuration."""
        parallel_config = ParallelEPRConfig(enabled=True, num_workers=8)
        config = CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=parallel_config,
        )

        assert config.parallel_config.enabled is True
        assert config.parallel_config.num_workers == 8

    def test_custom_network_config(self) -> None:
        """Test custom network configuration."""
        network_config = {"noise": 0.05, "distance_km": 10}
        config = CaligoConfig(
            num_epr_pairs=1000,
            network_config=network_config,
        )

        assert config.network_config["noise"] == 0.05
        assert config.network_config["distance_km"] == 10

    def test_custom_security_epsilon(self) -> None:
        """Test custom security parameter."""
        config = CaligoConfig(
            num_epr_pairs=1000,
            security_epsilon=1e-6,
        )

        assert config.security_epsilon == 1e-6


# =============================================================================
# TestEPRGenerationFactory
# =============================================================================


class TestEPRGenerationFactory:
    """Test EPRGenerationFactory strategy creation."""

    @pytest.fixture
    def sequential_caligo_config(self) -> CaligoConfig:
        """Config with parallel disabled."""
        return CaligoConfig(
            num_epr_pairs=1000,
            parallel_config=ParallelEPRConfig(enabled=False),
            network_config={"noise": 0.05},
        )

    @pytest.fixture
    def parallel_caligo_config(self) -> CaligoConfig:
        """Config with parallel enabled."""
        return CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=2500,
            ),
            network_config={"noise": 0.05},
        )

    def test_create_sequential_strategy_when_disabled(
        self, sequential_caligo_config: CaligoConfig
    ) -> None:
        """Test factory returns sequential strategy when parallel disabled."""
        factory = EPRGenerationFactory(sequential_caligo_config)
        strategy = factory.create_strategy()

        assert isinstance(strategy, SequentialEPRStrategy)

    def test_create_parallel_strategy_when_enabled(
        self, parallel_caligo_config: CaligoConfig
    ) -> None:
        """Test factory returns parallel strategy when parallel enabled."""
        factory = EPRGenerationFactory(parallel_caligo_config)
        strategy = factory.create_strategy()

        assert isinstance(strategy, ParallelEPRStrategy)

        # Clean up
        strategy.shutdown()

    def test_create_sequential_explicit(
        self, parallel_caligo_config: CaligoConfig
    ) -> None:
        """Test create_sequential() returns sequential even when parallel config."""
        factory = EPRGenerationFactory(parallel_caligo_config)
        strategy = factory.create_sequential()

        assert isinstance(strategy, SequentialEPRStrategy)

    def test_create_parallel_explicit(
        self, sequential_caligo_config: CaligoConfig
    ) -> None:
        """Test create_parallel() returns parallel even when sequential config."""
        factory = EPRGenerationFactory(sequential_caligo_config)
        strategy = factory.create_parallel()

        assert isinstance(strategy, ParallelEPRStrategy)

        # Clean up
        strategy.shutdown()

    def test_factory_passes_network_config(
        self, sequential_caligo_config: CaligoConfig
    ) -> None:
        """Test factory passes network config to strategy."""
        factory = EPRGenerationFactory(sequential_caligo_config)
        strategy = factory.create_sequential()

        assert strategy._network_config == {"noise": 0.05}


# =============================================================================
# TestSequentialEPRStrategy
# =============================================================================


class TestSequentialEPRStrategy:
    """Test SequentialEPRStrategy generation."""

    @pytest.fixture
    def network_config(self) -> dict:
        """Network configuration fixture."""
        return {"noise": 0.05}

    @pytest.fixture
    def strategy(self, network_config: dict) -> SequentialEPRStrategy:
        """Create sequential strategy."""
        return SequentialEPRStrategy(network_config)

    def test_generate_returns_four_lists(
        self, strategy: SequentialEPRStrategy
    ) -> None:
        """Test generate returns tuple of four lists."""
        results = strategy.generate(total_pairs=100)

        assert isinstance(results, tuple)
        assert len(results) == 4
        assert all(isinstance(r, list) for r in results)

    def test_generate_correct_count(
        self, strategy: SequentialEPRStrategy
    ) -> None:
        """Test generate returns correct number of pairs."""
        results = strategy.generate(total_pairs=100)
        alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

        assert len(alice_outcomes) == 100
        assert len(alice_bases) == 100
        assert len(bob_outcomes) == 100
        assert len(bob_bases) == 100

    def test_generate_binary_values(
        self, strategy: SequentialEPRStrategy
    ) -> None:
        """Test generate returns binary values."""
        results = strategy.generate(total_pairs=100)
        alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

        assert all(o in (0, 1) for o in alice_outcomes)
        assert all(b in (0, 1) for b in alice_bases)
        assert all(o in (0, 1) for o in bob_outcomes)
        assert all(b in (0, 1) for b in bob_bases)

    def test_generate_invalid_pairs_raises(
        self, strategy: SequentialEPRStrategy
    ) -> None:
        """Test generate with invalid pairs raises ValueError."""
        with pytest.raises(ValueError, match="total_pairs must be > 0"):
            strategy.generate(total_pairs=0)

        with pytest.raises(ValueError, match="total_pairs must be > 0"):
            strategy.generate(total_pairs=-10)

    def test_generate_noise_affects_qber(self, network_config: dict) -> None:
        """Test noise parameter affects QBER."""
        strategy_ideal = SequentialEPRStrategy({"noise": 0.0})
        strategy_noisy = SequentialEPRStrategy({"noise": 0.15})

        results_ideal = strategy_ideal.generate(1000)
        results_noisy = strategy_noisy.generate(1000)

        def calc_qber(results: Tuple) -> float:
            alice_out, alice_bases, bob_out, bob_bases = results
            matching = [
                i for i in range(len(alice_out))
                if alice_bases[i] == bob_bases[i]
            ]
            if not matching:
                return 0.0
            errors = sum(alice_out[i] != bob_out[i] for i in matching)
            return errors / len(matching)

        qber_ideal = calc_qber(results_ideal)
        qber_noisy = calc_qber(results_noisy)

        assert qber_ideal < 0.05  # Near-ideal
        assert qber_noisy > qber_ideal  # Noisy should be worse


# =============================================================================
# TestParallelEPRStrategy
# =============================================================================


class TestParallelEPRStrategy:
    """Test ParallelEPRStrategy generation."""

    @pytest.fixture
    def parallel_config(self) -> ParallelEPRConfig:
        """Parallel configuration fixture."""
        return ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=50,
        )

    @pytest.fixture
    def network_config(self) -> dict:
        """Network configuration fixture."""
        return {"noise": 0.05}

    @pytest.fixture
    def strategy(
        self, parallel_config: ParallelEPRConfig, network_config: dict
    ) -> ParallelEPRStrategy:
        """Create parallel strategy."""
        return ParallelEPRStrategy(parallel_config, network_config)

    def test_generate_returns_four_lists(
        self, strategy: ParallelEPRStrategy
    ) -> None:
        """Test generate returns tuple of four lists."""
        try:
            results = strategy.generate(total_pairs=100)

            assert isinstance(results, tuple)
            assert len(results) == 4
            assert all(isinstance(r, list) for r in results)
        finally:
            strategy.shutdown()

    def test_generate_correct_count(
        self, strategy: ParallelEPRStrategy
    ) -> None:
        """Test generate returns correct number of pairs."""
        try:
            results = strategy.generate(total_pairs=100)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            assert len(alice_outcomes) == 100
            assert len(alice_bases) == 100
            assert len(bob_outcomes) == 100
            assert len(bob_bases) == 100
        finally:
            strategy.shutdown()

    def test_shutdown_releases_resources(
        self, strategy: ParallelEPRStrategy
    ) -> None:
        """Test shutdown releases worker resources."""
        strategy.generate(total_pairs=50)
        strategy.shutdown()

        # Executor should be None after shutdown
        assert strategy._orchestrator._executor is None

    def test_context_manager(
        self, parallel_config: ParallelEPRConfig, network_config: dict
    ) -> None:
        """Test strategy as context manager."""
        with ParallelEPRStrategy(parallel_config, network_config) as strategy:
            results = strategy.generate(100)
            assert len(results[0]) == 100
        # Resources should be released after exit


# =============================================================================
# TestStrategyInterface
# =============================================================================


class TestStrategyInterface:
    """Test strategy implementations follow protocol."""

    def test_sequential_implements_protocol(self) -> None:
        """Test SequentialEPRStrategy implements EPRGenerationStrategy."""
        strategy = SequentialEPRStrategy({"noise": 0.0})

        # Protocol check via isinstance with runtime_checkable
        assert isinstance(strategy, EPRGenerationStrategy)

    def test_parallel_implements_protocol(self) -> None:
        """Test ParallelEPRStrategy implements EPRGenerationStrategy."""
        config = ParallelEPRConfig(enabled=True, num_workers=2)
        strategy = ParallelEPRStrategy(config, {"noise": 0.0})

        assert isinstance(strategy, EPRGenerationStrategy)

        strategy.shutdown()

    def test_strategies_have_same_signature(self) -> None:
        """Test both strategies have identical generate() signature."""
        seq_strategy = SequentialEPRStrategy({"noise": 0.0})
        par_config = ParallelEPRConfig(enabled=True, num_workers=2)
        par_strategy = ParallelEPRStrategy(par_config, {"noise": 0.0})

        try:
            # Both should accept total_pairs parameter
            seq_results = seq_strategy.generate(total_pairs=50)
            par_results = par_strategy.generate(total_pairs=50)

            # Both should return same structure
            assert len(seq_results) == len(par_results) == 4
            assert len(seq_results[0]) == len(par_results[0]) == 50
        finally:
            par_strategy.shutdown()

    def test_polymorphic_usage(self) -> None:
        """Test strategies can be used polymorphically."""
        def use_strategy(strategy: EPRGenerationStrategy, n: int) -> int:
            results = strategy.generate(n)
            return len(results[0])

        seq = SequentialEPRStrategy({"noise": 0.0})
        par_config = ParallelEPRConfig(enabled=True, num_workers=2)
        par = ParallelEPRStrategy(par_config, {"noise": 0.0})

        try:
            assert use_strategy(seq, 100) == 100
            assert use_strategy(par, 100) == 100
        finally:
            par.shutdown()


# =============================================================================
# TestFactoryWithStrategies
# =============================================================================


class TestFactoryWithStrategies:
    """Test factory creates working strategies."""

    def test_factory_sequential_generates_data(self) -> None:
        """Test factory-created sequential strategy generates data."""
        config = CaligoConfig(
            num_epr_pairs=100,
            parallel_config=ParallelEPRConfig(enabled=False),
            network_config={"noise": 0.0},
        )
        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        results = strategy.generate(config.num_epr_pairs)
        assert len(results[0]) == 100

    def test_factory_parallel_generates_data(self) -> None:
        """Test factory-created parallel strategy generates data."""
        config = CaligoConfig(
            num_epr_pairs=100,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=50,
            ),
            network_config={"noise": 0.0},
        )
        factory = EPRGenerationFactory(config)
        strategy = factory.create_strategy()

        try:
            results = strategy.generate(config.num_epr_pairs)
            assert len(results[0]) == 100
        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

    def test_end_to_end_workflow(self) -> None:
        """Test complete workflow from config to results."""
        # 1. Create configuration
        config = CaligoConfig(
            num_epr_pairs=500,
            parallel_config=ParallelEPRConfig(enabled=True, num_workers=2),
            network_config={"noise": 0.03},
        )

        # 2. Create factory
        factory = EPRGenerationFactory(config)

        # 3. Get strategy (automatic selection)
        strategy = factory.create_strategy()

        try:
            # 4. Generate EPR pairs
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = strategy.generate(
                config.num_epr_pairs
            )

            # 5. Verify results
            assert len(alice_outcomes) == 500
            assert len(alice_bases) == 500
            assert len(bob_outcomes) == 500
            assert len(bob_bases) == 500

            # 6. Verify statistical properties
            # Both bases should be used
            assert set(alice_bases) == {0, 1}
            assert set(bob_bases) == {0, 1}

            # Outcomes should be binary
            assert all(o in (0, 1) for o in alice_outcomes)
            assert all(o in (0, 1) for o in bob_outcomes)

        finally:
            if isinstance(strategy, ParallelEPRStrategy):
                strategy.shutdown()

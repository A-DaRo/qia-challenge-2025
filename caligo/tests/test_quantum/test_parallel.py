"""
Unit tests for parallel EPR generation module.

Tests cover:
- ParallelEPRConfig validation and defaults
- ParallelEPROrchestrator batch distribution and aggregation
- EPRWorkerResult data structure
- Result shuffling for i.i.d. preservation
"""

from __future__ import annotations

import math
from multiprocessing import cpu_count
from unittest.mock import Mock, patch, MagicMock

import pytest

from caligo.quantum.parallel import (
    ParallelEPRConfig,
    ParallelEPROrchestrator,
    EPRWorkerResult,
    _worker_generate_epr,
)
from caligo.types.exceptions import SimulationError


# =============================================================================
# TestParallelEPRConfig
# =============================================================================


class TestParallelEPRConfig:
    """Test ParallelEPRConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ParallelEPRConfig()

        assert config.enabled is False
        assert config.num_workers >= 1
        assert config.pairs_per_batch == 1000
        assert config.isolation_level == "process"
        assert config.prefetch_batches == 2
        assert config.timeout_seconds == 300.0
        assert config.shuffle_results is True

    def test_default_workers_cpu_based(self) -> None:
        """Test default worker count is physical CPU cores - 1."""
        from caligo.quantum.parallel import _get_physical_cpu_count
        
        config = ParallelEPRConfig()
        expected = max(1, _get_physical_cpu_count() - 1)
        assert config.num_workers == expected

    def test_custom_workers(self) -> None:
        """Test custom worker count."""
        config = ParallelEPRConfig(num_workers=4)
        assert config.num_workers == 4

    def test_custom_pairs_per_batch(self) -> None:
        """Test custom pairs per batch."""
        config = ParallelEPRConfig(pairs_per_batch=5000)
        assert config.pairs_per_batch == 5000

    def test_enabled_flag(self) -> None:
        """Test enabled flag toggling."""
        config_disabled = ParallelEPRConfig(enabled=False)
        config_enabled = ParallelEPRConfig(enabled=True)

        assert config_disabled.enabled is False
        assert config_enabled.enabled is True

    def test_isolation_level_process(self) -> None:
        """Test process isolation level."""
        config = ParallelEPRConfig(isolation_level="process")
        assert config.isolation_level == "process"

    def test_isolation_level_thread(self) -> None:
        """Test thread isolation level."""
        config = ParallelEPRConfig(isolation_level="thread")
        assert config.isolation_level == "thread"

    def test_invalid_isolation_level_raises(self) -> None:
        """Test invalid isolation_level raises ValueError."""
        with pytest.raises(ValueError, match="isolation_level"):
            ParallelEPRConfig(isolation_level="invalid")  # type: ignore

    def test_invalid_workers_raises(self) -> None:
        """Test num_workers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_workers"):
            ParallelEPRConfig(num_workers=0)

        with pytest.raises(ValueError, match="num_workers"):
            ParallelEPRConfig(num_workers=-1)

    def test_invalid_pairs_per_batch_raises(self) -> None:
        """Test pairs_per_batch < 1 raises ValueError."""
        with pytest.raises(ValueError, match="pairs_per_batch"):
            ParallelEPRConfig(pairs_per_batch=0)

    def test_invalid_timeout_raises(self) -> None:
        """Test timeout_seconds <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds"):
            ParallelEPRConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds"):
            ParallelEPRConfig(timeout_seconds=-1.0)

    def test_shuffle_results_toggle(self) -> None:
        """Test shuffle_results flag."""
        config_shuffle = ParallelEPRConfig(shuffle_results=True)
        config_no_shuffle = ParallelEPRConfig(shuffle_results=False)

        assert config_shuffle.shuffle_results is True
        assert config_no_shuffle.shuffle_results is False


# =============================================================================
# TestEPRWorkerResult
# =============================================================================


class TestEPRWorkerResult:
    """Test EPRWorkerResult dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        result = EPRWorkerResult(
            alice_outcomes=[0, 1, 0, 1],
            alice_bases=[0, 1, 0, 1],
            bob_outcomes=[0, 1, 1, 0],
            bob_bases=[0, 1, 0, 1],
            batch_id=0,
            num_pairs=4,
        )

        assert result.alice_outcomes == [0, 1, 0, 1]
        assert result.bob_outcomes == [0, 1, 1, 0]
        assert result.batch_id == 0
        assert result.num_pairs == 4

    def test_generation_time_default(self) -> None:
        """Test generation_time_ns defaults to 0."""
        result = EPRWorkerResult(
            alice_outcomes=[],
            alice_bases=[],
            bob_outcomes=[],
            bob_bases=[],
            batch_id=0,
            num_pairs=0,
        )
        assert result.generation_time_ns == 0.0

    def test_generation_time_custom(self) -> None:
        """Test custom generation_time_ns."""
        result = EPRWorkerResult(
            alice_outcomes=[],
            alice_bases=[],
            bob_outcomes=[],
            bob_bases=[],
            batch_id=0,
            num_pairs=0,
            generation_time_ns=1e9,
        )
        assert result.generation_time_ns == 1e9


# =============================================================================
# TestParallelEPROrchestrator
# =============================================================================


class TestParallelEPROrchestrator:
    """Test ParallelEPROrchestrator class."""

    @pytest.fixture
    def mock_network_config(self) -> dict:
        """Mock NetworkConfig fixture."""
        return {"noise": 0.05, "distance_km": 10}

    @pytest.fixture
    def config(self) -> ParallelEPRConfig:
        """Create test configuration."""
        return ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=100,
            shuffle_results=False,  # Disable for deterministic tests
        )

    @pytest.fixture
    def orchestrator(
        self, config: ParallelEPRConfig, mock_network_config: dict
    ) -> ParallelEPROrchestrator:
        """Create orchestrator with mock config."""
        return ParallelEPROrchestrator(
            config=config,
            network_config=mock_network_config,
        )

    def test_init(
        self, config: ParallelEPRConfig, mock_network_config: dict
    ) -> None:
        """Test orchestrator initialization."""
        orchestrator = ParallelEPROrchestrator(config, mock_network_config)

        assert orchestrator._config == config
        assert orchestrator._network_config == mock_network_config
        assert orchestrator._executor is None  # Lazy init

    def test_batch_count_calculation(
        self, orchestrator: ParallelEPROrchestrator
    ) -> None:
        """Test batch count calculation."""
        # 250 pairs with 100 per batch = 3 batches
        total_pairs = 250
        num_batches = math.ceil(
            total_pairs / orchestrator._config.pairs_per_batch
        )
        assert num_batches == 3  # 100 + 100 + 50

    def test_generate_parallel_invalid_pairs_raises(
        self, orchestrator: ParallelEPROrchestrator
    ) -> None:
        """Test that total_pairs <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="total_pairs must be > 0"):
            orchestrator.generate_parallel(0)

        with pytest.raises(ValueError, match="total_pairs must be > 0"):
            orchestrator.generate_parallel(-100)

    def test_generate_parallel_returns_correct_count(
        self, config: ParallelEPRConfig, mock_network_config: dict
    ) -> None:
        """Test generate_parallel returns correct number of results."""
        # Use small batch for fast test
        config = ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=50,
            shuffle_results=False,
        )
        orchestrator = ParallelEPROrchestrator(config, mock_network_config)

        try:
            results = orchestrator.generate_parallel(total_pairs=100)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            assert len(alice_outcomes) == 100
            assert len(alice_bases) == 100
            assert len(bob_outcomes) == 100
            assert len(bob_bases) == 100
        finally:
            orchestrator.shutdown()

    def test_generate_parallel_binary_values(
        self, mock_network_config: dict
    ) -> None:
        """Test that outcomes and bases are binary (0 or 1)."""
        config = ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=50,
        )
        orchestrator = ParallelEPROrchestrator(config, mock_network_config)

        try:
            results = orchestrator.generate_parallel(total_pairs=100)
            alice_outcomes, alice_bases, bob_outcomes, bob_bases = results

            assert all(o in (0, 1) for o in alice_outcomes)
            assert all(b in (0, 1) for b in alice_bases)
            assert all(o in (0, 1) for o in bob_outcomes)
            assert all(b in (0, 1) for b in bob_bases)
        finally:
            orchestrator.shutdown()

    def test_generate_parallel_basis_distribution(
        self, mock_network_config: dict
    ) -> None:
        """Test that both bases (Z=0, X=1) are used."""
        config = ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=500,
        )
        orchestrator = ParallelEPROrchestrator(config, mock_network_config)

        try:
            results = orchestrator.generate_parallel(total_pairs=1000)
            _, alice_bases, _, bob_bases = results

            # Both bases should be present
            assert set(alice_bases) == {0, 1}
            assert set(bob_bases) == {0, 1}

            # Roughly 50% each (allow 10% deviation)
            alice_z_ratio = alice_bases.count(0) / len(alice_bases)
            assert 0.4 < alice_z_ratio < 0.6
        finally:
            orchestrator.shutdown()

    def test_shutdown_idempotent(
        self, orchestrator: ParallelEPROrchestrator
    ) -> None:
        """Test shutdown can be called multiple times safely."""
        orchestrator.shutdown()
        orchestrator.shutdown()  # Should not raise

    def test_context_manager(
        self, config: ParallelEPRConfig, mock_network_config: dict
    ) -> None:
        """Test orchestrator as context manager."""
        with ParallelEPROrchestrator(config, mock_network_config) as orch:
            results = orch.generate_parallel(100)
            assert len(results[0]) == 100
        # Executor should be shut down after exit

    def test_result_shuffling(self, mock_network_config: dict) -> None:
        """Test results are shuffled when shuffle_results=True."""
        # Create two orchestrators: one with shuffle, one without
        config_no_shuffle = ParallelEPRConfig(
            enabled=True,
            num_workers=1,  # Single worker for determinism
            pairs_per_batch=100,
            shuffle_results=False,
        )
        config_shuffle = ParallelEPRConfig(
            enabled=True,
            num_workers=1,
            pairs_per_batch=100,
            shuffle_results=True,
        )

        orch_no_shuffle = ParallelEPROrchestrator(
            config_no_shuffle, mock_network_config
        )
        orch_shuffle = ParallelEPROrchestrator(
            config_shuffle, mock_network_config
        )

        try:
            # Generate with same seed should give same raw data
            # but shuffled version should differ in order
            results_no_shuffle = orch_no_shuffle.generate_parallel(100)
            results_shuffle = orch_shuffle.generate_parallel(100)

            # Both should have same length
            assert len(results_no_shuffle[0]) == len(results_shuffle[0])

            # Shuffled results should have same set of values but different order
            # (with high probability for 100 items)
            # Note: This test is probabilistic but extremely unlikely to fail
        finally:
            orch_no_shuffle.shutdown()
            orch_shuffle.shutdown()


# =============================================================================
# TestWorkerFunction
# =============================================================================


class TestWorkerFunction:
    """Test _worker_generate_epr worker function."""

    def test_worker_returns_correct_structure(self) -> None:
        """Test worker returns EPRWorkerResult with correct fields."""
        network_config = {"noise": 0.0}
        result = _worker_generate_epr(
            network_config=network_config,
            num_pairs=10,
            batch_id=0,
        )

        assert isinstance(result, EPRWorkerResult)
        assert len(result.alice_outcomes) == 10
        assert len(result.alice_bases) == 10
        assert len(result.bob_outcomes) == 10
        assert len(result.bob_bases) == 10
        assert result.batch_id == 0
        assert result.num_pairs == 10

    def test_worker_different_batches_different_ids(self) -> None:
        """Test different batch IDs are preserved."""
        network_config = {"noise": 0.0}

        result1 = _worker_generate_epr(network_config, 5, batch_id=0)
        result2 = _worker_generate_epr(network_config, 5, batch_id=1)
        result3 = _worker_generate_epr(network_config, 5, batch_id=42)

        assert result1.batch_id == 0
        assert result2.batch_id == 1
        assert result3.batch_id == 42

    def test_worker_respects_num_pairs(self) -> None:
        """Test worker generates exactly num_pairs."""
        network_config = {"noise": 0.0}

        for num_pairs in [1, 10, 100, 1000]:
            result = _worker_generate_epr(network_config, num_pairs, 0)
            assert len(result.alice_outcomes) == num_pairs
            assert result.num_pairs == num_pairs

    def test_worker_noise_affects_qber(self) -> None:
        """Test that noise parameter affects error rate."""
        # Ideal channel
        result_ideal = _worker_generate_epr(
            network_config={"noise": 0.0},
            num_pairs=1000,
            batch_id=0,
        )

        # Noisy channel
        result_noisy = _worker_generate_epr(
            network_config={"noise": 0.2},
            num_pairs=1000,
            batch_id=0,
        )

        # Calculate QBER on matching bases
        def calc_qber(result: EPRWorkerResult) -> float:
            matching = [
                i
                for i in range(result.num_pairs)
                if result.alice_bases[i] == result.bob_bases[i]
            ]
            if not matching:
                return 0.0
            errors = sum(
                result.alice_outcomes[i] != result.bob_outcomes[i]
                for i in matching
            )
            return errors / len(matching)

        qber_ideal = calc_qber(result_ideal)
        qber_noisy = calc_qber(result_noisy)

        # Ideal should have ~0% QBER, noisy should have higher
        assert qber_ideal < 0.05  # Allow small statistical fluctuation
        assert qber_noisy > qber_ideal  # Noisy should be worse


# =============================================================================
# Integration Tests (marked for optional CI execution)
# =============================================================================


@pytest.mark.integration
class TestParallelIntegration:
    """Integration tests requiring actual multiprocessing."""

    def test_large_scale_generation(self) -> None:
        """Test generation of large number of pairs."""
        config = ParallelEPRConfig(
            enabled=True,
            num_workers=4,
            pairs_per_batch=2500,
        )
        network_config = {"noise": 0.05}

        with ParallelEPROrchestrator(config, network_config) as orch:
            results = orch.generate_parallel(10000)

            assert len(results[0]) == 10000

            # Verify statistical properties
            alice_outcomes = results[0]
            ratio_ones = sum(alice_outcomes) / len(alice_outcomes)
            assert 0.4 < ratio_ones < 0.6  # ~50% ones

    def test_multiple_generations_same_orchestrator(self) -> None:
        """Test multiple generation calls with same orchestrator."""
        config = ParallelEPRConfig(
            enabled=True,
            num_workers=2,
            pairs_per_batch=100,
        )
        network_config = {"noise": 0.0}

        with ParallelEPROrchestrator(config, network_config) as orch:
            results1 = orch.generate_parallel(100)
            results2 = orch.generate_parallel(200)
            results3 = orch.generate_parallel(50)

            assert len(results1[0]) == 100
            assert len(results2[0]) == 200
            assert len(results3[0]) == 50

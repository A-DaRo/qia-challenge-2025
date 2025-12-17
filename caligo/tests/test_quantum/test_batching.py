"""
Unit tests for BatchingManager.
"""

import numpy as np
import pytest

from caligo.quantum.batching import (
    BatchingManager,
    BatchConfig,
    BatchResult,
    BatchState,
)
from caligo.types.exceptions import InvalidParameterError, ProtocolError


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_config(self):
        """Default configuration values."""
        config = BatchConfig()
        assert config.pairs_per_batch == 100
        assert config.max_batches == 100
        assert config.max_pairs == 10000

    def test_custom_config(self):
        """Custom configuration."""
        config = BatchConfig(
            pairs_per_batch=500,
            max_batches=20,
        )
        assert config.pairs_per_batch == 500
        assert config.max_pairs == 10000

    def test_invalid_pairs_per_batch_raises(self):
        """Zero or negative pairs_per_batch raises."""
        with pytest.raises(InvalidParameterError):
            BatchConfig(pairs_per_batch=0)
        
        with pytest.raises(InvalidParameterError):
            BatchConfig(pairs_per_batch=-10)

    def test_invalid_max_batches_raises(self):
        """Zero or negative max_batches raises."""
        with pytest.raises(InvalidParameterError):
            BatchConfig(max_batches=0)


class TestBatchingManager:
    """Tests for BatchingManager class."""

    def test_configure_returns_batch_count(self):
        """Configure returns correct batch count."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        
        num_batches = manager.configure(total_pairs=350)
        
        assert num_batches == 4  # 100 + 100 + 100 + 50

    def test_configure_exact_multiple(self):
        """Exact multiple of batch size."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        
        num_batches = manager.configure(total_pairs=500)
        
        assert num_batches == 5

    def test_configure_exceeds_max_raises(self):
        """Exceeding max_pairs raises."""
        manager = BatchingManager(
            BatchConfig(pairs_per_batch=100, max_batches=5)
        )
        
        with pytest.raises(InvalidParameterError):
            manager.configure(total_pairs=600)

    def test_configure_invalid_total_raises(self):
        """Zero or negative total raises."""
        manager = BatchingManager()
        
        with pytest.raises(InvalidParameterError):
            manager.configure(total_pairs=0)

    def test_start_batch(self):
        """Start batch creates correct structure."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=250)
        
        batch = manager.start_batch(batch_id=0)
        
        assert batch.batch_id == 0
        assert len(batch.round_ids) == 100
        assert batch.state == BatchState.GENERATING

    def test_start_batch_last_partial(self):
        """Last batch is partial."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=250)
        
        batch = manager.start_batch(batch_id=2)
        
        assert len(batch.round_ids) == 50  # 250 - 200

    def test_complete_batch(self):
        """Completing batch stores results."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=200)
        
        batch = manager.start_batch(0)
        batch.outcomes = np.random.randint(0, 2, 100, dtype=np.uint8)
        batch.bases = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        manager.complete_batch(batch)
        
        assert manager.completed_count == 1
        assert manager.total_outcomes == 100

    def test_iterate_batches(self):
        """Iterate over all batches."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=350)
        
        batch_ids = []
        for batch in manager.iterate_batches():
            batch_ids.append(batch.batch_id)
        
        assert batch_ids == [0, 1, 2, 3]

    def test_iterate_batches_unconfigured_raises(self):
        """Iterating without configure raises."""
        manager = BatchingManager()
        
        with pytest.raises(ProtocolError):
            list(manager.iterate_batches())

    def test_get_aggregated_results(self):
        """Aggregate all batch results."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=250)
        
        for batch in manager.iterate_batches():
            batch.outcomes = np.ones(len(batch.round_ids), dtype=np.uint8)
            batch.bases = np.zeros(len(batch.round_ids), dtype=np.uint8)
            manager.complete_batch(batch)
        
        result = manager.get_aggregated_results()
        
        assert result.num_pairs_requested == 250
        assert result.num_pairs_generated == 250
        assert len(result.measurement_outcomes) == 250

    def test_get_aggregated_no_batches_raises(self):
        """Aggregating with no batches raises."""
        manager = BatchingManager()
        
        with pytest.raises(ProtocolError):
            manager.get_aggregated_results()

    def test_reset(self):
        """Reset clears all state."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=200)
        
        for batch in manager.iterate_batches():
            batch.outcomes = np.zeros(len(batch.round_ids), dtype=np.uint8)
            batch.bases = np.zeros(len(batch.round_ids), dtype=np.uint8)
            manager.complete_batch(batch)
        
        manager.reset()
        
        assert manager.completed_count == 0
        assert manager.total_pairs == 0
        assert manager.total_outcomes == 0

    def test_estimate_memory_usage(self):
        """Memory usage estimate."""
        manager = BatchingManager(BatchConfig(pairs_per_batch=100))
        manager.configure(total_pairs=1000)
        
        for batch in manager.iterate_batches():
            batch.outcomes = np.zeros(len(batch.round_ids), dtype=np.uint8)
            batch.bases = np.zeros(len(batch.round_ids), dtype=np.uint8)
            manager.complete_batch(batch)
        
        memory = manager.estimate_memory_usage()
        
        # 1000 measurements × 10 bytes each
        assert memory == 10000


class TestBatchState:
    """Tests for BatchState enum."""

    def test_batch_states(self):
        """All batch states exist."""
        assert BatchState.PENDING.value == "pending"
        assert BatchState.GENERATING.value == "generating"
        assert BatchState.MEASURING.value == "measuring"
        assert BatchState.COMPLETED.value == "completed"
        assert BatchState.FAILED.value == "failed"

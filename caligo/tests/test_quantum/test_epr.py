"""
Unit tests for EPRGenerator.
"""

import pytest

from caligo.quantum.epr import (
    EPRGenerator,
    EPRGenerationConfig,
    EPRBatch,
)
from caligo.types.exceptions import EPRGenerationError


class TestEPRGenerationConfig:
    """Tests for EPRGenerationConfig dataclass."""

    def test_default_config(self):
        """Default configuration values."""
        config = EPRGenerationConfig()
        assert config.pairs_per_batch == 100
        assert config.timeout_ns == 1e9
        assert config.fidelity_threshold == 0.9
        assert config.retry_attempts == 3

    def test_custom_config(self):
        """Custom configuration."""
        config = EPRGenerationConfig(
            pairs_per_batch=500,
            timeout_ns=2e9,
            fidelity_threshold=0.95,
            retry_attempts=5,
        )
        assert config.pairs_per_batch == 500
        assert config.retry_attempts == 5


class TestEPRGenerator:
    """Tests for EPRGenerator class."""

    def test_initial_state(self):
        """Generator initializes with zero counts."""
        generator = EPRGenerator()
        assert generator.total_generated == 0

    def test_generate_batch_sync(self):
        """Synchronous batch generation."""
        generator = EPRGenerator()
        
        batch = generator.generate_batch_sync(num_pairs=50)
        
        assert batch.num_pairs == 50
        assert len(batch.qubit_refs) == 50
        assert batch.batch_id == 0
        assert generator.total_generated == 50

    def test_generate_multiple_batches(self):
        """Multiple batch generation."""
        generator = EPRGenerator()
        
        batch1 = generator.generate_batch_sync(num_pairs=100)
        batch2 = generator.generate_batch_sync(num_pairs=100)
        
        assert batch1.batch_id == 0
        assert batch2.batch_id == 1
        assert generator.total_generated == 200

    def test_reset_counters(self):
        """Reset counters clears state."""
        generator = EPRGenerator()
        generator.generate_batch_sync(num_pairs=100)
        
        generator.reset_counters()
        
        assert generator.total_generated == 0
        
        batch = generator.generate_batch_sync(num_pairs=50)
        assert batch.batch_id == 0

    def test_custom_config(self):
        """Generator uses custom config."""
        config = EPRGenerationConfig(pairs_per_batch=500)
        generator = EPRGenerator(config=config)
        
        assert generator.config.pairs_per_batch == 500


class TestEPRBatch:
    """Tests for EPRBatch dataclass."""

    def test_epr_batch_creation(self):
        """Create EPR batch."""
        batch = EPRBatch(
            qubit_refs=[0, 1, 2],
            generation_time=1000.0,
            num_pairs=3,
            batch_id=5,
        )
        
        assert batch.num_pairs == 3
        assert batch.batch_id == 5
        assert len(batch.qubit_refs) == 3

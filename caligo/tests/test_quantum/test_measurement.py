"""
Unit tests for MeasurementExecutor and MeasurementBuffer.
"""

import numpy as np
import pytest

from caligo.quantum.measurement import (
    MeasurementBuffer,
    MeasurementExecutor,
    MeasurementResult,
)


class TestMeasurementBuffer:
    """Tests for MeasurementBuffer class."""

    def test_initial_state(self):
        """Buffer initializes empty."""
        buffer = MeasurementBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_add_single_outcome(self):
        """Add single measurement."""
        buffer = MeasurementBuffer()
        buffer.add_outcome(outcome=1, basis=0, round_id=0)
        
        assert len(buffer) == 1
        outcomes, bases, ids, times = buffer.get_batch()
        assert outcomes[0] == 1
        assert bases[0] == 0
        assert ids[0] == 0

    def test_add_batch(self):
        """Add batch of measurements."""
        buffer = MeasurementBuffer()
        
        n = 100
        outcomes = np.random.randint(0, 2, n, dtype=np.uint8)
        bases = np.random.randint(0, 2, n, dtype=np.uint8)
        round_ids = np.arange(n, dtype=np.int64)
        
        buffer.add_batch(outcomes, bases, round_ids)
        
        assert len(buffer) == n
        out, bas, ids, _ = buffer.get_batch()
        assert np.array_equal(out, outcomes)
        assert np.array_equal(bas, bases)

    def test_buffer_grows_automatically(self):
        """Buffer grows when capacity exceeded."""
        buffer = MeasurementBuffer(capacity=10, growth_factor=2.0)
        
        # Add more than capacity
        for i in range(25):
            buffer.add_outcome(outcome=i % 2, basis=i % 2, round_id=i)
        
        assert len(buffer) == 25
        assert buffer.capacity >= 25

    def test_clear_resets_count(self):
        """Clear resets count but keeps capacity."""
        buffer = MeasurementBuffer(capacity=100)
        buffer.add_outcome(outcome=1, basis=0, round_id=0)
        buffer.add_outcome(outcome=0, basis=1, round_id=1)
        
        assert len(buffer) == 2
        
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_get_batch_returns_copies(self):
        """get_batch returns copies, not views."""
        buffer = MeasurementBuffer()
        buffer.add_outcome(outcome=1, basis=0, round_id=0)
        
        out1, _, _, _ = buffer.get_batch()
        out2, _, _, _ = buffer.get_batch()
        
        # Modifying one should not affect the other
        out1[0] = 99
        assert out2[0] == 1


class TestMeasurementExecutor:
    """Tests for MeasurementExecutor class."""

    def test_measure_qubit_sync(self):
        """Synchronous measurement returns valid outcome."""
        executor = MeasurementExecutor()
        
        outcome = executor.measure_qubit_sync(basis=0, round_id=0)
        
        assert outcome in (0, 1)
        assert executor.measurement_count == 1

    def test_measure_qubit_sync_predetermined(self):
        """Predetermined outcome returned."""
        executor = MeasurementExecutor()
        
        outcome = executor.measure_qubit_sync(
            basis=0, round_id=0, simulated_outcome=1
        )
        
        assert outcome == 1

    def test_measure_batch_sync(self):
        """Batch sync measurement."""
        executor = MeasurementExecutor()
        
        n = 100
        bases = np.random.randint(0, 2, n, dtype=np.uint8)
        round_ids = np.arange(n, dtype=np.int64)
        
        outcomes = executor.measure_batch_sync(bases, round_ids)
        
        assert len(outcomes) == n
        assert executor.measurement_count == n
        assert set(np.unique(outcomes)) <= {0, 1}

    def test_measure_batch_sync_predetermined(self):
        """Batch sync with predetermined outcomes."""
        executor = MeasurementExecutor()
        
        n = 50
        bases = np.zeros(n, dtype=np.uint8)
        round_ids = np.arange(n, dtype=np.int64)
        expected = np.ones(n, dtype=np.uint8)
        
        outcomes = executor.measure_batch_sync(
            bases, round_ids, simulated_outcomes=expected
        )
        
        assert np.array_equal(outcomes, expected)

    def test_get_results(self):
        """Get accumulated results."""
        executor = MeasurementExecutor()
        
        executor.measure_qubit_sync(basis=0, round_id=0, simulated_outcome=1)
        executor.measure_qubit_sync(basis=1, round_id=1, simulated_outcome=0)
        
        outcomes, bases, round_ids = executor.get_results()
        
        assert len(outcomes) == 2
        assert outcomes[0] == 1
        assert outcomes[1] == 0
        assert bases[0] == 0
        assert bases[1] == 1

    def test_clear_resets_state(self):
        """Clear resets executor state."""
        executor = MeasurementExecutor()
        executor.measure_qubit_sync(basis=0, round_id=0)
        
        executor.clear()
        
        assert executor.measurement_count == 0
        assert len(executor.buffer) == 0


class TestMeasurementResult:
    """Tests for MeasurementResult dataclass."""

    def test_measurement_result_creation(self):
        """Create measurement result."""
        result = MeasurementResult(
            outcome=1,
            basis=0,
            round_id=42,
            measurement_time=1000.0,
        )
        
        assert result.outcome == 1
        assert result.basis == 0
        assert result.round_id == 42
        assert result.measurement_time == 1000.0

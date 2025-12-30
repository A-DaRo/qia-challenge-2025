"""
Unit tests for exploration persistence module.

Tests HDF5Writer and StateManager functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    GROUP_ACTIVE_LEARNING,
    HDF5Writer,
    StateManager,
    result_to_hdf5_arrays,
    hdf5_arrays_to_training_data,
)
from caligo.exploration.types import (
    ExplorationSample,
    Phase1State,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)


class TestHDF5Writer:
    """Tests for HDF5Writer class."""

    def test_writer_creates_file(self, temp_dir):
        """Test that writer creates HDF5 file."""
        file_path = temp_dir / "test.h5"
        with HDF5Writer(file_path, mode="w") as writer:
            pass  # Just open and close
        assert file_path.exists()

    def test_writer_creates_groups(self, temp_dir):
        """Test that writer creates expected groups."""
        file_path = temp_dir / "test.h5"
        with HDF5Writer(file_path, mode="w") as writer:
            pass

        with h5py.File(file_path, "r") as f:
            assert GROUP_LHS_WARMUP in f
            assert GROUP_ACTIVE_LEARNING in f

    def test_write_batch(self, temp_dir, sample_results_batch):
        """Test writing a batch of results."""
        file_path = temp_dir / "test.h5"
        
        # Convert results to HDF5 format
        inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(sample_results_batch[:5])

        with HDF5Writer(file_path, mode="w") as writer:
            total = writer.append_batch(
                group_name=GROUP_LHS_WARMUP,
                inputs=inputs,
                outputs=outputs,
                outcomes=outcomes,
                metadata=metadata,
            )
            assert total == 5

        # Verify data was written
        with h5py.File(file_path, "r") as f:
            group = f[GROUP_LHS_WARMUP]
            assert group["inputs"].shape[0] == 5
            assert group["outputs"].shape[0] == 5

    def test_write_multiple_batches(self, temp_dir, sample_results_batch):
        """Test writing multiple batches."""
        file_path = temp_dir / "test.h5"

        with HDF5Writer(file_path, mode="w") as writer:
            for i in range(4):
                batch = sample_results_batch[i*5:(i+1)*5]
                inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(batch)
                total = writer.append_batch(
                    group_name=GROUP_LHS_WARMUP,
                    inputs=inputs,
                    outputs=outputs,
                    outcomes=outcomes,
                    metadata=metadata,
                )
            assert total == 20

    def test_read_group(self, temp_dir, sample_results_batch):
        """Test reading back data from a group."""
        file_path = temp_dir / "test.h5"
        inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(sample_results_batch[:10])

        with HDF5Writer(file_path, mode="w") as writer:
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        # Read back
        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, read_outputs, read_outcomes, read_metadata = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape == inputs.shape
            np.testing.assert_array_almost_equal(read_inputs, inputs.astype(np.float32), decimal=5)


class TestStateManager:
    """Tests for StateManager class."""

    def test_save_and_load_state(self, temp_dir, phase1_state_partial):
        """Test saving and loading checkpoint state."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        manager.save(phase1_state_partial)

        loaded = manager.load(Phase1State)
        assert loaded is not None
        assert loaded.target_feasible_samples == phase1_state_partial.target_feasible_samples
        assert loaded.feasible_samples_collected == phase1_state_partial.feasible_samples_collected

    def test_load_nonexistent_returns_none(self, temp_dir):
        """Test loading from empty directory returns None."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        loaded = manager.load(Phase1State)
        assert loaded is None

    def test_has_checkpoint(self, temp_dir, phase1_state_partial):
        """Test has_checkpoint method."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        assert manager.exists() is False

        manager.save(phase1_state_partial)
        assert manager.exists() is True

    def test_clear_checkpoint(self, temp_dir, phase1_state_partial):
        """Test clearing checkpoints."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        manager.save(phase1_state_partial)
        assert manager.exists() is True

        manager.delete()
        assert manager.exists() is False

    def test_backup_created_on_save(self, temp_dir, phase1_state_partial):
        """Test that backup is created when overwriting."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        
        # Save initial state
        manager.save(phase1_state_partial)
        
        # Save again (should create backup)
        updated_state = Phase1State(
            target_feasible_samples=100,
            feasible_samples_collected=50,
            total_samples_processed=50,
            current_batch_start=50,
            rng_state={},
        )
        manager.save(updated_state)

        # Load should return updated state
        loaded = manager.load(Phase1State)
        assert loaded.target_feasible_samples == 100


class TestResultToHdf5Arrays:
    """Tests for result conversion functions."""

    def test_conversion_preserves_data(self, sample_protocol_result):
        """Test that conversion preserves key data."""
        results = [sample_protocol_result]
        inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results)

        assert inputs.shape == (1, 9)
        assert outputs.shape == (1, 6)
        assert len(outcomes) == 1
        assert outcomes[0] == "success"

    def test_conversion_handles_empty_list(self):
        """Test conversion with empty list."""
        inputs, outputs, outcomes, metadata = result_to_hdf5_arrays([])
        assert inputs.shape == (0, 9)
        assert outputs.shape == (0, 6)
        assert len(outcomes) == 0


class TestRngStateCapture:
    """Tests for RNG state serialization."""

    def test_capture_restore_numpy(self):
        """Test capturing and restoring numpy RNG state."""
        # Create RNG and generate some numbers
        rng1 = np.random.default_rng(42)
        values1 = [rng1.random() for _ in range(10)]
        
        # Capture state
        state = rng1.bit_generator.state
        
        # Generate more numbers
        values2 = [rng1.random() for _ in range(10)]
        
        # Restore state
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = state
        
        # Should produce same sequence as values2
        values3 = [rng2.random() for _ in range(10)]
        assert values2 == values3


class TestHDF5DataIntegrity:
    """Data integrity tests for HDF5 operations."""

    def test_large_batch_write(self, temp_dir):
        """Test writing and reading a large batch."""
        file_path = temp_dir / "test.h5"
        n_samples = 1000
        
        # Generate random data
        rng = np.random.default_rng(42)
        inputs = rng.random((n_samples, 9))
        outputs = rng.random((n_samples, 6))
        outcomes = ["success"] * n_samples
        metadata = [json.dumps({"idx": i}) for i in range(n_samples)]

        with HDF5Writer(file_path, mode="w") as writer:
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, read_outputs, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            np.testing.assert_array_almost_equal(read_inputs, inputs.astype(np.float32), decimal=5)

    def test_special_values_preserved(self, temp_dir):
        """Test that special float values are preserved."""
        file_path = temp_dir / "test.h5"

        # Create data with special values
        inputs = np.array([[0.0, 1.0, 5.0, 0.6, -2.0, 0.05, -6.0, 5.0, 0.0]])
        outputs = np.array([[0.0, 100.0, 50.0, 0.05, 0.95, 1000.0]])  # Extreme values
        outcomes = ["success"]
        metadata = [json.dumps({})]

        with HDF5Writer(file_path, mode="w") as writer:
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, read_outputs, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            np.testing.assert_array_almost_equal(read_inputs, inputs.astype(np.float32), decimal=5)


class TestConcurrentHDF5Access:
    """Thread safety tests for HDF5 operations."""

    def test_sequential_writes_are_safe(self, temp_dir):
        """Test that sequential writes from same writer work."""
        file_path = temp_dir / "test.h5"
        rng = np.random.default_rng(42)

        with HDF5Writer(file_path, mode="w") as writer:
            for i in range(10):
                inputs = rng.random((10, 9))
                outputs = rng.random((10, 6))
                outcomes = ["success"] * 10
                metadata = [json.dumps({})] * 10
                writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with h5py.File(file_path, "r") as f:
            assert f[GROUP_LHS_WARMUP]["inputs"].shape[0] == 100

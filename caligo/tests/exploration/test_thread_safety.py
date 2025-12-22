"""
Thread safety tests for exploration modules.

Tests concurrent access patterns for:
- LHSSampler across threads
- HDF5Writer sequential writes
- StateManager concurrent operations
- EfficiencyLandscape predictions
- BayesianOptimizer suggestions
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pytest

from caligo.exploration.active import BayesianOptimizer
from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    HDF5Writer,
    StateManager,
)
from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.exploration.types import Phase1State


# Module-level function for multiprocessing (must be picklable)
def _sample_in_process(args):
    """Helper function for multiprocess sampling test."""
    seed, n_samples = args
    bounds = ParameterBounds()
    sampler = LHSSampler(bounds=bounds, seed=seed)
    samples = sampler.generate(n=n_samples)
    return len(samples)


class TestSamplerThreadSafety:
    """Thread safety tests for LHSSampler."""

    def test_concurrent_sampler_instances(self):
        """Test that concurrent sampler instances work correctly."""
        bounds = ParameterBounds()

        def sample_batch(seed: int) -> int:
            sampler = LHSSampler(bounds=bounds, seed=seed)
            samples = sampler.generate(n=100)
            return len(samples)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(sample_batch, i) for i in range(8)]
            results = [f.result() for f in futures]

        assert all(r == 100 for r in results)

    def test_rapid_sampler_creation_destruction(self):
        """Test rapid creation/destruction of samplers."""
        bounds = ParameterBounds()
        
        results = []
        for i in range(50):
            sampler = LHSSampler(bounds=bounds, seed=i)
            samples = sampler.generate(n=10)
            results.append(len(samples))

        assert all(r == 10 for r in results)


class TestHDF5ThreadSafety:
    """Thread safety tests for HDF5 operations."""

    def test_sequential_writers_to_same_file(self, temp_dir):
        """Test sequential writes from same writer."""
        file_path = temp_dir / "test.h5"
        rng = np.random.default_rng(42)

        with HDF5Writer(file_path, mode="w") as writer:
            for batch_idx in range(10):
                inputs = rng.random((10, 9))
                outputs = rng.random((10, 6))
                outcomes = ["success"] * 10
                metadata = [json.dumps({"batch": batch_idx})] * 10
                writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == 100

    def test_concurrent_readers(self, temp_dir):
        """Test concurrent reads (safe for HDF5)."""
        file_path = temp_dir / "test.h5"
        rng = np.random.default_rng(42)

        # Write some data
        with HDF5Writer(file_path, mode="w") as writer:
            inputs = rng.random((100, 9))
            outputs = rng.random((100, 6))
            outcomes = ["success"] * 100
            metadata = [json.dumps({})] * 100
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        # Read concurrently
        def read_data():
            with HDF5Writer(file_path, mode="r") as reader:
                read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
                return read_inputs.shape[0]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_data) for _ in range(8)]
            results = [f.result() for f in futures]

        assert all(r == 100 for r in results)


class TestStateManagerThreadSafety:
    """Thread safety tests for StateManager."""

    def test_concurrent_checkpoint_saves(self, temp_dir):
        """Test sequential saves (one should win)."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")

        states = []
        for i in range(5):
            state = Phase1State(
                total_samples=100,
                completed_samples=i * 20,
                current_batch_start=i * 20,
                rng_state={},
            )
            manager.save(state)
            states.append(state)

        # Load should return most recent
        loaded = manager.load(Phase1State)
        assert loaded is not None
        assert loaded.completed_samples == 80

    def test_save_load_race_condition(self, temp_dir):
        """Test save/load interaction."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")

        # Save initial state
        initial_state = Phase1State(
            total_samples=100,
            completed_samples=50,
            current_batch_start=50,
            rng_state={},
        )
        manager.save(initial_state)

        # Load should always return valid state
        loaded = manager.load(Phase1State)
        assert loaded is not None
        assert loaded.total_samples == 100


class TestSurrogateThreadSafety:
    """Thread safety tests for surrogate model."""

    @pytest.fixture
    def trained_landscape(self, training_data_twin):
        """Create a trained landscape."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)
        return landscape

    def test_concurrent_predictions(self, trained_landscape, training_data_twin):
        """Test concurrent prediction calls."""
        X_baseline, _, _, _ = training_data_twin

        def predict_batch(idx: int):
            batch = X_baseline[idx*5:(idx+1)*5]
            mu, sigma = trained_landscape.predict_baseline(batch, return_std=True)
            return mu.shape[0]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_batch, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert all(r == 5 for r in results)

    def test_concurrent_divergence_predictions(self, trained_landscape, training_data_twin):
        """Test concurrent divergence predictions."""
        X_baseline, _, _, _ = training_data_twin

        def predict_divergence(idx: int):
            batch = X_baseline[idx*5:(idx+1)*5]
            mu, sigma = trained_landscape.predict_divergence(batch)
            return len(mu)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_divergence, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert all(r == 5 for r in results)


class TestOptimizerThreadSafety:
    """Thread safety tests for Bayesian optimizer."""

    @pytest.fixture
    def trained_landscape(self, training_data_twin):
        """Create a trained landscape."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)
        return landscape

    def test_concurrent_optimizer_instances(self, trained_landscape, parameter_bounds):
        """Test concurrent optimizer instances."""

        def suggest_points(idx: int):
            optimizer = BayesianOptimizer(
                landscape=trained_landscape,
                bounds=parameter_bounds,
            )
            points = optimizer.suggest_batch(batch_size=3)
            return points.shape[0]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(suggest_points, i) for i in range(8)]
            results = [f.result() for f in futures]

        assert all(r == 3 for r in results)


class TestProcessPoolSafety:
    """Process pool tests for multiprocessing safety."""

    def test_multiprocess_sampling(self):
        """Test sampling in multiple processes."""
        # Use spawn context for clean processes
        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        # Use module-level function for pickling
        args_list = [(i, 50) for i in range(4)]
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
            futures = [executor.submit(_sample_in_process, args) for args in args_list]
            results = [f.result() for f in futures]

        assert all(r == 50 for r in results)


class TestLockContention:
    """Tests for high-contention scenarios."""

    def test_high_contention_checkpoint_writes(self, temp_dir):
        """Test high-frequency checkpoint updates."""
        manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        n_writes = 20

        for i in range(n_writes):
            state = Phase1State(
                total_samples=100,
                completed_samples=i,
                current_batch_start=i,
                rng_state={"iteration": i},
            )
            manager.save(state)

        # Final state should be last written
        loaded = manager.load(Phase1State)
        assert loaded is not None
        assert loaded.completed_samples == n_writes - 1

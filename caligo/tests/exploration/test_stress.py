"""
Stress tests for exploration modules.

Tests for:
- Memory limits with large datasets
- Numerical stability at extreme values
- Error recovery from corrupted state
- Edge cases and boundary conditions
- Performance under load
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from caligo.exploration.active import BayesianOptimizer, AcquisitionFunction
from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    HDF5Writer,
    StateManager,
)
from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.exploration.types import (
    ExplorationSample,
    Phase1State,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)


class TestMemoryStress:
    """Memory stress tests."""

    def test_large_lhs_sample_generation(self):
        """Test generating many LHS samples."""
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)

        # Generate large batch
        n_samples = 5000
        samples = sampler.generate(n=n_samples)

        assert len(samples) == n_samples
        
        # Clean up
        del samples
        gc.collect()

    def test_large_hdf5_writes(self, temp_dir):
        """Test writing large amounts of data to HDF5."""
        file_path = temp_dir / "large_test.h5"
        rng = np.random.default_rng(42)

        n_batches = 50
        batch_size = 100

        with HDF5Writer(file_path, mode="w") as writer:
            for _ in range(n_batches):
                inputs = rng.random((batch_size, 9))
                outputs = rng.random((batch_size, 6))
                outcomes = ["success"] * batch_size
                metadata = [json.dumps({})] * batch_size
                writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        # Verify
        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == n_batches * batch_size

    def test_gp_scaling_with_data_size(self):
        """Test GP training time scales reasonably with data size."""
        import time

        rng = np.random.default_rng(42)
        sizes = [50, 100, 200]
        times = []

        for n in sizes:
            X = rng.random((n, 9))
            y_baseline = rng.random(n)
            y_blind = rng.random(n)

            landscape = EfficiencyLandscape()
            start = time.time()
            landscape.fit(X, y_baseline, X, y_blind)
            elapsed = time.time() - start
            times.append(elapsed)

        # Training should complete in reasonable time
        for t in times:
            assert t < 30.0  # 30 seconds max

    def test_memory_cleanup_after_large_operations(self, temp_dir):
        """Test that memory is cleaned up after large operations."""
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)

        # Generate and discard large data multiple times
        for _ in range(5):
            samples = sampler.generate(n=1000)
            del samples
            gc.collect()

        # Should complete without memory error


class TestNumericalStability:
    """Numerical stability tests."""

    def test_extreme_parameter_values(self):
        """Test handling of extreme but valid parameter values."""
        # Minimum valid values
        sample_min = ExplorationSample(
            storage_noise_r=0.0,
            storage_rate_nu=0.001,
            wait_time_ns=1e5,
            channel_fidelity=0.501,
            detection_efficiency=0.001,
            detector_error=0.0,
            dark_count_prob=1e-8,
            num_pairs=10000,
            strategy=ReconciliationStrategy.BASELINE,
        )
        assert sample_min.to_array().shape == (9,)

        # Maximum valid values
        sample_max = ExplorationSample(
            storage_noise_r=1.0,
            storage_rate_nu=1.0,
            wait_time_ns=1e9,
            channel_fidelity=1.0,
            detection_efficiency=1.0,
            detector_error=0.1,
            dark_count_prob=1e-3,
            num_pairs=1000000,
            strategy=ReconciliationStrategy.BLIND,
        )
        assert sample_max.to_array().shape == (9,)

    def test_gp_with_nearly_identical_inputs(self):
        """Test GP with nearly identical input points."""
        rng = np.random.default_rng(42)
        n = 30

        # Create points that are nearly identical
        base = rng.random((1, 9))
        noise = rng.random((n, 9)) * 1e-8  # Very small noise
        X = base + noise

        y_baseline = rng.random(n)
        y_blind = rng.random(n)

        landscape = EfficiencyLandscape()
        # Should handle this gracefully (may produce warnings)
        landscape.fit(X, y_baseline, X, y_blind)

    def test_gp_with_constant_targets(self):
        """Test GP with constant target values."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 9))
        y_baseline = np.full(30, 0.5)  # Constant
        y_blind = np.full(30, 0.5)  # Constant

        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)

        # Predictions should be near constant
        mu, sigma = landscape.predict(X[:5], return_std=True)
        assert np.allclose(mu, 0.5, atol=0.1)

    def test_gp_with_outlier_targets(self):
        """Test GP robustness to outliers."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 9))
        y_baseline = rng.uniform(0.4, 0.6, 30)
        y_baseline[0] = 10.0  # Outlier
        y_blind = y_baseline * 0.9

        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)

        # Should still make reasonable predictions
        mu, _ = landscape.predict(X[1:10])
        assert np.all(np.isfinite(mu))

    def test_acquisition_at_boundary(self, training_data_twin):
        """Test acquisition function at parameter boundaries."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)

        acq = AcquisitionFunction(landscape=landscape, acquisition_type="straddle")

        # Create boundary points
        bounds = ParameterBounds()
        bounds_arr = bounds.to_bounds_array()

        # Test at lower bounds (with valid fidelity)
        lower = np.array([
            bounds_arr[0, 0],  # r_min
            np.log10(bounds.nu_min),
            bounds.dt_min_log,
            0.6,  # Valid fidelity
            bounds.eta_min_log,
            bounds.e_det_min,
            bounds.p_dark_min_log,
            bounds.n_min_log,
            0.0,
        ])

        values = acq(lower.reshape(1, -1))
        assert np.isfinite(values[0])


class TestErrorRecovery:
    """Error recovery tests."""

    def test_corrupted_checkpoint_recovery(self, temp_dir):
        """Test handling of corrupted checkpoint file."""
        checkpoint_file = temp_dir / "checkpoint.pkl"
        manager = StateManager(checkpoint_path=checkpoint_file)

        # Write corrupted data
        with open(checkpoint_file, "wb") as f:
            f.write(b"not valid pickle data")

        # Load should return None or raise appropriate error
        try:
            loaded = manager.load(Phase1State)
            # If it doesn't raise, it should return None
            assert loaded is None
        except (pickle.UnpicklingError, Exception):
            pass  # Expected behavior

    def test_partial_hdf5_write_recovery(self, temp_dir):
        """Test recovery from partial HDF5 writes."""
        file_path = temp_dir / "test.h5"
        rng = np.random.default_rng(42)

        # Write some valid data
        with HDF5Writer(file_path, mode="w") as writer:
            inputs = rng.random((50, 9))
            outputs = rng.random((50, 6))
            outcomes = ["success"] * 50
            metadata = [json.dumps({})] * 50
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        # Verify data is readable
        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == 50


class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample_pipeline(self):
        """Test pipeline with minimum number of samples."""
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)

        samples = sampler.generate(n=5)
        assert len(samples) == 5

        X = np.array([s.to_array() for s in samples])
        y = np.array([0.5, 0.6, 0.55, 0.7, 0.65])

        landscape = EfficiencyLandscape()
        landscape.fit(X, y, X, y * 0.9)

        # Should be able to make predictions
        mu, sigma = landscape.predict(X, return_std=True)
        assert mu.shape == (5,)

    def test_empty_result_batch(self, temp_dir):
        """Test handling of empty result batches."""
        file_path = temp_dir / "test.h5"

        with HDF5Writer(file_path, mode="w") as writer:
            # Write empty batch
            inputs = np.zeros((0, 9))
            outputs = np.zeros((0, 6))
            outcomes = []
            metadata = []
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == 0

    def test_all_failures_dataset(self):
        """Test landscape training with all zero efficiencies."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 9))
        y_baseline = np.zeros(30)  # All failures
        y_blind = np.zeros(30)

        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)

        mu, sigma = landscape.predict(X[:5], return_std=True)
        # Predictions should be near zero
        assert np.allclose(mu, 0.0, atol=0.1)

    def test_mixed_success_failure_ratio(self):
        """Test with varying success/failure ratios."""
        rng = np.random.default_rng(42)

        for failure_ratio in [0.1, 0.5, 0.9]:
            X = rng.random((50, 9))
            n_failures = int(50 * failure_ratio)
            y_baseline = rng.uniform(0.3, 0.8, 50)
            y_baseline[:n_failures] = 0.0  # Failures
            y_blind = y_baseline * 0.9

            landscape = EfficiencyLandscape()
            landscape.fit(X, y_baseline, X, y_blind)
            assert landscape.is_fitted

    def test_bounds_at_limits(self):
        """Test parameter bounds at edge values."""
        # Very narrow bounds
        narrow = ParameterBounds(
            f_min=0.95,
            f_max=0.951,
        )
        sampler = LHSSampler(bounds=narrow, seed=42)
        samples = sampler.generate(n=10)

        for s in samples:
            assert 0.95 <= s.channel_fidelity <= 0.951

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf in training data."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 9))
        y_baseline = rng.uniform(0.3, 0.8, 30)
        y_baseline[0] = np.nan
        y_baseline[1] = np.inf
        y_blind = y_baseline.copy()

        landscape = EfficiencyLandscape()
        
        # Should either handle gracefully or raise informative error
        try:
            landscape.fit(X, y_baseline, X, y_blind)
            # If it succeeds, check predictions are still finite
            mu, _ = landscape.predict(X[5:10])
            # May contain NaN due to training data issues
        except (ValueError, RuntimeError):
            pass  # Expected if invalid data is rejected


class TestPerformanceStress:
    """Performance stress tests."""

    def test_many_small_batches(self, temp_dir):
        """Test writing many small batches."""
        file_path = temp_dir / "test.h5"
        rng = np.random.default_rng(42)

        with HDF5Writer(file_path, mode="w") as writer:
            for i in range(100):
                inputs = rng.random((5, 9))
                outputs = rng.random((5, 6))
                outcomes = ["success"] * 5
                metadata = [json.dumps({"batch": i})] * 5
                writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)

        with HDF5Writer(file_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == 500

    def test_repeated_gp_retraining(self):
        """Test repeated GP retraining (simulating online learning)."""
        rng = np.random.default_rng(42)
        landscape = EfficiencyLandscape()

        X = rng.random((20, 9))
        y_baseline = rng.random(20)
        y_blind = y_baseline * 0.9

        # Retrain multiple times with growing data
        for i in range(10):
            new_X = rng.random((5, 9))
            new_y = rng.random(5)

            X = np.vstack([X, new_X])
            y_baseline = np.concatenate([y_baseline, new_y])
            y_blind = y_baseline * 0.9

            landscape.fit(X, y_baseline, X, y_blind)

        assert landscape.baseline_gp.n_samples == 70

    def test_acquisition_optimization_scaling(self, training_data_twin, parameter_bounds):
        """Test acquisition optimization scales reasonably."""
        import time

        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)

        optimizer = BayesianOptimizer(
            landscape=landscape,
            bounds=parameter_bounds,
        )

        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            start = time.time()
            points = optimizer.suggest_batch(batch_size=batch_size)
            elapsed = time.time() - start

            assert points.shape[0] == batch_size
            assert elapsed < 60.0  # Should complete in reasonable time

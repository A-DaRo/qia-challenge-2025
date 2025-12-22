"""
Integration tests for the exploration pipeline.

Tests end-to-end workflows including:
- Phase 1: LHS sampling and persistence
- Phase 2: Surrogate training
- Phase 3: Bayesian optimization
- Full pipeline mini integration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from caligo.exploration.active import BayesianOptimizer, AcquisitionConfig
from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    GROUP_ACTIVE_LEARNING,
    HDF5Writer,
    StateManager,
    result_to_hdf5_arrays,
)
from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.surrogate import EfficiencyLandscape, GPConfig
from caligo.exploration.types import (
    ExplorationConfig,
    ExplorationSample,
    Phase1State,
    Phase2State,
    Phase3State,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)


def optimizer_array_to_sample(arr: np.ndarray) -> ExplorationSample:
    """
    Convert optimizer output array to ExplorationSample.
    
    The optimizer works in a transformed space where some parameters
    are log-scaled. This function handles the inverse transform.
    
    Optimizer space (same as ParameterBounds.to_bounds_array()):
    - Index 0: storage_noise_r (linear)
    - Index 1: log10(storage_rate_nu)
    - Index 2: log10(wait_time_ns)
    - Index 3: channel_fidelity (linear)
    - Index 4: log10(detection_efficiency)
    - Index 5: detector_error (linear)
    - Index 6: log10(dark_count_prob)
    - Index 7: log10(num_pairs)
    - Index 8: strategy (0=baseline, 1=blind)
    """
    strategy = ReconciliationStrategy.BASELINE if arr[8] < 0.5 else ReconciliationStrategy.BLIND
    return ExplorationSample(
        storage_noise_r=float(arr[0]),
        storage_rate_nu=float(10 ** arr[1]),  # Inverse log transform
        wait_time_ns=float(10 ** arr[2]),
        channel_fidelity=float(arr[3]),
        detection_efficiency=float(10 ** arr[4]),
        detector_error=float(arr[5]),
        dark_count_prob=float(10 ** arr[6]),
        num_pairs=int(round(10 ** arr[7])),
        strategy=strategy,
    )


def mock_protocol_execution(sample: ExplorationSample) -> ProtocolResult:
    """
    Mock protocol execution for testing.
    
    Simulates success based on channel fidelity.
    """
    # Efficiency roughly correlated with fidelity
    base_efficiency = (sample.channel_fidelity - 0.5) * 2  # Maps 0.5->0, 1->1
    noise = np.random.uniform(-0.1, 0.1)
    efficiency = np.clip(base_efficiency + noise, 0, 1)
    
    # Simulate failures at low fidelity
    if sample.channel_fidelity < 0.6:
        outcome = ProtocolOutcome.FAILURE_QBER
        efficiency = 0.0
        error_msg = "QBER threshold exceeded"
    else:
        outcome = ProtocolOutcome.SUCCESS
        error_msg = None
    
    raw_key = int(sample.num_pairs * 0.5)
    final_key = int(raw_key * efficiency)
    
    return ProtocolResult(
        sample=sample,
        outcome=outcome,
        net_efficiency=efficiency,
        raw_key_length=raw_key,
        final_key_length=final_key,
        qber_measured=0.03 + (1 - sample.channel_fidelity) * 0.1,
        reconciliation_efficiency=0.95 if outcome == ProtocolOutcome.SUCCESS else 0.0,
        leakage_bits=int(raw_key * 0.1),
        execution_time_seconds=1.0,
        error_message=error_msg,
        metadata={},
    )


class TestPhase1Integration:
    """Integration tests for Phase 1: LHS sampling."""

    def test_generate_and_persist_lhs_samples(self, temp_dir):
        """Test generating LHS samples and persisting to HDF5."""
        n_samples = 50
        batch_size = 10
        
        # Initialize
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        hdf5_path = temp_dir / "exploration_data.h5"
        
        # Generate and execute
        samples = sampler.generate(n=n_samples)
        results = [mock_protocol_execution(s) for s in samples]
        
        # Persist in batches
        with HDF5Writer(hdf5_path, mode="w") as writer:
            for i in range(0, n_samples, batch_size):
                batch = results[i:i+batch_size]
                inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(batch)
                writer.append_batch(
                    GROUP_LHS_WARMUP,
                    inputs,
                    outputs,
                    outcomes,
                    metadata,
                )
        
        # Verify
        with HDF5Writer(hdf5_path, mode="r") as reader:
            read_inputs, read_outputs, read_outcomes, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == n_samples
            assert read_outputs.shape[0] == n_samples

    def test_checkpoint_resume_phase1(self, temp_dir):
        """Test checkpoint and resume for Phase 1."""
        n_samples = 30
        checkpoint_at = 15
        
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        hdf5_path = temp_dir / "exploration_data.h5"
        state_manager = StateManager(checkpoint_path=temp_dir / "checkpoint.pkl")
        
        # Run first half
        samples = sampler.generate(n=n_samples)
        results_part1 = [mock_protocol_execution(s) for s in samples[:checkpoint_at]]
        
        with HDF5Writer(hdf5_path, mode="w") as writer:
            inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results_part1)
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)
        
        # Save checkpoint
        rng = np.random.default_rng(42)
        state = Phase1State(
            total_samples=n_samples,
            completed_samples=checkpoint_at,
            current_batch_start=checkpoint_at,
            rng_state={"state": rng.bit_generator.state},
        )
        state_manager.save(state)
        
        # Resume (new sampler, load checkpoint)
        loaded_state = state_manager.load(Phase1State)
        assert loaded_state is not None
        assert loaded_state.completed_samples == checkpoint_at
        
        # Complete remaining samples
        results_part2 = [mock_protocol_execution(s) for s in samples[checkpoint_at:]]
        
        with HDF5Writer(hdf5_path, mode="a") as writer:
            inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results_part2)
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)
        
        # Verify complete dataset
        with HDF5Writer(hdf5_path, mode="r") as reader:
            read_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            assert read_inputs.shape[0] == n_samples


class TestPhase2Integration:
    """Integration tests for Phase 2: Surrogate training."""

    def test_load_data_and_train_surrogate(self, temp_dir):
        """Test loading HDF5 data and training surrogate."""
        n_samples = 50
        
        # Generate training data
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        samples = sampler.generate(n=n_samples)
        results = [mock_protocol_execution(s) for s in samples]
        
        # Persist
        hdf5_path = temp_dir / "exploration_data.h5"
        with HDF5Writer(hdf5_path, mode="w") as writer:
            inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results)
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)
        
        # Load and prepare for training
        with HDF5Writer(hdf5_path, mode="r") as reader:
            X, y_data, _, _ = reader.read_group(GROUP_LHS_WARMUP)
        
        # y_data has shape (n, 6) where first column is net_efficiency
        y_baseline = y_data[:, 0]
        y_blind = y_data[:, 0] * 0.9  # Simulate blind efficiency
        
        # Train surrogate
        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)
        
        assert landscape.is_fitted
        
        # Test predictions
        mu, sigma = landscape.predict(X[:5], return_std=True)
        assert mu.shape == (5,)
        assert sigma.shape == (5,)

    def test_surrogate_cross_validation(self, temp_dir):
        """Test surrogate prediction quality."""
        n_samples = 100
        
        # Generate data with known pattern
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        samples = sampler.generate(n=n_samples)
        
        # Create efficiency that depends on fidelity
        X = np.array([s.to_array() for s in samples])
        # Fidelity is at index 3 after transformation
        y_baseline = 0.8 * X[:, 3] + 0.1 * np.random.randn(n_samples)
        y_baseline = np.clip(y_baseline, 0, 1)
        y_blind = y_baseline * 0.9
        
        # Split train/test
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)
        
        landscape = EfficiencyLandscape()
        landscape.fit(X[train_idx], y_baseline[train_idx], X[train_idx], y_blind[train_idx])
        
        # Predict on test set
        y_pred, y_std = landscape.predict(X[test_idx], return_std=True)
        
        # Check predictions are reasonable
        rmse = np.sqrt(np.mean((y_pred - y_baseline[test_idx])**2))
        assert rmse < 0.3  # Reasonable error tolerance


class TestPhase3Integration:
    """Integration tests for Phase 3: Bayesian optimization."""

    def test_bayesian_optimization_loop(self, temp_dir):
        """Test one iteration of Bayesian optimization."""
        # Create initial data
        n_initial = 30
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        samples = sampler.generate(n=n_initial)
        results = [mock_protocol_execution(s) for s in samples]
        
        # Train landscape
        X = np.array([r.sample.to_array() for r in results])
        y_baseline = np.array([r.net_efficiency for r in results])
        y_blind = y_baseline * 0.9
        
        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)
        
        # Initialize optimizer
        optimizer = BayesianOptimizer(
            landscape=landscape,
            bounds=bounds,
        )
        
        # Suggest new points
        n_suggest = 5
        new_points = optimizer.suggest_batch(batch_size=n_suggest)
        
        assert new_points.shape == (n_suggest, 9)
        
        # Execute new points (convert from optimizer space)
        new_samples = [optimizer_array_to_sample(p) for p in new_points]
        new_results = [mock_protocol_execution(s) for s in new_samples]
        
        # Update landscape with new data
        X_new = np.array([r.sample.to_array() for r in new_results])
        y_new_baseline = np.array([r.net_efficiency for r in new_results])
        y_new_blind = y_new_baseline * 0.9
        
        X_all = np.vstack([X, X_new])
        y_all_baseline = np.concatenate([y_baseline, y_new_baseline])
        y_all_blind = np.concatenate([y_blind, y_new_blind])
        
        landscape.fit(X_all, y_all_baseline, X_all, y_all_blind)
        
        assert landscape.baseline_gp.n_samples == n_initial + n_suggest


class TestFullPipelineIntegration:
    """Full pipeline integration tests."""

    def test_full_pipeline_mini(self, temp_dir):
        """Test mini version of full exploration pipeline."""
        # Phase 1: LHS warmup
        n_lhs = 20
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        hdf5_path = temp_dir / "exploration.h5"
        
        lhs_samples = sampler.generate(n=n_lhs)
        lhs_results = [mock_protocol_execution(s) for s in lhs_samples]
        
        with HDF5Writer(hdf5_path, mode="w") as writer:
            inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(lhs_results)
            writer.append_batch(GROUP_LHS_WARMUP, inputs, outputs, outcomes, metadata)
        
        # Phase 2: Train surrogate
        with HDF5Writer(hdf5_path, mode="r") as reader:
            X, y_data, _, _ = reader.read_group(GROUP_LHS_WARMUP)
        
        y_baseline = y_data[:, 0]
        y_blind = y_baseline * 0.9
        
        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)
        
        # Phase 3: Active learning (2 iterations)
        optimizer = BayesianOptimizer(
            landscape=landscape,
            bounds=bounds,
        )
        
        n_iterations = 2
        batch_size = 3
        
        all_active_results = []
        for iteration in range(n_iterations):
            # Suggest points
            new_points = optimizer.suggest_batch(batch_size=batch_size)
            
            # Execute (convert from optimizer space)
            new_samples = [optimizer_array_to_sample(p) for p in new_points]
            new_results = [mock_protocol_execution(s) for s in new_samples]
            all_active_results.extend(new_results)
            
            # Update landscape
            X_new = np.array([r.sample.to_array() for r in new_results])
            y_new_baseline = np.array([r.net_efficiency for r in new_results])
            y_new_blind = y_new_baseline * 0.9
            
            X = np.vstack([X, X_new])
            y_baseline = np.concatenate([y_baseline, y_new_baseline])
            y_blind = np.concatenate([y_blind, y_new_blind])
            
            landscape.fit(X, y_baseline, X, y_blind)
        
        # Persist active results
        with HDF5Writer(hdf5_path, mode="a") as writer:
            inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(all_active_results)
            writer.append_batch(GROUP_ACTIVE_LEARNING, inputs, outputs, outcomes, metadata)
        
        # Verify final state
        with HDF5Writer(hdf5_path, mode="r") as reader:
            lhs_inputs, _, _, _ = reader.read_group(GROUP_LHS_WARMUP)
            active_inputs, _, _, _ = reader.read_group(GROUP_ACTIVE_LEARNING)
        
        assert lhs_inputs.shape[0] == n_lhs
        assert active_inputs.shape[0] == n_iterations * batch_size
        assert landscape.baseline_gp.n_samples == n_lhs + n_iterations * batch_size

    def test_pipeline_handles_failures_gracefully(self, temp_dir):
        """Test pipeline handles protocol failures."""
        bounds = ParameterBounds()
        sampler = LHSSampler(bounds=bounds, seed=42)
        
        # Generate samples including some that will fail
        samples = sampler.generate(n=30)
        
        # Execute and count failures
        results = []
        for sample in samples:
            result = mock_protocol_execution(sample)
            results.append(result)
        
        failures = sum(1 for r in results if not r.is_success())
        successes = sum(1 for r in results if r.is_success())
        
        # Should have mix of success and failure
        assert failures > 0
        assert successes > 0
        
        # Training should work even with failed results
        X = np.array([r.sample.to_array() for r in results])
        y_baseline = np.array([r.net_efficiency for r in results])
        y_blind = y_baseline * 0.9
        
        landscape = EfficiencyLandscape()
        landscape.fit(X, y_baseline, X, y_blind)
        
        assert landscape.is_fitted

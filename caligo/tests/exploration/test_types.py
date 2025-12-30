"""
Unit tests for exploration types module.

Tests ExplorationSample, ProtocolResult, phase states, and configuration.
"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from caligo.exploration.types import (
    ExplorationConfig,
    ExplorationPhase,
    ExplorationSample,
    Phase1State,
    Phase2State,
    Phase3State,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)


class TestExplorationSample:
    """Tests for ExplorationSample dataclass."""

    def test_valid_sample_creation(self, sample_parameters):
        """Test creating a valid sample."""
        sample = ExplorationSample(**sample_parameters)
        assert sample.storage_noise_r == 0.1
        assert sample.storage_rate_nu == 0.5
        assert sample.wait_time_ns == 1e6
        assert sample.channel_fidelity == 0.95
        assert sample.detection_efficiency == 0.8
        assert sample.detector_error == 0.01
        assert sample.dark_count_prob == 1e-5
        assert sample.num_pairs == 100000
        assert sample.strategy == ReconciliationStrategy.BASELINE

    def test_sample_is_frozen(self, sample_exploration_sample):
        """Test that samples are immutable."""
        with pytest.raises(FrozenInstanceError):
            sample_exploration_sample.channel_fidelity = 0.99

    def test_sample_hashable(self, sample_exploration_sample):
        """Test that samples can be used as dict keys."""
        d = {sample_exploration_sample: "value"}
        assert d[sample_exploration_sample] == "value"

    def test_sample_to_array(self, sample_exploration_sample):
        """Test conversion to numpy array."""
        arr = sample_exploration_sample.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (9,)
        # Float32 for memory efficiency and SIMD optimization
        assert arr.dtype == np.float32
        # First element should be storage_noise_r (0.1)
        assert arr[0] == pytest.approx(0.1, rel=1e-5)
        # Strategy should be encoded as 0 (BASELINE) or 1 (BLIND)
        assert arr[8] in [0.0, 1.0]

    def test_sample_from_array_roundtrip(self, sample_exploration_sample):
        """Test roundtrip conversion through array."""
        arr = sample_exploration_sample.to_array()
        reconstructed = ExplorationSample.from_array(arr)
        assert reconstructed.storage_noise_r == pytest.approx(sample_exploration_sample.storage_noise_r)
        assert reconstructed.channel_fidelity == pytest.approx(sample_exploration_sample.channel_fidelity)
        assert reconstructed.strategy == sample_exploration_sample.strategy

    def test_sample_validation_fidelity_bounds(self):
        """Test validation of channel_fidelity bounds."""
        with pytest.raises(ValueError, match="channel_fidelity"):
            ExplorationSample(
                storage_noise_r=0.1,
                storage_rate_nu=0.5,
                wait_time_ns=1e6,
                channel_fidelity=0.4,  # Invalid: must be > 0.5
                detection_efficiency=0.8,
                detector_error=0.01,
                dark_count_prob=1e-5,
                num_pairs=100000,
                strategy=ReconciliationStrategy.BASELINE,
            )

    def test_sample_validation_detection_efficiency(self):
        """Test validation of detection_efficiency bounds."""
        with pytest.raises(ValueError, match="detection_efficiency"):
            ExplorationSample(
                storage_noise_r=0.1,
                storage_rate_nu=0.5,
                wait_time_ns=1e6,
                channel_fidelity=0.95,
                detection_efficiency=0.0,  # Invalid: must be > 0
                detector_error=0.01,
                dark_count_prob=1e-5,
                num_pairs=100000,
                strategy=ReconciliationStrategy.BASELINE,
            )

    def test_sample_pickle_roundtrip(self, sample_exploration_sample):
        """Test pickle serialization."""
        pickled = pickle.dumps(sample_exploration_sample)
        unpickled = pickle.loads(pickled)
        assert unpickled == sample_exploration_sample


class TestProtocolResult:
    """Tests for ProtocolResult dataclass."""

    def test_valid_result_creation(self, sample_protocol_result):
        """Test creating a valid protocol result."""
        assert sample_protocol_result.outcome == ProtocolOutcome.SUCCESS
        assert sample_protocol_result.net_efficiency == 0.85
        assert sample_protocol_result.raw_key_length == 50000
        assert sample_protocol_result.final_key_length == 42500
        assert sample_protocol_result.qber_measured == pytest.approx(0.03)

    def test_result_is_success_property(self, sample_exploration_sample):
        """Test is_success() method."""
        success_result = ProtocolResult(
            sample=sample_exploration_sample,
            outcome=ProtocolOutcome.SUCCESS,
            net_efficiency=0.85,
            raw_key_length=50000,
            final_key_length=42500,
            qber_measured=0.03,
            reconciliation_efficiency=0.95,
            leakage_bits=10000,
            execution_time_seconds=2.5,
        )
        assert success_result.is_success() is True

        failure_result = ProtocolResult(
            sample=sample_exploration_sample,
            outcome=ProtocolOutcome.FAILURE_QBER,
            net_efficiency=0.0,
            raw_key_length=50000,
            final_key_length=0,
            qber_measured=0.15,
            reconciliation_efficiency=0.0,
            leakage_bits=0,
            execution_time_seconds=1.0,
        )
        assert failure_result.is_success() is False

    def test_result_to_training_pair(self, sample_protocol_result):
        """Test conversion to GP training pair."""
        X, y = sample_protocol_result.to_training_pair()
        assert isinstance(X, np.ndarray)
        assert X.shape == (9,)
        assert isinstance(y, float)
        assert y == pytest.approx(0.85)


class TestProtocolOutcome:
    """Tests for ProtocolOutcome enum."""

    def test_all_outcomes_defined(self):
        """Test all expected outcomes are defined."""
        expected = {
            "SUCCESS",
            "FAILURE_QBER",
            "FAILURE_RECONCILIATION",
            "FAILURE_SECURITY",
            "FAILURE_TIMEOUT",
            "FAILURE_ERROR",
            "SKIPPED_INFEASIBLE",
            "SKIPPED_PREDICTED_FAILURE",
        }
        actual = {o.name for o in ProtocolOutcome}
        assert actual == expected

    def test_outcome_values_are_strings(self):
        """Test outcome values are lowercase strings."""
        for outcome in ProtocolOutcome:
            assert isinstance(outcome.value, str)
            assert outcome.value == outcome.value.lower()


class TestReconciliationStrategy:
    """Tests for ReconciliationStrategy enum."""

    def test_strategies_defined(self):
        """Test both strategies are defined."""
        assert ReconciliationStrategy.BASELINE.value == "baseline"
        assert ReconciliationStrategy.BLIND.value == "blind"


class TestExplorationPhase:
    """Tests for ExplorationPhase enum."""

    def test_phases_order(self):
        """Test phases are defined in expected order."""
        phases = list(ExplorationPhase)
        assert phases[0] == ExplorationPhase.LHS
        assert phases[1] == ExplorationPhase.SURROGATE
        assert phases[2] == ExplorationPhase.ACTIVE


class TestPhaseStates:
    """Tests for phase state dataclasses."""

    def test_phase1_state_creation(self, phase1_state_partial):
        """Test Phase1State creation."""
        assert phase1_state_partial.target_feasible_samples == 50
        assert phase1_state_partial.feasible_samples_collected == 25
        assert phase1_state_partial.current_phase == "LHS"

    def test_phase1_state_progress(self, phase1_state_partial):
        """Test progress calculation."""
        progress = phase1_state_partial.progress_fraction()
        assert progress == pytest.approx(0.5)

    def test_phase1_state_is_complete(self, phase1_state_partial):
        """Test is_complete check."""
        assert phase1_state_partial.is_complete() is False
        
        complete_state = Phase1State(
            target_feasible_samples=50,
            feasible_samples_collected=50,
            total_samples_processed=60,
            current_batch_start=50,
            rng_state={},
        )
        assert complete_state.is_complete() is True

    def test_phase2_state_creation(self, phase2_state_trained):
        """Test Phase2State creation."""
        assert phase2_state_trained.training_samples_used == 1000
        assert phase2_state_trained.last_training_mse == pytest.approx(0.05)
        assert phase2_state_trained.current_phase == "SURROGATE"

    def test_phase3_state_creation(self, phase3_state_partial):
        """Test Phase3State creation."""
        assert phase3_state_partial.iteration == 10
        assert phase3_state_partial.total_active_samples == 160
        assert phase3_state_partial.current_phase == "ACTIVE"

    def test_phase3_cliff_found(self):
        """Test cliff_found method."""
        no_cliff = Phase3State(
            iteration=10,
            total_active_samples=100,
            best_cliff_point=None,
            best_cliff_efficiency=0.5,
        )
        assert no_cliff.cliff_found() is False

        found_cliff = Phase3State(
            iteration=50,
            total_active_samples=500,
            best_cliff_point=np.array([0.5] * 9),
            best_cliff_efficiency=0.05,
        )
        assert found_cliff.cliff_found() is True


class TestExplorationConfig:
    """Tests for ExplorationConfig dataclass."""

    def test_default_config(self, temp_dir):
        """Test default configuration values."""
        config = ExplorationConfig(output_dir=temp_dir)
        assert config.phase1_samples == 2000
        assert config.phase1_batch_size == 50
        assert config.phase3_iterations == 100
        assert config.random_seed == 42

    def test_config_paths(self, temp_dir):
        """Test derived path properties."""
        config = ExplorationConfig(output_dir=temp_dir)
        assert config.hdf5_path == temp_dir / "exploration_data.h5"
        assert config.checkpoint_path == temp_dir / "checkpoint.pkl"
        assert config.surrogate_path == temp_dir / "surrogate.pkl"

    def test_config_creates_output_dir(self, temp_dir):
        """Test that config creates output directory."""
        nested_dir = temp_dir / "nested" / "exploration"
        config = ExplorationConfig(output_dir=nested_dir)
        assert nested_dir.exists()

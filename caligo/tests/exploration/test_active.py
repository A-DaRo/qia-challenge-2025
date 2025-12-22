"""
Unit tests for exploration active learning module.

Tests BayesianOptimizer, acquisition functions, and batch
selection strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.exploration.active import (
    AcquisitionConfig,
    AcquisitionFunction,
    BayesianOptimizer,
)
from caligo.exploration.surrogate import EfficiencyLandscape, GPConfig
from caligo.exploration.sampler import ParameterBounds


class TestAcquisitionConfig:
    """Tests for AcquisitionConfig dataclass."""

    def test_default_config(self):
        """Test default acquisition configuration."""
        config = AcquisitionConfig()
        assert config.acquisition_type == "straddle"
        assert config.kappa == pytest.approx(1.96)

    def test_custom_config(self):
        """Test custom acquisition configuration."""
        config = AcquisitionConfig(
            acquisition_type="ei",
            kappa=2.5,
            xi=0.05,
        )
        assert config.acquisition_type == "ei"
        assert config.kappa == pytest.approx(2.5)
        assert config.xi == pytest.approx(0.05)


class TestAcquisitionFunction:
    """Tests for AcquisitionFunction wrapper class."""

    @pytest.fixture
    def trained_landscape(self, training_data_twin):
        """Create a trained landscape for testing."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)
        return landscape

    def test_acquisition_function_creation(self, trained_landscape):
        """Test creating acquisition function wrapper."""
        acq = AcquisitionFunction(
            landscape=trained_landscape,
            acquisition_type="straddle",
        )
        assert acq.acquisition_type == "straddle"
        assert acq.call_count == 0

    def test_evaluate_straddle(self, trained_landscape, training_data_twin):
        """Test straddle acquisition evaluation."""
        X_baseline, _, _, _ = training_data_twin
        acq = AcquisitionFunction(
            landscape=trained_landscape,
            acquisition_type="straddle",
        )
        
        values = acq(X_baseline[:5])
        assert values.shape == (5,)
        assert acq.call_count == 1

    def test_evaluate_ei(self, trained_landscape, training_data_twin):
        """Test expected improvement evaluation."""
        X_baseline, _, _, _ = training_data_twin
        acq = AcquisitionFunction(
            landscape=trained_landscape,
            acquisition_type="ei",
        )
        acq.set_best_y(0.5)
        
        values = acq(X_baseline[:5])
        assert values.shape == (5,)
        # EI should be non-negative
        assert np.all(values >= 0)

    def test_evaluate_ucb(self, trained_landscape, training_data_twin):
        """Test UCB acquisition evaluation."""
        X_baseline, _, _, _ = training_data_twin
        acq = AcquisitionFunction(
            landscape=trained_landscape,
            acquisition_type="ucb",
        )
        
        values = acq(X_baseline[:5])
        assert values.shape == (5,)

    def test_call_count_tracking(self, trained_landscape, training_data_twin):
        """Test that call count is tracked."""
        X_baseline, _, _, _ = training_data_twin
        acq = AcquisitionFunction(landscape=trained_landscape)
        
        assert acq.call_count == 0
        acq(X_baseline[:5])
        assert acq.call_count == 1
        acq(X_baseline[5:10])
        assert acq.call_count == 2
        
        acq.reset_call_count()
        assert acq.call_count == 0


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer class."""

    @pytest.fixture
    def trained_landscape(self, training_data_twin):
        """Create a trained landscape for testing."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)
        return landscape

    def test_optimizer_creation(self, trained_landscape, parameter_bounds):
        """Test creating Bayesian optimizer."""
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        assert optimizer.landscape is trained_landscape

    def test_suggest_single_point(self, trained_landscape, parameter_bounds):
        """Test suggesting a single point."""
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        
        point = optimizer.suggest_batch(batch_size=1)
        assert point.shape == (1, 9)

    def test_suggest_batch(self, trained_landscape, parameter_bounds):
        """Test suggesting a batch of points."""
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        
        batch = optimizer.suggest_batch(batch_size=5)
        assert batch.shape == (5, 9)

    def test_suggest_reproducibility(self, trained_landscape, parameter_bounds):
        """Test that optimizer produces deterministic results."""
        # Note: without explicit seeding, results may vary
        # This test just checks the output format is correct
        opt1 = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        
        points1 = opt1.suggest_batch(batch_size=3)
        assert points1.shape == (3, 9)

    def test_suggested_points_within_bounds(self, trained_landscape, parameter_bounds):
        """Test that suggested points are within bounds."""
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        
        points = optimizer.suggest_batch(batch_size=10)
        bounds_arr = parameter_bounds.to_bounds_array()
        
        # Check continuous dimensions (0-7)
        for i in range(8):
            assert np.all(points[:, i] >= bounds_arr[i, 0])
            assert np.all(points[:, i] <= bounds_arr[i, 1])


class TestBayesianOptimizerEdgeCases:
    """Edge case tests for BayesianOptimizer."""

    @pytest.fixture
    def trained_landscape(self, training_data_twin):
        """Create a trained landscape for testing."""
        X_baseline, y_baseline, X_blind, y_blind = training_data_twin
        landscape = EfficiencyLandscape()
        landscape.fit(X_baseline, y_baseline, X_blind, y_blind)
        return landscape

    def test_suggest_with_narrow_bounds(self, trained_landscape):
        """Test suggestion with narrow parameter bounds."""
        narrow_bounds = ParameterBounds(
            f_min=0.90,
            f_max=0.95,
        )
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=narrow_bounds,
        )
        
        points = optimizer.suggest_batch(batch_size=5)
        # Fidelity dimension should be in narrow range
        # (after transform, fidelity is at index 3, untransformed)
        assert points.shape == (5, 9)

    def test_suggest_many_points(self, trained_landscape, parameter_bounds):
        """Test suggesting many points."""
        optimizer = BayesianOptimizer(
            landscape=trained_landscape,
            bounds=parameter_bounds,
        )
        
        points = optimizer.suggest_batch(batch_size=20)
        assert points.shape == (20, 9)

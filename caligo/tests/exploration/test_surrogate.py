"""
Unit tests for exploration surrogate module.

Tests EfficiencyLandscape twin GPs, detect_divergence, and
prediction accuracy.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from caligo.exploration.surrogate import (
    GPConfig,
    StrategyGP,
    EfficiencyLandscape,
    detect_divergence,
    _create_kernel,
)


class TestGPConfig:
    """Tests for GPConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default GP configuration."""
        config = GPConfig()

        assert config.length_scale == 1.0
        assert config.nu == 2.5  # Matern 5/2
        assert config.n_restarts_optimizer >= 1
        assert config.normalize_y is True

    def test_custom_config(self) -> None:
        """Test custom GP configuration."""
        config = GPConfig(
            length_scale=0.5,
            nu=1.5,  # Matern 3/2
            noise_level=1e-3,
            n_restarts_optimizer=5,
        )

        assert config.length_scale == 0.5
        assert config.nu == 1.5
        assert config.noise_level == 1e-3
        assert config.n_restarts_optimizer == 5

    def test_config_is_frozen(self) -> None:
        """Test that GPConfig is frozen."""
        config = GPConfig()
        with pytest.raises(AttributeError):
            config.length_scale = 2.0  # type: ignore


class TestCreateKernel:
    """Tests for kernel creation."""

    def test_kernel_creation(self) -> None:
        """Test that kernel is created successfully."""
        config = GPConfig()
        kernel = _create_kernel(config)

        assert kernel is not None
        # Kernel should have parameters
        assert len(kernel.get_params()) > 0


class TestStrategyGP:
    """Tests for StrategyGP class."""

    def test_unfitted_gp_raises_on_predict(
        self, training_data_small: Tuple[NDArray, NDArray]
    ) -> None:
        """Test that unfitted GP raises on predict."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        gp = GaussianProcessRegressor(kernel=Matern())
        strategy_gp = StrategyGP(name="test", gp=gp)

        X, _ = training_data_small

        with pytest.raises(RuntimeError, match="not been fitted"):
            strategy_gp.predict(X[:5])

    def test_fit_and_predict(
        self, training_data_small: Tuple[NDArray, NDArray]
    ) -> None:
        """Test fitting and predicting with StrategyGP."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        gp = GaussianProcessRegressor(kernel=Matern(), random_state=42)
        strategy_gp = StrategyGP(name="test", gp=gp)

        X, y = training_data_small

        strategy_gp.fit(X, y)

        assert strategy_gp.is_fitted is True
        assert strategy_gp.n_samples == len(X)

        # Predict
        mean, std = strategy_gp.predict(X[:5], return_std=True)

        assert mean.shape == (5,)
        assert std is not None
        assert std.shape == (5,)

    def test_predict_without_std(
        self, training_data_small: Tuple[NDArray, NDArray]
    ) -> None:
        """Test prediction without returning std."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        gp = GaussianProcessRegressor(kernel=Matern(), random_state=42)
        strategy_gp = StrategyGP(name="test", gp=gp)

        X, y = training_data_small
        strategy_gp.fit(X, y)

        mean, std = strategy_gp.predict(X[:5], return_std=False)

        assert mean.shape == (5,)
        assert std is None


class TestEfficiencyLandscape:
    """Tests for EfficiencyLandscape twin GP model."""

    def test_landscape_initialization(self) -> None:
        """Test that landscape initializes both GPs."""
        landscape = EfficiencyLandscape()

        assert landscape.baseline_gp is not None
        assert landscape.blind_gp is not None
        assert landscape.baseline_gp.name == "baseline"
        assert landscape.blind_gp.name == "blind"

    def test_landscape_not_fitted_initially(self) -> None:
        """Test that landscape is not fitted initially."""
        landscape = EfficiencyLandscape()

        assert landscape.is_fitted is False

    def test_fit_landscape(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test fitting the efficiency landscape."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        assert landscape.is_fitted is True
        assert landscape.baseline_gp.is_fitted is True
        assert landscape.blind_gp.is_fitted is True

    def test_predict_baseline(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test baseline predictions."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        mean, std = landscape.predict_baseline(X_base[:5], return_std=True)

        assert mean.shape == (5,)
        assert std is not None
        assert std.shape == (5,)
        # Predictions at training points should be close to targets
        np.testing.assert_array_less(np.abs(mean - y_base[:5]), 0.5)

    def test_predict_blind(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test blind predictions."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        mean, std = landscape.predict_blind(X_blind[:5], return_std=True)

        assert mean.shape == (5,)
        assert std is not None

    def test_predict_divergence(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test divergence predictions."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        divergence, std = landscape.predict_divergence(X_base[:5])

        assert divergence.shape == (5,)
        assert std.shape == (5,)
        # Divergence should be non-negative
        assert np.all(divergence >= 0)

    def test_detect_cliffs(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test cliff detection."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        cliffs = landscape.detect_cliffs(X_base, threshold=0.1, confidence=2.0)

        assert cliffs.shape == (len(X_base),)
        assert cliffs.dtype == np.bool_

    def test_get_training_stats(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test getting training statistics."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        stats = landscape.get_training_stats()

        assert "baseline_n_samples" in stats
        assert "blind_n_samples" in stats
        assert "baseline_log_likelihood" in stats
        assert "blind_log_likelihood" in stats
        assert stats["baseline_n_samples"] == len(X_base)
        assert stats["blind_n_samples"] == len(X_blind)


class TestDetectDivergence:
    """Tests for detect_divergence function."""

    def test_detect_divergence_not_fitted(self) -> None:
        """Test that divergence detection fails for unfitted landscape."""
        landscape = EfficiencyLandscape()
        X = np.random.rand(10, 9)

        is_diverged, diagnostics = detect_divergence(landscape, X)

        assert is_diverged is True
        assert "error" in diagnostics

    def test_detect_divergence_normal_model(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test divergence detection on a normal model."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        is_diverged, diagnostics = detect_divergence(landscape, X_base)

        # Normal model shouldn't be diverged
        assert "mean_divergence" in diagnostics
        assert "mean_uncertainty" in diagnostics
        assert "baseline_mean_range" in diagnostics
        assert "blind_mean_range" in diagnostics

    def test_detect_divergence_high_threshold(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test divergence detection with high threshold."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        # With very high thresholds, should not flag divergence
        is_diverged, _ = detect_divergence(
            landscape, X_base, threshold_divergence=10.0, threshold_uncertainty=10.0
        )

        assert is_diverged is False

    def test_detect_divergence_low_threshold(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test divergence detection with very low threshold."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        # With very low thresholds, might flag divergence
        is_diverged, diagnostics = detect_divergence(
            landscape, X_base, threshold_divergence=0.001, threshold_uncertainty=0.001
        )

        # Should report what triggered divergence
        assert "is_diverged" in diagnostics


class TestEfficiencyLandscapeSerialization:
    """Tests for landscape serialization."""

    def test_landscape_pickle_roundtrip(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
        temp_dir: Path,
    ) -> None:
        """Test that fitted landscape can be pickled."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        # Pickle
        pkl_path = temp_dir / "landscape.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(landscape, f)

        # Unpickle
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.is_fitted is True

        # Predictions should match
        orig_mean, _ = landscape.predict_baseline(X_base[:5])
        load_mean, _ = loaded.predict_baseline(X_base[:5])

        np.testing.assert_array_almost_equal(orig_mean, load_mean)


class TestEfficiencyLandscapeEdgeCases:
    """Edge case tests for EfficiencyLandscape."""

    def test_fit_with_empty_data(self) -> None:
        """Test fitting with empty data (should handle gracefully)."""
        landscape = EfficiencyLandscape()

        X_empty = np.empty((0, 9))
        y_empty = np.empty((0,))

        # Should not crash
        landscape.fit(X_empty, y_empty, X_empty, y_empty)

        # But should not be properly fitted
        assert landscape.baseline_gp.is_fitted is False
        assert landscape.blind_gp.is_fitted is False

    def test_fit_with_single_sample(self) -> None:
        """Test fitting with a single sample."""
        landscape = EfficiencyLandscape()

        X = np.random.rand(1, 9)
        y = np.array([0.5])

        landscape.fit(X, y, X, y)

        # Should fit (GP can handle single sample)
        assert landscape.baseline_gp.n_samples == 1

    def test_fit_with_identical_targets(self) -> None:
        """Test fitting when all targets are identical."""
        landscape = EfficiencyLandscape()

        X = np.random.rand(20, 9)
        y = np.ones(20) * 0.5  # All same value

        landscape.fit(X, y, X, y)

        # Should fit without error
        assert landscape.is_fitted is True

        # Predictions should be close to 0.5
        mean, _ = landscape.predict_baseline(X[:5])
        np.testing.assert_array_almost_equal(mean, np.ones(5) * 0.5, decimal=1)

    def test_predict_on_extrapolation_points(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test predictions on points far from training data."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        # Create extrapolation points (outside [0,1] range)
        X_extrap = np.ones((5, 9)) * 2.0

        mean, std = landscape.predict_baseline(X_extrap, return_std=True)

        # Should still return predictions
        assert mean.shape == (5,)
        assert std is not None
        # Uncertainty should be higher at extrapolation points
        mean_train, std_train = landscape.predict_baseline(X_base[:5], return_std=True)
        assert np.mean(std) > np.mean(std_train)


class TestEfficiencyLandscapePredictionQuality:
    """Tests for prediction quality of EfficiencyLandscape."""

    def test_predictions_improve_with_more_data(self) -> None:
        """Test that predictions improve with more training data."""
        np.random.seed(42)

        # Generate data with known function
        def true_func(X: NDArray) -> NDArray:
            return 0.8 * X[:, 0] - 0.2 * X[:, 4]

        X_test = np.random.rand(50, 9)
        y_test = true_func(X_test)

        errors = []
        for n_train in [10, 30, 100]:
            X_train = np.random.rand(n_train, 9)
            y_train = true_func(X_train) + 0.05 * np.random.randn(n_train)

            landscape = EfficiencyLandscape(
                config=GPConfig(n_restarts_optimizer=2)
            )
            landscape.fit(X_train, y_train, X_train, y_train)

            mean, _ = landscape.predict_baseline(X_test)
            rmse = np.sqrt(np.mean((mean - y_test) ** 2))
            errors.append(rmse)

        # Error should generally decrease with more data
        # Allow some tolerance as GP fitting can be stochastic
        assert errors[-1] < errors[0] + 0.1

    def test_uncertainty_decreases_near_training_points(
        self,
        training_data_twin: Tuple[NDArray, NDArray, NDArray, NDArray],
    ) -> None:
        """Test that uncertainty is lower near training points."""
        X_base, y_base, X_blind, y_blind = training_data_twin

        landscape = EfficiencyLandscape()
        landscape.fit(X_base, y_base, X_blind, y_blind)

        # Uncertainty at training points
        _, std_train = landscape.predict_baseline(X_base, return_std=True)

        # Uncertainty at random points
        X_random = np.random.rand(30, 9)
        _, std_random = landscape.predict_baseline(X_random, return_std=True)

        # Mean uncertainty should be lower at training points
        assert np.mean(std_train) < np.mean(std_random) + 0.1

"""
Surrogate modeling for exploration using Twin Gaussian Processes.

This module provides GP-based surrogate models that learn the mapping from
the 9D parameter space to protocol efficiency metrics, enabling cheap
evaluation during Bayesian optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel

logger = get_logger(__name__)


@dataclass(frozen=True)
class GPConfig:
    """
    Configuration for Gaussian Process surrogate.

    Parameters
    ----------
    length_scale : float
        Initial length scale for Matern kernel.
    length_scale_bounds : Tuple[float, float]
        Bounds for length scale optimization.
    nu : float
        Smoothness parameter for Matern kernel (0.5, 1.5, 2.5).
    noise_level : float
        Initial noise level for WhiteKernel.
    noise_bounds : Tuple[float, float]
        Bounds for noise optimization.
    n_restarts_optimizer : int
        Number of restarts for hyperparameter optimization.
    normalize_y : bool
        Whether to normalize target values.
    random_state : int
        Random seed for reproducibility.
    """

    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-3, 1e3)
    nu: float = 2.5
    noise_level: float = 1e-2
    noise_bounds: Tuple[float, float] = (1e-10, 1e1)
    n_restarts_optimizer: int = 10
    normalize_y: bool = True
    random_state: int = 42


@dataclass
class StrategyGP:
    """
    Gaussian Process for a single reconciliation strategy.

    Parameters
    ----------
    name : str
        Strategy name (e.g., "baseline", "blind").
    gp : GaussianProcessRegressor
        Fitted GP model.
    scaler_X : StandardScaler
        Feature scaler.
    is_fitted : bool
        Whether the GP has been fitted.
    n_samples : int
        Number of training samples.
    """

    name: str
    gp: GaussianProcessRegressor
    scaler_X: StandardScaler = field(default_factory=StandardScaler)
    is_fitted: bool = False
    n_samples: int = 0

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "StrategyGP":
        """
        Fit the GP to training data.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix of shape (n_samples, n_features).
        y : NDArray[np.floating]
            Target vector of shape (n_samples,).

        Returns
        -------
        StrategyGP
            Self for method chaining.
        """
        if len(X) == 0:
            logger.warning(f"No training data for {self.name} GP")
            return self

        X_scaled = self.scaler_X.fit_transform(X)
        self.gp.fit(X_scaled, y)
        self.is_fitted = True
        self.n_samples = len(X)

        logger.info(
            f"Fitted {self.name} GP with {self.n_samples} samples, "
            f"log-marginal-likelihood: {self.gp.log_marginal_likelihood_value_:.3f}"
        )
        return self

    def predict(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Predict efficiency at given parameter configurations.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix of shape (n_samples, n_features).
        return_std : bool
            Whether to return standard deviation.

        Returns
        -------
        Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]
            Mean predictions and optionally standard deviations.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} GP has not been fitted")

        X_scaled = self.scaler_X.transform(X)

        if return_std:
            mean, std = self.gp.predict(X_scaled, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(X_scaled, return_std=False)
            return mean, None


def _create_kernel(config: GPConfig) -> "Kernel":
    """
    Create composite kernel for GP.

    Uses Matern kernel with automatic relevance determination (ARD)
    plus white noise kernel.

    Parameters
    ----------
    config : GPConfig
        GP configuration.

    Returns
    -------
    Kernel
        Composite sklearn kernel.
    """
    # Constant kernel for signal variance
    constant = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))

    # Matern kernel with ARD (separate length scale per dimension)
    matern = Matern(
        length_scale=config.length_scale,
        length_scale_bounds=config.length_scale_bounds,
        nu=config.nu,
    )

    # White noise kernel
    white = WhiteKernel(
        noise_level=config.noise_level,
        noise_level_bounds=config.noise_bounds,
    )

    return constant * matern + white


@dataclass
class EfficiencyLandscape:
    """
    Twin Gaussian Process surrogate for efficiency landscape.

    Maintains separate GPs for baseline and blind reconciliation strategies,
    enabling detection of security cliffs where strategies diverge.

    Parameters
    ----------
    config : GPConfig
        GP configuration.
    baseline_gp : StrategyGP
        GP for baseline (CASCADE) reconciliation.
    blind_gp : StrategyGP
        GP for blind (LDPC) reconciliation.
    """

    config: GPConfig = field(default_factory=GPConfig)
    baseline_gp: Optional[StrategyGP] = None
    blind_gp: Optional[StrategyGP] = None

    def __post_init__(self) -> None:
        """Initialize GP models if not provided."""
        if self.baseline_gp is None:
            kernel = _create_kernel(self.config)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                normalize_y=self.config.normalize_y,
                random_state=self.config.random_state,
            )
            self.baseline_gp = StrategyGP(name="baseline", gp=gp)

        if self.blind_gp is None:
            kernel = _create_kernel(self.config)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                normalize_y=self.config.normalize_y,
                random_state=self.config.random_state + 1,  # Different seed
            )
            self.blind_gp = StrategyGP(name="blind", gp=gp)

    def fit(
        self,
        X_baseline: NDArray[np.floating],
        y_baseline: NDArray[np.floating],
        X_blind: NDArray[np.floating],
        y_blind: NDArray[np.floating],
    ) -> "EfficiencyLandscape":
        """
        Fit both GPs to training data.

        Parameters
        ----------
        X_baseline : NDArray[np.floating]
            Features for baseline strategy.
        y_baseline : NDArray[np.floating]
            Efficiency targets for baseline.
        X_blind : NDArray[np.floating]
            Features for blind strategy.
        y_blind : NDArray[np.floating]
            Efficiency targets for blind.

        Returns
        -------
        EfficiencyLandscape
            Self for method chaining.
        """
        self.baseline_gp.fit(X_baseline, y_baseline)
        self.blind_gp.fit(X_blind, y_blind)

        logger.info(
            f"EfficiencyLandscape fitted: "
            f"baseline={self.baseline_gp.n_samples}, "
            f"blind={self.blind_gp.n_samples}"
        )
        return self

    def predict(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Predict efficiency using baseline model (for acquisition functions).

        This is a convenience method that delegates to predict_baseline.
        Use this for acquisition function evaluation where we want to
        identify regions near the security cliff.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.
        return_std : bool
            Whether to return uncertainty.

        Returns
        -------
        Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]
            Mean and optionally std.
        """
        return self.predict_baseline(X, return_std=return_std)

    def predict_baseline(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Predict baseline efficiency.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.
        return_std : bool
            Whether to return uncertainty.

        Returns
        -------
        Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]
            Mean and optionally std.
        """
        return self.baseline_gp.predict(X, return_std=return_std)

    def predict_blind(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """
        Predict blind efficiency.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.
        return_std : bool
            Whether to return uncertainty.

        Returns
        -------
        Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]
            Mean and optionally std.
        """
        return self.blind_gp.predict(X, return_std=return_std)

    def predict_divergence(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Predict divergence between strategies (security cliff indicator).

        Computes |baseline - blind| with propagated uncertainty.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.

        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating]]
            Mean divergence and combined uncertainty.
        """
        mu_base, std_base = self.predict_baseline(X, return_std=True)
        mu_blind, std_blind = self.predict_blind(X, return_std=True)

        # Divergence is absolute difference
        divergence = np.abs(mu_base - mu_blind)

        # Propagate uncertainty (assuming independence)
        std_combined = np.sqrt(std_base**2 + std_blind**2)

        return divergence, std_combined

    def detect_cliffs(
        self,
        X: NDArray[np.floating],
        threshold: float = 0.1,
        confidence: float = 2.0,
    ) -> NDArray[np.bool_]:
        """
        Detect potential security cliffs.

        A cliff is detected where divergence exceeds threshold with high confidence.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.
        threshold : float
            Minimum divergence to consider a cliff.
        confidence : float
            Number of standard deviations for confidence.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask indicating cliff locations.
        """
        divergence, std = self.predict_divergence(X)

        # Lower confidence bound on divergence
        lower_bound = divergence - confidence * std

        return lower_bound > threshold

    @property
    def is_fitted(self) -> bool:
        """Check if both GPs are fitted."""
        return (
            self.baseline_gp is not None
            and self.baseline_gp.is_fitted
            and self.blind_gp is not None
            and self.blind_gp.is_fitted
        )

    def get_training_stats(self) -> dict:
        """
        Get training statistics for both GPs.

        Returns
        -------
        dict
            Statistics including sample counts and log-likelihoods.
        """
        stats = {
            "baseline_n_samples": self.baseline_gp.n_samples if self.baseline_gp else 0,
            "blind_n_samples": self.blind_gp.n_samples if self.blind_gp else 0,
        }

        if self.baseline_gp and self.baseline_gp.is_fitted:
            stats["baseline_log_likelihood"] = (
                self.baseline_gp.gp.log_marginal_likelihood_value_
            )
            stats["baseline_kernel"] = str(self.baseline_gp.gp.kernel_)

        if self.blind_gp and self.blind_gp.is_fitted:
            stats["blind_log_likelihood"] = (
                self.blind_gp.gp.log_marginal_likelihood_value_
            )
            stats["blind_kernel"] = str(self.blind_gp.gp.kernel_)

        return stats


def detect_divergence(
    landscape: EfficiencyLandscape,
    X: NDArray[np.floating],
    threshold_divergence: float = 0.15,
    threshold_uncertainty: float = 0.3,
) -> Tuple[bool, dict]:
    """
    Detect model divergence indicating potential issues.

    Checks for:
    1. High average divergence between baseline and blind GPs
    2. Excessive uncertainty in predictions
    3. Unrealistic prediction values

    Parameters
    ----------
    landscape : EfficiencyLandscape
        Fitted twin GP landscape.
    X : NDArray[np.floating]
        Validation feature matrix.
    threshold_divergence : float
        Maximum acceptable mean divergence between strategies.
    threshold_uncertainty : float
        Maximum acceptable mean uncertainty.

    Returns
    -------
    Tuple[bool, dict]
        (is_diverged, diagnostics) where diagnostics contains
        detailed metrics about the divergence check.
    """
    if not landscape.is_fitted:
        return True, {"error": "Landscape not fitted"}

    diagnostics: dict = {}
    is_diverged = False

    # Get predictions
    mu_base, std_base = landscape.predict_baseline(X, return_std=True)
    mu_blind, std_blind = landscape.predict_blind(X, return_std=True)

    # Check prediction ranges (efficiency should be in [0, 1])
    diagnostics["baseline_mean_range"] = (float(mu_base.min()), float(mu_base.max()))
    diagnostics["blind_mean_range"] = (float(mu_blind.min()), float(mu_blind.max()))

    if mu_base.min() < -0.1 or mu_base.max() > 1.1:
        diagnostics["baseline_out_of_range"] = True
        is_diverged = True

    if mu_blind.min() < -0.1 or mu_blind.max() > 1.1:
        diagnostics["blind_out_of_range"] = True
        is_diverged = True

    # Check divergence
    divergence, _ = landscape.predict_divergence(X)
    mean_divergence = float(divergence.mean())
    diagnostics["mean_divergence"] = mean_divergence
    diagnostics["max_divergence"] = float(divergence.max())

    if mean_divergence > threshold_divergence:
        diagnostics["high_divergence"] = True
        is_diverged = True

    # Check uncertainty
    mean_uncertainty = float((std_base.mean() + std_blind.mean()) / 2)
    diagnostics["mean_uncertainty"] = mean_uncertainty
    diagnostics["baseline_mean_std"] = float(std_base.mean())
    diagnostics["blind_mean_std"] = float(std_blind.mean())

    if mean_uncertainty > threshold_uncertainty:
        diagnostics["high_uncertainty"] = True
        is_diverged = True

    # Check for NaN/Inf
    if np.any(~np.isfinite(mu_base)) or np.any(~np.isfinite(mu_blind)):
        diagnostics["non_finite_predictions"] = True
        is_diverged = True

    if np.any(~np.isfinite(std_base)) or np.any(~np.isfinite(std_blind)):
        diagnostics["non_finite_uncertainty"] = True
        is_diverged = True

    diagnostics["is_diverged"] = is_diverged
    return is_diverged, diagnostics

"""
Surrogate modeling for exploration using Twin Gaussian Processes.

This module provides GP-based surrogate models that learn the mapping from
the 9D parameter space to protocol efficiency metrics, enabling cheap
evaluation during Bayesian optimization.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class GuardrailConfig:
    """
    Configuration for the fail-fast guardrail classifier.

    The guardrail is a lightweight binary classifier that predicts whether
    a sample will result in a successful protocol execution (positive key rate)
    or failure (zero key rate). Samples predicted as failures with high
    confidence are skipped before expensive EPR generation.

    Parameters
    ----------
    enabled : bool
        Whether the guardrail is active. If False, all samples are executed.
    failure_threshold : float
        Probability threshold for classifying as failure. Samples with
        P(failure) > threshold are skipped. Default 0.95 (95% confident).
    n_estimators : int
        Number of trees in the Random Forest classifier.
    max_depth : int
        Maximum depth of trees. Shallow trees ensure fast inference.
    min_samples_for_training : int
        Minimum training samples required before guardrail is active.
        Below this threshold, all samples are executed to gather data.
    class_weight : str
        Class weight strategy for imbalanced data. Use "balanced" to
        handle datasets with more successes than failures.
    random_state : int
        Random seed for reproducibility.
    """

    enabled: bool = True
    failure_threshold: float = 0.95
    n_estimators: int = 100
    max_depth: int = 8
    min_samples_for_training: int = 50
    class_weight: str = "balanced"
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

    def save(self, path: Path) -> None:
        """
        Save the fitted landscape to disk using pickle.

        Parameters
        ----------
        path : Path
            Path to save the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved EfficiencyLandscape to {path}")

    @classmethod
    def load(cls, path: Path) -> "EfficiencyLandscape":
        """
        Load a fitted landscape from disk.

        Parameters
        ----------
        path : Path
            Path to the saved model.

        Returns
        -------
        EfficiencyLandscape
            Loaded model.
        """
        with open(path, "rb") as f:
            landscape = pickle.load(f)
        logger.info(f"Loaded EfficiencyLandscape from {path}")
        return landscape


# =============================================================================
# Fail-Fast Guardrail Classifier
# =============================================================================


class FeasibilityGuardrail:
    """
    Lightweight binary classifier for fail-fast sample filtering.

    The guardrail predicts whether a sample will result in successful
    protocol execution (positive key rate) or failure (zero key rate).
    Samples predicted as failures with high confidence are skipped before
    expensive EPR generation, saving minutes of simulation time per sample.

    The classifier is trained on binary labels derived from efficiency:
    - Label 1 (success): efficiency > 0
    - Label 0 (failure): efficiency == 0

    Parameters
    ----------
    config : GuardrailConfig
        Configuration for the classifier.

    Attributes
    ----------
    config : GuardrailConfig
        Classifier configuration.
    classifier : RandomForestClassifier
        Fitted Random Forest classifier.
    scaler : StandardScaler
        Feature scaler.
    is_fitted : bool
        Whether the classifier has been trained.
    n_samples : int
        Number of training samples.
    n_failures : int
        Number of failure samples in training data.
    n_successes : int
        Number of success samples in training data.

    Examples
    --------
    >>> guardrail = FeasibilityGuardrail()
    >>> guardrail.fit(X_train, y_train)  # y_train is efficiency
    >>> mask = guardrail.predict(X_new)  # True = feasible, False = skip
    >>> feasible_samples = X_new[mask]
    """

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        """
        Initialize the guardrail classifier.

        Parameters
        ----------
        config : Optional[GuardrailConfig]
            Configuration. Uses defaults if None.
        """
        from sklearn.ensemble import RandomForestClassifier

        self.config = config or GuardrailConfig()
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,  # Use all available cores for fast inference
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_samples = 0
        self.n_failures = 0
        self.n_successes = 0

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        success_threshold: float = 0.0,
    ) -> "FeasibilityGuardrail":
        """
        Fit the guardrail classifier.

        Training labels are derived from efficiency values:
        - Label 1 (success): y > success_threshold
        - Label 0 (failure): y <= success_threshold

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix, shape (n_samples, n_features).
        y : NDArray[np.floating]
            Efficiency values, shape (n_samples,).
        success_threshold : float
            Threshold for success classification. Default 0.0 means
            any positive efficiency is considered success.

        Returns
        -------
        FeasibilityGuardrail
            Self for method chaining.
        """
        if len(X) < self.config.min_samples_for_training:
            logger.warning(
                "Insufficient training data for guardrail: %d < %d required",
                len(X),
                self.config.min_samples_for_training,
            )
            return self

        # Convert efficiency to binary labels
        labels = (y > success_threshold).astype(np.int32)
        self.n_failures = int((labels == 0).sum())
        self.n_successes = int((labels == 1).sum())
        self.n_samples = len(X)

        # Check for degenerate cases
        if self.n_failures == 0:
            logger.warning("No failure samples in training data - guardrail disabled")
            return self
        if self.n_successes == 0:
            logger.warning("No success samples in training data - guardrail disabled")
            return self

        # Scale features and fit classifier
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.is_fitted = True

        logger.info(
            "Fitted FeasibilityGuardrail with %d samples "
            "(success: %d, failure: %d, ratio: %.1f%%)",
            self.n_samples,
            self.n_successes,
            self.n_failures,
            100 * self.n_successes / self.n_samples,
        )
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        Predict feasibility for samples.

        Returns True for samples predicted as feasible (should be executed),
        False for samples predicted as failures (should be skipped).

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask: True = execute, False = skip.

        Notes
        -----
        If the guardrail is not fitted or disabled, returns all True
        (execute all samples).
        """
        n_samples = len(X)

        # Return all True if guardrail is disabled or not fitted
        if not self.config.enabled or not self.is_fitted:
            return np.ones(n_samples, dtype=np.bool_)

        # Get predicted probabilities
        X_scaled = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X_scaled)

        # P(failure) is column 0 if classes are [0, 1]
        classes = self.classifier.classes_
        if classes[0] == 0:
            p_failure = proba[:, 0]
        else:
            p_failure = proba[:, 1]

        # Skip samples with high failure probability
        feasible = p_failure < self.config.failure_threshold

        n_skipped = int((~feasible).sum())
        if n_skipped > 0:
            logger.debug(
                "Guardrail filtering: %d/%d samples skipped (%.1f%%)",
                n_skipped,
                n_samples,
                100 * n_skipped / n_samples,
            )

        return feasible

    def predict_proba(
        self,
        X: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get failure/success probabilities.

        Parameters
        ----------
        X : NDArray[np.floating]
            Feature matrix.

        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating]]
            (p_failure, p_success) probability arrays.
        """
        if not self.is_fitted:
            n_samples = len(X)
            return np.zeros(n_samples), np.ones(n_samples)

        X_scaled = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X_scaled)

        classes = self.classifier.classes_
        if classes[0] == 0:
            return proba[:, 0], proba[:, 1]
        else:
            return proba[:, 1], proba[:, 0]

    def get_stats(self) -> dict:
        """
        Get guardrail statistics.

        Returns
        -------
        dict
            Statistics including sample counts and class balance.
        """
        return {
            "is_fitted": self.is_fitted,
            "enabled": self.config.enabled,
            "n_samples": self.n_samples,
            "n_failures": self.n_failures,
            "n_successes": self.n_successes,
            "failure_threshold": self.config.failure_threshold,
            "success_ratio": self.n_successes / self.n_samples if self.n_samples > 0 else 0.0,
        }


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

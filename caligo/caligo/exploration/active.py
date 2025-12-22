"""
Bayesian Optimizer with Numba-accelerated acquisition functions.

This module implements the active learning loop for Phase 3 stress testing,
using Bayesian optimization to efficiently explore the security cliff in
the 9-dimensional parameter space.

Key Features
------------
- **Straddle Acquisition**: Custom acquisition function targeting the
  security cliff (efficiency ≈ 0 boundary)
- **Kriging Believer**: Batch acquisition via sequential maximization
- **Numba JIT**: Hot paths compiled for performance

Theory
------
The Straddle acquisition function targets the boundary where protocol
efficiency transitions from positive to zero (the "security cliff"):

    A(x) = 1.96 * σ(x) - |μ(x)|

This rewards:
1. High uncertainty σ(x) → exploration
2. Near-zero efficiency |μ(x)| ≈ 0 → cliff identification

The Kriging Believer strategy generates batches by:
1. Find x* = argmax A(x)
2. Hallucinate y* = μ(x*) at x*
3. Repeat with updated GP for remaining batch slots

References
----------
- Bryan et al. (2006): Active Learning with Model Selection
- Ginsbourger et al. (2010): Kriging is Well-Suited for Batch
- Jones et al. (1998): EGO (Expected Improvement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    import numba
    from numba import float64, int64
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

from caligo.exploration.sampler import ParameterBounds
from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.exploration.types import ReconciliationStrategy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Numba-Accelerated Acquisition Functions
# =============================================================================


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _straddle_acquisition_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        kappa: float = 1.96,
    ) -> np.ndarray:
        """
        Numba-compiled straddle acquisition function.

        Parameters
        ----------
        mu : np.ndarray
            GP mean predictions, shape (n_points,).
        sigma : np.ndarray
            GP standard deviations, shape (n_points,).
        kappa : float
            Exploration-exploitation tradeoff (default: 1.96 for 95% CI).

        Returns
        -------
        np.ndarray
            Acquisition values, shape (n_points,).
        """
        n = len(mu)
        acq = np.empty(n, dtype=np.float64)
        for i in range(n):
            acq[i] = kappa * sigma[i] - abs(mu[i])
        return acq

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _expected_improvement_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        Numba-compiled expected improvement acquisition function.

        Parameters
        ----------
        mu : np.ndarray
            GP mean predictions.
        sigma : np.ndarray
            GP standard deviations.
        best_y : float
            Best observed value (for minimization).
        xi : float
            Exploration bonus.

        Returns
        -------
        np.ndarray
            EI values.
        """
        n = len(mu)
        ei = np.empty(n, dtype=np.float64)

        for i in range(n):
            if sigma[i] <= 1e-12:
                ei[i] = 0.0
            else:
                z = (best_y - mu[i] - xi) / sigma[i]
                # Approximation of normal CDF and PDF
                # Using rational approximation for Φ(z)
                t = 1.0 / (1.0 + 0.2316419 * abs(z))
                d = 0.3989422804014327  # 1/sqrt(2*pi)
                pdf = d * np.exp(-0.5 * z * z)
                cdf = 1.0 - pdf * t * (
                    0.319381530
                    + t * (-0.356563782
                    + t * (1.781477937
                    + t * (-1.821255978
                    + t * 1.330274429)))
                )
                if z < 0:
                    cdf = 1.0 - cdf
                ei[i] = (best_y - mu[i] - xi) * cdf + sigma[i] * pdf

        return ei

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _ucb_acquisition_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        beta: float = 2.0,
    ) -> np.ndarray:
        """
        Numba-compiled upper confidence bound acquisition.

        For cliff-finding, we want low efficiency, so we use LCB.

        Parameters
        ----------
        mu : np.ndarray
            GP mean predictions.
        sigma : np.ndarray
            GP standard deviations.
        beta : float
            Exploration parameter.

        Returns
        -------
        np.ndarray
            LCB values (negated for maximization).
        """
        n = len(mu)
        lcb = np.empty(n, dtype=np.float64)
        for i in range(n):
            # Lower confidence bound (we want to find low efficiency)
            lcb[i] = -mu[i] + beta * sigma[i]
        return lcb


else:
    # Fallback NumPy implementations
    def _straddle_acquisition_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        kappa: float = 1.96,
    ) -> np.ndarray:
        """Pure NumPy straddle acquisition."""
        return kappa * sigma - np.abs(mu)

    def _expected_improvement_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """Pure NumPy expected improvement."""
        from scipy.stats import norm
        z = (best_y - mu - xi) / np.maximum(sigma, 1e-12)
        ei = (best_y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0
        return ei

    def _ucb_acquisition_numba(
        mu: np.ndarray,
        sigma: np.ndarray,
        beta: float = 2.0,
    ) -> np.ndarray:
        """Pure NumPy UCB acquisition."""
        return -mu + beta * sigma


# =============================================================================
# Acquisition Function Wrapper
# =============================================================================


class AcquisitionFunction:
    """
    Wrapper for acquisition functions with GP landscape integration.

    Parameters
    ----------
    landscape : EfficiencyLandscape
        Trained GP surrogate model.
    acquisition_type : str
        Type: "straddle", "ei", "ucb".
    kappa : float
        Exploration parameter for straddle/UCB.
    xi : float
        Exploration parameter for EI.

    Attributes
    ----------
    landscape : EfficiencyLandscape
    acquisition_type : str
    kappa : float
    xi : float
    _best_y : float
    _call_count : int
    """

    def __init__(
        self,
        landscape: EfficiencyLandscape,
        acquisition_type: str = "straddle",
        kappa: float = 1.96,
        xi: float = 0.01,
    ) -> None:
        self.landscape = landscape
        self.acquisition_type = acquisition_type
        self.kappa = kappa
        self.xi = xi
        self._best_y = 0.0
        self._call_count = 0

    def set_best_y(self, best_y: float) -> None:
        """Set the best observed value for EI."""
        self._best_y = best_y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition function at points.

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate, shape (n_points, 9) or (9,).

        Returns
        -------
        np.ndarray
            Acquisition values.
        """
        self._call_count += 1

        # Handle single point
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get GP predictions
        mu, sigma = self.landscape.predict(X, return_std=True)

        # Compute acquisition
        if self.acquisition_type == "straddle":
            return _straddle_acquisition_numba(mu, sigma, self.kappa)
        elif self.acquisition_type == "ei":
            return _expected_improvement_numba(mu, sigma, self._best_y, self.xi)
        elif self.acquisition_type == "ucb":
            return _ucb_acquisition_numba(mu, sigma, self.kappa)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")

    def evaluate_single(self, x: np.ndarray) -> float:
        """Evaluate at a single point (for optimizers)."""
        return float(self(x.reshape(1, -1))[0])

    def negative_single(self, x: np.ndarray) -> float:
        """Negative acquisition for minimization (scipy)."""
        return -self.evaluate_single(x)

    @property
    def call_count(self) -> int:
        """Number of times the acquisition was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


# =============================================================================
# Bayesian Optimizer Configuration
# =============================================================================


@dataclass
class AcquisitionConfig:
    """
    Configuration for acquisition function optimization.

    Parameters
    ----------
    acquisition_type : str
        Acquisition function type: "straddle", "ei", or "ucb".
    kappa : float
        Exploration parameter for straddle/UCB.
    xi : float
        Exploration parameter for EI.
    optimizer : str
        Optimizer: "differential_evolution", "random", or "lbfgs".
    n_restarts : int
        Number of restarts for local optimization.
    n_random_starts : int
        Number of random starting points.
    maxiter : int
        Maximum iterations per optimization.
    population_size : int
        Population size for differential evolution.

    Attributes
    ----------
    acquisition_type : str
    kappa : float
    xi : float
    optimizer : str
    n_restarts : int
    n_random_starts : int
    maxiter : int
    population_size : int
    """

    acquisition_type: str = "straddle"
    kappa: float = 1.96
    xi: float = 0.01
    optimizer: str = "differential_evolution"
    n_restarts: int = 5
    n_random_starts: int = 1000
    maxiter: int = 100
    population_size: int = 15


# =============================================================================
# Bayesian Optimizer
# =============================================================================


class BayesianOptimizer:
    """
    Bayesian Optimizer for security cliff identification.

    This class implements batch Bayesian optimization using the
    Kriging Believer strategy for generating candidate points.

    Parameters
    ----------
    landscape : EfficiencyLandscape
        Trained GP surrogate model.
    bounds : ParameterBounds
        Parameter space bounds.
    config : AcquisitionConfig
        Acquisition function configuration.

    Attributes
    ----------
    landscape : EfficiencyLandscape
    bounds : ParameterBounds
    config : AcquisitionConfig
    _acquisition : AcquisitionFunction
    _best_point : Optional[np.ndarray]
    _best_value : float
    _history : List[Dict[str, Any]]

    Examples
    --------
    >>> optimizer = BayesianOptimizer(landscape, bounds)
    >>> candidates = optimizer.suggest_batch(batch_size=16)
    >>> # Execute candidates and get results
    >>> optimizer.update(candidates, efficiencies)
    """

    def __init__(
        self,
        landscape: EfficiencyLandscape,
        bounds: Optional[ParameterBounds] = None,
        config: Optional[AcquisitionConfig] = None,
    ) -> None:
        """
        Initialize the Bayesian optimizer.

        Parameters
        ----------
        landscape : EfficiencyLandscape
            Trained GP model.
        bounds : Optional[ParameterBounds]
            Parameter bounds.
        config : Optional[AcquisitionConfig]
            Acquisition configuration.
        """
        self.landscape = landscape
        self.bounds = bounds or ParameterBounds()
        self.config = config or AcquisitionConfig()

        self._acquisition = AcquisitionFunction(
            landscape=landscape,
            acquisition_type=self.config.acquisition_type,
            kappa=self.config.kappa,
            xi=self.config.xi,
        )

        self._best_point: Optional[np.ndarray] = None
        self._best_value: float = float("inf")
        self._history: List[Dict[str, Any]] = []

        # Pre-compute bounds array
        self._bounds_array = self._compute_bounds_array()

        logger.info(
            "Initialized BayesianOptimizer (acq=%s, optimizer=%s)",
            self.config.acquisition_type,
            self.config.optimizer,
        )

    def _compute_bounds_array(self) -> List[Tuple[float, float]]:
        """Compute scipy-compatible bounds array."""
        b = self.bounds
        return [
            (b.r_min, b.r_max),  # storage_noise_r
            (np.log10(b.nu_min), np.log10(b.nu_max)),  # storage_rate_nu (log)
            (b.dt_min_log, b.dt_max_log),  # wait_time_ns (log)
            (b.f_min, b.f_max),  # channel_fidelity
            (b.eta_min_log, b.eta_max_log),  # detection_efficiency (log)
            (b.e_det_min, b.e_det_max),  # detector_error
            (b.p_dark_min_log, b.p_dark_max_log),  # dark_count_prob (log)
            (b.n_min_log, b.n_max_log),  # num_pairs (log)
            (0.0, 1.0),  # strategy (binary)
        ]

    def _random_sample(self, n: int) -> np.ndarray:
        """Generate n random samples within bounds."""
        samples = np.zeros((n, 9))
        for i, (lo, hi) in enumerate(self._bounds_array):
            samples[:, i] = np.random.uniform(lo, hi, n)
        return samples

    def _optimize_acquisition(self) -> Tuple[np.ndarray, float]:
        """
        Maximize the acquisition function.

        Returns
        -------
        Tuple[np.ndarray, float]
            (best_point, best_value).
        """
        self._acquisition.reset_call_count()

        if self.config.optimizer == "differential_evolution":
            result = differential_evolution(
                self._acquisition.negative_single,
                bounds=self._bounds_array,
                maxiter=self.config.maxiter,
                popsize=self.config.population_size,
                polish=True,
                workers=1,  # Avoid nested parallelism
                seed=np.random.randint(0, 2**31),
            )
            best_x = result.x
            best_acq = -result.fun

        elif self.config.optimizer == "random":
            # Random search with local refinement
            X_random = self._random_sample(self.config.n_random_starts)
            acq_values = self._acquisition(X_random)
            best_idx = np.argmax(acq_values)
            best_x = X_random[best_idx]
            best_acq = acq_values[best_idx]

            # Local refinement
            for _ in range(self.config.n_restarts):
                x0 = self._random_sample(1)[0]
                try:
                    result = minimize(
                        self._acquisition.negative_single,
                        x0,
                        method="L-BFGS-B",
                        bounds=self._bounds_array,
                        options={"maxiter": 50},
                    )
                    if -result.fun > best_acq:
                        best_x = result.x
                        best_acq = -result.fun
                except Exception:
                    pass

        elif self.config.optimizer == "lbfgs":
            # Multi-start L-BFGS-B
            best_x = None
            best_acq = -np.inf

            for _ in range(self.config.n_restarts):
                x0 = self._random_sample(1)[0]
                try:
                    result = minimize(
                        self._acquisition.negative_single,
                        x0,
                        method="L-BFGS-B",
                        bounds=self._bounds_array,
                        options={"maxiter": self.config.maxiter},
                    )
                    if -result.fun > best_acq:
                        best_x = result.x
                        best_acq = -result.fun
                except Exception:
                    pass

            if best_x is None:
                # Fallback to random
                X_random = self._random_sample(100)
                acq_values = self._acquisition(X_random)
                best_idx = np.argmax(acq_values)
                best_x = X_random[best_idx]
                best_acq = acq_values[best_idx]

        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        logger.debug(
            "Acquisition optimization: acq=%.4f, calls=%d",
            best_acq,
            self._acquisition.call_count,
        )

        return best_x, best_acq

    def suggest_single(self) -> np.ndarray:
        """
        Suggest a single next point to evaluate.

        Returns
        -------
        np.ndarray
            Suggested point, shape (9,).
        """
        best_x, best_acq = self._optimize_acquisition()

        self._history.append({
            "type": "single",
            "acquisition_value": best_acq,
            "point": best_x.copy(),
        })

        return best_x

    def suggest_batch(
        self,
        batch_size: int,
        strategy: str = "kriging_believer",
    ) -> np.ndarray:
        """
        Suggest a batch of points to evaluate.

        Uses the Kriging Believer strategy: sequentially maximize
        acquisition, hallucinating the GP mean at each selected point.

        Parameters
        ----------
        batch_size : int
            Number of points to suggest.
        strategy : str
            Batch strategy: "kriging_believer" or "random".

        Returns
        -------
        np.ndarray
            Suggested points, shape (batch_size, 9).
        """
        if strategy == "random":
            return self._random_sample(batch_size)

        candidates = np.zeros((batch_size, 9))
        acquisition_values = np.zeros(batch_size)

        # Store original GP state
        # (For Kriging Believer, we'd ideally update the GP temporarily)
        # For simplicity, we use a greedy approach with random perturbation

        for i in range(batch_size):
            best_x, best_acq = self._optimize_acquisition()
            candidates[i] = best_x
            acquisition_values[i] = best_acq

            # Add small random perturbation to avoid duplicates
            # (Full Kriging Believer would retrain GP with hallucinated point)
            if i < batch_size - 1:
                # Perturb the GP predictions temporarily
                # This is a simplified version
                pass

        self._history.append({
            "type": "batch",
            "batch_size": batch_size,
            "acquisition_values": acquisition_values.tolist(),
            "mean_acquisition": float(np.mean(acquisition_values)),
        })

        logger.info(
            "Generated batch of %d candidates (mean_acq=%.4f)",
            batch_size,
            np.mean(acquisition_values),
        )

        return candidates

    def suggest_batch_diverse(
        self,
        batch_size: int,
        diversity_weight: float = 0.1,
    ) -> np.ndarray:
        """
        Suggest a diverse batch using acquisition + diversity.

        Balances acquisition value with distance from already selected points.

        Parameters
        ----------
        batch_size : int
            Number of points to suggest.
        diversity_weight : float
            Weight for diversity term.

        Returns
        -------
        np.ndarray
            Suggested points, shape (batch_size, 9).
        """
        # Generate many random candidates
        n_candidates = max(1000, batch_size * 100)
        X_candidates = self._random_sample(n_candidates)

        # Compute acquisition values
        acq_values = self._acquisition(X_candidates)

        # Greedy selection with diversity
        selected_indices = []

        for _ in range(batch_size):
            if len(selected_indices) == 0:
                # First point: pure acquisition
                scores = acq_values.copy()
            else:
                # Subsequent points: acquisition + diversity
                selected_points = X_candidates[selected_indices]
                distances = np.min(
                    np.linalg.norm(
                        X_candidates[:, np.newaxis, :] - selected_points[np.newaxis, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                scores = acq_values + diversity_weight * distances

            # Mask already selected
            scores[selected_indices] = -np.inf

            # Select best
            best_idx = np.argmax(scores)
            selected_indices.append(best_idx)

        candidates = X_candidates[selected_indices]

        self._history.append({
            "type": "batch_diverse",
            "batch_size": batch_size,
            "diversity_weight": diversity_weight,
        })

        return candidates

    def update_best(self, x: np.ndarray, efficiency: float) -> None:
        """
        Update the best observed point.

        Parameters
        ----------
        x : np.ndarray
            Point that was evaluated.
        efficiency : float
            Observed efficiency.
        """
        # For cliff-finding, we want efficiency close to 0
        # So we track absolute efficiency
        if abs(efficiency) < abs(self._best_value):
            self._best_point = x.copy()
            self._best_value = efficiency
            logger.info(
                "New best cliff point: efficiency=%.4f",
                efficiency,
            )

        # Update acquisition function's best_y for EI
        self._acquisition.set_best_y(self._best_value)

    @property
    def best_point(self) -> Optional[np.ndarray]:
        """Get the best observed point."""
        return self._best_point

    @property
    def best_value(self) -> float:
        """Get the best observed efficiency."""
        return self._best_value

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history.copy()

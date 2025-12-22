"""
Latin Hypercube Sampling for high-dimensional parameter exploration.

This module provides the LHSSampler class for generating space-filling
designs in the 9-dimensional QKD parameter space. LHS ensures better
coverage than random sampling while maintaining statistical properties.

Theory
------
Latin Hypercube Sampling divides each dimension into N equal strata and
places exactly one sample in each stratum. This guarantees full coverage
of the marginal distributions while providing better space-filling
properties than pure random sampling.

For log-uniform parameters, we sample uniformly in log-space then
transform back. For categorical parameters (strategy), we use stratified
assignment.

References
----------
- McKay et al. (1979): Original LHS formulation
- Iman & Conover (1982): LHS for computer experiments
- scipy.stats.qmc.LatinHypercube: Modern implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats.qmc import LatinHypercube

from caligo.exploration.types import ExplorationSample, ReconciliationStrategy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Parameter Bounds
# =============================================================================


@dataclass(frozen=True)
class ParameterBounds:
    """
    Bounds for the 9-dimensional parameter space.

    This dataclass defines the valid ranges for each parameter,
    matching Table 1 in parameter_explor.md.

    Parameters
    ----------
    r_min : float
        Minimum storage noise (default: 0.0).
    r_max : float
        Maximum storage noise (default: 1.0).
    nu_min : float
        Minimum storage rate in log10 scale (default: -3.0, i.e., 0.001).
    nu_max : float
        Maximum storage rate in log10 scale (default: 0.0, i.e., 1.0).
    dt_min_log : float
        Minimum wait time in log10(ns) (default: 5.0 = 10^5 ns).
    dt_max_log : float
        Maximum wait time in log10(ns) (default: 9.0 = 10^9 ns).
    f_min : float
        Minimum channel fidelity (default: 0.501).
    f_max : float
        Maximum channel fidelity (default: 1.0).
    eta_min_log : float
        Minimum detection efficiency in log10 (default: -3.0 = 0.001).
    eta_max_log : float
        Maximum detection efficiency in log10 (default: 0.0 = 1.0).
    e_det_min : float
        Minimum detector error (default: 0.0).
    e_det_max : float
        Maximum detector error (default: 0.1).
    p_dark_min_log : float
        Minimum dark count in log10 (default: -8.0 = 1e-8).
    p_dark_max_log : float
        Maximum dark count in log10 (default: -3.0 = 1e-3).
    n_min_log : float
        Minimum EPR pairs in log10 (default: 4.0 = 10^4).
    n_max_log : float
        Maximum EPR pairs in log10 (default: 6.0 = 10^6).

    Attributes
    ----------
    r_min, r_max : float
    nu_min, nu_max : float
    dt_min_log, dt_max_log : float
    f_min, f_max : float
    eta_min_log, eta_max_log : float
    e_det_min, e_det_max : float
    p_dark_min_log, p_dark_max_log : float
    n_min_log, n_max_log : float

    Examples
    --------
    >>> bounds = ParameterBounds()
    >>> bounds.f_max
    1.0

    >>> # Custom bounds for restricted exploration
    >>> bounds = ParameterBounds(f_min=0.8, f_max=0.99)
    """

    # Storage noise r ∈ [0, 1] (linear)
    r_min: float = 0.0
    r_max: float = 1.0

    # Storage rate ν ∈ [0.001, 1] (log-uniform)
    nu_min: float = 0.001
    nu_max: float = 1.0

    # Wait time Δt ∈ [10^5, 10^9] ns (log-uniform)
    dt_min_log: float = 5.0  # log10(1e5)
    dt_max_log: float = 9.0  # log10(1e9)

    # Channel fidelity F ∈ (0.5, 1] (beta distribution or linear)
    f_min: float = 0.501  # Just above 0.5 to ensure valid states
    f_max: float = 1.0

    # Detection efficiency η ∈ (0.001, 1] (log-uniform)
    eta_min_log: float = -3.0  # log10(0.001)
    eta_max_log: float = 0.0   # log10(1.0)

    # Detector error e_det ∈ [0, 0.1] (linear)
    e_det_min: float = 0.0
    e_det_max: float = 0.1

    # Dark count P_dark ∈ [10^-8, 10^-3] (log-uniform)
    p_dark_min_log: float = -8.0  # log10(1e-8)
    p_dark_max_log: float = -3.0  # log10(1e-3)

    # Number of pairs N ∈ [10^4, 10^6] (log-uniform)
    n_min_log: float = 4.0  # log10(1e4)
    n_max_log: float = 6.0  # log10(1e6)

    def to_bounds_array(self) -> np.ndarray:
        """
        Convert bounds to numpy array for optimization.

        Returns
        -------
        np.ndarray
            Shape (8, 2) array of [min, max] bounds for continuous parameters.
            Strategy (dim 9) is handled separately as categorical.
        """
        return np.array([
            [self.r_min, self.r_max],
            [np.log10(self.nu_min), np.log10(self.nu_max)],
            [self.dt_min_log, self.dt_max_log],
            [self.f_min, self.f_max],
            [self.eta_min_log, self.eta_max_log],
            [self.e_det_min, self.e_det_max],
            [self.p_dark_min_log, self.p_dark_max_log],
            [self.n_min_log, self.n_max_log],
        ], dtype=np.float64)


# =============================================================================
# LHS Sampler
# =============================================================================


class LHSSampler:
    """
    Latin Hypercube Sampler for the 9D parameter space.

    This class generates space-filling samples using scipy's LHS
    implementation, then transforms them to the parameter ranges
    defined by ParameterBounds.

    Parameters
    ----------
    bounds : ParameterBounds
        Parameter range bounds.
    seed : Optional[int]
        Random seed for reproducibility.
    fidelity_beta_params : Tuple[float, float]
        Beta distribution parameters (a, b) for channel fidelity.
        Default (2, 1) biases toward higher fidelity values.

    Attributes
    ----------
    bounds : ParameterBounds
        Parameter bounds.
    seed : Optional[int]
        Random seed.
    _rng : np.random.Generator
        Random number generator.
    _lhs : LatinHypercube
        Scipy LHS sampler.

    Examples
    --------
    >>> sampler = LHSSampler(seed=42)
    >>> samples = sampler.generate(n=1000)
    >>> len(samples)
    1000
    >>> samples[0].channel_fidelity
    0.9234...

    Notes
    -----
    The sampler uses different strategies for different parameter types:
    - Linear: storage_noise_r, detector_error
    - Log-uniform: storage_rate_nu, wait_time, detection_eff, dark_count, num_pairs
    - Beta-distributed: channel_fidelity (biased toward high values)
    - Stratified categorical: strategy (baseline/blind)
    """

    def __init__(
        self,
        bounds: Optional[ParameterBounds] = None,
        seed: Optional[int] = None,
        fidelity_beta_params: Tuple[float, float] = (2.0, 1.0),
    ) -> None:
        """
        Initialize the LHS sampler.

        Parameters
        ----------
        bounds : Optional[ParameterBounds]
            Parameter bounds. If None, uses defaults.
        seed : Optional[int]
            Random seed for reproducibility.
        fidelity_beta_params : Tuple[float, float]
            Beta(a, b) parameters for fidelity distribution.
        """
        self.bounds = bounds or ParameterBounds()
        self.seed = seed
        self.fidelity_beta_params = fidelity_beta_params
        self._rng = np.random.default_rng(seed)
        # 8 continuous dimensions (strategy handled separately)
        self._lhs = LatinHypercube(d=8, seed=seed)

        logger.debug(
            "Initialized LHSSampler with seed=%s, beta_params=%s",
            seed,
            fidelity_beta_params,
        )

    def generate(
        self,
        n: int,
        strategy_ratio: float = 0.5,
    ) -> List[ExplorationSample]:
        """
        Generate n LHS samples.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        strategy_ratio : float
            Fraction of samples using BLIND strategy. Default 0.5 = balanced.

        Returns
        -------
        List[ExplorationSample]
            List of n parameter samples.

        Raises
        ------
        ValueError
            If n <= 0 or strategy_ratio not in [0, 1].

        Examples
        --------
        >>> sampler = LHSSampler(seed=42)
        >>> samples = sampler.generate(100)
        >>> baseline_count = sum(1 for s in samples if s.strategy == ReconciliationStrategy.BASELINE)
        >>> blind_count = sum(1 for s in samples if s.strategy == ReconciliationStrategy.BLIND)
        >>> assert abs(baseline_count - blind_count) <= 2  # Roughly balanced
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if not 0.0 <= strategy_ratio <= 1.0:
            raise ValueError(f"strategy_ratio must be in [0, 1], got {strategy_ratio}")

        # Generate LHS samples in [0, 1]^8
        unit_samples = self._lhs.random(n)

        # Transform to parameter space
        samples = []
        bounds = self.bounds

        # Stratified strategy assignment
        n_blind = int(n * strategy_ratio)
        strategies = (
            [ReconciliationStrategy.BLIND] * n_blind +
            [ReconciliationStrategy.BASELINE] * (n - n_blind)
        )
        self._rng.shuffle(strategies)

        for i in range(n):
            u = unit_samples[i]

            # Dim 0: storage_noise_r (linear)
            r = bounds.r_min + u[0] * (bounds.r_max - bounds.r_min)

            # Dim 1: storage_rate_nu (log-uniform)
            log_nu = np.log10(bounds.nu_min) + u[1] * (np.log10(bounds.nu_max) - np.log10(bounds.nu_min))
            nu = 10 ** log_nu

            # Dim 2: wait_time_ns (log-uniform)
            dt_log = bounds.dt_min_log + u[2] * (bounds.dt_max_log - bounds.dt_min_log)
            dt = 10 ** dt_log

            # Dim 3: channel_fidelity (beta-transformed)
            # Use inverse CDF of Beta distribution
            a, b = self.fidelity_beta_params
            f_unit = beta_dist.ppf(u[3], a, b)
            f = bounds.f_min + f_unit * (bounds.f_max - bounds.f_min)
            f = np.clip(f, bounds.f_min, bounds.f_max)

            # Dim 4: detection_efficiency (log-uniform)
            eta_log = bounds.eta_min_log + u[4] * (bounds.eta_max_log - bounds.eta_min_log)
            eta = 10 ** eta_log

            # Dim 5: detector_error (linear)
            e_det = bounds.e_det_min + u[5] * (bounds.e_det_max - bounds.e_det_min)

            # Dim 6: dark_count_prob (log-uniform)
            p_dark_log = bounds.p_dark_min_log + u[6] * (bounds.p_dark_max_log - bounds.p_dark_min_log)
            p_dark = 10 ** p_dark_log

            # Dim 7: num_pairs (log-uniform, rounded to int)
            n_log = bounds.n_min_log + u[7] * (bounds.n_max_log - bounds.n_min_log)
            num_pairs = int(round(10 ** n_log))

            # Strategy from stratified assignment
            strategy = strategies[i]

            sample = ExplorationSample(
                storage_noise_r=float(r),
                storage_rate_nu=float(nu),
                wait_time_ns=float(dt),
                channel_fidelity=float(f),
                detection_efficiency=float(eta),
                detector_error=float(e_det),
                dark_count_prob=float(p_dark),
                num_pairs=num_pairs,
                strategy=strategy,
            )
            samples.append(sample)

        logger.info(
            "Generated %d LHS samples (BASELINE: %d, BLIND: %d)",
            n,
            n - n_blind,
            n_blind,
        )
        return samples

    def generate_array(
        self,
        n: int,
        strategy_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Generate n LHS samples as a numpy array.

        Parameters
        ----------
        n : int
            Number of samples.
        strategy_ratio : float
            Fraction using BLIND strategy.

        Returns
        -------
        np.ndarray
            Shape (n, 9) array of samples in transformed space.
        """
        samples = self.generate(n, strategy_ratio)
        return np.array([s.to_array() for s in samples], dtype=np.float64)

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        """
        Restore RNG state from a checkpoint.

        Parameters
        ----------
        state : Dict[str, Any]
            Serialized RNG state.
        """
        # For scipy's LatinHypercube, we need to recreate with the state
        # This is a limitation - we store the seed and sample index
        if "seed" in state:
            self._rng = np.random.default_rng(state["seed"])
            self._lhs = LatinHypercube(d=8, seed=state["seed"])
            logger.debug("Restored RNG state with seed=%s", state["seed"])

    def get_rng_state(self) -> Dict[str, Any]:
        """
        Get current RNG state for checkpointing.

        Returns
        -------
        Dict[str, Any]
            Serializable RNG state.
        """
        return {"seed": self.seed}


# =============================================================================
# Utility Functions
# =============================================================================


def samples_to_array(samples: List[ExplorationSample]) -> np.ndarray:
    """
    Convert a list of samples to a numpy array.

    Parameters
    ----------
    samples : List[ExplorationSample]
        List of parameter samples.

    Returns
    -------
    np.ndarray
        Shape (n, 9) array of transformed parameters.
    """
    return np.array([s.to_array() for s in samples], dtype=np.float64)


def array_to_samples(arr: np.ndarray) -> List[ExplorationSample]:
    """
    Convert a numpy array to a list of samples.

    Parameters
    ----------
    arr : np.ndarray
        Shape (n, 9) array of transformed parameters.

    Returns
    -------
    List[ExplorationSample]
        List of parameter samples.
    """
    return [ExplorationSample.from_array(arr[i]) for i in range(len(arr))]

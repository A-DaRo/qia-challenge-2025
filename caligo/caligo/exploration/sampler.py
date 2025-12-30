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

Performance Notes
-----------------
**Float32 Quantization:** All design matrices use `numpy.float32` for:
- 50% memory footprint reduction vs. float64
- 2x SIMD throughput (AVX processes more 32-bit values per cycle)
- GPU Tensor Core compatibility

**Numba Acceleration:** Feasibility checks use JIT-compiled kernels with
`fastmath=True` for vectorized SIMD processing.

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
from numba import jit, prange
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from scipy.stats.qmc import LatinHypercube

from caligo.exploration.types import (
    DTYPE_FLOAT,
    ExplorationSample,
    Float32Array,
    ReconciliationStrategy,
)
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

    def to_bounds_array(self) -> Float32Array:
        """
        Convert bounds to numpy array for optimization.

        Returns
        -------
        Float32Array
            Shape (8, 2) array of [min, max] bounds for continuous parameters.
            Strategy (dim 9) is handled separately as categorical.
            Uses Float32 for memory efficiency.
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
        ], dtype=DTYPE_FLOAT)


# =============================================================================
# Numba-Accelerated Feasibility Kernel
# =============================================================================


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def check_feasibility_batch(
    design_matrix: Float32Array,
    feasibility_out: NDArray[np.bool_],
    q_channel_out: Float32Array,
    q_storage_out: Float32Array,
    margin_out: Float32Array,
) -> int:
    """
    Vectorized feasibility check using SIMD-optimized Numba kernel.

    This kernel processes the entire design matrix in parallel, computing
    the theoretical QBER bounds for each sample and marking infeasible
    samples. Uses SIMD instructions (AVX/SSE) for maximum throughput.

    Parameters
    ----------
    design_matrix : Float32Array
        Shape (N, 9) array of samples in transformed space.
        Column order: [r, log_nu, log_dt, f, log_eta, e_det, log_p_dark, log_n, strategy]
    feasibility_out : NDArray[np.bool_]
        Output array for feasibility flags. Shape (N,).
    q_channel_out : Float32Array
        Output array for channel QBER values. Shape (N,).
    q_storage_out : Float32Array
        Output array for storage QBER bounds. Shape (N,).
    margin_out : Float32Array
        Output array for margin (q_storage - q_channel). Shape (N,).

    Returns
    -------
    int
        Number of feasible samples.

    Notes
    -----
    The feasibility condition requires:
    1. Q_channel < Q_storage (NSM constraint)
    2. Q_channel <= 0.22 (Lupo asymptotic bound)

    Q_channel is computed using the Erven formula:
        Q_channel = p_err / (2 * p_click)
    where:
        p_click = eta * (1 - p_dark) + 2 * p_dark * (1 - eta)
        p_err = e_det + (1 - e_det) * (1 - F) + p_dark * (1 - eta)

    Q_storage = (1 - r) / 2

    Performance
    -----------
    Processes ~1M samples in <10ms on modern CPUs with AVX-512.
    """
    n_samples = design_matrix.shape[0]
    n_feasible = 0

    for i in prange(n_samples):
        # Extract parameters (already in appropriate space)
        r = design_matrix[i, 0]  # storage_noise_r (linear)
        # log_nu = design_matrix[i, 1]  # Not used in feasibility
        # log_dt = design_matrix[i, 2]  # Not used in feasibility
        f = design_matrix[i, 3]  # channel_fidelity (linear)
        log_eta = design_matrix[i, 4]  # log10(detection_efficiency)
        e_det = design_matrix[i, 5]  # detector_error (linear)
        log_p_dark = design_matrix[i, 6]  # log10(dark_count_prob)
        # log_n = design_matrix[i, 7]  # Not used in feasibility

        # Convert from log space
        eta = 10.0 ** log_eta
        p_dark = 10.0 ** log_p_dark

        # Compute channel QBER using Erven formula
        # p_click = probability of detector click
        p_click = eta * (1.0 - p_dark) + 2.0 * p_dark * (1.0 - eta)

        # p_err = probability of error given click
        p_err = e_det + (1.0 - e_det) * (1.0 - f) + p_dark * (1.0 - eta)

        # Avoid division by zero
        if p_click > 1e-12:
            q_channel = p_err / (2.0 * p_click)
        else:
            q_channel = 0.5  # Maximum QBER if no clicks

        # Clamp to valid range
        q_channel = min(max(q_channel, 0.0), 0.5)

        # Compute storage QBER bound
        q_storage = (1.0 - r) / 2.0

        # Compute margin
        margin = q_storage - q_channel

        # Store outputs
        q_channel_out[i] = q_channel
        q_storage_out[i] = q_storage
        margin_out[i] = margin

        # Check feasibility (with epsilon for floating point safety)
        is_feasible = (margin > 1e-6) and (q_channel <= 0.22)
        feasibility_out[i] = is_feasible

        if is_feasible:
            n_feasible += 1

    return n_feasible


def compute_feasibility_vectorized(
    design_matrix: Float32Array,
) -> Tuple[NDArray[np.bool_], Float32Array, Float32Array, Float32Array, int]:
    """
    High-level wrapper for vectorized feasibility computation.

    Parameters
    ----------
    design_matrix : Float32Array
        Shape (N, 9) array of samples in transformed space.

    Returns
    -------
    Tuple[NDArray[np.bool_], Float32Array, Float32Array, Float32Array, int]
        (feasibility_mask, q_channel, q_storage, margin, n_feasible)
    """
    n_samples = design_matrix.shape[0]

    # Pre-allocate output arrays (Float32 for consistency)
    feasibility_out = np.empty(n_samples, dtype=np.bool_)
    q_channel_out = np.empty(n_samples, dtype=DTYPE_FLOAT)
    q_storage_out = np.empty(n_samples, dtype=DTYPE_FLOAT)
    margin_out = np.empty(n_samples, dtype=DTYPE_FLOAT)

    # Run the Numba kernel
    n_feasible = check_feasibility_batch(
        design_matrix.astype(DTYPE_FLOAT, copy=False),
        feasibility_out,
        q_channel_out,
        q_storage_out,
        margin_out,
    )

    return feasibility_out, q_channel_out, q_storage_out, margin_out, n_feasible


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

    def generate_design_matrix(
        self,
        n: int,
        strategy_ratio: float = 0.5,
    ) -> Tuple[Float32Array, NDArray[np.int8]]:
        """
        Generate n LHS samples directly as a contiguous Float32 design matrix.

        This is the preferred method for high-performance pipelines as it
        avoids object creation overhead. The design matrix stores parameters
        in transformed (log) space suitable for direct use in optimization
        and feasibility checking.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        strategy_ratio : float
            Fraction of samples using BLIND strategy. Default 0.5 = balanced.

        Returns
        -------
        Tuple[Float32Array, NDArray[np.int8]]
            (design_matrix, strategy_array) where:
            - design_matrix: Shape (n, 8) Float32 array of continuous parameters.
              Columns: [r, log_nu, log_dt, f, log_eta, e_det, log_p_dark, log_n]
            - strategy_array: Shape (n,) int8 array where 0=BASELINE, 1=BLIND.

        Notes
        -----
        The design matrix uses log-space for log-uniform parameters to
        maintain uniform density in transformed space. Use
        `inflate_to_samples()` for JIT object creation when needed.

        Performance
        -----------
        ~50% memory reduction vs. Float64 and enables SIMD vectorization.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if not 0.0 <= strategy_ratio <= 1.0:
            raise ValueError(f"strategy_ratio must be in [0, 1], got {strategy_ratio}")

        # Generate LHS samples in [0, 1]^8
        unit_samples = self._lhs.random(n).astype(DTYPE_FLOAT, copy=False)
        bounds = self.bounds

        # Pre-allocate output arrays
        design_matrix = np.empty((n, 8), dtype=DTYPE_FLOAT)

        # Stratified strategy assignment
        n_blind = int(n * strategy_ratio)
        strategy_array = np.zeros(n, dtype=np.int8)
        strategy_array[:n_blind] = 1
        self._rng.shuffle(strategy_array)

        # Vectorized transformation (all operations on Float32)
        # Dim 0: storage_noise_r (linear)
        design_matrix[:, 0] = (
            bounds.r_min + unit_samples[:, 0] * (bounds.r_max - bounds.r_min)
        ).astype(DTYPE_FLOAT)

        # Dim 1: storage_rate_nu (log-uniform) -> store in log space
        log_nu_min = np.float32(np.log10(bounds.nu_min))
        log_nu_max = np.float32(np.log10(bounds.nu_max))
        design_matrix[:, 1] = (
            log_nu_min + unit_samples[:, 1] * (log_nu_max - log_nu_min)
        ).astype(DTYPE_FLOAT)

        # Dim 2: wait_time_ns (log-uniform) -> store in log space
        design_matrix[:, 2] = (
            bounds.dt_min_log + unit_samples[:, 2] * (bounds.dt_max_log - bounds.dt_min_log)
        ).astype(DTYPE_FLOAT)

        # Dim 3: channel_fidelity (beta-transformed)
        a, b = self.fidelity_beta_params
        f_unit = beta_dist.ppf(unit_samples[:, 3], a, b).astype(DTYPE_FLOAT)
        design_matrix[:, 3] = np.clip(
            bounds.f_min + f_unit * (bounds.f_max - bounds.f_min),
            bounds.f_min,
            bounds.f_max,
        ).astype(DTYPE_FLOAT)

        # Dim 4: detection_efficiency (log-uniform) -> store in log space
        design_matrix[:, 4] = (
            bounds.eta_min_log + unit_samples[:, 4] * (bounds.eta_max_log - bounds.eta_min_log)
        ).astype(DTYPE_FLOAT)

        # Dim 5: detector_error (linear)
        design_matrix[:, 5] = (
            bounds.e_det_min + unit_samples[:, 5] * (bounds.e_det_max - bounds.e_det_min)
        ).astype(DTYPE_FLOAT)

        # Dim 6: dark_count_prob (log-uniform) -> store in log space
        design_matrix[:, 6] = (
            bounds.p_dark_min_log + unit_samples[:, 6] * (bounds.p_dark_max_log - bounds.p_dark_min_log)
        ).astype(DTYPE_FLOAT)

        # Dim 7: num_pairs (log-uniform) -> store in log space
        design_matrix[:, 7] = (
            bounds.n_min_log + unit_samples[:, 7] * (bounds.n_max_log - bounds.n_min_log)
        ).astype(DTYPE_FLOAT)

        logger.debug(
            "Generated %d-sample Float32 design matrix (BASELINE: %d, BLIND: %d)",
            n,
            n - n_blind,
            n_blind,
        )
        return design_matrix, strategy_array

    def inflate_to_samples(
        self,
        design_matrix: Float32Array,
        strategy_array: NDArray[np.int8],
        indices: Optional[NDArray[np.int64]] = None,
    ) -> List[ExplorationSample]:
        """
        JIT inflation: Convert design matrix rows to ExplorationSample objects.

        This method should only be called immediately before protocol execution,
        not during batch generation or filtering.

        Parameters
        ----------
        design_matrix : Float32Array
            Shape (N, 8) Float32 array from `generate_design_matrix()`.
        strategy_array : NDArray[np.int8]
            Shape (N,) int8 array of strategy assignments.
        indices : Optional[NDArray[np.int64]]
            Subset of row indices to inflate. If None, inflates all rows.

        Returns
        -------
        List[ExplorationSample]
            List of ExplorationSample objects for the selected indices.
        """
        if indices is None:
            indices = np.arange(design_matrix.shape[0])

        samples = []
        for i in indices:
            row = design_matrix[i]
            strategy = (
                ReconciliationStrategy.BLIND
                if strategy_array[i] == 1
                else ReconciliationStrategy.BASELINE
            )
            # Convert from log space where applicable
            sample = ExplorationSample.from_raw_params(
                r=float(row[0]),
                nu=float(10 ** row[1]),
                dt=float(10 ** row[2]),
                f=float(row[3]),
                eta=float(10 ** row[4]),
                e_det=float(row[5]),
                p_dark=float(10 ** row[6]),
                n=int(round(10 ** row[7])),
                strategy=strategy,
            )
            samples.append(sample)

        return samples

    def generate_array(
        self,
        n: int,
        strategy_ratio: float = 0.5,
    ) -> Float32Array:
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
        Float32Array
            Shape (n, 9) array of samples in transformed space.
            Uses Float32 for memory efficiency.

        Notes
        -----
        For high-performance pipelines, prefer `generate_design_matrix()`
        which avoids intermediate object creation.
        """
        samples = self.generate(n, strategy_ratio)
        return np.array([s.to_array() for s in samples], dtype=DTYPE_FLOAT)

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


def samples_to_array(samples: List[ExplorationSample]) -> Float32Array:
    """
    Convert a list of samples to a numpy array.

    Parameters
    ----------
    samples : List[ExplorationSample]
        List of parameter samples.

    Returns
    -------
    Float32Array
        Shape (n, 9) array of transformed parameters (Float32).
    """
    return np.array([s.to_array() for s in samples], dtype=DTYPE_FLOAT)


def array_to_samples(arr: NDArray) -> List[ExplorationSample]:
    """
    Convert a numpy array to a list of samples.

    Parameters
    ----------
    arr : NDArray
        Shape (n, 9) array of transformed parameters.

    Returns
    -------
    List[ExplorationSample]
        List of parameter samples.
    """
    return [ExplorationSample.from_array(arr[i]) for i in range(len(arr))]


def design_matrix_to_full(
    design_matrix: Float32Array,
    strategy_array: NDArray[np.int8],
) -> Float32Array:
    """
    Convert 8-column design matrix + strategy to 9-column full array.

    Parameters
    ----------
    design_matrix : Float32Array
        Shape (N, 8) design matrix from `generate_design_matrix()`.
    strategy_array : NDArray[np.int8]
        Shape (N,) strategy assignments (0=BASELINE, 1=BLIND).

    Returns
    -------
    Float32Array
        Shape (N, 9) full array with strategy as the 9th column.
    """
    n = design_matrix.shape[0]
    full_array = np.empty((n, 9), dtype=DTYPE_FLOAT)
    full_array[:, :8] = design_matrix
    full_array[:, 8] = strategy_array.astype(DTYPE_FLOAT)
    return full_array

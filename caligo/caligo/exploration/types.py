"""
Data structures for the Caligo Exploration Suite.

This module defines the canonical data types used throughout the exploration
pipeline, including parameter samples, protocol results, and phase states.

Type System
-----------
The exploration uses a strict type hierarchy:

- **ExplorationSample**: A single point in the 9D parameter space
- **ProtocolResult**: Outcome of running a protocol at a given sample
- **Phase*State**: Checkpoint state for fault-tolerant resumption

All types are frozen dataclasses to ensure immutability and hashability
where required for caching.

Performance Notes
-----------------
**Float32 Quantization:** All numerical arrays use `numpy.float32` for:
- 50% memory footprint reduction vs. float64
- 2x SIMD throughput (AVX processes more 32-bit values per cycle)
- GPU Tensor Core compatibility

**JIT-Friendly Design:** `ExplorationSample` is designed for just-in-time
inflation from raw Float32 arrays. Heavy object creation is deferred until
protocol execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Aliases (Float32-Int8 Enforcement)
# =============================================================================

# Canonical floating-point dtype for all numerical operations
DTYPE_FLOAT = np.float32

# Canonical integer dtype for all integer operations (e.g., LDPC syndromes, Quantum bits)
DTYPE_INT = np.int8

# Array type hints for Float32 arrays
Float32Array = NDArray[np.float32]

# Array type hints for Int8 arrays 
Int8Array = NDArray[np.int8]



# =============================================================================
# Enumerations
# =============================================================================


class ReconciliationStrategy(Enum):
    """
    Available reconciliation strategies for exploration.

    Attributes
    ----------
    BASELINE : str
        Standard rate-adaptive LDPC with QBER estimation.
    BLIND : str
        Blind reconciliation (Martinez-Mateo et al. 2012).
    """

    BASELINE = "baseline"
    BLIND = "blind"


class ExplorationPhase(Enum):
    """
    Current phase of the exploration campaign.

    Attributes
    ----------
    LHS : str
        Phase 1: Latin Hypercube Sampling warmup.
    SURROGATE : str
        Phase 2: Gaussian Process surrogate training.
    ACTIVE : str
        Phase 3: Bayesian active learning stress testing.
    """

    LHS = "lhs"
    SURROGATE = "surrogate"
    ACTIVE = "active"


class ProtocolOutcome(Enum):
    """
    Possible outcomes of a protocol execution.

    Attributes
    ----------
    SUCCESS : str
        Protocol completed successfully with positive key length.
    FAILURE_QBER : str
        Protocol aborted due to QBER threshold violation.
    FAILURE_RECONCILIATION : str
        Reconciliation failed to converge.
    FAILURE_SECURITY : str
        Security bounds violated (negative key rate).
    FAILURE_TIMEOUT : str
        Protocol execution timed out.
    FAILURE_ERROR : str
        Unexpected error during execution.
    SKIPPED_INFEASIBLE : str
        Sample breaches infeasibility constraints theoretical_qber >= (1-r)/2 and is skipped.
    """

    SUCCESS = "success"
    FAILURE_QBER = "failure_qber"
    FAILURE_RECONCILIATION = "failure_reconciliation"
    FAILURE_SECURITY = "failure_security"
    FAILURE_TIMEOUT = "failure_timeout"
    FAILURE_ERROR = "failure_error"
    SKIPPED_INFEASIBLE = "skipped_infeasible"
    SKIPPED_PREDICTED_FAILURE = "skipped_predicted_failure"


# =============================================================================
# Exploration Sample (9D Parameter Point)
# =============================================================================


@dataclass(frozen=True)
class ExplorationSample:
    """
    A single point in the 9-dimensional parameter space.

    This dataclass represents one configuration to be evaluated by the
    exploration framework. It maps directly to Table 1 in parameter_explor.md.

    Parameters
    ----------
    storage_noise_r : float
        Storage noise parameter r ∈ [0, 1]. Higher values indicate more noise.
    storage_rate_nu : float
        Storage rate ν ∈ [0, 1]. Memory decay rate parameter.
    wait_time_ns : float
        Wait time Δt in nanoseconds. Range: [1e5, 1e9] ns.
    channel_fidelity : float
        EPR pair fidelity F ∈ (0.5, 1]. Channel quality indicator.
    detection_efficiency : float
        Detection efficiency η ∈ (0, 1]. Probability of detector click.
    detector_error : float
        Intrinsic detector error e_det ∈ [0, 0.1].
    dark_count_prob : float
        Dark count probability P_dark ∈ [1e-8, 1e-3].
    num_pairs : int
        Number of input EPR pairs N ∈ [1e4, 1e6].
    strategy : ReconciliationStrategy
        Reconciliation strategy (BASELINE or BLIND).

    Attributes
    ----------
    storage_noise_r : float
    storage_rate_nu : float
    wait_time_ns : float
    channel_fidelity : float
    detection_efficiency : float
    detector_error : float
    dark_count_prob : float
    num_pairs : int
    strategy : ReconciliationStrategy

    Examples
    --------
    >>> sample = ExplorationSample(
    ...     storage_noise_r=0.1,
    ...     storage_rate_nu=0.5,
    ...     wait_time_ns=1e6,
    ...     channel_fidelity=0.95,
    ...     detection_efficiency=0.8,
    ...     detector_error=0.01,
    ...     dark_count_prob=1e-5,
    ...     num_pairs=100000,
    ...     strategy=ReconciliationStrategy.BASELINE,
    ... )

    Notes
    -----
    The sample is frozen (immutable) to allow use as dictionary keys
    and to ensure thread safety when shared across worker processes.
    """

    storage_noise_r: float
    storage_rate_nu: float
    wait_time_ns: float
    channel_fidelity: float
    detection_efficiency: float
    detector_error: float
    dark_count_prob: float
    num_pairs: int
    strategy: ReconciliationStrategy

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if not 0.0 <= self.storage_noise_r <= 1.0:
            raise ValueError(f"storage_noise_r must be in [0,1], got {self.storage_noise_r}")
        if not 0.0 <= self.storage_rate_nu <= 1.0:
            raise ValueError(f"storage_rate_nu must be in [0,1], got {self.storage_rate_nu}")
        if not 1e5 <= self.wait_time_ns <= 1e11:
            raise ValueError(f"wait_time_ns must be in [1e5,1e11], got {self.wait_time_ns}")
        if not 0.5 < self.channel_fidelity <= 1.0:
            raise ValueError(f"channel_fidelity must be in (0.5,1], got {self.channel_fidelity}")
        if not 0.0 < self.detection_efficiency <= 1.0:
            raise ValueError(f"detection_efficiency must be in (0,1], got {self.detection_efficiency}")
        if not 0.0 <= self.detector_error <= 0.1:
            raise ValueError(f"detector_error must be in [0,0.1], got {self.detector_error}")
        if not 1e-9 <= self.dark_count_prob <= 1e-3:
            raise ValueError(f"dark_count_prob must be in [1e-9,1e-3], got {self.dark_count_prob}")
        if not 1e4 <= self.num_pairs <= 1e9:
            raise ValueError(f"num_pairs must be in [1e4,1e9], got {self.num_pairs}")

    def to_array(self) -> Float32Array:
        """
        Convert sample to numpy array for GP training.

        Returns
        -------
        Float32Array
            Shape (9,) array with normalized continuous parameters and
            strategy encoded as 0 (BASELINE) or 1 (BLIND).

        Notes
        -----
        Parameter order matches Table 1 in the design document.
        Uses Float32 for memory efficiency and SIMD optimization.
        """
        strategy_encoded = 0.0 if self.strategy == ReconciliationStrategy.BASELINE else 1.0
        return np.array([
            self.storage_noise_r,
            np.log10(self.storage_rate_nu),  # Log-transform for uniform sampling
            np.log10(self.wait_time_ns),  # Log-transform for uniform sampling
            self.channel_fidelity,
            np.log10(self.detection_efficiency),
            self.detector_error,
            np.log10(self.dark_count_prob),
            np.log10(self.num_pairs),
            strategy_encoded,
        ], dtype=DTYPE_FLOAT)

    @classmethod
    def from_array(cls, arr: Union[Float32Array, NDArray]) -> "ExplorationSample":
        """
        Reconstruct sample from numpy array.

        Parameters
        ----------
        arr : Union[Float32Array, NDArray]
            Shape (9,) array as produced by `to_array()`.

        Returns
        -------
        ExplorationSample
            Reconstructed sample.

        Notes
        -----
        This is a JIT-friendly inflation point - only call when the full
        object is needed for protocol execution.
        """
        strategy = ReconciliationStrategy.BASELINE if arr[8] < 0.5 else ReconciliationStrategy.BLIND
        return cls(
            storage_noise_r=float(arr[0]),
            storage_rate_nu=float(10 ** arr[1]),  # Convert from log10 back to linear
            wait_time_ns=float(10 ** arr[2]),
            channel_fidelity=float(arr[3]),
            detection_efficiency=float(10 ** arr[4]),
            detector_error=float(arr[5]),
            dark_count_prob=float(10 ** arr[6]),
            num_pairs=int(round(10 ** arr[7])),
            strategy=strategy,
        )

    @staticmethod
    def from_raw_params(
        r: float,
        nu: float,
        dt: float,
        f: float,
        eta: float,
        e_det: float,
        p_dark: float,
        n: int,
        strategy: ReconciliationStrategy,
    ) -> "ExplorationSample":
        """
        Create sample from raw parameter values (non-log space).

        Parameters
        ----------
        r : float
            Storage noise parameter r.
        nu : float
            Storage rate ν (linear, not log).
        dt : float
            Wait time Δt in nanoseconds.
        f : float
            Channel fidelity F.
        eta : float
            Detection efficiency η.
        e_det : float
            Detector error.
        p_dark : float
            Dark count probability.
        n : int
            Number of EPR pairs.
        strategy : ReconciliationStrategy
            Reconciliation strategy.

        Returns
        -------
        ExplorationSample
            Validated exploration sample.

        Notes
        -----
        This factory method provides a fast path for sample creation
        from raw parameters without array intermediary.
        """
        return ExplorationSample(
            storage_noise_r=r,
            storage_rate_nu=nu,
            wait_time_ns=dt,
            channel_fidelity=f,
            detection_efficiency=eta,
            detector_error=e_det,
            dark_count_prob=p_dark,
            num_pairs=n,
            strategy=strategy,
        )


# =============================================================================
# Protocol Result
# =============================================================================


@dataclass(frozen=True)
class ProtocolResult:
    """
    Result of executing the Caligo protocol for one sample.

    This dataclass captures all metrics needed for surrogate training
    and security cliff identification.

    Parameters
    ----------
    sample : ExplorationSample
        The input parameter configuration.
    outcome : ProtocolOutcome
        Classification of the execution result.
    net_efficiency : float
        Net key efficiency: final_key_length / num_pairs.
        Range: [0, 1] for success, 0 for failures.
    raw_key_length : int
        Number of sifted bits before reconciliation.
    final_key_length : int
        Number of secure key bits after privacy amplification.
    qber_measured : float
        Measured QBER from test bits. NaN if not estimated.
    reconciliation_efficiency : float
        Reconciliation efficiency η ∈ [0, 1]. 1.0 = Shannon limit.
    leakage_bits : int
        Information leaked during reconciliation.
    execution_time_seconds : float
        Wall-clock time for protocol execution.
    error_message : Optional[str]
        Error details if outcome indicates failure.
    metadata : Dict[str, Any]
        Additional metrics (e.g., iteration counts, detector stats).

    Attributes
    ----------
    sample : ExplorationSample
    outcome : ProtocolOutcome
    net_efficiency : float
    raw_key_length : int
    final_key_length : int
    qber_measured : float
    reconciliation_efficiency : float
    leakage_bits : int
    execution_time_seconds : float
    error_message : Optional[str]
    metadata : Dict[str, Any]

    Examples
    --------
    >>> result = ProtocolResult(
    ...     sample=sample,
    ...     outcome=ProtocolOutcome.SUCCESS,
    ...     net_efficiency=0.85,
    ...     raw_key_length=50000,
    ...     final_key_length=42500,
    ...     qber_measured=0.03,
    ...     reconciliation_efficiency=0.95,
    ...     leakage_bits=10000,
    ...     execution_time_seconds=2.5,
    ...     error_message=None,
    ...     metadata={"ldpc_iterations": 15},
    ... )
    """

    sample: ExplorationSample
    outcome: ProtocolOutcome
    net_efficiency: float
    raw_key_length: int
    final_key_length: int
    qber_measured: float
    reconciliation_efficiency: float
    leakage_bits: int
    execution_time_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if the protocol execution was successful."""
        return self.outcome == ProtocolOutcome.SUCCESS

    def to_training_pair(self) -> Tuple[np.ndarray, float]:
        """
        Convert to (X, y) pair for GP training.

        Returns
        -------
        Tuple[np.ndarray, float]
            Input array (9D) and target (net_efficiency).
        """
        return self.sample.to_array(), self.net_efficiency


# =============================================================================
# Phase States (Checkpointing)
# =============================================================================


@dataclass
class Phase1State:
    """
    Persistent state for Phase 1 LHS sampling.

    This state enables fault-tolerant resumption of long-running
    LHS campaigns by tracking progress and RNG state.

    Parameters
    ----------
    target_feasible_samples : int
        Goal number of feasible samples to collect.
    feasible_samples_collected : int
        Number of theoretically feasible samples collected so far.
    total_samples_processed : int
        Total number of samples processed (including skipped infeasible ones).
    current_batch_start : int
        Index of the first sample in the current batch.
    rng_state : Dict[str, Any]
        Serialized numpy RNG state for reproducibility.
    current_phase : str
        Phase identifier (always "LHS" for Phase1State).
    hdf5_path : Path
        Path to the HDF5 data file.

    Attributes
    ----------
    target_feasible_samples : int
    feasible_samples_collected : int
    total_samples_processed : int
    current_batch_start : int
    rng_state : Dict[str, Any]
    current_phase : str
    hdf5_path : Path
    """

    target_feasible_samples: int
    feasible_samples_collected: int
    total_samples_processed: int
    current_batch_start: int
    rng_state: Dict[str, Any]
    current_phase: str = "LHS"
    hdf5_path: Path = field(default_factory=lambda: Path("exploration_data.h5"))

    def progress_fraction(self) -> float:
        """Return completion progress as a fraction in [0, 1]."""
        if self.target_feasible_samples == 0:
            return 0.0
        return self.feasible_samples_collected / self.target_feasible_samples

    def is_complete(self) -> bool:
        """Check if all samples have been processed."""
        return self.feasible_samples_collected >= self.target_feasible_samples


@dataclass
class Phase2State:
    """
    Persistent state for Phase 2 surrogate training.

    Parameters
    ----------
    training_samples_used : int
        Number of samples used for training.
    last_training_mse : float
        Mean squared error from last training epoch.
    model_version : int
        Monotonically increasing model version counter.
    divergence_detected : bool
        True if model divergence was detected.
    current_phase : str
        Phase identifier (always "SURROGATE").

    Attributes
    ----------
    training_samples_used : int
    last_training_mse : float
    model_version : int
    divergence_detected : bool
    current_phase : str
    """

    training_samples_used: int = 0
    last_training_mse: float = float("inf")
    model_version: int = 0
    divergence_detected: bool = False
    current_phase: str = "SURROGATE"


@dataclass
class Phase3State:
    """
    Persistent state for Phase 3 active learning.

    Parameters
    ----------
    iteration : int
        Current Bayesian optimization iteration.
    total_active_samples : int
        Total samples acquired during active learning.
    best_cliff_point : Optional[np.ndarray]
        Best security cliff point found so far.
    best_cliff_efficiency : float
        Efficiency at the best cliff point.
    acquisition_history : List[float]
        History of acquisition function values.
    current_phase : str
        Phase identifier (always "ACTIVE").

    Attributes
    ----------
    iteration : int
    total_active_samples : int
    best_cliff_point : Optional[np.ndarray]
    best_cliff_efficiency : float
    acquisition_history : List[float]
    current_phase : str
    """

    iteration: int = 0
    total_active_samples: int = 0
    best_cliff_point: Optional[np.ndarray] = None
    best_cliff_efficiency: float = 1.0
    acquisition_history: List[float] = field(default_factory=list)
    current_phase: str = "ACTIVE"

    def cliff_found(self) -> bool:
        """Check if a valid security cliff point has been identified."""
        return self.best_cliff_point is not None and self.best_cliff_efficiency < 0.1


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExplorationConfig:
    """
    Master configuration for the exploration suite.

    Parameters
    ----------
    output_dir : Path
        Directory for all output files (HDF5, checkpoints, models).
    phase1_samples : int
        Number of LHS warmup samples.
    phase1_batch_size : int
        Batch size for Phase 1 execution.
    phase3_iterations : int
        Number of Bayesian optimization iterations.
    phase3_batch_size : int
        Batch size for Phase 3 acquisition.
    num_workers : int
        Number of parallel workers for EPR generation.
    timeout_seconds : float
        Timeout per protocol execution.
    checkpoint_interval : int
        Save checkpoint every N batches.
    random_seed : Optional[int]
        Random seed for reproducibility. None for random initialization.

    Attributes
    ----------
    output_dir : Path
    phase1_samples : int
    phase1_batch_size : int
    phase3_iterations : int
    phase3_batch_size : int
    num_workers : int
    timeout_seconds : float
    checkpoint_interval : int
    random_seed : Optional[int]
    """

    output_dir: Path = field(default_factory=lambda: Path("./exploration_results"))
    phase1_samples: int = 2000
    phase1_batch_size: int = 50
    phase3_iterations: int = 100
    phase3_batch_size: int = 16
    num_workers: int = field(default_factory=lambda: max(1, __import__("os").cpu_count() - 1))
    timeout_seconds: float = 300.0
    checkpoint_interval: int = 5
    random_seed: Optional[int] = 42

    def __post_init__(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def hdf5_path(self) -> Path:
        """Path to the main HDF5 data file."""
        return self.output_dir / "exploration_data.h5"

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.output_dir / "checkpoint.pkl"

    @property
    def surrogate_path(self) -> Path:
        """Path to the serialized surrogate model."""
        return self.output_dir / "surrogate.pkl"

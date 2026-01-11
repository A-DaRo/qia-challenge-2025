"""
Parallel EPR generation orchestrator for simulation acceleration.

This module provides the infrastructure for generating EPR pairs across
multiple parallel worker processes, dramatically reducing wall-clock time
for large-scale QKD simulations while preserving statistical correctness.

The parallelization is physically valid because EPR pairs are independent
and identically distributed (i.i.d.) - the core assumption underlying all
QKD security proofs. Generation order has no physical significance.

Architecture
------------
```
    Main Process                    Worker Processes
    ─────────────                   ─────────────────
    ParallelEPROrchestrator
            │
            ├──► ProcessPoolExecutor
            │           │
            │           ├──► Worker 1: _worker_generate_epr()
            │           │         └──► Isolated NetSquid instance
            │           │
            │           ├──► Worker 2: _worker_generate_epr()
            │           │         └──► Isolated NetSquid instance
            │           │
            │           └──► Worker N: ...
            │
            └──► Aggregate & Shuffle Results
```

References
----------
- Tomamichel et al. (2012): i.i.d. assumption in finite-key analysis
- König et al. (2012): NSM security relies on pair independence
- NetSquid docs: Process isolation for discrete-event simulation
"""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type

from caligo.types.exceptions import SimulationError
from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from caligo.simulation.network_builder import NetworkConfig

logger = get_logger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def _get_physical_cpu_count() -> int:
    """
    Get the number of physical CPU cores (not hyperthreads).

    Returns
    -------
    int
        Number of physical CPU cores, minimum 1.

    Notes
    -----
    Uses os.sched_getaffinity() on Linux to respect CPU affinity masks.
    Falls back to os.cpu_count() // 2 as heuristic for hyperthreaded systems.
    """
    try:
        # On Linux, respect CPU affinity mask
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        # sched_getaffinity not available (Windows, macOS)
        available_cpus = os.cpu_count() or 2

    # Heuristic: assume hyperthreading (2 logical per physical)
    # This is conservative - better to underestimate than overload
    physical_cores = max(1, available_cpus // 2)

    logger.debug(
        f"CPU detection: available={available_cpus}, "
        f"estimated_physical={physical_cores}"
    )

    return physical_cores


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ParallelEPRConfig:
    """
    Configuration for parallel EPR generation.

    This dataclass encapsulates all settings for parallel generation,
    providing sensible defaults while allowing fine-grained control.

    Parameters
    ----------
    enabled : bool
        Whether parallel generation is enabled. If False, sequential
        generation is used (original behavior).
    num_workers : int
        Number of parallel worker processes. Defaults to CPU count - 1
        to leave one core for the main process.
    pairs_per_batch : int
        Number of EPR pairs generated per worker batch. Larger batches
        reduce overhead but increase memory usage.
    isolation_level : Literal["process", "thread"]
        Concurrency model. "process" is recommended for NetSquid due to
        global state issues. "thread" is experimental.
    prefetch_batches : int
        Number of batches to prefetch in the result queue.
    timeout_seconds : float
        Timeout for each worker batch in seconds.
    shuffle_results : bool
        Whether to shuffle aggregated results. Recommended for preserving
        i.i.d. assumption when batch boundaries might introduce artifacts.

    Attributes
    ----------
    enabled : bool
    num_workers : int
    pairs_per_batch : int
    isolation_level : Literal["process", "thread"]
    prefetch_batches : int
    timeout_seconds : float
    shuffle_results : bool

    Examples
    --------
    >>> # Default configuration
    >>> config = ParallelEPRConfig()
    >>> config.num_workers
    3  # On 8-core hyperthreaded machine (4 physical - 1)

    >>> # Custom configuration for large-scale simulation
    >>> config = ParallelEPRConfig(
    ...     enabled=True,
    ...     num_workers=16,
    ...     pairs_per_batch=5000,
    ... )

    Notes
    -----
    When `enabled=False`, this config is ignored and the factory will
    create a `SequentialEPRStrategy` instead.

    The default num_workers uses physical CPU cores (not hyperthreads)
    minus one, to leave headroom for the main process. This prevents
    oversubscription which can severely degrade NetSquid performance.
    """

    enabled: bool = False
    num_workers: int = field(
        default_factory=lambda: max(1, _get_physical_cpu_count() - 1)
    )
    pairs_per_batch: int = 1000
    isolation_level: Literal["process", "thread"] = "process"
    prefetch_batches: int = 2
    timeout_seconds: float = 300.0  # 5 minutes
    shuffle_results: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.pairs_per_batch < 1:
            raise ValueError(
                f"pairs_per_batch must be >= 1, got {self.pairs_per_batch}"
            )
        if self.isolation_level not in ("process", "thread"):
            raise ValueError(
                f"isolation_level must be 'process' or 'thread', "
                f"got '{self.isolation_level}'"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}"
            )


# =============================================================================
# Worker Result Data Structure
# =============================================================================


@dataclass
class EPRWorkerResult:
    """
    Result from a single parallel worker batch.

    Parameters
    ----------
    alice_outcomes : List[int]
        Alice's measurement outcomes (0 or 1) for this batch.
    alice_bases : List[int]
        Alice's measurement bases (0=Z, 1=X) for this batch.
    bob_outcomes : List[int]
        Bob's measurement outcomes (0 or 1) for this batch.
    bob_bases : List[int]
        Bob's measurement bases (0=Z, 1=X) for this batch.
    batch_id : int
        Unique identifier for this batch.
    num_pairs : int
        Number of EPR pairs in this batch.
    generation_time_ns : float
        Simulated time for generation (NetSquid time units).

    Attributes
    ----------
    alice_outcomes : List[int]
    alice_bases : List[int]
    bob_outcomes : List[int]
    bob_bases : List[int]
    batch_id : int
    num_pairs : int
    generation_time_ns : float
    """

    alice_outcomes: List[int]
    alice_bases: List[int]
    bob_outcomes: List[int]
    bob_bases: List[int]
    batch_id: int
    num_pairs: int
    generation_time_ns: float = 0.0


# =============================================================================
# Parallel EPR Orchestrator
# =============================================================================


class ParallelEPROrchestrator:
    """
    Manages parallel EPR pair generation across worker processes.

    This class coordinates the lifecycle of worker processes, distributes
    EPR generation tasks, and aggregates results while maintaining
    statistical equivalence to sequential generation.

    The orchestrator uses Python's `ProcessPoolExecutor` to spawn isolated
    worker processes, each running an independent NetSquid simulator.
    Results are aggregated and optionally shuffled to eliminate any
    ordering artifacts from batching.

    Parameters
    ----------
    config : ParallelEPRConfig
        Parallel execution configuration.
    network_config : Dict[str, Any]
        Quantum network topology and noise parameters. Must be serializable
        (no NetSquid objects) for multiprocessing.

    Attributes
    ----------
    _config : ParallelEPRConfig
        Stored configuration.
    _network_config : Dict[str, Any]
        Serialized network configuration.
    _executor : Optional[ProcessPoolExecutor]
        Worker pool (created lazily).
    _result_queue : Queue
        Thread-safe queue for collecting worker outputs.

    Examples
    --------
    >>> config = ParallelEPRConfig(enabled=True, num_workers=4)
    >>> network_config = {"distance_km": 10, "noise": 0.05}
    >>> orchestrator = ParallelEPROrchestrator(config, network_config)
    >>> results = orchestrator.generate_parallel(total_pairs=10000)
    >>> len(results[0])  # alice_outcomes
    10000
    >>> orchestrator.shutdown()

    Notes
    -----
    The orchestrator should be explicitly shut down via `shutdown()` to
    release worker resources. Alternatively, use as a context manager.

    **Thread Safety**: The orchestrator itself is not thread-safe. Use
    separate instances for concurrent generation from multiple threads.

    **Memory Considerations**: Each worker maintains its own NetSquid
    state. For very large simulations, consider reducing `num_workers`
    or `pairs_per_batch` to manage memory usage.

    References
    ----------
    - Python ProcessPoolExecutor documentation
    - NetSquid ns.sim_reset() for state isolation
    """

    def __init__(
        self,
        config: ParallelEPRConfig,
        network_config: Dict[str, Any],
    ) -> None:
        """
        Initialize the parallel EPR orchestrator.

        Parameters
        ----------
        config : ParallelEPRConfig
            Parallel execution configuration.
        network_config : Dict[str, Any]
            Serializable network configuration dictionary.
        """
        self._config = config
        self._network_config = network_config
        self._executor: Optional[ProcessPoolExecutor] = None
        self._result_queue: Queue = Queue()
        self._logger = get_logger(__name__)

    def _get_executor(self) -> ProcessPoolExecutor:
        """
        Get or create the process pool executor.

        Returns
        -------
        ProcessPoolExecutor
            The worker pool executor.

        Notes
        -----
        Executor is created lazily on first use.
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._config.num_workers
            )
        return self._executor

    def generate_parallel(
        self, total_pairs: int
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Generate EPR pairs using parallel workers.

        Distributes generation load across workers, aggregates results,
        and ensures random shuffling to preserve i.i.d. assumption.

        Parameters
        ----------
        total_pairs : int
            Total number of EPR pairs to generate.

        Returns
        -------
        alice_outcomes : List[int]
            Alice's measurement results (0 or 1).
        alice_bases : List[int]
            Alice's measurement bases (0=Z, 1=X).
        bob_outcomes : List[int]
            Bob's measurement results (0 or 1).
        bob_bases : List[int]
            Bob's measurement bases (0=Z, 1=X).

        Raises
        ------
        SimulationError
            If worker processes fail or timeout.
        ValueError
            If total_pairs <= 0.

        Notes
        -----
        Results are shuffled (if `config.shuffle_results=True`) to
        eliminate ordering artifacts from batching. The shuffling uses
        a shared index permutation to maintain alignment between all
        four output arrays.

        Examples
        --------
        >>> orchestrator = ParallelEPROrchestrator(config, network_config)
        >>> alice_out, alice_bases, bob_out, bob_bases = \\
        ...     orchestrator.generate_parallel(10000)
        >>> assert len(alice_out) == 10000
        """
        if total_pairs <= 0:
            raise ValueError(f"total_pairs must be > 0, got {total_pairs}")

        num_batches = math.ceil(total_pairs / self._config.pairs_per_batch)
        executor = self._get_executor()
        futures = []

        self._logger.debug(
            f"Launching {num_batches} batches across "
            f"{self._config.num_workers} workers for {total_pairs} pairs"
        )

        # Submit all batches to worker pool
        for batch_idx in range(num_batches):
            pairs_in_batch = min(
                self._config.pairs_per_batch,
                total_pairs - batch_idx * self._config.pairs_per_batch,
            )
            future = executor.submit(
                _worker_generate_epr,
                network_config=self._network_config,
                num_pairs=pairs_in_batch,
                batch_id=batch_idx,
            )
            futures.append(future)

        # Collect results from completed workers
        all_results: List[EPRWorkerResult] = []
        failed_batches: List[int] = []

        for future in as_completed(futures):
            try:
                result = future.result(timeout=self._config.timeout_seconds)
                all_results.append(result)
                self._logger.debug(
                    f"Batch {result.batch_id} completed: {result.num_pairs} pairs"
                )
            except Exception as e:
                # Track failed batch for error reporting
                batch_id = futures.index(future)
                failed_batches.append(batch_id)
                self._logger.error(f"Worker batch {batch_id} failed: {e}")

        if failed_batches:
            self.shutdown()
            raise SimulationError(
                f"Parallel generation failed for batches {failed_batches}. "
                f"Successfully completed {len(all_results)}/{num_batches} batches."
            )

        # Sort by batch_id to ensure deterministic ordering before shuffle
        all_results.sort(key=lambda r: r.batch_id)

        # Concatenate results
        alice_outcomes: List[int] = []
        alice_bases: List[int] = []
        bob_outcomes: List[int] = []
        bob_bases: List[int] = []

        for result in all_results:
            alice_outcomes.extend(result.alice_outcomes)
            alice_bases.extend(result.alice_bases)
            bob_outcomes.extend(result.bob_outcomes)
            bob_bases.extend(result.bob_bases)

        # Shuffle all lists with same permutation to preserve i.i.d.
        if self._config.shuffle_results and len(alice_outcomes) > 0:
            indices = list(range(len(alice_outcomes)))
            random.shuffle(indices)
            alice_outcomes = [alice_outcomes[i] for i in indices]
            alice_bases = [alice_bases[i] for i in indices]
            bob_outcomes = [bob_outcomes[i] for i in indices]
            bob_bases = [bob_bases[i] for i in indices]

        self._logger.debug(
            f"Generated {len(alice_outcomes)} pairs across {num_batches} batches"
        )

        return alice_outcomes, alice_bases, bob_outcomes, bob_bases

    def shutdown(self) -> None:
        """
        Gracefully shutdown worker pool.

        Waits for all pending tasks to complete before releasing resources.
        Safe to call multiple times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._logger.debug("Worker pool shutdown complete")

    def __enter__(self) -> "ParallelEPROrchestrator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()


# =============================================================================
# Network Configuration Helper
# =============================================================================


def _build_network_config_from_dict(
    config_dict: Dict[str, Any],
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 100,
) -> Any:
    """
    Build a StackNetworkConfig from a serialized configuration dictionary.

    This function reconstructs NSMParameters from the dictionary and uses
    CaligoNetworkBuilder to create a proper SquidASM network configuration.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary containing NSM parameters:
        - channel_fidelity: float (required, or 'fidelity' as fallback)
        - detection_efficiency: float (optional, default 1.0)
        - detector_error: float (optional, default 0.0)
        - dark_count_prob: float (optional, default 0.0)
        - storage_noise_r: float (optional, default 0.75)
        - storage_rate_nu: float (optional, default 0.01)
        - delta_t_ns: float (optional, default 1_000_000)
    alice_name : str
        Name for Alice's node. Default: "Alice".
    bob_name : str
        Name for Bob's node. Default: "Bob".
    num_qubits : int
        Number of qubit positions per node. Default: 100.

    Returns
    -------
    StackNetworkConfig
        Complete network configuration ready for simulation.

    Raises
    ------
    ImportError
        If SquidASM or caligo simulation modules are not available.

    Notes
    -----
    Link model selection follows the user's constraints:
    - Depolarise: Default model when detection_efficiency == 1.0 and
      dark_count_prob == 0.0
    - Heralded-double-click: When detection_efficiency < 1.0 or
      dark_count_prob > 0.0

    The function never uses "perfect" or simplified probabilistic models
    to ensure full NetSquid simulation fidelity.
    """
    from caligo.simulation.network_builder import (
        CaligoNetworkBuilder,
        ChannelModelSelection,
        ChannelParameters,
    )
    from caligo.simulation.physical_model import NSMParameters

    # Extract channel fidelity - support multiple naming conventions
    channel_fidelity = config_dict.get(
        "channel_fidelity",
        config_dict.get("fidelity", 1.0 - config_dict.get("noise", 0.0))
    )
    # Clamp to valid range (must be > 0.5 for NSMParameters)
    channel_fidelity = max(0.501, min(1.0, channel_fidelity))

    # Extract detector parameters
    detection_efficiency = config_dict.get("detection_efficiency", 1.0)
    detector_error = config_dict.get("detector_error", 0.0)
    dark_count_prob = config_dict.get("dark_count_prob", 0.0)

    # Extract storage parameters (with sensible defaults)
    storage_noise_r = config_dict.get("storage_noise_r", 0.75)
    storage_rate_nu = config_dict.get("storage_rate_nu", 0.01)
    delta_t_ns = config_dict.get("delta_t_ns", 1_000_000.0)

    # Construct NSMParameters
    nsm_params = NSMParameters(
        storage_noise_r=storage_noise_r,
        storage_rate_nu=storage_rate_nu,
        delta_t_ns=delta_t_ns,
        channel_fidelity=channel_fidelity,
        detection_eff_eta=detection_efficiency,
        detector_error=detector_error,
        dark_count_prob=dark_count_prob,
    )

    # Determine link model based on user constraints:
    # - Depolarise default
    # - Heralded when detection_efficiency < 1.0 or dark_count_prob > 0
    if detection_efficiency < 1.0 or dark_count_prob > 0.0:
        link_model = "heralded-double-click"
    else:
        link_model = "depolarise"

    model_selection = ChannelModelSelection(
        link_model=link_model,
        eta_semantics="detector_only",
    )

    # Build network configuration
    builder = CaligoNetworkBuilder(
        nsm_params=nsm_params,
        channel_params=ChannelParameters.for_testing(),
        model_selection=model_selection,
    )

    return builder.build_two_node_network(
        alice_name=alice_name,
        bob_name=bob_name,
        num_qubits=num_qubits,
        with_device_noise=False,  # Focus on channel noise for EPR generation
    )


# =============================================================================
# Worker Function (Top-Level for Pickling)
# =============================================================================


def _worker_generate_epr(
    network_config: Dict[str, Any],
    num_pairs: int,
    batch_id: int,
) -> EPRWorkerResult:
    """
    Worker function for isolated EPR generation using full SquidASM simulation.

    This function runs in a separate process with an independent NetSquid
    simulator instance. It must be a top-level function (not a method) to
    be picklable by multiprocessing.

    The function uses the complete SquidASM/NetSquid simulation stack:
    1. Builds a proper StackNetworkConfig from parameters
    2. Instantiates EPRGeneratorProgram for Alice and Bob
    3. Executes via squidasm.run.stack.run.run()
    4. Extracts measurement outcomes from simulation results

    Parameters
    ----------
    network_config : Dict[str, Any]
        Serialized network configuration dictionary containing NSM parameters:
        - channel_fidelity: float - EPR pair fidelity
        - detection_efficiency: float - Detector efficiency η
        - detector_error: float - Intrinsic detector error
        - dark_count_prob: float - Dark count probability
        - storage_noise_r: float - Storage noise parameter r
        - storage_rate_nu: float - Storage rate ν
        - delta_t_ns: float - Wait time Δt in nanoseconds
    num_pairs : int
        Number of EPR pairs to generate in this batch.
    batch_id : int
        Unique batch identifier for logging and ordering.

    Returns
    -------
    EPRWorkerResult
        Results containing measurement outcomes and bases for both parties.

    Raises
    ------
    SimulationError
        If the SquidASM simulation fails.

    Notes
    -----
    This function initializes a fresh NetSquid simulator via `ns.sim_reset()`
    to ensure complete isolation from other workers and the main process.

    The simulation uses proper quantum noise models:
    - DepolariseLinkConfig for fidelity-only noise
    - HeraldedDoubleClickConfig when detector effects are significant

    References
    ----------
    - SquidASM run() function: squidasm.run.stack.run.run
    - NetSquid simulator isolation: ns.sim_reset()
    """
    import time

    import netsquid as ns

    # Reset NetSquid simulator for complete isolation
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)

    # Import SquidASM components (late import for worker process)
    from squidasm.run.stack.run import run # type: ignore[import]

    from caligo.quantum.programs import create_epr_program_pair

    # Create deterministic but unique seed for this batch
    worker_seed = int(time.time() * 1000) + batch_id * 12345

    # Build network configuration from dict
    stack_config = _build_network_config_from_dict(
        config_dict=network_config,
        alice_name="Alice",
        bob_name="Bob",
        num_qubits=min(num_pairs + 10, 200),  # Allow headroom
    )

    # Create matched program pair
    alice_program, bob_program = create_epr_program_pair(
        alice_name="Alice",
        bob_name="Bob",
        num_pairs=num_pairs,
        seed=worker_seed,
    )

    # Execute simulation
    start_sim_time = ns.sim_time()

    results = run(
        config=stack_config,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=1,
    )

    end_sim_time = ns.sim_time()
    generation_time_ns = end_sim_time - start_sim_time

    # Extract results - results is List[List[Dict]] where outer is per-stack
    alice_result = None
    bob_result = None

    for stack_results in results:
        if stack_results and len(stack_results) > 0:
            result = stack_results[0]
            # Identify which node this result is from
            # The program stores 'outcomes' and 'bases' directly
            if "outcomes" in result and "bases" in result:
                # Need to determine if this is Alice or Bob
                # Alice is creator (first in programs dict typically)
                if alice_result is None:
                    alice_result = result
                else:
                    bob_result = result

    if alice_result is None or bob_result is None:
        # Fallback: try to access by index assuming order
        if len(results) >= 2:
            alice_result = results[0][0] if results[0] else None
            bob_result = results[1][0] if results[1] else None

    # Validate we have results
    if alice_result is None or bob_result is None:
        raise RuntimeError(
            f"Batch {batch_id}: Failed to extract results from simulation. "
            f"Got results: {results}"
        )

    return EPRWorkerResult(
        alice_outcomes=alice_result["outcomes"],
        alice_bases=alice_result["bases"],
        bob_outcomes=bob_result["outcomes"],
        bob_bases=bob_result["bases"],
        batch_id=batch_id,
        num_pairs=num_pairs,
        generation_time_ns=generation_time_ns,
    )

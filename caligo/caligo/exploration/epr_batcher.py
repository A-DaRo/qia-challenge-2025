"""
Batched EPR generation orchestrator for parallel exploration.

This module wraps the existing ParallelEPROrchestrator to support
batch generation across multiple parameter configurations in parallel.
It coordinates EPR generation for exploration samples efficiently.

Architecture
------------
The BatchedEPROrchestrator manages two levels of parallelism:

1. **Sample-level**: Multiple parameter configurations processed
   in parallel batches
2. **Pair-level**: Within each configuration, EPR pairs generated
   in parallel via ParallelEPROrchestrator

This two-level approach maximizes throughput while maintaining
isolation between different parameter configurations.

Zero-Copy Mode (HPC)
--------------------
For high-performance workloads, the orchestrator supports a zero-copy
mode using SharedMemoryArena. Workers write directly to shared memory,
eliminating serialization overhead and memory copies.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SharedMemoryArena                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│  │   Slot 0    │   Slot 1    │   Slot 2    │   Slot 3    │ ...  │
│  │ (Worker 0)  │ (Worker 1)  │ (Worker 2)  │ (Worker 3)  │      │
│  └─────────────┴─────────────┴─────────────┴─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from caligo.exploration.types import ExplorationSample, ReconciliationStrategy, DTYPE_INT
from caligo.protocol.base import PrecomputedEPRData
from caligo.quantum.parallel import ParallelEPRConfig, ParallelEPROrchestrator
from caligo.simulation.physical_model import NSMParameters
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Batched EPR Result
# =============================================================================


@dataclass
class BatchedEPRResult:
    """
    Result of generating EPR data for one ExplorationSample.

    Parameters
    ----------
    sample : ExplorationSample
        The parameter configuration used.
    epr_data : PrecomputedEPRData
        Generated EPR measurement data.
    generation_time_seconds : float
        Wall-clock time for generation.
    batch_id : int
        Identifier for this batch.
    error : Optional[str]
        Error message if generation failed.

    Attributes
    ----------
    sample : ExplorationSample
    epr_data : PrecomputedEPRData
    generation_time_seconds : float
    batch_id : int
    error : Optional[str]
    """

    sample: ExplorationSample
    epr_data: Optional[PrecomputedEPRData]
    generation_time_seconds: float
    batch_id: int
    error: Optional[str] = None

    def is_success(self) -> bool:
        """Check if EPR generation succeeded."""
        return self.epr_data is not None and self.error is None


# =============================================================================
# Network Configuration Builder
# =============================================================================


def build_network_config_for_sample(sample: ExplorationSample) -> Dict[str, Any]:
    """
    Build a serializable network configuration from an ExplorationSample.

    This function converts the high-level sample parameters into
    the network configuration format expected by ParallelEPROrchestrator.

    Parameters
    ----------
    sample : ExplorationSample
        Parameter configuration.

    Returns
    -------
    Dict[str, Any]
        Serializable network configuration dictionary.

    Notes
    -----
    The returned dictionary contains all parameters needed to configure
    the NetSquid simulation, but no actual NetSquid objects (which
    cannot be pickled for multiprocessing).
    """
    return {
        # NSM parameters
        "storage_noise_r": sample.storage_noise_r,
        "storage_rate_nu": sample.storage_rate_nu,
        "wait_time_ns": sample.wait_time_ns,
        # Channel parameters
        "channel_fidelity": sample.channel_fidelity,
        "detection_efficiency": sample.detection_efficiency,
        "detector_error": sample.detector_error,
        "dark_count_prob": sample.dark_count_prob,
        # Generation parameters
        "num_pairs": sample.num_pairs,
        # Network topology (default two-node)
        "alice_name": "Alice",
        "bob_name": "Bob",
        "distance_km": 0.0,  # Point-to-point for now
    }


def build_nsm_parameters_from_sample(sample: ExplorationSample) -> NSMParameters:
    """
    Build NSMParameters from an ExplorationSample.

    Parameters
    ----------
    sample : ExplorationSample
        Parameter configuration.

    Returns
    -------
    NSMParameters
        NSM parameter object for protocol execution.
    """
    # Map exploration sample fields to the NSMParameters dataclass fields
    return NSMParameters(
        storage_noise_r=sample.storage_noise_r,
        storage_rate_nu=sample.storage_rate_nu,
        delta_t_ns=sample.wait_time_ns,
        channel_fidelity=sample.channel_fidelity,
        detection_eff_eta=sample.detection_efficiency,
        detector_error=sample.detector_error,
        dark_count_prob=sample.dark_count_prob,
    )


# =============================================================================
# Worker Function (Process-Isolated)
# =============================================================================


def _generate_epr_for_sample(
    sample_dict: Dict[str, Any],
    batch_id: int,
    parallel_config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Worker function to generate EPR data for a single sample.

    This function runs in an isolated process to prevent NetSquid
    state contamination between different parameter configurations.

    Parameters
    ----------
    sample_dict : Dict[str, Any]
        Serialized ExplorationSample (as dict).
    batch_id : int
        Batch identifier.
    parallel_config_dict : Dict[str, Any]
        Serialized ParallelEPRConfig (as dict).

    Returns
    -------
    Dict[str, Any]
        Serialized BatchedEPRResult (as dict).

    Notes
    -----
    This function is designed for multiprocessing and only uses
    serializable inputs/outputs.
    """
    start_time = time.perf_counter()

    try:
        # Reconstruct sample from dict
        strategy = ReconciliationStrategy(sample_dict["strategy"])
        sample = ExplorationSample(
            storage_noise_r=sample_dict["storage_noise_r"],
            storage_rate_nu=sample_dict["storage_rate_nu"],
            wait_time_ns=sample_dict["wait_time_ns"],
            channel_fidelity=sample_dict["channel_fidelity"],
            detection_efficiency=sample_dict["detection_efficiency"],
            detector_error=sample_dict["detector_error"],
            dark_count_prob=sample_dict["dark_count_prob"],
            num_pairs=sample_dict["num_pairs"],
            strategy=strategy,
        )

        # Build network config
        network_config = build_network_config_for_sample(sample)

        # Reconstruct parallel config
        parallel_config = ParallelEPRConfig(
            enabled=parallel_config_dict.get("enabled", True),
            num_workers=parallel_config_dict.get("num_workers", 1),
            pairs_per_batch=parallel_config_dict.get("pairs_per_batch", 1000),
            shuffle_results=parallel_config_dict.get("shuffle_results", True),
        )

        # Generate EPR pairs
        # For exploration, we use sequential generation within each sample
        # to avoid nested parallelism overhead
        orchestrator = ParallelEPROrchestrator(
            config=ParallelEPRConfig(enabled=False),  # Sequential within sample
            network_config=network_config,
        )

        alice_out, alice_bases, bob_out, bob_bases = orchestrator.generate_parallel(
            total_pairs=sample.num_pairs
        )

        epr_data = {
            "alice_outcomes": alice_out,
            "alice_bases": alice_bases,
            "bob_outcomes": bob_out,
            "bob_bases": bob_bases,
        }

        elapsed = time.perf_counter() - start_time

        return {
            "sample_dict": sample_dict,
            "epr_data": epr_data,
            "generation_time_seconds": elapsed,
            "batch_id": batch_id,
            "error": None,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "sample_dict": sample_dict,
            "epr_data": None,
            "generation_time_seconds": elapsed,
            "batch_id": batch_id,
            "error": str(e),
        }


# =============================================================================
# Zero-Copy Worker Function (Shared Memory)
# =============================================================================


def _generate_epr_zero_copy(
    sample_dict: Dict[str, Any],
    batch_id: int,
    parallel_config_dict: Dict[str, Any],
    shm_name: str,
    slot_id: int,
    max_pairs: int,
    slot_size_bytes: int,
) -> Dict[str, Any]:
    """
    Worker function that writes EPR data directly to shared memory.

    This function is used in zero-copy mode to eliminate serialization
    overhead. Instead of returning large arrays through the pipe, it
    writes directly to a pre-allocated shared memory slot.

    Parameters
    ----------
    sample_dict : Dict[str, Any]
        Serialized ExplorationSample.
    batch_id : int
        Batch identifier.
    parallel_config_dict : Dict[str, Any]
        Serialized ParallelEPRConfig.
    shm_name : str
        Name of the shared memory segment.
    slot_id : int
        Slot ID for this worker's output.
    max_pairs : int
        Maximum pairs per slot.
    slot_size_bytes : int
        Size of each slot in bytes.

    Returns
    -------
    Dict[str, Any]
        Lightweight metadata (no EPR arrays).
    """
    # Import here to avoid circular imports
    from caligo.exploration.shared_memory import (
        SharedMemorySlotWriter,
        SlotMetadata,
    )

    start_time = time.perf_counter()
    start_ns = time.perf_counter_ns()

    try:
        # Reconstruct sample
        strategy = ReconciliationStrategy(sample_dict["strategy"])
        sample = ExplorationSample(
            storage_noise_r=sample_dict["storage_noise_r"],
            storage_rate_nu=sample_dict["storage_rate_nu"],
            wait_time_ns=sample_dict["wait_time_ns"],
            channel_fidelity=sample_dict["channel_fidelity"],
            detection_efficiency=sample_dict["detection_efficiency"],
            detector_error=sample_dict["detector_error"],
            dark_count_prob=sample_dict["dark_count_prob"],
            num_pairs=sample_dict["num_pairs"],
            strategy=strategy,
        )

        # Build network config
        network_config = build_network_config_for_sample(sample)

        # Generate EPR pairs (sequential within this worker)
        orchestrator = ParallelEPROrchestrator(
            config=ParallelEPRConfig(enabled=False),
            network_config=network_config,
        )

        alice_out, alice_bases, bob_out, bob_bases = orchestrator.generate_parallel(
            total_pairs=sample.num_pairs
        )

        # Attach to shared memory and write directly
        writer = SharedMemorySlotWriter(
            shm_name=shm_name,
            slot_id=slot_id,
            max_pairs=max_pairs,
            slot_size_bytes=slot_size_bytes,
        )

        try:
            n_pairs = writer.write_epr_data(
                alice_outcomes=alice_out,
                alice_bases=alice_bases,
                bob_outcomes=bob_out,
                bob_bases=bob_bases,
            )

            # Write success metadata
            elapsed_ns = time.perf_counter_ns() - start_ns
            writer.write_metadata(SlotMetadata(
                slot_id=slot_id,
                n_pairs=n_pairs,
                is_valid=True,
                generation_time_ns=elapsed_ns,
                error_code=0,
                sample_id=batch_id,
            ))

        finally:
            writer.close()

        elapsed = time.perf_counter() - start_time

        # Return lightweight metadata only (no EPR arrays!)
        return {
            "sample_dict": sample_dict,
            "slot_id": slot_id,
            "n_pairs": n_pairs,
            "generation_time_seconds": elapsed,
            "batch_id": batch_id,
            "error": None,
            "zero_copy": True,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "sample_dict": sample_dict,
            "slot_id": slot_id,
            "n_pairs": 0,
            "generation_time_seconds": elapsed,
            "batch_id": batch_id,
            "error": str(e),
            "zero_copy": True,
        }


# =============================================================================
# Batched EPR Orchestrator
# =============================================================================


@dataclass
class BatchedEPRConfig:
    """
    Configuration for batched EPR generation.

    Parameters
    ----------
    max_workers : int
        Maximum number of parallel worker processes.
    timeout_seconds : float
        Timeout per sample in seconds.
    inner_parallel : bool
        Whether to use parallel generation within each sample.
        Set to False for small num_pairs to reduce overhead.
    inner_workers : int
        Number of workers for inner parallel generation.
    inner_batch_size : int
        Batch size for inner parallel generation.
    use_zero_copy : bool
        If True, use shared memory for zero-copy data transfer.
        Significantly reduces memory overhead and serialization cost.
    max_pairs_per_slot : int
        Maximum EPR pairs per shared memory slot (only used if use_zero_copy=True).

    Attributes
    ----------
    max_workers : int
    timeout_seconds : float
    inner_parallel : bool
    inner_workers : int
    inner_batch_size : int
    use_zero_copy : bool
    max_pairs_per_slot : int
    """

    max_workers: int = field(default_factory=lambda: max(1, cpu_count() - 1))
    timeout_seconds: float = 600.0
    inner_parallel: bool = False
    inner_workers: int = 4
    inner_batch_size: int = 1000
    use_zero_copy: bool = True  # Enable by default for HPC
    max_pairs_per_slot: int = 1_000_000  # 1M pairs max


class BatchedEPROrchestrator:
    """
    Orchestrates parallel EPR generation across exploration samples.

    This class manages the generation of EPR data for multiple parameter
    configurations in parallel, coordinating worker processes and
    aggregating results.

    Zero-Copy Mode
    --------------
    When `config.use_zero_copy=True`, the orchestrator allocates a
    SharedMemoryArena and workers write directly to it. This eliminates
    serialization overhead and reduces memory usage.

    Parameters
    ----------
    config : BatchedEPRConfig
        Configuration for parallel execution.

    Attributes
    ----------
    config : BatchedEPRConfig
        Stored configuration.
    _executor : Optional[ProcessPoolExecutor]
        Worker pool (created lazily).
    _arena : Optional[SharedMemoryArena]
        Shared memory arena (only if use_zero_copy=True).

    Examples
    --------
    >>> orchestrator = BatchedEPROrchestrator(BatchedEPRConfig(max_workers=8))
    >>> samples = sampler.generate(100)
    >>> results = orchestrator.generate_batch(samples)
    >>> success_count = sum(1 for r in results if r.is_success())
    >>> orchestrator.shutdown()

    Notes
    -----
    The orchestrator should be shut down via `shutdown()` to release
    worker resources. Alternatively, use as a context manager:

    >>> with BatchedEPROrchestrator(config) as orch:
    ...     results = orch.generate_batch(samples)
    """

    def __init__(self, config: Optional[BatchedEPRConfig] = None) -> None:
        """
        Initialize the batched EPR orchestrator.

        Parameters
        ----------
        config : Optional[BatchedEPRConfig]
            Configuration. Uses defaults if None.
        """
        self.config = config or BatchedEPRConfig()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._arena: Optional["SharedMemoryArena"] = None

        logger.info(
            "Initialized BatchedEPROrchestrator with %d workers (zero_copy=%s)",
            self.config.max_workers,
            self.config.use_zero_copy,
        )

    def __enter__(self) -> "BatchedEPROrchestrator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create the process pool executor."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
        return self._executor

    def _get_arena(self) -> "SharedMemoryArena":
        """Get or create the shared memory arena."""
        if self._arena is None:
            from caligo.exploration.shared_memory import SharedMemoryArena
            self._arena = SharedMemoryArena(
                num_slots=self.config.max_workers,
                max_pairs_per_slot=self.config.max_pairs_per_slot,
            )
        return self._arena

    def shutdown(self) -> None:
        """Shut down the worker pool and release shared memory."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.debug("Shut down BatchedEPROrchestrator executor")

        if self._arena is not None:
            self._arena.cleanup()
            self._arena = None
            logger.debug("Cleaned up BatchedEPROrchestrator shared memory arena")

    def generate_batch(
        self,
        samples: List[ExplorationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BatchedEPRResult]:
        """
        Generate EPR data for a batch of samples in parallel.

        If `config.use_zero_copy=True`, uses shared memory for zero-copy
        data transfer. Otherwise, uses standard pickling.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Parameter configurations to generate EPR data for.
        progress_callback : Optional[Callable[[int, int], None]]
            Callback for progress updates (completed, total).

        Returns
        -------
        List[BatchedEPRResult]
            Results for each sample (same order as input).

        Examples
        --------
        >>> def on_progress(done, total):
        ...     print(f"Progress: {done}/{total}")
        >>> results = orchestrator.generate_batch(samples, progress_callback=on_progress)
        """
        n_samples = len(samples)
        if n_samples == 0:
            return []

        # Route to zero-copy or standard implementation
        if self.config.use_zero_copy:
            return self._generate_batch_zero_copy(samples, progress_callback)
        else:
            return self._generate_batch_standard(samples, progress_callback)

    def _generate_batch_standard(
        self,
        samples: List[ExplorationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BatchedEPRResult]:
        """Standard batch generation using pickling."""
        n_samples = len(samples)
        executor = self._get_executor()
        futures_to_idx: Dict[Any, int] = {}

        # Serialize samples and config for multiprocessing
        parallel_config_dict = {
            "enabled": self.config.inner_parallel,
            "num_workers": self.config.inner_workers,
            "pairs_per_batch": self.config.inner_batch_size,
            "shuffle_results": True,
        }

        # Submit all samples
        for idx, sample in enumerate(samples):
            sample_dict = {
                "storage_noise_r": sample.storage_noise_r,
                "storage_rate_nu": sample.storage_rate_nu,
                "wait_time_ns": sample.wait_time_ns,
                "channel_fidelity": sample.channel_fidelity,
                "detection_efficiency": sample.detection_efficiency,
                "detector_error": sample.detector_error,
                "dark_count_prob": sample.dark_count_prob,
                "num_pairs": sample.num_pairs,
                "strategy": sample.strategy.value,
            }
            future = executor.submit(
                _generate_epr_for_sample,
                sample_dict=sample_dict,
                batch_id=idx,
                parallel_config_dict=parallel_config_dict,
            )
            futures_to_idx[future] = idx

        # Collect results
        results: List[Optional[BatchedEPRResult]] = [None] * n_samples
        completed = 0

        for future in as_completed(futures_to_idx.keys()):
            idx = futures_to_idx[future]
            try:
                result_dict = future.result(timeout=self.config.timeout_seconds)

                # Reconstruct sample
                sample = samples[idx]

                # Reconstruct EPR data if successful
                epr_data = None
                if result_dict["epr_data"] is not None:
                    epr_dict = result_dict["epr_data"]
                    # CRITICAL FIX: Cast to int64 for protocol layer compatibility
                    # Int8 memory optimization is for storage/IPC only
                    # Note: data may be list (standard mode) or array (zero-copy mode)
                    epr_data = PrecomputedEPRData(
                        alice_outcomes=np.asarray(epr_dict["alice_outcomes"], dtype=np.int64),
                        alice_bases=np.asarray(epr_dict["alice_bases"], dtype=np.int64),
                        bob_outcomes=np.asarray(epr_dict["bob_outcomes"], dtype=np.int64),
                        bob_bases=np.asarray(epr_dict["bob_bases"], dtype=np.int64),
                    )

                results[idx] = BatchedEPRResult(
                    sample=sample,
                    epr_data=epr_data,
                    generation_time_seconds=result_dict["generation_time_seconds"],
                    batch_id=result_dict["batch_id"],
                    error=result_dict["error"],
                )

            except Exception as e:
                results[idx] = BatchedEPRResult(
                    sample=samples[idx],
                    epr_data=None,
                    generation_time_seconds=0.0,
                    batch_id=idx,
                    error=str(e),
                )
                logger.warning("Sample %d EPR generation failed: %s", idx, e)

            completed += 1
            if progress_callback is not None:
                progress_callback(completed, n_samples)

        # All results should be filled
        return [r for r in results if r is not None]

    def _generate_batch_zero_copy(
        self,
        samples: List[ExplorationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BatchedEPRResult]:
        """
        Zero-copy batch generation using shared memory.

        Workers write EPR data directly to shared memory slots, eliminating
        serialization overhead and memory copies.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Samples to process.
        progress_callback : Optional[Callable[[int, int], None]]
            Progress callback.

        Returns
        -------
        List[BatchedEPRResult]
            Results with EPR data loaded from shared memory.
        """
        from caligo.exploration.shared_memory import (
            SharedMemoryArena,
            create_zero_copy_epr_data,
        )

        n_samples = len(samples)
        arena = self._get_arena()
        executor = self._get_executor()

        # Serialize config for workers
        parallel_config_dict = {
            "enabled": self.config.inner_parallel,
            "num_workers": self.config.inner_workers,
            "pairs_per_batch": self.config.inner_batch_size,
            "shuffle_results": True,
        }

        # Process samples in chunks of num_slots
        results: List[Optional[BatchedEPRResult]] = [None] * n_samples
        completed = 0

        chunk_size = self.config.max_workers
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_samples = samples[chunk_start:chunk_end]

            futures_to_info: Dict[Any, Tuple[int, int]] = {}  # future -> (global_idx, slot_id)

            # Submit chunk with slot assignments
            for local_idx, sample in enumerate(chunk_samples):
                global_idx = chunk_start + local_idx
                slot_id = arena.acquire_slot(sample_id=global_idx)

                sample_dict = {
                    "storage_noise_r": sample.storage_noise_r,
                    "storage_rate_nu": sample.storage_rate_nu,
                    "wait_time_ns": sample.wait_time_ns,
                    "channel_fidelity": sample.channel_fidelity,
                    "detection_efficiency": sample.detection_efficiency,
                    "detector_error": sample.detector_error,
                    "dark_count_prob": sample.dark_count_prob,
                    "num_pairs": sample.num_pairs,
                    "strategy": sample.strategy.value,
                }

                future = executor.submit(
                    _generate_epr_zero_copy,
                    sample_dict=sample_dict,
                    batch_id=global_idx,
                    parallel_config_dict=parallel_config_dict,
                    shm_name=arena.shm_name,
                    slot_id=slot_id,
                    max_pairs=arena.max_pairs_per_slot,
                    slot_size_bytes=arena.slot_size_bytes,
                )
                futures_to_info[future] = (global_idx, slot_id)

            # Collect chunk results
            for future in as_completed(futures_to_info.keys()):
                global_idx, slot_id = futures_to_info[future]
                sample = samples[global_idx]

                try:
                    result_dict = future.result(timeout=self.config.timeout_seconds)

                    if result_dict["error"] is None:
                        # Read EPR data from shared memory (zero-copy view)
                        n_pairs = result_dict["n_pairs"]
                        zc_data = create_zero_copy_epr_data(arena, slot_id, n_pairs)
                        
                        # DEBUG: Check types
                        logger.info(f"DEBUG: zc_data type: {type(zc_data)}")
                        logger.info(f"DEBUG: alice_outcomes type: {type(zc_data.alice_outcomes)}")
                        
                        # Force conversion if it is a list/tuple (unexpected)
                        if isinstance(zc_data.alice_outcomes, (list, tuple)):
                            logger.warning(f"UNEXPECTED: alice_outcomes is a {type(zc_data.alice_outcomes)}! converting...")
                            zc_data.alice_outcomes = np.array(zc_data.alice_outcomes)
                            zc_data.alice_bases = np.array(zc_data.alice_bases)
                            zc_data.bob_outcomes = np.array(zc_data.bob_outcomes)
                            zc_data.bob_bases = np.array(zc_data.bob_bases)

                        # Copy data to PrecomputedEPRData (needed for protocol)
                        # The copy here is necessary because we'll release the slot
                        # CRITICAL FIX: Cast to int64 for protocol layer compatibility
                        # Int8 memory optimization is for storage/IPC only
                        epr_data = PrecomputedEPRData(
                            alice_outcomes=zc_data.alice_outcomes.astype(np.int64, copy=True),
                            alice_bases=zc_data.alice_bases.astype(np.int64, copy=True),
                            bob_outcomes=zc_data.bob_outcomes.astype(np.int64, copy=True),
                            bob_bases=zc_data.bob_bases.astype(np.int64, copy=True),
                        )

                        # Check for Zero-Data Corruption
                        # If we have pairs but the data is all zeros (highly unlikely for valid random bases),
                        # it indicates shared memory read failure (reading empty/wrong address).
                        if n_pairs > 0 and not np.any(epr_data.alice_bases):
                             # Check if it's truly all zeros (bases should be ~50% 0/1)
                             # A standard run won't produce 1000+ zeros in a row for bases.
                            error_msg = f"Zero-Data Corruption Detected: {n_pairs} pairs but Alice bases are all zero."
                            #logger.error(error_msg)
                            results[global_idx] = BatchedEPRResult(
                                sample=sample,
                                epr_data=None, # Invalidate data
                                generation_time_seconds=result_dict["generation_time_seconds"],
                                batch_id=result_dict["batch_id"],
                                error=error_msg,
                            )
                        else:
                            results[global_idx] = BatchedEPRResult(
                                sample=sample,
                                epr_data=epr_data,
                                generation_time_seconds=result_dict["generation_time_seconds"],
                                batch_id=result_dict["batch_id"],
                                error=None,
                            )
                    else:
                        results[global_idx] = BatchedEPRResult(
                            sample=sample,
                            epr_data=None,
                            generation_time_seconds=result_dict["generation_time_seconds"],
                            batch_id=result_dict["batch_id"],
                            error=result_dict["error"],
                        )

                except Exception as e:
                    results[global_idx] = BatchedEPRResult(
                        sample=sample,
                        epr_data=None,
                        generation_time_seconds=0.0,
                        batch_id=global_idx,
                        error=str(e),
                    )
                    logger.warning("Sample %d EPR generation failed: %s", global_idx, e)

                finally:
                    # Release slot immediately after reading
                    arena.release_slot(slot_id)

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n_samples)

        return [r for r in results if r is not None]

    def generate_batch_streaming(
        self,
        samples: List[ExplorationSample],
    ) -> "Generator[BatchedEPRResult, None, None]":
        """
        Stream EPR results as they complete using generators.

        Unlike `generate_batch`, this yields results as soon as each worker
        completes, enabling pipelined execution where downstream processing
        can begin immediately. Memory is released as each result is yielded.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Parameter configurations to generate EPR data for.

        Yields
        ------
        BatchedEPRResult
            EPR generation result. Order may differ from input order.

        Notes
        -----
        This method provides:
        - **Streaming semantics**: Results are yielded immediately on completion
        - **Memory efficiency**: Shared memory slots are released after yielding
        - **Pipelined execution**: Downstream processing starts before batch completes

        For strict ordering, use `generate_batch` instead.

        Examples
        --------
        >>> for result in orchestrator.generate_batch_streaming(samples):
        ...     if result.is_success():
        ...         process_immediately(result)  # Don't wait for full batch
        """
        from typing import Generator

        n_samples = len(samples)
        if n_samples == 0:
            return

        if self.config.use_zero_copy:
            yield from self._stream_zero_copy(samples)
        else:
            yield from self._stream_standard(samples)

    def _stream_standard(
        self,
        samples: List[ExplorationSample],
    ) -> "Generator[BatchedEPRResult, None, None]":
        """Stream results using standard pickling."""
        from typing import Generator

        n_samples = len(samples)
        executor = self._get_executor()
        futures_to_idx: Dict[Any, int] = {}

        parallel_config_dict = {
            "enabled": self.config.inner_parallel,
            "num_workers": self.config.inner_workers,
            "pairs_per_batch": self.config.inner_batch_size,
            "shuffle_results": True,
        }

        # Submit all samples
        for idx, sample in enumerate(samples):
            sample_dict = {
                "storage_noise_r": sample.storage_noise_r,
                "storage_rate_nu": sample.storage_rate_nu,
                "wait_time_ns": sample.wait_time_ns,
                "channel_fidelity": sample.channel_fidelity,
                "detection_efficiency": sample.detection_efficiency,
                "detector_error": sample.detector_error,
                "dark_count_prob": sample.dark_count_prob,
                "num_pairs": sample.num_pairs,
                "strategy": sample.strategy.value,
            }
            future = executor.submit(
                _generate_epr_for_sample,
                sample_dict=sample_dict,
                batch_id=idx,
                parallel_config_dict=parallel_config_dict,
            )
            futures_to_idx[future] = idx

        # Yield results as they complete
        for future in as_completed(futures_to_idx.keys()):
            idx = futures_to_idx[future]
            sample = samples[idx]

            try:
                result_dict = future.result(timeout=self.config.timeout_seconds)

                epr_data = None
                if result_dict["epr_data"] is not None:
                    epr_dict = result_dict["epr_data"]
                    # CRITICAL FIX: Cast to int64 for protocol layer compatibility
                    # Note: data may be list (standard mode) or array (zero-copy mode)
                    epr_data = PrecomputedEPRData(
                        alice_outcomes=np.asarray(epr_dict["alice_outcomes"], dtype=np.int64),
                        alice_bases=np.asarray(epr_dict["alice_bases"], dtype=np.int64),
                        bob_outcomes=np.asarray(epr_dict["bob_outcomes"], dtype=np.int64),
                        bob_bases=np.asarray(epr_dict["bob_bases"], dtype=np.int64),
                    )

                yield BatchedEPRResult(
                    sample=sample,
                    epr_data=epr_data,
                    generation_time_seconds=result_dict["generation_time_seconds"],
                    batch_id=result_dict["batch_id"],
                    error=result_dict["error"],
                )

            except Exception as e:
                yield BatchedEPRResult(
                    sample=sample,
                    epr_data=None,
                    generation_time_seconds=0.0,
                    batch_id=idx,
                    error=str(e),
                )
                logger.warning("Sample %d EPR generation failed: %s", idx, e)

    def _stream_zero_copy(
        self,
        samples: List[ExplorationSample],
    ) -> "Generator[BatchedEPRResult, None, None]":
        """
        Stream results using zero-copy shared memory.

        Memory slots are acquired per-sample and released immediately after
        the result is yielded, maintaining constant memory footprint.
        """
        from typing import Generator
        from caligo.exploration.shared_memory import (
            SharedMemoryArena,
            create_zero_copy_epr_data,
        )

        n_samples = len(samples)
        arena = self._get_arena()
        executor = self._get_executor()

        parallel_config_dict = {
            "enabled": self.config.inner_parallel,
            "num_workers": self.config.inner_workers,
            "pairs_per_batch": self.config.inner_batch_size,
            "shuffle_results": True,
        }

        # Process in chunks to limit concurrent slot usage
        chunk_size = self.config.max_workers

        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_samples = samples[chunk_start:chunk_end]

            futures_to_info: Dict[Any, Tuple[int, int]] = {}

            # Submit chunk
            for local_idx, sample in enumerate(chunk_samples):
                global_idx = chunk_start + local_idx
                slot_id = arena.acquire_slot(sample_id=global_idx)

                sample_dict = {
                    "storage_noise_r": sample.storage_noise_r,
                    "storage_rate_nu": sample.storage_rate_nu,
                    "wait_time_ns": sample.wait_time_ns,
                    "channel_fidelity": sample.channel_fidelity,
                    "detection_efficiency": sample.detection_efficiency,
                    "detector_error": sample.detector_error,
                    "dark_count_prob": sample.dark_count_prob,
                    "num_pairs": sample.num_pairs,
                    "strategy": sample.strategy.value,
                }

                future = executor.submit(
                    _generate_epr_zero_copy,
                    sample_dict=sample_dict,
                    batch_id=global_idx,
                    parallel_config_dict=parallel_config_dict,
                    shm_name=arena.shm_name,
                    slot_id=slot_id,
                    max_pairs=arena.max_pairs_per_slot,
                    slot_size_bytes=arena.slot_size_bytes,
                )
                futures_to_info[future] = (global_idx, slot_id)

            # Yield results as they complete
            for future in as_completed(futures_to_info.keys()):
                global_idx, slot_id = futures_to_info[future]
                sample = samples[global_idx]

                try:
                    result_dict = future.result(timeout=self.config.timeout_seconds)

                    if result_dict["error"] is None:
                        n_pairs = result_dict["n_pairs"]
                        zc_data = create_zero_copy_epr_data(arena, slot_id, n_pairs)

                        # DEBUG: Force check types
                        logger.info(f"DEBUG STREAM: zc_data type: {type(zc_data)}")
                        logger.info(f"DEBUG STREAM: alice_outcomes type: {type(zc_data.alice_outcomes)}")

                        # Ensure arrays are numpy arrays 
                        if not isinstance(zc_data.alice_outcomes, np.ndarray):
                            logger.warning(f"UNEXPECTED STREAM: alice_outcomes is {type(zc_data.alice_outcomes)}! converting...")
                            zc_data.alice_outcomes = np.array(zc_data.alice_outcomes)
                            zc_data.alice_bases = np.array(zc_data.alice_bases)
                            zc_data.bob_outcomes = np.array(zc_data.bob_outcomes)
                            zc_data.bob_bases = np.array(zc_data.bob_bases)

                        # Check for Zero-Data Corruption (all zeros)
                        # This happens if shared memory read fails silently or offsets are wrong
                        if n_pairs > 0 and not np.any(zc_data.alice_bases):
                             yield BatchedEPRResult(
                                sample=sample,
                                epr_data=None,
                                generation_time_seconds=result_dict["generation_time_seconds"],
                                batch_id=result_dict["batch_id"],
                                error="Zero-Data Corruption Detected: Shared memory contained all zeros",
                            )
                             # Release slot immediately
                             arena.release_slot(slot_id)
                             continue

                        # Copy to PrecomputedEPRData before releasing slot
                        # CRITICAL FIX: Cast to int64 for protocol layer compatibility
                        epr_data = PrecomputedEPRData(
                            alice_outcomes=zc_data.alice_outcomes.astype(np.int64, copy=True),
                            alice_bases=zc_data.alice_bases.astype(np.int64, copy=True),
                            bob_outcomes=zc_data.bob_outcomes.astype(np.int64, copy=True),
                            bob_bases=zc_data.bob_bases.astype(np.int64, copy=True),
                        )

                        yield BatchedEPRResult(
                            sample=sample,
                            epr_data=epr_data,
                            generation_time_seconds=result_dict["generation_time_seconds"],
                            batch_id=result_dict["batch_id"],
                            error=None,
                        )
                    else:
                        yield BatchedEPRResult(
                            sample=sample,
                            epr_data=None,
                            generation_time_seconds=result_dict["generation_time_seconds"],
                            batch_id=result_dict["batch_id"],
                            error=result_dict["error"],
                        )

                except Exception as e:
                    yield BatchedEPRResult(
                        sample=sample,
                        epr_data=None,
                        generation_time_seconds=0.0,
                        batch_id=global_idx,
                        error=str(e),
                    )
                    logger.warning("Sample %d EPR generation failed: %s", global_idx, e)

                finally:
                    # Release slot immediately after yielding
                    arena.release_slot(slot_id)

    def generate_single(
        self,
        sample: ExplorationSample,
    ) -> BatchedEPRResult:
        """
        Generate EPR data for a single sample.

        Parameters
        ----------
        sample : ExplorationSample
            Parameter configuration.

        Returns
        -------
        BatchedEPRResult
            Generation result.

        Notes
        -----
        For single samples, this runs synchronously without multiprocessing
        overhead. Use `generate_batch` for multiple samples.
        """
        results = self.generate_batch([sample])
        return results[0]

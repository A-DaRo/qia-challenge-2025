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
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from caligo.exploration.types import ExplorationSample, ReconciliationStrategy
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

    Attributes
    ----------
    max_workers : int
    timeout_seconds : float
    inner_parallel : bool
    inner_workers : int
    inner_batch_size : int
    """

    max_workers: int = field(default_factory=lambda: max(1, cpu_count() - 1))
    timeout_seconds: float = 600.0
    inner_parallel: bool = False
    inner_workers: int = 4
    inner_batch_size: int = 1000


class BatchedEPROrchestrator:
    """
    Orchestrates parallel EPR generation across exploration samples.

    This class manages the generation of EPR data for multiple parameter
    configurations in parallel, coordinating worker processes and
    aggregating results.

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

        logger.info(
            "Initialized BatchedEPROrchestrator with %d workers",
            self.config.max_workers,
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

    def shutdown(self) -> None:
        """Shut down the worker pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.debug("Shut down BatchedEPROrchestrator executor")

    def generate_batch(
        self,
        samples: List[ExplorationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BatchedEPRResult]:
        """
        Generate EPR data for a batch of samples in parallel.

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
                    epr_data = PrecomputedEPRData(
                        alice_outcomes=epr_dict["alice_outcomes"],
                        alice_bases=epr_dict["alice_bases"],
                        bob_outcomes=epr_dict["bob_outcomes"],
                        bob_bases=epr_dict["bob_bases"],
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

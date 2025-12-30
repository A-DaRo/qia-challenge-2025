"""
Phase 1 Executor: Latin Hypercube Sampling warmup campaign.

This module implements the Phase 1 exploration workflow, which generates
an initial dataset using Latin Hypercube Sampling to provide warmup
data for the Gaussian Process surrogate model.

Workflow
--------
1. Generate LHS design (or resume from checkpoint)
2. For each batch:
   a. Generate EPR data in parallel
   b. Execute protocols
   c. Write results to HDF5
   d. Update checkpoint
3. Summary statistics and validation

Features
--------
- **Fault Tolerance**: Full checkpoint/resume support
- **Progress Tracking**: TQDM progress bars with live metrics
- **Logging Integration**: TQDM-compatible logging output

Usage
-----
Command line:
    $ python -m caligo.exploration.lhs_executor --samples 2000 --batch-size 50

Python API:
    >>> executor = Phase1Executor(output_dir=Path("./results"))
    >>> executor.run(num_samples=2000, batch_size=50)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from caligo.exploration.epr_batcher import BatchedEPROrchestrator, BatchedEPRConfig
from caligo.exploration.harness import HarnessConfig, ProtocolHarness
from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    HDF5Writer,
    StateManager,
    capture_rng_state,
    restore_rng_state,
    result_to_hdf5_arrays,
)
from caligo.exploration.sampler import (
    LHSSampler,
    ParameterBounds,
    compute_feasibility_vectorized,
)
from caligo.exploration.types import (
    DTYPE_FLOAT,
    ExplorationConfig,
    ExplorationSample,
    Float32Array,
    Phase1State,
    ProtocolOutcome,
    ProtocolResult,
)
from caligo.utils.logging import get_logger
from caligo.utils.math import compute_qber_erven

logger = get_logger(__name__)


# =============================================================================
# TQDM-Compatible Logging Handler
# =============================================================================


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that writes through tqdm.write() for clean output.

    This handler ensures log messages don't interfere with progress bars.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record via tqdm.write()."""
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_tqdm_logging(log_level: int = logging.INFO) -> None:
    """
    Configure logging to work with tqdm progress bars.

    Parameters
    ----------
    log_level : int
        Logging level (default: INFO).
    """
    # Remove existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add tqdm-compatible handler
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


# =============================================================================
# Phase 1 Metrics
# =============================================================================


@dataclass
class Phase1Metrics:
    """
    Metrics tracked during Phase 1 execution.

    Attributes
    ----------
    target_feasible_samples : int
        Target number of feasible samples to collect.
    feasible_samples : int
        Number of feasible samples collected.
    skipped_infeasible : int
        Number of samples skipped due to infeasibility.
    total_samples_processed : int
        Total samples processed (feasible + skipped).
    failed_samples : int
        Failed protocol executions.
    yield_rate : float
        Fraction of feasible samples from total processed.
    current_efficiency : float
        Rolling average efficiency of recent samples.
    success_rate : float
        Fraction of successful executions.
    elapsed_seconds : float
        Total elapsed time.
    samples_per_second : float
        Throughput rate.

    Notes
    -----
    Backwards compatibility: `total_samples` and `completed_samples` are
    exposed as properties to avoid breaking callers that expect the old
    attribute names.
    """

    target_feasible_samples: int = 0
    feasible_samples: int = 0
    skipped_infeasible: int = 0
    total_samples_processed: int = 0
    failed_samples: int = 0
    yield_rate: float = 1.0
    current_efficiency: float = 0.0
    success_rate: float = 0.0
    elapsed_seconds: float = 0.0
    samples_per_second: float = 0.0

    def to_progress_dict(self) -> Dict[str, str]:
        """Convert to tqdm postfix dictionary."""
        return {
            "yield": f"{self.yield_rate:.1%}",
            "eff": f"{self.current_efficiency:.3f}",
            "success": f"{self.success_rate:.1%}",
            "rate": f"{self.samples_per_second:.1f}/s",
        }

    # Backwards-compatible aliases
    @property
    def total_samples(self) -> int:
        """Compatibility alias for old `total_samples` attribute."""
        return int(self.target_feasible_samples)

    @property
    def completed_samples(self) -> int:
        """Compatibility alias for old `completed_samples` attribute."""
        return int(self.feasible_samples)


# =============================================================================
# Phase 1 Executor
# =============================================================================


class Phase1Executor:
    """
    Executor for Phase 1 LHS warmup campaign.

    This class orchestrates the entire Phase 1 workflow, including
    checkpoint management, progress tracking, and result persistence.

    Parameters
    ----------
    output_dir : Path
        Directory for output files.
    bounds : Optional[ParameterBounds]
        Parameter space bounds.
    random_seed : Optional[int]
        Random seed for reproducibility.
    harness_config : Optional[HarnessConfig]
        Protocol execution configuration.
    epr_config : Optional[BatchedEPRConfig]
        EPR generation configuration.

    Attributes
    ----------
    output_dir : Path
    bounds : ParameterBounds
    random_seed : Optional[int]
    _sampler : LHSSampler
    _harness : ProtocolHarness
    _hdf5_writer : HDF5Writer
    _state_manager : StateManager
    _metrics : Phase1Metrics

    Examples
    --------
    >>> executor = Phase1Executor(
    ...     output_dir=Path("./exploration_results"),
    ...     random_seed=42,
    ... )
    >>> executor.run(num_samples=2000, batch_size=50)
    """

    def __init__(
        self,
        output_dir: Path,
        bounds: Optional[ParameterBounds] = None,
        random_seed: Optional[int] = 42,
        harness_config: Optional[HarnessConfig] = None,
        epr_config: Optional[BatchedEPRConfig] = None,
    ) -> None:
        """
        Initialize the Phase 1 executor.

        Parameters
        ----------
        output_dir : Path
            Directory for output files.
        bounds : Optional[ParameterBounds]
            Parameter space bounds.
        random_seed : Optional[int]
            Random seed for reproducibility.
        harness_config : Optional[HarnessConfig]
            Protocol execution configuration.
        epr_config : Optional[BatchedEPRConfig]
            EPR generation configuration.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bounds = bounds or ParameterBounds()
        self.random_seed = random_seed

        # Initialize components (lazy)
        self._sampler: Optional[LHSSampler] = None
        self._harness: Optional[ProtocolHarness] = None
        self._hdf5_writer: Optional[HDF5Writer] = None
        self._state_manager: Optional[StateManager] = None

        self._harness_config = harness_config or HarnessConfig()
        self._epr_config = epr_config or BatchedEPRConfig()

        self._metrics = Phase1Metrics()
        self._all_samples: List[ExplorationSample] = []
        self._start_time: float = 0.0

        logger.info(
            "Initialized Phase1Executor (output_dir=%s, seed=%s)",
            self.output_dir,
            self.random_seed,
        )

    @property
    def hdf5_path(self) -> Path:
        """Path to the HDF5 data file."""
        return self.output_dir / "exploration_data.h5"

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.output_dir / "phase1_checkpoint.pkl"

    def _init_components(self) -> None:
        """Initialize all components."""
        if self._sampler is None:
            self._sampler = LHSSampler(
                bounds=self.bounds,
                seed=self.random_seed,
            )

        if self._harness is None:
            epr_orchestrator = BatchedEPROrchestrator(self._epr_config)
            self._harness = ProtocolHarness(
                config=self._harness_config,
                epr_orchestrator=epr_orchestrator,
            )

        if self._hdf5_writer is None:
            self._hdf5_writer = HDF5Writer(self.hdf5_path, mode="a")
            self._hdf5_writer.open()

        if self._state_manager is None:
            self._state_manager = StateManager(self.checkpoint_path)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._harness is not None:
            self._harness.shutdown()
            self._harness = None

        if self._hdf5_writer is not None:
            self._hdf5_writer.close()
            self._hdf5_writer = None
            # Small delay to ensure HDF5 file handle is fully released
            time.sleep(0.1)

    def run(
        self,
        num_samples: int = 2000,
        batch_size: int = 50,
        checkpoint_interval: int = 5,
        verbose: bool = True,
        use_streaming: bool = True,
    ) -> Phase1Metrics:
        """
        Run the Phase 1 LHS campaign with adaptive feasibility filtering.

        Parameters
        ----------
        num_samples : int
            Target number of feasible samples to collect.
        batch_size : int
            Number of feasible samples per batch (target).
        checkpoint_interval : int
            Save checkpoint every N batches.
        verbose : bool
            Whether to show progress bars.
        use_streaming : bool
            If True, use streaming EPR generation with `as_completed()`.
            This enables pipelined execution and reduces memory footprint.

        Returns
        -------
        Phase1Metrics
            Final campaign metrics.

        Raises
        ------
        RuntimeError
            If the campaign cannot be started or resumed.
        """
        self._start_time = time.perf_counter()
        self._init_components()
        self._use_streaming = use_streaming

        try:
            # Check for existing checkpoint
            state = self._state_manager.load(Phase1State)
            if state is not None:
                logger.info(
                    "Resuming from checkpoint: %d/%d feasible samples (%d total processed)",
                    state.feasible_samples_collected,
                    state.target_feasible_samples,
                    state.total_samples_processed,
                )
                restore_rng_state(state.rng_state)
                # Pre-generate full LHS design for deterministic sampling
                self._all_samples = self._sampler.generate(num_samples * 20)  # Overgenerate
            else:
                logger.info("Starting new Phase 1 campaign: %d feasible samples target", num_samples)
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                # Pre-generate LHS design with oversampling buffer
                self._all_samples = self._sampler.generate(num_samples * 20)
                state = Phase1State(
                    target_feasible_samples=num_samples,
                    feasible_samples_collected=0,
                    total_samples_processed=0,
                    current_batch_start=0,
                    rng_state=capture_rng_state(),
                    hdf5_path=self.hdf5_path,
                )

            self._metrics.target_feasible_samples = num_samples
            self._metrics.feasible_samples = state.feasible_samples_collected
            self._metrics.total_samples_processed = state.total_samples_processed
            self._metrics.skipped_infeasible = (
                state.total_samples_processed - state.feasible_samples_collected
            )

            # Progress bar setup
            pbar = tqdm(
                total=num_samples,
                initial=state.feasible_samples_collected,
                desc="Phase 1 (LHS)",
                unit="samples",
                disable=not verbose,
                dynamic_ncols=True,
            )

            batches_completed = 0
            sample_idx = state.total_samples_processed

            # Main loop: collect until we reach target feasible samples
            while state.feasible_samples_collected < state.target_feasible_samples:
                # Adaptive oversampling: estimate how many raw samples needed
                remaining = state.target_feasible_samples - state.feasible_samples_collected
                current_yield = self._metrics.yield_rate if self._metrics.yield_rate > 0.01 else 0.1
                adjusted_batch_size = min(
                    int(batch_size / current_yield),
                    10000,  # Cap to prevent memory issues
                    len(self._all_samples) - sample_idx,  # Don't exceed buffer
                )
                adjusted_batch_size = max(adjusted_batch_size, batch_size)  # Min = requested batch_size

                # Extract raw samples
                batch_end = min(sample_idx + adjusted_batch_size, len(self._all_samples))
                if sample_idx >= batch_end:
                    logger.warning(
                        "LHS buffer exhausted at %d samples (collected %d/%d feasible). "
                        "Consider increasing oversampling factor.",
                        sample_idx,
                        state.feasible_samples_collected,
                        state.target_feasible_samples,
                    )
                    break

                raw_samples = self._all_samples[sample_idx:batch_end]

                # Step 1: Vectorized feasibility filtering using Numba kernel
                feasible_samples, infeasible_results = self._filter_feasible_batch(raw_samples)

                # Step 2: Execute feasible samples (slow path)
                # Use streaming or standard mode based on configuration
                feasible_results = []
                if feasible_samples:
                    if self._use_streaming:
                        feasible_results = self._execute_batch_streaming(feasible_samples, pbar)
                    else:
                        feasible_results = self._execute_batch(feasible_samples, pbar)

                # Step 3: Combine and persist all results
                all_results = infeasible_results + feasible_results

                if all_results:
                    inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(all_results)
                    self._hdf5_writer.append_batch(
                        GROUP_LHS_WARMUP,
                        inputs=inputs,
                        outputs=outputs,
                        outcomes=outcomes,
                        metadata=metadata,
                    )

                # Step 4: Update state
                state.total_samples_processed = batch_end
                state.feasible_samples_collected += len(feasible_samples)
                state.current_batch_start = batch_end
                state.rng_state = capture_rng_state()

                sample_idx = batch_end

                # Checkpoint
                batches_completed += 1
                if batches_completed % checkpoint_interval == 0:
                    self._state_manager.save(state)
                    logger.debug(
                        "Checkpoint: %d feasible / %d total processed",
                        state.feasible_samples_collected,
                        state.total_samples_processed,
                    )

                # Update metrics
                self._update_metrics(all_results, pbar)

            # Final save
            self._state_manager.save(state)
            pbar.close()

            # Summary
            self._print_summary()

            return self._metrics

        finally:
            self._cleanup()

    def _is_theoretically_feasible(
        self, sample: ExplorationSample
    ) -> tuple[bool, float, float, float]:
        """
        Check if a sample is theoretically feasible using NSM bounds.

        A sample is feasible only if BOTH:
          1) Q_channel < Q_storage (NSM constraint)
          2) Q_channel <= 0.22 (Lupo asymptotic bound for positive key rate)

        Parameters
        ----------
        sample : ExplorationSample
            The sample to check.

        Returns
        -------
        tuple[bool, float, float, float]
            (is_feasible, margin, q_channel, q_storage) where margin = Q_storage - Q_channel.
            Negative margin means infeasible.
        """
        # Calculate channel QBER using Erven formula
        q_channel = compute_qber_erven(
            fidelity=sample.channel_fidelity,
            detector_error=sample.detector_error,
            detection_efficiency=sample.detection_efficiency,
            dark_count_prob=sample.dark_count_prob,
        )

        # Calculate storage QBER bound: Q_storage = (1 - r) / 2
        q_storage = (1.0 - sample.storage_noise_r) / 2.0

        # Margin check (with small epsilon for floating point safety)
        margin = q_storage - q_channel
        is_feasible = (margin > 1e-6) and (q_channel <= 0.22)

        return is_feasible, margin, q_channel, q_storage

    def _filter_feasible_batch(
        self,
        samples: List[ExplorationSample],
    ) -> tuple[List[ExplorationSample], List[ProtocolResult]]:
        """
        Filter samples using vectorized Numba feasibility kernel.

        This method converts samples to a Float32 design matrix, runs the
        Numba-accelerated feasibility check, and returns separated lists
        of feasible samples and infeasible results.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Raw samples to filter.

        Returns
        -------
        tuple[List[ExplorationSample], List[ProtocolResult]]
            (feasible_samples, infeasible_results) where infeasible_results
            contain ProtocolResult objects with SKIPPED_INFEASIBLE outcome.

        Performance
        -----------
        Uses SIMD-optimized Numba kernel for ~100x speedup over Python loop.
        Processes ~100k samples in <10ms on modern CPUs.
        """
        if not samples:
            return [], []

        n = len(samples)

        # Build design matrix for Numba kernel (Float32 for SIMD)
        # Columns: [r, log_nu, log_dt, f, log_eta, e_det, log_p_dark, log_n, strategy]
        design_matrix = np.empty((n, 9), dtype=DTYPE_FLOAT)
        for i, s in enumerate(samples):
            design_matrix[i, 0] = s.storage_noise_r
            design_matrix[i, 1] = np.log10(s.storage_rate_nu)
            design_matrix[i, 2] = np.log10(s.wait_time_ns)
            design_matrix[i, 3] = s.channel_fidelity
            design_matrix[i, 4] = np.log10(s.detection_efficiency)
            design_matrix[i, 5] = s.detector_error
            design_matrix[i, 6] = np.log10(s.dark_count_prob)
            design_matrix[i, 7] = np.log10(s.num_pairs)
            design_matrix[i, 8] = 0.0 if s.strategy.value == "baseline" else 1.0

        # Run vectorized feasibility check
        feasibility_mask, q_channel, q_storage, margin, n_feasible = (
            compute_feasibility_vectorized(design_matrix)
        )

        # Separate feasible and infeasible samples
        feasible_samples = []
        infeasible_results = []

        for i, sample in enumerate(samples):
            if feasibility_mask[i]:
                feasible_samples.append(sample)
            else:
                infeasible_results.append(
                    ProtocolResult(
                        sample=sample,
                        outcome=ProtocolOutcome.SKIPPED_INFEASIBLE,
                        net_efficiency=0.0,
                        raw_key_length=0,
                        final_key_length=0,
                        qber_measured=float("nan"),
                        reconciliation_efficiency=0.0,
                        leakage_bits=0,
                        execution_time_seconds=0.0,
                        error_message=(
                            f"Infeasible: margin={margin[i]:.4f}; "
                            f"q_channel={q_channel[i]:.4f}; "
                            f"q_storage={q_storage[i]:.4f}"
                        ),
                        metadata={
                            "infeasibility_margin": float(margin[i]),
                            "q_channel": float(q_channel[i]),
                            "q_storage": float(q_storage[i]),
                        },
                    )
                )

        logger.debug(
            "Feasibility filter: %d/%d samples passed (%.1f%% yield)",
            n_feasible,
            n,
            100.0 * n_feasible / n if n > 0 else 0.0,
        )

        return feasible_samples, infeasible_results

    def _execute_batch(
        self,
        samples: List[ExplorationSample],
        pbar: tqdm,
    ) -> List[ProtocolResult]:
        """
        Execute a batch of feasible samples.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Feasible samples to execute.
        pbar : tqdm
            Progress bar to update.

        Returns
        -------
        List[ProtocolResult]
            Execution results.
        """
        results = []

        # Generate EPR data for all samples
        epr_results = self._harness.generate_epr_batch(samples)

        # Execute protocols
        for sample, epr_result in zip(samples, epr_results):
            if epr_result.is_success():
                result = self._harness.execute(sample, epr_data=epr_result.epr_data)
                # CRITICAL VISIBILITY: Escalate to WARNING to force visibility
                if not result.is_success():
                    logger.warning(
                        "Protocol FAILED | Sample %s | Error: %s | Eff: %.4f | KeyLen: %d | QBER: %.4f", 
                        epr_result.batch_id, 
                        result.error_message,
                        result.net_efficiency,
                        result.final_key_length,
                        result.qber_measured if not np.isnan(result.qber_measured) else -1.0
                    )
            else:
                 # CRITICAL ADDITION: Log why EPR generation failed
                logger.error(
                    "EPR Generation failed for sample (batch=%s): %s",
                    epr_result.batch_id,
                    epr_result.error
                )
                # EPR generation failed
                result = ProtocolResult(
                    sample=sample,
                    outcome=ProtocolOutcome.FAILURE_ERROR,
                    net_efficiency=0.0,
                    raw_key_length=0,
                    final_key_length=0,
                    qber_measured=float("nan"),
                    reconciliation_efficiency=0.0,
                    leakage_bits=0,
                    execution_time_seconds=0.0,
                    error_message=f"EPR generation failed: {epr_result.error}",
                )

            results.append(result)
            pbar.update(1)

        return results

    def _execute_batch_streaming(
        self,
        samples: List[ExplorationSample],
        pbar: tqdm,
    ) -> List[ProtocolResult]:
        """
        Execute a batch using streaming EPR generation.

        This method provides pipelined execution where protocol execution
        can begin as soon as EPR data is available, rather than waiting
        for all EPR generation to complete.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Feasible samples to execute.
        pbar : tqdm
            Progress bar to update.

        Returns
        -------
        List[ProtocolResult]
            Execution results (order may differ from input).
        """
        results = []

        # Get the EPR orchestrator from the harness
        epr_orchestrator = self._harness._epr_orchestrator

        # Stream EPR results and execute protocols immediately
        for epr_result in epr_orchestrator.generate_batch_streaming(samples):
            sample = epr_result.sample

            if epr_result.is_success():
                result = self._harness.execute(sample, epr_data=epr_result.epr_data)
                # CRITICAL VISIBILITY: Escalate to WARNING to force visibility
                if not result.is_success():
                    logger.warning(
                        "Protocol FAILED (streaming) | Sample %s | Error: %s | Eff: %.4f | KeyLen: %d | QBER: %.4f", 
                        epr_result.batch_id, 
                        result.error_message,
                        result.net_efficiency,
                        result.final_key_length,
                        result.qber_measured if not np.isnan(result.qber_measured) else -1.0
                    )
            else:
                logger.error(
                    "EPR Generation failed (streaming) for sample (batch=%s): %s",
                    epr_result.batch_id,
                    epr_result.error
                )
                result = ProtocolResult(
                    sample=sample,
                    outcome=ProtocolOutcome.FAILURE_ERROR,
                    net_efficiency=0.0,
                    raw_key_length=0,
                    final_key_length=0,
                    qber_measured=float("nan"),
                    reconciliation_efficiency=0.0,
                    leakage_bits=0,
                    execution_time_seconds=0.0,
                    error_message=f"EPR generation failed: {epr_result.error}",
                )

            results.append(result)
            pbar.update(1)

            # Explicit memory cleanup
            del epr_result

        return results

    def _update_metrics(
        self,
        results: List[ProtocolResult],
        pbar: tqdm,
    ) -> None:
        """
        Update campaign metrics with feasibility tracking.

        Parameters
        ----------
        results : List[ProtocolResult]
            Batch results (including skipped infeasible samples).
        pbar : tqdm
            Progress bar to update.
        """
        # Separate feasible from infeasible
        skipped = [r for r in results if r.outcome == ProtocolOutcome.SKIPPED_INFEASIBLE]
        executed = [r for r in results if r.outcome != ProtocolOutcome.SKIPPED_INFEASIBLE]
        successes = [r for r in executed if r.is_success()]
        failures = [r for r in executed if not r.is_success()]

        # Update counters
        self._metrics.skipped_infeasible += len(skipped)
        self._metrics.feasible_samples += len(executed)
        self._metrics.total_samples_processed += len(results)
        self._metrics.failed_samples += len(failures)

        # Update yield rate
        if self._metrics.total_samples_processed > 0:
            self._metrics.yield_rate = (
                self._metrics.feasible_samples / self._metrics.total_samples_processed
            )

        # Update success rate (of executed samples)
        total_executed = len(successes) + len(failures)
        if total_executed > 0:
            self._metrics.success_rate = len(successes) / total_executed

        # Update efficiency (rolling average of last 100 successful samples)
        if successes:
            recent_eff = np.mean([r.net_efficiency for r in successes[-100:]])
            self._metrics.current_efficiency = recent_eff

        # Update timing
        self._metrics.elapsed_seconds = time.perf_counter() - self._start_time
        if self._metrics.elapsed_seconds > 0:
            self._metrics.samples_per_second = (
                self._metrics.feasible_samples / self._metrics.elapsed_seconds
            )

        # Update progress bar
        pbar.set_postfix(self._metrics.to_progress_dict())

    def _print_summary(self) -> None:
        """Print campaign summary with feasibility metrics."""
        logger.info("=" * 60)
        logger.info("Phase 1 Campaign Complete")
        logger.info("=" * 60)
        logger.info("Target Feasible:    %d", self._metrics.target_feasible_samples)
        logger.info("Feasible Collected: %d", self._metrics.feasible_samples)
        logger.info("Skipped (Infeas):   %d", self._metrics.skipped_infeasible)
        logger.info("Total Processed:    %d", self._metrics.total_samples_processed)
        logger.info("Yield Rate:         %.1f%%", self._metrics.yield_rate * 100)
        logger.info("Failed Executions:  %d", self._metrics.failed_samples)
        logger.info("Success Rate:       %.1f%%", self._metrics.success_rate * 100)
        logger.info("Avg Efficiency:     %.3f", self._metrics.current_efficiency)
        logger.info("Elapsed Time:       %.1f seconds", self._metrics.elapsed_seconds)
        logger.info("Throughput:         %.1f samples/s", self._metrics.samples_per_second)
        logger.info("Output File:        %s", self.hdf5_path)
        logger.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """
    Command-line entry point for Phase 1 executor.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 1: LHS Warmup Campaign for Caligo Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./exploration_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of LHS samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for execution",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N batches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_tqdm_logging(getattr(logging, args.log_level))

    # Configure execution
    epr_config = None
    if args.workers is not None:
        epr_config = BatchedEPRConfig(max_workers=args.workers)

    # Run executor
    try:
        executor = Phase1Executor(
            output_dir=args.output_dir,
            random_seed=args.seed,
            epr_config=epr_config,
        )
        executor.run(
            num_samples=args.samples,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            verbose=not args.quiet,
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Phase 1 execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

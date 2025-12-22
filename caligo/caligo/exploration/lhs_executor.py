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
from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.types import (
    ExplorationConfig,
    ExplorationSample,
    Phase1State,
    ProtocolOutcome,
    ProtocolResult,
)
from caligo.utils.logging import get_logger

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
    total_samples : int
        Total samples in the campaign.
    completed_samples : int
        Successfully completed samples.
    failed_samples : int
        Failed protocol executions.
    current_efficiency : float
        Rolling average efficiency of recent samples.
    success_rate : float
        Fraction of successful executions.
    elapsed_seconds : float
        Total elapsed time.
    samples_per_second : float
        Throughput rate.
    """

    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    current_efficiency: float = 0.0
    success_rate: float = 0.0
    elapsed_seconds: float = 0.0
    samples_per_second: float = 0.0

    def to_progress_dict(self) -> Dict[str, str]:
        """Convert to tqdm postfix dictionary."""
        return {
            "eff": f"{self.current_efficiency:.3f}",
            "success": f"{self.success_rate:.1%}",
            "rate": f"{self.samples_per_second:.1f}/s",
        }


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

    def run(
        self,
        num_samples: int = 2000,
        batch_size: int = 50,
        checkpoint_interval: int = 5,
        verbose: bool = True,
    ) -> Phase1Metrics:
        """
        Run the Phase 1 LHS campaign.

        Parameters
        ----------
        num_samples : int
            Total number of samples to generate.
        batch_size : int
            Number of samples per batch.
        checkpoint_interval : int
            Save checkpoint every N batches.
        verbose : bool
            Whether to show progress bars.

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

        try:
            # Check for existing checkpoint
            state = self._state_manager.load(Phase1State)
            if state is not None:
                logger.info(
                    "Resuming from checkpoint: %d/%d samples",
                    state.completed_samples,
                    state.total_samples,
                )
                restore_rng_state(state.rng_state)
                start_sample = state.completed_samples
                self._all_samples = self._sampler.generate(num_samples)
            else:
                logger.info("Starting new Phase 1 campaign: %d samples", num_samples)
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                start_sample = 0
                self._all_samples = self._sampler.generate(num_samples)
                state = Phase1State(
                    total_samples=num_samples,
                    completed_samples=0,
                    current_batch_start=0,
                    rng_state=capture_rng_state(),
                    hdf5_path=self.hdf5_path,
                )

            self._metrics.total_samples = num_samples

            # Calculate batches
            num_batches = (num_samples - start_sample + batch_size - 1) // batch_size
            batches_completed = 0

            # Progress bar setup
            pbar = tqdm(
                total=num_samples,
                initial=start_sample,
                desc="Phase 1 (LHS)",
                unit="samples",
                disable=not verbose,
                dynamic_ncols=True,
            )

            # Process batches
            for batch_idx in range(start_sample // batch_size, (num_samples + batch_size - 1) // batch_size):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                batch_samples = self._all_samples[batch_start:batch_end]

                if batch_start < start_sample:
                    continue  # Skip already completed batches

                # Execute batch
                results = self._execute_batch(batch_samples, pbar)

                # Write to HDF5
                inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results)
                self._hdf5_writer.append_batch(
                    GROUP_LHS_WARMUP,
                    inputs=inputs,
                    outputs=outputs,
                    outcomes=outcomes,
                    metadata=metadata,
                )

                # Update state
                state.completed_samples = batch_end
                state.current_batch_start = batch_end
                state.rng_state = capture_rng_state()

                # Checkpoint
                batches_completed += 1
                if batches_completed % checkpoint_interval == 0:
                    self._state_manager.save(state)
                    logger.debug("Checkpoint saved at %d samples", batch_end)

                # Update metrics
                self._update_metrics(results, pbar)

            # Final save
            self._state_manager.save(state)
            pbar.close()

            # Summary
            self._print_summary()

            return self._metrics

        finally:
            self._cleanup()

    def _execute_batch(
        self,
        samples: List[ExplorationSample],
        pbar: tqdm,
    ) -> List[ProtocolResult]:
        """
        Execute a batch of samples.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Samples to execute.
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
            else:
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

    def _update_metrics(
        self,
        results: List[ProtocolResult],
        pbar: tqdm,
    ) -> None:
        """
        Update metrics from batch results.

        Parameters
        ----------
        results : List[ProtocolResult]
            Batch results.
        pbar : tqdm
            Progress bar to update.
        """
        successes = sum(1 for r in results if r.is_success())
        failures = len(results) - successes

        self._metrics.completed_samples += len(results)
        self._metrics.failed_samples += failures

        # Rolling efficiency (last batch)
        efficiencies = [r.net_efficiency for r in results if r.is_success()]
        if efficiencies:
            self._metrics.current_efficiency = np.mean(efficiencies)

        # Overall success rate
        if self._metrics.completed_samples > 0:
            self._metrics.success_rate = (
                (self._metrics.completed_samples - self._metrics.failed_samples)
                / self._metrics.completed_samples
            )

        # Throughput
        elapsed = time.perf_counter() - self._start_time
        self._metrics.elapsed_seconds = elapsed
        if elapsed > 0:
            self._metrics.samples_per_second = self._metrics.completed_samples / elapsed

        # Update progress bar
        pbar.set_postfix(self._metrics.to_progress_dict())

    def _print_summary(self) -> None:
        """Print campaign summary."""
        logger.info("=" * 60)
        logger.info("Phase 1 (LHS) Campaign Complete")
        logger.info("=" * 60)
        logger.info("  Total Samples:      %d", self._metrics.total_samples)
        logger.info("  Completed:          %d", self._metrics.completed_samples)
        logger.info("  Failed:             %d", self._metrics.failed_samples)
        logger.info("  Success Rate:       %.1f%%", self._metrics.success_rate * 100)
        logger.info("  Mean Efficiency:    %.4f", self._metrics.current_efficiency)
        logger.info("  Total Time:         %.1f s", self._metrics.elapsed_seconds)
        logger.info("  Throughput:         %.2f samples/s", self._metrics.samples_per_second)
        logger.info("  Output File:        %s", self.hdf5_path)
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

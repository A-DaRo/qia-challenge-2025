"""
Phase 3 Executor: Active learning stress testing.

This module implements the Phase 3 Bayesian optimization workflow for
active exploration of the security cliff in the QKD parameter space.

Workflow
--------
1. Load Phase 1 data and Phase 2 surrogate model
2. Initialize Bayesian optimizer
3. For each iteration:
   a. Generate batch of candidate points
   b. Execute protocols in parallel
   c. Update surrogate model
   d. Log metrics and checkpoint
4. Identify and characterize security cliff

Features
--------
- **Nested Progress Bars**: Outer (iterations) + inner (batch execution)
- **Adaptive Retraining**: Retrain GP after each batch
- **Cliff Detection**: Automatic identification of security boundary
- **Fault Tolerance**: Full checkpoint/resume support

Usage
-----
Command line:
    $ python -m caligo.exploration.active_executor \\
        --data exploration_data.h5 \\
        --surrogate surrogate.pkl \\
        --iterations 100

Python API:
    >>> executor = Phase3Executor(data_path, surrogate_path)
    >>> executor.run(num_iterations=100, batch_size=16)
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

from caligo.exploration.active import (
    AcquisitionConfig,
    BayesianOptimizer,
)
from caligo.exploration.epr_batcher import BatchedEPRConfig, BatchedEPROrchestrator
from caligo.exploration.harness import HarnessConfig, ProtocolHarness
from caligo.exploration.persistence import (
    GROUP_ACTIVE_LEARNING,
    GROUP_LHS_WARMUP,
    HDF5Writer,
    StateManager,
    capture_rng_state,
    hdf5_arrays_to_training_data,
    restore_rng_state,
    result_to_hdf5_arrays,
)
from caligo.exploration.sampler import ParameterBounds, array_to_samples
from caligo.exploration.surrogate import (
    EfficiencyLandscape,
    GPConfig,
    detect_divergence,
)
from caligo.exploration.types import (
    ExplorationSample,
    Phase3State,
    ProtocolOutcome,
    ProtocolResult,
)
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# TQDM-Compatible Logging
# =============================================================================


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write()."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_tqdm_logging(log_level: int = logging.INFO) -> None:
    """Configure logging to work with tqdm."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


# =============================================================================
# Phase 3 Metrics
# =============================================================================


@dataclass
class Phase3Metrics:
    """
    Metrics tracked during Phase 3 active learning.

    Attributes
    ----------
    iteration : int
        Current iteration number.
    total_iterations : int
        Total planned iterations.
    samples_acquired : int
        Total samples acquired in Phase 3.
    best_cliff_efficiency : float
        Best (closest to zero) efficiency found.
    mean_acquisition : float
        Mean acquisition value of last batch.
    success_rate : float
        Protocol success rate.
    gp_retrain_count : int
        Number of GP retraining rounds.
    divergence_count : int
        Number of divergence events.
    elapsed_seconds : float
        Total elapsed time.
    samples_per_second : float
        Throughput rate.
    """

    iteration: int = 0
    total_iterations: int = 0
    samples_acquired: int = 0
    best_cliff_efficiency: float = 1.0
    mean_acquisition: float = 0.0
    success_rate: float = 0.0
    gp_retrain_count: int = 0
    divergence_count: int = 0
    elapsed_seconds: float = 0.0
    samples_per_second: float = 0.0

    def to_progress_dict(self) -> Dict[str, str]:
        """Convert to tqdm postfix dictionary."""
        return {
            "cliff": f"{self.best_cliff_efficiency:.4f}",
            "acq": f"{self.mean_acquisition:.3f}",
            "success": f"{self.success_rate:.1%}",
            "retrain": str(self.gp_retrain_count),
        }


# =============================================================================
# Phase 3 Executor
# =============================================================================


class Phase3Executor:
    """
    Executor for Phase 3 active learning campaign.

    This class orchestrates the Bayesian optimization loop for
    identifying the security cliff in the QKD parameter space.

    Parameters
    ----------
    data_path : Path
        Path to the HDF5 data file.
    surrogate_path : Path
        Path to the trained surrogate model.
    output_dir : Optional[Path]
        Output directory. If None, uses data_path parent.
    bounds : Optional[ParameterBounds]
        Parameter space bounds.
    acquisition_config : Optional[AcquisitionConfig]
        Acquisition function configuration.
    harness_config : Optional[HarnessConfig]
        Protocol execution configuration.
    epr_config : Optional[BatchedEPRConfig]
        EPR generation configuration.
    random_seed : Optional[int]
        Random seed for reproducibility.

    Attributes
    ----------
    data_path : Path
    surrogate_path : Path
    output_dir : Path
    bounds : ParameterBounds

    Examples
    --------
    >>> executor = Phase3Executor(
    ...     data_path=Path("./exploration_data.h5"),
    ...     surrogate_path=Path("./surrogate.pkl"),
    ... )
    >>> executor.run(num_iterations=100, batch_size=16)
    """

    def __init__(
        self,
        data_path: Path,
        surrogate_path: Path,
        output_dir: Optional[Path] = None,
        bounds: Optional[ParameterBounds] = None,
        acquisition_config: Optional[AcquisitionConfig] = None,
        harness_config: Optional[HarnessConfig] = None,
        epr_config: Optional[BatchedEPRConfig] = None,
        random_seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the Phase 3 executor.

        Parameters
        ----------
        data_path : Path
            HDF5 data file path.
        surrogate_path : Path
            Surrogate model path.
        output_dir : Optional[Path]
            Output directory.
        bounds : Optional[ParameterBounds]
            Parameter bounds.
        acquisition_config : Optional[AcquisitionConfig]
            Acquisition configuration.
        harness_config : Optional[HarnessConfig]
            Harness configuration.
        epr_config : Optional[BatchedEPRConfig]
            EPR configuration.
        random_seed : Optional[int]
            Random seed.
        """
        self.data_path = Path(data_path)
        self.surrogate_path = Path(surrogate_path)
        self.output_dir = Path(output_dir) if output_dir else self.data_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bounds = bounds or ParameterBounds()
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self.harness_config = harness_config or HarnessConfig()
        self.epr_config = epr_config or BatchedEPRConfig()
        self.random_seed = random_seed

        # Components (lazy initialization)
        self._landscape: Optional[EfficiencyLandscape] = None
        self._optimizer: Optional[BayesianOptimizer] = None
        self._harness: Optional[ProtocolHarness] = None
        self._hdf5_writer: Optional[HDF5Writer] = None
        self._state_manager: Optional[StateManager] = None

        self._metrics = Phase3Metrics()
        self._start_time: float = 0.0

        # Training data accumulator
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

        logger.info(
            "Initialized Phase3Executor (data=%s, surrogate=%s)",
            self.data_path,
            self.surrogate_path,
        )

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.output_dir / "phase3_checkpoint.pkl"

    def _init_components(self) -> None:
        """Initialize all components."""
        # Load surrogate
        if self._landscape is None:
            self._landscape = EfficiencyLandscape.load(self.surrogate_path)

        # Load training data
        if self._X_train is None:
            self._load_training_data()

        # Initialize optimizer
        if self._optimizer is None:
            self._optimizer = BayesianOptimizer(
                landscape=self._landscape,
                bounds=self.bounds,
                config=self.acquisition_config,
            )

        # Initialize harness
        if self._harness is None:
            epr_orchestrator = BatchedEPROrchestrator(self.epr_config)
            self._harness = ProtocolHarness(
                config=self.harness_config,
                epr_orchestrator=epr_orchestrator,
            )

        # Initialize HDF5 writer
        if self._hdf5_writer is None:
            self._hdf5_writer = HDF5Writer(self.data_path, mode="a")
            self._hdf5_writer.open()

        # Initialize state manager
        if self._state_manager is None:
            self._state_manager = StateManager(self.checkpoint_path)

    def _load_training_data(self) -> None:
        """Load existing training data from HDF5."""
        with HDF5Writer(self.data_path, mode="r") as reader:
            # Load LHS warmup data
            inputs_lhs, outputs_lhs, outcomes_lhs, _ = reader.read_group(GROUP_LHS_WARMUP)
            X_lhs, y_lhs = hdf5_arrays_to_training_data(inputs_lhs, outputs_lhs, outcomes_lhs)

            # Load any existing active learning data
            try:
                inputs_al, outputs_al, outcomes_al, _ = reader.read_group(GROUP_ACTIVE_LEARNING)
                X_al, y_al = hdf5_arrays_to_training_data(inputs_al, outputs_al, outcomes_al)
            except (KeyError, Exception):
                X_al = np.empty((0, 9))
                y_al = np.empty((0,))

        # Combine
        self._X_train = np.vstack([X_lhs, X_al]) if len(X_al) > 0 else X_lhs
        self._y_train = np.concatenate([y_lhs, y_al]) if len(y_al) > 0 else y_lhs

        logger.info(
            "Loaded %d training samples (%d LHS + %d active)",
            len(self._X_train),
            len(X_lhs),
            len(X_al),
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._harness is not None:
            self._harness.shutdown()
            self._harness = None

        if self._hdf5_writer is not None:
            self._hdf5_writer.close()
            self._hdf5_writer = None

    def _retrain_surrogate(self) -> bool:
        """
        Retrain the surrogate model on accumulated data.

        Returns
        -------
        bool
            True if retraining succeeded.
        """
        try:
            self._landscape = EfficiencyLandscape(config=GPConfig())
            self._landscape.fit(self._X_train, self._y_train)

            # Update optimizer with new landscape
            self._optimizer = BayesianOptimizer(
                landscape=self._landscape,
                bounds=self.bounds,
                config=self.acquisition_config,
            )

            self._metrics.gp_retrain_count += 1
            return True

        except Exception as e:
            logger.error("Surrogate retraining failed: %s", e)
            return False

    def _execute_batch(
        self,
        candidates: np.ndarray,
        inner_pbar: tqdm,
    ) -> List[ProtocolResult]:
        """
        Execute a batch of candidate points.

        Parameters
        ----------
        candidates : np.ndarray
            Candidate points, shape (batch_size, 9).
        inner_pbar : tqdm
            Inner progress bar.

        Returns
        -------
        List[ProtocolResult]
            Execution results.
        """
        # Convert to ExplorationSamples
        samples = array_to_samples(candidates)

        # Generate EPR data
        inner_pbar.set_postfix({"step": "EPR generation"})
        epr_results = self._harness.generate_epr_batch(samples)
        inner_pbar.update(len(samples) // 2)

        # Execute protocols
        inner_pbar.set_postfix({"step": "Protocol execution"})
        results = []
        for sample, epr_result in zip(samples, epr_results):
            if epr_result.is_success():
                result = self._harness.execute(sample, epr_data=epr_result.epr_data)
            else:
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
            inner_pbar.update(1)

        return results

    def run(
        self,
        num_iterations: int = 100,
        batch_size: int = 16,
        retrain_interval: int = 5,
        checkpoint_interval: int = 5,
        verbose: bool = True,
    ) -> Phase3Metrics:
        """
        Run the Phase 3 active learning campaign.

        Parameters
        ----------
        num_iterations : int
            Number of Bayesian optimization iterations.
        batch_size : int
            Number of candidates per iteration.
        retrain_interval : int
            Retrain GP every N iterations.
        checkpoint_interval : int
            Save checkpoint every N iterations.
        verbose : bool
            Whether to show progress bars.

        Returns
        -------
        Phase3Metrics
            Final campaign metrics.
        """
        self._start_time = time.perf_counter()
        self._init_components()

        try:
            # Check for checkpoint
            state = self._state_manager.load(Phase3State)
            if state is not None:
                logger.info(
                    "Resuming from checkpoint: iteration %d",
                    state.iteration,
                )
                start_iteration = state.iteration
                self._metrics.samples_acquired = state.total_active_samples
                self._metrics.best_cliff_efficiency = state.best_cliff_efficiency
            else:
                logger.info(
                    "Starting Phase 3: %d iterations, batch_size=%d",
                    num_iterations,
                    batch_size,
                )
                start_iteration = 0
                state = Phase3State()

            self._metrics.total_iterations = num_iterations

            # Outer progress bar (iterations)
            outer_pbar = tqdm(
                total=num_iterations,
                initial=start_iteration,
                desc="Phase 3 (Active)",
                unit="iter",
                disable=not verbose,
                position=0,
            )

            for iteration in range(start_iteration, num_iterations):
                self._metrics.iteration = iteration

                # Generate candidates
                outer_pbar.set_postfix({"step": "Acquisition opt."})
                candidates = self._optimizer.suggest_batch_diverse(
                    batch_size=batch_size,
                    diversity_weight=0.1,
                )

                # Inner progress bar (batch execution)
                inner_pbar = tqdm(
                    total=batch_size + batch_size // 2,  # EPR + protocol
                    desc=f"  Iter {iteration+1}",
                    unit="samples",
                    disable=not verbose,
                    position=1,
                    leave=False,
                )

                # Execute batch
                results = self._execute_batch(candidates, inner_pbar)
                inner_pbar.close()

                # Write to HDF5
                inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results)
                self._hdf5_writer.append_batch(
                    GROUP_ACTIVE_LEARNING,
                    inputs=inputs,
                    outputs=outputs,
                    outcomes=outcomes,
                    metadata=metadata,
                )

                # Update training data
                efficiencies = outputs[:, 0]
                self._X_train = np.vstack([self._X_train, inputs])
                self._y_train = np.concatenate([self._y_train, efficiencies])

                # Update optimizer
                for x, eff in zip(candidates, efficiencies):
                    self._optimizer.update_best(x, eff)

                # Update metrics
                self._update_metrics(results, outer_pbar)

                # Retrain surrogate
                if (iteration + 1) % retrain_interval == 0:
                    outer_pbar.set_postfix({"step": "Retraining GP"})
                    if not self._retrain_surrogate():
                        self._metrics.divergence_count += 1

                # Checkpoint
                if (iteration + 1) % checkpoint_interval == 0:
                    state.iteration = iteration + 1
                    state.total_active_samples = self._metrics.samples_acquired
                    state.best_cliff_efficiency = self._metrics.best_cliff_efficiency
                    if self._optimizer.best_point is not None:
                        state.best_cliff_point = self._optimizer.best_point.copy()
                    self._state_manager.save(state)

                outer_pbar.update(1)

            # Final save
            state.iteration = num_iterations
            state.total_active_samples = self._metrics.samples_acquired
            state.best_cliff_efficiency = self._metrics.best_cliff_efficiency
            self._state_manager.save(state)

            outer_pbar.close()

            # Save final surrogate
            self._landscape.save(self.output_dir / "surrogate_final.pkl")

            # Print summary
            self._print_summary()

            return self._metrics

        finally:
            self._cleanup()

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
        self._metrics.samples_acquired += len(results)

        # Success rate
        successes = sum(1 for r in results if r.is_success())
        total = self._metrics.samples_acquired
        self._metrics.success_rate = successes / len(results) if results else 0.0

        # Best cliff point
        for result in results:
            eff = result.net_efficiency
            if abs(eff) < abs(self._metrics.best_cliff_efficiency) and eff < 0.5:
                self._metrics.best_cliff_efficiency = eff

        # Mean acquisition
        history = self._optimizer.history
        if history and "mean_acquisition" in history[-1]:
            self._metrics.mean_acquisition = history[-1]["mean_acquisition"]

        # Throughput
        elapsed = time.perf_counter() - self._start_time
        self._metrics.elapsed_seconds = elapsed
        if elapsed > 0:
            self._metrics.samples_per_second = self._metrics.samples_acquired / elapsed

        pbar.set_postfix(self._metrics.to_progress_dict())

    def _print_summary(self) -> None:
        """Print campaign summary."""
        m = self._metrics
        logger.info("=" * 60)
        logger.info("Phase 3 (Active Learning) Complete")
        logger.info("=" * 60)
        logger.info("  Iterations:           %d", m.iteration)
        logger.info("  Samples Acquired:     %d", m.samples_acquired)
        logger.info("  Best Cliff Eff:       %.6f", m.best_cliff_efficiency)
        logger.info("  Success Rate:         %.1f%%", m.success_rate * 100)
        logger.info("  GP Retrains:          %d", m.gp_retrain_count)
        logger.info("  Divergence Events:    %d", m.divergence_count)
        logger.info("  Total Time:           %.1f s", m.elapsed_seconds)
        logger.info("  Throughput:           %.2f samples/s", m.samples_per_second)
        logger.info("=" * 60)

        if self._optimizer.best_point is not None:
            logger.info("Best Cliff Point:")
            logger.info("  %s", self._optimizer.best_point)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """
    Command-line entry point for Phase 3 executor.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 3: Active Learning for Caligo Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to HDF5 data file",
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        required=True,
        help="Path to trained surrogate model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of Bayesian optimization iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per iteration",
    )
    parser.add_argument(
        "--retrain-interval",
        type=int,
        default=5,
        help="Retrain GP every N iterations",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="straddle",
        choices=["straddle", "ei", "ucb"],
        help="Acquisition function type",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.96,
        help="Exploration parameter",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Configure acquisition
    acquisition_config = AcquisitionConfig(
        acquisition_type=args.acquisition,
        kappa=args.kappa,
    )

    # Configure EPR
    epr_config = None
    if args.workers is not None:
        epr_config = BatchedEPRConfig(max_workers=args.workers)

    # Run executor
    try:
        executor = Phase3Executor(
            data_path=args.data,
            surrogate_path=args.surrogate,
            output_dir=args.output_dir,
            acquisition_config=acquisition_config,
            epr_config=epr_config,
            random_seed=args.seed,
        )
        executor.run(
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            retrain_interval=args.retrain_interval,
            checkpoint_interval=args.checkpoint_interval,
            verbose=not args.quiet,
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Phase 3 execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

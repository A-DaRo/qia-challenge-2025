"""
Phase 2 Executor: Surrogate model training.

This module implements the Phase 2 workflow, which trains the Twin Gaussian
Process surrogate model on the Phase 1 LHS warmup data.

Workflow
--------
1. Load Phase 1 data from HDF5
2. Split into training/validation sets
3. Train Twin GPs (baseline + blind)
4. Validate and detect divergence
5. Save trained model

Features
--------
- **Cross-validation**: Optional k-fold CV for hyperparameter selection
- **Divergence Detection**: Automatic detection of model pathologies
- **Incremental Training**: Can retrain on additional data

Usage
-----
Command line:
    $ python -m caligo.exploration.surrogate_trainer --data exploration_data.h5

Python API:
    >>> executor = Phase2Executor(data_path=Path("./exploration_data.h5"))
    >>> landscape = executor.run()
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from caligo.exploration.persistence import (
    GROUP_LHS_WARMUP,
    HDF5Writer,
    StateManager,
    hdf5_arrays_to_training_data,
)
from caligo.exploration.surrogate import (
    EfficiencyLandscape,
    GPConfig,
    detect_divergence,
)
from caligo.exploration.types import Phase2State
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
# Training Metrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """
    Metrics from surrogate model training.

    Attributes
    ----------
    n_samples : int
        Total training samples.
    n_baseline : int
        Baseline strategy samples.
    n_blind : int
        Blind strategy samples.
    train_r2_baseline : float
        R² score on training set (baseline).
    train_r2_blind : float
        R² score on training set (blind).
    val_r2_baseline : float
        R² score on validation set (baseline).
    val_r2_blind : float
        R² score on validation set (blind).
    train_rmse_baseline : float
        RMSE on training set (baseline).
    train_rmse_blind : float
        RMSE on training set (blind).
    val_rmse_baseline : float
        RMSE on validation set (baseline).
    val_rmse_blind : float
        RMSE on validation set (blind).
    training_time_seconds : float
        Total training time.
    divergence_detected : bool
        Whether model divergence was detected.
    """

    n_samples: int = 0
    n_baseline: int = 0
    n_blind: int = 0
    train_r2_baseline: float = 0.0
    train_r2_blind: float = 0.0
    val_r2_baseline: float = 0.0
    val_r2_blind: float = 0.0
    train_rmse_baseline: float = 0.0
    train_rmse_blind: float = 0.0
    val_rmse_baseline: float = 0.0
    val_rmse_blind: float = 0.0
    training_time_seconds: float = 0.0
    divergence_detected: bool = False


# =============================================================================
# Phase 2 Executor
# =============================================================================


class Phase2Executor:
    """
    Executor for Phase 2 surrogate model training.

    This class orchestrates loading Phase 1 data, training the Twin GP
    surrogate model, and validating the results.

    Parameters
    ----------
    data_path : Path
        Path to the HDF5 data file from Phase 1.
    output_dir : Optional[Path]
        Output directory for model files. If None, uses data_path parent.
    gp_config : Optional[GPConfig]
        GP configuration.
    validation_split : float
        Fraction of data for validation (default: 0.2).
    random_seed : Optional[int]
        Random seed for reproducibility.

    Attributes
    ----------
    data_path : Path
    output_dir : Path
    gp_config : GPConfig
    validation_split : float
    random_seed : Optional[int]

    Examples
    --------
    >>> executor = Phase2Executor(
    ...     data_path=Path("./exploration_data.h5"),
    ...     validation_split=0.2,
    ... )
    >>> landscape = executor.run()
    >>> landscape.save(Path("./surrogate.pkl"))
    """

    def __init__(
        self,
        data_path: Path,
        output_dir: Optional[Path] = None,
        gp_config: Optional[GPConfig] = None,
        validation_split: float = 0.2,
        random_seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the Phase 2 executor.

        Parameters
        ----------
        data_path : Path
            Path to HDF5 data file.
        output_dir : Optional[Path]
            Output directory.
        gp_config : Optional[GPConfig]
            GP configuration.
        validation_split : float
            Validation set fraction.
        random_seed : Optional[int]
            Random seed.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else self.data_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gp_config = gp_config or GPConfig()
        self.validation_split = validation_split
        self.random_seed = random_seed

        self._metrics = TrainingMetrics()
        self._landscape: Optional[EfficiencyLandscape] = None

        logger.info(
            "Initialized Phase2Executor (data=%s, val_split=%.1f%%)",
            self.data_path,
            self.validation_split * 100,
        )

    @property
    def surrogate_path(self) -> Path:
        """Path to the saved surrogate model."""
        return self.output_dir / "surrogate.pkl"

    @property
    def checkpoint_path(self) -> Path:
        """Path to the checkpoint file."""
        return self.output_dir / "phase2_checkpoint.pkl"

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from HDF5.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X, y) training data.
        """
        with HDF5Writer(self.data_path, mode="r") as writer:
            inputs, outputs, outcomes, _ = writer.read_group(GROUP_LHS_WARMUP)

        X, y = hdf5_arrays_to_training_data(inputs, outputs, outcomes)

        logger.info("Loaded %d samples from %s", len(X), self.data_path)
        return X, y

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_val, y_train, y_val).
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_seed,
        )
        return X_train, X_val, y_train, y_val

    def _compute_metrics(
        self,
        landscape: EfficiencyLandscape,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingMetrics:
        """
        Compute training and validation metrics.

        Parameters
        ----------
        landscape : EfficiencyLandscape
            Trained model.
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation targets.

        Returns
        -------
        TrainingMetrics
            Computed metrics.
        """
        metrics = TrainingMetrics()

        # Split by strategy
        train_baseline_mask = X_train[:, 8] < 0.5
        train_blind_mask = X_train[:, 8] >= 0.5
        val_baseline_mask = X_val[:, 8] < 0.5
        val_blind_mask = X_val[:, 8] >= 0.5

        metrics.n_samples = len(X_train) + len(X_val)
        metrics.n_baseline = int(np.sum(train_baseline_mask)) + int(np.sum(val_baseline_mask))
        metrics.n_blind = int(np.sum(train_blind_mask)) + int(np.sum(val_blind_mask))

        # Compute predictions (returns tuple of (mean, std))
        y_train_pred, _ = landscape.predict(X_train, return_std=False)
        y_val_pred, _ = landscape.predict(X_val, return_std=False)

        # Training metrics by strategy
        if np.any(train_baseline_mask):
            y_t_base = y_train[train_baseline_mask]
            y_p_base = y_train_pred[train_baseline_mask]
            ss_res = np.sum((y_t_base - y_p_base) ** 2)
            ss_tot = np.sum((y_t_base - np.mean(y_t_base)) ** 2)
            metrics.train_r2_baseline = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics.train_rmse_baseline = np.sqrt(np.mean((y_t_base - y_p_base) ** 2))

        if np.any(train_blind_mask):
            y_t_blind = y_train[train_blind_mask]
            y_p_blind = y_train_pred[train_blind_mask]
            ss_res = np.sum((y_t_blind - y_p_blind) ** 2)
            ss_tot = np.sum((y_t_blind - np.mean(y_t_blind)) ** 2)
            metrics.train_r2_blind = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics.train_rmse_blind = np.sqrt(np.mean((y_t_blind - y_p_blind) ** 2))

        # Validation metrics by strategy
        if np.any(val_baseline_mask):
            y_v_base = y_val[val_baseline_mask]
            y_p_base = y_val_pred[val_baseline_mask]
            ss_res = np.sum((y_v_base - y_p_base) ** 2)
            ss_tot = np.sum((y_v_base - np.mean(y_v_base)) ** 2)
            metrics.val_r2_baseline = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics.val_rmse_baseline = np.sqrt(np.mean((y_v_base - y_p_base) ** 2))

        if np.any(val_blind_mask):
            y_v_blind = y_val[val_blind_mask]
            y_p_blind = y_val_pred[val_blind_mask]
            ss_res = np.sum((y_v_blind - y_p_blind) ** 2)
            ss_tot = np.sum((y_v_blind - np.mean(y_v_blind)) ** 2)
            metrics.val_r2_blind = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics.val_rmse_blind = np.sqrt(np.mean((y_v_blind - y_p_blind) ** 2))

        return metrics

    def run(
        self,
        verbose: bool = True,
        save_model: bool = True,
    ) -> EfficiencyLandscape:
        """
        Run the Phase 2 training workflow.

        Parameters
        ----------
        verbose : bool
            Whether to show progress.
        save_model : bool
            Whether to save the trained model.

        Returns
        -------
        EfficiencyLandscape
            Trained surrogate model.
        """
        start_time = time.perf_counter()

        # Progress bar
        pbar = tqdm(
            total=5,
            desc="Phase 2 (Surrogate)",
            unit="step",
            disable=not verbose,
        )

        try:
            # Step 1: Load data
            pbar.set_postfix({"step": "Loading data"})
            X, y = self._load_data()
            pbar.update(1)

            # Step 2: Split data
            pbar.set_postfix({"step": "Splitting data"})
            X_train, X_val, y_train, y_val = self._split_data(X, y)
            
            # Split by strategy (column 8: 0=BASELINE, 1=BLIND)
            train_baseline_mask = X_train[:, 8] < 0.5
            train_blind_mask = X_train[:, 8] >= 0.5
            
            X_train_baseline = X_train[train_baseline_mask]
            y_train_baseline = y_train[train_baseline_mask]
            X_train_blind = X_train[train_blind_mask]
            y_train_blind = y_train[train_blind_mask]
            
            logger.info(
                "Split data: %d train (%d baseline, %d blind), %d validation",
                len(X_train),
                len(X_train_baseline),
                len(X_train_blind),
                len(X_val),
            )
            pbar.update(1)

            # Step 3: Train model
            pbar.set_postfix({"step": "Training GPs"})
            landscape = EfficiencyLandscape(config=self.gp_config)
            landscape.fit(X_train_baseline, y_train_baseline, X_train_blind, y_train_blind)
            pbar.update(1)

            # Step 4: Compute metrics
            pbar.set_postfix({"step": "Computing metrics"})
            self._metrics = self._compute_metrics(
                landscape, X_train, y_train, X_val, y_val
            )
            self._metrics.training_time_seconds = time.perf_counter() - start_time
            pbar.update(1)

            # Step 5: Divergence detection
            pbar.set_postfix({"step": "Checking divergence"})
            is_diverged, diagnostics = detect_divergence(landscape, X_val)
            self._metrics.divergence_detected = is_diverged
            if is_diverged:
                logger.warning("Model divergence detected: %s", diagnostics)
            pbar.update(1)

            # Save model
            if save_model:
                landscape.save(self.surrogate_path)

            self._landscape = landscape

            # Print summary
            pbar.close()
            self._print_summary()

            return landscape

        except Exception as e:
            pbar.close()
            logger.exception("Phase 2 training failed: %s", e)
            raise

    def run_cross_validation(
        self,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run k-fold cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of CV folds.
        verbose : bool
            Whether to show progress.

        Returns
        -------
        Dict[str, List[float]]
            CV scores for each fold.
        """
        X, y = self._load_data()

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        cv_scores: Dict[str, List[float]] = {
            "r2_baseline": [],
            "r2_blind": [],
            "rmse_baseline": [],
            "rmse_blind": [],
        }

        pbar = tqdm(
            total=n_folds,
            desc="Cross-validation",
            disable=not verbose,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            landscape = EfficiencyLandscape(config=self.gp_config)
            landscape.fit(X_train, y_train)

            metrics = self._compute_metrics(
                landscape, X_train, y_train, X_val, y_val
            )

            cv_scores["r2_baseline"].append(metrics.val_r2_baseline)
            cv_scores["r2_blind"].append(metrics.val_r2_blind)
            cv_scores["rmse_baseline"].append(metrics.val_rmse_baseline)
            cv_scores["rmse_blind"].append(metrics.val_rmse_blind)

            pbar.set_postfix({
                "fold": fold_idx + 1,
                "r2_b": f"{metrics.val_r2_baseline:.3f}",
                "r2_bl": f"{metrics.val_r2_blind:.3f}",
            })
            pbar.update(1)

        pbar.close()

        # Log summary
        logger.info("=" * 50)
        logger.info("Cross-Validation Results (%d folds)", n_folds)
        logger.info("=" * 50)
        logger.info(
            "  R² Baseline:  %.3f ± %.3f",
            np.mean(cv_scores["r2_baseline"]),
            np.std(cv_scores["r2_baseline"]),
        )
        logger.info(
            "  R² Blind:     %.3f ± %.3f",
            np.mean(cv_scores["r2_blind"]),
            np.std(cv_scores["r2_blind"]),
        )
        logger.info(
            "  RMSE Baseline: %.4f ± %.4f",
            np.mean(cv_scores["rmse_baseline"]),
            np.std(cv_scores["rmse_baseline"]),
        )
        logger.info(
            "  RMSE Blind:    %.4f ± %.4f",
            np.mean(cv_scores["rmse_blind"]),
            np.std(cv_scores["rmse_blind"]),
        )
        logger.info("=" * 50)

        return cv_scores

    def _print_summary(self) -> None:
        """Print training summary."""
        m = self._metrics
        logger.info("=" * 60)
        logger.info("Phase 2 (Surrogate Training) Complete")
        logger.info("=" * 60)
        logger.info("  Total Samples:        %d", m.n_samples)
        logger.info("  Baseline Samples:     %d", m.n_baseline)
        logger.info("  Blind Samples:        %d", m.n_blind)
        logger.info("-" * 60)
        logger.info("  Train R² (baseline):  %.4f", m.train_r2_baseline)
        logger.info("  Train R² (blind):     %.4f", m.train_r2_blind)
        logger.info("  Val R² (baseline):    %.4f", m.val_r2_baseline)
        logger.info("  Val R² (blind):       %.4f", m.val_r2_blind)
        logger.info("-" * 60)
        logger.info("  Train RMSE (baseline): %.4f", m.train_rmse_baseline)
        logger.info("  Train RMSE (blind):    %.4f", m.train_rmse_blind)
        logger.info("  Val RMSE (baseline):   %.4f", m.val_rmse_baseline)
        logger.info("  Val RMSE (blind):      %.4f", m.val_rmse_blind)
        logger.info("-" * 60)
        logger.info("  Training Time:        %.1f s", m.training_time_seconds)
        logger.info("  Divergence Detected:  %s", m.divergence_detected)
        logger.info("  Model Path:           %s", self.surrogate_path)
        logger.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """
    Command-line entry point for Phase 2 executor.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Phase 2: Surrogate Model Training for Caligo Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to HDF5 data file from Phase 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for model files",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation set fraction",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="matern52",
        choices=["rbf", "matern32", "matern52"],
        help="GP kernel type",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of optimizer restarts",
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run k-fold cross-validation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds",
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

    # Configure GP
    gp_config = GPConfig(
        kernel_type=args.kernel,
        n_restarts_optimizer=args.n_restarts,
    )

    # Run executor
    try:
        executor = Phase2Executor(
            data_path=args.data,
            output_dir=args.output_dir,
            gp_config=gp_config,
            validation_split=args.val_split,
            random_seed=args.seed,
        )

        if args.cross_validation:
            executor.run_cross_validation(
                n_folds=args.cv_folds,
                verbose=not args.quiet,
            )
        else:
            executor.run(verbose=not args.quiet)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Phase 2 execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

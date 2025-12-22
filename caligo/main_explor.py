#!/usr/bin/env python3
"""
Main exploration script for the Caligo QKD Security Cliff Campaign.

This script orchestrates the complete three-phase exploration pipeline:
- Phase 1: Latin Hypercube Sampling (LHS) warmup
- Phase 2: Surrogate model training (Twin GP)
- Phase 3: Bayesian optimization active learning

Usage
-----
Basic usage with defaults:
    $ python main_explor.py

Custom configuration:
    $ python main_explor.py --config explor_configs/qia_challenge_config.yaml

Override workers:
    $ python main_explor.py --config my_config.yaml --workers 32

Resume from checkpoint:
    $ python main_explor.py --config my_config.yaml --resume

Notes
-----
- TQDM progress bars are integrated into each phase executor
- Results are stored in `exploration_results/<campaign_name>_<timestamp>/`
- Checkpoints enable full fault tolerance and resume capability
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Exploration pipeline components
from caligo.exploration.lhs_executor import Phase1Executor, Phase1Metrics, setup_tqdm_logging
from caligo.exploration.surrogate_trainer import Phase2Executor, TrainingMetrics
from caligo.exploration.active_executor import Phase3Executor, Phase3Metrics
from caligo.exploration.sampler import ParameterBounds
from caligo.exploration.harness import HarnessConfig
from caligo.exploration.epr_batcher import BatchedEPRConfig
from caligo.exploration.surrogate import GPConfig
from caligo.exploration.active import AcquisitionConfig
from caligo.exploration.persistence import (
    GROUP_ACTIVE_LEARNING,
    GROUP_LHS_WARMUP,
    HDF5Writer,
    hdf5_arrays_to_training_data,
)

# Visualization and table utilities
from caligo.vis_tables import (
    generate_all_figures,
    generate_all_tables,
)

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================


@dataclass
class CampaignConfig:
    """
    Unified campaign configuration from YAML.

    Attributes
    ----------
    output_dir : Path
        Base output directory for results.
    phase1_config : Dict[str, Any]
        Phase 1 LHS configuration.
    phase2_config : Dict[str, Any]
        Phase 2 surrogate training configuration.
    phase3_config : Dict[str, Any]
        Phase 3 active learning configuration.
    bounds : ParameterBounds
        Parameter space bounds.
    execution : Dict[str, Any]
        Execution settings (workers, timeout, seed).
    visualization : Dict[str, Any]
        Visualization settings.
    tables : Dict[str, Any]
        Table generation settings.
    """

    output_dir: Path
    phase1_config: Dict[str, Any]
    phase2_config: Dict[str, Any]
    phase3_config: Dict[str, Any]
    bounds: ParameterBounds
    execution: Dict[str, Any]
    visualization: Dict[str, Any]
    tables: Dict[str, Any]


def load_config(config_path: Path, num_workers: int) -> CampaignConfig:
    """
    Load campaign configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file.
    num_workers : int
        Number of parallel workers (overrides config if specified).

    Returns
    -------
    CampaignConfig
        Parsed campaign configuration.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    yaml.YAMLError
        If configuration file is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Output directory with optional timestamp
    output_cfg = raw_config.get("output", {})
    base_dir = Path(output_cfg.get("base_dir", "exploration_results"))
    campaign_name = output_cfg.get("campaign_name", "campaign")

    if output_cfg.get("timestamped", True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"{campaign_name}_{timestamp}"
    else:
        output_dir = base_dir / campaign_name

    # Parameter bounds
    bounds_cfg = raw_config.get("bounds", {})
    
    # Handle storage_rate_nu which can be specified as linear or log
    nu_cfg = bounds_cfg.get("storage_rate_nu", {})
    nu_min = nu_cfg.get("min", 0.001)  # Linear value
    nu_max = nu_cfg.get("max", 1.0)
    
    bounds = ParameterBounds(
        r_min=bounds_cfg.get("storage_noise_r", {}).get("min", 0.0),
        r_max=bounds_cfg.get("storage_noise_r", {}).get("max", 1.0),
        nu_min=nu_min,
        nu_max=nu_max,
        dt_min_log=bounds_cfg.get("wait_time_ns", {}).get("min_log", 5.0),
        dt_max_log=bounds_cfg.get("wait_time_ns", {}).get("max_log", 9.0),
        f_min=bounds_cfg.get("channel_fidelity", {}).get("min", 0.501),
        f_max=bounds_cfg.get("channel_fidelity", {}).get("max", 1.0),
        eta_min_log=bounds_cfg.get("detection_efficiency", {}).get("min_log", -3.0),
        eta_max_log=bounds_cfg.get("detection_efficiency", {}).get("max_log", 0.0),
        e_det_min=bounds_cfg.get("detector_error", {}).get("min", 0.0),
        e_det_max=bounds_cfg.get("detector_error", {}).get("max", 0.1),
        p_dark_min_log=bounds_cfg.get("dark_count_prob", {}).get("min_log", -8.0),
        p_dark_max_log=bounds_cfg.get("dark_count_prob", {}).get("max_log", -3.0),
        n_min_log=bounds_cfg.get("num_pairs", {}).get("min_log", 4.0),
        n_max_log=bounds_cfg.get("num_pairs", {}).get("max_log", 6.0),
    )

    # Execution settings
    exec_cfg = raw_config.get("execution", {})
    execution = {
        "num_workers": num_workers if num_workers > 0 else exec_cfg.get("num_workers", 16),
        "timeout_seconds": exec_cfg.get("timeout_seconds", 300.0),
        "random_seed": exec_cfg.get("random_seed", 42),
        "log_level": exec_cfg.get("log_level", "INFO"),
    }

    return CampaignConfig(
        output_dir=output_dir,
        phase1_config=raw_config.get("phase1", {}),
        phase2_config=raw_config.get("phase2", {}),
        phase3_config=raw_config.get("phase3", {}),
        bounds=bounds,
        execution=execution,
        visualization=raw_config.get("visualization", {}),
        tables=raw_config.get("tables", {}),
    )


# =============================================================================
# Phase Execution
# =============================================================================


def run_phase1(config: CampaignConfig) -> Phase1Metrics:
    """
    Execute Phase 1: LHS Warmup Campaign.

    Parameters
    ----------
    config : CampaignConfig
        Campaign configuration.

    Returns
    -------
    Phase1Metrics
        Phase 1 execution metrics.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Latin Hypercube Sampling Warmup")
    logger.info("=" * 70)

    phase1_cfg = config.phase1_config
    exec_cfg = config.execution

    # Build component configs
    harness_config = HarnessConfig(
        timeout_seconds=exec_cfg["timeout_seconds"],
    )

    epr_config = BatchedEPRConfig(
        max_workers=exec_cfg["num_workers"],
        timeout_seconds=exec_cfg["timeout_seconds"],
    )

    # Create executor
    executor = Phase1Executor(
        output_dir=config.output_dir,
        bounds=config.bounds,
        random_seed=exec_cfg["random_seed"],
        harness_config=harness_config,
        epr_config=epr_config,
    )

    # Run campaign
    metrics = executor.run(
        num_samples=phase1_cfg.get("num_samples", 2000),
        batch_size=phase1_cfg.get("batch_size", 50),
        checkpoint_interval=phase1_cfg.get("checkpoint_interval", 5),
        verbose=True,
    )

    logger.info("-" * 70)
    logger.info("Phase 1 Complete:")
    logger.info(f"  Total Samples: {metrics.total_samples}")
    logger.info(f"  Success Rate: {metrics.success_rate:.1%}")
    logger.info(f"  Mean Efficiency: {metrics.current_efficiency:.4f}")
    logger.info(f"  Duration: {metrics.elapsed_seconds:.1f}s")
    logger.info("-" * 70)

    return metrics


def run_phase2(config: CampaignConfig) -> TrainingMetrics:
    """
    Execute Phase 2: Surrogate Model Training.

    Parameters
    ----------
    config : CampaignConfig
        Campaign configuration.

    Returns
    -------
    TrainingMetrics
        Phase 2 training metrics.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Surrogate Model Training")
    logger.info("=" * 70)

    phase2_cfg = config.phase2_config
    exec_cfg = config.execution

    # Build GP config
    gp_settings = phase2_cfg.get("gp", {})
    gp_config = GPConfig(
        n_restarts_optimizer=gp_settings.get("n_restarts_optimizer", 10),
        normalize_y=gp_settings.get("normalize_y", True),
        random_state=exec_cfg["random_seed"],
    )

    # Create executor
    data_path = config.output_dir / "exploration_data.h5"
    executor = Phase2Executor(
        data_path=data_path,
        output_dir=config.output_dir,
        gp_config=gp_config,
        validation_split=phase2_cfg.get("validation_split", 0.2),
        random_seed=exec_cfg["random_seed"],
    )

    # Run training
    landscape = executor.run(verbose=True)

    # Get metrics
    metrics = executor._metrics

    logger.info("-" * 70)
    logger.info("Phase 2 Complete:")
    logger.info(f"  Training Samples: {metrics.n_samples}")
    logger.info(f"  Baseline R²: {metrics.val_r2_baseline:.4f}")
    logger.info(f"  Blind R²: {metrics.val_r2_blind:.4f}")
    logger.info(f"  Duration: {metrics.training_time_seconds:.1f}s")
    logger.info("-" * 70)

    return metrics


def run_phase3(config: CampaignConfig) -> Phase3Metrics:
    """
    Execute Phase 3: Bayesian Optimization Active Learning.

    Parameters
    ----------
    config : CampaignConfig
        Campaign configuration.

    Returns
    -------
    Phase3Metrics
        Phase 3 execution metrics.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: Bayesian Optimization Active Learning")
    logger.info("=" * 70)

    phase3_cfg = config.phase3_config
    exec_cfg = config.execution

    # Build acquisition config
    acq_settings = phase3_cfg.get("acquisition", {})
    acquisition_config = AcquisitionConfig(
        acquisition_type=acq_settings.get("type", "straddle"),
        kappa=acq_settings.get("kappa", 1.96),
    )

    # Build component configs
    harness_config = HarnessConfig(
        timeout_seconds=exec_cfg["timeout_seconds"],
    )

    epr_config = BatchedEPRConfig(
        max_workers=exec_cfg["num_workers"],
        timeout_seconds=exec_cfg["timeout_seconds"],
    )

    # Create executor
    data_path = config.output_dir / "exploration_data.h5"
    surrogate_path = config.output_dir / "surrogate.pkl"

    executor = Phase3Executor(
        data_path=data_path,
        surrogate_path=surrogate_path,
        output_dir=config.output_dir,
        bounds=config.bounds,
        acquisition_config=acquisition_config,
        harness_config=harness_config,
        epr_config=epr_config,
        random_seed=exec_cfg["random_seed"],
    )

    # Run optimization
    metrics = executor.run(
        num_iterations=phase3_cfg.get("num_iterations", 100),
        batch_size=phase3_cfg.get("batch_size", 16),
        retrain_interval=phase3_cfg.get("retrain_interval", 5),
        checkpoint_interval=phase3_cfg.get("checkpoint_interval", 5),
        verbose=True,
    )

    logger.info("-" * 70)
    logger.info("Phase 3 Complete:")
    logger.info(f"  Iterations: {metrics.iteration}/{metrics.total_iterations}")
    logger.info(f"  Samples Acquired: {metrics.samples_acquired}")
    logger.info(f"  Best Cliff Efficiency: {metrics.best_cliff_efficiency:.4f}")
    logger.info(f"  GP Retraining: {metrics.gp_retrain_count} rounds")
    logger.info(f"  Duration: {metrics.elapsed_seconds:.1f}s")
    logger.info("-" * 70)

    return metrics


# =============================================================================
# Post-Processing: Visualization and Tables
# =============================================================================


def load_exploration_data(config: CampaignConfig) -> tuple:
    """
    Load all exploration data from HDF5.

    Parameters
    ----------
    config : CampaignConfig
        Campaign configuration.

    Returns
    -------
    tuple
        (X, y) arrays for visualization and analysis.
    """
    data_path = config.output_dir / "exploration_data.h5"

    with HDF5Writer(data_path, mode="r") as reader:
        # Load LHS warmup data
        inputs_lhs, outputs_lhs, outcomes_lhs, _ = reader.read_group(GROUP_LHS_WARMUP)
        X_lhs, y_lhs = hdf5_arrays_to_training_data(inputs_lhs, outputs_lhs, outcomes_lhs)

        # Load active learning data if available
        try:
            inputs_al, outputs_al, outcomes_al, _ = reader.read_group(GROUP_ACTIVE_LEARNING)
            X_al, y_al = hdf5_arrays_to_training_data(inputs_al, outputs_al, outcomes_al)
        except (KeyError, Exception):
            X_al = np.empty((0, 9))
            y_al = np.empty((0,))

    # Combine all data
    X = np.vstack([X_lhs, X_al]) if len(X_al) > 0 else X_lhs
    y = np.concatenate([y_lhs, y_al]) if len(y_al) > 0 else y_lhs

    logger.info(f"Loaded {len(X)} total samples for post-processing")
    return X, y


def run_post_processing(
    config: CampaignConfig,
    phase_metrics: Dict[str, Any],
) -> None:
    """
    Generate visualizations and tables from exploration results.

    Parameters
    ----------
    config : CampaignConfig
        Campaign configuration.
    phase_metrics : Dict[str, Any]
        Metrics from each phase.
    """
    logger.info("=" * 70)
    logger.info("POST-PROCESSING: Generating Visualizations and Tables")
    logger.info("=" * 70)

    # Load exploration data
    X, y = load_exploration_data(config)

    # Generate visualizations
    vis_cfg = config.visualization
    if vis_cfg.get("enabled", True):
        logger.info("Generating visualizations...")
        try:
            figures = generate_all_figures(
                X=X,
                y=y,
                output_path=config.output_dir,
                phase_metrics=phase_metrics,
                show=False,
            )
            logger.info(f"  Generated {len(figures)} figures")
        except ImportError as e:
            logger.warning(f"Visualization skipped: {e}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    # Generate tables
    table_cfg = config.tables
    if table_cfg.get("enabled", True):
        logger.info("Generating result tables...")
        try:
            tables = generate_all_tables(
                X=X,
                y=y,
                output_path=config.output_dir,
                config={
                    "formats": table_cfg.get("formats", ["csv", "markdown"]),
                    "qber_levels": table_cfg.get("qber_levels", [0.01, 0.03, 0.05, 0.08, 0.10]),
                    "storage_noise_levels": table_cfg.get("storage_noise_levels", [0.90, 0.85, 0.80, 0.75]),
                },
            )
            logger.info(f"  Generated {len(tables)} tables")
        except Exception as e:
            logger.error(f"Table generation failed: {e}")

    logger.info("-" * 70)
    logger.info("Post-processing complete")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("-" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """
    Main entry point for the exploration campaign.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Caligo QKD Security Cliff Exploration Campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_explor.py
  python main_explor.py --config explor_configs/qia_challenge_config.yaml
  python main_explor.py --workers 32 --skip-phase3
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("explor_configs/qia_challenge_config.yaml"),
        help="Path to YAML configuration file (default: explor_configs/qia_challenge_config.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16, overrides config)",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (assumes warmup data exists)",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (assumes surrogate model exists)",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 (only run warmup and training)",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip visualization and table generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_tqdm_logging(log_level)

    # Configure logging: only allow logs from caligo.exploration at given level, silence others to WARNING
    def _silence_external_info_logs(log_level: int = logging.INFO):
        """Allow logs only from caligo.exploration at `log_level`, set all other loggers to WARNING.

        This ensures only logs originating from `caligo.exploration` are shown at
        INFO/DEBUG level, while all other modules are silenced to WARNING.
        """
        # Set root logger to WARNING to silence other modules by default
        logging.getLogger().setLevel(logging.WARNING)

        # Ensure existing loggers are set appropriately
        logger_dict = logging.Logger.manager.loggerDict
        for name, logger_obj in logger_dict.items():
            # Skip placeholders in loggerDict
            if not isinstance(logger_obj, logging.Logger):
                continue
            if name.startswith("caligo.exploration"):
                logger_obj.setLevel(log_level)
            else:
                logger_obj.setLevel(logging.WARNING)

        # Also explicitly set commonly noisy external modules to WARNING for robustness
        noisy_modules = ("squidasm", "netsquid", "netqasm")
        for m in noisy_modules:
            try:
                logging.getLogger(m).setLevel(logging.WARNING)
            except Exception:
                pass

        # Try to set SquidASM internal log manager to WARNING if available
        try:
            from squidasm.sim.stack.common import LogManager as SquidLogManager  # type: ignore
            SquidLogManager.set_log_level("WARNING")
        except Exception:
            # ignore failures to be robust in environments without squidasm
            pass

    _silence_external_info_logs(log_level)

    # Banner
    logger.info("=" * 70)
    logger.info("   CALIGO EXPLORATION SUITE")
    logger.info("   QKD Security Cliff Detection Campaign")
    logger.info("=" * 70)

    start_time = time.perf_counter()

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config, args.workers)

        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Workers: {config.execution['num_workers']}")
        logger.info(f"Random seed: {config.execution['random_seed']}")
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Track phase metrics for post-processing
        phase_metrics = {}

        # Phase 1: LHS Warmup
        if not args.skip_phase1:
            metrics1 = run_phase1(config)
            phase_metrics["phase1"] = {
                "total_samples": metrics1.total_samples,
                "successful_samples": metrics1.completed_samples - metrics1.failed_samples,
                "success_rate": metrics1.success_rate,
                "elapsed_seconds": metrics1.elapsed_seconds,
            }
        else:
            logger.info("Skipping Phase 1 (--skip-phase1)")

        # Phase 2: Surrogate Training
        if not args.skip_phase2:
            metrics2 = run_phase2(config)
            phase_metrics["phase2"] = {
                "n_samples": metrics2.n_samples,
                "r2_score": (metrics2.val_r2_baseline + metrics2.val_r2_blind) / 2,
                "rmse": (metrics2.val_rmse_baseline + metrics2.val_rmse_blind) / 2,
                "training_time": metrics2.training_time_seconds,
            }
        else:
            logger.info("Skipping Phase 2 (--skip-phase2)")

        # Phase 3: Active Learning
        if not args.skip_phase3:
            metrics3 = run_phase3(config)
            phase_metrics["phase3"] = {
                "iterations": list(range(metrics3.iteration)),
                "best_y_history": [metrics3.best_cliff_efficiency],  # Simplified
                "samples_acquired": metrics3.samples_acquired,
                "best_cliff": metrics3.best_cliff_efficiency,
            }
        else:
            logger.info("Skipping Phase 3 (--skip-phase3)")

        # Post-processing
        if not args.skip_postprocess:
            run_post_processing(config, phase_metrics)
        else:
            logger.info("Skipping post-processing (--skip-postprocess)")

        # Final summary
        elapsed = time.perf_counter() - start_time
        logger.info("=" * 70)
        logger.info("   CAMPAIGN COMPLETE")
        logger.info(f"   Total Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"   Results: {config.output_dir}")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Campaign interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Campaign failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Generate LDPC matrices using PEG for all configured rates.

This script generates LDPC matrices for all configured code rates using the
Progressive Edge-Growth (PEG) algorithm. Matrices are saved to the ldpc_matrices/
directory for use by the reconciliation protocol.

This is an **offline tool** — matrices should be generated once and distributed
with the package, not regenerated at runtime.

Usage
-----
    python -m caligo.scripts.generate_ldpc_matrices [--log-show] [--log-level LEVEL]

Options
-------
    --log-show              Display logs in terminal (default: file only)
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
    --output-dir PATH       Custom output directory for matrices

Examples
--------
    # Generate matrices (silent terminal, logs to file only)
    python -m caligo.scripts.generate_ldpc_matrices

    # Generate with terminal output
    python -m caligo.scripts.generate_ldpc_matrices --log-show

    # Generate with debug output in terminal
    python -m caligo.scripts.generate_ldpc_matrices --log-show --log-level DEBUG

Log files are always created in ./logs/generate_ldpc_matrices.log
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp
import yaml

from caligo.reconciliation import constants
from caligo.scripts.peg_generator import DegreeDistribution, PEGMatrixGenerator
from caligo.utils.logging import setup_script_logging


def _load_degree_distributions() -> Dict[float, Dict[str, DegreeDistribution]]:
    """
    Load degree distributions from YAML configuration.

    Returns
    -------
    Dict[float, Dict[str, DegreeDistribution]]
        Mapping from code rate to lambda/rho distributions.

    Raises
    ------
    FileNotFoundError
        If the degree distributions YAML file is missing.
    """
    yaml_path = constants.LDPC_DEGREE_DISTRIBUTIONS_PATH
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Degree distributions file not found: {yaml_path}"
        )

    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)

    distributions: Dict[float, Dict[str, DegreeDistribution]] = {}
    for rate_str, dist_data in raw_data.items():
        rate = float(rate_str)
        distributions[rate] = {
            "lambda": DegreeDistribution(
                degrees=dist_data["lambda"]["degrees"],
                probabilities=dist_data["lambda"]["probabilities"],
            ),
            "rho": DegreeDistribution(
                degrees=dist_data["rho"]["degrees"],
                probabilities=dist_data["rho"]["probabilities"],
            ),
        }

    return distributions


def _get_distributions(
    rate: float,
    all_distributions: Dict[float, Dict[str, DegreeDistribution]],
    logger,
) -> Dict[str, DegreeDistribution]:
    """
    Get degree distributions for a specific code rate.

    Parameters
    ----------
    rate : float
        Target code rate.
    all_distributions : dict
        All loaded distributions from YAML.
    logger : logging.Logger
        Logger instance for warnings.

    Returns
    -------
    Dict[str, DegreeDistribution]
        Lambda and rho distributions for the rate.
    """
    if rate in all_distributions:
        dist = all_distributions[rate]
        logger.debug(
            "Loaded distributions for rate %.2f: VN degrees=%s, CN degrees=%s",
            rate,
            dist["lambda"].degrees,
            dist["rho"].degrees,
        )
        return dist

    # Fallback to base distribution if specific rate not provided
    fallback_rate = 0.50
    logger.warning(
        "Rate %.2f not found in degree distributions. Falling back to rate=%.2f",
        rate,
        fallback_rate,
    )
    return all_distributions[fallback_rate]


def generate_all(
    output_dir: Path,
    logger,
    frame_size: int = constants.LDPC_FRAME_SIZE,
    rates: Optional[List[float]] = None,
) -> None:
    """
    Generate LDPC matrices for the provided `rates` and `frame_size`.

    Parameters
    ----------
    output_dir : Path
        Directory to save generated matrices.
    logger : logging.Logger
        Logger instance for progress reporting.
    frame_size : int
        Number of variable nodes (codeword length).
    rates : list of float or None
        List of code rates to generate. If None, uses configured rates from
        `constants.LDPC_CODE_RATES`.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Output directory: %s", output_dir.resolve())

    rates_list = list(rates) if rates is not None else list(constants.LDPC_CODE_RATES)
    logger.info(
        "Generating matrices for %d rates: %s",
        len(rates_list),
        rates_list,
    )
    logger.info("-" * 70)

    # Load all distributions upfront
    all_distributions = _load_degree_distributions()
    logger.info("Loaded degree distributions from %s", constants.LDPC_DEGREE_DISTRIBUTIONS_PATH)

    total_start = time.perf_counter()
    results: List[Tuple[float, Tuple[int, int], int, float]] = []

    for idx, rate in enumerate(rates_list, 1):
        start = time.perf_counter()
        logger.info(
            "[%d/%d] Starting generation for rate=%.2f",
            idx,
            len(rates_list),
            rate,
        )

        # Load degree distributions
        dists = _get_distributions(rate, all_distributions, logger)

        # Create generator
        m = int(frame_size * (1 - rate))
        logger.debug(
            "Initializing PEG generator (n=%d, m=%d, max_depth=%d, seed=%d)",
            frame_size,
            m,
            constants.PEG_MAX_TREE_DEPTH,
            constants.PEG_DEFAULT_SEED,
        )

        generator = PEGMatrixGenerator(
            n=frame_size,
            rate=rate,
            lambda_dist=dists["lambda"],
            rho_dist=dists["rho"],
            max_tree_depth=constants.PEG_MAX_TREE_DEPTH,
            seed=constants.PEG_DEFAULT_SEED,
        )
        # Generate matrix
        logger.info(
            "[%d/%d] Running PEG algorithm...", idx, len(rates_list)
        )
        H = generator.generate()

        # Save to file
        filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
            frame_size=frame_size, rate=rate
        )
        path = output_dir / filename
        logger.debug(
            "[%d/%d] Saving matrix to %s", idx, len(rates_list), path.name
        )
        sp.save_npz(path, H)

        elapsed = time.perf_counter() - start
        logger.info(
            "[%d/%d] ✓ Generated %s (shape=%s, nnz=%d, density=%.4f) in %.2fs",
            idx,
            len(rates_list),
            filename,
            H.shape,
            H.nnz,
            H.nnz / (H.shape[0] * H.shape[1]),
            elapsed,
        )
        results.append((rate, H.shape, H.nnz, elapsed))
        logger.info("-" * 70)

    total_elapsed = time.perf_counter() - total_start

    # Summary statistics
    logger.info("=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total matrices generated: %d", len(results))
    logger.info(
        "Total time: %.2fs (avg: %.2fs per matrix)",
        total_elapsed,
        total_elapsed / len(results),
    )
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("")
    logger.info("Matrix Summary:")
    logger.info(
        "  %-10s %-15s %-10s %-10s %-10s",
        "Rate",
        "Shape",
        "NNZ",
        "Density",
        "Time (s)",
    )
    logger.info("-" * 70)
    for rate, shape, nnz, elapsed in results:
        density = nnz / (shape[0] * shape[1])
        logger.info(
            "  %-10.2f %-15s %-10d %-10.4f %-10.2f",
            rate,
            str(shape),
            nnz,
            density,
            elapsed,
        )
    logger.info("=" * 70)


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate LDPC matrices using PEG algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate matrices (silent terminal, logs to file only)
  python -m caligo.scripts.generate_ldpc_matrices

  # Generate with terminal output
  python -m caligo.scripts.generate_ldpc_matrices --log-show

  # Generate with debug output in terminal
  python -m caligo.scripts.generate_ldpc_matrices --log-show --log-level DEBUG

  # Generate for specific frame size and rates
  python -m caligo.scripts.generate_ldpc_matrices --frame-size 16 --rates 0.5 0.75

Log files are always created in ./logs/generate_ldpc_matrices.log
        """
    )
    parser.add_argument(
        "--log-show",
        action="store_true",
        help="Display logs in terminal (default: file only)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for matrices (default: caligo/configs/ldpc_matrices/)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=None,
        help="Frame size (n) for generated matrices. If omitted uses configured default.",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help="One or more code rates to generate (e.g. --rates 0.5 0.75). If omitted all configured rates are used.",
    )

    args = parser.parse_args()

    # Basic validation
    if args.frame_size is not None and args.frame_size <= 0:
        parser.error("--frame-size must be a positive integer")
    if args.rates is not None and any((r <= 0.0 or r >= 1.0) for r in args.rates):
        parser.error("--rates values must be floats in the interval (0, 1)")

    # Initialize logging
    logger = setup_script_logging(
        script_name="generate_ldpc_matrices",
        log_level=args.log_level,
        show_terminal=args.log_show,
    )

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = constants.LDPC_MATRICES_DIR

    # Use provided frame size or fallback to configured constant
    frame_size = args.frame_size if args.frame_size is not None else constants.LDPC_FRAME_SIZE
    rates = args.rates if args.rates is not None else list(constants.LDPC_CODE_RATES)

    try:
        logger.info("LDPC Matrix Generation Starting")
        logger.info("Configuration:")
        logger.info("  Frame size: %d", frame_size)
        logger.info("  Code rates: %s", rates)
        logger.info("  PEG max tree depth: %d", constants.PEG_MAX_TREE_DEPTH)
        logger.info("  PEG seed: %d", constants.PEG_DEFAULT_SEED)
        logger.info("=" * 70)

        generate_all(output_dir, logger, frame_size=frame_size, rates=rates)

        logger.info("Matrix generation completed successfully.")
        logger.info("All matrices available in: %s", output_dir.resolve())
    except Exception as e:
        logger.exception("Matrix generation failed: %s", e)
        raise


if __name__ == "__main__":
    main()

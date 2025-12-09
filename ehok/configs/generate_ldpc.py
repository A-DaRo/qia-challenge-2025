"""Generate LDPC matrices using PEG for all configured rates.

This script generates LDPC matrices for all configured code rates using the
Progressive Edge-Growth (PEG) algorithm. Matrices are saved to the ldpc_matrices/
directory for use by the reconciliation protocol.

Usage:
    python -m ehok.configs.generate_ldpc [--log-show] [--log-level LEVEL]

Options:
    --log-show              Display logs in terminal (default: file only)
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

from pathlib import Path
from typing import Dict
import time
import argparse

import scipy.sparse as sp

from ehok.core import constants
from ehok.implementations.reconciliation import DegreeDistribution, PEGMatrixGenerator
from ehok.utils.logging import setup_script_logging

# Logger will be initialized in main()


def _get_distributions(rate: float, logger) -> Dict[str, DegreeDistribution]:
    """Load degree distributions for a specific code rate."""
    dist = constants.LDPC_DEGREE_DISTRIBUTIONS.get(rate)
    if dist is None:
        # Fallback to base distribution if specific rate not provided
        dist = constants.LDPC_DEGREE_DISTRIBUTIONS[0.50]
        logger.warning(
            "Rate %.2f not found in degree distributions. Falling back to rate=0.50",
            rate
        )
    
    lambda_dist = DegreeDistribution(
        degrees=dist["lambda"]["degrees"], probabilities=dist["lambda"]["probabilities"]
    )
    rho_dist = DegreeDistribution(
        degrees=dist["rho"]["degrees"], probabilities=dist["rho"]["probabilities"]
    )
    
    logger.debug(
        "Loaded distributions for rate %.2f: VN degrees=%s, CN degrees=%s",
        rate, lambda_dist.degrees, rho_dist.degrees
    )
    return {"lambda": lambda_dist, "rho": rho_dist}


def generate_all(output_dir: Path, logger) -> None:
    """Generate all LDPC matrices for configured rates."""
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("Generating matrices for %d rates: %s", 
                len(constants.LDPC_CODE_RATES), 
                list(constants.LDPC_CODE_RATES))
    logger.info("-" * 70)
    
    total_start = time.perf_counter()
    results = []
    
    for idx, rate in enumerate(constants.LDPC_CODE_RATES, 1):
        start = time.perf_counter()
        logger.info("[%d/%d] Starting generation for rate=%.2f", 
                    idx, len(constants.LDPC_CODE_RATES), rate)
        
        # Load degree distributions
        dists = _get_distributions(rate, logger)
        
        # Create generator
        logger.debug("Initializing PEG generator (n=%d, m=%d, max_depth=%d, seed=%d)",
                     constants.LDPC_FRAME_SIZE,
                     int(constants.LDPC_FRAME_SIZE * (1 - rate)),
                     constants.PEG_MAX_TREE_DEPTH,
                     constants.PEG_DEFAULT_SEED)
        
        generator = PEGMatrixGenerator(
            n=constants.LDPC_FRAME_SIZE,
            rate=rate,
            lambda_dist=dists["lambda"],
            rho_dist=dists["rho"],
            max_tree_depth=constants.PEG_MAX_TREE_DEPTH,
            seed=constants.PEG_DEFAULT_SEED,
        )
        
        # Generate matrix
        logger.info("[%d/%d] Running PEG algorithm...", idx, len(constants.LDPC_CODE_RATES))
        H = generator.generate()
        
        # Save to file
        filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
            frame_size=constants.LDPC_FRAME_SIZE, rate=rate
        )
        path = output_dir / filename
        logger.debug("[%d/%d] Saving matrix to %s", idx, len(constants.LDPC_CODE_RATES), path.name)
        sp.save_npz(path, H)
        
        elapsed = time.perf_counter() - start
        logger.info(
            "[%d/%d] ✓ Generated %s (shape=%s, nnz=%d, density=%.4f) in %.2fs",
            idx,
            len(constants.LDPC_CODE_RATES),
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
    logger.info("Total time: %.2fs (avg: %.2fs per matrix)", 
                total_elapsed, total_elapsed / len(results))
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("")
    logger.info("Matrix Summary:")
    logger.info("  %-10s %-15s %-10s %-10s %-10s", "Rate", "Shape", "NNZ", "Density", "Time (s)")
    logger.info("-" * 70)
    for rate, shape, nnz, elapsed in results:
        density = nnz / (shape[0] * shape[1])
        logger.info("  %-10.2f %-15s %-10d %-10.4f %-10.2f", 
                    rate, str(shape), nnz, density, elapsed)
    logger.info("=" * 70)


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate LDPC matrices using PEG algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate matrices (silent terminal, logs to file only)
  python -m ehok.configs.generate_ldpc
  
  # Generate with terminal output
  python -m ehok.configs.generate_ldpc --log-show
  
  # Generate with debug output in terminal
  python -m ehok.configs.generate_ldpc --log-show --log-level DEBUG

Log files are always created in ./logs/generate_ldpc.log
        """
    )
    parser.add_argument(
        "--log-show",
        action="store_true",
        help="Display logs in terminal (default: file only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for matrices (default: ./ehok/configs/ldpc_matrices/)"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = setup_script_logging(
        script_name="generate_ldpc",
        log_level=args.log_level,
        show_terminal=args.log_show
    )
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(__file__).parent / "ldpc_matrices"
    
    try:
        logger.info("LDPC Matrix Generation Starting")
        logger.info("Configuration:")
        logger.info("  Frame size: %d", constants.LDPC_FRAME_SIZE)
        logger.info("  Code rates: %s", list(constants.LDPC_CODE_RATES))
        logger.info("  PEG max tree depth: %d", constants.PEG_MAX_TREE_DEPTH)
        logger.info("  PEG seed: %d", constants.PEG_DEFAULT_SEED)
        logger.info("=" * 70)
        
        generate_all(output_dir, logger)
        
        logger.info("Matrix generation completed successfully.")
        logger.info("All matrices available in: %s", output_dir.resolve())
        
    except Exception as e:
        logger.exception("Matrix generation failed: %s", e)
        raise


if __name__ == "__main__":
    main()

"""Run the complete E-HOK baseline protocol.

This script demonstrates the full 5-phase E-HOK protocol execution between
Alice and Bob using SquidASM simulation.

Usage:
    python -m ehok.examples.run_baseline [--log-show] [--log-level LEVEL] [OPTIONS]

Options:
    --log-show              Display logs in terminal (default: file only)
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
    --num-pairs N           Number of EPR pairs to generate
    --config PATH           Path to network configuration YAML file
"""

import logging
from pathlib import Path

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from ehok.protocols.alice import AliceEHOKProgram
from ehok.protocols.bob import BobEHOKProgram
from ehok.core.constants import TOTAL_EPR_PAIRS
from ehok.utils.logging import setup_script_logging, get_logger


# Logger will be initialized in main with proper configuration


def run_ehok_baseline(
    num_pairs: int = TOTAL_EPR_PAIRS,
    network_config_path: str = None,
    logger=None
):
    """
    Execute the E-HOK baseline protocol.
    
    Parameters
    ----------
    num_pairs : int
        Number of EPR pairs to generate.
    network_config_path : str, optional
        Path to network configuration YAML file. 
        If None, uses default baseline configuration.
    logger : logging.Logger, optional
        Logger instance. If None, uses module logger.
    
    Returns
    -------
    tuple
        (alice_results, bob_results) - Protocol execution results.
    """
    if logger is None:
        logger = get_logger("run_baseline")
    
    # Load network configuration
    if network_config_path is None:
        config_dir = Path(__file__).parent.parent / "configs"
        network_config_path = config_dir / "network_baseline.yaml"
    
    logger.info("Loading network configuration from: %s", network_config_path)
    network_cfg = StackNetworkConfig.from_file(str(network_config_path))
    
    # Create program instances
    logger.debug("Initializing protocol programs (Alice and Bob)")
    alice_program = AliceEHOKProgram(total_pairs=num_pairs)
    bob_program = BobEHOKProgram(total_pairs=num_pairs)
    
    # Run protocol
    logger.info("=" * 70)
    logger.info("E-HOK Baseline Protocol Execution")
    logger.info("=" * 70)
    logger.info("EPR pairs: %d", num_pairs)
    logger.info("Network config: %s", network_config_path)
    logger.info("=" * 70)
    
    logger.info("Starting protocol execution...")
    alice_results, bob_results = run(
        config=network_cfg,
        programs={"alice": alice_program, "bob": bob_program},
        num_times=1
    )
    logger.info("Protocol execution completed.")
    
    return alice_results, bob_results


def print_results(alice_results, bob_results, logger=None):
    """
    Print protocol execution results.
    
    Parameters
    ----------
    alice_results : list
        Alice's protocol results.
    bob_results : list
        Bob's protocol results.
    logger : logging.Logger, optional
        Logger instance.
    """
    if logger is None:
        logger = get_logger("run_baseline")
        
    logger.info("=" * 70)
    logger.info("Protocol Results")
    logger.info("=" * 70)
    
    for run_idx, (alice_result, bob_result) in enumerate(zip(alice_results, bob_results)):
        logger.info("Run %d:", run_idx + 1)
        logger.info("  %s", "-" * 66)
        
        # Alice's results
        logger.info("  Alice:")
        logger.info("    Success: %s", alice_result.get("success", False))
        logger.info("    Raw bits generated: %s", alice_result.get("raw_count", "N/A"))
        logger.info("    Sifted bits: %s", alice_result.get("sifted_count", "N/A"))
        logger.info(
            "    Final key length: %s bits",
            alice_result.get("final_count", "N/A"),
        )
        logger.info("    QBER: %.4f%%", alice_result.get("qber", 0) * 100)
        
        # Bob's results
        logger.info("  Bob:")
        logger.info("    Success: %s", bob_result.get("success", False))
        logger.info("    Raw bits generated: %s", bob_result.get("raw_count", "N/A"))
        logger.info("    Sifted bits: %s", bob_result.get("sifted_count", "N/A"))
        logger.info(
            "    Final key length: %s bits",
            bob_result.get("final_count", "N/A"),
        )
        logger.info("    QBER: %.4f%%", bob_result.get("qber", 0) * 100)
        
        # Verify keys match
        alice_key = alice_result.get('oblivious_key')
        bob_key = bob_result.get('oblivious_key')
        
        if alice_key and bob_key:
            import numpy as np
            keys_match = np.array_equal(alice_key.key_value, bob_key.key_value)
            logger.info("  Key Verification:")
            logger.info("    Keys match: %s", keys_match)
            logger.info(
                "    Alice knows: %d bits",
                np.sum(alice_key.knowledge_mask == 0),
            )
            logger.info(
                "    Bob knows: %d bits",
                np.sum(bob_key.knowledge_mask == 0),
            )
            logger.info(
                "    Bob unknown (oblivious): %d bits",
                np.sum(bob_key.knowledge_mask == 1),
            )
            
            if keys_match:
                logger.info("    ✓ Protocol SUCCESSFUL!")
            else:
                logger.error("    ✗ Key mismatch - protocol FAILED")
        logger.info("  %s", "-" * 66)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run E-HOK baseline protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters (silent terminal)
  python -m ehok.examples.run_baseline
  
  # Run with terminal output
  python -m ehok.examples.run_baseline --log-show
  
  # Run with debug output
  python -m ehok.examples.run_baseline --log-show --log-level DEBUG
  
  # Run with custom EPR pairs
  python -m ehok.examples.run_baseline --log-show --num-pairs 5000

Log files are always created in ./logs/run_baseline.log
        """
    )
    parser.add_argument(
        "--num-pairs", 
        type=int, 
        default=TOTAL_EPR_PAIRS,
        help=f"Number of EPR pairs to generate (default: {TOTAL_EPR_PAIRS})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to network configuration YAML file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-show",
        action="store_true",
        help="Display logs in terminal (default: file only)"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = setup_script_logging(
        script_name="run_baseline",
        log_level=args.log_level,
        show_terminal=args.log_show
    )
    
    try:
        logger.info("Starting E-HOK Baseline Protocol")
        logger.info("Configuration:")
        logger.info("  EPR pairs: %d", args.num_pairs)
        logger.info("  Network config: %s", args.config or "default (network_baseline.yaml)")
        logger.info("=" * 70)
        
        alice_results, bob_results = run_ehok_baseline(
            num_pairs=args.num_pairs,
            network_config_path=args.config,
            logger=logger
        )
        
        print_results(alice_results, bob_results, logger)
        
        logger.info("=" * 70)
        logger.info("E-HOK protocol execution completed successfully.")
        logger.info("=" * 70)

    except Exception as e:  # pragma: no cover - CLI failure path
        logger.exception("Protocol execution failed: %s", e)
        import traceback
        traceback.print_exc()
        exit(1)

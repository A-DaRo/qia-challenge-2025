"""
Run the complete E-HOK baseline protocol.

This script demonstrates the full 5-phase E-HOK protocol execution between
Alice and Bob using SquidASM simulation.
"""

import logging
from pathlib import Path

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from ehok.protocols.alice import AliceEHOKProgram
from ehok.protocols.bob import BobEHOKProgram
from ehok.core.constants import TOTAL_EPR_PAIRS
from ehok.utils import get_logger


logger = get_logger(__name__)


def run_ehok_baseline(
    num_pairs: int = TOTAL_EPR_PAIRS,
    network_config_path: str = None,
    log_level: int = logging.INFO
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
    log_level : int
        Logging level (logging.DEBUG, logging.INFO, etc.).
    
    Returns
    -------
    tuple
        (alice_results, bob_results) - Protocol execution results.
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load network configuration
    if network_config_path is None:
        config_dir = Path(__file__).parent.parent / "configs"
        network_config_path = config_dir / "network_baseline.yaml"
    
    network_cfg = StackNetworkConfig.from_file(str(network_config_path))
    
    # Create program instances
    alice_program = AliceEHOKProgram(total_pairs=num_pairs)
    bob_program = BobEHOKProgram(total_pairs=num_pairs)
    
    # Run protocol
    logger.info("%s", "=" * 60)
    logger.info("E-HOK Baseline Protocol Execution")
    logger.info("%s", "=" * 60)
    logger.info("EPR pairs: %s", num_pairs)
    logger.info("Network config: %s", network_config_path)
    logger.info("%s", "=" * 60)
    
    alice_results, bob_results = run(
        config=network_cfg,
        programs={"alice": alice_program, "bob": bob_program},
        num_times=1
    )
    
    return alice_results, bob_results


def print_results(alice_results, bob_results):
    """
    Print protocol execution results.
    
    Parameters
    ----------
    alice_results : list
        Alice's protocol results.
    bob_results : list
        Bob's protocol results.
    """
    logger.info("%s", "=" * 60)
    logger.info("Protocol Results")
    logger.info("%s", "=" * 60)
    
    for run_idx, (alice_result, bob_result) in enumerate(zip(alice_results, bob_results)):
        logger.info("Run %s:", run_idx + 1)
        logger.info("  %s", "-" * 56)
        
        # Alice's results
        logger.info("  Alice:")
        logger.info("    Success: %s", alice_result.get("success", False))
        logger.info("    Raw bits generated: %s", alice_result.get("raw_count", "N/A"))
        logger.info("    Sifted bits: %s", alice_result.get("sifted_count", "N/A"))
        logger.info(
            "    Final key length: %s bits",
            alice_result.get("final_count", "N/A"),
        )
        logger.info("    QBER: %.2f%%", alice_result.get("qber", 0) * 100)
        
        # Bob's results
        logger.info("  Bob:")
        logger.info("    Success: %s", bob_result.get("success", False))
        logger.info("    Raw bits generated: %s", bob_result.get("raw_count", "N/A"))
        logger.info("    Sifted bits: %s", bob_result.get("sifted_count", "N/A"))
        logger.info(
            "    Final key length: %s bits",
            bob_result.get("final_count", "N/A"),
        )
        logger.info("    QBER: %.2f%%", bob_result.get("qber", 0) * 100)
        
        # Verify keys match
        alice_key = alice_result.get('oblivious_key')
        bob_key = bob_result.get('oblivious_key')
        
        if alice_key and bob_key:
            import numpy as np
            keys_match = np.array_equal(alice_key.key_value, bob_key.key_value)
            logger.info("  Key Verification:")
            logger.info("    Keys match: %s", keys_match)
            logger.info(
                "    Alice knows: %s bits",
                np.sum(alice_key.knowledge_mask == 0),
            )
            logger.info(
                "    Bob knows: %s bits",
                np.sum(bob_key.knowledge_mask == 0),
            )
            logger.info(
                "    Bob unknown (oblivious): %s bits",
                np.sum(bob_key.knowledge_mask == 1),
            )
            
            if keys_match:
                logger.info("    Protocol successful!")
            else:
                logger.info("    Key mismatch - protocol failed")
        logger.info("  %s", "-" * 56)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E-HOK baseline protocol")
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
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level)
    
    try:
        alice_results, bob_results = run_ehok_baseline(
            num_pairs=args.num_pairs,
            network_config_path=args.config,
            log_level=log_level
        )
        
        print_results(alice_results, bob_results)

    except Exception as e:  # pragma: no cover - CLI failure path
        logger.exception("Protocol execution failed: %s", e)
        import traceback

        traceback.print_exc()
        exit(1)

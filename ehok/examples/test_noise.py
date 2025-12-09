"""Quick test to verify EPR noise is working.

This minimal test generates EPR pairs with a noisy channel and checks
if we observe the expected error rate. Useful for validating network
configuration and noise model parameters.

Usage:
    python -m ehok.examples.test_noise [--log-show] [--log-level LEVEL]

Options:
    --log-show              Display logs in terminal (default: file only)
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
    --num-pairs N           Number of EPR pairs to test (default: 1000)
"""

import numpy as np
from netqasm.sdk import EPRSocket, Qubit
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from pathlib import Path

from ehok.utils.logging import setup_script_logging, get_logger

class AliceTestProgram(Program):
    """Alice generates EPR pairs and measures in Z basis."""
    
    def __init__(self, num_pairs: int = 1000, logger=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.logger = logger or get_logger("test_noise.alice")
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_test",
            csockets=["bob"],
            epr_sockets=["bob"],
            max_qubits=2,
        )
    
    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets["bob"]
        connection = context.connection
        csocket = context.csockets["bob"]
        
        self.logger.info("Alice: Generating %d EPR pairs...", self.num_pairs)
        outcomes_alice = []
        
        for i in range(self.num_pairs):
            if (i + 1) % 200 == 0:
                self.logger.debug("Alice: Generated %d/%d EPR pairs", i + 1, self.num_pairs)
            
            # Create EPR pair
            q = epr_socket.create_keep()[0]
            
            # Measure in Z basis (no rotation)
            m = q.measure()
            yield from connection.flush()
            outcomes_alice.append(int(m))
        
        self.logger.info("Alice: Sending outcomes to Bob...")
        # Send outcomes to Bob
        for outcome in outcomes_alice:
            csocket.send(str(outcome))
        
        self.logger.info("Alice: Test complete.")
        return {"outcomes": outcomes_alice}


class BobTestProgram(Program):
    """Bob receives EPR pairs and measures in Z basis."""
    
    def __init__(self, num_pairs: int = 1000, logger=None):
        super().__init__()
        self.num_pairs = num_pairs
        self.logger = logger or get_logger("test_noise.bob")
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_test",
            csockets=["alice"],
            epr_sockets=["alice"],
            max_qubits=2,
        )
    
    def run(self, context: ProgramContext):
        epr_socket = context.epr_sockets["alice"]
        connection = context.connection
        csocket = context.csockets["alice"]
        
        self.logger.info("Bob: Receiving %d EPR pairs...", self.num_pairs)
        outcomes_bob = []
        
        for i in range(self.num_pairs):
            if (i + 1) % 200 == 0:
                self.logger.debug("Bob: Received %d/%d EPR pairs", i + 1, self.num_pairs)
            
            # Receive EPR pair
            q = epr_socket.recv_keep()[0]
            
            # Measure in Z basis (no rotation)
            m = q.measure()
            yield from connection.flush()
            outcomes_bob.append(int(m))
        
        # Receive Alice's outcomes
        self.logger.info("Bob: Receiving Alice's measurement outcomes...")
        outcomes_alice = []
        for _ in range(self.num_pairs):
            msg = yield from csocket.recv()
            outcomes_alice.append(int(msg))
        
        # Calculate error rate
        outcomes_alice = np.array(outcomes_alice)
        outcomes_bob = np.array(outcomes_bob)
        errors = outcomes_alice != outcomes_bob
        error_count = np.sum(errors)
        error_rate = error_count / self.num_pairs
        
        # Analyze error distribution
        error_positions = np.where(errors)[0]
        if len(error_positions) > 0:
            avg_gap = np.mean(np.diff(error_positions)) if len(error_positions) > 1 else 0
        else:
            avg_gap = 0
        
        # Log detailed results
        self.logger.info("=" * 70)
        self.logger.info("EPR Noise Test Results")
        self.logger.info("=" * 70)
        self.logger.info("Total pairs tested: %d", self.num_pairs)
        self.logger.info("Errors detected: %d", error_count)
        self.logger.info("Error rate: %.4f%% (%.4f)", error_rate * 100, error_rate)
        self.logger.info("Expected (fidelity=0.97): ~2.25%% (0.0225)")
        
        # Error analysis
        if error_count > 0:
            self.logger.info("Error Analysis:")
            self.logger.info("  First error at position: %d", error_positions[0])
            self.logger.info("  Last error at position: %d", error_positions[-1])
            if len(error_positions) > 1:
                self.logger.info("  Average gap between errors: %.2f", avg_gap)
                self.logger.info("  Min gap: %d", np.min(np.diff(error_positions)))
                self.logger.info("  Max gap: %d", np.max(np.diff(error_positions)))
        
        # Verdict
        expected_error_rate = 0.0225
        tolerance = 0.01  # 1% tolerance
        deviation = abs(error_rate - expected_error_rate)
        
        if deviation < tolerance:
            self.logger.info("Verdict: ✓ PASS (within %.1f%% of expected)", tolerance * 100)
        else:
            self.logger.warning(
                "Verdict: ✗ FAIL (deviation %.4f%% exceeds tolerance %.1f%%)",
                deviation * 100, tolerance * 100
            )
        
        self.logger.info("=" * 70)
        
        return {
            "outcomes": outcomes_bob,
            "error_rate": error_rate,
            "error_count": error_count,
            "pass": deviation < tolerance
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test EPR pair noise model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test with default parameters (silent terminal)
  python -m ehok.examples.test_noise
  
  # Run with terminal output
  python -m ehok.examples.test_noise --log-show
  
  # Run with debug output and more pairs
  python -m ehok.examples.test_noise --log-show --log-level DEBUG --num-pairs 5000

Log files are always created in ./logs/test_noise.log
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
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=1000,
        help="Number of EPR pairs to test (default: 1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to network configuration YAML file (default: ehok/configs/network_baseline.yaml)"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = setup_script_logging(
        script_name="test_noise",
        log_level=args.log_level,
        show_terminal=args.log_show
    )
    
    try:
        logger.info("Starting EPR Noise Test")
        logger.info("Configuration:")
        logger.info("  Test pairs: %d", args.num_pairs)
        logger.info("  Expected fidelity: 0.97 (2.25%% error rate)")
        logger.info("=" * 70)
        
        # Load network configuration
        if args.config:
            config_path = args.config
        else:
            config_path = Path(__file__).parent.parent / "configs" / "network_baseline.yaml"
        
        logger.info("Loading network configuration: %s", config_path)
        cfg = StackNetworkConfig.from_file(str(config_path))
        
        # Create programs
        alice_program = AliceTestProgram(num_pairs=args.num_pairs, logger=logger)
        bob_program = BobTestProgram(num_pairs=args.num_pairs, logger=logger)
        
        logger.info("Running test...")
        alice_results, bob_results = run(
            config=cfg,
            programs={"alice": alice_program, "bob": bob_program},
            num_times=1,
        )
        
        # Final summary
        error_rate = bob_results[0]['error_rate']
        passed = bob_results[0]['pass']
        
        logger.info("=" * 70)
        logger.info("Test Complete")
        logger.info("Final error rate: %.4f%% (%.4f)", error_rate * 100, error_rate)
        logger.info("Test result: %s", "PASS" if passed else "FAIL")
        logger.info("=" * 70)
        
        exit(0 if passed else 1)
        
    except Exception as e:
        logger.exception("Test failed with exception: %s", e)
        import traceback
        traceback.print_exc()
        exit(2)

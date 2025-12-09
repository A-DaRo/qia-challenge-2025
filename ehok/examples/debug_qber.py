"""
Debug script to trace QBER calculation in E-HOK protocol.
"""
import logging

import numpy as np

from ehok.core.sifting import SiftingManager
from ehok.utils import get_logger


logger = get_logger(__name__)

# Simulate scenario with noise
logging.basicConfig(level=logging.INFO, format="%(message)s")

np.random.seed(42)

n = 1000
# Alice's outcomes (random)
outcomes_alice = np.random.randint(0, 2, n, dtype=np.uint8)

# Bob's outcomes - identical to Alice's but with some errors
outcomes_bob = outcomes_alice.copy()
error_rate = 0.02  # 2% errors
n_errors = int(n * error_rate)
error_positions = np.random.choice(n, n_errors, replace=False)
outcomes_bob[error_positions] ^= 1  # Flip bits at error positions

logger.info("Generated %s measurements", n)
logger.info("Introduced %s errors (%.2f%%)", n_errors, error_rate * 100)
logger.info("Verification: %s mismatches", np.sum(outcomes_alice != outcomes_bob))

# Generate random bases
bases_alice = np.random.randint(0, 2, n, dtype=np.uint8)
bases_bob = np.random.randint(0, 2, n, dtype=np.uint8)

# Sifting
sifting = SiftingManager()
I_0, I_1 = sifting.identify_matching_bases(bases_alice, bases_bob)

logger.info("Sifting results:")
logger.info("  Matched bases (I_0): %s (%.1f%%)", len(I_0), len(I_0) / n * 100)
logger.info("  Mismatched bases (I_1): %s (%.1f%%)", len(I_1), len(I_1) / n * 100)

# Select test set
test_set, key_set = sifting.select_test_set(I_0)

logger.info("Test set selection:")
logger.info("  Test set size: %s", len(test_set))
logger.info("  Key set size: %s", len(key_set))

# Estimate QBER
qber = sifting.estimate_qber(outcomes_alice, outcomes_bob, test_set)

logger.info("QBER on test set: %.2f%%", qber * 100)

# Debug: check test set contents
logger.info("Debug test set:")
logger.info("  First 10 test indices: %s", test_set[:10])
logger.info("  First 10 I_0 indices: %s", I_0[:10])
logger.info("  Are test indices in I_0? %s", np.all(np.isin(test_set, I_0)))

# Check actual errors in test set
alice_test = outcomes_alice[test_set]
bob_test = outcomes_bob[test_set]
test_errors = np.sum(alice_test != bob_test)
logger.info("  Manual count of errors in test set: %s/%s", test_errors, len(test_set))

# Check errors in matched vs mismatched bases
matched_errors = np.sum(outcomes_alice[I_0] != outcomes_bob[I_0])
mismatched_errors = np.sum(outcomes_alice[I_1] != outcomes_bob[I_1])

logger.info(
	"Error distribution:\n  Errors in matched bases: %s/%s (%.2f%%)",
	matched_errors,
	len(I_0),
	matched_errors / len(I_0) * 100,
)
logger.info(
	"  Errors in mismatched bases: %s/%s (%.2f%%)",
	mismatched_errors,
	len(I_1),
	mismatched_errors / len(I_1) * 100,
)

# Find where errors are in I_0
error_positions_in_I_0 = np.where(outcomes_alice[I_0] != outcomes_bob[I_0])[0]
actual_error_indices = I_0[error_positions_in_I_0]
logger.info("Actual error positions in I_0: %s", actual_error_indices)
logger.info("Test set: %s", test_set)
logger.info("Overlap: %s", np.intersect1d(test_set, actual_error_indices))

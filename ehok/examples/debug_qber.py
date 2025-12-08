"""
Debug script to trace QBER calculation in E-HOK protocol.
"""
import numpy as np
from ehok.core.sifting import SiftingManager

# Simulate scenario with noise
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

print(f"Generated {n} measurements")
print(f"Introduced {n_errors} errors ({error_rate*100}%)")
print(f"Verification: {np.sum(outcomes_alice != outcomes_bob)} mismatches")

# Generate random bases
bases_alice = np.random.randint(0, 2, n, dtype=np.uint8)
bases_bob = np.random.randint(0, 2, n, dtype=np.uint8)

# Sifting
sifting = SiftingManager()
I_0, I_1 = sifting.identify_matching_bases(bases_alice, bases_bob)

print(f"\nSifting results:")
print(f"  Matched bases (I_0): {len(I_0)} ({len(I_0)/n*100:.1f}%)")
print(f"  Mismatched bases (I_1): {len(I_1)} ({len(I_1)/n*100:.1f}%)")

# Select test set
test_set, key_set = sifting.select_test_set(I_0)

print(f"\nTest set selection:")
print(f"  Test set size: {len(test_set)}")
print(f"  Key set size: {len(key_set)}")

# Estimate QBER
qber = sifting.estimate_qber(outcomes_alice, outcomes_bob, test_set)

print(f"\nQBER on test set: {qber*100:.2f}%")

# Debug: check test set contents
print(f"\nDebug test set:")
print(f"  First 10 test indices: {test_set[:10]}")
print(f"  First 10 I_0 indices: {I_0[:10]}")
print(f"  Are test indices in I_0? {np.all(np.isin(test_set, I_0))}")

# Check actual errors in test set
alice_test = outcomes_alice[test_set]
bob_test = outcomes_bob[test_set]
test_errors = np.sum(alice_test != bob_test)
print(f"  Manual count of errors in test set: {test_errors}/{len(test_set)}")

# Check errors in matched vs mismatched bases
matched_errors = np.sum(outcomes_alice[I_0] != outcomes_bob[I_0])
mismatched_errors = np.sum(outcomes_alice[I_1] != outcomes_bob[I_1])

print(f"\nError distribution:")
print(f"  Errors in matched bases: {matched_errors}/{len(I_0)} ({matched_errors/len(I_0)*100:.2f}%)")
print(f"  Errors in mismatched bases: {mismatched_errors}/{len(I_1)} ({mismatched_errors/len(I_1)*100:.2f}%)")

# Find where errors are in I_0
error_positions_in_I_0 = np.where(outcomes_alice[I_0] != outcomes_bob[I_0])[0]
actual_error_indices = I_0[error_positions_in_I_0]
print(f"\nActual error positions in I_0: {actual_error_indices}")
print(f"Test set: {test_set}")
print(f"Overlap: {np.intersect1d(test_set, actual_error_indices)}")

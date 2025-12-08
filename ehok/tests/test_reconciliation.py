"""
Tests for Phase 4: Information Reconciliation.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from ehok.implementations.reconciliation.ldpc_reconciliator import LDPCReconciliator

# --- Test Case 6.1: Syndrome Computation ---

def test_syndrome_computation():
    """
    Test ID: test_reconciliation::test_syndrome_computation
    Requirement: Syndrome must satisfy S = H * k mod 2.
    """
    # Simple parity-check matrix: 3x6
    # H = [
    #   [1, 1, 0, 1, 0, 0],
    #   [0, 1, 1, 0, 1, 0],
    #   [1, 0, 1, 0, 0, 1]
    # ]
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    indices = np.array([0, 1, 3, 1, 2, 4, 0, 2, 5])
    indptr = np.array([0, 3, 6, 9])
    H = sp.csr_matrix((data, indices, indptr), shape=(3, 6))
    
    reconciliator = LDPCReconciliator(H)
    key = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    
    # Expected Syndrome (manual calculation):
    # S[0] = (1*1 + 1*0 + 0*1 + 1*1 + 0*0 + 0*1) mod 2 = (1 + 1) mod 2 = 0
    # S[1] = (0*1 + 1*0 + 1*1 + 0*1 + 1*0 + 0*1) mod 2 = (1) mod 2 = 1
    # S[2] = (1*1 + 0*0 + 1*1 + 0*1 + 0*0 + 1*1) mod 2 = (1 + 1 + 1) mod 2 = 1
    expected_syndrome = np.array([0, 1, 1], dtype=np.uint8)
    
    syndrome = reconciliator.compute_syndrome(key)
    
    np.testing.assert_array_equal(syndrome, expected_syndrome, 
                                  err_msg="Computed syndrome does not match expected syndrome")

# --- Test Case 6.2: Error Correction ---

def load_ldpc_matrix(filename):
    path = Path(__file__).parents[1] / "configs" / "ldpc_matrices" / filename
    if not path.exists():
        pytest.skip(f"LDPC matrix file not found: {path}")
    return sp.load_npz(path)

def test_error_correction_success():
    """
    Test ID: test_reconciliation::test_error_correction (Success Case)
    Requirement: LDPC BP decoder must correct errors below code capacity.
    """
    # Load regular LDPC matrix with rate 0.5
    # We use the 5000 bit matrix which should have better properties
    H = load_ldpc_matrix("ldpc_5000_rate05.npz")
    reconciliator = LDPCReconciliator(H)
    
    n = H.shape[1]
    np.random.seed(42)
    alice_key = np.random.randint(0, 2, size=n, dtype=np.uint8)
    
    # Introduce 2% errors (realistic for baseline with simple codes)
    num_errors = int(0.02 * n)
    error_positions = np.random.choice(n, size=num_errors, replace=False)
    bob_key = alice_key.copy()
    bob_key[error_positions] ^= 1 # Flip bits
    
    # Alice computes syndrome
    syndrome = reconciliator.compute_syndrome(alice_key)
    
    # Bob reconciles
    bob_corrected_key = reconciliator.reconcile(bob_key, syndrome)
    
    # Verification
    np.testing.assert_array_equal(bob_corrected_key, alice_key,
                                  err_msg="Reconciliation failed to correct errors")

def test_error_correction_failure():
    """
    Test ID: test_reconciliation::test_error_correction (Failure Case)
    Requirement: Decoder should fail (not converge to correct key) at high QBER.
    """
    H = load_ldpc_matrix("ldpc_1000_rate05.npz")
    reconciliator = LDPCReconciliator(H)
    
    n = H.shape[1]
    np.random.seed(43) # Different seed
    alice_key = np.random.randint(0, 2, size=n, dtype=np.uint8)
    
    # Introduce 15% errors (likely above capacity for this simple code/decoder)
    num_errors = int(0.15 * n)
    error_positions = np.random.choice(n, size=num_errors, replace=False)
    bob_key = alice_key.copy()
    bob_key[error_positions] ^= 1
    
    syndrome = reconciliator.compute_syndrome(alice_key)
    
    bob_corrected_key = reconciliator.reconcile(bob_key, syndrome)
    
    # We expect it to NOT match perfectly
    # Or at least have some errors remaining
    errors_remaining = np.sum(bob_corrected_key != alice_key)
    assert errors_remaining > 0, "Decoder should have failed at high error rate"

# --- Test Case 6.3: Leakage Estimation ---

def test_leakage_estimation():
    """
    Test ID: test_reconciliation::test_leakage_estimation
    Requirement: Information leakage must be bounded.
    """
    # We don't need a real matrix for this test, just the class
    # But the class requires a matrix in __init__. 
    # Let's create a dummy one.
    H = sp.csr_matrix((500, 1000), dtype=np.uint8)
    reconciliator = LDPCReconciliator(H)
    
    m = 500
    qber = 0.05
    
    leakage = reconciliator.estimate_leakage(m, qber)
    
    # Acceptance Criterion: m <= leakage <= m * 1.2 + margin
    # My implementation uses: leakage = syndrome_length + 100
    # So leakage = 500 + 100 = 600.
    
    assert leakage >= m, "Leakage cannot be less than syndrome length"
    # The implementation adds a fixed margin of 100. 
    # 600 <= 500 * 1.2 = 600. It's exactly on the border if we use 1.2 factor.
    # Let's be slightly more lenient or check the specific logic.
    # Logic: actual_leakage = syndrome_length + 100
    
    expected_leakage = m + 100
    assert leakage == expected_leakage

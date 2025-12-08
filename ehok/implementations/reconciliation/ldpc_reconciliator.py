"""
LDPC-based information reconciliation.

Uses scipy.sparse for matrix operations and custom BP decoder.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
from ehok.interfaces.reconciliation import IReconciliator
from ehok.core.constants import LDPC_MAX_ITERATIONS, LDPC_BP_THRESHOLD
from ehok.utils.logging import get_logger

logger = get_logger("ldpc_reconciliation")

class LDPCReconciliator(IReconciliator):
    """LDPC-based error correction."""
    
    def __init__(self, parity_check_matrix: sp.spmatrix):
        """
        Initialize with parity check matrix.
        
        Parameters
        ----------
        parity_check_matrix : scipy.sparse matrix
            H matrix, shape (m, n), GF(2).
        """
        self.H = parity_check_matrix.astype(np.uint8)
        self.m, self.n = self.H.shape
        # Estimate column weight (assuming regular or near-regular)
        # We take the max column weight to be safe
        self.w_c = self.H.getnnz(axis=0).max()
        logger.info(f"LDPC: H shape=({self.m}, {self.n}), rate≈{1-self.m/self.n:.2f}, w_c={self.w_c}")
    
    def compute_syndrome(self, key: np.ndarray) -> np.ndarray:
        """Compute syndrome S = H @ key mod 2."""
        # Ensure key is a column vector or 1D array
        if key.ndim == 1:
            key_vec = key
        else:
            key_vec = key.flatten()
            
        if len(key_vec) != self.n:
             raise ValueError(f"Key length {len(key_vec)} does not match LDPC code length {self.n}")

        syndrome = (self.H @ key_vec) % 2
        logger.debug(f"Syndrome computed: {np.sum(syndrome)} non-zero entries")
        return syndrome
    
    def reconcile(self, key: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode using Belief Propagation.
        
        Algorithm: Sum-Product on Tanner graph.
        """
        # Initialize with Bob's noisy key
        decoded = key.copy().astype(np.uint8)
        
        if len(decoded) != self.n:
             raise ValueError(f"Key length {len(decoded)} does not match LDPC code length {self.n}")

        # BP decoder (simplified bit-flipping for hard decision)
        # Note: A full BP decoder would use log-likelihood ratios (LLRs).
        # For the baseline, we implement a hard-decision bit-flipping algorithm 
        
        for iteration in range(LDPC_MAX_ITERATIONS):
            # Compute current syndrome
            current_syndrome = (self.H @ decoded) % 2
            
            # Check convergence
            if np.array_equal(current_syndrome, syndrome):
                logger.info(f"BP converged in {iteration+1} iterations")
                return decoded
            
            # Message passing step (simplified bit-flipping)
            unsatisfied = np.where(current_syndrome != syndrome)[0]
            if len(unsatisfied) == 0:
                break
            
            # Identify bits to flip (greedy)
            # Calculate how many unsatisfied checks each bit participates in
            # This is effectively the "vote" for flipping
            
            # We need to map unsatisfied checks back to variable nodes (bits)
            # H is (m, n). Rows are checks, cols are bits.
            # We want to sum rows where check is unsatisfied.
            
            # Create a vector of unsatisfied checks
            check_vector = np.zeros(self.m)
            check_vector[unsatisfied] = 1
            
            # Multiply by H.T to get bit scores
            # bit_scores[j] = sum of unsatisfied checks connected to bit j
            bit_scores = self.H.T @ check_vector
            
            # Find the bit(s) with the maximum score
            max_score = np.max(bit_scores)
            
            if max_score == 0:
                # Should not happen if there are unsatisfied checks and graph is connected
                logger.warning("Max score is 0 but checks are unsatisfied. Stalled.")
                break
                
            # Gallager A / Bit Flipping with Threshold
            # Threshold is typically majority of checks. For w_c=3, threshold=2.
            # threshold = self.w_c // 2 + 1
            
            # Serial Bit Flipping with Randomness to break cycles
            # If we are stuck (max_score < threshold), or just generally to avoid cycles:
            # Pick a random bit among those with the highest score.
            
            candidates = np.where(bit_scores == max_score)[0]
            if len(candidates) > 0:
                flip_idx = np.random.choice(candidates)
                decoded[flip_idx] ^= 1
        
        logger.warning(
            f"BP did not converge after {LDPC_MAX_ITERATIONS} iterations"
        )
        # Return best effort
        return decoded
    
    def estimate_leakage(self, syndrome_length: int, qber: float) -> float:
        """
        Estimate information leakage.
        
        Conservative: leakage ≈ syndrome_length + margin.
        """
        # Binary entropy function
        h = lambda p: -p*np.log2(p) - (1-p)*np.log2(1-p) if 0 < p < 1 else 0
        
        # Shannon bound leakage
        shannon_leakage = self.n * h(qber)
        
        # Actual leakage (syndrome + inefficiency)
        # We use the actual syndrome length sent
        actual_leakage = syndrome_length + 100  # Safety margin
        
        logger.debug(
            f"Leakage estimate: {actual_leakage} bits "
            f"(Shannon bound: {shannon_leakage:.1f})"
        )
        return float(actual_leakage)

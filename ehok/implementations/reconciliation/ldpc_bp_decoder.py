"""
Belief-propagation decoder for LDPC codes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from ehok.core import constants
from ehok.utils.logging import get_logger

logger = get_logger("reconciliation.ldpc_bp_decoder")


class LDPCBeliefPropagation:
    """
    Sum-product belief-propagation decoder operating in the log domain.
    """

    def __init__(self, max_iterations: int = constants.LDPC_MAX_ITERATIONS, threshold: float = constants.LDPC_BP_THRESHOLD) -> None:
        self.max_iterations = max_iterations
        self.threshold = threshold

    def decode(
        self, H: sp.csr_matrix, llr: np.ndarray, syndrome: np.ndarray
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Decode a codeword given LLRs and target syndrome.

        Parameters
        ----------
        H : sp.csr_matrix
            Parity-check matrix (m x n).
        llr : np.ndarray
            Log-likelihood ratios for each variable node.
        syndrome : np.ndarray
            Target syndrome vector in {0,1}^m.

        Returns
        -------
        decoded : np.ndarray
            Hard-decision bit estimate after decoding.
        converged : bool
            True if all parity checks satisfied.
        iterations : int
            Number of iterations executed.
        """

        H = H.tocsr()
        m, n = H.shape
        llr = llr.astype(float)
        if llr.shape[0] != n:
            raise ValueError("llr length must equal number of columns in H")
        if syndrome.shape[0] != m:
            raise ValueError("syndrome length must equal number of rows in H")

        # Build adjacency
        var_neighbors: List[List[int]] = [[] for _ in range(n)]
        check_neighbors: List[List[int]] = [[] for _ in range(m)]
        edge_c = []
        edge_v = []
        for c in range(m):
            start, end = H.indptr[c], H.indptr[c + 1]
            cols = H.indices[start:end]
            for v in cols:
                idx = len(edge_c)
                edge_c.append(c)
                edge_v.append(v)
                var_neighbors[v].append(idx)
                check_neighbors[c].append(idx)

        edge_count = len(edge_c)
        if edge_count == 0:
            raise ValueError("Parity-check matrix has no edges")

        # Messages
        r = np.zeros(edge_count, dtype=float)  # check -> var
        q = np.zeros(edge_count, dtype=float)  # var -> check

        # Initialize variable to check messages with channel LLR
        for v, edges in enumerate(var_neighbors):
            for e in edges:
                q[e] = llr[v]

        converged = False
        decoded = np.zeros(n, dtype=np.uint8)

        for iteration in range(1, self.max_iterations + 1):
            # Check node update
            for c, edges in enumerate(check_neighbors):
                if not edges:
                    continue
                tanh_vals = np.tanh(q[edges] / 2.0)
                prod_all = np.prod(tanh_vals)
                for idx, e in enumerate(edges):
                    prod_excl = prod_all / tanh_vals[idx] if len(edges) > 1 else prod_all
                    prod_excl = np.clip(prod_excl, -0.999999, 0.999999)
                    sign = -1.0 if syndrome[c] == 1 else 1.0
                    r[e] = 2.0 * np.arctanh(sign * prod_excl)

            # Variable node update and decisions
            for v, edges in enumerate(var_neighbors):
                total = llr[v] + np.sum(r[edges])
                # Update q messages
                for e in edges:
                    q[e] = total - r[e]
                decoded[v] = 1 if total < 0 else 0

            # Check convergence
            current_syndrome = (H @ decoded) % 2
            if np.array_equal(current_syndrome, syndrome):
                converged = True
                break

            # Early stopping if messages stabilise
            if np.max(np.abs(r)) < self.threshold:
                logger.debug("Stopping early due to message stability")
                break

        return decoded, converged, iteration

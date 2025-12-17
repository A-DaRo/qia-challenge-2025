"""
Belief-propagation decoder for LDPC codes.

This module implements the sum-product belief-propagation algorithm operating
in the log-likelihood ratio (LLR) domain for efficient numerical computation.
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

    Implements iterative message-passing on the Tanner graph representation of
    an LDPC code to decode syndrome-based error correction.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of BP iterations, by default constants.LDPC_MAX_ITERATIONS.
    threshold : float, optional
        Message stability threshold for early stopping, by default constants.LDPC_BP_THRESHOLD.

    Attributes
    ----------
    max_iterations : int
        Maximum iteration limit.
    threshold : float
        Convergence threshold.

    Notes
    -----
    The decoder uses the tanh-domain update rule for check node messages:

    .. math::
        \mu_{c \to v} = 2 \cdot \text{arctanh}\left(\prod_{v' \in N(c) \setminus v} 
                       \tanh(\mu_{v' \to c} / 2)\right)

    Variable node messages accumulate incoming check node messages plus channel LLR.
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
            Parity-check matrix of shape (m, n) in CSR format.
        llr : np.ndarray
            Log-likelihood ratios for each variable node (length n).
            Positive LLR indicates high confidence in bit=0.
            Negative LLR indicates high confidence in bit=1.
        syndrome : np.ndarray
            Target syndrome vector in {0,1}^m.
            syndrome[i]=1 indicates parity check i is unsatisfied.

        Returns
        -------
        decoded : np.ndarray
            Hard-decision bit estimate after decoding (uint8 array).
        converged : bool
            True if all parity checks satisfied (H · decoded = syndrome mod 2).
        iterations : int
            Number of iterations executed before convergence or timeout.

        Raises
        ------
        ValueError
            If LLR length doesn't match H columns or syndrome length doesn't match H rows.
            If parity-check matrix has no edges.

        Notes
        -----
        The decoder alternates between check node updates and variable node updates:

        **Check Node Update** (horizontal step):
            For each check c and connected variable v, compute message
            based on product of incoming tanh values from other variables.
            If syndrome[c]=1, flip the message sign to encode unsatisfied constraint.

        **Variable Node Update** (vertical step):
            For each variable v, sum channel LLR and all incoming check messages.
            Update outgoing messages by subtracting the corresponding check contribution.

        **Hard Decision**: Bit v is decided as 1 if total LLR < 0, else 0.

        **Early Stopping**: Decoding halts if messages stabilize (max |r| < threshold)
        or all parity checks are satisfied.

        References
        ----------
        - Kschischang, Frey, Loeliger, "Factor graphs and the sum-product algorithm",
          IEEE Trans. Inform. Theory (2001).
        - MacKay, "Information Theory, Inference, and Learning Algorithms" (2003).
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
                # Avoid division by zero and numerical instability by computing
                # the product of tanh values excluding the current index directly.
                for idx, e in enumerate(edges):
                    if len(edges) > 1:
                        # product excluding current index
                        prod_excl = np.prod(np.delete(tanh_vals, idx))
                    else:
                        prod_excl = 1.0
                    prod_excl = np.clip(prod_excl, -0.999999, 0.999999)
                    sign = -1.0 if syndrome[c] == 1 else 1.0
                    # Use safe arctanh domain
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

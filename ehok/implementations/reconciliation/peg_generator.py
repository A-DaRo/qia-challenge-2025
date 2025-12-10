"""
PEG-based LDPC matrix generation.

Implements the Progressive Edge-Growth (PEG) algorithm with edge-perspective
DegreeDistribution inputs to maximise local girth and produce sparse parity-
check matrices suitable for belief-propagation decoding.

References
----------
Hu, X. Y., Eleftheriou, E., & Arnold, D. M. (2005). "Regular and irregular
progressive edge-growth tanner graphs." IEEE Transactions on Information Theory,
51(1), 386-398.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import random

import numpy as np
import scipy.sparse as sp

from ehok.core import constants
from ehok.utils.logging import get_logger

logger = get_logger("reconciliation.peg_generator")


@dataclass
class DegreeDistribution:
    """
    Edge-perspective degree distribution for LDPC nodes.

    Parameters
    ----------
    degrees : Sequence[int]
        Distinct node degrees.
    probabilities : Sequence[float]
        Edge-perspective probability for each degree; must sum to 1.
    """

    degrees: Sequence[int]
    probabilities: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.degrees) != len(self.probabilities):
            raise ValueError("degrees and probabilities must have same length")
        if any(d < 1 for d in self.degrees):
            raise ValueError("degrees must be positive")
        if any(p < 0.0 or p > 1.0 for p in self.probabilities):
            raise ValueError("probabilities must be within [0, 1]")
        total = float(sum(self.probabilities))
        if total <= 0.0:
            raise ValueError("probabilities sum must be positive")
        # Perform L1-normalization if sums deviate from 1.0 from source or typos
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            logger.warning(
                "DegreeDistribution: probabilities sum to %.6f, normalizing to 1.0",
                total,
            )
            self.probabilities = [float(p) / total for p in self.probabilities]
        # Final check
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1 after normalization")


class PEGMatrixGenerator:
    """
    Progressive Edge-Growth (PEG) LDPC matrix generator.

    Parameters
    ----------
    n : int
        Codeword length (number of variable nodes).
    rate : float
        Code rate in (0, 1).
    lambda_dist : DegreeDistribution
        Variable node degree distribution.
    rho_dist : DegreeDistribution
        Check node degree distribution.
    max_tree_depth : int, optional
        Maximum depth for BFS tree expansion during girth maximization,
        by default constants.PEG_MAX_TREE_DEPTH.
    seed : int or None, optional
        Random seed for deterministic generation. If None, uses
        constants.PEG_DEFAULT_SEED.

    Attributes
    ----------
    n : int
        Codeword length.
    rate : float
        Code rate.
    m : int
        Number of check nodes (parity checks).
    lambda_dist : DegreeDistribution
        Variable node degree distribution.
    rho_dist : DegreeDistribution
        Check node degree distribution.
    max_tree_depth : int
        Maximum BFS depth.
    random : random.Random
        Seeded random generator.

    Raises
    ------
    ValueError
        If n is not positive, rate is not in (0,1), or computed check count is non-positive.

    Notes
    -----
    The PEG algorithm builds the Tanner graph incrementally, placing each edge
    to maximize the local girth (shortest cycle) around the current variable node.
    This is achieved via breadth-first search to identify check nodes at maximum
    distance from the variable node's existing neighborhood.

    - Uses edge-perspective degree distributions for variable and check nodes.
    - Builds Tanner graph incrementally to maximise local girth.
    - Returns parity-check matrix in CSR format.
    """

    def __init__(
        self,
        n: int,
        rate: float,
        lambda_dist: DegreeDistribution,
        rho_dist: DegreeDistribution,
        max_tree_depth: int = constants.PEG_MAX_TREE_DEPTH,
        seed: Optional[int] = None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0 < rate < 1:
            raise ValueError("rate must be in (0, 1)")
        self.n = n
        self.rate = rate
        self.lambda_dist = lambda_dist
        self.rho_dist = rho_dist
        self.max_tree_depth = max_tree_depth
        self.random = random.Random(seed if seed is not None else constants.PEG_DEFAULT_SEED)

        # Compute number of checks (rows)
        self.m = int(round(self.n * (1.0 - rate)))
        if self.m <= 0:
            raise ValueError("computed check node count must be positive")

    def generate(self) -> sp.csr_matrix:
        """
        Generate an LDPC parity-check matrix using PEG.

        Returns
        -------
        scipy.sparse.csr_matrix
            Parity-check matrix of shape (m, n) over GF(2).
        """
        import time
        
        logger.debug("PEG: Starting matrix generation (n=%d, m=%d, rate=%.2f)", 
                     self.n, self.m, self.rate)
        start_time = time.perf_counter()

        # Assign degrees
        logger.debug("PEG: Assigning node degrees from distributions")
        vn_degrees, cn_target_degrees = self._assign_node_degrees()
        total_edges = sum(vn_degrees)
        logger.debug("PEG: VN degrees assigned (total edges: %d)", total_edges)
        logger.debug(
            "PEG: CN target degrees: min=%d, max=%d, avg=%.2f",
            min(cn_target_degrees),
            max(cn_target_degrees),
            np.mean(cn_target_degrees),
        )
        
        check_adjacency = [set() for _ in range(self.m)]
        var_adjacency = [set() for _ in range(self.n)]
        current_cn_degrees = [0 for _ in range(self.m)]

        # Process variable nodes in increasing degree to favour low-degree nodes first
        logger.debug("PEG: Sorting variable nodes by degree")
        order = np.argsort(vn_degrees)
        
        # Track progress
        edges_placed = 0
        progress_interval = max(1, total_edges // 10)  # Report every 10%
        
        logger.debug("PEG: Placing edges using girth-maximization strategy")
        for v_idx, v in enumerate(order):
            deg_v = vn_degrees[v]
            for edge_idx in range(deg_v):
                if edge_idx == 0:
                    # First edge: prefer underfull checks based on target degrees
                    c = self._select_check_node(current_cn_degrees, cn_target_degrees)
                else:
                    # Subsequent edges: maximize girth via BFS on current adjacency
                    reachable = self._bfs_reachable(v, var_adjacency, check_adjacency)
                    candidates = [c_idx for c_idx in range(self.m) if c_idx not in reachable]
                    if not candidates:
                        candidates = list(range(self.m))
                    c = self._select_check_node(current_cn_degrees, cn_target_degrees, candidates)

                check_adjacency[c].add(v)
                var_adjacency[v].add(c)
                current_cn_degrees[c] += 1
                edges_placed += 1
                
                # Progress logging
                if edges_placed % progress_interval == 0:
                    progress_pct = (edges_placed / total_edges) * 100
                    logger.debug("PEG: Progress: %d/%d edges placed (%.1f%%)", 
                                 edges_placed, total_edges, progress_pct)

        logger.debug("PEG: All %d edges placed. Building sparse matrix...", edges_placed)
        
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        for c_idx, vars_for_check in enumerate(check_adjacency):
            for v in vars_for_check:
                rows.append(c_idx)
                cols.append(v)
                data.append(1)
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.m, self.n), dtype=np.uint8)
        
        elapsed = time.perf_counter() - start_time
        density = H.nnz / (self.m * self.n)
        
        logger.info(
            "PEG: Generated matrix n=%d, m=%d, rate=%.2f, nnz=%d, density=%.4f in %.2fs", 
            self.n, self.m, self.rate, H.nnz, density, elapsed
        )
        logger.debug("PEG: Actual CN degrees: min=%d, max=%d, avg=%.2f",
                     min(current_cn_degrees), max(current_cn_degrees), np.mean(current_cn_degrees))
        
        return H

    def _assign_node_degrees(self) -> Tuple[List[int], List[int]]:
        """
        Convert edge-perspective distributions to node degree assignments.

        Returns
        -------
        vn_degrees : list of int
            Variable-node degrees (length n).
        cn_degrees : list of int
            Check-node target degrees (length m).

        Notes
        -----
        Samples degrees from the provided distributions and ensures the total
        edge count matches between variable and check nodes for a valid graph.
        """

        vn_degrees = self._sample_degrees(self.n, self.lambda_dist)
        cn_degrees = self._sample_degrees(self.m, self.rho_dist, target_edges=sum(vn_degrees))
        return vn_degrees, cn_degrees

    def _sample_degrees(
        self, node_count: int, dist: DegreeDistribution, target_edges: Optional[int] = None
    ) -> List[int]:
        """
        Sample node degrees from edge-perspective distribution.

        Parameters
        ----------
        node_count : int
            Number of nodes to assign degrees to.
        dist : DegreeDistribution
            Edge-perspective degree distribution.
        target_edges : int or None, optional
            Target total edge count. If None, computed from distribution.

        Returns
        -------
        list of int
            List of degrees (length = node_count), shuffled randomly.

        Notes
        -----
        Converts edge-perspective probabilities to node counts by distributing
        edges across degree classes. Handles fractional node counts through
        rounding and remainder distribution to match exact node_count.
        """
        probs = np.array(dist.probabilities, dtype=float)
        degrees = np.array(dist.degrees, dtype=int)
        edge_factor = float(np.sum(probs / degrees))
        total_edges = target_edges if target_edges is not None else int(round(node_count / edge_factor))
        expected_counts = (probs * total_edges / degrees).astype(float)

        counts = np.floor(expected_counts).astype(int)
        remainder = node_count - counts.sum()
        if remainder > 0:
            # Distribute remaining nodes to degrees with largest fractional parts
            fractional = expected_counts - counts
            order = np.argsort(fractional)[::-1]
            for idx in order[:remainder]:
                counts[idx] += 1
        elif remainder < 0:
            # Trim excess starting from highest degrees
            order = np.argsort(degrees)[::-1]
            excess = -remainder
            for idx in order:
                take = min(excess, counts[idx])
                counts[idx] -= take
                excess -= take
                if excess == 0:
                    break

        # If still mismatch, pad with minimum degree
        while counts.sum() < node_count:
            counts[np.argmin(degrees)] += 1
        while counts.sum() > node_count:
            idx = np.argmax(degrees)
            if counts[idx] > 0:
                counts[idx] -= 1
            else:
                break

        node_degrees: List[int] = []
        for deg, cnt in zip(degrees, counts):
            node_degrees.extend([int(deg)] * int(cnt))
        self.random.shuffle(node_degrees)
        return node_degrees

    def _bfs_reachable(
        self, v: int, var_adj: List[set], check_adj: List[set]
    ) -> set:
        """
        Compute reachable check nodes from variable v up to max_tree_depth using cached adjacency.

        Parameters
        ----------
        v : int
            Starting variable node index.
        var_adj : list of set
            Variable node adjacency lists.
        check_adj : list of set
            Check node adjacency lists.

        Returns
        -------
        set of int
            Set of check node indices reachable within max_tree_depth.

        Notes
        -----
        Performs alternating BFS expansion: variable nodes on even depths,
        check nodes on odd depths. This identifies check nodes in the
        neighborhood of variable v to avoid creating short cycles.
        """

        visited_vars = {v}
        visited_checks = set()

        frontier_vars = {v}
        frontier_checks = set()

        for depth in range(self.max_tree_depth):
            if depth % 2 == 0:
                if not frontier_vars:
                    break
                next_checks = set()
                for var in frontier_vars:
                    for c_idx in var_adj[var]:
                        if c_idx not in visited_checks:
                            visited_checks.add(c_idx)
                            next_checks.add(c_idx)
                frontier_vars = set()
                frontier_checks = next_checks
            else:
                if not frontier_checks:
                    break
                next_vars = set()
                for c_idx in frontier_checks:
                    for var in check_adj[c_idx]:
                        if var not in visited_vars:
                            visited_vars.add(var)
                            next_vars.add(var)
                frontier_checks = set()
                frontier_vars = next_vars

        return visited_checks

    def _select_check_node(
        self,
        current_degrees: Sequence[int],
        target_degrees: Sequence[int],
        candidates: Optional[Sequence[int]] = None,
    ) -> int:
        """
        Select check node preferring those furthest below their target degree.

        Parameters
        ----------
        current_degrees : sequence of int
            Current degree of each check node.
        target_degrees : sequence of int
            Target degree for each check node.
        candidates : sequence of int or None, optional
            Restricted set of check nodes to consider. If None, considers all.

        Returns
        -------
        int
            Selected check node index.

        Notes
        -----
        Selection criterion: minimize (current_degree / target_degree, current_degree).
        This prioritizes filling underfull check nodes while maintaining degree balance.
        Ties are broken randomly.
        """

        if candidates is None or len(candidates) == 0:
            candidates = range(len(current_degrees))

        best = []
        best_score: Optional[Tuple[float, int]] = None

        for idx in candidates:
            target = max(int(target_degrees[idx]), 1)
            fill_ratio = current_degrees[idx] / target
            score = (fill_ratio, current_degrees[idx])
            if best_score is None or score < best_score:
                best_score = score
                best = [idx]
            elif score == best_score:
                best.append(idx)

        return self.random.choice(best)

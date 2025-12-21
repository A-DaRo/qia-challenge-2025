"""
PEG-based LDPC matrix generation.

Implements the Progressive Edge-Growth (PEG) algorithm with edge-perspective
DegreeDistribution inputs to maximise local girth and produce sparse parity-
check matrices suitable for belief-propagation decoding.

This module is an **offline tool** intended for pre-runtime matrix generation.
Generated matrices are stored as compressed .npz files for runtime loading by
the MatrixManager.

References
----------
Hu, X. Y., Eleftheriou, E., & Arnold, D. M. (2005). "Regular and irregular
progressive edge-growth tanner graphs." IEEE Transactions on Information Theory,
51(1), 386-398.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.scripts import numba_kernels
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


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

    Raises
    ------
    ValueError
        If degrees and probabilities have different lengths, contain
        invalid values, or probabilities do not sum to a positive value.
    """

    degrees: Sequence[int]
    probabilities: Sequence[float]

    def __post_init__(self) -> None:
        # Basic shape checks
        if len(self.degrees) != len(self.probabilities):
            raise ValueError("degrees and probabilities must have same length")
        # Value checks
        if any(int(d) < 1 for d in self.degrees):
            raise ValueError("degrees must be positive")
        if any(float(p) < 0.0 or float(p) > 1.0 for p in self.probabilities):
            raise ValueError("probabilities must be within [0, 1]")

        # Normalise and final validation
        total = float(sum(self.probabilities))
        if total <= 0.0:
            raise ValueError("probabilities sum must be positive")

        # Perform L1-normalization if sums deviate from 1.0
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            logger.warning(
                "DegreeDistribution: probabilities sum to %.6f, normalizing to 1.0",
                total,
            )
            self.probabilities = [float(p) / total for p in self.probabilities]
        else:
            # Make sure we have a concrete list of floats
            self.probabilities = [float(p) for p in self.probabilities]

        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1 after normalization")


class PEGMatrixGenerator:
    """
    Progressive Edge-Growth (PEG) LDPC matrix generator.

    The PEG algorithm builds the Tanner graph incrementally, placing each edge
    to maximize the local girth (shortest cycle) around the current variable node.
    This is achieved via breadth-first search to identify check nodes at maximum
    distance from the variable node's existing neighborhood.

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
        Maximum depth for BFS tree expansion during girth maximization.
        Defaults to constants.PEG_MAX_TREE_DEPTH.
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

    Raises
    ------
    ValueError
        If n is not positive, rate is not in (0,1), or computed check count
        is non-positive.
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
        self._seed = seed if seed is not None else constants.PEG_DEFAULT_SEED
        self.random = random.Random(self._seed)

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
        logger.debug(
            "PEG: Starting matrix generation (n=%d, m=%d, rate=%.2f)",
            self.n,
            self.m,
            self.rate,
        )
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

        if numba_kernels.numba_available():
            H = self._generate_numba(vn_degrees, cn_target_degrees, total_edges)
        else:
            H = self._generate_python(vn_degrees, cn_target_degrees, total_edges)

        elapsed = time.perf_counter() - start_time
        density = H.nnz / (self.m * self.n)

        logger.info(
            "PEG: Generated matrix n=%d, m=%d, rate=%.2f, nnz=%d, density=%.4f "
            "in %.2fs",
            self.n,
            self.m,
            self.rate,
            H.nnz,
            density,
            elapsed,
        )

        return H

    def _generate_python(
        self,
        vn_degrees: List[int],
        cn_target_degrees: List[int],
        total_edges: int,
    ) -> sp.csr_matrix:
        """Original PEG implementation using Python sets.

        Kept as a correctness fallback if Numba is unavailable.
        """
        check_adjacency: List[set] = [set() for _ in range(self.m)]
        var_adjacency: List[set] = [set() for _ in range(self.n)]
        current_cn_degrees = [0 for _ in range(self.m)]

        # Process variable nodes in increasing degree to favour low-degree first
        logger.debug("PEG: Sorting variable nodes by degree")
        order = np.argsort(vn_degrees)

        # Track progress
        edges_placed = 0
        progress_interval = max(1, total_edges // 10)  # Report every 10%

        logger.debug("PEG: Placing edges using girth-maximization strategy")
        for v in order:
            deg_v = vn_degrees[v]
            for edge_idx in range(deg_v):
                if edge_idx == 0:
                    # First edge: prefer underfull checks based on target degrees
                    c = self._select_check_node(
                        current_cn_degrees, cn_target_degrees
                    )
                else:
                    # Subsequent edges: maximize girth via BFS on current adjacency
                    reachable = self._bfs_reachable(v, var_adjacency, check_adjacency)
                    candidates = [
                        c_idx for c_idx in range(self.m) if c_idx not in reachable
                    ]
                    if not candidates:
                        candidates = list(range(self.m))
                    c = self._select_check_node(
                        current_cn_degrees, cn_target_degrees, candidates
                    )

                check_adjacency[c].add(v)
                var_adjacency[v].add(c)
                current_cn_degrees[c] += 1
                edges_placed += 1

                # Progress logging
                if edges_placed % progress_interval == 0:
                    progress_pct = (edges_placed / total_edges) * 100
                    logger.debug(
                        "PEG: Progress: %d/%d edges placed (%.1f%%)",
                        edges_placed,
                        total_edges,
                        progress_pct,
                    )

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
        logger.debug(
            "PEG: Actual CN degrees: min=%d, max=%d, avg=%.2f",
            min(current_cn_degrees),
            max(current_cn_degrees),
            np.mean(current_cn_degrees),
        )
        return H

    def _generate_numba(
        self,
        vn_degrees: List[int],
        cn_target_degrees: List[int],
        total_edges: int,
    ) -> sp.csr_matrix:
        """Fast PEG implementation using fixed-width adjacency arrays + Numba.

        Notes
        -----
        The Numba kernels operate on fixed-width adjacency arrays. For large
        frames and/or irregular degree distributions, some check nodes can
        temporarily exceed their target degree during PEG fallback. To avoid
        brittle failures (IndexError: capacity exceeded), we start with a
        conservative capacity estimate and automatically retry with larger
        check-node adjacency width if needed.
        """
        logger.debug("PEG: Using Numba-accelerated graph builder")

        vn_target = np.asarray(vn_degrees, dtype=np.int32)
        cn_target = np.asarray(cn_target_degrees, dtype=np.int32)
        order = np.argsort(vn_target).astype(np.int32)

        max_vn_degree = int(vn_target.max(initial=0))
        # Small VN headroom: protects against any off-by-one in kernels and
        # keeps retry logic symmetric with CN capacity growth.
        vn_capacity = max_vn_degree + 1

        # Initial CN capacity estimate.
        # - cn_target.max()+8: baseline headroom beyond target degrees
        # - avg_cn_deg*6+16: robust for skewed load during PEG fallback
        # - >=32: avoids frequent retries for moderate frames
        avg_cn_deg = int(np.ceil(total_edges / max(1, self.m)))
        max_cn_degree = max(int(cn_target.max(initial=0)) + 8, avg_cn_deg * 6 + 16, 32)

        # Round up to a power of two (cheap doubling strategy on retry).
        max_cn_degree = 1 << (int(max_cn_degree) - 1).bit_length()

        max_retries = 6
        max_cn_degree_limit = 8192
        max_vn_degree_limit = max(256, vn_capacity * 4)
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            vn_adj = np.full((self.n, vn_capacity), -1, dtype=np.int32)
            cn_adj = np.full((self.m, max_cn_degree), -1, dtype=np.int32)
            vn_deg = np.zeros(self.n, dtype=np.int32)
            cn_deg = np.zeros(self.m, dtype=np.int32)

            # BFS workspaces
            visited_vars = np.zeros(self.n, dtype=np.int32)
            visited_checks = np.zeros(self.m, dtype=np.int32)
            frontier_vars = np.empty(self.n, dtype=np.int32)
            next_frontier_vars = np.empty(self.n, dtype=np.int32)
            frontier_checks = np.empty(self.m, dtype=np.int32)
            next_frontier_checks = np.empty(self.m, dtype=np.int32)

            rng_state = np.uint64(self._seed if self._seed is not None else 1)

            try:
                numba_kernels.build_peg_graph(
                    order=order,
                    vn_target_deg=vn_target,
                    cn_target_deg=cn_target,
                    vn_adj=vn_adj,
                    cn_adj=cn_adj,
                    vn_deg=vn_deg,
                    cn_deg=cn_deg,
                    max_tree_depth=np.int32(self.max_tree_depth),
                    visited_vars=visited_vars,
                    visited_checks=visited_checks,
                    frontier_vars=frontier_vars,
                    frontier_checks=frontier_checks,
                    next_frontier_vars=next_frontier_vars,
                    next_frontier_checks=next_frontier_checks,
                    rng_state=rng_state,
                )
                break
            except IndexError as exc:
                # Numba kernels raise IndexError on overflow.
                msg = str(exc)
                last_error = exc

                overflow_vn = "vn_adj capacity exceeded" in msg
                overflow_cn = "cn_adj capacity exceeded" in msg

                # If the kernel cannot find any eligible check node, it implies
                # all checks are considered at capacity under the current
                # fixed-width representation.
                if "no available check node" in msg:
                    overflow_cn = True

                if not overflow_vn and not overflow_cn:
                    # Backwards compatibility with older error message.
                    if "fixed-width adjacency capacity exceeded" not in msg:
                        raise
                    overflow_cn = True

                if overflow_cn:
                    if max_cn_degree >= max_cn_degree_limit:
                        logger.error(
                            "PEG: Numba CN adjacency capacity exceeded (width=%d) and hit limit=%d",
                            max_cn_degree,
                            max_cn_degree_limit,
                        )
                        raise
                    new_cn = max_cn_degree * 2
                    logger.warning(
                        "PEG: Numba CN adjacency capacity exceeded (width=%d). Retrying with width=%d (%d/%d)",
                        max_cn_degree,
                        new_cn,
                        attempt + 1,
                        max_retries,
                    )
                    max_cn_degree = new_cn

                if overflow_vn:
                    if vn_capacity >= max_vn_degree_limit:
                        logger.error(
                            "PEG: Numba VN adjacency capacity exceeded (width=%d) and hit limit=%d",
                            vn_capacity,
                            max_vn_degree_limit,
                        )
                        raise
                    new_vn = vn_capacity * 2
                    logger.warning(
                        "PEG: Numba VN adjacency capacity exceeded (width=%d). Retrying with width=%d (%d/%d)",
                        vn_capacity,
                        new_vn,
                        attempt + 1,
                        max_retries,
                    )
                    vn_capacity = new_vn
        else:
            # Defensive: loop exhausted without break.
            if last_error is not None:
                raise last_error

        # Build COO arrays in Numba, then CSR
        rows = np.empty(total_edges, dtype=np.int32)
        cols = np.empty(total_edges, dtype=np.int32)
        numba_kernels.fill_edges_from_cn_adj(cn_adj=cn_adj, cn_deg=cn_deg, rows=rows, cols=cols)
        data = np.ones(total_edges, dtype=np.uint8)
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.m, self.n), dtype=np.uint8)

        logger.debug(
            "PEG: Actual CN degrees: min=%d, max=%d, avg=%.2f",
            int(cn_deg.min(initial=0)),
            int(cn_deg.max(initial=0)),
            float(cn_deg.mean()) if cn_deg.size else 0.0,
        )
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
        cn_degrees = self._sample_degrees(
            self.m, self.rho_dist, target_edges=sum(vn_degrees)
        )
        return vn_degrees, cn_degrees

    def _sample_degrees(
        self,
        node_count: int,
        dist: DegreeDistribution,
        target_edges: Optional[int] = None,
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
        total_edges = (
            target_edges
            if target_edges is not None
            else int(round(node_count / edge_factor))
        )
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
        Compute reachable check nodes from variable v up to max_tree_depth.

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
        visited_checks: set = set()

        frontier_vars = {v}
        frontier_checks: set = set()

        for depth in range(self.max_tree_depth):
            if depth % 2 == 0:
                if not frontier_vars:
                    break
                next_checks: set = set()
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
                next_vars: set = set()
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
            candidates = list(range(len(current_degrees)))

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

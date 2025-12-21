"""
ACE-optimized PEG LDPC matrix generation for rate-compatible reconciliation.

Implements the ACE-PEG algorithm combining:
1. Progressive Edge-Growth (PEG) for girth maximization
2. ACE (Approximate Cycle EMD) conditioning to minimize trapping sets

This generator produces robust mother codes (R₀=0.5) with superior structural
properties for rate-compatible puncturing schemes.

References
----------
Tian, T., Jones, C., Villasenor, J. D., & Wesel, R. D. (2003). "Construction of
irregular LDPC codes with low error floors." In IEEE International Conference on
Communications (Vol. 5, pp. 3125-3129).

Hu, X. Y., Eleftheriou, E., & Arnold, D. M. (2005). "Regular and irregular
progressive edge-growth tanner graphs." IEEE Transactions on Information Theory,
51(1), 386-398.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import yaml

from caligo.scripts.peg_generator import DegreeDistribution, PEGMatrixGenerator
from caligo.scripts import numba_kernels
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ACEConfig:
    """
    ACE algorithm configuration.
    
    Parameters
    ----------
    d_ACE : int
        Maximum cycle length to condition (cycles up to 2*d_ACE are checked).
        Default 8 → cycles up to length 16.
    eta : int
        Minimum cycle ACE threshold. All cycles ≤ 2*d_ACE must have ACE ≥ eta.
        Default 4 (Tian et al., 2003 recommendation).
    bypass_threshold : int
        Variables with degree ≥ bypass_threshold skip ACE check (their ACE is
        automatically ≥ eta). Default: eta + 2 = 6.
    
    References
    ----------
    Tian et al. (2003), Section III: ACE metric definition and thresholds.
    """
    d_ACE: int = 8
    eta: int = 4
    bypass_threshold: int = 6
    
    def __post_init__(self) -> None:
        """Validate ACE configuration."""
        if self.d_ACE < 2:
            raise ValueError("d_ACE must be at least 2")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if self.bypass_threshold < self.eta:
            raise ValueError("bypass_threshold must be >= eta")


class ACEPEGGenerator(PEGMatrixGenerator):
    """
    ACE-enhanced PEG generator for low error floor LDPC codes.
    
    Extends standard PEG with ACE conditioning to minimize trapping sets
    that cause high error floors in finite-length codes.
    
    Parameters
    ----------
    n : int
        Codeword length (default 4096 for Caligo).
    rate : float
        Mother code rate (default 0.5).
    lambda_dist : DegreeDistribution
        Variable node degree distribution (irregular optimized).
    rho_dist : DegreeDistribution
        Check node degree distribution.
    ace_config : ACEConfig
        ACE conditioning parameters.
    max_tree_depth : int, optional
        BFS depth for girth maximization (PEG baseline).
    seed : int, optional
        Random seed for deterministic generation.
    
    Attributes
    ----------
    ace_config : ACEConfig
        ACE algorithm configuration.
    
    References
    ----------
    Tian et al. (2003): ACE algorithm, Section III-IV.
    Hu et al. (2005): PEG baseline algorithm.
    """
    
    def __init__(
        self,
        n: int,
        rate: float,
        lambda_dist: DegreeDistribution,
        rho_dist: DegreeDistribution,
        ace_config: ACEConfig,
        max_tree_depth: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n, rate, lambda_dist, rho_dist, max_tree_depth, seed)
        self.ace_config = ace_config
        logger.info(
            "ACE-PEG: Initialized generator (n=%d, m=%d, R=%.2f, "
            "d_ACE=%d, eta=%d, bypass=%d)",
            self.n,
            self.m,
            self.rate,
            ace_config.d_ACE,
            ace_config.eta,
            ace_config.bypass_threshold,
        )
    
    def generate(self) -> sp.csr_matrix:
        """
        Generate ACE-conditioned LDPC parity-check matrix.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Parity-check matrix H of shape (m, n) over GF(2).
        
        Notes
        -----
        Uses hybrid PEG + ACE algorithm:
        1. Assign node degrees from distributions
        2. For each edge, select check node using PEG (girth maximization)
        3. Before finalizing edge, verify ACE constraint is satisfied
        4. If ACE fails, try next-best check node
        5. Build sparse matrix from final adjacency lists
        """
        logger.info(
            "ACE-PEG: Starting ACE-conditioned matrix generation "
            "(n=%d, m=%d, rate=%.2f)",
            self.n,
            self.m,
            self.rate,
        )
        start_time = time.perf_counter()
        
        # Assign degrees
        logger.debug("ACE-PEG: Assigning node degrees from distributions")
        vn_degrees, cn_target_degrees = self._assign_node_degrees()
        total_edges = sum(vn_degrees)
        logger.info(
            "ACE-PEG: VN degrees assigned (total edges: %d, avg VN degree: %.2f)",
            total_edges,
            total_edges / self.n,
        )
        logger.debug(
            "ACE-PEG: CN target degrees: min=%d, max=%d, avg=%.2f",
            min(cn_target_degrees),
            max(cn_target_degrees),
            np.mean(cn_target_degrees),
        )

        if numba_kernels.numba_available():
            H, cn_deg, ace_checks_performed, ace_fallbacks, vn_adj, cn_adj, vn_deg = (
                self._generate_numba(vn_degrees, cn_target_degrees, total_edges)
            )
            # Validate ACE properties (sample-based)
            self._validate_ace_properties_numba(
                H=H,
                vn_adj=vn_adj,
                cn_adj=cn_adj,
                vn_deg=vn_deg,
                cn_deg=cn_deg,
            )
        else:
            (
                H,
                var_adjacency,
                check_adjacency,
                current_cn_degrees,
                ace_checks_performed,
                ace_fallbacks,
            ) = self._generate_python(vn_degrees, cn_target_degrees, total_edges)
            self._validate_ace_properties(H, var_adjacency, check_adjacency)

        elapsed = time.perf_counter() - start_time
        density = H.nnz / (self.m * self.n)
        logger.info(
            "ACE-PEG: Matrix generation complete in %.2fs "
            "(n=%d, m=%d, nnz=%d, density=%.4f)",
            elapsed,
            self.n,
            self.m,
            H.nnz,
            density,
        )

        logger.info(
            "ACE-PEG: ACE checks performed: %d, fallbacks: %d (%.2f%%)",
            ace_checks_performed,
            ace_fallbacks,
            100.0 * ace_fallbacks / max(1, ace_checks_performed),
        )

        return H

    def _generate_python(
        self,
        vn_degrees: List[int],
        cn_target_degrees: List[int],
        total_edges: int,
    ) -> Tuple[
        sp.csr_matrix,
        List[Set[int]],
        List[Set[int]],
        List[int],
        int,
        int,
    ]:
        """Original ACE-PEG implementation using Python sets.

        Returned values include adjacency lists for validation.
        """
        check_adjacency: List[Set[int]] = [set() for _ in range(self.m)]
        var_adjacency: List[Set[int]] = [set() for _ in range(self.n)]
        current_cn_degrees = [0 for _ in range(self.m)]

        logger.debug("ACE-PEG: Sorting variable nodes by degree")
        order = np.argsort(vn_degrees)

        edges_placed = 0
        ace_checks_performed = 0
        ace_fallbacks = 0
        progress_interval = max(1, total_edges // 20)

        logger.info("ACE-PEG: Placing edges with ACE conditioning...")
        for v in order:
            deg_v = vn_degrees[v]
            for edge_idx in range(deg_v):
                c = self._place_edge_with_ace(
                    v=v,
                    edge_idx=edge_idx,
                    var_adj=var_adjacency,
                    check_adj=check_adjacency,
                    current_cn_degrees=current_cn_degrees,
                    target_cn_degrees=cn_target_degrees,
                    ace_config=self.ace_config,
                )

                check_adjacency[c].add(v)
                var_adjacency[v].add(c)
                current_cn_degrees[c] += 1
                edges_placed += 1

                current_degree = len(var_adjacency[v])
                if current_degree < self.ace_config.bypass_threshold:
                    ace_checks_performed += 1

                if edges_placed % progress_interval == 0:
                    progress_pct = (edges_placed / total_edges) * 100
                    logger.info(
                        "ACE-PEG: Progress: %d/%d edges placed (%.1f%%), "
                        "ACE checks: %d",
                        edges_placed,
                        total_edges,
                        progress_pct,
                        ace_checks_performed,
                    )

        logger.debug("ACE-PEG: Building sparse CSR matrix...")
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
            "ACE-PEG: Actual CN degrees: min=%d, max=%d, avg=%.2f",
            min(current_cn_degrees),
            max(current_cn_degrees),
            np.mean(current_cn_degrees),
        )
        return (
            H,
            var_adjacency,
            check_adjacency,
            current_cn_degrees,
            ace_checks_performed,
            ace_fallbacks,
        )

    def _generate_numba(
        self,
        vn_degrees: List[int],
        cn_target_degrees: List[int],
        total_edges: int,
    ) -> Tuple[sp.csr_matrix, np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
        """Fast ACE-PEG implementation using fixed-width adjacency arrays + Numba."""
        logger.info("ACE-PEG: Using Numba-accelerated graph builder")

        vn_target = np.asarray(vn_degrees, dtype=np.int32)
        cn_target = np.asarray(cn_target_degrees, dtype=np.int32)
        order = np.argsort(vn_target).astype(np.int32)

        max_vn_degree = int(vn_target.max(initial=0))
        # Check degrees can exceed their targets (esp. when reachable set covers
        # all checks); allocate headroom and let kernels skip full checks.
        # For large block lengths and ACE constraints, local check node degrees
        # can spike significantly above the average.
        max_cn_degree = max(
            int(cn_target.max(initial=0)) * 4,
            int(np.ceil(total_edges / max(1, self.m))) * 6,
            256,
        )

        vn_adj = np.full((self.n, max_vn_degree), -1, dtype=np.int32)
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

        # ACE workspaces
        p_var = np.empty(self.n, dtype=np.int32)
        p_check = np.empty(self.m, dtype=np.int32)
        pvar_seen = np.zeros(self.n, dtype=np.int32)
        pcheck_seen = np.zeros(self.m, dtype=np.int32)
        active_vars = np.empty(self.n, dtype=np.int32)
        next_active_vars = np.empty(self.n, dtype=np.int32)
        active_checks = np.empty(self.m, dtype=np.int32)
        next_active_checks = np.empty(self.m, dtype=np.int32)
        
        # Anti-duplicate arrays for Viterbi
        var_in_next = np.zeros(self.n, dtype=np.int32)
        check_in_next = np.zeros(self.m, dtype=np.int32)

        seed = getattr(self, "_seed", 1)
        rng_state = np.uint64(seed if seed is not None else 1)

        (
            _rng_state,
            ace_checks_performed,
            ace_fallbacks,
        ) = numba_kernels.build_ace_peg_graph(
            order=order,
            vn_target_deg=vn_target,
            cn_target_deg=cn_target,
            vn_adj=vn_adj,
            cn_adj=cn_adj,
            vn_deg=vn_deg,
            cn_deg=cn_deg,
            max_tree_depth=np.int32(self.max_tree_depth),
            d_ace=np.int32(self.ace_config.d_ACE),
            eta=np.int32(self.ace_config.eta),
            bypass_threshold=np.int32(self.ace_config.bypass_threshold),
            visited_vars=visited_vars,
            visited_checks=visited_checks,
            frontier_vars=frontier_vars,
            frontier_checks=frontier_checks,
            next_frontier_vars=next_frontier_vars,
            next_frontier_checks=next_frontier_checks,
            p_var=p_var,
            p_check=p_check,
            pvar_seen=pvar_seen,
            pcheck_seen=pcheck_seen,
            active_vars=active_vars,
            active_checks=active_checks,
            next_active_vars=next_active_vars,
            next_active_checks=next_active_checks,
            var_in_next=var_in_next,
            check_in_next=check_in_next,
            rng_state=rng_state,
        )

        rows = np.empty(total_edges, dtype=np.int32)
        cols = np.empty(total_edges, dtype=np.int32)
        numba_kernels.fill_edges_from_cn_adj(cn_adj=cn_adj, cn_deg=cn_deg, rows=rows, cols=cols)
        data = np.ones(total_edges, dtype=np.uint8)
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.m, self.n), dtype=np.uint8)

        logger.debug(
            "ACE-PEG: Actual CN degrees: min=%d, max=%d, avg=%.2f",
            int(cn_deg.min(initial=0)),
            int(cn_deg.max(initial=0)),
            float(cn_deg.mean()) if cn_deg.size else 0.0,
        )

        return (
            H,
            cn_deg,
            int(ace_checks_performed),
            int(ace_fallbacks),
            vn_adj,
            cn_adj,
            vn_deg,
        )

    def _validate_ace_properties_numba(
        self,
        H: sp.csr_matrix,
        vn_adj: np.ndarray,
        cn_adj: np.ndarray,
        vn_deg: np.ndarray,
        cn_deg: np.ndarray,
    ) -> None:
        """Validate ACE constraints using Numba workspaces (sample-based)."""
        logger.info("ACE-PEG: Validating ACE properties (sample-based, numba)...")
        sample_size = min(100, max(10, self.n // 10))
        sampled_vars = self.random.sample(range(self.n), sample_size)

        # ACE workspaces
        p_var = np.empty(self.n, dtype=np.int32)
        p_check = np.empty(self.m, dtype=np.int32)
        pvar_seen = np.zeros(self.n, dtype=np.int32)
        pcheck_seen = np.zeros(self.m, dtype=np.int32)
        active_vars = np.empty(self.n, dtype=np.int32)
        next_active_vars = np.empty(self.n, dtype=np.int32)
        active_checks = np.empty(self.m, dtype=np.int32)
        next_active_checks = np.empty(self.m, dtype=np.int32)
        
        var_in_next = np.zeros(self.n, dtype=np.int32)
        check_in_next = np.zeros(self.m, dtype=np.int32)

        visit_token = np.int32(1)
        violations = 0

        for v in sampled_vars:
            visit_token = np.int32(visit_token + 1)
            passes = numba_kernels.ace_detection_viterbi(
                np.int32(v),
                vn_adj,
                cn_adj,
                vn_deg,
                cn_deg,
                np.int32(self.ace_config.d_ACE),
                np.int32(self.ace_config.eta),
                p_var,
                p_check,
                pvar_seen,
                pcheck_seen,
                active_vars,
                active_checks,
                next_active_vars,
                next_active_checks,
                var_in_next,
                check_in_next,
                visit_token,
            )
            if passes == 0:
                violations += 1
                logger.warning(
                    "ACE-PEG: Variable %d violates ACE constraint (numba check)",
                    v,
                )

        compliance_rate = (
            100.0 * (sample_size - violations) / sample_size if sample_size > 0 else 0.0
        )
        logger.info(
            "ACE-PEG: ACE compliance: %.1f%% (%d/%d sampled variables pass)",
            compliance_rate,
            sample_size - violations,
            sample_size,
        )

        if violations > sample_size * 0.05:
            logger.warning(
                "ACE-PEG: High ACE violation rate (%.1f%%). "
                "Matrix may have suboptimal error floor.",
                100.0 * violations / sample_size,
            )
    
    def _compute_ace(
        self,
        v: int,
        var_adj: List[Set[int]],
        check_adj: List[Set[int]],
    ) -> int:
        """
        Compute ACE (Approximate Cycle EMD) of variable node v.
        
        The ACE metric quantifies the connectivity strength of a variable node
        within cycles. Low ACE indicates vulnerability to trapping sets.
        
        Parameters
        ----------
        v : int
            Variable node index.
        var_adj : List[Set[int]]
            Variable adjacency lists: var_adj[v] = {checks adjacent to v}.
        check_adj : List[Set[int]]
            Check adjacency lists: check_adj[c] = {variables adjacent to c}.
        
        Returns
        -------
        int
            ACE value for variable v.
        
        Notes
        -----
        ACE definition (Tian et al., 2003, Eq. 1):
            ACE(v) = Σ_{c∈N(v)} Σ_{w∈N(c)\{v}} (d_w - 2)
        
        where:
        - N(v) = checks adjacent to v
        - N(c) = variables adjacent to c
        - d_w = current degree of variable w
        
        Interpretation:
        - High ACE → v is connected to checks with many high-degree variables
          → strong connectivity → decoder can recover
        - Low ACE → v is in cycles with other low-degree variables
          → weak spot → prone to trapping sets
        
        References
        ----------
        Tian et al. (2003), Section III.A: ACE metric definition.
        """
        ace = 0
        for c in var_adj[v]:
            for w in check_adj[c]:
                if w != v:
                    degree_w = len(var_adj[w])  # Current degree
                    ace += max(0, degree_w - 2)
        return ace
    
    def _ace_detection_viterbi(
        self,
        v_root: int,
        var_adj: List[Set[int]],
        check_adj: List[Set[int]],
        d_ACE: int,
        eta: int,
    ) -> Tuple[bool, List[Tuple[int, ...]]]:
        """
        Viterbi-like ACE detection for cycles up to length 2*d_ACE.
        
        Checks if all cycles involving v_root with length ≤ 2*d_ACE satisfy
        ACE(cycle) ≥ eta. Uses dynamic programming to prune search tree.
        
        Parameters
        ----------
        v_root : int
            Root variable node to check.
        var_adj : List[Set[int]]
            Variable adjacency lists.
        check_adj : List[Set[int]]
            Check adjacency lists.
        d_ACE : int
            Maximum cycle depth (checks cycles up to length 2*d_ACE).
        eta : int
            Minimum required cycle ACE.
        
        Returns
        -------
        passes : bool
            True if all cycles have ACE ≥ eta.
        violations : List[Tuple[int, ...]]
            List of violating cycles (empty if passes=True).
        
        Notes
        -----
        Algorithm (Tian et al., 2003, Fig. 6):
        1. Initialize path ACE tracking: p_t[node] = minimal ACE to reach node
        2. Expand tree in alternating layers (var → check → var → ...)
        3. At each node, update minimal path ACE
        4. When paths meet (cycle detected), check cycle ACE
        5. Prune non-minimal paths (Viterbi-style dynamic programming)
        
        Cycle ACE calculation:
            ACE(cycle) = p_temp + p_t - ACE(v_root) - ACE(junction)
        
        Complexity: O(d_ACE × n × d_max) due to pruning
        
        References
        ----------
        Tian et al. (2003), Fig. 5-6: Viterbi-like ACE detection algorithm.
        """
        # Path ACE tracking: minimal ACE to reach each node from v_root
        p_var: Dict[int, int] = {v_root: self._compute_ace(v_root, var_adj, check_adj)}
        p_check: Dict[int, int] = {}
        
        # Active node tracking (frontier of search)
        active_vars = {v_root}
        active_checks: Set[int] = set()
        
        violations: List[Tuple[int, ...]] = []
        
        for level in range(1, d_ACE + 1):
            # Alternating BFS: var → check → var → check → ...
            if level % 2 == 1:  # Odd level: expand variables to checks
                if not active_vars:
                    break
                next_checks: Set[int] = set()
                for v in active_vars:
                    for c in var_adj[v]:
                        if c not in p_check:
                            p_check[c] = float('inf')  # type: ignore
                        
                        # Path ACE = previous ACE + 0 (checks have ACE=0)
                        p_temp = p_var[v]
                        
                        # Update minimal path ACE (Viterbi pruning)
                        if p_temp < p_check[c]:
                            p_check[c] = p_temp
                            next_checks.add(c)
                
                active_vars = set()
                active_checks = next_checks
            
            else:  # Even level: expand checks to variables
                if not active_checks:
                    break
                next_vars: Set[int] = set()
                for c in active_checks:
                    for w in check_adj[c]:
                        # Compute ACE contribution of w
                        ace_w = self._compute_ace(w, var_adj, check_adj)
                        
                        if w == v_root:
                            # Cycle detected! Check cycle ACE
                            # Cycle ACE = path ACE + root ACE - double-counted root
                            cycle_ace = p_check[c] + ace_w
                            if cycle_ace < eta:
                                # Violating cycle found
                                violations.append((v_root, c, cycle_ace))
                                return False, violations
                            # Don't expand further from root (cycle complete)
                            continue
                        
                        if w not in p_var:
                            p_var[w] = float('inf')  # type: ignore
                        
                        # Path ACE = previous ACE + ACE(w)
                        p_temp = p_check[c] + ace_w
                        
                        # Viterbi pruning: only keep minimal path
                        if p_temp < p_var[w]:
                            p_var[w] = p_temp
                            next_vars.add(w)
                
                active_checks = set()
                active_vars = next_vars
        
        # All cycles checked, none violated
        return True, []
    
    def _place_edge_with_ace(
        self,
        v: int,
        edge_idx: int,
        var_adj: List[Set[int]],
        check_adj: List[Set[int]],
        current_cn_degrees: List[int],
        target_cn_degrees: List[int],
        ace_config: ACEConfig,
    ) -> int:
        """
        Select check node for edge, enforcing ACE constraint.
        
        Hybrid PEG + ACE selection strategy:
        1. Use standard PEG to get candidates (maximize girth)
        2. For each candidate check c, tentatively add edge (v, c)
        3. Run ACE detection on v
        4. If ACE passes: keep c in viable set
        5. Among viable checks, select using PEG criterion (minimize degree)
        
        Parameters
        ----------
        v : int
            Variable node being processed.
        edge_idx : int
            Edge index (0 = first edge, 1 = second edge, ...).
        var_adj, check_adj : List[Set[int]]
            Current adjacency lists.
        current_cn_degrees, target_cn_degrees : List[int]
            Check node degree tracking.
        ace_config : ACEConfig
            ACE parameters (d_ACE, eta, bypass_threshold).
        
        Returns
        -------
        int
            Selected check node index.
        
        Raises
        ------
        RuntimeError
            If no check satisfies ACE constraint (should be rare with proper
            degree distributions).
        
        Notes
        -----
        Bypass optimization (Tian et al., 2003):
        - Variables with degree ≥ eta + 2 automatically satisfy ACE ≥ eta
        - Skip ACE detection for these high-degree variables
        
        References
        ----------
        Tian et al. (2003), Section IV: ACE-PEG algorithm.
        """
        # Current degree includes edges already placed
        current_degree_v = len(var_adj[v]) + 1  # +1 for edge being placed
        
        # Bypass ACE check for high-degree variables
        if current_degree_v >= ace_config.bypass_threshold:
            # ACE(v) ≥ (degree - 2) × neighbors ≥ (bypass_threshold - 2) = eta
            return self._select_check_node(
                current_cn_degrees, target_cn_degrees
            )
        
        # Get PEG candidates (maximize girth)
        if edge_idx == 0:
            # First edge: any check is valid (no cycles yet)
            candidates = list(range(self.m))
        else:
            # Subsequent edges: avoid creating short cycles
            reachable = self._bfs_reachable(v, var_adj, check_adj)
            candidates = [c for c in range(self.m) if c not in reachable]
            if not candidates:
                # All checks reachable (dense graph) → fallback to all checks
                candidates = list(range(self.m))
        
        # Filter candidates by ACE constraint
        viable_checks: List[int] = []
        for c in candidates:
            # Tentatively add edge
            var_adj[v].add(c)
            check_adj[c].add(v)
            
            # Run ACE detection
            passes_ace, _ = self._ace_detection_viterbi(
                v_root=v,
                var_adj=var_adj,
                check_adj=check_adj,
                d_ACE=ace_config.d_ACE,
                eta=ace_config.eta,
            )
            
            # Undo tentative edge
            var_adj[v].remove(c)
            check_adj[c].remove(v)
            
            if passes_ace:
                viable_checks.append(c)
        
        if not viable_checks:
            # No ACE-compliant check found (rare for proper distributions)
            logger.warning(
                "ACE-PEG: Variable %d (degree %d) has no ACE-compliant check. "
                "Falling back to standard PEG (may create trapping sets).",
                v,
                current_degree_v,
            )
            viable_checks = candidates
        
        # Among viable checks, select using PEG criterion (minimize degree)
        return self._select_check_node(
            current_cn_degrees, target_cn_degrees, viable_checks
        )
    
    def _validate_ace_properties(
        self,
        H: sp.csr_matrix,
        var_adj: List[Set[int]],
        check_adj: List[Set[int]],
    ) -> None:
        """
        Validate ACE properties of generated matrix (sample-based).
        
        Parameters
        ----------
        H : scipy.sparse.csr_matrix
            Generated parity-check matrix.
        var_adj : List[Set[int]]
            Variable adjacency lists.
        check_adj : List[Set[int]]
            Check adjacency lists.
        
        Notes
        -----
        Samples random variable nodes and verifies ACE constraint.
        Full validation is O(n × d_ACE × d_max) → prohibitive for large codes.
        Sample size: min(100, n/10) variables.
        """
        logger.info("ACE-PEG: Validating ACE properties (sample-based)...")
        sample_size = min(100, max(10, self.n // 10))
        sampled_vars = self.random.sample(range(self.n), sample_size)
        
        violations = 0
        total_cycles_checked = 0
        
        for v in sampled_vars:
            passes, cycles = self._ace_detection_viterbi(
                v_root=v,
                var_adj=var_adj,
                check_adj=check_adj,
                d_ACE=self.ace_config.d_ACE,
                eta=self.ace_config.eta,
            )
            if not passes:
                violations += len(cycles)
                logger.warning(
                    "ACE-PEG: Variable %d violates ACE constraint (%d cycles)",
                    v,
                    len(cycles),
                )
            total_cycles_checked += 1
        
        compliance_rate = (
            100.0 * (sample_size - violations) / sample_size
            if sample_size > 0
            else 0.0
        )
        
        logger.info(
            "ACE-PEG: ACE compliance: %.1f%% (%d/%d sampled variables pass)",
            compliance_rate,
            sample_size - violations,
            sample_size,
        )
        
        if violations > sample_size * 0.05:  # >5% violation rate
            logger.warning(
                "ACE-PEG: High ACE violation rate (%.1f%%). "
                "Matrix may have suboptimal error floor.",
                100.0 * violations / sample_size,
            )
        
        # Estimate girth (sample-based)
        self._estimate_girth(var_adj, check_adj, sample_size=50)
    
    def _estimate_girth(
        self,
        var_adj: List[Set[int]],
        check_adj: List[Set[int]],
        sample_size: int = 50,
    ) -> int:
        """
        Estimate girth via BFS from sampled variable nodes.
        
        Parameters
        ----------
        var_adj : List[Set[int]]
            Variable adjacency lists.
        check_adj : List[Set[int]]
            Check adjacency lists.
        sample_size : int
            Number of variables to sample.
        
        Returns
        -------
        int
            Estimated girth (shortest cycle length).
        
        Notes
        -----
        Girth estimation is approximate. Full girth computation is NP-hard.
        """
        sampled_vars = self.random.sample(
            range(self.n), min(sample_size, self.n)
        )
        min_cycle = float('inf')
        
        for v in sampled_vars:
            # BFS to find shortest cycle
            visited_vars = {v}
            visited_checks: Set[int] = set()
            queue = [(v, 0)]  # (node, depth)
            
            while queue:
                current_v, depth = queue.pop(0)
                if depth > 20:  # Stop if cycles are very long
                    break
                
                for c in var_adj[current_v]:
                    if c in visited_checks:
                        continue
                    visited_checks.add(c)
                    
                    for w in check_adj[c]:
                        if w == v and depth > 0:
                            # Cycle found
                            cycle_length = 2 * (depth + 1)
                            min_cycle = min(min_cycle, cycle_length)
                        elif w not in visited_vars:
                            visited_vars.add(w)
                            queue.append((w, depth + 1))
        
        girth = int(min_cycle) if min_cycle != float('inf') else -1
        logger.info(
            "ACE-PEG: Estimated girth: %s (sampled %d nodes)",
            girth if girth > 0 else "unknown",
            sample_size,
        )
        return girth


def load_degree_distributions(
    config_path: Path, rate: float
) -> Tuple[DegreeDistribution, DegreeDistribution]:
    """
    Load irregular degree distributions from YAML config.
    
    Parameters
    ----------
    config_path : Path
        Path to ldpc_degree_distributions.yaml.
    rate : float
        Code rate (e.g., 0.5).
    
    Returns
    -------
    lambda_dist : DegreeDistribution
        Variable node degree distribution.
    rho_dist : DegreeDistribution
        Check node degree distribution.
    
    Raises
    ------
    FileNotFoundError
        If config file not found.
    KeyError
        If rate not found in config.
    
    Notes
    -----
    The distributions in ldpc_degree_distributions.yaml are optimized for
    the Binary Symmetric Channel (BSC) via Differential Evolution
    (Elkouss et al., 2009, Table I).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rate_key = f"{rate:.2f}"
    if rate_key not in config:
        available = ', '.join(config.keys())
        raise KeyError(
            f"Rate {rate} not found in config. Available: {available}"
        )
    
    rate_config = config[rate_key]
    
    lambda_dist = DegreeDistribution(
        degrees=rate_config['lambda']['degrees'],
        probabilities=rate_config['lambda']['probabilities'],
    )
    rho_dist = DegreeDistribution(
        degrees=rate_config['rho']['degrees'],
        probabilities=rate_config['rho']['probabilities'],
    )
    
    return lambda_dist, rho_dist


def main() -> int:
    """
    Command-line entry point for ACE mother code generation.
    
    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="Generate ACE-optimized LDPC mother code for rate-compatible reconciliation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=4096,
        help="Codeword length n",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.5,
        help="Mother code rate R₀",
    )
    parser.add_argument(
        "--irregular",
        action="store_true",
        help="Use irregular degree distribution (optimized for BSC)",
    )
    parser.add_argument(
        "--degree-config",
        type=Path,
        default=Path("caligo/configs/ldpc_degree_distributions.yaml"),
        help="Path to degree distribution config",
    )
    parser.add_argument(
        "--d-ace",
        type=int,
        default=8,
        help="Maximum cycle depth for ACE conditioning",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=4,
        help="Minimum cycle ACE threshold",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory or file path. Final filename will follow 'ldpc_{n}_rate{rate:.2f}.npz' convention",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Load degree distributions
        if args.irregular:
            logger.info(
                "Loading irregular degree distributions for rate %.2f from %s",
                args.rate,
                args.degree_config,
            )
            lambda_dist, rho_dist = load_degree_distributions(
                args.degree_config, args.rate
            )
            logger.info(
                "Loaded lambda dist: degrees=%s", lambda_dist.degrees
            )
            logger.info(
                "Loaded rho dist: degrees=%s", rho_dist.degrees
            )
        else:
            # Regular code (for comparison)
            logger.info("Using regular degree distribution (rate %.2f)", args.rate)
            avg_vn_degree = 3
            avg_cn_degree = int(avg_vn_degree / (1 - args.rate))
            lambda_dist = DegreeDistribution(
                degrees=[avg_vn_degree], probabilities=[1.0]
            )
            rho_dist = DegreeDistribution(
                degrees=[avg_cn_degree], probabilities=[1.0]
            )
        
        # Create ACE config
        ace_config = ACEConfig(
            d_ACE=args.d_ace,
            eta=args.eta,
            bypass_threshold=args.eta + 2,
        )
        
        # Generate matrix
        logger.info(
            "Generating ACE-PEG mother code (n=%d, R=%.2f, ACE: d=%d, eta=%d)...",
            args.block_length,
            args.rate,
            args.d_ace,
            args.eta,
        )
        generator = ACEPEGGenerator(
            n=args.block_length,
            rate=args.rate,
            lambda_dist=lambda_dist,
            rho_dist=rho_dist,
            ace_config=ace_config,
            max_tree_depth=50,
            seed=args.seed,
        )
        H = generator.generate()
        
        # Save matrix using enforced naming convention: ldpc_{n}_rate{rate:.2f}.npz
        desired_fname = f"ldpc_{args.block_length}_rate{args.rate:.2f}.npz"
        if args.output_path.exists() and args.output_path.is_dir():
            out_dir = args.output_path
        else:
            out_dir = args.output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / desired_fname
        if out_path != args.output_path:
            logger.info(
                "Output filename adjusted to follow convention: %s -> %s",
                args.output_path,
                out_path,
            )
        sp.save_npz(out_path, H.tocsr())
        logger.info("Saved ACE mother code to %s", out_path)

        logger.info(
            "Matrix properties: shape=%s, nnz=%d, density=%.4f",
            H.shape,
            H.nnz,
            H.nnz / (H.shape[0] * H.shape[1]),
        )
        
        return 0
    
    except Exception as e:
        logger.error("Failed to generate ACE mother code: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

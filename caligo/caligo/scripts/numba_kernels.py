"""Numba-accelerated kernels for PEG / ACE-PEG Tanner graph construction.

This module provides low-level kernels that operate on fixed-width adjacency
arrays. It is intentionally dependency-light and does not import Caligo
internals, so it can be used from multiple offline generation scripts.

Notes
-----
The core representation is:

- ``vn_adj``: int32 array of shape (n, max_vn_degree), filled with neighbor
  check indices or -1 for empty.
- ``cn_adj``: int32 array of shape (m, max_cn_degree), filled with neighbor
  variable indices or -1 for empty.
- ``vn_deg`` / ``cn_deg``: int32 arrays containing the current degree for each
  node (i.e., how many entries in the corresponding adjacency row are valid).

All kernels assume the adjacency rows are *packed* (valid entries stored in
positions ``0..deg-1``).

The kernels are written to avoid allocation in the hot loops. Workspaces should
be allocated in the Python driver and passed in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import numba
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    numba = None  # type: ignore

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

    _NUMBA_AVAILABLE = False


@njit(cache=True)
def _xorshift64star(state: np.uint64) -> np.uint64:
    """Fast deterministic PRNG step (xorshift64*)."""
    x = state
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    return x * np.uint64(2685821657736338717)


@njit(cache=True)
def bfs_mark_reachable_checks(
    v_root: np.int32,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
    max_depth: np.int32,
    visited_vars: np.ndarray,
    visited_checks: np.ndarray,
    frontier_vars: np.ndarray,
    frontier_checks: np.ndarray,
    next_frontier_vars: np.ndarray,
    next_frontier_checks: np.ndarray,
    visit_token: np.int32,
) -> None:
    """Mark all reachable check nodes from ``v_root`` up to ``max_depth``.

    This matches the alternating BFS semantics of ``PEGMatrixGenerator._bfs_reachable``.

    Parameters
    ----------
    v_root : int32
        Starting variable node.
    vn_adj, cn_adj : ndarray
        Fixed-width adjacency arrays.
    vn_deg, cn_deg : ndarray
        Current degrees.
    max_depth : int32
        Maximum BFS depth.
    visited_vars, visited_checks : ndarray
        int32 stamp arrays; a node is considered visited if its entry equals
        ``visit_token``.
    frontier_vars, frontier_checks, next_frontier_vars, next_frontier_checks : ndarray
        Preallocated buffers used as frontiers.
    visit_token : int32
        Stamp value for this BFS.
    """

    visited_vars[v_root] = visit_token
    frontier_vars[0] = v_root
    n_frontier_vars = np.int32(1)
    n_frontier_checks = np.int32(0)

    for depth in range(max_depth):
        if depth % 2 == 0:
            if n_frontier_vars == 0:
                break
            n_next_checks = np.int32(0)
            for i in range(n_frontier_vars):
                v = frontier_vars[i]
                deg_v = vn_deg[v]
                for k in range(deg_v):
                    c = vn_adj[v, k]
                    if c >= 0 and visited_checks[c] != visit_token:
                        visited_checks[c] = visit_token
                        next_frontier_checks[n_next_checks] = c
                        n_next_checks += 1
            n_frontier_vars = np.int32(0)
            # swap
            for i in range(n_next_checks):
                frontier_checks[i] = next_frontier_checks[i]
            n_frontier_checks = n_next_checks

        else:
            if n_frontier_checks == 0:
                break
            n_next_vars = np.int32(0)
            for i in range(n_frontier_checks):
                c = frontier_checks[i]
                deg_c = cn_deg[c]
                for k in range(deg_c):
                    w = cn_adj[c, k]
                    if w >= 0 and visited_vars[w] != visit_token:
                        visited_vars[w] = visit_token
                        next_frontier_vars[n_next_vars] = w
                        n_next_vars += 1
            n_frontier_checks = np.int32(0)
            # swap
            for i in range(n_next_vars):
                frontier_vars[i] = next_frontier_vars[i]
            n_frontier_vars = n_next_vars


@njit(cache=True)
def add_edge_packed(
    v: np.int32,
    c: np.int32,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
) -> None:
    """Add edge (v, c) into packed fixed-width adjacency arrays."""
    if c < 0:
        print("DEBUG: Invalid check node index c=", c)
        raise IndexError("Invalid check node index")
    dv = vn_deg[v]
    dc = cn_deg[c]
    if dv >= vn_adj.shape[1]:
        raise IndexError("vn_adj capacity exceeded")
    if dc >= cn_adj.shape[1]:
        raise IndexError("cn_adj capacity exceeded")
    vn_adj[v, dv] = c
    cn_adj[c, dc] = v
    vn_deg[v] = dv + 1
    cn_deg[c] = dc + 1


@njit(cache=True)
def remove_last_edge_packed(
    v: np.int32,
    c: np.int32,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
) -> None:
    """Remove the most recently added edge for v and c.

    Assumes the last entries in both adjacency rows are the edge (v,c).
    """
    dv = vn_deg[v] - 1
    dc = cn_deg[c] - 1
    vn_adj[v, dv] = np.int32(-1)
    cn_adj[c, dc] = np.int32(-1)
    vn_deg[v] = dv
    cn_deg[c] = dc


@njit(cache=True)
def select_check_min_fill_ratio(
    cn_deg: np.ndarray,
    cn_target_deg: np.ndarray,
    cn_capacity: np.int32,
    visited_checks: np.ndarray,
    visit_token: np.int32,
    restrict_to_unreachable: np.uint8,
    rng_state: np.uint64,
) -> Tuple[np.int32, np.uint64]:
    """Select a check node minimizing (deg/target, deg) with random tie breaks.

    Uses exact fraction comparison via cross-multiplication to avoid float
    instability.

    Parameters
    ----------
    restrict_to_unreachable : uint8
        If 1, ignores checks with visited_checks[c] == visit_token.

    Returns
    -------
    (c, rng_state)
        Selected check index and updated RNG state.
    """
    m = cn_deg.shape[0]
    best_c = np.int32(-1)
    # Initialize with the first eligible check node. Using a large sentinel for
    # best_num can overflow in the cross-multiplication (best_num * tgt) when
    # tgt is moderately large (e.g. 9-13), which can lead to returning -1.
    best_num = np.int64(0)  # current degree numerator
    best_den = np.int64(1)  # target degree denominator
    best_deg = np.int32(0)
    ties = np.int32(0)

    for c in range(m):
        if restrict_to_unreachable == 1 and visited_checks[c] == visit_token:
            continue
        if cn_deg[c] >= cn_capacity:
            continue
        cur = np.int32(cn_deg[c])
        tgt = np.int64(cn_target_deg[c])
        if tgt < 1:
            tgt = np.int64(1)

        if best_c == -1:
            best_c = np.int32(c)
            best_num = np.int64(cur)
            best_den = tgt
            best_deg = cur
            ties = np.int32(1)
            continue

        # Compare cur/tgt vs best_num/best_den via cross-multiplication
        left = np.int64(cur) * best_den
        right = best_num * tgt

        better = False
        if left < right:
            better = True
        elif left == right and cur < best_deg:
            better = True

        if better:
            best_c = np.int32(c)
            best_num = np.int64(cur)
            best_den = tgt
            best_deg = cur
            ties = np.int32(1)
        else:
            # Tie if ratios equal AND degrees equal
            if left == right and cur == best_deg:
                ties += 1
                rng_state = _xorshift64star(rng_state)
                if np.int64(rng_state % np.uint64(ties)) == 0:
                    best_c = np.int32(c)

    if best_c == -1:
        pass

    return best_c, rng_state


@njit(cache=True)
def compute_ace_value(
    v: np.int32,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
) -> np.int32:
    """Compute ACE(v) using the same definition as the Python implementation."""
    ace = np.int32(0)
    deg_v = vn_deg[v]
    for i in range(deg_v):
        c = vn_adj[v, i]
        if c < 0:
            continue
        deg_c = cn_deg[c]
        for j in range(deg_c):
            w = cn_adj[c, j]
            if w < 0 or w == v:
                continue
            dw = vn_deg[w]
            contrib = dw - 2
            if contrib > 0:
                ace += np.int32(contrib)
    return ace


@njit(cache=True)
def ace_detection_viterbi(
    v_root: np.int32,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
    d_ace: np.int32,
    eta: np.int32,
    p_var: np.ndarray,
    p_check: np.ndarray,
    pvar_seen: np.ndarray,
    pcheck_seen: np.ndarray,
    active_vars: np.ndarray,
    active_checks: np.ndarray,
    next_active_vars: np.ndarray,
    next_active_checks: np.ndarray,
    var_in_next: np.ndarray,
    check_in_next: np.ndarray,
    visit_token: np.int32,
) -> np.uint8:
    """Numba port of ACEPEGGenerator._ace_detection_viterbi.

    Returns
    -------
    uint8
        1 if passes, 0 if a violating cycle is detected.
    """
    INF = np.int32(1 << 30)

    root_ace = compute_ace_value(v_root, vn_adj, cn_adj, vn_deg, cn_deg)
    p_var[v_root] = root_ace
    pvar_seen[v_root] = visit_token

    active_vars[0] = v_root
    n_active_vars = np.int32(1)
    n_active_checks = np.int32(0)

    for level in range(1, d_ace + 1):
        if level % 2 == 1:
            if n_active_vars == 0:
                break
            n_next_checks = np.int32(0)
            for i in range(n_active_vars):
                v = active_vars[i]
                pv = p_var[v]
                deg_v = vn_deg[v]
                for k in range(deg_v):
                    c = vn_adj[v, k]
                    if c < 0:
                        continue
                    if pcheck_seen[c] != visit_token:
                        p_check[c] = INF
                        pcheck_seen[c] = visit_token
                    p_temp = pv
                    if p_temp < p_check[c]:
                        p_check[c] = p_temp
                        if check_in_next[c] != visit_token:
                            check_in_next[c] = visit_token
                            next_active_checks[n_next_checks] = c
                            n_next_checks += 1
            n_active_vars = np.int32(0)
            for i in range(n_next_checks):
                active_checks[i] = next_active_checks[i]
            n_active_checks = n_next_checks

        else:
            if n_active_checks == 0:
                break
            n_next_vars = np.int32(0)
            for i in range(n_active_checks):
                c = active_checks[i]
                pc = p_check[c]
                deg_c = cn_deg[c]
                for k in range(deg_c):
                    w = cn_adj[c, k]
                    if w < 0:
                        continue
                    ace_w = compute_ace_value(w, vn_adj, cn_adj, vn_deg, cn_deg)
                    if w == v_root:
                        cycle_ace = pc + ace_w
                        if cycle_ace < eta:
                            return np.uint8(0)
                        continue
                    if pvar_seen[w] != visit_token:
                        p_var[w] = INF
                        pvar_seen[w] = visit_token
                    p_temp = pc + ace_w
                    if p_temp < p_var[w]:
                        p_var[w] = p_temp
                        if var_in_next[w] != visit_token:
                            var_in_next[w] = visit_token
                            next_active_vars[n_next_vars] = w
                            n_next_vars += 1
            n_active_checks = np.int32(0)
            for i in range(n_next_vars):
                active_vars[i] = next_active_vars[i]
            n_active_vars = n_next_vars

    return np.uint8(1)


@njit(cache=True)
def build_peg_graph(
    order: np.ndarray,
    vn_target_deg: np.ndarray,
    cn_target_deg: np.ndarray,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
    max_tree_depth: np.int32,
    visited_vars: np.ndarray,
    visited_checks: np.ndarray,
    frontier_vars: np.ndarray,
    frontier_checks: np.ndarray,
    next_frontier_vars: np.ndarray,
    next_frontier_checks: np.ndarray,
    rng_state: np.uint64,
) -> np.uint64:
    """Build a Tanner graph using the PEG edge placement strategy."""
    visit_token = np.int32(1)
    cn_capacity = np.int32(cn_adj.shape[1])
    for idx in range(order.shape[0]):
        v = np.int32(order[idx])
        deg_v = np.int32(vn_target_deg[v])
        for edge_idx in range(deg_v):
            if edge_idx == 0:
                c, rng_state = select_check_min_fill_ratio(
                    cn_deg,
                    cn_target_deg,
                    cn_capacity,
                    visited_checks,
                    visit_token,
                    np.uint8(0),
                    rng_state,
                )
            else:
                visit_token += 1
                bfs_mark_reachable_checks(
                    v,
                    vn_adj,
                    cn_adj,
                    vn_deg,
                    cn_deg,
                    max_tree_depth,
                    visited_vars,
                    visited_checks,
                    frontier_vars,
                    frontier_checks,
                    next_frontier_vars,
                    next_frontier_checks,
                    visit_token,
                )
                c, rng_state = select_check_min_fill_ratio(
                    cn_deg,
                    cn_target_deg,
                    cn_capacity,
                    visited_checks,
                    visit_token,
                    np.uint8(1),
                    rng_state,
                )
                if c == -1:
                    c, rng_state = select_check_min_fill_ratio(
                        cn_deg,
                        cn_target_deg,
                        cn_capacity,
                        visited_checks,
                        visit_token,
                        np.uint8(0),
                        rng_state,
                    )

            # Defensive: never allow invalid check index.
            # If this triggers, it indicates there is no check node with
            # remaining capacity.
            if c == -1:
                raise IndexError("no available check node")

            add_edge_packed(v, c, vn_adj, cn_adj, vn_deg, cn_deg)

    return rng_state


@njit(cache=True)
def build_ace_peg_graph(
    order: np.ndarray,
    vn_target_deg: np.ndarray,
    cn_target_deg: np.ndarray,
    vn_adj: np.ndarray,
    cn_adj: np.ndarray,
    vn_deg: np.ndarray,
    cn_deg: np.ndarray,
    max_tree_depth: np.int32,
    d_ace: np.int32,
    eta: np.int32,
    bypass_threshold: np.int32,
    visited_vars: np.ndarray,
    visited_checks: np.ndarray,
    frontier_vars: np.ndarray,
    frontier_checks: np.ndarray,
    next_frontier_vars: np.ndarray,
    next_frontier_checks: np.ndarray,
    # ACE workspaces
    p_var: np.ndarray,
    p_check: np.ndarray,
    pvar_seen: np.ndarray,
    pcheck_seen: np.ndarray,
    active_vars: np.ndarray,
    active_checks: np.ndarray,
    next_active_vars: np.ndarray,
    next_active_checks: np.ndarray,
    var_in_next: np.ndarray,
    check_in_next: np.ndarray,
    rng_state: np.uint64,
) -> Tuple[np.uint64, np.int64, np.int64]:
    """Build a Tanner graph using ACE-PEG selection.

    Returns
    -------
    (rng_state, ace_checks_performed, ace_fallbacks)
    """
    bfs_token = np.int32(1)
    ace_token = np.int32(1)
    ace_checks_performed = np.int64(0)
    ace_fallbacks = np.int64(0)
    cn_capacity = np.int32(cn_adj.shape[1])

    for idx in range(order.shape[0]):
        v = np.int32(order[idx])
        deg_v = np.int32(vn_target_deg[v])
        for edge_idx in range(deg_v):
            # current_degree_v in the Python code is len(var_adj[v]) + 1
            current_degree_v = np.int32(vn_deg[v] + 1)

            # Bypass optimization
            if current_degree_v >= bypass_threshold:
                c, rng_state = select_check_min_fill_ratio(
                    cn_deg,
                    cn_target_deg,
                    cn_capacity,
                    visited_checks,
                    bfs_token,
                    np.uint8(0),
                    rng_state,
                )
                add_edge_packed(v, c, vn_adj, cn_adj, vn_deg, cn_deg)
                continue

            ace_checks_performed += 1

            restrict = np.uint8(0)
            if edge_idx != 0:
                bfs_token += 1
                bfs_mark_reachable_checks(
                    v,
                    vn_adj,
                    cn_adj,
                    vn_deg,
                    cn_deg,
                    max_tree_depth,
                    visited_vars,
                    visited_checks,
                    frontier_vars,
                    frontier_checks,
                    next_frontier_vars,
                    next_frontier_checks,
                    bfs_token,
                )
                restrict = np.uint8(1)

            # Filter candidates by ACE constraint and select best by PEG criterion
            best_c = np.int32(-1)
            best_num = np.int64(1 << 60)
            best_den = np.int64(1)
            best_deg = np.int32(1 << 30)
            ties = np.int32(0)
            found_viable = np.uint8(0)

            m = cn_deg.shape[0]
            for c_idx in range(m):
                if restrict == 1 and visited_checks[c_idx] == bfs_token:
                    continue
                if cn_deg[c_idx] >= cn_capacity:
                    continue

                c_idx32 = np.int32(c_idx)

                # Tentatively add edge
                add_edge_packed(v, c_idx32, vn_adj, cn_adj, vn_deg, cn_deg)

                # ACE detection
                ace_token += 1
                passes = ace_detection_viterbi(
                    v,
                    vn_adj,
                    cn_adj,
                    vn_deg,
                    cn_deg,
                    d_ace,
                    eta,
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
                    ace_token,
                )

                # Undo tentative edge
                remove_last_edge_packed(v, c_idx32, vn_adj, cn_adj, vn_deg, cn_deg)

                if passes == 0:
                    continue

                found_viable = np.uint8(1)

                cur = np.int32(cn_deg[c_idx])
                tgt = np.int64(cn_target_deg[c_idx])
                if tgt < 1:
                    tgt = np.int64(1)

                if best_c == -1:
                    best_c = c_idx32
                    best_num = np.int64(cur)
                    best_den = tgt
                    best_deg = cur
                    ties = np.int32(1)
                    continue

                left = np.int64(cur) * best_den
                right = best_num * tgt

                better = False
                if left < right:
                    better = True
                elif left == right and cur < best_deg:
                    better = True

                if better:
                    best_c = c_idx32
                    best_num = np.int64(cur)
                    best_den = tgt
                    best_deg = cur
                    ties = np.int32(1)
                else:
                    if left == right and cur == best_deg:
                        ties += 1
                        rng_state = _xorshift64star(rng_state)
                        if np.int64(rng_state % np.uint64(ties)) == 0:
                            best_c = c_idx32

            final_c = best_c

            if found_viable == 0:
                ace_fallbacks += 1
                final_c, rng_state = select_check_min_fill_ratio(
                    cn_deg,
                    cn_target_deg,
                    cn_capacity,
                    visited_checks,
                    bfs_token,
                    restrict,
                    rng_state,
                )
                if final_c == -1:
                    final_c, rng_state = select_check_min_fill_ratio(
                        cn_deg,
                        cn_target_deg,
                        cn_capacity,
                        visited_checks,
                        bfs_token,
                        np.uint8(0),
                        rng_state,
                    )
                
                if final_c == -1:
                    raise IndexError("No check node found")

            add_edge_packed(v, final_c, vn_adj, cn_adj, vn_deg, cn_deg)

    return rng_state, ace_checks_performed, ace_fallbacks


@njit(cache=True)
def fill_edges_from_cn_adj(
    cn_adj: np.ndarray,
    cn_deg: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    """Fill COO row/col arrays from packed cn adjacency."""
    idx = np.int64(0)
    m = cn_adj.shape[0]
    for c in range(m):
        deg_c = cn_deg[c]
        for k in range(deg_c):
            v = cn_adj[c, k]
            if v < 0:
                continue
            rows[idx] = np.int32(c)
            cols[idx] = np.int32(v)
            idx += 1


def numba_available() -> bool:
    """Return True if Numba is importable in this environment."""
    return _NUMBA_AVAILABLE

# ============================================================================
# RECONCILIATION KERNELS (Phase 0)
# ============================================================================

@njit(cache=True)
def encode_bitpacked_kernel(
    packed_frame: np.ndarray,
    check_row_ptr: np.ndarray,
    check_col_idx: np.ndarray,
    n_checks: int,
) -> np.ndarray:
    """
    Compute syndrome using bit-packed SpMV kernel.
    
    Parameters
    ----------
    packed_frame : np.ndarray
        Input frame packed into uint64 words.
    check_row_ptr : np.ndarray
        CSR row pointers for parity check matrix.
    check_col_idx : np.ndarray
        CSR column indices for parity check matrix.
    n_checks : int
        Number of check nodes (rows).
        
    Returns
    -------
    np.ndarray
        Packed syndrome (uint64 array).
    """
    # Calculate output size (packed)
    n_words_out = (n_checks + 63) // 64
    packed_syndrome = np.zeros(n_words_out, dtype=np.uint64)
    
    for r in range(n_checks):
        # Compute parity for row r
        row_parity = np.uint64(0)
        
        # Iterate over non-zero columns in this row
        start = check_row_ptr[r]
        end = check_row_ptr[r + 1]
        
        for k in range(start, end):
            c = check_col_idx[k]
            
            # Extract bit c from packed_frame
            word_idx = c // 64
            bit_idx = c % 64
            bit_val = (packed_frame[word_idx] >> np.uint64(bit_idx)) & np.uint64(1)
            
            row_parity ^= bit_val
            
        # Store parity bit in packed_syndrome
        if row_parity:
            out_word_idx = r // 64
            out_bit_idx = r % 64
            packed_syndrome[out_word_idx] |= (np.uint64(1) << np.uint64(out_bit_idx))
            
    return packed_syndrome


@njit(cache=True, fastmath=True)
def decode_bp_virtual_graph_kernel(
    llr: np.ndarray,
    syndrome: np.ndarray,
    messages: np.ndarray,
    check_row_ptr: np.ndarray,
    check_col_idx: np.ndarray,
    var_col_ptr: np.ndarray,
    var_row_idx: np.ndarray,
    edge_c2v: np.ndarray,
    edge_v2c: np.ndarray,
    max_iterations: int,
) -> Tuple[np.ndarray, bool, int]:
    """
    Belief Propagation decoder for Baseline protocol (Virtual Graph).
    """
    n_vars = len(llr)
    n_checks = len(syndrome)
    n_edges = len(check_col_idx)
    
    OFFSET_C2V = 0
    OFFSET_V2C = n_edges
    
    converged = False
    iterations = 0
    corrected_bits = np.zeros(n_vars, dtype=np.uint8)
    
    # Initialize C2V to 0
    messages[OFFSET_C2V : OFFSET_C2V + n_edges] = 0.0
    
    for it in range(max_iterations):
        iterations += 1
        
        # --- Variable Node Step ---
        for v in range(n_vars):
            start = var_col_ptr[v]
            end = var_col_ptr[v+1]
            
            # Compute sum of incoming C2V messages
            c2v_sum = 0.0
            for k in range(start, end):
                edge_idx = edge_v2c[k]
                c2v_sum += messages[OFFSET_C2V + edge_idx]
            
            # Total LLR for this variable
            total_llr = llr[v] + c2v_sum
            
            # Hard decision
            corrected_bits[v] = 1 if total_llr < 0 else 0
            
            # Compute outgoing V2C messages
            for k in range(start, end):
                edge_idx = edge_v2c[k]
                val = total_llr - messages[OFFSET_C2V + edge_idx]
                
                # Clip to avoid overflow
                if val > 30.0: val = 30.0
                elif val < -30.0: val = -30.0
                
                messages[OFFSET_V2C + edge_idx] = val
                
        # --- Syndrome Check ---
        syndrome_ok = True
        for c in range(n_checks):
            start = check_row_ptr[c]
            end = check_row_ptr[c+1]
            parity = 0
            for k in range(start, end):
                v = check_col_idx[k]
                parity ^= corrected_bits[v]
            
            if parity != syndrome[c]:
                syndrome_ok = False
                break
        
        if syndrome_ok:
            converged = True
            break
            
        # --- Check Node Step ---
        # All messages indexed by CSR edge index (k is already CSR index)
        for c in range(n_checks):
            start = check_row_ptr[c]
            end = check_row_ptr[c+1]
            
            total_tanh = 1.0
            zero_count = 0
            
            for k in range(start, end):
                val = messages[OFFSET_V2C + k]
                t = np.tanh(val / 2.0)
                if np.abs(t) < 1e-12:
                    zero_count += 1
                else:
                    total_tanh *= t
            
            for k in range(start, end):
                val = messages[OFFSET_V2C + k]
                t = np.tanh(val / 2.0)
                
                res = 0.0
                if zero_count > 1:
                    res = 0.0
                elif zero_count == 1:
                    if np.abs(t) < 1e-12:
                        sub_prod = 1.0
                        for k2 in range(start, end):
                            if k2 == k: continue
                            val2 = messages[OFFSET_V2C + k2]
                            sub_prod *= np.tanh(val2 / 2.0)
                        res = 2.0 * np.arctanh(sub_prod)
                    else:
                        res = 0.0
                else:
                    sub_prod = total_tanh / t
                    if sub_prod > 0.999999999999: sub_prod = 0.999999999999
                    if sub_prod < -0.999999999999: sub_prod = -0.999999999999
                    res = 2.0 * np.arctanh(sub_prod)
                
                if syndrome[c] == 1:
                    res = -res
                    
                messages[OFFSET_C2V + k] = res

    return corrected_bits, converged, iterations


@njit(cache=True, fastmath=True)
def decode_bp_hotstart_kernel(
    llr: np.ndarray,
    syndrome: np.ndarray,
    messages: np.ndarray,
    frozen_mask: np.ndarray,
    check_row_ptr: np.ndarray,
    check_col_idx: np.ndarray,
    var_col_ptr: np.ndarray,
    var_row_idx: np.ndarray,
    edge_c2v: np.ndarray,
    edge_v2c: np.ndarray,
    max_iterations: int,
) -> Tuple[np.ndarray, bool, int, np.ndarray]:
    """
    Hot-Start BP decoder with Freeze optimization.
    """
    n_vars = len(llr)
    n_checks = len(syndrome)
    n_edges = len(check_col_idx)
    
    OFFSET_C2V = 0
    OFFSET_V2C = n_edges
    
    converged = False
    iterations = 0
    corrected_bits = np.zeros(n_vars, dtype=np.uint8)
    
    for it in range(max_iterations):
        iterations += 1
        
        # --- Variable Node Step ---
        for v in range(n_vars):
            start = var_col_ptr[v]
            end = var_col_ptr[v+1]
            
            if frozen_mask[v]:
                val = llr[v]
                if val > 30.0: val = 30.0
                elif val < -30.0: val = -30.0
                
                for k in range(start, end):
                    edge_idx = edge_v2c[k]
                    messages[OFFSET_V2C + edge_idx] = val
                
                corrected_bits[v] = 1 if llr[v] < 0 else 0
                continue
            
            c2v_sum = 0.0
            for k in range(start, end):
                edge_idx = edge_v2c[k]
                c2v_sum += messages[OFFSET_C2V + edge_idx]
            
            total_llr = llr[v] + c2v_sum
            corrected_bits[v] = 1 if total_llr < 0 else 0
            
            for k in range(start, end):
                edge_idx = edge_v2c[k]
                val = total_llr - messages[OFFSET_C2V + edge_idx]
                if val > 30.0: val = 30.0
                elif val < -30.0: val = -30.0
                messages[OFFSET_V2C + edge_idx] = val
                
        # --- Syndrome Check ---
        syndrome_ok = True
        for c in range(n_checks):
            start = check_row_ptr[c]
            end = check_row_ptr[c+1]
            parity = 0
            for k in range(start, end):
                v = check_col_idx[k]
                parity ^= corrected_bits[v]
            if parity != syndrome[c]:
                syndrome_ok = False
                break
        
        if syndrome_ok:
            converged = True
            break
            
        # --- Check Node Step ---
        # All messages indexed by CSR edge index (k is already CSR index)
        for c in range(n_checks):
            start = check_row_ptr[c]
            end = check_row_ptr[c+1]
            
            total_tanh = 1.0
            zero_count = 0
            
            for k in range(start, end):
                val = messages[OFFSET_V2C + k]
                t = np.tanh(val / 2.0)
                if np.abs(t) < 1e-12:
                    zero_count += 1
                else:
                    total_tanh *= t
            
            for k in range(start, end):
                val = messages[OFFSET_V2C + k]
                t = np.tanh(val / 2.0)
                
                res = 0.0
                if zero_count > 1:
                    res = 0.0
                elif zero_count == 1:
                    if np.abs(t) < 1e-12:
                        sub_prod = 1.0
                        for k2 in range(start, end):
                            if k2 == k: continue
                            val2 = messages[OFFSET_V2C + k2]
                            sub_prod *= np.tanh(val2 / 2.0)
                        res = 2.0 * np.arctanh(sub_prod)
                    else:
                        res = 0.0
                else:
                    sub_prod = total_tanh / t
                    if sub_prod > 0.999999999999: sub_prod = 0.999999999999
                    if sub_prod < -0.999999999999: sub_prod = -0.999999999999
                    res = 2.0 * np.arctanh(sub_prod)
                
                if syndrome[c] == 1:
                    res = -res
                    
                messages[OFFSET_C2V + k] = res

    return corrected_bits, converged, iterations, messages

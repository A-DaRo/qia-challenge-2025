import numpy as np
import pytest

from caligo.scripts.generate_ace_mother_code import ACEConfig, ACEPEGGenerator
from caligo.scripts.peg_generator import DegreeDistribution, PEGMatrixGenerator
from caligo.scripts import numba_kernels


@pytest.mark.skipif(not numba_kernels.numba_available(), reason="numba not available")
def test_bfs_reachable_matches_python() -> None:
    """Numba BFS reachable set matches Python set-based BFS."""
    pytest.importorskip("numba")

    rng = np.random.default_rng(123)
    n = 30
    m = 15
    max_v = 4
    max_c = 7

    # Build a small random Tanner graph with both representations.
    var_adj = [set() for _ in range(n)]
    check_adj = [set() for _ in range(m)]

    vn_adj = np.full((n, max_v), -1, dtype=np.int32)
    cn_adj = np.full((m, max_c), -1, dtype=np.int32)
    vn_deg = np.zeros(n, dtype=np.int32)
    cn_deg = np.zeros(m, dtype=np.int32)

    target_edges = min(n * 2, n * max_v, m * max_c)
    attempts = 0
    while sum(vn_deg) < target_edges and attempts < target_edges * 50:
        attempts += 1
        v = int(rng.integers(0, n))
        c = int(rng.integers(0, m))
        if c in var_adj[v]:
            continue
        if vn_deg[v] >= max_v or cn_deg[c] >= max_c:
            continue
        var_adj[v].add(c)
        check_adj[c].add(v)
        vn_adj[v, vn_deg[v]] = np.int32(c)
        cn_adj[c, cn_deg[c]] = np.int32(v)
        vn_deg[v] += 1
        cn_deg[c] += 1

    gen = PEGMatrixGenerator(
        n=n,
        rate=0.5,
        lambda_dist=DegreeDistribution(degrees=[2], probabilities=[1.0]),
        rho_dist=DegreeDistribution(degrees=[4], probabilities=[1.0]),
        max_tree_depth=10,
        seed=1,
    )

    v_root = int(rng.integers(0, n))
    reachable_py = gen._bfs_reachable(v_root, var_adj, check_adj)

    visited_vars = np.zeros(n, dtype=np.int32)
    visited_checks = np.zeros(m, dtype=np.int32)
    frontier_vars = np.empty(n, dtype=np.int32)
    next_frontier_vars = np.empty(n, dtype=np.int32)
    frontier_checks = np.empty(m, dtype=np.int32)
    next_frontier_checks = np.empty(m, dtype=np.int32)

    token = np.int32(42)
    numba_kernels.bfs_mark_reachable_checks(
        np.int32(v_root),
        vn_adj,
        cn_adj,
        vn_deg,
        cn_deg,
        np.int32(gen.max_tree_depth),
        visited_vars,
        visited_checks,
        frontier_vars,
        frontier_checks,
        next_frontier_vars,
        next_frontier_checks,
        token,
    )

    reachable_nb = {c for c in range(m) if visited_checks[c] == token}
    assert reachable_nb == reachable_py


@pytest.mark.skipif(not numba_kernels.numba_available(), reason="numba not available")
def test_ace_viterbi_matches_python_pass_fail() -> None:
    """Numba ACE Viterbi pass/fail matches Python implementation."""
    pytest.importorskip("numba")

    rng = np.random.default_rng(456)
    n = 24
    m = 12
    max_v = 5
    max_c = 8

    var_adj = [set() for _ in range(n)]
    check_adj = [set() for _ in range(m)]

    vn_adj = np.full((n, max_v), -1, dtype=np.int32)
    cn_adj = np.full((m, max_c), -1, dtype=np.int32)
    vn_deg = np.zeros(n, dtype=np.int32)
    cn_deg = np.zeros(m, dtype=np.int32)

    target_edges = min(n * 2, n * max_v, m * max_c)
    attempts = 0
    while sum(vn_deg) < target_edges and attempts < target_edges * 80:
        attempts += 1
        v = int(rng.integers(0, n))
        c = int(rng.integers(0, m))
        if c in var_adj[v]:
            continue
        if vn_deg[v] >= max_v or cn_deg[c] >= max_c:
            continue
        var_adj[v].add(c)
        check_adj[c].add(v)
        vn_adj[v, vn_deg[v]] = np.int32(c)
        cn_adj[c, cn_deg[c]] = np.int32(v)
        vn_deg[v] += 1
        cn_deg[c] += 1

    ace_gen = ACEPEGGenerator(
        n=n,
        rate=0.5,
        lambda_dist=DegreeDistribution(degrees=[2], probabilities=[1.0]),
        rho_dist=DegreeDistribution(degrees=[4], probabilities=[1.0]),
        ace_config=ACEConfig(d_ACE=6, eta=3, bypass_threshold=5),
        max_tree_depth=10,
        seed=7,
    )

    v_root = int(rng.integers(0, n))
    passes_py, _ = ace_gen._ace_detection_viterbi(
        v_root=v_root,
        var_adj=var_adj,
        check_adj=check_adj,
        d_ACE=ace_gen.ace_config.d_ACE,
        eta=ace_gen.ace_config.eta,
    )

    p_var = np.empty(n, dtype=np.int32)
    p_check = np.empty(m, dtype=np.int32)
    pvar_seen = np.zeros(n, dtype=np.int32)
    pcheck_seen = np.zeros(m, dtype=np.int32)
    active_vars = np.empty(n, dtype=np.int32)
    next_active_vars = np.empty(n, dtype=np.int32)
    active_checks = np.empty(m, dtype=np.int32)
    next_active_checks = np.empty(m, dtype=np.int32)

    token = np.int32(100)
    passes_nb = numba_kernels.ace_detection_viterbi(
        np.int32(v_root),
        vn_adj,
        cn_adj,
        vn_deg,
        cn_deg,
        np.int32(ace_gen.ace_config.d_ACE),
        np.int32(ace_gen.ace_config.eta),
        p_var,
        p_check,
        pvar_seen,
        pcheck_seen,
        active_vars,
        active_checks,
        next_active_vars,
        next_active_checks,
        token,
    )

    assert bool(passes_nb) is bool(passes_py)

import numpy as np
import scipy.sparse as sp

from caligo.scripts.peg_generator import PEGMatrixGenerator, DegreeDistribution
from caligo.scripts import numba_kernels


def test_peg_numba_retries_on_capacity_exceeded(monkeypatch):
    # Small synthetic instance
    gen = PEGMatrixGenerator(
        n=8,
        rate=0.5,
        lambda_dist=DegreeDistribution(degrees=[1], probabilities=[1.0]),
        rho_dist=DegreeDistribution(degrees=[2], probabilities=[1.0]),
        max_tree_depth=2,
        seed=1,
    )

    vn_degrees = [1] * gen.n
    cn_target_degrees = [2] * gen.m
    total_edges = sum(vn_degrees)

    calls = {"count": 0}

    def fake_build_peg_graph(
        *,
        order,
        vn_target_deg,
        cn_target_deg,
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
        rng_state,
    ):
        calls["count"] += 1

        # Force a single retry: initial width is expected to be 32, so fail it.
        if cn_adj.shape[1] < 64:
            raise IndexError("fixed-width adjacency capacity exceeded")

        # Fill in a simple deterministic graph: connect v -> (v % m)
        vn_deg[:] = 0
        cn_deg[:] = 0
        vn_adj[:, :] = -1
        cn_adj[:, :] = -1

        for v in range(int(vn_target_deg.size)):
            c = v % int(cn_deg.size)
            vn_adj[v, vn_deg[v]] = c
            vn_deg[v] += 1
            cn_adj[c, cn_deg[c]] = v
            cn_deg[c] += 1

    def fake_fill_edges_from_cn_adj(*, cn_adj, cn_deg, rows, cols):
        idx = 0
        for c in range(cn_adj.shape[0]):
            for j in range(int(cn_deg[c])):
                rows[idx] = c
                cols[idx] = cn_adj[c, j]
                idx += 1

    monkeypatch.setattr(numba_kernels, "build_peg_graph", fake_build_peg_graph)
    monkeypatch.setattr(numba_kernels, "fill_edges_from_cn_adj", fake_fill_edges_from_cn_adj)

    H = gen._generate_numba(vn_degrees, cn_target_degrees, total_edges)

    assert isinstance(H, sp.csr_matrix)
    assert H.shape == (gen.m, gen.n)
    assert H.nnz == total_edges
    assert calls["count"] >= 2

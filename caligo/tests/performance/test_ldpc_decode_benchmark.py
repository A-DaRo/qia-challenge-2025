"""Micro-benchmark harness for LDPC decoding.

This is intentionally skip-by-default to avoid flaky timing assertions.
Run with:

    RUN_PERF=1 /home/adaro/projects/qia_25/qia/bin/python -m pytest -m performance -k ldpc_decode_benchmark -s
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.ldpc_decoder import BeliefPropagationDecoder, build_channel_llr
from caligo.reconciliation.matrix_manager import MatrixManager


@pytest.mark.performance
def test_ldpc_decode_benchmark() -> None:
    mm = MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
    H = mm.get_matrix(0.70)
    compiled = mm.get_compiled(0.70)
    decoder = BeliefPropagationDecoder(parity_check_matrix=H, max_iterations=40)

    n = int(H.shape[1])
    m = int(H.shape[0])
    rng = np.random.default_rng(1234)
    bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    syndrome = np.zeros(m, dtype=np.uint8)
    punctured_mask = np.zeros(n, dtype=bool)
    llr = build_channel_llr(bits, qber=0.03, punctured_mask=punctured_mask)

    # Always run a lightweight non-skipping smoke check.
    res = decoder.decode(llr, syndrome, H=compiled)
    assert res.corrected_bits.shape[0] == n
    assert res.syndrome_errors >= 0

    # Optional timing loop (does not affect pass/fail).
    if os.environ.get("RUN_PERF") == "1":
        runs = 10
        t0 = time.perf_counter()
        for _ in range(runs):
            decoder.decode(llr, syndrome, H=compiled)
        dt = time.perf_counter() - t0

        per = dt / runs
        print(f"LDPC decode: {per*1e3:.2f} ms per call (n={n})")

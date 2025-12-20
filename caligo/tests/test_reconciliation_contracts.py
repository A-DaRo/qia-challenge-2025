"""Contract tests for Phase III reconciliation + LDPC.

These tests focus on deterministic construction, algebraic invariants, and
failure semantics. They are intentionally lightweight and avoid end-to-end
simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.block_reconciler import BlockReconciler, BlockReconcilerConfig
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.ldpc_decoder import BeliefPropagationDecoder
from caligo.reconciliation.ldpc_encoder import generate_padding, prepare_frame
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.types.exceptions import LeakageBudgetExceeded


def _matrix_manager() -> MatrixManager:
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


def test_prepare_frame_deterministic_for_seed() -> None:
    rng = np.random.default_rng(123)
    payload_len = 1024
    payload = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
    n_shortened = 4096 - payload_len
    seed = 7 + constants.SEED_OFFSET

    frame_1 = prepare_frame(payload, n_shortened=n_shortened, prng_seed=seed)
    frame_2 = prepare_frame(payload, n_shortened=n_shortened, prng_seed=seed)

    assert np.array_equal(frame_1, frame_2)

    padding = generate_padding(n_shortened, seed)
    assert np.array_equal(frame_1[payload_len:], padding)


def test_syndrome_linearity_xor() -> None:
    mm = _matrix_manager()
    compiled = mm.get_compiled(0.70)

    n = int(compiled.n)
    rng = np.random.default_rng(321)
    x = rng.integers(0, 2, size=n, dtype=np.uint8)
    y = rng.integers(0, 2, size=n, dtype=np.uint8)

    sx = compiled.compute_syndrome(x)
    sy = compiled.compute_syndrome(y)
    sxy = compiled.compute_syndrome(np.bitwise_xor(x, y))

    assert np.array_equal(sxy, np.bitwise_xor(sx, sy))


def test_decoder_converged_implies_syndrome_match() -> None:
    mm = _matrix_manager()
    compiled = mm.get_compiled(0.70)
    decoder = BeliefPropagationDecoder(max_iterations=25)

    n = int(compiled.n)
    rng = np.random.default_rng(999)
    true_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    target = compiled.compute_syndrome(true_bits)

    # Strong, deterministic LLRs matching the true bits.
    llr = np.where(true_bits == 0, 50.0, -50.0).astype(np.float64)

    res = decoder.decode(llr, target, H=compiled)
    if res.converged:
        assert compiled.count_syndrome_errors(res.corrected_bits, target) == 0


def test_leakage_budget_exceeded_is_fatal() -> None:
    mm = _matrix_manager()
    H = mm.get_matrix(0.70)

    decoder = BeliefPropagationDecoder(max_iterations=10)
    hv = PolynomialHashVerifier(hash_bits=constants.LDPC_HASH_BITS)

    # Tiny cap ensures we exceed it as soon as we record a block.
    tracker = LeakageTracker(safety_cap=1)

    reconciler = BlockReconciler(
        matrix_manager=mm,
        decoder=decoder,
        hash_verifier=hv,
        leakage_tracker=tracker,
        config=BlockReconcilerConfig(
            frame_size=constants.LDPC_FRAME_SIZE,
            max_iterations=10,
            max_retries=0,
            hash_bits=constants.LDPC_HASH_BITS,
            f_crit=constants.LDPC_F_CRIT,
            verify=True,
            enforce_leakage_cap=True,
        ),
    )

    rng = np.random.default_rng(2025)
    payload_len = 1024
    alice = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
    bob = alice.copy()

    with pytest.raises(LeakageBudgetExceeded):
        reconciler.reconcile_baseline(alice_key=alice, bob_key=bob, qber_estimate=0.01, block_id=0)

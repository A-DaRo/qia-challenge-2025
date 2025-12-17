"""Unit tests for LDPC reconciliation components."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ehok.core import constants
from ehok.core.exceptions import MatrixSynchronizationError
from ehok.implementations.reconciliation import (
    LDPCMatrixManager,
    LDPCReconciliator,
    PEGMatrixGenerator,
    DegreeDistribution,
)


FRAME_SIZE = 32
RATE = 0.50


def _write_test_matrix(tmp_path) -> LDPCMatrixManager:
    lambda_dist = DegreeDistribution(degrees=[2, 3], probabilities=[0.5, 0.5])
    rho_dist = DegreeDistribution(degrees=[4, 5], probabilities=[0.6, 0.4])
    generator = PEGMatrixGenerator(
        n=FRAME_SIZE,
        rate=RATE,
        lambda_dist=lambda_dist,
        rho_dist=rho_dist,
        max_tree_depth=3,
        seed=constants.PEG_DEFAULT_SEED,
    )
    H = generator.generate()
    path = tmp_path / constants.LDPC_MATRIX_FILE_PATTERN.format(frame_size=FRAME_SIZE, rate=RATE)
    sp.save_npz(path, H)
    return LDPCMatrixManager.from_directory(tmp_path, frame_size=FRAME_SIZE, rates=(RATE,))


def test_peg_generator_respects_regular_degrees():
    lambda_dist = DegreeDistribution(degrees=[2], probabilities=[1.0])
    rho_dist = DegreeDistribution(degrees=[4], probabilities=[1.0])
    generator = PEGMatrixGenerator(
        n=12,
        rate=0.50,
        lambda_dist=lambda_dist,
        rho_dist=rho_dist,
        max_tree_depth=4,
        seed=123,
    )

    H = generator.generate()

    row_degrees = np.array(H.sum(axis=1)).ravel()
    col_degrees = np.array(H.sum(axis=0)).ravel()

    assert H.shape == (6, 12)
    assert np.all(row_degrees == 4)
    assert np.all(col_degrees == 2)


def test_degree_distribution_normalization():
    # Provide invalid sum > 1.0 and expect normalization to 1.0
    dist = DegreeDistribution(degrees=[1, 2], probabilities=[0.6, 0.6])
    assert abs(sum(dist.probabilities) - 1.0) < 1e-6
    # Ensure sampled degrees for generator do not error
    gen = PEGMatrixGenerator(n=10, rate=0.5, lambda_dist=dist, rho_dist=DegreeDistribution(degrees=[3], probabilities=[1.0]), max_tree_depth=3, seed=1)
    H = gen.generate()
    assert H.shape[1] == 10


def test_constants_load_normalized_distributions():
    # Ensure distributions loaded from constants were normalized
    for rate, d in constants.LDPC_DEGREE_DISTRIBUTIONS.items():
        lam = d["lambda"]["probabilities"]
        rho = d["rho"]["probabilities"]
        assert abs(sum(lam) - 1.0) < 1e-6
        assert abs(sum(rho) - 1.0) < 1e-6


def test_matrix_manager_checksum_and_access(tmp_path):
    manager = _write_test_matrix(tmp_path)
    assert manager.frame_size == FRAME_SIZE
    assert np.isclose(manager.rates, np.array([RATE])).all()

    H = manager.get_matrix(RATE)
    assert H.shape == (int(round(FRAME_SIZE * (1 - RATE))), FRAME_SIZE)
    assert manager.checksum

    with pytest.raises(MatrixSynchronizationError):
        manager.verify_checksum("deadbeef")


def test_ldpc_reconciliator_block_roundtrip(tmp_path):
    manager = _write_test_matrix(tmp_path)
    reconciliator = LDPCReconciliator(manager)
    rng = np.random.default_rng(1234)

    payload_len = 20
    key_block = rng.integers(0, 2, size=payload_len, dtype=np.uint8)

    rate = reconciliator.select_rate(0.05)
    n_short = reconciliator.compute_shortening(rate, 0.05, payload_len)
    syndrome = reconciliator.compute_syndrome_block(key_block, rate, n_short, prng_seed=7)

    corrected, converged, error_count = reconciliator.reconcile_block(
        key_block, syndrome, rate, n_short, prng_seed=7
    )

    assert converged
    assert error_count == 0
    assert np.array_equal(corrected, key_block)


def test_ldpc_reconciliator_hash_and_leakage(tmp_path):
    manager = _write_test_matrix(tmp_path)
    reconciliator = LDPCReconciliator(manager)

    key_block = np.zeros(16, dtype=np.uint8)
    rate = reconciliator.select_rate(0.02)
    n_short = reconciliator.compute_shortening(rate, 0.02, len(key_block))
    syndrome = reconciliator.compute_syndrome_block(key_block, rate, n_short, prng_seed=11)

    # Basic leakage (without enhanced accounting)
    leak = reconciliator.estimate_leakage_block(len(syndrome), reconciliator.hash_verifier.hash_length_bits)
    # Enhanced leakage includes: base (syndrome + hash) + log2(|rates|)
    # The leakage should be >= len(syndrome) + hash_bits
    base_leakage = len(syndrome) + reconciliator.hash_verifier.hash_length_bits
    assert leak >= base_leakage, f"Leakage {leak} should be >= base {base_leakage}"

    hash_value = reconciliator.hash_verifier.compute_hash(key_block, seed=11)
    assert isinstance(hash_value, bytes)
    assert len(hash_value) == (constants.LDPC_HASH_BITS // 8) + (1 if constants.LDPC_HASH_BITS % 8 else 0)

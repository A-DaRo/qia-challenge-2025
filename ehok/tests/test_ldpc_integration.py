"""Integration-style tests for LDPC reconciliation handshake."""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pytest
import scipy.sparse as sp

from ehok.core import constants
from ehok.core.exceptions import MatrixSynchronizationError, ReconciliationFailedError
from ehok.implementations.reconciliation import (
    LDPCMatrixManager,
    LDPCReconciliator,
    PEGMatrixGenerator,
    DegreeDistribution,
)


FRAME_SIZE = 32
RATE = 0.50


def _build_manager(tmp_path) -> LDPCMatrixManager:
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


def _simulate_bob(reconciliator: LDPCReconciliator, key_block: np.ndarray) -> Tuple[str, str, str]:
    checksum = reconciliator.matrix_manager.checksum  # type: ignore[attr-defined]
    rate = reconciliator.select_rate(0.05)
    n_short = reconciliator.compute_shortening(rate, 0.05, len(key_block))
    seed = 9
    syndrome = reconciliator.compute_syndrome_block(key_block, rate, n_short, seed)
    block_hash = reconciliator.hash_verifier.compute_hash(key_block, seed)  # type: ignore[attr-defined]
    payload = {
        "rate": rate,
        "n_short": n_short,
        "seed": seed,
        "payload_len": len(key_block),
        "syndrome": syndrome.tobytes().hex(),
        "hash": block_hash.hex(),
    }
    return checksum, json.dumps(payload), block_hash.hex()


def test_alice_bob_handshake_success(tmp_path):
    manager = _build_manager(tmp_path)
    reconciliator_bob = LDPCReconciliator(manager)
    reconciliator_alice = LDPCReconciliator(manager)

    rng = np.random.default_rng(2025)
    key_block = rng.integers(0, 2, size=24, dtype=np.uint8)

    checksum, payload_str, _block_hash_hex = _simulate_bob(reconciliator_bob, key_block)

    # Alice verifies checksum and reconciles
    reconciliator_alice.matrix_manager.verify_checksum(checksum)  # type: ignore[attr-defined]
    payload = json.loads(payload_str)
    syndrome = np.frombuffer(bytes.fromhex(payload["syndrome"]), dtype=np.uint8)
    corrected, converged, error_count = reconciliator_alice.reconcile_block(
        key_block, syndrome, float(payload["rate"]), int(payload["n_short"]), int(payload["seed"])
    )

    alice_hash = reconciliator_alice.hash_verifier.compute_hash(corrected, int(payload["seed"]))  # type: ignore[attr-defined]
    verified = alice_hash.hex() == payload["hash"] and converged

    assert verified
    assert error_count == 0
    assert np.array_equal(corrected, key_block)


def test_matrix_sync_failure(tmp_path):
    manager = _build_manager(tmp_path)
    reconciliator = LDPCReconciliator(manager)
    bad_checksum = "deadbeef"

    with pytest.raises(MatrixSynchronizationError):
        reconciliator.matrix_manager.verify_checksum(bad_checksum)  # type: ignore[attr-defined]


def test_reconciliation_failure_on_hash_mismatch(tmp_path):
    manager = _build_manager(tmp_path)
    reconciliator = LDPCReconciliator(manager)

    key_block = np.zeros(24, dtype=np.uint8)
    rate = reconciliator.select_rate(0.05)
    n_short = reconciliator.compute_shortening(rate, 0.05, len(key_block))
    syndrome = reconciliator.compute_syndrome_block(key_block, rate, n_short, prng_seed=3)

    # Simulate tampered hash from Bob
    wrong_hash = bytes([1] * ((constants.LDPC_HASH_BITS + 7) // 8))
    corrected, converged, error_count = reconciliator.reconcile_block(
        key_block, syndrome, rate, n_short, prng_seed=3
    )

    alice_hash = reconciliator.hash_verifier.compute_hash(corrected, seed=3)  # type: ignore[attr-defined]

    assert np.array_equal(corrected, key_block)
    assert converged
    assert alice_hash != wrong_hash or error_count > 0

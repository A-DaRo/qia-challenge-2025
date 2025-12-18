"""Focused unit tests for MatrixManager and LDPC pool loading.

These tests use tmp_path and small synthetic matrices to avoid relying on the
full pre-generated LDPC asset set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import (
    compile_parity_check_matrix,
    compiled_cache_path,
    compute_sparse_checksum,
    save_compiled_cache,
)


def _write_small_matrix(directory: Path, frame_size: int, rate: float) -> sp.csr_matrix:
    # Build a small sparse parity check matrix of shape (m, n).
    n = frame_size
    m = max(1, int(round((1.0 - rate) * n)))

    rng = np.random.default_rng(0)
    rows = rng.integers(0, m, size=3 * n, dtype=np.int32)
    cols = rng.integers(0, n, size=3 * n, dtype=np.int32)
    data = np.ones(rows.shape[0], dtype=np.uint8)

    H = sp.coo_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8).tocsr()
    filename = constants.LDPC_MATRIX_FILE_PATTERN.format(frame_size=frame_size, rate=rate)
    sp.save_npz(directory / filename, H)
    return H


def test_from_directory_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Matrix directory not found"):
        MatrixManager.from_directory(directory=missing, frame_size=8, rates=(0.50,))


def test_from_directory_missing_matrix_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing LDPC matrix"):
        MatrixManager.from_directory(directory=tmp_path, frame_size=8, rates=(0.50,))


def test_from_directory_loads_small_pool_and_caches_compiled(tmp_path: Path) -> None:
    frame_size = 8
    rate = 0.50

    H = _write_small_matrix(tmp_path, frame_size=frame_size, rate=rate)

    # Pre-create compiled cache sidecar for this matrix.
    checksum = compute_sparse_checksum(H)
    compiled = compile_parity_check_matrix(H, checksum=checksum)

    matrix_path = tmp_path / constants.LDPC_MATRIX_FILE_PATTERN.format(
        frame_size=frame_size, rate=rate
    )
    save_compiled_cache(compiled_cache_path(matrix_path), compiled)

    manager = MatrixManager.from_directory(directory=tmp_path, frame_size=frame_size, rates=(rate,))

    assert manager.frame_size == frame_size
    assert manager.rates == (rate,)

    # Ensure compiled cache can be retrieved.
    compiled2 = manager.get_compiled(rate)
    assert compiled2.m == compiled.m
    assert compiled2.n == compiled.n
    assert compiled2.checksum == checksum


def test_get_matrix_unknown_rate_raises(tmp_path: Path) -> None:
    _write_small_matrix(tmp_path, frame_size=8, rate=0.50)
    manager = MatrixManager.from_directory(directory=tmp_path, frame_size=8, rates=(0.50,))

    with pytest.raises(KeyError, match="Available"):
        manager.get_matrix(0.90)


def test_verify_checksum_matches_local(tmp_path: Path) -> None:
    _write_small_matrix(tmp_path, frame_size=8, rate=0.50)
    manager = MatrixManager.from_directory(directory=tmp_path, frame_size=8, rates=(0.50,))

    assert manager.verify_checksum(manager.checksum) is True
    assert manager.verify_checksum("not-the-checksum") is False


def test_write_compiled_caches_writes_sidecar(tmp_path: Path) -> None:
    _write_small_matrix(tmp_path, frame_size=8, rate=0.50)
    manager = MatrixManager.from_directory(directory=tmp_path, frame_size=8, rates=(0.50,))

    written = manager.write_compiled_caches(overwrite=False)
    assert written == 1

    # Second call without overwrite should write nothing.
    written2 = manager.write_compiled_caches(overwrite=False)
    assert written2 == 0

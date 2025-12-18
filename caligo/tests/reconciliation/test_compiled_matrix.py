"""Unit tests for compiled LDPC matrix helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from caligo.reconciliation.compiled_matrix import (
    compile_parity_check_matrix,
    compiled_cache_path,
    compute_sparse_checksum,
    load_compiled_cache,
    save_compiled_cache,
)


def _small_parity_check_matrix() -> sp.csr_matrix:
    # 4 checks, 8 variables
    data = np.ones(10, dtype=np.uint8)
    rows = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32)
    cols = np.array([0, 3, 1, 2, 7, 2, 5, 0, 6, 7], dtype=np.int32)
    H = sp.coo_matrix((data, (rows, cols)), shape=(4, 8), dtype=np.uint8).tocsr()
    return H


def test_compute_sparse_checksum_is_deterministic() -> None:
    H = _small_parity_check_matrix()
    assert compute_sparse_checksum(H) == compute_sparse_checksum(H.copy())


def test_compile_parity_check_matrix_syndrome_matches_sparse_matmul() -> None:
    H = _small_parity_check_matrix()
    compiled = compile_parity_check_matrix(H)

    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=compiled.n, dtype=np.uint8)

    s_compiled = compiled.compute_syndrome(bits)

    # Sparse matmul, then mod 2.
    s_sparse = (H.dot(bits.astype(np.uint8)) % 2).astype(np.uint8)
    np.testing.assert_array_equal(s_compiled, s_sparse)


def test_count_syndrome_errors_zero_when_matching_target() -> None:
    H = _small_parity_check_matrix()
    compiled = compile_parity_check_matrix(H)

    bits = np.zeros(compiled.n, dtype=np.uint8)
    target = compiled.compute_syndrome(bits)

    assert compiled.count_syndrome_errors(bits, target) == 0


def test_count_syndrome_errors_positive_when_target_flipped() -> None:
    H = _small_parity_check_matrix()
    compiled = compile_parity_check_matrix(H)

    bits = np.zeros(compiled.n, dtype=np.uint8)
    target = compiled.compute_syndrome(bits)
    target[0] ^= 1

    assert compiled.count_syndrome_errors(bits, target) >= 1


def test_compiled_cache_roundtrip_and_checksum_mismatch(tmp_path: Path) -> None:
    H = _small_parity_check_matrix()
    checksum = compute_sparse_checksum(H)
    compiled = compile_parity_check_matrix(H, checksum=checksum)

    matrix_path = tmp_path / "ldpc_8_rate0.50.npz"
    cache_path = compiled_cache_path(matrix_path)

    save_compiled_cache(cache_path, compiled)

    loaded = load_compiled_cache(cache_path, expected_checksum=checksum)
    assert loaded is not None
    assert loaded.m == compiled.m
    assert loaded.n == compiled.n
    assert loaded.edge_count == compiled.edge_count
    assert loaded.checksum == checksum

    # Mismatched checksum should yield None.
    assert load_compiled_cache(cache_path, expected_checksum="deadbeef") is None


def test_compute_syndrome_rejects_wrong_length() -> None:
    H = _small_parity_check_matrix()
    compiled = compile_parity_check_matrix(H)

    with pytest.raises(ValueError, match="bits length"):
        compiled.compute_syndrome(np.zeros(compiled.n + 1, dtype=np.uint8))

"""Compiled LDPC matrix representation.

This module provides a lightweight, preprocessed representation of an LDPC
parity-check matrix for performance-critical runtime operations.

At runtime, belief propagation repeatedly needs:
- Tanner graph adjacency (check→edges and variable→edges)
- Fast syndrome computations / syndrome error counts

Building adjacency from a sparse matrix is expensive and was previously done on
*every decode() call*. Compiling once and caching the result avoids that cost.

The compiled form can also be stored as a small sidecar `.npz` next to the
original matrix file, enabling offline precomputation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompiledParityCheckMatrix:
    """Preprocessed parity-check matrix for fast LDPC operations.

    Parameters
    ----------
    m : int
        Number of check nodes (rows).
    n : int
        Number of variable nodes (columns).
    check_ptr : np.ndarray
        CSR row pointer array of shape (m+1,), dtype int32.
        Edges for check `c` are in the range `[check_ptr[c], check_ptr[c+1])`.
    check_var : np.ndarray
        CSR column indices array of shape (E,), dtype int32. For edge `e`,
        `check_var[e]` is the variable node index.
    var_ptr : np.ndarray
        Variable pointer array of shape (n+1,), dtype int32.
        Incident edges for variable `v` are in `[var_ptr[v], var_ptr[v+1])`.
    var_edges : np.ndarray
        Flattened list of incident edge ids for variables, shape (E,),
        dtype int32.
    checksum : str
        SHA-256 checksum of the source sparse matrix structure.

    Notes
    -----
    This object is intentionally independent of SciPy sparse types to keep
    decoding inner loops on contiguous NumPy arrays.
    """

    m: int
    n: int
    check_ptr: np.ndarray
    check_var: np.ndarray
    var_ptr: np.ndarray
    var_edges: np.ndarray
    checksum: str = ""

    @property
    def edge_count(self) -> int:
        """Total number of Tanner graph edges."""

        return int(self.check_var.shape[0])

    @property
    def max_variable_degree(self) -> int:
        """Maximum variable node degree."""

        degrees = np.diff(self.var_ptr)
        if degrees.size == 0:
            return 0
        return int(degrees.max())

    def compute_syndrome(self, bits: np.ndarray) -> np.ndarray:
        """Compute syndrome $s = H x \bmod 2$ using compiled adjacency.

        Parameters
        ----------
        bits : np.ndarray
            Bit vector of shape (n,), values in {0,1}.

        Returns
        -------
        np.ndarray
            Syndrome vector of shape (m,), dtype uint8.
        """

        if bits.shape[0] != self.n:
            raise ValueError(f"bits length {bits.shape[0]} != n {self.n}")

        bits_u8 = bits.astype(np.uint8, copy=False)
        syndrome = np.empty(self.m, dtype=np.uint8)

        # LDPC checks are low-degree; tight Python loop is typically faster
        # than sparse matmul in repeated decode convergence checks.
        for c in range(self.m):
            start = int(self.check_ptr[c])
            end = int(self.check_ptr[c + 1])
            parity = 0
            for v in self.check_var[start:end]:
                parity ^= int(bits_u8[int(v)])
            syndrome[c] = parity

        return syndrome

    def count_syndrome_errors(self, bits: np.ndarray, target_syndrome: np.ndarray) -> int:
        """Count unsatisfied parity checks for the given target syndrome.

        Parameters
        ----------
        bits : np.ndarray
            Bit vector of shape (n,), values in {0,1}.
        target_syndrome : np.ndarray
            Target syndrome vector of shape (m,), values in {0,1}.

        Returns
        -------
        int
            Number of parity checks where computed syndrome differs.
        """

        if target_syndrome.shape[0] != self.m:
            raise ValueError(
                f"target_syndrome length {target_syndrome.shape[0]} != m {self.m}"
            )

        bits_u8 = bits.astype(np.uint8, copy=False)
        target_u8 = target_syndrome.astype(np.uint8, copy=False)

        errors = 0
        for c in range(self.m):
            start = int(self.check_ptr[c])
            end = int(self.check_ptr[c + 1])
            parity = 0
            for v in self.check_var[start:end]:
                parity ^= int(bits_u8[int(v)])
            if parity != int(target_u8[c]):
                errors += 1

        return errors


def compute_sparse_checksum(H: sp.csr_matrix) -> str:
    """Compute a deterministic SHA-256 checksum for a sparse CSR matrix."""

    H = H.tocsr()
    digest = hashlib.sha256()
    digest.update(H.indptr.tobytes())
    digest.update(H.indices.tobytes())
    if H.data is not None:
        digest.update(H.data.tobytes())
    return digest.hexdigest()


def compile_parity_check_matrix(
    H: sp.csr_matrix,
    checksum: Optional[str] = None,
) -> CompiledParityCheckMatrix:
    """Compile a SciPy CSR parity-check matrix into fast adjacency arrays.

    Parameters
    ----------
    H : sp.csr_matrix
        Parity-check matrix in CSR format.
    checksum : str, optional
        Precomputed checksum. If omitted, computed from H.

    Returns
    -------
    CompiledParityCheckMatrix
        Compiled adjacency.
    """

    H = H.tocsr()
    m, n = H.shape

    if checksum is None:
        checksum = compute_sparse_checksum(H)

    check_ptr = H.indptr.astype(np.int32, copy=True)
    check_var = H.indices.astype(np.int32, copy=True)
    edge_count = int(check_var.shape[0])

    # Build variable→edges adjacency once.
    var_lists = [[] for _ in range(n)]
    for edge_id, v in enumerate(check_var.tolist()):
        var_lists[int(v)].append(int(edge_id))

    var_ptr = np.zeros(n + 1, dtype=np.int32)
    for v in range(n):
        var_ptr[v + 1] = var_ptr[v] + len(var_lists[v])

    var_edges = np.empty(edge_count, dtype=np.int32)
    cursor = 0
    for v in range(n):
        edges = var_lists[v]
        if edges:
            var_edges[cursor : cursor + len(edges)] = np.asarray(edges, dtype=np.int32)
        cursor += len(edges)

    return CompiledParityCheckMatrix(
        m=m,
        n=n,
        check_ptr=check_ptr,
        check_var=check_var,
        var_ptr=var_ptr,
        var_edges=var_edges,
        checksum=checksum,
    )


def compiled_cache_path(matrix_path: Path) -> Path:
    """Return sidecar cache path for a given LDPC matrix file."""

    return matrix_path.with_suffix(matrix_path.suffix + ".compiled.npz")


def load_compiled_cache(
    cache_path: Path,
    expected_checksum: str,
) -> Optional[CompiledParityCheckMatrix]:
    """Load a compiled cache file if it matches the expected checksum."""

    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=False)
        cached_checksum = str(payload["checksum"][0])
        if cached_checksum != expected_checksum:
            logger.debug(
                "Compiled cache checksum mismatch (%s != %s): %s",
                cached_checksum[:16], expected_checksum[:16], cache_path,
            )
            return None

        m = int(payload["m"][0])
        n = int(payload["n"][0])
        return CompiledParityCheckMatrix(
            m=m,
            n=n,
            check_ptr=payload["check_ptr"].astype(np.int32, copy=False),
            check_var=payload["check_var"].astype(np.int32, copy=False),
            var_ptr=payload["var_ptr"].astype(np.int32, copy=False),
            var_edges=payload["var_edges"].astype(np.int32, copy=False),
            checksum=cached_checksum,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to load compiled cache %s: %s", cache_path, exc)
        return None


def save_compiled_cache(cache_path: Path, compiled: CompiledParityCheckMatrix) -> None:
    """Persist compiled adjacency arrays to a sidecar `.npz` file."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        checksum=np.asarray([compiled.checksum]),
        m=np.asarray([compiled.m], dtype=np.int32),
        n=np.asarray([compiled.n], dtype=np.int32),
        check_ptr=compiled.check_ptr,
        check_var=compiled.check_var,
        var_ptr=compiled.var_ptr,
        var_edges=compiled.var_edges,
    )

"""
LDPC Matrix Pool Management.

Handles loading, caching, and synchronization of LDPC parity-check
matrices for reconciliation.

Matrices are pre-generated offline using PEG algorithm and loaded
at runtime. Alice and Bob must use identical matrices, verified
via SHA-256 checksum.

References
----------
- Hu et al. (2005): Progressive Edge-Growth construction
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import (
    CompiledParityCheckMatrix,
    compile_parity_check_matrix,
    compiled_cache_path,
    compute_sparse_checksum,
    load_compiled_cache,
    save_compiled_cache,
)
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Matrix Pool Dataclass
# =============================================================================


@dataclass
class MatrixPool:
    """
    Immutable pool of LDPC parity-check matrices.

    Attributes
    ----------
    frame_size : int
        Common frame size n for all matrices.
    matrices : Dict[float, sp.csr_matrix]
        Rate → parity-check matrix mapping.
    rates : Tuple[float, ...]
        Sorted tuple of available rates.
    checksum : str
        SHA-256 checksum for synchronization.
    puncture_patterns : Dict[float, np.ndarray]
        Rate → untainted puncturing pattern mapping.
        Pattern is binary array of shape (n,) where 1 indicates punctured position.
    """

    frame_size: int
    matrices: Dict[float, sp.csr_matrix] = field(repr=False)
    rates: Tuple[float, ...]
    checksum: str
    puncture_patterns: Dict[float, np.ndarray] = field(default_factory=dict, repr=False)


# =============================================================================
# Matrix Manager
# =============================================================================


class MatrixManager:
    """
    Thread-safe LDPC matrix pool accessor.

    Provides matrix loading, caching, and checksum verification
    for Alice-Bob synchronization.

    Parameters
    ----------
    pool : MatrixPool
        Loaded matrix pool.
    """

    def __init__(self, pool: MatrixPool) -> None:
        self._pool = pool
        self._compiled_by_rate: Dict[float, CompiledParityCheckMatrix] = {}
        self._matrix_paths: Dict[float, Path] = {}
        self._pattern_paths: Dict[float, Path] = {}

    @classmethod
    def from_directory(
        cls,
        directory: Optional[Path] = None,
        frame_size: int = constants.LDPC_FRAME_SIZE,
        rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    ) -> "MatrixManager":
        """
        Load matrix pool from directory.

        Parameters
        ----------
        directory : Path, optional
            Directory containing .npz matrix files.
            Defaults to package ldpc_matrices directory.
        frame_size : int
            Expected frame size.
        rates : Tuple[float, ...]
            Rates to load.

        Returns
        -------
        MatrixManager
            Initialized manager with loaded pool.

        Raises
        ------
        FileNotFoundError
            If directory or required matrices not found.
        ValueError
            If matrix dimensions don't match frame_size.
        """
        if directory is None:
            directory = constants.LDPC_MATRICES_DIR
        directory = Path(directory).expanduser().resolve()

        logger.info("Loading LDPC matrices from: %s", directory)

        if not directory.exists():
            raise FileNotFoundError(f"Matrix directory not found: {directory}")

        matrices: Dict[float, sp.csr_matrix] = {}
        sorted_rates = tuple(sorted(rates))

        compiled: Dict[float, CompiledParityCheckMatrix] = {}
        matrix_paths: Dict[float, Path] = {}
        puncture_patterns: Dict[float, np.ndarray] = {}
        pattern_paths: Dict[float, Path] = {}

        for idx, rate in enumerate(sorted_rates, 1):
            filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
                frame_size=frame_size, rate=rate
            )
            path = directory / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing LDPC matrix for rate {rate:.2f}: {path}"
                )

            logger.debug("[%d/%d] Loading %s", idx, len(sorted_rates), filename)
            matrix = sp.load_npz(path).tocsr().astype(np.uint8)

            if matrix.shape[1] != frame_size:
                raise ValueError(
                    f"Matrix {filename} has frame size {matrix.shape[1]}, "
                    f"expected {frame_size}"
                )

            matrices[rate] = matrix
            matrix_paths[rate] = path

            # Optional: load (offline) compiled adjacency if present.
            matrix_checksum = compute_sparse_checksum(matrix)
            cache_path = compiled_cache_path(path)
            compiled_matrix = load_compiled_cache(cache_path, matrix_checksum)
            if compiled_matrix is not None:
                compiled[rate] = compiled_matrix

            logger.info(
                "Loaded rate=%.2f: shape=%s, nnz=%d",
                rate, matrix.shape, matrix.nnz
            )

        # Load puncture patterns from puncture_patterns/ subdirectory
        pattern_dir = directory / "puncture_patterns"
        if pattern_dir.exists():
            logger.debug("Scanning for puncture patterns in: %s", pattern_dir)
            for rate in sorted_rates:
                pattern_filename = f"puncture_pattern_rate{rate:.2f}.npy"
                pattern_path = pattern_dir / pattern_filename
                if pattern_path.exists():
                    try:
                        pattern = np.load(pattern_path)
                        if pattern.shape[0] != frame_size:
                            logger.warning(
                                "Pattern %s has size %d, expected %d. Skipping.",
                                pattern_filename, pattern.shape[0], frame_size
                            )
                            continue
                        puncture_patterns[rate] = pattern.astype(np.uint8)
                        pattern_paths[rate] = pattern_path
                        logger.info(
                            "Loaded puncture pattern for rate=%.2f: %d punctured bits",
                            rate, int(pattern.sum())
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load pattern %s: %s",
                            pattern_filename, exc
                        )
        else:
            logger.debug("No puncture_patterns/ subdirectory found")

        checksum = cls._compute_checksum(matrices, puncture_patterns)
        logger.info("Matrix pool checksum: %s...", checksum[:16])

        pool = MatrixPool(
            frame_size=frame_size,
            matrices=matrices,
            rates=sorted_rates,
            checksum=checksum,
            puncture_patterns=puncture_patterns,
        )

        manager = cls(pool)
        manager._compiled_by_rate.update(compiled)
        manager._matrix_paths.update(matrix_paths)
        manager._pattern_paths.update(pattern_paths)

        # Optional: precompute and persist compiled caches for future runs.
        if os.getenv("CALIGO_LDPC_WRITE_COMPILED_CACHE", "0") == "1":
            written = manager.write_compiled_caches(overwrite=False)
            if written > 0:
                logger.info("Wrote %d compiled LDPC cache files", written)
        return manager

    @staticmethod
    def _compute_checksum(
        matrices: Dict[float, sp.csr_matrix],
        puncture_patterns: Dict[float, np.ndarray] = None,
    ) -> str:
        """
        Compute SHA-256 checksum for matrix pool.

        Processes matrices and patterns in sorted rate order for determinism.
        Including patterns in checksum ensures Alice-Bob synchronization.

        Parameters
        ----------
        matrices : Dict[float, sp.csr_matrix]
            Rate → matrix mapping.
        puncture_patterns : Dict[float, np.ndarray], optional
            Rate → pattern mapping.

        Returns
        -------
        str
            SHA-256 checksum hex digest.
        """
        digest = hashlib.sha256()
        for rate in sorted(matrices.keys()):
            matrix = matrices[rate].tocsr()
            digest.update(str(rate).encode())
            digest.update(matrix.indptr.tobytes())
            digest.update(matrix.indices.tobytes())
            if matrix.data is not None:
                digest.update(matrix.data.tobytes())

        # Include puncture patterns in checksum for synchronization
        if puncture_patterns:
            for rate in sorted(puncture_patterns.keys()):
                pattern = puncture_patterns[rate]
                digest.update(f"pattern_{rate}".encode())
                digest.update(pattern.tobytes())

        return digest.hexdigest()

    def get_matrix(self, rate: float) -> sp.csr_matrix:
        """
        Retrieve parity-check matrix for rate.

        Parameters
        ----------
        rate : float
            Desired code rate.

        Returns
        -------
        sp.csr_matrix
            Parity-check matrix H.

        Raises
        ------
        KeyError
            If rate not in pool.
        """
        if rate not in self._pool.matrices:
            available = list(self._pool.rates)
            raise KeyError(f"Rate {rate} not in pool. Available: {available}")
        return self._pool.matrices[rate]

    def get_compiled(self, rate: float) -> CompiledParityCheckMatrix:
        """Retrieve (and cache) the compiled parity-check representation.

        Parameters
        ----------
        rate : float
            Desired code rate.

        Returns
        -------
        CompiledParityCheckMatrix
            Compiled adjacency representation.
        """

        cached = self._compiled_by_rate.get(rate)
        if cached is not None:
            return cached

        H = self.get_matrix(rate)
        compiled = compile_parity_check_matrix(H)
        self._compiled_by_rate[rate] = compiled
        return compiled

    def write_compiled_caches(self, overwrite: bool = False) -> int:
        """Write sidecar compiled caches for loaded matrices.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite existing cache files.

        Returns
        -------
        int
            Number of cache files written.
        """

        written = 0
        for rate in self._pool.rates:
            matrix_path = self._matrix_paths.get(rate)
            if matrix_path is None:
                continue

            cache_path = compiled_cache_path(matrix_path)
            if cache_path.exists() and not overwrite:
                continue

            compiled = self.get_compiled(rate)
            save_compiled_cache(cache_path, compiled)
            written += 1

        return written

    def verify_checksum(self, remote_checksum: str) -> bool:
        """
        Verify local checksum matches remote.

        Parameters
        ----------
        remote_checksum : str
            Checksum from remote party.

        Returns
        -------
        bool
            True if checksums match.
        """
        return self._pool.checksum == remote_checksum

    @property
    def checksum(self) -> str:
        """Local pool checksum."""
        return self._pool.checksum

    @property
    def rates(self) -> Tuple[float, ...]:
        """Available rates (sorted)."""
        return self._pool.rates

    @property
    def frame_size(self) -> int:
        """Common frame size for all matrices."""
        return self._pool.frame_size

    def get_puncture_pattern(self, rate: float) -> Optional[np.ndarray]:
        """
        Retrieve untainted puncturing pattern for rate.

        Returns None if no pattern is available for the rate, in which case
        the caller should fall back to legacy random padding.

        Parameters
        ----------
        rate : float
            Desired code rate.

        Returns
        -------
        np.ndarray or None
            Binary puncturing pattern of shape (n,), or None if unavailable.
            Pattern[i] = 1 indicates position i should be punctured.
        """
        return self._pool.puncture_patterns.get(rate)

    @property
    def available_pattern_rates(self) -> Tuple[float, ...]:
        """Rates for which puncture patterns are available."""
        return tuple(sorted(self._pool.puncture_patterns.keys()))

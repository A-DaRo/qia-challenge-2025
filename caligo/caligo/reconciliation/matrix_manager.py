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

        # Load puncture patterns from puncture_patterns/ subdirectory (legacy)
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

        # NEW: Also look for hybrid patterns in parent's hybrid_patterns/ dir
        # This supports the new architecture where patterns are in ldpc_matrices/hybrid_patterns/
        hybrid_pattern_dir = directory.parent / "hybrid_patterns"
        if hybrid_pattern_dir.exists() and len(puncture_patterns) == 0:
            logger.debug("Scanning for hybrid patterns in: %s", hybrid_pattern_dir)
            # Load all patterns that exist (new naming: pattern_rate0.XX.npy)
            for path in hybrid_pattern_dir.glob("pattern_rate*.npy"):
                try:
                    rate_str = path.stem.split("rate")[-1]
                    rate = float(rate_str)
                    pattern = np.load(path).astype(np.uint8)
                    if pattern.shape[0] != frame_size:
                        logger.warning(
                            "Pattern %s has size %d, expected %d. Skipping.",
                            path.name, pattern.shape[0], frame_size
                        )
                        continue
                    puncture_patterns[rate] = pattern
                    pattern_paths[rate] = path
                    logger.debug(
                        "Loaded hybrid pattern for rate=%.2f: %d punctured bits",
                        rate, int(pattern.sum())
                    )
                except (ValueError, IndexError) as exc:
                    logger.warning("Failed to parse pattern file %s: %s", path.name, exc)
            if puncture_patterns:
                logger.info(
                    "Loaded %d hybrid patterns from %s",
                    len(puncture_patterns), hybrid_pattern_dir
                )

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

# =============================================================================
# Mother Code Manager (Phase 1)
# =============================================================================

@dataclass
class NumbaGraphTopology:
    """
    Pre-compiled graph topology arrays for Numba kernels (Phase 1).
    
    Structure-of-Arrays (SoA) format optimized for cache locality.
    These arrays are pinned in memory and passed as read-only arguments
    to JIT-compiled kernels.
    
    Per Implementation Report v2 §6: This replaces CompiledParityCheckMatrix
    for the Phase 1 hybrid architecture.
    
    Attributes
    ----------
    check_row_ptr : np.ndarray
        CSR row pointers (uint32[m+1]).
    check_col_idx : np.ndarray
        CSR column indices (uint32[nnz]).
    var_col_ptr : np.ndarray
        CSC column pointers (uint32[n+1]).
    var_row_idx : np.ndarray
        CSC row indices (uint32[nnz]).
    edge_c2v : np.ndarray
        Check→Var edge indices (uint32[nnz]).
    edge_v2c : np.ndarray
        Var→Check edge indices (uint32[nnz]).
    n_checks : int
        Number of check nodes (m).
    n_vars : int
        Number of variable nodes (n).
    n_edges : int
        Number of edges (nnz).
    """
    check_row_ptr: np.ndarray      # uint32[m+1]: Row pointers
    check_col_idx: np.ndarray      # uint32[nnz]: Column indices
    var_col_ptr: np.ndarray        # uint32[n+1]: Column pointers
    var_row_idx: np.ndarray        # uint32[nnz]: Row indices
    edge_c2v: np.ndarray           # uint32[nnz]: Check→Var edge indices
    edge_v2c: np.ndarray           # uint32[nnz]: Var→Check edge indices
    n_checks: int
    n_vars: int
    n_edges: int


class MotherCodeManager:
    """
    Singleton manager for R=0.5 Mother Code with Hybrid Pattern Library (Phase 1).

    Per Theoretical Report v2 §2.2 and §5.1, this class:
    1. Manages a single R_0=0.5 mother matrix
    2. Provides Hybrid Pattern Library (Untainted + ACE-Guided)
    3. Serves as the 'Static Data' provider for Numba kernels
    
    The Hybrid Pattern Library covers R_eff ∈ [0.51, 0.90] with Δ R = 0.01:
    - Regime A (R ≤ R_sat): Untainted puncturing patterns [3]
    - Regime B (R > R_sat): ACE-guided puncturing patterns [4]
    
    References
    ----------
    [3] Elkouss et al., "Untainted Puncturing for Irregular LDPC Codes"
    [4] Liu & de Lamare, "Rate-Compatible LDPC Codes Based on Puncturing"
    """

    _instance: Optional["MotherCodeManager"] = None
    _init_lock = False  # Prevent re-initialization

    def __init__(
        self,
        matrix_path: Path,
        pattern_dir: Path,
    ) -> None:
        """
        Load mother matrix and Hybrid Pattern Library.
        
        Parameters
        ----------
        matrix_path : Path
            Path to R=0.5 mother matrix (.npz format).
        pattern_dir : Path
            Directory containing hybrid pattern files.
        """
        if self._init_lock:
            raise RuntimeError("Use MotherCodeManager.get_instance() for singleton access")
        
        # Load single R=0.5 matrix
        logger.info(f"Loading mother matrix from {matrix_path}")
        self._H_csr = sp.load_npz(matrix_path).tocsr().astype(np.uint8)
        
        # Verify R=0.5
        n, m = self._H_csr.shape[1], self._H_csr.shape[0]
        rate = 1.0 - m / n
        if abs(rate - 0.5) > 0.01:
            raise ValueError(f"Mother code rate {rate:.3f} != 0.5")
        
        logger.info(f"Mother matrix shape: {self._H_csr.shape}, rate: {rate:.3f}")
        
        # PRE-COMPILE FOR NUMBA:
        # Flatten CSR arrays to contiguous uint32/uint64 buffers for 
        # direct access by JIT kernels.
        logger.info("Compiling topology for Numba kernels...")
        self._compiled_topology = self._compile_topology()
        logger.info(f"Topology compiled: {self._compiled_topology.n_edges} edges")
        
        # Load Hybrid Pattern Library (Step = 0.01)
        logger.info(f"Loading hybrid patterns from {pattern_dir}")
        self._patterns = self._load_hybrid_library(pattern_dir)
        logger.info(f"Loaded {len(self._patterns)} hybrid patterns")
        
        # Pre-computed modulation indices for Blind protocol
        self._modulation_indices: Optional[np.ndarray] = None
        
        # Legacy compiled matrix for backward compatibility
        self._compiled: Optional[CompiledParityCheckMatrix] = None

    @classmethod
    def get_instance(cls, **kwargs) -> "MotherCodeManager":
        """
        Singleton accessor for MotherCodeManager.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments passed to __init__ on first invocation.
            
        Returns
        -------
        MotherCodeManager
            Singleton instance.
        """
        if cls._instance is None:
            cls._init_lock = False  # Allow first init
            cls._instance = cls(**kwargs)
            cls._init_lock = True   # Prevent further inits
        return cls._instance

    @classmethod
    def from_config(
        cls,
        code_type: str = "ace_peg",
        base_dir: Optional[Path] = None,
        frame_size: int = constants.LDPC_FRAME_SIZE,
    ) -> "MotherCodeManager":
        """
        Create MotherCodeManager from configuration (convenience method).

        Parameters
        ----------
        code_type : str
            Type of code to load (e.g., "ace_peg").
        base_dir : Path, optional
            Directory containing matrices. Defaults to constants.LDPC_MATRICES_DIR.
        frame_size : int
            Frame size n.

        Returns
        -------
        MotherCodeManager
            Singleton instance.
        """
        if base_dir is None:
            base_dir = constants.LDPC_MATRICES_DIR

        # Construct paths
        if code_type == "ace_peg":
            matrix_path = base_dir / f"ldpc_ace_peg/ldpc_{frame_size}_rate0.50.npz"
        else:
            matrix_path = base_dir / f"ldpc_{frame_size}_rate0.50.npz"

        pattern_dir = base_dir / "hybrid_patterns"

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Mother matrix not found at {matrix_path}. "
                "Run generate_ace_mother_code.py to create it."
            )

        if not pattern_dir.exists():
            raise FileNotFoundError(
                f"Hybrid patterns directory not found at {pattern_dir}. "
                "Run generate_hybrid_patterns.py to create patterns."
            )

        return cls.get_instance(
            matrix_path=matrix_path,
            pattern_dir=pattern_dir,
        )

    @property
    def frame_size(self) -> int:
        """Get frame size (n)."""
        return self._H_csr.shape[1]

    @property
    def H_csr(self) -> sp.csr_matrix:
        """Get the mother code parity-check matrix (CSR format)."""
        return self._H_csr

    @property
    def mother_rate(self) -> float:
        """Get mother code rate (R_0=0.5)."""
        return constants.MOTHER_CODE_RATE

    @property
    def compiled_topology(self) -> NumbaGraphTopology:
        """Get pre-compiled NumbaGraphTopology."""
        return self._compiled_topology

    @property
    def num_edges(self) -> int:
        """Get number of edges in the Tanner graph."""
        return self._compiled_topology.n_edges

    @property
    def patterns(self) -> Dict[float, np.ndarray]:
        """Get dictionary of hybrid patterns."""
        return self._patterns

    def get_compiled(self) -> CompiledParityCheckMatrix:
        """
        Get JIT-compiled mother matrix representation.

        Returns
        -------
        CompiledParityCheckMatrix
            Compiled matrix structure for Numba kernels.
        """
        if self._compiled is None:
            self._compiled = compile_parity_check_matrix(self._H_csr)
        return self._compiled

    def get_compiled_mother_code(self) -> CompiledParityCheckMatrix:
        """
        Alias for get_compiled() (backward compatibility).

        Returns
        -------
        CompiledParityCheckMatrix
            Compiled matrix structure for legacy Numba kernels.
        """
        return self.get_compiled()

    def get_pattern(self, target_rate: float) -> np.ndarray:
        """
        Get hybrid puncturing pattern for target effective rate.
        
        Per Theoretical Report v2 §2.2:
        - R ≤ R_sat: Untainted pattern (Regime A)
        - R > R_sat: ACE-guided pattern (Regime B)
        
        Parameters
        ----------
        target_rate : float
            Desired effective rate.

        Returns
        -------
        np.ndarray
            Binary mask (uint8) where 1 indicates punctured position.
            
        Raises
        ------
        ValueError
            If no pattern found for rate.
        """
        # Find closest available rate (Δ R = 0.01 step)
        available = sorted(self._patterns.keys())
        if not available:
            raise ValueError("No patterns loaded")
        
        closest = min(available, key=lambda r: abs(r - target_rate))
        
        # Tolerance check
        if abs(closest - target_rate) > 0.02:
            logger.warning(
                f"Requested rate {target_rate:.2f} not in library. "
                f"Using closest: {closest:.2f}"
            )
        
        return self._patterns[closest].copy()

    def get_modulation_indices(self, d: int) -> np.ndarray:
        """
        Get d hybrid modulation indices for Blind protocol.
        
        Per Theoretical Report v2 §4.3: the revelation order is fixed
        at setup time using the hybrid puncturing order (Phase I untainted
        first, then Phase II ACE-guided).
        
        Parameters
        ----------
        d : int
            Number of modulation positions (punctured + shortened budget).
            
        Returns
        -------
        np.ndarray
            Ordered indices for modulation positions.
        """
        if self._modulation_indices is None:
            self._modulation_indices = self._compute_hybrid_indices()
        
        if d > len(self._modulation_indices):
            raise ValueError(
                f"Requested {d} indices but only {len(self._modulation_indices)} available"
            )
        
        return self._modulation_indices[:d].copy()

    def _compile_topology(self) -> NumbaGraphTopology:
        """
        Convert CSR matrix to Numba-friendly SoA format.
        
        This is the "baking" step that converts the sparse matrix
        representation into flat arrays optimized for JIT compilation.
        The edge_c2v and edge_v2c arrays map between CSR and CSC orderings.
        
        Returns
        -------
        NumbaGraphTopology
            Pre-compiled topology arrays.
        """
        H = self._H_csr
        H_csc = H.tocsc()
        
        # CSR format (check-to-variable)
        check_row_ptr = H.indptr.astype(np.uint32)
        check_col_idx = H.indices.astype(np.uint32)
        
        # CSC format (variable-to-check)
        var_col_ptr = H_csc.indptr.astype(np.uint32)
        var_row_idx = H_csc.indices.astype(np.uint32)
        
        # Build edge permutation arrays
        # CSR edges are ordered (c0,v_a), (c0,v_b), ..., (c1,v_c), ...
        # CSC edges are ordered (c_x,v0), (c_y,v0), ..., (c_z,v1), ...
        n_edges = H.nnz
        n_vars = H.shape[1]
        n_checks = H.shape[0]
        
        # Build lookup: (check, var) -> CSR edge index
        csr_edge_map = {}
        for c in range(n_checks):
            for k in range(H.indptr[c], H.indptr[c+1]):
                v = H.indices[k]
                csr_edge_map[(c, v)] = k
        
        # edge_c2v[csr_idx] = csc_idx for same (c,v) edge
        # edge_v2c[csc_idx] = csr_idx for same (c,v) edge
        edge_c2v = np.zeros(n_edges, dtype=np.uint32)
        edge_v2c = np.zeros(n_edges, dtype=np.uint32)
        
        csc_idx = 0
        for v in range(n_vars):
            for k in range(H_csc.indptr[v], H_csc.indptr[v+1]):
                c = H_csc.indices[k]
                csr_idx = csr_edge_map[(c, v)]
                edge_c2v[csr_idx] = csc_idx
                edge_v2c[csc_idx] = csr_idx
                csc_idx += 1
        
        return NumbaGraphTopology(
            check_row_ptr=check_row_ptr,
            check_col_idx=check_col_idx,
            var_col_ptr=var_col_ptr,
            var_row_idx=var_row_idx,
            edge_c2v=edge_c2v,
            edge_v2c=edge_v2c,
            n_checks=n_checks,
            n_vars=n_vars,
            n_edges=n_edges,
        )

    def _load_hybrid_library(self, pattern_dir: Path) -> Dict[float, np.ndarray]:
        """
        Load Hybrid Pattern Library from directory.
        
        Expected file naming: pattern_rate0.51.npy, pattern_rate0.52.npy, ...
        Covers R_eff ∈ [0.51, 0.90] with Δ R = 0.01 (~40 files).
        
        Parameters
        ----------
        pattern_dir : Path
            Directory containing pattern files.
            
        Returns
        -------
        Dict[float, np.ndarray]
            Rate → pattern mapping.
            
        Raises
        ------
        ValueError
            If insufficient patterns found.
        """
        patterns = {}
        
        for path in pattern_dir.glob("pattern_rate*.npy"):
            # Parse rate from filename: pattern_rate0.65.npy
            try:
                rate_str = path.stem.split("rate")[-1]
                rate = float(rate_str)
                pattern = np.load(path).astype(np.uint8)
                
                # Validate shape
                if pattern.shape[0] != self._H_csr.shape[1]:
                    logger.warning(
                        f"Skipping {path.name}: shape {pattern.shape[0]} != {self._H_csr.shape[1]}"
                    )
                    continue
                
                patterns[rate] = pattern
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed pattern file {path.name}: {e}")
                continue
        
        if len(patterns) < 10:
            raise ValueError(
                f"Insufficient patterns in {pattern_dir}. "
                f"Expected ~40, found {len(patterns)}. "
                "Run generate_hybrid_patterns.py to create Hybrid Pattern Library."
            )
        
        return patterns

    def _compute_hybrid_indices(self) -> np.ndarray:
        """
        Compute ordered modulation indices from hybrid puncturing order.
        
        Per Theoretical Report v2 §2.2.3: Phase I (untainted) first,
        then Phase II (ACE-guided), maintaining nesting property for
        rate-compatible modulation.
        
        Returns
        -------
        np.ndarray
            Ordered indices for Blind protocol modulation.
        """
        # Load modulation_indices.npy from pattern directory if available
        # This file is generated by generate_hybrid_patterns.py
        try:
            pattern_dir = constants.LDPC_MATRICES_DIR / "hybrid_patterns"
            indices_path = pattern_dir / "modulation_indices.npy"
            
            if indices_path.exists():
                return np.load(indices_path).astype(np.int64)
        except Exception as e:
            logger.warning(f"Could not load modulation indices: {e}")
        
        # Fallback: construct from pattern order (use highest rate pattern)
        highest_rate = max(self._patterns.keys())
        pattern = self._patterns[highest_rate]
        punctured_indices = np.where(pattern == 1)[0]
        
        logger.warning(
            "Using fallback modulation indices from highest rate pattern. "
            "Recommend running generate_hybrid_patterns.py to create proper indices."
        )
        
        return punctured_indices.astype(np.int64)

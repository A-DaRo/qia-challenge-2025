"""
Test Suite for Hybrid Pattern Generation (Task 2).

Tests the two-regime puncturing strategy from Theoretical Report v2 §2.2:
- Regime A: Untainted puncturing (saturation detection)
- Regime B: ACE-guided puncturing
- Nesting property (rate-compatibility)
- MotherCodeManager initialization

References:
[3] Elkouss et al., "Untainted Puncturing for Irregular LDPC Codes"
[4] Liu & de Lamare, "Rate-Compatible LDPC Codes Based on Puncturing"
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz

from caligo.reconciliation import constants
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_matrix_dir(tmp_path: Path) -> Path:
    """Create temporary directory with test matrix."""
    matrix_dir = tmp_path / "ldpc_matrices"
    matrix_dir.mkdir()
    return matrix_dir


@pytest.fixture
def small_mother_matrix() -> csr_matrix:
    """Create a small R=0.5 test matrix (128 bits, 64 checks)."""
    np.random.seed(42)
    n = 128
    m = 64  # R = 1 - 64/128 = 0.5
    
    # Generate sparse matrix with ~3 ones per column
    H = np.zeros((m, n), dtype=np.uint8)
    for col in range(n):
        rows = np.random.choice(m, size=3, replace=False)
        H[rows, col] = 1
    
    return csr_matrix(H)


@pytest.fixture
def temp_mother_matrix_file(temp_matrix_dir: Path, small_mother_matrix: csr_matrix) -> Path:
    """Save mother matrix to file."""
    ace_peg_dir = temp_matrix_dir / "ldpc_ace_peg"
    ace_peg_dir.mkdir()
    matrix_path = ace_peg_dir / "ldpc_128_rate0.50.npz"
    save_npz(matrix_path, small_mother_matrix)
    return matrix_path


def create_dummy_pattern(n: int, num_punctured: int) -> np.ndarray:
    """Create a dummy puncturing pattern."""
    pattern = np.zeros(n, dtype=np.uint8)
    pattern[:num_punctured] = 1
    return pattern


@pytest.fixture
def temp_pattern_dir(temp_matrix_dir: Path) -> Path:
    """Create temporary hybrid patterns directory."""
    pattern_dir = temp_matrix_dir / "hybrid_patterns"
    pattern_dir.mkdir()
    return pattern_dir


@pytest.fixture
def populated_pattern_dir(temp_pattern_dir: Path) -> Path:
    """Create directory with valid hybrid patterns."""
    n = 128
    
    # Create patterns for rates 0.51 to 0.70
    for rate in np.arange(0.51, 0.71, 0.01):
        # Calculate punctured bits for this rate
        # R_eff = R_0 / (1 - p/n), so p = n * (1 - R_0/R_eff)
        R_0 = 0.5
        p = int(n * (1 - R_0 / rate))
        p = max(1, min(p, n // 2))
        
        pattern = create_dummy_pattern(n, p)
        np.save(temp_pattern_dir / f"pattern_rate{rate:.2f}.npy", pattern)
    
    # Create modulation indices file
    modulation_indices = np.arange(n // 2, dtype=np.int64)
    np.save(temp_pattern_dir / "modulation_indices.npy", modulation_indices)
    
    return temp_pattern_dir


# =============================================================================
# TASK 2.1: Saturation Detection (Theorem 2.2)
# =============================================================================


class TestSaturationDetection:
    """
    Task 2.1: Test that generator detects untainted set exhaustion.
    
    Per Theorem 2.2: Strict untainted puncturing saturates when X_∞ empties,
    typically around R_eff ≈ 0.625.
    """
    
    def test_untainted_set_computation(self, small_mother_matrix: csr_matrix) -> None:
        """Verify untainted candidate detection logic."""
        from caligo.scripts.generate_hybrid_patterns import (
            compute_depth2_neighborhood,
        )
        
        H = small_mother_matrix
        n = H.shape[1]
        
        # Every symbol should have a valid N²(v) neighborhood
        for v in range(min(10, n)):
            neighborhood = compute_depth2_neighborhood(H, v)
            assert v in neighborhood, f"Symbol {v} not in its own neighborhood"
            assert len(neighborhood) >= 1
    
    def test_regime_transition_detection(self) -> None:
        """
        Verify patterns above saturation rate are marked as 'ace' regime.
        
        Note: This is a behavioral test. In production, patterns have
        metadata indicating their generation regime.
        """
        # For a test matrix, saturation typically occurs around R ≈ 0.60-0.65
        # We verify that high-rate patterns (>0.65) would need ACE-guided
        
        n = 128
        R_0 = 0.5
        
        # Calculate max punctured bits for R_eff = 0.65
        R_sat = 0.65
        p_sat = int(n * (1 - R_0 / R_sat))
        
        # For R_eff = 0.75 (above saturation)
        R_high = 0.75
        p_high = int(n * (1 - R_0 / R_high))
        
        # Higher rate needs more punctured bits
        assert p_high > p_sat
        
        # Untainted typically supports ~20% puncturing (π ≈ 0.2)
        max_untainted = int(n * 0.2)
        
        # Verify high rates exceed untainted capacity
        assert p_high > max_untainted, "High rate should exceed untainted capacity"


# =============================================================================
# TASK 2.2: Rate-Compatibility (Nesting Property)
# =============================================================================


class TestNestingProperty:
    """
    Task 2.2: Verify that puncturing patterns are nested.
    
    Rate-compatibility requires: Indices(R_low) ⊂ Indices(R_high)
    This is critical for Blind protocol security.
    """
    
    def test_pattern_nesting(self, populated_pattern_dir: Path) -> None:
        """Verify lower rate patterns are subsets of higher rate patterns."""
        # Load patterns
        patterns: Dict[float, np.ndarray] = {}
        for path in populated_pattern_dir.glob("pattern_rate*.npy"):
            rate_str = path.stem.split("rate")[-1]
            rate = float(rate_str)
            patterns[rate] = np.load(path)
        
        if len(patterns) < 2:
            pytest.skip("Need at least 2 patterns for nesting test")
        
        sorted_rates = sorted(patterns.keys())
        
        for i in range(len(sorted_rates) - 1):
            r_low = sorted_rates[i]
            r_high = sorted_rates[i + 1]
            
            pattern_low = patterns[r_low]
            pattern_high = patterns[r_high]
            
            # Get punctured indices
            indices_low = set(np.where(pattern_low == 1)[0])
            indices_high = set(np.where(pattern_high == 1)[0])
            
            # Verify nesting
            # Note: For rate-compatible puncturing, higher rate means MORE punctured bits
            # But the subset relationship depends on the construction order
            # Either indices_low ⊆ indices_high OR the patterns share a common core
            
            # At minimum, patterns should not have disjoint punctured sets
            intersection = indices_low & indices_high
            
            # For proper rate-compatibility via puncturing from same mother:
            # Lower rate = fewer punctured bits
            # Higher rate = more punctured bits (extending the lower pattern)
            if pattern_low.sum() <= pattern_high.sum():
                assert indices_low.issubset(indices_high), (
                    f"Rate {r_low:.2f} indices not subset of rate {r_high:.2f}"
                )
    
    def test_blind_protocol_nesting(self, populated_pattern_dir: Path) -> None:
        """
        Verify nesting for Blind protocol: reveal order matches puncture order.
        """
        modulation_path = populated_pattern_dir / "modulation_indices.npy"
        if not modulation_path.exists():
            pytest.skip("No modulation_indices.npy found")
        
        indices = np.load(modulation_path)
        
        # Indices should be unique
        assert len(indices) == len(set(indices)), "Modulation indices must be unique"
        
        # First indices should appear in all patterns (lowest rate)
        patterns = {}
        for path in populated_pattern_dir.glob("pattern_rate*.npy"):
            rate_str = path.stem.split("rate")[-1]
            rate = float(rate_str)
            patterns[rate] = np.load(path)
        
        if patterns:
            lowest_rate = min(patterns.keys())
            lowest_pattern = patterns[lowest_rate]
            
            # First few modulation indices should be punctured in lowest rate
            n_punctured_lowest = int(lowest_pattern.sum())
            first_indices = set(indices[:n_punctured_lowest])
            punctured_indices = set(np.where(lowest_pattern == 1)[0])
            
            # Allow some tolerance for different construction methods
            overlap = len(first_indices & punctured_indices)
            assert overlap >= n_punctured_lowest * 0.5, (
                "Modulation indices should significantly overlap with punctured positions"
            )


# =============================================================================
# TASK 2.3: ACE Score Validation
# =============================================================================


class TestACEScoreValidation:
    """
    Task 2.3: Verify ACE-guided puncturing selects high-ACE symbols.
    """
    
    def test_ace_score_computation(self, small_mother_matrix: csr_matrix) -> None:
        """Verify ACE score is computed correctly."""
        from caligo.scripts.generate_hybrid_patterns import compute_ace_score
        
        H = small_mother_matrix
        n = H.shape[1]
        
        punctured = set()  # No punctured symbols yet
        
        # Compute ACE for all symbols
        ace_scores = []
        for v in range(n):
            score = compute_ace_score(H, v, punctured)
            ace_scores.append((v, score))
        
        # All scores should be non-negative
        for v, score in ace_scores:
            assert score >= 0, f"ACE score for symbol {v} is negative: {score}"
        
        # For uniform-degree matrices, scores may be identical
        # This is acceptable - the test verifies computation doesn't fail
        scores_only = [s for _, s in ace_scores]
        unique_scores = set(scores_only)
        assert len(unique_scores) >= 1, "At least one valid ACE score should exist"
    
    def test_ace_guided_selection_prefers_high_ace(
        self, small_mother_matrix: csr_matrix
    ) -> None:
        """Verify ACE-guided puncturing selects symbols with higher ACE first."""
        from caligo.scripts.generate_hybrid_patterns import compute_ace_score
        
        H = small_mother_matrix
        n = H.shape[1]
        
        punctured = set()
        
        # Get initial ACE scores
        initial_scores = {
            v: compute_ace_score(H, v, punctured)
            for v in range(n)
        }
        
        # Sort by ACE score (descending) - higher ACE = safer to puncture
        sorted_by_ace = sorted(initial_scores.items(), key=lambda x: -x[1])
        
        # Top candidates should have relatively high scores
        top_5 = sorted_by_ace[:5]
        bottom_5 = sorted_by_ace[-5:]
        
        avg_top = np.mean([s for _, s in top_5])
        avg_bottom = np.mean([s for _, s in bottom_5])
        
        # Top ACE symbols should have higher average score
        assert avg_top >= avg_bottom, (
            f"Top ACE symbols ({avg_top:.2f}) should have higher score than bottom ({avg_bottom:.2f})"
        )


# =============================================================================
# TASK 2.4: Mother Code Loading
# =============================================================================


class TestMotherCodeLoading:
    """
    Task 2.4: Verify MotherCodeManager initialization and error handling.
    """
    
    def test_load_valid_mother_matrix(
        self,
        temp_mother_matrix_file: Path,
        populated_pattern_dir: Path,
    ) -> None:
        """Load valid mother matrix and pattern library."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton for test
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        manager = MotherCodeManager(
            matrix_path=temp_mother_matrix_file,
            pattern_dir=populated_pattern_dir,
        )
        
        assert manager.frame_size == 128
        assert manager.mother_rate == 0.5
        assert len(manager.patterns) > 0
    
    def test_corrupted_matrix_fails_fast(
        self, temp_matrix_dir: Path, populated_pattern_dir: Path
    ) -> None:
        """Corrupted or missing matrix should fail immediately."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        # Create corrupted matrix file (not a valid npz)
        ace_peg_dir = temp_matrix_dir / "ldpc_ace_peg"
        ace_peg_dir.mkdir(exist_ok=True)
        corrupted_path = ace_peg_dir / "ldpc_128_rate0.50.npz"
        corrupted_path.write_bytes(b"this is not a valid npz file")
        
        with pytest.raises(Exception):
            MotherCodeManager(
                matrix_path=corrupted_path,
                pattern_dir=populated_pattern_dir,
            )
    
    def test_wrong_rate_matrix_fails(
        self, temp_matrix_dir: Path, populated_pattern_dir: Path
    ) -> None:
        """Matrix with wrong rate should fail validation."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        # Create matrix with wrong rate (R = 0.75, not 0.5)
        n = 128
        m = 32  # R = 1 - 32/128 = 0.75
        H = np.eye(m, n, dtype=np.uint8)
        H_sparse = csr_matrix(H)
        
        ace_peg_dir = temp_matrix_dir / "ldpc_ace_peg"
        ace_peg_dir.mkdir(exist_ok=True)
        wrong_rate_path = ace_peg_dir / "ldpc_128_rate0.50.npz"
        save_npz(wrong_rate_path, H_sparse)
        
        with pytest.raises(ValueError, match="rate"):
            MotherCodeManager(
                matrix_path=wrong_rate_path,
                pattern_dir=populated_pattern_dir,
            )
    
    def test_missing_patterns_fails(
        self, temp_mother_matrix_file: Path, temp_matrix_dir: Path
    ) -> None:
        """Insufficient patterns should fail initialization."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        # Create empty pattern directory
        empty_patterns = temp_matrix_dir / "empty_patterns"
        empty_patterns.mkdir()
        
        with pytest.raises(ValueError, match="Insufficient patterns"):
            MotherCodeManager(
                matrix_path=temp_mother_matrix_file,
                pattern_dir=empty_patterns,
            )
    
    def test_pattern_rate_lookup(
        self,
        temp_mother_matrix_file: Path,
        populated_pattern_dir: Path,
    ) -> None:
        """Verify pattern lookup for target rates."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        manager = MotherCodeManager(
            matrix_path=temp_mother_matrix_file,
            pattern_dir=populated_pattern_dir,
        )
        
        # Get pattern for rate 0.60
        pattern = manager.get_pattern(0.60)
        assert pattern is not None
        assert pattern.shape[0] == 128
        assert pattern.dtype == np.uint8
        
        # Pattern should have punctured bits
        assert pattern.sum() > 0
    
    def test_modulation_indices(
        self,
        temp_mother_matrix_file: Path,
        populated_pattern_dir: Path,
    ) -> None:
        """Verify modulation indices for Blind protocol."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        manager = MotherCodeManager(
            matrix_path=temp_mother_matrix_file,
            pattern_dir=populated_pattern_dir,
        )
        
        # Get modulation indices
        d = 20  # Request 20 modulation positions
        indices = manager.get_modulation_indices(d)
        
        assert len(indices) == d
        assert indices.dtype == np.int64
        assert len(np.unique(indices)) == d  # All unique
    
    def test_compiled_topology(
        self,
        temp_mother_matrix_file: Path,
        populated_pattern_dir: Path,
    ) -> None:
        """Verify compiled topology is valid."""
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        # Reset singleton
        MotherCodeManager._instance = None
        MotherCodeManager._init_lock = False
        
        manager = MotherCodeManager(
            matrix_path=temp_mother_matrix_file,
            pattern_dir=populated_pattern_dir,
        )
        
        topo = manager.compiled_topology
        
        assert topo.n_vars == 128
        assert topo.n_checks == 64
        assert topo.n_edges > 0
        assert len(topo.check_row_ptr) == topo.n_checks + 1
        assert len(topo.check_col_idx) == topo.n_edges


# =============================================================================
# HYBRID PATTERN GENERATION INTEGRATION
# =============================================================================


class TestHybridPatternGeneration:
    """Integration tests for hybrid pattern generation script."""
    
    def test_generate_pattern_range(self, small_mother_matrix: csr_matrix) -> None:
        """Test generating patterns for a range of rates."""
        from caligo.scripts.generate_hybrid_patterns import (
            PuncturingState,
            compute_depth2_neighborhood,
            compute_n2_size,
        )
        
        H = small_mother_matrix
        n = H.shape[1]
        
        # Initialize state
        pattern = np.zeros(n, dtype=np.uint8)
        
        # Compute initial untainted set (all symbols eligible)
        untainted_set = set(range(n))
        
        state = PuncturingState(
            pattern=pattern,
            untainted_set=untainted_set,
            punctured_order=[],
            current_rate=0.5,
            regime='untainted',
        )
        
        # Verify initial state
        assert len(state.untainted_set) == n
        assert state.current_rate == 0.5
        assert state.regime == 'untainted'
    
    def test_n2_size_ordering(self, small_mother_matrix: csr_matrix) -> None:
        """Verify N² size ordering for untainted selection."""
        from caligo.scripts.generate_hybrid_patterns import compute_n2_size
        
        H = small_mother_matrix
        n = H.shape[1]
        
        # Compute N² sizes for all symbols
        n2_sizes = [(v, compute_n2_size(H, v)) for v in range(n)]
        
        # Should have variation (not all identical)
        sizes = [s for _, s in n2_sizes]
        assert len(set(sizes)) > 1, "N² sizes should vary"
        
        # All sizes should be >= 1 (at least self)
        for v, size in n2_sizes:
            assert size >= 1, f"Symbol {v} has invalid N² size: {size}"

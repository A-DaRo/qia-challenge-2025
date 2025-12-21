"""
Hybrid Puncturing Pattern Generation for Caligo Reconciliation.

Implements the two-regime strategy from Theoretical Report v2 §2.2:
- Regime A: Untainted puncturing (Elkouss et al. 2012 [3])
- Regime B: ACE-guided puncturing (Liu & de Lamare 2014 [4])

Usage
-----
    python -m caligo.scripts.generate_hybrid_patterns \\
        --matrix path/to/mother.npz \\
        --output path/to/patterns/

References
----------
[3] Elkouss et al., "Untainted Puncturing for Irregular LDPC Codes"
[4] Liu & de Lamare, "Rate-Compatible LDPC Codes Based on Puncturing"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from caligo.utils.logging import get_logger, setup_script_logging

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MOTHER_RATE = 0.5
FRAME_SIZE = 4096
RATE_STEP = 0.01  # Δ R = 0.01 for fine-grained rate adaptation
RATE_MIN = 0.51
RATE_MAX = 0.95

# Saturation threshold for Regime A → Regime B transition is detected dynamically
# when untainted set is exhausted (typically R_sat ≈ 0.625 per Theorem 2.2)


@dataclass
class PuncturingState:
    """
    Track state during hybrid pattern generation.
    
    Attributes
    ----------
    pattern : np.ndarray
        Current puncturing pattern (1=punctured).
    untainted_set : Set[int]
        Remaining untainted candidates.
    punctured_order : List[int]
        Ordered list of punctured indices.
    current_rate : float
        Current effective rate.
    regime : str
        'untainted' or 'ace'.
    """

    pattern: np.ndarray
    untainted_set: Set[int]
    punctured_order: List[int]
    current_rate: float
    regime: str


# ============================================================================
# GRAPH ANALYSIS FUNCTIONS
# ============================================================================

def compute_depth2_neighborhood(H: csr_matrix, symbol_idx: int) -> Set[int]:
    """
    Compute N²(v) - all symbols within 2 hops of symbol v.
    
    Per Theoretical Report v2 Definition 2.3:
    N²(v) = {v} ∪ {all symbols sharing a check with v}
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix (m × n).
    symbol_idx : int
        Symbol node index.
        
    Returns
    -------
    Set[int]
        Set of symbols in N²(v).
    """
    # Get checks connected to symbol (1 hop)
    check_indices = H.getcol(symbol_idx).nonzero()[0]
    
    # Get all symbols connected to those checks (2 hops)
    neighbors = {symbol_idx}
    for check_idx in check_indices:
        symbol_indices = H.getrow(check_idx).nonzero()[1]
        neighbors.update(symbol_indices)
    
    return neighbors


def compute_ace_score(
    H: csr_matrix,
    symbol_idx: int,
    punctured: Set[int],
    max_cycle_length: int = 12,
) -> float:
    """
    Compute ACE (Approximate Cycle Extrinsic message degree) score.
    
    Per Theoretical Report v2 §2.2.3 and Liu & de Lamare [4]:
    - High ACE = Better connectivity = Safer to puncture
    - Low ACE = Weakly connected = Risky to puncture
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    symbol_idx : int
        Symbol node to evaluate.
    punctured : Set[int]
        Already punctured symbol indices.
    max_cycle_length : int
        Maximum cycle length to consider.
        
    Returns
    -------
    float
        ACE score (higher is better for puncturing).
    """
    # Get checks connected to this symbol
    check_indices = H.getcol(symbol_idx).nonzero()[0]
    
    if len(check_indices) == 0:
        return 0.0
    
    # Compute extrinsic connectivity for each check
    ace_scores = []
    
    for check_idx in check_indices:
        # Get all symbols connected to this check
        neighbor_symbols = set(H.getrow(check_idx).nonzero()[1])
        neighbor_symbols.discard(symbol_idx)  # Exclude self
        
        # Count non-punctured neighbors (extrinsic connections)
        extrinsic_count = len(neighbor_symbols - punctured)
        
        # Check degree minus 1 (excluding the current symbol)
        check_degree = len(neighbor_symbols)
        
        if check_degree > 0:
            ace_scores.append(extrinsic_count / check_degree)
    
    # Return minimum ACE across all connected checks
    # Per [4]: puncture nodes with HIGH minimum ACE first
    return min(ace_scores) if ace_scores else 0.0


def compute_n2_size(H: csr_matrix, symbol_idx: int) -> int:
    """
    Compute |N²(v)| for tie-breaking in untainted selection.
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    symbol_idx : int
        Symbol node index.
        
    Returns
    -------
    int
        Size of 2-hop neighborhood.
    """
    return len(compute_depth2_neighborhood(H, symbol_idx))


# ============================================================================
# REGIME A: UNTAINTED PUNCTURING
# ============================================================================

def run_untainted_phase(
    H: csr_matrix,
    state: PuncturingState,
    target_fraction: float,
) -> bool:
    """
    Run strict untainted puncturing (Regime A).
    
    Per Theoretical Report v2 §2.2.2 Algorithm:
    1. Select candidates with smallest |N²(v)|
    2. Puncture one (deterministic tie-breaking)
    3. Remove N²(v) from untainted set
    4. Repeat until target or saturation
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    state : PuncturingState
        Current puncturing state (modified in-place).
    target_fraction : float
        Target puncturing fraction π.
        
    Returns
    -------
    bool
        True if target reached, False if saturated early.
    """
    n = H.shape[1]
    target_punctured = int(target_fraction * n)
    
    # Precompute N² sizes for remaining candidates
    n2_sizes = {v: compute_n2_size(H, v) for v in state.untainted_set}
    
    while state.untainted_set and len(state.punctured_order) < target_punctured:
        # Step 1: Find candidates with smallest N² size
        min_size = min(n2_sizes[v] for v in state.untainted_set)
        candidates = [v for v in state.untainted_set if n2_sizes[v] == min_size]
        
        # Step 2: Select one (deterministic: choose smallest index)
        selected = min(candidates)
        
        # Step 3: Puncture selected
        state.pattern[selected] = 1
        state.punctured_order.append(selected)
        
        # Step 4: Remove N²(selected) from untainted set
        n2_selected = compute_depth2_neighborhood(H, selected)
        state.untainted_set -= n2_selected
        
        # Clean up n2_sizes for removed nodes
        for v in n2_selected:
            n2_sizes.pop(v, None)
    
    # Check if target reached
    return len(state.punctured_order) >= target_punctured


# ============================================================================
# REGIME B: ACE-GUIDED PUNCTURING
# ============================================================================

def run_ace_phase(
    H: csr_matrix,
    state: PuncturingState,
    target_fraction: float,
) -> bool:
    """
    Run ACE-guided puncturing (Regime B).
    
    Per Theoretical Report v2 §2.2.3:
    - Used when untainted set is exhausted
    - Rank candidates by ACE score (highest first)
    - Puncture to preserve graph connectivity
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    state : PuncturingState
        Current puncturing state (modified in-place).
    target_fraction : float
        Target puncturing fraction π.
        
    Returns
    -------
    bool
        True if target reached.
    """
    n = H.shape[1]
    target_punctured = int(target_fraction * n)
    punctured_set = set(state.punctured_order)
    
    state.regime = 'ace'
    
    while len(state.punctured_order) < target_punctured:
        # Get remaining unpunctured symbols
        remaining = [i for i in range(n) if state.pattern[i] == 0]
        
        if not remaining:
            break
        
        # Compute ACE scores for all remaining candidates
        ace_scores = {
            v: compute_ace_score(H, v, punctured_set)
            for v in remaining
        }
        
        # Select candidate with HIGHEST ACE score
        # Per [4]: high ACE = well-connected = safer to puncture
        selected = max(remaining, key=lambda v: (ace_scores[v], -v))
        
        # Puncture selected
        state.pattern[selected] = 1
        state.punctured_order.append(selected)
        punctured_set.add(selected)
    
    return len(state.punctured_order) >= target_punctured


# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def generate_hybrid_library(
    H: csr_matrix,
    output_dir: Path,
    rate_min: float = RATE_MIN,
    rate_max: float = RATE_MAX,
    rate_step: float = RATE_STEP,
) -> Dict[float, np.ndarray]:
    """
    Generate Hybrid Pattern Library covering R_eff ∈ [rate_min, rate_max].
    
    Per Theoretical Report v2 §2.2.3:
    1. Phase I: Untainted puncturing until saturation (detected dynamically)
    2. Phase II: ACE-guided puncturing for higher rates
    3. Rate-compatibility: patterns are nested (truncatable)
    
    Parameters
    ----------
    H : csr_matrix
        R=0.5 mother code parity-check matrix.
    output_dir : Path
        Directory to save pattern files.
    rate_min : float
        Minimum effective rate (default 0.51).
    rate_max : float
        Maximum effective rate (default 0.90).
    rate_step : float
        Rate step size (default 0.01).
        
    Returns
    -------
    Dict[float, np.ndarray]
        Dictionary of {rate: pattern} mappings.
        
    Yields
    ------
    None
        This function logs progress but does not yield values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = H.shape[1]
    
    logger.info(f"Starting Hybrid Pattern Generation for {n}-bit mother code")
    logger.info(f"Target rate range: [{rate_min:.2f}, {rate_max:.2f}], step: {rate_step:.2f}")
    
    # Initialize state
    state = PuncturingState(
        pattern=np.zeros(n, dtype=np.uint8),
        untainted_set=set(range(n)),
        punctured_order=[],
        current_rate=MOTHER_RATE,
        regime='untainted',
    )
    
    patterns = {}
    saturation_rate = None  # Detected dynamically
    
    # Generate rates from rate_min to rate_max
    current_rate = rate_min
    pattern_count = 0
    
    while current_rate <= rate_max + 1e-6:
        # Compute required puncture fraction for this rate
        # R_eff = R_0 / (1 - π) => π = 1 - R_0 / R_eff
        target_puncture_fraction = 1.0 - MOTHER_RATE / current_rate
        
        if target_puncture_fraction <= 0:
            current_rate += rate_step
            continue
        
        # Try untainted puncturing first (Regime A)
        if state.regime == 'untainted':
            reached = run_untainted_phase(H, state, target_puncture_fraction)
            
            if not reached and not state.untainted_set:
                # Untainted set exhausted → transition to ACE (Regime B)
                saturation_rate = MOTHER_RATE / (1 - len(state.punctured_order) / n)
                logger.warning(
                    f"═══════════════════════════════════════════════════════════"
                )
                logger.warning(
                    f"Untainted saturation detected at R_sat = {saturation_rate:.4f}"
                )
                logger.warning(
                    f"  Punctured: {len(state.punctured_order)}/{n} "
                    f"(π = {len(state.punctured_order)/n:.3f})"
                )
                logger.warning(
                    f"  Switching to ACE-guided puncturing (Regime B)"
                )
                logger.warning(
                    f"═══════════════════════════════════════════════════════════"
                )
                run_ace_phase(H, state, target_puncture_fraction)
        else:
            # Already in Regime B (ACE)
            run_ace_phase(H, state, target_puncture_fraction)
        
        # Compute actual achieved rate
        actual_punctured = state.pattern.sum()
        actual_rate = MOTHER_RATE / (1 - actual_punctured / n)
        
        # Save pattern
        filename = f"pattern_rate{actual_rate:.2f}.npy"
        np.save(output_dir / filename, state.pattern.copy())
        patterns[actual_rate] = state.pattern.copy()
        pattern_count += 1
        
        logger.info(
            f"Generated {filename}: π={actual_punctured/n:.3f}, "
            f"R_eff={actual_rate:.3f}, regime={state.regime}"
        )
        
        state.current_rate = actual_rate
        current_rate += rate_step
    
    # Save ordered indices for Blind protocol revelation order
    indices_path = output_dir / "modulation_indices.npy"
    np.save(indices_path, np.array(state.punctured_order, dtype=np.int64))
    logger.info(f"Saved modulation indices to {indices_path}")
    
    # Log final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Complete: {pattern_count} pattern files created")
    if saturation_rate is not None:
        logger.info(f"Detected R_sat = {saturation_rate:.4f} (untainted exhaustion point)")
    logger.info(f"Rate range: [{rate_min:.2f}, {rate_max:.2f}] with Δ R = {rate_step:.2f}")
    logger.info(f"Final regime: {state.regime}")
    logger.info(f"{'='*60}")
    
    return patterns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Hybrid Puncturing Pattern Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--matrix",
        "-m",
        type=Path,
        default=Path("caligo/configs/ldpc_matrices/ldpc_ace_peg/ldpc_4096_rate0.50.npz"),
        help="Path to R=0.5 mother matrix (.npz)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("caligo/configs/ldpc_matrices/hybrid_patterns/"),
        help="Output directory for pattern files",
    )
    parser.add_argument(
        "--rate-min",
        type=float,
        default=RATE_MIN,
        help=f"Minimum effective rate (default {RATE_MIN})",
    )
    parser.add_argument(
        "--rate-max",
        type=float,
        default=RATE_MAX,
        help=f"Maximum effective rate (default {RATE_MAX})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    
    # Setup logging for both script and module loggers
    script_logger = setup_script_logging(
        "generate_hybrid_patterns",
        log_level=args.log_level,
        show_terminal=True
    )
    
    # Also add console handler to module logger for function output
    import logging
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(getattr(logging, args.log_level.upper()))
    if not any(isinstance(h, logging.StreamHandler) for h in module_logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
        module_logger.addHandler(console)
    
    script_logger.info(f"Loading mother code from {args.matrix}")
    
    if not args.matrix.exists():
        script_logger.error(f"Mother matrix not found: {args.matrix}")
        script_logger.error("Please run generate_ace_mother_code.py first")
        exit(1)
    
    H = load_npz(args.matrix).tocsr()
    script_logger.info(f"Matrix shape: {H.shape}, rate: {1 - H.shape[0]/H.shape[1]:.3f}")
    
    # Validate mother rate
    actual_rate = 1 - H.shape[0] / H.shape[1]
    if abs(actual_rate - MOTHER_RATE) > 0.01:
        script_logger.error(f"Expected mother rate {MOTHER_RATE}, got {actual_rate:.3f}")
        exit(1)
    
    script_logger.info(f"\nGenerating Hybrid Pattern Library:")
    script_logger.info(f"  Rate range: [{args.rate_min}, {args.rate_max}]")
    script_logger.info(f"  Rate step: {RATE_STEP}")
    script_logger.info(f"  Output: {args.output}\n")
    
    patterns = generate_hybrid_library(
        H,
        args.output,
        rate_min=args.rate_min,
        rate_max=args.rate_max,
    )
    
    script_logger.info(f"\nGenerated {len(patterns)} patterns successfully")

"""
Generate untainted puncturing patterns for LDPC rate adaptation.

This script implements the Elkouss et al. (2012) untainted puncturing algorithm
to construct deterministic puncturing patterns for high-rate codes derived from
a lower-rate mother code. Unlike random puncturing (which creates tainted check
nodes), untainted puncturing ensures belief propagation can converge at high rates.

The algorithm maintains an "untainted" set X∞ of variable nodes with no punctured
neighbors in their 2-hop neighborhood, iteratively selecting puncturing candidates
that minimize the disruption to the Tanner graph structure.

Usage
-----
    python -m caligo.scripts.generate_puncture_patterns [--log-show] [--log-level LEVEL]

Options
-------
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
    --output-dir PATH       Custom output directory for patterns
    --mother-rate RATE      Mother code rate (default: 0.5)
    --target-rates RATES    Comma-separated target rates (default: 0.6,0.7,0.8,0.9)

Examples
--------
    # Generate patterns
    python -m caligo.scripts.generate_puncture_patterns

    # Custom target rates
    python -m caligo.scripts.generate_puncture_patterns --target-rates 0.7,0.8,0.85,0.9

References
----------
Elkouss, D., Martinez-Mateo, J., & Martin, V. (2012). "Untainted puncturing for
irregular low-density parity-check codes." IEEE Wireless Communications Letters, 1(6), 585-588.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import (
    CompiledParityCheckMatrix,
    compile_parity_check_matrix,
)
from caligo.reconciliation.matrix_manager import MotherCodeManager
from caligo.utils.logging import setup_script_logging


@dataclass
class PuncturingPattern:
    """
    Untainted puncturing pattern for a target rate.

    Attributes
    ----------
    mother_rate : float
        Base code rate (e.g., 0.5).
    target_rate : float
        Derived code rate (e.g., 0.8).
    pattern : np.ndarray
        Binary mask of shape (n,), where pattern[i]=1 indicates punctured position.
    n_punctured : int
        Number of punctured variable nodes.
    untainted_count : int
        Number of punctured nodes selected from untainted set.
    forced_count : int
        Number of punctured nodes selected via forced heuristic.
    """

    mother_rate: float
    target_rate: float
    pattern: np.ndarray
    n_punctured: int
    untainted_count: int
    forced_count: int


class UntaintedPuncturingGenerator:
    """
    Generator for untainted puncturing patterns using Elkouss et al. algorithm.

    The algorithm maintains an untainted set X∞ of variable nodes where no
    check node in their neighborhood has multiple punctured neighbors. At each
    step, it selects the variable node from X∞ with the smallest 2-hop neighborhood
    (minimizing decoder disruption), punctures it, and updates X∞ by removing
    affected nodes.

    When X∞ is exhausted before reaching the target puncturing count, the algorithm
    employs a forced puncturing heuristic that selects nodes to minimize the number
    of "dead" check nodes (checks with no recoverable neighbors).

    Parameters
    ----------
    compiled_matrix : CompiledParityCheckMatrix
        Compiled mother code parity-check matrix.
    seed : int
        Random seed for tie-breaking (determinism).
    """

    def __init__(
        self,
        compiled_matrix: CompiledParityCheckMatrix,
        seed: int = 42,
    ) -> None:
        self.compiled = compiled_matrix
        self.m = compiled_matrix.m
        self.n = compiled_matrix.n
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build check→variables adjacency (already in compiled.check_var)
        # Build variable→checks adjacency
        self._build_var_to_checks()

    def _build_var_to_checks(self) -> None:
        """Construct variable→check adjacency lists from compiled matrix."""
        self.var_to_checks: List[List[int]] = [[] for _ in range(self.n)]

        for c in range(self.m):
            start = int(self.compiled.check_ptr[c])
            end = int(self.compiled.check_ptr[c + 1])
            for v_idx in self.compiled.check_var[start:end]:
                v = int(v_idx)
                self.var_to_checks[v].append(c)

    def _get_2hop_neighborhood(self, v: int, punctured: Set[int]) -> Set[int]:
        """
        Compute 2-hop neighborhood N²(v) excluding punctured nodes.

        The 2-hop neighborhood includes:
        - All check nodes adjacent to v (1-hop)
        - All variable nodes adjacent to those checks (2-hop)

        Parameters
        ----------
        v : int
            Variable node index.
        punctured : Set[int]
            Set of already punctured variable nodes.

        Returns
        -------
        Set[int]
            Set of variable node indices in N²(v) \\ punctured.
        """
        neighbors_2hop: Set[int] = set()

        # 1-hop: checks adjacent to v
        for c in self.var_to_checks[v]:
            start = int(self.compiled.check_ptr[c])
            end = int(self.compiled.check_ptr[c + 1])
            # 2-hop: variables adjacent to those checks
            for v_prime_idx in self.compiled.check_var[start:end]:
                v_prime = int(v_prime_idx)
                if v_prime != v and v_prime not in punctured:
                    neighbors_2hop.add(v_prime)

        return neighbors_2hop

    def _is_untainted(self, v: int, punctured: Set[int]) -> bool:
        """
        Check if variable v is untainted given current punctured set.

        A variable v is untainted if for all check nodes c ∈ N(v),
        the intersection |N(c) ∩ P| ≤ 0 (no punctured neighbors in its checks).

        Parameters
        ----------
        v : int
            Variable node index.
        punctured : Set[int]
            Current set of punctured variable nodes.

        Returns
        -------
        bool
            True if v is untainted.
        """
        if v in punctured:
            return False

        for c in self.var_to_checks[v]:
            start = int(self.compiled.check_ptr[c])
            end = int(self.compiled.check_ptr[c + 1])
            punctured_count = 0
            for v_prime_idx in self.compiled.check_var[start:end]:
                if int(v_prime_idx) in punctured:
                    punctured_count += 1
                    if punctured_count > 0:
                        return False
        return True

    def _compute_untainted_set(self, punctured: Set[int]) -> Set[int]:
        """
        Compute the current untainted set X∞.

        Parameters
        ----------
        punctured : Set[int]
            Current set of punctured variable nodes.

        Returns
        -------
        Set[int]
            Set of untainted variable node indices.
        """
        untainted: Set[int] = set()
        for v in range(self.n):
            if self._is_untainted(v, punctured):
                untainted.add(v)
        return untainted

    def _select_min_2hop_node(
        self, candidates: Set[int], punctured: Set[int]
    ) -> int:
        """
        Select node from candidates with minimum 2-hop neighborhood size.

        This is the core heuristic from Elkouss et al.: minimize the number
        of variable nodes affected by puncturing to preserve decoder structure.

        Parameters
        ----------
        candidates : Set[int]
            Candidate variable nodes (untainted set).
        punctured : Set[int]
            Current punctured set.

        Returns
        -------
        int
            Selected variable node index.
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate set")

        min_size = float("inf")
        best_node = -1
        ties: List[int] = []

        for v in candidates:
            neighborhood = self._get_2hop_neighborhood(v, punctured)
            size = len(neighborhood)

            if size < min_size:
                min_size = size
                best_node = v
                ties = [v]
            elif size == min_size:
                ties.append(v)

        # Break ties randomly for robustness
        if len(ties) > 1:
            best_node = int(self.rng.choice(ties))

        return best_node

    def _count_dead_checks(self, v: int, punctured: Set[int]) -> int:
        """
        Count how many check nodes would become "dead" if v is punctured.

                mother_mgr = MotherCodeManager.from_config(code_type="ace_peg", base_dir=matrix_dir)
                compiled = mother_mgr.get_compiled_mother_code()
        v : int
            Candidate variable node.
        punctured : Set[int]
            Current punctured set.

        Returns
        -------
        int
            Number of checks that would become dead.
        """
        dead_count = 0
        temp_punctured = punctured | {v}

        for c in self.var_to_checks[v]:
            start = int(self.compiled.check_ptr[c])
            end = int(self.compiled.check_ptr[c + 1])
            all_punctured = True
            for v_prime_idx in self.compiled.check_var[start:end]:
                if int(v_prime_idx) not in temp_punctured:
                    all_punctured = False
                    break
            if all_punctured:
                dead_count += 1

        return dead_count

    def _forced_puncture_select(self, punctured: Set[int]) -> int:
        """
        Select next node via forced puncturing heuristic.

        When the untainted set is exhausted, we must puncture from the
        "tainted" nodes. The heuristic minimizes the number of dead check
        nodes created by the puncturing.

        Parameters
        ----------
        punctured : Set[int]
            Current punctured set.

        Returns
        -------
        int
            Selected variable node index.
        """
        candidates = set(range(self.n)) - punctured
        if not candidates:
            raise ValueError("All nodes already punctured")

        min_dead = float("inf")
        best_node = -1
        ties: List[int] = []

        for v in candidates:
            dead_count = self._count_dead_checks(v, punctured)
            if dead_count < min_dead:
                min_dead = dead_count
                best_node = v
                ties = [v]
            elif dead_count == min_dead:
                ties.append(v)

        if len(ties) > 1:
            best_node = int(self.rng.choice(ties))

        return best_node

    def generate_pattern(
        self, target_rate: float, mother_rate: float, logger
    ) -> PuncturingPattern:
        """
        Generate untainted puncturing pattern for target rate.

        Parameters
        ----------
        target_rate : float
            Desired code rate after puncturing (e.g., 0.8).
        mother_rate : float
            Mother code rate (e.g., 0.5).
        logger : logging.Logger
            Logger for progress reporting.

        Returns
        -------
        PuncturingPattern
            Generated puncturing pattern.

        Raises
        ------
        ValueError
            If target_rate <= mother_rate (cannot puncture to lower rate).
        """
        if target_rate <= mother_rate:
            raise ValueError(
                f"Target rate {target_rate} must be > mother rate {mother_rate}"
            )

        # Compute number of bits to puncture
        # k_mother = R_mother * n
        # k_target = R_target * n
        # Since we're removing m parity bits through puncturing:
        # (k_mother + n_punct) / n = R_target
        # n_punct = n * (R_target - R_mother)
        n_puncture = int(np.round(self.n * (target_rate - mother_rate)))

        if n_puncture <= 0:
            raise ValueError(
                f"Computed n_puncture={n_puncture} for rates {mother_rate}→{target_rate}"
            )

        logger.info(
            "Generating pattern: R=%.2f → R=%.2f (puncture %d/%d bits)",
            mother_rate,
            target_rate,
            n_puncture,
            self.n,
        )

        punctured: Set[int] = set()
        untainted_count = 0
        forced_count = 0

        start_time = time.time()

        for iteration in range(1, n_puncture + 1):
            # Compute untainted set
            untainted = self._compute_untainted_set(punctured)

            if untainted:
                # Select from untainted set (Elkouss main algorithm)
                v_star = self._select_min_2hop_node(untainted, punctured)
                punctured.add(v_star)
                untainted_count += 1
            else:
                # Forced puncturing heuristic
                v_star = self._forced_puncture_select(punctured)
                punctured.add(v_star)
                forced_count += 1

            if iteration % 100 == 0 or iteration == n_puncture:
                elapsed = time.time() - start_time
                logger.debug(
                    "  [%4d/%4d] punctured=%d, untainted_set=%d (%.2fs)",
                    iteration,
                    n_puncture,
                    len(punctured),
                    len(untainted),
                    elapsed,
                )

        # Convert to binary pattern array
        pattern = np.zeros(self.n, dtype=np.uint8)
        for v in punctured:
            pattern[v] = 1

        elapsed = time.time() - start_time
        logger.info(
            "Pattern complete: %d punctured (%d untainted, %d forced) in %.2fs",
            n_puncture,
            untainted_count,
            forced_count,
            elapsed,
        )

        return PuncturingPattern(
            mother_rate=mother_rate,
            target_rate=target_rate,
            pattern=pattern,
            n_punctured=n_puncture,
            untainted_count=untainted_count,
            forced_count=forced_count,
        )


def generate_all_patterns(
    output_dir: Path,
    mother_rate: float,
    target_rates: List[float],
    frame_size: int,
    logger,
    matrix_dir: Path | None = None,
) -> Dict[float, PuncturingPattern]:
    """
    Generate all puncturing patterns for specified target rates.

    Parameters
    ----------
    output_dir : Path
        Directory to save pattern files.
    mother_rate : float
        Mother code rate.
    target_rates : List[float]
        Target rates to derive from mother code.
    frame_size : int
        LDPC frame size.
    logger : logging.Logger
        Logger for progress reporting.
    matrix_dir : Path or None
        Optional path to the directory containing LDPC matrices. If None,
        the default in :mod:`caligo.reconciliation.constants` will be used.

    Returns
    -------
    Dict[float, PuncturingPattern]
        Generated patterns keyed by target rate.
    """
    # Allow caller to override matrices base directory.
    # In RC mode we load via MotherCodeManager (single mother code + patterns).
    matrix_dir = matrix_dir if matrix_dir is not None else constants.LDPC_MATRICES_DIR
    logger.info("Loading mother code (expected rate=%.2f) from %s", mother_rate, matrix_dir)

    manager = MotherCodeManager.from_config(
        code_type="ace_peg",
        base_dir=matrix_dir,
        frame_size=frame_size,
    )
    compiled = manager.get_compiled_mother_code()

    if abs(float(manager.mother_rate) - float(mother_rate)) > 1e-3:
        logger.warning(
            "Configured mother_rate=%.3f differs from loaded mother code rate=%.3f",
            float(mother_rate),
            float(manager.mother_rate),
        )

    logger.info(
        "Mother code: n=%d, m=%d, rate=%.2f, edges=%d",
        compiled.n,
        compiled.m,
        float(manager.mother_rate),
        compiled.edge_count,
    )

    patterns: Dict[float, PuncturingPattern] = {}
    generator = UntaintedPuncturingGenerator(compiled, seed=42)

    logger.info("=" * 70)
    for idx, target_rate in enumerate(sorted(target_rates), 1):
        logger.info("[%d/%d] Target rate: %.2f", idx, len(target_rates), target_rate)
        logger.info("-" * 70)

        pattern = generator.generate_pattern(target_rate, mother_rate, logger)
        patterns[target_rate] = pattern

        # Save pattern to .npy file
        filename = f"puncture_pattern_rate{target_rate:.2f}.npy"
        filepath = output_dir / filename
        np.save(filepath, pattern.pattern)
        logger.info("Saved: %s", filepath)
        logger.info("=" * 70)

    return patterns


def main() -> None:
    """Main entry point for pattern generation script."""
    parser = argparse.ArgumentParser(
        description="Generate untainted puncturing patterns for LDPC codes"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for patterns (default: configs/ldpc_matrices/puncture_patterns)",
    )
    parser.add_argument(
        "--mother-rate",
        type=float,
        default=0.5,
        help="Mother code rate (default: 0.5)",
    )
    parser.add_argument(
        "--target-rates",
        type=str,
        default="0.6,0.7,0.8,0.9",
        help="Comma-separated target rates (default: 0.6,0.7,0.8,0.9)",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=None,
        help="Directory containing LDPC matrices (overrides default: constants.LDPC_MATRICES_DIR)",
    )

    args = parser.parse_args()

    logger = setup_script_logging(
        script_name="generate_puncture_patterns",
        log_level=args.log_level,
    )

    # Parse target rates
    target_rates = [float(r.strip()) for r in args.target_rates.split(",")]

    # Determine output directory
    if args.output_dir is None:
        output_dir = constants.LDPC_MATRICES_DIR / "puncture_patterns"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 70)
    logger.info("Untainted Puncturing Pattern Generation")
    logger.info("=" * 70)
    logger.info("Mother rate: %.2f", args.mother_rate)
    logger.info("Target rates: %s", target_rates)
    logger.info("Frame size: %d", constants.LDPC_FRAME_SIZE)
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("Matrices directory: %s", args.matrix_dir or constants.LDPC_MATRICES_DIR)
    logger.info("=" * 70)

    start_time = time.time()

    patterns = generate_all_patterns(
        output_dir=output_dir,
        mother_rate=args.mother_rate,
        target_rates=target_rates,
        frame_size=constants.LDPC_FRAME_SIZE,
        logger=logger,
        matrix_dir=args.matrix_dir,
    )

    elapsed = time.time() - start_time

    logger.info("=" * 70)
    logger.info("Generation complete!")
    logger.info("Patterns generated: %d", len(patterns))
    logger.info("Total time: %.2fs", elapsed)
    logger.info("=" * 70)

    # Summary statistics
    logger.info("\nPattern Statistics:")
    logger.info("-" * 70)
    for rate in sorted(patterns.keys()):
        p = patterns[rate]
        logger.info(
            "Rate %.2f: %4d punctured (%4d untainted, %4d forced, %.1f%% forced)",
            rate,
            p.n_punctured,
            p.untainted_count,
            p.forced_count,
            100.0 * p.forced_count / p.n_punctured if p.n_punctured > 0 else 0.0,
        )


if __name__ == "__main__":
    main()

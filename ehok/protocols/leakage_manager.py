"""
Leakage Safety Manager for E-HOK Protocol.

This module implements the leakage budget enforcement for Phase III
reconciliation (TASK-LEAKAGE-MGR-001), tracking syndrome and hash
leakage with hard cap enforcement.

Security Rationale
------------------
Every transmitted bit of syndrome information and verification hash
constitutes information leaked to a potential adversary. The NSM security
proofs require this "wiretap cost" to be subtracted from the extractable
entropy. Additionally, a hard cap prevents "feigned failure" attacks where
a dishonest party forces repeated reconciliation attempts to exhaust the
security margin.

Wiretap Cost Model
------------------
Following Lupo et al. (2023) and Schaffner et al. (2009):
    |Σ| = Σᵢ |Sᵢ| + Σᵢ |hᵢ|

where Sᵢ is the LDPC syndrome for block i and hᵢ is the verification hash.

Integration
-----------
1. LDPCReconciliator calls `account_syndrome` after each syndrome transmission
2. Verification step calls `account_hash` after each hash transmission
3. Protocol checks `is_cap_exceeded` and aborts if True
4. `wiretap_cost_bits` is passed to Phase IV for key length calculation

References
----------
- Lupo et al. (2023): Eq. (3) wiretap cost formalization
- Erven et al. (2014): One-way LDPC implementation
- Phase III analysis: Wiretap Channel Model
- sprint_2_specification.md Section 4.2-4.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default maximum leakage (conservative value)
# Can be derived from security margin calculations
DEFAULT_MAX_LEAKAGE_BITS = 1_000_000  # 1 Mbit default cap


# Abort code
ABORT_CODE_LEAKAGE_CAP_EXCEEDED = "ABORT-III-LEAK-001"


# =============================================================================
# Block Reconciliation Report
# =============================================================================


@dataclass(frozen=True)
class BlockReconciliationReport:
    """
    Report for a single LDPC reconciliation block.

    This is the per-block output from LDPCReconciliator that feeds
    into LeakageSafetyManager for cumulative accounting.

    Attributes
    ----------
    block_index : int
        Index of this block in the sequence.
    syndrome_bits : int
        Number of syndrome bits transmitted for this block.
    hash_bits : int
        Number of verification hash bits transmitted for this block.
    decode_converged : bool
        Whether BP decoder converged successfully.
    hash_verified : bool
        Whether hash verification passed.
    iterations : int
        Number of BP decoder iterations used.

    Properties
    ----------
    total_leakage_bits : int
        syndrome_bits + hash_bits for this block.
    """

    block_index: int
    syndrome_bits: int
    hash_bits: int
    decode_converged: bool
    hash_verified: bool
    iterations: int = 0

    @property
    def total_leakage_bits(self) -> int:
        """Total leakage from this block."""
        return self.syndrome_bits + self.hash_bits


@dataclass
class LeakageState:
    """
    Mutable state for leakage tracking.

    Attributes
    ----------
    total_syndrome_bits : int
        Cumulative syndrome bits transmitted.
    total_hash_bits : int
        Cumulative hash bits transmitted.
    block_reports : List[BlockReconciliationReport]
        All block reports received.
    cap_exceeded : bool
        Whether leakage cap has been exceeded.
    """

    total_syndrome_bits: int = 0
    total_hash_bits: int = 0
    block_reports: List[BlockReconciliationReport] = field(default_factory=list)
    cap_exceeded: bool = False


# =============================================================================
# Leakage Safety Manager (TASK-LEAKAGE-MGR-001)
# =============================================================================


class LeakageSafetyManager:
    """
    Enforces hard leakage cap for reconciliation phase.

    This class tracks cumulative syndrome and hash leakage, enforces
    the maximum allowed leakage cap, and produces the wiretap cost
    value required by Phase IV.

    Attributes
    ----------
    max_leakage_bits : int
        Maximum allowed total leakage (L_max).

    Security Model
    --------------
    Every bit transmitted (syndrome or hash) is assumed to be observed
    by the adversary. The total forms the "wiretap cost" that must be
    subtracted from extractable entropy.

    References
    ----------
    - sprint_2_specification.md Section 4.2-4.4
    - Phase III analysis: Wiretap Channel Model
    - Lupo et al. (2023): Wiretap cost formalization

    Examples
    --------
    >>> manager = LeakageSafetyManager(max_leakage_bits=100000)
    >>> manager.account_syndrome(block_index=0, syndrome_bits=5000)
    True
    >>> manager.account_hash(block_index=0, hash_bits=128)
    True
    >>> manager.wiretap_cost_bits
    5128
    >>> manager.is_cap_exceeded
    False
    """

    def __init__(self, max_leakage_bits: int = DEFAULT_MAX_LEAKAGE_BITS) -> None:
        """
        Initialize leakage safety manager.

        Parameters
        ----------
        max_leakage_bits : int
            Maximum allowed total leakage L_max. Default: 1,000,000 bits.

        Raises
        ------
        ValueError
            If max_leakage_bits <= 0.
        """
        if max_leakage_bits <= 0:
            raise ValueError(
                f"max_leakage_bits must be positive, got {max_leakage_bits}"
            )

        self._max_leakage_bits = max_leakage_bits
        self._state = LeakageState()

        logger.debug(
            "LeakageSafetyManager initialized with L_max=%d bits",
            max_leakage_bits,
        )

    @property
    def max_leakage_bits(self) -> int:
        """Get maximum allowed leakage."""
        return self._max_leakage_bits

    @property
    def total_syndrome_bits(self) -> int:
        """Get cumulative syndrome leakage."""
        return self._state.total_syndrome_bits

    @property
    def total_hash_bits(self) -> int:
        """Get cumulative hash leakage."""
        return self._state.total_hash_bits

    @property
    def total_leakage_bits(self) -> int:
        """Get total leakage (syndrome + hash)."""
        return self._state.total_syndrome_bits + self._state.total_hash_bits

    @property
    def wiretap_cost_bits(self) -> int:
        """
        Get wiretap cost for Phase IV.

        This is the value |Σ| to be subtracted from extractable entropy.

        Returns
        -------
        int
            Total leakage in bits.
        """
        return self.total_leakage_bits

    @property
    def is_cap_exceeded(self) -> bool:
        """Check if leakage cap has been exceeded."""
        return self._state.cap_exceeded

    @property
    def remaining_budget_bits(self) -> int:
        """Get remaining leakage budget."""
        remaining = self._max_leakage_bits - self.total_leakage_bits
        return max(0, remaining)

    @property
    def block_reports(self) -> List[BlockReconciliationReport]:
        """Get all block reports."""
        return list(self._state.block_reports)

    @property
    def num_blocks_processed(self) -> int:
        """Get number of blocks processed."""
        return len(self._state.block_reports)

    def account_syndrome(self, block_index: int, syndrome_bits: int) -> bool:
        """
        Account for syndrome transmission leakage.

        Parameters
        ----------
        block_index : int
            Index of the block.
        syndrome_bits : int
            Number of syndrome bits transmitted.

        Returns
        -------
        bool
            True if within budget, False if cap exceeded.

        Raises
        ------
        ValueError
            If syndrome_bits < 0.
        """
        if syndrome_bits < 0:
            raise ValueError(f"syndrome_bits must be non-negative, got {syndrome_bits}")

        self._state.total_syndrome_bits += syndrome_bits

        logger.debug(
            "LEAKAGE_SYNDROME block=%d bits=%d cumulative_syndrome=%d",
            block_index,
            syndrome_bits,
            self._state.total_syndrome_bits,
        )

        return self._check_cap()

    def account_hash(self, block_index: int, hash_bits: int) -> bool:
        """
        Account for hash transmission leakage.

        Parameters
        ----------
        block_index : int
            Index of the block.
        hash_bits : int
            Number of hash bits transmitted.

        Returns
        -------
        bool
            True if within budget, False if cap exceeded.

        Raises
        ------
        ValueError
            If hash_bits < 0.
        """
        if hash_bits < 0:
            raise ValueError(f"hash_bits must be non-negative, got {hash_bits}")

        self._state.total_hash_bits += hash_bits

        logger.debug(
            "LEAKAGE_HASH block=%d bits=%d cumulative_hash=%d",
            block_index,
            hash_bits,
            self._state.total_hash_bits,
        )

        return self._check_cap()

    def account_block(self, report: BlockReconciliationReport) -> bool:
        """
        Account for a complete block reconciliation.

        Parameters
        ----------
        report : BlockReconciliationReport
            Block reconciliation report.

        Returns
        -------
        bool
            True if within budget, False if cap exceeded.
        """
        self._state.block_reports.append(report)
        self._state.total_syndrome_bits += report.syndrome_bits
        self._state.total_hash_bits += report.hash_bits

        logger.info(
            "LEAKAGE_BLOCK block=%d syndrome=%d hash=%d converged=%s verified=%s "
            "cumulative_total=%d",
            report.block_index,
            report.syndrome_bits,
            report.hash_bits,
            report.decode_converged,
            report.hash_verified,
            self.total_leakage_bits,
        )

        return self._check_cap()

    def _check_cap(self) -> bool:
        """
        Check if leakage cap has been exceeded.

        Returns
        -------
        bool
            True if within budget, False if cap exceeded.
        """
        if self.total_leakage_bits > self._max_leakage_bits:
            if not self._state.cap_exceeded:
                self._state.cap_exceeded = True
                logger.warning(
                    "LEAKAGE_CAP_EXCEEDED total=%d > max=%d abort_code=%s",
                    self.total_leakage_bits,
                    self._max_leakage_bits,
                    ABORT_CODE_LEAKAGE_CAP_EXCEEDED,
                )
            return False
        return True

    def check_abort(self) -> Optional[str]:
        """
        Check if protocol should abort due to leakage.

        Returns
        -------
        Optional[str]
            Abort code if cap exceeded, None otherwise.
        """
        if self.is_cap_exceeded:
            return ABORT_CODE_LEAKAGE_CAP_EXCEEDED
        return None

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary of leakage accounting.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary for logging/reporting.
        """
        return {
            "total_syndrome_bits": self.total_syndrome_bits,
            "total_hash_bits": self.total_hash_bits,
            "total_leakage_bits": self.total_leakage_bits,
            "wiretap_cost_bits": self.wiretap_cost_bits,
            "max_leakage_bits": self._max_leakage_bits,
            "remaining_budget_bits": self.remaining_budget_bits,
            "cap_exceeded": self.is_cap_exceeded,
            "num_blocks_processed": self.num_blocks_processed,
        }

    def reset(self) -> None:
        """
        Reset leakage state for a new session.

        Clears all accumulated leakage but keeps the cap configuration.
        """
        self._state = LeakageState()
        logger.debug("LeakageSafetyManager reset")


# =============================================================================
# Leakage Budget Calculator
# =============================================================================


def compute_max_leakage_budget(
    n_sifted_bits: int,
    min_entropy_rate: float,
    epsilon_sec: float,
    safety_margin: float = 0.1,
) -> int:
    """
    Compute maximum leakage budget from security parameters.

    This derives L_max such that some secure key can still be extracted
    after accounting for leakage and privacy amplification costs.

    Formula:
        L_max = floor((h_min * n - 2*log2(1/ε_sec)) * (1 - safety_margin))

    Parameters
    ----------
    n_sifted_bits : int
        Number of sifted key bits.
    min_entropy_rate : float
        Min-entropy rate per sifted bit (from NSMBoundsCalculator).
    epsilon_sec : float
        Security parameter.
    safety_margin : float
        Reserve fraction to ensure positive key length. Default: 0.1 (10%).

    Returns
    -------
    int
        Maximum allowed leakage in bits.

    Raises
    ------
    ValueError
        If computed budget would be non-positive.

    Examples
    --------
    >>> budget = compute_max_leakage_budget(
    ...     n_sifted_bits=100000,
    ...     min_entropy_rate=0.5,
    ...     epsilon_sec=1e-10,
    ...     safety_margin=0.1
    ... )
    >>> budget > 0
    True
    """
    import math

    if n_sifted_bits <= 0:
        raise ValueError(f"n_sifted_bits must be positive, got {n_sifted_bits}")
    if not 0 < min_entropy_rate <= 1:
        raise ValueError(
            f"min_entropy_rate must be in (0, 1], got {min_entropy_rate}"
        )
    if not 0 < epsilon_sec < 1:
        raise ValueError(f"epsilon_sec must be in (0, 1), got {epsilon_sec}")
    if not 0 <= safety_margin < 1:
        raise ValueError(f"safety_margin must be in [0, 1), got {safety_margin}")

    # Total extractable entropy
    total_entropy = min_entropy_rate * n_sifted_bits

    # Privacy amplification cost
    pa_cost = 2 * math.log2(1.0 / epsilon_sec)

    # Available for leakage + key
    available = total_entropy - pa_cost

    if available <= 0:
        raise ValueError(
            f"No extractable entropy: h_min*n={total_entropy:.1f}, PA_cost={pa_cost:.1f}"
        )

    # Apply safety margin
    max_leakage = int(available * (1 - safety_margin))

    if max_leakage <= 0:
        raise ValueError(
            f"Computed max_leakage is non-positive: {max_leakage}. "
            f"Consider reducing safety_margin or increasing n_sifted_bits."
        )

    logger.debug(
        "LEAKAGE_BUDGET_COMPUTED L_max=%d (n=%d, h_min=%.4f, ε_sec=%.2e, margin=%.2f)",
        max_leakage,
        n_sifted_bits,
        min_entropy_rate,
        epsilon_sec,
        safety_margin,
    )

    return max_leakage


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_MAX_LEAKAGE_BITS",
    "ABORT_CODE_LEAKAGE_CAP_EXCEEDED",
    # Data structures
    "BlockReconciliationReport",
    "LeakageState",
    # Classes
    "LeakageSafetyManager",
    # Functions
    "compute_max_leakage_budget",
]

"""
Leakage Tracker for Wiretap Cost Accounting.

Tracks cumulative information leakage during reconciliation and
enforces the safety cap to prevent protocol compromise.

In E-HOK, syndromes leak directly to Bob (potential adversary),
not an external eavesdropper. This leakage directly reduces the
extractable secure key length.

References
----------
- König et al. (2012): Min-entropy bounds
- Schaffner et al. (2009): Wiretap cost model
- Lupo et al. (2023): Error-tolerant OT bounds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from caligo.reconciliation import constants
from caligo.types.exceptions import LeakageBudgetExceeded
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Leakage Record
# =============================================================================


@dataclass
class LeakageRecord:
    """
    Record of a single leakage event.

    Attributes
    ----------
    syndrome_bits : int
        Syndrome bits transmitted.
    hash_bits : int
        Verification hash bits.
    retry_penalty_bits : int
        Conservative penalty for retry/interaction metadata.
    shortening_bits : float
        Shortening position leakage (log2 combinatorial bound).
    block_id : int
        Associated block identifier.
    iteration : int
        Blind iteration number (1-indexed).
    """

    syndrome_bits: int
    hash_bits: int = constants.LDPC_HASH_BITS
    retry_penalty_bits: int = 0
    shortening_bits: float = 0.0
    block_id: int = 0
    iteration: int = 1

    @property
    def total_leakage(self) -> int:
        """Total leakage for this record: syndrome + hash + retry + shortening."""
        return int(math.ceil(
            self.syndrome_bits
            + self.hash_bits
            + self.retry_penalty_bits
            + self.shortening_bits
        ))


# =============================================================================
# Leakage Tracker
# =============================================================================


class LeakageTracker:
    """
    Accumulate and enforce reconciliation leakage bounds with circuit breaker.

    Tracks all information disclosed during Phase III and immediately aborts
    if the safety cap is exceeded (circuit breaker pattern).

    Per Implementation Report v2 §7: Circuit breaker raises LeakageBudgetExceeded
    exception before security violation occurs.

    Parameters
    ----------
    safety_cap : int
        Maximum allowed total leakage in bits.
    abort_on_exceed : bool, optional
        If True (default), immediately raise exception when cap exceeded.
        If False, only log warnings (for testing/debugging).

    Attributes
    ----------
    records : List[LeakageRecord]
        History of leakage events.
    safety_cap : int
        Configured maximum leakage.
    """

    def __init__(self, safety_cap: int, abort_on_exceed: bool = True) -> None:
        if safety_cap < 0:
            raise ValueError("safety_cap must be non-negative")
        self.safety_cap = safety_cap
        self.abort_on_exceed = abort_on_exceed
        self.records: List[LeakageRecord] = []

    def record(self, event: LeakageRecord) -> None:
        """
        Add a leakage event with immediate circuit breaker enforcement.

        Parameters
        ----------
        event : LeakageRecord
            Leakage event to record.

        Raises
        ------
        LeakageBudgetExceeded
            If cumulative leakage exceeds safety_cap and abort_on_exceed=True.

        Notes
        -----
        This is the CIRCUIT BREAKER implementation per Implementation Report v2 §7.
        Raises exception BEFORE security violation propagates through protocol.
        """
        self.records.append(event)
        current_leakage = self.total_leakage
        
        logger.debug(
            "Leakage recorded: block=%d, syndrome=%d, hash=%d, short=%.1f, iter=%d, "
            "total=%d/%d",
            event.block_id, event.syndrome_bits, event.hash_bits, 
            event.shortening_bits, event.iteration, current_leakage, self.safety_cap
        )

        # CIRCUIT BREAKER: Immediate enforcement
        if self.abort_on_exceed and current_leakage > self.safety_cap:
            logger.error(
                "Leakage budget EXCEEDED: %d > %d (margin: %d)",
                current_leakage, self.safety_cap, current_leakage - self.safety_cap
            )
            raise LeakageBudgetExceeded(
                f"Cumulative leakage {current_leakage} bits exceeds "
                f"safety cap {self.safety_cap} bits",
                actual_leakage=current_leakage,
                max_allowed=self.safety_cap,
            )

    def record_block(
        self,
        syndrome_bits: int = 0,
        hash_bits: int = constants.LDPC_HASH_BITS,
        retry_penalty_bits: int = 0,
        block_id: int = 0,
        iteration: int = 1,
        syndrome_length: int = None,  # Alias for syndrome_bits
        n_shortened: int = 0,
        frame_size: int = constants.LDPC_FRAME_SIZE,
    ) -> None:
        """
        Record leakage for a reconciled block.

        Computes shortening leakage as upper bound.

        Parameters
        ----------
        syndrome_bits : int
            Syndrome bits transmitted.
        hash_bits : int
            Hash bits transmitted.
        retry_penalty_bits : int
            Conservative retry penalty bits.
        block_id : int
            Block identifier.
        iteration : int
            Blind iteration number.
        syndrome_length : int, optional
            Alias for syndrome_bits (backward compatibility).
        n_shortened : int
            Number of shortened bits.
        frame_size : int
            LDPC frame size.
        """
        # Support both syndrome_bits and syndrome_length
        actual_syndrome = syndrome_bits if syndrome_length is None else syndrome_length
        
        # Shortening leakage: log2(C(n, n_s)) ≈ n_s * log2(n/n_s)
        shortening_bits = 0.0
        if n_shortened > 0 and frame_size > n_shortened:
            ratio = frame_size / n_shortened
            shortening_bits = n_shortened * math.log2(ratio)

        event = LeakageRecord(
            syndrome_bits=actual_syndrome,
            hash_bits=hash_bits,
            retry_penalty_bits=retry_penalty_bits,
            shortening_bits=shortening_bits,
            block_id=block_id,
            iteration=iteration,
        )
        self.record(event)

    def record_reveal(
        self,
        block_id: int,
        iteration: int,
        revealed_bits: int,
    ) -> None:
        """
        Record Blind iteration reveal leakage (syndrome already counted).

        Per Implementation Report v2 §5.3: Blind protocol reveals additional
        bits in iterations 2+. This method records the revealed_bits leakage
        without double-counting the syndrome.

        Parameters
        ----------
        block_id : int
            Block identifier.
        iteration : int
            Blind iteration number (≥2).
        revealed_bits : int
            Number of bits revealed in this iteration (Δ_i).

        Raises
        ------
        LeakageBudgetExceeded
            If cumulative leakage exceeds safety_cap.
        """
        event = LeakageRecord(
            syndrome_bits=0,  # Syndrome already counted in iteration 1
            hash_bits=0,       # Hash already counted
            retry_penalty_bits=revealed_bits,  # Use retry_penalty for reveal bits
            shortening_bits=0.0,
            block_id=block_id,
            iteration=iteration,
        )
        self.record(event)

    @property
    def total_leakage(self) -> int:
        """
        Compute total cumulative leakage.

        Returns
        -------
        int
            Total leakage in bits (ceiling of float sum).
        """
        total = sum(
            r.syndrome_bits + r.hash_bits + r.retry_penalty_bits + r.shortening_bits
            for r in self.records
        )
        return int(math.ceil(total))

    @property
    def remaining_budget(self) -> int:
        """
        Remaining leakage budget before abort.

        Returns
        -------
        int
            Bits remaining (may be negative if exceeded).
        """
        return self.safety_cap - self.total_leakage

    def check_safety(self) -> bool:
        """
        Check if leakage is within safety cap.

        Returns
        -------
        bool
            True if total_leakage <= safety_cap.
        """
        return self.total_leakage <= self.safety_cap

    def should_abort(self) -> bool:
        """
        Check if protocol should abort due to leakage.

        Returns
        -------
        bool
            True if safety cap exceeded.
        """
        return not self.check_safety()

    @property
    def num_blocks(self) -> int:
        """Number of blocks recorded."""
        return len(self.records)


# =============================================================================
# Safety Cap Computation
# =============================================================================


def compute_safety_cap(
    n_sifted: int,
    qber: float,
    epsilon: float = 0.0,
    raw_key_length: int = None,  # Alias for n_sifted
    min_entropy_rate: float = None,  # Alternative interface
    target_key_length: int = None,
    security_parameter: float = 1e-10,
) -> int:
    """
    Compute maximum safe syndrome leakage.

    Supports two interfaces:
    1. Simple (test style): n_sifted, qber, epsilon
    2. Detailed: raw_key_length, min_entropy_rate, target_key_length

    Parameters
    ----------
    n_sifted : int
        Length of sifted key (bits).
    qber : float
        Quantum bit error rate.
    epsilon : float, optional
        Additional security margin. Default 0.
    raw_key_length : int, optional
        Alias for n_sifted.
    min_entropy_rate : float, optional
        Min-entropy per bit (alternative to QBER).
    target_key_length : int, optional
        Desired secure output length.
    security_parameter : float
        Statistical security ε.

    Returns
    -------
    int
        Maximum allowed leakage L_max.

    Notes
    -----
    Simple formula (König et al. style):
        L_max = n * (1 - h(QBER) - ε)
    
    Where h(p) is binary entropy: -p·log2(p) - (1-p)·log2(1-p)
    """
    # Use n_sifted or raw_key_length
    n = n_sifted if n_sifted is not None else (raw_key_length or 0)
    
    if min_entropy_rate is not None and target_key_length is not None:
        # Detailed interface
        total_min_entropy = n * min_entropy_rate
        finite_size_penalty = 2 * math.log2(1 / security_parameter)
        l_max = int(total_min_entropy - target_key_length - finite_size_penalty)
    else:
        # Simple interface: l_max = n * (1 - h(QBER) - ε)
        # Binary entropy
        if qber <= 0 or qber >= 1:
            h_qber = 0.0
        else:
            h_qber = -qber * math.log2(qber) - (1 - qber) * math.log2(1 - qber)
        
        l_max = int(n * (1.0 - h_qber - epsilon))
    
    return max(0, l_max)

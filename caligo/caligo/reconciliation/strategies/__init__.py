"""
Strategy Pattern for Reconciliation Protocols (Phase 2).

This module implements the Strategy Pattern to support multiple reconciliation
protocols (Baseline and Blind) with a unified interface.

Per Implementation Report v2 §5:
- ReconciliationStrategy: ABC defining the protocol interface
- ReconciliationContext: Immutable context dataclass
- BlockResult: Result dataclass with exact leakage accounting
- BaselineStrategy: Elkouss et al. (2010) rate-compatible protocol
- BlindStrategy: Martinez-Mateo et al. (2012) blind protocol

References
----------
[1] Elkouss et al. (2010), "Rate Compatible Protocol for Information Reconciliation"
[2] Martinez-Mateo et al. (2012), "Blind Reconciliation"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import numpy as np

from caligo.reconciliation.strategies.codec import LDPCCodec

__all__ = [
    "ReconciliationContext",
    "BlockResult",
    "ReconciliationStrategy",
    "DecoderResult",
    "BlindDecoderState",
    "LDPCCodec",
    "BaselineStrategy",
    "BlindStrategy",
]


# Lazy imports for concrete strategies to avoid circular imports
def __getattr__(name: str):
    if name == "BaselineStrategy":
        from caligo.reconciliation.strategies.baseline import BaselineStrategy
        return BaselineStrategy
    if name == "BlindStrategy":
        from caligo.reconciliation.strategies.blind import BlindStrategy
        return BlindStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class ReconciliationContext:
    """
    Immutable context passed to reconciliation strategies.
    
    Per Implementation Report v2 §5.1: This dataclass encapsulates all
    configuration parameters needed by a strategy, enabling stateless
    protocol implementations.
    
    Attributes
    ----------
    session_id : int
        Unique session identifier for logging/debugging.
    frame_size : int
        LDPC frame size (n = 4096).
    mother_rate : float
        Mother code rate (R_0 = 0.5).
    max_iterations : int
        BP decoder max iterations.
    hash_bits : int
        Verification hash length (typically 64 bits).
    f_crit : float
        Target efficiency threshold (f ∈ [1.05, 1.2]).
    qber_measured : Optional[float]
        Measured QBER from test bit exchange (Baseline only).
    qber_heuristic : Optional[float]
        Heuristic QBER from NSM parameters (Blind optional).
    modulation_delta : float
        Modulation budget δ = (p+s)/n (determines R_eff range).
    """
    session_id: int
    frame_size: int
    mother_rate: float
    max_iterations: int
    hash_bits: int
    f_crit: float
    qber_measured: Optional[float] = None
    qber_heuristic: Optional[float] = None
    modulation_delta: float = 0.1
    
    @property
    def qber_for_baseline(self) -> float:
        """
        QBER for Baseline rate selection (required).
        
        Raises
        ------
        ValueError
            If qber_measured is None.
        """
        if self.qber_measured is None:
            raise ValueError("Baseline protocol requires measured QBER from test bits")
        return self.qber_measured
    
    @property
    def qber_for_blind_gating(self) -> float:
        """
        QBER for Blind NSM-gating (optional optimization).
        
        Falls back to conservative default if no heuristic available.
        """
        return self.qber_heuristic if self.qber_heuristic is not None else 0.05


@dataclass
class BlockResult:
    """
    Result of single block reconciliation with exact leakage accounting.
    
    Per Theoretical Report v2 §1.2 and Implementation Report v2 §5.1:
    leak_EC = syndrome_leakage + hash_leakage + revealed_leakage
    
    This dataclass enforces the exact accounting required for NSM security proofs.
    
    Attributes
    ----------
    corrected_payload : np.ndarray
        Corrected bits (payload only, excluding padding).
    verified : bool
        True if hash verification passed.
    converged : bool
        True if BP decoder converged.
    iterations_used : int
        BP iterations until convergence (or max).
    syndrome_leakage : int
        Syndrome bits leaked = (1 - R_0) × n (constant per block).
    revealed_leakage : int
        Revealed bits leaked (Blind only, 0 for Baseline).
    hash_leakage : int
        Hash bits leaked.
    retry_count : int
        Number of blind iterations (1 for Baseline).
    effective_rate : float
        Effective code rate R_eff used for this block.
    """
    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    iterations_used: int
    syndrome_leakage: int
    revealed_leakage: int
    hash_leakage: int
    retry_count: int
    effective_rate: float
    estimated_qber: float = 0.5
    
    @property
    def total_leakage(self) -> int:
        """
        Total information leakage for this block.
        
        Per Theoretical Report v2 Eq. (leak_EC):
        leak = |Σ| + |Hash| + |Revealed|
        
        Returns
        -------
        int
            Total bits leaked to Bob.
        """
        return self.syndrome_leakage + self.revealed_leakage + self.hash_leakage


@dataclass
class DecoderResult:
    """
    Low-level decoder result from Numba kernels.
    
    Attributes
    ----------
    corrected_bits : np.ndarray
        Decoded bit array (full frame).
    converged : bool
        True if BP converged.
    iterations : int
        Number of BP iterations used.
    messages : np.ndarray
        Edge messages (for Hot-Start persistence).
    """
    corrected_bits: np.ndarray
    converged: bool
    iterations: int
    messages: np.ndarray


@dataclass
class BlindDecoderState:
    """
    Persistent decoder state across Blind iterations.
    
    Per Theoretical Report v2 §4.1 (Theorem 4.1), the syndrome is computed
    ONCE and reused. This state maintains LLR arrays and messages across
    network round-trips for the "Hot-Start" decoder kernel.
    
    Attributes
    ----------
    llr : np.ndarray
        Current LLR array (updated as bits are revealed).
    messages : np.ndarray
        BP edge messages for Hot-Start continuation.
    puncture_indices : np.ndarray
        Originally punctured positions (from hybrid pattern).
    shortened_indices : np.ndarray
        Currently shortened positions (grows each iteration).
    shortened_values : np.ndarray
        Values at shortened positions (grows each iteration).
    frozen_mask : np.ndarray
        Boolean mask for "freeze" optimization (revealed bits).
    iteration : int
        Current iteration number.
    syndrome : np.ndarray
        Fixed syndrome (computed once by Alice).
    """
    llr: np.ndarray
    messages: np.ndarray
    puncture_indices: np.ndarray
    shortened_indices: np.ndarray
    shortened_values: np.ndarray
    frozen_mask: np.ndarray
    iteration: int
    syndrome: np.ndarray


class ReconciliationStrategy(ABC):
    """
    Abstract base for reconciliation strategies.
    
    Implements the Strategy Pattern to support both Baseline [1] and
    Blind [2] reconciliation protocols with a unified interface.
    
    Strategies are stateless protocol implementations. Per-block state
    (e.g., LLRs across Blind iterations) is managed via yielded messages
    to the network layer.
    
    The generator-based interface supports SquidASM's cooperative scheduling.
    
    References
    ----------
    [1] Elkouss et al. (2010), "Rate Compatible Protocol for Information Reconciliation"
    [2] Martinez-Mateo et al. (2012), "Blind Reconciliation"
    """
    
    @property
    @abstractmethod
    def requires_qber_estimation(self) -> bool:
        """
        Whether this strategy requires QBER pre-estimation.
        
        Per Theoretical Report v2:
        - Baseline (§3): Requires explicit QBER sampling (t disclosed bits)
        - Blind (§4): Uses optional heuristic from compute_qber_erven
        
        Returns
        -------
        bool
            True for Baseline (needs rate selection), False for Blind.
        """
        pass
    
    @abstractmethod
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice-side block reconciliation.
        
        Parameters
        ----------
        payload : np.ndarray
            Alice's payload bits (uint8).
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
            
        Yields
        ------
        Dict[str, Any]
            Outgoing message to Bob (send to network).
            
        Receives
        --------
        Dict[str, Any]
            Response from Bob (received from network).
            
        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        pass
    
    @abstractmethod
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob-side block reconciliation.
        
        Parameters
        ----------
        payload : np.ndarray
            Bob's received payload bits (uint8).
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
            
        Yields
        ------
        Dict[str, Any]
            Outgoing message to Alice (ACK/NACK for Blind).
            
        Receives
        --------
        Dict[str, Any]
            Incoming message from Alice (syndrome, revealed bits).
            
        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        pass

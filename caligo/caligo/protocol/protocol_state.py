"""
Protocol State Machine for Reconciliation.

Provides explicit enum-based state tracking to prevent race conditions
in the generator-based protocol drivers.

Per Audit Report P1-1:
- Generator-based protocol has implicit state machine with asymmetric yield semantics
- Alice and Bob can reach inconsistent terminal states
- Explicit FSM provides clear state transitions and termination handling

References
----------
- Audit Report §3.1: "Asymmetric State Machine Between Alice and Bob"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class BlindPhase(Enum):
    """
    Explicit states for blind reconciliation protocol.
    
    State transitions:
    
    Alice: INIT -> SYNDROME_SENT -> (REVEAL_SENT)* -> DONE|FAILED
    Bob:   INIT -> AWAIT_SYNDROME -> (AWAIT_REVEAL)* -> DONE|FAILED
    """
    # Initial state
    INIT = auto()
    
    # Alice states
    SYNDROME_SENT = auto()      # Alice sent initial syndrome
    REVEAL_SENT = auto()        # Alice sent reveal iteration
    AWAIT_RESPONSE = auto()     # Alice waiting for Bob's response
    
    # Bob states
    AWAIT_SYNDROME = auto()     # Bob waiting for initial syndrome
    AWAIT_REVEAL = auto()       # Bob waiting for reveal iteration
    DECODING = auto()           # Bob performing BP decode
    
    # Terminal states
    VERIFIED = auto()           # Hash verified successfully
    FAILED = auto()             # Protocol failed (max iterations or error)
    DONE = auto()               # Protocol completed (sent/received done signal)


class BaselinePhase(Enum):
    """
    Explicit states for baseline reconciliation protocol.
    
    Baseline is single-round: syndrome -> decode -> verify
    """
    INIT = auto()
    SYNDROME_SENT = auto()
    AWAIT_RESPONSE = auto()
    AWAIT_SYNDROME = auto()
    DECODING = auto()
    VERIFIED = auto()
    FAILED = auto()
    DONE = auto()


@dataclass
class ProtocolState:
    """
    Tracks protocol state for race condition prevention.
    
    Used by protocol drivers to ensure:
    1. No sending to exhausted generator
    2. Proper termination synchronization
    3. Clear error reporting on state mismatch
    
    Attributes
    ----------
    phase : BlindPhase | BaselinePhase
        Current protocol phase.
    iteration : int
        Current blind iteration (1-indexed).
    generator_active : bool
        Whether the generator is still active (not returned).
    last_message_kind : Optional[str]
        Kind of last message sent/received.
    block_id : int
        Current block identifier.
    """
    phase: BlindPhase | BaselinePhase
    iteration: int = 1
    generator_active: bool = True
    last_message_kind: Optional[str] = None
    block_id: int = 0
    
    def transition_to(self, new_phase: BlindPhase | BaselinePhase) -> None:
        """
        Transition to new phase with validation.
        
        Parameters
        ----------
        new_phase : BlindPhase | BaselinePhase
            Target phase.
            
        Raises
        ------
        StateTransitionError
            If transition is invalid from current state.
        """
        # Terminal states cannot transition
        if self.phase in (BlindPhase.DONE, BlindPhase.FAILED, 
                          BlindPhase.VERIFIED, BaselinePhase.DONE,
                          BaselinePhase.FAILED, BaselinePhase.VERIFIED):
            if new_phase not in (BlindPhase.DONE, BaselinePhase.DONE):
                raise StateTransitionError(
                    f"Cannot transition from terminal state {self.phase} to {new_phase}"
                )
        self.phase = new_phase
    
    def mark_generator_exhausted(self) -> None:
        """Mark the generator as exhausted (returned)."""
        self.generator_active = False
    
    def is_terminal(self) -> bool:
        """Check if protocol is in a terminal state."""
        return self.phase in (
            BlindPhase.DONE, BlindPhase.FAILED, BlindPhase.VERIFIED,
            BaselinePhase.DONE, BaselinePhase.FAILED, BaselinePhase.VERIFIED,
        )
    
    def can_send_to_generator(self) -> bool:
        """Check if it's safe to send to the generator."""
        return self.generator_active and not self.is_terminal()


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class ProtocolSynchronizationError(Exception):
    """Raised when Alice/Bob states become desynchronized."""
    pass


def create_alice_state(block_id: int, is_blind: bool = True) -> ProtocolState:
    """
    Create initial protocol state for Alice.
    
    Parameters
    ----------
    block_id : int
        Block identifier.
    is_blind : bool
        True for blind protocol, False for baseline.
        
    Returns
    -------
    ProtocolState
        Initial Alice state.
    """
    phase = BlindPhase.INIT if is_blind else BaselinePhase.INIT
    return ProtocolState(phase=phase, block_id=block_id)


def create_bob_state(block_id: int, is_blind: bool = True) -> ProtocolState:
    """
    Create initial protocol state for Bob.
    
    Parameters
    ----------
    block_id : int
        Block identifier.
    is_blind : bool
        True for blind protocol, False for baseline.
        
    Returns
    -------
    ProtocolState
        Initial Bob state.
    """
    phase = BlindPhase.AWAIT_SYNDROME if is_blind else BaselinePhase.AWAIT_SYNDROME
    return ProtocolState(phase=phase, block_id=block_id)


def safe_generator_send(
    gen,
    msg: Optional[Dict[str, Any]],
    state: ProtocolState,
) -> tuple[Any, bool]:
    """
    Safely send a message to a generator with state tracking.
    
    Prevents race condition by checking generator_active before send.
    
    Parameters
    ----------
    gen : Generator
        Protocol generator.
    msg : Optional[Dict[str, Any]]
        Message to send. Use None for initial generator start.
    state : ProtocolState
        Current protocol state.
        
    Returns
    -------
    tuple[Any, bool]
        (response, generator_exhausted) - response from generator and
        whether it returned (StopIteration).
        
    Raises
    ------
    ProtocolSynchronizationError
        If attempting to send to exhausted generator.
    """
    if not state.can_send_to_generator():
        raise ProtocolSynchronizationError(
            f"Cannot send to generator in state {state.phase} "
            f"(generator_active={state.generator_active})"
        )
    
    try:
        response = gen.send(msg)
        if msg is not None:
            state.last_message_kind = msg.get("kind")
        return response, False
    except StopIteration as e:
        state.mark_generator_exhausted()
        return e.value, True

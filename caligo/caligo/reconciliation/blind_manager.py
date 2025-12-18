"""
Blind Reconciliation Manager (Phase 2 Extension).

Implements the Martinez-Mateo et al. blind reconciliation protocol
using iterative puncturing/shortening for QBER-free rate discovery.

The blind protocol eliminates QBER pre-estimation by starting with
maximum puncturing (highest rate) and progressively converting
punctured bits to shortened bits until decoding succeeds.

References
----------
- Martinez-Mateo et al. (2012): Blind Reconciliation Protocol
- Elkouss et al. (2009): Rate-compatible LDPC codes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from caligo.reconciliation import constants
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BlindConfig:
    """
    Configuration for blind reconciliation.

    Attributes
    ----------
    max_iterations : int
        Maximum retry iterations (t).
    modulation_fraction : float
        Fraction δ = d/n of frame for puncture/shorten modulation.
    frame_size : int
        LDPC codeword length.
    delta : float, optional
        Alias for modulation_fraction (test compatibility).
    """

    max_iterations: int = constants.BLIND_MAX_ITERATIONS
    modulation_fraction: float = constants.BLIND_MODULATION_FRACTION
    frame_size: int = constants.LDPC_FRAME_SIZE
    
    def __init__(
        self,
        max_iterations: int = None,
        modulation_fraction: float = None,
        frame_size: int = None,
        delta: float = None,  # Alias for test compatibility
    ):
        """Initialize with support for delta alias."""
        self.max_iterations = max_iterations if max_iterations is not None else constants.BLIND_MAX_ITERATIONS
        # delta takes precedence if provided
        if delta is not None:
            self.modulation_fraction = delta
        else:
            self.modulation_fraction = modulation_fraction if modulation_fraction is not None else constants.BLIND_MODULATION_FRACTION
        self.frame_size = frame_size if frame_size is not None else constants.LDPC_FRAME_SIZE

    @property
    def delta(self) -> float:
        """Alias for modulation_fraction (test compatibility)."""
        return self.modulation_fraction

    @property
    def modulation_bits(self) -> int:
        """Total modulation bits d = δ·n."""
        return int(self.modulation_fraction * self.frame_size)

    @property
    def delta_per_iteration(self) -> int:
        """Bits converted per iteration: Δ = d/t."""
        return max(1, self.modulation_bits // self.max_iterations)


# =============================================================================
# Iteration State
# =============================================================================


@dataclass
class BlindIterationState:
    """
    Track puncture/shorten state across iterations.

    Supports two interface styles:
    1. Martinez-Mateo style: n_punctured, n_shortened, shortened_values
    2. Simple style (tests): decoded_bits, syndrome_errors, converged

    Attributes
    ----------
    iteration : int
        Current iteration number (0 = initial).
    n_punctured : int
        Currently punctured positions (Martinez-Mateo).
    n_shortened : int
        Currently shortened positions (Martinez-Mateo).
    shortened_values : List[np.ndarray]
        Values revealed at each iteration (Martinez-Mateo).
    syndrome_leakage : int
        Fixed syndrome bits (constant across iterations).
    decoded_bits : np.ndarray, optional
        Current decoded bits (simple interface).
    syndrome_errors : int
        Number of unsatisfied syndrome checks (simple interface).
    converged : bool
        Whether decoding has converged (simple interface).
    """

    iteration: int = 0
    n_punctured: int = 0
    n_shortened: int = 0
    shortened_values: List[np.ndarray] = field(default_factory=list)
    syndrome_leakage: int = 0
    decoded_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    syndrome_errors: int = 0
    converged: bool = False

    @property
    def total_leakage(self) -> int:
        """Total bits leaked: syndrome + revealed shortened values."""
        shortened_bits = sum(len(v) for v in self.shortened_values)
        return self.syndrome_leakage + shortened_bits

    @property
    def total_modulation(self) -> int:
        """Total modulation bits: p + s."""
        return self.n_punctured + self.n_shortened


# =============================================================================
# Blind Reconciliation Manager
# =============================================================================


class BlindReconciliationManager:
    """
    Manage blind reconciliation iteration protocol.

    Protocol Flow:
    1. Initialize: all d modulation bits punctured (max rate)
    2. Alice sends syndrome (one-time)
    3. Bob attempts decode
       - Success → return corrected key
       - Failure → Alice reveals Δ shortened values
    4. Repeat step 3 until success or iteration limit

    Key Properties:
    - One-way information flow preserved (Alice → Bob)
    - No error-position feedback (Bob's failure is local)
    - Leakage monotonically increases with iterations

    Parameters
    ----------
    config : BlindConfig, optional
        Protocol configuration. Defaults to BlindConfig().
    """

    def __init__(self, config: BlindConfig = None) -> None:
        self.config = config or BlindConfig()

    def initialize(
        self,
        initial_bits_or_syndrome_length: np.ndarray | int = None,
    ) -> BlindIterationState:
        """
        Start new blind reconciliation.

        Supports two calling conventions:
        1. initialize(syndrome_length: int) - Martinez-Mateo style
        2. initialize(initial_bits: np.ndarray) - Simple test style

        Parameters
        ----------
        initial_bits_or_syndrome_length : np.ndarray or int
            Either the initial bits array (simple style) or
            the syndrome length (Martinez-Mateo style).

        Returns
        -------
        BlindIterationState
            Initial state with max puncturing.
        """
        if isinstance(initial_bits_or_syndrome_length, np.ndarray):
            # Simple test interface: initialize with bits
            # No puncturing/shortening for simple test interface
            initial_bits = initial_bits_or_syndrome_length
            return BlindIterationState(
                iteration=0,
                n_punctured=0,  # No puncturing for simple interface
                n_shortened=0,
                shortened_values=[],
                syndrome_leakage=0,
                decoded_bits=initial_bits.copy(),
                syndrome_errors=len(initial_bits),  # Assume all errors initially
                converged=False,
            )
        else:
            # Martinez-Mateo interface: initialize with syndrome length
            syndrome_length = initial_bits_or_syndrome_length or 0
            return BlindIterationState(
                iteration=0,
                n_punctured=self.config.modulation_bits,
                n_shortened=0,
                shortened_values=[],
                syndrome_leakage=syndrome_length,
                decoded_bits=np.array([], dtype=np.int8),
                syndrome_errors=0,
                converged=False,
            )

    def should_continue(self, state: BlindIterationState) -> bool:
        """
        Check if more iterations are allowed.

        Considers both convergence status and iteration limits.

        Parameters
        ----------
        state : BlindIterationState
            Current iteration state.

        Returns
        -------
        bool
            False if converged, iteration limit reached, or all bits shortened.
        """
        # Check if converged (simple interface)
        if state.converged:
            return False
        # Check iteration limit
        if state.iteration >= self.config.max_iterations:
            return False
        # Martinez-Mateo: check if all bits shortened
        if state.n_punctured <= 0 and state.total_modulation > 0:
            return False
        return True

    def advance_iteration(
        self,
        state: BlindIterationState,
        shortened_values: np.ndarray = None,
        new_decoded_bits: np.ndarray = None,
        new_syndrome_errors: int = None,
    ) -> BlindIterationState:
        """
        Advance to next iteration.

        Supports two interfaces:
        1. Martinez-Mateo: shortened_values provided
        2. Simple test: new_decoded_bits and new_syndrome_errors provided

        Parameters
        ----------
        state : BlindIterationState
            Current state.
        shortened_values : np.ndarray, optional
            Values of newly-shortened positions (Martinez-Mateo).
        new_decoded_bits : np.ndarray, optional
            Updated decoded bits (simple interface).
        new_syndrome_errors : int, optional
            New syndrome error count (simple interface).

        Returns
        -------
        BlindIterationState
            Updated state for next iteration.
        """
        if new_decoded_bits is not None:
            # Simple test interface
            converged = new_syndrome_errors == 0 if new_syndrome_errors is not None else False
            new_state = BlindIterationState(
                iteration=state.iteration + 1,
                n_punctured=state.n_punctured,
                n_shortened=state.n_shortened,
                shortened_values=state.shortened_values,
                syndrome_leakage=state.syndrome_leakage,
                decoded_bits=new_decoded_bits,
                syndrome_errors=new_syndrome_errors if new_syndrome_errors is not None else state.syndrome_errors,
                converged=converged,
            )
        else:
            # Martinez-Mateo interface
            delta = min(self.config.delta_per_iteration, state.n_punctured)
            new_state = BlindIterationState(
                iteration=state.iteration + 1,
                n_punctured=state.n_punctured - delta,
                n_shortened=state.n_shortened + delta,
                shortened_values=state.shortened_values + ([shortened_values] if shortened_values is not None else []),
                syndrome_leakage=state.syndrome_leakage,
                decoded_bits=state.decoded_bits,
                syndrome_errors=state.syndrome_errors,
                converged=state.converged,
            )

        logger.debug(
            "Blind iteration %d→%d: converged=%s, errors=%d",
            state.iteration, new_state.iteration,
            new_state.converged, new_state.syndrome_errors,
        )

        return new_state

    def build_llr_for_state(
        self,
        state: BlindIterationState,
        bob_bits: np.ndarray = None,
        qber: float = 0.05,
        received_bits: np.ndarray = None,  # Alias for test compatibility
    ) -> np.ndarray:
        """
        Construct LLRs for current blind state.

        LLR structure: [payload | shortened | punctured]
        - Payload: channel LLR from QBER
        - Shortened: high confidence (values known)
        - Punctured: zero confidence (unknown)

        Parameters
        ----------
        state : BlindIterationState
            Current iteration state.
        bob_bits : np.ndarray, optional
            Bob's payload bits (uint8). Use received_bits as alias.
        qber : float
            Estimated QBER for channel LLR.
        received_bits : np.ndarray, optional
            Alias for bob_bits (test compatibility).

        Returns
        -------
        np.ndarray
            LLR array for full frame.
        """
        # Use received_bits if bob_bits not provided (test compatibility)
        bits = bob_bits if bob_bits is not None else received_bits
        if bits is None:
            raise ValueError("Either bob_bits or received_bits must be provided")
            
        payload_len = len(bits)
        n = payload_len + state.n_shortened + state.n_punctured

        # Clamp QBER for numerical stability
        qber_clamped = np.clip(qber, 1e-6, 0.5 - 1e-6)
        channel_llr = np.log((1 - qber_clamped) / qber_clamped)

        llr = np.zeros(n, dtype=np.float64)

        # Payload: channel LLR with sign from received bit
        llr[:payload_len] = channel_llr * (1 - 2 * bits.astype(np.float64))

        # Shortened: high confidence (assuming zeros by convention)
        if state.n_shortened > 0:
            llr[payload_len:payload_len + state.n_shortened] = constants.LDPC_LLR_SHORTENED

        # Punctured: zero confidence (remain at 0.0)
        # llr[payload_len + state.n_shortened:] already 0.0

        return llr

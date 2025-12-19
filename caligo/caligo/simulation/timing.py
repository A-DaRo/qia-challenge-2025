"""
Timing barrier for NSM wait time Δt enforcement.

This module implements the TimingBarrier — a simulation-aware mechanism
that enforces the wait time Δt as a causal barrier in the discrete-event
timeline, preventing basis revelation before the required time has elapsed.

References
----------
- Erven et al. (2014): "Both parties now wait a time, Δt..."
- König et al. (2012): Markovian storage noise assumption
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Generator, Optional

from caligo.types.exceptions import TimingViolationError
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# State Machine
# =============================================================================


class TimingBarrierState(Enum):
    """
    State machine states for TimingBarrier.

    State Transitions:
        IDLE ──[mark_quantum_complete()]──► WAITING
        WAITING ──[wait_delta_t()]──► READY
        READY ──[reset()]──► IDLE
    """

    IDLE = auto()  # Initial state, waiting for quantum phase
    WAITING = auto()  # Quantum complete, waiting for Δt
    READY = auto()  # Δt elapsed, can proceed


# =============================================================================
# Timing Utilities
# =============================================================================


def _get_sim_time() -> float:
    """
    Get current simulation time from NetSquid.

    Returns
    -------
    float
        Current simulation time in nanoseconds.

    Notes
    -----
    Returns 0.0 if NetSquid is not available (for unit testing).
    """
    try:
        import netsquid as ns

        return ns.sim_time()
    except ImportError:
        return 0.0


def _sim_run(duration: float) -> None:
    """
    Advance simulation time by specified duration.

    Parameters
    ----------
    duration : float
        Time to advance in nanoseconds.

    Notes
    -----
    No-op if NetSquid is not available (for unit testing).
    """
    try:
        import netsquid as ns

        ns.sim_run(duration=duration)
    except ImportError:
        pass


# =============================================================================
# TimingBarrier Class
# =============================================================================


class TimingBarrier:
    """
    Enforces NSM timing constraint Δt in discrete-event simulation.

    The TimingBarrier ensures that Alice cannot reveal her basis choices
    until time Δt has elapsed since the quantum phase completed. This
    allows any quantum information stored by a dishonest Bob to decohere
    according to the NSM assumption.

    Parameters
    ----------
    delta_t_ns : float
        Required wait time in nanoseconds.
    strict_mode : bool
        If True, raises TimingViolationError on premature access.
        If False, logs warning but allows continuation. Default: True.

    Attributes
    ----------
    state : TimingBarrierState
        Current state machine state.
    quantum_complete_time : Optional[float]
        Simulation time when quantum phase ended (ns).
    delta_t_ns : float
        Configured wait time.
    timing_compliant : bool
        True if protocol respected timing constraints.

    Examples
    --------
    In a SquidASM Program:

    >>> barrier = TimingBarrier(delta_t_ns=1_000_000)  # 1 ms
    >>>
    >>> # After quantum measurements complete:
    >>> barrier.mark_quantum_complete()
    >>>
    >>> # Wait for Δt (yields control to simulator)
    >>> yield from barrier.wait_delta_t()
    >>>
    >>> # Now safe to reveal basis
    >>> if barrier.can_reveal_basis():
    >>>     socket.send(bases)

    References
    ----------
    - phase_I_analysis.md Section 7.2: Timing barrier requirements
    - König et al. (2012): Markovian storage noise assumption
    """

    def __init__(self, delta_t_ns: float, strict_mode: bool = True) -> None:
        """
        Initialize the timing barrier.

        Parameters
        ----------
        delta_t_ns : float
            Required wait time in nanoseconds.
        strict_mode : bool
            If True, raises on violations. Default: True.

        Raises
        ------
        ValueError
            If delta_t_ns <= 0.
        """
        if delta_t_ns <= 0:
            raise ValueError(f"delta_t_ns={delta_t_ns} must be > 0")

        self._delta_t_ns = delta_t_ns
        self._strict_mode = strict_mode
        self._state = TimingBarrierState.IDLE
        self._quantum_complete_time: Optional[float] = None
        self._timing_compliant = True
        self._timer_entity: Optional[Any] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> TimingBarrierState:
        """Get current state machine state."""
        return self._state

    @property
    def delta_t_ns(self) -> float:
        """Get configured wait time in nanoseconds."""
        return self._delta_t_ns

    @property
    def quantum_complete_time(self) -> Optional[float]:
        """Get time when quantum phase completed (ns), or None if not yet."""
        return self._quantum_complete_time

    @property
    def timing_compliant(self) -> bool:
        """Check if protocol has remained timing-compliant."""
        return self._timing_compliant

    @property
    def strict_mode(self) -> bool:
        """Check if strict mode is enabled."""
        return self._strict_mode

    # =========================================================================
    # State Transitions
    # =========================================================================

    def mark_quantum_complete(self) -> None:
        """
        Mark the end of the quantum phase.

        Records the current simulation time as the start of the Δt
        wait period. Must be called after all quantum measurements
        are complete but before any basis information is exchanged.

        Raises
        ------
        TimingViolationError
            If called when not in IDLE state (in strict mode).

        Side Effects
        ------------
        - Sets quantum_complete_time to current simulation time
        - Transitions state from IDLE to WAITING
        """
        if self._state != TimingBarrierState.IDLE:
            msg = (
                f"mark_quantum_complete() called in state {self._state.name}. "
                f"Expected IDLE state."
            )
            if self._strict_mode:
                raise TimingViolationError(msg)
            logger.warning(msg)

        self._quantum_complete_time = _get_sim_time()
        self._state = TimingBarrierState.WAITING
        logger.debug(
            f"Quantum phase complete at t={self._quantum_complete_time} ns. "
            f"Waiting {self._delta_t_ns} ns before basis revelation."
        )

    def wait_delta_t(self) -> Generator[None, None, None]:
        """
        Generator that waits until Δt has elapsed.

        Yields control back to the SquidASM simulation engine until
        the required wait time has passed. Compatible with the
        generator-based programming model.

        Yields
        ------
        None
            Control is yielded until Δt elapses.

        Raises
        ------
        TimingViolationError
            If called when not in WAITING state (in strict mode).

        Notes
        -----
        In SquidASM, use: `yield from barrier.wait_delta_t()`

        This does NOT block the simulation — other events can occur.
        The barrier simply tracks when the protocol is allowed to
        proceed with basis revelation.
        """
        if self._state != TimingBarrierState.WAITING:
            msg = (
                f"wait_delta_t() called in state {self._state.name}. "
                f"Expected WAITING state. Call mark_quantum_complete() first."
            )
            if self._strict_mode:
                raise TimingViolationError(msg)
            logger.warning(msg)

        # Calculate remaining time
        elapsed = self.get_elapsed_ns()
        remaining = self._delta_t_ns - elapsed

        if remaining > 0:
            # In NetSquid/SquidASM we must yield an EventExpression to allow the
            # discrete-event simulator to advance time.
            try:
                from pydynaa import Entity, EventExpression, EventType

                if self._timer_entity is None:

                    class _BarrierTimer(Entity):
                        def __init__(self) -> None:
                            self._event = EventType(
                                "CALIGO_BARRIER_TIMER",
                                "TimingBarrier scheduled wait event",
                            )

                        def wait(self, duration_ns: float) -> EventExpression:
                            self._schedule_after(duration_ns, self._event)
                            return EventExpression(
                                source=self, event_type=self._event
                            )

                    self._timer_entity = _BarrierTimer()

                yield self._timer_entity.wait(remaining)
            except ImportError:
                # Unit-test / non-simulation fallback.
                _sim_run(remaining)
                # Yield at least once to preserve generator semantics.
                yield

        self._state = TimingBarrierState.READY
        logger.debug(
            f"Wait complete. Elapsed: {self.get_elapsed_ns()} ns. "
            f"Basis revelation permitted."
        )

        return

    def can_reveal_basis(self) -> bool:
        """
        Check if basis revelation is permitted.

        Returns
        -------
        bool
            True if Δt has elapsed and state is READY.

        Side Effects
        ------------
        In strict_mode, raises TimingViolationError if called in
        WAITING state before Δt has elapsed.
        """
        if self._state == TimingBarrierState.READY:
            return True

        if self._state == TimingBarrierState.WAITING:
            elapsed = self.get_elapsed_ns()
            if elapsed >= self._delta_t_ns:
                # Automatically transition to READY
                self._state = TimingBarrierState.READY
                return True

            # Not enough time has passed
            self._timing_compliant = False
            msg = (
                f"Basis revelation attempted after only {elapsed:.0f} ns. "
                f"Required: {self._delta_t_ns:.0f} ns."
            )
            if self._strict_mode:
                raise TimingViolationError(msg)
            logger.warning(msg)
            return False

        # IDLE state
        return False

    def get_elapsed_ns(self) -> float:
        """
        Get time elapsed since quantum phase completed.

        Returns
        -------
        float
            Elapsed time in nanoseconds, or 0.0 if not started.
        """
        if self._quantum_complete_time is None:
            return 0.0
        return _get_sim_time() - self._quantum_complete_time

    def assert_timing_compliant(self) -> None:
        """
        Assert that timing constraints are satisfied.

        Raises
        ------
        TimingViolationError
            If Δt has not elapsed since quantum phase completion.
            Error includes diagnostic information about elapsed time
            and required wait time.

        Notes
        -----
        Call this method before revealing basis information.
        Unlike can_reveal_basis(), this method ALWAYS raises if
        timing is violated (no strict_mode flag needed).

        Examples
        --------
        >>> barrier.mark_quantum_complete()
        >>> yield from barrier.wait_delta_t()
        >>> barrier.assert_timing_compliant()  # Raises if Δt not satisfied
        >>> socket.send(bases)  # Safe to reveal
        """
        if self._state == TimingBarrierState.IDLE:
            raise TimingViolationError(
                "Cannot assert timing compliance: quantum phase not yet marked complete."
            )

        elapsed = self.get_elapsed_ns()
        if elapsed < self._delta_t_ns:
            self._timing_compliant = False
            raise TimingViolationError(
                f"Timing constraint violated: {elapsed:.0f} ns elapsed, "
                f"{self._delta_t_ns:.0f} ns required. "
                f"Deficit: {self._delta_t_ns - elapsed:.0f} ns."
            )

    def reset(self) -> None:
        """
        Reset barrier for next protocol run.

        Transitions state back to IDLE and clears timing records.
        Use this when starting a new protocol instance.
        """
        self._state = TimingBarrierState.IDLE
        self._quantum_complete_time = None
        self._timing_compliant = True
        logger.debug("TimingBarrier reset to IDLE state.")

    # =========================================================================
    # Diagnostic Methods
    # =========================================================================

    def get_diagnostic_info(self) -> dict:
        """
        Get diagnostic information about barrier state.

        Returns
        -------
        dict
            Dictionary with state, timing, and compliance info.
        """
        return {
            "state": self._state.name,
            "delta_t_ns": self._delta_t_ns,
            "quantum_complete_time": self._quantum_complete_time,
            "elapsed_ns": self.get_elapsed_ns(),
            "timing_compliant": self._timing_compliant,
            "strict_mode": self._strict_mode,
            "can_reveal": self._state == TimingBarrierState.READY,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TimingBarrier(delta_t_ns={self._delta_t_ns}, "
            f"state={self._state.name}, "
            f"elapsed={self.get_elapsed_ns():.0f}ns)"
        )

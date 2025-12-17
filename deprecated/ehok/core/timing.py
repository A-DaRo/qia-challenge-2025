"""
Timing Enforcer for NSM Causal Barrier.

This module implements the mandatory wait window Δt as specified in the
Noisy Storage Model (NSM). The timing barrier ensures that an adversary's
quantum storage decoheres before basis information is revealed.

Security Rationale
------------------
The NSM security model requires strict temporal ordering:
1. Bob receives and stores qubits
2. Bob acknowledges receipt (commitment)
3. **Timer Δt elapses** (adversary's storage decoheres)
4. Alice reveals basis information

If basis revelation occurs before Δt has elapsed since commitment, the
adversary may retain coherent quantum information, breaking security.

Implementation Notes
--------------------
- TimingEnforcer operates on simulation time (integer nanoseconds)
- The enforcer itself is deterministic; time is provided externally
- Integration layer provides time via NetSquid's ns.sim_time()
- Timing events are logged for audit and verification

References
----------
- König et al. (2012): "Whenever the protocol requires the adversary to wait
  for a time Δt, he has to measure/discard all his quantum information except
  what he can encode [...] This information then undergoes noise described by F."
- Erven et al. (2014): Uses Δt = 1 second before basis reveal.
- sprint_1_specification.md Section 3.1 (TASK-TIMING-001)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default timing values (from literature)
DEFAULT_DELTA_T_NS = 1_000_000_000  # 1 second in nanoseconds (Erven et al. 2014)


class TimingEvent(Enum):
    """
    Enumeration of timing events for structured logging.

    These events form the audit trail for NSM timing barrier verification.
    """

    TIMING_COMMIT_ACK_RECEIVED = auto()
    TIMING_BASIS_REVEAL_BLOCKED = auto()
    TIMING_BASIS_REVEAL_ALLOWED = auto()
    TIMING_BARRIER_SATISFIED = auto()


# =============================================================================
# Timing Configuration
# =============================================================================


@dataclass(frozen=True)
class TimingConfig:
    """
    Configuration for the NSM timing barrier.

    Attributes
    ----------
    delta_t_ns : int
        Mandatory wait window in nanoseconds (simulation time).
        Must be > 0. Default: 1 second (from Erven et al. 2014).

    Raises
    ------
    ValueError
        If delta_t_ns <= 0.

    References
    ----------
    - Erven et al. (2014): "Both parties now wait a time, Δt, long enough for
      any stored quantum information of a dishonest party to decohere."

    Examples
    --------
    >>> config = TimingConfig(delta_t_ns=1_000_000_000)  # 1 second
    >>> config.delta_t_ns
    1000000000
    """

    delta_t_ns: int = DEFAULT_DELTA_T_NS

    def __post_init__(self) -> None:
        """Validate timing configuration."""
        if self.delta_t_ns <= 0:
            raise ValueError(
                f"delta_t_ns must be positive, got {self.delta_t_ns}"
            )


# =============================================================================
# Timing State
# =============================================================================


class TimingState(Enum):
    """
    State machine for timing barrier progression.

    States
    ------
    UNINITIALIZED
        No timing events recorded yet.
    COMMIT_RECEIVED
        Commitment acknowledgment received; timer started.
    BARRIER_SATISFIED
        Δt has elapsed; basis reveal is permitted.
    BASIS_REVEALED
        Basis information has been revealed.
    """

    UNINITIALIZED = auto()
    COMMIT_RECEIVED = auto()
    BARRIER_SATISFIED = auto()
    BASIS_REVEALED = auto()


# =============================================================================
# Timing Exceptions
# =============================================================================


class TimingViolationError(Exception):
    """
    Raised when the NSM timing barrier is violated.

    This is a security-critical exception indicating that basis information
    was revealed (or attempted) before the mandatory wait period elapsed.
    """

    pass


class TimingStateError(Exception):
    """
    Raised when timing operations are invoked in invalid state.

    For example, attempting to mark basis reveal before commit is received.
    """

    pass


# =============================================================================
# Timing Enforcer
# =============================================================================


class TimingEnforcer:
    """
    Enforces the NSM mandatory wait window Δt.

    This class implements the timing barrier as a state machine that tracks
    protocol events and validates temporal ordering. It operates solely on
    simulation time provided by the integration layer.

    The enforcer guarantees the invariant:
        t_basis - t_commit_ack >= Δt

    Attributes
    ----------
    config : TimingConfig
        Timing parameters including Δt.

    Methods
    -------
    mark_commit_received(sim_time_ns)
        Record commitment acknowledgment timestamp.
    mark_basis_reveal_attempt(sim_time_ns)
        Validate and record basis reveal attempt.
    is_basis_reveal_allowed(sim_time_ns)
        Check if basis reveal is permitted at given time.
    required_release_time_ns()
        Get the earliest time when basis reveal is allowed.

    References
    ----------
    - sprint_1_specification.md Section 3.1 (TASK-TIMING-001)
    - König et al. (2012) Section I-C (timing semantics)

    Examples
    --------
    >>> config = TimingConfig(delta_t_ns=1_000_000_000)  # 1 second
    >>> enforcer = TimingEnforcer(config)
    >>> enforcer.mark_commit_received(sim_time_ns=100_000_000)  # 0.1s
    >>> enforcer.is_basis_reveal_allowed(sim_time_ns=500_000_000)  # 0.5s
    False
    >>> enforcer.is_basis_reveal_allowed(sim_time_ns=1_200_000_000)  # 1.2s
    True
    """

    def __init__(self, config: TimingConfig) -> None:
        """
        Initialize the timing enforcer.

        Parameters
        ----------
        config : TimingConfig
            Timing configuration with Δt specification.
        """
        self._config = config
        self._state = TimingState.UNINITIALIZED
        self._commit_ack_time_ns: Optional[int] = None
        self._basis_reveal_time_ns: Optional[int] = None

        logger.debug(
            "TimingEnforcer initialized with delta_t_ns=%d",
            config.delta_t_ns,
        )

    @property
    def config(self) -> TimingConfig:
        """Get timing configuration."""
        return self._config

    @property
    def state(self) -> TimingState:
        """Get current timing state."""
        return self._state

    @property
    def commit_ack_time_ns(self) -> Optional[int]:
        """Get commitment acknowledgment timestamp, if recorded."""
        return self._commit_ack_time_ns

    def mark_commit_received(self, *, sim_time_ns: int) -> None:
        """
        Record that commitment acknowledgment has been received.

        This marks the start of the Δt timing window. The basis reveal
        will be blocked until sim_time >= commit_ack_time + Δt.

        Parameters
        ----------
        sim_time_ns : int
            Current simulation time in nanoseconds when commit was received.

        Raises
        ------
        ValueError
            If sim_time_ns is negative.
        TimingStateError
            If commit has already been marked.

        Notes
        -----
        This method should be called exactly once per protocol session,
        when Bob's measurement commitment is acknowledged.
        """
        if sim_time_ns < 0:
            raise ValueError(f"sim_time_ns must be non-negative, got {sim_time_ns}")

        if self._state != TimingState.UNINITIALIZED:
            raise TimingStateError(
                f"Cannot mark commit in state {self._state.name}; "
                "commit has already been received"
            )

        self._commit_ack_time_ns = sim_time_ns
        self._state = TimingState.COMMIT_RECEIVED

        logger.info(
            "TIMING_COMMIT_ACK_RECEIVED at t_commit_ack_ns=%d",
            sim_time_ns,
        )

    def mark_basis_reveal_attempt(self, *, sim_time_ns: int) -> None:
        """
        Mark an attempt to reveal basis information.

        This method validates that the timing barrier has been satisfied
        before allowing basis reveal to proceed.

        Parameters
        ----------
        sim_time_ns : int
            Current simulation time in nanoseconds.

        Raises
        ------
        ValueError
            If sim_time_ns is negative.
        TimingStateError
            If commit has not been received yet.
        TimingViolationError
            If basis reveal is attempted before Δt has elapsed.

        Notes
        -----
        If the timing barrier is not satisfied, this method logs a
        TIMING_BASIS_REVEAL_BLOCKED event and raises TimingViolationError.
        """
        if sim_time_ns < 0:
            raise ValueError(f"sim_time_ns must be non-negative, got {sim_time_ns}")

        if self._commit_ack_time_ns is None:
            raise TimingStateError(
                "Cannot attempt basis reveal before commit is received; "
                "call mark_commit_received first"
            )

        if not self.is_basis_reveal_allowed(sim_time_ns=sim_time_ns):
            required_time = self.required_release_time_ns()
            remaining_ns = required_time - sim_time_ns

            logger.warning(
                "TIMING_BASIS_REVEAL_BLOCKED at t=%d; "
                "required_release=%d, remaining=%d ns",
                sim_time_ns,
                required_time,
                remaining_ns,
            )

            raise TimingViolationError(
                f"Basis reveal blocked: current time {sim_time_ns} ns < "
                f"required release time {required_time} ns. "
                f"Remaining: {remaining_ns} ns ({remaining_ns / 1e9:.6f} s)"
            )

        # Timing barrier satisfied
        self._basis_reveal_time_ns = sim_time_ns

        if self._state == TimingState.COMMIT_RECEIVED:
            self._state = TimingState.BARRIER_SATISFIED

        self._state = TimingState.BASIS_REVEALED

        elapsed_ns = sim_time_ns - self._commit_ack_time_ns
        logger.info(
            "TIMING_BASIS_REVEAL_ALLOWED at t=%d; elapsed=%d ns (%.6f s)",
            sim_time_ns,
            elapsed_ns,
            elapsed_ns / 1e9,
        )

    def is_basis_reveal_allowed(self, *, sim_time_ns: int) -> bool:
        """
        Check if basis reveal is permitted at the given simulation time.

        Parameters
        ----------
        sim_time_ns : int
            Current simulation time in nanoseconds.

        Returns
        -------
        bool
            True if t_basis - t_commit_ack >= Δt, False otherwise.

        Notes
        -----
        Returns False if commit has not been received yet.
        This is a pure query method with no side effects.
        """
        if self._commit_ack_time_ns is None:
            return False

        elapsed = sim_time_ns - self._commit_ack_time_ns
        return elapsed >= self._config.delta_t_ns

    def required_release_time_ns(self) -> int:
        """
        Get the earliest simulation time when basis reveal is allowed.

        Returns
        -------
        int
            t_commit_ack + Δt in nanoseconds.

        Raises
        ------
        TimingStateError
            If commit has not been received yet.

        Notes
        -----
        This provides the target time for simulation scheduling when
        implementing the wait in SquidASM protocol coroutines.
        """
        if self._commit_ack_time_ns is None:
            raise TimingStateError(
                "Cannot compute release time before commit is received"
            )

        return self._commit_ack_time_ns + self._config.delta_t_ns

    def remaining_wait_ns(self, *, sim_time_ns: int) -> int:
        """
        Compute remaining wait time until barrier is satisfied.

        Parameters
        ----------
        sim_time_ns : int
            Current simulation time in nanoseconds.

        Returns
        -------
        int
            Remaining nanoseconds until basis reveal is allowed.
            Returns 0 if barrier is already satisfied.

        Raises
        ------
        TimingStateError
            If commit has not been received yet.
        """
        release_time = self.required_release_time_ns()
        remaining = release_time - sim_time_ns
        return max(0, remaining)

    def reset(self) -> None:
        """
        Reset the enforcer to initial state.

        This is primarily useful for testing. In production, a new
        enforcer should be created for each protocol session.
        """
        self._state = TimingState.UNINITIALIZED
        self._commit_ack_time_ns = None
        self._basis_reveal_time_ns = None
        logger.debug("TimingEnforcer reset to UNINITIALIZED state")

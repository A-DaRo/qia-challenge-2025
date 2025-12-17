"""
Exception hierarchy and protocol enumerations for Caligo.

This module defines all exceptions used throughout the Caligo codebase,
organized in a hierarchy reflecting the domain structure, along with
protocol phase and abort reason enumerations.
"""

from __future__ import annotations

from enum import Enum


# =============================================================================
# Base Exception
# =============================================================================


class CaligoError(Exception):
    """
    Base exception for all Caligo-related errors.

    All exceptions in the Caligo package inherit from this class,
    enabling catch-all handling at the protocol orchestration level.
    """

    pass


# =============================================================================
# Simulation Errors
# =============================================================================


class SimulationError(CaligoError):
    """Base exception for simulation-related errors."""

    pass


class TimingViolationError(SimulationError):
    """
    Raised when NSM timing constraints (Δt) are violated.

    This is a security-critical error indicating that basis revelation
    occurred before the required wait time elapsed.
    """

    pass


class NetworkConfigError(SimulationError):
    """Raised when SquidASM network setup fails."""

    pass


class EPRGenerationError(SimulationError):
    """Raised when EPR pair generation fails."""

    pass


# =============================================================================
# Security Errors
# =============================================================================


class SecurityError(CaligoError):
    """Base exception for security-related errors."""

    pass


class QBERThresholdExceeded(SecurityError):
    """
    Raised when QBER exceeds security threshold.

    The hard limit is 22% (security impossible beyond).
    The conservative limit is 11% (recommended operating point).

    References
    ----------
    - König et al. (2012): 22% hard limit
    - Schaffner et al. (2009), Corollary 7: 11% optimal for protocol
    """

    pass


class NSMViolationError(SecurityError):
    """Raised when Noisy Storage Model assumptions are violated."""

    pass


class FeasibilityError(SecurityError):
    """
    Raised when pre-flight security check fails.

    Indicates that secure OT is not feasible with current parameters.
    """

    pass


class EntropyDepletedError(SecurityError):
    """
    Raised when no extractable entropy remains.

    After accounting for error correction leakage and finite-size
    penalties, min-entropy h_min ≤ 0.
    """

    pass


class CommitmentVerificationError(SecurityError):
    """Raised when commitment hash verification fails (potential attack)."""

    pass


# =============================================================================
# Protocol Errors
# =============================================================================


class ProtocolError(CaligoError):
    """Base exception for protocol logic errors."""

    pass


class PhaseOrderViolation(ProtocolError):
    """Raised when phases are executed out of order."""

    pass


class ContractViolation(ProtocolError):
    """
    Raised when a phase contract invariant is violated.

    This typically indicates a programming error where data passed
    between phases does not satisfy the required constraints.
    """

    pass


class ReconciliationError(ProtocolError):
    """Raised when LDPC decoding fails after maximum iterations."""

    pass


# =============================================================================
# Connection Errors
# =============================================================================


class ConnectionError(CaligoError):
    """Base exception for classical communication errors."""

    pass


class OrderingViolationError(ConnectionError):
    """Raised when message ordering protocol is violated."""

    pass


class AckTimeoutError(ConnectionError):
    """Raised when acknowledgment is not received in time."""

    pass


class SessionMismatchError(ConnectionError):
    """Raised when session ID does not match expected value."""

    pass


class OutOfOrderError(ConnectionError):
    """Raised when sequence number is out of order."""

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(CaligoError):
    """Base exception for configuration-related errors."""

    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a configuration parameter is out of valid range."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when a required configuration is not provided."""

    pass


# =============================================================================
# Protocol Phase Enumeration
# =============================================================================


class ProtocolPhase(Enum):
    """
    Enumeration of E-HOK protocol phases.

    Used for state machine tracking and transcript annotation.
    Phase transitions must follow the order:
    INIT → QUANTUM → SIFTING → RECONCILIATION → AMPLIFICATION → COMPLETED

    Any phase may transition to ABORTED.
    """

    INIT = "init"
    QUANTUM = "quantum"  # Phase I
    SIFTING = "sifting"  # Phase II
    RECONCILIATION = "reconciliation"  # Phase III
    AMPLIFICATION = "amplification"  # Phase IV
    COMPLETED = "completed"
    ABORTED = "aborted"


# =============================================================================
# Abort Reason Enumeration
# =============================================================================


class AbortReason(Enum):
    """
    Enumeration of protocol abort reasons with diagnostic codes.

    Code taxonomy:
    - ABORT-I-*: Phase I (Quantum) abort conditions
    - ABORT-II-*: Phase II (Sifting) abort conditions
    - ABORT-III-*: Phase III (Reconciliation) abort conditions
    - ABORT-IV-*: Phase IV (Amplification) abort conditions
    """

    # Phase I abort conditions
    FEASIBILITY_HARD_LIMIT = "ABORT-I-FEAS-001"  # Q_total > 22%
    TIMING_VIOLATION = "ABORT-I-TIMING-001"  # Basis revealed before Δt

    # Phase II abort conditions
    DETECTION_ANOMALY = "ABORT-II-DET-001"  # Chernoff bound violated
    QBER_HARD_LIMIT = "ABORT-II-QBER-001"  # Adjusted QBER > 22%
    MISSING_ROUNDS_INVALID = "ABORT-II-MISS-001"  # Invalid loss report

    # Phase III abort conditions
    LEAKAGE_CAP_EXCEEDED = "ABORT-III-LEAK-001"  # |Σ| > L_max
    RECONCILIATION_FAILED = "ABORT-III-REC-001"  # Decoder failure
    VERIFICATION_FAILED = "ABORT-III-VER-001"  # Hash mismatch

    # Phase IV abort conditions
    ENTROPY_DEPLETED = "ABORT-IV-ENT-001"  # h_min ≤ 0
    KEY_LENGTH_ZERO = "ABORT-IV-LEN-001"  # ℓ = 0

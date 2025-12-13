"""
Ordered Protocol Socket for NSM Commit-then-Reveal Semantics.

This module implements ordered message delivery with acknowledgment
enforcement, ensuring that Bob's detection report is committed before
Alice reveals basis information (TASK-ORDERED-MSG-001).

Security Rationale
------------------
The Commit-then-Reveal ordering is fundamental to NSM security. If Bob
receives basis information before his detection report is acknowledged,
he can selectively claim "missing" only rounds where his quantum storage
failed, effectively post-selecting a lower-noise sub-key.

Design
------
The OrderedProtocolSocket wraps a classical socket and provides:
- `send_with_ack`: Blocks until matching ACK is received
- `recv_and_ack`: Receives message and automatically sends ACK
- Monotonic sequence numbering per session
- Timeout-based abort on missing ACKs

State Machine
-------------
- IDLE: Ready for send or receive
- SENT_WAIT_ACK: Message sent, awaiting ACK
- RECV_PROCESSING: Processing received message
- VIOLATION: Unrecoverable ordering failure (terminal state)

References
----------
- Phase II analysis Section 3.1: Ordered Message Protocol Flow
- sprint_2_specification.md Section 2.1-2.4: OrderedProtocolSocket
- König et al. (2012): Commit-then-Reveal timing semantics
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ACK_TIMEOUT_NS = 5_000_000_000  # 5 seconds in nanoseconds


# =============================================================================
# Message Types
# =============================================================================


class MessageType(Enum):
    """
    Enumeration of ordered message types.

    These discriminators identify message semantics within the envelope.
    """

    DETECTION_REPORT = "DETECTION_REPORT"
    BASIS_REVEAL = "BASIS_REVEAL"
    INDEX_LISTS = "INDEX_LISTS"
    TEST_CHALLENGE = "TEST_CHALLENGE"
    TEST_RESPONSE = "TEST_RESPONSE"
    ACK = "ACK"


# =============================================================================
# Socket States
# =============================================================================


class SocketState(Enum):
    """
    State machine states for OrderedProtocolSocket.

    Attributes
    ----------
    IDLE : auto
        Ready to send or receive.
    SENT_WAIT_ACK : auto
        Message sent, waiting for acknowledgment.
    RECV_PROCESSING : auto
        Processing a received message.
    VIOLATION : auto
        Unrecoverable ordering failure (terminal state).
    """

    IDLE = auto()
    SENT_WAIT_ACK = auto()
    RECV_PROCESSING = auto()
    VIOLATION = auto()


# =============================================================================
# Exceptions
# =============================================================================


class OrderingViolationError(Exception):
    """
    Raised when message ordering constraints are violated.

    This indicates a security-critical failure where the commit-then-reveal
    semantics cannot be guaranteed.
    """

    pass


class AckTimeoutError(Exception):
    """
    Raised when ACK is not received within the timeout period.

    This triggers protocol abort as the ordering guarantee cannot be verified.
    """

    pass


class DuplicateMessageError(Exception):
    """
    Raised when a duplicate (session_id, seq) is detected.

    Note: This is typically handled internally by re-sending ACK.
    """

    pass


class OutOfOrderError(Exception):
    """
    Raised when a message with unexpected sequence number is received.

    Sprint 2 specification: reject and abort on out-of-order delivery.
    """

    pass


# =============================================================================
# Data Structures: Message Envelope
# =============================================================================


@dataclass(frozen=True)
class MessageEnvelope:
    """
    Envelope for ordered protocol messages.

    All ordered messages are wrapped in this envelope to enable
    sequence tracking and acknowledgment.

    Attributes
    ----------
    session_id : str
        Opaque identifier for a single protocol execution.
    seq : int
        Monotonically increasing integer per direction per session.
    msg_type : MessageType
        Discriminator identifying message semantics.
    payload : Dict[str, Any]
        The message payload (dataclass serialized to dict).

    Notes
    -----
    The envelope is serialized to JSON for transmission over
    ClassicalSocket.
    """

    session_id: str
    seq: int
    msg_type: MessageType
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """
        Serialize envelope to JSON string for transmission.

        Returns
        -------
        str
            JSON-encoded envelope.
        """
        return json.dumps(
            {
                "session_id": self.session_id,
                "seq": self.seq,
                "msg_type": self.msg_type.value,
                "payload": self.payload,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "MessageEnvelope":
        """
        Deserialize envelope from JSON string.

        Parameters
        ----------
        data : str
            JSON-encoded envelope.

        Returns
        -------
        MessageEnvelope
            Reconstructed envelope.

        Raises
        ------
        ValueError
            If JSON is malformed or missing required fields.
        """
        try:
            obj = json.loads(data)
            return cls(
                session_id=obj["session_id"],
                seq=obj["seq"],
                msg_type=MessageType(obj["msg_type"]),
                payload=obj["payload"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid message envelope: {e}") from e


@dataclass(frozen=True)
class AckPayload:
    """
    Acknowledgment payload for ordered messages.

    Attributes
    ----------
    ack_seq : int
        Sequence number being acknowledged.
    ack_msg_type : MessageType
        Message type being acknowledged (defensive disambiguation).
    """

    ack_seq: int
    ack_msg_type: MessageType

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for envelope payload."""
        return {"ack_seq": self.ack_seq, "ack_msg_type": self.ack_msg_type.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AckPayload":
        """Reconstruct from dictionary."""
        return cls(
            ack_seq=data["ack_seq"], ack_msg_type=MessageType(data["ack_msg_type"])
        )


# =============================================================================
# Ordered Protocol Socket (Non-async version for unit testing)
# =============================================================================


@dataclass
class OrderedSocketState:
    """
    Mutable state for OrderedProtocolSocket.

    This is separated from the socket class for easier testing
    and state inspection.

    Attributes
    ----------
    session_id : str
        Unique session identifier.
    state : SocketState
        Current state machine state.
    send_seq : int
        Next sequence number to use for sending.
    recv_seq : int
        Next expected sequence number for receiving.
    pending_ack_seq : Optional[int]
        Sequence number awaiting acknowledgment (if any).
    pending_ack_type : Optional[MessageType]
        Message type awaiting acknowledgment.
    processed_seqs : Set[int]
        Set of already-processed sequence numbers (for duplicate detection).
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SocketState = SocketState.IDLE
    send_seq: int = 0
    recv_seq: int = 0
    pending_ack_seq: Optional[int] = None
    pending_ack_type: Optional[MessageType] = None
    processed_seqs: Set[int] = field(default_factory=set)


class OrderedProtocolSocket:
    """
    Socket wrapper enforcing ordered message delivery with acknowledgments.

    This class implements the TASK-ORDERED-MSG-001 requirements, providing
    send-with-ACK and receive-and-ACK semantics to enforce commit-then-reveal
    ordering.

    Attributes
    ----------
    socket_state : OrderedSocketState
        Mutable state tracking sequence numbers and socket state.

    Notes
    -----
    This is a base implementation for testing. The SquidASM integration
    version extends this with generator-based async operations.

    Examples
    --------
    >>> sock = OrderedProtocolSocket()
    >>> # Create envelope for detection report
    >>> envelope = sock.create_envelope(
    ...     MessageType.DETECTION_REPORT,
    ...     {"total_rounds": 1000, "detected_indices": [...]}
    ... )
    >>> # In SquidASM: yield from sock.send_with_ack(socket, envelope, timeout)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Initialize ordered socket.

        Parameters
        ----------
        session_id : Optional[str]
            Session identifier. If None, generates UUID.
        """
        self.socket_state = OrderedSocketState(
            session_id=session_id or str(uuid.uuid4())
        )
        logger.debug(
            "OrderedProtocolSocket initialized with session_id=%s",
            self.socket_state.session_id,
        )

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return self.socket_state.session_id

    @property
    def state(self) -> SocketState:
        """Get current socket state."""
        return self.socket_state.state

    def create_envelope(
        self, msg_type: MessageType, payload: Dict[str, Any]
    ) -> MessageEnvelope:
        """
        Create a message envelope with current sequence number.

        Parameters
        ----------
        msg_type : MessageType
            Type of message.
        payload : Dict[str, Any]
            Message payload.

        Returns
        -------
        MessageEnvelope
            Envelope ready for transmission.

        Raises
        ------
        OrderingViolationError
            If socket is in VIOLATION state.
        """
        if self.socket_state.state == SocketState.VIOLATION:
            raise OrderingViolationError(
                "Cannot create envelope: socket in VIOLATION state"
            )

        envelope = MessageEnvelope(
            session_id=self.socket_state.session_id,
            seq=self.socket_state.send_seq,
            msg_type=msg_type,
            payload=payload,
        )
        return envelope

    def create_ack(self, for_envelope: MessageEnvelope) -> MessageEnvelope:
        """
        Create an ACK envelope for a received message.

        Parameters
        ----------
        for_envelope : MessageEnvelope
            The envelope to acknowledge.

        Returns
        -------
        MessageEnvelope
            ACK envelope.
        """
        ack_payload = AckPayload(
            ack_seq=for_envelope.seq, ack_msg_type=for_envelope.msg_type
        )
        return MessageEnvelope(
            session_id=self.socket_state.session_id,
            seq=self.socket_state.send_seq,  # ACKs also have sequence numbers
            msg_type=MessageType.ACK,
            payload=ack_payload.to_dict(),
        )

    def mark_sent(self, envelope: MessageEnvelope) -> None:
        """
        Mark that a message has been sent, update state to await ACK.

        Parameters
        ----------
        envelope : MessageEnvelope
            The sent envelope.

        Raises
        ------
        OrderingViolationError
            If socket is not in IDLE state or already awaiting ACK.
        """
        if self.socket_state.state != SocketState.IDLE:
            raise OrderingViolationError(
                f"Cannot send: socket in {self.socket_state.state.name} state"
            )

        self.socket_state.send_seq += 1
        self.socket_state.pending_ack_seq = envelope.seq
        self.socket_state.pending_ack_type = envelope.msg_type
        self.socket_state.state = SocketState.SENT_WAIT_ACK

        logger.info(
            "ORDERED_SEND msg_type=%s seq=%d session=%s",
            envelope.msg_type.value,
            envelope.seq,
            self.socket_state.session_id[:8],
        )

    def process_received(self, envelope: MessageEnvelope) -> Optional[MessageEnvelope]:
        """
        Process a received envelope, returning ACK if needed.

        Parameters
        ----------
        envelope : MessageEnvelope
            Received envelope.

        Returns
        -------
        Optional[MessageEnvelope]
            ACK envelope to send back, or None if envelope was an ACK.

        Raises
        ------
        OrderingViolationError
            If session ID doesn't match.
        OutOfOrderError
            If sequence number is out of expected order.
        DuplicateMessageError
            If message was already processed (returns ACK anyway).
        """
        # Validate session
        if envelope.session_id != self.socket_state.session_id:
            self.socket_state.state = SocketState.VIOLATION
            raise OrderingViolationError(
                f"Session mismatch: expected {self.socket_state.session_id[:8]}, "
                f"got {envelope.session_id[:8]}"
            )

        # Handle ACK messages
        if envelope.msg_type == MessageType.ACK:
            return self._process_ack(envelope)

        # Check for duplicate
        if envelope.seq in self.socket_state.processed_seqs:
            logger.warning(
                "Duplicate message seq=%d, resending ACK", envelope.seq
            )
            # Re-send ACK for duplicate
            return self.create_ack(envelope)

        # Check sequence order (strict: reject out-of-order)
        expected_seq = self.socket_state.recv_seq
        if envelope.seq != expected_seq:
            self.socket_state.state = SocketState.VIOLATION
            raise OutOfOrderError(
                f"Out-of-order message: expected seq={expected_seq}, got seq={envelope.seq}"
            )

        # Process normally
        self.socket_state.recv_seq += 1
        self.socket_state.processed_seqs.add(envelope.seq)

        logger.info(
            "ORDERED_RECV msg_type=%s seq=%d session=%s",
            envelope.msg_type.value,
            envelope.seq,
            self.socket_state.session_id[:8],
        )

        # Generate ACK
        return self.create_ack(envelope)

    def _process_ack(self, envelope: MessageEnvelope) -> None:
        """
        Process an ACK message.

        Parameters
        ----------
        envelope : MessageEnvelope
            ACK envelope.

        Returns
        -------
        None
            ACKs don't generate responses.

        Raises
        ------
        OrderingViolationError
            If ACK doesn't match pending message.
        """
        if self.socket_state.state != SocketState.SENT_WAIT_ACK:
            logger.warning(
                "Received unexpected ACK in state %s",
                self.socket_state.state.name,
            )
            return None

        ack_payload = AckPayload.from_dict(envelope.payload)

        # Validate ACK matches pending
        if ack_payload.ack_seq != self.socket_state.pending_ack_seq:
            self.socket_state.state = SocketState.VIOLATION
            raise OrderingViolationError(
                f"ACK seq mismatch: expected {self.socket_state.pending_ack_seq}, "
                f"got {ack_payload.ack_seq}"
            )

        if ack_payload.ack_msg_type != self.socket_state.pending_ack_type:
            self.socket_state.state = SocketState.VIOLATION
            raise OrderingViolationError(
                f"ACK type mismatch: expected {self.socket_state.pending_ack_type}, "
                f"got {ack_payload.ack_msg_type}"
            )

        # ACK valid, return to IDLE
        self.socket_state.pending_ack_seq = None
        self.socket_state.pending_ack_type = None
        self.socket_state.state = SocketState.IDLE

        logger.info(
            "ORDERED_ACK_RECEIVED for seq=%d type=%s",
            ack_payload.ack_seq,
            ack_payload.ack_msg_type.value,
        )
        return None

    def mark_timeout(self) -> None:
        """
        Mark that ACK timeout has occurred.

        Transitions socket to VIOLATION state.

        Raises
        ------
        AckTimeoutError
            Always raised to signal timeout.
        """
        self.socket_state.state = SocketState.VIOLATION
        logger.error(
            "ACK_TIMEOUT for seq=%d type=%s session=%s",
            self.socket_state.pending_ack_seq,
            self.socket_state.pending_ack_type.value if self.socket_state.pending_ack_type else "None",
            self.socket_state.session_id[:8],
        )
        raise AckTimeoutError(
            f"ACK timeout for seq={self.socket_state.pending_ack_seq}"
        )

    def reset(self) -> None:
        """
        Reset socket state for a new session.

        Creates new session ID and clears all state.
        """
        new_session_id = str(uuid.uuid4())
        self.socket_state = OrderedSocketState(session_id=new_session_id)
        logger.debug(
            "OrderedProtocolSocket reset with new session_id=%s",
            new_session_id[:8],
        )

    def is_ack_pending(self) -> bool:
        """Check if socket is waiting for an ACK."""
        return self.socket_state.state == SocketState.SENT_WAIT_ACK

    def can_send(self) -> bool:
        """Check if socket can send a new message."""
        return self.socket_state.state == SocketState.IDLE

    def is_violated(self) -> bool:
        """Check if socket is in violation state."""
        return self.socket_state.state == SocketState.VIOLATION


# =============================================================================
# Protocol Message Payloads
# =============================================================================


@dataclass(frozen=True)
class DetectionReport:
    """
    Bob's detection report payload.

    Reports which rounds had successful photon detection vs. missing
    (no detection). This commits Bob to his detection pattern before
    Alice reveals basis information.

    Attributes
    ----------
    total_rounds : int
        Total number of quantum transmission rounds (M).
    detected_indices : List[int]
        Indices of rounds where Bob detected a photon.
    missing_indices : List[int]
        Indices of rounds where Bob did not detect a photon.

    Invariants
    ----------
    - len(detected_indices) + len(missing_indices) == total_rounds
    - detected_indices ∩ missing_indices = ∅
    - All indices in [0, total_rounds - 1]

    References
    ----------
    - sprint_2_specification.md Section 2.5.1
    - Phase II analysis: DetectionReport dataclass
    """

    total_rounds: int
    detected_indices: List[int]
    missing_indices: List[int]

    def __post_init__(self) -> None:
        """Validate invariants."""
        # Check count invariant
        if len(self.detected_indices) + len(self.missing_indices) != self.total_rounds:
            raise ValueError(
                f"Invariant violated: detected ({len(self.detected_indices)}) + "
                f"missing ({len(self.missing_indices)}) != total_rounds ({self.total_rounds})"
            )

        # Check disjoint invariant
        detected_set = set(self.detected_indices)
        missing_set = set(self.missing_indices)
        overlap = detected_set & missing_set
        if overlap:
            raise ValueError(
                f"Invariant violated: detected ∩ missing is non-empty: {overlap}"
            )

        # Check range invariant
        all_indices = detected_set | missing_set
        if all_indices:
            if min(all_indices) < 0:
                raise ValueError(
                    f"Invariant violated: index < 0: {min(all_indices)}"
                )
            if max(all_indices) >= self.total_rounds:
                raise ValueError(
                    f"Invariant violated: index >= total_rounds: {max(all_indices)}"
                )

    @property
    def num_detected(self) -> int:
        """Number of detected rounds (S)."""
        return len(self.detected_indices)

    @property
    def num_missing(self) -> int:
        """Number of missing rounds."""
        return len(self.missing_indices)

    @property
    def detection_rate(self) -> float:
        """Detection rate S/M."""
        if self.total_rounds == 0:
            return 0.0
        return self.num_detected / self.total_rounds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for envelope payload."""
        return {
            "total_rounds": self.total_rounds,
            "detected_indices": list(self.detected_indices),
            "missing_indices": list(self.missing_indices),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionReport":
        """Reconstruct from dictionary."""
        return cls(
            total_rounds=data["total_rounds"],
            detected_indices=data["detected_indices"],
            missing_indices=data["missing_indices"],
        )


# Basis encoding constants
BASIS_Z = 0  # Computational basis (standard)
BASIS_X = 1  # Hadamard basis (diagonal)


@dataclass(frozen=True)
class BasisReveal:
    """
    Alice's basis revelation payload.

    Reveals Alice's basis choices for all rounds after Bob has
    committed his detection report and Δt has elapsed.

    Attributes
    ----------
    total_rounds : int
        Total number of quantum transmission rounds (M).
    bases : List[int]
        Basis choice for each round, encoded as:
        - 0 (BASIS_Z): Computational basis (Z)
        - 1 (BASIS_X): Hadamard basis (X)

    Invariants
    ----------
    - len(bases) == total_rounds
    - All elements in {0, 1}

    References
    ----------
    - sprint_2_specification.md Section 2.5.2
    - Phase II analysis: α^m basis string
    """

    total_rounds: int
    bases: List[int]

    def __post_init__(self) -> None:
        """Validate invariants."""
        if len(self.bases) != self.total_rounds:
            raise ValueError(
                f"Invariant violated: len(bases) ({len(self.bases)}) != "
                f"total_rounds ({self.total_rounds})"
            )

        for i, b in enumerate(self.bases):
            if b not in (BASIS_Z, BASIS_X):
                raise ValueError(
                    f"Invariant violated: bases[{i}] = {b}, expected 0 (Z) or 1 (X)"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for envelope payload."""
        return {
            "total_rounds": self.total_rounds,
            "bases": list(self.bases),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasisReveal":
        """Reconstruct from dictionary."""
        return cls(
            total_rounds=data["total_rounds"],
            bases=data["bases"],
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_ACK_TIMEOUT_NS",
    "BASIS_Z",
    "BASIS_X",
    # Enums
    "MessageType",
    "SocketState",
    # Exceptions
    "OrderingViolationError",
    "AckTimeoutError",
    "DuplicateMessageError",
    "OutOfOrderError",
    # Data structures
    "MessageEnvelope",
    "AckPayload",
    "OrderedSocketState",
    "OrderedProtocolSocket",
    "DetectionReport",
    "BasisReveal",
]

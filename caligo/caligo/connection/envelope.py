"""Message envelope types for ordered messaging.

This module provides a small JSON-serializable envelope that binds messages to
an execution session and enforces per-direction ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Protocol message type discriminators."""

    # Phase II
    DETECTION_COMMITMENT = "DETECTION_COMMITMENT"
    BASIS_REVEAL = "BASIS_REVEAL"
    COMMITMENT_OPENING = "COMMITMENT_OPENING"

    # Sifting
    INDEX_LISTS = "INDEX_LISTS"
    TEST_OUTCOMES = "TEST_OUTCOMES"

    # Reconciliation
    SYNDROME = "SYNDROME"
    SYNDROME_RESPONSE = "SYNDROME_RESPONSE"

    # Amplification
    TOEPLITZ_SEED = "TOEPLITZ_SEED"

    # Control
    ACK = "ACK"
    ABORT = "ABORT"


@dataclass(frozen=True)
class AckPayload:
    """Acknowledgment payload for ordered messages.

    Parameters
    ----------
    ack_seq : int
        Sequence number of the message being acknowledged.
    ack_msg_type : MessageType
        Type of message being acknowledged.
    """

    ack_seq: int
    ack_msg_type: MessageType

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""

        return {"ack_seq": int(self.ack_seq), "ack_msg_type": self.ack_msg_type.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AckPayload":
        """Reconstruct from dictionary."""

        return cls(
            ack_seq=int(data["ack_seq"]),
            ack_msg_type=MessageType(str(data["ack_msg_type"])),
        )


@dataclass(frozen=True)
class MessageEnvelope:
    """Envelope for ordered protocol messages.

    Parameters
    ----------
    session_id : str
        Opaque identifier binding messages to a single protocol run.
    seq : int
        Monotonically increasing sequence number per direction.
    msg_type : MessageType
        Discriminator identifying message semantics.
    payload : Dict[str, Any]
        JSON-serializable payload.
    """

    session_id: str
    seq: int
    msg_type: MessageType
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize envelope to a compact JSON string."""

        return json.dumps(
            {
                "session_id": self.session_id,
                "seq": int(self.seq),
                "msg_type": self.msg_type.value,
                "payload": self.payload,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> "MessageEnvelope":
        """Deserialize envelope from a JSON string.

        Raises
        ------
        ValueError
            If JSON is malformed or missing required fields.
        """

        try:
            obj = json.loads(data)
            session_id = str(obj["session_id"])
            seq = int(obj["seq"])
            msg_type = MessageType(str(obj["msg_type"]))
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be a dict")
            return cls(session_id=session_id, seq=seq, msg_type=msg_type, payload=payload)
        except Exception as exc:
            logger.error("Failed to parse MessageEnvelope JSON: %s", exc)
            raise ValueError(f"Malformed MessageEnvelope JSON: {exc}") from exc

"""Caligo connection package: ordered protocol messaging.

This package implements the commit-then-reveal semantics required for NSM
security via an ACK-enforced ordered messaging wrapper.

Exports
-------
MessageType
    Protocol message type discriminators.
MessageEnvelope
    Message wrapper with sequence tracking.
AckPayload
    Acknowledgment payload structure.
OrderedSocket
    ACK-enforced ordered messaging wrapper.

Exceptions
----------
ConnectionError
OrderingViolationError
AckTimeoutError
SessionMismatchError
OutOfOrderError
    Connection-layer exceptions defined in :mod:`caligo.types.exceptions`.
"""

from caligo.connection.envelope import AckPayload, MessageEnvelope, MessageType
from caligo.connection.ordered_socket import OrderedSocket, SocketState
from caligo.connection.exceptions import (
    AckTimeoutError,
    ConnectionError,
    OrderingViolationError,
    OutOfOrderError,
    SessionMismatchError,
)

__all__ = [
    "MessageType",
    "MessageEnvelope",
    "AckPayload",
    "SocketState",
    "OrderedSocket",
    "ConnectionError",
    "OrderingViolationError",
    "AckTimeoutError",
    "SessionMismatchError",
    "OutOfOrderError",
]

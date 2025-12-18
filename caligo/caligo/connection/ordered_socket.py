"""Ordered classical socket wrapper.

SquidASM classical sockets are unordered in the sense that application code can
send arbitrary messages back-to-back without causal coupling. For Caligo we need
an explicit commit-then-reveal discipline. This wrapper enforces per-direction
ordering with an ACK handshake.

Notes
-----
This wrapper assumes the underlying socket offers:
- send(str) -> None
- recv() -> generator yielding to the simulator and returning a str

That matches SquidASM's socket API pattern ("yield from sock.recv()").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generator, Optional

from caligo.connection.envelope import AckPayload, MessageEnvelope, MessageType
from caligo.types.exceptions import (
    AckTimeoutError,
    OrderingViolationError,
    OutOfOrderError,
    SessionMismatchError,
)
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class SocketState(str, Enum):
    """Internal state of the ordered socket."""

    IDLE = "IDLE"
    SENT_WAIT_ACK = "SENT_WAIT_ACK"
    VIOLATION = "VIOLATION"


@dataclass
class _PendingSend:
    seq: int
    msg_type: MessageType


class OrderedSocket:
    """Ordered messaging on top of a SquidASM classical socket.

    Parameters
    ----------
    socket : Any
        Underlying SquidASM classical socket.
    session_id : str
        Identifier tying messages to a single protocol instance.
    ack_timeout_rounds : int
        Maximum number of receive attempts while waiting for an ACK.
    """

    def __init__(self, socket, session_id: str, ack_timeout_rounds: int = 10):
        self._sock = socket
        self._session_id = session_id
        self._ack_timeout_rounds = int(ack_timeout_rounds)

        self._send_seq = 0
        self._recv_seq = 0

        self._state: SocketState = SocketState.IDLE
        self._pending: Optional[_PendingSend] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SocketState:
        return self._state

    def send(self, msg_type: MessageType, payload: dict) -> Generator:
        """Send a message and wait for its ACK.

        This is a generator and must be used as `yield from`.

        Parameters
        ----------
        msg_type : MessageType
            Type discriminator.
        payload : dict
            JSON-serializable payload.

        Raises
        ------
        OrderingViolationError
            If called while a previous send is awaiting ACK.
        AckTimeoutError
            If ACK isn't received within `ack_timeout_rounds` receive iterations.
        SessionMismatchError
            If a received envelope has the wrong session id.
        OutOfOrderError
            If a received envelope sequence is inconsistent.
        """

        if self._state != SocketState.IDLE:
            self._state = SocketState.VIOLATION
            raise OrderingViolationError(
                f"OrderedSocket.send called in state {self._state}; previous send not acked"
            )

        seq = self._send_seq
        self._send_seq += 1
        self._pending = _PendingSend(seq=seq, msg_type=msg_type)
        self._state = SocketState.SENT_WAIT_ACK

        env = MessageEnvelope(
            session_id=self._session_id,
            seq=seq,
            msg_type=msg_type,
            payload=payload,
        )
        self._sock.send(env.to_json())

        # Wait for ACK; while waiting we may also receive and ack application
        # messages from the peer.
        tries = 0
        while True:
            tries += 1
            if tries > self._ack_timeout_rounds:
                self._state = SocketState.VIOLATION
                raise AckTimeoutError(
                    f"Timed out waiting for ACK for seq={seq}, msg_type={msg_type}"
                )

            incoming = yield from self._sock.recv()
            env_in = MessageEnvelope.from_json(incoming)
            self._validate_session(env_in)

            if env_in.msg_type == MessageType.ACK:
                ack = AckPayload.from_dict(env_in.payload)
                if ack.ack_seq == seq and ack.ack_msg_type == msg_type:
                    self._pending = None
                    self._state = SocketState.IDLE
                    return
                # Unexpected ACK: treat as ordering violation.
                self._state = SocketState.VIOLATION
                raise OutOfOrderError(
                    f"Unexpected ACK ack_seq={ack.ack_seq} ack_msg_type={ack.ack_msg_type}; "
                    f"expected seq={seq} msg_type={msg_type}"
                )

            # Application message while we're waiting: enforce recv ordering and ACK it.
            self._handle_incoming_application_message(env_in)

    def recv(self, expected_type: MessageType) -> Generator[object, object, dict]:
        """Receive the next application message of a given type.

        Parameters
        ----------
        expected_type : MessageType
            Expected incoming message type.

        Returns
        -------
        dict
            The received payload.

        Raises
        ------
        SessionMismatchError
            If message belongs to a different session.
        OutOfOrderError
            If sequence numbers are inconsistent.
        """

        while True:
            raw = yield from self._sock.recv()
            env = MessageEnvelope.from_json(raw)
            self._validate_session(env)

            if env.msg_type == MessageType.ACK:
                # ACKs are only meaningful for send(), so ignore here.
                continue

            payload = self._consume_application_message(env)
            if env.msg_type != expected_type:
                raise OutOfOrderError(
                    f"Expected msg_type={expected_type}, got {env.msg_type} at seq={env.seq}"
                )
            return payload

    def _validate_session(self, env: MessageEnvelope) -> None:
        if env.session_id != self._session_id:
            self._state = SocketState.VIOLATION
            raise SessionMismatchError(
                f"Session mismatch; expected {self._session_id}, got {env.session_id}"
            )

    def _consume_application_message(self, env: MessageEnvelope) -> dict:
        if env.seq != self._recv_seq:
            self._state = SocketState.VIOLATION
            raise OutOfOrderError(
                f"Out of order receive; expected seq={self._recv_seq}, got seq={env.seq}"
            )
        self._recv_seq += 1

        # ACK it.
        ack_env = MessageEnvelope(
            session_id=self._session_id,
            seq=env.seq,
            msg_type=MessageType.ACK,
            payload=AckPayload(ack_seq=env.seq, ack_msg_type=env.msg_type).to_dict(),
        )
        self._sock.send(ack_env.to_json())
        return env.payload

    def _handle_incoming_application_message(self, env: MessageEnvelope) -> None:
        """Handle an application message received while waiting for ACK.

        We can't deliver it to the caller here, but we must still enforce
        sequencing and ACK it so the peer can progress.

        Notes
        -----
        This is intentionally strict: any unexpected sequencing triggers a
        violation to avoid subtle causal ordering bugs.
        """

        _ = self._consume_application_message(env)
        logger.debug(
            "Received application message %s seq=%s while waiting for ACK; acked and dropped",
            env.msg_type,
            env.seq,
        )

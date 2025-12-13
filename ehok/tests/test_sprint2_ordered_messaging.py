"""
Unit tests for Sprint 2 Ordered Messaging.

Tests the OrderedProtocolSocket, message envelope, and payload contracts
per sprint_2_specification.md Section 2 and Section 5.1.
"""

import json
import pytest
from typing import List

from ehok.protocols.ordered_messaging import (
    # Constants
    DEFAULT_ACK_TIMEOUT_NS,
    BASIS_Z,
    BASIS_X,
    # Enums
    MessageType,
    SocketState,
    # Exceptions
    OrderingViolationError,
    AckTimeoutError,
    DuplicateMessageError,
    OutOfOrderError,
    # Data structures
    MessageEnvelope,
    AckPayload,
    OrderedSocketState,
    OrderedProtocolSocket,
    DetectionReport,
    BasisReveal,
)


# =============================================================================
# Message Envelope Tests
# =============================================================================


class TestMessageEnvelope:
    """Tests for MessageEnvelope serialization and deserialization."""

    def test_to_json_round_trip(self) -> None:
        """Envelope should serialize and deserialize correctly."""
        envelope = MessageEnvelope(
            session_id="test-session-123",
            seq=5,
            msg_type=MessageType.DETECTION_REPORT,
            payload={"total_rounds": 1000, "detected": [1, 2, 3]},
        )

        json_str = envelope.to_json()
        restored = MessageEnvelope.from_json(json_str)

        assert restored.session_id == envelope.session_id
        assert restored.seq == envelope.seq
        assert restored.msg_type == envelope.msg_type
        assert restored.payload == envelope.payload

    def test_all_message_types(self) -> None:
        """All message types should serialize correctly."""
        for msg_type in MessageType:
            envelope = MessageEnvelope(
                session_id="sess",
                seq=0,
                msg_type=msg_type,
                payload={},
            )
            restored = MessageEnvelope.from_json(envelope.to_json())
            assert restored.msg_type == msg_type

    def test_from_json_invalid(self) -> None:
        """Invalid JSON should raise ValueError."""
        with pytest.raises(ValueError):
            MessageEnvelope.from_json("not json")

    def test_from_json_missing_field(self) -> None:
        """Missing required field should raise ValueError."""
        incomplete = json.dumps({"session_id": "x", "seq": 0})
        with pytest.raises(ValueError):
            MessageEnvelope.from_json(incomplete)


class TestAckPayload:
    """Tests for AckPayload."""

    def test_to_dict_round_trip(self) -> None:
        """AckPayload should convert to/from dict."""
        ack = AckPayload(ack_seq=10, ack_msg_type=MessageType.DETECTION_REPORT)
        restored = AckPayload.from_dict(ack.to_dict())
        assert restored.ack_seq == ack.ack_seq
        assert restored.ack_msg_type == ack.ack_msg_type


# =============================================================================
# OrderedProtocolSocket Tests
# =============================================================================


class TestOrderedProtocolSocket:
    """Tests for OrderedProtocolSocket state machine."""

    @pytest.fixture
    def socket(self) -> OrderedProtocolSocket:
        """Create a fresh socket for testing."""
        return OrderedProtocolSocket(session_id="test-session")

    def test_initial_state(self, socket: OrderedProtocolSocket) -> None:
        """New socket should be in IDLE state."""
        assert socket.state == SocketState.IDLE
        assert socket.can_send()
        assert not socket.is_ack_pending()
        assert not socket.is_violated()

    def test_create_envelope(self, socket: OrderedProtocolSocket) -> None:
        """create_envelope should produce valid envelope."""
        envelope = socket.create_envelope(
            MessageType.DETECTION_REPORT,
            {"data": "test"},
        )
        assert envelope.session_id == socket.session_id
        assert envelope.seq == 0
        assert envelope.msg_type == MessageType.DETECTION_REPORT

    def test_mark_sent_transitions_state(self, socket: OrderedProtocolSocket) -> None:
        """mark_sent should transition to SENT_WAIT_ACK."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        assert socket.state == SocketState.SENT_WAIT_ACK
        assert socket.is_ack_pending()
        assert not socket.can_send()

    def test_mark_sent_increments_seq(self, socket: OrderedProtocolSocket) -> None:
        """mark_sent should increment send sequence number."""
        # Create and send first message
        env1 = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        assert env1.seq == 0
        socket.mark_sent(env1)

        # Simulate receiving ACK to return to IDLE
        ack_envelope = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.ACK,
            payload=AckPayload(ack_seq=0, ack_msg_type=MessageType.DETECTION_REPORT).to_dict(),
        )
        socket.process_received(ack_envelope)

        # Create second message
        env2 = socket.create_envelope(MessageType.BASIS_REVEAL, {})
        assert env2.seq == 1

    def test_cannot_send_while_waiting_ack(self, socket: OrderedProtocolSocket) -> None:
        """Cannot send another message while waiting for ACK."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        # Trying to send again should fail
        envelope2 = socket.create_envelope(MessageType.BASIS_REVEAL, {})
        with pytest.raises(OrderingViolationError):
            socket.mark_sent(envelope2)

    def test_process_received_generates_ack(self, socket: OrderedProtocolSocket) -> None:
        """Processing a received message should generate ACK."""
        incoming = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.DETECTION_REPORT,
            payload={"data": "test"},
        )

        ack = socket.process_received(incoming)

        assert ack is not None
        assert ack.msg_type == MessageType.ACK
        ack_payload = AckPayload.from_dict(ack.payload)
        assert ack_payload.ack_seq == 0
        assert ack_payload.ack_msg_type == MessageType.DETECTION_REPORT

    def test_process_ack_returns_to_idle(self, socket: OrderedProtocolSocket) -> None:
        """Valid ACK should return socket to IDLE."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)
        assert socket.state == SocketState.SENT_WAIT_ACK

        # Process matching ACK
        ack_envelope = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.ACK,
            payload=AckPayload(ack_seq=0, ack_msg_type=MessageType.DETECTION_REPORT).to_dict(),
        )
        result = socket.process_received(ack_envelope)

        assert result is None  # ACKs don't generate responses
        assert socket.state == SocketState.IDLE
        assert socket.can_send()

    def test_wrong_session_id_raises(self, socket: OrderedProtocolSocket) -> None:
        """Message with wrong session_id should raise and violate."""
        wrong_session = MessageEnvelope(
            session_id="wrong-session",
            seq=0,
            msg_type=MessageType.DETECTION_REPORT,
            payload={},
        )

        with pytest.raises(OrderingViolationError, match="Session mismatch"):
            socket.process_received(wrong_session)

        assert socket.is_violated()

    def test_out_of_order_raises(self, socket: OrderedProtocolSocket) -> None:
        """Out-of-order sequence number should raise and violate."""
        # Expected seq is 0, but we send seq=5
        out_of_order = MessageEnvelope(
            session_id=socket.session_id,
            seq=5,
            msg_type=MessageType.DETECTION_REPORT,
            payload={},
        )

        with pytest.raises(OutOfOrderError, match="Out-of-order"):
            socket.process_received(out_of_order)

        assert socket.is_violated()

    def test_duplicate_message_resends_ack(self, socket: OrderedProtocolSocket) -> None:
        """Duplicate message should resend ACK without reprocessing."""
        incoming = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.DETECTION_REPORT,
            payload={"data": "test"},
        )

        # Process first time
        ack1 = socket.process_received(incoming)
        assert ack1 is not None

        # Process duplicate
        ack2 = socket.process_received(incoming)
        assert ack2 is not None  # ACK is re-sent

        # Sequence should still be 1 (not incremented by duplicate)
        assert socket.socket_state.recv_seq == 1

    def test_ack_mismatch_seq_raises(self, socket: OrderedProtocolSocket) -> None:
        """ACK with wrong seq should raise."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        wrong_ack = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.ACK,
            payload=AckPayload(ack_seq=999, ack_msg_type=MessageType.DETECTION_REPORT).to_dict(),
        )

        with pytest.raises(OrderingViolationError, match="ACK seq mismatch"):
            socket.process_received(wrong_ack)

    def test_ack_mismatch_type_raises(self, socket: OrderedProtocolSocket) -> None:
        """ACK with wrong msg_type should raise."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        wrong_ack = MessageEnvelope(
            session_id=socket.session_id,
            seq=0,
            msg_type=MessageType.ACK,
            payload=AckPayload(ack_seq=0, ack_msg_type=MessageType.BASIS_REVEAL).to_dict(),
        )

        with pytest.raises(OrderingViolationError, match="ACK type mismatch"):
            socket.process_received(wrong_ack)

    def test_mark_timeout_transitions_to_violation(self, socket: OrderedProtocolSocket) -> None:
        """mark_timeout should raise and transition to VIOLATION."""
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        with pytest.raises(AckTimeoutError):
            socket.mark_timeout()

        assert socket.is_violated()

    def test_reset(self, socket: OrderedProtocolSocket) -> None:
        """reset should clear state and generate new session."""
        old_session = socket.session_id

        # Put socket in non-idle state
        envelope = socket.create_envelope(MessageType.DETECTION_REPORT, {})
        socket.mark_sent(envelope)

        socket.reset()

        assert socket.state == SocketState.IDLE
        assert socket.session_id != old_session
        assert socket.socket_state.send_seq == 0
        assert socket.socket_state.recv_seq == 0

    def test_cannot_create_envelope_in_violation_state(self) -> None:
        """Cannot create envelope when in VIOLATION state."""
        socket = OrderedProtocolSocket()

        # Force violation
        wrong_session = MessageEnvelope(
            session_id="wrong",
            seq=0,
            msg_type=MessageType.DETECTION_REPORT,
            payload={},
        )
        try:
            socket.process_received(wrong_session)
        except OrderingViolationError:
            pass

        with pytest.raises(OrderingViolationError, match="VIOLATION state"):
            socket.create_envelope(MessageType.BASIS_REVEAL, {})


# =============================================================================
# DetectionReport Tests
# =============================================================================


class TestDetectionReport:
    """Tests for DetectionReport payload."""

    def test_valid_report(self) -> None:
        """Valid report should construct without error."""
        report = DetectionReport(
            total_rounds=1000,
            detected_indices=list(range(100)),
            missing_indices=list(range(100, 1000)),
        )
        assert report.num_detected == 100
        assert report.num_missing == 900
        assert report.detection_rate == 0.1

    def test_empty_detected(self) -> None:
        """Report with no detections should be valid."""
        report = DetectionReport(
            total_rounds=100,
            detected_indices=[],
            missing_indices=list(range(100)),
        )
        assert report.num_detected == 0
        assert report.detection_rate == 0.0

    def test_all_detected(self) -> None:
        """Report with all detections should be valid."""
        report = DetectionReport(
            total_rounds=100,
            detected_indices=list(range(100)),
            missing_indices=[],
        )
        assert report.num_detected == 100
        assert report.detection_rate == 1.0

    def test_invariant_count_mismatch(self) -> None:
        """Report with count mismatch should raise ValueError."""
        with pytest.raises(ValueError, match="Invariant violated"):
            DetectionReport(
                total_rounds=100,
                detected_indices=[0, 1, 2],
                missing_indices=[3, 4, 5],  # Only 6 total, not 100
            )

    def test_invariant_overlap(self) -> None:
        """Report with overlapping indices should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            DetectionReport(
                total_rounds=10,
                detected_indices=[0, 1, 2, 3, 4],
                missing_indices=[4, 5, 6, 7, 8],  # 4 appears in both
            )

    def test_invariant_index_out_of_range(self) -> None:
        """Report with out-of-range index should raise ValueError."""
        with pytest.raises(ValueError, match="index >= total_rounds"):
            DetectionReport(
                total_rounds=10,
                detected_indices=[0, 1, 2, 10],  # 10 is out of range
                missing_indices=[3, 4, 5, 6, 7, 8],
            )

    def test_invariant_negative_index(self) -> None:
        """Report with negative index should raise ValueError."""
        with pytest.raises(ValueError, match="index < 0"):
            DetectionReport(
                total_rounds=10,
                detected_indices=[-1, 0, 1, 2, 3],
                missing_indices=[4, 5, 6, 7, 8],
            )

    def test_to_dict_round_trip(self) -> None:
        """DetectionReport should serialize and deserialize."""
        report = DetectionReport(
            total_rounds=100,
            detected_indices=list(range(10)),
            missing_indices=list(range(10, 100)),
        )
        restored = DetectionReport.from_dict(report.to_dict())
        assert restored.total_rounds == report.total_rounds
        assert restored.detected_indices == report.detected_indices
        assert restored.missing_indices == report.missing_indices


# =============================================================================
# BasisReveal Tests
# =============================================================================


class TestBasisReveal:
    """Tests for BasisReveal payload."""

    def test_valid_reveal(self) -> None:
        """Valid reveal should construct without error."""
        bases = [BASIS_Z, BASIS_X, BASIS_Z, BASIS_X, BASIS_Z]
        reveal = BasisReveal(total_rounds=5, bases=bases)
        assert len(reveal.bases) == 5

    def test_invariant_length_mismatch(self) -> None:
        """Reveal with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="len\\(bases\\)"):
            BasisReveal(total_rounds=10, bases=[0, 1, 0])

    def test_invariant_invalid_basis(self) -> None:
        """Reveal with invalid basis value should raise ValueError."""
        with pytest.raises(ValueError, match="expected 0 \\(Z\\) or 1 \\(X\\)"):
            BasisReveal(total_rounds=3, bases=[0, 2, 1])

    def test_to_dict_round_trip(self) -> None:
        """BasisReveal should serialize and deserialize."""
        reveal = BasisReveal(
            total_rounds=5,
            bases=[BASIS_Z, BASIS_X, BASIS_Z, BASIS_X, BASIS_Z],
        )
        restored = BasisReveal.from_dict(reveal.to_dict())
        assert restored.total_rounds == reveal.total_rounds
        assert restored.bases == reveal.bases


# =============================================================================
# Commit-then-Reveal Protocol Test
# =============================================================================


class TestCommitThenRevealProtocol:
    """Integration test for commit-then-reveal ordering."""

    def test_correct_ordering_flow(self) -> None:
        """Test correct protocol flow: detection report → ACK → basis reveal."""
        alice = OrderedProtocolSocket(session_id="protocol-test")
        bob = OrderedProtocolSocket(session_id="protocol-test")

        # Step 1: Bob creates and "sends" detection report
        detection_payload = DetectionReport(
            total_rounds=100,
            detected_indices=list(range(10)),
            missing_indices=list(range(10, 100)),
        )
        bob_envelope = bob.create_envelope(
            MessageType.DETECTION_REPORT,
            detection_payload.to_dict(),
        )
        bob.mark_sent(bob_envelope)
        assert bob.state == SocketState.SENT_WAIT_ACK

        # Step 2: Alice receives and ACKs
        # Simulate receiving on Alice's side
        alice_sock = OrderedProtocolSocket(session_id="protocol-test")
        ack = alice_sock.process_received(bob_envelope)
        assert ack is not None
        assert ack.msg_type == MessageType.ACK

        # Step 3: Bob processes ACK
        bob_sock2 = OrderedProtocolSocket(session_id="protocol-test")
        bob_sock2.mark_sent(bob_envelope)  # Simulate sent state
        bob_sock2.process_received(ack)
        assert bob_sock2.state == SocketState.IDLE

        # Step 4: Now Alice can send basis reveal
        # (After Δt - tested in timing module)
        basis_payload = BasisReveal(
            total_rounds=100,
            bases=[BASIS_Z] * 50 + [BASIS_X] * 50,
        )
        alice_reveal = alice_sock.create_envelope(
            MessageType.BASIS_REVEAL,
            basis_payload.to_dict(),
        )
        alice_sock.mark_sent(alice_reveal)
        assert alice_sock.is_ack_pending()

    def test_basis_reveal_before_ack_blocked(self) -> None:
        """Basis reveal cannot be sent while waiting for detection ACK."""
        socket = OrderedProtocolSocket(session_id="test")

        # Send detection report
        detection_payload = DetectionReport(
            total_rounds=100,
            detected_indices=list(range(10)),
            missing_indices=list(range(10, 100)),
        )
        envelope = socket.create_envelope(
            MessageType.DETECTION_REPORT,
            detection_payload.to_dict(),
        )
        socket.mark_sent(envelope)

        # Attempting to send basis reveal without ACK should fail
        basis_payload = BasisReveal(total_rounds=100, bases=[BASIS_Z] * 100)
        basis_envelope = socket.create_envelope(
            MessageType.BASIS_REVEAL,
            basis_payload.to_dict(),
        )

        with pytest.raises(OrderingViolationError):
            socket.mark_sent(basis_envelope)

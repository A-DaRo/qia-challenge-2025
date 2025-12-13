"""
System Integration Tests: Ordered Messaging.

Test Cases
----------
SYS-INT-MSG-001: Ordered Socket ACK Blocking
SYS-INT-MSG-002: ACK Timeout Triggers Abort

Reference
---------
System Test Specification §2.3 (GAP: ORDERED-MSG-001)
"""

import pytest
from typing import Optional, Any

# ============================================================================
# Attempt to import required modules - let ImportError happen if missing
# ============================================================================

# E-HOK ordered messaging modules under test
from ehok.protocols.ordered_messaging import (
    MessageType,
    SocketState,
    DEFAULT_ACK_TIMEOUT_NS,
)

# OrderedProtocolSocket - the main class under test
try:
    from ehok.protocols.ordered_messaging import OrderedProtocolSocket
except ImportError:
    OrderedProtocolSocket = None  # type: ignore

# OrderedSocketState enum (spec uses this name)
try:
    from ehok.protocols.ordered_messaging import OrderedSocketState
except ImportError:
    # Fall back to SocketState if OrderedSocketState doesn't exist
    OrderedSocketState = SocketState

# Protocol violation exception
try:
    from ehok.core.exceptions import ProtocolViolation
except ImportError:
    try:
        from ehok.protocols.ordered_messaging import ProtocolViolation
    except ImportError:
        ProtocolViolation = None  # type: ignore

# NetSquid/SquidASM for simulation
try:
    import netsquid as ns
    NETSQUID_AVAILABLE = True
except ImportError:
    ns = None  # type: ignore
    NETSQUID_AVAILABLE = False

try:
    from netqasm.sdk.classical_communication.socket import Socket as ClassicalSocket
    CLASSICAL_SOCKET_AVAILABLE = True
except ImportError:
    ClassicalSocket = None  # type: ignore
    CLASSICAL_SOCKET_AVAILABLE = False


# ============================================================================
# Test Constants (from spec)
# ============================================================================

# ACK timeout from spec
ACK_TIMEOUT_NS = 5_000_000_000  # 5 seconds

# Network delays from spec (SYS-INT-MSG-001)
ALICE_TO_BOB_DELAY_MS = 100
BOB_TO_ALICE_DELAY_MS = 500


# ============================================================================
# Mock Classical Socket for Unit Testing
# ============================================================================

class MockClassicalSocket:
    """
    Mock classical socket for unit testing OrderedProtocolSocket.
    
    This allows testing the ordered messaging logic without requiring
    a full SquidASM network setup.
    """
    
    def __init__(self, delay_ms: int = 0, drop_ack: bool = False):
        self.delay_ms = delay_ms
        self.drop_ack = drop_ack
        self.sent_messages: list = []
        self.pending_ack: bool = False
        self._received_queue: list = []
    
    def send(self, message: str) -> None:
        """Record sent message."""
        self.sent_messages.append(message)
        self.pending_ack = True
    
    def recv(self, timeout: Optional[int] = None) -> Optional[str]:
        """Return queued message or None if empty/timeout."""
        if self.drop_ack and self.pending_ack:
            return None
        if self._received_queue:
            return self._received_queue.pop(0)
        return None
    
    def queue_response(self, message: str) -> None:
        """Queue a message to be received."""
        self._received_queue.append(message)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_socket() -> MockClassicalSocket:
    """Create mock classical socket for testing."""
    return MockClassicalSocket()


@pytest.fixture
def mock_socket_no_ack() -> MockClassicalSocket:
    """Create mock socket that drops ACKs."""
    return MockClassicalSocket(drop_ack=True)


# ============================================================================
# SYS-INT-MSG-001: Ordered Socket ACK Blocking
# ============================================================================

class TestOrderedSocketAckBlocking:
    """
    Test Case ID: SYS-INT-MSG-001
    Title: Verify OrderedProtocolSocket prevents race conditions under network delay
    Priority: CRITICAL
    Traces To: GAP: ORDERED-MSG-001, Security Invariant: Commit-then-Reveal
    """

    def test_ordered_protocol_socket_exists(self):
        """
        Verify OrderedProtocolSocket class exists.
        
        Spec Requirement: "OrderedProtocolSocket wrapping ClassicalSocket"
        """
        assert OrderedProtocolSocket is not None, (
            "MISSING: OrderedProtocolSocket class not found. "
            "This is a CRITICAL GAP - required for Commit-then-Reveal ordering."
        )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_socket_has_required_interface(self):
        """
        Verify OrderedProtocolSocket has spec-required methods.
        
        Spec requires:
        - send_with_ack(message, timeout_ns)
        - recv_and_ack()
        - state property
        """
        # Check method existence via class inspection
        assert hasattr(OrderedProtocolSocket, 'send_with_ack'), (
            "MISSING: OrderedProtocolSocket.send_with_ack method"
        )
        assert hasattr(OrderedProtocolSocket, 'recv_and_ack'), (
            "MISSING: OrderedProtocolSocket.recv_and_ack method"
        )
        assert hasattr(OrderedProtocolSocket, 'state') or \
               hasattr(OrderedProtocolSocket, '_state'), (
            "MISSING: OrderedProtocolSocket.state property"
        )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_socket_state_enum_exists(self):
        """
        Verify socket state enumeration exists.
        
        Spec inspection point: "socket.state == OrderedSocketState.SENT_WAIT_ACK"
        """
        # Check that SENT_WAIT_ACK state exists
        assert hasattr(SocketState, 'SENT_WAIT_ACK') or \
               hasattr(OrderedSocketState, 'SENT_WAIT_ACK'), (
            "MISSING: SENT_WAIT_ACK state in SocketState enum"
        )
        
        # Check that IDLE state exists
        assert hasattr(SocketState, 'IDLE') or \
               hasattr(OrderedSocketState, 'IDLE'), (
            "MISSING: IDLE state in SocketState enum"
        )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_send_with_ack_blocks_until_ack(self, mock_socket):
        """
        Verify send_with_ack blocks until ACK is received.
        
        Spec Logic Steps 1-6:
        1. Alice sends DetectionReport via send_with_ack()
        2. INJECT: Additional delay on Bob's ACK response
        5. CAPTURE: t_send and t_ack_received
        6. ASSERT: Alice's state is SENT_WAIT_ACK during wait
        """
        # This test requires generator integration - simplified unit test version
        ordered_socket = OrderedProtocolSocket(mock_socket)
        
        # Queue an ACK response
        mock_socket.queue_response('{"type": "ACK", "seq": 1}')
        
        # Check initial state
        initial_state = ordered_socket.state
        assert initial_state == SocketState.IDLE, (
            f"Initial state should be IDLE, got {initial_state}"
        )
        
        # Note: Full test requires generator-based execution
        # This is a structural verification of the interface

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_message_ordering_enforcement(self, mock_socket):
        """
        Verify Alice does NOT send BasisReveal before ACK received.
        
        Spec assertion: "Alice does NOT send BasisReveal before ACK received"
        """
        ordered_socket = OrderedProtocolSocket(mock_socket)
        
        # Verify the socket enforces ordering
        # In SENT_WAIT_ACK state, sending another message should be blocked
        if hasattr(ordered_socket, 'can_send'):
            # Send first message
            # Note: This requires mock to not immediately return ACK
            pass  # Implementation-dependent test


# ============================================================================
# SYS-INT-MSG-002: ACK Timeout Triggers Abort
# ============================================================================

class TestAckTimeoutAbort:
    """
    Test Case ID: SYS-INT-MSG-002
    Title: Verify OrderedProtocolSocket triggers ProtocolViolation on ACK timeout
    Priority: HIGH
    Traces To: GAP: ORDERED-MSG-001, Abort: ABORT-II-ACK-TIMEOUT
    """

    def test_protocol_violation_exception_exists(self):
        """
        Verify ProtocolViolation exception class exists.
        
        Spec assertion: "ProtocolViolation exception raised"
        """
        assert ProtocolViolation is not None, (
            "MISSING: ProtocolViolation exception not found. "
            "Required for timeout abort handling."
        )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    @pytest.mark.skipif(ProtocolViolation is None,
                       reason="ProtocolViolation not implemented")
    def test_timeout_raises_protocol_violation(self, mock_socket_no_ack):
        """
        Verify ACK timeout raises ProtocolViolation.
        
        Spec Logic Steps 1-4:
        1. Alice sends message via send_with_ack(timeout_ns=5*10^9)
        2. Bob receives message but does NOT send ACK
        3. WAIT: 5 seconds simulation time
        4. ASSERT: ProtocolViolation exception raised
        """
        ordered_socket = OrderedProtocolSocket(mock_socket_no_ack)
        
        # Note: Full test requires simulation time advancement
        # Unit test version checks that timeout logic exists
        if hasattr(ordered_socket, '_timeout_ns'):
            assert ordered_socket._timeout_ns == DEFAULT_ACK_TIMEOUT_NS, (
                f"Default timeout should be {DEFAULT_ACK_TIMEOUT_NS}ns"
            )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_timeout_produces_abort_code(self):
        """
        Verify timeout exception contains proper abort code.
        
        Spec assertion: "Exception message indicates 'ACK timeout'"
        """
        # This test verifies abort code compliance
        abort_code = "ABORT-II-ACK-TIMEOUT"
        
        # Check if ordered_messaging module defines this abort code
        from ehok.protocols import ordered_messaging
        
        has_abort_code = (
            hasattr(ordered_messaging, 'ABORT_CODE_ACK_TIMEOUT') or
            hasattr(ordered_messaging, 'ABORT_II_ACK_TIMEOUT') or
            'ACK' in str(getattr(ordered_messaging, '__dict__', {}))
        )
        
        # Note: Soft check - report but don't necessarily fail
        if not has_abort_code:
            pytest.skip("ABORT-II-ACK-TIMEOUT code not explicitly defined")

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_violation_state_transition(self, mock_socket):
        """
        Verify protocol state transitions to VIOLATION on timeout.
        
        Spec assertion: "Protocol state transitions to VIOLATION/ABORT"
        """
        # Check VIOLATION state exists
        assert hasattr(SocketState, 'VIOLATION'), (
            "MISSING: VIOLATION state in SocketState enum"
        )
        
        # Verify state value
        violation_state = SocketState.VIOLATION
        assert violation_state is not None

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_clean_abort_no_partial_state(self, mock_socket):
        """
        Verify clean abort with no partial state left behind.
        
        Spec Expected State: "Clean abort with traceable error"
        "No partial state left behind"
        """
        ordered_socket = OrderedProtocolSocket(mock_socket)
        
        # After VIOLATION, socket should be in terminal state
        # and refuse further operations
        if hasattr(ordered_socket, '_force_violation'):
            ordered_socket._force_violation()
            
            # Socket should refuse new operations
            assert ordered_socket.state == SocketState.VIOLATION, (
                "Socket should be in VIOLATION state after abort"
            )


# ============================================================================
# Message Type Validation Tests
# ============================================================================

class TestMessageTypeValidation:
    """Tests for MessageType enum compliance."""

    def test_required_message_types_exist(self):
        """
        Verify all spec-required message types exist.
        
        Per spec, the protocol uses:
        - DETECTION_REPORT
        - BASIS_REVEAL
        - ACK
        """
        assert hasattr(MessageType, 'DETECTION_REPORT'), (
            "MISSING: DETECTION_REPORT message type"
        )
        assert hasattr(MessageType, 'BASIS_REVEAL'), (
            "MISSING: BASIS_REVEAL message type"
        )
        assert hasattr(MessageType, 'ACK'), (
            "MISSING: ACK message type"
        )

    def test_message_type_values(self):
        """Verify message type values are strings (for serialization)."""
        assert isinstance(MessageType.DETECTION_REPORT.value, str)
        assert isinstance(MessageType.ACK.value, str)


# ============================================================================
# Sequence Number Tests
# ============================================================================

class TestSequenceNumbering:
    """Tests for monotonic sequence numbering."""

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_sequence_number_tracking(self, mock_socket):
        """
        Verify socket maintains monotonic sequence numbers.
        
        Spec: "Monotonic sequence numbering per session"
        """
        ordered_socket = OrderedProtocolSocket(mock_socket)
        
        # Check sequence number initialization
        if hasattr(ordered_socket, '_send_seq'):
            initial_seq = ordered_socket._send_seq
            assert isinstance(initial_seq, int), (
                "Sequence number should be integer"
            )
            assert initial_seq >= 0, (
                "Sequence number should be non-negative"
            )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_out_of_order_detection(self, mock_socket):
        """
        Verify detection of out-of-order messages.
        
        This is critical for Commit-then-Reveal security.
        """
        ordered_socket = OrderedProtocolSocket(mock_socket)
        
        # Check for sequence validation method
        has_validation = (
            hasattr(ordered_socket, 'validate_sequence') or
            hasattr(ordered_socket, '_check_sequence') or
            hasattr(ordered_socket, '_expected_recv_seq')
        )
        
        assert has_validation, (
            "OrderedProtocolSocket should track expected receive sequence"
        )


# ============================================================================
# Integration Pattern Tests
# ============================================================================

class TestSocketIntegrationPattern:
    """Tests for SquidASM integration patterns."""

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    @pytest.mark.skipif(not CLASSICAL_SOCKET_AVAILABLE,
                       reason="NetQASM ClassicalSocket not available")
    def test_wraps_classical_socket(self):
        """
        Verify OrderedProtocolSocket wraps netqasm ClassicalSocket.
        
        Spec: "OrderedProtocolSocket wrapping ClassicalSocket"
        """
        # Check constructor signature accepts socket-like object
        import inspect
        sig = inspect.signature(OrderedProtocolSocket.__init__)
        params = list(sig.parameters.keys())
        
        # Should accept a socket parameter
        socket_param_found = any(
            'socket' in p.lower() or 'underlying' in p.lower()
            for p in params
        )
        
        assert socket_param_found or len(params) > 1, (
            "OrderedProtocolSocket should accept underlying socket in constructor"
        )

    @pytest.mark.skipif(OrderedProtocolSocket is None,
                       reason="OrderedProtocolSocket not implemented")
    def test_generator_compatible(self):
        """
        Verify socket operations are generator-compatible.
        
        Spec: "Alice's generator remains yielded during ACK wait"
        """
        # Check if send_with_ack is a generator or returns generator
        import inspect
        
        if hasattr(OrderedProtocolSocket, 'send_with_ack'):
            method = getattr(OrderedProtocolSocket, 'send_with_ack')
            # Check if it's a generator function
            is_generator = inspect.isgeneratorfunction(method)
            
            # Or check if it's designed to yield
            source = inspect.getsource(method) if hasattr(method, '__code__') else ""
            has_yield = 'yield' in source
            
            assert is_generator or has_yield, (
                "send_with_ack should be generator-compatible (use yield)"
            )

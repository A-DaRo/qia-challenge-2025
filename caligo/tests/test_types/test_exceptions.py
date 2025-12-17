"""Unit tests for caligo.types.exceptions module."""

import pytest

from caligo.types.exceptions import (
    CaligoError,
    SimulationError,
    TimingViolationError,
    SecurityError,
    QBERThresholdExceeded,
    ProtocolError,
    ContractViolation,
    ConnectionError,
    ConfigurationError,
    ProtocolPhase,
    AbortReason,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_caligo_error_is_base(self):
        """Test that CaligoError is the base for all custom exceptions."""
        assert issubclass(SimulationError, CaligoError)
        assert issubclass(SecurityError, CaligoError)
        assert issubclass(ProtocolError, CaligoError)
        assert issubclass(ConnectionError, CaligoError)
        assert issubclass(ConfigurationError, CaligoError)

    def test_simulation_error_subclasses(self):
        """Test SimulationError subclass hierarchy."""
        assert issubclass(TimingViolationError, SimulationError)

    def test_security_error_subclasses(self):
        """Test SecurityError subclass hierarchy."""
        assert issubclass(QBERThresholdExceeded, SecurityError)

    def test_protocol_error_subclasses(self):
        """Test ProtocolError subclass hierarchy."""
        assert issubclass(ContractViolation, ProtocolError)

    def test_exception_can_be_raised(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(CaligoError):
            raise TimingViolationError("Δt violation")

        with pytest.raises(SecurityError):
            raise QBERThresholdExceeded("QBER > 22%")

        with pytest.raises(ProtocolError):
            raise ContractViolation("POST-Q-001 failed")

    def test_exception_message(self):
        """Test that exception messages are preserved."""
        msg = "Test error message"
        exc = ContractViolation(msg)
        assert str(exc) == msg


class TestProtocolPhase:
    """Tests for ProtocolPhase enumeration."""

    def test_all_phases_defined(self):
        """Test that all expected phases are defined."""
        phases = [p.value for p in ProtocolPhase]
        assert "init" in phases
        assert "quantum" in phases
        assert "sifting" in phases
        assert "reconciliation" in phases
        assert "amplification" in phases
        assert "completed" in phases
        assert "aborted" in phases

    def test_phase_values(self):
        """Test specific phase values."""
        assert ProtocolPhase.INIT.value == "init"
        assert ProtocolPhase.QUANTUM.value == "quantum"
        assert ProtocolPhase.SIFTING.value == "sifting"
        assert ProtocolPhase.RECONCILIATION.value == "reconciliation"
        assert ProtocolPhase.AMPLIFICATION.value == "amplification"
        assert ProtocolPhase.COMPLETED.value == "completed"
        assert ProtocolPhase.ABORTED.value == "aborted"


class TestAbortReason:
    """Tests for AbortReason enumeration."""

    def test_phase_i_abort_reasons(self):
        """Test Phase I abort reasons have correct prefix."""
        assert AbortReason.FEASIBILITY_HARD_LIMIT.value.startswith("ABORT-I-")
        assert AbortReason.TIMING_VIOLATION.value.startswith("ABORT-I-")

    def test_phase_ii_abort_reasons(self):
        """Test Phase II abort reasons have correct prefix."""
        assert AbortReason.DETECTION_ANOMALY.value.startswith("ABORT-II-")
        assert AbortReason.QBER_HARD_LIMIT.value.startswith("ABORT-II-")
        assert AbortReason.MISSING_ROUNDS_INVALID.value.startswith("ABORT-II-")

    def test_phase_iii_abort_reasons(self):
        """Test Phase III abort reasons have correct prefix."""
        assert AbortReason.LEAKAGE_CAP_EXCEEDED.value.startswith("ABORT-III-")
        assert AbortReason.RECONCILIATION_FAILED.value.startswith("ABORT-III-")
        assert AbortReason.VERIFICATION_FAILED.value.startswith("ABORT-III-")

    def test_phase_iv_abort_reasons(self):
        """Test Phase IV abort reasons have correct prefix."""
        assert AbortReason.ENTROPY_DEPLETED.value.startswith("ABORT-IV-")
        assert AbortReason.KEY_LENGTH_ZERO.value.startswith("ABORT-IV-")

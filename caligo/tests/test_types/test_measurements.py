"""Unit tests for caligo.types.measurements module."""

import pytest

from caligo.types.measurements import (
    MeasurementRecord,
    RoundResult,
    DetectionEvent,
)
from caligo.types.exceptions import ContractViolation


class TestMeasurementRecord:
    """Tests for MeasurementRecord dataclass."""

    def test_valid_record(self):
        """Test creating a valid MeasurementRecord."""
        record = MeasurementRecord(
            round_id=0,
            outcome=1,
            basis=0,
            timestamp_ns=1000.0,
        )
        assert record.round_id == 0
        assert record.outcome == 1
        assert record.basis == 0
        assert record.timestamp_ns == 1000.0
        assert record.detected is True  # Default

    def test_inv_meas_001_invalid_outcome(self):
        """INV-MEAS-001: outcome must be in {0, 1}."""
        with pytest.raises(ContractViolation, match="INV-MEAS-001"):
            MeasurementRecord(round_id=0, outcome=2, basis=0, timestamp_ns=0.0)

    def test_inv_meas_002_invalid_basis(self):
        """INV-MEAS-002: basis must be in {0, 1}."""
        with pytest.raises(ContractViolation, match="INV-MEAS-002"):
            MeasurementRecord(round_id=0, outcome=0, basis=2, timestamp_ns=0.0)

    def test_inv_meas_003_negative_round_id(self):
        """INV-MEAS-003: round_id must be >= 0."""
        with pytest.raises(ContractViolation, match="INV-MEAS-003"):
            MeasurementRecord(round_id=-1, outcome=0, basis=0, timestamp_ns=0.0)

    def test_inv_meas_004_negative_timestamp(self):
        """INV-MEAS-004: timestamp_ns must be >= 0."""
        with pytest.raises(ContractViolation, match="INV-MEAS-004"):
            MeasurementRecord(round_id=0, outcome=0, basis=0, timestamp_ns=-1.0)

    def test_detected_false(self):
        """Test record with detected=False."""
        record = MeasurementRecord(
            round_id=0, outcome=0, basis=0, timestamp_ns=0.0, detected=False
        )
        assert record.detected is False


class TestRoundResult:
    """Tests for RoundResult dataclass."""

    def test_valid_round_result(self):
        """Test creating a valid RoundResult."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
        )
        assert result.round_id == 0
        assert result.alice_outcome == 1
        assert result.bob_outcome == 1

    def test_is_valid_both_detected(self):
        """Test is_valid property when both parties detect."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
            alice_detected=True,
            bob_detected=True,
        )
        assert result.is_valid is True

    def test_is_valid_alice_not_detected(self):
        """Test is_valid property when Alice doesn't detect."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
            alice_detected=False,
            bob_detected=True,
        )
        assert result.is_valid is False

    def test_bases_match_true(self):
        """Test bases_match when bases are equal."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=0,
            alice_basis=1,
            bob_basis=1,
        )
        assert result.bases_match is True

    def test_bases_match_false(self):
        """Test bases_match when bases differ."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=0,
            alice_basis=0,
            bob_basis=1,
        )
        assert result.bases_match is False

    def test_outcomes_match_true(self):
        """Test outcomes_match when outcomes are equal."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
        )
        assert result.outcomes_match is True

    def test_outcomes_match_false(self):
        """Test outcomes_match when outcomes differ."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=0,
            alice_basis=0,
            bob_basis=0,
        )
        assert result.outcomes_match is False

    def test_contributes_to_sifted_key_true(self):
        """Test contributes_to_sifted_key when valid and bases match."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
            alice_detected=True,
            bob_detected=True,
        )
        assert result.contributes_to_sifted_key is True

    def test_contributes_to_sifted_key_false_bases_differ(self):
        """Test contributes_to_sifted_key when bases differ."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=1,
            alice_detected=True,
            bob_detected=True,
        )
        assert result.contributes_to_sifted_key is False

    def test_has_error_true(self):
        """Test has_error when bases match but outcomes differ."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=0,
            alice_basis=0,
            bob_basis=0,
        )
        assert result.has_error is True

    def test_has_error_false_outcomes_match(self):
        """Test has_error when outcomes match."""
        result = RoundResult(
            round_id=0,
            alice_outcome=1,
            bob_outcome=1,
            alice_basis=0,
            bob_basis=0,
        )
        assert result.has_error is False

    def test_invalid_alice_outcome(self):
        """Test rejection of invalid alice_outcome."""
        with pytest.raises(ContractViolation, match="alice_outcome"):
            RoundResult(
                round_id=0,
                alice_outcome=2,
                bob_outcome=0,
                alice_basis=0,
                bob_basis=0,
            )


class TestDetectionEvent:
    """Tests for DetectionEvent dataclass."""

    def test_valid_detection_event(self):
        """Test creating a valid DetectionEvent."""
        event = DetectionEvent(round_id=0, detected=True, timestamp_ns=1000.0)
        assert event.round_id == 0
        assert event.detected is True
        assert event.timestamp_ns == 1000.0

    def test_invalid_round_id(self):
        """Test rejection of negative round_id."""
        with pytest.raises(ContractViolation, match="round_id"):
            DetectionEvent(round_id=-1, detected=True, timestamp_ns=0.0)

    def test_invalid_timestamp(self):
        """Test rejection of negative timestamp."""
        with pytest.raises(ContractViolation, match="timestamp_ns"):
            DetectionEvent(round_id=0, detected=True, timestamp_ns=-1.0)

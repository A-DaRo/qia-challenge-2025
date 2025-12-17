"""Unit tests for caligo.types.phase_contracts module."""

import pytest
import numpy as np
from bitarray import bitarray

from caligo.types.phase_contracts import (
    QuantumPhaseResult,
    SiftingPhaseResult,
    ReconciliationPhaseResult,
    AmplificationPhaseResult,
    ObliviousTransferOutput,
    QBER_HARD_LIMIT,
)
from caligo.types.keys import AliceObliviousKey, BobObliviousKey
from caligo.types.exceptions import ContractViolation


class TestQuantumPhaseResult:
    """Tests for QuantumPhaseResult contract."""

    def test_valid_quantum_result(self, sample_quantum_phase_result):
        """Test valid QuantumPhaseResult passes all checks."""
        # Should not raise
        assert sample_quantum_phase_result.num_pairs_generated == 1000

    def test_post_q_001_outcomes_length_mismatch(self):
        """POST-Q-001: measurement_outcomes length must match num_pairs_generated."""
        with pytest.raises(ContractViolation, match="POST-Q-001"):
            QuantumPhaseResult(
                measurement_outcomes=np.array([0, 1, 0], dtype=np.uint8),
                basis_choices=np.array([0, 1, 0, 1, 0], dtype=np.uint8),
                round_ids=np.arange(5, dtype=np.int64),
                generation_timestamp=1000.0,
                num_pairs_requested=5,
                num_pairs_generated=5,  # But outcomes only has 3
            )

    def test_post_q_002_bases_length_mismatch(self):
        """POST-Q-002: basis_choices length must match num_pairs_generated."""
        with pytest.raises(ContractViolation, match="POST-Q-002"):
            QuantumPhaseResult(
                measurement_outcomes=np.array([0, 1, 0, 1, 0], dtype=np.uint8),
                basis_choices=np.array([0, 1], dtype=np.uint8),  # Only 2
                round_ids=np.arange(5, dtype=np.int64),
                generation_timestamp=1000.0,
                num_pairs_requested=5,
                num_pairs_generated=5,
            )

    def test_post_q_003_invalid_outcomes(self):
        """POST-Q-003: All outcomes must be in {0, 1}."""
        with pytest.raises(ContractViolation, match="POST-Q-003"):
            QuantumPhaseResult(
                measurement_outcomes=np.array([0, 1, 2], dtype=np.uint8),  # 2 invalid
                basis_choices=np.array([0, 1, 0], dtype=np.uint8),
                round_ids=np.arange(3, dtype=np.int64),
                generation_timestamp=1000.0,
                num_pairs_requested=3,
                num_pairs_generated=3,
            )

    def test_post_q_004_invalid_bases(self):
        """POST-Q-004: All bases must be in {0, 1}."""
        with pytest.raises(ContractViolation, match="POST-Q-004"):
            QuantumPhaseResult(
                measurement_outcomes=np.array([0, 1, 0], dtype=np.uint8),
                basis_choices=np.array([0, 1, 3], dtype=np.uint8),  # 3 invalid
                round_ids=np.arange(3, dtype=np.int64),
                generation_timestamp=1000.0,
                num_pairs_requested=3,
                num_pairs_generated=3,
            )

    def test_empty_result_valid(self):
        """Test that empty result (0 pairs) is valid."""
        result = QuantumPhaseResult(
            measurement_outcomes=np.array([], dtype=np.uint8),
            basis_choices=np.array([], dtype=np.uint8),
            round_ids=np.array([], dtype=np.int64),
            generation_timestamp=1000.0,
            num_pairs_requested=100,
            num_pairs_generated=0,  # All lost
        )
        assert result.num_pairs_generated == 0


class TestSiftingPhaseResult:
    """Tests for SiftingPhaseResult contract."""

    def test_valid_sifting_result(self, sample_sifting_result):
        """Test valid SiftingPhaseResult passes all checks."""
        assert sample_sifting_result.qber_adjusted <= QBER_HARD_LIMIT

    def test_post_s_001_key_length_mismatch(self):
        """POST-S-001: Alice and Bob sifted keys must have same length."""
        with pytest.raises(ContractViolation, match="POST-S-001"):
            SiftingPhaseResult(
                sifted_key_alice=bitarray("10101010"),  # 8 bits
                sifted_key_bob=bitarray("1010"),  # 4 bits
                matching_indices=np.array([0, 1, 2, 3], dtype=np.int64),
                i0_indices=np.array([0, 1], dtype=np.int64),
                i1_indices=np.array([2, 3], dtype=np.int64),
                test_set_indices=np.array([0], dtype=np.int64),
                qber_estimate=0.05,
                qber_adjusted=0.06,
                finite_size_penalty=0.01,
                test_set_size=1,
            )

    def test_post_s_002_qber_adjusted_calculation(self):
        """POST-S-002: qber_adjusted must equal qber_estimate + penalty."""
        with pytest.raises(ContractViolation, match="POST-S-002"):
            SiftingPhaseResult(
                sifted_key_alice=bitarray("1010"),
                sifted_key_bob=bitarray("1010"),
                matching_indices=np.array([0, 1, 2, 3], dtype=np.int64),
                i0_indices=np.array([0, 1], dtype=np.int64),
                i1_indices=np.array([2, 3], dtype=np.int64),
                test_set_indices=np.array([0], dtype=np.int64),
                qber_estimate=0.05,
                qber_adjusted=0.10,  # Should be 0.06
                finite_size_penalty=0.01,
                test_set_size=1,
            )

    def test_post_s_003_qber_exceeds_limit(self):
        """POST-S-003: qber_adjusted must not exceed hard limit."""
        with pytest.raises(ContractViolation, match="POST-S-003"):
            SiftingPhaseResult(
                sifted_key_alice=bitarray("1010"),
                sifted_key_bob=bitarray("1010"),
                matching_indices=np.array([0, 1, 2, 3], dtype=np.int64),
                i0_indices=np.array([0, 1], dtype=np.int64),
                i1_indices=np.array([2, 3], dtype=np.int64),
                test_set_indices=np.array([0], dtype=np.int64),
                qber_estimate=0.20,
                qber_adjusted=0.25,  # Above 22%
                finite_size_penalty=0.05,
                test_set_size=1,
            )


class TestReconciliationPhaseResult:
    """Tests for ReconciliationPhaseResult contract."""

    def test_valid_reconciliation_result(self, sample_reconciliation_result):
        """Test valid ReconciliationPhaseResult passes all checks."""
        assert sample_reconciliation_result.hash_verified is True

    def test_post_r_002_hash_not_verified(self):
        """POST-R-002: hash_verified must be True."""
        with pytest.raises(ContractViolation, match="POST-R-002"):
            ReconciliationPhaseResult(
                reconciled_key=bitarray("10101010"),
                num_blocks=1,
                blocks_succeeded=0,
                blocks_failed=1,
                total_syndrome_bits=10,
                effective_rate=0.5,
                hash_verified=False,  # Failure
            )


class TestAmplificationPhaseResult:
    """Tests for AmplificationPhaseResult contract."""

    def test_valid_amplification_result(self, sample_amplification_result):
        """Test valid AmplificationPhaseResult passes all checks."""
        assert sample_amplification_result.key_length > 0

    def test_post_amp_001_zero_key_length(self, sample_alice_key):
        """POST-AMP-001: key_length must be > 0."""
        with pytest.raises(ContractViolation, match="POST-AMP-001"):
            AmplificationPhaseResult(
                oblivious_key=sample_alice_key,
                qber=0.06,
                key_length=0,  # Invalid
                entropy_consumed=16.0,
                entropy_rate=0.0,
            )

    def test_post_amp_002_insufficient_entropy(self, sample_alice_key):
        """POST-AMP-002: entropy_consumed must be >= key_length."""
        with pytest.raises(ContractViolation, match="POST-AMP-002"):
            AmplificationPhaseResult(
                oblivious_key=sample_alice_key,
                qber=0.06,
                key_length=100,
                entropy_consumed=50.0,  # Less than key_length
                entropy_rate=0.1,
            )


class TestObliviousTransferOutput:
    """Tests for ObliviousTransferOutput contract."""

    def test_valid_ot_output(self, sample_ot_output):
        """Test valid ObliviousTransferOutput passes all checks."""
        assert sample_ot_output.protocol_succeeded is True

    def test_post_ot_001_alice_key_length_mismatch(self):
        """POST-OT-001: Alice key lengths must match final_key_length."""
        alice = AliceObliviousKey(
            s0=bitarray("1010"),  # 4 bits
            s1=bitarray("0101"),  # 4 bits
            key_length=4,
        )
        bob = BobObliviousKey(
            sc=bitarray("1010"),  # Must match s0
            choice_bit=0,
            key_length=4,
        )
        with pytest.raises(ContractViolation, match="POST-OT-001"):
            ObliviousTransferOutput(
                alice_key=alice,
                bob_key=bob,
                protocol_succeeded=True,
                total_rounds=100,
                final_key_length=8,  # Says 8 but keys are 4
            )

    def test_post_ot_003_bob_key_mismatch(self):
        """POST-OT-003: Bob's key must match Alice's key at choice_bit."""
        alice = AliceObliviousKey(
            s0=bitarray("10101010"),
            s1=bitarray("01010101"),
            key_length=8,
        )
        bob = BobObliviousKey(
            sc=bitarray("11111111"),  # Doesn't match either s0 or s1
            choice_bit=0,
            key_length=8,
        )
        with pytest.raises(ContractViolation, match="POST-OT-003"):
            ObliviousTransferOutput(
                alice_key=alice,
                bob_key=bob,
                protocol_succeeded=True,
                total_rounds=100,
                final_key_length=8,
            )

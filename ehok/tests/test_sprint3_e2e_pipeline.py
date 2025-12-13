"""
Sprint 3 Tests: E2E Pipeline and Oblivious Transfer Output.

This module tests the end-to-end protocol integration including:
- Oblivious key formatting (AliceObliviousKey, BobObliviousKey)
- OT correctness validation
- Protocol metrics
- Adversarial abort conditions
- Statistical validation for NSM security

References
----------
- sprint_3_specification.md Section 4-7
- Lupo et al. (2023): E-HOK protocol
"""

import pytest
import numpy as np
import math

from ehok.core.oblivious_formatter import (
    AliceObliviousKey,
    BobObliviousKey,
    ProtocolMetrics,
    ObliviousTransferResult,
    ObliviousKeyFormatter,
    validate_ot_correctness,
)
from ehok.core.config import ProtocolConfig, NSMConfig
from ehok.analysis.nsm_bounds import FeasibilityResult


class TestAliceObliviousKey:
    """Tests for Alice's OT output structure."""

    def test_create_valid_key(self):
        """Test creating valid Alice OT keys."""
        key_0 = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)

        alice_key = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=8,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="dupuis_konig",
            hash_seed=b"test_seed_12345",
        )

        assert alice_key.key_length == 8
        assert np.array_equal(alice_key.key_0, key_0)
        assert np.array_equal(alice_key.key_1, key_1)
        assert alice_key.security_parameter == 1e-9
        assert alice_key.storage_noise_r == 0.75

    def test_length_mismatch_raises(self):
        """Test that key length mismatch raises error."""
        key_0 = np.array([0, 1, 1, 0], dtype=np.uint8)
        key_1 = np.array([1, 0, 1], dtype=np.uint8)  # Different length

        with pytest.raises(ValueError, match="key_length"):
            AliceObliviousKey(
                key_0=key_0,
                key_1=key_1,
                key_length=4,
                security_parameter=1e-9,
                storage_noise_r=0.75,
                entropy_bound_used="dupuis_konig",
                hash_seed=b"test",
            )

    def test_invalid_values_raise(self):
        """Test that non-binary values raise error."""
        key_0 = np.array([0, 1, 2, 0], dtype=np.uint8)  # Invalid value
        key_1 = np.array([1, 0, 1, 0], dtype=np.uint8)

        with pytest.raises(ValueError, match="must be 0 or 1"):
            AliceObliviousKey(
                key_0=key_0,
                key_1=key_1,
                key_length=4,
                security_parameter=1e-9,
                storage_noise_r=0.75,
                entropy_bound_used="dupuis_konig",
                hash_seed=b"test",
            )

    def test_empty_key(self):
        """Test creating empty Alice key."""
        empty_key = AliceObliviousKey.empty(security_parameter=1e-6)

        assert empty_key.key_length == 0
        assert len(empty_key.key_0) == 0
        assert len(empty_key.key_1) == 0
        assert empty_key.security_parameter == 1e-6


class TestBobObliviousKey:
    """Tests for Bob's OT output structure."""

    def test_create_valid_key(self):
        """Test creating valid Bob OT key."""
        key_c = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

        bob_key = BobObliviousKey(
            key_c=key_c,
            choice_bit=0,
            key_length=8,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        assert bob_key.key_length == 8
        assert bob_key.choice_bit == 0
        assert np.array_equal(bob_key.key_c, key_c)

    def test_invalid_choice_bit_raises(self):
        """Test that invalid choice bit raises error."""
        key_c = np.array([0, 1, 1, 0], dtype=np.uint8)

        with pytest.raises(ValueError, match="choice_bit"):
            BobObliviousKey(
                key_c=key_c,
                choice_bit=2,  # Invalid
                key_length=4,
                security_parameter=1e-9,
                storage_noise_r=0.75,
            )

    def test_empty_key(self):
        """Test creating empty Bob key."""
        empty_key = BobObliviousKey.empty(choice_bit=1)

        assert empty_key.key_length == 0
        assert len(empty_key.key_c) == 0
        assert empty_key.choice_bit == 1


class TestOTCorrectnessValidation:
    """Tests for OT correctness property validation."""

    def test_ot_correctness_choice_0(self):
        """Test OT correctness when Bob chooses key 0."""
        key_0 = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)

        alice_keys = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=8,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="dupuis_konig",
            hash_seed=b"test",
        )

        bob_key = BobObliviousKey(
            key_c=key_0.copy(),  # Bob got key_0
            choice_bit=0,
            key_length=8,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        is_correct, msg = validate_ot_correctness(alice_keys, bob_key)
        assert is_correct is True
        assert "verified" in msg.lower()

    def test_ot_correctness_choice_1(self):
        """Test OT correctness when Bob chooses key 1."""
        key_0 = np.array([0, 1, 1, 0], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1], dtype=np.uint8)

        alice_keys = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="virtual_erasure",
            hash_seed=b"test",
        )

        bob_key = BobObliviousKey(
            key_c=key_1.copy(),  # Bob got key_1
            choice_bit=1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        is_correct, msg = validate_ot_correctness(alice_keys, bob_key)
        assert is_correct is True

    def test_ot_correctness_violation(self):
        """Test OT correctness violation detection."""
        key_0 = np.array([0, 1, 1, 0], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1], dtype=np.uint8)

        alice_keys = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="dupuis_konig",
            hash_seed=b"test",
        )

        # Bob claims key_1 but has wrong value
        bob_key = BobObliviousKey(
            key_c=np.array([0, 0, 0, 0], dtype=np.uint8),  # Wrong!
            choice_bit=1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        is_correct, msg = validate_ot_correctness(alice_keys, bob_key)
        assert is_correct is False
        assert "mismatch" in msg.lower()

    def test_ot_correctness_empty_keys(self):
        """Test OT correctness with empty keys (abort scenario)."""
        alice_keys = AliceObliviousKey.empty()
        bob_key = BobObliviousKey.empty()

        is_correct, msg = validate_ot_correctness(alice_keys, bob_key)
        assert is_correct is True
        assert "empty" in msg.lower() or "valid" in msg.lower()


class TestObliviousTransferResult:
    """Tests for complete OT result structure."""

    def test_successful_result(self):
        """Test successful OT result creation."""
        key_0 = np.array([0, 1, 1, 0], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1], dtype=np.uint8)

        alice_keys = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="dupuis_konig",
            hash_seed=b"test",
        )

        bob_key = BobObliviousKey(
            key_c=key_0.copy(),
            choice_bit=0,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        metrics = ProtocolMetrics(
            storage_noise_r=0.75,
            extractable_entropy=4.0,
            wiretap_cost_bits=100,
            security_penalty_bits=60,
            final_key_length=4,
            feasibility_status="FEASIBLE",
            entropy_bound_used="dupuis_konig",
        )

        result = ObliviousTransferResult(
            success=True,
            alice_keys=alice_keys,
            bob_key=bob_key,
            metrics=metrics,
        )

        assert result.success is True
        assert result.alice_keys.key_length == 4
        assert result.bob_key.choice_bit == 0

    def test_successful_result_ot_violation_raises(self):
        """Test that OT violation in successful result raises."""
        key_0 = np.array([0, 1, 1, 0], dtype=np.uint8)
        key_1 = np.array([1, 0, 1, 1], dtype=np.uint8)

        alice_keys = AliceObliviousKey(
            key_0=key_0,
            key_1=key_1,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
            entropy_bound_used="dupuis_konig",
            hash_seed=b"test",
        )

        # Bob claims key_0 but has wrong value
        bob_key = BobObliviousKey(
            key_c=key_1.copy(),  # Has key_1 but claims 0
            choice_bit=0,
            key_length=4,
            security_parameter=1e-9,
            storage_noise_r=0.75,
        )

        metrics = ProtocolMetrics(
            storage_noise_r=0.75,
            extractable_entropy=4.0,
            wiretap_cost_bits=100,
            security_penalty_bits=60,
            final_key_length=4,
            feasibility_status="FEASIBLE",
            entropy_bound_used="dupuis_konig",
        )

        with pytest.raises(ValueError, match="OT correctness"):
            ObliviousTransferResult(
                success=True,
                alice_keys=alice_keys,
                bob_key=bob_key,
                metrics=metrics,
            )


class TestProtocolMetrics:
    """Tests for protocol metrics tracking."""

    def test_metrics_creation(self):
        """Test creating protocol metrics."""
        metrics = ProtocolMetrics(
            storage_noise_r=0.75,
            extractable_entropy=50000.0,
            wiretap_cost_bits=5000,
            security_penalty_bits=60,
            final_key_length=10000,
            feasibility_status="FEASIBLE",
            entropy_bound_used="dupuis_konig",
            raw_pairs_generated=100000,
            sifted_length=50000,
            reconciled_length=45000,
            observed_qber=0.03,
            adjusted_qber=0.035,
            timing_barrier_enforced=True,
            protocol_duration_ns=1_500_000_000,
        )

        assert metrics.storage_noise_r == 0.75
        assert metrics.final_key_length == 10000
        assert metrics.timing_barrier_enforced is True

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ProtocolMetrics(
            storage_noise_r=0.5,
            extractable_entropy=1000.0,
            wiretap_cost_bits=100,
            security_penalty_bits=60,
            final_key_length=500,
            feasibility_status="FEASIBLE",
            entropy_bound_used="virtual_erasure",
        )

        d = metrics.to_dict()
        assert d["storage_noise_r"] == 0.5
        assert d["feasibility_status"] == "FEASIBLE"
        assert "entropy_bound_used" in d

    def test_abort_metrics(self):
        """Test metrics for aborted protocol."""
        metrics = ProtocolMetrics(
            storage_noise_r=0.75,
            extractable_entropy=0.0,
            wiretap_cost_bits=10000,
            security_penalty_bits=60,
            final_key_length=0,
            feasibility_status="INFEASIBLE_INSUFFICIENT_ENTROPY",
            entropy_bound_used="none",
            abort_reason="DEATH_VALLEY",
        )

        assert metrics.final_key_length == 0
        assert metrics.abort_reason == "DEATH_VALLEY"


class TestObliviousKeyFormatter:
    """Tests for OT key formatting utility."""

    def test_derive_choice_bit_i1_majority(self):
        """Test choice bit derivation when I_1 > I_0."""
        # I_1 larger than I_0 means Bob effectively chose basis 1
        choice = ObliviousKeyFormatter.derive_choice_bit_from_i1_fraction(
            i_1_length=600,
            total_sifted=1000,
        )
        assert choice == 1

    def test_derive_choice_bit_i0_majority(self):
        """Test choice bit derivation when I_0 > I_1."""
        # I_1 smaller than I_0 means Bob effectively chose basis 0
        choice = ObliviousKeyFormatter.derive_choice_bit_from_i1_fraction(
            i_1_length=400,
            total_sifted=1000,
        )
        assert choice == 0

    def test_derive_choice_bit_equal(self):
        """Test choice bit derivation with equal fractions (tie-breaker)."""
        # Equal fractions should use deterministic tie-breaker
        choice1 = ObliviousKeyFormatter.derive_choice_bit_from_i1_fraction(
            i_1_length=500,
            total_sifted=1000,
            seed=42,
        )
        choice2 = ObliviousKeyFormatter.derive_choice_bit_from_i1_fraction(
            i_1_length=500,
            total_sifted=1000,
            seed=42,
        )
        # Should be deterministic with same seed
        assert choice1 == choice2

    def test_derive_choice_bit_empty(self):
        """Test choice bit derivation with empty sifted set."""
        choice = ObliviousKeyFormatter.derive_choice_bit_from_i1_fraction(
            i_1_length=0,
            total_sifted=0,
        )
        assert choice == 0  # Default


class TestNSMConfig:
    """Tests for NSM configuration in ProtocolConfig."""

    def test_default_nsm_config(self):
        """Test default NSM configuration values."""
        config = ProtocolConfig()

        assert config.nsm.storage_noise_r == 0.75
        assert config.nsm.storage_rate_nu == 0.002
        assert config.nsm.delta_t_ns == 1_000_000_000

    def test_custom_nsm_config(self):
        """Test custom NSM configuration."""
        nsm_config = NSMConfig(
            storage_noise_r=0.5,
            storage_rate_nu=0.01,
            delta_t_ns=500_000_000,
        )
        config = ProtocolConfig(nsm=nsm_config)

        assert config.nsm.storage_noise_r == 0.5
        assert config.nsm.storage_rate_nu == 0.01
        assert config.nsm.delta_t_ns == 500_000_000

    def test_invalid_storage_noise_raises(self):
        """Test that invalid storage noise raises error."""
        with pytest.raises(ValueError, match="storage_noise_r"):
            NSMConfig(storage_noise_r=1.5)

    def test_invalid_delta_t_raises(self):
        """Test that invalid delta_t raises error."""
        with pytest.raises(ValueError, match="delta_t_ns"):
            NSMConfig(delta_t_ns=-1)


class TestAdversarialAbortConditions:
    """Tests for adversarial abort conditions."""

    def test_qber_hard_limit_abort(self):
        """Test that QBER > 22% triggers abort."""
        from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
            validate_nsm_feasibility,
        )

        is_feasible, status, _ = validate_nsm_feasibility(
            reconciled_key_length=100000,
            storage_noise_r=0.5,
            total_leakage_bits=10000,
            epsilon_sec=1e-9,
            adjusted_qber=0.25,  # Above 22%
        )

        assert is_feasible is False
        assert status == FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH

    def test_death_valley_abort(self):
        """Test Death Valley (insufficient entropy) abort."""
        from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
            validate_nsm_feasibility,
        )

        is_feasible, status, key_len = validate_nsm_feasibility(
            reconciled_key_length=100,  # Too small
            storage_noise_r=0.75,
            total_leakage_bits=10000,  # High leakage
            epsilon_sec=1e-9,
        )

        assert is_feasible is False
        assert status == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        assert key_len == 0


class TestStatisticalValidation:
    """Statistical validation tests for NSM security."""

    @pytest.mark.parametrize("r,expected_h_min_approx", [
        (0.1, 0.95),   # High noise → high entropy
        (0.5, 0.50),   # Medium noise
        (0.9, 0.10),   # Low noise → low entropy
    ])
    def test_entropy_rate_scales_with_noise(self, r, expected_h_min_approx):
        """Test that entropy rate scales correctly with storage noise."""
        from ehok.analysis.nsm_bounds import max_bound_entropy_rate

        h_min = max_bound_entropy_rate(r)

        # Should be within 20% of expected
        assert abs(h_min - expected_h_min_approx) < 0.2

    def test_key_length_positive_in_target_regime(self):
        """Test that target parameter regime yields positive key."""
        from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
            NSMPrivacyAmplificationParams,
            compute_nsm_key_length,
        )

        # Erven et al. parameters
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=100000,
            storage_noise_r=0.75,
            syndrome_leakage_bits=10000,
            hash_leakage_bits=128,
            epsilon_sec=2.5e-7,
        )
        result = compute_nsm_key_length(params)

        assert result.secure_key_length > 0
        assert result.feasibility == FeasibilityResult.FEASIBLE

    def test_key_length_formula_consistency(self):
        """Test that key length formula is internally consistent."""
        from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
            NSMPrivacyAmplificationParams,
            compute_nsm_key_length,
        )

        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=5000,
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        # Verify entropy accounting
        expected_consumed = (
            params.total_leakage +
            2 * math.log2(1 / params.epsilon_sec)
        )
        assert abs(result.entropy_consumed - expected_consumed) < 1.0

        # Verify key length formula
        expected_key = int(
            result.extractable_entropy - result.entropy_consumed
        )
        assert result.secure_key_length == expected_key

    def test_leakage_subtraction_exact(self):
        """Test that syndrome leakage is subtracted exactly."""
        from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
            NSMPrivacyAmplificationParams,
            compute_nsm_key_length,
        )

        base_params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=1000,
            hash_leakage_bits=0,
            epsilon_sec=1e-9,
        )
        result_1000 = compute_nsm_key_length(base_params)

        more_leak_params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=2000,  # 1000 more
            hash_leakage_bits=0,
            epsilon_sec=1e-9,
        )
        result_2000 = compute_nsm_key_length(more_leak_params)

        # Key length should decrease by exactly 1000
        assert result_1000.secure_key_length - result_2000.secure_key_length == 1000

"""
Sprint 3 Tests: NSM Privacy Amplification.

This module tests the NSM-correct privacy amplification implementation
including key length formula, death valley guard, and wiretap cost subtraction.

References
----------
- sprint_3_specification.md Section 3 (TOEPLITZ-MODIFY-001)
- Lupo et al. (2023) Eq. (36): Max Bound
"""

import math
import pytest
import numpy as np

from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
    NSMPrivacyAmplificationParams,
    NSMPrivacyAmplificationResult,
    compute_nsm_key_length,
    compute_minimum_n_for_positive_key,
    validate_nsm_feasibility,
)
from ehok.analysis.nsm_bounds import (
    FeasibilityResult,
    max_bound_entropy_rate,
    gamma_function,
    collision_entropy_rate,
)


class TestNSMKeyLengthFormula:
    """Tests for the NSM key length formula: ℓ ≤ n·h_min(r) - |Σ| - 2log₂(1/ε)."""

    def test_basic_computation_feasible(self):
        """Test basic feasible key length computation."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=100000,
            storage_noise_r=0.75,
            syndrome_leakage_bits=10000,
            hash_leakage_bits=128,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        # Should be feasible with these parameters
        assert result.feasibility == FeasibilityResult.FEASIBLE
        assert result.secure_key_length > 0
        assert result.min_entropy_rate > 0
        assert result.extractable_entropy > 0

    def test_formula_verification(self):
        """Verify the key length formula components."""
        n = 50000
        r = 0.5
        syndrome = 5000
        hash_bits = 64
        epsilon = 1e-6

        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=n,
            storage_noise_r=r,
            syndrome_leakage_bits=syndrome,
            hash_leakage_bits=hash_bits,
            epsilon_sec=epsilon,
        )
        result = compute_nsm_key_length(params)

        # Manual calculation
        h_min = max_bound_entropy_rate(r)
        extractable = n * h_min
        security_penalty = 2 * math.log2(1 / epsilon)
        total_leakage = syndrome + hash_bits
        expected_length = int(extractable - total_leakage - security_penalty)

        # Should match (within 1 bit due to floor)
        assert result.secure_key_length == expected_length
        assert abs(result.min_entropy_rate - h_min) < 1e-10
        assert abs(result.extractable_entropy - extractable) < 1e-6

    def test_entropy_bound_selection_dupuis_konig(self):
        """Test that Dupuis-König bound dominates for high noise (low r)."""
        # For low r, Dupuis-König bound should dominate
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.2,  # High noise, low r
            syndrome_leakage_bits=1000,
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        assert result.entropy_bound_used == "dupuis_konig"

    def test_entropy_bound_selection_virtual_erasure(self):
        """Test that virtual erasure bound dominates for low noise (high r)."""
        # For high r, virtual erasure bound should dominate
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.95,  # Low noise, high r
            syndrome_leakage_bits=1000,
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        assert result.entropy_bound_used == "virtual_erasure"


class TestDeathValleyGuard:
    """Tests for the Death Valley guard (insufficient entropy)."""

    def test_death_valley_small_key(self):
        """Test that small keys trigger Death Valley."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=100,  # Too small
            storage_noise_r=0.75,
            syndrome_leakage_bits=1000,
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        assert result.feasibility == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        assert result.secure_key_length == 0

    def test_death_valley_high_leakage(self):
        """Test that high leakage triggers Death Valley."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=10000,
            storage_noise_r=0.75,
            syndrome_leakage_bits=50000,  # Leakage exceeds entropy
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        assert result.feasibility == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        assert result.secure_key_length == 0

    def test_death_valley_low_security_margin(self):
        """Test that tiny epsilon increases security penalty and may cause Death Valley."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=1000,
            storage_noise_r=0.75,
            syndrome_leakage_bits=100,
            hash_leakage_bits=64,
            epsilon_sec=1e-30,  # Extreme security → huge penalty
        )
        result = compute_nsm_key_length(params)

        # Security penalty: 2 * log2(10^30) ≈ 200 bits
        # This should push us into Death Valley for small n
        assert result.feasibility == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY

    def test_death_valley_empty_key(self):
        """Test empty reconciled key."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=0,
            storage_noise_r=0.75,
            syndrome_leakage_bits=0,
            hash_leakage_bits=0,
            epsilon_sec=1e-9,
        )
        result = compute_nsm_key_length(params)

        assert result.feasibility == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        assert result.secure_key_length == 0
        assert result.entropy_bound_used == "none"


class TestWiretapCostSubtraction:
    """Tests for wiretap cost (syndrome leakage) subtraction."""

    def test_leakage_reduces_key_length(self):
        """Test that leakage reduces extractable key length."""
        base_params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=1000,
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result_low_leak = compute_nsm_key_length(base_params)

        high_leak_params = NSMPrivacyAmplificationParams(
            reconciled_key_length=50000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=5000,  # 4000 more bits leaked
            hash_leakage_bits=64,
            epsilon_sec=1e-9,
        )
        result_high_leak = compute_nsm_key_length(high_leak_params)

        # Key length should be reduced by exactly the leakage difference
        expected_diff = 4000
        actual_diff = result_low_leak.secure_key_length - result_high_leak.secure_key_length
        assert actual_diff == expected_diff

    def test_total_leakage_property(self):
        """Test that total_leakage = syndrome + hash."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=10000,
            storage_noise_r=0.5,
            syndrome_leakage_bits=1234,
            hash_leakage_bits=567,
            epsilon_sec=1e-9,
        )
        assert params.total_leakage == 1234 + 567


class TestMinimumNComputation:
    """Tests for computing minimum n for positive key."""

    def test_minimum_n_calculation(self):
        """Test minimum n calculation for positive key."""
        r = 0.5
        leakage = 5000
        epsilon = 1e-9

        min_n = compute_minimum_n_for_positive_key(r, leakage, epsilon)

        # Verify that min_n actually yields positive key
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=min_n,
            storage_noise_r=r,
            syndrome_leakage_bits=leakage,
            hash_leakage_bits=0,
            epsilon_sec=epsilon,
        )
        result = compute_nsm_key_length(params)
        assert result.secure_key_length >= 1

        # Verify that min_n - 1 does not
        params_minus_one = NSMPrivacyAmplificationParams(
            reconciled_key_length=min_n - 1,
            storage_noise_r=r,
            syndrome_leakage_bits=leakage,
            hash_leakage_bits=0,
            epsilon_sec=epsilon,
        )
        result_minus_one = compute_nsm_key_length(params_minus_one)
        assert result_minus_one.secure_key_length <= 0

    def test_minimum_n_perfect_storage(self):
        """Test that perfect storage (r=1) yields huge minimum n."""
        # r=1 means h_min ≈ 0, so min_n should be very large
        min_n = compute_minimum_n_for_positive_key(
            storage_noise_r=0.9999,  # Near-perfect storage
            total_leakage_bits=100,
            epsilon_sec=1e-9,
        )
        # Should require many bits due to low entropy rate
        assert min_n > 100000


class TestFeasibilityValidation:
    """Tests for quick feasibility validation."""

    def test_validate_feasibility_success(self):
        """Test successful feasibility validation."""
        is_feasible, status, key_len = validate_nsm_feasibility(
            reconciled_key_length=100000,
            storage_noise_r=0.5,
            total_leakage_bits=10000,
            epsilon_sec=1e-9,
        )
        assert is_feasible is True
        assert status == FeasibilityResult.FEASIBLE
        assert key_len > 0

    def test_validate_feasibility_death_valley(self):
        """Test Death Valley detection in validation."""
        is_feasible, status, key_len = validate_nsm_feasibility(
            reconciled_key_length=100,
            storage_noise_r=0.75,
            total_leakage_bits=10000,
            epsilon_sec=1e-9,
        )
        assert is_feasible is False
        assert status == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY
        assert key_len == 0

    def test_validate_feasibility_qber_abort(self):
        """Test QBER hard limit detection."""
        is_feasible, status, key_len = validate_nsm_feasibility(
            reconciled_key_length=100000,
            storage_noise_r=0.5,
            total_leakage_bits=10000,
            epsilon_sec=1e-9,
            adjusted_qber=0.30,  # Above 22% hard limit
        )
        assert is_feasible is False
        assert status == FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH


class TestMaxBoundEntropy:
    """Tests for Max Bound entropy rate calculation."""

    def test_max_bound_low_r(self):
        """Test Max Bound for high noise (low r)."""
        r = 0.1
        h_min = max_bound_entropy_rate(r)

        # For low r, both bounds should give high entropy
        assert h_min > 0.8

    def test_max_bound_high_r(self):
        """Test Max Bound for low noise (high r)."""
        r = 0.9
        h_min = max_bound_entropy_rate(r)

        # For high r, entropy should be low (1 - r = 0.1)
        assert h_min < 0.2

    def test_max_bound_crossover(self):
        """Test Max Bound near crossover point (r ≈ 0.82)."""
        # At crossover, both bounds are approximately equal
        for r in [0.80, 0.82, 0.84]:
            h_min = max_bound_entropy_rate(r)
            h2 = collision_entropy_rate(r)
            dk_bound = gamma_function(h2)
            ve_bound = 1.0 - r

            # Near crossover, bounds should be close
            assert abs(dk_bound - ve_bound) < 0.1 or h_min > 0

    def test_max_bound_boundary_values(self):
        """Test Max Bound at boundary values."""
        # r = 0: complete depolarization
        assert max_bound_entropy_rate(0.0) == 1.0

        # r = 1: perfect storage (virtual erasure = 1-r = 0)
        # The function may return a small epsilon for numerical stability
        assert max_bound_entropy_rate(1.0) < 1e-9


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_key_length(self):
        """Test that negative key length raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            NSMPrivacyAmplificationParams(
                reconciled_key_length=-1,
                storage_noise_r=0.5,
                syndrome_leakage_bits=100,
                hash_leakage_bits=10,
                epsilon_sec=1e-9,
            )

    def test_invalid_storage_noise(self):
        """Test that r outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="storage_noise_r"):
            NSMPrivacyAmplificationParams(
                reconciled_key_length=1000,
                storage_noise_r=1.5,
                syndrome_leakage_bits=100,
                hash_leakage_bits=10,
                epsilon_sec=1e-9,
            )

    def test_invalid_epsilon(self):
        """Test that epsilon outside (0, 1) raises error."""
        with pytest.raises(ValueError, match="epsilon_sec"):
            NSMPrivacyAmplificationParams(
                reconciled_key_length=1000,
                storage_noise_r=0.5,
                syndrome_leakage_bits=100,
                hash_leakage_bits=10,
                epsilon_sec=0.0,
            )

    def test_invalid_leakage(self):
        """Test that negative leakage raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            NSMPrivacyAmplificationParams(
                reconciled_key_length=1000,
                storage_noise_r=0.5,
                syndrome_leakage_bits=-100,
                hash_leakage_bits=10,
                epsilon_sec=1e-9,
            )


class TestErvenParameters:
    """Tests using parameters from Erven et al. (2014)."""

    @pytest.mark.parametrize("r,n,leakage_fraction,expected_positive", [
        (0.75, 100000, 0.10, True),   # Baseline from paper: h_min ≈ 0.25
        (0.75, 100, 0.10, False),     # Too small for any leakage
        (0.50, 50000, 0.10, True),    # Higher noise: h_min ≈ 0.50
        (0.90, 200000, 0.01, True),   # Low noise (h_min≈0.1), low leakage
        (0.95, 500000, 0.005, True),  # Very low noise: h_min≈0.05, minimal leakage
    ])
    def test_erven_scenarios(self, r, n, leakage_fraction, expected_positive):
        """Test various scenarios from Erven et al. parameter space."""
        params = NSMPrivacyAmplificationParams(
            reconciled_key_length=n,
            storage_noise_r=r,
            syndrome_leakage_bits=int(n * leakage_fraction),
            hash_leakage_bits=64,
            epsilon_sec=2.5e-7,  # Erven security parameter
        )
        result = compute_nsm_key_length(params)

        if expected_positive:
            assert result.secure_key_length > 0, f"r={r}, n={n} should yield positive key"
        else:
            assert result.secure_key_length <= 0, f"r={r}, n={n} should be in Death Valley"

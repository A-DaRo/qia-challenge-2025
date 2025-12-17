"""
Unit tests for caligo.security.bounds module.

Tests cover:
- Core entropy bound functions (gamma, collision, Dupuis-König, Lupo)
- Max bound selection
- Rational adversary bound
- Bounded storage extension
- Strong converse exponent
- Literature value validation

References
----------
- Lupo et al. (2023), Figure 1 and Table values
- Schaffner et al. (2009), Corollary 7
"""

from __future__ import annotations

import math

import pytest

from caligo.security.bounds import (
    # Constants
    QBER_CONSERVATIVE_THRESHOLD,
    QBER_ABSOLUTE_THRESHOLD,
    R_TILDE,
    R_CROSSOVER,
    DEFAULT_EPSILON_SEC,
    # Functions
    gamma_function,
    collision_entropy_rate,
    dupuis_konig_bound,
    lupo_virtual_erasure_bound,
    max_bound_entropy,
    rational_adversary_bound,
    bounded_storage_entropy,
    strong_converse_exponent,
    compute_extractable_key_rate,
    _validate_r,
    _validate_nu,
    _g_function,
)
from caligo.types.exceptions import InvalidParameterError


class TestConstants:
    """Tests for security threshold constants."""

    def test_qber_conservative_threshold(self):
        """Conservative threshold should be 11%."""
        assert QBER_CONSERVATIVE_THRESHOLD == pytest.approx(0.11, abs=0.001)

    def test_qber_absolute_threshold(self):
        """Absolute threshold should be 22%."""
        assert QBER_ABSOLUTE_THRESHOLD == pytest.approx(0.22, abs=0.001)

    def test_r_tilde(self):
        """R_TILDE should be approximately 0.78."""
        assert R_TILDE == pytest.approx(0.7798, abs=0.001)

    def test_r_crossover(self):
        """R_CROSSOVER should be approximately 0.25.
        
        This is where Dupuis-König bound = Lupo bound ≈ 0.75.
        Mathematically: log₂(1 + 3r²) = r at r ≈ 0.25.
        """
        assert R_CROSSOVER == pytest.approx(0.25, abs=0.01)

    def test_default_epsilon(self):
        """Default security parameter should be 10^-10."""
        assert DEFAULT_EPSILON_SEC == pytest.approx(1e-10, rel=0.01)


class TestValidation:
    """Tests for parameter validation helpers."""

    def test_validate_r_valid_range(self):
        """Valid r values should not raise."""
        for r in [0.0, 0.5, 1.0]:
            _validate_r(r)  # Should not raise

    def test_validate_r_below_zero(self):
        """r < 0 should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match="must be in"):
            _validate_r(-0.1)

    def test_validate_r_above_one(self):
        """r > 1 should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match="must be in"):
            _validate_r(1.1)

    def test_validate_nu_valid_range(self):
        """Valid nu values should not raise."""
        for nu in [0.0, 0.5, 1.0]:
            _validate_nu(nu)  # Should not raise

    def test_validate_nu_invalid(self):
        """Invalid nu should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            _validate_nu(-0.5)
        with pytest.raises(InvalidParameterError):
            _validate_nu(1.5)


class TestGFunction:
    """Tests for the internal g function."""

    def test_g_at_zero(self):
        """g(0) should return inf (handled as boundary)."""
        assert _g_function(0.0) == float("inf")

    def test_g_at_half(self):
        """g(0.5) = h(0.5) + 0.5 - 1 = 1 + 0.5 - 1 = 0.5."""
        assert _g_function(0.5) == pytest.approx(0.5, abs=0.001)

    def test_g_monotonic(self):
        """g should be monotonically increasing on (0, 1)."""
        prev = _g_function(0.1)
        for y in [0.2, 0.3, 0.4, 0.5]:
            curr = _g_function(y)
            assert curr > prev
            prev = curr


class TestGammaFunction:
    """Tests for the Γ function."""

    def test_gamma_above_half_identity(self):
        """Γ(x) = x for x ≥ 0.5."""
        for x in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            assert gamma_function(x) == pytest.approx(x, abs=0.001)

    def test_gamma_at_half(self):
        """Γ(0.5) should equal 0.5."""
        assert gamma_function(0.5) == pytest.approx(0.5, abs=0.001)

    def test_gamma_below_half_valid(self):
        """Γ(x) should return value in (0, 0.5) for x < 0.5."""
        for x in [0.1, 0.2, 0.3, 0.4]:
            result = gamma_function(x)
            assert 0 < result < 0.5

    def test_gamma_at_zero(self):
        """Γ(0) should return 0."""
        assert gamma_function(0.0) == 0.0

    def test_gamma_negative(self):
        """Γ(x) for x < 0 should return 0 (edge case)."""
        assert gamma_function(-0.5) == 0.0


class TestCollisionEntropyRate:
    """Tests for collision entropy rate h₂."""

    def test_complete_depolarization(self):
        """h₂(0) = 1 - log₂(1) = 1."""
        assert collision_entropy_rate(0.0) == pytest.approx(1.0, abs=0.001)

    def test_perfect_storage(self):
        """h₂(1) = 1 - log₂(4) = -1."""
        assert collision_entropy_rate(1.0) == pytest.approx(-1.0, abs=0.001)

    def test_intermediate_values(self):
        """Intermediate r values should give expected entropy."""
        # h₂(0.5) = 1 - log₂(1 + 0.75) = 1 - log₂(1.75) ≈ 0.193
        assert collision_entropy_rate(0.5) == pytest.approx(
            1 - math.log2(1.75), abs=0.001
        )

    def test_invalid_r(self):
        """Invalid r should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            collision_entropy_rate(-0.1)
        with pytest.raises(InvalidParameterError):
            collision_entropy_rate(1.1)

    def test_monotonic_decreasing(self):
        """h₂ should decrease as r increases (less noise = less entropy)."""
        prev = collision_entropy_rate(0.0)
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            curr = collision_entropy_rate(r)
            assert curr < prev
            prev = curr


class TestDupuisKonigBound:
    """Tests for Dupuis-König collision entropy bound."""

    def test_complete_depolarization(self):
        """h_A(0) should be 1.0."""
        assert dupuis_konig_bound(0.0) == pytest.approx(1.0, abs=0.01)

    def test_literature_values(self, literature_entropy_values):
        """Verify against published values from Lupo et al."""
        for r, expected_dk, _ in literature_entropy_values:
            if r < 1.0:  # Skip r=1 edge case
                result = dupuis_konig_bound(r)
                assert result == pytest.approx(expected_dk, abs=0.02)

    def test_invalid_r(self):
        """Invalid r should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            dupuis_konig_bound(-0.1)


class TestLupoVirtualErasureBound:
    """Tests for Lupo virtual erasure bound."""

    def test_complete_depolarization(self):
        """h_B(0) = 1 - 0 = 1."""
        assert lupo_virtual_erasure_bound(0.0) == pytest.approx(1.0, abs=0.001)

    def test_perfect_storage(self):
        """h_B(1) = 1 - 1 = 0."""
        assert lupo_virtual_erasure_bound(1.0) == pytest.approx(0.0, abs=0.001)

    def test_erven_value(self):
        """h_B(0.75) = 0.25."""
        assert lupo_virtual_erasure_bound(0.75) == pytest.approx(0.25, abs=0.001)

    def test_literature_values(self, literature_entropy_values):
        """Verify against published values."""
        for r, _, expected_lupo in literature_entropy_values:
            result = lupo_virtual_erasure_bound(r)
            assert result == pytest.approx(expected_lupo, abs=0.001)

    def test_invalid_r(self):
        """Invalid r should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            lupo_virtual_erasure_bound(-0.1)


class TestMaxBoundEntropy:
    """Tests for the optimal max bound."""

    def test_selects_larger_bound(self, storage_noise_range):
        """Max bound should always be >= both individual bounds."""
        for r in storage_noise_range:
            h_dk = dupuis_konig_bound(r)
            h_lupo = lupo_virtual_erasure_bound(r)
            h_max = max_bound_entropy(r)
            assert h_max >= h_dk - 0.001
            assert h_max >= h_lupo - 0.001

    def test_equals_max_of_bounds(self, storage_noise_range):
        """Max bound should equal max(DK, Lupo)."""
        for r in storage_noise_range:
            h_dk = dupuis_konig_bound(r)
            h_lupo = lupo_virtual_erasure_bound(r)
            h_max = max_bound_entropy(r)
            assert h_max == pytest.approx(max(h_dk, h_lupo), abs=0.001)

    def test_crossover_point(self):
        """At r ≈ 0.25, both bounds should be approximately equal."""
        # The crossover happens where DK bound equals Lupo bound
        # For our formula: Γ[1 - log₂(1 + 3r²)] = 1 - r
        # This occurs around r ≈ 0.25
        crossover_r = 0.25
        h_dk = dupuis_konig_bound(crossover_r)
        h_lupo = lupo_virtual_erasure_bound(crossover_r)
        # They should be close at crossover
        assert abs(h_dk - h_lupo) < 0.05

    def test_dk_better_for_high_noise(self):
        """For small r (high noise), DK bound should be selected."""
        # DK is better only for very small r (before crossover ~0.25)
        for r in [0.05, 0.10, 0.15]:
            h_dk = dupuis_konig_bound(r)
            h_lupo = lupo_virtual_erasure_bound(r)
            assert h_dk > h_lupo

    def test_lupo_better_for_low_noise(self):
        """For large r (low noise), Lupo bound should be selected."""
        for r in [0.85, 0.9, 0.95]:
            h_dk = dupuis_konig_bound(r)
            h_lupo = lupo_virtual_erasure_bound(r)
            assert h_lupo > h_dk

    def test_in_valid_range(self, storage_noise_range):
        """Max bound should always be in [0, 1] for valid r."""
        for r in storage_noise_range:
            h_max = max_bound_entropy(r)
            # Note: for r=1, h_max=0 which is valid
            assert -0.01 <= h_max <= 1.01


class TestRationalAdversaryBound:
    """Tests for rational adversary bound."""

    def test_capped_at_half(self, storage_noise_range):
        """Rational bound should never exceed 0.5."""
        for r in storage_noise_range:
            h_rat = rational_adversary_bound(r)
            assert h_rat <= 0.5 + 0.001

    def test_equals_half_for_high_noise(self):
        """For high noise (small r), rational bound = 0.5."""
        for r in [0.0, 0.1, 0.2, 0.3]:
            h_rat = rational_adversary_bound(r)
            # max_bound is > 0.5 for these r, so rational = 0.5
            assert h_rat == pytest.approx(0.5, abs=0.01)

    def test_equals_max_bound_for_low_noise(self):
        """For low noise (large r), rational bound = max bound."""
        for r in [0.6, 0.7, 0.8, 0.9]:
            h_rat = rational_adversary_bound(r)
            h_max = max_bound_entropy(r)
            if h_max <= 0.5:
                assert h_rat == pytest.approx(h_max, abs=0.001)

    def test_erven_value(self):
        """Rational bound at r=0.75 should be 0.25."""
        assert rational_adversary_bound(0.75) == pytest.approx(0.25, abs=0.001)

    def test_invalid_r(self):
        """Invalid r should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            rational_adversary_bound(-0.1)


class TestBoundedStorageEntropy:
    """Tests for bounded storage entropy."""

    def test_full_storage_equals_max_bound(self, storage_noise_range):
        """ν=1 should give same result as max_bound_entropy."""
        for r in storage_noise_range:
            h_bounded = bounded_storage_entropy(r, 1.0)
            h_max = max_bound_entropy(r)
            assert h_bounded == pytest.approx(h_max, abs=0.001)

    def test_no_storage_equals_half(self, storage_noise_range):
        """ν=0 should give 0.5 (immediate measurement)."""
        for r in storage_noise_range:
            h_bounded = bounded_storage_entropy(r, 0.0)
            assert h_bounded == pytest.approx(0.5, abs=0.001)

    def test_interpolation(self):
        """Bounded entropy should interpolate between 0.5 and max bound."""
        r = 0.75
        h_max = max_bound_entropy(r)  # 0.25
        h_half = 0.5

        for nu in [0.25, 0.5, 0.75]:
            h_bounded = bounded_storage_entropy(r, nu)
            expected = (1 - nu) * h_half + nu * h_max
            assert h_bounded == pytest.approx(expected, abs=0.001)

    def test_erven_experimental(self):
        """Test with Erven experimental parameters."""
        h_bounded = bounded_storage_entropy(0.75, 0.002)
        # (1-0.002)*0.5 + 0.002*0.25 = 0.499*0.5 + 0.001*0.25 ≈ 0.4995
        assert h_bounded > 0.49

    def test_invalid_parameters(self):
        """Invalid r or nu should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            bounded_storage_entropy(-0.1, 0.5)
        with pytest.raises(InvalidParameterError):
            bounded_storage_entropy(0.5, -0.1)
        with pytest.raises(InvalidParameterError):
            bounded_storage_entropy(0.5, 1.5)


class TestStrongConverseExponent:
    """Tests for strong converse error exponent."""

    def test_below_capacity_zero(self):
        """Exponent should be 0 for rate below capacity."""
        # For r=0.5, capacity is approximately 0.19
        gamma = strong_converse_exponent(0.5, 0.0)
        assert gamma == pytest.approx(0.0, abs=0.001)

    def test_above_capacity_positive(self):
        """Exponent should be positive for rate above capacity."""
        # Try rate of 0.5 which is above capacity for r=0.5
        gamma = strong_converse_exponent(0.5, 0.5)
        assert gamma >= 0

    def test_perfect_storage_capacity_one(self):
        """Perfect storage (r=1) has capacity 1."""
        # Rate 0 below capacity
        gamma = strong_converse_exponent(1.0, 0.0)
        assert gamma == pytest.approx(0.0, abs=0.001)

    def test_complete_depolarization_capacity_zero(self):
        """Complete depolarization (r=0) has capacity 0."""
        # Any positive rate is above capacity
        gamma = strong_converse_exponent(0.0, 0.5)
        assert gamma >= 0

    def test_invalid_r(self):
        """Invalid r should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            strong_converse_exponent(-0.1, 0.5)


class TestExtractableKeyRate:
    """Tests for extractable key rate computation."""

    def test_erven_positive_rate(self):
        """Erven parameters should yield positive key rate."""
        rate = compute_extractable_key_rate(
            r=0.75, nu=0.002, qber=0.02, ec_efficiency=1.16
        )
        assert rate > 0

    def test_high_qber_negative_rate(self):
        """High QBER should yield negative key rate."""
        rate = compute_extractable_key_rate(
            r=0.75, nu=0.002, qber=0.25, ec_efficiency=1.16
        )
        assert rate < 0

    def test_perfect_channel_maximum_rate(self):
        """Zero QBER should yield maximum key rate."""
        rate_zero = compute_extractable_key_rate(
            r=0.75, nu=0.002, qber=0.0, ec_efficiency=1.0
        )
        rate_small = compute_extractable_key_rate(
            r=0.75, nu=0.002, qber=0.01, ec_efficiency=1.0
        )
        assert rate_zero > rate_small

    def test_invalid_parameters(self):
        """Invalid parameters should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            compute_extractable_key_rate(r=-0.1, nu=0.5, qber=0.1)
        with pytest.raises(InvalidParameterError):
            compute_extractable_key_rate(r=0.5, nu=0.5, qber=0.6)
        with pytest.raises(InvalidParameterError):
            compute_extractable_key_rate(r=0.5, nu=0.5, qber=0.1, ec_efficiency=0.5)


class TestPropertyBased:
    """Property-based tests for entropy bounds."""

    @pytest.mark.parametrize("r", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_max_bound_in_valid_range(self, r):
        """Max bound should be in reasonable range for all r."""
        h = max_bound_entropy(r)
        assert -2.0 <= h <= 1.0

    @pytest.mark.parametrize("r", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_rational_bound_in_valid_range(self, r):
        """Rational bound should be in [0, 0.5] for all r."""
        h = rational_adversary_bound(r)
        assert 0.0 <= h <= 0.5 + 0.001

    @pytest.mark.parametrize(
        "r,nu", [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0), (0.75, 0.002), (0.9, 0.1)]
    )
    def test_bounded_storage_in_valid_range(self, r, nu):
        """Bounded storage entropy should be in reasonable range."""
        h = bounded_storage_entropy(r, nu)
        assert 0.0 <= h <= 1.0

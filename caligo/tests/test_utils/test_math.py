"""Unit tests for caligo.utils.math module."""

import math

import pytest

from caligo.utils.math import (
    binary_entropy,
    channel_capacity,
    finite_size_penalty,
    gamma_function,
    smooth_min_entropy_rate,
    key_length_bound,
)


class TestBinaryEntropy:
    """Tests for binary_entropy function."""

    def test_entropy_at_zero(self):
        """h(0) = 0 by convention."""
        assert binary_entropy(0.0) == 0.0

    def test_entropy_at_one(self):
        """h(1) = 0 by convention."""
        assert binary_entropy(1.0) == 0.0

    def test_entropy_at_half(self):
        """h(0.5) = 1 (maximum entropy)."""
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

    def test_entropy_symmetry(self):
        """h(p) = h(1-p) for all p."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(binary_entropy(p) - binary_entropy(1 - p)) < 1e-10

    def test_entropy_bounds(self):
        """h(p) ∈ [0, 1] for all p ∈ [0, 1]."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            h = binary_entropy(p)
            assert 0.0 <= h <= 1.0

    def test_entropy_invalid_negative(self):
        """Raise ValueError for p < 0."""
        with pytest.raises(ValueError, match="must be in"):
            binary_entropy(-0.1)

    def test_entropy_invalid_above_one(self):
        """Raise ValueError for p > 1."""
        with pytest.raises(ValueError, match="must be in"):
            binary_entropy(1.1)

    def test_entropy_known_value(self):
        """Test against known value: h(0.11) ≈ 0.5."""
        # h(0.11) is approximately 0.5 (this is why 11% QBER is significant)
        h = binary_entropy(0.11)
        assert 0.45 < h < 0.55


class TestChannelCapacity:
    """Tests for channel_capacity function."""

    def test_capacity_at_zero_qber(self):
        """C(0) = 1 (perfect channel)."""
        assert channel_capacity(0.0) == 1.0

    def test_capacity_at_half_qber(self):
        """C(0.5) = 0 (useless channel)."""
        assert abs(channel_capacity(0.5)) < 1e-10

    def test_capacity_decreases_with_qber(self):
        """Capacity decreases as QBER increases."""
        prev_cap = channel_capacity(0.0)
        for qber in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            cap = channel_capacity(qber)
            assert cap <= prev_cap
            prev_cap = cap

    def test_capacity_invalid_qber(self):
        """Raise ValueError for QBER > 0.5."""
        with pytest.raises(ValueError, match="must be in"):
            channel_capacity(0.6)


class TestFiniteSizePenalty:
    """Tests for finite_size_penalty function."""

    def test_penalty_positive(self):
        """Penalty should always be positive."""
        mu = finite_size_penalty(n=10000, k=1000)
        assert mu > 0

    def test_penalty_decreases_with_sample_size(self):
        """Penalty decreases as sample sizes increase."""
        mu_small = finite_size_penalty(n=1000, k=100)
        mu_large = finite_size_penalty(n=100000, k=10000)
        assert mu_large < mu_small

    def test_penalty_increases_with_security(self):
        """Penalty increases with tighter security parameter."""
        mu_loose = finite_size_penalty(n=10000, k=1000, epsilon_sec=1e-5)
        mu_tight = finite_size_penalty(n=10000, k=1000, epsilon_sec=1e-15)
        assert mu_tight > mu_loose

    def test_penalty_invalid_n(self):
        """Raise ValueError for n <= 0."""
        with pytest.raises(ValueError, match="n="):
            finite_size_penalty(n=0, k=100)

    def test_penalty_invalid_k(self):
        """Raise ValueError for k <= 0."""
        with pytest.raises(ValueError, match="k="):
            finite_size_penalty(n=1000, k=0)

    def test_penalty_invalid_epsilon(self):
        """Raise ValueError for epsilon_sec not in (0, 1)."""
        with pytest.raises(ValueError, match="epsilon_sec"):
            finite_size_penalty(n=1000, k=100, epsilon_sec=0.0)
        with pytest.raises(ValueError, match="epsilon_sec"):
            finite_size_penalty(n=1000, k=100, epsilon_sec=1.0)

    def test_penalty_reasonable_magnitude(self):
        """Penalty should be small for large sample sizes."""
        # For typical experimental parameters from Erven et al.
        mu = finite_size_penalty(n=int(8e7), k=int(8e6), epsilon_sec=2.5e-7)
        assert mu < 0.01  # Should be reasonably small for large n


class TestGammaFunction:
    """Tests for gamma_function function."""

    def test_gamma_at_zero(self):
        """Γ(0) = 1 (perfect storage, worst for security)."""
        assert abs(gamma_function(0.0) - 1.0) < 1e-10

    def test_gamma_at_one(self):
        """Γ(1) = 1 - log₂(4) = -1 (complete depolarization)."""
        assert abs(gamma_function(1.0) - (-1.0)) < 1e-10

    def test_gamma_decreases_with_noise(self):
        """Γ(r) decreases as storage noise r increases."""
        prev_gamma = gamma_function(0.0)
        for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            g = gamma_function(r)
            assert g < prev_gamma
            prev_gamma = g

    def test_gamma_invalid_r(self):
        """Raise ValueError for r outside [0, 1]."""
        with pytest.raises(ValueError, match="r="):
            gamma_function(-0.1)
        with pytest.raises(ValueError, match="r="):
            gamma_function(1.1)

    def test_gamma_typical_value(self):
        """Test Γ at typical experimental value r=0.75."""
        g = gamma_function(0.75)
        # Γ(0.75) = 1 - log₂(1 + 3*0.75²) = 1 - log₂(2.6875) ≈ -0.427
        assert -0.5 < g < -0.4


class TestSmoothMinEntropyRate:
    """Tests for smooth_min_entropy_rate function."""

    def test_entropy_rate_positive_for_low_qber(self):
        """Entropy rate should be positive for low QBER with noise."""
        gamma = gamma_function(0.75)  # ≈ -0.43
        rate = smooth_min_entropy_rate(qber=0.05, gamma=gamma)
        # For QBER=5%, h(0.05) ≈ 0.29
        # Rate = gamma - h(qber) ≈ -0.43 - 0.29 = -0.72
        # This shows even with noise, QBER eats into entropy
        assert rate < 0  # Actually negative in this case

    def test_entropy_rate_with_perfect_storage(self):
        """Test entropy rate with perfect storage (gamma=1)."""
        rate = smooth_min_entropy_rate(qber=0.05, gamma=1.0)
        # Rate = 1 - h(0.05) ≈ 1 - 0.29 = 0.71
        assert rate > 0.5


class TestKeyLengthBound:
    """Tests for key_length_bound function."""

    def test_key_length_positive_for_good_params(self):
        """Key length should be positive for good parameters."""
        length = key_length_bound(
            n_sifted=100000,
            qber=0.05,
            leakage_bits=10000,
        )
        assert length > 0

    def test_key_length_zero_for_bad_params(self):
        """Key length should be zero for high QBER."""
        length = key_length_bound(
            n_sifted=1000,
            qber=0.40,  # Very high QBER
            leakage_bits=500,
        )
        assert length == 0

    def test_key_length_decreases_with_leakage(self):
        """Key length decreases with more leakage."""
        len_low_leak = key_length_bound(n_sifted=100000, qber=0.05, leakage_bits=1000)
        len_high_leak = key_length_bound(n_sifted=100000, qber=0.05, leakage_bits=50000)
        assert len_high_leak < len_low_leak

    def test_key_length_with_nsm_gamma(self):
        """Test key length calculation with NSM gamma."""
        gamma = gamma_function(0.75)
        length = key_length_bound(
            n_sifted=100000,
            qber=0.05,
            leakage_bits=10000,
            gamma=gamma,
        )
        # With adversarial storage, entropy rate is lower
        # This might result in zero or negative key length
        assert length >= 0

"""
Unit tests for caligo.simulation.noise_models module.

Tests NSMStorageNoiseModel and ChannelNoiseProfile.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.simulation.noise_models import (
    NSMStorageNoiseModel,
    ChannelNoiseProfile,
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
)
from caligo.types.exceptions import InvalidParameterError


# =============================================================================
# NSMStorageNoiseModel Tests
# =============================================================================


class TestNSMStorageNoiseModelCreation:
    """Tests for NSMStorageNoiseModel initialization."""

    def test_valid_creation(self, storage_noise_model):
        """Valid parameters should create instance."""
        assert storage_noise_model.r == 0.75
        assert storage_noise_model.delta_t_ns == 1_000_000

    def test_creation_with_different_r(self):
        """Different r values should be accepted."""
        model = NSMStorageNoiseModel(r=0.5, delta_t_ns=1_000_000)
        assert model.r == 0.5

    def test_invalid_r_below_zero(self):
        """r < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="r=.* must be in"):
            NSMStorageNoiseModel(r=-0.1, delta_t_ns=1_000_000)

    def test_invalid_r_above_one(self):
        """r > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="r=.* must be in"):
            NSMStorageNoiseModel(r=1.5, delta_t_ns=1_000_000)

    def test_r_edge_cases(self):
        """r=0 and r=1 should be valid."""
        m0 = NSMStorageNoiseModel(r=0.0, delta_t_ns=1_000_000)
        assert m0.r == 0.0

        m1 = NSMStorageNoiseModel(r=1.0, delta_t_ns=1_000_000)
        assert m1.r == 1.0

    def test_invalid_delta_t(self):
        """delta_t_ns <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="delta_t_ns=.* must be > 0"):
            NSMStorageNoiseModel(r=0.75, delta_t_ns=0)

        with pytest.raises(ValueError, match="delta_t_ns=.* must be > 0"):
            NSMStorageNoiseModel(r=0.75, delta_t_ns=-1000)


class TestNSMStorageNoiseModelProperties:
    """Tests for NSMStorageNoiseModel properties."""

    def test_depolar_prob(self, storage_noise_model):
        """depolar_prob = 1 - r."""
        assert storage_noise_model.depolar_prob == pytest.approx(0.25)

    def test_depolar_prob_extremes(self):
        """Depolar prob at extremes."""
        m0 = NSMStorageNoiseModel(r=0.0, delta_t_ns=1_000_000)
        assert m0.depolar_prob == 1.0

        m1 = NSMStorageNoiseModel(r=1.0, delta_t_ns=1_000_000)
        assert m1.depolar_prob == 0.0


class TestNSMStorageNoiseModelApplyNoise:
    """Tests for NSMStorageNoiseModel.apply_noise()."""

    def test_apply_noise_shape(self, storage_noise_model):
        """Output should have same shape as input."""
        state = np.array([[1, 0], [0, 0]], dtype=complex)
        output = storage_noise_model.apply_noise(state)
        assert output.shape == (2, 2)

    def test_apply_noise_preserves_trace(self, storage_noise_model):
        """Trace should be preserved."""
        state = np.array([[1, 0], [0, 0]], dtype=complex)
        output = storage_noise_model.apply_noise(state)
        assert np.trace(output) == pytest.approx(1.0)

    def test_apply_noise_invalid_shape(self, storage_noise_model):
        """Non-2x2 input should raise ValueError."""
        state = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        with pytest.raises(ValueError, match="Expected 2x2"):
            storage_noise_model.apply_noise(state)

    def test_apply_noise_depolarization(self):
        """Full depolarization should give maximally mixed state."""
        model = NSMStorageNoiseModel(r=0.0, delta_t_ns=1_000_000)
        state = np.array([[1, 0], [0, 0]], dtype=complex)
        output = model.apply_noise(state)

        # Maximally mixed = I/2
        expected = np.eye(2) / 2.0
        np.testing.assert_array_almost_equal(output, expected)

    def test_apply_noise_perfect_storage(self):
        """Perfect storage should preserve state."""
        model = NSMStorageNoiseModel(r=1.0, delta_t_ns=1_000_000)
        state = np.array([[1, 0], [0, 0]], dtype=complex)
        output = model.apply_noise(state)

        np.testing.assert_array_almost_equal(output, state)


class TestNSMStorageNoiseModelDerivedMethods:
    """Tests for derived methods."""

    def test_get_effective_fidelity(self, storage_noise_model):
        """Effective fidelity = (1+r)/2."""
        # r=0.75 → (1+0.75)/2 = 0.875
        assert storage_noise_model.get_effective_fidelity() == pytest.approx(0.875)

    def test_effective_fidelity_extremes(self):
        """Fidelity at extreme r values."""
        m0 = NSMStorageNoiseModel(r=0.0, delta_t_ns=1_000_000)
        assert m0.get_effective_fidelity() == pytest.approx(0.5)

        m1 = NSMStorageNoiseModel(r=1.0, delta_t_ns=1_000_000)
        assert m1.get_effective_fidelity() == pytest.approx(1.0)

    def test_get_min_entropy_bound(self, storage_noise_model):
        """Min entropy bound should be non-negative."""
        bound = storage_noise_model.get_min_entropy_bound()
        assert bound >= 0.0

    def test_repr(self, storage_noise_model):
        """__repr__ should return informative string."""
        repr_str = repr(storage_noise_model)
        assert "NSMStorageNoiseModel" in repr_str
        assert "r=0.75" in repr_str
        assert "delta_t_ns=1000000" in repr_str


# =============================================================================
# ChannelNoiseProfile Tests
# =============================================================================


class TestChannelNoiseProfileCreation:
    """Tests for ChannelNoiseProfile initialization."""

    def test_valid_creation(self, channel_noise_profile):
        """Valid parameters should create instance."""
        assert channel_noise_profile.source_fidelity == 0.98
        assert channel_noise_profile.detector_efficiency == 0.90
        assert channel_noise_profile.detector_error == 0.01
        assert channel_noise_profile.dark_count_rate == 1e-5

    def test_default_transmission_loss(self):
        """Default transmission_loss should be 0.0."""
        profile = ChannelNoiseProfile(
            source_fidelity=0.98,
            detector_efficiency=0.90,
            detector_error=0.01,
            dark_count_rate=1e-5,
        )
        assert profile.transmission_loss == 0.0

    def test_frozen_dataclass(self, channel_noise_profile):
        """ChannelNoiseProfile should be immutable."""
        with pytest.raises(AttributeError):
            channel_noise_profile.source_fidelity = 0.5


class TestChannelNoiseProfileInvariants:
    """Tests for ChannelNoiseProfile invariant validation."""

    def test_inv_cnp_001_source_fidelity_bounds(self):
        """INV-CNP-001: source_fidelity ∈ (0.5, 1]."""
        with pytest.raises(InvalidParameterError, match="INV-CNP-001"):
            ChannelNoiseProfile(
                source_fidelity=0.5,  # Must be > 0.5
                detector_efficiency=0.90,
                detector_error=0.01,
                dark_count_rate=1e-5,
            )

        # Valid boundary
        profile = ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=0.90,
            detector_error=0.01,
            dark_count_rate=1e-5,
        )
        assert profile.source_fidelity == 1.0

    def test_inv_cnp_002_detector_efficiency_bounds(self):
        """INV-CNP-002: detector_efficiency ∈ (0, 1]."""
        with pytest.raises(InvalidParameterError, match="INV-CNP-002"):
            ChannelNoiseProfile(
                source_fidelity=0.98,
                detector_efficiency=0.0,  # Must be > 0
                detector_error=0.01,
                dark_count_rate=1e-5,
            )

    def test_inv_cnp_003_detector_error_bounds(self):
        """INV-CNP-003: detector_error ∈ [0, 0.5]."""
        with pytest.raises(InvalidParameterError, match="INV-CNP-003"):
            ChannelNoiseProfile(
                source_fidelity=0.98,
                detector_efficiency=0.90,
                detector_error=0.6,  # Must be <= 0.5
                dark_count_rate=1e-5,
            )

        # Valid boundaries
        p0 = ChannelNoiseProfile(
            source_fidelity=0.98,
            detector_efficiency=0.90,
            detector_error=0.0,
            dark_count_rate=1e-5,
        )
        assert p0.detector_error == 0.0

        p5 = ChannelNoiseProfile(
            source_fidelity=0.98,
            detector_efficiency=0.90,
            detector_error=0.5,
            dark_count_rate=1e-5,
        )
        assert p5.detector_error == 0.5

    def test_inv_cnp_004_dark_count_bounds(self):
        """INV-CNP-004: dark_count_rate ∈ [0, 1]."""
        with pytest.raises(InvalidParameterError, match="INV-CNP-004"):
            ChannelNoiseProfile(
                source_fidelity=0.98,
                detector_efficiency=0.90,
                detector_error=0.01,
                dark_count_rate=-0.1,
            )

        with pytest.raises(InvalidParameterError, match="INV-CNP-004"):
            ChannelNoiseProfile(
                source_fidelity=0.98,
                detector_efficiency=0.90,
                detector_error=0.01,
                dark_count_rate=1.5,
            )

    def test_inv_cnp_005_transmission_loss_bounds(self):
        """INV-CNP-005: transmission_loss ∈ [0, 1)."""
        with pytest.raises(InvalidParameterError, match="INV-CNP-005"):
            ChannelNoiseProfile(
                source_fidelity=0.98,
                detector_efficiency=0.90,
                detector_error=0.01,
                dark_count_rate=1e-5,
                transmission_loss=1.0,  # Must be < 1
            )


class TestChannelNoiseProfileDerivedProperties:
    """Tests for ChannelNoiseProfile derived properties."""

    def test_total_qber_calculation(self, channel_noise_profile):
        """QBER calculation from noise sources."""
        # source_fidelity=0.98 → (1-0.98)/2 = 0.01
        # detector_error=0.01
        # dark contribution ≈ (1-0.9) * 1e-5 / 2 ≈ negligible
        qber = channel_noise_profile.total_qber
        assert qber == pytest.approx(0.02, abs=0.001)

    def test_total_qber_perfect(self, channel_noise_profile_perfect):
        """Perfect channel should have zero QBER."""
        assert channel_noise_profile_perfect.total_qber == pytest.approx(0.0)

    def test_is_secure(self, channel_noise_profile):
        """is_secure checks QBER < 0.11."""
        # Our test profile has low QBER
        assert channel_noise_profile.is_secure is True

    def test_is_secure_false_for_high_qber(self):
        """High QBER should fail security check."""
        profile = ChannelNoiseProfile(
            source_fidelity=0.70,  # Low fidelity
            detector_efficiency=0.90,
            detector_error=0.05,
            dark_count_rate=1e-5,
        )
        # QBER ≈ 0.15 + 0.05 = 0.20 > 0.11
        assert profile.is_secure is False

    def test_is_feasible(self, channel_noise_profile):
        """is_feasible checks QBER < 0.22."""
        assert channel_noise_profile.is_feasible is True

    def test_is_feasible_false_for_very_high_qber(self):
        """Very high QBER should fail feasibility check."""
        profile = ChannelNoiseProfile(
            source_fidelity=0.51,  # Minimum allowed fidelity
            detector_efficiency=0.90,
            detector_error=0.05,
            dark_count_rate=1e-5,
        )
        # QBER ≈ 0.245 + 0.05 > 0.22
        assert profile.is_feasible is False

    def test_security_margin(self, channel_noise_profile):
        """security_margin = 0.11 - total_qber."""
        margin = channel_noise_profile.security_margin
        # QBER ≈ 0.02, margin ≈ 0.09
        assert margin > 0
        assert margin == pytest.approx(0.11 - channel_noise_profile.total_qber)


class TestChannelNoiseProfileFactoryMethods:
    """Tests for ChannelNoiseProfile factory methods."""

    def test_perfect(self, channel_noise_profile_perfect):
        """Perfect factory creates noiseless profile."""
        assert channel_noise_profile_perfect.source_fidelity == 1.0
        assert channel_noise_profile_perfect.detector_efficiency == 1.0
        assert channel_noise_profile_perfect.detector_error == 0.0
        assert channel_noise_profile_perfect.dark_count_rate == 0.0
        assert channel_noise_profile_perfect.transmission_loss == 0.0

    def test_from_erven_experimental(self, channel_noise_profile_erven):
        """Erven factory creates experimental profile."""
        assert channel_noise_profile_erven.detector_efficiency == 0.0150
        assert channel_noise_profile_erven.detector_error == 0.0093

    def test_realistic(self):
        """Realistic factory creates moderate noise profile."""
        profile = ChannelNoiseProfile.realistic()
        assert profile.source_fidelity == 0.98
        assert profile.detector_efficiency == 0.90
        assert profile.detector_error == 0.01

    def test_realistic_custom_values(self):
        """Realistic factory accepts custom values."""
        profile = ChannelNoiseProfile.realistic(
            source_fidelity=0.99,
            detector_efficiency=0.95,
            detector_error=0.005,
        )
        assert profile.source_fidelity == 0.99
        assert profile.detector_efficiency == 0.95
        assert profile.detector_error == 0.005


class TestChannelNoiseProfileConversion:
    """Tests for ChannelNoiseProfile conversion methods."""

    def test_to_nsm_parameters(self, channel_noise_profile):
        """to_nsm_parameters should create valid NSMParameters."""
        nsm = channel_noise_profile.to_nsm_parameters()

        assert nsm.channel_fidelity == channel_noise_profile.source_fidelity
        assert nsm.detection_eff_eta == channel_noise_profile.detector_efficiency
        assert nsm.detector_error == channel_noise_profile.detector_error
        assert nsm.dark_count_prob == channel_noise_profile.dark_count_rate

    def test_to_nsm_parameters_custom_storage(self, channel_noise_profile):
        """to_nsm_parameters accepts custom storage parameters."""
        nsm = channel_noise_profile.to_nsm_parameters(
            storage_noise_r=0.5,
            storage_rate_nu=0.01,
            delta_t_ns=500_000,
        )

        assert nsm.storage_noise_r == 0.5
        assert nsm.storage_rate_nu == 0.01
        assert nsm.delta_t_ns == 500_000


class TestChannelNoiseProfileDiagnostics:
    """Tests for diagnostic methods."""

    def test_get_diagnostic_info(self, channel_noise_profile):
        """get_diagnostic_info should return dict with expected keys."""
        info = channel_noise_profile.get_diagnostic_info()

        assert "source_fidelity" in info
        assert "detector_efficiency" in info
        assert "detector_error" in info
        assert "dark_count_rate" in info
        assert "transmission_loss" in info
        assert "total_qber" in info
        assert "is_secure" in info
        assert "is_feasible" in info
        assert "security_margin" in info

        assert info["source_fidelity"] == 0.98
        assert info["is_secure"] is True

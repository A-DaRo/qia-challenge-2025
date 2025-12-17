"""
Unit tests for caligo.simulation.physical_model module.

Tests NSMParameters, ChannelParameters, derived properties, and invariants.
"""

from __future__ import annotations

import pytest

from caligo.simulation.physical_model import (
    NSMParameters,
    ChannelParameters,
    NANOSECOND,
    MICROSECOND,
    MILLISECOND,
    SECOND,
    TYPICAL_DELTA_T_NS,
    TYPICAL_T1_NS,
    TYPICAL_T2_NS,
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
    create_depolar_noise_model,
    create_t1t2_noise_model,
)
from caligo.types.exceptions import InvalidParameterError


# =============================================================================
# Time Constants Tests
# =============================================================================


class TestTimeConstants:
    """Tests for time unit constants."""

    def test_nanosecond_is_base_unit(self):
        """NANOSECOND should be 1.0."""
        assert NANOSECOND == 1.0

    def test_microsecond_conversion(self):
        """MICROSECOND = 1000 ns."""
        assert MICROSECOND == 1e3

    def test_millisecond_conversion(self):
        """MILLISECOND = 1e6 ns."""
        assert MILLISECOND == 1e6

    def test_second_conversion(self):
        """SECOND = 1e9 ns."""
        assert SECOND == 1e9


# =============================================================================
# NSMParameters Tests
# =============================================================================


class TestNSMParametersCreation:
    """Tests for NSMParameters initialization and validation."""

    def test_valid_creation(self, nsm_params):
        """Valid parameters should create instance."""
        assert nsm_params.storage_noise_r == 0.75
        assert nsm_params.storage_rate_nu == 0.002
        assert nsm_params.delta_t_ns == 1_000_000
        assert nsm_params.channel_fidelity == 0.95

    def test_default_values(self):
        """Default values should be applied correctly."""
        params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.95,
        )
        assert params.detection_eff_eta == 1.0
        assert params.detector_error == 0.0
        assert params.dark_count_prob == 0.0
        assert params.storage_dimension_d == 2

    def test_frozen_dataclass(self, nsm_params):
        """NSMParameters should be immutable."""
        with pytest.raises(AttributeError):
            nsm_params.storage_noise_r = 0.5


class TestNSMParametersInvariants:
    """Tests for NSMParameters invariant validation."""

    def test_inv_nsm_001_storage_noise_lower_bound(self):
        """INV-NSM-001: storage_noise_r >= 0."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-001"):
            NSMParameters(
                storage_noise_r=-0.1,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
            )

    def test_inv_nsm_001_storage_noise_upper_bound(self):
        """INV-NSM-001: storage_noise_r <= 1."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-001"):
            NSMParameters(
                storage_noise_r=1.5,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
            )

    def test_inv_nsm_001_edge_cases(self):
        """INV-NSM-001: boundary values should work."""
        # r=0 (full depolarization)
        p0 = NSMParameters(
            storage_noise_r=0.0,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.95,
        )
        assert p0.storage_noise_r == 0.0

        # r=1 (perfect storage)
        p1 = NSMParameters(
            storage_noise_r=1.0,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.95,
        )
        assert p1.storage_noise_r == 1.0

    def test_inv_nsm_002_storage_rate_bounds(self):
        """INV-NSM-002: storage_rate_nu ∈ [0, 1]."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-002"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=-0.1,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
            )

        with pytest.raises(InvalidParameterError, match="INV-NSM-002"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=1.5,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
            )

    def test_inv_nsm_003_dimension_must_be_2(self):
        """INV-NSM-003: storage_dimension_d must be 2."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-003"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
                storage_dimension_d=3,
            )

    def test_inv_nsm_004_delta_t_positive(self):
        """INV-NSM-004: delta_t_ns must be > 0."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-004"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=0.002,
                delta_t_ns=0.0,
                channel_fidelity=0.95,
            )

        with pytest.raises(InvalidParameterError, match="INV-NSM-004"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=0.002,
                delta_t_ns=-1000,
                channel_fidelity=0.95,
            )

    def test_inv_nsm_005_fidelity_bounds(self):
        """INV-NSM-005: channel_fidelity ∈ (0.5, 1]."""
        # Too low
        with pytest.raises(InvalidParameterError, match="INV-NSM-005"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.5,  # Must be > 0.5
            )

        # Valid boundary
        p = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=1.0,
        )
        assert p.channel_fidelity == 1.0

    def test_inv_nsm_006_detection_efficiency_bounds(self):
        """INV-NSM-006: detection_eff_eta ∈ (0, 1]."""
        with pytest.raises(InvalidParameterError, match="INV-NSM-006"):
            NSMParameters(
                storage_noise_r=0.75,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.95,
                detection_eff_eta=0.0,  # Must be > 0
            )


class TestNSMParametersDerivedProperties:
    """Tests for NSMParameters derived properties."""

    def test_depolar_prob(self, nsm_params):
        """depolar_prob = 1 - r."""
        assert nsm_params.depolar_prob == pytest.approx(0.25)

    def test_depolar_prob_extremes(self):
        """depolar_prob boundary cases."""
        # r=0 → depolar=1
        p0 = NSMParameters.for_testing(storage_noise_r=0.0)
        assert p0.depolar_prob == 1.0

        # r=1 → depolar=0
        p1 = NSMParameters.for_testing(storage_noise_r=1.0)
        assert p1.depolar_prob == 0.0

    def test_qber_channel(self, nsm_params):
        """qber_channel = (1-F)/2 + e_det."""
        # F=0.95, e_det=0 → (1-0.95)/2 = 0.025
        assert nsm_params.qber_channel == pytest.approx(0.025)

    def test_qber_channel_with_detector_error(self):
        """QBER includes detector error."""
        params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.95,
            detector_error=0.01,
        )
        # (1-0.95)/2 + 0.01 = 0.025 + 0.01 = 0.035
        assert params.qber_channel == pytest.approx(0.035)

    def test_storage_capacity(self, nsm_params):
        """storage_capacity = 1 - h(depolar_prob)."""
        # r=0.75 → depolar=0.25
        # h(0.25) ≈ 0.811
        # C ≈ 1 - 0.811 ≈ 0.189
        assert 0.1 < nsm_params.storage_capacity < 0.3

    def test_storage_capacity_edge_cases(self):
        """Storage capacity at extreme r values."""
        # r=1 (perfect storage) → depolar=0 → h(0)=0 → C=1
        p1 = NSMParameters.for_testing(storage_noise_r=1.0)
        assert p1.storage_capacity == 1.0

        # r=0 (full depolar) → depolar=1 → h(1)=0 → C=1
        p0 = NSMParameters.for_testing(storage_noise_r=0.0)
        assert p0.storage_capacity == 1.0

    def test_security_possible(self, nsm_params):
        """security_possible checks QBER < 0.11."""
        # qber_channel = 0.025 < 0.11
        assert nsm_params.security_possible is True

    def test_security_impossible_high_qber(self):
        """High QBER should fail security check."""
        params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.70,  # Low fidelity → high QBER
        )
        # (1-0.70)/2 = 0.15 > 0.11
        assert params.security_possible is False

    def test_storage_security_satisfied(self, nsm_params):
        """storage_security_satisfied checks C*ν < 0.5."""
        # C ≈ 0.19, ν = 0.002 → C*ν ≈ 0.00038 < 0.5
        assert nsm_params.storage_security_satisfied is True


class TestNSMParametersFactoryMethods:
    """Tests for NSMParameters factory methods."""

    def test_from_erven_experimental(self, nsm_params_erven):
        """Erven factory should create valid params."""
        assert nsm_params_erven.storage_noise_r == 0.75
        assert nsm_params_erven.storage_rate_nu == 0.002
        assert nsm_params_erven.detection_eff_eta == 0.0150
        assert nsm_params_erven.detector_error == 0.0093

    def test_for_testing(self):
        """Testing factory should create simplified params."""
        params = NSMParameters.for_testing()
        assert params.storage_noise_r == 0.75
        assert params.channel_fidelity == 0.95
        assert params.delta_t_ns == 1_000_000

    def test_for_testing_custom_values(self):
        """Testing factory accepts custom values."""
        params = NSMParameters.for_testing(
            storage_noise_r=0.5,
            channel_fidelity=0.99,
            delta_t_ns=500_000,
        )
        assert params.storage_noise_r == 0.5
        assert params.channel_fidelity == 0.99
        assert params.delta_t_ns == 500_000


# =============================================================================
# ChannelParameters Tests
# =============================================================================


class TestChannelParametersCreation:
    """Tests for ChannelParameters initialization."""

    def test_valid_creation(self, channel_params):
        """Valid parameters should create instance."""
        assert channel_params.length_km == 0.0
        assert channel_params.t1_ns == TYPICAL_T1_NS
        assert channel_params.t2_ns == TYPICAL_T2_NS

    def test_default_values(self):
        """Default values should be applied correctly."""
        params = ChannelParameters()
        assert params.length_km == 0.0
        assert params.attenuation_db_per_km == 0.2
        assert params.speed_of_light_km_s == 200_000.0

    def test_frozen_dataclass(self, channel_params):
        """ChannelParameters should be immutable."""
        with pytest.raises(AttributeError):
            channel_params.length_km = 10.0


class TestChannelParametersInvariants:
    """Tests for ChannelParameters invariant validation."""

    def test_inv_ch_001_length_non_negative(self):
        """INV-CH-001: length_km >= 0."""
        with pytest.raises(InvalidParameterError, match="INV-CH-001"):
            ChannelParameters(length_km=-1.0)

    def test_inv_ch_002_attenuation_non_negative(self):
        """INV-CH-002: attenuation_db_per_km >= 0."""
        with pytest.raises(InvalidParameterError, match="INV-CH-002"):
            ChannelParameters(attenuation_db_per_km=-0.1)

    def test_inv_ch_003_speed_positive(self):
        """INV-CH-003: speed_of_light_km_s > 0."""
        with pytest.raises(InvalidParameterError, match="INV-CH-003"):
            ChannelParameters(speed_of_light_km_s=0.0)

    def test_inv_ch_004_t1_positive(self):
        """INV-CH-004: t1_ns > 0."""
        with pytest.raises(InvalidParameterError, match="INV-CH-004"):
            ChannelParameters(t1_ns=0.0)

    def test_inv_ch_005_t2_positive_and_less_than_t1(self):
        """INV-CH-005: t2_ns > 0 and t2_ns <= t1_ns."""
        with pytest.raises(InvalidParameterError, match="INV-CH-005"):
            ChannelParameters(t2_ns=0.0)

        with pytest.raises(InvalidParameterError, match="INV-CH-005"):
            ChannelParameters(t1_ns=1_000_000, t2_ns=2_000_000)

    def test_inv_ch_006_cycle_time_positive(self):
        """INV-CH-006: cycle_time_ns > 0."""
        with pytest.raises(InvalidParameterError, match="INV-CH-006"):
            ChannelParameters(cycle_time_ns=0.0)


class TestChannelParametersDerivedProperties:
    """Tests for ChannelParameters derived properties."""

    def test_propagation_delay_zero_length(self, channel_params):
        """Zero length should have zero delay."""
        assert channel_params.propagation_delay_ns == 0.0

    def test_propagation_delay_with_length(self, channel_params_with_loss):
        """Propagation delay calculation."""
        # 10 km / 200_000 km/s * 1e9 ns/s = 50_000 ns = 50 μs
        assert channel_params_with_loss.propagation_delay_ns == pytest.approx(50_000)

    def test_total_loss_db(self, channel_params_with_loss):
        """Total loss = length * attenuation."""
        # 10 km * 0.2 dB/km = 2 dB
        assert channel_params_with_loss.total_loss_db == pytest.approx(2.0)

    def test_total_loss_zero_length(self, channel_params):
        """Zero length should have zero loss."""
        assert channel_params.total_loss_db == 0.0

    def test_transmittance_zero_loss(self, channel_params):
        """Zero loss should have unit transmittance."""
        assert channel_params.transmittance == 1.0

    def test_transmittance_with_loss(self, channel_params_with_loss):
        """Transmittance = 10^(-loss_db/10)."""
        # 10^(-2/10) ≈ 0.631
        assert channel_params_with_loss.transmittance == pytest.approx(0.631, rel=0.01)


class TestChannelParametersFactoryMethods:
    """Tests for ChannelParameters factory methods."""

    def test_for_testing(self):
        """Testing factory creates zero-length channel."""
        params = ChannelParameters.for_testing()
        assert params.length_km == 0.0

    def test_from_erven_experimental(self):
        """Erven factory creates experimental config."""
        params = ChannelParameters.from_erven_experimental()
        assert params.length_km == 0.0
        assert params.t1_ns == 100_000_000  # 100 ms


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestCreateDepolarNoiseModel:
    """Tests for create_depolar_noise_model factory."""

    def test_requires_netsquid(self, nsm_params):
        """Should raise ImportError if NetSquid not available."""
        # This test will pass if NetSquid is not installed
        # and fail gracefully if it is installed
        try:
            model = create_depolar_noise_model(nsm_params)
            # If NetSquid is available, verify the model
            assert model.depolar_rate == pytest.approx(0.25)
        except ImportError as e:
            assert "NetSquid is required" in str(e)


class TestCreateT1T2NoiseModel:
    """Tests for create_t1t2_noise_model factory."""

    def test_requires_netsquid(self, channel_params):
        """Should raise ImportError if NetSquid not available."""
        try:
            model = create_t1t2_noise_model(channel_params)
            # If NetSquid is available, verify the model
            assert hasattr(model, "T1")
        except ImportError as e:
            assert "NetSquid is required" in str(e)

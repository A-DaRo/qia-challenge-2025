"""
Unit tests for ProtocolConfig and PhysicalAdapter (NOISE-PARAMS-001, TASK-NOISE-ADAPTER-001).

These tests verify:
1. Config dataclasses have correct defaults from Erven Table I
2. Validation catches invalid parameters
3. Physical-to-simulator translation produces expected values

References
----------
- Erven et al. (2014): Table I experimental parameters
- sprint_1_specification.md Section 4
"""

import pytest

from ehok.configs.protocol_config import (
    PhysicalParameters,
    NSMSecurityParameters,
    ProtocolParameters,
    ProtocolConfig,
)
from ehok.quantum.noise_adapter import (
    SimulatorNoiseParams,
    physical_to_simulator,
    estimate_qber_from_physical,
    estimate_sifted_rate,
    validate_physical_params_for_simulation,
)


# =============================================================================
# PhysicalParameters Tests
# =============================================================================


class TestPhysicalParameters:
    """Tests for PhysicalParameters dataclass."""

    def test_erven_defaults(self) -> None:
        """Default values should match Erven et al. Table I."""
        params = PhysicalParameters()

        # Erven Table I values
        assert params.mu_pair_per_coherence == pytest.approx(3.145e-5, rel=0.01)
        assert params.eta_total_transmittance == pytest.approx(0.0150, rel=0.01)
        assert params.e_det == pytest.approx(0.0093, rel=0.01)
        assert params.p_dark == pytest.approx(1.50e-8, rel=0.01)

    def test_custom_values(self) -> None:
        """Custom values should be accepted."""
        params = PhysicalParameters(
            mu_pair_per_coherence=0.001,
            eta_total_transmittance=0.1,
            e_det=0.05,
            p_dark=1e-7,
        )
        assert params.mu_pair_per_coherence == 0.001
        assert params.eta_total_transmittance == 0.1
        assert params.e_det == 0.05
        assert params.p_dark == 1e-7

    def test_validation_mu_positive(self) -> None:
        """μ must be positive."""
        with pytest.raises(ValueError, match="mu_pair_per_coherence must be"):
            PhysicalParameters(mu_pair_per_coherence=0.0)

    def test_validation_eta_range(self) -> None:
        """η must be in (0, 1]."""
        with pytest.raises(ValueError, match="eta_total_transmittance must be"):
            PhysicalParameters(eta_total_transmittance=1.5)

    def test_validation_e_det_range(self) -> None:
        """e_det must be in [0, 0.5]."""
        with pytest.raises(ValueError, match="e_det must be"):
            PhysicalParameters(e_det=0.6)

    def test_validation_p_dark_non_negative(self) -> None:
        """P_dark must be non-negative."""
        with pytest.raises(ValueError, match="p_dark must be"):
            PhysicalParameters(p_dark=-1e-10)


# =============================================================================
# NSMSecurityParameters Tests
# =============================================================================


class TestNSMSecurityParameters:
    """Tests for NSMSecurityParameters dataclass."""

    def test_erven_defaults(self) -> None:
        """Default values should match Erven Table I and literature."""
        params = NSMSecurityParameters()

        assert params.storage_noise_r == pytest.approx(0.75, rel=0.01)
        assert params.storage_rate_nu == pytest.approx(0.002, rel=0.01)
        assert params.delta_t_s == pytest.approx(1.0, rel=0.01)
        assert params.delta_t_ns == 1_000_000_000

    def test_custom_values(self) -> None:
        """Custom values should be accepted."""
        params = NSMSecurityParameters(
            storage_noise_r=0.9,
            storage_rate_nu=0.001,
            delta_t_s=2.0,
        )
        assert params.storage_noise_r == 0.9
        assert params.storage_rate_nu == 0.001
        assert params.delta_t_s == 2.0

    def test_validation_storage_noise_r_range(self) -> None:
        """storage_noise_r must be in [0, 1]."""
        with pytest.raises(ValueError, match="storage_noise_r must be"):
            NSMSecurityParameters(storage_noise_r=-0.1)

    def test_validation_storage_rate_nu_range(self) -> None:
        """storage_rate_nu must be in [0, 1]."""
        with pytest.raises(ValueError, match="storage_rate_nu must be"):
            NSMSecurityParameters(storage_rate_nu=1.5)

    def test_validation_delta_t_positive(self) -> None:
        """Δt must be positive."""
        with pytest.raises(ValueError, match="delta_t_s must be"):
            NSMSecurityParameters(delta_t_s=-1.0)


# =============================================================================
# ProtocolParameters Tests
# =============================================================================


class TestProtocolParameters:
    """Tests for ProtocolParameters dataclass."""

    def test_defaults(self) -> None:
        """Default protocol values should be reasonable."""
        params = ProtocolParameters()
        # Verify it constructs without error
        assert params is not None


# =============================================================================
# ProtocolConfig Tests
# =============================================================================


class TestProtocolConfig:
    """Tests for ProtocolConfig composite dataclass."""

    def test_default_construction(self) -> None:
        """Should construct with all defaults."""
        config = ProtocolConfig()
        assert isinstance(config.physical, PhysicalParameters)
        assert isinstance(config.nsm_security, NSMSecurityParameters)
        assert isinstance(config.protocol, ProtocolParameters)


# =============================================================================
# SimulatorNoiseParams Tests
# =============================================================================


class TestSimulatorNoiseParams:
    """Tests for SimulatorNoiseParams dataclass."""

    def test_valid_params(self) -> None:
        """Valid params should not raise."""
        params = SimulatorNoiseParams(
            link_fidelity=0.99,
            measurement_bitflip_prob=0.01,
            expected_detection_prob=0.001,
        )
        assert params.link_fidelity == 0.99

    def test_validation_link_fidelity(self) -> None:
        """link_fidelity must be in [0, 1]."""
        with pytest.raises(ValueError, match="link_fidelity must be"):
            SimulatorNoiseParams(
                link_fidelity=1.5,
                measurement_bitflip_prob=0.01,
                expected_detection_prob=0.001,
            )

    def test_validation_measurement_bitflip(self) -> None:
        """measurement_bitflip_prob must be in [0, 0.5]."""
        with pytest.raises(ValueError, match="measurement_bitflip_prob must be"):
            SimulatorNoiseParams(
                link_fidelity=0.99,
                measurement_bitflip_prob=0.6,
                expected_detection_prob=0.001,
            )

    def test_validation_expected_detection(self) -> None:
        """expected_detection_prob must be in [0, 1]."""
        with pytest.raises(ValueError, match="expected_detection_prob must be"):
            SimulatorNoiseParams(
                link_fidelity=0.99,
                measurement_bitflip_prob=0.01,
                expected_detection_prob=1.5,
            )


# =============================================================================
# Physical-to-Simulator Translation Tests
# =============================================================================


class TestPhysicalToSimulator:
    """Tests for physical_to_simulator translation function."""

    def test_erven_defaults_translation(self) -> None:
        """Translation of Erven defaults should produce valid params."""
        physical = PhysicalParameters()
        sim = physical_to_simulator(physical)

        assert isinstance(sim, SimulatorNoiseParams)
        assert 0 <= sim.link_fidelity <= 1
        assert 0 <= sim.measurement_bitflip_prob <= 0.5
        assert 0 <= sim.expected_detection_prob <= 1

    def test_measurement_bitflip_equals_e_det(self) -> None:
        """Measurement bitflip should directly map to e_det."""
        physical = PhysicalParameters(e_det=0.05)
        sim = physical_to_simulator(physical)
        assert sim.measurement_bitflip_prob == 0.05

    def test_link_fidelity_from_e_det(self) -> None:
        """Link fidelity should be approximately 1 - e_det."""
        physical = PhysicalParameters(e_det=0.02)
        sim = physical_to_simulator(physical)
        # Conservative approximation: F = 1 - e_det
        assert sim.link_fidelity == pytest.approx(0.98, rel=0.01)

    def test_detection_prob_from_eta_mu(self) -> None:
        """Detection prob should include η × μ contribution."""
        physical = PhysicalParameters(
            eta_total_transmittance=0.1,
            mu_pair_per_coherence=0.01,
            p_dark=0.0,
        )
        sim = physical_to_simulator(physical)
        # P_detect ≈ η × μ = 0.001
        assert sim.expected_detection_prob == pytest.approx(0.001, rel=0.01)

    def test_detection_prob_includes_dark_count(self) -> None:
        """Detection prob should include dark count contribution."""
        physical = PhysicalParameters(
            eta_total_transmittance=0.01,
            mu_pair_per_coherence=0.001,
            p_dark=1e-5,
        )
        sim = physical_to_simulator(physical)
        # P_detect ≈ η × μ + P_dark = 1e-5 + 1e-5 = 2e-5
        assert sim.expected_detection_prob >= 1e-5


# =============================================================================
# QBER Estimation Tests
# =============================================================================


class TestEstimateQBER:
    """Tests for estimate_qber_from_physical function."""

    def test_qber_includes_e_det(self) -> None:
        """Estimated QBER should include e_det contribution."""
        physical = PhysicalParameters(e_det=0.05, p_dark=0.0)
        qber = estimate_qber_from_physical(physical)
        assert qber >= 0.05

    def test_qber_bounded_by_half(self) -> None:
        """QBER should never exceed 0.5."""
        physical = PhysicalParameters(
            e_det=0.3,
            p_dark=1e-3,  # High dark count
        )
        qber = estimate_qber_from_physical(physical)
        assert qber <= 0.5

    def test_erven_defaults_reasonable_qber(self) -> None:
        """Erven defaults should produce reasonable QBER estimate."""
        physical = PhysicalParameters()
        qber = estimate_qber_from_physical(physical)
        # Should be in reasonable range for typical experimental setup
        assert 0.0 < qber < 0.1


# =============================================================================
# Sifted Rate Estimation Tests
# =============================================================================


class TestEstimateSiftedRate:
    """Tests for estimate_sifted_rate function."""

    def test_rate_scales_with_detection(self) -> None:
        """Sifted rate should scale with detection probability."""
        physical1 = PhysicalParameters(eta_total_transmittance=0.01)
        physical2 = PhysicalParameters(eta_total_transmittance=0.02)

        rate1 = estimate_sifted_rate(physical1)
        rate2 = estimate_sifted_rate(physical2)

        # Double η should approximately double rate
        assert rate2 > rate1

    def test_rate_includes_basis_matching(self) -> None:
        """Sifted rate should account for 50% basis matching."""
        physical = PhysicalParameters(
            eta_total_transmittance=1.0,  # Perfect transmission
            mu_pair_per_coherence=1.0,  # Deterministic source
        )
        rate = estimate_sifted_rate(physical, source_rate_hz=1_000_000)
        # With perfect detection, rate = source_rate * 0.5 (basis matching)
        assert rate == pytest.approx(500_000, rel=0.01)


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestValidatePhysicalParams:
    """Tests for validate_physical_params_for_simulation function."""

    def test_erven_defaults_no_warnings(self) -> None:
        """Erven defaults should produce no warnings."""
        physical = PhysicalParameters()
        warnings = validate_physical_params_for_simulation(physical)
        assert len(warnings) == 0

    def test_very_low_detection_warning(self) -> None:
        """Very low detection rate should warn."""
        physical = PhysicalParameters(
            eta_total_transmittance=1e-10,
            mu_pair_per_coherence=1e-10,
        )
        warnings = validate_physical_params_for_simulation(physical)
        assert any("detection" in w.lower() for w in warnings)

    def test_high_e_det_warning(self) -> None:
        """High e_det should warn about security limits."""
        physical = PhysicalParameters(e_det=0.15)
        warnings = validate_physical_params_for_simulation(physical)
        assert any("error" in w.lower() or "e_det" in w for w in warnings)

    def test_high_mu_warning(self) -> None:
        """High μ should warn about multi-photon emissions."""
        physical = PhysicalParameters(mu_pair_per_coherence=0.2)
        warnings = validate_physical_params_for_simulation(physical)
        assert any("μ" in w or "multi-photon" in w.lower() for w in warnings)

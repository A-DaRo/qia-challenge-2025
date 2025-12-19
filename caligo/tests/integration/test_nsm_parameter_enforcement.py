"""
Integration tests for NSM parameter physical enforcement.

These tests validate that NSM parameters flow correctly through the
SquidASM/netsquid-netbuilder stack and produce the expected physical
noise models in the simulation.

Test Categories:
- Test A: Stack-runner config conversion preserves fields
- Test B: Depolarise link builds correct MagicDistributor
- Test C: Heralded-double-click installs detector and dark count params
- Test D: Device noise is installed into QuantumProcessor

References
----------
- nsm_parameters_enforcement.md Section 7.5.4
- netsquid_netbuilder/modules/qlinks/: Link builders
- netsquid_magic/magic_distributor.py: Distributor classes
"""

from __future__ import annotations

import math

import pytest

# Skip entire module if SquidASM or NetSquid not available
squidasm = pytest.importorskip("squidasm")
netsquid = pytest.importorskip("netsquid")

from squidasm.run.stack.config import (
    _convert_stack_network_config,
    StackNetworkConfig,
    StackConfig,
    LinkConfig,
    GenericQDeviceConfig,
    DepolariseLinkConfig,
    HeraldedLinkConfig,
)
from squidasm.run.stack.build import create_stack_network_builder

from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel
from netsquid_magic.magic_distributor import (
    DepolariseWithFailureMagicDistributor,
    DoubleClickMagicDistributor,
)
from netsquid_netbuilder.util.fidelity import fidelity_to_prob_max_mixed

from caligo.simulation.network_builder import CaligoNetworkBuilder
from caligo.simulation.physical_model import (
    NSMParameters,
    ChannelParameters,
    ChannelModelSelection,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nsm_params_depolarise() -> NSMParameters:
    """NSM parameters for depolarise link testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.01,
        delta_t_ns=1_000_000,
        channel_fidelity=0.92,
        detection_eff_eta=1.0,  # Ideal detection for depolarise
        detector_error=0.0,
        dark_count_prob=0.0,
    )


@pytest.fixture
def nsm_params_heralded() -> NSMParameters:
    """NSM parameters for heralded link testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.01,
        delta_t_ns=1_000_000,
        channel_fidelity=0.95,
        detection_eff_eta=0.90,
        detector_error=0.01,
        dark_count_prob=1e-5,
    )


@pytest.fixture
def channel_params() -> ChannelParameters:
    """Standard channel parameters for testing."""
    return ChannelParameters(
        length_km=0.0,
        attenuation_db_per_km=0.2,
        speed_of_light_km_s=200_000.0,
        t1_ns=20_000_000,
        t2_ns=2_000_000,
        cycle_time_ns=10_000,
    )


# =============================================================================
# Test A: Stack-runner config conversion preserves fields
# =============================================================================


@pytest.mark.integration
class TestStackRunnerConfigConversion:
    """Test that Caligo configs survive stack-runner conversion."""

    def test_depolarise_config_preserves_fidelity(
        self, nsm_params_depolarise: NSMParameters
    ) -> None:
        """Depolarise link config should preserve fidelity field."""
        model_selection = ChannelModelSelection(link_model="depolarise")
        builder = CaligoNetworkBuilder(
            nsm_params_depolarise,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=4)

        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        assert len(netbuilder_cfg.qlinks) == 1
        qlink = netbuilder_cfg.qlinks[0]
        assert qlink.typ == "depolarise"

        # cfg may be a dict or DepolariseQLinkConfig instance
        cfg = qlink.cfg
        if isinstance(cfg, dict):
            fidelity = cfg["fidelity"]
            prob_success = cfg.get("prob_success", 1.0)
            t_cycle = cfg.get("t_cycle", 0)
        else:
            fidelity = cfg.fidelity
            prob_success = cfg.prob_success
            t_cycle = cfg.t_cycle

        assert math.isclose(float(fidelity), 0.92)
        assert math.isclose(float(prob_success), 1.0)
        assert float(t_cycle) > 0

    def test_heralded_config_preserves_detector_fields(
        self,
        nsm_params_heralded: NSMParameters,
        channel_params: ChannelParameters,
    ) -> None:
        """Heralded link config should preserve detector_efficiency and dark_count."""
        model_selection = ChannelModelSelection(link_model="heralded-double-click")
        builder = CaligoNetworkBuilder(
            nsm_params_heralded,
            channel_params,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=4)

        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        assert len(netbuilder_cfg.qlinks) == 1
        qlink = netbuilder_cfg.qlinks[0]
        assert qlink.typ == "heralded-double-click"

        cfg = qlink.cfg
        if isinstance(cfg, dict):
            detector_eff = cfg["detector_efficiency"]
            dark_count = cfg["dark_count_probability"]
            emission_fid = cfg.get("emission_fidelity", 1.0)
        else:
            detector_eff = cfg.detector_efficiency
            dark_count = cfg.dark_count_probability
            emission_fid = cfg.emission_fidelity

        assert math.isclose(float(detector_eff), 0.90)
        assert math.isclose(float(dark_count), 1e-5, rel_tol=1e-6)
        assert math.isclose(float(emission_fid), 0.95)


# =============================================================================
# Test B: Depolarise link builds correct MagicDistributor
# =============================================================================


@pytest.mark.integration
class TestDepolariseLinkBuild:
    """Test that depolarise link builds expected MagicDistributor."""

    def test_depolarise_installs_expected_magic_model_params(
        self, nsm_params_depolarise: NSMParameters
    ) -> None:
        """Depolarise link should create DepolariseWithFailureMagicDistributor."""
        model_selection = ChannelModelSelection(link_model="depolarise")
        builder = CaligoNetworkBuilder(
            nsm_params_depolarise,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=2)
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        network_builder = create_stack_network_builder()
        network = network_builder.build(netbuilder_cfg)

        # Access the link
        link = network.qlinks[("Alice", "Bob")]
        dist = link.magic_distributor

        assert isinstance(dist, DepolariseWithFailureMagicDistributor)

        # Access model parameters
        model = dist._model_parameters[0]

        # Verify prob_success
        assert math.isclose(float(model.prob_success), 1.0)

        # Verify prob_max_mixed matches fidelity conversion
        expected_prob_max_mixed = fidelity_to_prob_max_mixed(0.92)
        assert math.isclose(
            float(model.prob_max_mixed),
            expected_prob_max_mixed,
            rel_tol=1e-6,
        )

    def test_depolarise_auto_selection(
        self, nsm_params_depolarise: NSMParameters
    ) -> None:
        """Auto selection should choose depolarise when η=1 and P_dark=0."""
        model_selection = ChannelModelSelection(link_model="auto")
        builder = CaligoNetworkBuilder(
            nsm_params_depolarise,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=2)
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        # Should select depolarise
        assert netbuilder_cfg.qlinks[0].typ == "depolarise"


# =============================================================================
# Test C: Heralded-double-click installs detector and dark count params
# =============================================================================


@pytest.mark.integration
class TestHeraldedLinkBuild:
    """Test that heralded-double-click link builds correct MagicDistributor."""

    def test_heralded_installs_detector_and_dark_counts(
        self,
        nsm_params_heralded: NSMParameters,
        channel_params: ChannelParameters,
    ) -> None:
        """Heralded link should create DoubleClickMagicDistributor with params."""
        model_selection = ChannelModelSelection(link_model="heralded-double-click")
        builder = CaligoNetworkBuilder(
            nsm_params_heralded,
            channel_params,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=2)
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        network_builder = create_stack_network_builder()
        network = network_builder.build(netbuilder_cfg)

        link = network.qlinks[("Alice", "Bob")]
        dist = link.magic_distributor

        assert isinstance(dist, DoubleClickMagicDistributor)

        model = dist._model_parameters[0]

        # Verify detector efficiency
        assert math.isclose(float(model.detector_efficiency), 0.90)

        # Verify dark count probability
        assert math.isclose(float(model.dark_count_probability), 1e-5, rel_tol=1e-6)

    def test_heralded_auto_selection_with_low_eta(
        self,
        nsm_params_heralded: NSMParameters,
        channel_params: ChannelParameters,
    ) -> None:
        """Auto selection should choose heralded when η < 1."""
        model_selection = ChannelModelSelection(link_model="auto")
        builder = CaligoNetworkBuilder(
            nsm_params_heralded,
            channel_params,
            model_selection=model_selection,
        )
        stack_cfg = builder.build_two_node_network(num_qubits=2)
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        # Should select heralded (converted to heralded-double-click)
        assert netbuilder_cfg.qlinks[0].typ == "heralded-double-click"

    def test_heralded_with_dark_count_only(self) -> None:
        """Auto selection should choose heralded when P_dark > 0 even with η=1."""
        nsm_params = NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.01,
            delta_t_ns=1_000_000,
            channel_fidelity=0.95,
            detection_eff_eta=1.0,  # Ideal detection
            detector_error=0.0,
            dark_count_prob=1e-6,  # But has dark counts
        )
        model_selection = ChannelModelSelection(link_model="auto")
        builder = CaligoNetworkBuilder(nsm_params, model_selection=model_selection)
        stack_cfg = builder.build_two_node_network(num_qubits=2)
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        # Should select heralded due to dark counts
        assert netbuilder_cfg.qlinks[0].typ == "heralded-double-click"


# =============================================================================
# Test D: Device noise is installed into QuantumProcessor
# =============================================================================


@pytest.mark.integration
class TestDeviceNoiseInstallation:
    """Test that device noise is installed into NetSquid QuantumProcessor."""

    def test_t1t2_noise_installed_in_memory(
        self,
        nsm_params_depolarise: NSMParameters,
        channel_params: ChannelParameters,
    ) -> None:
        """T1T2 noise should be installed when with_device_noise=True."""
        builder = CaligoNetworkBuilder(nsm_params_depolarise, channel_params)
        stack_cfg = builder.build_two_node_network(
            num_qubits=3,
            with_device_noise=True,
        )
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        network_builder = create_stack_network_builder()
        network = network_builder.build(netbuilder_cfg)

        # Check Alice's qmemory
        qproc = network.end_nodes["Alice"].qmemory
        assert qproc is not None

        # Memory noise should be T1T2NoiseModel (stored in pos.models)
        # NetSquid stores noise models in memory positions
        pos = qproc.mem_positions[0]
        assert pos is not None
        assert hasattr(pos, "models")
        noise_model = pos.models.get("noise_model")
        assert noise_model is not None, "No noise_model found in memory position"
        assert isinstance(noise_model, T1T2NoiseModel)

        # Verify T1/T2 values match channel_params
        assert math.isclose(float(noise_model.T1), channel_params.t1_ns)
        assert math.isclose(float(noise_model.T2), channel_params.t2_ns)

    def test_gate_depolar_installed(
        self,
        nsm_params_heralded: NSMParameters,
        channel_params: ChannelParameters,
    ) -> None:
        """Gate depolarization should be installed when with_device_noise=True."""
        builder = CaligoNetworkBuilder(nsm_params_heralded, channel_params)
        stack_cfg = builder.build_two_node_network(
            num_qubits=3,
            with_device_noise=True,
        )
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        network_builder = create_stack_network_builder()
        network = network_builder.build(netbuilder_cfg)

        qproc = network.end_nodes["Alice"].qmemory
        assert qproc is not None

        # Gate depolarization stored in _phys_instrs[-1][gate_name].q_noise_model
        inner_dict = qproc._phys_instrs.get(-1, {})
        assert len(inner_dict) > 0, "No physical instructions found"

        # Find gates with DepolarNoiseModel
        depolar_models = []
        for gate_name, phys_instr in inner_dict.items():
            if hasattr(phys_instr, "q_noise_model"):
                noise_model = phys_instr.q_noise_model
                if isinstance(noise_model, DepolarNoiseModel):
                    depolar_models.append(noise_model)

        assert len(depolar_models) > 0, "No DepolarNoiseModel found in gates"

        # Expected gate depolar from detector_error mapping
        # _map_detector_error_to_gate_depolar(0.01) = 0.02
        expected_depolar = 0.02
        assert any(
            math.isclose(float(m.depolar_rate), expected_depolar, rel_tol=1e-6)
            for m in depolar_models
        ), f"Expected depolar_rate {expected_depolar}, got {[m.depolar_rate for m in depolar_models]}"

    def test_no_noise_without_flag(
        self, nsm_params_depolarise: NSMParameters
    ) -> None:
        """No device noise should be installed when with_device_noise=False."""
        builder = CaligoNetworkBuilder(nsm_params_depolarise)
        stack_cfg = builder.build_two_node_network(
            num_qubits=3,
            with_device_noise=False,
        )
        netbuilder_cfg = _convert_stack_network_config(stack_cfg)

        network_builder = create_stack_network_builder()
        network = network_builder.build(netbuilder_cfg)

        qproc = network.end_nodes["Alice"].qmemory

        # Check for perfect config indicators
        # T1=0, T2=0 disables T1T2 noise in NetSquid
        qdevice_cfg = stack_cfg.stacks[0].qdevice_cfg
        assert qdevice_cfg.T1 == 0
        assert qdevice_cfg.T2 == 0
        assert qdevice_cfg.single_qubit_gate_depolar_prob == 0

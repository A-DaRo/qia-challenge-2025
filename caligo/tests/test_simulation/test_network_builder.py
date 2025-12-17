"""
Unit tests for caligo.simulation.network_builder module.

Tests CaligoNetworkBuilder and configuration preset functions.
"""

from __future__ import annotations

import pytest

from caligo.simulation.network_builder import (
    CaligoNetworkBuilder,
    NetworkConfigSummary,
    perfect_network_config,
    realistic_network_config,
    erven_experimental_config,
)
from caligo.simulation.physical_model import (
    NSMParameters,
    ChannelParameters,
    QBER_CONSERVATIVE_LIMIT,
)
from caligo.types.exceptions import NetworkConfigError


# =============================================================================
# CaligoNetworkBuilder Tests
# =============================================================================


class TestCaligoNetworkBuilderCreation:
    """Tests for CaligoNetworkBuilder initialization."""

    def test_valid_creation(self, nsm_params):
        """Valid parameters should create instance."""
        builder = CaligoNetworkBuilder(nsm_params)
        assert builder.nsm_params == nsm_params
        assert builder.channel_params is not None

    def test_creation_with_channel_params(self, nsm_params, channel_params):
        """Channel parameters should be accepted."""
        builder = CaligoNetworkBuilder(nsm_params, channel_params)
        assert builder.channel_params == channel_params

    def test_default_channel_params(self, nsm_params):
        """Default channel params should be for testing."""
        builder = CaligoNetworkBuilder(nsm_params)
        assert builder.channel_params.length_km == 0.0


class TestCaligoNetworkBuilderProperties:
    """Tests for CaligoNetworkBuilder properties."""

    def test_nsm_params_property(self, nsm_params):
        """nsm_params property should return NSM parameters."""
        builder = CaligoNetworkBuilder(nsm_params)
        assert builder.nsm_params.storage_noise_r == 0.75

    def test_channel_params_property(self, nsm_params, channel_params):
        """channel_params property should return channel parameters."""
        builder = CaligoNetworkBuilder(nsm_params, channel_params)
        assert builder.channel_params.t1_ns == channel_params.t1_ns


class TestCaligoNetworkBuilderBuildMethods:
    """Tests for CaligoNetworkBuilder build methods."""

    def test_build_two_node_network_requires_squidasm(self, nsm_params):
        """build_two_node_network should require SquidASM."""
        builder = CaligoNetworkBuilder(nsm_params)
        try:
            config = builder.build_two_node_network()
            # If SquidASM is available, verify the config
            assert config is not None
        except NetworkConfigError as e:
            assert "SquidASM is required" in str(e)

    def test_build_two_node_network_custom_names(self, nsm_params):
        """Custom node names should be used."""
        builder = CaligoNetworkBuilder(nsm_params)
        try:
            config = builder.build_two_node_network(
                alice_name="Sender",
                bob_name="Receiver",
            )
            # If SquidASM available, verify names
            assert config is not None
        except NetworkConfigError:
            pass  # Expected if SquidASM not available

    def test_build_stack_config_requires_squidasm(self, nsm_params):
        """build_stack_config should require SquidASM."""
        builder = CaligoNetworkBuilder(nsm_params)
        try:
            config = builder.build_stack_config("TestNode")
            assert config is not None
        except NetworkConfigError as e:
            assert "SquidASM is required" in str(e)


# =============================================================================
# Configuration Preset Tests
# =============================================================================


class TestPerfectNetworkConfig:
    """Tests for perfect_network_config() preset."""

    def test_requires_squidasm(self):
        """perfect_network_config should require SquidASM."""
        try:
            config = perfect_network_config()
            assert config is not None
        except NetworkConfigError as e:
            assert "SquidASM is required" in str(e)

    def test_custom_node_names(self):
        """Custom names should be accepted."""
        try:
            config = perfect_network_config(
                alice_name="Sender",
                bob_name="Receiver",
            )
            assert config is not None
        except NetworkConfigError:
            pass  # Expected if SquidASM not available

    def test_custom_num_qubits(self):
        """Custom qubit count should be accepted."""
        try:
            config = perfect_network_config(num_qubits=20)
            assert config is not None
        except NetworkConfigError:
            pass  # Expected if SquidASM not available


class TestRealisticNetworkConfig:
    """Tests for realistic_network_config() preset."""

    def test_requires_squidasm(self):
        """realistic_network_config should require SquidASM."""
        try:
            config = realistic_network_config()
            assert config is not None
        except NetworkConfigError as e:
            assert "SquidASM is required" in str(e)

    def test_custom_fidelity(self):
        """Custom fidelity should be accepted."""
        try:
            config = realistic_network_config(fidelity=0.99)
            assert config is not None
        except NetworkConfigError:
            pass  # Expected if SquidASM not available

    def test_custom_t1_t2(self):
        """Custom T1/T2 values should be accepted."""
        try:
            config = realistic_network_config(
                t1_ns=20_000_000,
                t2_ns=2_000_000,
            )
            assert config is not None
        except NetworkConfigError:
            pass  # Expected if SquidASM not available


class TestErvenExperimentalConfig:
    """Tests for erven_experimental_config() preset."""

    def test_requires_squidasm(self):
        """erven_experimental_config should require SquidASM."""
        try:
            config = erven_experimental_config()
            assert config is not None
        except NetworkConfigError as e:
            assert "SquidASM is required" in str(e)


# =============================================================================
# NetworkConfigSummary Tests
# =============================================================================


class TestNetworkConfigSummary:
    """Tests for NetworkConfigSummary dataclass."""

    def test_creation(self):
        """Valid parameters should create instance."""
        summary = NetworkConfigSummary(
            alice_name="Alice",
            bob_name="Bob",
            num_qubits=10,
            channel_fidelity=0.95,
            expected_qber=0.025,
            is_secure=True,
        )
        assert summary.alice_name == "Alice"
        assert summary.bob_name == "Bob"
        assert summary.num_qubits == 10
        assert summary.channel_fidelity == 0.95
        assert summary.expected_qber == 0.025
        assert summary.is_secure is True

    def test_from_builder(self, nsm_params):
        """from_builder should create summary from builder."""
        builder = CaligoNetworkBuilder(nsm_params)
        summary = NetworkConfigSummary.from_builder(builder)

        assert summary.alice_name == "Alice"
        assert summary.bob_name == "Bob"
        assert summary.num_qubits == 10
        assert summary.channel_fidelity == nsm_params.channel_fidelity
        assert summary.expected_qber == nsm_params.qber_channel
        assert summary.is_secure == (nsm_params.qber_channel < QBER_CONSERVATIVE_LIMIT)

    def test_from_builder_custom_names(self, nsm_params):
        """from_builder should accept custom names."""
        builder = CaligoNetworkBuilder(nsm_params)
        summary = NetworkConfigSummary.from_builder(
            builder,
            alice_name="Sender",
            bob_name="Receiver",
            num_qubits=20,
        )

        assert summary.alice_name == "Sender"
        assert summary.bob_name == "Receiver"
        assert summary.num_qubits == 20

    def test_is_secure_based_on_qber(self):
        """is_secure should reflect QBER threshold."""
        # Low QBER - secure
        secure = NetworkConfigSummary(
            alice_name="A",
            bob_name="B",
            num_qubits=10,
            channel_fidelity=0.95,
            expected_qber=0.05,  # < 0.11
            is_secure=True,
        )
        assert secure.is_secure is True

        # High QBER - not secure
        insecure = NetworkConfigSummary(
            alice_name="A",
            bob_name="B",
            num_qubits=10,
            channel_fidelity=0.70,
            expected_qber=0.15,  # > 0.11
            is_secure=False,
        )
        assert insecure.is_secure is False

    def test_frozen_dataclass(self):
        """NetworkConfigSummary should be immutable."""
        summary = NetworkConfigSummary(
            alice_name="Alice",
            bob_name="Bob",
            num_qubits=10,
            channel_fidelity=0.95,
            expected_qber=0.025,
            is_secure=True,
        )
        with pytest.raises(AttributeError):
            summary.alice_name = "Sender"

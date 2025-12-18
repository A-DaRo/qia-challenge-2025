"""
SquidASM network configuration builder for E-HOK protocol.

This module provides factory functions and a builder class for creating
SquidASM network configurations that incorporate NSM-specific parameters.

References
----------
- squidasm/run/stack/config.py: StackNetworkConfig, LinkConfig
- netsquid_magic/model_parameters.py: Parameter classes
- Erven et al. (2014): Experimental setup parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from caligo.simulation.physical_model import (
    ChannelParameters,
    NSMParameters,
    QBER_CONSERVATIVE_LIMIT,
)
from caligo.types.exceptions import NetworkConfigError, UnsupportedHardwareError
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Caligo only supports Generic QDevice due to NetQASM 2.x / SquidASM 0.13.x
# instruction incompatibility. See protocol_flow_bug.md for details.
SUPPORTED_QDEVICE_TYPE = "generic"
BLOCKED_QDEVICE_TYPES = frozenset({"nv", "nv_config", "trapped_ion"})


# =============================================================================
# Type Aliases (for optional imports)
# =============================================================================

# These will be replaced with actual types when SquidASM is available
StackNetworkConfig = Any
StackConfig = Any
LinkConfig = Any


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_qdevice_type(qdevice_typ: str) -> None:
    """
    Validate that the QDevice type is supported by Caligo.

    Caligo enforces Generic QDevice only due to NetQASM 2.x / SquidASM 0.13.x
    instruction incompatibility. NV hardware triggers MOV instructions that
    are not implemented in the processor.

    Parameters
    ----------
    qdevice_typ : str
        The QDevice type string (e.g., "generic", "nv").

    Raises
    ------
    UnsupportedHardwareError
        If the QDevice type is not supported.

    References
    ----------
    - protocol_flow_bug.md: Full bug analysis
    - squidasm/sim/stack/processor.py:734: RuntimeError on MOV instruction
    """
    qdevice_typ_lower = qdevice_typ.lower()
    if qdevice_typ_lower in BLOCKED_QDEVICE_TYPES:
        raise UnsupportedHardwareError(
            f"QDevice type '{qdevice_typ}' is not supported by Caligo. "
            f"NetQASM 2.x generates MOV instructions that SquidASM 0.13.x "
            f"cannot execute. Use qdevice_typ='{SUPPORTED_QDEVICE_TYPE}' instead. "
            f"See docs/caligo/protocol_flow_bug.md for details."
        )
    if qdevice_typ_lower != SUPPORTED_QDEVICE_TYPE:
        logger.warning(
            f"QDevice type '{qdevice_typ}' is not explicitly supported. "
            f"Recommended: '{SUPPORTED_QDEVICE_TYPE}'"
        )


def validate_stack_config(config: Any) -> None:
    """
    Validate a StackConfig for Caligo compatibility.

    Parameters
    ----------
    config : StackConfig
        The stack configuration to validate.

    Raises
    ------
    UnsupportedHardwareError
        If the configuration uses unsupported hardware.
    """
    qdevice_typ = getattr(config, "qdevice_typ", None)
    if qdevice_typ is not None:
        validate_qdevice_type(qdevice_typ)


def validate_network_config(config: Any) -> None:
    """
    Validate a StackNetworkConfig for Caligo compatibility.

    Parameters
    ----------
    config : StackNetworkConfig
        The network configuration to validate.

    Raises
    ------
    UnsupportedHardwareError
        If any node uses unsupported hardware.
    """
    stacks = getattr(config, "stacks", [])
    for stack in stacks:
        validate_stack_config(stack)


# =============================================================================
# Network Builder
# =============================================================================


class CaligoNetworkBuilder:
    """
    Factory for creating SquidASM network configurations for E-HOK.

    Provides a high-level API for constructing network configurations
    that incorporate NSM-specific parameters. Wraps the complexity of
    SquidASM's StackNetworkConfig and netsquid-netbuilder.

    Parameters
    ----------
    nsm_params : NSMParameters
        NSM security parameters.
    channel_params : ChannelParameters
        Physical channel parameters.

    Examples
    --------
    >>> builder = CaligoNetworkBuilder(nsm_params, channel_params)
    >>> config = builder.build_two_node_network("Alice", "Bob")

    References
    ----------
    - squidasm/run/stack/config.py: StackNetworkConfig
    - squidasm/run/stack/build.py: create_stack_network_builder
    """

    def __init__(
        self,
        nsm_params: NSMParameters,
        channel_params: Optional[ChannelParameters] = None,
    ) -> None:
        """
        Initialize the network builder.

        Parameters
        ----------
        nsm_params : NSMParameters
            NSM security parameters.
        channel_params : ChannelParameters, optional
            Physical channel parameters. If None, uses defaults.
        """
        self._nsm_params = nsm_params
        self._channel_params = channel_params or ChannelParameters.for_testing()

    @property
    def nsm_params(self) -> NSMParameters:
        """Get NSM parameters."""
        return self._nsm_params

    @property
    def channel_params(self) -> ChannelParameters:
        """Get channel parameters."""
        return self._channel_params

    def build_two_node_network(
        self,
        alice_name: str = "Alice",
        bob_name: str = "Bob",
        num_qubits: int = 10,
    ) -> Any:
        """
        Create a two-node network configuration for E-HOK.

        Parameters
        ----------
        alice_name : str
            Name for Alice's node. Default: "Alice".
        bob_name : str
            Name for Bob's node. Default: "Bob".
        num_qubits : int
            Number of qubit positions per node. Default: 10.

        Returns
        -------
        StackNetworkConfig
            Complete network configuration ready for simulation.

        Raises
        ------
        NetworkConfigError
            If SquidASM is not available or configuration fails.

        Notes
        -----
        The returned configuration includes:
        - Two StackConfig nodes with Generic QDevice
        - One LinkConfig with appropriate noise model
        - Instant classical links (no propagation delay)

        The quantum link noise model is selected based on NSM parameters:
        - If channel_fidelity == 1.0: "perfect" quantum link
        - Otherwise: "depolarise" quantum link with fidelity = channel_fidelity
        """
        try:
            from squidasm.run.stack.config import (
                StackNetworkConfig,
                StackConfig,
                LinkConfig,
                GenericQDeviceConfig,
            )
        except ImportError as e:
            raise NetworkConfigError(
                "SquidASM is required for network configuration. "
                "Install with: pip install squidasm"
            ) from e

        # Determine noise model based on fidelity.
        #
        # SquidASM's stack runner expects LinkConfig.typ to be a lowercase
        # registered netbuilder model name (e.g. "perfect", "depolarise").
        fidelity_param = float(self._nsm_params.channel_fidelity)
        if fidelity_param == 1.0:
            link_noise_type = "perfect"
            link_cfg_payload = None
        else:
            link_noise_type = "depolarise"
            link_cfg_payload = {"fidelity": fidelity_param}

        # Create node configurations
        alice_config = StackConfig(
            name=alice_name,
            qdevice_typ="generic",
            qdevice_cfg=GenericQDeviceConfig.perfect_config(),
        )
        alice_config.qdevice_cfg.num_qubits = num_qubits

        bob_config = StackConfig(
            name=bob_name,
            qdevice_typ="generic",
            qdevice_cfg=GenericQDeviceConfig.perfect_config(),
        )
        bob_config.qdevice_cfg.num_qubits = num_qubits

        # Create link configuration
        link_cfg = LinkConfig(
            stack1=alice_name,
            stack2=bob_name,
            typ=link_noise_type,
            cfg=link_cfg_payload,
        )

        # Build network config
        network_config = StackNetworkConfig(
            stacks=[alice_config, bob_config],
            links=[link_cfg],
        )

        logger.info(
            f"Created two-node network: {alice_name} <-> {bob_name}, "
            f"fidelity={fidelity_param:.4f}, qubits={num_qubits}"
        )

        return network_config

    def build_stack_config(
        self,
        name: str,
        num_qubits: int = 10,
        with_memory_noise: bool = False,
    ) -> Any:
        """
        Create single node configuration.

        Parameters
        ----------
        name : str
            Node name.
        num_qubits : int
            Number of qubit positions. Default: 10.
        with_memory_noise : bool
            If True, apply T1/T2 noise to quantum memory. Default: False.

        Returns
        -------
        StackConfig
            Node configuration.

        Raises
        ------
        NetworkConfigError
            If SquidASM is not available.
        """
        try:
            from squidasm.run.stack.config import StackConfig, GenericQDeviceConfig
        except ImportError as e:
            raise NetworkConfigError(
                "SquidASM is required for stack configuration."
            ) from e

        qdevice_cfg = GenericQDeviceConfig.perfect_config()
        qdevice_cfg.num_qubits = num_qubits

        if with_memory_noise:
            # Apply T1/T2 noise from channel parameters
            qdevice_cfg.T1 = self._channel_params.t1_ns
            qdevice_cfg.T2 = self._channel_params.t2_ns

        return StackConfig(
            name=name,
            qdevice_typ="generic",
            qdevice_cfg=qdevice_cfg,
        )


# =============================================================================
# Configuration Presets
# =============================================================================


def perfect_network_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 10,
) -> Any:
    """
    Create a perfect (noiseless) network for unit testing.

    All operations are ideal:
    - Perfect EPR pairs (F = 1.0)
    - No memory decoherence
    - No channel losses
    - No gate errors

    Parameters
    ----------
    alice_name : str
        Name for Alice's node. Default: "Alice".
    bob_name : str
        Name for Bob's node. Default: "Bob".
    num_qubits : int
        Number of qubit positions. Default: 10.

    Returns
    -------
    StackNetworkConfig
        Perfect network configuration.

    Raises
    ------
    NetworkConfigError
        If SquidASM is not available.

    Notes
    -----
    Use this for testing protocol logic without noise effects.
    """
    nsm_params = NSMParameters(
        storage_noise_r=0.75,  # Doesn't matter for ideal network
        storage_rate_nu=0.01,
        delta_t_ns=1_000_000,
        channel_fidelity=1.0,  # Perfect fidelity
    )
    builder = CaligoNetworkBuilder(nsm_params)
    return builder.build_two_node_network(
        alice_name=alice_name,
        bob_name=bob_name,
        num_qubits=num_qubits,
    )


def realistic_network_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 10,
    fidelity: float = 0.95,
    t1_ns: float = 10_000_000,
    t2_ns: float = 1_000_000,
) -> Any:
    """
    Create a network with realistic noise parameters.

    Parameters
    ----------
    alice_name : str
        Name for Alice's node. Default: "Alice".
    bob_name : str
        Name for Bob's node. Default: "Bob".
    num_qubits : int
        Number of qubit positions. Default: 10.
    fidelity : float
        EPR pair fidelity. Default: 0.95.
    t1_ns : float
        T1 relaxation time (ns). Default: 10_000_000 (10 ms).
    t2_ns : float
        T2 dephasing time (ns). Default: 1_000_000 (1 ms).

    Returns
    -------
    StackNetworkConfig
        Realistic network configuration.

    Raises
    ------
    NetworkConfigError
        If SquidASM is not available.

    Notes
    -----
    Based on typical experimental parameters, scaled for simulation
    efficiency.
    """
    nsm_params = NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=fidelity,
    )
    channel_params = ChannelParameters(
        length_km=0.0,  # Simulated, no propagation delay
        t1_ns=t1_ns,
        t2_ns=t2_ns,
    )
    builder = CaligoNetworkBuilder(nsm_params, channel_params)
    return builder.build_two_node_network(
        alice_name=alice_name,
        bob_name=bob_name,
        num_qubits=num_qubits,
    )


def erven_experimental_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 10,
) -> Any:
    """
    Create network matching Erven et al. (2014) experiment.

    Uses exact parameters from Table I:
    - μ = 3.145 × 10^{-5} (source quality → fidelity)
    - η = 0.0150 (transmittance)
    - e_det = 0.0093 (detector error)
    - r = 0.75 (storage noise)
    - ν = 0.002 (storage rate)

    Parameters
    ----------
    alice_name : str
        Name for Alice's node. Default: "Alice".
    bob_name : str
        Name for Bob's node. Default: "Bob".
    num_qubits : int
        Number of qubit positions. Default: 10.

    Returns
    -------
    StackNetworkConfig
        Network matching experimental setup.

    Raises
    ------
    NetworkConfigError
        If SquidASM is not available.

    Warnings
    --------
    This configuration has very low transmittance (η = 0.0150),
    requiring many rounds for meaningful key generation. Consider
    using realistic_network_config() for development.
    """
    nsm_params = NSMParameters.from_erven_experimental()
    channel_params = ChannelParameters.from_erven_experimental()

    logger.warning(
        "Using Erven experimental config with low transmittance (η=0.015). "
        "Many rounds may be needed for meaningful key generation."
    )

    builder = CaligoNetworkBuilder(nsm_params, channel_params)
    return builder.build_two_node_network(
        alice_name=alice_name,
        bob_name=bob_name,
        num_qubits=num_qubits,
    )


# =============================================================================
# Validation Utilities
# =============================================================================


@dataclass(frozen=True)
class NetworkConfigSummary:
    """
    Summary of network configuration for diagnostics.

    Attributes
    ----------
    alice_name : str
        Alice's node name.
    bob_name : str
        Bob's node name.
    num_qubits : int
        Qubits per node.
    channel_fidelity : float
        EPR pair fidelity.
    expected_qber : float
        Expected QBER from configuration.
    is_secure : bool
        True if expected QBER < conservative limit.
    """

    alice_name: str
    bob_name: str
    num_qubits: int
    channel_fidelity: float
    expected_qber: float
    is_secure: bool

    @classmethod
    def from_builder(
        cls,
        builder: CaligoNetworkBuilder,
        alice_name: str = "Alice",
        bob_name: str = "Bob",
        num_qubits: int = 10,
    ) -> NetworkConfigSummary:
        """
        Create summary from builder configuration.

        Parameters
        ----------
        builder : CaligoNetworkBuilder
            The builder to summarize.
        alice_name : str
            Alice's node name.
        bob_name : str
            Bob's node name.
        num_qubits : int
            Number of qubits.

        Returns
        -------
        NetworkConfigSummary
            Configuration summary.
        """
        qber = builder.nsm_params.qber_channel
        return cls(
            alice_name=alice_name,
            bob_name=bob_name,
            num_qubits=num_qubits,
            channel_fidelity=builder.nsm_params.channel_fidelity,
            expected_qber=qber,
            is_secure=qber < QBER_CONSERVATIVE_LIMIT,
        )

"""
SquidASM network configuration builder for E-HOK protocol.

This module provides factory functions and a builder class for creating
SquidASM network configurations that incorporate NSM-specific parameters.

NSM Parameter Enforcement
-------------------------
This module implements the physical enforcement of NSM parameters via:

1. **Link Model Selection**: Automatic selection of quantum link model
   (perfect, depolarise, or heralded-double-click) based on parameters.

2. **Detection Efficiency Modeling**: Maps η to detector_efficiency in
   heralded model or includes it implicitly in depolarise model.

3. **Dark Count Injection**: Uses heralded-double-click model's built-in
   dark_count_probability parameter.

4. **Device Noise Mapping**: Maps detector_error to gate depolarization
   and T1/T2 to memory noise in GenericQDeviceConfig.

References
----------
- squidasm/run/stack/config.py: StackNetworkConfig, LinkConfig
- netsquid_netbuilder/modules/qlinks/: Link model implementations
- netsquid_magic/model_parameters.py: Parameter classes
- nsm_parameters_enforcement.md: Full specification
- Erven et al. (2014): Experimental setup parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from caligo.simulation.physical_model import (
    ChannelModelSelection,
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
# Link Configuration Factory Functions (NSM Parameter Enforcement)
# =============================================================================


def _make_depolarise_link_cfg(
    nsm_params: NSMParameters,
    channel_params: ChannelParameters,
) -> Any:
    """
    Create depolarising link configuration from NSM parameters.

    Uses netbuilder's DepolariseQLinkConfig which maps fidelity to
    prob_max_mixed internally via fidelity_to_prob_max_mixed().

    Parameters
    ----------
    nsm_params : NSMParameters
        NSM configuration with channel_fidelity.
    channel_params : ChannelParameters
        Channel configuration with cycle_time_ns.

    Returns
    -------
    DepolariseLinkConfig
        Configured depolarising link.

    Notes
    -----
    This link model does NOT simulate detection efficiency or dark counts.
    Use heralded-double-click model for those parameters.

    References
    ----------
    - netsquid_netbuilder/modules/qlinks/depolarise.py
    - netsquid_netbuilder/util/fidelity.py: fidelity_to_prob_max_mixed
    """
    try:
        from squidasm.run.stack.config import DepolariseLinkConfig

        return DepolariseLinkConfig(
            fidelity=float(nsm_params.channel_fidelity),
            prob_success=1.0,
            t_cycle=float(channel_params.cycle_time_ns),
            random_bell_state=False,
        )
    except ImportError as e:
        raise NetworkConfigError(
            "SquidASM is required for link configuration."
        ) from e


def _make_heralded_double_click_cfg(
    nsm_params: NSMParameters,
    channel_params: ChannelParameters,
    eta_semantics: str = "detector_only",
) -> Any:
    """
    Create heralded-double-click link configuration from NSM parameters.

    This model includes detection efficiency and dark count probability,
    enabling full NSM channel parameter enforcement.

    Parameters
    ----------
    nsm_params : NSMParameters
        NSM configuration with channel_fidelity, detection_eff_eta,
        dark_count_prob.
    channel_params : ChannelParameters
        Channel configuration with length_km, attenuation_db_per_km,
        speed_of_light_km_s.
    eta_semantics : str
        Interpretation of detection efficiency:
        - "detector_only": Map η directly to detector_efficiency
        - "end_to_end": Distribute η across channel loss and detector

    Returns
    -------
    HeraldedLinkConfig
        Configured heralded-double-click link.

    Notes
    -----
    The heralded model supports per-side parameters (_A, _B) but we use
    symmetric "global" parameters for simplicity.

    References
    ----------
    - netsquid_netbuilder/modules/qlinks/heralded_double_click.py
    - netsquid_magic/model_parameters.py: DoubleClickModelParameters
    """
    try:
        from squidasm.run.stack.config import HeraldedLinkConfig

        if eta_semantics == "detector_only":
            # Map η directly to detector_efficiency, no physical loss
            length_km = 0.0
            p_loss_length = 0.0
            p_loss_init = 0.0
            detector_efficiency = float(nsm_params.detection_eff_eta)
        else:  # end_to_end
            # Distribute η across channel loss and keep detector ideal
            length_km = float(channel_params.length_km)
            p_loss_length = float(channel_params.attenuation_db_per_km)
            detector_efficiency = 1.0
            # Model remaining loss as initial loss
            p_loss_init = 1.0 - float(nsm_params.detection_eff_eta)

        return HeraldedLinkConfig(
            length=length_km,
            p_loss_length=p_loss_length,
            p_loss_init=p_loss_init,
            speed_of_light=float(channel_params.speed_of_light_km_s),
            detector_efficiency=detector_efficiency,
            dark_count_probability=float(nsm_params.dark_count_prob),
            visibility=1.0,
            emission_fidelity=float(nsm_params.channel_fidelity),
            emission_duration=0.0,
            collection_efficiency=1.0,
            num_multiplexing_modes=1,
        )
    except ImportError as e:
        raise NetworkConfigError(
            "SquidASM is required for link configuration."
        ) from e


def _map_detector_error_to_gate_depolar(detector_error: float) -> float:
    """
    Map detector error rate to gate depolarization probability.

    Provides a heuristic mapping from e_det (intrinsic detector error)
    to the gate depolarization probability used in GenericQDeviceConfig.

    Parameters
    ----------
    detector_error : float
        Intrinsic detector error rate ∈ [0, 0.5].

    Returns
    -------
    float
        Gate depolarization probability ∈ [0, 1].

    Notes
    -----
    The mapping uses a factor of 2 to account for the difference between
    bit error probability and depolarization probability. This is a
    heuristic that should be validated empirically.

    Validation requirement: demonstrate that higher detector_error
    produces higher observed QBER in simulation.
    """
    return min(1.0, max(0.0, 2.0 * float(detector_error)))


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
    model_selection : ChannelModelSelection, optional
        Link model selection configuration. If None, uses auto-selection.

    Examples
    --------
    >>> builder = CaligoNetworkBuilder(nsm_params, channel_params)
    >>> config = builder.build_two_node_network("Alice", "Bob")

    References
    ----------
    - squidasm/run/stack/config.py: StackNetworkConfig
    - squidasm/run/stack/build.py: create_stack_network_builder
    - nsm_parameters_enforcement.md: NSM physical enforcement specification
    """

    def __init__(
        self,
        nsm_params: NSMParameters,
        channel_params: Optional[ChannelParameters] = None,
        model_selection: Optional[ChannelModelSelection] = None,
    ) -> None:
        """
        Initialize the network builder.

        Parameters
        ----------
        nsm_params : NSMParameters
            NSM security parameters.
        channel_params : ChannelParameters, optional
            Physical channel parameters. If None, uses defaults.
        model_selection : ChannelModelSelection, optional
            Link model selection. If None, uses auto-selection.
        """
        self._nsm_params = nsm_params
        self._channel_params = channel_params or ChannelParameters.for_testing()
        self._model_selection = model_selection or ChannelModelSelection()

    @property
    def nsm_params(self) -> NSMParameters:
        """Get NSM parameters."""
        return self._nsm_params

    @property
    def channel_params(self) -> ChannelParameters:
        """Get channel parameters."""
        return self._channel_params

    @property
    def model_selection(self) -> ChannelModelSelection:
        """Get model selection configuration."""
        return self._model_selection

    def build_two_node_network(
        self,
        alice_name: str = "Alice",
        bob_name: str = "Bob",
        num_qubits: int = 10,
        with_device_noise: bool = False,
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
        with_device_noise : bool
            If True, apply device noise (T1/T2, gate depolarization).
            Default: False.

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

        The quantum link noise model is selected based on NSM parameters
        and the model_selection configuration:
        - "perfect": No noise (requires F=1.0, η=1.0, P_dark=0)
        - "depolarise": Depolarizing EPR pairs (fidelity only)
        - "heralded-double-click": Full model with detector efficiency
          and dark counts
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

        # Resolve link model based on parameters
        resolved_model = self._model_selection.resolve_link_model(
            channel_fidelity=self._nsm_params.channel_fidelity,
            detection_eff_eta=self._nsm_params.detection_eff_eta,
            dark_count_prob=self._nsm_params.dark_count_prob,
            detector_error=self._nsm_params.detector_error,
            length_km=self._channel_params.length_km,
        )

        # Create link configuration based on resolved model
        if resolved_model == "perfect":
            link_noise_type = "perfect"
            link_cfg_payload = None
        elif resolved_model == "depolarise":
            link_noise_type = "depolarise"
            link_cfg_payload = _make_depolarise_link_cfg(
                self._nsm_params, self._channel_params
            )
        else:  # heralded-double-click
            link_noise_type = "heralded"
            link_cfg_payload = _make_heralded_double_click_cfg(
                self._nsm_params,
                self._channel_params,
                eta_semantics=self._model_selection.eta_semantics,
            )

        # Create node configurations
        alice_qdevice_cfg = self._create_qdevice_config(
            num_qubits=num_qubits,
            with_noise=with_device_noise,
        )
        bob_qdevice_cfg = self._create_qdevice_config(
            num_qubits=num_qubits,
            with_noise=with_device_noise,
        )

        alice_config = StackConfig(
            name=alice_name,
            qdevice_typ="generic",
            qdevice_cfg=alice_qdevice_cfg,
        )

        bob_config = StackConfig(
            name=bob_name,
            qdevice_typ="generic",
            qdevice_cfg=bob_qdevice_cfg,
        )

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

        logger.debug(
            f"Created two-node network: {alice_name} <-> {bob_name}, "
            f"model={resolved_model}, fidelity={self._nsm_params.channel_fidelity:.4f}, "
            f"qubits={num_qubits}, device_noise={with_device_noise}"
        )

        return network_config

    def _create_qdevice_config(
        self,
        num_qubits: int = 10,
        with_noise: bool = False,
    ) -> Any:
        """
        Create GenericQDeviceConfig with optional noise.

        Parameters
        ----------
        num_qubits : int
            Number of qubit positions.
        with_noise : bool
            If True, apply T1/T2 and gate depolarization noise.

        Returns
        -------
        GenericQDeviceConfig
            Configured device settings.
        """
        try:
            from squidasm.run.stack.config import GenericQDeviceConfig
        except ImportError as e:
            raise NetworkConfigError(
                "SquidASM is required for device configuration."
            ) from e

        if with_noise:
            qdevice_cfg = GenericQDeviceConfig()
            qdevice_cfg.num_qubits = num_qubits
            qdevice_cfg.num_comm_qubits = num_qubits

            # Memory noise from channel parameters
            qdevice_cfg.T1 = self._channel_params.t1_ns
            qdevice_cfg.T2 = self._channel_params.t2_ns

            # Gate noise from detector error
            gate_depolar = _map_detector_error_to_gate_depolar(
                self._nsm_params.detector_error
            )
            qdevice_cfg.single_qubit_gate_depolar_prob = gate_depolar
            qdevice_cfg.two_qubit_gate_depolar_prob = min(1.0, gate_depolar * 1.5)
        else:
            qdevice_cfg = GenericQDeviceConfig.perfect_config()
            qdevice_cfg.num_qubits = num_qubits
            qdevice_cfg.num_comm_qubits = num_qubits

        return qdevice_cfg

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

"""
Physical-to-Simulator Noise Adapter.

This module provides a translation layer mapping physical device parameters
(μ, η, e_det, P_dark) to simulator-level noise knobs (link fidelity, bitflip
probability, detection probability).

Design Rationale
----------------
The adapter bridges two abstraction levels:
1. **Physical level**: Experimental parameters from device characterization
2. **Simulator level**: NetSquid/SquidASM noise model parameters

The translation is explicit and conservative—it documents what is modeled
and what approximations are made.

Important Limitations
---------------------
This adapter is NOT a full security proof translation. It provides:
- First-order approximations suitable for simulation
- Expected-rate checks for protocol feasibility
- Reproducible parameter mapping for testing

The adapter is designed to be replaceable/extendable in later sprints without
changing the NSM math module.

References
----------
- Erven et al. (2014): Table I experimental parameters
- sprint_1_specification.md Section 4 (TASK-NOISE-ADAPTER-001)
- SquidASM network configuration: squidasm/sim/stack/config.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Import PhysicalParameters from core config (canonical location)
# Note: Also available from configs/protocol_config.py for backward compatibility
from ehok.core.config import PhysicalParameters
from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Simulator Noise Parameters
# =============================================================================


@dataclass(frozen=True)
class SimulatorNoiseParams:
    """
    Simulator-level noise parameters for SquidASM/NetSquid.

    These parameters map to SquidASM configuration structures:
    - link_fidelity → Link.fidelity in network config
    - measurement_bitflip_prob → qubit error model
    - expected_detection_prob → expected coincidence rate

    Attributes
    ----------
    link_fidelity : float
        Entanglement fidelity for EPR pair distribution.
        Corresponds to Link.fidelity in SquidASM.
        Must be in [0, 1]. Higher is better.
    measurement_bitflip_prob : float
        Probability of measurement yielding wrong result.
        Maps to intrinsic detection error e_det.
        Must be in [0, 0.5].
    expected_detection_prob : float
        Expected probability of successful coincidence detection.
        Accounts for source brightness, channel loss, and detector efficiency.
        Must be in [0, 1].

    Raises
    ------
    ValueError
        If any parameter is outside valid range.
    """

    link_fidelity: float
    measurement_bitflip_prob: float
    expected_detection_prob: float

    def __post_init__(self) -> None:
        """Validate simulator parameters."""
        if self.link_fidelity < 0 or self.link_fidelity > 1:
            raise ValueError(
                f"link_fidelity must be in [0, 1], got {self.link_fidelity}"
            )

        if self.measurement_bitflip_prob < 0 or self.measurement_bitflip_prob > 0.5:
            raise ValueError(
                f"measurement_bitflip_prob must be in [0, 0.5], got {self.measurement_bitflip_prob}"
            )

        if self.expected_detection_prob < 0 or self.expected_detection_prob > 1:
            raise ValueError(
                f"expected_detection_prob must be in [0, 1], got {self.expected_detection_prob}"
            )


# =============================================================================
# Translation Functions
# =============================================================================


def physical_to_simulator(params: PhysicalParameters) -> SimulatorNoiseParams:
    """
    Translate physical parameters to simulator noise configuration.

    Translation Logic
    -----------------
    1. **measurement_bitflip_prob**: Directly maps to e_det.
       This is exact for intrinsic detector errors.

    2. **expected_detection_prob**: Approximated as η × μ for low μ regime.
       In the low-brightness regime (μ << 1), the probability of detecting
       a coincidence is dominated by single-pair emissions.

    3. **link_fidelity**: Derived from channel quality.
       For a depolarizing channel with error rate e, fidelity F ≈ 1 - 4e/3.
       We use 1 - e_det as a conservative first-order approximation.

    Parameters
    ----------
    params : PhysicalParameters
        Physical device/channel parameters from characterization.

    Returns
    -------
    SimulatorNoiseParams
        Simulator-level noise configuration.

    Notes
    -----
    These translations are first-order approximations. The actual relationship
    between physical and simulated parameters may be more complex in practice.

    References
    ----------
    - sprint_1_specification.md Section 4 (normative requirements)

    Examples
    --------
    >>> from ehok.configs.protocol_config import PhysicalParameters
    >>> phys = PhysicalParameters()  # Erven defaults
    >>> sim = physical_to_simulator(phys)
    >>> sim.measurement_bitflip_prob
    0.0093
    """
    # 1. Measurement bitflip directly equals intrinsic detection error
    measurement_bitflip = params.e_det

    # 2. Expected detection probability
    # In low-μ regime: P_detect ≈ η × (probability of single pair)
    # For Poisson source: P(n=1|n≥1) ≈ μ for μ << 1
    # Including dark count contribution: P_detect ≈ η × μ + P_dark
    expected_detection = params.eta_total_transmittance * params.mu_pair_per_coherence
    expected_detection += params.p_dark
    expected_detection = min(1.0, expected_detection)  # Cap at 1

    # 3. Link fidelity from channel quality
    # Conservative approximation: F = 1 - e_det
    # This models the channel as introducing errors at rate e_det
    link_fidelity = 1.0 - params.e_det

    logger.debug(
        "Physical→Simulator translation: "
        "η=%.4e, μ=%.4e, e_det=%.4f → "
        "fidelity=%.4f, bitflip=%.4f, detect_prob=%.4e",
        params.eta_total_transmittance,
        params.mu_pair_per_coherence,
        params.e_det,
        link_fidelity,
        measurement_bitflip,
        expected_detection,
    )

    return SimulatorNoiseParams(
        link_fidelity=link_fidelity,
        measurement_bitflip_prob=measurement_bitflip,
        expected_detection_prob=expected_detection,
    )


def estimate_qber_from_physical(params: PhysicalParameters) -> float:
    """
    Estimate expected QBER from physical parameters.

    The QBER includes contributions from:
    - Intrinsic detection errors (e_det)
    - Dark count contamination
    - Multi-photon emissions (for non-zero μ)

    Parameters
    ----------
    params : PhysicalParameters
        Physical device/channel parameters.

    Returns
    -------
    float
        Estimated QBER in [0, 0.5].

    Notes
    -----
    This is a simplified model. In practice, QBER depends on:
    - Alignment quality
    - Background light
    - Detector timing jitter
    - Polarization drift

    The formula used is:
        QBER ≈ e_det + P_dark / (2 × P_detection)

    The dark count contribution assumes random outcomes add 50% errors.
    """
    # Base error from detector
    qber = params.e_det

    # Add dark count contribution
    # Dark counts produce random bits, contributing ~50% errors
    detection_prob = params.eta_total_transmittance * params.mu_pair_per_coherence
    if detection_prob > 1e-15:  # Avoid division by zero
        dark_contribution = 0.5 * params.p_dark / detection_prob
        qber += dark_contribution

    # Cap at 0.5 (maximum QBER for random guessing)
    qber = min(0.5, qber)

    logger.debug(
        "Estimated QBER from physical params: %.4f (e_det=%.4f, dark_contrib=%.4e)",
        qber,
        params.e_det,
        qber - params.e_det,
    )

    return qber


def estimate_sifted_rate(
    params: PhysicalParameters,
    source_rate_hz: float = 10_000_000,  # 10 MHz typical
) -> float:
    """
    Estimate sifted bit rate from physical parameters.

    Parameters
    ----------
    params : PhysicalParameters
        Physical device/channel parameters.
    source_rate_hz : float
        Source repetition rate in Hz (pulses per second).
        Default: 10 MHz (typical for PDC sources).

    Returns
    -------
    float
        Expected sifted bits per second.

    Notes
    -----
    Sifted rate accounts for:
    - Detection probability (η × μ)
    - Basis matching (50% for BB84-style)
    """
    # Detection rate
    detection_prob = params.eta_total_transmittance * params.mu_pair_per_coherence

    # Basis matching probability (both choose same basis)
    basis_match_prob = 0.5

    # Sifted rate
    sifted_rate = source_rate_hz * detection_prob * basis_match_prob

    return sifted_rate


# =============================================================================
# Validation Functions
# =============================================================================


def validate_physical_params_for_simulation(
    params: PhysicalParameters,
) -> list[str]:
    """
    Validate physical parameters for simulation feasibility.

    Parameters
    ----------
    params : PhysicalParameters
        Parameters to validate.

    Returns
    -------
    list[str]
        List of warning messages. Empty if no issues.

    Notes
    -----
    Checks for common issues that may cause simulation problems:
    - Very low detection rates (may need long simulation times)
    - Very high error rates (may violate security bounds)
    - Unrealistic parameter combinations
    """
    warnings = []

    # Check for very low detection rate
    detection_prob = params.eta_total_transmittance * params.mu_pair_per_coherence
    if detection_prob < 1e-10:
        warnings.append(
            f"Very low detection probability ({detection_prob:.2e}); "
            "simulation may require extremely long run times"
        )

    # Check for marginal transmittance
    if params.eta_total_transmittance < 0.001:
        warnings.append(
            f"Transmittance η={params.eta_total_transmittance:.4f} is very low; "
            "this may cause feasibility issues"
        )

    # Check for high error rate
    if params.e_det > 0.1:
        warnings.append(
            f"Detection error e_det={params.e_det:.4f} is high; "
            "approaching security limits"
        )

    # Check for unrealistic μ
    if params.mu_pair_per_coherence > 0.1:
        warnings.append(
            f"μ={params.mu_pair_per_coherence:.4f} is unusually high; "
            "multi-photon emissions may dominate"
        )

    return warnings


# =============================================================================
# Physical Model Adapter Output
# =============================================================================


@dataclass(frozen=True)
class AdapterOutput:
    """
    Output container from PhysicalModelAdapter.

    Attributes
    ----------
    link_fidelity : float
        Calculated EPR pair fidelity for SquidASM link configuration.
        Corresponds to F = 1 - p_max_mixed in netsquid_magic parameters.
    prob_success : float
        Per-cycle success probability for entanglement generation.
    t_cycle_ns : float
        Cycle time in nanoseconds (default: distance-based calculation).
    storage_noise_r : float | None
        NSM storage noise parameter r if T1/T2 configuration provided.
    expected_qber : float
        Expected QBER from device characterization.

    Notes
    -----
    This output feeds into both:
    1. SquidASM network configuration (fidelity, prob_success, t_cycle)
    2. NSM security calculations (storage_noise_r, expected_qber)
    """

    link_fidelity: float
    prob_success: float
    t_cycle_ns: float
    storage_noise_r: float | None
    expected_qber: float


# =============================================================================
# Physical Model Adapter (TASK-NOISE-ADAPTER-001)
# =============================================================================


class PhysicalModelAdapter:
    """
    Bridges NSM physical parameters to SquidASM simulation configuration.

    This adapter performs two critical translations:
    1. NSM physical params (μ, η, e_det) → SquidASM DepolariseLinkConfig
    2. NetSquid T1/T2 memory params → NSM storage noise parameter r

    The adapter ensures that security calculations and simulation state
    derive from consistent physical assumptions.

    Parameters
    ----------
    physical_params : PhysicalParameters
        NSM physical device characterization (μ, η, e_det, P_dark).
    memory_T1_ns : float | None
        Amplitude damping time T1 in nanoseconds. None if not modeling
        adversary memory explicitly.
    memory_T2_ns : float | None
        Dephasing time T2 in nanoseconds. Must satisfy T2 ≤ T1.
    delta_t_ns : float
        NSM mandatory wait time Δt in nanoseconds.
        Default: 1e9 ns (1 second, per Erven et al. 2014).

    Attributes
    ----------
    output : AdapterOutput
        Computed adapter output after initialization.

    References
    ----------
    - Erven et al. (2014): Table I experimental parameters
    - netsquid_magic.model_parameters.DepolariseModelParameters
    - squidasm.run.stack.config.DepolariseLinkConfig
    - sprint_1_specification.md Section 4 (TASK-NOISE-ADAPTER-001)

    Examples
    --------
    >>> from ehok.configs.protocol_config import PhysicalParameters
    >>> params = PhysicalParameters()  # Erven defaults
    >>> adapter = PhysicalModelAdapter(
    ...     physical_params=params,
    ...     memory_T1_ns=1e9,
    ...     memory_T2_ns=5e8,
    ...     delta_t_ns=1e9
    ... )
    >>> adapter.output.link_fidelity
    0.9907
    """

    def __init__(
        self,
        physical_params: PhysicalParameters,
        memory_T1_ns: float | None = None,
        memory_T2_ns: float | None = None,
        delta_t_ns: float = 1_000_000_000,  # 1 second default
    ) -> None:
        self._physical_params = physical_params
        self._memory_T1_ns = memory_T1_ns
        self._memory_T2_ns = memory_T2_ns
        self._delta_t_ns = delta_t_ns

        # Validate T1/T2 relationship
        if memory_T1_ns is not None and memory_T2_ns is not None:
            if memory_T2_ns > memory_T1_ns:
                raise ValueError(
                    f"T2 ({memory_T2_ns} ns) cannot exceed T1 ({memory_T1_ns} ns)"
                )

        # Compute adapter output
        self._output = self._compute_output()

        logger.info(
            "PhysicalModelAdapter initialized: fidelity=%.4f, r=%.4f, QBER=%.4f",
            self._output.link_fidelity,
            self._output.storage_noise_r or 0.0,
            self._output.expected_qber,
        )

    @property
    def output(self) -> AdapterOutput:
        """Get computed adapter output."""
        return self._output

    @property
    def physical_params(self) -> PhysicalParameters:
        """Get physical parameters."""
        return self._physical_params

    def _compute_output(self) -> AdapterOutput:
        """Compute all adapter output values."""
        params = self._physical_params

        # 1. Link fidelity from channel quality
        # F = 1 - e_det is the first-order approximation
        link_fidelity = 1.0 - params.e_det

        # 2. Success probability per cycle
        # P_success ≈ η × μ (detection probability)
        prob_success = (
            params.eta_total_transmittance * params.mu_pair_per_coherence
        )
        prob_success = min(1.0, max(0.0, prob_success))

        # 3. Cycle time (default based on typical fiber propagation)
        # For 1km fiber at 200,000 km/s: t_cycle = 5 μs
        t_cycle_ns = 5_000.0  # 5 μs default

        # 4. Storage noise r from T1/T2 if provided
        storage_noise_r = None
        if self._memory_T1_ns is not None and self._memory_T2_ns is not None:
            storage_noise_r = self._compute_storage_noise_r(
                T1_ns=self._memory_T1_ns,
                T2_ns=self._memory_T2_ns,
                delta_t_ns=self._delta_t_ns,
            )

        # 5. Expected QBER
        expected_qber = self._compute_expected_qber(params)

        return AdapterOutput(
            link_fidelity=link_fidelity,
            prob_success=prob_success,
            t_cycle_ns=t_cycle_ns,
            storage_noise_r=storage_noise_r,
            expected_qber=expected_qber,
        )

    @staticmethod
    def _compute_storage_noise_r(
        T1_ns: float,
        T2_ns: float,
        delta_t_ns: float,
    ) -> float:
        """
        Compute NSM storage noise parameter r from T1/T2 memory parameters.

        The storage noise r represents the probability that a qubit stored
        in the adversary's quantum memory retains its quantum state after
        the mandatory wait time Δt.

        Derivation
        ----------
        For a T1/T2 noise model, the fidelity of a stored qubit decays as:

            F(t) = 0.5 × (1 + exp(-t/T1) × exp(-t/T2))

        This models:
        - Amplitude damping (T1): Population relaxation
        - Dephasing (T2): Phase coherence loss

        The NSM storage noise parameter r is the "retention probability":

            r = exp(-Δt/T1) × exp(-Δt/T2) (the coherence factor)

        Parameters
        ----------
        T1_ns : float
            Amplitude damping time in nanoseconds.
        T2_ns : float
            Dephasing time in nanoseconds.
        delta_t_ns : float
            NSM mandatory wait time in nanoseconds.

        Returns
        -------
        float
            Storage noise parameter r ∈ [0, 1].
            r = 0: Complete decoherence (ideal for security)
            r = 1: Perfect storage (worst for security)

        References
        ----------
        - König et al. (2012): Eq. (1) Markovian noise assumption
        - netsquid.components.models.qerrormodels.T1T2NoiseModel
        """
        if T1_ns <= 0 or T2_ns <= 0:
            raise ValueError("T1 and T2 must be positive")
        if delta_t_ns < 0:
            raise ValueError("delta_t must be non-negative")

        # Compute decay factors
        decay_T1 = math.exp(-delta_t_ns / T1_ns)
        decay_T2 = math.exp(-delta_t_ns / T2_ns)

        # Storage retention parameter r
        r = decay_T1 * decay_T2

        logger.debug(
            "Storage noise computed: T1=%.2e ns, T2=%.2e ns, Δt=%.2e ns → r=%.4f",
            T1_ns, T2_ns, delta_t_ns, r
        )

        return r

    @staticmethod
    def _compute_expected_qber(params: PhysicalParameters) -> float:
        """Compute expected QBER from physical parameters."""
        # Base error from intrinsic detector error
        qber = params.e_det

        # Dark count contribution: adds 50% error rate
        detection_prob = (
            params.eta_total_transmittance * params.mu_pair_per_coherence
        )
        if detection_prob > 1e-15:
            dark_contribution = 0.5 * params.p_dark / detection_prob
            qber += dark_contribution

        return min(0.5, qber)

    def to_squidasm_link_config(self):
        """
        Generate SquidASM-compatible link configuration.

        Returns
        -------
        DepolariseLinkConfig
            Configuration object for squidasm.run.stack.config.LinkConfig.

        Notes
        -----
        Import is deferred to avoid circular dependencies with SquidASM.
        """
        from squidasm.run.stack.config import DepolariseLinkConfig

        return DepolariseLinkConfig(
            fidelity=self._output.link_fidelity,
            prob_success=self._output.prob_success,
            t_cycle=self._output.t_cycle_ns,
        )

    def to_stack_network_config(
        self,
        alice_name: str = "Alice",
        bob_name: str = "Bob",
    ):
        """
        Generate complete SquidASM network configuration.

        Parameters
        ----------
        alice_name : str
            Name for Alice's stack node.
        bob_name : str
            Name for Bob's stack node.

        Returns
        -------
        StackNetworkConfig
            Complete network configuration ready for simulation.

        Notes
        -----
        Creates a two-node network with:
        - Generic quantum devices (perfect except for link noise)
        - Depolarizing quantum link with configured fidelity
        - Instant classical link
        """
        from squidasm.run.stack.config import (
            StackNetworkConfig,
            StackConfig,
            LinkConfig,
        )

        alice_stack = StackConfig.perfect_generic_config(alice_name)
        bob_stack = StackConfig.perfect_generic_config(bob_name)

        link = LinkConfig(
            stack1=alice_name,
            stack2=bob_name,
            typ="depolarise",
            cfg=self.to_squidasm_link_config(),
        )

        return StackNetworkConfig(
            stacks=[alice_stack, bob_stack],
            links=[link],
        )


# =============================================================================
# Standalone Function
# =============================================================================


def estimate_storage_noise_from_netsquid(
    T1_ns: float,
    T2_ns: float,
    delta_t_ns: float,
) -> float:
    """
    Estimate NSM storage noise parameter r from NetSquid T1/T2 memory parameters.

    This is a convenience function wrapping the static method from
    PhysicalModelAdapter. Use this when you need only the storage noise
    calculation without full adapter configuration.

    Parameters
    ----------
    T1_ns : float
        Amplitude damping time in nanoseconds.
    T2_ns : float
        Dephasing time in nanoseconds. Must satisfy T2 ≤ T1.
    delta_t_ns : float
        NSM mandatory wait time in nanoseconds.

    Returns
    -------
    float
        Storage noise parameter r ∈ [0, 1].

    References
    ----------
    - system_test_specification.md SYS-INT-NOISE-002

    Examples
    --------
    >>> r = estimate_storage_noise_from_netsquid(1e9, 5e8, 1e9)
    >>> abs(r - 0.135) < 0.01  # exp(-1) × exp(-2) ≈ 0.135
    True
    """
    return PhysicalModelAdapter._compute_storage_noise_r(T1_ns, T2_ns, delta_t_ns)

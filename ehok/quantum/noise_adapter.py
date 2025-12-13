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

from ehok.configs.protocol_config import PhysicalParameters
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

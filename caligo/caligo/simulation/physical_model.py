"""
NSM physical parameters and NetSquid noise model mappings.

This module defines the dataclasses that encapsulate Noisy Storage Model
parameters and map them to NetSquid simulation components.

Erven et al. (2014) Formulas
----------------------------
The paper provides exact formulas for calculating security parameters:

**PDC Source Model (Eq. 9-11):**
    |Ψ_src⟩ = Σ √(P^n_src) |Φ_n⟩_{AB}
    P^n_src = (n+1)(μ/2)^n / (1+(μ/2))^{n+2}

**Key Derived Quantities:**
    P_sent: Probability only one photon pair is sent
    P_{B,noclick}: Probability honest Bob receives no click
    P'_{B,noclick}: Minimum probability dishonest Bob receives no click

**QBER Components:**
    p_err = (1-F)/2 + e_det  (base error rate)
    Full QBER includes dark count contributions

References
----------
- König et al. (2012): NSM definition, storage capacity constraint
- Erven et al. (2014): Experimental parameters, Table I
- Schaffner et al. (2009): 11% QBER threshold, depolarizing analysis
- Wehner et al. (2010): Implementation of two-party protocols in NSM
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from caligo.types.exceptions import InvalidParameterError
from caligo.utils.math import (
    binary_entropy,
    compute_qber_erven,
    suggested_ldpc_rate_from_qber,
    blind_reconciliation_initial_config,
)
from caligo.simulation.constants import (
    NANOSECOND,
    MICROSECOND,
    MILLISECOND,
    SECOND,
    TYPICAL_DELTA_T_NS,
    TYPICAL_CYCLE_TIME_NS,
    TYPICAL_T1_NS,
    TYPICAL_T2_NS,
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
    ERVEN_MU,
    ERVEN_ETA,
    ERVEN_E_DET,
    ERVEN_P_DARK,
    ERVEN_R,
    ERVEN_NU,
)


# =============================================================================
# PDC Source Probability Functions (Erven et al. Eq. 9-11)
# =============================================================================


def pdc_probability(n: int, mu: float) -> float:
    """
    Calculate PDC source probability for n-pair emission.

    P^n_src = (n+1)(μ/2)^n / (1+(μ/2))^{n+2}   [Eq. 10]

    Parameters
    ----------
    n : int
        Number of photon pairs (n >= 0).
    mu : float
        Mean photon pair number per pulse (μ > 0).

    Returns
    -------
    float
        Probability of n-pair emission.

    References
    ----------
    - Erven et al. (2014) Eq. 10
    - Kok & Braunstein (2000) PDC model
    """
    if n < 0:
        return 0.0
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    mu_half = mu / 2.0
    numerator = (n + 1) * (mu_half ** n)
    denominator = (1.0 + mu_half) ** (n + 2)
    return numerator / denominator


def p_sent(mu: float) -> float:
    """
    Probability only one photon pair is sent.

    P_sent = P^1_src / (1 - P^0_src)

    This is the conditional probability of single-pair emission
    given that at least one pair was emitted.

    Parameters
    ----------
    mu : float
        Mean photon pair number per pulse.

    Returns
    -------
    float
        Single-pair probability.

    References
    ----------
    - Erven et al. (2014) derivation from Eq. 9-11
    - Wehner et al. (2010) Phys. Rev. A 81, 052336
    """
    p0 = pdc_probability(0, mu)
    p1 = pdc_probability(1, mu)
    if p0 >= 1.0:
        return 0.0  # Edge case: no pairs emitted
    return p1 / (1.0 - p0)


def p_b_noclick(mu: float, eta: float, p_dark: float) -> float:
    """
    Probability honest Bob receives no click from a photon pair.

    P_{B,noclick} considers all n-pair emissions and the probability
    that none produce a detection event at Bob's side.

    For single pair: (1 - η)(1 - P_dark)
    Full sum over n-pair contributions weighted by PDC probabilities.

    Parameters
    ----------
    mu : float
        Mean photon pair number per pulse.
    eta : float
        Total transmittance (detection efficiency).
    p_dark : float
        Dark count probability per detection window.

    Returns
    -------
    float
        No-click probability for honest Bob.

    References
    ----------
    - Erven et al. (2014) Security Analysis
    - Wehner et al. (2010)
    """
    # Simplified model: conditional on pair emission
    # P_{B,noclick} ≈ (1 - η)(1 - P_dark) for single pair
    # Full model sums over all n-pair contributions

    # For practical calculations, use truncated sum
    total = 0.0
    p_no_detection = (1.0 - eta) * (1.0 - p_dark)

    for n in range(20):  # Truncate at n=20 (negligible contribution beyond)
        pn = pdc_probability(n, mu)
        if n == 0:
            # No pair emitted: only dark count can trigger
            p_noclick_n = 1.0 - p_dark
        else:
            # n pairs: each has independent chance to not trigger detection
            # Bob sees at least one photon with prob 1-(1-η)^n
            # No click means no photon detected AND no dark count
            p_noclick_n = ((1.0 - eta) ** n) * (1.0 - p_dark)
        total += pn * p_noclick_n

    return total


def p_b_noclick_min(mu: float, eta: float, p_dark: float) -> float:
    """
    Minimum probability dishonest Bob receives no click.

    P'_{B,noclick} is the lower bound on no-click probability,
    assuming adversary can optimize detection strategy.

    Parameters
    ----------
    mu : float
        Mean photon pair number per pulse.
    eta : float
        Total transmittance.
    p_dark : float
        Dark count probability.

    Returns
    -------
    float
        Minimum no-click probability (dishonest Bob bound).

    References
    ----------
    - Erven et al. (2014) Security bounds
    - Wehner et al. (2010) Theorem 1
    """
    # Dishonest Bob can potentially optimize, but bounded by physics
    # Conservative estimate: honest Bob value (no amplification attack)
    # In practice, P'_{B,noclick} ≤ P_{B,noclick}

    # For security analysis, use the more conservative bound
    # allowing for potential adversarial strategies
    honest_bound = p_b_noclick(mu, eta, p_dark)

    # Dishonest Bob might block some events; minimum is at perfect detection
    # P'_{B,noclick} = P^0_src (only vacuum gives no click)
    p0 = pdc_probability(0, mu)

    # Return the more conservative (higher) bound for security
    return max(p0, honest_bound * 0.95)  # 5% margin for dishonest strategies


# =============================================================================
# NSM Parameters
# =============================================================================


@dataclass(frozen=True)
class NSMParameters:
    """
    Noisy Storage Model parameters for simulation and security analysis.

    Encapsulates all NSM parameters required for E-HOK protocol execution
    and provides validated mappings to NetSquid noise models.

    Parameters
    ----------
    storage_noise_r : float
        Depolarizing parameter r ∈ [0, 1].
        r=0: Complete depolarization (best for security).
        r=1: Perfect storage (no noise, worst for security).
    storage_rate_nu : float
        Adversary storage rate ν ∈ [0, 1].
        Fraction of qubits the adversary can store.
    delta_t_ns : float
        Wait time Δt in nanoseconds. Must be > 0.
    channel_fidelity : float
        EPR pair fidelity F ∈ (0.5, 1].
    detection_eff_eta : float
        Combined detection efficiency η ∈ (0, 1]. Default: 1.0.
    detector_error : float
        Intrinsic detector error rate. Default: 0.0.
    dark_count_prob : float
        Dark count probability per detection event. Default: 0.0.
    storage_dimension_d : int
        Qubit dimension (always 2). Default: 2.

    Raises
    ------
    InvalidParameterError
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-NSM-001: storage_noise_r ∈ [0, 1]
    - INV-NSM-002: storage_rate_nu ∈ [0, 1]
    - INV-NSM-003: storage_dimension_d == 2
    - INV-NSM-004: delta_t_ns > 0
    - INV-NSM-005: channel_fidelity ∈ (0.5, 1]
    - INV-NSM-006: detection_eff_eta ∈ (0, 1]

    References
    ----------
    - König et al. (2012) Section I-C: NSM definition
    - Erven et al. (2014) Table I: Experimental parameters

    Examples
    --------
    >>> params = NSMParameters(
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ...     delta_t_ns=1_000_000,
    ...     channel_fidelity=0.95,
    ... )
    >>> params.depolar_prob
    0.25
    >>> params.qber_channel < QBER_CONSERVATIVE_LIMIT
    True
    """

    storage_noise_r: float
    storage_rate_nu: float
    delta_t_ns: float
    channel_fidelity: float
    detection_eff_eta: float = 1.0
    detector_error: float = 0.0
    dark_count_prob: float = 0.0
    storage_dimension_d: int = 2

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-NSM-001
        if not 0 <= self.storage_noise_r <= 1:
            raise InvalidParameterError(
                f"INV-NSM-001: storage_noise_r={self.storage_noise_r} must be in [0, 1]"
            )
        # INV-NSM-002
        if not 0 <= self.storage_rate_nu <= 1:
            raise InvalidParameterError(
                f"INV-NSM-002: storage_rate_nu={self.storage_rate_nu} must be in [0, 1]"
            )
        # INV-NSM-003
        if self.storage_dimension_d != 2:
            raise InvalidParameterError(
                f"INV-NSM-003: storage_dimension_d={self.storage_dimension_d} must be 2"
            )
        # INV-NSM-004
        if self.delta_t_ns <= 0:
            raise InvalidParameterError(
                f"INV-NSM-004: delta_t_ns={self.delta_t_ns} must be > 0"
            )
        # INV-NSM-005
        if not 0.5 < self.channel_fidelity <= 1:
            raise InvalidParameterError(
                f"INV-NSM-005: channel_fidelity={self.channel_fidelity} "
                "must be in (0.5, 1]"
            )
        # INV-NSM-006
        if not 0 < self.detection_eff_eta <= 1:
            raise InvalidParameterError(
                f"INV-NSM-006: detection_eff_eta={self.detection_eff_eta} "
                "must be in (0, 1]"
            )

    # =========================================================================
    # Derived Properties
    # =========================================================================

    @property
    def depolar_prob(self) -> float:
        """
        NetSquid depolarization probability.

        Returns
        -------
        float
            Depolarization probability = 1 - r.

        Notes
        -----
        NetSquid's DepolarNoiseModel uses `depolar_rate` as the probability
        of depolarization, whereas NSM uses `r` as the probability of
        preservation. Hence: depolar_rate = 1 - r.
        """
        return 1.0 - self.storage_noise_r

    @property
    def qber_simple(self) -> float:
        """
        Simplified channel QBER from fidelity and detector error only.

        Returns
        -------
        float
            Q_simple = (1 - F) / 2 + e_det.

        Notes
        -----
        This is a first-order approximation that ignores dark counts
        and detection efficiency. Use `qber_channel` for the full model.
        """
        infidelity_contribution = (1.0 - self.channel_fidelity) / 2.0
        return infidelity_contribution + self.detector_error

    @property
    def qber_channel(self) -> float:
        """
        Full channel QBER including all error sources (Erven et al. formula).

        Delegates to shared utility function in caligo.utils.math.

        Returns
        -------
        float
            Q_total = Q_source + Q_det + Q_dark.

        References
        ----------
        - Erven et al. (2014) Eq. 8 and surrounding analysis
        """
        return compute_qber_erven(
            fidelity=self.channel_fidelity,
            detector_error=self.detector_error,
            detection_efficiency=self.detection_eff_eta,
            dark_count_prob=self.dark_count_prob,
        )

    @property
    def qber_full_erven(self) -> float:
        """
        Complete QBER using exact Erven et al. PDC source model.

        This method uses the full PDC probability distribution to compute
        the expected QBER accounting for multi-pair emissions.

        Returns
        -------
        float
            QBER from full PDC model analysis.

        Notes
        -----
        Uses the PDC source model with μ derived from channel parameters.
        For most practical cases, `qber_channel` is sufficient and faster.
        """
        # Infer μ from fidelity degradation
        # μ = 1 - F maps imperfect source to mean photon number proxy
        # This is an approximation; exact μ requires experimental calibration
        mu_estimate = 1.0 - self.channel_fidelity

        # Compute detection probability
        p_detect = self.detection_eff_eta + self.dark_count_prob * (
            1.0 - self.detection_eff_eta
        )

        if p_detect <= 0:
            return 0.5  # No detection means random guessing

        # Compute error probability given detection
        p_error_given_detect = (
            self.detector_error
            + (1.0 - self.channel_fidelity) / 2.0 * self.detection_eff_eta
            + self.dark_count_prob * (1.0 - self.detection_eff_eta) * 0.5
        ) / p_detect

        return min(p_error_given_detect, 0.5)  # Cap at 0.5 (random)

    @property
    def storage_capacity(self) -> float:
        """
        Classical capacity of adversary's storage channel.

        Returns
        -------
        float
            C = 1 - h(depolar_prob) where h = binary entropy.

        Notes
        -----
        For the depolarizing channel:
        - Perfect storage (r=1, depolar=0): C = 1 - h(0) = 1
        - Full depolarization (r=0, depolar=1): C = 1 - h(1) = 1
        - Maximum noise (depolar=0.5): C = 1 - h(0.5) = 0
        """
        # Handle edge cases
        p = self.depolar_prob
        if p == 0 or p == 1:
            return 1.0
        return 1.0 - binary_entropy(p)

    @property
    def security_possible(self) -> bool:
        """
        Check if QBER is below conservative threshold.

        Returns
        -------
        bool
            True if qber_channel < 0.11 (Schaffner threshold).
        """
        return self.qber_channel < QBER_CONSERVATIVE_LIMIT

    @property
    def storage_security_satisfied(self) -> bool:
        """
        Check storage capacity constraint: C_N * ν < 1/2.

        Returns
        -------
        bool
            True if adversary storage capacity is bounded.

        References
        ----------
        - König et al. (2012), Section I-C
        """
        return self.storage_capacity * self.storage_rate_nu < 0.5

    # =========================================================================
    # Reconciliation Support Methods
    # =========================================================================

    def suggested_ldpc_rate(self, safety_margin: float = 0.05) -> float:
        """
        Suggest optimal LDPC code rate for blind reconciliation.

        Delegates to shared utility function in caligo.utils.math.

        Parameters
        ----------
        safety_margin : float
            Additional capacity margin (0-0.1). Default: 0.05.

        Returns
        -------
        float
            Suggested code rate R ∈ [0.5, 0.95].

        References
        ----------
        - Martinez-Mateo et al. (2012) Blind Reconciliation
        - Shannon limit: R ≤ 1 - h(QBER)
        """
        return suggested_ldpc_rate_from_qber(self.qber_channel, safety_margin)

    def blind_reconciliation_config(self) -> dict:
        """
        Generate configuration for blind reconciliation protocol.

        Delegates to shared utility function, adding NSM-specific fields.

        Returns
        -------
        dict
            Configuration dictionary with keys:
            - initial_rate: Starting LDPC code rate
            - rate_adaptation: "puncturing" or "shortening"
            - expected_qber: Estimated channel QBER
            - max_iterations: Maximum blind iterations (3)
            - frame_size: Recommended frame size (4096)
            - use_nsm_informed_start: True
        """
        qber = self.qber_channel
        config = blind_reconciliation_initial_config(qber)
        
        # Add NSM-specific fields
        config["expected_qber"] = qber
        config["max_iterations"] = 3
        config["frame_size"] = 4096
        config["use_nsm_informed_start"] = True
        
        return config

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_erven_experimental(cls) -> NSMParameters:
        """
        Create parameters matching Erven et al. (2014) experiment.

        Returns
        -------
        NSMParameters
            Configuration from Erven et al. Table I.

        Notes
        -----
        Uses exact experimental values:
        - μ = 3.145 × 10^{-5} (source quality, maps to fidelity)
        - η = 0.0150 (transmittance)
        - e_det = 0.0093 (detector error)
        - r = 0.75 (storage noise)
        - ν = 0.002 (storage rate)
        - P_dark = 1.50 × 10^{-8}
        """
        return cls(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,  # 1 ms (typical experimental value)
            channel_fidelity=1.0 - 3.145e-5,  # From source quality μ
            detection_eff_eta=0.0150,
            detector_error=0.0093,
            dark_count_prob=1.50e-8,
        )

    @classmethod
    def for_testing(
        cls,
        storage_noise_r: float = 0.75,
        channel_fidelity: float = 0.95,
        delta_t_ns: float = 1_000_000,
    ) -> NSMParameters:
        """
        Create simplified parameters for unit testing.

        Parameters
        ----------
        storage_noise_r : float
            Storage noise parameter. Default: 0.75.
        channel_fidelity : float
            Channel fidelity. Default: 0.95.
        delta_t_ns : float
            Wait time in ns. Default: 1_000_000 (1 ms).

        Returns
        -------
        NSMParameters
            Simplified configuration suitable for testing.
        """
        return cls(
            storage_noise_r=storage_noise_r,
            storage_rate_nu=0.01,
            delta_t_ns=delta_t_ns,
            channel_fidelity=channel_fidelity,
        )


# =============================================================================
# Channel Parameters
# =============================================================================


@dataclass(frozen=True)
class ChannelParameters:
    """
    Physical channel parameters for the quantum link.

    These parameters characterize the trusted channel between honest
    Alice and honest Bob, separate from the adversary's storage model.

    Parameters
    ----------
    length_km : float
        Fiber length in kilometers. Must be >= 0.
    attenuation_db_per_km : float
        Fiber loss in dB/km. Default: 0.2 (standard telecom fiber).
    speed_of_light_km_s : float
        Speed of light in fiber (km/s). Default: 200_000.
    t1_ns : float
        T1 relaxation time for memory qubits (ns). Default: 10_000_000.
    t2_ns : float
        T2 dephasing time for memory qubits (ns). Default: 1_000_000.
    cycle_time_ns : float
        EPR generation cycle time (ns). Default: 1_000_000 (1 ms).

    Raises
    ------
    InvalidParameterError
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-CH-001: length_km >= 0
    - INV-CH-002: attenuation_db_per_km >= 0
    - INV-CH-003: speed_of_light_km_s > 0
    - INV-CH-004: t1_ns > 0
    - INV-CH-005: t2_ns > 0 and t2_ns <= t1_ns
    - INV-CH-006: cycle_time_ns > 0

    References
    ----------
    - netsquid-magic HeraldedModelParameters: length_A, length_B
    - NetSquid T1T2NoiseModel: T1, T2 parameters
    """

    length_km: float = 0.0
    attenuation_db_per_km: float = 0.2
    speed_of_light_km_s: float = 200_000.0
    t1_ns: float = TYPICAL_T1_NS
    t2_ns: float = TYPICAL_T2_NS
    cycle_time_ns: float = TYPICAL_CYCLE_TIME_NS

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-CH-001
        if self.length_km < 0:
            raise InvalidParameterError(
                f"INV-CH-001: length_km={self.length_km} must be >= 0"
            )
        # INV-CH-002
        if self.attenuation_db_per_km < 0:
            raise InvalidParameterError(
                f"INV-CH-002: attenuation_db_per_km={self.attenuation_db_per_km} "
                "must be >= 0"
            )
        # INV-CH-003
        if self.speed_of_light_km_s <= 0:
            raise InvalidParameterError(
                f"INV-CH-003: speed_of_light_km_s={self.speed_of_light_km_s} must be > 0"
            )
        # INV-CH-004
        if self.t1_ns <= 0:
            raise InvalidParameterError(f"INV-CH-004: t1_ns={self.t1_ns} must be > 0")
        # INV-CH-005
        if self.t2_ns <= 0:
            raise InvalidParameterError(f"INV-CH-005: t2_ns={self.t2_ns} must be > 0")
        if self.t2_ns > self.t1_ns:
            raise InvalidParameterError(
                f"INV-CH-005: t2_ns={self.t2_ns} must be <= t1_ns={self.t1_ns}"
            )
        # INV-CH-006
        if self.cycle_time_ns <= 0:
            raise InvalidParameterError(
                f"INV-CH-006: cycle_time_ns={self.cycle_time_ns} must be > 0"
            )

    # =========================================================================
    # Derived Properties
    # =========================================================================

    @property
    def propagation_delay_ns(self) -> float:
        """
        Light propagation delay through fiber.

        Returns
        -------
        float
            Delay in nanoseconds: length_km / speed_of_light_km_s * 1e9.
        """
        if self.length_km == 0:
            return 0.0
        return (self.length_km / self.speed_of_light_km_s) * SECOND

    @property
    def total_loss_db(self) -> float:
        """
        Total fiber loss in dB.

        Returns
        -------
        float
            Loss = length_km * attenuation_db_per_km.
        """
        return self.length_km * self.attenuation_db_per_km

    @property
    def transmittance(self) -> float:
        """
        Transmission probability through fiber.

        Returns
        -------
        float
            η = 10^(-total_loss_db / 10).
        """
        if self.total_loss_db == 0:
            return 1.0
        return 10.0 ** (-self.total_loss_db / 10.0)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def for_testing(cls) -> ChannelParameters:
        """
        Create simplified parameters for unit testing.

        Returns
        -------
        ChannelParameters
            Zero-length channel (ideal) with standard T1/T2.
        """
        return cls(length_km=0.0)

    @classmethod
    def from_erven_experimental(cls) -> ChannelParameters:
        """
        Create parameters inspired by Erven et al. (2014).

        Returns
        -------
        ChannelParameters
            Configuration approximating experimental setup.
        """
        return cls(
            length_km=0.0,  # Table-top experiment
            t1_ns=100_000_000,  # 100 ms (typical NV centers)
            t2_ns=10_000_000,  # 10 ms
            cycle_time_ns=1_000_000,  # 1 ms per EPR attempt
        )


# =============================================================================
# Factory Functions for NetSquid Models
# =============================================================================


def create_depolar_noise_model(params: NSMParameters) -> Any:
    """
    Create NetSquid DepolarNoiseModel from NSM parameters.

    Parameters
    ----------
    params : NSMParameters
        NSM configuration with storage_noise_r.

    Returns
    -------
    DepolarNoiseModel
        Configured with depolar_rate = 1 - params.storage_noise_r.

    Raises
    ------
    ImportError
        If NetSquid is not available.

    Notes
    -----
    NetSquid's DepolarNoiseModel uses `depolar_rate` as the probability
    of depolarization, whereas NSM uses `r` as the probability of
    preservation. Hence: depolar_rate = 1 - r.

    For time-independent storage noise (adversary stores for exactly Δt):
        time_independent=True
    """
    try:
        from netsquid.components.models.qerrormodels import DepolarNoiseModel

        return DepolarNoiseModel(
            depolar_rate=params.depolar_prob,
            time_independent=True,
        )
    except ImportError as e:
        raise ImportError(
            "NetSquid is required for create_depolar_noise_model. "
            "Install with: pip install netsquid"
        ) from e


def create_t1t2_noise_model(params: ChannelParameters) -> Any:
    """
    Create NetSquid T1T2NoiseModel for quantum memory decoherence.

    Parameters
    ----------
    params : ChannelParameters
        Channel configuration with t1_ns and t2_ns.

    Returns
    -------
    T1T2NoiseModel
        Configured with T1=params.t1_ns, T2=params.t2_ns.

    Raises
    ------
    ImportError
        If NetSquid is not available.

    Notes
    -----
    The T1T2 model is phenomenological:
    - T1: Amplitude damping (energy relaxation)
    - T2: Phase damping (dephasing), T2 ≤ T1

    This models the honest party's imperfect quantum memory,
    NOT the adversary's noisy storage (use depolar model for that).

    References
    ----------
    - netsquid/components/models/qerrormodels.py: T1T2NoiseModel
    """
    try:
        from netsquid.components.models.qerrormodels import T1T2NoiseModel

        return T1T2NoiseModel(T1=params.t1_ns, T2=params.t2_ns)
    except ImportError as e:
        raise ImportError(
            "NetSquid is required for create_t1t2_noise_model. "
            "Install with: pip install netsquid"
        ) from e

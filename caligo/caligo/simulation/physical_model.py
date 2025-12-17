"""
NSM physical parameters and NetSquid noise model mappings.

This module defines the dataclasses that encapsulate Noisy Storage Model
parameters and map them to NetSquid simulation components.

References
----------
- König et al. (2012): NSM definition, storage capacity constraint
- Erven et al. (2014): Experimental parameters, Table I
- Schaffner et al. (2009): 11% QBER threshold, depolarizing analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from caligo.types.exceptions import InvalidParameterError
from caligo.utils.math import binary_entropy


# =============================================================================
# Time Unit Constants (NetSquid compatibility)
# =============================================================================

NANOSECOND: float = 1.0
MICROSECOND: float = 1e3
MILLISECOND: float = 1e6
SECOND: float = 1e9

# Typical timing values
TYPICAL_DELTA_T_NS: float = 1_000_000  # 1 ms (Δt for NSM)
TYPICAL_CYCLE_TIME_NS: float = 10_000  # 10 μs (EPR generation)
TYPICAL_T1_NS: float = 10_000_000  # 10 ms (T1 relaxation)
TYPICAL_T2_NS: float = 1_000_000  # 1 ms (T2 dephasing)

# Security thresholds
QBER_HARD_LIMIT: float = 0.22  # König et al. (2012)
QBER_CONSERVATIVE_LIMIT: float = 0.11  # Schaffner et al. (2009)


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
    def qber_channel(self) -> float:
        """
        Channel QBER from fidelity and detector error.

        Returns
        -------
        float
            Q_channel = (1 - F) / 2 + e_det.

        Notes
        -----
        This is a simplified model. The full Erven formula includes
        detection efficiency and dark counts, but for simulation purposes
        we use this approximation.
        """
        infidelity_contribution = (1.0 - self.channel_fidelity) / 2.0
        return infidelity_contribution + self.detector_error

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

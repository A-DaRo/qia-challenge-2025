"""
Protocol Configuration Schema for E-HOK.

This module provides typed configuration dataclasses for the E-HOK protocol,
exposing NSM security parameters and physical model parameters through a
validated, immutable schema.

Configuration Categories
------------------------
1. **Physical Parameters**: Hardware/channel characteristics (μ, η, e_det, P_dark)
2. **NSM Security Parameters**: Adversary model assumptions (r, ν, Δt)
3. **Protocol Parameters**: Security/correctness targets (ε_sec, ε_cor)

Default Values
--------------
Physical parameter defaults are taken from Erven et al. (2014) Table I:
- μ (mean photon pairs): 3.145 × 10⁻⁵
- η (total transmittance): 0.0150
- e_det (intrinsic error rate): 0.0093
- P_dark (dark count probability): 1.50 × 10⁻⁸

NSM parameter defaults from the same paper:
- r (storage retention): 0.75
- ν (storage rate): 0.002
- Δt (wait time): 1 second

References
----------
- Erven et al. (2014): "An Experimental Implementation of Oblivious Transfer
  in the Noisy Storage Model", Table I.
- sprint_1_specification.md Section 3.3 (NOISE-PARAMS-001)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Default Values from Erven et al. (2014) Table I
# =============================================================================

# Physical parameters (experimental characterization)
DEFAULT_MU_PAIR_PER_COHERENCE = 3.145e-5  # Mean photon pair number per coherence time
DEFAULT_ETA_TOTAL_TRANSMITTANCE = 0.0150  # Total transmission efficiency
DEFAULT_E_DET = 0.0093  # Intrinsic detection error rate
DEFAULT_P_DARK = 1.50e-8  # Dark count probability per coherence time

# Adversary's memory limitations
DEFAULT_STORAGE_NOISE_R = 0.75  # Probability memory retains state
DEFAULT_STORAGE_RATE_NU = 0.002  # Fraction of qubits storable

# Timing parameters
DEFAULT_DELTA_T_S = 1.0  # Wait time in seconds
DEFAULT_DELTA_T_NS = 1_000_000_000  # Wait time in nanoseconds

# Security parameters
DEFAULT_EPSILON_SEC = 2.5e-7  # Security error parameter
DEFAULT_EPSILON_COR = 3.09e-3  # Correctness error (dominated by EC failure)


# =============================================================================
# Physical Parameters Dataclass
# =============================================================================


@dataclass(frozen=True)
class PhysicalParameters:
    """
    Physical/hardware parameters for channel characterization.

    These parameters describe the honest parties' devices and the quantum
    channel. They are estimated before protocol execution and determine
    the error correction requirements.

    Attributes
    ----------
    mu_pair_per_coherence : float
        Mean photon pair number per coherence time (pulse).
        Controls multi-photon emission probability.
        Default: 3.145 × 10⁻⁵ (Erven et al. 2014 Table I)
    eta_total_transmittance : float
        Total transmission efficiency from source to detector.
        Includes source coupling, fiber loss, and detector efficiency.
        Must be in (0, 1]. Default: 0.0150
    e_det : float
        Intrinsic detection error rate of the system.
        Probability of click in wrong detector.
        Must be in [0, 0.5]. Default: 0.0093
    p_dark : float
        Dark count probability per coherence time.
        Must be in [0, 1]. Default: 1.50 × 10⁻⁸

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    References
    ----------
    - Erven et al. (2014) Table I: Experimental parameters.

    Examples
    --------
    >>> params = PhysicalParameters()  # Use defaults
    >>> params.mu_pair_per_coherence
    3.145e-05
    >>> params = PhysicalParameters(eta_total_transmittance=0.1)
    >>> params.eta_total_transmittance
    0.1
    """

    mu_pair_per_coherence: float = DEFAULT_MU_PAIR_PER_COHERENCE
    eta_total_transmittance: float = DEFAULT_ETA_TOTAL_TRANSMITTANCE
    e_det: float = DEFAULT_E_DET
    p_dark: float = DEFAULT_P_DARK

    def __post_init__(self) -> None:
        """Validate physical parameters."""
        if self.mu_pair_per_coherence <= 0:
            raise ValueError(
                f"mu_pair_per_coherence must be positive, got {self.mu_pair_per_coherence}"
            )

        if self.eta_total_transmittance <= 0 or self.eta_total_transmittance > 1:
            raise ValueError(
                f"eta_total_transmittance must be in (0, 1], got {self.eta_total_transmittance}"
            )

        if self.e_det < 0 or self.e_det > 0.5:
            raise ValueError(
                f"e_det must be in [0, 0.5], got {self.e_det}"
            )

        if self.p_dark < 0 or self.p_dark > 1:
            raise ValueError(
                f"p_dark must be in [0, 1], got {self.p_dark}"
            )


# =============================================================================
# NSM Security Parameters Dataclass
# =============================================================================


@dataclass(frozen=True)
class NSMSecurityParameters:
    """
    Noisy Storage Model security parameters.

    These parameters define the adversary model assumptions for security
    analysis. They are not directly measurable but represent assumptions
    about the adversary's quantum storage capabilities.

    Attributes
    ----------
    storage_noise_r : float
        Depolarizing channel retention parameter r ∈ [0, 1].
        Probability that stored qubit is preserved without noise.
        r = 0: Complete depolarization (maximum security)
        r = 1: Perfect storage (minimum security)
        Default: 0.75 (Erven et al. 2014)
    storage_rate_nu : float
        Storage rate ν ∈ [0, 1].
        Fraction of received qubits adversary can store.
        ν = 0: No storage capability
        ν = 1: Can store all qubits
        Default: 0.002 (Erven et al. 2014)
    delta_t_s : float
        Mandatory wait time Δt in seconds.
        Time between commitment and basis reveal.
        Default: 1.0 second (Erven et al. 2014)
    delta_t_ns : int
        Mandatory wait time Δt in nanoseconds (derived from delta_t_s).
        For simulation time integration.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    References
    ----------
    - Erven et al. (2014) Table I: Adversary's Memory Limitations.
    - König et al. (2012): NSM model definition.

    Notes
    -----
    The delta_t_ns is automatically derived from delta_t_s if not explicitly
    provided. If both are provided, they must be consistent.

    Examples
    --------
    >>> params = NSMSecurityParameters()
    >>> params.storage_noise_r
    0.75
    >>> params.delta_t_ns
    1000000000
    """

    storage_noise_r: float = DEFAULT_STORAGE_NOISE_R
    storage_rate_nu: float = DEFAULT_STORAGE_RATE_NU
    delta_t_s: float = DEFAULT_DELTA_T_S
    delta_t_ns: int = field(default=-1)  # Sentinel, will be computed

    def __post_init__(self) -> None:
        """Validate NSM security parameters and derive delta_t_ns."""
        if self.storage_noise_r < 0 or self.storage_noise_r > 1:
            raise ValueError(
                f"storage_noise_r must be in [0, 1], got {self.storage_noise_r}"
            )

        if self.storage_rate_nu < 0 or self.storage_rate_nu > 1:
            raise ValueError(
                f"storage_rate_nu must be in [0, 1], got {self.storage_rate_nu}"
            )

        if self.delta_t_s <= 0:
            raise ValueError(
                f"delta_t_s must be positive, got {self.delta_t_s}"
            )

        # Derive delta_t_ns from delta_t_s if not explicitly set
        if self.delta_t_ns == -1:
            derived_ns = int(self.delta_t_s * 1_000_000_000)
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(self, "delta_t_ns", derived_ns)
        elif self.delta_t_ns <= 0:
            raise ValueError(
                f"delta_t_ns must be positive, got {self.delta_t_ns}"
            )

        # Validate consistency if both were provided
        expected_ns = int(self.delta_t_s * 1_000_000_000)
        if abs(self.delta_t_ns - expected_ns) > 1000:  # 1 microsecond tolerance
            logger.warning(
                "delta_t_ns (%d) differs from delta_t_s-derived value (%d); "
                "using explicit delta_t_ns",
                self.delta_t_ns,
                expected_ns,
            )


# =============================================================================
# Protocol Parameters Dataclass
# =============================================================================


@dataclass(frozen=True)
class ProtocolParameters:
    """
    Protocol execution parameters.

    These parameters control protocol behavior including security/correctness
    targets and operational settings.

    Attributes
    ----------
    epsilon_sec : float
        Security parameter ε_sec ∈ (0, 1).
        Target trace distance from ideal functionality.
        Smaller values mean stronger security but require more resources.
        Default: 2.5 × 10⁻⁷ (Erven et al. 2014)
    epsilon_cor : float
        Correctness parameter ε_cor ∈ (0, 1).
        Upper bound on protocol failure probability.
        Default: 3.09 × 10⁻³ (dominated by EC failure)
    target_key_length : int | None
        Target final key length in bits.
        None means maximize extractable key.
    min_sifted_bits : int
        Minimum acceptable sifted bits before abort.
        Default: 10000

    Raises
    ------
    ValueError
        If parameters are outside valid ranges.
    """

    epsilon_sec: float = DEFAULT_EPSILON_SEC
    epsilon_cor: float = DEFAULT_EPSILON_COR
    target_key_length: Optional[int] = None
    min_sifted_bits: int = 10000

    def __post_init__(self) -> None:
        """Validate protocol parameters."""
        if self.epsilon_sec <= 0 or self.epsilon_sec >= 1:
            raise ValueError(
                f"epsilon_sec must be in (0, 1), got {self.epsilon_sec}"
            )

        if self.epsilon_cor <= 0 or self.epsilon_cor >= 1:
            raise ValueError(
                f"epsilon_cor must be in (0, 1), got {self.epsilon_cor}"
            )

        if self.target_key_length is not None and self.target_key_length <= 0:
            raise ValueError(
                f"target_key_length must be positive or None, got {self.target_key_length}"
            )

        if self.min_sifted_bits < 0:
            raise ValueError(
                f"min_sifted_bits must be non-negative, got {self.min_sifted_bits}"
            )


# =============================================================================
# Combined Protocol Configuration
# =============================================================================


@dataclass(frozen=True)
class ProtocolConfig:
    """
    Complete E-HOK protocol configuration.

    This is the top-level configuration object combining all parameter
    categories. It provides a single point of configuration for protocol
    execution.

    Attributes
    ----------
    physical : PhysicalParameters
        Hardware/channel characterization parameters.
    nsm_security : NSMSecurityParameters
        Noisy Storage Model adversary assumptions.
    protocol : ProtocolParameters
        Protocol execution settings.

    Class Methods
    -------------
    default()
        Create configuration with all default values from Erven et al. 2014.
    from_dict(d)
        Create configuration from a dictionary (e.g., loaded from YAML).

    References
    ----------
    - sprint_1_specification.md Section 3.3 (NOISE-PARAMS-001)

    Examples
    --------
    >>> config = ProtocolConfig.default()
    >>> config.physical.mu_pair_per_coherence
    3.145e-05
    >>> config.nsm_security.storage_noise_r
    0.75
    >>> config.protocol.epsilon_sec
    2.5e-07
    """

    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    nsm_security: NSMSecurityParameters = field(default_factory=NSMSecurityParameters)
    protocol: ProtocolParameters = field(default_factory=ProtocolParameters)

    @classmethod
    def default(cls) -> "ProtocolConfig":
        """
        Create configuration with default values from Erven et al. 2014.

        Returns
        -------
        ProtocolConfig
            Configuration with all literature default values.
        """
        return cls(
            physical=PhysicalParameters(),
            nsm_security=NSMSecurityParameters(),
            protocol=ProtocolParameters(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolConfig":
        """
        Create configuration from a dictionary.

        Parameters
        ----------
        d : dict
            Configuration dictionary, optionally with nested 'physical',
            'nsm_security', and 'protocol' sub-dictionaries.

        Returns
        -------
        ProtocolConfig
            Validated configuration object.

        Examples
        --------
        >>> config = ProtocolConfig.from_dict({
        ...     "physical": {"eta_total_transmittance": 0.1},
        ...     "nsm_security": {"storage_noise_r": 0.5}
        ... })
        """
        physical_dict = d.get("physical", {})
        nsm_dict = d.get("nsm_security", {})
        protocol_dict = d.get("protocol", {})

        return cls(
            physical=PhysicalParameters(**physical_dict),
            nsm_security=NSMSecurityParameters(**nsm_dict),
            protocol=ProtocolParameters(**protocol_dict),
        )

    def validate(self) -> list[str]:
        """
        Perform cross-field validation.

        Returns
        -------
        list[str]
            List of validation warnings (empty if no issues).

        Notes
        -----
        Individual field validation is done in __post_init__ of each
        dataclass. This method checks cross-field constraints.
        """
        warnings = []

        # Check that storage_noise_r makes security achievable
        # with the given epsilon_sec
        if self.nsm_security.storage_noise_r > 0.99:
            warnings.append(
                f"storage_noise_r={self.nsm_security.storage_noise_r:.4f} is very high; "
                "adversary's storage is nearly perfect"
            )

        # Check that storage_rate_nu is meaningful
        if self.nsm_security.storage_rate_nu > 0.5:
            warnings.append(
                f"storage_rate_nu={self.nsm_security.storage_rate_nu:.4f} is high; "
                "adversary can store significant fraction of qubits"
            )

        # Check that error rate is below hard limit
        if self.physical.e_det > 0.22:
            warnings.append(
                f"e_det={self.physical.e_det:.4f} exceeds 22% hard limit; "
                "protocol will likely be infeasible"
            )

        return warnings

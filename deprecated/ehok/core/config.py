"""
Protocol configuration objects for the extensible E-HOK framework.

This module centralises configuration so protocol runs are reproducible and
strategy selection is explicit. Values default to the conservative baselines in
``core.constants`` but can be overridden per simulation.

Configuration Categories
------------------------
1. **Physical Parameters**: Hardware/channel characteristics (μ, η, e_det, P_dark)
2. **NSM Config**: Adversary model assumptions (r, ν, Δt)
3. **Protocol Config**: Security/correctness targets (ε_sec, ε_cor)

Note
----
Physical parameters are consolidated from `configs/protocol_config.py` for
unified access. The original module is deprecated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, Any

from . import constants


# =============================================================================
# Physical Parameters (consolidated from configs/protocol_config.py)
# =============================================================================

# Default values from Erven et al. (2014) Table I
DEFAULT_MU_PAIR_PER_COHERENCE = 3.145e-5  # Mean photon pair number per coherence time
DEFAULT_ETA_TOTAL_TRANSMITTANCE = 0.0150  # Total transmission efficiency
DEFAULT_E_DET = 0.0093  # Intrinsic detection error rate
DEFAULT_P_DARK = 1.50e-8  # Dark count probability per coherence time


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
# Quantum Config
# =============================================================================


@dataclass
class QuantumConfig:
    """Quantum-layer parameters."""

    total_pairs: int = constants.TOTAL_EPR_PAIRS
    batch_size: int = constants.BATCH_SIZE
    max_qubits: int = 5

    def __post_init__(self) -> None:
        if self.total_pairs <= 0:
            raise ValueError("total_pairs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_qubits <= 0:
            raise ValueError("max_qubits must be positive")


@dataclass
class SecurityConfig:
    """Security thresholds and sampling parameters."""

    qber_threshold: float = constants.QBER_THRESHOLD
    target_epsilon: float = constants.TARGET_EPSILON_SEC
    test_set_fraction: float = constants.TEST_SET_FRACTION
    min_test_set_size: int = constants.MIN_TEST_SET_SIZE

    def __post_init__(self) -> None:
        if not 0 < self.qber_threshold < 1:
            raise ValueError("qber_threshold must be in (0,1)")
        if not 0 < self.target_epsilon < 1:
            raise ValueError("target_epsilon must be in (0,1)")
        if not 0 < self.test_set_fraction <= 1:
            raise ValueError("test_set_fraction must be in (0,1]")
        if self.min_test_set_size <= 0:
            raise ValueError("min_test_set_size must be positive")


@dataclass
class ReconciliationConfig:
    """Information reconciliation parameters."""

    code_rate: float = constants.LDPC_CODE_RATE
    max_iterations: int = constants.LDPC_MAX_ITERATIONS
    bp_threshold: float = constants.LDPC_BP_THRESHOLD
    matrix_path: Optional[str] = None
    
    testing_mode: bool = False
    """Enable test-specific LDPC matrices (USE ONLY IN TESTS)."""
    
    ldpc_test_frame_size: int | None = None
    """Test frame size override. Must be in constants.LDPC_TEST_FRAME_SIZES.
    
    Only effective when testing_mode=True. Production code ignores this field.
    """

    def __post_init__(self) -> None:
        if not 0 < self.code_rate < 1:
            raise ValueError("code_rate must be in (0,1)")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.bp_threshold <= 0:
            raise ValueError("bp_threshold must be positive")
        
        if self.testing_mode:
            if self.ldpc_test_frame_size is not None:
                if self.ldpc_test_frame_size not in constants.LDPC_TEST_FRAME_SIZES:
                    raise ValueError(
                        f"Invalid test frame size: {self.ldpc_test_frame_size}. "
                        f"Must be one of {constants.LDPC_TEST_FRAME_SIZES}"
                    )



@dataclass
class NSMConfig:
    """
    Noisy Storage Model security parameters.

    These parameters define the adversary model assumptions for NSM security.

    Attributes
    ----------
    storage_noise_r : float
        Depolarizing channel retention parameter r ∈ [0, 1].
        r = 0: Complete depolarization (maximum security).
        r = 1: Perfect storage (minimum security).
        Default: 0.75 (Erven et al. 2014).
    storage_rate_nu : float
        Fraction of qubits adversary can store ν ∈ [0, 1].
        Default: 0.002.
    delta_t_ns : int
        Mandatory wait time in nanoseconds for timing barrier.
        Default: 1_000_000_000 (1 second).
    """

    storage_noise_r: float = 0.75
    storage_rate_nu: float = 0.002
    delta_t_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.storage_noise_r <= 1.0:
            raise ValueError(
                f"storage_noise_r must be in [0, 1], got {self.storage_noise_r}"
            )
        if not 0.0 <= self.storage_rate_nu <= 1.0:
            raise ValueError(
                f"storage_rate_nu must be in [0, 1], got {self.storage_rate_nu}"
            )
        if self.delta_t_ns <= 0:
            raise ValueError(
                f"delta_t_ns must be positive, got {self.delta_t_ns}"
            )


@dataclass
class PrivacyAmplificationConfig:
    """
    Privacy amplification parameters with NSM finite-key security.

    The NSM finite-key formula automatically computes secure output lengths
    using the Max Bound: h_min(r) = max { Γ[1 - log₂(1 + 3r²)], 1 - r }.

    Attributes
    ----------
    target_epsilon_sec : float
        Target security parameter (trace distance from ideal key).
    target_epsilon_cor : float
        Correctness parameter (probability of key mismatch).
    test_bits_override : int, optional
        Override the number of test bits for finite-key calculation.
        If None, estimated from TEST_SET_FRACTION.
    use_fft_compression : bool
        Whether to use FFT-based O(n log n) compression for large keys.
    fft_threshold : int
        Key length above which FFT compression is used.

    Notes
    -----
    Legacy parameters (security_margin, fixed_output_length, target_epsilon)
    have been removed as part of NSM compliance. All key length calculations
    now use the NSM Max Bound formula exclusively.
    """

    target_epsilon_sec: float = constants.TARGET_EPSILON_SEC
    target_epsilon_cor: float = 1e-15
    test_bits_override: Optional[int] = None
    use_fft_compression: bool = False
    fft_threshold: int = 10000

    def __post_init__(self) -> None:
        if not 0 < self.target_epsilon_sec < 1:
            raise ValueError("target_epsilon_sec must be in (0,1)")
        if not 0 < self.target_epsilon_cor < 1:
            raise ValueError("target_epsilon_cor must be in (0,1)")
        if self.test_bits_override is not None and self.test_bits_override <= 0:
            raise ValueError("test_bits_override must be positive when set")
        if self.fft_threshold <= 0:
            raise ValueError("fft_threshold must be positive")


@dataclass
class ProtocolConfig:
    """Aggregated configuration for a protocol execution."""

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    privacy_amplification: PrivacyAmplificationConfig = field(
        default_factory=PrivacyAmplificationConfig
    )
    nsm: NSMConfig = field(default_factory=NSMConfig)
    sampling_seed: Optional[int] = None
    noise_model: Optional[str] = None
    approximate_knowledge_mask: bool = True

    def __post_init__(self) -> None:
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative when provided")
        if not isinstance(self.approximate_knowledge_mask, bool):
            raise ValueError("approximate_knowledge_mask must be a boolean")

    @classmethod
    def baseline(cls) -> "ProtocolConfig":
        """Return a configuration populated with baseline defaults."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)

    def copy_with(self, **kwargs: Any) -> "ProtocolConfig":
        """Create a modified copy with selected fields overridden."""
        return replace(self, **kwargs)
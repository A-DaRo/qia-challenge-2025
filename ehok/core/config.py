"""
Protocol configuration objects for the extensible E-HOK framework.

This module centralises configuration so protocol runs are reproducible and
strategy selection is explicit. Values default to the conservative baselines in
``core.constants`` but can be overridden per simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, Any

from . import constants


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
class PrivacyAmplificationConfig:
    """
    Privacy amplification parameters with finite-key security.

    The finite-key formula automatically computes secure output lengths
    without requiring arbitrary security margins.

    Attributes
    ----------
    target_epsilon_sec : float
        Target security parameter (trace distance from ideal key).
    target_epsilon_cor : float
        Correctness parameter (probability of key mismatch).
    test_bits_override : int, optional
        Override the number of test bits for finite-key calculation.
        If None, estimated from TEST_SET_FRACTION.
    use_finite_key : bool
        Whether to use the rigorous finite-key formula (recommended).
    use_fft_compression : bool
        Whether to use FFT-based O(n log n) compression for large keys.
    fft_threshold : int
        Key length above which FFT compression is used.

    Deprecated Attributes
    ---------------------
    security_margin : int
        **DEPRECATED**. Ignored by finite-key formula.
    fixed_output_length : int, optional
        **DEPRECATED**. No longer needed with finite-key formula.
    target_epsilon : float
        **DEPRECATED**. Use target_epsilon_sec instead.
    """

    # New finite-key parameters
    target_epsilon_sec: float = constants.TARGET_EPSILON_SEC
    target_epsilon_cor: float = 1e-15
    test_bits_override: Optional[int] = None
    use_finite_key: bool = True  # Enable finite-key formula by default
    use_fft_compression: bool = False
    fft_threshold: int = 10000

    # Deprecated parameters (kept for backwards compatibility)
    security_margin: int = constants.PA_SECURITY_MARGIN
    target_epsilon: float = constants.TARGET_EPSILON_SEC
    fixed_output_length: Optional[int] = None

    def __post_init__(self) -> None:
        # Validate new parameters
        if not 0 < self.target_epsilon_sec < 1:
            raise ValueError("target_epsilon_sec must be in (0,1)")
        if not 0 < self.target_epsilon_cor < 1:
            raise ValueError("target_epsilon_cor must be in (0,1)")
        if self.test_bits_override is not None and self.test_bits_override <= 0:
            raise ValueError("test_bits_override must be positive when set")
        if self.fft_threshold <= 0:
            raise ValueError("fft_threshold must be positive")

        # Validate deprecated parameters (for backwards compatibility)
        if self.security_margin < 0:
            raise ValueError("security_margin must be non-negative")
        if not 0 < self.target_epsilon < 1:
            raise ValueError("target_epsilon must be in (0,1)")
        if self.fixed_output_length is not None and self.fixed_output_length <= 0:
            raise ValueError("fixed_output_length must be positive when set")


@dataclass
class ProtocolConfig:
    """Aggregated configuration for a protocol execution."""

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    privacy_amplification: PrivacyAmplificationConfig = field(
        default_factory=PrivacyAmplificationConfig
    )
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
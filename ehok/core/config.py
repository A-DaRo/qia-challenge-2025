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

    def __post_init__(self) -> None:
        if not 0 < self.code_rate < 1:
            raise ValueError("code_rate must be in (0,1)")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.bp_threshold <= 0:
            raise ValueError("bp_threshold must be positive")


@dataclass
class PrivacyAmplificationConfig:
    """Privacy amplification parameters."""

    security_margin: int = constants.PA_SECURITY_MARGIN
    target_epsilon: float = constants.TARGET_EPSILON_SEC
    # Optional explicit output length override
    fixed_output_length: Optional[int] = None

    def __post_init__(self) -> None:
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
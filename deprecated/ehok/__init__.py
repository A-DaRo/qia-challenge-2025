"""
E-HOK: Entangled Hybrid Oblivious Key Distribution

This package implements the E-HOK protocol for generating oblivious keys
suitable for secure multiparty computation applications.

Based on:
- Lemus et al., "Quantum Oblivious Key Distribution" (arXiv:1909.11701)
- Lemus et al., "Revised E-HOK" (arXiv:2501.03973)

Package Structure
-----------------
core/
    Core data structures, exceptions, and constants
interfaces/
    Abstract base classes defining component interfaces
implementations/
    Concrete implementations of commitment, reconciliation, and privacy amplification
quantum/
    Quantum operations: EPR generation, measurement, basis selection
protocols/
    Alice and Bob protocol implementations (SquidASM Programs)
utils/
    Logging and helper utilities
configs/
    Network configurations and LDPC matrices
"""

__version__ = "0.1.0"
__author__ = "QIA Challenge 2025 Team"

# Expose key classes for convenient imports
from ehok.core.data_structures import (
    ObliviousKey,
    MeasurementRecord,
    ProtocolResult,
    ExecutionMetrics,
)
from ehok.core.config import ProtocolConfig
from ehok.core.exceptions import (
    EHOKException,
    SecurityException,
    ProtocolError,
    QBERTooHighError,
    ReconciliationFailedError,
    CommitmentVerificationError,
)
from ehok.core.constants import (
    QBER_THRESHOLD,
    TARGET_EPSILON_SEC,
    TEST_SET_FRACTION,
    TOTAL_EPR_PAIRS,
    BATCH_SIZE,
)

__all__ = [
    # Data structures
    "ObliviousKey",
    "MeasurementRecord",
    "ProtocolResult",
    "ExecutionMetrics",
    "ProtocolConfig",
    # Exceptions
    "EHOKException",
    "SecurityException",
    "ProtocolError",
    "QBERTooHighError",
    "ReconciliationFailedError",
    "CommitmentVerificationError",
    # Constants
    "QBER_THRESHOLD",
    "TARGET_EPSILON_SEC",
    "TEST_SET_FRACTION",
    "TOTAL_EPR_PAIRS",
    "BATCH_SIZE",
]

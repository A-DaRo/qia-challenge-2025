"""
Caligo types package: Domain primitives and phase contracts.

This package defines the canonical vocabulary for the Caligo codebase,
including dataclasses for phase boundaries, measurement records, and keys.
"""

from caligo.types.keys import (
    ObliviousKey,
    AliceObliviousKey,
    BobObliviousKey,
)

from caligo.types.measurements import (
    MeasurementRecord,
    RoundResult,
    DetectionEvent,
)

from caligo.types.phase_contracts import (
    QuantumPhaseResult,
    SiftingPhaseResult,
    ReconciliationPhaseResult,
    AmplificationPhaseResult,
    ObliviousTransferOutput,
)

from caligo.types.exceptions import (
    CaligoError,
    SimulationError,
    TimingViolationError,
    NetworkConfigError,
    UnsupportedHardwareError,
    EPRGenerationError,
    SecurityError,
    QBERThresholdExceeded,
    NSMViolationError,
    FeasibilityError,
    EntropyDepletedError,
    CommitmentVerificationError,
    ProtocolError,
    PhaseOrderViolation,
    ContractViolation,
    ReconciliationError,
    ConnectionError,
    OrderingViolationError,
    AckTimeoutError,
    SessionMismatchError,
    OutOfOrderError,
    ConfigurationError,
    InvalidParameterError,
    MissingConfigError,
    ProtocolPhase,
    AbortReason,
)

__all__ = [
    # Keys
    "ObliviousKey",
    "AliceObliviousKey",
    "BobObliviousKey",
    # Measurements
    "MeasurementRecord",
    "RoundResult",
    "DetectionEvent",
    # Phase contracts
    "QuantumPhaseResult",
    "SiftingPhaseResult",
    "ReconciliationPhaseResult",
    "AmplificationPhaseResult",
    "ObliviousTransferOutput",
    # Exceptions
    "CaligoError",
    "SimulationError",
    "TimingViolationError",
    "NetworkConfigError",
    "UnsupportedHardwareError",
    "EPRGenerationError",
    "SecurityError",
    "QBERThresholdExceeded",
    "NSMViolationError",
    "FeasibilityError",
    "EntropyDepletedError",
    "CommitmentVerificationError",
    "ProtocolError",
    "PhaseOrderViolation",
    "ContractViolation",
    "ReconciliationError",
    "ConnectionError",
    "OrderingViolationError",
    "AckTimeoutError",
    "SessionMismatchError",
    "OutOfOrderError",
    "ConfigurationError",
    "InvalidParameterError",
    "MissingConfigError",
    # Enums
    "ProtocolPhase",
    "AbortReason",
]

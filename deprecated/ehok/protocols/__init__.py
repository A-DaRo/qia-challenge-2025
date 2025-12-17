"""
E-HOK protocol implementations.

This subpackage contains the SquidASM Program implementations for Alice and Bob,
orchestrating the complete E-HOK protocol execution.

Sprint 2 Additions
------------------
- ordered_messaging: Commit-then-Reveal ordered socket with ACK semantics
- statistical_validation: Detection validation, finite-size penalty, QBER adjustment
- leakage_manager: Reconciliation leakage tracking and safety cap enforcement
"""

from ehok.protocols.ordered_messaging import (
    # Constants
    DEFAULT_ACK_TIMEOUT_NS,
    BASIS_Z,
    BASIS_X,
    # Enums
    MessageType,
    SocketState,
    # Exceptions
    OrderingViolationError,
    AckTimeoutError,
    DuplicateMessageError,
    OutOfOrderError,
    # Data structures
    MessageEnvelope,
    AckPayload,
    OrderedSocketState,
    OrderedProtocolSocket,
    DetectionReport,
    BasisReveal,
)

from ehok.protocols.statistical_validation import (
    # Abort codes
    ABORT_CODE_DETECTION_ANOMALY,
    ABORT_CODE_QBER_HIGH,
    # Enums
    ValidationStatus,
    # Result dataclasses
    DetectionValidationResult,
    FiniteSizePenaltyResult,
    QBERAdjustmentResult,
    # Classes
    DetectionValidator,
    QBERAdjuster,
    # Functions
    compute_finite_size_penalty,
    adjust_qber,
)

from ehok.protocols.leakage_manager import (
    # Constants
    DEFAULT_MAX_LEAKAGE_BITS,
    ABORT_CODE_LEAKAGE_CAP_EXCEEDED,
    # Data structures
    BlockReconciliationReport,
    LeakageState,
    # Classes
    LeakageSafetyManager,
    # Functions
    compute_max_leakage_budget,
)

__all__ = [
    # Ordered Messaging
    "DEFAULT_ACK_TIMEOUT_NS",
    "BASIS_Z",
    "BASIS_X",
    "MessageType",
    "SocketState",
    "OrderingViolationError",
    "AckTimeoutError",
    "DuplicateMessageError",
    "OutOfOrderError",
    "MessageEnvelope",
    "AckPayload",
    "OrderedSocketState",
    "OrderedProtocolSocket",
    "DetectionReport",
    "BasisReveal",
    # Statistical Validation
    "ABORT_CODE_DETECTION_ANOMALY",
    "ABORT_CODE_QBER_HIGH",
    "ValidationStatus",
    "DetectionValidationResult",
    "FiniteSizePenaltyResult",
    "QBERAdjustmentResult",
    "DetectionValidator",
    "QBERAdjuster",
    "compute_finite_size_penalty",
    "adjust_qber",
    # Leakage Manager
    "DEFAULT_MAX_LEAKAGE_BITS",
    "ABORT_CODE_LEAKAGE_CAP_EXCEEDED",
    "BlockReconciliationReport",
    "LeakageState",
    "LeakageSafetyManager",
    "compute_max_leakage_budget",
]

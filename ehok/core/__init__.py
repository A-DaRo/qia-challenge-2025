"""
Core E-HOK components.

This subpackage contains the fundamental data structures, exceptions, and
constants that form the foundation of the E-HOK implementation.
"""

from ehok.core.data_structures import ObliviousKey, MeasurementRecord, ProtocolResult
from ehok.core.exceptions import (
    EHOKException,
    SecurityException,
    ProtocolError,
    QBERTooHighError,
    ReconciliationFailedError,
    CommitmentVerificationError,
)
from ehok.core import constants
from ehok.core.timing import (
    TimingConfig,
    TimingEnforcer,
    TimingState,
    TimingViolationError,
    TimingStateError,
)
from ehok.core.feasibility import (
    FeasibilityInputs,
    FeasibilityDecision,
    FeasibilityChecker,
    ABORT_CODE_QBER_TOO_HIGH,
    ABORT_CODE_STRICT_LESS_VIOLATED,
    ABORT_CODE_CAPACITY_RATE_VIOLATED,
    ABORT_CODE_DEATH_VALLEY,
    ABORT_CODE_INVALID_PARAMETERS,
)
from ehok.core.oblivious_formatter import (
    AliceObliviousKey,
    BobObliviousKey,
    ProtocolMetrics,
    ObliviousTransferResult,
    ObliviousKeyFormatter,
    validate_ot_correctness,
)

__all__ = [
    "ObliviousKey",
    "MeasurementRecord",
    "ProtocolResult",
    "EHOKException",
    "SecurityException",
    "ProtocolError",
    "QBERTooHighError",
    "ReconciliationFailedError",
    "CommitmentVerificationError",
    "constants",
    "TimingConfig",
    "TimingEnforcer",
    "TimingState",
    "TimingViolationError",
    "TimingStateError",
    "FeasibilityInputs",
    "FeasibilityDecision",
    "FeasibilityChecker",
    "ABORT_CODE_QBER_TOO_HIGH",
    "ABORT_CODE_STRICT_LESS_VIOLATED",
    "ABORT_CODE_CAPACITY_RATE_VIOLATED",
    "ABORT_CODE_DEATH_VALLEY",
    "ABORT_CODE_INVALID_PARAMETERS",
    # Sprint 3 OT output structures
    "AliceObliviousKey",
    "BobObliviousKey",
    "ProtocolMetrics",
    "ObliviousTransferResult",
    "ObliviousKeyFormatter",
    "validate_ot_correctness",
]

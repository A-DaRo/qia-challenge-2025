"""
Reconciliation Subpackage for Caligo Information Reconciliation.

This package provides LDPC-based reconciliation with support for:
- Rate-adaptive reconciliation (puncturing/shortening)
- Blind reconciliation protocol (Martinez-Mateo et al.)
- Runtime reconciliation type selection via factory pattern
- Wiretap cost accounting and safety cap enforcement

Modules
-------
constants
    Reconciliation parameters and thresholds.
ldpc_decoder
    Belief-propagation decoder for LDPC codes.
ldpc_encoder
    Syndrome computation for Alice side.
hash_verifier
    Polynomial hash for block verification.
rate_selector
    Adaptive rate selection based on QBER.
matrix_manager
    LDPC matrix pool loading and caching.
leakage_tracker
    Wiretap cost accumulation and safety cap.
blind_manager
    Blind reconciliation iteration protocol.
orchestrator
    Phase III coordinator.
factory
    Runtime reconciliation type selection.

References
----------
- Martinez-Mateo et al. (2012): Blind Reconciliation
- Elkouss et al. (2009): Rate-compatible LDPC codes for QKD
- Kiktenko et al. (2016): Industrial QKD post-processing
- Erven et al. (2014): Experimental NSM implementation
"""

# Constants
from caligo.reconciliation.constants import (
    LDPC_FRAME_SIZE,
    LDPC_CODE_RATES,
    LDPC_DEFAULT_RATE,
    LDPC_MAX_ITERATIONS,
    LDPC_F_CRIT,
    LDPC_HASH_BITS,
    BLIND_MAX_ITERATIONS,
    BLIND_MODULATION_FRACTION,
    # Phase 1: Architectural constants
    MOTHER_CODE_RATE,
    RATE_STEP,
    UNTAINTED_SATURATION_RATE,
    RATE_MIN,
    RATE_MAX,
    NUMBA_CACHE_ENABLED,
    NUMBA_PARALLEL_ENABLED,
    NUMBA_FASTMATH_ENABLED,
)

# Decoder
from caligo.reconciliation.ldpc_decoder import (
    BeliefPropagationDecoder,
    DecodeResult,
    build_channel_llr,
    syndrome_guided_refinement,
)

# Encoder
from caligo.reconciliation.ldpc_encoder import (
    SyndromeBlock,
    compute_syndrome,
    encode_block_from_payload,
    prepare_frame,
)

# Hash Verifier
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier

# Rate Selector
from caligo.reconciliation.rate_selector import (
    RateSelection,
    select_rate,
    compute_shortening,
    compute_puncturing,
    select_rate_with_parameters,
)

# Matrix Manager
from caligo.reconciliation.matrix_manager import (
    MatrixPool,
    MatrixManager,
)

# Leakage Tracker
from caligo.reconciliation.leakage_tracker import (
    LeakageRecord,
    LeakageTracker,
    compute_safety_cap,
)

# Blind Manager
from caligo.reconciliation.blind_manager import (
    BlindConfig,
    BlindIterationState,
    BlindReconciliationManager,
)

# Orchestrator
from caligo.reconciliation.orchestrator import (
    ReconciliationOrchestratorConfig,
    BlockResult,
    ReconciliationOrchestrator,
    partition_key,
)

# Single-block reconciler
from caligo.reconciliation.block_reconciler import (
    BlockReconciler,
    BlockReconcilerConfig,
)

# Factory (existing)
from caligo.reconciliation.factory import (
    ReconciliationType,
    ReconciliationConfig,
    create_reconciler,
    create_strategy,  # Phase 1: Strategy factory
)

# Strategy Pattern (Phase 1: Foundation) - DISABLED until strategies implemented
# from caligo.reconciliation.strategies import (
#     BlockResult as StrategyBlockResult,
#     ReconciliationContext,
#     ReconciliationStrategy,
# )

__all__ = [
    # Constants
    "LDPC_FRAME_SIZE",
    "LDPC_CODE_RATES",
    "LDPC_DEFAULT_RATE",
    "LDPC_MAX_ITERATIONS",
    "LDPC_F_CRIT",
    "LDPC_HASH_BITS",
    "BLIND_MAX_ITERATIONS",
    "BLIND_MODULATION_FRACTION",
    # Phase 1: Architectural constants
    "MOTHER_CODE_RATE",
    "RATE_STEP",
    "UNTAINTED_SATURATION_RATE",
    "RATE_MIN",
    "RATE_MAX",
    "NUMBA_CACHE_ENABLED",
    "NUMBA_PARALLEL_ENABLED",
    "NUMBA_FASTMATH_ENABLED",
    # Decoder
    "BeliefPropagationDecoder",
    "DecodeResult",
    "build_channel_llr",
    "syndrome_guided_refinement",
    # Encoder
    "SyndromeBlock",
    "compute_syndrome",
    "encode_block_from_payload",
    "prepare_frame",
    # Hash Verifier
    "PolynomialHashVerifier",
    # Rate Selector
    "RateSelection",
    "select_rate",
    "compute_shortening",
    "compute_puncturing",
    "select_rate_with_parameters",
    # Matrix Manager
    "MatrixPool",
    "MatrixManager",
    # Leakage Tracker
    "LeakageRecord",
    "LeakageTracker",
    "compute_safety_cap",
    # Blind Manager
    "BlindConfig",
    "BlindIterationState",
    "BlindReconciliationManager",
    # Orchestrator
    "ReconciliationOrchestratorConfig",
    "BlockResult",
    "ReconciliationOrchestrator",
    "partition_key",
    # Block reconciler
    "BlockReconciler",
    "BlockReconcilerConfig",
    # Factory
    "ReconciliationType",
    "ReconciliationConfig",
    "create_reconciler",
    "create_strategy",  # Phase 1: Strategy factory
    # Strategy Pattern (Phase 1)
    "StrategyBlockResult",
    "ReconciliationContext",
    "ReconciliationStrategy",
]

"""
Caligo: E-HOK Protocol Implementation with Native SquidASM Integration.

Caligo (Latin: "fog/mist") implements the E-HOK (Efficient High-dimensional
Oblivious Key) protocol for 1-out-of-2 Oblivious Transfer using the Noisy
Storage Model (NSM).

Public API
----------
Types (from caligo.types):
    ObliviousKey, AliceObliviousKey, BobObliviousKey
    MeasurementRecord, RoundResult, DetectionEvent
    QuantumPhaseResult, SiftingPhaseResult, ReconciliationPhaseResult
    AmplificationPhaseResult, ObliviousTransferOutput
    ProtocolPhase, AbortReason
    CaligoError, SecurityError, ProtocolError (and subclasses)

Utilities (from caligo.utils):
    get_logger, setup_script_logging
    binary_entropy, channel_capacity, finite_size_penalty, gamma_function
    xor_bitarrays, hamming_distance, random_bitarray

Simulation (from caligo.simulation):
    NSMParameters, ChannelParameters
    TimingBarrier, TimingBarrierState
    NSMStorageNoiseModel, ChannelNoiseProfile
    CaligoNetworkBuilder
    perfect_network_config, realistic_network_config, erven_experimental_config

Quantum (from caligo.quantum):
    EPRGenerator, BasisSelector, MeasurementExecutor, MeasurementBuffer
    BatchingManager, BatchConfig, BatchResult

Sifting (from caligo.sifting):
    SHA256Commitment, Sifter, QBEREstimator, DetectionValidator

Amplification (from caligo.amplification):
    NSMEntropyCalculator, SecureKeyLengthCalculator
    ToeplitzHasher, OTOutputFormatter
"""

from caligo.types import (
    # Keys
    ObliviousKey,
    AliceObliviousKey,
    BobObliviousKey,
    # Measurements
    MeasurementRecord,
    RoundResult,
    DetectionEvent,
    # Phase contracts
    QuantumPhaseResult,
    SiftingPhaseResult,
    ReconciliationPhaseResult,
    AmplificationPhaseResult,
    ObliviousTransferOutput,
    # Exceptions and enums
    CaligoError,
    SimulationError,
    SecurityError,
    ProtocolError,
    ConnectionError,
    ConfigurationError,
    ProtocolPhase,
    AbortReason,
)

from caligo.utils import (
    get_logger,
    setup_script_logging,
    binary_entropy,
    channel_capacity,
    finite_size_penalty,
    gamma_function,
    smooth_min_entropy_rate,
    key_length_bound,
    xor_bitarrays,
    hamming_distance,
    random_bitarray,
    bitarray_to_bytes,
    bytes_to_bitarray,
    slice_bitarray,
    bitarray_from_numpy,
    bitarray_to_numpy,
)

from caligo.simulation import (
    # Physical model
    NSMParameters,
    ChannelParameters,
    create_depolar_noise_model,
    create_t1t2_noise_model,
    # Timing
    TimingBarrier,
    TimingBarrierState,
    # Noise models
    NSMStorageNoiseModel,
    ChannelNoiseProfile,
    # Network builder
    CaligoNetworkBuilder,
    perfect_network_config,
    realistic_network_config,
    erven_experimental_config,
)

from caligo.security import (
    # Constants
    QBER_CONSERVATIVE_THRESHOLD,
    QBER_ABSOLUTE_THRESHOLD,
    R_TILDE,
    R_CROSSOVER,
    DEFAULT_EPSILON_SEC,
    DEFAULT_EPSILON_COR,
    # Bounds
    collision_entropy_rate,
    dupuis_konig_bound,
    lupo_virtual_erasure_bound,
    max_bound_entropy,
    rational_adversary_bound,
    bounded_storage_entropy,
    strong_converse_exponent,
    # Feasibility
    FeasibilityChecker,
    FeasibilityResult,
    PreflightReport,
    compute_expected_qber,
    # Finite-key
    compute_statistical_fluctuation,
    hoeffding_detection_interval,
    compute_finite_key_length,
)

# Phase D: Quantum operations
from caligo.quantum import (
    EPRGenerator,
    EPRGenerationConfig,
    BasisSelector,
    MeasurementExecutor,
    MeasurementBuffer,
    BatchingManager,
    BatchConfig,
    BatchResult,
)

# Phase D: Sifting
from caligo.sifting import (
    SHA256Commitment,
    CommitmentResult,
    Sifter,
    SiftingResult,
    QBEREstimator,
    QBEREstimate,
    DetectionValidator,
    ValidationResult,
    HoeffdingBound,
)

# Phase D: Privacy Amplification
from caligo.amplification import (
    NSMEntropyCalculator,
    SecureKeyLengthCalculator,
    KeyLengthResult,
    ToeplitzHasher,
    OTOutputFormatter,
    AliceOTOutput,
    BobOTOutput,
)

__version__ = "0.1.0"
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
    # Exceptions and enums
    "CaligoError",
    "SimulationError",
    "SecurityError",
    "ProtocolError",
    "ConnectionError",
    "ConfigurationError",
    "ProtocolPhase",
    "AbortReason",
    # Utility functions
    "get_logger",
    "setup_script_logging",
    "binary_entropy",
    "channel_capacity",
    "finite_size_penalty",
    "gamma_function",
    "smooth_min_entropy_rate",
    "key_length_bound",
    "xor_bitarrays",
    "hamming_distance",
    "random_bitarray",
    "bitarray_to_bytes",
    "bytes_to_bitarray",
    "slice_bitarray",
    "bitarray_from_numpy",
    "bitarray_to_numpy",
    # Simulation - Physical model
    "NSMParameters",
    "ChannelParameters",
    "create_depolar_noise_model",
    "create_t1t2_noise_model",
    # Simulation - Timing
    "TimingBarrier",
    "TimingBarrierState",
    # Simulation - Noise models
    "NSMStorageNoiseModel",
    "ChannelNoiseProfile",
    # Simulation - Network builder
    "CaligoNetworkBuilder",
    "perfect_network_config",
    "realistic_network_config",
    "erven_experimental_config",
    # Security - Constants
    "QBER_CONSERVATIVE_THRESHOLD",
    "QBER_ABSOLUTE_THRESHOLD",
    "R_TILDE",
    "R_CROSSOVER",
    "DEFAULT_EPSILON_SEC",
    "DEFAULT_EPSILON_COR",
    # Security - Bounds
    "collision_entropy_rate",
    "dupuis_konig_bound",
    "lupo_virtual_erasure_bound",
    "max_bound_entropy",
    "rational_adversary_bound",
    "bounded_storage_entropy",
    "strong_converse_exponent",
    # Security - Feasibility
    "FeasibilityChecker",
    "FeasibilityResult",
    "PreflightReport",
    "compute_expected_qber",
    # Security - Finite-key
    "compute_statistical_fluctuation",
    "hoeffding_detection_interval",
    "compute_finite_key_length",
    # Quantum operations (Phase D)
    "EPRGenerator",
    "EPRGenerationConfig",
    "BasisSelector",
    "MeasurementExecutor",
    "MeasurementBuffer",
    "BatchingManager",
    "BatchConfig",
    "BatchResult",
    # Sifting (Phase D)
    "SHA256Commitment",
    "CommitmentResult",
    "Sifter",
    "SiftingResult",
    "QBEREstimator",
    "QBEREstimate",
    "DetectionValidator",
    "ValidationResult",
    "HoeffdingBound",
    # Amplification (Phase D)
    "NSMEntropyCalculator",
    "SecureKeyLengthCalculator",
    "KeyLengthResult",
    "ToeplitzHasher",
    "OTOutputFormatter",
    "AliceOTOutput",
    "BobOTOutput",
]

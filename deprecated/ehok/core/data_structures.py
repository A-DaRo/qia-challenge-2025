"""
Core data structures for the E-HOK protocol.

This module defines the fundamental data structures used throughout the E-HOK
baseline implementation, including the oblivious key representation, measurement
records, and protocol execution results.

Phase Boundary Contracts
------------------------
This module defines canonical dataclass contracts at phase boundaries for the
E-HOK protocol, enabling:
- Deterministic parity tests between legacy and new implementations
- Type-safe data exchange between protocol phases
- Runtime validation of protocol invariants via Design-by-Contract

Contract Map
------------
===============================  ===================================  ===============================
Phase Boundary                   Input Contract                       Output Contract
===============================  ===================================  ===============================
Phase I → Phase II               —                                    QuantumPhaseOutput
Phase II → Phase III             QuantumPhaseOutput                   SiftedKeyData
Phase III → Phase IV             SiftedKeyData                        ReconciledKeyData
Phase IV output                  ReconciledKeyData                    ObliviousTransferOutput
===============================  ===================================  ===============================

References
----------
- sprint_0_specification.md (INFRA-002)
- phase_I_analysis.md (timing semantics, NSM invariants)
- phase_II_analysis.md (ordered acknowledgment, Chernoff validation)
- phase_III_analysis.md (wiretap cost, leakage accounting)
- phase_IV_analysis.md (NSM max bound, min-entropy)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .exceptions import EHOKException


# =============================================================================
# Enumerations for Protocol State
# =============================================================================


class ProtocolPhase(Enum):
    """
    Enumeration of E-HOK protocol phases.

    Used for protocol state tracking and transcript annotation.
    """

    PHASE_I_QUANTUM = auto()
    PHASE_II_SIFTING = auto()
    PHASE_III_RECONCILIATION = auto()
    PHASE_IV_AMPLIFICATION = auto()
    COMPLETED = auto()
    ABORTED = auto()


class AbortReason(Enum):
    """
    Enumeration of protocol abort reasons with associated codes.

    Codes follow the taxonomy defined in phase_I_analysis.md Section 7.3:
    - ABORT-I-*: Phase I abort conditions
    - ABORT-II-*: Phase II abort conditions
    - ABORT-III-*: Phase III abort conditions
    - ABORT-IV-*: Phase IV abort conditions
    """

    # Phase I abort reasons
    FEASIBILITY_HARD_LIMIT = "ABORT-I-FEAS-001"  # Q_total > 22%
    TIMING_VIOLATION = "ABORT-I-TIMING-001"  # Basis revealed before Δt

    # Phase II abort reasons
    DETECTION_ANOMALY = "ABORT-II-DET-001"  # Chernoff bound violated
    QBER_HARD_LIMIT = "ABORT-II-QBER-001"  # Adjusted QBER > 22%
    MISSING_ROUNDS_INVALID = "ABORT-II-MISS-001"  # Invalid missing rounds report

    # Phase III abort reasons
    LEAKAGE_CAP_EXCEEDED = "ABORT-III-LEAK-001"  # Syndrome leakage > safety cap
    RECONCILIATION_FAILED = "ABORT-III-REC-001"  # Decoder failure rate too high
    VERIFICATION_FAILED = "ABORT-III-VER-001"  # Hash verification failed

    # Phase IV abort reasons
    ENTROPY_DEPLETED = "ABORT-IV-ENT-001"  # No extractable entropy after penalties
    KEY_LENGTH_ZERO = "ABORT-IV-LEN-001"  # Final key length is zero


class WarningCode(Enum):
    """
    Enumeration of protocol warning codes (non-fatal).

    Warnings indicate suboptimal conditions that do not require abort.
    """

    QBER_CONSERVATIVE_LIMIT = "WARN-I-FEAS-001"  # 11% < Q_total ≤ 22%
    LOW_DETECTION_RATE = "WARN-II-DET-001"  # Detection below expected but within Chernoff
    HIGH_RECONCILIATION_LEAKAGE = "WARN-III-LEAK-001"  # Leakage approaching safety cap


# =============================================================================
# Phase I → Phase II Contract: QuantumPhaseOutput
# =============================================================================


@dataclass
class TimingMarker:
    """
    Marker recording a protocol timing event for NSM causal barrier enforcement.

    The NSM security model requires strict ordering of protocol events. This
    dataclass captures timestamps for enforcement and audit.

    Attributes
    ----------
    event_type : str
        Type of timing event (e.g., "COMMITMENT_SENT", "TIMING_BARRIER_START").
    timestamp_ns : int
        Simulation timestamp in nanoseconds when event occurred.
    description : str
        Human-readable description of the event.

    References
    ----------
    - phase_I_analysis.md Section 7.1 (INV-PHI-001: Causality Barrier)
    - König et al. (2012) Section I-C (timing semantics)
    """

    event_type: str
    timestamp_ns: int
    description: str = ""

    def __post_init__(self) -> None:
        """Validate timing marker invariants."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not self.event_type:
            raise ValueError("event_type must be a non-empty string")


@dataclass
class CommitmentRecord:
    """
    Record of a cryptographic commitment for protocol transcript.

    Captures the commitment hash, timing, and verification status for audit.

    Attributes
    ----------
    commitment_hash : bytes
        The cryptographic commitment value (e.g., SHA-256 hash).
    salt : bytes
        Random salt used in commitment generation.
    timestamp_ns : int
        Simulation timestamp when commitment was created.
    verified : Optional[bool]
        Verification status (None if not yet verified).
    data_length : int
        Length of committed data in bytes.

    References
    ----------
    - phase_I_analysis.md Section 4.3.2 (Commitment Scheme)
    - Lemus et al. (2020) (hybrid commitment architecture)
    """

    commitment_hash: bytes
    salt: bytes
    timestamp_ns: int
    verified: Optional[bool] = None
    data_length: int = 0

    def __post_init__(self) -> None:
        """Validate commitment record invariants."""
        if not isinstance(self.commitment_hash, bytes) or len(self.commitment_hash) == 0:
            raise ValueError("commitment_hash must be non-empty bytes")
        if not isinstance(self.salt, bytes) or len(self.salt) == 0:
            raise ValueError("salt must be non-empty bytes")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.data_length < 0:
            raise ValueError("data_length must be non-negative")


@dataclass
class QuantumPhaseOutput:
    """
    Output contract for Phase I: Quantum Generation.

    This dataclass represents the complete output of Phase I, providing all
    necessary data for Phase II sifting and estimation. It enforces NSM-specific
    invariants including timing markers for causal barrier verification.

    Attributes
    ----------
    outcomes_alice : np.ndarray
        Alice's measurement outcomes, shape (n_pairs,), dtype uint8, values 0/1.
    outcomes_bob : np.ndarray
        Bob's measurement outcomes, shape (n_pairs,), dtype uint8, values 0/1.
    bases_alice : np.ndarray
        Alice's basis choices, shape (n_pairs,), dtype uint8, values 0 (Z) / 1 (X).
    bases_bob : np.ndarray
        Bob's basis choices, shape (n_pairs,), dtype uint8, values 0 (Z) / 1 (X).
    n_pairs : int
        Total number of EPR pairs generated (length of all arrays).
    start_timestamp_ns : int
        Simulation timestamp when quantum generation started.
    end_timestamp_ns : int
        Simulation timestamp when quantum generation completed.
    timing_markers : List[TimingMarker]
        Ordered list of timing events for causal barrier verification.
    commitment : Optional[CommitmentRecord]
        Bob's detection commitment (if commit-then-reveal enabled).
    missing_rounds : Optional[np.ndarray]
        Indices where Bob reported no detection, dtype int64.
    expected_detection_rate : float
        Expected detection rate from channel calibration (for Chernoff validation).
    channel_fidelity : float
        Configured channel fidelity (1 - depolarizing probability).

    Invariants
    ----------
    - POST-PHI-001: len(outcomes_*) == len(bases_*) == n_pairs
    - POST-PHI-002: All arrays have dtype uint8
    - INV-PHI-001: Timing markers must show commitment before basis reveal

    References
    ----------
    - sprint_0_specification.md (INFRA-002)
    - phase_I_analysis.md Section 7.2 (POST-PHI-001, POST-PHI-002)
    """

    outcomes_alice: np.ndarray
    outcomes_bob: np.ndarray
    bases_alice: np.ndarray
    bases_bob: np.ndarray
    n_pairs: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    timing_markers: List[TimingMarker] = field(default_factory=list)
    commitment: Optional[CommitmentRecord] = None
    missing_rounds: Optional[np.ndarray] = None
    expected_detection_rate: float = 1.0
    channel_fidelity: float = 1.0

    def __post_init__(self) -> None:
        """Validate quantum phase output invariants (POST-PHI-001, POST-PHI-002)."""
        # Check array lengths (POST-PHI-001)
        arrays = [
            ("outcomes_alice", self.outcomes_alice),
            ("outcomes_bob", self.outcomes_bob),
            ("bases_alice", self.bases_alice),
            ("bases_bob", self.bases_bob),
        ]

        for name, arr in arrays:
            if len(arr) != self.n_pairs:
                raise ValueError(
                    f"{name} length {len(arr)} does not match n_pairs {self.n_pairs}"
                )

        # Check dtypes (POST-PHI-002)
        for name, arr in arrays:
            if arr.dtype != np.uint8:
                raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")

        # Check value ranges
        for name, arr in [
            ("outcomes_alice", self.outcomes_alice),
            ("outcomes_bob", self.outcomes_bob),
        ]:
            if not np.all((arr == 0) | (arr == 1)):
                raise ValueError(f"{name} values must be 0 or 1")

        for name, arr in [
            ("bases_alice", self.bases_alice),
            ("bases_bob", self.bases_bob),
        ]:
            if not np.all((arr == 0) | (arr == 1)):
                raise ValueError(f"{name} values must be 0 (Z) or 1 (X)")

        # Check timestamp ordering
        if self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("end_timestamp_ns must be >= start_timestamp_ns")

        # Check detection rate
        if not 0.0 <= self.expected_detection_rate <= 1.0:
            raise ValueError("expected_detection_rate must be in [0, 1]")

        # Check fidelity
        if not 0.0 <= self.channel_fidelity <= 1.0:
            raise ValueError("channel_fidelity must be in [0, 1]")

        # Validate missing_rounds if present
        if self.missing_rounds is not None:
            if self.missing_rounds.dtype not in (np.int64, np.int32):
                raise ValueError("missing_rounds must have integer dtype")
            if len(self.missing_rounds) > 0:
                if np.min(self.missing_rounds) < 0:
                    raise ValueError("missing_rounds indices must be non-negative")
                if np.max(self.missing_rounds) >= self.n_pairs:
                    raise ValueError("missing_rounds indices must be < n_pairs")


# =============================================================================
# Phase II → Phase III Contract: SiftedKeyData
# =============================================================================


@dataclass
class SiftedKeyData:
    """
    Output contract for Phase II: Sifting & Estimation.

    This dataclass represents the sifted key material ready for reconciliation,
    including QBER estimates with finite-size statistical penalties.

    Attributes
    ----------
    key_alice : np.ndarray
        Alice's sifted key bits (matching basis positions), dtype uint8.
    key_bob : np.ndarray
        Bob's sifted key bits (matching basis positions, may have errors), dtype uint8.
    sifted_length : int
        Length of sifted key (|I_0| + |I_1| = total matching positions).
    matching_basis_indices : np.ndarray
        Original indices in QuantumPhaseOutput where bases matched, dtype int64.
    i_0_indices : np.ndarray
        Indices in sifted key corresponding to I_0 (Alice's chosen basis), dtype int64.
    i_1_indices : np.ndarray
        Indices in sifted key corresponding to I_1 (Alice's unchosen basis), dtype int64.
    test_indices : np.ndarray
        Indices used for QBER estimation (subset of I_0), dtype int64.
    key_indices : np.ndarray
        Indices used for key generation (I_0 \\ test_indices), dtype int64.
    observed_qber : float
        Raw QBER measured on test set.
    adjusted_qber : float
        QBER with finite-size penalty μ applied: e_adj = e_obs + μ.
    statistical_penalty : float
        Finite-size penalty μ from Erven et al. Eq. (2).
    test_set_size : int
        Number of bits used for testing (|T|).
    detection_validation_passed : bool
        Whether Chernoff bound validation passed.
    detected_rounds : int
        Number of rounds where Bob reported detection.
    timing_markers : List[TimingMarker]
        Timing markers from Phase II for audit.

    Invariants
    ----------
    - len(key_alice) == len(key_bob) == sifted_length
    - adjusted_qber = observed_qber + statistical_penalty
    - |i_0_indices| + |i_1_indices| == sifted_length
    - |test_indices| + |key_indices| == |i_0_indices|

    References
    ----------
    - sprint_0_specification.md (INFRA-002)
    - phase_II_analysis.md Section 2.3 (finite-size penalty)
    - Erven et al. (2014) Theorem 2 (μ formula)
    """

    key_alice: np.ndarray
    key_bob: np.ndarray
    sifted_length: int
    matching_basis_indices: np.ndarray
    i_0_indices: np.ndarray
    i_1_indices: np.ndarray
    test_indices: np.ndarray
    key_indices: np.ndarray
    observed_qber: float
    adjusted_qber: float
    statistical_penalty: float
    test_set_size: int
    detection_validation_passed: bool
    detected_rounds: int
    timing_markers: List[TimingMarker] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate sifted key data invariants."""
        # Check key lengths
        if len(self.key_alice) != self.sifted_length:
            raise ValueError(
                f"key_alice length {len(self.key_alice)} != sifted_length {self.sifted_length}"
            )
        if len(self.key_bob) != self.sifted_length:
            raise ValueError(
                f"key_bob length {len(self.key_bob)} != sifted_length {self.sifted_length}"
            )

        # Check dtypes
        if self.key_alice.dtype != np.uint8:
            raise ValueError(f"key_alice must have dtype uint8, got {self.key_alice.dtype}")
        if self.key_bob.dtype != np.uint8:
            raise ValueError(f"key_bob must have dtype uint8, got {self.key_bob.dtype}")

        # Check key values
        if not np.all((self.key_alice == 0) | (self.key_alice == 1)):
            raise ValueError("key_alice values must be 0 or 1")
        if not np.all((self.key_bob == 0) | (self.key_bob == 1)):
            raise ValueError("key_bob values must be 0 or 1")

        # Check index set sizes
        total_indices = len(self.i_0_indices) + len(self.i_1_indices)
        if total_indices != self.sifted_length:
            raise ValueError(
                f"|i_0_indices| + |i_1_indices| = {total_indices} != sifted_length {self.sifted_length}"
            )

        # Check test/key partition of I_0
        if len(self.test_indices) + len(self.key_indices) != len(self.i_0_indices):
            raise ValueError(
                "|test_indices| + |key_indices| must equal |i_0_indices|"
            )

        # Check test_set_size matches
        if self.test_set_size != len(self.test_indices):
            raise ValueError(
                f"test_set_size {self.test_set_size} != len(test_indices) {len(self.test_indices)}"
            )

        # Check QBER values
        if not 0.0 <= self.observed_qber <= 1.0:
            raise ValueError("observed_qber must be in [0, 1]")
        if not 0.0 <= self.adjusted_qber <= 1.0:
            raise ValueError("adjusted_qber must be in [0, 1]")
        if self.statistical_penalty < 0.0:
            raise ValueError("statistical_penalty must be non-negative")

        # Verify penalty relationship (with floating point tolerance)
        expected_adjusted = self.observed_qber + self.statistical_penalty
        if abs(self.adjusted_qber - expected_adjusted) > 1e-9:
            raise ValueError(
                f"adjusted_qber {self.adjusted_qber} != observed_qber + penalty "
                f"({self.observed_qber} + {self.statistical_penalty} = {expected_adjusted})"
            )

        # Check detection counts
        if self.detected_rounds < 0:
            raise ValueError("detected_rounds must be non-negative")


# =============================================================================
# Phase III → Phase IV Contract: ReconciledKeyData
# =============================================================================


@dataclass
class ReconciledKeyData:
    """
    Output contract for Phase III: Information Reconciliation.

    This dataclass represents the error-corrected key material with full
    leakage accounting for privacy amplification.

    Attributes
    ----------
    reconciled_key : np.ndarray
        Error-corrected key bits, dtype uint8.
    reconciled_length : int
        Length of reconciled key (may be shorter than sifted due to block failures).
    total_syndrome_bits : int
        Total syndrome bits transmitted (wiretap cost |Σ|).
    total_hash_bits : int
        Total verification hash bits transmitted.
    total_leakage : int
        Total information leakage: syndrome_bits + hash_bits.
    blocks_processed : int
        Number of LDPC blocks processed.
    blocks_verified : int
        Number of blocks that passed hash verification.
    blocks_discarded : int
        Number of blocks discarded due to decoder/verification failure.
    integrated_qber : float
        QBER estimate from reconciliation error counts.
    block_results : List["LDPCBlockResult"]
        Per-block reconciliation results for detailed analysis.
    safety_cap_bits : int
        Maximum allowed leakage before abort (L_max).
    safety_cap_utilization : float
        Fraction of safety cap used: total_leakage / safety_cap_bits.

    Invariants
    ----------
    - total_leakage = total_syndrome_bits + total_hash_bits
    - blocks_verified + blocks_discarded <= blocks_processed
    - safety_cap_utilization = total_leakage / safety_cap_bits

    References
    ----------
    - sprint_0_specification.md (INFRA-002)
    - phase_III_analysis.md Section 1.3 (wiretap cost)
    - Lupo et al. (2023) Eq. (3) (leakage subtraction)
    """

    reconciled_key: np.ndarray
    reconciled_length: int
    total_syndrome_bits: int
    total_hash_bits: int
    total_leakage: int
    blocks_processed: int
    blocks_verified: int
    blocks_discarded: int
    integrated_qber: float
    block_results: List["LDPCBlockResult"] = field(default_factory=list)
    safety_cap_bits: int = 0
    safety_cap_utilization: float = 0.0

    def __post_init__(self) -> None:
        """Validate reconciled key data invariants."""
        # Check key length
        if len(self.reconciled_key) != self.reconciled_length:
            raise ValueError(
                f"reconciled_key length {len(self.reconciled_key)} != "
                f"reconciled_length {self.reconciled_length}"
            )

        # Check dtype
        if self.reconciled_key.dtype != np.uint8:
            raise ValueError(
                f"reconciled_key must have dtype uint8, got {self.reconciled_key.dtype}"
            )

        # Check key values
        if len(self.reconciled_key) > 0:
            if not np.all((self.reconciled_key == 0) | (self.reconciled_key == 1)):
                raise ValueError("reconciled_key values must be 0 or 1")

        # Check leakage accounting
        expected_leakage = self.total_syndrome_bits + self.total_hash_bits
        if self.total_leakage != expected_leakage:
            raise ValueError(
                f"total_leakage {self.total_leakage} != syndrome + hash "
                f"({self.total_syndrome_bits} + {self.total_hash_bits} = {expected_leakage})"
            )

        # Check block counts
        if self.blocks_processed < 0:
            raise ValueError("blocks_processed must be non-negative")
        if self.blocks_verified < 0:
            raise ValueError("blocks_verified must be non-negative")
        if self.blocks_discarded < 0:
            raise ValueError("blocks_discarded must be non-negative")
        if self.blocks_verified + self.blocks_discarded > self.blocks_processed:
            raise ValueError("verified + discarded cannot exceed processed blocks")

        # Check QBER
        if not 0.0 <= self.integrated_qber <= 1.0:
            raise ValueError("integrated_qber must be in [0, 1]")

        # Check safety cap utilization
        if self.safety_cap_bits > 0:
            expected_util = self.total_leakage / self.safety_cap_bits
            if abs(self.safety_cap_utilization - expected_util) > 1e-9:
                raise ValueError(
                    f"safety_cap_utilization {self.safety_cap_utilization} != "
                    f"total_leakage / safety_cap_bits ({expected_util})"
                )


# =============================================================================
# Phase IV Output Contract: ObliviousTransferOutput
# =============================================================================


@dataclass
class ObliviousTransferOutput:
    """
    Final output contract for Phase IV: Privacy Amplification.

    This dataclass represents the complete 1-out-of-2 Oblivious Transfer output
    where Alice possesses two keys (S_0, S_1) and Bob possesses exactly one (S_C).

    Attributes
    ----------
    alice_key_0 : np.ndarray
        Alice's first key S_0, dtype uint8.
    alice_key_1 : np.ndarray
        Alice's second key S_1, dtype uint8.
    bob_key : np.ndarray
        Bob's chosen key S_C, dtype uint8.
    bob_choice_bit : int
        Bob's choice bit C (0 or 1), determining which key he received.
    final_length : int
        Length of final keys in bits.
    extractable_entropy : float
        Min-entropy available before extraction (bits).
    entropy_consumed : float
        Min-entropy consumed by privacy amplification.
    security_parameter : float
        Achieved security parameter ε_sec.
    correctness_parameter : float
        Achieved correctness parameter ε_cor.
    hash_seed : bytes
        Seed used for Toeplitz hashing (shared between parties).
    storage_noise_parameter : float
        Assumed adversary storage noise r used in entropy calculation.
    entropy_bound_used : str
        Which entropy bound was used: "dupuis_konig" or "lupo" (max bound selection).

    Invariants
    ----------
    - len(alice_key_0) == len(alice_key_1) == len(bob_key) == final_length
    - bob_key == alice_key_0 if bob_choice_bit == 0, else alice_key_1
    - final_length <= extractable_entropy - security_margins

    References
    ----------
    - sprint_0_specification.md (INFRA-002)
    - phase_IV_analysis.md Section 1.3 (Max Bound)
    - Lupo et al. (2023) Eq. (36) (min-entropy bound)
    """

    alice_key_0: np.ndarray
    alice_key_1: np.ndarray
    bob_key: np.ndarray
    bob_choice_bit: int
    final_length: int
    extractable_entropy: float
    entropy_consumed: float
    security_parameter: float
    correctness_parameter: float
    hash_seed: bytes
    storage_noise_parameter: float
    entropy_bound_used: str

    def __post_init__(self) -> None:
        """Validate oblivious transfer output invariants."""
        # Check key lengths
        if len(self.alice_key_0) != self.final_length:
            raise ValueError(
                f"alice_key_0 length {len(self.alice_key_0)} != final_length {self.final_length}"
            )
        if len(self.alice_key_1) != self.final_length:
            raise ValueError(
                f"alice_key_1 length {len(self.alice_key_1)} != final_length {self.final_length}"
            )
        if len(self.bob_key) != self.final_length:
            raise ValueError(
                f"bob_key length {len(self.bob_key)} != final_length {self.final_length}"
            )

        # Check dtypes
        for name, arr in [
            ("alice_key_0", self.alice_key_0),
            ("alice_key_1", self.alice_key_1),
            ("bob_key", self.bob_key),
        ]:
            if arr.dtype != np.uint8:
                raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")
            if len(arr) > 0 and not np.all((arr == 0) | (arr == 1)):
                raise ValueError(f"{name} values must be 0 or 1")

        # Check choice bit
        if self.bob_choice_bit not in (0, 1):
            raise ValueError("bob_choice_bit must be 0 or 1")

        # Check OT correctness: Bob's key should match Alice's chosen key
        if self.final_length > 0:
            expected_bob_key = (
                self.alice_key_0 if self.bob_choice_bit == 0 else self.alice_key_1
            )
            if not np.array_equal(self.bob_key, expected_bob_key):
                raise ValueError(
                    f"bob_key does not match alice_key_{self.bob_choice_bit} (OT correctness violation)"
                )

        # Check entropy values
        if self.extractable_entropy < 0:
            raise ValueError("extractable_entropy must be non-negative")
        if self.entropy_consumed < 0:
            raise ValueError("entropy_consumed must be non-negative")

        # Check security parameters
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if self.correctness_parameter <= 0:
            raise ValueError("correctness_parameter must be positive")

        # Check hash seed
        if not isinstance(self.hash_seed, bytes) or len(self.hash_seed) == 0:
            raise ValueError("hash_seed must be non-empty bytes")

        # Check storage noise
        if not 0.0 <= self.storage_noise_parameter <= 1.0:
            raise ValueError("storage_noise_parameter must be in [0, 1]")

        # Check entropy bound
        valid_bounds = {"dupuis_konig", "lupo", "max_bound"}
        if self.entropy_bound_used not in valid_bounds:
            raise ValueError(f"entropy_bound_used must be one of {valid_bounds}")


# =============================================================================
# Protocol Transcript for Audit
# =============================================================================


@dataclass
class ProtocolTranscript:
    """
    Complete protocol execution transcript for audit and debugging.

    Captures all phase transitions, timing markers, and abort conditions
    for post-execution analysis.

    Attributes
    ----------
    session_id : str
        Unique identifier for this protocol execution.
    start_timestamp_ns : int
        Simulation timestamp when protocol started.
    end_timestamp_ns : int
        Simulation timestamp when protocol ended.
    final_phase : ProtocolPhase
        Final protocol state (COMPLETED or ABORTED).
    abort_reason : Optional[AbortReason]
        Reason for abort if final_phase is ABORTED.
    warnings : List[WarningCode]
        List of warnings encountered during execution.
    timing_markers : List[TimingMarker]
        Complete ordered list of timing events.
    phase_durations_ns : Dict[ProtocolPhase, int]
        Duration of each phase in nanoseconds.

    References
    ----------
    - sprint_0_specification.md (INFRA-002)
    - phase_I_analysis.md Section 6.2.1 (TASK-TRANSCRIPT-001)
    """

    session_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    final_phase: ProtocolPhase
    abort_reason: Optional[AbortReason] = None
    warnings: List[WarningCode] = field(default_factory=list)
    timing_markers: List[TimingMarker] = field(default_factory=list)
    phase_durations_ns: Dict[ProtocolPhase, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate protocol transcript invariants."""
        if not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("end_timestamp_ns must be >= start_timestamp_ns")
        if self.final_phase == ProtocolPhase.ABORTED and self.abort_reason is None:
            raise ValueError("abort_reason must be set when final_phase is ABORTED")


# =============================================================================
# Existing Data Structures (preserved for backward compatibility)
# =============================================================================


@dataclass
class ObliviousKey:
    """
    Represents the output of the E-HOK protocol.
    
    An oblivious key is a shared secret where one party (Bob) has partial
    knowledge and the other party (Alice) has complete knowledge. This is the
    fundamental output of the E-HOK protocol used for secure multiparty
    computation.
    
    Attributes
    ----------
    key_value : np.ndarray
        Final key bits as uint8 array (values 0 or 1).
    knowledge_mask : np.ndarray
        Mask indicating knowledge: 0 = known, 1 = unknown (oblivious).
        For Alice: all zeros. For Bob: 1s at positions corresponding to I_1.
    security_param : float
        Estimated epsilon security parameter (εsec).
    qber : float
        Measured quantum bit error rate on the test set.
    final_length : int
        Length of the final key after privacy amplification.
    
    Notes
    -----
    The knowledge mask enables the oblivious key property: Alice knows all bits,
    while Bob knows only a subset. This asymmetric knowledge is crucial for
    certain secure computation protocols. Baseline simulations may set
    ``ProtocolConfig.approximate_knowledge_mask`` to ``True``, in which case the
    mask encodes an approximation rather than an exact mapping through the
    privacy amplification matrix.
    """
    key_value: np.ndarray
    knowledge_mask: np.ndarray
    security_param: float
    qber: float
    final_length: int
    
    def __post_init__(self) -> None:
        """Validate data structure consistency."""
        if self.key_value.shape != self.knowledge_mask.shape:
            raise ValueError("Key and mask must have same shape")
        if self.key_value.dtype != np.uint8:
            raise ValueError("Key must be uint8")
        if self.knowledge_mask.dtype != np.uint8:
            raise ValueError("Mask must be uint8")
        if not np.all((self.key_value == 0) | (self.key_value == 1)):
            raise ValueError("Key values must be 0 or 1")
        if not np.all((self.knowledge_mask == 0) | (self.knowledge_mask == 1)):
            raise ValueError("Mask values must be 0 or 1")
        if len(self.key_value) != self.final_length:
            raise ValueError("final_length must equal key length")
        if len(self.knowledge_mask) != self.final_length:
            raise ValueError("knowledge_mask length must equal final_length")
        if not 0.0 <= float(self.qber) <= 1.0:
            raise ValueError("qber must be in [0, 1]")
        if float(self.security_param) <= 0:
            raise ValueError("security_param must be positive")


@dataclass
class MeasurementRecord:
    """
    Record of a single EPR measurement.
    
    This structure captures all relevant information about a quantum measurement
    performed during the E-HOK protocol's quantum phase.
    
    Attributes
    ----------
    outcome : int
        Measurement outcome (0 or 1).
    basis : int
        Measurement basis (0 = Z, 1 = X).
    timestamp : float
        Simulation time when measurement occurred (ns).
    
    Notes
    -----
    Basis encoding follows standard quantum information conventions:
    - 0 = Z-basis (computational basis: |0⟩, |1⟩)
    - 1 = X-basis (Hadamard basis: |+⟩, |-⟩)
    """
    outcome: int
    basis: int
    timestamp: float
    
    def __post_init__(self) -> None:
        """Validate measurement record."""
        if self.outcome not in (0, 1):
            raise ValueError("Outcome must be 0 or 1")
        if self.basis not in (0, 1):
            raise ValueError("Basis must be 0 (Z) or 1 (X)")


@dataclass
class ProtocolResult:
    """
    Complete protocol execution result with statistics.
    
    This structure encapsulates all outcomes and metrics from a complete E-HOK
    protocol execution, enabling detailed analysis and debugging.
    
    Attributes
    ----------
    oblivious_key : Optional[ObliviousKey]
        The final oblivious key (None if protocol aborted).
    success : bool
        Whether protocol completed successfully.
    abort_reason : Optional[str]
        Reason for abort (if success=False).
    raw_count : int
        Number of raw EPR pairs generated.
    sifted_count : int
        Number of sifted bits (matching bases, |I_0|).
    test_count : int
        Number of bits used for error estimation (|T|).
    final_count : int
        Number of bits after privacy amplification.
    qber : float
        Quantum bit error rate measured on test set.
    execution_time_ms : float
        Total protocol execution time (simulation time).
    
    Notes
    -----
    The counts provide insight into protocol efficiency:
    - raw_count: Total quantum resources consumed
    - sifted_count: Effective bits after basis reconciliation
    - test_count: Bits sacrificed for security verification
    - final_count: Secure key length after all post-processing
    """
    oblivious_key: Optional[ObliviousKey]
    success: bool
    abort_reason: Optional[str]
    raw_count: int
    sifted_count: int
    test_count: int
    final_count: int
    qber: float
    execution_time_ms: float

    def __post_init__(self) -> None:
        """Validate protocol result invariants."""
        if self.raw_count < 0:
            raise ValueError("raw_count must be non-negative")
        if self.sifted_count < 0 or self.sifted_count > self.raw_count:
            raise ValueError("sifted_count must satisfy 0 <= sifted_count <= raw_count")
        if self.test_count < 0:
            raise ValueError("test_count must be non-negative")
        if self.final_count < 0:
            raise ValueError("final_count must be non-negative")
        if self.sifted_count < self.test_count + self.final_count:
            raise ValueError("sifted_count must be >= test_count + final_count")
        if not 0.0 <= float(self.qber) <= 1.0:
            raise ValueError("qber must be in [0, 1]")
        if self.success and self.oblivious_key is None:
            raise ValueError("oblivious_key must be set when success is True")
        if self.oblivious_key is not None and self.final_count != self.oblivious_key.final_length:
            raise ValueError("final_count must equal oblivious_key.final_length when key is present")


@dataclass
class ExecutionMetrics(ProtocolResult):
    """Extended protocol metrics derived from :class:`ProtocolResult`.

    This dataclass enriches the base result with commonly used derived
    quantities so later analysis layers do not need to recompute them.
    """

    raw_to_sifted_ratio: Optional[float] = None
    sifted_to_final_ratio: Optional[float] = None
    leakage_bits: Optional[float] = None

    def __post_init__(self) -> None:
        # Validate base fields first
        super().__post_init__()

        if self.raw_count > 0:
            self.raw_to_sifted_ratio = self.sifted_count / self.raw_count
        if self.sifted_count > 0:
            self.sifted_to_final_ratio = (
                self.final_count / self.sifted_count
            )

        # leakage_bits is left to upstream calculators; ensure non-negative if set
        if self.leakage_bits is not None and self.leakage_bits < 0:
            raise ValueError("leakage_bits must be non-negative when provided")


@dataclass
class LDPCBlockResult:
    """
    Result of processing a single LDPC block during reconciliation.

    Attributes
    ----------
    verified : bool
        True if hash verification succeeded for the block.
    error_count : int
        Number of errors corrected (Hamming weight of error vector).
    block_length : int
        Length of the payload portion (excludes padding) in bits.
    syndrome_length : int
        Number of syndrome bits transmitted for this block.
    hash_bits : int
        Number of verification hash bits transmitted (default 50).
    """

    verified: bool
    error_count: int
    block_length: int
    syndrome_length: int
    hash_bits: int = 50

    def __post_init__(self) -> None:
        if self.block_length < 0:
            raise ValueError("block_length must be non-negative")
        if self.error_count < 0:
            raise ValueError("error_count must be non-negative")
        if self.error_count > self.block_length:
            raise ValueError("error_count cannot exceed block_length")
        if self.syndrome_length < 0:
            raise ValueError("syndrome_length must be non-negative")
        if self.hash_bits <= 0:
            raise ValueError("hash_bits must be positive")
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be a boolean")


@dataclass
class LDPCMatrixPool:
    """
    Pool of pre-generated LDPC matrices at different code rates.

    Attributes
    ----------
    frame_size : int
        Fixed frame size ``n`` for all matrices.
    matrices : Dict[float, sp.spmatrix]
        Mapping from code rate to parity-check matrix in CSR format.
    rates : np.ndarray
        Sorted array of available code rates for selection.
    checksum : str
        SHA-256 checksum over the matrix pool used for synchronization.
    """

    frame_size: int
    matrices: Dict[float, sp.spmatrix]
    rates: np.ndarray
    checksum: str

    def __post_init__(self) -> None:
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if self.rates.size == 0:
            raise ValueError("rates must not be empty")
        if not np.all(np.diff(self.rates) > 0):
            raise ValueError("rates must be strictly increasing")
        for rate, matrix in self.matrices.items():
            if matrix.shape[1] != self.frame_size:
                raise ValueError(
                    f"Matrix for rate {rate} has incompatible frame size {matrix.shape[1]}"
                )
        if not isinstance(self.checksum, str) or not self.checksum:
            raise ValueError("checksum must be a non-empty string")


@dataclass
class LDPCReconciliationResult:
    """
    Aggregate result of LDPC reconciliation across all processed blocks.

    Attributes
    ----------
    corrected_key : np.ndarray
        Concatenated verified payload bits after reconciliation.
    qber_estimate : float
        Integrated QBER estimate computed from block outcomes.
    total_leakage : int
        Total information leakage (syndrome plus hash bits).
    blocks_processed : int
        Number of LDPC blocks processed.
    blocks_verified : int
        Number of blocks that passed verification.
    blocks_discarded : int
        Number of blocks discarded due to decoder failure or hash mismatch.
    """

    corrected_key: np.ndarray
    qber_estimate: float
    total_leakage: int
    blocks_processed: int
    blocks_verified: int
    blocks_discarded: int

    def __post_init__(self) -> None:
        if self.total_leakage < 0:
            raise ValueError("total_leakage must be non-negative")
        if self.blocks_processed < 0:
            raise ValueError("blocks_processed must be non-negative")
        if self.blocks_verified < 0:
            raise ValueError("blocks_verified must be non-negative")
        if self.blocks_discarded < 0:
            raise ValueError("blocks_discarded must be non-negative")
        if self.blocks_verified + self.blocks_discarded > self.blocks_processed:
            raise ValueError("verified + discarded cannot exceed processed blocks")
        if not 0.0 <= float(self.qber_estimate) <= 1.0:
            raise ValueError("qber_estimate must be in [0, 1]")
        if self.corrected_key.dtype != np.uint8:
            raise ValueError("corrected_key must be uint8")

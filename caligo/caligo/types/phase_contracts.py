"""
Phase boundary contracts for E-HOK protocol.

This module defines the canonical data contracts at each protocol phase
boundary, enabling type-safe data flow and runtime validation.

Contract Map:
    Phase I (Quantum) ──► QuantumPhaseResult ──► Phase II (Sifting)
    Phase II (Sifting) ──► SiftingPhaseResult ──► Phase III (Reconciliation)
    Phase III (Reconciliation) ──► ReconciliationPhaseResult ──► Phase IV
    Phase IV (Amplification) ──► ObliviousTransferOutput ──► Application

References
----------
- Erven et al. (2014): Experimental validation
- Schaffner et al. (2009): Protocol definition
- Lupo et al. (2023): Key length formula
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
from bitarray import bitarray

from caligo.types.exceptions import ContractViolation
from caligo.types.keys import AliceObliviousKey, BobObliviousKey
from caligo.types.measurements import DetectionEvent


# =============================================================================
# Security Constants
# =============================================================================

# QBER hard limit: security impossible beyond 22% (König et al. 2012)
QBER_HARD_LIMIT: float = 0.22

# QBER conservative limit: recommended operating point (Schaffner et al. 2009)
QBER_CONSERVATIVE_LIMIT: float = 0.11

# Default security parameter (Erven et al. 2014, Table I)
DEFAULT_EPSILON_SEC: float = 1e-10


# =============================================================================
# Phase I Contract: Quantum Phase Result
# =============================================================================


@dataclass
class QuantumPhaseResult:
    """
    Contract: Phase I → Phase II data transfer.

    Contains all quantum measurement data and metadata required for
    the sifting and estimation phase. Enforces NSM timing invariants.

    Parameters
    ----------
    measurement_outcomes : np.ndarray
        Array of measurement outcomes (0/1), shape (n_pairs,), dtype uint8.
    basis_choices : np.ndarray
        Array of basis choices (0=Z, 1=X), shape (n_pairs,), dtype uint8.
    round_ids : np.ndarray
        Array of round identifiers, shape (n_pairs,), dtype int64.
    generation_timestamp : float
        Simulation time (ns) when quantum phase completed.
    num_pairs_requested : int
        Number of EPR pairs requested.
    num_pairs_generated : int
        Number of EPR pairs actually generated (may differ due to losses).
    detection_events : List[DetectionEvent]
        Detection event records for validation.
    timing_barrier_marked : bool
        True if TimingBarrier.mark_quantum_complete() was called.

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-Q-001: len(measurement_outcomes) == num_pairs_generated
    - POST-Q-002: len(basis_choices) == num_pairs_generated
    - POST-Q-003: All outcomes ∈ {0, 1}
    - POST-Q-004: All bases ∈ {0, 1}
    - POST-Q-005: timing_barrier_marked == True (for NSM security)

    References
    ----------
    - phase_I_analysis.md Section 7.2
    """

    measurement_outcomes: np.ndarray
    basis_choices: np.ndarray
    round_ids: np.ndarray
    generation_timestamp: float
    num_pairs_requested: int
    num_pairs_generated: int
    detection_events: List[DetectionEvent] = field(default_factory=list)
    timing_barrier_marked: bool = True

    def __post_init__(self) -> None:
        """Validate post-conditions after initialization."""
        # POST-Q-001
        if len(self.measurement_outcomes) != self.num_pairs_generated:
            raise ContractViolation(
                f"POST-Q-001: len(measurement_outcomes)={len(self.measurement_outcomes)} "
                f"!= num_pairs_generated={self.num_pairs_generated}"
            )
        # POST-Q-002
        if len(self.basis_choices) != self.num_pairs_generated:
            raise ContractViolation(
                f"POST-Q-002: len(basis_choices)={len(self.basis_choices)} "
                f"!= num_pairs_generated={self.num_pairs_generated}"
            )
        # POST-Q-003
        if self.num_pairs_generated > 0:
            unique_outcomes = set(np.unique(self.measurement_outcomes))
            if not unique_outcomes.issubset({0, 1}):
                raise ContractViolation(
                    f"POST-Q-003: measurement_outcomes contains invalid values: "
                    f"{unique_outcomes - {0, 1}}"
                )
        # POST-Q-004
        if self.num_pairs_generated > 0:
            unique_bases = set(np.unique(self.basis_choices))
            if not unique_bases.issubset({0, 1}):
                raise ContractViolation(
                    f"POST-Q-004: basis_choices contains invalid values: "
                    f"{unique_bases - {0, 1}}"
                )
        # POST-Q-005 is a soft check (warning if False in production)


# =============================================================================
# Phase II Contract: Sifting Phase Result
# =============================================================================


@dataclass
class SiftingPhaseResult:
    """
    Contract: Phase II → Phase III data transfer.

    Contains sifted key material with QBER estimates and statistical
    penalties accounting for finite-size effects.

    Parameters
    ----------
    sifted_key_alice : bitarray
        Alice's sifted key bits (matching basis positions).
    sifted_key_bob : bitarray
        Bob's sifted key bits (may contain errors).
    matching_indices : np.ndarray
        Original round indices where bases matched, dtype int64.
    i0_indices : np.ndarray
        Indices for I₀ partition (Alice's random subset).
    i1_indices : np.ndarray
        Indices for I₁ partition (complement of I₀).
    test_set_indices : np.ndarray
        Indices sacrificed for QBER estimation.
    qber_estimate : float
        Observed QBER on test set: e_obs.
    qber_adjusted : float
        QBER with finite-size penalty: e_adj = e_obs + μ.
    finite_size_penalty : float
        Statistical penalty μ from Erven et al. Eq. (2).
    test_set_size : int
        Number of bits used for testing |T|.
    timing_compliant : bool
        True if Δt was properly enforced before basis revelation.

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-S-001: len(sifted_key_alice) == len(sifted_key_bob)
    - POST-S-002: qber_adjusted ≈ qber_estimate + finite_size_penalty
    - POST-S-003: qber_adjusted ≤ QBER_HARD_LIMIT (else would abort)

    References
    ----------
    - phase_II_analysis.md Section 1.1
    - Erven et al. (2014) Theorem 2, Eq. (2) for μ calculation
    """

    sifted_key_alice: bitarray
    sifted_key_bob: bitarray
    matching_indices: np.ndarray
    i0_indices: np.ndarray
    i1_indices: np.ndarray
    test_set_indices: np.ndarray
    qber_estimate: float
    qber_adjusted: float
    finite_size_penalty: float
    test_set_size: int
    timing_compliant: bool = True

    def __post_init__(self) -> None:
        """Validate post-conditions after initialization."""
        # POST-S-001
        if len(self.sifted_key_alice) != len(self.sifted_key_bob):
            raise ContractViolation(
                f"POST-S-001: len(sifted_key_alice)={len(self.sifted_key_alice)} "
                f"!= len(sifted_key_bob)={len(self.sifted_key_bob)}"
            )
        # POST-S-002 (tolerance for floating point)
        expected_adjusted = self.qber_estimate + self.finite_size_penalty
        if abs(self.qber_adjusted - expected_adjusted) > 1e-10:
            raise ContractViolation(
                f"POST-S-002: qber_adjusted={self.qber_adjusted} != "
                f"qber_estimate + finite_size_penalty={expected_adjusted}"
            )
        # POST-S-003
        if self.qber_adjusted > QBER_HARD_LIMIT:
            raise ContractViolation(
                f"POST-S-003: qber_adjusted={self.qber_adjusted} > "
                f"QBER_HARD_LIMIT={QBER_HARD_LIMIT}"
            )


# =============================================================================
# Key Material + Optional Channel Estimate
# =============================================================================


@dataclass
class SiftedKeyMaterial:
    """
    Contract: Phase II → Phase III key material (Option A).

    This contract intentionally excludes any channel estimate so that
    blind reconciliation can run without test-bit sacrifice.

    Parameters
    ----------
    sifted_key_alice : bitarray
        Alice's sifted key bits after removing any test subset.
    sifted_key_bob : bitarray
        Bob's sifted key bits after removing any test subset.
    matching_indices : np.ndarray
        Original round indices used to form the sifted key (post-test removal).
    i0_indices : np.ndarray
        Original indices for the I₀ partition (post-test removal).
    i1_indices : np.ndarray
        Original indices for the I₁ partition (post-test removal).
    test_set_indices : np.ndarray
        Original round indices sacrificed for testing (may be empty).
    timing_compliant : bool
        True if Δt was enforced before basis revelation.

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-SKM-001: len(sifted_key_alice) == len(sifted_key_bob)
    """

    sifted_key_alice: bitarray
    sifted_key_bob: bitarray
    matching_indices: np.ndarray
    i0_indices: np.ndarray
    i1_indices: np.ndarray
    test_set_indices: np.ndarray
    timing_compliant: bool = True

    def __post_init__(self) -> None:
        if len(self.sifted_key_alice) != len(self.sifted_key_bob):
            raise ContractViolation(
                f"POST-SKM-001: len(sifted_key_alice)={len(self.sifted_key_alice)} "
                f"!= len(sifted_key_bob)={len(self.sifted_key_bob)}"
            )


@dataclass
class ChannelEstimate:
    """
    Optional channel estimate produced during Phase II.

    Parameters
    ----------
    qber_estimate : float
        Observed QBER on test set.
    qber_adjusted : float
        QBER with finite-size penalty.
    finite_size_penalty : float
        Statistical penalty μ.
    test_set_size : int
        Number of tested bits.

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-CE-001: qber_adjusted ≈ qber_estimate + finite_size_penalty
    - POST-CE-002: qber_adjusted ≤ QBER_HARD_LIMIT
    """

    qber_estimate: float
    qber_adjusted: float
    finite_size_penalty: float
    test_set_size: int

    def __post_init__(self) -> None:
        expected_adjusted = self.qber_estimate + self.finite_size_penalty
        if abs(self.qber_adjusted - expected_adjusted) > 1e-10:
            raise ContractViolation(
                f"POST-CE-001: qber_adjusted={self.qber_adjusted} != "
                f"qber_estimate + finite_size_penalty={expected_adjusted}"
            )
        if self.qber_adjusted > QBER_HARD_LIMIT:
            raise ContractViolation(
                f"POST-CE-002: qber_adjusted={self.qber_adjusted} > "
                f"QBER_HARD_LIMIT={QBER_HARD_LIMIT}"
            )


@dataclass
class PhaseIIResult:
    """
    Contract: Phase II → Phase III (Option A) bundle.

    Parameters
    ----------
    key_material : SiftedKeyMaterial
        Sifted key bits and index metadata.
    channel_estimate : ChannelEstimate, optional
        Present for baseline reconciliation; omitted for blind.
    """

    key_material: SiftedKeyMaterial
    channel_estimate: Optional[ChannelEstimate] = None

    @property
    def has_channel_estimate(self) -> bool:
        """Whether a QBER/channel estimate is present."""

        return self.channel_estimate is not None

    @classmethod
    def from_sifting_phase_result(cls, dto: SiftingPhaseResult) -> "PhaseIIResult":
        """Convert legacy SiftingPhaseResult into Option-A bundle."""

        key_material = SiftedKeyMaterial(
            sifted_key_alice=dto.sifted_key_alice,
            sifted_key_bob=dto.sifted_key_bob,
            matching_indices=dto.matching_indices,
            i0_indices=dto.i0_indices,
            i1_indices=dto.i1_indices,
            test_set_indices=dto.test_set_indices,
            timing_compliant=dto.timing_compliant,
        )
        channel_estimate = ChannelEstimate(
            qber_estimate=float(dto.qber_estimate),
            qber_adjusted=float(dto.qber_adjusted),
            finite_size_penalty=float(dto.finite_size_penalty),
            test_set_size=int(dto.test_set_size),
        )
        return cls(key_material=key_material, channel_estimate=channel_estimate)


# =============================================================================
# Phase III Contract: Reconciliation Phase Result
# =============================================================================


@dataclass
class ReconciliationPhaseResult:
    """
    Contract: Phase III → Phase IV data transfer.

    Contains error-corrected key material with leakage accounting
    for privacy amplification entropy calculation.

    Parameters
    ----------
    reconciled_key : bitarray
        Error-corrected key (Alice's perspective).
    num_blocks : int
        Number of LDPC blocks processed.
    blocks_succeeded : int
        Number of blocks that passed verification.
    blocks_failed : int
        Number of blocks that failed (discarded).
    total_syndrome_bits : int
        Total syndrome leakage |Σ| in bits.
    effective_rate : float
        Achieved code rate R = (n - |Σ|) / n.
    hash_verified : bool
        True if final hash verification passed.
    leakage_within_cap : bool
        True if |Σ| ≤ L_max (safety cap).
    leakage_cap : Optional[int]
        Maximum allowed leakage L_max (None if uncapped).

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-R-001: total_syndrome_bits ≤ leakage_cap (if cap set)
    - POST-R-002: hash_verified == True (else would abort)

    References
    ----------
    - phase_III_analysis.md Section 1.1 (Wiretap Cost)
    - Schaffner et al. (2009): "length of syndrome must be subtracted"
    """

    reconciled_key: bitarray
    num_blocks: int
    blocks_succeeded: int
    blocks_failed: int
    total_syndrome_bits: int
    effective_rate: float
    hash_verified: bool = True
    leakage_within_cap: bool = True
    leakage_cap: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate post-conditions after initialization."""
        # POST-R-001
        if self.leakage_cap is not None and not self.leakage_within_cap:
            if self.total_syndrome_bits > self.leakage_cap:
                raise ContractViolation(
                    f"POST-R-001: total_syndrome_bits={self.total_syndrome_bits} > "
                    f"leakage_cap={self.leakage_cap}"
                )
        # POST-R-002
        if not self.hash_verified:
            raise ContractViolation(
                "POST-R-002: hash_verified=False indicates reconciliation failure"
            )


# =============================================================================
# Phase IV Contract: Amplification Phase Result
# =============================================================================


@dataclass
class AmplificationPhaseResult:
    """
    Contract: Phase IV → Final protocol output (role-specific).

    Contains the privacy-amplified output for a single party,
    plus diagnostic metrics. This is the role-specific view;
    the aggregate protocol output is ObliviousTransferOutput.

    Parameters
    ----------
    oblivious_key : Union[AliceObliviousKey, BobObliviousKey]
        Role-dependent output key(s).
    qber : float
        Final adjusted QBER used for security calculations.
    key_length : int
        Length of extracted key(s) in bits.
    entropy_consumed : float
        Total min-entropy consumed (h_min * n - leakage).
    entropy_rate : float
        Efficiency: key_length / raw_bits.
    metrics : dict
        Diagnostic data (timing, block stats, etc.).

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-AMP-001: key_length > 0 (else abort before reaching here)
    - POST-AMP-002: entropy_consumed >= key_length (entropy bound)

    References
    ----------
    - Lupo et al. (2023), Eq. (43): Key length formula
    - Erven et al. (2014): Experimental validation
    """

    oblivious_key: Union[AliceObliviousKey, BobObliviousKey]
    qber: float
    key_length: int
    entropy_consumed: float
    entropy_rate: float
    metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate post-conditions after initialization."""
        # POST-AMP-001
        if self.key_length <= 0:
            raise ContractViolation(
                f"POST-AMP-001: key_length={self.key_length} <= 0"
            )
        # POST-AMP-002 (with some margin for security parameter)
        if self.entropy_consumed < self.key_length:
            raise ContractViolation(
                f"POST-AMP-002: entropy_consumed={self.entropy_consumed} < "
                f"key_length={self.key_length}"
            )


# =============================================================================
# Final Output: Oblivious Transfer Output
# =============================================================================


@dataclass
class ObliviousTransferOutput:
    """
    Final protocol output: 1-out-of-2 OT keys.

    The terminal output of the E-HOK protocol, containing:
    - For Alice: Two keys (S₀, S₁)
    - For Bob: One key (Sᴄ) and his choice bit C

    Parameters
    ----------
    alice_key : AliceObliviousKey
        Alice's output containing S₀ and S₁.
    bob_key : BobObliviousKey
        Bob's output containing Sᴄ and C.
    protocol_succeeded : bool
        True if protocol completed without abort.
    total_rounds : int
        Total EPR pairs used in the protocol.
    final_key_length : int
        Length of extracted keys in bits.
    security_parameter : float
        ε_sec achieved (trace distance from ideal).
    entropy_rate : float
        Bits of key per input bit: ℓ / n.

    Raises
    ------
    ContractViolation
        If any post-condition is violated.

    Notes
    -----
    Post-conditions:
    - POST-OT-001: len(alice_key.s0) == len(alice_key.s1) == final_key_length
    - POST-OT-002: len(bob_key.sc) == final_key_length
    - POST-OT-003: bob_key.sc == alice_key.s_{choice_bit}

    References
    ----------
    - Schaffner et al. (2009) Definition 1
    - Erven et al. (2014) "Results" section
    """

    alice_key: AliceObliviousKey
    bob_key: BobObliviousKey
    protocol_succeeded: bool
    total_rounds: int
    final_key_length: int
    security_parameter: float = DEFAULT_EPSILON_SEC
    entropy_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate post-conditions after initialization."""
        # POST-OT-001
        if (
            len(self.alice_key.s0) != self.final_key_length
            or len(self.alice_key.s1) != self.final_key_length
        ):
            raise ContractViolation(
                f"POST-OT-001: Alice key lengths ({len(self.alice_key.s0)}, "
                f"{len(self.alice_key.s1)}) != final_key_length={self.final_key_length}"
            )
        # POST-OT-002
        if len(self.bob_key.sc) != self.final_key_length:
            raise ContractViolation(
                f"POST-OT-002: len(bob_key.sc)={len(self.bob_key.sc)} != "
                f"final_key_length={self.final_key_length}"
            )
        # POST-OT-003
        expected_key = (
            self.alice_key.s0 if self.bob_key.choice_bit == 0 else self.alice_key.s1
        )
        if self.bob_key.sc != expected_key:
            raise ContractViolation(
                f"POST-OT-003: bob_key.sc != alice_key.s{self.bob_key.choice_bit}"
            )

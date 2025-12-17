"""
Measurement records and detection events for quantum phase.

This module defines records for quantum measurement outcomes and detection
events used in Phase I (Quantum Generation) of the E-HOK protocol.

References
----------
- Erven et al. (2014): Detection validation via Chernoff bounds
"""

from __future__ import annotations

from dataclasses import dataclass

from caligo.types.exceptions import ContractViolation


@dataclass
class MeasurementRecord:
    """
    Record of a single quantum measurement event from EPR pair generation.

    Parameters
    ----------
    round_id : int
        Unique identifier for this round (must be ≥ 0).
    outcome : int
        Measurement result: 0 or 1.
    basis : int
        Basis used: 0 (Z/computational) or 1 (X/Hadamard).
    timestamp_ns : float
        Simulation time when measured in nanoseconds.
    detected : bool
        True if photon was detected (default: True).

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-MEAS-001: outcome ∈ {0, 1}
    - INV-MEAS-002: basis ∈ {0, 1}
    - INV-MEAS-003: round_id ≥ 0
    - INV-MEAS-004: timestamp_ns ≥ 0

    Domain Mapping (BB84):
    - basis=0 (Z): Computational basis {|0⟩, |1⟩}
    - basis=1 (X): Hadamard basis {|+⟩, |-⟩}
    """

    round_id: int
    outcome: int
    basis: int
    timestamp_ns: float
    detected: bool = True

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-MEAS-001
        if self.outcome not in {0, 1}:
            raise ContractViolation(
                f"INV-MEAS-001: outcome={self.outcome} not in {{0, 1}}"
            )
        # INV-MEAS-002
        if self.basis not in {0, 1}:
            raise ContractViolation(
                f"INV-MEAS-002: basis={self.basis} not in {{0, 1}}"
            )
        # INV-MEAS-003
        if self.round_id < 0:
            raise ContractViolation(f"INV-MEAS-003: round_id={self.round_id} < 0")
        # INV-MEAS-004
        if self.timestamp_ns < 0:
            raise ContractViolation(
                f"INV-MEAS-004: timestamp_ns={self.timestamp_ns} < 0"
            )


@dataclass
class RoundResult:
    """
    Combined result for a single EPR pair round (both Alice and Bob).

    Aggregates measurement outcomes from both parties for a single
    EPR pair, enabling sifting and QBER calculation.

    Parameters
    ----------
    round_id : int
        Unique round identifier.
    alice_outcome : int
        Alice's measurement outcome (0 or 1).
    bob_outcome : int
        Bob's measurement outcome (0 or 1).
    alice_basis : int
        Alice's basis choice (0=Z, 1=X).
    bob_basis : int
        Bob's basis choice (0=Z, 1=X).
    alice_detected : bool
        True if Alice detected a photon.
    bob_detected : bool
        True if Bob detected a photon.

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    Notes
    -----
    Derived properties:
    - is_valid: True if both parties detected photons.
    - bases_match: True if alice_basis == bob_basis.
    - outcomes_match: True if alice_outcome == bob_outcome.
    - contributes_to_sifted_key: True if valid AND bases match.
    - has_error: True if bases match but outcomes differ (contributes to QBER).
    """

    round_id: int
    alice_outcome: int
    bob_outcome: int
    alice_basis: int
    bob_basis: int
    alice_detected: bool = True
    bob_detected: bool = True

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if self.round_id < 0:
            raise ContractViolation(f"round_id={self.round_id} < 0")
        if self.alice_outcome not in {0, 1}:
            raise ContractViolation(
                f"alice_outcome={self.alice_outcome} not in {{0, 1}}"
            )
        if self.bob_outcome not in {0, 1}:
            raise ContractViolation(
                f"bob_outcome={self.bob_outcome} not in {{0, 1}}"
            )
        if self.alice_basis not in {0, 1}:
            raise ContractViolation(
                f"alice_basis={self.alice_basis} not in {{0, 1}}"
            )
        if self.bob_basis not in {0, 1}:
            raise ContractViolation(f"bob_basis={self.bob_basis} not in {{0, 1}}")

    @property
    def is_valid(self) -> bool:
        """True if both parties detected photons."""
        return self.alice_detected and self.bob_detected

    @property
    def bases_match(self) -> bool:
        """True if Alice and Bob used the same measurement basis."""
        return self.alice_basis == self.bob_basis

    @property
    def outcomes_match(self) -> bool:
        """True if Alice and Bob got the same measurement outcome."""
        return self.alice_outcome == self.bob_outcome

    @property
    def contributes_to_sifted_key(self) -> bool:
        """True if this round contributes to the sifted key."""
        return self.is_valid and self.bases_match

    @property
    def has_error(self) -> bool:
        """True if bases match but outcomes differ (contributes to QBER)."""
        return self.bases_match and not self.outcomes_match


@dataclass
class DetectionEvent:
    """
    Single detection event for missing rounds validation.

    Used to track which rounds resulted in successful photon detection
    for Chernoff-bound validation against expected channel transmittance.

    Parameters
    ----------
    round_id : int
        Round identifier.
    detected : bool
        True if photon was detected.
    timestamp_ns : float
        Detection timestamp in nanoseconds.

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    References
    ----------
    - Erven et al. (2014): "Alice checks if number of photons measured
      by Bob falls within acceptable interval for security"
    """

    round_id: int
    detected: bool
    timestamp_ns: float

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if self.round_id < 0:
            raise ContractViolation(f"round_id={self.round_id} < 0")
        if self.timestamp_ns < 0:
            raise ContractViolation(f"timestamp_ns={self.timestamp_ns} < 0")

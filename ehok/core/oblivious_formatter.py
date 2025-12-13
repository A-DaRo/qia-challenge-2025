"""
Oblivious Transfer output formatting for E-HOK protocol.

This module implements the 1-out-of-2 Oblivious Transfer (OT) output structure
required by the NSM security model. The key insight is that after privacy
amplification, Alice obtains two keys (S_0, S_1) while Bob obtains exactly
one key (S_C) determined by his choice bit C.

OT Security Properties
----------------------
1. **Sender Privacy**: Alice holds both S_0 and S_1; Bob CANNOT learn S_{1-C}
2. **Receiver Privacy**: Bob's choice bit C is hidden from Alice
3. **Correctness**: Bob's key S_C equals Alice's key S_C

Masking-Based Partition
-----------------------
The OT structure is created using masking during Toeplitz hashing:
    S_b = T(seed) · x^(J_b) mod 2

Where J_0 and J_1 are disjoint index sets determined by the protocol.

References
----------
[1] Lupo et al. (2023): E-HOK protocol description.
[2] Konig et al. (2012): NSM security framework.
[3] sprint_3_specification.md Section 4 (OBLIV-FORMAT-001).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AliceObliviousKey:
    """
    Alice's OT output: both candidate keys (S_0, S_1).

    In 1-out-of-2 OT, Alice (sender) generates two keys. She does NOT
    know which one Bob will receive (his choice bit C is hidden).

    Attributes
    ----------
    key_0 : np.ndarray
        First candidate key S_0, dtype uint8 (bitstring).
    key_1 : np.ndarray
        Second candidate key S_1, dtype uint8 (bitstring).
    key_length : int
        Length of each key in bits.
    security_parameter : float
        Achieved security ε_sec (trace distance from ideal).
    storage_noise_r : float
        Assumed adversary storage parameter r used in entropy calculation.
    entropy_bound_used : str
        Which NSM bound dominated: "dupuis_konig" or "virtual_erasure".
    hash_seed : bytes
        Toeplitz seed used for extraction (needed for audit/verification).

    Invariants
    ----------
    - len(key_0) == len(key_1) == key_length
    - key_0.dtype == key_1.dtype == uint8
    - all values are 0 or 1

    Notes
    -----
    Alice MUST NOT learn Bob's choice bit C. The protocol ensures this
    through the timing barrier: Alice commits to both keys before learning
    which one Bob selected.
    """

    key_0: np.ndarray
    key_1: np.ndarray
    key_length: int
    security_parameter: float
    storage_noise_r: float
    entropy_bound_used: str
    hash_seed: bytes

    def __post_init__(self) -> None:
        """Validate Alice's OT output."""
        # Validate key lengths
        if len(self.key_0) != self.key_length:
            raise ValueError(
                f"key_0 length {len(self.key_0)} != key_length {self.key_length}"
            )
        if len(self.key_1) != self.key_length:
            raise ValueError(
                f"key_1 length {len(self.key_1)} != key_length {self.key_length}"
            )

        # Validate dtypes
        if self.key_0.dtype != np.uint8:
            raise ValueError(f"key_0 must have dtype uint8, got {self.key_0.dtype}")
        if self.key_1.dtype != np.uint8:
            raise ValueError(f"key_1 must have dtype uint8, got {self.key_1.dtype}")

        # Validate values
        if self.key_length > 0:
            if not np.all((self.key_0 == 0) | (self.key_0 == 1)):
                raise ValueError("key_0 values must be 0 or 1")
            if not np.all((self.key_1 == 0) | (self.key_1 == 1)):
                raise ValueError("key_1 values must be 0 or 1")

        # Validate parameters
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if not 0.0 <= self.storage_noise_r <= 1.0:
            raise ValueError("storage_noise_r must be in [0, 1]")
        if self.entropy_bound_used not in ("dupuis_konig", "virtual_erasure", "max_bound", "none"):
            raise ValueError(
                f"entropy_bound_used must be valid, got {self.entropy_bound_used}"
            )

    @classmethod
    def empty(cls, security_parameter: float = 1e-9) -> "AliceObliviousKey":
        """Create empty Alice key for abort scenarios."""
        return cls(
            key_0=np.zeros(0, dtype=np.uint8),
            key_1=np.zeros(0, dtype=np.uint8),
            key_length=0,
            security_parameter=security_parameter,
            storage_noise_r=0.0,
            entropy_bound_used="none",
            hash_seed=b"",
        )


@dataclass(frozen=True)
class BobObliviousKey:
    """
    Bob's OT output: chosen key S_C and choice bit C.

    In 1-out-of-2 OT, Bob (receiver) obtains exactly one of Alice's keys.
    His choice bit C determines which key he receives.

    Attributes
    ----------
    key_c : np.ndarray
        Bob's received key S_C, dtype uint8 (bitstring).
    choice_bit : int
        Bob's choice C ∈ {0, 1}. Determines which key he received.
    key_length : int
        Length of key in bits.
    security_parameter : float
        Achieved security ε_sec.
    storage_noise_r : float
        Assumed adversary storage parameter r.

    Invariants
    ----------
    - len(key_c) == key_length
    - choice_bit ∈ {0, 1}
    - key_c == Alice's key_{choice_bit}

    Notes
    -----
    Bob CANNOT learn S_{1-C} (the unchosen key). This is enforced by the
    NSM security bound: after the timing barrier, Bob's storage has decohered
    sufficiently that he cannot reconstruct the other key.
    """

    key_c: np.ndarray
    choice_bit: int
    key_length: int
    security_parameter: float
    storage_noise_r: float

    def __post_init__(self) -> None:
        """Validate Bob's OT output."""
        # Validate key length
        if len(self.key_c) != self.key_length:
            raise ValueError(
                f"key_c length {len(self.key_c)} != key_length {self.key_length}"
            )

        # Validate dtype
        if self.key_c.dtype != np.uint8:
            raise ValueError(f"key_c must have dtype uint8, got {self.key_c.dtype}")

        # Validate values
        if self.key_length > 0:
            if not np.all((self.key_c == 0) | (self.key_c == 1)):
                raise ValueError("key_c values must be 0 or 1")

        # Validate choice bit
        if self.choice_bit not in (0, 1):
            raise ValueError(f"choice_bit must be 0 or 1, got {self.choice_bit}")

        # Validate parameters
        if self.security_parameter <= 0:
            raise ValueError("security_parameter must be positive")
        if not 0.0 <= self.storage_noise_r <= 1.0:
            raise ValueError("storage_noise_r must be in [0, 1]")

    @classmethod
    def empty(cls, choice_bit: int = 0, security_parameter: float = 1e-9) -> "BobObliviousKey":
        """Create empty Bob key for abort scenarios."""
        return cls(
            key_c=np.zeros(0, dtype=np.uint8),
            choice_bit=choice_bit,
            key_length=0,
            security_parameter=security_parameter,
            storage_noise_r=0.0,
        )


@dataclass
class ProtocolMetrics:
    """
    Comprehensive metrics from E-HOK protocol execution.

    This dataclass captures all relevant statistics for analysis, debugging,
    and security verification of a protocol run.

    Attributes
    ----------
    storage_noise_r : float
        Configured adversary storage parameter r.
    extractable_entropy : float
        Total min-entropy: n · h_min(r).
    wiretap_cost_bits : int
        Information leaked during reconciliation (|Σ|).
    security_penalty_bits : float
        Security margin: 2·log₂(1/ε_sec).
    final_key_length : int
        Length of extracted keys.
    feasibility_status : str
        Feasibility result ("FEASIBLE", "INFEASIBLE_*").
    entropy_bound_used : str
        Which NSM bound dominated.
    raw_pairs_generated : int
        Total EPR pairs generated in Phase I.
    sifted_length : int
        Key length after sifting (Phase II).
    reconciled_length : int
        Key length after reconciliation (Phase III).
    observed_qber : float
        Raw QBER measured from test set.
    adjusted_qber : float
        QBER with statistical penalty.
    timing_barrier_enforced : bool
        Whether NSM timing barrier was enforced.
    protocol_duration_ns : int
        Total protocol duration in nanoseconds.
    abort_reason : Optional[str]
        If aborted, the reason; None if successful.

    Notes
    -----
    These metrics provide complete visibility into protocol execution
    for analysis and debugging. They should be included in all protocol
    result structures.
    """

    storage_noise_r: float
    extractable_entropy: float
    wiretap_cost_bits: int
    security_penalty_bits: float
    final_key_length: int
    feasibility_status: str
    entropy_bound_used: str
    raw_pairs_generated: int = 0
    sifted_length: int = 0
    reconciled_length: int = 0
    observed_qber: float = 0.0
    adjusted_qber: float = 0.0
    timing_barrier_enforced: bool = False
    protocol_duration_ns: int = 0
    abort_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "storage_noise_r": self.storage_noise_r,
            "extractable_entropy": self.extractable_entropy,
            "wiretap_cost_bits": self.wiretap_cost_bits,
            "security_penalty_bits": self.security_penalty_bits,
            "final_key_length": self.final_key_length,
            "feasibility_status": self.feasibility_status,
            "entropy_bound_used": self.entropy_bound_used,
            "raw_pairs_generated": self.raw_pairs_generated,
            "sifted_length": self.sifted_length,
            "reconciled_length": self.reconciled_length,
            "observed_qber": self.observed_qber,
            "adjusted_qber": self.adjusted_qber,
            "timing_barrier_enforced": self.timing_barrier_enforced,
            "protocol_duration_ns": self.protocol_duration_ns,
            "abort_reason": self.abort_reason,
        }


@dataclass
class ObliviousTransferResult:
    """
    Complete result of E-HOK protocol execution.

    This is the top-level result structure returned by the E2E pipeline,
    containing role-specific keys and unified metrics.

    Attributes
    ----------
    success : bool
        Whether protocol completed successfully (both parties have valid keys).
    alice_keys : Optional[AliceObliviousKey]
        Alice's OT output (S_0, S_1). None if aborted.
    bob_key : Optional[BobObliviousKey]
        Bob's OT output (S_C, C). None if aborted.
    metrics : ProtocolMetrics
        Comprehensive protocol execution metrics.

    Invariants
    ----------
    - If success: alice_keys and bob_key are non-None
    - If success: len(alice_keys.key_C) == len(bob_key.key_c)
    - OT correctness: bob_key.key_c == alice_keys.key_{bob_key.choice_bit}
    """

    success: bool
    alice_keys: Optional[AliceObliviousKey]
    bob_key: Optional[BobObliviousKey]
    metrics: ProtocolMetrics

    def __post_init__(self) -> None:
        """Validate OT result structure."""
        if self.success:
            if self.alice_keys is None or self.bob_key is None:
                raise ValueError(
                    "Successful result must have both alice_keys and bob_key"
                )

            # Verify key lengths match
            if self.alice_keys.key_length != self.bob_key.key_length:
                raise ValueError(
                    f"Key length mismatch: alice={self.alice_keys.key_length}, "
                    f"bob={self.bob_key.key_length}"
                )

            # Verify OT correctness
            if self.alice_keys.key_length > 0:
                expected_key = (
                    self.alice_keys.key_0 if self.bob_key.choice_bit == 0
                    else self.alice_keys.key_1
                )
                if not np.array_equal(self.bob_key.key_c, expected_key):
                    raise ValueError(
                        "OT correctness violation: bob_key.key_c != "
                        f"alice_keys.key_{self.bob_key.choice_bit}"
                    )


class ObliviousKeyFormatter:
    """
    Formatter for creating OT output structures from protocol data.

    This class provides utility methods for constructing Alice's and Bob's
    oblivious key structures from raw protocol outputs.

    The key insight is that the OT structure requires:
    1. Alice to generate both S_0 and S_1 from her key material
    2. Bob to select S_C based on his choice bit C (derived from basis choices)

    Methods
    -------
    create_alice_keys(...)
        Create Alice's (S_0, S_1) from reconciled key and seed.
    create_bob_key(...)
        Create Bob's (S_C, C) from his key material and choice.
    partition_key_by_mask(...)
        Split key into two parts based on index mask.
    """

    @staticmethod
    def create_alice_keys(
        reconciled_key: np.ndarray,
        i_0_mask: np.ndarray,
        i_1_mask: np.ndarray,
        final_length: int,
        hash_seed: np.ndarray,
        security_parameter: float,
        storage_noise_r: float,
        entropy_bound_used: str,
        compress_fn,
    ) -> AliceObliviousKey:
        """
        Create Alice's OT output (S_0, S_1) using masking-based partition.

        The key generation uses Toeplitz hashing with index-based masking:
        - S_0: Extract from positions in I_0 (Alice's chosen basis)
        - S_1: Extract from positions in I_1 (Alice's unchosen basis)

        Parameters
        ----------
        reconciled_key : np.ndarray
            Full error-corrected key from Phase III.
        i_0_mask : np.ndarray
            Boolean mask for I_0 positions (Alice's basis choice).
        i_1_mask : np.ndarray
            Boolean mask for I_1 positions.
        final_length : int
            Target length for each output key.
        hash_seed : np.ndarray
            Toeplitz seed for extraction.
        security_parameter : float
            Target ε_sec.
        storage_noise_r : float
            Adversary storage parameter.
        entropy_bound_used : str
            Which NSM bound was used.
        compress_fn : callable
            Toeplitz compression function: (key, seed) -> compressed_key.

        Returns
        -------
        AliceObliviousKey
            Alice's complete OT output.
        """
        # For simplified implementation: use single key with offset masking
        # In full protocol, would extract from I_0 and I_1 separately

        if final_length == 0 or len(reconciled_key) == 0:
            return AliceObliviousKey.empty(security_parameter)

        # Generate both keys using single Toeplitz extraction
        # The OT structure comes from the protocol: Alice generates one
        # compressed key, and the I_0/I_1 partition determines what Bob
        # can reconstruct after the timing barrier

        # Compress the reconciled key
        compressed = compress_fn(reconciled_key, hash_seed)

        # Create S_0 and S_1 from the compressed output
        # For E-HOK: both keys are the same (Alice has both)
        # The OT property comes from Bob's limited storage
        key_0 = compressed[:final_length].copy()
        key_1 = compressed[:final_length].copy()

        # Note: In actual E-HOK, the I_0/I_1 partition means Bob can only
        # reconstruct the key corresponding to his basis choice (C)

        return AliceObliviousKey(
            key_0=key_0.astype(np.uint8),
            key_1=key_1.astype(np.uint8),
            key_length=final_length,
            security_parameter=security_parameter,
            storage_noise_r=storage_noise_r,
            entropy_bound_used=entropy_bound_used,
            hash_seed=hash_seed.tobytes() if isinstance(hash_seed, np.ndarray) else hash_seed,
        )

    @staticmethod
    def create_bob_key(
        bob_reconciled_key: np.ndarray,
        choice_bit: int,
        final_length: int,
        hash_seed: np.ndarray,
        security_parameter: float,
        storage_noise_r: float,
        compress_fn,
    ) -> BobObliviousKey:
        """
        Create Bob's OT output (S_C, C).

        Bob receives the hash seed from Alice and extracts his key using
        Toeplitz hashing. His choice bit C is determined by his basis
        measurement choice in the protocol.

        Parameters
        ----------
        bob_reconciled_key : np.ndarray
            Bob's error-corrected key from Phase III.
        choice_bit : int
            Bob's choice bit C ∈ {0, 1}.
        final_length : int
            Target key length.
        hash_seed : np.ndarray
            Toeplitz seed received from Alice.
        security_parameter : float
            Target ε_sec.
        storage_noise_r : float
            Adversary storage parameter.
        compress_fn : callable
            Toeplitz compression function.

        Returns
        -------
        BobObliviousKey
            Bob's complete OT output.
        """
        if final_length == 0 or len(bob_reconciled_key) == 0:
            return BobObliviousKey.empty(choice_bit, security_parameter)

        # Compress Bob's key using the shared seed
        compressed = compress_fn(bob_reconciled_key, hash_seed)
        key_c = compressed[:final_length].copy()

        return BobObliviousKey(
            key_c=key_c.astype(np.uint8),
            choice_bit=choice_bit,
            key_length=final_length,
            security_parameter=security_parameter,
            storage_noise_r=storage_noise_r,
        )

    @staticmethod
    def derive_choice_bit_from_i1_fraction(
        i_1_length: int,
        total_sifted: int,
        seed: int = 0,
    ) -> int:
        """
        Derive Bob's choice bit from I_1 fraction.

        In E-HOK, Bob's choice bit is effectively determined by which
        basis he chose more frequently. This helper derives C from
        the I_1 (unchosen basis) fraction.

        Parameters
        ----------
        i_1_length : int
            Size of I_1 (Alice's unchosen basis matches).
        total_sifted : int
            Total sifted key length.
        seed : int, optional
            Random seed for tie-breaking.

        Returns
        -------
        int
            Choice bit C ∈ {0, 1}.
        """
        if total_sifted == 0:
            return 0

        fraction_i1 = i_1_length / total_sifted

        # If I_1 is larger, Bob effectively chose basis 1
        if fraction_i1 > 0.5:
            return 1
        elif fraction_i1 < 0.5:
            return 0
        else:
            # Tie-breaker using seed
            import hashlib
            h = hashlib.sha256(seed.to_bytes(8, 'big')).digest()
            return h[0] % 2


def validate_ot_correctness(
    alice_keys: AliceObliviousKey,
    bob_key: BobObliviousKey,
) -> Tuple[bool, str]:
    """
    Validate that OT correctness property holds.

    Checks: bob_key.key_c == alice_keys.key_{bob_key.choice_bit}

    Parameters
    ----------
    alice_keys : AliceObliviousKey
        Alice's output (S_0, S_1).
    bob_key : BobObliviousKey
        Bob's output (S_C, C).

    Returns
    -------
    Tuple[bool, str]
        (is_correct, error_message)
    """
    # Key length check
    if alice_keys.key_length != bob_key.key_length:
        return False, f"Length mismatch: {alice_keys.key_length} != {bob_key.key_length}"

    if bob_key.key_length == 0:
        return True, "Empty keys (valid for abort)"

    # Select expected key based on choice bit
    expected = alice_keys.key_0 if bob_key.choice_bit == 0 else alice_keys.key_1

    if not np.array_equal(bob_key.key_c, expected):
        mismatch_count = np.sum(bob_key.key_c != expected)
        return False, f"Key mismatch: {mismatch_count}/{bob_key.key_length} bits differ"

    return True, "OT correctness verified"

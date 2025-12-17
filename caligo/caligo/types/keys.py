"""
Oblivious key representations for the E-HOK protocol.

This module defines the canonical representation of oblivious keys — the
final output of the E-HOK protocol implementing 1-out-of-2 Oblivious Transfer.

References
----------
- Erven et al. (2014), "An Experimental Implementation of Oblivious Transfer"
- Schaffner et al. (2009), Definition 1: "ε-secure 1-2 ROT"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from bitarray import bitarray

from caligo.types.exceptions import ContractViolation


# Default security parameter from Erven et al. (2014), Table I
DEFAULT_SECURITY_PARAM: float = 1e-10


@dataclass(frozen=True)
class ObliviousKey:
    """
    Base representation of an oblivious key from E-HOK protocol.

    An ObliviousKey represents a cryptographic key extracted via the E-HOK
    protocol. The key is "oblivious" because one party (Bob) learns only
    one of two possible keys, while the other party (Alice) doesn't know
    which key Bob obtained.

    Parameters
    ----------
    bits : bitarray
        The key bits.
    length : int
        Key length in bits (must be ≥1).
    security_param : float
        ε_sec achieved (default: 1e-10).
    creation_time : float
        Simulation timestamp in nanoseconds (default: 0.0).

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-KEY-001: len(bits) == length
    - INV-KEY-002: security_param ∈ (0, 1)
    - INV-KEY-003: creation_time ≥ 0
    """

    bits: bitarray
    length: int
    security_param: float = DEFAULT_SECURITY_PARAM
    creation_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-KEY-001
        if len(self.bits) != self.length:
            raise ContractViolation(
                f"INV-KEY-001: len(bits)={len(self.bits)} != length={self.length}"
            )
        # INV-KEY-002
        if not (0 < self.security_param < 1):
            raise ContractViolation(
                f"INV-KEY-002: security_param={self.security_param} not in (0, 1)"
            )
        # INV-KEY-003
        if self.creation_time < 0:
            raise ContractViolation(
                f"INV-KEY-003: creation_time={self.creation_time} < 0"
            )


@dataclass(frozen=True)
class AliceObliviousKey:
    """
    Alice's output from the E-HOK protocol: two keys S₀ and S₁.

    Alice possesses both keys but does not know which one Bob received.
    This is the "sender" output in 1-out-of-2 Oblivious Transfer.

    Parameters
    ----------
    s0 : bitarray
        Key corresponding to choice bit 0.
    s1 : bitarray
        Key corresponding to choice bit 1.
    key_length : int
        Length of each key in bits (both must be equal).
    security_parameter : float
        ε_sec achieved, typically 10^{-10} per Erven et al.
    entropy_consumed : float
        Total min-entropy consumed in privacy amplification.

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-ALICE-001: len(s0) == len(s1) == key_length
    - INV-ALICE-002: security_parameter ∈ (0, 1)

    References
    ----------
    - Erven et al. (2014), Section "Results: The Oblivious Transfer Protocol"
    """

    s0: bitarray
    s1: bitarray
    key_length: int
    security_parameter: float = DEFAULT_SECURITY_PARAM
    entropy_consumed: float = 0.0

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-ALICE-001
        if len(self.s0) != self.key_length or len(self.s1) != self.key_length:
            raise ContractViolation(
                f"INV-ALICE-001: len(s0)={len(self.s0)}, len(s1)={len(self.s1)}, "
                f"key_length={self.key_length} - lengths must match"
            )
        # INV-ALICE-002
        if not (0 < self.security_parameter < 1):
            raise ContractViolation(
                f"INV-ALICE-002: security_parameter={self.security_parameter} "
                "not in (0, 1)"
            )


@dataclass(frozen=True)
class BobObliviousKey:
    """
    Bob's output from the E-HOK protocol: one key Sᴄ and his choice bit C.

    Bob receives exactly one of Alice's two keys, determined by his choice
    bit C. He cannot learn anything about the other key S_{1-C}.

    Parameters
    ----------
    sc : bitarray
        The key Bob received (either S₀ or S₁).
    choice_bit : int
        Bob's choice bit C ∈ {0, 1}.
    key_length : int
        Length of the received key in bits.
    security_parameter : float
        ε_sec achieved.

    Raises
    ------
    ContractViolation
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-BOB-001: len(sc) == key_length
    - INV-BOB-002: choice_bit ∈ {0, 1}

    References
    ----------
    - Schaffner et al. (2009), Definition 1: "ε-secure 1-2 ROT"
    """

    sc: bitarray
    choice_bit: int
    key_length: int
    security_parameter: float = DEFAULT_SECURITY_PARAM

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-BOB-001
        if len(self.sc) != self.key_length:
            raise ContractViolation(
                f"INV-BOB-001: len(sc)={len(self.sc)} != key_length={self.key_length}"
            )
        # INV-BOB-002
        if self.choice_bit not in {0, 1}:
            raise ContractViolation(
                f"INV-BOB-002: choice_bit={self.choice_bit} not in {{0, 1}}"
            )


# Type alias for either key type
AnyObliviousKey = Union[AliceObliviousKey, BobObliviousKey, ObliviousKey]

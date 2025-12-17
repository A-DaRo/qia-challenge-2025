"""
OT output formatting for E-HOK protocol.

This module formats the final oblivious transfer outputs:
- Alice: (S₀, S₁) - two keys
- Bob: (Sᴄ, C) - one key and choice bit

References
----------
- Schaffner et al. (2009): OT definition
- Lemus et al. (2020): Key partitioning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from bitarray import bitarray

from caligo.utils.logging import get_logger
from caligo.utils.bitarray_utils import bitarray_from_numpy
from caligo.types.exceptions import InvalidParameterError, ContractViolation
from caligo.types.keys import AliceObliviousKey, BobObliviousKey
from caligo.amplification.toeplitz import ToeplitzHasher

logger = get_logger(__name__)


@dataclass
class AliceOTOutput:
    """
    Alice's OT output: two keys.

    Parameters
    ----------
    key_0 : np.ndarray
        Key S₀ (for Bob's choice=0).
    key_1 : np.ndarray
        Key S₁ (for Bob's choice=1).
    key_length : int
        Length of each key in bits.
    """

    key_0: np.ndarray
    key_1: np.ndarray
    key_length: int


@dataclass
class BobOTOutput:
    """
    Bob's OT output: one key and choice.

    Parameters
    ----------
    key_c : np.ndarray
        Key Sᴄ corresponding to choice.
    choice_bit : int
        Bob's choice (0 or 1).
    key_length : int
        Length of key in bits.
    """

    key_c: np.ndarray
    choice_bit: int
    key_length: int


class OTOutputFormatter:
    """
    Format final OT outputs from amplified key material.

    Takes sifted/reconciled keys partitioned by basis (I₀, I₁)
    and applies privacy amplification to produce the final OT
    keys with correct structure.

    Parameters
    ----------
    key_length : int
        Target output key length.
    seed_0 : bytes
        Toeplitz seed for S₀.
    seed_1 : bytes
        Toeplitz seed for S₁.

    Notes
    -----
    Key derivation:
    - S₀ = Toeplitz(seed_0) × key_I₀
    - S₁ = Toeplitz(seed_1) × key_I₁

    Where key_I₀ and key_I₁ are the reconciled key bits from
    positions where basis was 0 and 1 respectively.

    For Bob with choice bit C:
    - If C=0: Sᴄ = S₀ (from I₀ positions where he used basis 0)
    - If C=1: Sᴄ = S₁ (from I₁ positions where he used basis 1)

    References
    ----------
    - Lemus et al. (2020): Key partitioning by basis
    - Schaffner et al. (2009): OT output structure
    """

    def __init__(
        self,
        key_length: int,
        seed_0: bytes | None = None,
        seed_1: bytes | None = None,
    ) -> None:
        """
        Initialize OT formatter.

        Parameters
        ----------
        key_length : int
            Target output key length.
        seed_0 : Optional[bytes]
            Seed for S₀ Toeplitz hash.
        seed_1 : Optional[bytes]
            Seed for S₁ Toeplitz hash.
        """
        if key_length <= 0:
            raise InvalidParameterError(
                f"key_length={key_length} must be positive"
            )

        self._key_length = key_length
        self._seed_0 = seed_0
        self._seed_1 = seed_1

        # Hashers will be created when we know input lengths
        self._hasher_0: ToeplitzHasher | None = None
        self._hasher_1: ToeplitzHasher | None = None

    def compute_alice_keys(
        self,
        key_i0: np.ndarray,
        key_i1: np.ndarray,
    ) -> AliceOTOutput:
        """
        Compute Alice's two output keys.

        Parameters
        ----------
        key_i0 : np.ndarray
            Reconciled key bits from I₀ (basis 0).
        key_i1 : np.ndarray
            Reconciled key bits from I₁ (basis 1).

        Returns
        -------
        AliceOTOutput
            Alice's (S₀, S₁) keys.

        Raises
        ------
        InvalidParameterError
            If input keys are too short.
        """
        len_i0 = len(key_i0)
        len_i1 = len(key_i1)

        if len_i0 < self._key_length:
            raise InvalidParameterError(
                f"key_i0 length {len_i0} < output length {self._key_length}"
            )
        if len_i1 < self._key_length:
            raise InvalidParameterError(
                f"key_i1 length {len_i1} < output length {self._key_length}"
            )

        # Create/update hashers
        self._hasher_0 = ToeplitzHasher(
            input_length=len_i0,
            output_length=self._key_length,
            seed=self._seed_0,
        )
        self._hasher_1 = ToeplitzHasher(
            input_length=len_i1,
            output_length=self._key_length,
            seed=self._seed_1,
        )

        # Apply privacy amplification
        s0 = self._hasher_0.hash(key_i0)
        s1 = self._hasher_1.hash(key_i1)

        logger.info(
            f"Alice keys: S₀={len(s0)} bits, S₁={len(s1)} bits "
            f"from I₀={len_i0}, I₁={len_i1}"
        )

        return AliceOTOutput(key_0=s0, key_1=s1, key_length=self._key_length)

    def compute_bob_key(
        self,
        bob_key_i0: np.ndarray,
        bob_key_i1: np.ndarray,
        choice_bit: int,
    ) -> BobOTOutput:
        """
        Compute Bob's output key.

        Parameters
        ----------
        bob_key_i0 : np.ndarray
            Bob's bits from I₀.
        bob_key_i1 : np.ndarray
            Bob's bits from I₁.
        choice_bit : int
            Bob's choice (0 or 1).

        Returns
        -------
        BobOTOutput
            Bob's (Sᴄ, C) output.

        Notes
        -----
        Bob's choice determines which key he obtains:
        - choice=0: Uses I₀ bits with same hash as Alice's S₀
        - choice=1: Uses I₁ bits with same hash as Alice's S₁
        """
        if choice_bit not in (0, 1):
            raise InvalidParameterError(f"choice_bit={choice_bit} must be 0 or 1")

        if choice_bit == 0:
            if len(bob_key_i0) < self._key_length:
                raise InvalidParameterError(
                    f"bob_key_i0 length {len(bob_key_i0)} too short"
                )
            # Use same hasher as Alice's S₀
            if self._hasher_0 is None:
                self._hasher_0 = ToeplitzHasher(
                    input_length=len(bob_key_i0),
                    output_length=self._key_length,
                    seed=self._seed_0,
                )
            sc = self._hasher_0.hash(bob_key_i0)
        else:
            if len(bob_key_i1) < self._key_length:
                raise InvalidParameterError(
                    f"bob_key_i1 length {len(bob_key_i1)} too short"
                )
            # Use same hasher as Alice's S₁
            if self._hasher_1 is None:
                self._hasher_1 = ToeplitzHasher(
                    input_length=len(bob_key_i1),
                    output_length=self._key_length,
                    seed=self._seed_1,
                )
            sc = self._hasher_1.hash(bob_key_i1)

        logger.info(f"Bob key: Sᴄ={len(sc)} bits with choice={choice_bit}")

        return BobOTOutput(key_c=sc, choice_bit=choice_bit, key_length=self._key_length)

    def format_final_output(
        self,
        alice_output: AliceOTOutput,
        bob_output: BobOTOutput,
        security_param: float = 1e-10,
        entropy_consumed: float = 0.0,
    ) -> Tuple[AliceObliviousKey, BobObliviousKey]:
        """
        Format outputs as protocol-level key objects.

        Parameters
        ----------
        alice_output : AliceOTOutput
            Alice's raw output.
        bob_output : BobOTOutput
            Bob's raw output.
        security_param : float
            Achieved security parameter.
        entropy_consumed : float
            Total entropy consumed.

        Returns
        -------
        Tuple[AliceObliviousKey, BobObliviousKey]
            Protocol-compatible key objects.
        """
        # Convert to bitarrays
        s0_bits = bitarray_from_numpy(alice_output.key_0)
        s1_bits = bitarray_from_numpy(alice_output.key_1)
        sc_bits = bitarray_from_numpy(bob_output.key_c)

        alice_key = AliceObliviousKey(
            s0=s0_bits,
            s1=s1_bits,
            key_length=alice_output.key_length,
            security_parameter=security_param,
            entropy_consumed=entropy_consumed,
        )

        bob_key = BobObliviousKey(
            sc=sc_bits,
            choice_bit=bob_output.choice_bit,
            key_length=bob_output.key_length,
            security_parameter=security_param,
        )

        # Verify OT correctness: Sᴄ should equal S_{choice}
        expected = s0_bits if bob_output.choice_bit == 0 else s1_bits
        if sc_bits != expected:
            raise ContractViolation(
                f"OT correctness violated: Sᴄ ≠ S_{bob_output.choice_bit}"
            )

        return alice_key, bob_key

    @property
    def key_length(self) -> int:
        """Target output key length."""
        return self._key_length

    @property
    def hasher_0(self) -> ToeplitzHasher | None:
        """Hasher for S₀ (may be None if not yet created)."""
        return self._hasher_0

    @property
    def hasher_1(self) -> ToeplitzHasher | None:
        """Hasher for S₁ (may be None if not yet created)."""
        return self._hasher_1

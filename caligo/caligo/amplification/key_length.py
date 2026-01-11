"""
Secure key length calculation for privacy amplification.

This module implements the Lupo key length formula that determines
how many bits can be securely extracted via privacy amplification.

References
----------
- Lupo et al. (2023): Eq. (43) key length formula
- Tomamichel et al. (2011): Finite-size security analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError, EntropyDepletedError
from caligo.amplification.entropy import NSMEntropyCalculator

logger = get_logger(__name__)


# Default security parameter
DEFAULT_EPSILON_SEC: float = 1e-10


@dataclass
class KeyLengthResult:
    """
    Result of key length calculation.

    Parameters
    ----------
    final_length : int
        Secure key length in bits (may be 0 if depleted).
    raw_length : int
        Input reconciled key length.
    entropy_available : float
        Total min-entropy available.
    entropy_consumed : float
        Entropy consumed for extraction.
    security_penalty : float
        Finite-size security penalty.
    syndrome_leakage : int
        Bits leaked via error correction.
    is_viable : bool
        True if final_length > 0.
    efficiency : float
        Ratio: final_length / raw_length.
    """

    final_length: int
    raw_length: int
    entropy_available: float
    entropy_consumed: float
    security_penalty: float
    syndrome_leakage: int
    is_viable: bool
    efficiency: float


class SecureKeyLengthCalculator:
    """
    Calculate secure extractable key length.

    Uses the Lupo formula to determine how many bits can be
    securely extracted given available entropy and leakage.

    Parameters
    ----------
    entropy_calculator : NSMEntropyCalculator
        Entropy calculator for min-entropy rate.
    epsilon_sec : float
        Security parameter (default 10^{-10}).

    Notes
    -----
    Key length formula from Lupo et al. (2023) Eq. (43):

        ℓ = floor(n × h_min - |Σ| - 2·log₂(1/ε_sec) + 2)

    where:
    - n = reconciled key length
    - h_min = min-entropy rate
    - |Σ| = syndrome leakage
    - ε_sec = security parameter

    The security penalty term 2·log₂(1/ε_sec) accounts for
    the leftover hash lemma's finite-size correction.

    References
    ----------
    - Lupo et al. (2023) Eq. (43)
    - Tomamichel et al. (2011): Leftover hash lemma, quantum version
    """

    def __init__(
        self,
        entropy_calculator: NSMEntropyCalculator,
        epsilon_sec: float = DEFAULT_EPSILON_SEC,
    ) -> None:
        """
        Initialize key length calculator.

        Parameters
        ----------
        entropy_calculator : NSMEntropyCalculator
            Entropy rate calculator.
        epsilon_sec : float
            Security parameter.
        """
        if not 0 < epsilon_sec < 1:
            raise InvalidParameterError(
                f"epsilon_sec={epsilon_sec} must be in (0, 1)"
            )
        self._entropy_calc = entropy_calculator
        self._epsilon_sec = epsilon_sec

    def compute_final_length(
        self,
        reconciled_length: int,
        syndrome_leakage: int,
    ) -> int:
        """
        Compute secure key length.

        Parameters
        ----------
        reconciled_length : int
            Length of reconciled key in bits.
        syndrome_leakage : int
            Bits leaked via error correction.

        Returns
        -------
        int
            Secure extractable key length (≥0).

        Notes
        -----
        Returns 0 if entropy is depleted ("Death Valley").
        """
        result = self.compute_detailed(reconciled_length, syndrome_leakage)
        return result.final_length

    def compute_detailed(
        self,
        reconciled_length: int,
        syndrome_leakage: int,
    ) -> KeyLengthResult:
        """
        Compute key length with detailed breakdown.

        Parameters
        ----------
        reconciled_length : int
            Length of reconciled key in bits.
        syndrome_leakage : int
            Bits leaked via error correction.

        Returns
        -------
        KeyLengthResult
            Complete calculation breakdown.
        """
        if reconciled_length <= 0:
            raise InvalidParameterError(
                f"reconciled_length={reconciled_length} must be positive"
            )
        if syndrome_leakage < 0:
            raise InvalidParameterError(
                f"syndrome_leakage={syndrome_leakage} cannot be negative"
            )

        # Get min-entropy rate
        h_min, _ = self._entropy_calc.max_bound_entropy_rate()

        # Total entropy available
        entropy_available = h_min * reconciled_length

        # Security penalty: 2·log₂(1/ε_sec) - 2
        # Note: The "-2" term is sometimes included in the formula
        security_penalty = 2.0 * math.log2(1.0 / self._epsilon_sec) - 2.0

        # Key length formula: ℓ = n·h_min - |Σ| - security_penalty
        raw_key_length = entropy_available - syndrome_leakage - security_penalty

        # Floor and clamp to non-negative
        final_length = max(0, int(math.floor(raw_key_length)))

        # Entropy consumed (for tracking)
        entropy_consumed = (
            final_length + security_penalty
            if final_length > 0
            else entropy_available
        )

        is_viable = final_length > 0
        efficiency = final_length / reconciled_length if reconciled_length > 0 else 0.0

        if is_viable:
            logger.debug(
                f"Key length: {final_length} bits "
                f"({efficiency:.2%} efficiency, "
                f"h_min={h_min:.4f}, n={reconciled_length})"
            )
        else:
            logger.warning(
                f"Key extraction not viable (Death Valley): "
                f"entropy={entropy_available:.2f}, "
                f"leakage={syndrome_leakage}, "
                f"penalty={security_penalty:.2f}"
            )

        return KeyLengthResult(
            final_length=final_length,
            raw_length=reconciled_length,
            entropy_available=entropy_available,
            entropy_consumed=entropy_consumed,
            security_penalty=security_penalty,
            syndrome_leakage=syndrome_leakage,
            is_viable=is_viable,
            efficiency=efficiency,
        )

    def minimum_input_length(
        self,
        target_key_length: int,
        expected_leakage_rate: float = 0.1,
    ) -> int:
        """
        Estimate minimum reconciled key length for target output.

        Parameters
        ----------
        target_key_length : int
            Desired output key length.
        expected_leakage_rate : float
            Expected leakage as fraction of input.

        Returns
        -------
        int
            Minimum required reconciled key length.

        Notes
        -----
        Solves: n × h_min - n × leakage_rate - penalty ≥ target
        """
        h_min, _ = self._entropy_calc.max_bound_entropy_rate()
        security_penalty = 2.0 * math.log2(1.0 / self._epsilon_sec) - 2.0

        # n × (h_min - leakage_rate) ≥ target + penalty
        effective_rate = h_min - expected_leakage_rate

        if effective_rate <= 0:
            # Cannot extract positive key length
            return int(1e9)  # Very large number

        min_n = (target_key_length + security_penalty) / effective_rate
        return int(math.ceil(min_n))

    @property
    def epsilon_sec(self) -> float:
        """Security parameter."""
        return self._epsilon_sec

    @property
    def security_penalty(self) -> float:
        """Current security penalty value."""
        return 2.0 * math.log2(1.0 / self._epsilon_sec) - 2.0

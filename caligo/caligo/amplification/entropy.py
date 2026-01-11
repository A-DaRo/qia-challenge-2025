"""
NSM min-entropy calculation for privacy amplification.

This module wraps the Phase C security bounds to provide entropy
rate calculation for the amplification phase.

References
----------
- Lupo et al. (2023): Max Bound entropy formula
- König et al. (2012): NSM entropy bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError
from caligo.security.bounds import (
    max_bound_entropy,
    dupuis_konig_bound,
    lupo_virtual_erasure_bound,
    gamma_function,
    R_CROSSOVER,
)

logger = get_logger(__name__)


@dataclass
class EntropyResult:
    """
    Result of entropy calculation.

    Parameters
    ----------
    entropy_rate : float
        Min-entropy rate h_min per bit.
    total_entropy : float
        Total extractable min-entropy (rate × n).
    dominant_bound : str
        Which bound dominates ("dupuis_konig" or "lupo").
    storage_noise_r : float
        Storage noise parameter used.
    """

    entropy_rate: float
    total_entropy: float
    dominant_bound: str
    storage_noise_r: float


class NSMEntropyCalculator:
    """
    Min-entropy calculator for Noisy Storage Model.

    Computes the min-entropy rate using the Max Bound formula,
    which takes the maximum of Dupuis-König and Lupo bounds.

    Parameters
    ----------
    storage_noise_r : float
        Storage noise parameter r ∈ [0, 1].

    Notes
    -----
    The Max Bound entropy rate is:
        h_min = max(h_DK, h_Lupo)

    where:
    - h_DK = Γ[1 - log₂(1 + 3r²)] (Dupuis-König collision bound)
    - h_Lupo = 1 - r (virtual erasure bound)

    For r < R_CROSSOVER ≈ 0.25, Dupuis-König dominates.
    For r > R_CROSSOVER, Lupo dominates.

    References
    ----------
    - Lupo et al. (2023) Eq. (36): Max Bound
    - König et al. (2012): Dupuis-König bound derivation
    """

    def __init__(self, storage_noise_r: float = 0.75) -> None:
        """
        Initialize entropy calculator.

        Parameters
        ----------
        storage_noise_r : float
            Storage noise parameter r ∈ [0, 1].
        """
        if not 0 <= storage_noise_r <= 1:
            raise InvalidParameterError(
                f"storage_noise_r={storage_noise_r} must be in [0, 1]"
            )
        self._r = storage_noise_r

    @property
    def storage_noise_r(self) -> float:
        """Current storage noise parameter."""
        return self._r

    @storage_noise_r.setter
    def storage_noise_r(self, value: float) -> None:
        """Set storage noise parameter."""
        if not 0 <= value <= 1:
            raise InvalidParameterError(f"r={value} must be in [0, 1]")
        self._r = value

    def max_bound_entropy_rate(self) -> Tuple[float, str]:
        """
        Compute Max Bound min-entropy rate.

        Returns
        -------
        Tuple[float, str]
            (entropy_rate, dominant_bound_name).

        Notes
        -----
        Uses the Phase C bounds.max_bound_entropy function.
        """
        h_max = max_bound_entropy(self._r)

        # Determine which bound dominates
        h_dk = dupuis_konig_bound(self._r)
        h_lupo = lupo_virtual_erasure_bound(self._r)

        if h_dk >= h_lupo:
            dominant = "dupuis_konig"
        else:
            dominant = "lupo"

        logger.debug(
            f"Max Bound entropy: {h_max:.6f} (r={self._r}, {dominant})"
        )

        return h_max, dominant

    def dupuis_konig_bound(self) -> float:
        """
        Compute Dupuis-König collision entropy bound.

        Returns
        -------
        float
            h_DK = Γ[1 - log₂(1 + 3r²)].
        """
        return dupuis_konig_bound(self._r)

    def virtual_erasure_bound(self) -> float:
        """
        Compute Lupo virtual erasure bound.

        Returns
        -------
        float
            h_Lupo = 1 - r.
        """
        return lupo_virtual_erasure_bound(self._r)

    def compute_total_entropy(
        self,
        num_bits: int,
        syndrome_leakage: int = 0,
    ) -> EntropyResult:
        """
        Compute total extractable entropy.

        Parameters
        ----------
        num_bits : int
            Number of reconciled key bits.
        syndrome_leakage : int
            Bits leaked via error correction syndromes.

        Returns
        -------
        EntropyResult
            Complete entropy calculation result.

        Notes
        -----
        Total entropy = h_min × n - leakage
        """
        h_rate, dominant = self.max_bound_entropy_rate()

        # Total entropy available
        total = h_rate * num_bits - syndrome_leakage

        # Clamp to non-negative
        total = max(0.0, total)

        logger.debug(
            f"Total entropy: {total:.2f} bits "
            f"({h_rate:.4f} × {num_bits} - {syndrome_leakage})"
        )

        return EntropyResult(
            entropy_rate=h_rate,
            total_entropy=total,
            dominant_bound=dominant,
            storage_noise_r=self._r,
        )

    @staticmethod
    def crossover_point() -> float:
        """
        Return the crossover point where bounds are equal.

        Returns
        -------
        float
            r ≈ 0.25 where h_DK = h_Lupo.
        """
        return R_CROSSOVER

    @staticmethod
    def gamma(x: float) -> float:
        """
        Compute Γ function for collision entropy regularization.

        Parameters
        ----------
        x : float
            Collision entropy rate.

        Returns
        -------
        float
            Regularized min-entropy rate.
        """
        return gamma_function(x)

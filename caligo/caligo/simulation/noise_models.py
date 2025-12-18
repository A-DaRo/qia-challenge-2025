"""
Custom noise model wrappers for NSM-specific behavior.

This module provides noise model wrappers that combine NSM theoretical
parameters with NetSquid simulation components.

Erven et al. (2014) QBER Model
------------------------------
The total QBER is composed of three contributions:

1. **Source Error (Q_source)**: From imperfect entanglement generation
   Q_source = (1 - F) / 2
   where F is the fidelity of the prepared Bell state.

2. **Detector Error (Q_det)**: Intrinsic measurement error rate
   Q_det = e_det
   This includes polarization misalignment and detector imperfections.

3. **Dark Count Error (Q_dark)**: False positive detections
   Q_dark = (1 - η) × P_dark × 0.5
   where η is detection efficiency and P_dark is dark count probability.

Full QBER: Q_total = Q_source + Q_det + Q_dark

References
----------
- König et al. (2012): Markovian storage noise model
- Schaffner et al. (2009): Depolarizing channel analysis
- Erven et al. (2014): Experimental parameters, Table I
- Wehner et al. (2010): Implementation of two-party protocols in NSM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from caligo.types.exceptions import InvalidParameterError
from caligo.utils.math import (
    binary_entropy,
    compute_qber_erven,
    suggested_ldpc_rate_from_qber,
    blind_reconciliation_initial_config,
)
from caligo.simulation.constants import (
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
)


# =============================================================================
# NSM Storage Noise Model
# =============================================================================


class NSMStorageNoiseModel:
    """
    Wrapper combining depolarization and timing for NSM storage noise.

    Models the adversary's quantum storage as experiencing depolarizing
    noise over the wait time Δt. This is used for theoretical analysis
    and test oracle generation, NOT for actual simulation (where the
    adversary's storage is not simulated).

    Parameters
    ----------
    r : float
        Depolarizing parameter (preservation probability) ∈ [0, 1].
        r=1: Perfect storage (no noise).
        r=0: Complete depolarization (maximum noise).
    delta_t_ns : float
        Wait time in nanoseconds.

    Attributes
    ----------
    r : float
        Depolarizing parameter.
    delta_t_ns : float
        Wait time.

    References
    ----------
    - König et al. (2012): Markovian storage noise model
    - Schaffner et al. (2009): Depolarizing channel analysis
    """

    def __init__(self, r: float, delta_t_ns: float) -> None:
        """
        Initialize the NSM storage noise model.

        Parameters
        ----------
        r : float
            Depolarizing parameter ∈ [0, 1].
        delta_t_ns : float
            Wait time in nanoseconds.

        Raises
        ------
        ValueError
            If parameters are out of range.
        """
        if not 0 <= r <= 1:
            raise ValueError(f"r={r} must be in [0, 1]")
        if delta_t_ns <= 0:
            raise ValueError(f"delta_t_ns={delta_t_ns} must be > 0")

        self._r = r
        self._delta_t_ns = delta_t_ns

    @property
    def r(self) -> float:
        """Get depolarizing parameter (preservation probability)."""
        return self._r

    @property
    def delta_t_ns(self) -> float:
        """Get wait time in nanoseconds."""
        return self._delta_t_ns

    @property
    def depolar_prob(self) -> float:
        """Get depolarization probability = 1 - r."""
        return 1.0 - self._r

    def apply_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Apply depolarizing channel to quantum state.

        For a depolarizing channel with parameter r:
        D_r(ρ) = r·ρ + (1-r)·I/d

        Parameters
        ----------
        state : np.ndarray
            Input density matrix (2x2 for qubit).

        Returns
        -------
        np.ndarray
            Output density matrix after depolarizing noise.

        Notes
        -----
        This is a theoretical model for analysis, not actual simulation.
        """
        if state.shape != (2, 2):
            raise ValueError(f"Expected 2x2 density matrix, got {state.shape}")

        identity = np.eye(2) / 2.0  # Maximally mixed state
        return self._r * state + (1.0 - self._r) * identity

    def get_effective_fidelity(self) -> float:
        """
        Calculate fidelity after noise application.

        For depolarizing channel, if input is pure state |ψ⟩:
        F = r + (1-r)/2 = (1+r)/2

        Returns
        -------
        float
            Expected fidelity of output state.
        """
        return (1.0 + self._r) / 2.0

    def get_min_entropy_bound(self) -> float:
        """
        Calculate adversary's min-entropy bound per qubit.

        Returns
        -------
        float
            Min-entropy bound from NSM analysis.

        Notes
        -----
        For depolarizing storage with noise r, the adversary's
        information is bounded. This returns a simplified bound.
        """
        # Simplified bound: entropy grows with depolarization
        return binary_entropy(self.depolar_prob / 2.0)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NSMStorageNoiseModel(r={self._r}, delta_t_ns={self._delta_t_ns})"


# =============================================================================
# Channel Noise Profile
# =============================================================================


@dataclass(frozen=True)
class ChannelNoiseProfile:
    """
    Aggregate noise profile for the trusted quantum channel.

    Combines multiple noise sources into a single QBER estimate.
    This dataclass provides a convenient way to specify and validate
    channel noise parameters.

    Parameters
    ----------
    source_fidelity : float
        EPR source fidelity (intrinsic pair quality) ∈ (0.5, 1].
    detector_efficiency : float
        Probability of detection given arrival ∈ (0, 1].
    detector_error : float
        Probability of measurement error ∈ [0, 0.5].
    dark_count_rate : float
        Dark count probability per detection window ∈ [0, 1].
    transmission_loss : float
        Probability of photon loss in channel ∈ [0, 1). Default: 0.0.

    Raises
    ------
    InvalidParameterError
        If any invariant is violated.

    Notes
    -----
    Invariants:
    - INV-CNP-001: source_fidelity ∈ (0.5, 1]
    - INV-CNP-002: detector_efficiency ∈ (0, 1]
    - INV-CNP-003: detector_error ∈ [0, 0.5]
    - INV-CNP-004: dark_count_rate ∈ [0, 1]
    - INV-CNP-005: transmission_loss ∈ [0, 1)

    References
    ----------
    - Erven et al. (2014) Table I: Experimental parameters
    """

    source_fidelity: float
    detector_efficiency: float
    detector_error: float
    dark_count_rate: float
    transmission_loss: float = 0.0

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # INV-CNP-001
        if not 0.5 < self.source_fidelity <= 1:
            raise InvalidParameterError(
                f"INV-CNP-001: source_fidelity={self.source_fidelity} "
                "must be in (0.5, 1]"
            )
        # INV-CNP-002
        if not 0 < self.detector_efficiency <= 1:
            raise InvalidParameterError(
                f"INV-CNP-002: detector_efficiency={self.detector_efficiency} "
                "must be in (0, 1]"
            )
        # INV-CNP-003
        if not 0 <= self.detector_error <= 0.5:
            raise InvalidParameterError(
                f"INV-CNP-003: detector_error={self.detector_error} "
                "must be in [0, 0.5]"
            )
        # INV-CNP-004
        if not 0 <= self.dark_count_rate <= 1:
            raise InvalidParameterError(
                f"INV-CNP-004: dark_count_rate={self.dark_count_rate} "
                "must be in [0, 1]"
            )
        # INV-CNP-005
        if not 0 <= self.transmission_loss < 1:
            raise InvalidParameterError(
                f"INV-CNP-005: transmission_loss={self.transmission_loss} "
                "must be in [0, 1)"
            )

    @property
    def total_qber(self) -> float:
        """
        Combined QBER from all noise sources (Erven et al. formula).

        Delegates to shared utility function in caligo.utils.math.

        Returns
        -------
        float
            Total QBER = Q_source + Q_det + Q_dark.

        References
        ----------
        - Erven et al. (2014) Table I and Eq. 8
        """
        return compute_qber_erven(
            fidelity=self.source_fidelity,
            detector_error=self.detector_error,
            detection_efficiency=self.detector_efficiency,
            dark_count_prob=self.dark_count_rate,
        )

    @property
    def qber_conditional(self) -> float:
        """
        Conditional QBER given a detection event occurred.

        This is the QBER conditioned on Bob actually detecting something,
        which is more relevant for post-selected analysis.

        Returns
        -------
        float
            Conditional QBER = E[error | detection].

        Notes
        -----
        Formula:
            P(detect) = η + (1-η) × P_dark
            Q_cond = [η × (Q_source + Q_det) + (1-η) × P_dark × 0.5] / P(detect)
        """
        q_source = (1.0 - self.source_fidelity) / 2.0
        q_det = self.detector_error
        eta = self.detector_efficiency
        p_dark = self.dark_count_rate

        # Probability of any detection event
        p_detect = eta + (1.0 - eta) * p_dark

        if p_detect <= 0:
            return 0.5  # No detection means random guessing

        # Error contribution from real photon detection
        error_from_photon = eta * (q_source + q_det)

        # Error contribution from dark count (random bit)
        error_from_dark = (1.0 - eta) * p_dark * 0.5

        return (error_from_photon + error_from_dark) / p_detect

    @property
    def detection_probability(self) -> float:
        """
        Total probability of a detection event.

        Returns
        -------
        float
            P(detect) = η + (1 - η) × P_dark.
        """
        eta = self.detector_efficiency
        p_dark = self.dark_count_rate
        return eta + (1.0 - eta) * p_dark

    @property
    def signal_to_noise_ratio(self) -> float:
        """
        Ratio of real detections to dark counts.

        Returns
        -------
        float
            SNR = η / [(1 - η) × P_dark].
            Returns inf if dark count contribution is zero.
        """
        eta = self.detector_efficiency
        dark_contribution = (1.0 - eta) * self.dark_count_rate

        if dark_contribution <= 0:
            return float("inf")

        return eta / dark_contribution

    @property
    def is_secure(self) -> bool:
        """
        Check if QBER is below conservative threshold.

        Returns
        -------
        bool
            True if total_qber < 0.11 (Schaffner threshold).
        """
        return self.total_qber < QBER_CONSERVATIVE_LIMIT

    @property
    def is_feasible(self) -> bool:
        """
        Check if QBER is below hard limit.

        Returns
        -------
        bool
            True if total_qber < 0.22 (König hard limit).
        """
        return self.total_qber < QBER_HARD_LIMIT

    @property
    def security_margin(self) -> float:
        """
        Distance from QBER to conservative threshold.

        Returns
        -------
        float
            Margin = 0.11 - total_qber. Positive means secure.
        """
        return QBER_CONSERVATIVE_LIMIT - self.total_qber

    # =========================================================================
    # Reconciliation Support Methods
    # =========================================================================

    def suggested_ldpc_rate(self, safety_margin: float = 0.05) -> float:
        """
        Suggest optimal LDPC code rate based on channel QBER.

        Delegates to shared utility function in caligo.utils.math.

        Parameters
        ----------
        safety_margin : float
            Capacity margin (0-0.1). Default: 0.05.

        Returns
        -------
        float
            Suggested rate R ∈ [0.5, 0.95].
        """
        return suggested_ldpc_rate_from_qber(self.total_qber, safety_margin)

    def blind_reconciliation_config(self) -> dict:
        """
        Generate configuration for blind reconciliation.

        Returns configuration with channel-informed initial rate.

        Returns
        -------
        dict
            Configuration dict with initial_rate, max_iterations, etc.
        """
        config = blind_reconciliation_initial_config(self.total_qber)
        config["expected_qber"] = self.total_qber
        return config

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def perfect(cls) -> ChannelNoiseProfile:
        """
        Create perfect (noiseless) channel profile.

        Returns
        -------
        ChannelNoiseProfile
            Profile with zero errors.
        """
        return cls(
            source_fidelity=1.0,
            detector_efficiency=1.0,
            detector_error=0.0,
            dark_count_rate=0.0,
            transmission_loss=0.0,
        )

    @classmethod
    def from_erven_experimental(cls) -> ChannelNoiseProfile:
        """
        Create profile from Erven et al. (2014) experimental parameters.

        Returns
        -------
        ChannelNoiseProfile
            Profile matching Table I values.
        """
        return cls(
            source_fidelity=1.0 - 3.145e-5,  # From μ (source quality)
            detector_efficiency=0.0150,  # η (transmittance)
            detector_error=0.0093,  # e_det
            dark_count_rate=1.50e-8,  # P_dark
            transmission_loss=1.0 - 0.0150,  # From η
        )

    @classmethod
    def realistic(
        cls,
        source_fidelity: float = 0.98,
        detector_efficiency: float = 0.90,
        detector_error: float = 0.01,
    ) -> ChannelNoiseProfile:
        """
        Create realistic channel profile for development.

        Parameters
        ----------
        source_fidelity : float
            Source fidelity. Default: 0.98.
        detector_efficiency : float
            Detection efficiency. Default: 0.90.
        detector_error : float
            Detector error rate. Default: 0.01.

        Returns
        -------
        ChannelNoiseProfile
            Realistic profile with moderate noise.
        """
        return cls(
            source_fidelity=source_fidelity,
            detector_efficiency=detector_efficiency,
            detector_error=detector_error,
            dark_count_rate=1e-5,
            transmission_loss=0.0,
        )

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_nsm_parameters(
        self,
        storage_noise_r: float = 0.75,
        storage_rate_nu: float = 0.002,
        delta_t_ns: float = 1_000_000,
    ) -> Any:
        """
        Convert to NSMParameters for simulation.

        Parameters
        ----------
        storage_noise_r : float
            NSM storage noise parameter. Default: 0.75.
        storage_rate_nu : float
            NSM storage rate. Default: 0.002.
        delta_t_ns : float
            Wait time in ns. Default: 1_000_000 (1 ms).

        Returns
        -------
        NSMParameters
            Complete NSM parameter set.
        """
        from caligo.simulation.physical_model import NSMParameters

        return NSMParameters(
            storage_noise_r=storage_noise_r,
            storage_rate_nu=storage_rate_nu,
            delta_t_ns=delta_t_ns,
            channel_fidelity=self.source_fidelity,
            detection_eff_eta=self.detector_efficiency,
            detector_error=self.detector_error,
            dark_count_prob=self.dark_count_rate,
        )

    def get_diagnostic_info(self) -> dict:
        """
        Get diagnostic information about the noise profile.

        Returns
        -------
        dict
            Dictionary with all noise parameters and derived values.
        """
        return {
            "source_fidelity": self.source_fidelity,
            "detector_efficiency": self.detector_efficiency,
            "detector_error": self.detector_error,
            "dark_count_rate": self.dark_count_rate,
            "transmission_loss": self.transmission_loss,
            "total_qber": self.total_qber,
            "is_secure": self.is_secure,
            "is_feasible": self.is_feasible,
            "security_margin": self.security_margin,
        }

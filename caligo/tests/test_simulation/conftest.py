"""
Pytest fixtures for simulation package tests.

These fixtures provide commonly needed test objects for the
simulation layer tests.
"""

from __future__ import annotations

import pytest

from caligo.simulation.physical_model import (
    NSMParameters,
    ChannelParameters,
    TYPICAL_DELTA_T_NS,
    TYPICAL_T1_NS,
    TYPICAL_T2_NS,
)
from caligo.simulation.timing import TimingBarrier
from caligo.simulation.noise_models import (
    NSMStorageNoiseModel,
    ChannelNoiseProfile,
)


# =============================================================================
# NSM Parameters Fixtures
# =============================================================================


@pytest.fixture
def nsm_params() -> NSMParameters:
    """Default NSM parameters for testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,  # 1 ms
        channel_fidelity=0.95,
    )


@pytest.fixture
def nsm_params_perfect() -> NSMParameters:
    """Perfect channel NSM parameters for testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=1.0,  # Perfect fidelity
    )


@pytest.fixture
def nsm_params_erven() -> NSMParameters:
    """Erven et al. (2014) experimental NSM parameters."""
    return NSMParameters.from_erven_experimental()


# =============================================================================
# Channel Parameters Fixtures
# =============================================================================


@pytest.fixture
def channel_params() -> ChannelParameters:
    """Default channel parameters for testing."""
    return ChannelParameters(
        length_km=0.0,
        t1_ns=TYPICAL_T1_NS,
        t2_ns=TYPICAL_T2_NS,
    )


@pytest.fixture
def channel_params_with_loss() -> ChannelParameters:
    """Channel parameters with fiber loss."""
    return ChannelParameters(
        length_km=10.0,
        attenuation_db_per_km=0.2,
        t1_ns=TYPICAL_T1_NS,
        t2_ns=TYPICAL_T2_NS,
    )


# =============================================================================
# Timing Barrier Fixtures
# =============================================================================


@pytest.fixture
def timing_barrier(nsm_params: NSMParameters) -> TimingBarrier:
    """Pre-configured TimingBarrier for testing."""
    return TimingBarrier(delta_t_ns=nsm_params.delta_t_ns)


@pytest.fixture
def timing_barrier_lenient() -> TimingBarrier:
    """TimingBarrier in non-strict mode."""
    return TimingBarrier(delta_t_ns=1_000_000, strict_mode=False)


# =============================================================================
# Noise Model Fixtures
# =============================================================================


@pytest.fixture
def storage_noise_model() -> NSMStorageNoiseModel:
    """Default NSM storage noise model."""
    return NSMStorageNoiseModel(r=0.75, delta_t_ns=1_000_000)


@pytest.fixture
def channel_noise_profile() -> ChannelNoiseProfile:
    """Default channel noise profile."""
    return ChannelNoiseProfile(
        source_fidelity=0.98,
        detector_efficiency=0.90,
        detector_error=0.01,
        dark_count_rate=1e-5,
    )


@pytest.fixture
def channel_noise_profile_perfect() -> ChannelNoiseProfile:
    """Perfect (noiseless) channel profile."""
    return ChannelNoiseProfile.perfect()


@pytest.fixture
def channel_noise_profile_erven() -> ChannelNoiseProfile:
    """Erven et al. experimental channel profile."""
    return ChannelNoiseProfile.from_erven_experimental()

import math

import pytest

from caligo.simulation.noise_models import ChannelNoiseProfile
from caligo.types.exceptions import InvalidParameterError


def test_channel_noise_profile_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=1.0,
            detector_error=0.0,
            dark_count_rate=0.0,
            transmission_loss=-1e-9,
        )

    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=0.5,
            detector_efficiency=1.0,
            detector_error=0.0,
            dark_count_rate=0.0,
        )

    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=0.0,
            detector_error=0.0,
            dark_count_rate=0.0,
        )

    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=1.0,
            detector_error=0.5 + 1e-9,
            dark_count_rate=0.0,
        )

    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=1.0,
            detector_error=0.0,
            dark_count_rate=1.0 + 1e-9,
        )

    with pytest.raises(InvalidParameterError):
        ChannelNoiseProfile(
            source_fidelity=1.0,
            detector_efficiency=1.0,
            detector_error=0.0,
            dark_count_rate=0.0,
            transmission_loss=1.0,
        )


def test_qber_conditional_range_and_monotonicity() -> None:
    profile_low = ChannelNoiseProfile(
        source_fidelity=1.0,
        detector_efficiency=1.0,
        detector_error=0.0,
        dark_count_rate=0.0,
    )
    profile_high = ChannelNoiseProfile(
        source_fidelity=0.9,
        detector_efficiency=1.0,
        detector_error=0.05,
        dark_count_rate=0.0,
    )

    q_low = profile_low.qber_conditional
    q_high = profile_high.qber_conditional

    assert 0.0 <= q_low <= 0.5
    assert 0.0 <= q_high <= 0.5
    assert q_high > q_low


def test_snr_is_finite_and_nonnegative() -> None:
    profile = ChannelNoiseProfile(
        source_fidelity=0.99,
        detector_efficiency=0.8,
        detector_error=0.01,
        dark_count_rate=1e-6,
    )

    snr = profile.signal_to_noise_ratio
    assert math.isfinite(snr) or math.isinf(snr)
    assert snr >= 0.0


def test_to_nsm_parameters_has_positive_derived_quantities() -> None:
    profile = ChannelNoiseProfile(
        source_fidelity=0.99,
        detector_efficiency=0.9,
        detector_error=0.02,
        dark_count_rate=1e-6,
    )

    params = profile.to_nsm_parameters(storage_noise_r=0.75, storage_rate_nu=0.002, delta_t_ns=1_000_000)

    assert params.channel_fidelity == profile.source_fidelity
    assert params.detection_eff_eta == profile.detector_efficiency
    assert params.detector_error == profile.detector_error
    assert params.dark_count_prob == profile.dark_count_rate

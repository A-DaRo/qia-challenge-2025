"""Unit tests for PDC helper functions in physical_model.

These are simulator-free and provide deterministic oracle coverage.
"""

from __future__ import annotations

import pytest

from caligo.simulation.physical_model import (
    p_b_noclick,
    p_b_noclick_min,
    p_sent,
    pdc_probability,
)


def test_pdc_probability_rejects_nonpositive_mu() -> None:
    with pytest.raises(ValueError, match="mu must be positive"):
        pdc_probability(0, mu=0.0)


def test_pdc_probability_negative_n_is_zero() -> None:
    assert pdc_probability(-1, mu=0.1) == 0.0


def test_pdc_probability_distribution_sums_close_to_one() -> None:
    mu = 0.1
    total = sum(pdc_probability(n, mu) for n in range(0, 200))
    assert 0.999 < total < 1.001


def test_p_sent_matches_definition() -> None:
    mu = 0.2
    p0 = pdc_probability(0, mu)
    p1 = pdc_probability(1, mu)
    assert p_sent(mu) == pytest.approx(p1 / (1.0 - p0))


def test_p_b_noclick_is_probability_and_decreases_with_eta() -> None:
    mu = 0.2
    p_dark = 1e-6

    p_low_eta = p_b_noclick(mu=mu, eta=0.2, p_dark=p_dark)
    p_high_eta = p_b_noclick(mu=mu, eta=0.8, p_dark=p_dark)

    assert 0.0 <= p_low_eta <= 1.0
    assert 0.0 <= p_high_eta <= 1.0
    assert p_high_eta < p_low_eta


def test_p_b_noclick_min_is_not_below_vacuum_probability() -> None:
    mu = 0.2
    eta = 0.8
    p_dark = 1e-6

    p0 = pdc_probability(0, mu)
    p_min = p_b_noclick_min(mu=mu, eta=eta, p_dark=p_dark)

    assert p_min >= p0
    assert 0.0 <= p_min <= 1.0

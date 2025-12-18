import math

import pytest

from caligo.utils.math import (
    binary_entropy,
    channel_capacity,
    suggested_ldpc_rate_from_qber,
    blind_reconciliation_initial_config,
    finite_size_penalty,
    gamma_function,
    key_length_bound,
)


def test_binary_entropy_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        binary_entropy(-1e-9)
    with pytest.raises(ValueError):
        binary_entropy(1.0 + 1e-9)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_binary_entropy_edges_are_zero(p: float) -> None:
    assert binary_entropy(p) == 0.0


def test_binary_entropy_midpoint_is_one() -> None:
    assert binary_entropy(0.5) == 1.0


def test_channel_capacity_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        channel_capacity(-1e-9)
    with pytest.raises(ValueError):
        channel_capacity(0.5 + 1e-9)


def test_suggested_ldpc_rate_low_qber_returns_highest_rate() -> None:
    rate = suggested_ldpc_rate_from_qber(0.0)
    assert math.isclose(rate, 0.90)


def test_suggested_ldpc_rate_near_hard_limit_returns_lowest_rate() -> None:
    # At ~11% QBER the Shannon capacity is ~0.5, so the best available
    # rate in our discrete set should be 0.50.
    rate = suggested_ldpc_rate_from_qber(0.11)
    assert math.isclose(rate, 0.50)


def test_suggested_ldpc_rate_safety_margin_reduces_rate_or_equal() -> None:
    base = suggested_ldpc_rate_from_qber(0.03, safety_margin=0.0)
    safer = suggested_ldpc_rate_from_qber(0.03, safety_margin=0.05)
    assert safer <= base


@pytest.mark.parametrize(
    "qber,expected",
    [
        (0.0, {"initial_rate": 0.90, "rate_adaptation": "puncturing"}),
        (0.019999, {"initial_rate": 0.90, "rate_adaptation": "puncturing"}),
        (0.02, {"initial_rate": 0.80, "rate_adaptation": "puncturing"}),
        (0.049999, {"initial_rate": 0.80, "rate_adaptation": "puncturing"}),
        (0.05, {"initial_rate": 0.70, "rate_adaptation": "shortening"}),
        (0.079999, {"initial_rate": 0.70, "rate_adaptation": "shortening"}),
        (0.08, {"initial_rate": 0.60, "rate_adaptation": "shortening"}),
    ],
)
def test_blind_reconciliation_initial_config_thresholds(qber: float, expected: dict) -> None:
    assert blind_reconciliation_initial_config(qber) == expected


def test_finite_size_penalty_preconditions() -> None:
    with pytest.raises(ValueError):
        finite_size_penalty(0, 1)
    with pytest.raises(ValueError):
        finite_size_penalty(1, 0)
    with pytest.raises(ValueError):
        finite_size_penalty(1, 1, epsilon_sec=0.0)
    with pytest.raises(ValueError):
        finite_size_penalty(1, 1, epsilon_sec=1.0)


def test_finite_size_penalty_decreases_with_larger_test_set() -> None:
    n = 10_000
    mu_small_k = finite_size_penalty(n=n, k=100, epsilon_sec=1e-10)
    mu_large_k = finite_size_penalty(n=n, k=1000, epsilon_sec=1e-10)
    assert mu_large_k < mu_small_k


def test_gamma_function_preconditions_and_edges() -> None:
    with pytest.raises(ValueError):
        gamma_function(-1e-9)
    with pytest.raises(ValueError):
        gamma_function(1.0 + 1e-9)

    g0 = gamma_function(0.0)
    g1 = gamma_function(1.0)

    assert math.isfinite(g0)
    assert math.isfinite(g1)


def test_key_length_bound_gamma_branches_floor_at_zero() -> None:
    # QKD-style branch (gamma=0.0)
    ell_qkd = key_length_bound(n_sifted=1000, qber=0.49, leakage_bits=10_000, epsilon_sec=1e-10, gamma=0.0)
    assert ell_qkd == 0

    # NSM-style branch (gamma!=0.0)
    ell_nsm = key_length_bound(n_sifted=1000, qber=0.01, leakage_bits=0, epsilon_sec=1e-10, gamma=0.5)
    assert ell_nsm >= 0

"""
Pytest fixtures for security module tests.

Provides reusable test configurations and parameters.
"""

from __future__ import annotations

import pytest

from caligo.security.bounds import (
    QBER_CONSERVATIVE_THRESHOLD,
    QBER_ABSOLUTE_THRESHOLD,
    R_TILDE,
    R_CROSSOVER,
    DEFAULT_EPSILON_SEC,
)
from caligo.security.feasibility import FeasibilityChecker


# =============================================================================
# Parameter Fixtures
# =============================================================================


@pytest.fixture
def erven_experimental_params() -> dict:
    """
    Parameters matching Erven et al. (2014) experiment.

    These are the baseline experimental parameters used for validation.
    """
    return {
        "storage_noise_r": 0.75,
        "storage_rate_nu": 0.002,
        "expected_qber": 0.02,
        "security_parameter": 1e-10,
    }


@pytest.fixture
def high_noise_params() -> dict:
    """Parameters for high storage noise (good for security)."""
    return {
        "storage_noise_r": 0.3,
        "storage_rate_nu": 0.01,
        "expected_qber": 0.05,
        "security_parameter": 1e-10,
    }


@pytest.fixture
def low_noise_params() -> dict:
    """Parameters for low storage noise (challenging for security)."""
    return {
        "storage_noise_r": 0.9,
        "storage_rate_nu": 0.1,
        "expected_qber": 0.03,
        "security_parameter": 1e-10,
    }


@pytest.fixture
def borderline_params() -> dict:
    """Parameters near security thresholds."""
    return {
        "storage_noise_r": 0.75,
        "storage_rate_nu": 0.002,
        "expected_qber": 0.10,  # Close to 11% limit
        "security_parameter": 1e-10,
    }


@pytest.fixture
def infeasible_qber_params() -> dict:
    """Parameters with QBER above hard limit."""
    return {
        "storage_noise_r": 0.75,
        "storage_rate_nu": 0.002,
        "expected_qber": 0.25,  # Above 22% limit
        "security_parameter": 1e-10,
    }


@pytest.fixture
def infeasible_storage_params() -> dict:
    """Parameters violating storage capacity constraint."""
    return {
        "storage_noise_r": 0.99,  # Near-perfect storage
        "storage_rate_nu": 0.6,  # High storage rate
        "expected_qber": 0.02,
        "security_parameter": 1e-10,
    }


# =============================================================================
# Checker Fixtures
# =============================================================================


@pytest.fixture
def erven_checker(erven_experimental_params) -> FeasibilityChecker:
    """FeasibilityChecker with Erven experimental parameters."""
    return FeasibilityChecker(**erven_experimental_params)


@pytest.fixture
def high_noise_checker(high_noise_params) -> FeasibilityChecker:
    """FeasibilityChecker with high storage noise parameters."""
    return FeasibilityChecker(**high_noise_params)


@pytest.fixture
def borderline_checker(borderline_params) -> FeasibilityChecker:
    """FeasibilityChecker with borderline parameters."""
    return FeasibilityChecker(**borderline_params)


# =============================================================================
# Reference Value Fixtures
# =============================================================================


@pytest.fixture
def literature_entropy_values() -> list[tuple[float, float, float]]:
    """
    Reference entropy values for entropy bounds.

    Format: (r, expected_dk_bound, expected_lupo_bound)

    Notes
    -----
    DK bound uses h_A = Γ[1 - log₂(1 + 3r²)] from Lupo et al. (2023).
    Lupo bound uses h_B = 1 - r (virtual erasure).
    """
    return [
        # r, DK bound (Γ[1 - log₂(1 + 3r²)]), Lupo bound (1-r)
        (0.0, 1.0, 1.0),  # Complete depolarization
        (0.1, 0.957, 0.9),  # High noise, DK better
        (0.3, 0.655, 0.7),  # Lupo becomes better
        (0.5, 0.305, 0.5),  # Lupo much better
        (0.7, 0.0, 0.3),  # DK negative → 0, Lupo better
        (0.75, 0.0, 0.25),  # DK ≈ 0, use Lupo
        (0.8, 0.0, 0.2),  # DK negative → 0
        (0.9, 0.0, 0.1),  # Low noise, DK ≈ 0
    ]


@pytest.fixture
def storage_noise_range() -> list[float]:
    """Standard range of r values for testing."""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.9, 1.0]


@pytest.fixture
def qber_range() -> list[float]:
    """Standard range of QBER values for testing."""
    return [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.11, 0.15, 0.20, 0.22, 0.25]


# =============================================================================
# Finite-Key Fixtures
# =============================================================================


@pytest.fixture
def typical_batch_sizes() -> list[int]:
    """Typical batch sizes for finite-key tests."""
    return [1000, 10_000, 100_000, 1_000_000]


@pytest.fixture
def test_fractions() -> list[float]:
    """Test sample fractions for optimization tests."""
    return [0.01, 0.05, 0.10, 0.20, 0.50]

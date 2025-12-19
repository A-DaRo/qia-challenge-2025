"""
Unit tests for rate selector module.

Tests rate selection logic, efficiency criterion, and shortening computation.
"""

from __future__ import annotations

import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.rate_selector import (
    binary_entropy,
    select_rate,
    compute_shortening,
    compute_puncturing,
    select_rate_with_parameters,
    RateSelection,
)


class TestBinaryEntropy:
    """Tests for binary entropy function."""

    def test_entropy_zero(self) -> None:
        """h(0) = 0."""
        assert binary_entropy(0.0) == 0.0

    def test_entropy_one(self) -> None:
        """h(1) = 0."""
        assert binary_entropy(1.0) == 0.0

    def test_entropy_half(self) -> None:
        """h(0.5) = 1.0."""
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

    def test_entropy_symmetric(self) -> None:
        """h(p) = h(1-p)."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(binary_entropy(p) - binary_entropy(1 - p)) < 1e-10

    def test_entropy_typical_qber(self) -> None:
        """Entropy for typical QBER values."""
        # QBER 5% → h(0.05) ≈ 0.286
        h_05 = binary_entropy(0.05)
        assert 0.28 < h_05 < 0.30

        # QBER 11% → h(0.11) ≈ 0.50
        h_11 = binary_entropy(0.11)
        assert 0.49 < h_11 < 0.51


class TestRateSelection:
    """Tests for rate selection logic.
    
    Note: Due to BP decoder limitations with current LDPC matrices, rate 0.5
    is always selected for reliability (~90% success rate vs ~50% for higher rates).
    """

    def test_low_qber_reliable_rate(self) -> None:
        """Low QBER should select most reliable rate (0.5).
        
        Note: Higher rates (0.6, 0.7, 0.85, 0.9) have poor BP convergence
        and are not used even for low QBER.
        """
        rate = select_rate(0.01, constants.LDPC_CODE_RATES)
        assert rate == 0.5, f"Expected 0.5 for reliability, got {rate}"

    def test_moderate_qber_reliable_rate(self) -> None:
        """Moderate QBER (5%) selects most reliable rate."""
        rate = select_rate(0.05, constants.LDPC_CODE_RATES)
        # Rate 0.5 is the most reliable rate
        assert rate == 0.5

    def test_high_qber_low_rate(self) -> None:
        """High QBER (10%) selects low rate."""
        rate = select_rate(0.10, constants.LDPC_CODE_RATES)
        assert rate <= 0.55

    def test_reliability_over_efficiency(self) -> None:
        """Rate selector prioritizes decoder reliability over efficiency.
        
        Due to BP decoder limitations with available LDPC matrices, we always
        use rate 0.5 which has the highest decoder reliability (~90% success rate).
        Higher rates (0.6, 0.7) have lower success rates (~50-55%) and are not
        used even for low QBER.
        """
        # All QBER values should select rate 0.5 for reliability
        for qber in [0.01, 0.03, 0.05, 0.08, 0.10]:
            rate = select_rate(qber, constants.LDPC_CODE_RATES)
            assert rate == 0.5, f"Expected rate 0.5 at QBER {qber} for reliability, got {rate}"

    @pytest.mark.parametrize("qber,expected_rate", [
        (0.01, 0.50),  # Low QBER: use most reliable rate
        (0.03, 0.50),  # Low QBER: use most reliable rate
        (0.06, 0.50),  # Medium: use most reliable rate
        (0.09, 0.50),  # High: use most reliable rate
    ])
    def test_qber_rate_mapping(self, qber: float, expected_rate: float) -> None:
        """Verify rate selection always returns most reliable rate."""
        rate = select_rate(qber, constants.LDPC_CODE_RATES)
        assert rate == expected_rate, f"Expected {expected_rate} at QBER {qber}, got {rate}"


class TestShortening:
    """Tests for shortening computation."""

    def test_no_shortening_low_qber(self) -> None:
        """Low QBER with matching payload needs minimal shortening."""
        n_s = compute_shortening(
            rate=0.90,
            qber_estimate=0.01,
            payload_length=3686,  # ~90% of 4096
            frame_size=4096,
        )
        # Shortening should be reasonable
        assert 0 <= n_s <= 1000

    def test_shortening_fits_frame(self) -> None:
        """payload + shortening fits within frame."""
        for payload in [1000, 2000, 3000]:
            n_s = compute_shortening(
                rate=0.70,
                qber_estimate=0.05,
                payload_length=payload,
                frame_size=4096,
            )
            assert payload + n_s <= 4096


class TestPuncturing:
    """Tests for puncturing computation."""

    def test_no_puncturing_same_rate(self) -> None:
        """No puncturing when target equals base rate."""
        n_p = compute_puncturing(0.70, 0.70, 4096)
        assert n_p == 0

    def test_no_puncturing_lower_target(self) -> None:
        """No puncturing when target < base."""
        n_p = compute_puncturing(0.70, 0.60, 4096)
        assert n_p == 0

    def test_puncturing_higher_target(self) -> None:
        """Positive puncturing when target > base."""
        n_p = compute_puncturing(0.70, 0.80, 4096)
        assert n_p > 0


class TestRateSelectionWithParameters:
    """Tests for combined rate selection."""

    def test_returns_rate_selection(self) -> None:
        """Returns RateSelection dataclass."""
        result = select_rate_with_parameters(
            qber_estimate=0.05,
            payload_length=2000,
            frame_size=4096,
        )
        assert isinstance(result, RateSelection)

    def test_includes_syndrome_length(self) -> None:
        """Syndrome length computed correctly."""
        result = select_rate_with_parameters(
            qber_estimate=0.05,
            payload_length=2000,
            frame_size=4096,
        )
        expected_syndrome = int(4096 * (1 - result.rate))
        assert result.syndrome_length == expected_syndrome

    def test_efficiency_computed(self) -> None:
        """Expected efficiency is calculated."""
        result = select_rate_with_parameters(
            qber_estimate=0.05,
            payload_length=2000,
            frame_size=4096,
        )
        assert result.expected_efficiency > 0
        assert result.expected_efficiency < 2.0  # Reasonable bound

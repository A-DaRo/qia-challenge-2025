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
    """Tests for rate selection using efficiency model.
    
    Rate selection follows R = 1 - f_crit × h(QBER) as per Elkouss et al. (2010)
    and Martinez-Mateo et al. (2012). Assumes correct decoder implementation.
    """

    def test_low_qber_high_rate(self) -> None:
        """Low QBER should select high rate via efficiency model.
        
        For QBER=0.01, h(0.01) ≈ 0.081, with f_crit=1.1:
        R = 1 - 1.1 × 0.081 ≈ 0.91 → quantized to nearest available rate
        """
        rate = select_rate(0.01, constants.LDPC_CODE_RATES, f_crit=1.1)
        # Should select high rate (expect 0.85 or 0.9)
        assert rate >= 0.85, f"Expected high rate for low QBER, got {rate}"

    def test_moderate_qber_medium_rate(self) -> None:
        """Moderate QBER (5%) selects medium rate.
        
        For QBER=0.05, h(0.05) ≈ 0.286, with f_crit=1.1:
        R = 1 - 1.1 × 0.286 ≈ 0.69 → quantized to 0.70
        """
        rate = select_rate(0.05, constants.LDPC_CODE_RATES, f_crit=1.1)
        # Should be in medium range [0.65, 0.75]
        assert 0.65 <= rate <= 0.75, f"Expected medium rate for QBER 0.05, got {rate}"

    def test_high_qber_low_rate(self) -> None:
        """High QBER (10%) selects low rate.
        
        For QBER=0.10, h(0.10) ≈ 0.469, with f_crit=1.1:
        R = 1 - 1.1 × 0.469 ≈ 0.48 → quantized to 0.50
        """
        rate = select_rate(0.10, constants.LDPC_CODE_RATES, f_crit=1.1)
        assert rate <= 0.55, f"Expected low rate for high QBER, got {rate}"

    def test_efficiency_criterion_scaling(self) -> None:
        """Rate scales with efficiency criterion.
        
        Higher f_crit means more leakage tolerance → lower rate selected.
        """
        qber = 0.05
        rate_strict = select_rate(qber, constants.LDPC_CODE_RATES, f_crit=1.0)
        rate_relaxed = select_rate(qber, constants.LDPC_CODE_RATES, f_crit=1.2)
        
        # Relaxed efficiency allows lower rate (more error correction)
        assert rate_relaxed <= rate_strict, \
            f"Relaxed f_crit should give lower/equal rate: {rate_relaxed} vs {rate_strict}"

    @pytest.mark.parametrize("qber,f_crit,expected_min,expected_max", [
        (0.01, 1.1, 0.85, 0.90),  # Low QBER: high rate
        (0.05, 1.1, 0.65, 0.75),  # Moderate QBER: medium rate
        (0.10, 1.1, 0.50, 0.55),  # High QBER: low rate
    ])
    def test_qber_rate_mapping(
        self, qber: float, f_crit: float, expected_min: float, expected_max: float
    ) -> None:
        """Verify rate selection follows efficiency model."""
        rate = select_rate(qber, constants.LDPC_CODE_RATES, f_crit=f_crit)
        assert expected_min <= rate <= expected_max, \
            f"Rate {rate} outside expected range [{expected_min}, {expected_max}] for QBER {qber}"
    
    def test_edge_case_zero_qber(self) -> None:
        """Zero QBER returns highest available rate."""
        rate = select_rate(0.0, constants.LDPC_CODE_RATES)
        assert rate == max(constants.LDPC_CODE_RATES)
    
    def test_edge_case_maximum_qber(self) -> None:
        """QBER at 0.5 returns lowest available rate."""
        rate = select_rate(0.5, constants.LDPC_CODE_RATES)
        assert rate == min(constants.LDPC_CODE_RATES)


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
        """Syndrome length computed correctly from mother code rate.
        
        Critical: Syndrome length uses R_0 (mother rate), not R_eff.
        For R_0 = 0.5, syndrome_length = (1 - 0.5) × n = n/2 = 2048 bits.
        """
        result = select_rate_with_parameters(
            qber_estimate=0.05,
            payload_length=2000,
            frame_size=4096,
        )
        # Syndrome length is ALWAYS (1 - R_0) × n for mother code R_0 = 0.5
        expected_syndrome = int(4096 * (1 - 0.5))  # = 2048 bits (constant)
        assert result.syndrome_length == expected_syndrome, \
            f"Syndrome length must be constant 2048 for R_0=0.5, got {result.syndrome_length}"

    def test_efficiency_computed(self) -> None:
        """Expected efficiency is calculated."""
        result = select_rate_with_parameters(
            qber_estimate=0.05,
            payload_length=2000,
            frame_size=4096,
        )
        assert result.expected_efficiency > 0
        assert result.expected_efficiency < 2.0  # Reasonable bound

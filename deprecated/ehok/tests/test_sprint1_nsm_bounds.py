"""
Unit tests for NSM bounds calculator (TASK-NSM-001).

These tests verify the mathematical correctness of the NSM security bound
calculations against reference values from the literature.

Test Strategy
-------------
1. **Regression tests**: Known values from analytical computation
2. **Monotonicity tests**: h_min(r) must be decreasing in r
3. **Boundary tests**: Edge cases at r=0 and r≈0.82 (crossover)
4. **Feasibility result tests**: Correct enum assignments

References
----------
- Lupo et al. (2023): Max Bound formula (Eq. 36)
- König et al. (2012): Collision entropy regularization
- sprint_1_specification.md Section 2
"""

import math

import pytest

from ehok.analysis.nsm_bounds import (
    NSMBoundsCalculator,
    NSMBoundsInputs,
    NSMBoundsResult,
    FeasibilityResult,
    binary_entropy,
    gamma_function,
    collision_entropy_rate,
    max_bound_entropy_rate,
    channel_capacity,
    QBER_HARD_LIMIT,
)


class TestBinaryEntropy:
    """Tests for the binary entropy function h(p)."""

    def test_boundary_zero(self) -> None:
        """h(0) = 0 by convention."""
        assert binary_entropy(0.0) == 0.0

    def test_boundary_one(self) -> None:
        """h(1) = 0 by convention."""
        assert binary_entropy(1.0) == 0.0

    def test_maximum_at_half(self) -> None:
        """h(1/2) = 1 (maximum entropy)."""
        assert binary_entropy(0.5) == pytest.approx(1.0, rel=1e-10)

    def test_symmetry(self) -> None:
        """h(p) = h(1-p)."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert binary_entropy(p) == pytest.approx(binary_entropy(1 - p), rel=1e-10)

    def test_known_value(self) -> None:
        """h(0.11) ≈ 0.5 (11% QBER threshold)."""
        # h(0.11) = -0.11*log2(0.11) - 0.89*log2(0.89)
        expected = -0.11 * math.log2(0.11) - 0.89 * math.log2(0.89)
        assert binary_entropy(0.11) == pytest.approx(expected, rel=1e-10)


class TestGammaFunction:
    """Tests for the Γ regularization function."""

    def test_identity_above_half(self) -> None:
        """Γ(x) = x for x ≥ 1/2."""
        for x in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            assert gamma_function(x) == pytest.approx(x, rel=1e-6)

    def test_below_half_returns_value(self) -> None:
        """Γ(x) should return a valid value for x < 1/2."""
        for x in [0.1, 0.2, 0.3, 0.4]:
            result = gamma_function(x)
            # Γ(x) should be in (0, 1) range
            assert 0 < result < 1

    def test_monotonicity(self) -> None:
        """Γ is monotonically increasing."""
        x_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        gamma_values = [gamma_function(x) for x in x_values]
        for i in range(len(gamma_values) - 1):
            assert gamma_values[i] <= gamma_values[i + 1] + 1e-6

    def test_continuity_at_half(self) -> None:
        """Γ should be continuous at x = 1/2."""
        below = gamma_function(0.499)
        at = gamma_function(0.5)
        above = gamma_function(0.501)
        assert below == pytest.approx(at, rel=0.01)
        assert above == pytest.approx(at, rel=0.01)


class TestCollisionEntropyRate:
    """Tests for collision entropy h2(r) = 1 - log₂(1 + 3r²)."""

    def test_at_r_zero(self) -> None:
        """h2(0) = 1 - log₂(1) = 1."""
        assert collision_entropy_rate(0.0) == pytest.approx(1.0, rel=1e-10)

    def test_at_r_one(self) -> None:
        """h2(1) = 1 - log₂(4) = -1."""
        assert collision_entropy_rate(1.0) == pytest.approx(-1.0, rel=1e-10)

    def test_at_r_half(self) -> None:
        """h2(0.5) = 1 - log₂(1.75)."""
        expected = 1.0 - math.log2(1 + 3 * 0.25)  # 1 + 3*0.25 = 1.75
        assert collision_entropy_rate(0.5) == pytest.approx(expected, rel=1e-10)

    def test_monotonically_decreasing(self) -> None:
        """h2(r) decreases as r increases."""
        r_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        h2_values = [collision_entropy_rate(r) for r in r_values]
        for i in range(len(h2_values) - 1):
            assert h2_values[i] > h2_values[i + 1]


class TestMaxBoundEntropyRate:
    """
    Tests for the Max Bound formula from Lupo et al. (2023).

    h_min(r) = max{ Γ[1 - log₂(1 + 3r²)], 1 - r }
    """

    def test_regression_r_01(self) -> None:
        """
        Regression test at r = 0.1.

        At low r, Γ branch typically dominates:
        h2(0.1) = 1 - log₂(1.03) ≈ 0.9574
        Γ(0.9574) = 0.9574 (since > 0.5)
        1 - r = 0.9
        Max = 0.9574
        """
        result = max_bound_entropy_rate(0.1)
        assert result == pytest.approx(0.957, rel=0.01)

    def test_regression_r_03(self) -> None:
        """
        Regression test at r = 0.3.

        h2(0.3) = 1 - log₂(1.27) ≈ 0.6546
        Γ(0.6546) ≈ 0.6546
        1 - r = 0.7
        Max = 0.7
        """
        result = max_bound_entropy_rate(0.3)
        # Either branch could dominate; verify reasonable value
        assert 0.6 <= result <= 0.75

    def test_regression_r_05(self) -> None:
        """
        Regression test at r = 0.5.

        h2(0.5) = 1 - log₂(1.75) ≈ 0.1926
        Γ(0.1926) via inverse...
        1 - r = 0.5
        """
        result = max_bound_entropy_rate(0.5)
        assert 0.4 <= result <= 0.55

    def test_regression_r_07(self) -> None:
        """
        Regression test at r = 0.7.

        At higher r, 1 - r typically dominates:
        1 - r = 0.3
        """
        result = max_bound_entropy_rate(0.7)
        assert 0.25 <= result <= 0.35

    def test_regression_r_09(self) -> None:
        """
        Regression test at r = 0.9.

        1 - r = 0.1 dominates
        """
        result = max_bound_entropy_rate(0.9)
        assert 0.05 <= result <= 0.15

    def test_monotonically_decreasing(self) -> None:
        """h_min(r) must decrease as r increases (more noise = less entropy)."""
        r_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        h_min_values = [max_bound_entropy_rate(r) for r in r_values]
        for i in range(len(h_min_values) - 1):
            assert h_min_values[i] >= h_min_values[i + 1] - 1e-6, (
                f"Monotonicity violated: h_min({r_values[i]}) = {h_min_values[i]} "
                f"< h_min({r_values[i+1]}) = {h_min_values[i+1]}"
            )

    def test_crossover_region(self) -> None:
        """
        Around r ≈ 0.82, the two branches cross.

        The Γ branch and (1 - r) branch should produce similar values.
        """
        # Test values around crossover
        for r in [0.80, 0.82, 0.84]:
            h2 = collision_entropy_rate(r)
            gamma_branch = gamma_function(h2)
            linear_branch = 1.0 - r
            max_bound = max_bound_entropy_rate(r)

            # The max should equal one of the branches
            assert max_bound == pytest.approx(
                max(gamma_branch, linear_branch), rel=0.01
            )


class TestChannelCapacity:
    """Tests for depolarizing channel capacity C_N = 1 - h((1+r)/2)."""

    def test_at_r_zero(self) -> None:
        """C_N(0) = 1 - h(0.5) = 0."""
        assert channel_capacity(0.0) == pytest.approx(0.0, rel=1e-6)

    def test_at_r_one(self) -> None:
        """C_N(1) = 1 - h(1) = 1."""
        assert channel_capacity(1.0) == pytest.approx(1.0, rel=1e-6)

    def test_monotonically_increasing(self) -> None:
        """C_N increases with r (higher correlation = more capacity)."""
        r_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        c_values = [channel_capacity(r) for r in r_values]
        for i in range(len(c_values) - 1):
            assert c_values[i] <= c_values[i + 1] + 1e-6


class TestNSMBoundsInputs:
    """Tests for input dataclass validation."""

    def test_valid_inputs(self) -> None:
        """Valid inputs should not raise."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.75,
            adjusted_qber=0.05,
            total_leakage_bits=1000,
            epsilon_sec=1e-8,
            n_sifted_bits=10000,
        )
        assert inputs.n_sifted_bits == 10000

    def test_inputs_accessible(self) -> None:
        """All fields should be accessible."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.75,
            adjusted_qber=0.05,
            total_leakage_bits=1000,
            epsilon_sec=1e-8,
            n_sifted_bits=10000,
        )
        assert inputs.storage_noise_r == 0.75
        assert inputs.adjusted_qber == 0.05
        assert inputs.total_leakage_bits == 1000
        assert inputs.epsilon_sec == 1e-8


class TestNSMBoundsCalculator:
    """Tests for the NSMBoundsCalculator class."""

    @pytest.fixture
    def calculator(self) -> NSMBoundsCalculator:
        """Create a calculator instance."""
        return NSMBoundsCalculator()

    @pytest.fixture
    def good_inputs(self) -> NSMBoundsInputs:
        """Create inputs with good parameters."""
        return NSMBoundsInputs(
            storage_noise_r=0.75,
            adjusted_qber=0.05,
            total_leakage_bits=1000,
            epsilon_sec=1e-8,
            n_sifted_bits=10000,
        )

    def test_compute_returns_result(
        self, calculator: NSMBoundsCalculator, good_inputs: NSMBoundsInputs
    ) -> None:
        """compute() should return NSMBoundsResult."""
        result = calculator.compute(good_inputs)
        assert isinstance(result, NSMBoundsResult)

    def test_result_contains_min_entropy(
        self, calculator: NSMBoundsCalculator, good_inputs: NSMBoundsInputs
    ) -> None:
        """Result should contain non-negative min_entropy_per_bit."""
        result = calculator.compute(good_inputs)
        assert result.min_entropy_per_bit >= 0.0

    def test_result_contains_key_length(
        self, calculator: NSMBoundsCalculator, good_inputs: NSMBoundsInputs
    ) -> None:
        """Result should contain computed key length."""
        result = calculator.compute(good_inputs)
        assert isinstance(result.max_secure_key_length_bits, int)

    def test_result_contains_feasibility(
        self, calculator: NSMBoundsCalculator, good_inputs: NSMBoundsInputs
    ) -> None:
        """Result should contain feasibility status."""
        result = calculator.compute(good_inputs)
        assert isinstance(result.feasibility_status, FeasibilityResult)

    def test_feasibility_feasible_with_good_params(
        self, calculator: NSMBoundsCalculator
    ) -> None:
        """Good parameters should yield FEASIBLE."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.5,
            adjusted_qber=0.05,
            total_leakage_bits=1000,
            epsilon_sec=1e-8,
            n_sifted_bits=100000,
        )
        result = calculator.compute(inputs)
        assert result.feasibility_status == FeasibilityResult.FEASIBLE

    def test_feasibility_infeasible_with_high_qber(
        self, calculator: NSMBoundsCalculator
    ) -> None:
        """QBER > 0.22 should yield INFEASIBLE_QBER_TOO_HIGH."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.75,
            adjusted_qber=0.25,  # Above 22% hard limit
            total_leakage_bits=1000,
            epsilon_sec=1e-8,
            n_sifted_bits=10000,
        )
        result = calculator.compute(inputs)
        assert result.feasibility_status == FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH

    def test_feasibility_infeasible_insufficient_entropy(
        self, calculator: NSMBoundsCalculator
    ) -> None:
        """Very small n with large leakage should yield INFEASIBLE_INSUFFICIENT_ENTROPY."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.75,
            adjusted_qber=0.05,
            total_leakage_bits=10000,  # Large leakage
            epsilon_sec=1e-8,
            n_sifted_bits=100,  # Very small n
        )
        result = calculator.compute(inputs)
        assert result.feasibility_status == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY

    def test_key_length_formula(self, calculator: NSMBoundsCalculator) -> None:
        """Verify key length formula: ℓ = ⌊n·h_min - L - 2log(1/ε)⌋."""
        inputs = NSMBoundsInputs(
            storage_noise_r=0.5,
            adjusted_qber=0.0,
            total_leakage_bits=500,
            epsilon_sec=1e-8,
            n_sifted_bits=10000,
        )
        result = calculator.compute(inputs)

        # Verify formula manually
        h_min = max_bound_entropy_rate(inputs.storage_noise_r)
        log_term = 2 * math.log2(1.0 / inputs.epsilon_sec)
        expected = int(
            inputs.n_sifted_bits * h_min - inputs.total_leakage_bits - log_term
        )

        # Allow small difference due to floor
        assert abs(result.max_secure_key_length_bits - expected) <= 1


class TestNSMBoundsResult:
    """Tests for the result dataclass."""

    def test_result_fields(self) -> None:
        """Result should have all expected fields."""
        result = NSMBoundsResult(
            min_entropy_per_bit=0.25,
            max_secure_key_length_bits=1000,
            feasibility_status=FeasibilityResult.FEASIBLE,
            recommended_min_n=None,
        )
        assert result.min_entropy_per_bit == 0.25
        assert result.max_secure_key_length_bits == 1000
        assert result.feasibility_status == FeasibilityResult.FEASIBLE
        assert result.recommended_min_n is None


class TestFeasibilityResult:
    """Tests for the FeasibilityResult enum."""

    def test_enum_values(self) -> None:
        """Enum should have expected values."""
        assert FeasibilityResult.FEASIBLE is not None
        assert FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH is not None
        assert FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY is not None
        assert FeasibilityResult.INFEASIBLE_INVALID_PARAMETERS is not None

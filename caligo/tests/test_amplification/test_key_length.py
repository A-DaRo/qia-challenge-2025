"""
Unit tests for SecureKeyLengthCalculator.
"""

import math
import pytest

from caligo.amplification.key_length import (
    SecureKeyLengthCalculator,
    KeyLengthResult,
    DEFAULT_EPSILON_SEC,
)
from caligo.amplification.entropy import NSMEntropyCalculator
from caligo.types.exceptions import InvalidParameterError


class TestSecureKeyLengthCalculator:
    """Tests for SecureKeyLengthCalculator class."""

    @pytest.fixture
    def entropy_calc(self):
        """Entropy calculator fixture."""
        return NSMEntropyCalculator(storage_noise_r=0.5)

    @pytest.fixture
    def key_calc(self, entropy_calc):
        """Key length calculator fixture."""
        return SecureKeyLengthCalculator(entropy_calc, epsilon_sec=1e-10)

    def test_compute_final_length_positive(self, key_calc):
        """Positive key length for large input."""
        length = key_calc.compute_final_length(
            reconciled_length=10000,
            syndrome_leakage=500,
        )
        
        assert length > 0
        assert isinstance(length, int)

    def test_compute_final_length_death_valley(self, key_calc):
        """Death Valley: zero key length for small input."""
        length = key_calc.compute_final_length(
            reconciled_length=100,
            syndrome_leakage=50,
        )
        
        # Small input + leakage = Death Valley
        assert length == 0

    def test_compute_final_length_monotonic(self, key_calc):
        """Key length increases with input length."""
        lengths = []
        for n in [1000, 2000, 4000, 8000]:
            length = key_calc.compute_final_length(
                reconciled_length=n,
                syndrome_leakage=int(0.1 * n),
            )
            lengths.append(length)
        
        # Should be monotonically increasing
        assert lengths == sorted(lengths)

    def test_compute_final_length_leakage_reduces(self, key_calc):
        """More leakage reduces key length."""
        l1 = key_calc.compute_final_length(
            reconciled_length=5000,
            syndrome_leakage=100,
        )
        l2 = key_calc.compute_final_length(
            reconciled_length=5000,
            syndrome_leakage=500,
        )
        
        assert l2 < l1

    def test_compute_detailed(self, key_calc):
        """Detailed computation returns full breakdown."""
        result = key_calc.compute_detailed(
            reconciled_length=10000,
            syndrome_leakage=500,
        )
        
        assert isinstance(result, KeyLengthResult)
        assert result.raw_length == 10000
        assert result.syndrome_leakage == 500
        assert result.entropy_available > 0
        assert result.security_penalty > 0

    def test_compute_detailed_efficiency(self, key_calc):
        """Efficiency calculation."""
        result = key_calc.compute_detailed(
            reconciled_length=10000,
            syndrome_leakage=500,
        )
        
        expected_efficiency = result.final_length / 10000
        assert result.efficiency == pytest.approx(expected_efficiency)

    def test_compute_detailed_is_viable(self, key_calc):
        """is_viable flag correctness."""
        good = key_calc.compute_detailed(
            reconciled_length=10000,
            syndrome_leakage=500,
        )
        assert good.is_viable is True
        
        bad = key_calc.compute_detailed(
            reconciled_length=100,
            syndrome_leakage=90,
        )
        assert bad.is_viable is False

    def test_invalid_reconciled_length_raises(self, key_calc):
        """Zero/negative reconciled length raises."""
        with pytest.raises(InvalidParameterError):
            key_calc.compute_final_length(
                reconciled_length=0,
                syndrome_leakage=0,
            )

    def test_negative_leakage_raises(self, key_calc):
        """Negative leakage raises."""
        with pytest.raises(InvalidParameterError):
            key_calc.compute_final_length(
                reconciled_length=1000,
                syndrome_leakage=-10,
            )

    def test_minimum_input_length(self, key_calc):
        """Minimum input length calculation."""
        min_n = key_calc.minimum_input_length(
            target_key_length=128,
            expected_leakage_rate=0.1,
        )
        
        assert min_n > 128
        
        # Verify it actually works
        length = key_calc.compute_final_length(
            reconciled_length=min_n,
            syndrome_leakage=int(0.1 * min_n),
        )
        assert length >= 128

    def test_minimum_input_length_zero_entropy(self):
        """Minimum input for zero effective rate."""
        # Very noisy storage means no extractable entropy
        entropy_calc = NSMEntropyCalculator(storage_noise_r=0.95)
        key_calc = SecureKeyLengthCalculator(entropy_calc)
        
        min_n = key_calc.minimum_input_length(
            target_key_length=128,
            expected_leakage_rate=0.2,  # Exceeds entropy rate
        )
        
        # Should return very large number
        assert min_n > 1e6

    def test_security_penalty_property(self, key_calc):
        """Security penalty property."""
        penalty = key_calc.security_penalty
        
        # 2·log₂(1/ε_sec) - 2 for ε_sec = 1e-10
        expected = 2 * math.log2(1 / 1e-10) - 2
        assert penalty == pytest.approx(expected)

    def test_epsilon_sec_property(self, key_calc):
        """Epsilon sec property."""
        assert key_calc.epsilon_sec == 1e-10

    def test_invalid_epsilon_raises(self):
        """Invalid epsilon raises."""
        entropy_calc = NSMEntropyCalculator()
        
        with pytest.raises(InvalidParameterError):
            SecureKeyLengthCalculator(entropy_calc, epsilon_sec=0)
        
        with pytest.raises(InvalidParameterError):
            SecureKeyLengthCalculator(entropy_calc, epsilon_sec=1.5)


class TestKeyLengthResult:
    """Tests for KeyLengthResult dataclass."""

    def test_key_length_result_creation(self):
        """Create KeyLengthResult."""
        result = KeyLengthResult(
            final_length=500,
            raw_length=1000,
            entropy_available=600.0,
            entropy_consumed=566.5,
            security_penalty=66.5,
            syndrome_leakage=100,
            is_viable=True,
            efficiency=0.5,
        )
        
        assert result.final_length == 500
        assert result.is_viable is True

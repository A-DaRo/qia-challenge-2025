"""
Unit tests for NSMEntropyCalculator.
"""

import pytest

from caligo.amplification.entropy import NSMEntropyCalculator, EntropyResult
from caligo.types.exceptions import InvalidParameterError
from caligo.security.bounds import R_CROSSOVER


class TestNSMEntropyCalculator:
    """Tests for NSMEntropyCalculator class."""

    def test_init_default(self):
        """Default initialization."""
        calc = NSMEntropyCalculator()
        assert calc.storage_noise_r == 0.75

    def test_init_custom_r(self):
        """Custom r initialization."""
        calc = NSMEntropyCalculator(storage_noise_r=0.5)
        assert calc.storage_noise_r == 0.5

    def test_init_invalid_r_raises(self):
        """Invalid r raises error."""
        with pytest.raises(InvalidParameterError):
            NSMEntropyCalculator(storage_noise_r=-0.1)
        
        with pytest.raises(InvalidParameterError):
            NSMEntropyCalculator(storage_noise_r=1.5)

    def test_set_storage_noise_r(self):
        """Setting r property."""
        calc = NSMEntropyCalculator()
        calc.storage_noise_r = 0.3
        assert calc.storage_noise_r == 0.3

    def test_set_invalid_r_raises(self):
        """Setting invalid r raises."""
        calc = NSMEntropyCalculator()
        with pytest.raises(InvalidParameterError):
            calc.storage_noise_r = -0.1

    @pytest.mark.parametrize("r", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_max_bound_entropy_rate_in_range(self, r):
        """Entropy rate is in [0, 1]."""
        calc = NSMEntropyCalculator(storage_noise_r=r)
        h_min, _ = calc.max_bound_entropy_rate()
        assert 0.0 <= h_min <= 1.0

    def test_max_bound_at_r_zero(self):
        """At r=0, entropy rate should be 1."""
        calc = NSMEntropyCalculator(storage_noise_r=0.0)
        h_min, _ = calc.max_bound_entropy_rate()
        assert h_min == pytest.approx(1.0)

    def test_max_bound_at_r_one(self):
        """At r=1, entropy rate should be 0."""
        calc = NSMEntropyCalculator(storage_noise_r=1.0)
        h_min, _ = calc.max_bound_entropy_rate()
        assert h_min == pytest.approx(0.0)

    def test_dominant_bound_low_r(self):
        """Dupuis-König dominates for low r."""
        calc = NSMEntropyCalculator(storage_noise_r=0.1)
        _, dominant = calc.max_bound_entropy_rate()
        assert dominant == "dupuis_konig"

    def test_dominant_bound_high_r(self):
        """Lupo dominates for high r."""
        calc = NSMEntropyCalculator(storage_noise_r=0.9)
        _, dominant = calc.max_bound_entropy_rate()
        assert dominant == "lupo"

    def test_dupuis_konig_bound(self):
        """Dupuis-König bound calculation."""
        calc = NSMEntropyCalculator(storage_noise_r=0.5)
        h_dk = calc.dupuis_konig_bound()
        assert 0.0 <= h_dk <= 1.0

    def test_virtual_erasure_bound(self):
        """Virtual erasure bound calculation."""
        calc = NSMEntropyCalculator(storage_noise_r=0.5)
        h_lupo = calc.virtual_erasure_bound()
        
        # h_Lupo = 1 - r = 0.5
        assert h_lupo == pytest.approx(0.5)

    def test_compute_total_entropy_no_leakage(self):
        """Total entropy without leakage."""
        calc = NSMEntropyCalculator(storage_noise_r=0.5)
        h_rate, _ = calc.max_bound_entropy_rate()
        
        result = calc.compute_total_entropy(num_bits=1000, syndrome_leakage=0)
        
        assert result.entropy_rate == h_rate
        assert result.total_entropy == pytest.approx(h_rate * 1000)

    def test_compute_total_entropy_with_leakage(self):
        """Total entropy with leakage."""
        calc = NSMEntropyCalculator(storage_noise_r=0.5)
        h_rate, _ = calc.max_bound_entropy_rate()
        
        result = calc.compute_total_entropy(num_bits=1000, syndrome_leakage=200)
        
        expected = h_rate * 1000 - 200
        assert result.total_entropy == pytest.approx(expected)
        assert result.storage_noise_r == 0.5

    def test_compute_total_entropy_depleted(self):
        """Total entropy depleted returns zero."""
        calc = NSMEntropyCalculator(storage_noise_r=0.9)  # Low entropy
        
        result = calc.compute_total_entropy(num_bits=100, syndrome_leakage=200)
        
        # Leakage exceeds available entropy
        assert result.total_entropy == 0.0

    def test_crossover_point(self):
        """Crossover point constant."""
        assert NSMEntropyCalculator.crossover_point() == pytest.approx(R_CROSSOVER)

    def test_gamma_function(self):
        """Gamma function wrapper."""
        # Γ(0) = 0
        assert NSMEntropyCalculator.gamma(0) == 0.0
        
        # Γ(1) = 1
        assert NSMEntropyCalculator.gamma(1) == pytest.approx(1.0)
        
        # Γ(x) = x for x >= 0.5
        assert NSMEntropyCalculator.gamma(0.6) == pytest.approx(0.6)


class TestEntropyResult:
    """Tests for EntropyResult dataclass."""

    def test_entropy_result_creation(self):
        """Create EntropyResult."""
        result = EntropyResult(
            entropy_rate=0.5,
            total_entropy=500.0,
            dominant_bound="lupo",
            storage_noise_r=0.5,
        )
        
        assert result.entropy_rate == 0.5
        assert result.total_entropy == 500.0
        assert result.dominant_bound == "lupo"

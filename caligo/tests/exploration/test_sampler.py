"""
Unit tests for exploration sampler module.

Tests ParameterBounds and LHSSampler functionality.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.types import ExplorationSample, ReconciliationStrategy


class TestParameterBounds:
    """Tests for ParameterBounds dataclass."""

    def test_default_bounds(self):
        """Test default parameter bounds."""
        bounds = ParameterBounds()
        assert bounds.r_min == 0.0
        assert bounds.r_max == 1.0
        assert bounds.f_min == 0.501
        assert bounds.f_max == 1.0
        assert bounds.e_det_min == 0.0
        assert bounds.e_det_max == 0.1

    def test_custom_bounds(self):
        """Test custom parameter bounds."""
        bounds = ParameterBounds(f_min=0.8, f_max=0.99)
        assert bounds.f_min == 0.8
        assert bounds.f_max == 0.99

    def test_to_bounds_array(self):
        """Test conversion to numpy array."""
        bounds = ParameterBounds()
        arr = bounds.to_bounds_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 2)
        # Check first row (storage_noise_r bounds)
        assert arr[0, 0] == 0.0
        assert arr[0, 1] == 1.0


class TestLHSSampler:
    """Tests for LHSSampler class."""

    def test_sample_count(self, parameter_bounds):
        """Test that sampler generates the correct number of samples."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=100)
        assert len(samples) == 100

    def test_sample_type(self, parameter_bounds):
        """Test that samples are ExplorationSample instances."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=10)
        assert all(isinstance(s, ExplorationSample) for s in samples)

    def test_samples_within_bounds(self, parameter_bounds):
        """Test that all samples are within bounds."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=100)

        for sample in samples:
            assert 0.0 <= sample.storage_noise_r <= 1.0
            assert 0.001 <= sample.storage_rate_nu <= 1.0
            assert 1e5 <= sample.wait_time_ns <= 1e9
            assert 0.501 <= sample.channel_fidelity <= 1.0
            assert 0.001 <= sample.detection_efficiency <= 1.0
            assert 0.0 <= sample.detector_error <= 0.1
            assert 1e-8 <= sample.dark_count_prob <= 1e-3
            assert 1e4 <= sample.num_pairs <= 1e6

    def test_reproducibility_with_seed(self, parameter_bounds):
        """Test that same seed produces same samples."""
        sampler1 = LHSSampler(bounds=parameter_bounds, seed=42)
        sampler2 = LHSSampler(bounds=parameter_bounds, seed=42)
        samples1 = sampler1.generate(n=10)
        samples2 = sampler2.generate(n=10)
        assert samples1 == samples2

    def test_different_seeds_produce_different_samples(self, parameter_bounds):
        """Test that different seeds produce different samples."""
        sampler1 = LHSSampler(bounds=parameter_bounds, seed=42)
        sampler2 = LHSSampler(bounds=parameter_bounds, seed=123)
        samples1 = sampler1.generate(n=10)
        samples2 = sampler2.generate(n=10)
        assert samples1 != samples2

    def test_strategy_ratio(self, parameter_bounds):
        """Test strategy ratio distribution."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=1000, strategy_ratio=0.7)

        baseline_count = sum(1 for s in samples if s.strategy == ReconciliationStrategy.BASELINE)
        blind_count = sum(1 for s in samples if s.strategy == ReconciliationStrategy.BLIND)

        # strategy_ratio=0.7 means 70% should be BLIND
        blind_ratio = blind_count / len(samples)
        assert 0.65 <= blind_ratio <= 0.75  # Allow some variance

    def test_space_filling_property(self, parameter_bounds):
        """Test that LHS has better space-filling than pure random."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=100)

        # Convert to array and test uniformity of storage_noise_r (linear)
        values = np.array([s.storage_noise_r for s in samples])

        # KS test for uniformity
        stat, p_value = kstest(values, "uniform", args=(0, 1))
        # LHS should produce approximately uniform marginals
        assert p_value > 0.01  # Not strong evidence against uniformity

    def test_generate_as_array(self, parameter_bounds):
        """Test generating samples directly as array."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        arr = sampler.generate_array(n=50)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (50, 9)


class TestLHSSamplerEdgeCases:
    """Edge case tests for LHSSampler."""

    def test_sample_single(self, parameter_bounds):
        """Test generating a single sample."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=1)
        assert len(samples) == 1
        assert isinstance(samples[0], ExplorationSample)

    def test_sample_large_batch(self, parameter_bounds):
        """Test generating a large batch."""
        sampler = LHSSampler(bounds=parameter_bounds, seed=42)
        samples = sampler.generate(n=5000)
        assert len(samples) == 5000

    def test_narrow_bounds(self):
        """Test with narrow parameter bounds."""
        narrow_bounds = ParameterBounds(
            f_min=0.90,
            f_max=0.95,
        )
        sampler = LHSSampler(bounds=narrow_bounds, seed=42)
        samples = sampler.generate(n=50)

        for sample in samples:
            assert 0.90 <= sample.channel_fidelity <= 0.95


class TestLHSSamplerThreadSafety:
    """Thread safety tests for LHSSampler."""

    def test_concurrent_sampling_different_instances(self, parameter_bounds):
        """Test that concurrent instances work correctly."""
        from concurrent.futures import ThreadPoolExecutor

        def sample_batch(seed):
            sampler = LHSSampler(bounds=parameter_bounds, seed=seed)
            return sampler.generate(n=100)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(sample_batch, seed) for seed in range(4)]
            results = [f.result() for f in futures]

        # Each result should have 100 samples
        for result in results:
            assert len(result) == 100

        # Different seeds should produce different results
        assert results[0] != results[1]

    def test_same_seed_across_threads_deterministic(self, parameter_bounds):
        """Test that same seed produces same samples even in threads."""
        from concurrent.futures import ThreadPoolExecutor

        def sample_batch():
            sampler = LHSSampler(bounds=parameter_bounds, seed=42)
            return sampler.generate(n=50)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(sample_batch) for _ in range(2)]
            results = [f.result() for f in futures]

        # Same seed should produce same samples
        assert results[0] == results[1]

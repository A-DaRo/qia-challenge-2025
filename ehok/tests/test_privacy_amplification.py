"""
Tests for Phase 5: Privacy Amplification.

This module implements the test suite for the privacy amplification phase,
verifying the Toeplitz matrix implementation, security parameter calculations,
and statistical properties of the output keys.
"""

import pytest
import numpy as np
from collections import Counter
from scipy.stats import chisquare

from ehok.implementations.privacy_amplification.toeplitz_amplifier import ToeplitzAmplifier
from ehok.core.constants import TARGET_EPSILON_SEC, PA_SECURITY_MARGIN


class TestPrivacyAmplification:
    """Test suite for Privacy Amplification phase."""

    def setup_method(self):
        """Initialize amplifier for each test."""
        self.amplifier = ToeplitzAmplifier()

    def test_toeplitz_construction(self):
        """
        Test Case 7.1.1: Matrix Structure and Seed Generation.
        
        Verifies that the seed is generated with the correct length and type.
        """
        input_length = 6
        output_length = 4
        
        seed = self.amplifier.generate_hash_seed(input_length, output_length)
        
        # 1. Check length: m + n - 1
        expected_length = output_length + input_length - 1
        assert len(seed) == expected_length
        
        # 2. Check type
        assert seed.dtype == np.uint8
        
        # 3. Check binary values
        assert np.all(np.isin(seed, [0, 1]))

    def test_compression_length_calculation(self):
        """
        Test Case 7.2.1: Length Reduction Calculation.
        
        Verifies that the final key length is calculated correctly according
        to the leftover hash lemma.
        """
        sifted_length = 1000
        qber = 0.05
        leakage = 500
        epsilon = 1e-9
        
        final_length = self.amplifier.compute_final_length(
            sifted_length, qber, leakage, epsilon
        )
        
        # Manual calculation check
        # h(0.05) = -0.05*log2(0.05) - 0.95*log2(0.95)
        h_qber = -0.05 * np.log2(0.05) - 0.95 * np.log2(0.95)
        min_entropy = sifted_length * (1 - h_qber)
        epsilon_cost = 2 * np.log2(1.0 / epsilon)
        
        expected_float = min_entropy - leakage - epsilon_cost - PA_SECURITY_MARGIN
        expected_int = int(np.floor(expected_float))
        
        # Allow for small floating point differences, but integer result should match
        assert final_length == expected_int
        assert final_length > 0
        assert final_length < sifted_length

    def test_compression_execution(self):
        """
        Test Case 7.2.2: Actual Compression Execution.
        
        Verifies that the compression function produces output of the correct
        shape and type, and is deterministic.
        """
        n = 1000
        m = 54
        
        key = np.random.randint(0, 2, size=n, dtype=np.uint8)
        seed = self.amplifier.generate_hash_seed(n, m)
        
        compressed = self.amplifier.compress(key, seed)
        
        # 1. Check length
        assert len(compressed) == m
        
        # 2. Check type
        assert compressed.dtype == np.uint8
        
        # 3. Check binary values
        assert np.all(np.isin(compressed, [0, 1]))
        
        # 4. Check determinism
        compressed2 = self.amplifier.compress(key, seed)
        assert np.array_equal(compressed, compressed2)

    def test_compression_correctness_small_example(self):
        """
        Verify mathematical correctness with a small manual example.
        
        Let n=3, m=2. Seed length = 3+2-1 = 4.
        Seed = [s0, s1, s2, s3]
        Key = [k0, k1, k2]
        
        Toeplitz Matrix T (2x3):
        Row 0: seed[0:3] -> [s0, s1, s2]
        Row 1: seed[1:4] -> [s1, s2, s3]
        
        Output y = T @ k
        y0 = s0*k0 + s1*k1 + s2*k2
        y1 = s1*k0 + s2*k1 + s3*k2
        """
        n = 3
        m = 2
        
        # Define specific inputs
        key = np.array([1, 0, 1], dtype=np.uint8)
        seed = np.array([1, 1, 0, 1], dtype=np.uint8)
        
        # Expected calculation
        # Row 0: [1, 1, 0] . [1, 0, 1] = 1*1 + 1*0 + 0*1 = 1
        # Row 1: [1, 0, 1] . [1, 0, 1] = 1*1 + 0*0 + 1*1 = 2 -> 0 (mod 2)
        expected_output = np.array([1, 0], dtype=np.uint8)
        
        output = self.amplifier.compress(key, seed)
        
        assert np.array_equal(output, expected_output)

    def test_output_uniformity(self):
        """
        Test Case 7.3.1: Chi-Square Uniformity Test.
        
        Verifies that the output of the privacy amplification is statistically
        close to uniform random.
        """
        num_trials = 10000
        input_length = 100
        output_length = 8  # Use 8 bits (256 bins) for better statistics/performance balance
        
        outputs = []
        
        # We need to vary the key and seed to test the family of hash functions
        # or vary the key with a fixed seed?
        # The test spec says: "key = random, seed = generated (random)" inside the loop.
        # This tests that for random inputs and random seeds, the output is uniform.
        
        for _ in range(num_trials):
            key = np.random.randint(0, 2, size=input_length, dtype=np.uint8)
            seed = self.amplifier.generate_hash_seed(input_length, output_length)
            compressed = self.amplifier.compress(key, seed)
            
            # Convert binary array to integer for counting
            # e.g. [1, 0, 1] -> 5
            val = 0
            for bit in compressed:
                val = (val << 1) | int(bit)
            outputs.append(val)
            
        # Expected frequency for each of 2^8 = 256 bins
        num_bins = 2**output_length
        expected_freq = num_trials / num_bins
        
        # Count observed frequencies
        counts = Counter(outputs)
        observed = [counts.get(i, 0) for i in range(num_bins)]
        
        # Perform Chi-Square test
        chi2_stat, p_value = chisquare(observed, f_exp=expected_freq)
        
        # Critical value for df=255 at alpha=0.01 is approx 310
        # Or just check p-value > 0.01 (fail to reject null hypothesis of uniformity)
        
        # The spec mentions: "Acceptance Criterion: chi2 < threshold"
        # For 256 bins (df=255), 95% confidence threshold is ~293.
        # 99% confidence is ~310.
        
        # We use a soft assertion here because statistical tests can fail by chance.
        # However, with 10000 trials and 256 bins, it should be robust.
        
        print(f"Chi-square statistic: {chi2_stat}, p-value: {p_value}")
        
        # Fail if p-value is extremely low (e.g. < 0.001), indicating strong non-uniformity
        assert p_value > 0.001, f"Output distribution is not uniform (p={p_value})"

    def test_invalid_seed_length(self):
        """Test that invalid seed lengths raise an error."""
        n = 10
        key = np.zeros(n, dtype=np.uint8)
        
        # Seed too short to produce even 1 bit of output
        # Required: len(seed) >= n (since m = len(seed) - n + 1 >= 1)
        # Try len(seed) = n - 1 = 9
        short_seed = np.zeros(n - 1, dtype=np.uint8)
        
        with pytest.raises(ValueError):
            self.amplifier.compress(key, short_seed)

    def test_zero_length_output(self):
        """Test behavior when calculated length is zero or negative."""
        # High leakage/QBER resulting in 0 length
        length = self.amplifier.compute_final_length(
            sifted_length=100,
            qber=0.5,  # Very high error
            leakage=100,
            epsilon=1e-9
        )
        assert length == 0

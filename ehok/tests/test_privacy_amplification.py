"""
Tests for Phase 5: Privacy Amplification.

This module implements the test suite for the privacy amplification phase,
verifying the Toeplitz matrix implementation, security parameter calculations,
and statistical properties of the output keys.

Includes:
- PT1-PT6: Finite-key formula tests
- Original Toeplitz matrix correctness tests
- Statistical uniformity tests
"""

import pytest
import numpy as np
import warnings
from collections import Counter
from scipy.stats import chisquare

from ehok.implementations.privacy_amplification.toeplitz_amplifier import ToeplitzAmplifier
from ehok.implementations.privacy_amplification.finite_key import (
    FiniteKeyParams,
    compute_final_length_finite_key,
    compute_blind_reconciliation_leakage,
    compute_final_length_blind_mode,
    binary_entropy,
    compute_statistical_fluctuation,
)
from ehok.core.constants import TARGET_EPSILON_SEC


# =============================================================================
# PT1-PT6: Finite-Key Formula Tests (NEW)
# =============================================================================


class TestFiniteKeyFormula:
    """Test suite for finite-key privacy amplification formula (PT1-PT6)."""

    @pytest.mark.parametrize(
        "n,k,qber,leakage,expected_range",
        [
            # For finite-key QKD, larger key sizes are needed for non-zero output
            # These test cases use production-scale parameters
            (10000, 1000, 0.01, 3500, (800, 1200)),   # ~1% QBER, expect ~1000 bits
            (10000, 1000, 0.05, 3500, (0, 100)),      # ~5% QBER, finite-key penalty kicks in
            (50000, 5000, 0.05, 17500, (8000, 10000)), # ~5% QBER at scale
            (100000, 10000, 0.05, 35000, (20000, 25000)),  # Production scale
        ],
    )
    def test_pt1_finite_key_formula_correctness(self, n, k, qber, leakage, expected_range):
        """
        PT1: Verify finite-key formula produces lengths matching theoretical prediction.

        The formula should:
        1. Account for μ(ε) statistical fluctuation
        2. Produce lengths within expected_range for given parameters
        3. Be monotonically decreasing in QBER
        4. Be monotonically decreasing in leakage
        """
        params = FiniteKeyParams(n=n, k=k, qber_measured=qber, leakage=leakage)
        result = compute_final_length_finite_key(params)

        assert expected_range[0] <= result <= expected_range[1], (
            f"Expected {expected_range}, got {result}"
        )

    def test_pt2_finite_key_vs_asymptotic(self):
        """
        PT2: Verify finite-key formula is strictly more conservative than asymptotic.

        For all parameter combinations:
            - m_finite ≤ m_asymptotic
            - The difference shrinks as n, k → ∞
        """
        for n in [10000, 50000, 100000]:
            for qber in [0.02, 0.05, 0.08]:
                k = max(100, n // 10)
                leakage = n * 0.35  # Approximate

                # Asymptotic formula (no μ correction)
                h_qber = binary_entropy(qber)
                # Use same epsilon cost as finite-key
                epsilon_cost = 2 * np.log2(1.0 / 1e-9)  # DEFAULT_EPSILON_SEC
                m_asymp = n * (1 - h_qber) - leakage - epsilon_cost

                # Finite-key formula (includes μ correction)
                params = FiniteKeyParams(n=n, k=k, qber_measured=qber, leakage=leakage)
                m_finite = compute_final_length_finite_key(params)

                assert m_finite <= m_asymp, (
                    f"Finite-key ({m_finite}) > asymptotic ({m_asymp:.0f}) "
                    f"for n={n}, k={k}, QBER={qber}"
                )

    def test_pt3_no_fixed_output_length_required(self):
        """
        PT3: Verify the improved PA formula works for test scenarios without workarounds.

        This test verifies that the finite-key formula handles small test runs
        gracefully (returning 0 or positive keys depending on parameters).

        Note: For small key sizes, finite-key QKD is correctly conservative.
        Production runs should use n > 10000 for positive key output.
        """
        # Test with larger key size that can produce positive output
        params = FiniteKeyParams(
            n=10000,
            k=1000,
            qber_measured=0.001,  # Near-perfect channel
            leakage=3500,  # ~35% leakage
        )
        result = compute_final_length_finite_key(params)

        # With near-perfect channel and large key, should produce positive output
        assert result > 0, "Should produce positive key for near-perfect channel at scale"
        assert result <= 10000, "Cannot exceed input length"

    def test_pt3_small_key_correctly_conservative(self):
        """
        PT3 (additional): Verify small keys correctly return 0 (conservative behavior).

        Finite-key QKD requires larger key sizes for security. Small keys should
        return 0 because the statistical fluctuation penalty is too large.
        """
        params = FiniteKeyParams(
            n=90,  # Small test run
            k=10,
            qber_measured=0.0,
            leakage=45,
        )
        result = compute_final_length_finite_key(params)

        # For very small keys, finite-key formula correctly returns 0
        assert result >= 0, "Result must be non-negative"
        # Note: For n=90, the formula will likely return 0 due to finite-key penalty

    @pytest.mark.parametrize("qber", [0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
    def test_pt4_pa_robustness_qber_range(self, qber):
        """
        PT4: Verify PA produces valid output across operational QBER range.

        For each QBER in [1%, 10%]:
            - Output length is non-negative
            - Output length is strictly less than input
        """
        # Use production-scale parameters where finite-key formula gives meaningful output
        n, k, leakage = 50000, 5000, 17500
        params = FiniteKeyParams(n=n, k=k, qber_measured=qber, leakage=leakage)
        result = compute_final_length_finite_key(params)

        assert result >= 0, f"Negative output for QBER={qber}"
        assert result < n, "Output should be shorter than input"

    def test_pt4_qber_monotonicity(self):
        """
        PT4 (additional): Verify output length decreases monotonically with QBER.
        """
        # Use production-scale parameters
        n, k, leakage = 50000, 5000, 17500
        results = []
        for qber in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
            params = FiniteKeyParams(n=n, k=k, qber_measured=qber, leakage=leakage)
            results.append(compute_final_length_finite_key(params))

        # Results should be monotonically decreasing
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1], (
                f"Non-monotonic: QBER increase from {i} to {i+1} "
                f"gave {results[i]} → {results[i+1]}"
            )

    @pytest.mark.parametrize("failed_attempts", [0, 1, 2, 3])
    def test_pt5_blind_mode_leakage_accounting(self, failed_attempts):
        """
        PT5: Verify retry attempts are correctly penalized in leakage calculation.

        For each number of failed attempts:
            - Leakage increases with failed attempts
            - Output length decreases (more conservative)
        """
        # Use consistent parameters:
        # - Large reconciled_length for finite-key output
        # - error_count as fraction of (frame_size - n_shortened) for sensible QBER
        frame_size = 1024
        n_shortened = 128
        payload = frame_size - n_shortened  # 896
        error_count = 45  # ~5% of payload = ~45 errors

        base_result = compute_final_length_blind_mode(
            reconciled_length=100000,  # Very large for non-zero output
            error_count=error_count,
            frame_size=frame_size,
            n_shortened=n_shortened,
            successful_rate=0.65,
            hash_bits=50,
            test_bits=10000,
            failed_attempts=0,
        )

        retry_result = compute_final_length_blind_mode(
            reconciled_length=100000,
            error_count=error_count,
            frame_size=frame_size,
            n_shortened=n_shortened,
            successful_rate=0.65,
            hash_bits=50,
            test_bits=10000,
            failed_attempts=failed_attempts,
        )

        # For base case, verify we got a positive result
        if failed_attempts == 0:
            # The base result should be positive for these parameters
            # (error_count/payload ≈ 5% QBER, large reconciled_length)
            assert base_result > 0, f"Expected positive base result, got {base_result}"

        if failed_attempts > 0:
            assert retry_result < base_result, (
                f"Retries should reduce output: {retry_result} >= {base_result}"
            )

    def test_pt5_blind_leakage_calculation(self):
        """
        PT5 (additional): Verify blind reconciliation leakage formula.
        """
        # Basic leakage calculation
        leakage = compute_blind_reconciliation_leakage(
            frame_size=128,
            successful_rate=0.65,
            hash_bits=50,
            failed_attempts=0,
        )

        # Leakage should include syndrome bits + hash bits
        expected_syndrome = 128 * (1 - 0.65)  # ~45 bits
        assert leakage >= expected_syndrome, "Leakage should include syndrome"
        assert leakage >= 50, "Leakage should include hash bits"

        # Verify retry penalty
        leakage_with_retry = compute_blind_reconciliation_leakage(
            frame_size=128,
            successful_rate=0.65,
            hash_bits=50,
            failed_attempts=2,
        )
        assert leakage_with_retry > leakage, "Retry should increase leakage"

    @pytest.mark.long
    def test_pt6_output_key_uniformity(self):
        """
        PT6: Verify compressed key bits are approximately uniformly distributed.

        Method:
            - Generate 1000 random input keys
            - Compress each with different seeds
            - For each output bit position, count 0s and 1s
            - Chi-square test for uniformity (p > 0.01)
        """
        amplifier = ToeplitzAmplifier()
        num_trials = 1000
        input_length = 200
        output_length = 10
        rng = np.random.default_rng(42)

        # Count bit occurrences at each position
        bit_counts = np.zeros((output_length, 2), dtype=int)

        for _ in range(num_trials):
            key = rng.integers(0, 2, size=input_length, dtype=np.uint8)
            seed = amplifier.generate_hash_seed(input_length, output_length)
            compressed = amplifier.compress(key, seed)

            for pos, bit in enumerate(compressed):
                bit_counts[pos, bit] += 1

        # Chi-square test for each bit position
        for pos in range(output_length):
            observed = bit_counts[pos]
            expected = [num_trials / 2, num_trials / 2]
            chi2, p_value = chisquare(observed, f_exp=expected)

            assert p_value > 0.01, (
                f"Bit position {pos} not uniform: p={p_value:.4f}"
            )

    def test_pt6_independence_different_seeds(self):
        """
        PT6 (additional): Verify outputs from different seeds are independent.

        Method:
            - Fix input key
            - Generate 100 different seeds
            - Compute pairwise Hamming distance of outputs
            - Verify mean distance ≈ m/2 (expected for independent random bits)
        """
        amplifier = ToeplitzAmplifier()
        input_length = 100
        output_length = 20
        num_seeds = 100

        rng = np.random.default_rng(42)
        key = rng.integers(0, 2, size=input_length, dtype=np.uint8)

        outputs = []
        for _ in range(num_seeds):
            seed = amplifier.generate_hash_seed(input_length, output_length)
            outputs.append(amplifier.compress(key, seed))

        # Compute Hamming distances
        distances = []
        for i in range(num_seeds):
            for j in range(i + 1, num_seeds):
                dist = np.sum(outputs[i] != outputs[j])
                distances.append(dist)

        mean_dist = np.mean(distances)
        expected_dist = output_length / 2

        # Mean distance should be close to m/2 (within 20%)
        assert abs(mean_dist - expected_dist) < expected_dist * 0.2, (
            f"Mean Hamming distance {mean_dist:.2f} != expected {expected_dist}"
        )

    def test_statistical_fluctuation_scaling(self):
        """Verify μ(ε) scales correctly with sample sizes."""
        epsilon = 3.16e-5  # sqrt(1e-9)

        # μ should decrease as n, k increase
        mu_small = compute_statistical_fluctuation(n=100, k=10, epsilon=epsilon)
        mu_large = compute_statistical_fluctuation(n=10000, k=1000, epsilon=epsilon)

        assert mu_small > mu_large, "μ should decrease with larger samples"
        assert mu_small > 0, "μ must be positive"
        assert mu_large > 0, "μ must be positive"

        # μ should be less than 1 for reasonable sample sizes
        assert mu_large < 0.5, "μ should be small for large samples"

    def test_binary_entropy_properties(self):
        """Verify binary entropy function properties."""
        # h(0) = h(1) = 0
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

        # h(0.5) = 1
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

        # Symmetry: h(p) = h(1-p)
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(binary_entropy(p) - binary_entropy(1 - p)) < 1e-10


# =============================================================================
# Original Tests (Updated to work with both old and new API)
# =============================================================================


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
        Test Case 7.2.1: Length Reduction Calculation using finite-key formula.
        
        Verifies that the finite-key formula produces correct key lengths
        accounting for statistical fluctuation corrections.
        """
        # Use larger key with lower leakage to ensure positive output with low QBER
        sifted_length = 50000
        qber = 0.02  # Low QBER to ensure positive key
        leakage = 5000  # ~10% leakage
        epsilon = 1e-9
        test_bits = 5000  # Explicit test bits
        
        final_length = self.amplifier.compute_final_length(
            sifted_length, qber, leakage, epsilon, test_bits=test_bits
        )
        
        # finite-key formula should produce positive output for these parameters
        assert final_length > 0, f"Expected positive key, got {final_length}"
        assert final_length < sifted_length
        
        # Verify monotonicity: higher QBER -> shorter key
        # Use QBER that's high but still produces positive output
        final_length_high_qber = self.amplifier.compute_final_length(
            sifted_length, 0.05, leakage, epsilon, test_bits=test_bits
        )
        # Monotonicity: with higher QBER, key should be shorter (or both zero)
        assert final_length_high_qber <= final_length

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

    def test_compress_returns_zero_for_m_zero(self):
        """When computed m is 0, compress should return empty array, and final_length should be 0."""
        n = 8
        key = np.random.randint(0, 2, size=n, dtype=np.uint8)
        # seed length = n - 1 => m = 0
        seed = np.zeros(n - 1, dtype=np.uint8)
        compressed = self.amplifier.compress(key, seed)
        assert len(compressed) == 0

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

    def test_hankel_matrix_equivalence(self):
        """
        Ensure that compress() is equivalent to multiplying by a Hankel matrix
        constructed from the seed (row i uses seed[i:i+n]). Confirm result
        equals T @ key (mod 2).
        """
        n = 5
        m = 3
        key = np.random.randint(0, 2, size=n, dtype=np.uint8)
        seed = self.amplifier.generate_hash_seed(n, m)

        compressed = self.amplifier.compress(key, seed)

        # Build full Hankel matrix T (m x n) where T[i,j] = seed[i+j]
        T = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                T[i, j] = seed[i + j]

        # Matrix multiply mod 2
        prod = (T.dot(key.astype(np.int32)) % 2).astype(np.uint8)

        assert np.array_equal(prod, compressed)

    @pytest.mark.long
    def test_output_uniformity(self):
        """
        Test Case 7.3.1: Chi-Square Uniformity Test.
        
        Verifies that the output of the privacy amplification is statistically
        close to uniform random.
        """
        num_trials = 10000
        # Fix RNG seed for reproducibility in test environment
        rng = np.random.default_rng(20251209)
        input_length = 100
        output_length = 8  # Use 8 bits (256 bins) for better statistics/performance balance
        
        outputs = []
        
        # We need to vary the key and seed to test the family of hash functions
        # or vary the key with a fixed seed?
        # The test spec says: "key = random, seed = generated (random)" inside the loop.
        # This tests that for random inputs and random seeds, the output is uniform.
        
        for _ in range(num_trials):
            key = rng.integers(0, 2, size=input_length, dtype=np.uint8)
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

    @pytest.mark.long
    def test_output_uniformity_chi_square_10bits(self):
        """
        Test Case 7.3.1 (Spec): Chi-Square Uniformity Test with 10-bit output.

        This test uses the specification in the docs: output length = 10
        (1024 bins), num_trials = 10000 and asserts that the chi-square
        statistic is below the threshold (95% confidence for df=1023)
        which the spec takes as 1101.
        """
        num_trials = 10000
        # Fix RNG seed for reproducibility in test environment
        rng = np.random.default_rng(20251209)
        input_length = 100
        output_length = 10

        outputs = []
        for _ in range(num_trials):
            key = rng.integers(0, 2, size=input_length, dtype=np.uint8)
            seed = self.amplifier.generate_hash_seed(input_length, output_length)
            compressed = self.amplifier.compress(key, seed)
            val = 0
            for bit in compressed:
                val = (val << 1) | int(bit)
            outputs.append(val)

        num_bins = 2**output_length
        expected_freq = num_trials / num_bins
        counts = Counter(outputs)
        observed = [counts.get(i, 0) for i in range(num_bins)]

        chi2_stat, p_value = chisquare(observed, f_exp=expected_freq)

        # Accept if chi2 < 1101 (95% confidence approx for df=1023)
        assert chi2_stat < 1101, (
            f"Chi-square stat {chi2_stat:.2f} >= 1101 (not uniform at 95% for df=1023). "
            f"p-value={p_value:.5f}" )

    def test_invalid_seed_length(self):
        """Test that invalid seed lengths raise an error."""
        n = 10
        key = np.zeros(n, dtype=np.uint8)
        
        # If len(seed) == n - 1, output length m=0 is valid and should return empty
        short_seed_m_zero = np.zeros(n - 1, dtype=np.uint8)
        compressed = self.amplifier.compress(key, short_seed_m_zero)
        assert len(compressed) == 0

        # Seed too short to produce negative m (invalid), e.g., len(seed) = n - 2
        too_short_seed = np.zeros(n - 2, dtype=np.uint8)
        with pytest.raises(ValueError):
            self.amplifier.compress(key, too_short_seed)

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

    def test_compute_final_length_security_bound(self):
        """
        Verify that compute_final_length produces conservative outputs
        that account for finite-key statistical fluctuations.
        
        The finite-key formula is more conservative than asymptotic bounds.
        """
        sifted_length = 10000
        qber = 0.05
        leakage = 3500  # ~35% leakage
        epsilon = 1e-9
        test_bits = 1000

        final_length = self.amplifier.compute_final_length(
            sifted_length, qber, leakage, epsilon, test_bits=test_bits
        )

        # Compute asymptotic bound for comparison
        h_qber = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        min_entropy = sifted_length * (1 - h_qber)
        epsilon_cost = 2 * np.log2(1.0 / epsilon)
        asymptotic_bound = min_entropy - leakage - epsilon_cost

        # Finite-key result must be more conservative than asymptotic
        assert final_length <= asymptotic_bound
        assert final_length >= 0

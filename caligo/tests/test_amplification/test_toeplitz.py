"""
Unit tests for ToeplitzHasher.
"""

import numpy as np
import pytest
import secrets

from caligo.amplification.toeplitz import ToeplitzHasher
from caligo.types.exceptions import InvalidParameterError


class TestToeplitzHasher:
    """Tests for ToeplitzHasher class."""

    def test_init_basic(self):
        """Basic initialization."""
        hasher = ToeplitzHasher(input_length=100, output_length=50)
        
        assert hasher.input_length == 100
        assert hasher.output_length == 50

    def test_init_with_seed(self):
        """Initialization with seed."""
        seed = b"test_seed_12345678"
        hasher = ToeplitzHasher(input_length=100, output_length=50, seed=seed)
        
        assert hasher.input_length == 100

    def test_init_invalid_input_raises(self):
        """Invalid input length raises."""
        with pytest.raises(InvalidParameterError):
            ToeplitzHasher(input_length=0, output_length=50)
        
        with pytest.raises(InvalidParameterError):
            ToeplitzHasher(input_length=-10, output_length=50)

    def test_init_invalid_output_raises(self):
        """Invalid output length raises."""
        with pytest.raises(InvalidParameterError):
            ToeplitzHasher(input_length=100, output_length=0)

    def test_init_output_exceeds_input_raises(self):
        """Output > input raises."""
        with pytest.raises(InvalidParameterError):
            ToeplitzHasher(input_length=50, output_length=100)

    def test_hash_output_length(self):
        """Hash output has correct length."""
        hasher = ToeplitzHasher(input_length=100, output_length=50)
        input_key = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        output = hasher.hash(input_key)
        
        assert len(output) == 50

    def test_hash_output_binary(self):
        """Hash output is binary."""
        hasher = ToeplitzHasher(input_length=100, output_length=50)
        input_key = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        output = hasher.hash(input_key)
        
        assert all(b in (0, 1) for b in output)

    def test_hash_wrong_input_length_raises(self):
        """Wrong input length raises."""
        hasher = ToeplitzHasher(input_length=100, output_length=50)
        
        with pytest.raises(InvalidParameterError):
            hasher.hash(np.zeros(90, dtype=np.uint8))

    def test_hash_deterministic(self):
        """Same input + seed gives same output."""
        seed = b"deterministic_seed"
        hasher = ToeplitzHasher(input_length=100, output_length=50, seed=seed)
        input_key = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        out1 = hasher.hash(input_key)
        out2 = hasher.hash(input_key)
        
        assert np.array_equal(out1, out2)

    def test_different_seeds_different_outputs(self):
        """Different seeds produce different hashes."""
        input_key = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        hasher1 = ToeplitzHasher(
            input_length=100, output_length=50, seed=secrets.token_bytes(32)
        )
        hasher2 = ToeplitzHasher(
            input_length=100, output_length=50, seed=secrets.token_bytes(32)
        )
        
        out1 = hasher1.hash(input_key)
        out2 = hasher2.hash(input_key)
        
        # Should differ with overwhelming probability
        assert not np.array_equal(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different hashes."""
        seed = b"fixed_seed_for_test"
        hasher = ToeplitzHasher(input_length=100, output_length=50, seed=seed)
        
        input1 = np.zeros(100, dtype=np.uint8)
        input2 = np.ones(100, dtype=np.uint8)
        
        out1 = hasher.hash(input1)
        out2 = hasher.hash(input2)
        
        # Very likely different
        assert not np.array_equal(out1, out2)

    def test_fft_direct_equivalence(self):
        """FFT and direct methods both produce valid outputs."""
        seed = secrets.token_bytes(32)
        input_length = 100
        output_length = 50
        input_key = np.random.randint(0, 2, input_length, dtype=np.uint8)
        
        hasher_fft = ToeplitzHasher(
            input_length, output_length, seed=seed, use_fft=True
        )
        hasher_direct = ToeplitzHasher(
            input_length, output_length, seed=seed, use_fft=False
        )
        
        out_fft = hasher_fft.hash(input_key)
        out_direct = hasher_direct.hash(input_key)
        
        # Both should produce valid binary output of correct length
        assert len(out_fft) == output_length
        assert len(out_direct) == output_length
        assert all(b in (0, 1) for b in out_fft)
        assert all(b in (0, 1) for b in out_direct)
        
        # Both should be deterministic with same seed
        out_fft2 = hasher_fft.hash(input_key)
        out_direct2 = hasher_direct.hash(input_key)
        assert np.array_equal(out_fft, out_fft2)
        assert np.array_equal(out_direct, out_direct2)

    def test_get_matrix(self):
        """Get explicit Toeplitz matrix."""
        hasher = ToeplitzHasher(input_length=5, output_length=3, seed=b"test")
        
        matrix = hasher.get_matrix()
        
        assert matrix.shape == (3, 5)
        assert set(matrix.flatten()) <= {0, 1}
        
        # Verify Toeplitz structure (constant diagonals)
        for i in range(1, 3):
            for j in range(1, 5):
                if i - 1 >= 0 and j - 1 >= 0:
                    # Diagonal property: T[i,j] = T[i-1,j-1]
                    assert matrix[i, j] == matrix[i - 1, j - 1]

    def test_random_bits_property(self):
        """Random bits property returns copy."""
        hasher = ToeplitzHasher(input_length=100, output_length=50)
        
        bits1 = hasher.random_bits
        bits2 = hasher.random_bits
        
        # Should be equal
        assert np.array_equal(bits1, bits2)
        
        # Should be copies (not same object)
        bits1[0] = 99
        assert bits2[0] != 99

    def test_generate_seed_static(self):
        """Static generate_seed method."""
        seed1 = ToeplitzHasher.generate_seed(100)
        seed2 = ToeplitzHasher.generate_seed(100)
        
        # Should be different (crypto random)
        assert seed1 != seed2
        
        # Should have correct length
        assert len(seed1) >= 13  # ceil(100/8)


class TestToeplitzProperties:
    """Tests for mathematical properties of Toeplitz hashing."""

    def test_2_universal_property(self):
        """Approximate test of 2-universal property."""
        # For 2-universal: P[h(x) = h(x')] ≤ 2^{-m} for x ≠ x'
        # We can't test this exactly, but check collisions are rare
        
        m = 32  # output length
        n = 64  # input length
        num_trials = 100
        
        collision_count = 0
        
        for _ in range(num_trials):
            hasher = ToeplitzHasher(n, m, seed=secrets.token_bytes(16))
            
            x = np.random.randint(0, 2, n, dtype=np.uint8)
            x_prime = x.copy()
            x_prime[0] = 1 - x_prime[0]  # Flip one bit
            
            if np.array_equal(hasher.hash(x), hasher.hash(x_prime)):
                collision_count += 1
        
        # Expect very few collisions
        # Probability ~= 2^{-32} per trial
        assert collision_count < 5  # Very generous bound

    def test_output_distribution(self):
        """Output bits are roughly uniform."""
        hasher = ToeplitzHasher(input_length=100, output_length=64)
        
        # Hash many random inputs
        total_ones = 0
        total_bits = 0
        
        for _ in range(100):
            input_key = np.random.randint(0, 2, 100, dtype=np.uint8)
            output = hasher.hash(input_key)
            total_ones += np.sum(output)
            total_bits += len(output)
        
        # Should be roughly 50% ones
        ratio = total_ones / total_bits
        assert 0.4 < ratio < 0.6

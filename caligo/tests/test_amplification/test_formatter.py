"""
Unit tests for OTOutputFormatter.
"""

import numpy as np
import pytest

from caligo.amplification.formatter import (
    OTOutputFormatter,
    AliceOTOutput,
    BobOTOutput,
)
from caligo.types.exceptions import InvalidParameterError, ContractViolation


class TestAliceOTOutput:
    """Tests for AliceOTOutput dataclass."""

    def test_creation(self):
        """Basic creation."""
        key0 = np.array([0, 1, 0, 1], dtype=np.uint8)
        key1 = np.array([1, 0, 1, 0], dtype=np.uint8)
        
        output = AliceOTOutput(
            key_0=key0,
            key_1=key1,
            key_length=4,
        )
        
        assert np.array_equal(output.key_0, key0)
        assert np.array_equal(output.key_1, key1)
        assert output.key_length == 4


class TestBobOTOutput:
    """Tests for BobOTOutput dataclass."""

    def test_creation(self):
        """Basic creation."""
        key = np.array([1, 1, 0, 0], dtype=np.uint8)
        
        output = BobOTOutput(
            key_c=key,
            choice_bit=1,
            key_length=4,
        )
        
        assert np.array_equal(output.key_c, key)
        assert output.choice_bit == 1
        assert output.key_length == 4


class TestOTOutputFormatter:
    """Tests for OTOutputFormatter class."""

    def test_init_basic(self):
        """Basic initialization."""
        formatter = OTOutputFormatter(key_length=64)
        
        assert formatter.key_length == 64
        assert formatter.hasher_0 is None
        assert formatter.hasher_1 is None

    def test_init_with_seeds(self):
        """Initialization with seeds."""
        formatter = OTOutputFormatter(
            key_length=64,
            seed_0=b"seed_for_s0_key",
            seed_1=b"seed_for_s1_key",
        )
        
        assert formatter.key_length == 64

    def test_init_invalid_length_raises(self):
        """Invalid key length raises."""
        with pytest.raises(InvalidParameterError):
            OTOutputFormatter(key_length=0)
        
        with pytest.raises(InvalidParameterError):
            OTOutputFormatter(key_length=-10)

    def test_compute_alice_keys(self):
        """Compute Alice's OT output keys."""
        formatter = OTOutputFormatter(key_length=32, seed_0=b"s0", seed_1=b"s1")
        
        key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        output = formatter.compute_alice_keys(key_i0, key_i1)
        
        assert isinstance(output, AliceOTOutput)
        assert len(output.key_0) == 32
        assert len(output.key_1) == 32
        assert output.key_length == 32

    def test_compute_alice_keys_input_too_short(self):
        """Alice keys input too short raises."""
        formatter = OTOutputFormatter(key_length=64)
        
        key_i0 = np.zeros(50, dtype=np.uint8)  # Too short
        key_i1 = np.zeros(100, dtype=np.uint8)
        
        with pytest.raises(InvalidParameterError):
            formatter.compute_alice_keys(key_i0, key_i1)

    def test_compute_bob_key_choice_0(self):
        """Compute Bob's key with choice=0."""
        seed_0 = b"deterministic_seed_0"
        formatter = OTOutputFormatter(key_length=32, seed_0=seed_0)
        
        bob_key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        bob_key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        output = formatter.compute_bob_key(bob_key_i0, bob_key_i1, choice_bit=0)
        
        assert isinstance(output, BobOTOutput)
        assert len(output.key_c) == 32
        assert output.choice_bit == 0

    def test_compute_bob_key_choice_1(self):
        """Compute Bob's key with choice=1."""
        seed_1 = b"deterministic_seed_1"
        formatter = OTOutputFormatter(key_length=32, seed_1=seed_1)
        
        bob_key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        bob_key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        output = formatter.compute_bob_key(bob_key_i0, bob_key_i1, choice_bit=1)
        
        assert output.choice_bit == 1
        assert len(output.key_c) == 32

    def test_compute_bob_key_invalid_choice_raises(self):
        """Invalid choice bit raises."""
        formatter = OTOutputFormatter(key_length=32)
        bob_key = np.zeros(100, dtype=np.uint8)
        
        with pytest.raises(InvalidParameterError):
            formatter.compute_bob_key(bob_key, bob_key, choice_bit=2)
        
        with pytest.raises(InvalidParameterError):
            formatter.compute_bob_key(bob_key, bob_key, choice_bit=-1)


class TestOTCorrectness:
    """Tests for OT correctness verification."""

    def test_ot_correctness_choice_0(self):
        """OT correctness with choice=0: Bob gets S₀."""
        seed_0 = b"seed_for_key_zero"
        seed_1 = b"seed_for_key_one"
        formatter = OTOutputFormatter(key_length=32, seed_0=seed_0, seed_1=seed_1)
        
        # Same input keys (simulating perfect reconciliation)
        key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        # Alice computes both keys
        alice = formatter.compute_alice_keys(key_i0, key_i1)
        
        # Bob with choice=0 uses I₀ key
        bob = formatter.compute_bob_key(key_i0, key_i1, choice_bit=0)
        
        # Bob's key should equal Alice's S₀
        assert np.array_equal(bob.key_c, alice.key_0)

    def test_ot_correctness_choice_1(self):
        """OT correctness with choice=1: Bob gets S₁."""
        seed_0 = b"seed_for_key_zero"
        seed_1 = b"seed_for_key_one"
        formatter = OTOutputFormatter(key_length=32, seed_0=seed_0, seed_1=seed_1)
        
        key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        alice = formatter.compute_alice_keys(key_i0, key_i1)
        bob = formatter.compute_bob_key(key_i0, key_i1, choice_bit=1)
        
        # Bob's key should equal Alice's S₁
        assert np.array_equal(bob.key_c, alice.key_1)

    def test_format_final_output_success(self):
        """Format final output with correct keys."""
        seed_0 = b"seed_0"
        seed_1 = b"seed_1"
        formatter = OTOutputFormatter(key_length=32, seed_0=seed_0, seed_1=seed_1)
        
        key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        key_i1 = np.random.randint(0, 2, 100, dtype=np.uint8)
        
        alice = formatter.compute_alice_keys(key_i0, key_i1)
        bob = formatter.compute_bob_key(key_i0, key_i1, choice_bit=0)
        
        alice_key, bob_key = formatter.format_final_output(alice, bob)
        
        assert alice_key.key_length == 32
        assert bob_key.key_length == 32
        assert bob_key.choice_bit == 0

    def test_format_final_output_ot_violation_raises(self):
        """OT correctness violation raises ContractViolation."""
        formatter = OTOutputFormatter(key_length=32, seed_0=b"s0", seed_1=b"s1")
        
        # Create mismatched outputs manually
        alice = AliceOTOutput(
            key_0=np.zeros(32, dtype=np.uint8),
            key_1=np.ones(32, dtype=np.uint8),
            key_length=32,
        )
        bob = BobOTOutput(
            key_c=np.array([1, 0, 1, 0] * 8, dtype=np.uint8),  # Doesn't match
            choice_bit=0,
            key_length=32,
        )
        
        with pytest.raises(ContractViolation):
            formatter.format_final_output(alice, bob)


class TestDeterministicSeeds:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_same_output(self):
        """Same seeds produce same outputs."""
        seed = b"deterministic_test_seed"
        
        formatter1 = OTOutputFormatter(key_length=32, seed_0=seed)
        formatter2 = OTOutputFormatter(key_length=32, seed_0=seed)
        
        key_i0 = np.random.randint(0, 2, 100, dtype=np.uint8)
        key_i1 = np.zeros(100, dtype=np.uint8)
        
        out1 = formatter1.compute_alice_keys(key_i0, key_i1)
        out2 = formatter2.compute_alice_keys(key_i0, key_i1)
        
        assert np.array_equal(out1.key_0, out2.key_0)

    def test_different_seeds_different_output(self):
        """Different seeds produce different outputs."""
        # Fixed key - same for both formatters
        key_i0 = np.ones(100, dtype=np.uint8)
        key_i0[::2] = 0  # Alternating pattern
        key_i1 = np.zeros(100, dtype=np.uint8)
        
        # Use distinctly different seeds
        formatter1 = OTOutputFormatter(key_length=32, seed_0=b"alpha_seed_1234")
        formatter2 = OTOutputFormatter(key_length=32, seed_0=b"beta_seed_5678")
        
        out1 = formatter1.compute_alice_keys(key_i0.copy(), key_i1.copy())
        out2 = formatter2.compute_alice_keys(key_i0.copy(), key_i1.copy())
        
        # Same input + different seeds = different S₀ keys
        assert not np.array_equal(out1.key_0, out2.key_0)

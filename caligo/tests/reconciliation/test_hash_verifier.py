"""
Unit tests for hash verifier module.

Tests polynomial hash computation, collision probability, and verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.reconciliation.hash_verifier import PolynomialHashVerifier


class TestHashComputation:
    """Tests for hash computation."""

    def test_deterministic(self) -> None:
        """Same input produces same hash."""
        verifier = PolynomialHashVerifier()
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        
        hash1 = verifier.compute_hash(bits)
        hash2 = verifier.compute_hash(bits)
        
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        verifier = PolynomialHashVerifier()
        bits1 = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        bits2 = np.array([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)  # Last bit different
        
        hash1 = verifier.compute_hash(bits1)
        hash2 = verifier.compute_hash(bits2)
        
        assert hash1 != hash2

    def test_output_length(self) -> None:
        """Hash output is 50 bits (within range)."""
        verifier = PolynomialHashVerifier(output_bits=50)
        bits = np.random.default_rng(42).integers(0, 2, size=1000, dtype=np.int8)
        
        hash_value = verifier.compute_hash(bits)
        
        # 50-bit hash: max value is 2^50 - 1
        assert 0 <= hash_value < 2**50

    def test_empty_input(self) -> None:
        """Empty input produces valid hash."""
        verifier = PolynomialHashVerifier()
        bits = np.array([], dtype=np.int8)
        
        hash_value = verifier.compute_hash(bits)
        
        assert isinstance(hash_value, int)


class TestHashVerification:
    """Tests for hash verification."""

    def test_matching_blocks_verify(self) -> None:
        """Identical blocks verify successfully."""
        verifier = PolynomialHashVerifier()
        local_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
        remote_bits = local_bits.copy()
        
        remote_hash = verifier.compute_hash(remote_bits)
        
        assert verifier.verify(local_bits, remote_hash) is True

    def test_mismatched_blocks_fail(self) -> None:
        """Different blocks fail verification."""
        verifier = PolynomialHashVerifier()
        local_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
        remote_bits = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)  # Different
        
        remote_hash = verifier.compute_hash(remote_bits)
        
        assert verifier.verify(local_bits, remote_hash) is False

    def test_single_bit_difference(self) -> None:
        """Single bit difference detected."""
        verifier = PolynomialHashVerifier()
        
        # Large block with single bit difference
        rng = np.random.default_rng(42)
        local_bits = rng.integers(0, 2, size=4096, dtype=np.int8)
        remote_bits = local_bits.copy()
        remote_bits[2048] = 1 - remote_bits[2048]  # Flip one bit
        
        remote_hash = verifier.compute_hash(remote_bits)
        
        assert verifier.verify(local_bits, remote_hash) is False


class TestCollisionProbability:
    """Tests for collision probability bounds."""

    @pytest.mark.slow
    def test_collision_rate_empirical(self) -> None:
        """Empirical collision rate below theoretical bound."""
        verifier = PolynomialHashVerifier(output_bits=20)  # Smaller for test
        rng = np.random.default_rng(42)
        
        n_trials = 10000
        hashes = set()
        
        for _ in range(n_trials):
            bits = rng.integers(0, 2, size=100, dtype=np.int8)
            hashes.add(verifier.compute_hash(bits))
        
        # With 20-bit hash and 10k samples, expect ~some collisions
        # but collision rate should be reasonable
        collision_rate = 1 - len(hashes) / n_trials
        
        # For 2^20 ≈ 1M possible values, expect ~0.5% collisions with 10k samples
        # (birthday paradox approximation)
        assert collision_rate < 0.10  # Allow 10% margin


class TestDifferentHashSizes:
    """Tests for different hash output sizes."""

    @pytest.mark.parametrize("output_bits", [32, 40, 50, 60])
    def test_output_bits_respected(self, output_bits: int) -> None:
        """Hash respects output bit size."""
        verifier = PolynomialHashVerifier(output_bits=output_bits)
        bits = np.random.default_rng(42).integers(0, 2, size=500, dtype=np.int8)
        
        hash_value = verifier.compute_hash(bits)
        
        assert 0 <= hash_value < 2**output_bits

    def test_default_50_bits(self) -> None:
        """Default hash size is 50 bits."""
        verifier = PolynomialHashVerifier()
        assert verifier.output_bits == 50

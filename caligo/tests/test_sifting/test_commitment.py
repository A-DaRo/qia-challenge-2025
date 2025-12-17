"""
Unit tests for SHA256Commitment.
"""

import hashlib
import pytest

import numpy as np

from caligo.sifting.commitment import (
    SHA256Commitment,
    CommitmentResult,
    HASH_LENGTH_BYTES,
)
from caligo.types.exceptions import (
    CommitmentVerificationError,
    InvalidParameterError,
)


class TestSHA256Commitment:
    """Tests for SHA256Commitment class."""

    def test_commit_returns_valid_result(self):
        """Commit returns proper structure."""
        scheme = SHA256Commitment()
        data = b"test_data_12345"
        
        result = scheme.commit(data)
        
        assert isinstance(result, CommitmentResult)
        assert len(result.commitment) == HASH_LENGTH_BYTES
        assert len(result.nonce) == 32
        assert len(result.data_hash) == HASH_LENGTH_BYTES

    def test_commit_different_data_different_commitment(self):
        """Different data produces different commitments."""
        scheme = SHA256Commitment()
        
        result1 = scheme.commit(b"data_one")
        result2 = scheme.commit(b"data_two")
        
        assert result1.commitment != result2.commitment

    def test_commit_same_data_different_nonce(self):
        """Same data with different nonce produces different commitment."""
        scheme = SHA256Commitment()
        
        result1 = scheme.commit(b"same_data")
        result2 = scheme.commit(b"same_data")
        
        # Different nonces should produce different commitments
        assert result1.nonce != result2.nonce
        assert result1.commitment != result2.commitment

    def test_verify_valid_commitment(self):
        """Valid commitment verifies successfully."""
        scheme = SHA256Commitment()
        data = b"test_message"
        
        result = scheme.commit(data)
        
        # Verification should succeed
        assert scheme.verify(result.commitment, result.nonce, data) is True

    def test_verify_wrong_data_fails(self):
        """Wrong data fails verification."""
        scheme = SHA256Commitment()
        data = b"correct_data"
        
        result = scheme.commit(data)
        
        with pytest.raises(CommitmentVerificationError):
            scheme.verify(result.commitment, result.nonce, b"wrong_data")

    def test_verify_wrong_nonce_fails(self):
        """Wrong nonce fails verification."""
        scheme = SHA256Commitment()
        data = b"test_data"
        
        result = scheme.commit(data)
        wrong_nonce = b"x" * len(result.nonce)
        
        with pytest.raises(CommitmentVerificationError):
            scheme.verify(result.commitment, wrong_nonce, data)

    def test_verify_wrong_commitment_fails(self):
        """Wrong commitment fails verification."""
        scheme = SHA256Commitment()
        data = b"test_data"
        
        result = scheme.commit(data)
        wrong_commitment = b"x" * HASH_LENGTH_BYTES
        
        with pytest.raises(CommitmentVerificationError):
            scheme.verify(wrong_commitment, result.nonce, data)

    def test_commit_bases(self):
        """Commit to basis choices."""
        scheme = SHA256Commitment()
        bases = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        
        result = scheme.commit_bases(bases)
        
        assert len(result.commitment) == HASH_LENGTH_BYTES

    def test_verify_bases(self):
        """Verify basis commitment."""
        scheme = SHA256Commitment()
        bases = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        
        result = scheme.commit_bases(bases)
        
        assert scheme.verify_bases(result.commitment, result.nonce, bases) is True

    def test_verify_bases_wrong_bases_fails(self):
        """Wrong bases fail verification."""
        scheme = SHA256Commitment()
        bases = np.array([0, 1, 0, 1], dtype=np.uint8)
        wrong_bases = np.array([1, 0, 1, 0], dtype=np.uint8)
        
        result = scheme.commit_bases(bases)
        
        with pytest.raises(CommitmentVerificationError):
            scheme.verify_bases(result.commitment, result.nonce, wrong_bases)

    def test_short_nonce_raises(self):
        """Too short nonce length raises."""
        with pytest.raises(InvalidParameterError):
            SHA256Commitment(nonce_length=8)

    def test_custom_nonce_length(self):
        """Custom nonce length is respected."""
        scheme = SHA256Commitment(nonce_length=64)
        result = scheme.commit(b"data")
        
        assert len(result.nonce) == 64

    def test_hash_data_static(self):
        """Static hash_data method."""
        data = b"test"
        expected = hashlib.sha256(data).digest()
        
        assert SHA256Commitment.hash_data(data) == expected

    def test_generate_nonce_static(self):
        """Static generate_nonce method."""
        nonce1 = SHA256Commitment.generate_nonce(32)
        nonce2 = SHA256Commitment.generate_nonce(32)
        
        assert len(nonce1) == 32
        assert len(nonce2) == 32
        assert nonce1 != nonce2  # Should be different (crypto random)


class TestCommitmentBinding:
    """Tests for binding property of commitment."""

    def test_binding_property(self):
        """Cannot find different message with same commitment."""
        scheme = SHA256Commitment()
        
        # Create many commitments to different data
        # None should collide (with overwhelming probability)
        commitments = set()
        for i in range(1000):
            result = scheme.commit(f"data_{i}".encode())
            assert result.commitment not in commitments
            commitments.add(result.commitment)


class TestCommitmentHiding:
    """Tests for hiding property of commitment."""

    def test_commitment_reveals_nothing(self):
        """Commitment hash doesn't leak data."""
        scheme = SHA256Commitment()
        
        # Commit to similar data
        result0 = scheme.commit(b"secret_key_0")
        result1 = scheme.commit(b"secret_key_1")
        
        # Commitments should look random (no obvious pattern)
        # Check they have roughly equal bit distribution
        bits0 = bin(int.from_bytes(result0.commitment, "big")).count("1")
        bits1 = bin(int.from_bytes(result1.commitment, "big")).count("1")
        
        # Each should have roughly 128 ones (256 bits, 50% expected)
        assert 96 < bits0 < 160
        assert 96 < bits1 < 160

"""Unit tests for caligo.types.keys module."""

import pytest
from bitarray import bitarray

from caligo.types.keys import (
    ObliviousKey,
    AliceObliviousKey,
    BobObliviousKey,
    DEFAULT_SECURITY_PARAM,
)
from caligo.types.exceptions import ContractViolation


class TestObliviousKey:
    """Tests for ObliviousKey dataclass."""

    def test_valid_key_creation(self):
        """Test creating a valid ObliviousKey."""
        bits = bitarray("10101010")
        key = ObliviousKey(bits=bits, length=8)
        assert key.bits == bits
        assert key.length == 8
        assert key.security_param == DEFAULT_SECURITY_PARAM
        assert key.creation_time == 0.0

    def test_custom_security_param(self):
        """Test key with custom security parameter."""
        bits = bitarray("1010")
        key = ObliviousKey(bits=bits, length=4, security_param=1e-5)
        assert key.security_param == 1e-5

    def test_custom_creation_time(self):
        """Test key with custom creation time."""
        bits = bitarray("1010")
        key = ObliviousKey(bits=bits, length=4, creation_time=1000.0)
        assert key.creation_time == 1000.0

    def test_inv_key_001_length_mismatch(self):
        """INV-KEY-001: len(bits) must equal length."""
        bits = bitarray("10101010")  # 8 bits
        with pytest.raises(ContractViolation, match="INV-KEY-001"):
            ObliviousKey(bits=bits, length=4)  # Claims 4

    def test_inv_key_002_security_param_zero(self):
        """INV-KEY-002: security_param must be in (0, 1)."""
        bits = bitarray("1010")
        with pytest.raises(ContractViolation, match="INV-KEY-002"):
            ObliviousKey(bits=bits, length=4, security_param=0.0)

    def test_inv_key_002_security_param_one(self):
        """INV-KEY-002: security_param must be in (0, 1)."""
        bits = bitarray("1010")
        with pytest.raises(ContractViolation, match="INV-KEY-002"):
            ObliviousKey(bits=bits, length=4, security_param=1.0)

    def test_inv_key_002_security_param_negative(self):
        """INV-KEY-002: security_param must be in (0, 1)."""
        bits = bitarray("1010")
        with pytest.raises(ContractViolation, match="INV-KEY-002"):
            ObliviousKey(bits=bits, length=4, security_param=-0.1)

    def test_inv_key_003_negative_creation_time(self):
        """INV-KEY-003: creation_time must be >= 0."""
        bits = bitarray("1010")
        with pytest.raises(ContractViolation, match="INV-KEY-003"):
            ObliviousKey(bits=bits, length=4, creation_time=-1.0)

    def test_frozen_immutable(self):
        """Test that ObliviousKey is immutable (frozen)."""
        bits = bitarray("1010")
        key = ObliviousKey(bits=bits, length=4)
        with pytest.raises(AttributeError):
            key.length = 8  # type: ignore


class TestAliceObliviousKey:
    """Tests for AliceObliviousKey dataclass."""

    def test_valid_alice_key(self):
        """Test creating a valid AliceObliviousKey."""
        s0 = bitarray("10101010")
        s1 = bitarray("01010101")
        key = AliceObliviousKey(s0=s0, s1=s1, key_length=8)
        assert key.s0 == s0
        assert key.s1 == s1
        assert key.key_length == 8

    def test_inv_alice_001_s0_length_mismatch(self):
        """INV-ALICE-001: s0 length must equal key_length."""
        s0 = bitarray("1010")  # 4 bits
        s1 = bitarray("01010101")  # 8 bits
        with pytest.raises(ContractViolation, match="INV-ALICE-001"):
            AliceObliviousKey(s0=s0, s1=s1, key_length=8)

    def test_inv_alice_001_s1_length_mismatch(self):
        """INV-ALICE-001: s1 length must equal key_length."""
        s0 = bitarray("10101010")  # 8 bits
        s1 = bitarray("0101")  # 4 bits
        with pytest.raises(ContractViolation, match="INV-ALICE-001"):
            AliceObliviousKey(s0=s0, s1=s1, key_length=8)

    def test_inv_alice_002_invalid_security_param(self):
        """INV-ALICE-002: security_parameter must be in (0, 1)."""
        s0 = bitarray("1010")
        s1 = bitarray("0101")
        with pytest.raises(ContractViolation, match="INV-ALICE-002"):
            AliceObliviousKey(s0=s0, s1=s1, key_length=4, security_parameter=0.0)


class TestBobObliviousKey:
    """Tests for BobObliviousKey dataclass."""

    def test_valid_bob_key_choice_0(self):
        """Test creating a valid BobObliviousKey with choice_bit=0."""
        sc = bitarray("10101010")
        key = BobObliviousKey(sc=sc, choice_bit=0, key_length=8)
        assert key.sc == sc
        assert key.choice_bit == 0

    def test_valid_bob_key_choice_1(self):
        """Test creating a valid BobObliviousKey with choice_bit=1."""
        sc = bitarray("01010101")
        key = BobObliviousKey(sc=sc, choice_bit=1, key_length=8)
        assert key.choice_bit == 1

    def test_inv_bob_001_length_mismatch(self):
        """INV-BOB-001: sc length must equal key_length."""
        sc = bitarray("1010")  # 4 bits
        with pytest.raises(ContractViolation, match="INV-BOB-001"):
            BobObliviousKey(sc=sc, choice_bit=0, key_length=8)

    def test_inv_bob_002_invalid_choice_bit(self):
        """INV-BOB-002: choice_bit must be in {0, 1}."""
        sc = bitarray("10101010")
        with pytest.raises(ContractViolation, match="INV-BOB-002"):
            BobObliviousKey(sc=sc, choice_bit=2, key_length=8)

    def test_inv_bob_002_negative_choice_bit(self):
        """INV-BOB-002: choice_bit must be in {0, 1}."""
        sc = bitarray("10101010")
        with pytest.raises(ContractViolation, match="INV-BOB-002"):
            BobObliviousKey(sc=sc, choice_bit=-1, key_length=8)

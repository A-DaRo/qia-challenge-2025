"""Unit tests for caligo.utils.bitarray_utils module."""

import numpy as np
import pytest
from bitarray import bitarray

from caligo.utils.bitarray_utils import (
    xor_bitarrays,
    bitarray_to_bytes,
    bytes_to_bitarray,
    random_bitarray,
    hamming_distance,
    slice_bitarray,
    bitarray_from_numpy,
    bitarray_to_numpy,
)


class TestXorBitarrays:
    """Tests for xor_bitarrays function."""

    def test_xor_basic(self):
        """Test basic XOR operation."""
        a = bitarray("1010")
        b = bitarray("1100")
        result = xor_bitarrays(a, b)
        assert result == bitarray("0110")

    def test_xor_all_zeros(self):
        """XOR with all zeros returns original."""
        a = bitarray("10101010")
        b = bitarray("00000000")
        assert xor_bitarrays(a, b) == a

    def test_xor_with_self(self):
        """XOR with self returns all zeros."""
        a = bitarray("10101010")
        result = xor_bitarrays(a, a)
        assert result == bitarray("00000000")

    def test_xor_length_mismatch(self):
        """Raise ValueError for different lengths."""
        a = bitarray("1010")
        b = bitarray("10101010")
        with pytest.raises(ValueError, match="lengths differ"):
            xor_bitarrays(a, b)

    def test_xor_empty(self):
        """XOR of empty bitarrays."""
        a = bitarray()
        b = bitarray()
        assert xor_bitarrays(a, b) == bitarray()


class TestBitarrayToBytes:
    """Tests for bitarray_to_bytes function."""

    def test_byte_aligned(self):
        """Test conversion of byte-aligned bitarray."""
        bits = bitarray("00001111")  # 0x0F
        assert bitarray_to_bytes(bits) == b"\x0f"

    def test_multiple_bytes(self):
        """Test conversion of multiple bytes."""
        bits = bitarray("0000111110101010")  # 0x0F 0xAA
        assert bitarray_to_bytes(bits) == b"\x0f\xaa"

    def test_empty(self):
        """Test conversion of empty bitarray."""
        assert bitarray_to_bytes(bitarray()) == b""


class TestBytesToBitarray:
    """Tests for bytes_to_bitarray function."""

    def test_single_byte(self):
        """Test conversion of single byte."""
        result = bytes_to_bitarray(b"\x0f")
        assert result == bitarray("00001111")

    def test_multiple_bytes(self):
        """Test conversion of multiple bytes."""
        result = bytes_to_bitarray(b"\x0f\xaa")
        assert result == bitarray("0000111110101010")

    def test_empty(self):
        """Test conversion of empty bytes."""
        assert bytes_to_bitarray(b"") == bitarray()

    def test_roundtrip(self):
        """Test bytes → bitarray → bytes roundtrip."""
        original = b"\xde\xad\xbe\xef"
        bits = bytes_to_bitarray(original)
        recovered = bitarray_to_bytes(bits)
        assert recovered == original


class TestRandomBitarray:
    """Tests for random_bitarray function."""

    def test_correct_length(self):
        """Generated bitarray has correct length."""
        for length in [0, 1, 7, 8, 15, 16, 100, 1000]:
            bits = random_bitarray(length)
            assert len(bits) == length

    def test_randomness(self):
        """Generated bitarrays are not all zeros or ones."""
        # For a 1000-bit random array, probability of all same is negligible
        bits = random_bitarray(1000)
        assert bits.count() > 0  # Has some 1s
        assert bits.count() < 1000  # Has some 0s

    def test_different_each_call(self):
        """Different calls produce different results."""
        bits1 = random_bitarray(256)
        bits2 = random_bitarray(256)
        # Probability of collision is 2^{-256}, effectively impossible
        assert bits1 != bits2

    def test_negative_length(self):
        """Raise ValueError for negative length."""
        with pytest.raises(ValueError, match="length="):
            random_bitarray(-1)


class TestHammingDistance:
    """Tests for hamming_distance function."""

    def test_identical_arrays(self):
        """Distance between identical arrays is 0."""
        a = bitarray("10101010")
        assert hamming_distance(a, a) == 0

    def test_completely_different(self):
        """Distance between complements equals length."""
        a = bitarray("10101010")
        b = bitarray("01010101")
        assert hamming_distance(a, b) == 8

    def test_known_distance(self):
        """Test known Hamming distance."""
        a = bitarray("1010")
        b = bitarray("1001")
        assert hamming_distance(a, b) == 2

    def test_length_mismatch(self):
        """Raise ValueError for different lengths."""
        a = bitarray("1010")
        b = bitarray("10101010")
        with pytest.raises(ValueError, match="lengths differ"):
            hamming_distance(a, b)

    def test_empty(self):
        """Distance between empty arrays is 0."""
        assert hamming_distance(bitarray(), bitarray()) == 0


class TestSliceBitarray:
    """Tests for slice_bitarray function."""

    def test_basic_slice(self):
        """Test basic slicing operation."""
        bits = bitarray("10110100")
        # bits[0]=1, bits[2]=1, bits[4]=0, bits[6]=0
        indices = np.array([0, 2, 4, 6])
        result = slice_bitarray(bits, indices)
        assert result == bitarray("1100")

    def test_single_index(self):
        """Test slicing with single index."""
        bits = bitarray("10110100")
        result = slice_bitarray(bits, [3])
        assert result == bitarray("1")

    def test_empty_indices(self):
        """Test slicing with no indices."""
        bits = bitarray("10110100")
        result = slice_bitarray(bits, [])
        assert result == bitarray()

    def test_out_of_bounds(self):
        """Raise IndexError for out-of-bounds index."""
        bits = bitarray("1010")
        with pytest.raises(IndexError):
            slice_bitarray(bits, [4])

    def test_negative_index(self):
        """Raise IndexError for negative index."""
        bits = bitarray("1010")
        with pytest.raises(IndexError):
            slice_bitarray(bits, [-1])

    def test_preserves_order(self):
        """Indices are processed in given order."""
        bits = bitarray("10110100")
        # bits[6]=0, bits[4]=0, bits[2]=1, bits[0]=1
        indices = np.array([6, 4, 2, 0])  # Reverse order
        result = slice_bitarray(bits, indices)
        assert result == bitarray("0011")


class TestBitarrayFromNumpy:
    """Tests for bitarray_from_numpy function."""

    def test_basic_conversion(self):
        """Test basic numpy to bitarray conversion."""
        arr = np.array([1, 0, 1, 1, 0])
        result = bitarray_from_numpy(arr)
        assert result == bitarray("10110")

    def test_uint8_array(self):
        """Test conversion from uint8 array."""
        arr = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        result = bitarray_from_numpy(arr)
        assert result == bitarray("10110")

    def test_empty_array(self):
        """Test conversion of empty array."""
        arr = np.array([], dtype=np.uint8)
        result = bitarray_from_numpy(arr)
        assert result == bitarray()

    def test_invalid_values(self):
        """Raise ValueError for values other than 0 or 1."""
        arr = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="invalid values"):
            bitarray_from_numpy(arr)


class TestBitarrayToNumpy:
    """Tests for bitarray_to_numpy function."""

    def test_basic_conversion(self):
        """Test basic bitarray to numpy conversion."""
        bits = bitarray("10110")
        result = bitarray_to_numpy(bits)
        expected = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_dtype(self):
        """Result has uint8 dtype."""
        bits = bitarray("1010")
        result = bitarray_to_numpy(bits)
        assert result.dtype == np.uint8

    def test_empty(self):
        """Test conversion of empty bitarray."""
        result = bitarray_to_numpy(bitarray())
        assert len(result) == 0

    def test_roundtrip(self):
        """Test numpy → bitarray → numpy roundtrip."""
        original = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        bits = bitarray_from_numpy(original)
        recovered = bitarray_to_numpy(bits)
        np.testing.assert_array_equal(recovered, original)

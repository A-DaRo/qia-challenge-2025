"""
Bitarray manipulation helpers for key processing.

This module provides utility functions for working with bitarrays,
which are used throughout the protocol for efficient bit manipulation.
"""

from __future__ import annotations

import secrets
from typing import Union

import numpy as np
from bitarray import bitarray


def xor_bitarrays(a: bitarray, b: bitarray) -> bitarray:
    """
    XOR two bitarrays of equal length.

    Parameters
    ----------
    a : bitarray
        First bitarray.
    b : bitarray
        Second bitarray.

    Returns
    -------
    bitarray
        Result of a XOR b.

    Raises
    ------
    ValueError
        If bitarrays have different lengths.

    Examples
    --------
    >>> a = bitarray('1010')
    >>> b = bitarray('1100')
    >>> xor_bitarrays(a, b)
    bitarray('0110')
    """
    if len(a) != len(b):
        raise ValueError(f"Bitarray lengths differ: {len(a)} != {len(b)}")

    return a ^ b


def bitarray_to_bytes(bits: bitarray) -> bytes:
    """
    Convert bitarray to bytes (big-endian).

    Parameters
    ----------
    bits : bitarray
        Input bitarray.

    Returns
    -------
    bytes
        Byte representation.

    Notes
    -----
    If the bitarray length is not a multiple of 8, it is
    padded with zeros on the right before conversion.

    Examples
    --------
    >>> bits = bitarray('00001111')
    >>> bitarray_to_bytes(bits)
    b'\\x0f'
    """
    return bits.tobytes()


def bytes_to_bitarray(data: bytes) -> bitarray:
    """
    Convert bytes to bitarray.

    Parameters
    ----------
    data : bytes
        Input bytes.

    Returns
    -------
    bitarray
        Bitarray representation.

    Examples
    --------
    >>> bytes_to_bitarray(b'\\x0f')
    bitarray('00001111')
    """
    result = bitarray()
    result.frombytes(data)
    return result


def random_bitarray(length: int) -> bitarray:
    """
    Generate cryptographically secure random bitarray.

    Parameters
    ----------
    length : int
        Length of bitarray in bits.

    Returns
    -------
    bitarray
        Random bitarray of specified length.

    Raises
    ------
    ValueError
        If length < 0.

    Notes
    -----
    Uses secrets.token_bytes for cryptographic randomness.

    Examples
    --------
    >>> bits = random_bitarray(16)
    >>> len(bits)
    16
    """
    if length < 0:
        raise ValueError(f"length={length} must be >= 0")
    if length == 0:
        return bitarray()

    # Generate enough random bytes
    num_bytes = (length + 7) // 8
    random_bytes = secrets.token_bytes(num_bytes)

    result = bitarray()
    result.frombytes(random_bytes)

    # Trim to exact length
    return result[:length]


def hamming_distance(a: bitarray, b: bitarray) -> int:
    """
    Compute Hamming distance between two bitarrays.

    Parameters
    ----------
    a : bitarray
        First bitarray.
    b : bitarray
        Second bitarray.

    Returns
    -------
    int
        Number of positions where bits differ.

    Raises
    ------
    ValueError
        If bitarrays have different lengths.

    Examples
    --------
    >>> a = bitarray('1010')
    >>> b = bitarray('1001')
    >>> hamming_distance(a, b)
    2
    """
    if len(a) != len(b):
        raise ValueError(f"Bitarray lengths differ: {len(a)} != {len(b)}")

    return (a ^ b).count()


def slice_bitarray(bits: bitarray, indices: Union[np.ndarray, list]) -> bitarray:
    """
    Extract bits at specified indices.

    Parameters
    ----------
    bits : bitarray
        Source bitarray.
    indices : Union[np.ndarray, list]
        Array or list of indices to extract.

    Returns
    -------
    bitarray
        Bitarray containing only bits at specified indices.

    Raises
    ------
    IndexError
        If any index is out of bounds.

    Examples
    --------
    >>> bits = bitarray('10110100')
    >>> indices = np.array([0, 2, 4, 6])
    >>> slice_bitarray(bits, indices)
    bitarray('1110')
    """
    result = bitarray()
    for idx in indices:
        if idx < 0 or idx >= len(bits):
            raise IndexError(f"Index {idx} out of bounds for bitarray of length {len(bits)}")
        result.append(bits[idx])
    return result


def bitarray_from_numpy(arr: np.ndarray) -> bitarray:
    """
    Convert numpy array of 0/1 values to bitarray.

    Parameters
    ----------
    arr : np.ndarray
        Array containing 0/1 values.

    Returns
    -------
    bitarray
        Equivalent bitarray.

    Raises
    ------
    ValueError
        If array contains values other than 0 or 1.

    Examples
    --------
    >>> arr = np.array([1, 0, 1, 1, 0])
    >>> bitarray_from_numpy(arr)
    bitarray('10110')
    """
    if arr.size > 0:
        unique_vals = set(np.unique(arr))
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Array contains invalid values: {unique_vals - {0, 1}}")

    return bitarray(arr.tolist())


def bitarray_to_numpy(bits: bitarray) -> np.ndarray:
    """
    Convert bitarray to numpy array of uint8.

    Parameters
    ----------
    bits : bitarray
        Input bitarray.

    Returns
    -------
    np.ndarray
        Array of 0/1 values, dtype uint8.

    Examples
    --------
    >>> bits = bitarray('10110')
    >>> bitarray_to_numpy(bits)
    array([1, 0, 1, 1, 0], dtype=uint8)
    """
    return np.array(bits.tolist(), dtype=np.uint8)

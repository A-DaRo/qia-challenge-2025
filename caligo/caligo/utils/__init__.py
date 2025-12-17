"""
Caligo utilities package: Cross-cutting concerns.

This package provides logging, mathematical functions, and bitarray
manipulation helpers used throughout the protocol implementation.
"""

from caligo.utils.logging import (
    get_logger,
    setup_script_logging,
)

from caligo.utils.math import (
    binary_entropy,
    channel_capacity,
    finite_size_penalty,
    gamma_function,
    smooth_min_entropy_rate,
    key_length_bound,
)

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

__all__ = [
    # Logging
    "get_logger",
    "setup_script_logging",
    # Math
    "binary_entropy",
    "channel_capacity",
    "finite_size_penalty",
    "gamma_function",
    "smooth_min_entropy_rate",
    "key_length_bound",
    # Bitarray
    "xor_bitarrays",
    "bitarray_to_bytes",
    "bytes_to_bitarray",
    "random_bitarray",
    "hamming_distance",
    "slice_bitarray",
    "bitarray_from_numpy",
    "bitarray_to_numpy",
]

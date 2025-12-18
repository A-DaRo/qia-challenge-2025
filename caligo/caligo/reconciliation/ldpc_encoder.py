"""
LDPC Encoder for Syndrome Computation (Alice Side).

Computes syndromes for Alice's key blocks, handling padding
generation and frame construction for reconciliation.

The encoder is used by Alice to generate syndrome information
that Bob uses to correct errors in his received key.

References
----------
- MacKay (2003): LDPC encoding
- Martinez-Mateo et al. (2012): Rate-adaptive reconciliation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import CompiledParityCheckMatrix
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class SyndromeBlock:
    """
    Syndrome data for a single LDPC block.

    Attributes
    ----------
    syndrome : np.ndarray
        Binary syndrome vector s = H·x mod 2.
    rate : float
        Code rate used for this block.
    n_shortened : int
        Number of shortened (padding) bits.
    prng_seed : int
        Seed for deterministic padding generation.
    payload_length : int
        Original payload length (excluding padding).
    leakage_bits : int
        Syndrome bits leaked (for accounting).
    """

    syndrome: np.ndarray
    rate: float
    n_shortened: int
    prng_seed: int
    payload_length: int
    leakage_bits: int


# =============================================================================
# Padding Generation
# =============================================================================


def generate_padding(length: int, seed: int) -> np.ndarray:
    """
    Generate deterministic padding bits.

    Both Alice and Bob must use identical seeds to generate
    matching padding for frame construction.

    Parameters
    ----------
    length : int
        Number of padding bits to generate.
    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Binary padding array (uint8).

    Notes
    -----
    Uses NumPy's default_rng for strong pseudo-random generation.
    Seed must be shared out-of-band or derived from protocol state.
    """
    if length <= 0:
        return np.array([], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, dtype=np.uint8)


# =============================================================================
# Syndrome Computation
# =============================================================================


def compute_syndrome(
    key_bits: np.ndarray,
    H: "sp.csr_matrix | CompiledParityCheckMatrix",
) -> np.ndarray:
    """
    Compute syndrome s = H·x mod 2.

    Parameters
    ----------
    key_bits : np.ndarray
        Full frame bits (payload + padding), uint8.
    H : sp.csr_matrix
        Parity-check matrix (m × n).

    Returns
    -------
    np.ndarray
        Syndrome vector (uint8, length m).

    Raises
    ------
    ValueError
        If key_bits length doesn't match H columns.
    """
    if isinstance(H, CompiledParityCheckMatrix):
        if key_bits.shape[0] != H.n:
            raise ValueError(
                f"Key length {key_bits.shape[0]} != H columns {H.n}"
            )
        return H.compute_syndrome(key_bits)

    H = H.tocsr()
    if key_bits.shape[0] != H.shape[1]:
        raise ValueError(
            f"Key length {key_bits.shape[0]} != H columns {H.shape[1]}"
        )
    return ((H @ key_bits) % 2).astype(np.uint8)


def encode_block(
    alice_key_or_frame: np.ndarray,
    H: sp.csr_matrix,
    rate: float = None,
    n_shortened: int = None,
    prng_seed: int = None,
) -> SyndromeBlock:
    """
    Encode a key block to produce syndrome.

    Supports two calling conventions:
    1. encode_block(alice_key, H, rate, n_shortened, prng_seed) - full API
    2. encode_block(frame, H) - simple API (frame already prepared)

    Parameters
    ----------
    alice_key_or_frame : np.ndarray
        Alice's payload bits or complete frame (uint8).
    H : sp.csr_matrix
        Parity-check matrix for selected rate.
    rate : float, optional
        LDPC code rate (for full API).
    n_shortened : int, optional
        Number of shortened bits (for full API).
    prng_seed : int, optional
        Seed for padding generation (for full API).

    Returns
    -------
    SyndromeBlock
        Complete syndrome data for transmission.

    Raises
    ------
    ValueError
        If payload + padding doesn't match frame size.
    """
    frame_size = H.shape[1]
    input_len = len(alice_key_or_frame)

    # Simple API: input is already the complete frame
    if input_len == frame_size and rate is None:
        full_frame = alice_key_or_frame
        n_shortened = 0
        prng_seed = 0
        payload_len = frame_size
        # Estimate rate from matrix dimensions
        rate = 1.0 - H.shape[0] / frame_size
    else:
        # Full API: construct frame from payload
        alice_key = alice_key_or_frame
        payload_len = len(alice_key)
        
        if n_shortened is None:
            n_shortened = frame_size - payload_len
            
        if rate is None:
            rate = 1.0 - H.shape[0] / frame_size
            
        if prng_seed is None:
            prng_seed = 0

        # Validate dimensions
        if payload_len + n_shortened != frame_size:
            n_shortened = frame_size - payload_len
            logger.debug(
                "Adjusted n_shortened to %d for frame_size=%d, payload=%d",
                n_shortened, frame_size, payload_len
            )

        if n_shortened < 0:
            raise ValueError(
                f"Payload {payload_len} exceeds frame size {frame_size}"
            )

        # Generate padding and construct full frame
        padding = generate_padding(n_shortened, prng_seed)
        full_frame = np.concatenate([alice_key, padding])

    # Compute syndrome
    syndrome = compute_syndrome(full_frame, H)
    syndrome_length = len(syndrome)

    return SyndromeBlock(
        syndrome=syndrome,
        rate=rate,
        n_shortened=n_shortened,
        prng_seed=prng_seed,
        payload_length=payload_len,
        leakage_bits=syndrome_length,
    )


def prepare_frame(
    key_bits: np.ndarray,
    n_shortened: int = None,
    prng_seed: int = None,
    frame_size: int = None,
) -> np.ndarray:
    """
    Prepare full frame from payload and generated padding.

    Utility function for frame construction without syndrome
    computation (used by decoder).

    Supports two calling conventions:
    1. prepare_frame(key_bits, n_shortened, prng_seed) - explicit padding
    2. prepare_frame(key_bits, frame_size=N) - auto-compute padding

    Parameters
    ----------
    key_bits : np.ndarray
        Payload bits (uint8).
    n_shortened : int, optional
        Number of padding bits (explicit mode).
    prng_seed : int, optional
        Padding generation seed (defaults to 0).
    frame_size : int, optional
        Target frame size (auto-compute mode).

    Returns
    -------
    np.ndarray
        Full frame: [payload | padding].
    """
    payload_len = len(key_bits)
    
    # Determine shortening from frame_size if provided
    if frame_size is not None:
        n_shortened = frame_size - payload_len
        if n_shortened < 0:
            raise ValueError(f"Payload {payload_len} exceeds frame_size {frame_size}")
    
    if n_shortened is None:
        n_shortened = 0
    if prng_seed is None:
        prng_seed = 0
        
    if n_shortened == 0:
        return key_bits.copy()
        
    padding = generate_padding(n_shortened, prng_seed)
    return np.concatenate([key_bits, padding])

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
from typing import Optional, Tuple

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
    payload_length : int
        Original payload length (excluding punctured bits).
    leakage_bits : int
        Syndrome bits leaked (for accounting).
    puncture_pattern : np.ndarray
        Untainted puncturing pattern used.
    """

    syndrome: np.ndarray
    rate: float
    payload_length: int
    leakage_bits: int
    puncture_pattern: np.ndarray


# =============================================================================
# Pattern-Based Frame Construction
# =============================================================================


def apply_puncture_pattern(
    payload: np.ndarray,
    pattern: np.ndarray,
    frame_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct LDPC frame using untainted puncturing pattern.

    The pattern indicates which positions should be punctured (unknown to Bob).
    Punctured positions are zero-initialized; non-punctured positions are filled
    with payload bits in order.

    Parameters
    ----------
    payload : np.ndarray
        Alice's payload bits (uint8), length k.
    pattern : np.ndarray
        Binary puncturing pattern (uint8), shape (frame_size,).
        pattern[i] = 1 indicates position i is punctured.
    frame_size : int
        Target LDPC frame size n.

    Returns
    -------
    frame : np.ndarray
        Full LDPC frame with payload and punctured bits, shape (frame_size,).
    mask : np.ndarray
        Binary mask where mask[i] = 1 indicates punctured position.

    Raises
    ------
    ValueError
        If payload length + punctured positions != frame_size.

    Notes
    -----
    This implements the Elkouss et al. (2012) untainted puncturing scheme.
    Punctured bits are set to zero (arbitrary choice; Bob won't use these values).
    The decoder will use mask to set punctured LLRs to zero (neutral/infinite uncertainty).
    """
    if pattern.shape[0] != frame_size:
        raise ValueError(
            f"Pattern size {pattern.shape[0]} != frame_size {frame_size}"
        )

    n_punctured = int(pattern.sum())
    n_payload = frame_size - n_punctured

    if len(payload) != n_payload:
        raise ValueError(
            f"Payload length {len(payload)} != expected {n_payload} "
            f"(frame_size={frame_size} - n_punctured={n_punctured})"
        )

    frame = np.zeros(frame_size, dtype=np.uint8)
    mask = pattern.astype(np.uint8, copy=True)

    # Fill non-punctured positions with payload bits
    non_punctured_indices = np.where(pattern == 0)[0]
    frame[non_punctured_indices] = payload

    return frame, mask


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
    frame: np.ndarray,
    H: sp.csr_matrix,
) -> SyndromeBlock:
    """
    Encode a complete LDPC frame to produce syndrome.

    Parameters
    ----------
    frame : np.ndarray
        Complete LDPC frame (uint8).
    H : sp.csr_matrix
        Parity-check matrix for selected rate.

    Returns
    -------
    SyndromeBlock
        Complete syndrome data for transmission.

    Raises
    ------
    ValueError
        If frame length doesn't match H columns.
    """
    return encode_block_from_frame(frame=frame, H=H)


def encode_block_from_frame(
    frame: np.ndarray,
    H: sp.csr_matrix,
) -> SyndromeBlock:
    """Compute syndrome for a full pre-constructed LDPC frame.

    Parameters
    ----------
    frame : np.ndarray
        Full frame bits (payload + padding), uint8.
    H : sp.csr_matrix
        Parity-check matrix for the selected rate.

    Returns
    -------
    SyndromeBlock
        Syndrome data for transmission.
    """
    frame_size = int(H.shape[1])
    if int(frame.shape[0]) != frame_size:
        raise ValueError(f"Frame length {int(frame.shape[0])} != frame_size {frame_size}")

    syndrome = compute_syndrome(frame, H)
    rate = 1.0 - float(H.shape[0]) / float(frame_size)
    # For pre-constructed frames, we don't track puncture pattern
    return SyndromeBlock(
        syndrome=syndrome,
        rate=rate,
        payload_length=frame_size,
        leakage_bits=int(syndrome.shape[0]),
        puncture_pattern=np.zeros(frame_size, dtype=np.uint8),  # No puncturing
    )


def encode_block_from_payload(
    payload: np.ndarray,
    H: sp.csr_matrix,
    puncture_pattern: np.ndarray,
) -> SyndromeBlock:
    """Construct frame using untainted puncturing pattern and compute syndrome.

    Parameters
    ----------
    payload : np.ndarray
        Payload bits (uint8).
    H : sp.csr_matrix
        Parity-check matrix for the selected rate.
    puncture_pattern : np.ndarray
        Untainted puncturing pattern.

    Returns
    -------
    SyndromeBlock
        Syndrome data for transmission.
    """
    frame_size = int(H.shape[1])
    payload_len = int(payload.shape[0])
    rate = 1.0 - float(H.shape[0]) / float(frame_size)

    # Pattern-based mode: use untainted puncturing
    full_frame, mask = apply_puncture_pattern(payload, puncture_pattern, frame_size)
    
    syndrome = compute_syndrome(full_frame, H)
    return SyndromeBlock(
        syndrome=syndrome,
        rate=rate,
        payload_length=payload_len,
        leakage_bits=int(syndrome.shape[0]),
        puncture_pattern=puncture_pattern,
    )


def prepare_frame(
    key_bits: np.ndarray,
    puncture_pattern: np.ndarray,
) -> np.ndarray:
    """
    Prepare full frame from payload using untainted puncturing pattern.

    Utility function for frame construction without syndrome
    computation (used by decoder).

    Parameters
    ----------
    key_bits : np.ndarray
        Payload bits (uint8).
    puncture_pattern : np.ndarray
        Untainted puncturing pattern.

    Returns
    -------
    np.ndarray
        Full frame with payload in non-punctured positions.
    """
    pattern_frame_size = int(puncture_pattern.shape[0])
    frame, _ = apply_puncture_pattern(key_bits, puncture_pattern, pattern_frame_size)
    return frame

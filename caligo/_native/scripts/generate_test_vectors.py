#!/usr/bin/env python3
"""
Generate synthetic test vectors for Polar codec validation.

This script creates test vectors for:
1. Encoder verification (message → codeword)
2. SC decoder verification (LLR → message)

Test vectors are saved as JSON for consumption by Rust tests.

Usage:
    python generate_test_vectors.py --output tests/vectors/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class EncoderTestVector:
    """Test vector for encoder validation."""
    name: str
    block_length: int
    message_length: int
    frozen_indices: List[int]
    message: List[int]
    expected_codeword: List[int]


@dataclass
class DecoderTestVector:
    """Test vector for decoder validation."""
    name: str
    block_length: int
    message_length: int
    frozen_indices: List[int]
    channel_llr: List[float]
    expected_message: List[int]
    expected_converged: bool


def butterfly_encode(u: np.ndarray) -> np.ndarray:
    """
    Polar encoding via butterfly transform.
    
    Implements x = u · G_N where G_N = B_N · F^⊗n.
    
    Parameters
    ----------
    u : np.ndarray
        Input vector with frozen bits set to 0.
    
    Returns
    -------
    np.ndarray
        Encoded codeword.
    """
    x = u.copy()
    n = len(x)
    n_stages = int(np.log2(n))
    
    for stage in range(n_stages):
        stride = 1 << stage
        block_size = stride << 1
        
        for block_start in range(0, n, block_size):
            for i in range(stride):
                idx_a = block_start + i
                idx_b = idx_a + stride
                x[idx_a] ^= x[idx_b]
    
    return x


def compute_bhattacharyya_reliabilities(n: int, design_snr_db: float) -> np.ndarray:
    """
    Compute channel reliabilities via Bhattacharyya parameter evolution.
    
    Parameters
    ----------
    n : int
        Block length (power of 2).
    design_snr_db : float
        Design SNR in dB.
    
    Returns
    -------
    np.ndarray
        Reliability values (higher = more reliable).
    """
    n_stages = int(np.log2(n))
    snr_linear = 10 ** (design_snr_db / 10)
    z_init = min(np.exp(-snr_linear), 1 - 1e-10)
    
    z = np.full(n, z_init)
    
    for stage in range(n_stages):
        half = 1 << stage
        z_new = np.zeros(n)
        
        for i in range(n):
            pair_idx = i ^ half
            if i < pair_idx:
                z_minus = 2 * z[i] - z[i] ** 2
                z_plus = z[i] ** 2
                z_new[i] = z_minus
                z_new[pair_idx] = z_plus
        
        z = z_new
    
    # Convert to reliability: -log(Z)
    reliability = np.where(z > 0, -np.log(z), np.inf)
    reliability = np.where(z >= 1, 0, reliability)
    
    return reliability


def select_frozen_indices(n: int, k: int, design_snr_db: float) -> List[int]:
    """
    Select frozen bit indices based on channel reliability.
    
    Parameters
    ----------
    n : int
        Block length.
    k : int
        Number of information bits.
    design_snr_db : float
        Design SNR in dB.
    
    Returns
    -------
    List[int]
        Indices of frozen bit positions (n - k positions).
    """
    reliability = compute_bhattacharyya_reliabilities(n, design_snr_db)
    sorted_indices = np.argsort(reliability)
    frozen_indices = sorted(sorted_indices[:n - k].tolist())
    return frozen_indices


def generate_encoder_test_vector(
    name: str,
    n: int,
    k: int,
    design_snr_db: float,
    seed: int,
) -> EncoderTestVector:
    """Generate a single encoder test vector."""
    rng = np.random.default_rng(seed)
    
    frozen_indices = select_frozen_indices(n, k, design_snr_db)
    info_indices = [i for i in range(n) if i not in frozen_indices]
    
    # Random message
    message = rng.integers(0, 2, size=k, dtype=np.uint8)
    
    # Build u vector
    u = np.zeros(n, dtype=np.uint8)
    for idx, info_idx in enumerate(info_indices):
        u[info_idx] = message[idx]
    
    # Encode
    codeword = butterfly_encode(u)
    
    return EncoderTestVector(
        name=name,
        block_length=n,
        message_length=k,
        frozen_indices=frozen_indices,
        message=message.tolist(),
        expected_codeword=codeword.tolist(),
    )


def generate_decoder_test_vector(
    name: str,
    n: int,
    k: int,
    design_snr_db: float,
    channel_snr_db: float,
    seed: int,
) -> DecoderTestVector:
    """Generate a single decoder test vector."""
    rng = np.random.default_rng(seed)
    
    frozen_indices = select_frozen_indices(n, k, design_snr_db)
    info_indices = [i for i in range(n) if i not in frozen_indices]
    
    # Random message
    message = rng.integers(0, 2, size=k, dtype=np.uint8)
    
    # Build u vector and encode
    u = np.zeros(n, dtype=np.uint8)
    for idx, info_idx in enumerate(info_indices):
        u[info_idx] = message[idx]
    
    codeword = butterfly_encode(u)
    
    # Add AWGN noise to create channel LLRs
    # BPSK: x = 1 - 2*c, y = x + noise
    # LLR = 2y/σ² = 2(x + noise)/σ²
    snr_linear = 10 ** (channel_snr_db / 10)
    noise_std = 1.0 / np.sqrt(2 * snr_linear)
    
    bpsk = 1 - 2 * codeword.astype(np.float32)
    noise = rng.normal(0, noise_std, size=n).astype(np.float32)
    received = bpsk + noise
    
    # LLR = 2 * received / noise_variance = 2 * received * snr_linear
    llr = (2 * received * snr_linear).tolist()
    
    return DecoderTestVector(
        name=name,
        block_length=n,
        message_length=k,
        frozen_indices=frozen_indices,
        channel_llr=llr,
        expected_message=message.tolist(),
        expected_converged=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Polar codec test vectors")
    parser.add_argument("--output", type=Path, default=Path("tests/vectors"))
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Encoder test vectors
    encoder_vectors = [
        generate_encoder_test_vector("enc_n8_k4_seed0", 8, 4, 2.0, seed=0),
        generate_encoder_test_vector("enc_n8_k4_seed1", 8, 4, 2.0, seed=1),
        generate_encoder_test_vector("enc_n1024_k512_seed0", 1024, 512, 2.0, seed=0),
        generate_encoder_test_vector("enc_n1024_k512_seed1", 1024, 512, 2.0, seed=1),
        generate_encoder_test_vector("enc_n4096_k2048_seed0", 4096, 2048, 2.0, seed=0),
    ]
    
    with open(args.output / "encoder_vectors.json", "w") as f:
        json.dump([asdict(v) for v in encoder_vectors], f, indent=2)
    
    print(f"Generated {len(encoder_vectors)} encoder test vectors")
    
    # Decoder test vectors (high SNR for reliable decoding)
    decoder_vectors = [
        generate_decoder_test_vector("dec_n8_k4_highsnr", 8, 4, 2.0, 10.0, seed=0),
        generate_decoder_test_vector("dec_n1024_k512_highsnr", 1024, 512, 2.0, 5.0, seed=0),
        generate_decoder_test_vector("dec_n1024_k512_medsnr", 1024, 512, 2.0, 2.0, seed=0),
        generate_decoder_test_vector("dec_n4096_k2048_highsnr", 4096, 2048, 2.0, 4.0, seed=0),
    ]
    
    with open(args.output / "decoder_vectors.json", "w") as f:
        json.dump([asdict(v) for v in decoder_vectors], f, indent=2)
    
    print(f"Generated {len(decoder_vectors)} decoder test vectors")
    print(f"Test vectors saved to {args.output}")


if __name__ == "__main__":
    main()

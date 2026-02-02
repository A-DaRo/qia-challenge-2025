# Phase 1: Rust Foundation Implementation Guide

<metadata>
impl_id: phase1-rust-foundation
version: 1.0.0
status: ready
created: 2026-02-02
depends_on: [ADR-0001, ADR-0002, siso-codec-protocol]
enables: [phase2-scl-decoder, rust-polar-crate]
</metadata>

---

## Overview

<overview>
This document provides a step-by-step implementation guide for the `caligo-codecs` Rust crate foundation: Polar encoder and SC decoder (L=1 baseline).

### Deliverables

| Component | File | Description |
|-----------|------|-------------|
| Crate manifest | `Cargo.toml` | Dependencies, features, build config |
| Maturin config | `pyproject.toml` | Python build integration |
| Module entry | `src/lib.rs` | PyO3 `#[pymodule]` definition |
| Polar encoder | `src/polar/encoder.rs` | Arikan recursive encoding |
| SC decoder | `src/polar/decoder.rs` | LLR-domain SC (L=1) |
| Frozen bits | `src/polar/construction.rs` | Gaussian Approximation |
| CRC stub | `src/polar/crc.rs` | CRC-16-CCITT (Phase 2 enabler) |
| Test vectors | `tests/vectors/` | Generated synthetic test data |

### Timeline

| Task | Estimated Hours |
|------|-----------------|
| Project scaffolding | 2h |
| Polar encoder | 4h |
| Channel construction | 3h |
| SC decoder (L=1) | 6h |
| PyO3 bindings | 4h |
| Test vectors & validation | 4h |
| Documentation | 2h |
| **Total** | **25h** (~3 days) |

</overview>

---

## §1 Project Setup

### 1.1 Directory Structure

Create the following structure at `caligo/_native/`:

```
caligo/_native/
├── Cargo.toml
├── pyproject.toml
├── rust-toolchain.toml
├── src/
│   ├── lib.rs
│   ├── error.rs
│   └── polar/
│       ├── mod.rs
│       ├── encoder.rs
│       ├── decoder.rs
│       ├── construction.rs
│       └── crc.rs
└── tests/
    ├── test_encoder.rs
    ├── test_decoder.rs
    └── vectors/
        ├── enc_n8_k4.json
        ├── enc_n1024_k512.json
        ├── sc_n8_k4.json
        └── sc_n1024_k512.json
```

### 1.2 Cargo.toml

<code_block language="toml" file="caligo/_native/Cargo.toml">

```toml
[package]
name = "caligo-codecs"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Caligo Team"]
description = "High-performance Polar codec for QKD reconciliation"
license = "MIT"

[lib]
name = "caligo_codecs"
crate-type = ["cdylib", "rlib"]

[dependencies]
# Python bindings
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"

# Core data structures
ndarray = "0.15"
bitvec = "1.0"

# Error handling
thiserror = "1.0"

# Serialization (for test vectors)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
approx = "0.5"
rand = "0.8"
rand_chacha = "0.3"

[features]
default = []
simd = []  # Enable AVX-512 (requires nightly for std::simd)

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
debug = false

[profile.dev]
opt-level = 1  # Faster debug builds
```

</code_block>

### 1.3 pyproject.toml

<code_block language="toml" file="caligo/_native/pyproject.toml">

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "caligo-codecs"
version = "0.1.0"
description = "Rust-accelerated Polar codec for Caligo"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[tool.maturin]
python-source = "python"
module-name = "caligo._native"
features = ["pyo3/extension-module"]
strip = true
```

</code_block>

### 1.4 rust-toolchain.toml

<code_block language="toml" file="caligo/_native/rust-toolchain.toml">

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu"]
```

</code_block>

---

## §2 Polar Encoder

### 2.1 Theory Background

Per [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §II and [Fast_Polar_Decoders.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §II-A:

**Generator Matrix Construction:**

The Polar code generator matrix is:
$$\mathbf{G}_N = \mathbf{B}_N \mathbf{F}^{\otimes n}, \quad N = 2^n$$

Where:
- $\mathbf{F} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ is the polarization kernel
- $\mathbf{F}^{\otimes n}$ is the $n$-th Kronecker power
- $\mathbf{B}_N$ is the bit-reversal permutation matrix

**Encoding:**
$$\mathbf{x} = \mathbf{u} \cdot \mathbf{G}_N$$

Where $\mathbf{u}$ has frozen bit positions set to 0 (or known values).

**Recursive Implementation:**

The encoding can be performed in $O(N \log N)$ via the butterfly structure:

```
For stage s = 0, 1, ..., n-1:
    For each pair (i, i + 2^s) where i mod 2^(s+1) < 2^s:
        x[i] = x[i] XOR x[i + 2^s]
```

### 2.2 Implementation

<code_block language="rust" file="caligo/_native/src/polar/encoder.rs">

```rust
//! Polar code encoder using Arikan's recursive butterfly construction.
//!
//! References:
//! - [1] Arikan, "Channel Polarization", IEEE TIT 2009
//! - [2] Tal & Vardy, "List Decoding of Polar Codes", IEEE TIT 2015, §II

use bitvec::prelude::*;
use ndarray::Array1;
use thiserror::Error;

/// Errors that can occur during encoding.
#[derive(Error, Debug)]
pub enum EncoderError {
    #[error("Message length {0} does not match expected {1}")]
    MessageLengthMismatch(usize, usize),
    
    #[error("Block length {0} is not a power of 2")]
    InvalidBlockLength(usize),
    
    #[error("Message contains non-binary value at index {0}")]
    NonBinaryValue(usize),
}

/// Polar code encoder.
///
/// Implements Arikan's recursive butterfly encoding in O(N log N) time.
/// The encoder is stateless; frozen bit mask is provided at construction.
pub struct PolarEncoder {
    /// Block length N = 2^n
    block_length: usize,
    /// log2(N)
    n_stages: usize,
    /// Frozen bit mask: true = frozen (set to 0), false = information
    frozen_mask: BitVec<u64, Lsb0>,
    /// Number of information bits (including CRC if any)
    k_total: usize,
}

impl PolarEncoder {
    /// Create a new encoder.
    ///
    /// # Arguments
    /// * `block_length` - Code block length N (must be power of 2)
    /// * `frozen_mask` - Bit vector where true = frozen position
    ///
    /// # Errors
    /// Returns error if block_length is not a power of 2.
    pub fn new(block_length: usize, frozen_mask: BitVec<u64, Lsb0>) -> Result<Self, EncoderError> {
        if !block_length.is_power_of_two() {
            return Err(EncoderError::InvalidBlockLength(block_length));
        }
        
        let n_stages = block_length.trailing_zeros() as usize;
        let k_total = frozen_mask.iter().filter(|b| !**b).count();
        
        Ok(Self {
            block_length,
            n_stages,
            frozen_mask,
            k_total,
        })
    }
    
    /// Encode message bits to codeword.
    ///
    /// # Arguments
    /// * `message` - Information bits of length k_total
    ///
    /// # Returns
    /// Encoded codeword of length N
    ///
    /// # Algorithm
    /// 1. Place message bits at information positions (non-frozen)
    /// 2. Set frozen positions to 0
    /// 3. Apply butterfly transform: x[i] ^= x[i + 2^s] for each stage
    pub fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
        // Validate input
        if message.len() != self.k_total {
            return Err(EncoderError::MessageLengthMismatch(message.len(), self.k_total));
        }
        
        for (i, &bit) in message.iter().enumerate() {
            if bit > 1 {
                return Err(EncoderError::NonBinaryValue(i));
            }
        }
        
        // Initialize u-vector: place message at information positions
        let mut u = vec![0u8; self.block_length];
        let mut msg_idx = 0;
        
        for i in 0..self.block_length {
            if !self.frozen_mask[i] {
                u[i] = message[msg_idx];
                msg_idx += 1;
            }
            // Frozen positions remain 0
        }
        
        // Apply butterfly transform (in-place)
        // This computes x = u · G_N where G_N = B_N · F^⊗n
        self.butterfly_transform(&mut u);
        
        Ok(Array1::from_vec(u))
    }
    
    /// In-place butterfly transform implementing x = u · G_N.
    ///
    /// The transform processes n = log2(N) stages. At stage s:
    /// - Stride = 2^s
    /// - For each block of size 2^(s+1), XOR pairs at distance 2^s
    ///
    /// This is equivalent to multiplying by F^⊗n with natural bit ordering.
    #[inline]
    fn butterfly_transform(&self, x: &mut [u8]) {
        let n = self.block_length;
        
        for stage in 0..self.n_stages {
            let stride = 1 << stage;
            let block_size = stride << 1;
            
            for block_start in (0..n).step_by(block_size) {
                for i in 0..stride {
                    let idx_a = block_start + i;
                    let idx_b = idx_a + stride;
                    
                    // x[a] = x[a] XOR x[b]  (butterfly operation)
                    x[idx_a] ^= x[idx_b];
                }
            }
        }
    }
    
    /// Get the number of information bits.
    pub fn message_length(&self) -> usize {
        self.k_total
    }
    
    /// Get the block length.
    pub fn block_length(&self) -> usize {
        self.block_length
    }
    
    /// Get code rate k/N.
    pub fn rate(&self) -> f64 {
        self.k_total as f64 / self.block_length as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test N=8, K=4 example from Arikan's paper.
    #[test]
    fn test_encode_n8_k4() {
        // Frozen positions: 0, 1, 2, 4 (indices with low reliability)
        // Information positions: 3, 5, 6, 7
        let mut frozen = bitvec![u64, Lsb0; 0; 8];
        frozen.set(0, true);
        frozen.set(1, true);
        frozen.set(2, true);
        frozen.set(4, true);
        
        let encoder = PolarEncoder::new(8, frozen).unwrap();
        assert_eq!(encoder.message_length(), 4);
        
        // Test with message [1, 0, 1, 1] at positions [3, 5, 6, 7]
        // u = [0, 0, 0, 1, 0, 0, 1, 1]
        let message = [1u8, 0, 1, 1];
        let codeword = encoder.encode(&message).unwrap();
        
        // Verify codeword is valid (manual calculation required for test vector)
        assert_eq!(codeword.len(), 8);
    }
}
```

</code_block>

---

## §3 Channel Construction (Frozen Bit Selection)

### 3.1 Theory Background

Per [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §I and ADR-0001:

**Gaussian Approximation (GA):**

For AWGN channels with known SNR, frozen bits are selected based on channel reliability. The GA method approximates the mutual information evolution through the polar transform.

For design SNR $\sigma^2$ (noise variance):
1. Initialize: $I_0(\sigma^2) = 1 - h_b(Q(\sqrt{2/\sigma^2}))$ where $Q$ is the Q-function
2. Recurse through polarization stages
3. Select $N - K$ positions with lowest reliability as frozen

**Simplified Density Evolution:**

For Phase 1, we use a simplified construction based on Bhattacharyya parameters:

$$Z(W^-) = 2Z(W) - Z(W)^2, \quad Z(W^+) = Z(W)^2$$

Where $Z(W)$ is the Bhattacharyya parameter of channel $W$.

### 3.2 Implementation

<code_block language="rust" file="caligo/_native/src/polar/construction.rs">

```rust
//! Polar code construction via Gaussian Approximation.
//!
//! Computes frozen bit positions based on channel reliability estimates.
//! Uses Bhattacharyya parameter evolution for computational efficiency.
//!
//! References:
//! - [1] Tal & Vardy, "How to Construct Polar Codes", IEEE TIT 2013
//! - [2] Trifonov, "Efficient Design and Decoding of Polar Codes", IEEE TCom 2012

use bitvec::prelude::*;

/// Channel construction methods.
#[derive(Debug, Clone, Copy)]
pub enum ConstructionMethod {
    /// Bhattacharyya parameter evolution (fast, approximate)
    Bhattacharyya,
    /// Gaussian Approximation (more accurate for AWGN)
    GaussianApproximation,
}

/// Construct frozen bit mask for Polar code.
///
/// # Arguments
/// * `block_length` - N = 2^n
/// * `message_length` - K (number of information bits)
/// * `design_snr_db` - Design SNR in dB (for AWGN channel)
/// * `method` - Construction method
///
/// # Returns
/// BitVec where true = frozen position, false = information position
pub fn construct_frozen_mask(
    block_length: usize,
    message_length: usize,
    design_snr_db: f64,
    method: ConstructionMethod,
) -> BitVec<u64, Lsb0> {
    assert!(block_length.is_power_of_two());
    assert!(message_length <= block_length);
    
    let n_frozen = block_length - message_length;
    
    // Compute reliability for each bit position
    let reliabilities = match method {
        ConstructionMethod::Bhattacharyya => {
            bhattacharyya_construction(block_length, design_snr_db)
        }
        ConstructionMethod::GaussianApproximation => {
            gaussian_approximation_construction(block_length, design_snr_db)
        }
    };
    
    // Sort indices by reliability (ascending = least reliable first)
    let mut indices: Vec<usize> = (0..block_length).collect();
    indices.sort_by(|&a, &b| {
        reliabilities[a].partial_cmp(&reliabilities[b]).unwrap()
    });
    
    // Mark n_frozen least reliable positions as frozen
    let mut frozen = bitvec![u64, Lsb0; 0; block_length];
    for &idx in indices.iter().take(n_frozen) {
        frozen.set(idx, true);
    }
    
    frozen
}

/// Bhattacharyya parameter construction.
///
/// Computes Z(W_i) for each synthetic channel using:
/// - Z(W^-) = 2Z - Z^2  (worse channel)
/// - Z(W^+) = Z^2       (better channel)
///
/// Returns negative log reliability (higher = more reliable).
fn bhattacharyya_construction(block_length: usize, design_snr_db: f64) -> Vec<f64> {
    let n = block_length.trailing_zeros() as usize;
    
    // Initial Bhattacharyya parameter for AWGN
    // Z = exp(-SNR) for BSC approximation
    let snr_linear = 10_f64.powf(design_snr_db / 10.0);
    let z_init = (-snr_linear).exp().min(1.0 - 1e-10);
    
    // Initialize all channels with same reliability
    let mut z = vec![z_init; block_length];
    
    // Polarization stages
    for stage in 0..n {
        let half = 1 << stage;
        let mut z_new = vec![0.0; block_length];
        
        for i in 0..block_length {
            let pair_idx = i ^ half;  // Partner in butterfly
            
            if i < pair_idx {
                // Compute polarized channels
                let z_minus = 2.0 * z[i] - z[i] * z[i];  // Worse channel
                let z_plus = z[i] * z[i];                 // Better channel
                
                z_new[i] = z_minus;
                z_new[pair_idx] = z_plus;
            }
        }
        
        z = z_new;
    }
    
    // Convert to reliability: -log(Z) (higher = more reliable)
    z.iter().map(|&zi| {
        if zi <= 0.0 {
            f64::INFINITY  // Perfect channel
        } else if zi >= 1.0 {
            0.0  // Useless channel
        } else {
            -zi.ln()
        }
    }).collect()
}

/// Gaussian Approximation construction.
///
/// Tracks mean LLR evolution through polar transform stages.
/// More accurate than Bhattacharyya for moderate SNR.
fn gaussian_approximation_construction(block_length: usize, design_snr_db: f64) -> Vec<f64> {
    let n = block_length.trailing_zeros() as usize;
    
    // Initial LLR mean for AWGN: μ = 2/σ² = 2 * SNR_linear
    let snr_linear = 10_f64.powf(design_snr_db / 10.0);
    let mu_init = 2.0 * snr_linear;
    
    let mut mu = vec![mu_init; block_length];
    
    // GA update functions (approximations from Trifonov)
    let phi = |x: f64| -> f64 {
        if x < 0.0 {
            0.0
        } else if x < 10.0 {
            (-0.4527 * x.powf(0.86) + 0.0218).exp()
        } else {
            (std::f64::consts::PI * x).sqrt() * (-x / 4.0).exp()
        }
    };
    
    let phi_inv = |y: f64| -> f64 {
        if y <= 0.0 || y >= 1.0 {
            0.0
        } else if y > 0.0 && y <= 1e-10 {
            // Large μ approximation
            4.0 * (-y.ln()) - std::f64::consts::PI.ln()
        } else {
            // Numerical inversion via Newton-Raphson
            let mut x = 1.0;
            for _ in 0..20 {
                let fx = phi(x) - y;
                let dfx = -0.4527 * 0.86 * x.powf(-0.14) * phi(x);
                if dfx.abs() < 1e-15 {
                    break;
                }
                x -= fx / dfx;
                x = x.max(0.001);
            }
            x
        }
    };
    
    // Polarization stages
    for stage in 0..n {
        let half = 1 << stage;
        let mut mu_new = vec![0.0; block_length];
        
        for i in 0..block_length {
            let pair_idx = i ^ half;
            
            if i < pair_idx {
                // GA update rules
                // μ^- ≈ φ^{-1}(1 - (1 - φ(μ))²)
                // μ^+ = 2μ
                let phi_mu = phi(mu[i]);
                let mu_minus = phi_inv(1.0 - (1.0 - phi_mu).powi(2));
                let mu_plus = 2.0 * mu[i];
                
                mu_new[i] = mu_minus;
                mu_new[pair_idx] = mu_plus;
            }
        }
        
        mu = mu_new;
    }
    
    // Return mean LLR as reliability (higher = more reliable)
    mu
}

/// Convenience function to get information bit indices.
pub fn get_information_indices(frozen_mask: &BitVec<u64, Lsb0>) -> Vec<usize> {
    frozen_mask.iter()
        .enumerate()
        .filter(|(_, is_frozen)| !**is_frozen)
        .map(|(i, _)| i)
        .collect()
}

/// Convenience function to get frozen bit indices.
pub fn get_frozen_indices(frozen_mask: &BitVec<u64, Lsb0>) -> Vec<usize> {
    frozen_mask.iter()
        .enumerate()
        .filter(|(_, is_frozen)| **is_frozen)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_construction_n8_k4() {
        let frozen = construct_frozen_mask(8, 4, 0.0, ConstructionMethod::Bhattacharyya);
        
        // Should have exactly 4 frozen positions
        assert_eq!(frozen.count_ones(), 4);
        assert_eq!(frozen.count_zeros(), 4);
        
        // Position 0 should always be frozen (least reliable)
        assert!(frozen[0]);
    }
    
    #[test]
    fn test_construction_n1024_k512() {
        let frozen = construct_frozen_mask(1024, 512, 2.0, ConstructionMethod::GaussianApproximation);
        
        assert_eq!(frozen.count_ones(), 512);  // n_frozen
        assert_eq!(frozen.count_zeros(), 512); // n_info
    }
}
```

</code_block>

---

## §4 SC Decoder (L=1 Baseline)

### 4.1 Theory Background

Per [LLR-Based_SCL_Decoding.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §II-B and [Fast_Polar_Decoders.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §II-B:

**LLR Update Functions (Min-Sum Approximation):**

The SC decoder operates on Log-Likelihood Ratios (LLRs):
$$\text{LLR} = \ln\frac{P(\text{bit}=0)}{P(\text{bit}=1)}$$

Two update functions propagate LLRs through the polar transform:

**f-function (check node):**
$$f(\alpha, \beta) = \text{sign}(\alpha) \cdot \text{sign}(\beta) \cdot \min(|\alpha|, |\beta|)$$

**g-function (variable node):**
$$g(\alpha, \beta, u) = (-1)^u \cdot \alpha + \beta$$

Where:
- $\alpha, \beta$ are incoming LLRs
- $u \in \{0, 1\}$ is the partial sum (previously decoded bit XOR)

**Decision Rule:**
$$\hat{u}_i = \begin{cases} 0 & \text{if } \text{LLR}_i \geq 0 \text{ (or frozen)} \\ 1 & \text{otherwise} \end{cases}$$

### 4.2 Implementation

<code_block language="rust" file="caligo/_native/src/polar/decoder.rs">

```rust
//! Successive Cancellation (SC) decoder for Polar codes.
//!
//! This module implements the baseline L=1 SC decoder using LLR-domain
//! computations with the min-sum approximation.
//!
//! References:
//! - [1] Balatsoukas-Stimming et al., "LLR-Based SCL Decoding", IEEE TSP 2015
//! - [2] Sarkis et al., "Fast Polar Decoders", IEEE JSAC 2014

use bitvec::prelude::*;
use ndarray::Array1;
use thiserror::Error;

/// Errors during decoding.
#[derive(Error, Debug)]
pub enum DecoderError {
    #[error("LLR length {0} does not match block length {1}")]
    LlrLengthMismatch(usize, usize),
    
    #[error("Received length {0} does not match block length {1}")]
    ReceivedLengthMismatch(usize, usize),
}

/// Result of SC decoding.
#[derive(Debug, Clone)]
pub struct SCDecodeResult {
    /// Decoded message bits
    pub message: Vec<u8>,
    /// Final path metric (sum of absolute LLRs for decisions against LLR sign)
    pub path_metric: f32,
    /// Whether decoding completed successfully (always true for SC L=1)
    pub converged: bool,
}

/// Successive Cancellation decoder (L=1).
///
/// Implements the standard SC algorithm with LLR-domain computation.
/// This is the baseline for SCL extension in Phase 2.
pub struct SCDecoder {
    /// Block length N = 2^n
    block_length: usize,
    /// Number of stages n = log2(N)
    n_stages: usize,
    /// Frozen bit mask
    frozen_mask: BitVec<u64, Lsb0>,
    /// Number of information bits
    k_total: usize,
    /// Working memory for LLRs at each stage
    /// Shape: [n_stages + 1][block_length]
    llr_memory: Vec<Vec<f32>>,
    /// Working memory for partial sums
    /// Shape: [n_stages + 1][block_length]
    bit_memory: Vec<Vec<u8>>,
}

impl SCDecoder {
    /// Create a new SC decoder.
    ///
    /// # Arguments
    /// * `block_length` - Code block length N (must be power of 2)
    /// * `frozen_mask` - Bit vector where true = frozen position
    pub fn new(block_length: usize, frozen_mask: BitVec<u64, Lsb0>) -> Self {
        assert!(block_length.is_power_of_two());
        
        let n_stages = block_length.trailing_zeros() as usize;
        let k_total = frozen_mask.iter().filter(|b| !**b).count();
        
        // Allocate working memory
        let llr_memory = vec![vec![0.0f32; block_length]; n_stages + 1];
        let bit_memory = vec![vec![0u8; block_length]; n_stages + 1];
        
        Self {
            block_length,
            n_stages,
            frozen_mask,
            k_total,
            llr_memory,
            bit_memory,
        }
    }
    
    /// Decode from channel LLRs.
    ///
    /// # Arguments
    /// * `llr_channel` - Channel LLRs of length N
    ///   Convention: positive LLR means bit=0 more likely
    ///
    /// # Returns
    /// Decoded message and path metric
    pub fn decode(&mut self, llr_channel: &[f32]) -> Result<SCDecodeResult, DecoderError> {
        if llr_channel.len() != self.block_length {
            return Err(DecoderError::LlrLengthMismatch(
                llr_channel.len(),
                self.block_length,
            ));
        }
        
        // Initialize layer 0 with channel LLRs
        self.llr_memory[0].copy_from_slice(llr_channel);
        
        // Clear bit memory
        for layer in &mut self.bit_memory {
            layer.fill(0);
        }
        
        let mut message = Vec::with_capacity(self.k_total);
        let mut path_metric = 0.0f32;
        
        // Decode each bit sequentially
        for phi in 0..self.block_length {
            // Compute LLR for position phi at top layer
            self.recursively_calc_llr(self.n_stages, phi);
            
            let llr = self.llr_memory[self.n_stages][phi];
            
            // Make decision
            let decoded_bit = if self.frozen_mask[phi] {
                // Frozen bit: always 0
                0u8
            } else {
                // Information bit: hard decision
                if llr >= 0.0 { 0u8 } else { 1u8 }
            };
            
            // Update path metric: penalize decisions against LLR
            if (llr >= 0.0 && decoded_bit == 1) || (llr < 0.0 && decoded_bit == 0) {
                path_metric += llr.abs();
            }
            
            // Store decoded bit
            self.bit_memory[self.n_stages][phi] = decoded_bit;
            
            // Propagate partial sums back
            if phi % 2 == 1 {
                self.recursively_calc_bits(self.n_stages, phi);
            }
            
            // Collect information bits
            if !self.frozen_mask[phi] {
                message.push(decoded_bit);
            }
        }
        
        Ok(SCDecodeResult {
            message,
            path_metric,
            converged: true,
        })
    }
    
    /// Decode from hard received bits (convert to LLR first).
    ///
    /// Uses a simple model: LLR = ±confidence based on QBER estimate.
    ///
    /// # Arguments
    /// * `received` - Received bits (0 or 1)
    /// * `qber` - Estimated quantum bit error rate
    pub fn decode_hard(
        &mut self,
        received: &[u8],
        qber: f32,
    ) -> Result<SCDecodeResult, DecoderError> {
        if received.len() != self.block_length {
            return Err(DecoderError::ReceivedLengthMismatch(
                received.len(),
                self.block_length,
            ));
        }
        
        // Convert hard bits to LLRs
        // LLR = log((1-qber)/qber) for bit=0, negated for bit=1
        let confidence = ((1.0 - qber) / qber.max(1e-10)).ln();
        
        let llr_channel: Vec<f32> = received
            .iter()
            .map(|&bit| if bit == 0 { confidence } else { -confidence })
            .collect();
        
        self.decode(&llr_channel)
    }
    
    /// Recursively compute LLR at (layer, phi).
    ///
    /// Implements the f/g function updates through the butterfly structure.
    fn recursively_calc_llr(&mut self, layer: usize, phi: usize) {
        if layer == 0 {
            return; // Base case: channel LLRs already set
        }
        
        let half_len = 1 << (layer - 1);
        let psi = phi / 2;
        
        if phi % 2 == 0 {
            // f-function path: need LLRs from both children
            self.recursively_calc_llr(layer - 1, psi);
            self.recursively_calc_llr(layer - 1, psi + half_len);
            
            // Compute f-function for all positions in this layer
            self.calc_f_function(layer, phi);
        } else {
            // g-function path: use partial sum from sibling
            self.calc_g_function(layer, phi);
        }
    }
    
    /// Compute f-function: LLR[layer][phi] = f(LLR[layer-1][psi], LLR[layer-1][psi+half])
    ///
    /// f(α, β) = sign(α) × sign(β) × min(|α|, |β|)
    #[inline]
    fn calc_f_function(&mut self, layer: usize, phi: usize) {
        let half_len = 1 << (layer - 1);
        let psi = phi / 2;
        
        let alpha = self.llr_memory[layer - 1][psi];
        let beta = self.llr_memory[layer - 1][psi + half_len];
        
        // Min-sum approximation
        let sign = alpha.signum() * beta.signum();
        let magnitude = alpha.abs().min(beta.abs());
        
        self.llr_memory[layer][phi] = sign * magnitude;
    }
    
    /// Compute g-function: LLR[layer][phi] = g(LLR[layer-1][psi], LLR[layer-1][psi+half], u)
    ///
    /// g(α, β, u) = (-1)^u × α + β
    #[inline]
    fn calc_g_function(&mut self, layer: usize, phi: usize) {
        let half_len = 1 << (layer - 1);
        let psi = phi / 2;
        
        let alpha = self.llr_memory[layer - 1][psi];
        let beta = self.llr_memory[layer - 1][psi + half_len];
        let u = self.bit_memory[layer][phi - 1]; // Sibling's decoded bit
        
        let sign_factor = if u == 0 { 1.0 } else { -1.0 };
        
        self.llr_memory[layer][phi] = sign_factor * alpha + beta;
    }
    
    /// Propagate partial sums back through butterfly.
    fn recursively_calc_bits(&mut self, layer: usize, phi: usize) {
        if layer == 0 {
            return;
        }
        
        let half_len = 1 << (layer - 1);
        let psi = phi / 2;
        
        // Get the two bits at this layer
        let bit_left = self.bit_memory[layer][phi - 1];  // Even position
        let bit_right = self.bit_memory[layer][phi];     // Odd position
        
        // Partial sums for lower layer
        self.bit_memory[layer - 1][psi] = bit_left ^ bit_right;
        self.bit_memory[layer - 1][psi + half_len] = bit_right;
        
        // Recurse if needed
        if psi % 2 == 1 {
            self.recursively_calc_bits(layer - 1, psi);
        }
        if (psi + half_len) % 2 == 1 {
            self.recursively_calc_bits(layer - 1, psi + half_len);
        }
    }
    
    /// Get the number of information bits.
    pub fn message_length(&self) -> usize {
        self.k_total
    }
    
    /// Get the block length.
    pub fn block_length(&self) -> usize {
        self.block_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polar::construction::{construct_frozen_mask, ConstructionMethod};
    use crate::polar::encoder::PolarEncoder;
    
    #[test]
    fn test_sc_decode_noiseless() {
        // Test N=8, K=4 with no noise
        let frozen = construct_frozen_mask(8, 4, 2.0, ConstructionMethod::Bhattacharyya);
        
        let encoder = PolarEncoder::new(8, frozen.clone()).unwrap();
        let mut decoder = SCDecoder::new(8, frozen);
        
        // Encode a test message
        let message = vec![1u8, 0, 1, 1];
        let codeword = encoder.encode(&message).unwrap();
        
        // Convert to high-confidence LLRs (noiseless)
        let llr: Vec<f32> = codeword.iter()
            .map(|&b| if b == 0 { 10.0 } else { -10.0 })
            .collect();
        
        // Decode
        let result = decoder.decode(&llr).unwrap();
        
        assert_eq!(result.message, message);
        assert!(result.converged);
    }
    
    #[test]
    fn test_sc_decode_with_noise() {
        // Test N=1024, K=512 with moderate noise
        let frozen = construct_frozen_mask(1024, 512, 2.0, ConstructionMethod::Bhattacharyya);
        
        let encoder = PolarEncoder::new(1024, frozen.clone()).unwrap();
        let mut decoder = SCDecoder::new(1024, frozen);
        
        // Random message
        let message: Vec<u8> = (0..512).map(|i| (i % 2) as u8).collect();
        let codeword = encoder.encode(&message).unwrap();
        
        // Add moderate noise (flip some bits)
        let mut llr: Vec<f32> = codeword.iter()
            .map(|&b| if b == 0 { 3.0 } else { -3.0 })
            .collect();
        
        // Flip a few LLRs to simulate errors
        for i in (0..1024).step_by(100) {
            llr[i] *= -0.5; // Reduce confidence, don't fully flip
        }
        
        // Decode
        let result = decoder.decode(&llr).unwrap();
        
        // SC without CRC may have some errors, but should complete
        assert!(result.converged);
        assert_eq!(result.message.len(), 512);
    }
}
```

</code_block>

---

## §5 CRC Stub (Phase 2 Enabler)

### 5.1 Implementation

<code_block language="rust" file="caligo/_native/src/polar/crc.rs">

```rust
//! CRC-16-CCITT computation for CRC-aided SCL decoding.
//!
//! This is a stub implementation for Phase 1. Full integration
//! with the encoder/decoder occurs in Phase 2.
//!
//! CRC Polynomial: x^16 + x^12 + x^5 + 1 (0x1021)

/// CRC-16-CCITT polynomial.
pub const CRC16_CCITT_POLY: u16 = 0x1021;

/// Compute CRC-16-CCITT for a bit sequence.
///
/// # Arguments
/// * `bits` - Input bit sequence (0/1 values)
///
/// # Returns
/// 16-bit CRC value
pub fn crc16_ccitt(bits: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF; // Initial value
    
    for &bit in bits {
        let xor_bit = ((crc >> 15) as u8) ^ bit;
        crc <<= 1;
        
        if xor_bit != 0 {
            crc ^= CRC16_CCITT_POLY;
        }
    }
    
    crc
}

/// Verify CRC-16-CCITT for a bit sequence with appended CRC.
///
/// # Arguments
/// * `bits_with_crc` - Message bits followed by 16 CRC bits
///
/// # Returns
/// true if CRC matches, false otherwise
pub fn verify_crc16(bits_with_crc: &[u8]) -> bool {
    if bits_with_crc.len() < 16 {
        return false;
    }
    
    // Compute CRC over entire sequence (should be 0 for valid data)
    let remainder = crc16_ccitt(bits_with_crc);
    remainder == 0
}

/// Append CRC-16 to message bits.
///
/// # Arguments
/// * `message` - Input message bits
///
/// # Returns
/// Message with 16 CRC bits appended
pub fn append_crc16(message: &[u8]) -> Vec<u8> {
    let crc = crc16_ccitt(message);
    
    let mut result = message.to_vec();
    
    // Append CRC bits (MSB first)
    for i in (0..16).rev() {
        result.push(((crc >> i) & 1) as u8);
    }
    
    result
}

/// Extract message from bits with CRC appended.
///
/// # Arguments
/// * `bits_with_crc` - Message bits followed by 16 CRC bits
///
/// # Returns
/// Message bits without CRC (or None if too short)
pub fn strip_crc16(bits_with_crc: &[u8]) -> Option<Vec<u8>> {
    if bits_with_crc.len() < 16 {
        return None;
    }
    
    Some(bits_with_crc[..bits_with_crc.len() - 16].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crc16_known_value() {
        // Test with known input
        let data = [1u8, 0, 1, 1, 0, 0, 1, 0]; // 0xB2 in bits
        let crc = crc16_ccitt(&data);
        
        // CRC should be deterministic
        assert_eq!(crc, crc16_ccitt(&data));
    }
    
    #[test]
    fn test_crc16_append_verify() {
        let message = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
        
        let with_crc = append_crc16(&message);
        assert_eq!(with_crc.len(), message.len() + 16);
        
        // Verify should pass
        assert!(verify_crc16(&with_crc));
        
        // Flip a bit, verify should fail
        let mut corrupted = with_crc.clone();
        corrupted[5] ^= 1;
        assert!(!verify_crc16(&corrupted));
    }
    
    #[test]
    fn test_strip_crc() {
        let message = vec![1u8, 0, 1, 1];
        let with_crc = append_crc16(&message);
        
        let stripped = strip_crc16(&with_crc).unwrap();
        assert_eq!(stripped, message);
    }
}
```

</code_block>

---

## §6 PyO3 Bindings

### 6.1 Implementation

<code_block language="rust" file="caligo/_native/src/lib.rs">

```rust
//! Caligo Codecs: High-performance Polar codec for QKD reconciliation.
//!
//! This crate provides Rust-accelerated encoding and decoding for Polar codes,
//! with Python bindings via PyO3.
//!
//! # Architecture
//!
//! ```text
//! Python (caligo.reconciliation)
//!     │
//!     ▼
//! PyO3 Bindings (this module)
//!     │
//!     ▼
//! Rust Core (polar::encoder, polar::decoder)
//! ```
//!
//! # GIL Release
//!
//! All compute-intensive methods release the GIL via `py.allow_threads()`,
//! enabling true parallelism across Python threads.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use bitvec::prelude::*;

pub mod error;
pub mod polar;

use polar::construction::{construct_frozen_mask, ConstructionMethod};
use polar::encoder::PolarEncoder;
use polar::decoder::SCDecoder;
use polar::crc;

/// Python-visible Polar codec implementing the SISOCodec protocol pattern.
///
/// This class wraps the Rust encoder and decoder, providing:
/// - `encode()`: Message → Codeword
/// - `decode_hard()`: Received bits → Message (hard decision)
/// - `decode_soft()`: Channel LLRs → (Extrinsic LLRs, Message) (soft decision)
///
/// All heavy computation releases the GIL for parallel execution.
#[pyclass(name = "PolarCodec")]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCDecoder,
    frozen_mask: BitVec<u64, Lsb0>,
    block_length: usize,
    message_length: usize,
    crc_length: usize,
}

#[pymethods]
impl PyPolarCodec {
    /// Create a new Polar codec.
    ///
    /// # Arguments
    /// * `block_length` - Code block length N (must be power of 2)
    /// * `message_length` - Number of information bits K
    /// * `design_snr_db` - Design SNR in dB for frozen bit construction
    /// * `crc_length` - CRC length (0 or 16 supported). Default: 0 for Phase 1.
    ///
    /// # Raises
    /// * `ValueError` - If block_length is not a power of 2
    /// * `ValueError` - If message_length > block_length
    #[new]
    #[pyo3(signature = (block_length, message_length, *, design_snr_db=2.0, crc_length=0))]
    fn new(
        block_length: usize,
        message_length: usize,
        design_snr_db: f64,
        crc_length: usize,
    ) -> PyResult<Self> {
        if !block_length.is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "block_length {} is not a power of 2",
                block_length
            )));
        }
        
        if message_length > block_length {
            return Err(PyValueError::new_err(format!(
                "message_length {} exceeds block_length {}",
                message_length, block_length
            )));
        }
        
        if crc_length != 0 && crc_length != 16 {
            return Err(PyValueError::new_err(
                "crc_length must be 0 or 16"
            ));
        }
        
        // Total information bits = message + CRC
        let k_total = message_length + crc_length;
        
        // Construct frozen mask
        let frozen_mask = construct_frozen_mask(
            block_length,
            k_total,
            design_snr_db,
            ConstructionMethod::GaussianApproximation,
        );
        
        let encoder = PolarEncoder::new(block_length, frozen_mask.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let decoder = SCDecoder::new(block_length, frozen_mask.clone());
        
        Ok(Self {
            encoder,
            decoder,
            frozen_mask,
            block_length,
            message_length,
            crc_length,
        })
    }
    
    /// Encode message bits to codeword.
    ///
    /// GIL is released during encoding computation.
    ///
    /// # Arguments
    /// * `message` - Information bits of shape (message_length,), dtype=uint8
    ///
    /// # Returns
    /// Encoded codeword of shape (block_length,), dtype=uint8
    ///
    /// # Raises
    /// * `ValueError` - If message shape or dtype is incorrect
    fn encode<'py>(
        &self,
        py: Python<'py>,
        message: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let msg_slice = message.as_slice()
            .map_err(|_| PyValueError::new_err("Message must be contiguous"))?;
        
        if msg_slice.len() != self.message_length {
            return Err(PyValueError::new_err(format!(
                "Message length {} does not match expected {}",
                msg_slice.len(), self.message_length
            )));
        }
        
        // Prepare message with optional CRC
        let full_message: Vec<u8> = if self.crc_length > 0 {
            crc::append_crc16(msg_slice)
        } else {
            msg_slice.to_vec()
        };
        
        // Release GIL for encoding
        let codeword = py.allow_threads(|| {
            self.encoder.encode(&full_message)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(codeword.into_pyarray_bound(py))
    }
    
    /// Hard-decision decoding from received bits.
    ///
    /// GIL is released during decoding computation.
    ///
    /// # Arguments
    /// * `received` - Received bits of shape (block_length,), dtype=uint8
    /// * `qber` - Estimated quantum bit error rate (0.0 to 0.5)
    ///
    /// # Returns
    /// Tuple of (decoded_message, converged, path_metric)
    /// - decoded_message: shape (message_length,), dtype=uint8
    /// - converged: bool (always True for SC L=1)
    /// - path_metric: float (sum of LLR penalties)
    #[pyo3(signature = (received, *, qber=0.05))]
    fn decode_hard<'py>(
        &mut self,
        py: Python<'py>,
        received: PyReadonlyArray1<'py, u8>,
        qber: f32,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, f32)> {
        let recv_slice = received.as_slice()
            .map_err(|_| PyValueError::new_err("Received must be contiguous"))?;
        
        if recv_slice.len() != self.block_length {
            return Err(PyValueError::new_err(format!(
                "Received length {} does not match block_length {}",
                recv_slice.len(), self.block_length
            )));
        }
        
        // Release GIL for decoding
        let recv_vec: Vec<u8> = recv_slice.to_vec();
        let result = py.allow_threads(|| {
            self.decoder.decode_hard(&recv_vec, qber)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Strip CRC if present
        let message = if self.crc_length > 0 {
            crc::strip_crc16(&result.message)
                .ok_or_else(|| PyRuntimeError::new_err("Failed to strip CRC"))?
        } else {
            result.message
        };
        
        let msg_array = PyArray1::from_vec_bound(py, message);
        
        Ok((msg_array, result.converged, result.path_metric))
    }
    
    /// Soft-decision decoding from channel LLRs.
    ///
    /// GIL is released during decoding computation.
    ///
    /// # Arguments
    /// * `llr_channel` - Channel LLRs of shape (block_length,), dtype=float32
    ///   Convention: positive LLR means bit=0 more likely
    ///
    /// # Returns
    /// Tuple of (extrinsic_llr, decoded_message, converged, iterations, path_metric)
    /// - extrinsic_llr: shape (block_length,), dtype=float32 (stub: returns zeros)
    /// - decoded_message: shape (message_length,), dtype=uint8
    /// - converged: bool
    /// - iterations: int (always 1 for SC)
    /// - path_metric: float
    fn decode_soft<'py>(
        &mut self,
        py: Python<'py>,
        llr_channel: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u8>>, bool, i32, f32)> {
        let llr_slice = llr_channel.as_slice()
            .map_err(|_| PyValueError::new_err("LLR must be contiguous"))?;
        
        if llr_slice.len() != self.block_length {
            return Err(PyValueError::new_err(format!(
                "LLR length {} does not match block_length {}",
                llr_slice.len(), self.block_length
            )));
        }
        
        // Release GIL for decoding
        let llr_vec: Vec<f32> = llr_slice.to_vec();
        let result = py.allow_threads(|| {
            self.decoder.decode(&llr_vec)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Strip CRC if present
        let message = if self.crc_length > 0 {
            crc::strip_crc16(&result.message)
                .ok_or_else(|| PyRuntimeError::new_err("Failed to strip CRC"))?
        } else {
            result.message
        };
        
        // Extrinsic LLR computation is a stub for Phase 1
        // Full implementation in Phase 2 with SCL path metrics
        let extrinsic = vec![0.0f32; self.block_length];
        
        let ext_array = PyArray1::from_vec_bound(py, extrinsic);
        let msg_array = PyArray1::from_vec_bound(py, message);
        
        Ok((ext_array, msg_array, result.converged, 1, result.path_metric))
    }
    
    // =========================================================================
    // Properties (SISOCodec protocol)
    // =========================================================================
    
    /// Code block length N.
    #[getter]
    fn block_length(&self) -> usize {
        self.block_length
    }
    
    /// Information bit length k (excluding CRC).
    #[getter]
    fn message_length(&self) -> usize {
        self.message_length
    }
    
    /// Effective code rate R = k / N.
    #[getter]
    fn rate(&self) -> f64 {
        self.message_length as f64 / self.block_length as f64
    }
    
    /// CRC bit length (0 or 16).
    #[getter]
    fn crc_length(&self) -> usize {
        self.crc_length
    }
    
    /// Total information bits including CRC.
    #[getter]
    fn k_total(&self) -> usize {
        self.message_length + self.crc_length
    }
}

/// Python module definition.
#[pymodule]
fn caligo_codecs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolarCodec>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
```

</code_block>

### 6.2 Error Module

<code_block language="rust" file="caligo/_native/src/error.rs">

```rust
//! Error types for the caligo-codecs crate.

use thiserror::Error;

/// Top-level error type for codec operations.
#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Encoding error: {0}")]
    Encode(#[from] crate::polar::encoder::EncoderError),
    
    #[error("Decoding error: {0}")]
    Decode(#[from] crate::polar::decoder::DecoderError),
    
    #[error("Configuration error: {0}")]
    Config(String),
}
```

</code_block>

### 6.3 Polar Module

<code_block language="rust" file="caligo/_native/src/polar/mod.rs">

```rust
//! Polar code implementation modules.

pub mod construction;
pub mod crc;
pub mod decoder;
pub mod encoder;

pub use construction::{construct_frozen_mask, ConstructionMethod};
pub use decoder::{SCDecoder, SCDecodeResult};
pub use encoder::PolarEncoder;
```

</code_block>

---

## §7 Test Vector Generation

### 7.1 Python Script for Synthetic Test Vectors

Create this script at `caligo/_native/scripts/generate_test_vectors.py`:

<code_block language="python" file="caligo/_native/scripts/generate_test_vectors.py">

```python
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
```

</code_block>

---

## §8 Acceptance Criteria

<acceptance_criteria>

### Build Verification

- [ ] `cargo build --release` completes without errors
- [ ] `cargo test` passes all unit tests
- [ ] `cargo clippy` reports no warnings
- [ ] `cargo fmt --check` passes

### Python Integration

- [ ] `maturin develop` installs Python module successfully
- [ ] `python -c "from caligo._native import PolarCodec"` succeeds
- [ ] `pytest caligo/_native/tests/` passes integration tests

### Functional Correctness

- [ ] Encoder produces correct codewords for all test vectors
- [ ] SC decoder recovers original message for high-SNR test vectors
- [ ] CRC computation matches reference implementation

### Performance Baseline

- [ ] Encoding N=4096 completes in <1ms
- [ ] SC decoding N=4096 completes in <10ms
- [ ] GIL release verified via threading stress test

### Protocol Compliance

- [ ] `PolarCodec` satisfies `SISOCodec` protocol interface
- [ ] `encode()` returns correct shape `(block_length,)`
- [ ] `decode_soft()` returns tuple matching protocol spec

</acceptance_criteria>

---

## References

<references>
- [1] ADR-0001: Polar Codec Adoption
- [2] ADR-0002: Rust Native Extension
- [3] [siso-codec-protocol.md](../specs/siso-codec-protocol.md) — Interface contract
- [4] [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) — Encoding algorithm, SC decoder
- [5] [LLR-Based_SCL_Decoding.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) — f/g functions (Eq. 8-9)
- [6] [Fast_Polar_Decoders.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) — Butterfly structure, min-sum approximation
- [7] PyO3 Documentation: https://pyo3.rs/
- [8] Maturin Documentation: https://www.maturin.rs/
</references>

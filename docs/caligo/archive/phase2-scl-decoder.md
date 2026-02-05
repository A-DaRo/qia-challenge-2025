# Phase 2: SCL Decoder Implementation Guide

<metadata>
impl_id: phase2-scl-decoder
version: 1.0.0
status: ready
created: 2026-02-02
depends_on: [ADR-0001, ADR-0002, phase1-rust-foundation, rust-polar-crate]
enables: [phase3-strategy-refactor, phase4-integration]
</metadata>

---

## Overview

<overview>

This document provides step-by-step implementation guidance for extending the Phase 1 SC decoder (L=1) to a full Successive Cancellation List (SCL) decoder with:

- **List-based path management** — Track L concurrent decoding paths
- **CRC-aided selection** — Use CRC-16-CCITT to identify correct path
- **LLR-domain metrics** — Numerically stable path metric computation
- **GIL release** — Python parallelism via `py.allow_threads()`

### Deliverables

| Component | File | Description |
|-----------|------|-------------|
| Bit-reversal fix | `src/polar/encoder.rs` | Align encoder/decoder indexing |
| SCL decoder | `src/polar/scl_decoder.rs` | List decoding with L paths |
| CRC integration | `src/polar/crc.rs` | CA-SCL path selection |
| PyO3 bindings | `src/lib.rs` | `PyPolarCodec` class |
| Test vectors | `tests/vectors/scl_*.json` | Validation data |

### Timeline

| Task | Estimated Hours |
|------|-----------------|
| Phase 2.0: Bit-reversal fix | 2h |
| Phase 2.1: SCL core algorithm | 8h |
| Phase 2.2: CRC integration | 3h |
| Phase 2.3: PyO3 bindings | 4h |
| Phase 2.4: Testing & validation | 4h |
| **Total** | **21h** (~2.5 days) |

### Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bit-reversal location | Encoder (`encode()`) | Single modification point |
| Memory management | Simple deep-copy | Correctness first; lazy-copy deferred |
| SIMD optimizations | Deferred to Phase 2.5 | SCL correctness priority |
| Soft output format | Per-bit LLRs from winning path | Standard CA-SCL convention |

</overview>

---

## §1 Prerequisites & Bit-Reversal Fix (Phase 2.0)

### §1.1 Problem Analysis

**Reference:** [phase2-scl-decoder-plan.md](phase2-scl-decoder-plan.md) — Critical Issue section

The Phase 1 `test_sc_decode_noiseless` revealed a structural mismatch:

```
Expected: [1, 0, 1, 1]
Actual:   [1, 1, 1, 0]
```

**Root Cause:** The SC decoder's recursive tree structure expects bit-reversed input indices, but the encoder uses natural order.

**Literature:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) Eq. (1):

$$\mathbf{G}_n \triangleq \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}^{\otimes n} \mathbf{B}_n$$

where $\mathbf{B}_n$ is the bit-reversal permutation matrix.

### §1.2 Implementation

**File:** [encoder.rs](../../../caligo/_native/src/polar/encoder.rs)

<code_block language="rust" file="caligo/_native/src/polar/encoder.rs" action="modify">

```rust
// Add to construction.rs or encoder.rs

/// Compute bit-reversal permutation of index.
///
/// # Arguments
/// * `index` - Original index in [0, 2^n_bits)
/// * `n_bits` - Number of bits (log2 of block length)
///
/// # Returns
/// Bit-reversed index
///
/// # Example
/// ```
/// assert_eq!(bit_reverse_index(1, 3), 4);  // 001 -> 100
/// assert_eq!(bit_reverse_index(3, 3), 6);  // 011 -> 110
/// ```
#[inline]
pub fn bit_reverse_index(index: usize, n_bits: usize) -> usize {
    // Use built-in reverse_bits and shift to get n_bits result
    index.reverse_bits() >> (usize::BITS as usize - n_bits)
}

// Modify PolarEncoder::encode() to apply bit-reversal:

impl PolarEncoder {
    pub fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
        // Validate input
        if message.len() != self.k_total {
            return Err(EncoderError::MessageLengthMismatch(
                message.len(),
                self.k_total,
            ));
        }
        
        for (i, &bit) in message.iter().enumerate() {
            if bit > 1 {
                return Err(EncoderError::NonBinaryValue(bit, i));
            }
        }
        
        // Initialize u-vector with zeros
        let mut u = vec![0u8; self.block_length];
        let mut msg_idx = 0;
        
        // Place message bits at information positions
        // KEY FIX: Apply bit-reversal permutation to align with decoder
        for i in 0..self.block_length {
            if !self.frozen_mask[i] {
                // frozen_mask is indexed in decoder's bit-reversed order
                // Place message at bit-reversed position for butterfly compatibility
                let br_i = bit_reverse_index(i, self.n_stages);
                u[br_i] = message[msg_idx];
                msg_idx += 1;
            }
            // Frozen positions remain 0
        }
        
        // Apply butterfly transform (unchanged)
        self.butterfly_transform(&mut u);
        
        Ok(Array1::from_vec(u))
    }
}
```

</code_block>

### §1.3 Validation

**Test Vector:** TV-BRP-01

```rust
#[test]
fn test_encoder_decoder_roundtrip_n8() {
    // N=8, K=4, design SNR=2.0dB
    let frozen_mask = construct_frozen_mask(8, 4, 2.0, ConstructionMethod::GaussianApproximation);
    
    let encoder = PolarEncoder::new(8, frozen_mask.clone()).unwrap();
    let mut decoder = SCDecoder::new(8, frozen_mask);
    
    // Test message
    let message = vec![1u8, 0, 1, 1];
    
    // Encode
    let codeword = encoder.encode(&message).unwrap();
    
    // Create "perfect" channel LLRs (very high confidence)
    let llr: Vec<f32> = codeword.iter()
        .map(|&b| if b == 0 { 10.0 } else { -10.0 })
        .collect();
    
    // Decode
    let result = decoder.decode(&llr).unwrap();
    
    // MUST match exactly after bit-reversal fix
    assert_eq!(result.message, message, "Round-trip failed!");
    assert!(result.converged);
}
```

**Acceptance Criteria:**
- [ ] AC-2.0.1: `test_encoder_decoder_roundtrip_n8` passes
- [ ] AC-2.0.2: `bit_reverse_index` is O(1) per call

---

## §2 SCL Algorithm

### §2.1 Path Splitting at Information Bits

**Reference:** [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-A

> "Our decoder doubles the number of decoding paths for each information bit $u_i$ to be decoded, thus pursuing both $u_i = 0$ and $u_i = 1$ options."

**Algorithm:**

```
At information bit position φ:
  For each active path l ∈ [0, num_active):
    1. Compute LLR at decision layer: L = llr_memory[l][n][φ]
    2. Create two child paths:
       - Path l:   decide û = 0, metric += penalty(L, 0)
       - Path l':  decide û = 1, metric += penalty(L, 1)
    3. num_active *= 2
```

**Implementation:**

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
/// Process an information bit: split all active paths.
fn process_info_bit(&mut self, phi: usize) {
    // Snapshot current active paths
    let active_indices: Vec<usize> = (0..self.list_size)
        .filter(|&l| self.active_paths[l])
        .collect();
    
    let num_to_split = active_indices.len();
    
    // Find inactive slots for new paths
    let mut inactive_slots: Vec<usize> = (0..self.list_size)
        .filter(|&l| !self.active_paths[l])
        .take(num_to_split)
        .collect();
    
    for (i, &l) in active_indices.iter().enumerate() {
        let llr = self.llr_memory[l][self.n_stages][phi];
        
        // Original path decides 0
        let metric_0 = self.path_metrics[l] + path_metric_increment(llr, 0);
        self.bit_memory[l][self.n_stages][phi] = 0;
        self.path_metrics[l] = metric_0;
        
        // Clone to new path, decide 1
        if let Some(&l_new) = inactive_slots.get(i) {
            // Deep copy path state (simple approach per design decision)
            self.clone_path(l, l_new);
            
            let metric_1 = self.path_metrics[l_new] + path_metric_increment(llr, 1);
            self.bit_memory[l_new][self.n_stages][phi] = 1;
            self.path_metrics[l_new] = metric_1;
            
            self.active_paths.set(l_new, true);
            self.num_active += 1;
        }
    }
}

/// Deep copy path state from src to dst.
fn clone_path(&mut self, src: usize, dst: usize) {
    // Copy LLR memory
    for layer in 0..=self.n_stages {
        self.llr_memory[dst][layer].copy_from_slice(&self.llr_memory[src][layer]);
    }
    
    // Copy bit memory
    for layer in 0..=self.n_stages {
        self.bit_memory[dst][layer].copy_from_slice(&self.bit_memory[src][layer]);
    }
    
    // Copy path metric
    self.path_metrics[dst] = self.path_metrics[src];
}
```

</code_block>

### §2.2 Pruning to L Best Paths

**Reference:** [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-B

> "uses a pruning procedure to discard all but the $L$ most likely paths"

**Algorithm:**

```
If num_active > L:
  1. Collect (path_index, metric) for all active paths
  2. Sort by metric (ascending = best first)
  3. Keep L paths with lowest metrics
  4. Deactivate remaining paths
```

**Implementation:**

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
/// Prune paths to keep only L best (lowest metric).
fn prune_paths(&mut self) {
    if self.num_active <= self.list_size {
        return;
    }
    
    // Collect active paths with metrics
    let mut path_metrics: Vec<(usize, f32)> = (0..self.list_size)
        .filter(|&l| self.active_paths[l])
        .map(|l| (l, self.path_metrics[l]))
        .collect();
    
    // Sort by metric (ascending)
    path_metrics.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Keep only L best paths
    let paths_to_keep: std::collections::HashSet<usize> = path_metrics
        .iter()
        .take(self.list_size)
        .map(|&(l, _)| l)
        .collect();
    
    // Deactivate pruned paths
    for l in 0..self.list_size {
        if self.active_paths[l] && !paths_to_keep.contains(&l) {
            self.active_paths.set(l, false);
        }
    }
    
    self.num_active = self.list_size;
}
```

</code_block>

### §2.3 Path Metric Updates (LLR Domain)

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §III-A, Eq. (15)

$$PM^{(l)} \leftarrow PM^{(l)} + \ln(1 + e^{-|L_i^{(l)}|})$$

when decision disagrees with LLR sign.

**Approximation (§III-B):** For $|L| > T \approx 10$:

$$\ln(1 + e^{-|L|}) \approx 0$$

**Implementation:**

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
/// Compute path metric increment for a decision.
///
/// Reference: Balatsoukas-Stimming et al., IEEE TSP 2015, Eq. (15)
///
/// # Arguments
/// * `llr` - Log-likelihood ratio at decision point
/// * `decision` - Decoded bit (0 or 1)
///
/// # Returns
/// Non-negative metric increment
#[inline]
fn path_metric_increment(llr: f32, decision: u8) -> f32 {
    const THRESHOLD: f32 = 10.0;
    
    // Determine if decision matches LLR sign
    // LLR > 0 suggests bit = 0, LLR < 0 suggests bit = 1
    let llr_suggests = if llr >= 0.0 { 0u8 } else { 1u8 };
    
    if decision == llr_suggests {
        // Decision matches LLR: no penalty
        0.0
    } else {
        // Decision against LLR: penalize
        let abs_llr = llr.abs();
        if abs_llr > THRESHOLD {
            // Approximation for large |LLR|
            // ln(1 + e^-x) ≈ e^-x ≈ 0 for large x
            // But we still want some penalty proportional to confidence
            // Use |LLR| directly as penalty (matches soft-decision intuition)
            abs_llr
        } else {
            // Exact computation for moderate |LLR|
            (1.0 + (-abs_llr).exp()).ln()
        }
    }
}

/// Alternative: Exact LLR-domain metric (no approximation).
/// Use for validation; may have numerical issues for very large |LLR|.
#[inline]
fn path_metric_increment_exact(llr: f32, decision: u8) -> f32 {
    // PM += ln(1 + e^{-(1-2u)*L})
    let sign = if decision == 0 { 1.0 } else { -1.0 };
    let exponent = -sign * llr;
    
    if exponent > 20.0 {
        // Avoid overflow: ln(1 + e^x) ≈ x for large x
        exponent
    } else if exponent < -20.0 {
        // Avoid underflow: ln(1 + e^x) ≈ 0 for very negative x
        0.0
    } else {
        (1.0 + exponent.exp()).ln()
    }
}
```

</code_block>

---

## §3 Memory Management

### §3.1 Simple Deep-Copy Approach

**Design Decision:** Use simple deep-copy for Phase 2 (correctness over complexity).

**Memory Budget:** For N=4096, L=32, n=12:

| Component | Size | Calculation |
|-----------|------|-------------|
| LLR memory | 52 MB | 32 × 13 × 4096 × 4 bytes |
| Bit memory | 13 MB | 32 × 13 × 4096 × 1 byte |
| Path metrics | 128 B | 32 × 4 bytes |
| Active mask | 4 B | 32 bits |
| **Total** | ~65 MB | Within AC-2.3.2 (≤50 MB for L=32)¹ |

¹ Note: Budget may need adjustment. Consider L=16 as default if memory constrained.

**Implementation:**

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
impl SCLDecoder {
    /// Construct SCL decoder with pre-allocated memory.
    pub fn new(code: PolarCode, list_size: usize) -> Result<Self, ConstructionError> {
        // Validate list size
        if !list_size.is_power_of_two() || list_size > 32 {
            return Err(ConstructionError::InvalidListSize(list_size));
        }
        
        let n_stages = code.n_stages;
        let block_length = code.block_length;
        
        // Pre-allocate LLR memory: [L][n+1][N]
        let llr_memory: Vec<Vec<Vec<f32>>> = (0..list_size)
            .map(|_| {
                (0..=n_stages)
                    .map(|_| vec![0.0f32; block_length])
                    .collect()
            })
            .collect();
        
        // Pre-allocate bit memory: [L][n+1][N]
        let bit_memory: Vec<Vec<Vec<u8>>> = (0..list_size)
            .map(|_| {
                (0..=n_stages)
                    .map(|_| vec![0u8; block_length])
                    .collect()
            })
            .collect();
        
        // Path metrics: [L]
        let path_metrics = vec![0.0f32; list_size];
        
        // Active path mask: [L]
        let active_paths = bitvec![u64, Lsb0; 0; list_size];
        
        Ok(Self {
            code,
            list_size,
            n_stages,
            llr_memory,
            bit_memory,
            path_metrics,
            active_paths,
            num_active: 0,
        })
    }
    
    /// Reset decoder state for new decode operation.
    pub fn reset(&mut self) {
        // Clear path metrics
        self.path_metrics.fill(0.0);
        
        // Clear active paths
        self.active_paths.fill(false);
        self.num_active = 0;
        
        // Note: LLR and bit memory don't need clearing
        // They will be overwritten during decode
    }
}
```

</code_block>

### §3.2 Future: Lazy-Copy Optimization

**Reference:** [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-C

> "Specifically, our implementation maintains for every active path a set of pointers to arrays, rather than a set of arrays."

**Deferred to Phase 2.5:** This optimization reduces memory from O(LN log N) to O(LN) but adds significant complexity. Implement after validating correctness with deep-copy approach.

---

## §4 CRC Integration

### §4.1 Encoder: Append CRC

**Reference:** [crc.rs](../../../caligo/_native/src/polar/crc.rs)

<code_block language="rust" file="caligo/_native/src/polar/crc.rs">

```rust
/// Append CRC-16-CCITT to message bits.
///
/// # Arguments
/// * `message` - Information bits (K bits)
///
/// # Returns
/// Message with 16 CRC bits appended (K+16 bits)
pub fn append_crc16(message: &[u8]) -> Vec<u8> {
    let crc = crc16_ccitt(message);
    
    let mut result = message.to_vec();
    
    // Append CRC bits (MSB first)
    for i in (0..16).rev() {
        result.push(((crc >> i) & 1) as u8);
    }
    
    result
}
```

</code_block>

**Encoder Integration:**

```rust
impl PyPolarCodec {
    fn encode<'py>(&self, py: Python<'py>, message: PyReadonlyArray1<'py, u8>) 
        -> PyResult<Bound<'py, PyArray1<u8>>> 
    {
        let msg = message.as_slice()?;
        
        // Append CRC if configured
        let msg_with_crc = if self.decoder.code.crc_bits > 0 {
            append_crc16(msg)
        } else {
            msg.to_vec()
        };
        
        // Encode
        let codeword = self.encoder.encode(&msg_with_crc)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyArray1::from_vec_bound(py, codeword.to_vec()))
    }
}
```

### §4.2 Decoder: CRC-Aided Path Selection

**Reference:** [CRC-Aided_Decoding_of_Polar_Codes.md](../../literature/CRC-Aided_Decoding_of_Polar_Codes.md) §III

> "the decoder outputs the candidate sequences into CRC detector and the latter feeds the check results back"

**Algorithm (CA-SCL):**

```
After decoding all N bits:
  1. For each active path l:
     a. Extract message bits from bit_memory[l][n][info_positions]
     b. Verify CRC: remainder should be 0
     c. If CRC passes, track (l, metric)
  2. If any path passes CRC:
     Return path with lowest metric among CRC-passing paths
  3. Else (no CRC match - "list decoding failure"):
     Return path with lowest metric overall
```

**Implementation:**

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
impl SCLDecoder {
    /// Select best path using CRC-aided selection.
    fn select_best_path(&self) -> Result<SCLDecodeResult, DecoderError> {
        let mut best_crc_path: Option<(usize, f32)> = None;
        let mut best_any_path: Option<(usize, f32)> = None;
        let mut paths_explored = 0;
        
        for l in 0..self.list_size {
            if !self.active_paths[l] {
                continue;
            }
            paths_explored += 1;
            
            let metric = self.path_metrics[l];
            
            // Track best overall path
            match best_any_path {
                None => best_any_path = Some((l, metric)),
                Some((_, best_metric)) if metric < best_metric => {
                    best_any_path = Some((l, metric));
                }
                _ => {}
            }
            
            // Check CRC if configured
            if self.code.crc_bits > 0 {
                let message_with_crc = self.extract_info_bits(l);
                if verify_crc16(&message_with_crc) {
                    match best_crc_path {
                        None => best_crc_path = Some((l, metric)),
                        Some((_, best_metric)) if metric < best_metric => {
                            best_crc_path = Some((l, metric));
                        }
                        _ => {}
                    }
                }
            }
        }
        
        // Determine selected path and CRC status
        let (selected_path, crc_valid) = match (best_crc_path, best_any_path) {
            (Some((l, _)), _) => (l, Some(true)),
            (None, Some((l, _))) if self.code.crc_bits > 0 => (l, Some(false)),
            (None, Some((l, _))) => (l, None),
            (None, None) => return Err(DecoderError::NoActivePaths),
        };
        
        // Build result
        let message_with_crc = self.extract_info_bits(selected_path);
        let message = if self.code.crc_bits > 0 {
            message_with_crc[..message_with_crc.len() - self.code.crc_bits].to_vec()
        } else {
            message_with_crc
        };
        
        let soft_output = self.extract_soft_output(selected_path);
        
        Ok(SCLDecodeResult {
            message,
            soft_output,
            path_metric: self.path_metrics[selected_path],
            crc_valid,
            paths_explored,
            converged: true,
        })
    }
    
    /// Extract information bits from a path.
    fn extract_info_bits(&self, path: usize) -> Vec<u8> {
        let mut bits = Vec::with_capacity(self.code.k_total);
        
        for phi in 0..self.code.block_length {
            if !self.code.frozen_mask[phi] {
                bits.push(self.bit_memory[path][self.n_stages][phi]);
            }
        }
        
        bits
    }
    
    /// Extract soft output (LLRs from winning path).
    fn extract_soft_output(&self, path: usize) -> Vec<f32> {
        // Return decision-layer LLRs for SISO interface
        self.llr_memory[path][self.n_stages].clone()
    }
}
```

</code_block>

---

## §5 Rust Implementation

### §5.1 SCLDecoder Struct

**File:** `src/polar/scl_decoder.rs`

<code_block language="rust" file="caligo/_native/src/polar/scl_decoder.rs">

```rust
//! Successive Cancellation List (SCL) decoder for Polar codes.
//!
//! References:
//! - [1] Tal & Vardy, "List Decoding of Polar Codes", IEEE TIT 2015
//! - [2] Balatsoukas-Stimming et al., "LLR-Based SCL Decoding", IEEE TSP 2015
//! - [3] Niu & Chen, "CRC-Aided Decoding of Polar Codes", IEEE CL 2012

use bitvec::prelude::*;
use crate::polar::crc::{verify_crc16, CRC16_BITS};
use crate::polar::construction::bit_reverse_index;
use crate::polar::{PolarCode, DecoderError, ConstructionError};

/// SCL decoder result.
#[derive(Debug, Clone)]
pub struct SCLDecodeResult {
    /// Decoded message bits (excluding CRC)
    pub message: Vec<u8>,
    /// Soft output LLRs for SISO interface
    pub soft_output: Vec<f32>,
    /// Path metric of selected path
    pub path_metric: f32,
    /// CRC validation result
    pub crc_valid: Option<bool>,
    /// Number of paths explored
    pub paths_explored: usize,
    /// Decode completed successfully
    pub converged: bool,
}

/// Successive Cancellation List decoder.
pub struct SCLDecoder {
    /// Polar code configuration
    code: PolarCode,
    /// List size L
    list_size: usize,
    /// Number of stages n = log2(N)
    n_stages: usize,
    /// LLR memory: [L][n+1][N]
    llr_memory: Vec<Vec<Vec<f32>>>,
    /// Bit memory: [L][n+1][N]
    bit_memory: Vec<Vec<Vec<u8>>>,
    /// Path metrics: [L]
    path_metrics: Vec<f32>,
    /// Active path mask
    active_paths: BitVec<u64, Lsb0>,
    /// Current number of active paths
    num_active: usize,
}

impl SCLDecoder {
    /// Create new SCL decoder.
    pub fn new(code: PolarCode, list_size: usize) -> Result<Self, ConstructionError> {
        // Validate list size (must be power of 2, max 32)
        if !list_size.is_power_of_two() || list_size > 32 || list_size == 0 {
            return Err(ConstructionError::InvalidListSize(list_size));
        }
        
        let n_stages = code.n_stages;
        let block_length = code.block_length;
        
        // Allocate memory
        let llr_memory = Self::allocate_llr_memory(list_size, n_stages, block_length);
        let bit_memory = Self::allocate_bit_memory(list_size, n_stages, block_length);
        let path_metrics = vec![0.0f32; list_size];
        let active_paths = bitvec![u64, Lsb0; 0; list_size];
        
        Ok(Self {
            code,
            list_size,
            n_stages,
            llr_memory,
            bit_memory,
            path_metrics,
            active_paths,
            num_active: 0,
        })
    }
    
    fn allocate_llr_memory(l: usize, n: usize, block_len: usize) -> Vec<Vec<Vec<f32>>> {
        (0..l).map(|_| {
            (0..=n).map(|_| vec![0.0f32; block_len]).collect()
        }).collect()
    }
    
    fn allocate_bit_memory(l: usize, n: usize, block_len: usize) -> Vec<Vec<Vec<u8>>> {
        (0..l).map(|_| {
            (0..=n).map(|_| vec![0u8; block_len]).collect()
        }).collect()
    }
    
    /// Decode from channel LLRs.
    pub fn decode(&mut self, llr_channel: &[f32]) -> Result<SCLDecodeResult, DecoderError> {
        // Validate input
        if llr_channel.len() != self.code.block_length {
            return Err(DecoderError::LlrLengthMismatch(
                llr_channel.len(),
                self.code.block_length,
            ));
        }
        
        // Reset state
        self.reset();
        
        // Initialize path 0 with channel LLRs
        self.llr_memory[0][0].copy_from_slice(llr_channel);
        self.active_paths.set(0, true);
        self.num_active = 1;
        self.path_metrics[0] = 0.0;
        
        // Decode each bit position
        for phi in 0..self.code.block_length {
            // Compute LLRs at decision layer for all active paths
            self.compute_llrs(phi);
            
            if self.code.is_frozen(phi) {
                self.process_frozen_bit(phi);
            } else {
                self.process_info_bit(phi);
                
                // Prune if needed
                if self.num_active > self.list_size {
                    self.prune_paths();
                }
            }
            
            // Propagate partial sums
            if phi % 2 == 1 {
                self.propagate_partial_sums(phi);
            }
        }
        
        // Select best path (CRC-aided if configured)
        self.select_best_path()
    }
    
    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.path_metrics.fill(0.0);
        self.active_paths.fill(false);
        self.num_active = 0;
    }
    
    /// Get list size.
    pub fn list_size(&self) -> usize {
        self.list_size
    }
    
    /// Get code reference.
    pub fn code(&self) -> &PolarCode {
        &self.code
    }
}
```

</code_block>

### §5.2 Core Methods

**LLR Computation (f/g functions):**

<code_block language="rust">

```rust
impl SCLDecoder {
    /// Compute LLRs at position phi for all active paths.
    fn compute_llrs(&mut self, phi: usize) {
        for l in 0..self.list_size {
            if !self.active_paths[l] {
                continue;
            }
            self.recursively_calc_llr(l, self.n_stages, phi);
        }
    }
    
    /// Recursively compute LLR at (layer, phi) for path l.
    fn recursively_calc_llr(&mut self, l: usize, layer: usize, phi: usize) {
        if layer == 0 {
            return; // Base case: channel LLRs already set
        }
        
        let psi = phi >> 1;
        let layer_size = 1 << layer;
        
        if phi % 2 == 0 {
            // Even position: need both LLRs from previous layer
            self.recursively_calc_llr(l, layer - 1, psi);
            self.calc_f_function(l, layer, phi);
        } else {
            // Odd position: use partial sum from even
            self.calc_g_function(l, layer, phi);
        }
    }
    
    /// f-function: combine LLRs for even position.
    /// f(α, β) = sign(α)·sign(β)·min(|α|, |β|)
    #[inline]
    fn calc_f_function(&mut self, l: usize, layer: usize, phi: usize) {
        let psi = phi >> 1;
        let half = 1 << (layer - 1);
        
        let alpha = self.llr_memory[l][layer - 1][psi];
        let beta = self.llr_memory[l][layer - 1][psi + half];
        
        // Min-sum approximation
        let sign = alpha.signum() * beta.signum();
        let min_abs = alpha.abs().min(beta.abs());
        
        self.llr_memory[l][layer][phi] = sign * min_abs;
    }
    
    /// g-function: combine LLRs for odd position using partial sum.
    /// g(α, β, u) = (-1)^u · α + β
    #[inline]
    fn calc_g_function(&mut self, l: usize, layer: usize, phi: usize) {
        let psi = phi >> 1;
        let half = 1 << (layer - 1);
        
        let alpha = self.llr_memory[l][layer - 1][psi];
        let beta = self.llr_memory[l][layer - 1][psi + half];
        let u = self.bit_memory[l][layer][phi - 1]; // Even bit decision
        
        let sign = if u == 0 { 1.0 } else { -1.0 };
        
        self.llr_memory[l][layer][phi] = sign * alpha + beta;
    }
    
    /// Process frozen bit (all paths decide 0).
    fn process_frozen_bit(&mut self, phi: usize) {
        for l in 0..self.list_size {
            if !self.active_paths[l] {
                continue;
            }
            
            let llr = self.llr_memory[l][self.n_stages][phi];
            
            // Frozen bit is always 0
            self.bit_memory[l][self.n_stages][phi] = 0;
            
            // Update path metric (penalize if LLR suggests 1)
            self.path_metrics[l] += path_metric_increment(llr, 0);
        }
    }
    
    /// Propagate partial sums through butterfly.
    fn propagate_partial_sums(&mut self, phi: usize) {
        for l in 0..self.list_size {
            if !self.active_paths[l] {
                continue;
            }
            self.recursively_calc_bits(l, self.n_stages, phi);
        }
    }
    
    /// Recursively propagate partial sums for path l.
    fn recursively_calc_bits(&mut self, l: usize, layer: usize, phi: usize) {
        if layer == 0 || phi % 2 == 0 {
            return;
        }
        
        let psi = phi >> 1;
        let half = 1 << (layer - 1);
        
        // XOR propagation
        let u_even = self.bit_memory[l][layer][phi - 1];
        let u_odd = self.bit_memory[l][layer][phi];
        
        self.bit_memory[l][layer - 1][psi] = u_even ^ u_odd;
        self.bit_memory[l][layer - 1][psi + half] = u_odd;
        
        // Continue recursion
        if psi % 2 == 1 {
            self.recursively_calc_bits(l, layer - 1, psi);
        }
    }
}
```

</code_block>

### §5.3 SIMD Opportunities (Deferred)

**Reference:** [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §III-B

**Deferred optimizations for Phase 2.5:**

1. **Vectorized f-function:** Process multiple paths with SIMD
2. **Node specialization:** Rate-0, Rate-1, Rep, SPC fast paths
3. **Memory prefetching:** Improve cache utilization

---

## §6 PyO3 Bindings

### §6.1 GIL Release Pattern

**Reference:** [numba_kernels.py](../../../caligo/caligo/scripts/numba_kernels.py) L718-771

<code_block language="rust" file="caligo/_native/src/lib.rs">

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass(name = "PolarCodec")]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCLDecoder,
}

#[pymethods]
impl PyPolarCodec {
    #[new]
    #[pyo3(signature = (block_length, message_length, list_size=8, crc_bits=16, design_snr_db=2.0))]
    fn new(
        block_length: usize,
        message_length: usize,
        list_size: usize,
        crc_bits: usize,
        design_snr_db: f64,
    ) -> PyResult<Self> {
        // Construct PolarCode
        let code = PolarCode::try_new(block_length, message_length, crc_bits, design_snr_db)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Construct encoder and decoder
        let encoder = PolarEncoder::new(code.block_length, code.frozen_mask.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let decoder = SCLDecoder::new(code, list_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(Self { encoder, decoder })
    }
    
    /// Decode with GIL release.
    fn decode_soft<'py>(
        &mut self,
        py: Python<'py>,
        llr: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        f32,
        Option<bool>,
    )> {
        // Extract LLR data
        let llr_slice = llr.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid LLR array: {}", e)))?;
        
        // Copy to owned buffer for GIL release
        let llr_owned: Vec<f32> = llr_slice.to_vec();
        
        // GIL RELEASE: perform decode without holding Python lock
        let result = py.allow_threads(|| {
            self.decoder.decode(&llr_owned)
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert results to numpy arrays (GIL re-acquired)
        let soft_output = PyArray1::from_vec_bound(py, result.soft_output);
        let message = PyArray1::from_vec_bound(py, result.message);
        
        Ok((soft_output, message, result.path_metric, result.crc_valid))
    }
    
    fn encode<'py>(
        &self,
        py: Python<'py>,
        message: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let msg = message.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid message array: {}", e)))?;
        
        // Append CRC if configured
        let msg_with_crc = if self.decoder.code().crc_bits > 0 {
            crate::polar::crc::append_crc16(msg)
        } else {
            msg.to_vec()
        };
        
        let codeword = self.encoder.encode(&msg_with_crc)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(PyArray1::from_vec_bound(py, codeword.to_vec()))
    }
    
    fn frozen_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let mask: Vec<u8> = self.decoder.code().frozen_mask
            .iter()
            .map(|b| if *b { 1u8 } else { 0u8 })
            .collect();
        Ok(PyArray1::from_vec_bound(py, mask))
    }
    
    #[getter]
    fn block_length(&self) -> usize {
        self.decoder.code().block_length
    }
    
    #[getter]
    fn message_length(&self) -> usize {
        self.decoder.code().message_length
    }
    
    #[getter]
    fn list_size(&self) -> usize {
        self.decoder.list_size()
    }
    
    #[getter]
    fn crc_bits(&self) -> usize {
        self.decoder.code().crc_bits
    }
    
    #[getter]
    fn rate(&self) -> f64 {
        self.decoder.code().rate()
    }
}

/// Python module definition.
#[pymodule]
fn caligo_codecs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolarCodec>()?;
    Ok(())
}
```

</code_block>

### §6.2 Buffer Protocol

The `numpy` crate handles buffer protocol automatically for `PyArray1`. Key considerations:

1. **Input arrays:** Use `PyReadonlyArray1` to avoid unnecessary copies
2. **Output arrays:** Use `PyArray1::from_vec_bound` for owned data
3. **Dtype enforcement:** Rust types map directly (f32 → float32, u8 → uint8)

---

## §7 Test Vectors & Validation

### Test Vector Table

| ID | N | K | L | CRC | Design SNR | Noise SNR | Target FER | Purpose |
|----|---|---|---|-----|------------|-----------|------------|---------|
| TV-BRP-01 | 8 | 4 | 1 | 0 | 2.0 dB | ∞ | 0 (exact) | Bit-reversal alignment |
| TV-SC-01 | 1024 | 512 | 1 | 0 | 2.0 dB | 2.0 dB | < 0.1 | SC baseline regression |
| TV-SCL-01 | 1024 | 512 | 1 | 0 | 2.0 dB | 2.0 dB | < 0.1 | SCL L=1 = SC |
| TV-SCL-02 | 1024 | 512 | 8 | 0 | 2.0 dB | 2.0 dB | < 0.01 | Default L |
| TV-SCL-03 | 1024 | 512 | 32 | 0 | 2.0 dB | 2.0 dB | < 0.001 | Max L |
| TV-CASCL-01 | 1024 | 496 | 8 | 16 | 2.0 dB | 1.5 dB | < 0.001 | CRC-aided |
| TV-CASCL-02 | 4096 | 2032 | 8 | 16 | 2.0 dB | 1.0 dB | < 0.0001 | Production |

### Acceptance Criteria

```
Phase 2.0 (Bit-Reversal):
- [ ] AC-2.0.1: test_encoder_decoder_roundtrip passes with exact match
- [ ] AC-2.0.2: bit_reverse_index is O(1) per call

Phase 2.1 (SCL Core):
- [ ] AC-2.1.1: SCLDecoder(L=1) produces identical output to SCDecoder
- [ ] AC-2.1.2: SCLDecoder(L=8) achieves FER < 0.01 at SNR=2dB, N=1024, K=512
- [ ] AC-2.1.3: SCLDecoder(L=32) achieves FER < 0.001 at SNR=2dB, N=1024, K=512

Phase 2.2 (CRC):
- [ ] AC-2.2.1: CRC-aided selection reduces FER by ≥10× vs non-CRC at L=8
- [ ] AC-2.2.2: CRC verification matches crc16_ccitt() for all decoded paths

Phase 2.3 (Performance):
- [ ] AC-2.3.1: Decode throughput ≥ 5 Mbps/core on AMD EPYC for N=4096, L=8
- [ ] AC-2.3.2: Memory usage ≤ 50 MB for N=4096, L=32

Phase 2.4 (PyO3):
- [ ] AC-2.4.1: PyO3 bindings release GIL during decode
- [ ] AC-2.4.2: decode_soft() returns (soft_llr, message, metric, crc_valid) tuple
```

### Statistical Validation Script

```python
# tests/test_scl_statistical.py

import numpy as np
from caligo_codecs import PolarCodec

def test_scl_fer(block_length, message_length, list_size, crc_bits, 
                 design_snr_db, noise_snr_db, target_fer, num_trials=10000):
    """Statistical FER validation."""
    codec = PolarCodec(
        block_length=block_length,
        message_length=message_length,
        list_size=list_size,
        crc_bits=crc_bits,
        design_snr_db=design_snr_db,
    )
    
    # Noise parameters
    snr_linear = 10 ** (noise_snr_db / 10)
    noise_std = 1 / np.sqrt(2 * snr_linear)
    
    errors = 0
    for _ in range(num_trials):
        # Random message
        message = np.random.randint(0, 2, message_length, dtype=np.uint8)
        
        # Encode
        codeword = codec.encode(message)
        
        # BPSK modulation: 0 -> +1, 1 -> -1
        tx = 1 - 2 * codeword.astype(np.float32)
        
        # AWGN channel
        noise = np.random.randn(block_length).astype(np.float32) * noise_std
        rx = tx + noise
        
        # LLR: 2 * rx / σ² (for BPSK)
        llr = (2 * rx / (noise_std ** 2)).astype(np.float32)
        
        # Decode
        soft_out, decoded, metric, crc_valid = codec.decode_soft(llr)
        
        # Check for frame error
        if not np.array_equal(decoded, message):
            errors += 1
    
    fer = errors / num_trials
    print(f"FER: {fer:.6f} (target: {target_fer})")
    assert fer <= target_fer * 1.1, f"FER {fer} exceeds target {target_fer}"
```

---

## Appendix: Performance Tuning (Future)

**Deferred to Phase 2.5:**

1. **Lazy-copy optimization** — Reduce memory from O(LN log N) to O(LN)
2. **Fast-SSC nodes** — Rate-0, Rate-1, Rep, SPC specialization
3. **SIMD vectorization** — AVX-512 for parallel f-function
4. **Memory pooling** — Reuse decoder instances across calls
5. **Batch decoding** — Decode multiple frames in parallel

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-02 | Context Engineer | Initial implementation guide from confirmed plan |

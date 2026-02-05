# Phase 2: SCL Decoder Implementation Strategy

<metadata>
impl_id: phase2-scl-decoder-theory
version: 2.0.0
status: draft
created: 2026-02-03
revised: 2026-02-04
depends_on: [specs/polar-math-spec.md, specs/scl-algorithm-ref.md, impl/phase1-rust-foundation.md]
enables: [phase3-reconciliation-integration]
architecture: [vm-control-flow, hybrid-lazy-soa]
</metadata>

---

## Executive Summary

<overview>

This document bridges the gap between abstract specifications ([polar-math-spec.md](../specs/polar-math-spec.md), [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md)) and concrete Rust implementation. It defines:

1. **Control Flow Architecture** — VM-based instruction loop (not recursion)
2. **Memory Architecture** — Hybrid Lazy-SoA: SIMD-friendly LLRs + lazy-copy bits
3. **Rust Data Structures** — Type-safe mapping with HPC optimizations
4. **SIMD Strategy** — AVX-512 vectorization via Structure-of-Arrays layout
5. **CRC Integration Points** — Hooks for CRC-aided path selection
6. **Program Compiler** — Offline frozen_set → instruction tape compilation

**Target Platform:** AMD Genoa HPC (192 cores, 2GB/core, no GPU)

**Key Architectural Decisions:**
| Decision | Rationale | Reference |
|----------|-----------|----------|
| VM control flow | Branch prediction + instruction cache locality | [Fast_Polar_Decoders.md] §V |
| Hybrid Lazy-SoA | SIMD LLRs + space-efficient bits | [Fast_Polar_Decoders.md] §VII + [List_Decoding.md] §IV.A |
| Node specialization | Reserved ISA slots for Phase 2.5 | [Fast_Polar_Decoders.md] §IV Table IV |

**Complexity Invariants:**
- Time: $O(L \cdot N \log N)$
- Space: $O(L \cdot N)$ via lazy-copy memory sharing

</overview>

---

## 1. Specification-to-Implementation Mapping

<spec_mapping>

This section explicitly maps every Rust type to its specification counterpart.

### 1.1 Cross-Reference Table

| Rust Field | Spec Variable | Spec Document | Notes |
|------------|---------------|---------------|-------|
| `program` | (instruction tape) | [Fast_Polar_Decoders.md] §V | VM bytecode for decode loop |
| `inactive_path_stack` | `inactivePathIndices` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | Stack of free path indices |
| `active_paths` | `activePath` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | BitVec for $O(1)$ lookup |
| `soa_llr_memory` | `arrayPointer_L` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.3 | **SoA layout:** paths contiguous per β |
| `bit_arrays` | `arrayPointer_C` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | Lazy-copy bit-pair arrays |
| `path_to_bit_array` | `pathIndexToArrayIndex` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | Path → bit array mapping (lazy) |
| `inactive_bit_stacks` | `inactiveArrayIndices` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | Per-layer free bit array stacks |
| `bit_ref_counts` | `arrayReferenceCount` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.2 | Reference counting (bits only) |
| `path_metrics` | `pathMetric` | [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) §1.4 | $PM_\ell^{(i)}$ — dense array |

**Architectural Notes:**
- LLR arrays use **Structure-of-Arrays** (SoA) layout for SIMD vectorization
- Bit arrays retain **lazy-copy** semantics for space efficiency
- Control flow uses **VM instruction dispatch** (not recursion)

**Reference:** [List_Decoding_of_Polar_Codes.md] Algorithm 5, [Fast_Polar_Decoders.md] §V.

</spec_mapping>

---

## 2. Rust Struct Definitions

<rust_structs>

### 2.1 Instruction Set Architecture (VM Opcodes)

**Design Decision:** Replace recursive control flow with a pre-compiled instruction tape.

**Reference:** [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §V:
> "By using a pre-compiled list of instructions, the controller is reduced to fetching and decoding instructions."

```rust
/// Pre-compiled decoding instruction.
///
/// # Specification Reference
/// - [Fast_Polar_Decoders.md] §V Table III
/// - [scl-algorithm-ref.md] Algorithms 10-13
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SclOp {
    // === Standard Operations (Phase 2) ===
    
    /// F-function: calculate α_left from α_parent.
    /// Corresponds to even-phase LLR update in Algorithm 10.
    /// Reference: [LLR-Based_SCL.md] §II Eq. (8a)
    F { lambda: u8 },
    
    /// G-function: calculate α_right from α_parent and β_left.
    /// Corresponds to odd-phase LLR update in Algorithm 10.
    /// Reference: [LLR-Based_SCL.md] §II Eq. (8b)
    G { lambda: u8 },
    
    /// Combine: merge β_left and β_right into β_parent.
    /// Corresponds to partial sum update in Algorithm 11.
    Combine { lambda: u8 },
    
    /// G with β_left = 0 (rate-0 left child optimization).
    /// Reference: [Fast_Polar_Decoders.md] §IV.D
    G0R { lambda: u8 },
    
    /// Process frozen bit at position φ.
    /// Sets C_m[0][φ mod 2] = 0, updates path metric.
    Frozen { phi: u16 },
    
    /// Fork/prune at information bit position φ.
    /// Implements Algorithm 13 (continuePaths_UnfrozenBit).
    Info { phi: u16 },
    
    // === Specialized Nodes (Phase 2.5 — Reserved) ===
    
    /// Rate-1 node: direct copy (all bits unfrozen).
    /// Reference: [Fast_Polar_Decoders.md] §IV.A
    #[cfg(feature = "fast-nodes")]
    Rate1 { start: u16, len: u16 },
    
    /// SPC node: single parity check.
    /// Reference: [Fast_Polar_Decoders.md] §IV.A Eq. (7)
    #[cfg(feature = "fast-nodes")]
    Spc { start: u16, len: u16 },
    
    /// REP node: repetition code.
    /// Reference: [Fast_Polar_Decoders.md] §IV.B Eq. (8)
    #[cfg(feature = "fast-nodes")]
    Rep { start: u16, len: u16 },
}

/// Pre-compiled instruction sequence for a specific (N, K, frozen_set).
///
/// Compiled once at decoder construction; reused for all decode() calls.
/// See §9 for compilation algorithm.
pub struct SclProgram {
    /// Instruction tape (offline-compiled)
    pub ops: Vec<SclOp>,
    /// Block length N
    pub block_len: usize,
    /// Number of information bits K
    pub info_bits: usize,
    /// Frozen bit positions (for validation)
    pub frozen_indices: Vec<u16>,
}
```

### 2.2 Top-Level Decoder Struct

```rust
/// Successive Cancellation List (SCL) Decoder
///
/// Implements LLR-domain SCL decoding with:
/// - **VM-based control flow** (instruction dispatch, not recursion)
/// - **Hybrid Lazy-SoA memory** (SIMD LLRs + lazy-copy bits)
///
/// # Specification Reference
/// - [scl-algorithm-ref.md] Algorithm 12 (`SCL_Decode`)
/// - [Fast_Polar_Decoders.md] §V (instruction-based architecture)
/// - [List_Decoding_of_Polar_Codes.md] §IV.A (lazy-copy)
pub struct SclDecoder {
    // === Code Parameters ===
    /// Block length: $N = 2^m$
    block_len: usize,
    /// Polarization depth: $m = \log_2 N$
    depth: usize,
    /// List size: max concurrent decoding paths
    list_size: usize,
    /// Frozen bit indices: $\mathcal{F}$
    frozen_set: BitVec,
    /// Information bit indices: $\mathcal{A}$
    info_set: BitVec,
    
    // === VM Control Flow ===
    /// Pre-compiled instruction tape.
    /// Reference: [Fast_Polar_Decoders.md] §V
    program: SclProgram,
    
    // === Path Management ===
    /// Stack of inactive path indices.
    /// Spec: `inactivePathIndices` [scl-algorithm-ref.md §1.2]
    inactive_path_stack: Vec<usize>,
    
    /// Active path bitmap.
    /// Spec: `activePath[ℓ]` [scl-algorithm-ref.md §1.2]
    active_paths: BitVec,
    
    // === Hybrid Memory: SoA LLRs ===
    /// Structure-of-Arrays LLR storage for SIMD.
    /// Layout: llr[λ][β][path] — paths contiguous per branch.
    /// See §3.2 for detailed layout.
    soa_llr: SoaLlrMemory,
    
    // === Hybrid Memory: Lazy-Copy Bits ===
    /// Bit-pair arrays with lazy-copy semantics.
    /// Spec: `arrayPointer_C` [scl-algorithm-ref.md §1.2]
    bit_arrays: BitArrayBank,
    
    // === Path Metrics ===
    /// Cumulative path metrics (dense array, no indirection).
    /// Spec: `pathMetric[ℓ]` [scl-algorithm-ref.md §1.4]
    /// Lower metric = more likely path
    path_metrics: Vec<f32>,
    
    // === Working Buffers ===
    /// Pre-allocated buffers for path fork/prune.
    continuation_state: PathContinuationState,
    
    // === CRC State (Optional) ===
    /// CRC polynomial for CA-SCL (None for standard SCL)
    crc_poly: Option<u32>,
    /// CRC length in bits
    crc_len: Option<usize>,
    /// Per-path incremental CRC state
    crc_states: Vec<u32>,
}
```

### 2.2 Path State Struct (Internal)

```rust
/// Transient state for a single decoding path during `continuePaths_UnfrozenBit`.
///
/// This struct is used during the path fork/prune operation (Algorithm 13)
/// to collect candidate metrics before selection.
///
/// # Specification Reference
/// - [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) Algorithm 13 (`continuePaths_UnfrozenBit`)
#[derive(Clone, Copy)]
struct PathCandidate {
    /// Path index $\ell$
    path_idx: usize,
    /// Candidate bit value (0 or 1)
    bit_value: u8,
    /// Candidate path metric: $\phi(PM_\ell, L_m[0], u)$
    metric: f32,
}

/// Working buffers for path continuation.
/// Pre-allocated to avoid per-bit allocation.
struct PathContinuationState {
    /// Candidate metrics for all (path, bit) combinations
    /// Capacity: 2*L candidates
    candidates: Vec<PathCandidate>,
    
    /// Indices of selected candidates after pruning
    /// Capacity: L
    selected: Vec<usize>,
    
    /// Temporary buffer for sorting/selection
    /// Capacity: 2*L
    sort_buffer: Vec<f32>,
}
```

### 2.3 Decoder Configuration

```rust
/// Configuration for SCL decoder instantiation.
#[derive(Debug, Clone)]
pub struct SclConfig {
    /// Block length $N$ (must be power of 2)
    pub block_len: usize,
    /// Number of information bits $K$
    pub info_bits: usize,
    /// List size $L$ (max concurrent paths)
    pub list_size: usize,
    /// Design SNR for frozen bit selection (dB)
    pub design_snr: f32,
    /// CRC polynomial (e.g., 0x8005 for CRC-16-CCITT)
    /// None for standard SCL without CRC
    pub crc_poly: Option<u32>,
    /// CRC length in bits (must match polynomial degree)
    pub crc_len: Option<usize>,
    /// Enable min-sum approximation for f-function
    pub use_minsum: bool,
}
```

</rust_structs>

---

## 3. Memory Architecture: Hybrid Lazy-SoA

<memory_architecture>

### 3.1 Design Principles: Why Hybrid?

**Problem Statement:**
Tal & Vardy's "lazy-copy" ([List_Decoding.md] §IV.A) achieves $O(L \cdot N)$ space complexity but induces pointer-chasing patterns that break SIMD auto-vectorization. Sarkis et al. ([Fast_Polar_Decoders.md] §VII) achieve HPC throughput via Structure-of-Arrays (SoA) but increase memory footprint.

**Solution: Hybrid Lazy-SoA**
Apply SoA where SIMD matters (LLRs), retain lazy-copy where it doesn't (bits).

| Memory Type | Layout | Rationale | Reference |
|-------------|--------|-----------|----------|
| **LLRs** | Structure-of-Arrays | F/G functions dominate runtime; SIMD vectorizes across paths | [Fast_Polar_Decoders.md] §VII |
| **Bits** | Array-of-Structures + Lazy Copy | Accessed sequentially per path; rarely hot | [List_Decoding.md] §IV.A |

**Space Complexity:**
$$
\text{Total Memory} = \underbrace{L \cdot (2N - 1) \cdot 4}_{\text{LLR (SoA)}} + \underbrace{L \cdot (2N - 1) \cdot 1}_{\text{Bits (lazy)}} = O(L \cdot N)
$$

### 3.2 Structure-of-Arrays LLR Memory

**Key Insight:** The F and G functions operate on *all L paths simultaneously* for each β position. SoA layout makes L values contiguous for SIMD.

#### 3.2.1 SoA Memory Layout

```
SoA LLR Layout: soa_llr[λ][β] → [L contiguous f32]

┌─────────────────────────────────────────────────────────────────────┐
│ Layer λ=0: N positions, L paths per position                        │
├─────────────────────────────────────────────────────────────────────┤
│ β=0: [path_0, path_1, ..., path_{L-1}]  ← 16-path SIMD vector       │
│ β=1: [path_0, path_1, ..., path_{L-1}]                              │
│ ...                                                                 │
│ β=N-1: [path_0, path_1, ..., path_{L-1}]                            │
├─────────────────────────────────────────────────────────────────────┤
│ Layer λ=1: N/2 positions, L paths per position                      │
├─────────────────────────────────────────────────────────────────────┤
│ β=0: [path_0, path_1, ..., path_{L-1}]                              │
│ ...                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Reference:** [Fast_Polar_Decoders.md] §VII.A:
> "The memory for LLR values is arranged such that the values for all paths at a given bit position are stored contiguously."

#### 3.2.2 Rust Data Structure

```rust
/// Structure-of-Arrays LLR memory for SIMD-friendly access.
///
/// # Layout Invariant
/// For layer λ at position β, the L path values are stored at:
///   base_offset(λ) + β * L + path_idx
///
/// This enables `f32xL` SIMD operations across all paths.
///
/// # Reference
/// [Fast_Polar_Decoders.md] §VII.A
pub struct SoaLlrMemory {
    /// Flat backing buffer.
    /// Total size: L × Σ_{λ=0}^{m} 2^{m-λ} = L × (2N - 1)
    data: Vec<f32>,
    
    /// Pre-computed offsets into `data` for each layer.
    /// layer_offsets[λ] = L × Σ_{l=0}^{λ-1} 2^{m-l}
    layer_offsets: Vec<usize>,
    
    /// List size L (for stride calculation)
    list_size: usize,
    
    /// Depth m = log₂(N)
    depth: usize,
}

impl SoaLlrMemory {
    /// Access LLR slice for all L paths at (λ, β).
    ///
    /// Returns a slice of L contiguous f32 values.
    /// Suitable for SIMD vectorization.
    #[inline]
    pub fn get_path_slice(&self, lambda: usize, beta: usize) -> &[f32] {
        let offset = self.layer_offsets[lambda] + beta * self.list_size;
        &self.data[offset..offset + self.list_size]
    }
    
    /// Mutable access for LLR updates.
    #[inline]
    pub fn get_path_slice_mut(&mut self, lambda: usize, beta: usize) -> &mut [f32] {
        let offset = self.layer_offsets[lambda] + beta * self.list_size;
        &mut self.data[offset..offset + self.list_size]
    }
    
    /// Access single LLR value: llr[λ][β][ℓ]
    #[inline]
    pub fn get(&self, lambda: usize, beta: usize, path: usize) -> f32 {
        let offset = self.layer_offsets[lambda] + beta * self.list_size + path;
        self.data[offset]
    }
    
    /// Set single LLR value: llr[λ][β][ℓ] = value
    #[inline]
    pub fn set(&mut self, lambda: usize, beta: usize, path: usize, value: f32) {
        let offset = self.layer_offsets[lambda] + beta * self.list_size + path;
        self.data[offset] = value;
    }
}
```

### 3.3 Lazy-Copy Bit Memory

Bit arrays retain Tal & Vardy's lazy-copy semantics because:
1. Bit access is sequential per path (not SIMD-beneficial)
2. Space savings from sharing dominate runtime cost
3. Copy-on-write occurs only at path forks (K times per decode)

```rust
/// Lazy-copy bit array management.
///
/// # Specification Reference
/// [scl-algorithm-ref.md] §1.2, Algorithms 6-9
pub struct BitArrayBank {
    /// Physical storage: [layer][slot] → bit array
    /// Each bit array has 2^{m-λ} bit-pairs (2 bits each)
    arrays: Vec<Vec<Vec<u8>>>,  // [layer][slot][β] → 2-bit packed
    
    /// Indirection: [layer][path] → slot index
    /// Spec: `pathIndexToArrayIndex[λ][ℓ]`
    path_to_slot: Vec<Vec<usize>>,
    
    /// Reference counts: [layer][slot] → count
    /// Spec: `arrayReferenceCount[λ][s]`
    ref_counts: Vec<Vec<u8>>,
    
    /// Free slot stacks: [layer] → stack of free indices
    /// Spec: `inactiveArrayIndices[λ]`
    free_slots: Vec<Vec<usize>>,
}

impl BitArrayBank {
    /// Get immutable reference (no copy needed).
    pub fn get_read(&self, lambda: usize, path: usize) -> &[u8] {
        let slot = self.path_to_slot[lambda][path];
        &self.arrays[lambda][slot]
    }
    
    /// Get mutable reference with copy-on-write.
    ///
    /// If ref_count > 1, copies to a private slot first.
    /// Implements Algorithm 9 (`getArrayPointer_C`).
    pub fn get_write(&mut self, lambda: usize, path: usize) -> &mut [u8] {
        let slot = self.path_to_slot[lambda][path];
        
        if self.ref_counts[lambda][slot] == 1 {
            // Private — return directly
            return &mut self.arrays[lambda][slot];
        }
        
        // Shared — copy-on-write
        let new_slot = self.free_slots[lambda].pop()
            .expect("INV-LAZY: No free slots");
        
        // Clone data
        self.arrays[lambda][new_slot].clone_from(&self.arrays[lambda][slot]);
        
        // Update bookkeeping
        self.ref_counts[lambda][slot] -= 1;
        self.ref_counts[lambda][new_slot] = 1;
        self.path_to_slot[lambda][path] = new_slot;
        
        &mut self.arrays[lambda][new_slot]
    }
}
```

**Reference:** [List_Decoding_of_Polar_Codes.md] Algorithm 9 (`getArrayPointer_C`).

</memory_architecture>

---

## 4. SIMD Strategy with SoA Layout

<simd_strategy>

### 4.1 Vectorization via SoA Memory

**Core Principle:** With SoA layout, the F and G functions operate on L paths simultaneously. When L=16, each operation naturally maps to a single AVX-512 vector instruction.

| List Size | SIMD Width | Vectorization |
|-----------|------------|---------------|
| L=16 | AVX-512 (512-bit) | Perfect: 16×f32 per vector |
| L=8 | AVX2 (256-bit) | Perfect: 8×f32 per vector |
| L=32 | AVX-512 | 2 vectors per operation |

**Reference:** [Fast_Polar_Decoders.md] §VII.A:
> "We restructure the memory so that the L LLRs corresponding to different paths [...] are stored in consecutive memory locations."

### 4.2 Vectorized F/G Operations

The SoA layout enables vectorized F and G functions without gather/scatter:

```rust
/// SIMD-vectorized F-function across all L paths.
///
/// Computes: α_child[β][ℓ] = f(α_parent[2β][ℓ], α_parent[2β+1][ℓ])
/// for all ℓ ∈ [0, L) simultaneously.
///
/// # Reference
/// [LLR-Based_SCL.md] §II Eq. (8a)
/// [Fast_Polar_Decoders.md] §VII.A
#[cfg(feature = "simd")]
fn f_function_simd(
    soa_llr: &mut SoaLlrMemory,
    lambda: usize,
    beta_range: std::ops::Range<usize>,
) {
    use std::simd::{f32x16, SimdFloat};
    
    let l = soa_llr.list_size;
    debug_assert!(l <= 16, "L > 16 requires loop unrolling");
    
    for beta in beta_range {
        // Load L values from α_parent[2β] and α_parent[2β+1]
        let alpha_even = f32x16::from_slice(
            soa_llr.get_path_slice(lambda - 1, 2 * beta)
        );
        let alpha_odd = f32x16::from_slice(
            soa_llr.get_path_slice(lambda - 1, 2 * beta + 1)
        );
        
        // Vectorized min-sum: sign(a) * sign(b) * min(|a|, |b|)
        let sign = alpha_even.signum() * alpha_odd.signum();
        let min_abs = alpha_even.abs().simd_min(alpha_odd.abs());
        let result = sign * min_abs;
        
        // Store L results to α_child[β]
        result.copy_to_slice(
            soa_llr.get_path_slice_mut(lambda, beta)
        );
    }
}

/// SIMD-vectorized G-function across all L paths.
///
/// Computes: α_child[β][ℓ] = g(α_parent[2β][ℓ], α_parent[2β+1][ℓ], β_left[β][ℓ])
/// for all ℓ ∈ [0, L) simultaneously.
///
/// # Reference
/// [LLR-Based_SCL.md] §II Eq. (8b)
/// [Fast_Polar_Decoders.md] §VII.A
#[cfg(feature = "simd")]
fn g_function_simd(
    soa_llr: &mut SoaLlrMemory,
    bit_bank: &BitArrayBank,
    lambda: usize,
    beta_range: std::ops::Range<usize>,
) {
    use std::simd::f32x16;
    
    for beta in beta_range {
        let alpha_even = f32x16::from_slice(
            soa_llr.get_path_slice(lambda - 1, 2 * beta)
        );
        let alpha_odd = f32x16::from_slice(
            soa_llr.get_path_slice(lambda - 1, 2 * beta + 1)
        );
        
        // Load β_left bits for all L paths (requires bit-to-f32 expansion)
        let beta_left = load_bits_as_f32x16(bit_bank, lambda, beta);
        
        // Vectorized: α_odd + (1 - 2*β_left) * α_even
        let sign_factor = f32x16::splat(1.0) - f32x16::splat(2.0) * beta_left;
        let result = alpha_odd + sign_factor * alpha_even;
        
        result.copy_to_slice(
            soa_llr.get_path_slice_mut(lambda, beta)
        );
    }
}
```

### 4.3 Alignment Requirements

| Component | Required Alignment | Reason |
|-----------|-------------------|--------|
| `SoaLlrMemory.data` | 64 bytes | AVX-512 loads/stores |
| `layer_offsets[λ]` | Multiple of L×4 | Path slice alignment |

```rust
/// Allocate 64-byte aligned LLR buffer for AVX-512.
#[cfg(feature = "simd")]
fn allocate_aligned_llr(total_elements: usize) -> Vec<f32> {
    // Use aligned_vec crate or manual allocation
    aligned_vec::AVec::<f32, aligned_vec::A64>::new()
        .with_capacity(total_elements)
        .into_iter()
        .chain(std::iter::repeat(0.0f32))
        .take(total_elements)
        .collect()
}
```

### 4.4 Performance Expectation

With SoA + AVX-512 (L=16, N=1024):
- F/G operations: 16 paths × 1 vector op = **16× throughput** vs. scalar
- Expected decode throughput: **>100 Mbps** per core

**Reference:** [Fast_Polar_Decoders.md] Table VI reports ~250 Mbps at N=2048, L=4 with SIMD.

</simd_strategy>

---

## 5. CRC Integration

<crc_integration>

### 5.1 CRC Hook Points

Per [CRC-Aided_Decoding_of_Polar_Codes.md] §III.A, CRC checking occurs at path selection:

```rust
impl SclDecoder {
    /// Final path selection with optional CRC.
    ///
    /// # Algorithm
    /// 1. Sort active paths by metric (ascending = most likely first)
    /// 2. For each path, extract info bits and check CRC
    /// 3. Return first path passing CRC, or best path if none pass
    ///
    /// # Reference
    /// [CRC-Aided_Decoding_of_Polar_Codes.md] §III.A Step (A.4)
    fn select_best_path(&self) -> (Vec<u8>, bool) {
        // Collect active paths with metrics
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(self.list_size);
        for l in 0..self.list_size {
            if self.active_paths[l] {
                candidates.push((l, self.path_metrics[l]));
            }
        }
        
        // Sort by metric (ascending — lower is better)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        if let Some(crc_poly) = self.crc_poly {
            // CA-SCL: Check CRC for each path in order
            for (path_idx, _metric) in &candidates {
                let info_bits = self.extract_info_bits(*path_idx);
                if self.verify_crc(&info_bits, crc_poly, self.crc_len.unwrap()) {
                    return (info_bits, true);  // CRC passed
                }
            }
            // No path passed CRC — return best anyway with failure flag
            let info_bits = self.extract_info_bits(candidates[0].0);
            (info_bits, false)
        } else {
            // Standard SCL: Return path with lowest metric
            let info_bits = self.extract_info_bits(candidates[0].0);
            (info_bits, true)
        }
    }
    
    /// Extract information bits from decoded path.
    fn extract_info_bits(&self, path_idx: usize) -> Vec<u8> {
        let c_0 = self.get_bit_array(0, path_idx);
        let mut info_bits = Vec::with_capacity(self.info_set.count_ones());
        
        for beta in 0..self.block_len {
            if self.info_set[beta] {
                info_bits.push(read_bit(c_0, beta, 0));
            }
        }
        
        info_bits
    }
    
    /// Verify CRC checksum.
    ///
    /// # Reference
    /// [CRC-Aided_Decoding_of_Polar_Codes.md] §III.A
    fn verify_crc(&self, data: &[u8], poly: u32, crc_len: usize) -> bool {
        // CRC computation (standard polynomial division)
        let mut crc: u32 = 0;
        for &bit in data.iter().take(data.len() - crc_len) {
            let msb = (crc >> (crc_len - 1)) & 1;
            crc = ((crc << 1) | (bit as u32)) & ((1 << crc_len) - 1);
            if msb == 1 {
                crc ^= poly;
            }
        }
        
        // Compare with transmitted CRC
        let mut transmitted_crc: u32 = 0;
        for &bit in data.iter().skip(data.len() - crc_len) {
            transmitted_crc = (transmitted_crc << 1) | (bit as u32);
        }
        
        crc == transmitted_crc
    }
}
```

### 5.2 Incremental CRC (Optimization)

For efficiency, CRC can be updated incrementally as each information bit is decoded:

```rust
/// Update CRC state with a new information bit.
///
/// Called during `continuePaths_UnfrozenBit` when a path survives.
#[inline]
fn update_crc_state(&mut self, path_idx: usize, bit: u8) {
    if let Some(poly) = self.crc_poly {
        let crc_len = self.crc_len.unwrap();
        let state = &mut self.crc_states[path_idx];
        let msb = (*state >> (crc_len - 1)) & 1;
        *state = ((*state << 1) | (bit as u32)) & ((1 << crc_len) - 1);
        if msb == 1 {
            *state ^= poly;
        }
    }
}
```

</crc_integration>

---

## 6. SclProgram Compiler

<scl_program_compiler>

### 6.1 Design Rationale

**Problem:** Recursive control flow in `recursivelyCalcL()` and `recursivelyUpdateC()` causes:
1. **Stack thrashing** — 10 recursive calls for N=1024
2. **Branch misprediction** — Conditional frozen/info bit checks at each φ
3. **Poor instruction cache locality** — Pointer-chasing through call graph

**Solution:** Pre-compile the decode traversal as a flat instruction tape.

**Reference:** [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §V:
> "By using a pre-compiled list of instructions, the controller is reduced to fetching and decoding instructions, making it much simpler than the state machine-based solution."

### 6.2 Compiler Algorithm

The compiler transforms `(N, frozen_set)` → `Vec<SclOp>` at decoder construction time:

```rust
/// Compile frozen_set into instruction tape.
///
/// # Algorithm (Offline)
/// 1. Simulate recursive decode traversal (DFS order)
/// 2. Emit F/G/Combine opcodes at each node visit
/// 3. Emit Frozen/Info opcodes at leaf nodes
/// 4. (Phase 2.5) Recognize specialized node patterns
///
/// # Reference
/// [Fast_Polar_Decoders.md] §V Table III
pub fn compile_scl_program(
    block_len: usize,
    frozen_set: &BitVec,
) -> SclProgram {
    let depth = block_len.trailing_zeros() as usize;
    let mut ops = Vec::with_capacity(3 * block_len);  // ~3N ops typical
    
    // Recursive compilation (runs once at construction, not per-decode)
    compile_subtree(&mut ops, depth, 0, block_len, frozen_set);
    
    SclProgram {
        ops,
        block_len,
        info_bits: block_len - frozen_set.count_ones(),
        frozen_indices: frozen_set.iter_ones().map(|i| i as u16).collect(),
    }
}

/// Compile a subtree rooted at layer λ, covering positions [phi_start, phi_start + size).
///
/// Emits instructions in decode traversal order (pre-order DFS with interleaved ops).
fn compile_subtree(
    ops: &mut Vec<SclOp>,
    lambda: usize,
    phi_start: usize,
    size: usize,
    frozen_set: &BitVec,
) {
    if lambda == 0 {
        // Leaf node: emit Frozen or Info
        let phi = phi_start;
        if frozen_set[phi] {
            ops.push(SclOp::Frozen { phi: phi as u16 });
        } else {
            ops.push(SclOp::Info { phi: phi as u16 });
        }
        return;
    }
    
    let half = size / 2;
    
    // === Left child (even phase) ===
    // 1. Compute LLRs for left child: F-function
    ops.push(SclOp::F { lambda: lambda as u8 });
    
    // 2. Recurse into left child
    compile_subtree(ops, lambda - 1, phi_start, half, frozen_set);
    
    // === Right child (odd phase) ===
    // 3. Compute LLRs for right child: G-function
    ops.push(SclOp::G { lambda: lambda as u8 });
    
    // 4. Recurse into right child
    compile_subtree(ops, lambda - 1, phi_start + half, half, frozen_set);
    
    // 5. Combine partial sums after both children complete
    ops.push(SclOp::Combine { lambda: lambda as u8 });
}
```

### 6.3 VM Dispatch Loop

The decode loop becomes a simple instruction dispatch:

```rust
impl SclDecoder {
    /// Execute pre-compiled program to decode LLRs.
    ///
    /// Replaces recursive `recursivelyCalcL` / `recursivelyUpdateC`.
    pub fn execute_program(&mut self, channel_llrs: &[f32]) -> Result<(Vec<u8>, bool), DecodeError> {
        // Load channel LLRs into layer 0
        self.load_channel_llrs(channel_llrs);
        
        // VM dispatch loop — flat, no recursion
        for op in &self.program.ops {
            match op {
                SclOp::F { lambda } => self.exec_f(*lambda as usize),
                SclOp::G { lambda } => self.exec_g(*lambda as usize),
                SclOp::Combine { lambda } => self.exec_combine(*lambda as usize),
                SclOp::G0R { lambda } => self.exec_g0r(*lambda as usize),
                SclOp::Frozen { phi } => self.exec_frozen(*phi as usize),
                SclOp::Info { phi } => self.exec_info(*phi as usize),
                
                // Phase 2.5: Specialized nodes (feature-gated)
                #[cfg(feature = "fast-nodes")]
                SclOp::Rate1 { start, len } => self.exec_rate1(*start, *len),
                #[cfg(feature = "fast-nodes")]
                SclOp::Spc { start, len } => self.exec_spc(*start, *len),
                #[cfg(feature = "fast-nodes")]
                SclOp::Rep { start, len } => self.exec_rep(*start, *len),
            }
        }
        
        Ok(self.select_best_path())
    }
}
```

### 6.4 Example: N=8 Instruction Tape

For a simple N=8 polar code with frozen_set = {0, 1, 2, 4}:

```
Instruction Tape (15 ops):
┌────┬────────────────┬───────────────────────────────────────┐
│ PC │ Instruction    │ Comment                               │
├────┼────────────────┼───────────────────────────────────────┤
│  0 │ F { λ=3 }      │ Compute α[2] from α[1]               │
│  1 │ F { λ=2 }      │ Compute α[1] from α[0]               │
│  2 │ F { λ=1 }      │ Compute α[0] for φ=0                 │
│  3 │ Frozen { φ=0 } │ Process frozen bit 0                 │
│  4 │ G { λ=1 }      │ Compute α[0] for φ=1                 │
│  5 │ Frozen { φ=1 } │ Process frozen bit 1                 │
│  6 │ Combine { λ=1 }│ Update β[1] from β[0]                │
│  7 │ G { λ=2 }      │ Compute α[1] for φ=2,3               │
│  8 │ F { λ=1 }      │ Compute α[0] for φ=2                 │
│  9 │ Frozen { φ=2 } │ Process frozen bit 2                 │
│ 10 │ G { λ=1 }      │ Compute α[0] for φ=3                 │
│ 11 │ Info { φ=3 }   │ Fork/prune at info bit 3             │
│ 12 │ Combine { λ=1 }│ Update β[1]                          │
│ 13 │ Combine { λ=2 }│ Update β[2]                          │
│ ... │ (right half)   │ Similar pattern for φ=4..7           │
└────┴────────────────┴───────────────────────────────────────┘
```

### 6.5 Phase 2.5: Node Specialization (Reserved)

The compiler can be extended to recognize specialized node patterns:

| Pattern | Detection Condition | Replacement | Reference |
|---------|---------------------|-------------|-----------|
| Rate-0 | All children frozen | G0R opcode | [Fast_Polar_Decoders.md] §IV.D |
| Rate-1 | All children info | Rate1 opcode | [Fast_Polar_Decoders.md] §IV.A |
| REP | First child frozen, rest info | Rep opcode | [Fast_Polar_Decoders.md] §IV.B |
| SPC | First child info, rest frozen | Spc opcode | [Fast_Polar_Decoders.md] §IV.A |

```rust
/// (Phase 2.5) Detect specialized node at compile time.
#[cfg(feature = "fast-nodes")]
fn detect_node_type(
    frozen_set: &BitVec,
    phi_start: usize,
    size: usize,
) -> NodeType {
    let frozen_count = frozen_set[phi_start..phi_start + size]
        .iter()
        .filter(|&&b| b)
        .count();
    
    match frozen_count {
        n if n == size => NodeType::Rate0,
        0 => NodeType::Rate1,
        n if n == size - 1 && !frozen_set[phi_start] => NodeType::Spc,
        n if n == size - 1 && !frozen_set[phi_start + size - 1] => NodeType::Rep,
        _ => NodeType::General,
    }
}
```

**Implementation Note:** Node specialization is reserved for Phase 2.5 and feature-gated behind `fast-nodes`.

### 6.6 Literature Citations for Further Development

| Concept | Primary Source | Key Section |
|---------|----------------|-------------|
| Instruction-based architecture | [Fast_Polar_Decoders.md] | §V "Decoder Architecture" |
| Node function table | [Fast_Polar_Decoders.md] | Table III |
| Simplified SC decoder tree | [Fast_Polar_Decoders.md] | §IV.E, Fig. 4 |
| Rate-0/Rate-1 nodes | [Fast_Polar_Decoders.md] | §IV.A, §IV.D |
| REP/SPC nodes | [Fast_Polar_Decoders.md] | §IV.A-B, Eq. (7)-(8) |
| Memory layout for SIMD | [Fast_Polar_Decoders.md] | §VII.A |

</scl_program_compiler>

---

## 7. API Surface

<api_surface>

### 7.1 Public Interface

```rust
impl SclDecoder {
    /// Create a new SCL decoder with given configuration.
    ///
    /// Compiles the SclProgram (instruction tape) at construction time.
    ///
    /// # Arguments
    /// * `config` - Decoder parameters (N, K, L, design_snr, CRC options)
    ///
    /// # Panics
    /// Panics if `block_len` is not a power of 2.
    pub fn new(config: SclConfig) -> Self {
        assert!(config.block_len.is_power_of_two(), "N must be power of 2");
        
        // Compile instruction tape (offline, once per code configuration)
        let program = compile_scl_program(config.block_len, &config.frozen_set);
        
        // ... initialization per Algorithm 5 ...
    }
    
    /// Decode received LLRs to information bits.
    ///
    /// Uses VM-based control flow (instruction dispatch) instead of recursion.
    ///
    /// # Arguments
    /// * `channel_llrs` - Channel LLR values: $L_i = \ln\frac{W(y_i|0)}{W(y_i|1)}$
    ///
    /// # Returns
    /// * `Ok((info_bits, crc_passed))` - Decoded information bits and CRC status
    /// * `Err(DecodeError)` - Decoding failure (e.g., all paths pruned)
    ///
    /// # Specification Reference
    /// [scl-algorithm-ref.md](../specs/scl-algorithm-ref.md) Algorithm 12 (`SCL_Decode`)
    /// [Fast_Polar_Decoders.md] §V (instruction-based architecture)
    pub fn decode(&mut self, channel_llrs: &[f32]) -> Result<(Vec<u8>, bool), DecodeError> {
        assert_eq!(channel_llrs.len(), self.block_len, "LLR length mismatch");
        
        // Reset decoder state
        self.reset();
        
        // Initialize first path and load channel LLRs
        let l = self.assign_initial_path();
        self.load_channel_llrs(channel_llrs, l);
        
        // Execute pre-compiled instruction tape (VM dispatch loop)
        self.execute_program()
    }
    
    /// Reset decoder state for reuse.
    ///
    /// Clears all path state without deallocating memory.
    pub fn reset(&mut self) {
        // Reset path tracking
        self.inactive_path_stack.clear();
        for l in (0..self.list_size).rev() {
            self.inactive_path_stack.push(l);
        }
        self.active_paths.fill(false);
        
        // Reset array management
        for lambda in 0..=self.depth {
            self.inactive_array_stacks[lambda].clear();
            for s in (0..self.list_size).rev() {
                self.inactive_array_stacks[lambda].push(s);
            }
            self.array_ref_counts[lambda].fill(0);
        }
        
        // Reset metrics
        self.path_metrics.fill(0.0);
        
        // Reset CRC states
        self.crc_states.fill(0);
    }
}
```

### 7.2 Error Types

```rust
/// Errors that can occur during SCL decoding.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("All decoding paths were pruned")]
    AllPathsPruned,
    
    #[error("Input LLR length {got} does not match block length {expected}")]
    LengthMismatch { got: usize, expected: usize },
    
    #[error("Internal invariant violation: {0}")]
    InternalError(String),
}
```

### 7.3 PyO3 Binding Stub

```rust
#[pyfunction]
#[pyo3(signature = (channel_llrs, block_len, info_bits, list_size, design_snr, crc_poly=None))]
fn scl_decode(
    channel_llrs: PyReadonlyArrayDyn<f32>,
    block_len: usize,
    info_bits: usize,
    list_size: usize,
    design_snr: f32,
    crc_poly: Option<u32>,
) -> PyResult<(Vec<u8>, bool)> {
    let config = SclConfig {
        block_len,
        info_bits,
        list_size,
        design_snr,
        crc_poly,
        crc_len: crc_poly.map(|p| 32 - p.leading_zeros() as usize - 1),
        use_minsum: true,
    };
    
    let mut decoder = SclDecoder::new(config);
    let llrs = channel_llrs.as_slice()?;
    
    decoder.decode(llrs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
```

</api_surface>

---

## 8. Complexity Analysis

<complexity_analysis>

### 8.1 Time Complexity

| Operation | Per-Call Cost | Calls | Total |
|-----------|---------------|-------|-------|
| F/G ops (via VM) | $O(L \cdot 2^{m-\lambda})$ | $O(N)$ | $O(L \cdot N \log N)$ |
| Combine ops | $O(L \cdot 2^{m-\lambda})$ | $O(N)$ | $O(L \cdot N \log N)$ |
| Info bit (fork/prune) | $O(L \log L)$ | $K$ | $O(K \cdot L \log L)$ |
| Bit array copy (lazy) | $O(2^{m-\lambda})$ | $O(L \cdot m)$ | $O(L \cdot N)$ amortized |

**Total:** $O(L \cdot N \log N)$

**VM Overhead:** Instruction dispatch is $O(1)$ per op; total ops $\approx 3N$.

**Reference:** [List_Decoding_of_Polar_Codes.md] §IV.C Theorem 8.

### 8.2 Space Complexity

| Component | Size |
|-----------|------|
| SoA LLR memory | $L \cdot (2N - 1) \cdot 4$ bytes |
| Lazy-copy bit memory | $L \cdot (2N - 1)$ bytes |
| SclProgram (instruction tape) | $\approx 3N \cdot 4$ bytes |
| Path management | $O(L \cdot m)$ |
| **Total** | $O(L \cdot N)$ |

**Example:** $N = 1024$, $L = 16$:
- SoA LLR memory: $16 \times 2047 \times 4 = 131$ KB
- Bit memory: $16 \times 2047 = 32$ KB
- Instruction tape: $3 \times 1024 \times 4 = 12$ KB
- **Total:** ~175 KB per decoder instance

</complexity_analysis>

---

## 9. Verification Checklist

<verification_checklist>

Before Phase 3 integration, verify:

**VM Architecture Invariants:**
- [ ] **INV-VM-1:** `program.ops.len() ≈ 3N` (expected instruction count)
- [ ] **INV-VM-2:** Each F is followed by its subtree, then corresponding G
- [ ] **INV-VM-3:** VM dispatch loop visits all N leaf positions exactly once

**Hybrid Memory Invariants:**
- [ ] **INV-SOA-1:** `soa_llr.get_path_slice(λ, β)` returns L contiguous f32 values
- [ ] **INV-SOA-2:** SoA slice addresses are 64-byte aligned (when SIMD enabled)
- [ ] **INV-LAZY-1:** Bit array `ref_count > 1` triggers copy-on-write
- [ ] **INV-LAZY-2:** `killPath` decrements ref counts and returns slots to free stacks
- [ ] **INV-LAZY-3:** Sum of ref counts = number of active paths (per layer)

**Path Metric Invariants:**
- [ ] **INV-METRIC-1:** Path metrics updated at both frozen and info bits
- [ ] **INV-METRIC-2:** Path selection chooses minimum metric (not maximum)
- [ ] **INV-CRC-1:** CRC-aided selection iterates paths in metric-ascending order

**Test Vectors:**
1. **SC equivalence:** L=1 output matches Phase 1 SC decoder
2. **Known codeword:** All-zero codeword at high SNR decodes correctly
3. **CRC detection:** Valid vs. corrupted CRC distinction
4. **Memory invariants:** Ref count sums remain consistent through decode
5. **VM round-trip:** Compiled program decodes same as recursive reference

</verification_checklist>

---

## Appendix A: Literature Reference Summary

<literature_references>

| Citation | Section Used | Key Content |
|----------|--------------|-------------|
| [List_Decoding_of_Polar_Codes.md] §IV.A | Lazy-copy bits (§3.3) | Algorithms 5-9, copy-on-write mechanism |
| [List_Decoding_of_Polar_Codes.md] §III | Space reduction | Eq. (11)-(12), phase-independent arrays |
| [LLR-Based_SCL.md] §II | SIMD F/G (§4.2) | Eq. (7)-(9), f/g functions |
| [LLR-Based_SCL.md] §III | Path metrics | Eq. (10)-(12), φ function |
| [Fast_Polar_Decoders.md] §V | **VM architecture (§6)** | Instruction-based control flow |
| [Fast_Polar_Decoders.md] §VII.A | **SoA memory (§3.2)** | SIMD-friendly LLR layout |
| [Fast_Polar_Decoders.md] §IV | Node specialization (§6.5) | Rate-0/1, REP, SPC patterns |
| [CRC-Aided_Decoding.md] §III.A | CRC integration | CA-SCL algorithm steps A.1-A.4 |

**Primary Architectural References:**
- **VM Control Flow:** [Fast_Polar_Decoders_Algorithm_and_Implementation.md] §V, Table III
- **Hybrid Memory:** [Fast_Polar_Decoders.md] §VII.A + [List_Decoding.md] §IV.A
- **Node Specialization (Phase 2.5):** [Fast_Polar_Decoders.md] §IV.A-E

</literature_references>

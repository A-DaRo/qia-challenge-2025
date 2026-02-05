# Phase 2: SCL Decoder — Structured Completion Plan

<metadata>
plan_id: phase2-scl-decoder-plan
version: 1.1.0
status: confirmed
created: 2026-02-02
author: context-engineer
purpose: Structured plan enabling document creation for P2: SCL Decoder phase
</metadata>

---

## Executive Summary

This plan addresses the completion of two P2 documents:

1. **SPEC: Rust Polar Crate API** (`specs/rust-polar-crate.md`)
2. **IMPL: Phase 2 - SCL Decoder** (`impl/phase2-scl-decoder.md`)

A **critical prerequisite** emerged from Phase 1 testing: the Encoder/Decoder indexing mismatch (Natural vs. Bit-Reversed order) must be resolved before SCL implementation. This plan incorporates that fix as Phase 2.0.

---

## Critical Issue: Bit-Reversal Permutation Alignment

### Analysis from System Architect

> The `AssertionError` in `test_sc_decode_noiseless` revealed a fundamental structural mismatch between the standard Polar code generator matrix definition (implied by the `PolarEncoder`) and the recursive structure expected by the `SCDecoder`.

**Root Cause:**

| Component | Index Order | Convention |
|-----------|-------------|------------|
| `PolarEncoder` | Natural order | $\mathbf{x} = \mathbf{u} \cdot \mathbf{G}_N$ where $\mathbf{G}_N = \mathbf{F}^{\otimes n}$ |
| `SCDecoder` | Bit-reversed order | Tree recursion expects $\mathbf{G}_N = \mathbf{B}_N \cdot \mathbf{F}^{\otimes n}$ |
| `construction.rs` | Bit-reversed (correct for decoder) | Reliability indices in decoder-aligned order |

**Literature Reference:**
- [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md), Eq. (1):
  > $\mathbf{G}_n \triangleq \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}^{\otimes n} \mathbf{B}_n$, where $\mathbf{B}_n$ is the **bit-reversal permutation**.

### Resolution Strategy

Two options (recommend **Option A**):

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A** | Permute input in `PolarEncoder.encode()` | Single modification point; decoder unchanged | Slightly non-standard encoder interface |
| **B** | Permute output in `SCDecoder.decode()` | Encoder remains textbook | Permutation cost in decoder hot path |

**Phase 2.0 Action:** Implement `bit_reverse_permutation(index, n_bits)` utility and apply to encoder message placement.

---

## Document 1: SPEC — Rust Polar Crate API

### Purpose

Detailed specification of the `caligo-codecs` Rust crate's public API for Phase 2, including:
- Memory layout for path management (SCL)
- Error handling taxonomy
- GIL release points for Python parallelism
- Thread-safety guarantees

### Document Structure

```
specs/rust-polar-crate.md
├── §1 Overview & Scope
├── §2 Type System
│   ├── §2.1 Core Rust Types
│   ├── §2.2 PyO3 Exports
│   └── §2.3 Error Types
├── §3 Memory Layout
│   ├── §3.1 LLR Storage (SCL)
│   ├── §3.2 Path Metrics
│   └── §3.3 Partial Sum Trees
├── §4 API Surface
│   ├── §4.1 PolarCode Configuration
│   ├── §4.2 SCLDecoder Interface
│   └── §4.3 CRC Integration
├── §5 GIL & Parallelism
├── §6 Test Vector Format
└── Appendix: Bit-Reversal Permutation
```

### Codebase References (Precise)

| Section | File | Lines | Content |
|---------|------|-------|---------|
| §2.1 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 24-52 | `SCDecoder` struct, `SCDecodeResult` |
| §2.1 | [encoder.rs](../../../caligo/_native/src/polar/encoder.rs) | 23-37 | `PolarEncoder` struct |
| §2.1 | [construction.rs](../../../caligo/_native/src/polar/construction.rs) | 12-19 | `ConstructionMethod` enum |
| §2.2 | [mod.rs](../../../caligo/_native/src/polar/mod.rs) | 1-11 | Public re-exports |
| §2.3 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 14-22 | `DecoderError` enum |
| §2.3 | [encoder.rs](../../../caligo/_native/src/polar/encoder.rs) | 12-21 | `EncoderError` enum |
| §3.1 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 46-51 | `llr_memory`, `bit_memory` layout |
| §4.3 | [crc.rs](../../../caligo/_native/src/polar/crc.rs) | 1-80 | CRC-16 API (Phase 1 stub) |
| §5 | [codec.py](../../../caligo/caligo/reconciliation/strategies/codec.py) | 36-70 | Python invocation patterns |

### Literature References (Precise)

| Section | Reference | Key Content |
|---------|-----------|-------------|
| §3.1 | [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV | $P_\lambda[\phi, \beta]$ probabilities array, $B_\lambda[\phi, \beta]$ bit array |
| §3.2 | [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §III | LLR-domain path metric: $PM^{(l)} = PM^{(l)} + \ln(1 + e^{-(1-2\hat{u}_i^{(l)})L_i^{(l)}})$ |
| §3.3 | [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §III | Partial sum propagation for Rate-0, Rate-1, Rep, SPC nodes |
| §4.3 | [CRC-Aided_Decoding_of_Polar_Codes.md](../../literature/CRC-Aided_Decoding_of_Polar_Codes.md) §III | CA-SCL algorithm: CRC check on $L$ candidate paths |
| Appendix | [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) Eq. (1) | $\mathbf{B}_n$ bit-reversal permutation definition |

### Interface Preview (Expanded from initialization.md)

```rust
// === Core Types ===

/// Polar code configuration.
pub struct PolarCode {
    /// Block length N = 2^n
    pub block_length: usize,
    /// Information + CRC length K
    pub message_length: usize,
    /// Frozen bit mask (true = frozen)
    pub frozen_mask: BitVec<u64, Lsb0>,
    /// CRC polynomial (None = no CRC)
    pub crc_poly: Option<u32>,
}

/// SCL decoder with configurable list size.
pub struct SCLDecoder {
    /// Polar code parameters
    code: PolarCode,
    /// List size L
    list_size: usize,
    /// LLR memory: [L][n+1][N]
    llr_memory: Vec<Vec<Vec<f32>>>,
    /// Partial sum memory: [L][n+1][N]
    bit_memory: Vec<Vec<Vec<u8>>>,
    /// Path metrics: [L]
    path_metrics: Vec<f32>,
    /// Active path mask
    active_paths: BitVec<u64, Lsb0>,
}

/// Decode result with soft information.
pub struct SCLDecodeResult {
    /// Decoded message bits (K - CRC bits)
    pub message: Vec<u8>,
    /// Soft output LLRs (for SISO interface)
    pub soft_output: Vec<f32>,
    /// Best path metric
    pub path_metric: f32,
    /// CRC passed (if CRC configured)
    pub crc_valid: Option<bool>,
    /// Number of paths explored
    pub paths_explored: usize,
}

// === PyO3 Exports ===

#[pyclass]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCLDecoder,
}

#[pymethods]
impl PyPolarCodec {
    /// Construct codec with Gaussian Approximation.
    #[new]
    fn new(
        n: usize,           // Block length
        k: usize,           // Message length (including CRC)
        list_size: usize,   // L
        crc_poly: Option<u32>,
        design_snr_db: f64,
    ) -> PyResult<Self>;

    /// Encode message to codeword.
    fn encode(&self, message: PyReadonlyArray1<u8>) -> PyResult<Py<PyArray1<u8>>>;

    /// Decode LLRs to message + soft output.
    /// Releases GIL during computation.
    fn decode_soft(
        &self,
        py: Python,
        llr: PyReadonlyArray1<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u8>>, f32)>;
    
    /// Get frozen mask as numpy array.
    fn frozen_mask(&self) -> PyResult<Py<PyArray1<u8>>>;
}
```

---

## Document 2: IMPL — Phase 2 SCL Decoder

### Purpose

Implementation guide for extending the Phase 1 SC decoder to full SCL with:
- List-based path management (L paths)
- CRC-aided path selection
- LLR-domain numerically stable metrics
- GIL release for Python parallelism

### Document Structure

```
impl/phase2-scl-decoder.md
├── §1 Overview & Prerequisites
│   └── §1.1 Bit-Reversal Fix (Phase 2.0)
├── §2 SCL Algorithm
│   ├── §2.1 Path Splitting at Information Bits
│   ├── §2.2 Pruning to L Best Paths
│   └── §2.3 Path Metric Updates (LLR Domain)
├── §3 Memory Management
│   ├── §3.1 Lazy Copy Optimization
│   └── §3.2 Stack-Allocated Path Arrays
├── §4 CRC Integration
│   ├── §4.1 Message Bit Extraction
│   └── §4.2 CRC-Aided Selection
├── §5 Rust Implementation
│   ├── §5.1 SCLDecoder Struct
│   ├── §5.2 Core Methods
│   └── §5.3 SIMD Opportunities
├── §6 PyO3 Bindings
│   ├── §6.1 GIL Release Pattern
│   └── §6.2 Buffer Protocol
├── §7 Test Vectors & Validation
└── Appendix: Performance Tuning
```

### Codebase References (Precise)

| Section | File | Lines | Content |
|---------|------|-------|---------|
| §1.1 | [encoder.rs](../../../caligo/_native/src/polar/encoder.rs) | 100-115 | `butterfly_transform()` — modification point |
| §1.1 | [construction.rs](../../../caligo/_native/src/polar/construction.rs) | 52-63 | Frozen mask generation (already bit-reversed) |
| §2 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 84-148 | `decode()` — extend to SCL |
| §2.3 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 115-123 | Path metric update (currently SC-style) |
| §3 | [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) | 65-70 | Memory allocation pattern |
| §4 | [crc.rs](../../../caligo/_native/src/polar/crc.rs) | 18-45 | CRC computation, verification |
| §6 | [codec.py](../../../caligo/caligo/reconciliation/strategies/codec.py) | 95-140 | Current `decode_baseline()` for pattern reference |
| §6.1 | [numba_kernels.py](../../../caligo/caligo/scripts/numba_kernels.py) | 718-771 | GIL-released kernel pattern |

### Literature References (Precise)

| Section | Reference | Key Content |
|---------|-----------|-------------|
| §2.1 | [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-A | Path doubling: "doubles the number of decoding paths for each information bit $u_i$" |
| §2.2 | [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-B | Pruning: "uses a pruning procedure to discard all but the $L$ most likely paths" |
| §2.3 | [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §III-A | LLR path metric: Eq. (15) $PM^{(l)} \leftarrow PM^{(l)} + \ln(1 + e^{-|L_i^{(l)}|})$ when $\hat{u}_i^{(l)} \neq \frac{1-\text{sign}(L_i^{(l)})}{2}$ |
| §2.3 | [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §III-B | Approximation: $\ln(1 + e^{-|x|}) \approx 0$ for $|x| > T$ (threshold ~10) |
| §3.1 | [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV-C | Lazy copy: "Specifically, our implementation maintains for every active path a set of pointers to arrays, rather than a set of arrays" |
| §4.2 | [CRC-Aided_Decoding_of_Polar_Codes.md](../../literature/CRC-Aided_Decoding_of_Polar_Codes.md) §III | CA-SCL: "the decoder outputs the candidate sequences into CRC detector and the latter feeds the check results back" |
| §5.3 | [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §III-B | SIMD-friendly f-function: $f(\alpha, \beta) = \text{sign}(\alpha)\text{sign}(\beta)\min(|\alpha|, |\beta|)$ |

### Test Vectors (Expanded from initialization.md)

| ID | N | K | L | Design SNR | Noise SNR | Expected FER | Source | Validation |
|----|---|---|---|------------|-----------|--------------|--------|------------|
| TV-BRP-01 | 8 | 4 | 1 | 2.0 dB | — | Exact match | Phase 2.0 | Bit-reversal alignment |
| TV-SC-01 | 1024 | 512 | 1 | 2.0 dB | 2.0 dB | < 0.1 | Tal & Vardy | Regression vs Phase 1 |
| TV-SCL-01 | 1024 | 512 | 1 | 2.0 dB | 2.0 dB | < 0.1 | Tal & Vardy | SCL L=1 = SC baseline |
| TV-SCL-02 | 1024 | 512 | 8 | 2.0 dB | 2.0 dB | < 0.01 | Tal & Vardy | Default list size |
| TV-SCL-03 | 1024 | 512 | 32 | 2.0 dB | 2.0 dB | < 0.001 | Tal & Vardy | Maximum list size |
| TV-CASCL-01 | 1024 | 496+16 | 8 | 2.0 dB | 1.5 dB | < 0.001 | Niu & Chen | CRC-16-CCITT |
| TV-CASCL-02 | 4096 | 2032+16 | 8 | 2.0 dB | 1.0 dB | < 0.0001 | ADR-0001 target | Production config |

### Acceptance Criteria

```
- [ ] AC-2.0.1: `test_encoder_decoder_roundtrip` passes with bit-reversal fix
- [ ] AC-2.0.2: Bit-reversal permutation utility is O(1) per index

- [ ] AC-2.1.1: SCLDecoder(L=1) produces identical output to SCDecoder
- [ ] AC-2.1.2: SCLDecoder(L=8) achieves FER < 0.01 at SNR=2dB, N=1024, K=512
- [ ] AC-2.1.3: SCLDecoder(L=32) achieves FER < 0.001 at SNR=2dB, N=1024, K=512

- [ ] AC-2.2.1: CRC-aided selection reduces FER by ≥10× vs non-CRC at L=8
- [ ] AC-2.2.2: CRC verification matches `crc16_ccitt()` for all decoded paths

- [ ] AC-2.3.1: Decode throughput ≥ 5 Mbps/core on AMD EPYC for N=4096, L=8
- [ ] AC-2.3.2: Memory usage ≤ 50 MB for N=4096, L=32

- [ ] AC-2.4.1: PyO3 bindings release GIL during decode
- [ ] AC-2.4.2: `PyPolarCodec.decode_soft()` returns (soft_llr, message, metric) tuple
```

---

## Implementation Sequence

```
Phase 2.0: Bit-Reversal Alignment (Prerequisite)
├── 2.0.1: Add `bit_reverse_index(idx, n_bits)` utility to `construction.rs`
├── 2.0.2: Modify `PolarEncoder.encode()` to permute message indices
├── 2.0.3: Verify `test_sc_decode_noiseless` passes with exact match
└── 2.0.4: Add TV-BRP-01 test vector

Phase 2.1: SCL Core Algorithm
├── 2.1.1: Extend `SCDecoder` to `SCLDecoder` with `list_size` parameter
├── 2.1.2: Implement path splitting at information bits
├── 2.1.3: Implement LLR-domain path metric updates
├── 2.1.4: Implement path pruning (keep L best)
└── 2.1.5: Validate TV-SCL-01, TV-SCL-02, TV-SCL-03

Phase 2.2: CRC Integration
├── 2.2.1: Integrate `crc.rs` with encoder (append CRC to message)
├── 2.2.2: Implement CRC-aided path selection in decoder
├── 2.2.3: Handle "no CRC match" fallback (return best metric path)
└── 2.2.4: Validate TV-CASCL-01, TV-CASCL-02

Phase 2.3: PyO3 Bindings
├── 2.3.1: Create `PyPolarCodec` pyclass
├── 2.3.2: Implement GIL release wrapper for `decode_soft()`
├── 2.3.3: Add numpy buffer protocol for soft output
└── 2.3.4: Integration test from Python

Phase 2.4: Documentation
├── 2.4.1: Create `specs/rust-polar-crate.md`
└── 2.4.2: Create `impl/phase2-scl-decoder.md`
```

---

## Dependencies & Blockers

| Dependency | Status | Resolution Path |
|------------|--------|-----------------|
| ADR-0001 (Polar Codec Adoption) | ✅ Accepted | — |
| ADR-0002 (Rust Extension) | ✅ Accepted | — |
| Phase 1 (Rust Foundation) | ✅ Complete* | *Requires bit-reversal fix |
| `specs/siso-codec-protocol.md` | ✅ Created | — |
| Literature: Tal & Vardy | ✅ Available | `docs/literature/List_Decoding_of_Polar_Codes.md` |
| Literature: Balatsoukas-Stimming | ✅ Available | `docs/literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md` |

---

## Design Decisions (Confirmed 2026-02-02)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Bit-Reversal Fix Location | **Option A: `encode()`** | Single modification point; decoder unchanged |
| 2 | Lazy Copy Complexity | **Simple deep-copy** | Correctness first; lazy-copy deferred to post-launch |
| 3 | SIMD Scope | **Defer to Phase 2.5** | SCL correctness first, then throughput optimization |
| 4 | Soft Output Format | **Per-bit LLRs from winning path** | Standard CA-SCL convention |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-02 | Context Engineer | Initial plan incorporating architect's bit-reversal analysis |
| 1.1.0 | 2026-02-02 | Context Engineer | Confirmed design decisions; plan ready for execution |

# Rust Polar Crate API Specification

<metadata>
spec_id: rust-polar-crate
version: 1.0.0
status: ready
created: 2026-02-02
depends_on: [ADR-0001, ADR-0002, siso-codec-protocol, phase1-rust-foundation]
enables: [phase2-scl-decoder, phase3-strategy-refactor]
</metadata>

---

## §1 Overview & Scope

<overview>

This specification defines the public API of the `caligo-codecs` Rust crate for Phase 2 (SCL Decoder). The crate provides:

1. **Polar code configuration** — Block length, frozen mask, CRC polynomial
2. **SCL decoder** — Successive Cancellation List decoding with CRC-aided selection
3. **PyO3 bindings** — Python interface with GIL release for parallelism
4. **SISO interface** — Soft-input soft-output for iterative reconciliation

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Zero-copy where possible** | NumPy buffer protocol for LLR arrays |
| **GIL release** | `py.allow_threads()` during decode computation |
| **Memory predictability** | Pre-allocated path arrays; no runtime allocation in hot path |
| **Error transparency** | Rust errors map to Python exceptions with context |

### Scope Boundaries

| In Scope | Out of Scope |
|----------|--------------|
| SCL decoding (L=1,2,4,8,16,32) | Belief propagation decoding |
| CRC-16-CCITT integration | Arbitrary CRC polynomials |
| LLR-domain computation | Probability-domain computation |
| Single-threaded decode | Multi-threaded batch decode (Phase 3) |

</overview>

---

## §2 Type System

### §2.1 Core Rust Types

<interface language="rust">

```rust
//! Core types for Polar code representation.
//!
//! Reference: [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) L24-52

use bitvec::prelude::*;

/// Polar code configuration.
///
/// Immutable after construction. Thread-safe (implements `Send + Sync`).
#[derive(Debug, Clone)]
pub struct PolarCode {
    /// Block length N = 2^n (must be power of 2)
    pub block_length: usize,
    
    /// Number of stages n = log2(N)
    pub n_stages: usize,
    
    /// Information bit length K (excluding CRC)
    pub message_length: usize,
    
    /// Total information positions K_total = K + CRC_bits
    pub k_total: usize,
    
    /// Frozen bit mask: true = frozen position, false = information
    /// Length = N, indexed in bit-reversed order for decoder compatibility
    pub frozen_mask: BitVec<u64, Lsb0>,
    
    /// CRC polynomial (None = no CRC)
    /// Default: CRC-16-CCITT (0x1021)
    pub crc_poly: Option<u16>,
    
    /// CRC length in bits (0 if no CRC)
    pub crc_bits: usize,
}

/// SCL decoder state.
///
/// Mutable during decode; single-threaded access required.
/// Memory is pre-allocated at construction for predictable performance.
pub struct SCLDecoder {
    /// Polar code configuration (shared reference)
    code: PolarCode,
    
    /// List size L (1, 2, 4, 8, 16, or 32)
    list_size: usize,
    
    /// LLR memory for each path: shape [L][n_stages+1][N]
    /// Stores log-likelihood ratios at each layer of the decoding tree
    llr_memory: Vec<Vec<Vec<f32>>>,
    
    /// Partial sum memory for each path: shape [L][n_stages+1][N]
    /// Stores decoded bit estimates propagated through butterfly
    bit_memory: Vec<Vec<Vec<u8>>>,
    
    /// Path metrics: shape [L]
    /// Lower metric = more likely path
    path_metrics: Vec<f32>,
    
    /// Active path bitmap: shape [L]
    /// true = path is active and being tracked
    active_paths: BitVec<u64, Lsb0>,
    
    /// Number of currently active paths (≤ L)
    num_active: usize,
}

/// Decode result with soft information for SISO interface.
///
/// Reference: [siso-codec-protocol.md](siso-codec-protocol.md)
#[derive(Debug, Clone)]
pub struct SCLDecodeResult {
    /// Decoded message bits (length = K, excluding CRC)
    pub message: Vec<u8>,
    
    /// Soft output LLRs for SISO interface (length = N)
    /// Per-bit LLRs from the winning path
    /// Convention: positive = bit 0 more likely
    pub soft_output: Vec<f32>,
    
    /// Path metric of selected path (lower = better)
    pub path_metric: f32,
    
    /// CRC validation result (None if no CRC configured)
    pub crc_valid: Option<bool>,
    
    /// Number of paths explored during decode
    pub paths_explored: usize,
    
    /// Whether decode completed successfully
    pub converged: bool,
}
```

</interface>

### §2.2 PyO3 Exports

<interface language="rust">

```rust
//! PyO3 bindings for Python interoperability.
//!
//! Reference: [codec.py](../../../caligo/caligo/reconciliation/strategies/codec.py) L36-70

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Python-visible Polar codec combining encoder and decoder.
///
/// Thread-safe for concurrent Python calls (GIL released during decode).
#[pyclass(name = "PolarCodec")]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCLDecoder,
}

#[pymethods]
impl PyPolarCodec {
    /// Construct a new Polar codec.
    ///
    /// # Arguments
    /// * `block_length` - N = 2^n (must be power of 2, 8 ≤ N ≤ 32768)
    /// * `message_length` - K (number of information bits, excluding CRC)
    /// * `list_size` - L (1, 2, 4, 8, 16, or 32; default 8)
    /// * `crc_bits` - CRC length (0 or 16; default 16)
    /// * `design_snr_db` - Design SNR for frozen set construction (default 2.0)
    ///
    /// # Raises
    /// * `ValueError` - Invalid parameters (non-power-of-2, K > N, etc.)
    ///
    /// # Example
    /// ```python
    /// codec = PolarCodec(
    ///     block_length=4096,
    ///     message_length=2032,
    ///     list_size=8,
    ///     crc_bits=16,
    ///     design_snr_db=2.0
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (block_length, message_length, list_size=8, crc_bits=16, design_snr_db=2.0))]
    fn new(
        block_length: usize,
        message_length: usize,
        list_size: usize,
        crc_bits: usize,
        design_snr_db: f64,
    ) -> PyResult<Self>;

    /// Encode message bits to codeword.
    ///
    /// CRC is automatically appended before encoding if configured.
    ///
    /// # Arguments
    /// * `message` - Information bits (length = K, dtype=uint8, values 0 or 1)
    ///
    /// # Returns
    /// Codeword bits (length = N, dtype=uint8)
    ///
    /// # Raises
    /// * `ValueError` - Wrong message length or non-binary values
    fn encode<'py>(
        &self,
        py: Python<'py>,
        message: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>>;

    /// Decode from soft LLR input.
    ///
    /// **GIL is released** during the computation for parallelism.
    ///
    /// # Arguments
    /// * `llr` - Channel LLRs (length = N, dtype=float32)
    ///   Convention: positive LLR = bit 0 more likely
    ///
    /// # Returns
    /// Tuple of:
    /// * `soft_output` - Extrinsic LLRs (length = N, dtype=float32)
    /// * `message` - Decoded message bits (length = K, dtype=uint8)
    /// * `path_metric` - Best path metric (float32)
    /// * `crc_valid` - CRC check result (bool or None)
    ///
    /// # Raises
    /// * `ValueError` - Wrong LLR length
    fn decode_soft<'py>(
        &self,
        py: Python<'py>,
        llr: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,  // soft_output
        Bound<'py, PyArray1<u8>>,   // message
        f32,                         // path_metric
        Option<bool>,                // crc_valid
    )>;

    /// Get frozen bit mask as numpy array.
    ///
    /// # Returns
    /// Frozen mask (length = N, dtype=uint8, 1=frozen, 0=information)
    fn frozen_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>>;

    /// Get code parameters.
    #[getter]
    fn block_length(&self) -> usize;
    
    #[getter]
    fn message_length(&self) -> usize;
    
    #[getter]
    fn list_size(&self) -> usize;
    
    #[getter]
    fn crc_bits(&self) -> usize;
    
    #[getter]
    fn rate(&self) -> f64;
}
```

</interface>

### §2.3 Error Types

<interface language="rust">

```rust
//! Error taxonomy for Polar codec operations.
//!
//! Reference: [decoder.rs](../../../caligo/_native/src/polar/decoder.rs) L14-22
//! Reference: [encoder.rs](../../../caligo/_native/src/polar/encoder.rs) L12-21

use thiserror::Error;

/// Errors during Polar code construction.
#[derive(Error, Debug)]
pub enum ConstructionError {
    #[error("Block length {0} is not a power of 2")]
    InvalidBlockLength(usize),
    
    #[error("Block length {0} out of range [8, 32768]")]
    BlockLengthOutOfRange(usize),
    
    #[error("Message length {0} exceeds block length {1}")]
    MessageTooLong(usize, usize),
    
    #[error("List size {0} must be power of 2 in range [1, 32]")]
    InvalidListSize(usize),
    
    #[error("CRC bits {0} must be 0 or 16")]
    InvalidCrcBits(usize),
    
    #[error("Message length {0} + CRC {1} exceeds block length {2}")]
    TotalInfoTooLong(usize, usize, usize),
}

/// Errors during encoding.
#[derive(Error, Debug)]
pub enum EncoderError {
    #[error("Message length {0} does not match expected {1}")]
    MessageLengthMismatch(usize, usize),
    
    #[error("Message contains non-binary value {0} at index {1}")]
    NonBinaryValue(u8, usize),
}

/// Errors during decoding.
#[derive(Error, Debug)]
pub enum DecoderError {
    #[error("LLR length {0} does not match block length {1}")]
    LlrLengthMismatch(usize, usize),
    
    #[error("No active paths remain (internal error)")]
    NoActivePaths,
    
    #[error("Path index {0} out of range [0, {1})")]
    PathIndexOutOfRange(usize, usize),
}

/// Unified codec error (for PyO3 conversion).
#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Construction error: {0}")]
    Construction(#[from] ConstructionError),
    
    #[error("Encoder error: {0}")]
    Encoder(#[from] EncoderError),
    
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError),
}

// PyO3 error conversion
impl From<CodecError> for PyErr {
    fn from(err: CodecError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
```

</interface>

---

## §3 Memory Layout

### §3.1 LLR Storage (SCL)

<invariants>

**Memory Structure:**

For SCL decoding with list size $L$, block length $N = 2^n$:

```
llr_memory[l][λ][i]  where:
  l ∈ [0, L)       — path index
  λ ∈ [0, n]       — layer (0 = channel, n = decision)
  i ∈ [0, N)       — position within layer
```

**Total LLR memory:** $L \times (n+1) \times N \times 4$ bytes (f32)

For N=4096, L=8: $8 \times 13 \times 4096 \times 4 = 1.7$ MB

**Allocation Strategy:**

```rust
// Pre-allocate at construction (simple deep-copy approach)
let llr_memory: Vec<Vec<Vec<f32>>> = (0..list_size)
    .map(|_| {
        (0..=n_stages)
            .map(|_| vec![0.0f32; block_length])
            .collect()
    })
    .collect();
```

**Invariant:** Once allocated, no reallocation occurs during decode.

</invariants>

**Literature Reference:** [List_Decoding_of_Polar_Codes.md](../../literature/List_Decoding_of_Polar_Codes.md) §IV — $P_\lambda[\phi, \beta]$ probabilities array structure.

### §3.2 Path Metrics

<invariants>

**Path Metric Convention:**

Per [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) §III-A, Eq. (15):

$$PM^{(l)} \leftarrow PM^{(l)} + \ln(1 + e^{-|L_i^{(l)}|})$$

when $\hat{u}_i^{(l)} \neq \frac{1 - \text{sign}(L_i^{(l)})}{2}$ (decision against LLR sign).

**Approximation (§III-B):** For $|L| > T$ (threshold ≈ 10):

$$\ln(1 + e^{-|L|}) \approx 0$$

**Implementation:**

```rust
/// Update path metric for a decision.
///
/// # Arguments
/// * `llr` - LLR value at decision point
/// * `decision` - Decoded bit (0 or 1)
///
/// # Returns
/// Metric increment (always ≥ 0)
#[inline]
fn path_metric_increment(llr: f32, decision: u8) -> f32 {
    const THRESHOLD: f32 = 10.0;
    
    let llr_sign_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
    
    if decision == llr_sign_bit {
        // Decision matches LLR sign: no penalty
        0.0
    } else {
        // Decision against LLR sign: penalize
        let abs_llr = llr.abs();
        if abs_llr > THRESHOLD {
            abs_llr  // Approximation: ln(1 + e^-x) ≈ 0 for large x
        } else {
            (1.0 + (-abs_llr).exp()).ln()
        }
    }
}
```

**Invariant:** Path metrics are monotonically non-decreasing during decode.

**Invariant:** Lower path metric = higher likelihood path.

</invariants>

### §3.3 Partial Sum Trees

<invariants>

**Partial Sum Structure:**

```
bit_memory[l][λ][i]  where:
  l ∈ [0, L)       — path index
  λ ∈ [0, n]       — layer
  i ∈ [0, N)       — position
```

**Propagation Rules:**

At layer $\lambda$, position $\phi$, after decoding bits $\hat{u}_{2\psi}$ and $\hat{u}_{2\psi+1}$:

```rust
// XOR propagation (butterfly)
bit_memory[l][λ-1][ψ] = bit_memory[l][λ][2*ψ] ^ bit_memory[l][λ][2*ψ + 1];
bit_memory[l][λ-1][ψ + half] = bit_memory[l][λ][2*ψ + 1];
```

where $\psi = \lfloor \phi / 2 \rfloor$ and $\text{half} = 2^{\lambda-1}$.

**Invariant:** Partial sums are only updated after completing both bits of a pair.

</invariants>

---

## §4 API Surface

### §4.1 PolarCode Configuration

<interface language="rust">

```rust
impl PolarCode {
    /// Construct Polar code with Gaussian Approximation frozen set.
    ///
    /// # Arguments
    /// * `block_length` - N = 2^n
    /// * `message_length` - K (excluding CRC)
    /// * `crc_bits` - 0 or 16
    /// * `design_snr_db` - Design point for frozen set
    ///
    /// # Panics
    /// Panics if parameters are invalid (use `try_new` for fallible construction).
    pub fn new(
        block_length: usize,
        message_length: usize,
        crc_bits: usize,
        design_snr_db: f64,
    ) -> Self;
    
    /// Fallible construction with detailed errors.
    pub fn try_new(
        block_length: usize,
        message_length: usize,
        crc_bits: usize,
        design_snr_db: f64,
    ) -> Result<Self, ConstructionError>;
    
    /// Check if position is frozen.
    #[inline]
    pub fn is_frozen(&self, position: usize) -> bool {
        self.frozen_mask[position]
    }
    
    /// Get code rate K/N (excluding CRC from numerator).
    pub fn rate(&self) -> f64 {
        self.message_length as f64 / self.block_length as f64
    }
    
    /// Get effective rate (K + CRC) / N.
    pub fn effective_rate(&self) -> f64 {
        self.k_total as f64 / self.block_length as f64
    }
}
```

</interface>

### §4.2 SCLDecoder Interface

<interface language="rust">

```rust
impl SCLDecoder {
    /// Construct SCL decoder with given list size.
    ///
    /// # Arguments
    /// * `code` - Polar code configuration
    /// * `list_size` - L (must be power of 2 in [1, 32])
    pub fn new(code: PolarCode, list_size: usize) -> Result<Self, ConstructionError>;
    
    /// Decode from channel LLRs.
    ///
    /// # Arguments
    /// * `llr_channel` - Channel LLRs (length = N)
    ///   Convention: positive = bit 0 more likely
    ///
    /// # Returns
    /// Decode result with message, soft output, and metrics.
    pub fn decode(&mut self, llr_channel: &[f32]) -> Result<SCLDecodeResult, DecoderError>;
    
    /// Reset decoder state for next decode.
    ///
    /// Called automatically at start of `decode()`.
    pub fn reset(&mut self);
    
    /// Get current list size.
    pub fn list_size(&self) -> usize;
    
    /// Get reference to code configuration.
    pub fn code(&self) -> &PolarCode;
}
```

</interface>

**Core Algorithm (simplified):**

```rust
pub fn decode(&mut self, llr_channel: &[f32]) -> Result<SCLDecodeResult, DecoderError> {
    self.reset();
    
    // Initialize path 0 with channel LLRs
    self.llr_memory[0][0].copy_from_slice(llr_channel);
    self.active_paths.set(0, true);
    self.num_active = 1;
    
    // Decode each bit position
    for phi in 0..self.code.block_length {
        // Compute LLRs at decision layer for all active paths
        self.compute_llrs_for_position(phi);
        
        if self.code.is_frozen(phi) {
            // Frozen bit: all paths decide 0
            self.process_frozen_bit(phi);
        } else {
            // Information bit: split paths (decide both 0 and 1)
            self.process_info_bit(phi);
            
            // Prune to L best paths if needed
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
```

### §4.3 CRC Integration

<interface language="rust">

```rust
//! CRC-16-CCITT integration for CA-SCL decoding.
//!
//! Reference: [crc.rs](../../../caligo/_native/src/polar/crc.rs) L1-80
//! Reference: [CRC-Aided_Decoding_of_Polar_Codes.md](../../literature/CRC-Aided_Decoding_of_Polar_Codes.md) §III

/// CRC-16-CCITT polynomial: x^16 + x^12 + x^5 + 1
pub const CRC16_CCITT_POLY: u16 = 0x1021;

/// CRC-16-CCITT initial value
pub const CRC16_INIT: u16 = 0xFFFF;

impl PolarCode {
    /// Append CRC to message before encoding.
    ///
    /// # Arguments
    /// * `message` - Information bits (length = K)
    ///
    /// # Returns
    /// Message with CRC appended (length = K + crc_bits)
    pub fn append_crc(&self, message: &[u8]) -> Vec<u8>;
    
    /// Verify CRC of decoded message.
    ///
    /// # Arguments
    /// * `message_with_crc` - Decoded bits (length = K + crc_bits)
    ///
    /// # Returns
    /// true if CRC matches, false otherwise
    pub fn verify_crc(&self, message_with_crc: &[u8]) -> bool;
    
    /// Strip CRC from decoded message.
    ///
    /// # Arguments
    /// * `message_with_crc` - Message with CRC (length = K + crc_bits)
    ///
    /// # Returns
    /// Message without CRC (length = K)
    pub fn strip_crc(&self, message_with_crc: &[u8]) -> Vec<u8>;
}

impl SCLDecoder {
    /// Select best path using CRC-aided selection.
    ///
    /// Algorithm (CA-SCL):
    /// 1. For each active path, extract message bits
    /// 2. Verify CRC for each candidate
    /// 3. If any path passes CRC, return best (lowest metric) passing path
    /// 4. If no path passes CRC, return path with lowest metric
    ///
    /// Reference: [CRC-Aided_Decoding_of_Polar_Codes.md](../../literature/CRC-Aided_Decoding_of_Polar_Codes.md) §III
    fn select_best_path(&self) -> Result<SCLDecodeResult, DecoderError> {
        let mut best_crc_path: Option<(usize, f32)> = None;
        let mut best_any_path: Option<(usize, f32)> = None;
        
        for l in 0..self.list_size {
            if !self.active_paths[l] {
                continue;
            }
            
            let metric = self.path_metrics[l];
            
            // Track best overall path
            if best_any_path.is_none() || metric < best_any_path.unwrap().1 {
                best_any_path = Some((l, metric));
            }
            
            // Check CRC if configured
            if self.code.crc_bits > 0 {
                let message = self.extract_message(l);
                if self.code.verify_crc(&message) {
                    if best_crc_path.is_none() || metric < best_crc_path.unwrap().1 {
                        best_crc_path = Some((l, metric));
                    }
                }
            }
        }
        
        // Prefer CRC-passing path, fallback to best metric
        let (selected_path, crc_valid) = if let Some((l, _)) = best_crc_path {
            (l, Some(true))
        } else if let Some((l, _)) = best_any_path {
            (l, if self.code.crc_bits > 0 { Some(false) } else { None })
        } else {
            return Err(DecoderError::NoActivePaths);
        };
        
        self.build_result(selected_path, crc_valid)
    }
}
```

</interface>

---

## §5 GIL & Parallelism

<invariants>

**GIL Release Pattern:**

```rust
fn decode_soft<'py>(
    &self,
    py: Python<'py>,
    llr: PyReadonlyArray1<'py, f32>,
) -> PyResult<(...)> {
    // Extract data while holding GIL
    let llr_slice = llr.as_slice()?;
    
    // Release GIL for computation
    let result = py.allow_threads(|| {
        // Clone decoder for thread-safety (or use interior mutability)
        let mut decoder = self.decoder.clone();
        decoder.decode(llr_slice)
    })?;
    
    // Re-acquire GIL for result conversion
    let soft_output = PyArray1::from_vec_bound(py, result.soft_output);
    let message = PyArray1::from_vec_bound(py, result.message);
    
    Ok((soft_output, message, result.path_metric, result.crc_valid))
}
```

**Thread Safety Guarantees:**

| Component | Thread Safety | Notes |
|-----------|---------------|-------|
| `PolarCode` | `Send + Sync` | Immutable after construction |
| `SCLDecoder` | `Send` only | Mutable state; clone for parallel use |
| `PyPolarCodec` | GIL-protected | Python object; GIL released during decode |

**Memory Ordering:**

- No atomic operations required (single-threaded decode)
- `clone()` performs deep copy of all path state
- Future optimization: `Arc<RwLock<>>` for shared code, per-thread decoder pools

</invariants>

---

## §6 Test Vector Format

<test_vector_spec>

Test vectors are stored as JSON files in `tests/vectors/`:

```json
{
  "metadata": {
    "id": "TV-SCL-02",
    "description": "SCL L=8 decode at SNR=2dB",
    "created": "2026-02-02",
    "source": "Tal & Vardy"
  },
  "parameters": {
    "block_length": 1024,
    "message_length": 512,
    "list_size": 8,
    "crc_bits": 0,
    "design_snr_db": 2.0,
    "noise_snr_db": 2.0
  },
  "test_cases": [
    {
      "message": [1, 0, 1, 1, ...],
      "codeword": [0, 1, 0, 0, ...],
      "channel_llr": [2.1, -1.5, 0.8, ...],
      "expected_decoded": [1, 0, 1, 1, ...],
      "expected_crc_valid": null,
      "max_path_metric": 15.0
    }
  ],
  "statistical": {
    "num_trials": 10000,
    "target_fer": 0.01,
    "tolerance": 0.002
  }
}
```

**Validation Criteria:**

| Test Type | Pass Condition |
|-----------|----------------|
| Deterministic | `decoded == expected_decoded` |
| CRC | `crc_valid == expected_crc_valid` |
| Metric | `path_metric <= max_path_metric` |
| Statistical | `measured_fer <= target_fer + tolerance` |

</test_vector_spec>

---

## Appendix: Bit-Reversal Permutation

<invariants>

**Problem Statement:**

The Polar encoder and decoder use different index conventions:

| Component | Convention | Generator Matrix |
|-----------|------------|------------------|
| Standard textbook | Natural order | $\mathbf{G}_N = \mathbf{F}^{\otimes n}$ |
| SC/SCL decoder | Bit-reversed | $\mathbf{G}_N = \mathbf{B}_N \cdot \mathbf{F}^{\otimes n}$ |

**Reference:** [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) Eq. (1)

**Resolution (Decision: Option A):**

Apply bit-reversal permutation to message indices in encoder:

```rust
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
/// For n_bits=3: 0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7
#[inline]
pub fn bit_reverse_index(index: usize, n_bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - n_bits)
}

// In PolarEncoder::encode():
fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
    let mut u = vec![0u8; self.block_length];
    let mut msg_idx = 0;
    
    for i in 0..self.block_length {
        // Apply bit-reversal when placing message bits
        let br_i = bit_reverse_index(i, self.n_stages);
        
        if !self.frozen_mask[i] {  // frozen_mask is in bit-reversed order
            u[br_i] = message[msg_idx];
            msg_idx += 1;
        }
    }
    
    self.butterfly_transform(&mut u);
    Ok(Array1::from_vec(u))
}
```

**Invariant:** After this fix, `decode(encode(m)) == m` for all valid messages.

**Test Vector:** TV-BRP-01 validates exact round-trip for N=8, K=4.

</invariants>

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-02 | Context Engineer | Initial specification from confirmed plan |

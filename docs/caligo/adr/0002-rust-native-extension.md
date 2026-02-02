# [ADR-0002] Rust Native Extension for Polar Codec

<metadata>
adr_id: 0002
status: accepted
superseded_by: null
date_created: 2026-02-02
date_decided: 2026-02-02
deciders: [User, Context Engineer Agent]
</metadata>

## Status

**Accepted**

---

## Context

<context>
The decision to adopt Polar codes (ADR-0001) introduces a new implementation requirement: a high-performance SCL decoder capable of exploiting the AMD Genoa HPC target (192 cores, 2GB/core, AVX-512). This ADR addresses the implementation technology selection.

### Current Numba Architecture Limitations

The existing LDPC reconciliation kernels ([numba_kernels.py](../../../caligo/caligo/scripts/numba_kernels.py)) use Numba's `@njit(nogil=True)` pattern:

```python
@njit(cache=True, fastmath=True)
def decode_bp_virtual_graph_kernel(
    llr: np.ndarray,
    syndrome: np.ndarray,
    messages: np.ndarray,
    # ... topology arrays ...
    max_iterations: int,
) -> Tuple[np.ndarray, bool, int]:
```

While this releases the GIL during kernel execution, empirical analysis reveals systematic limitations:

| Issue | Impact | Evidence |
|-------|--------|----------|
| **Memory fragmentation** | OOM after ~10K iterations | Long-running exploration jobs show gradual RSS growth despite Numba's `nogil` |
| **Limited SIMD control** | ~50% AVX-512 utilization | Numba's auto-vectorization misses opportunities in min-sum BP inner loops |
| **JIT warmup penalty** | 2-5s per worker | Each ProcessPoolExecutor worker recompiles; 192 workers = cumulative overhead |
| **Debug opacity** | High dev cost | Numba errors are cryptic; no stack traces in `nopython` mode |

### HPC Platform Constraints

Per [hpc_migration_roadmap.md](../archive/hpc_migration_roadmap.md) §1.2:

| Resource | Constraint | Implication |
|----------|------------|-------------|
| **CPU** | 192 cores @ 2.4GHz | Must achieve near-linear scaling via GIL release |
| **Memory** | 2 GiB/core | SCL(L=32, N=4096) must fit in <100MB working memory |
| **SIMD** | AVX-512 | Critical for LLR updates (f/g functions, path metrics) |
| **GPU** | None | All compute on CPU; no CUDA fallback |

### Throughput Requirements

Per ADR-0001 §Consequences, the target is >10 Mbps/core for SCL decoding. Reference benchmarks from [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) §VIII indicate:

- FPGA at 100 MHz: 910 Mbit/s for Fast-SSC (N=32768, R=0.9)
- Extrapolated CPU (AVX-512, 2.4GHz): ~50-100 Mbps/core achievable with optimized implementation

### Rust Ecosystem Maturity

The Rust ecosystem now provides mature tools for this workload:

| Component | Crate | Maturity |
|-----------|-------|----------|
| Python bindings | `pyo3` 0.21+ | Production (used by Polars, cryptography.io) |
| Build system | `maturin` 1.4+ | Production (PyPI standard) |
| SIMD intrinsics | `std::simd` (nightly) or `packed_simd2` | Stable via feature flags |
| Bit manipulation | `bitvec` 1.0+ | Production |
| ndarray interop | `ndarray` + `numpy` | Production |

</context>

---

## Decision

<decision>
We adopt **Rust with PyO3 bindings** (via Maturin) as the implementation technology for the Polar codec core. The crate will be named `caligo-codecs` and reside at `caligo/_native/`.

### Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIGO-CODECS CRATE STRUCTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  caligo/_native/                                                            │
│  ├── Cargo.toml                    # Crate manifest                         │
│  ├── pyproject.toml                # Maturin build config                   │
│  └── src/                                                                   │
│      ├── lib.rs                    # PyO3 module entry (#[pymodule])        │
│      ├── error.rs                  # Error types (PyErr conversion)         │
│      └── polar/                                                             │
│          ├── mod.rs                # Module exports                         │
│          ├── encoder.rs            # Polar encoder (Arikan recursive)       │
│          ├── decoder.rs            # SC/SCL decoder (LLR-domain)            │
│          ├── construction.rs       # Frozen bit selection (GA)              │
│          ├── crc.rs                # CRC-16-CCITT (stub → Phase 2)          │
│          └── simd.rs               # AVX-512 LLR kernels                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Dependencies

```toml
# Cargo.toml [dependencies]
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"                    # PyO3 numpy interop
ndarray = "0.15"                  # N-dimensional arrays
bitvec = "1.0"                    # Bit-level operations
thiserror = "1.0"                 # Error handling

[features]
default = []
simd = []                         # Enable AVX-512 via std::simd (nightly)

[profile.release]
lto = "fat"                       # Link-time optimization
codegen-units = 1                 # Single codegen unit for better inlining
opt-level = 3
```

### PyO3 Export Surface

```rust
/// Python-visible Polar codec implementing SISOCodec protocol.
#[pyclass(name = "PolarCodec")]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCLDecoder,
    frozen_mask: BitVec,
    crc_poly: Option<u16>,
}

#[pymethods]
impl PyPolarCodec {
    #[new]
    #[pyo3(signature = (block_length, message_length, *, list_size=8, crc_poly=None))]
    fn new(
        block_length: usize,
        message_length: usize,
        list_size: usize,
        crc_poly: Option<u16>,
    ) -> PyResult<Self>;
    
    /// Encode message to codeword. GIL released during computation.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        message: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>>;
    
    /// Hard-decision SC/SCL decoding. GIL released.
    fn decode_hard<'py>(
        &self,
        py: Python<'py>,
        received: PyReadonlyArray1<'py, u8>,
        syndrome: Option<PyReadonlyArray1<'py, u8>>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, i32)>;
    
    /// Soft-decision SCL decoding with extrinsic LLR output. GIL released.
    fn decode_soft<'py>(
        &self,
        py: Python<'py>,
        llr_channel: PyReadonlyArray1<'py, f32>,
        llr_prior: Option<PyReadonlyArray1<'py, f32>>,
        list_size: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u8>>, bool, i32, f32)>;
    
    // Properties matching SISOCodec protocol
    #[getter]
    fn block_length(&self) -> usize;
    #[getter]
    fn message_length(&self) -> usize;
    #[getter]
    fn rate(&self) -> f64;
    #[getter]
    fn crc_length(&self) -> usize;
}
```

### GIL Release Pattern

All compute-intensive methods use `Python::allow_threads()`:

```rust
fn decode_soft<'py>(&self, py: Python<'py>, llr: PyReadonlyArray1<'py, f32>, ...) 
    -> PyResult<...> 
{
    let llr_slice = llr.as_slice()?;
    
    // Release GIL for the heavy computation
    let (extrinsic, message, converged, iters, metric) = py.allow_threads(|| {
        self.decoder.decode_soft_inner(llr_slice, ...)
    });
    
    // Re-acquire GIL for array construction
    let extrinsic_arr = PyArray1::from_slice_bound(py, &extrinsic);
    let message_arr = PyArray1::from_slice_bound(py, &message);
    
    Ok((extrinsic_arr, message_arr, converged, iters, metric))
}
```

### Scope Boundaries

**This ADR covers:**
- Technology selection (Rust + PyO3)
- Crate structure and dependencies
- PyO3 API surface design
- GIL release strategy

**This ADR does NOT cover:**
- Detailed algorithm implementation (see [impl/phase1-rust-foundation.md](../impl/phase1-rust-foundation.md))
- Fast-SSC optimizations (Phase 2+)
- LDPC-Polar concatenation (future work)

</decision>

---

## Consequences

<consequences>

### Positive

1. **True Parallelism:** `py.allow_threads()` releases GIL, enabling 192-core scaling. Unlike Numba, Rust code runs entirely outside Python's runtime.

2. **Deterministic Memory:** Rust's ownership model eliminates memory fragmentation. Working set is bounded at compile time: SCL(L=32, N=4096) uses exactly `32 × 4096 × 4 bytes = 512KB` for path LLRs.

3. **AVX-512 Control:** Direct access to SIMD intrinsics via `std::simd` or `core::arch::x86_64`. Min-sum f-function can be vectorized explicitly:

   ```rust
   // 16 LLR pairs per AVX-512 operation
   #[cfg(target_feature = "avx512f")]
   fn f_function_avx512(alpha: &[f32; 16], beta: &[f32; 16]) -> [f32; 16] {
       // sign(α) × sign(β) × min(|α|, |β|)
       unsafe {
           let a = _mm512_loadu_ps(alpha.as_ptr());
           let b = _mm512_loadu_ps(beta.as_ptr());
           let sign_a = _mm512_and_ps(a, _mm512_set1_ps(-0.0));
           let sign_b = _mm512_and_ps(b, _mm512_set1_ps(-0.0));
           let abs_a = _mm512_andnot_ps(_mm512_set1_ps(-0.0), a);
           let abs_b = _mm512_andnot_ps(_mm512_set1_ps(-0.0), b);
           let min_ab = _mm512_min_ps(abs_a, abs_b);
           let sign = _mm512_xor_ps(sign_a, sign_b);
           let result = _mm512_or_ps(sign, min_ab);
           std::mem::transmute(result)
       }
   }
   ```

4. **Ecosystem Compatibility:** Maturin produces standard wheels; no runtime dependency on Rust toolchain for end users. CI builds via `maturin build --release`.

5. **Debuggability:** Full stack traces, `#[cfg(debug_assertions)]` for bounds checking, integration with standard profilers (perf, flamegraph).

### Negative

1. **Build Complexity:** Requires Rust toolchain (rustc 1.75+) for development. CI must cache Cargo dependencies.

2. **Development Overhead:** Estimated 3-4 days for Phase 1 (encoder + SC decoder); learning curve for PyO3 patterns.

3. **Cross-Platform Testing:** Must verify x86_64-linux (HPC target), x86_64-darwin (dev), aarch64-linux (CI). SIMD fallbacks needed for non-AVX512.

4. **Two-Language Debugging:** Errors may cross Python/Rust boundary; requires familiarity with both ecosystems.

### Neutral

1. **Maturin Integration:** Build configuration via `pyproject.toml` is standard; integrates with existing `caligo/pyproject.toml` as workspace member.

2. **Backward Compatibility:** `LDPCCodecAdapter` wraps existing Numba kernels; no migration required for LDPC users.

</consequences>

---

## Alternatives Considered

<alternatives>

### Alternative 1: Pure Numba (Status Quo)

**Description:** Continue using Numba `@njit(nogil=True)` kernels for Polar codec, mirroring the LDPC implementation.

**Rejected because:**
- Memory fragmentation persists across long runs (observed in LDPC exploration)
- Limited SIMD control; Numba's auto-vectorizer underperforms on min-sum inner loops
- JIT warmup on 192 workers creates cumulative overhead (~6-10 minutes wasted per exploration batch)
- Debugging `nopython` mode is extremely difficult

**Quantitative comparison:**
| Metric | Numba (estimated) | Rust (estimated) |
|--------|-------------------|------------------|
| Throughput/core | 5-10 Mbps | 20-50 Mbps |
| Memory stability | Degraded after ~10K iters | Constant |
| AVX-512 utilization | ~50% | ~90% (explicit intrinsics) |
| Warmup time | 2-5s/worker | 0 (AOT compiled) |

### Alternative 2: Cython

**Description:** Implement Polar codec in Cython with `nogil` annotations and explicit SIMD via `cython.parallel`.

**Rejected because:**
- Memory management remains manual (C-style); no ownership guarantees
- SIMD requires inline C/assembly; loses Cython's simplicity advantage
- Build system more fragile than Maturin (setuptools extensions are error-prone)
- Error messages less informative than Rust's borrow checker

### Alternative 3: C++ with pybind11

**Description:** Implement in C++17/20 with pybind11 Python bindings.

**Rejected because:**
- Memory safety requires discipline (no compiler enforcement)
- Build system complexity (CMake + pybind11 is more fragile than Cargo + Maturin)
- Less mature ecosystem for bit manipulation (no `bitvec` equivalent)
- Rust's `Result<T, E>` pattern maps cleanly to Python exceptions via PyO3

### Alternative 4: JAX/XLA

**Description:** Implement Polar codec as JAX custom primitives with XLA compilation.

**Rejected because:**
- XLA designed for GPU/TPU; CPU backend underperforms vs native code
- SCL's path management (dynamic branching, list pruning) maps poorly to XLA's static graph model
- Adds significant dependency (JAX ecosystem) for marginal benefit
- No advantage over Rust for CPU-only HPC target

</alternatives>

---

## References

<references>
- [1] ADR-0001: Polar Codec Adoption — Establishes Polar CA-SCL as primary codec
- [2] [hpc_migration_roadmap.md](../archive/hpc_migration_roadmap.md) — HPC platform constraints (192 cores, 2GB/core, AVX-512)
- [3] [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) — Throughput benchmarks (§VIII)
- [4] [numba_kernels.py](../../../caligo/caligo/scripts/numba_kernels.py) — Current Numba patterns (reference for comparison)
- [5] [siso-codec-protocol.md](../specs/siso-codec-protocol.md) — Interface contract to implement
- [6] PyO3 Documentation: https://pyo3.rs/
- [7] Maturin Documentation: https://www.maturin.rs/
</references>

---

## Implementation Notes

<implementation_notes>

### Migration Path

1. **Phase 1:** Create `caligo/_native/` crate with encoder + SC decoder (L=1). Verify against test vectors.
2. **Phase 2:** Extend to full SCL (L=1..32) with CRC-aided selection.
3. **Phase 3:** Create `PolarCodecAdapter` implementing `SISOCodec` protocol.
4. **Phase 4:** Integration with `ReconciliationStrategy` via configuration.

### Rollback Strategy

If Rust implementation proves problematic:
1. `SISOCodec` protocol allows runtime codec selection
2. Fall back to `LDPCCodecAdapter` wrapping existing Numba kernels
3. No code changes required outside configuration

### Verification Criteria

- [ ] `cargo test` passes all unit tests
- [ ] `maturin develop` installs Python module
- [ ] `pytest caligo/_native/tests/` passes integration tests
- [ ] GIL release verified via concurrent decode stress test
- [ ] Memory profile shows constant RSS over 10K iterations
- [ ] AVX-512 codegen verified via `RUSTFLAGS="-C target-feature=+avx512f" cargo build`

### Build Integration

Add to `caligo/pyproject.toml`:

```toml
[tool.maturin]
python-source = "caligo"
module-name = "caligo._native"
features = ["pyo3/extension-module"]
```

CI pipeline:
```yaml
- name: Build Rust extension
  run: |
    pip install maturin
    cd caligo && maturin build --release
    pip install target/wheels/*.whl
```

</implementation_notes>

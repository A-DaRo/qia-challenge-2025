# Polar Codec Refactor: Initialization Document

<metadata>
version: 1.1.0
status: active
created: 2026-01-30
updated: 2026-02-02
purpose: Reference mapping and context aggregation for multi-agent development
</metadata>

## Executive Summary

<overview>
This document serves as the **central reference hub** for the "HPC-Ready Polar Codec with LDPC Deprecation Refactor" initiative. It maps each planned document to its codebase dependencies and literature foundations, enabling AI agents to load precise context before generating content.

**Project Goal:** Replace the current LDPC-only reconciliation system with a hybrid architecture supporting Polar codes (SCL decoding), while maintaining backward compatibility and optimizing for the AMD Genoa HPC target (192 cores, no GPU, 2GB/core).

**Timeline:** < 1 week (no scope extensions)
</overview>

---

## Document Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT DEPENDENCY GRAPH                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐                                                        │
│  │   initialization.md │ ◄─── YOU ARE HERE                                      │
│  │   (Reference Hub)   │                                                        │
│  └──────────┬──────────┘                                                        │
│             │                                                                   │
│             ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           P0: FOUNDATION                                  │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────────────┐  │  │
│  │  │ ADR-0001            │───▶│ specs/siso-codec-protocol.md            │  │  │
│  │  │ Polar Codec Adoption│    │ (Requires ADR rationale)                │  │  │
│  │  └─────────────────────┘    └─────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│             │                                                                   │
│             ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           P1: RUST LAYER                                  │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────────────┐  │  │
│  │  │ ADR-0002            │───▶│ impl/phase1-rust-foundation.md          │  │  │
│  │  │ Rust Extension      │    │ (Requires ADR + SISO spec)              │  │  │
│  │  └─────────────────────┘    └─────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│             │                                                                   │
│             ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           P2: SCL DECODER                                 │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│  │  │ specs/rust-polar-crate.md ──▶ impl/phase2-scl-decoder.md            │ │  │
│  │  │ (Rust API details)           (SCL + CRC implementation)             │ │  │
│  │  └─────────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│             │                                                                   │
│             ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           P3: PYTHON REFACTOR                             │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────────────┐  │  │
│  │  │ ADR-0003            │───▶│ impl/phase3-strategy-refactor.md        │  │  │
│  │  │ Kiktenko Baseline   │    │ + impl/phase4-integration.md            │  │  │
│  │  └─────────────────────┘    │ + specs/reconciliation-strategy.md      │  │  │
│  │                             └─────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## P0: Foundation Documents

### ADR-0001: Polar Codec Adoption

<document_spec id="adr-0001">

**Purpose:** Justify the strategic decision to adopt Polar codes (SCL decoding) as the primary error correction mechanism, deprecating the LDPC-only architecture.

**Status:** ✅ **ACCEPTED** — [adr/0001-polar-codec-adoption.md](adr/0001-polar-codec-adoption.md)

#### Key Decisions (2026-02-02)

| Parameter | Decision | Rationale |
|-----------|----------|----------|
| Default list size L | **8** | Optimal throughput/performance for typical QBER |
| Maximum list size L | **32** | Configurable for extreme noise |
| CRC polynomial | **CRC-16-CCITT** | 16-bit integrity, 0.4% overhead |
| Block length N | **4096** (Phase 1/2) | Drop-in LDPC replacement; Rust core generic N=2^n |
| Backward compat | **Adapter pattern** | Clean SISOCodec interface; LDPCCodecAdapter wraps legacy |

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/reconciliation/strategies/__init__.py](../../caligo/caligo/reconciliation/strategies/__init__.py) | Current strategy protocol | `ReconciliationStrategy` ABC, `BlockResult` dataclass |
| [caligo/reconciliation/strategies/codec.py](../../caligo/caligo/reconciliation/strategies/codec.py) | Existing LDPC codec interface | `LDPCCodec` class (lines 36-70) |
| [caligo/reconciliation/constants.py](../../caligo/caligo/reconciliation/constants.py) | Frame sizes, rate parameters | `LDPC_FRAME_SIZE=4096`, `MOTHER_CODE_RATE=0.5` |
| [caligo/reconciliation/ldpc_decoder.py](../../caligo/caligo/reconciliation/ldpc_decoder.py) | BP decoder implementation | Numba kernel invocation patterns |

#### Literature References

| Reference | Contribution | Key Sections |
|-----------|--------------|--------------|
| [List_Decoding_of_Polar_Codes.md](../literature/List_Decoding_of_Polar_Codes.md) | SCL algorithm foundation | §I Introduction, §II Algorithm, §III Pruning |
| [CRC-Aided_Decoding_of_Polar_Codes.md](../literature/CRC-Aided_Decoding_of_Polar_Codes.md) | CA-SCL integrity mechanism | §III CRC-aided algorithms |
| [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) | Numerical stability (LLR domain) | §II-B LLR update equations |
| [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) | Fast-SSC optimization | §III Simplified nodes (Rate-0, Rate-1, Rep, SPC) |
| [Rate-Adaptive_Polar-Coding-Based_Reconciliation.md](../literature/Rate-Adaptive_Polar-Coding-Based_Reconciliation.md) | CV-QKD context | §I-II Polar codes in QKD reconciliation |

#### Decision Drivers

<decision_drivers>
1. **Finite-length performance:** LDPC exhibits error floor at FER < 10⁻⁴; Polar+CRC achieves 10⁻⁶
2. **Throughput:** SCL with Fast-SSC achieves 10+ Mbps/core on AMD EPYC
3. **Memory efficiency:** SCL(L=32, N=2²⁰) ≈ 128MB fits 2GB/core HPC constraint
4. **SISO composability:** Enables future IC-LDPC-Polar concatenation
</decision_drivers>

</document_spec>

---

### SPEC: SISO Codec Protocol

<document_spec id="siso-codec-protocol">

**Purpose:** Define the Soft-Input Soft-Output codec interface that abstracts over LDPC and Polar implementations, enabling strategy-agnostic reconciliation.

**Status:** ✅ **ACCEPTED** — [specs/siso-codec-protocol.md](specs/siso-codec-protocol.md)

#### Key Interface Methods

| Method | Purpose | Returns |
|--------|---------|--------|
| `encode()` | Message → Codeword | `NDArray[np.uint8]` |
| `decode_hard()` | Hard-decision decode | `SISOHardDecodeResult` |
| `decode_soft()` | Soft-decision with extrinsic LLRs | `SISODecodeResult` |
| `compute_syndrome()` | Syndrome for Alice | `NDArray[np.uint8]` |

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/reconciliation/strategies/codec.py](../../caligo/caligo/reconciliation/strategies/codec.py) | Current `LDPCCodec` to refactor | `encode()`, `decode_baseline()`, `decode_blind()` methods |
| [caligo/reconciliation/strategies/__init__.py](../../caligo/caligo/reconciliation/strategies/__init__.py) | `DecoderResult` dataclass | Lines 176-193 |
| [caligo/reconciliation/matrix_manager.py](../../caligo/caligo/reconciliation/matrix_manager.py) | Matrix topology patterns | `NumbaGraphTopology`, `MotherCodeManager` |

#### Literature References

| Reference | Contribution | Key Sections |
|-----------|--------------|--------------|
| [Concatenated_LDPC-Polar_Codes.md](../literature/Concatenated_LDPC-Polar_Codes.md) | SISO message passing for concatenation | §III Belief propagation interface |
| [RC-LDPC-Polar_Codes.md](../literature/RC-LDPC-Polar_Codes.md) | Rate-compatible SISO requirements | §2 Decoder interface requirements |

#### Interface Contract Preview

<interface_preview>

```python
@runtime_checkable
class SISOCodec(Protocol):
    """Soft-Input Soft-Output Codec for composable reconciliation."""
    
    @property
    def block_length(self) -> int: ...
    
    @property
    def rate(self) -> float: ...
    
    def encode(self, message: NDArray[np.uint8], *, frozen_values: NDArray[np.uint8] | None = None) -> NDArray[np.uint8]: ...
    
    def decode_hard(self, received: NDArray[np.uint8], *, syndrome: NDArray[np.uint8] | None = None) -> tuple[NDArray[np.uint8], bool]: ...
    
    def decode_soft(self, llr_channel: NDArray[np.float32], *, syndrome: NDArray[np.uint8] | None = None, list_size: int = 1) -> tuple[NDArray[np.float32], NDArray[np.uint8], float]: ...
```
</interface_preview>

</document_spec>

---

## P1: Rust Layer Documents

### ADR-0002: Rust Native Extension

<document_spec id="adr-0002">

**Purpose:** Justify the use of Rust + PyO3 for the Polar codec core, versus pure Python/Numba alternatives.

**Status:** ✅ **ACCEPTED** — [adr/0002-rust-native-extension.md](adr/0002-rust-native-extension.md)

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/scripts/numba_kernels.py](../../caligo/caligo/scripts/numba_kernels.py) | Current Numba approach (to compare) | `@njit(nogil=True)` kernels |
| [archive/hpc_migration_roadmap.md](archive/hpc_migration_roadmap.md) | HPC constraints | §1.2 Gap Analysis, §1.3 Target Architecture |

#### Literature References

| Reference | Contribution | Key Sections |
|-----------|--------------|--------------|
| [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) | Hardware implementation patterns | §V FPGA results (throughput benchmarks) |

#### Decision Drivers

<decision_drivers>
1. **GIL release:** PyO3 allows `Python::allow_threads()` for true parallelism across 192 cores
2. **Memory control:** Rust's ownership model prevents the memory fragmentation seen in Numba on long runs
3. **SIMD intrinsics:** Direct access to AVX-512 via `std::simd` or `packed_simd`
4. **Future-proofing:** IC-LDPC-Polar hybrid requires tight control over memory layout
</decision_drivers>

</document_spec>

---

### IMPL: Phase 1 - Rust Foundation

<document_spec id="phase1-rust-foundation">

**Purpose:** Step-by-step guide for creating the `caligo-codecs` Rust crate with Maturin, implementing basic Polar encoder and SC decoder (L=1).

**Status:** ✅ **READY** — [impl/phase1-rust-foundation.md](impl/phase1-rust-foundation.md)

#### Codebase References (Target Locations)

| Target Path | Description |
|-------------|-------------|
| `caligo/_native/Cargo.toml` | Rust crate manifest |
| `caligo/_native/pyproject.toml` | Maturin build configuration |
| `caligo/_native/src/lib.rs` | PyO3 module entry point |
| `caligo/_native/src/polar/encoder.rs` | Polar encoder (frozen-bit insertion) |
| `caligo/_native/src/polar/decoder.rs` | SC decoder (baseline, L=1) |
| `caligo/_native/src/polar/construction.rs` | Channel polarization, frozen-bit selection |

#### Literature References

| Reference | Contribution | Implementation Target |
|-----------|--------------|----------------------|
| [List_Decoding_of_Polar_Codes.md](../literature/List_Decoding_of_Polar_Codes.md) | Encoding algorithm | `encoder.rs`: Arikan's recursive construction |
| [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) | LLR update functions | `decoder.rs`: f-function (check), g-function (variable) |

#### Test Vectors Required

<test_vector_sources>
**Approach:** Synthetic generation via Python script (`generate_test_vectors.py`)

| Vector ID | N | K | Purpose |
|-----------|---|---|--------|
| enc_n8_k4 | 8 | 4 | Encoder validation (small) |
| enc_n1024_k512 | 1024 | 512 | Encoder validation (standard) |
| enc_n4096_k2048 | 4096 | 2048 | Encoder validation (target size) |
| dec_n8_k4_highsnr | 8 | 4 | SC decoder validation (noiseless) |
| dec_n1024_k512_highsnr | 1024 | 512 | SC decoder validation (high SNR) |
| dec_n4096_k2048_highsnr | 4096 | 2048 | SC decoder validation (target size) |
</test_vector_sources>

</document_spec>

---

## P2: SCL Decoder Documents

### SPEC: Rust Polar Crate API

<document_spec id="rust-polar-crate">

**Purpose:** Detailed specification of the Rust crate's public API, including memory layout, error handling, and GIL release points.

**Status:** To be created

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/reconciliation/strategies/codec.py](../../caligo/caligo/reconciliation/strategies/codec.py) | Python-side invocation patterns | Buffer management in `decode_baseline()` |

#### Literature References

| Reference | Contribution | Implementation Target |
|-----------|--------------|----------------------|
| [List_Decoding_of_Polar_Codes.md](../literature/List_Decoding_of_Polar_Codes.md) | SCL path management | Path metric data structures |
| [CRC-Aided_Decoding_of_Polar_Codes.md](../literature/CRC-Aided_Decoding_of_Polar_Codes.md) | CRC integration | CRC polynomial configuration |
| [Fast_Polar_Decoders_Algorithm_and_Implementation.md](../literature/Fast_Polar_Decoders_Algorithm_and_Implementation.md) | Node specialization | Rate-0, Rate-1, Rep, SPC node handlers |

#### API Surface Preview

<api_preview>

```rust
// Core types
pub struct PolarCode { /* frozen mask, CRC polynomial */ }
pub struct SCLDecoder { /* list size, path metrics */ }

// PyO3 exports
#[pyclass]
pub struct PyPolarCodec { /* wraps PolarCode + SCLDecoder */ }

#[pymethods]
impl PyPolarCodec {
    #[new]
    fn new(n: usize, k: usize, list_size: usize, crc_poly: Option<u32>) -> PyResult<Self>;
    
    fn encode(&self, message: PyReadonlyArray1<u8>) -> PyResult<Py<PyArray1<u8>>>;
    
    fn decode_soft(&self, py: Python, llr: PyReadonlyArray1<f32>) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u8>>, f32)>;
}
```
</api_preview>

</document_spec>

---

### IMPL: Phase 2 - SCL Decoder

<document_spec id="phase2-scl-decoder">

**Purpose:** Implementation guide for the full SCL decoder with CRC-aided selection, LLR-domain computation, and GIL release for parallelism.

**Status:** To be created

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| Phase 1 deliverables | Foundation to extend | `decoder.rs` SC baseline |

#### Literature References

| Reference | Contribution | Implementation Target |
|-----------|--------------|----------------------|
| [List_Decoding_of_Polar_Codes.md](../literature/List_Decoding_of_Polar_Codes.md) | Path splitting & pruning | `SCLDecoder::expand_paths()` |
| [CRC-Aided_Decoding_of_Polar_Codes.md](../literature/CRC-Aided_Decoding_of_Polar_Codes.md) | CRC-aided final selection | `SCLDecoder::select_best_path()` |
| [LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md](../literature/LLR-Based_Successive_Cancellation_List_Decoding_of_Polar_Codes.md) | Numerically stable metrics | Path metric update without renormalization |

#### Test Vectors Required

<test_vector_spec>
| ID | N | K | L | SNR (dB) | Expected FER | Source |
|----|---|---|---|----------|--------------|--------|
| TV-SCL-01 | 1024 | 512 | 1 | 2.0 | < 0.1 | Tal & Vardy |
| TV-SCL-02 | 1024 | 512 | 8 | 2.0 | < 0.01 | Tal & Vardy |
| TV-SCL-03 | 1024 | 512 | 32 | 2.0 | < 0.001 | Tal & Vardy |
| TV-CASCL-01 | 1024 | 512+CRC16 | 8 | 1.5 | < 0.001 | Niu & Chen |
</test_vector_spec>

</document_spec>

---

## P3: Python Refactor Documents

### ADR-0003: Kiktenko Baseline Strategy

<document_spec id="adr-0003">

**Purpose:** Document the decision to introduce a simplified "Kiktenko-style" 4096-frame baseline strategy with fixed rate ladder (R=0.9 → 0.3), distinct from the dynamic Untainted approach.

**Status:** To be created

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/reconciliation/strategies/baseline.py](../../caligo/caligo/reconciliation/strategies/baseline.py) | Current baseline (to rename) | `BaselineStrategy` class |
| [caligo/reconciliation/rate_selector.py](../../caligo/caligo/reconciliation/rate_selector.py) | Rate selection logic | `select_rate_for_qber()` |
| [caligo/reconciliation/constants.py](../../caligo/caligo/reconciliation/constants.py) | Rate parameters | `RATE_MIN`, `RATE_MAX`, `RATE_STEP` |

#### Literature References

| Reference | Contribution | Key Sections |
|-----------|--------------|--------------|
| [Post-processing procedure for industrial quantum key distribution systems.md](../literature/Post-processing%20procedure%20for%20industrial%20quantum%20key%20distribution%20systems.md) | Kiktenko industrial protocol | §III Rate ladder, §IV Syndrome exchange |
| [Efficient reconciliation protocol for discrete-variable quantum key distribution.md](../literature/Efficient%20reconciliation%20protocol%20for%20discrete-variable%20quantum%20key%20distribution.md) | Elkouss rate-compatible foundation | §II-III Rate adaptation |

#### Decision Drivers

<decision_drivers>
1. **Simplicity:** Fixed rate ladder (0.9, 0.8, ..., 0.3) reduces complexity for industrial deployment
2. **Predictable leakage:** Known syndrome sizes enable exact security proofs
3. **Separation of concerns:** Kiktenko (simple) vs Untainted (optimal) vs Polar (next-gen)
</decision_drivers>

</document_spec>

---

### SPEC: Reconciliation Strategy Protocol (Updated)

<document_spec id="reconciliation-strategy-spec">

**Purpose:** Update the `ReconciliationStrategy` protocol to support both LDPC and Polar backends via the new `SISOCodec` abstraction.

**Status:** To be created

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/reconciliation/strategies/__init__.py](../../caligo/caligo/reconciliation/strategies/__init__.py) | Current protocol definition | `ReconciliationStrategy` ABC (lines 236-342) |
| [caligo/reconciliation/orchestrator.py](../../caligo/caligo/reconciliation/orchestrator.py) | Strategy consumer | Generator-based invocation |
| [caligo/reconciliation/factory.py](../../caligo/caligo/reconciliation/factory.py) | Strategy instantiation | `create_strategy()` factory function |

#### Literature References

| Reference | Contribution | Key Sections |
|-----------|--------------|--------------|
| [Blind Reconciliation.md](../literature/Blind%20Reconciliation.md) | Blind protocol specification | §II-III Iterative revelation |
| [Rate Compatible Protocol for Information.md](../literature/Rate%20Compatible%20Protocol%20for%20Information.md) | Baseline protocol specification | §II Rate adaptation |

</document_spec>

---

### IMPL: Phase 3 - Strategy Refactor

<document_spec id="phase3-strategy-refactor">

**Purpose:** Implementation guide for refactoring the Python strategy layer: rename `baseline.py` → `untainted.py`, create `kiktenko.py`, create `polar.py`, deprecate `blind.py`.

**Status:** To be created

#### Codebase References

| Current File | Action | Target File |
|--------------|--------|-------------|
| `strategies/baseline.py` | Rename | `strategies/untainted.py` |
| (new) | Create | `strategies/kiktenko.py` |
| (new) | Create | `strategies/polar.py` |
| `strategies/blind.py` | Deprecate | Add `@deprecated` decorator |
| `strategies/__init__.py` | Update | New exports, lazy imports |
| `factory.py` | Update | New strategy routing |

#### Acceptance Criteria

<acceptance_criteria>
- [ ] AC-3.1: `from caligo.reconciliation.strategies import UntaintedStrategy` works
- [ ] AC-3.2: `from caligo.reconciliation.strategies import KiktenkoStrategy` works
- [ ] AC-3.3: `from caligo.reconciliation.strategies import PolarStrategy` works
- [ ] AC-3.4: `BlindStrategy` emits `DeprecationWarning` on instantiation
- [ ] AC-3.5: All existing tests pass with renamed imports
- [ ] AC-3.6: Factory routes `"untainted"`, `"kiktenko"`, `"polar"` correctly
</acceptance_criteria>

</document_spec>

---

### IMPL: Phase 4 - Integration & Validation

<document_spec id="phase4-integration">

**Purpose:** End-to-end validation guide ensuring the new Polar codec integrates correctly with SquidASM simulation and meets HPC performance targets.

**Status:** To be created

#### Codebase References

| File | Relevance | Key Sections |
|------|-----------|--------------|
| [caligo/protocol/](../../caligo/caligo/protocol/) | OT protocol integration | Alice/Bob program entry points |
| [caligo/simulation/](../../caligo/caligo/simulation/) | SquidASM runner | Simulation configuration |
| [archive/hpc_migration_roadmap.md](archive/hpc_migration_roadmap.md) | Performance targets | §7 Success Metrics |

#### Validation Targets

<validation_targets>
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Throughput | ≥10 Mbps/core | Benchmark 1M-bit blocks, SCL L=32 |
| Efficiency | ≥90% Shannon | QBER 1-5% test vectors |
| FER | ≤10⁻⁴ | 10K block statistical test |
| Memory | ≤128MB peak | `tracemalloc` profiling |
| Compatibility | 100% existing tests | `pytest` regression suite |
</validation_targets>

</document_spec>

---

## Cross-Reference Matrix

<cross_reference>

| Document | Depends On | Required By |
|----------|-----------|-------------|
| ADR-0001 | (none) | SISO Protocol, Phase 1 |
| ADR-0002 | ADR-0001 | Phase 1, Rust API |
| ADR-0003 | ADR-0001 | Phase 3 |
| SISO Protocol | ADR-0001 | Phase 1, Phase 2, Strategy Protocol |
| Rust API | ADR-0002, SISO Protocol | Phase 2 |
| Strategy Protocol | SISO Protocol, ADR-0003 | Phase 3, Phase 4 |
| Phase 1 | ADR-0001, ADR-0002, SISO Protocol | Phase 2 |
| Phase 2 | Phase 1, Rust API | Phase 3 |
| Phase 3 | Phase 2, ADR-0003, Strategy Protocol | Phase 4 |
| Phase 4 | Phase 3 | (terminal) |

</cross_reference>

---

## AI Agent Context Loading Instructions

<agent_instructions>

When generating content for any document in this project, load context in this order:

1. **Always load:** This file (`initialization.md`) for reference mapping
2. **Load ADR dependencies:** Check the cross-reference matrix
3. **Load codebase files:** As listed in the document's "Codebase References"
4. **Load literature:** As listed in the document's "Literature References"
5. **Load HPC constraints:** `archive/hpc_migration_roadmap.md` for performance context

**Context Budget Guidance:**
- ADR documents: ~2000 tokens literature, ~1000 tokens codebase
- Spec documents: ~1000 tokens literature, ~2000 tokens codebase
- Impl documents: ~1000 tokens literature, ~3000 tokens codebase, test vectors

</agent_instructions>

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-30 | AI Agent | Initial creation |
| 1.1.0 | 2026-02-02 | Context Engineer | P0 complete: ADR-0001 accepted, SISO Codec Protocol accepted |
| 1.2.0 | 2026-02-02 | Context Engineer | P1 complete: ADR-0002 accepted, Phase 1 IMPL ready; synthetic test vectors specified |
# [ADR-0001] Polar Codec Adoption for QKD Reconciliation

<metadata>
adr_id: 0001
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
The current reconciliation subsystem relies exclusively on LDPC codes with belief propagation (BP) decoding. While functional, this architecture exhibits fundamental limitations that constrain both error correction performance and throughput on the target HPC platform.

### Current Architecture Limitations

1. **Error Floor Phenomenon:** The LDPC BP decoder (N=4096, R₀=0.5) exhibits an error floor at Frame Error Rate (FER) < 10⁻⁴. Beyond this threshold, increasing BP iterations yields diminishing returns [1, §I]. This limitation is intrinsic to the BP algorithm's handling of short cycles in the Tanner graph.

2. **Finite-Length Performance Gap:** At moderate block lengths (N ≤ 4096), LDPC codes under BP decoding underperform compared to capacity-achieving codes. Empirical studies show a ~0.5 dB gap from Shannon limit at FER = 10⁻⁴ [2, Fig. 2].

3. **Throughput Constraints:** The iterative nature of BP decoding, combined with the lack of GPU acceleration on the AMD Genoa target (192 cores, 2GB/core), limits aggregate throughput. Current implementation achieves ~5 Mbps/core with 60 iterations.

### Target Requirements

| Requirement | Current LDPC | Target |
|-------------|--------------|--------|
| FER floor | 10⁻⁴ | 10⁻⁶ |
| Throughput/core | ~5 Mbps | >10 Mbps |
| Memory/core | ~50 MB | <128 MB |
| Rate adaptability | Puncturing only | Native + Puncturing |

### Polar Code Advantages

Polar codes, introduced by Arıkan [3], are the first provably capacity-achieving codes with explicit construction. Recent advances in decoding algorithms address their historical weaknesses:

1. **Successive Cancellation List (SCL) Decoding:** Tal & Vardy [1] demonstrated that SCL with list size L=32 approaches maximum-likelihood (ML) performance, with complexity O(LN log N) and space O(LN).

2. **CRC-Aided SCL (CA-SCL):** Niu & Chen [4] showed that CRC precoding enables list disambiguation, achieving FER < 10⁻⁶ and outperforming WiMAX LDPC codes at comparable rates [1, Fig. 2].

3. **Fast-SSC Optimization:** Sarkis et al. [5] achieved 40× throughput improvement over standard SC by exploiting Rate-0, Rate-1, Repetition, and Single-Parity-Check node patterns.

4. **QKD-Specific Validation:** Cao et al. [6] demonstrated polar code reconciliation achieving FER < 10⁻³ across SNR range -0.5 to -4.5 dB, validating applicability to CV-QKD channels.

### HPC Platform Constraints

The AMD Genoa target imposes specific constraints:

- **No GPU:** CUDA/ROCm unavailable; all computation on CPU cores
- **Memory:** 2GB per core; SCL(L=32, N=2²⁰) requires ~128MB, fitting constraint
- **SIMD:** AVX-512 available; Rust SIMD intrinsics can exploit this
- **Parallelism:** 192 cores; block-parallel processing viable

</context>

---

## Decision

<decision>
We adopt Polar codes with CRC-Aided Successive Cancellation List (CA-SCL) decoding as the primary error correction mechanism for QKD reconciliation, while maintaining LDPC as a fallback option via a unified SISO interface.

### Core Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block length N | 4096 (2¹²) | Drop-in replacement for existing LDPC frame size |
| Default list size L | 8 | Optimal throughput/performance trade-off for typical QBER |
| Maximum list size L | 32 | Configurable for extreme noise conditions |
| CRC polynomial | CRC-16-CCITT | 16-bit integrity with low overhead (0.4% at N=4096) |
| Frozen bit construction | Gaussian Approximation | Channel-adaptive, O(N log N) complexity |

### Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RECONCILIATION LAYER                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              SISOCodec Protocol (Python)                ││
│  │   encode() | decode_hard() | decode_soft()              ││
│  └─────────────────────────────────────────────────────────┘│
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │  LDPCCodecAdapter│           │   PolarCodec    │          │
│  │  (Legacy Wrapper)│           │   (New Primary) │          │
│  └────────┬────────┘           └────────┬────────┘          │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │  Numba BP Kernel │           │  Rust SCL Crate │          │
│  │  (Existing)      │           │  (PyO3 Bindings)│          │
│  └─────────────────┘           └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Scope Boundaries

**This decision DOES cover:**
- Primary codec selection (Polar CA-SCL)
- Interface abstraction (SISOCodec protocol)
- Parameter defaults (L=8, CRC-16, N=4096)
- Backward compatibility strategy (adapter pattern)

**This decision does NOT cover:**
- Rust crate implementation details (see ADR-0002)
- Concatenated LDPC-Polar schemes (future work)
- Rate-adaptive frozen bit recalculation (implementation detail)

</decision>

---

## Consequences

<consequences>

### Positive

1. **FER Improvement:** CA-SCL(L=8) achieves FER ≈ 10⁻⁵ at design SNR, breaking LDPC error floor [1, Fig. 1]. With L=32, FER < 10⁻⁶ is achievable [4, §III].

2. **Throughput Gain:** Fast-SSC optimizations enable 10-40× throughput improvement over naive SC [5, §I]. Conservative estimate: >10 Mbps/core on AMD EPYC with AVX-512.

3. **Memory Efficiency:** SCL(L=32, N=4096) requires ~4MB working memory, well within 2GB/core constraint. Even N=2²⁰ at L=32 fits in ~128MB.

4. **SISO Composability:** The `decode_soft()` interface returning extrinsic LLRs enables future IC-LDPC-Polar concatenation [7, §III] without architectural changes.

5. **Explicit Construction:** Polar codes have deterministic frozen bit patterns (no random ensemble), ensuring reproducibility and simplifying verification.

### Negative

1. **New Dependency:** Requires Rust extension crate with PyO3 bindings. Adds build complexity and cross-platform testing burden.

2. **Development Overhead:** Estimated 3-5 days for core SCL decoder implementation, plus 2 days for Fast-SSC optimizations.

3. **Frozen Bit Sensitivity:** Polar code performance depends on accurate channel estimation. Mismatched frozen bits degrade FER significantly.

4. **List Size Trade-off:** Higher L improves FER but increases latency linearly. Must tune L per-application.

### Neutral

1. **Parallel Compatibility:** Both LDPC (via adapter) and Polar codecs share the SISOCodec interface; strategies remain agnostic.

2. **Rate Adaptation:** Polar supports native rate adaptation via frozen bit count, complementing puncturing/shortening used by LDPC.

3. **Testing Infrastructure:** Existing reconciliation test harness applies to both codecs via unified interface.

</consequences>

---

## Alternatives Considered

<alternatives>

### Alternative 1: Enhanced LDPC (Higher Iteration Count / Improved Scheduling)

**Description:** Increase BP iterations beyond 60, implement layered scheduling, or use min-sum with offset correction to improve LDPC performance.

**Rejected because:**
- Error floor is fundamental to BP on short cycles; more iterations cannot break it [1, §I]
- Layered scheduling provides ~30% speedup but no FER improvement at floor
- Does not address finite-length performance gap vs. capacity

### Alternative 2: Turbo Codes

**Description:** Adopt 3GPP-style turbo codes with iterative BCJR decoding.

**Rejected because:**
- Patent encumbrance concerns (though largely expired, licensing uncertainty remains)
- Inferior finite-length performance compared to CA-SCL at N < 8192 [1, Fig. 3]
- Higher memory footprint due to trellis storage
- No clear SISO interface advantage over Polar

### Alternative 3: Raptor/Fountain Codes

**Description:** Use rateless Raptor codes for natural rate adaptation.

**Rejected because:**
- Designed for erasure channels, not AWGN/BSC typical of QKD
- Overhead for small block sizes (N=4096) is prohibitive
- No native soft-output capability for concatenation

### Alternative 4: LDPC-Polar Concatenation as Primary

**Description:** Skip standalone Polar and implement concatenated IC-LDPC-Polar [7] directly.

**Deferred because:**
- Requires validated SISO interface first (this ADR enables that)
- Higher complexity; should validate Polar alone before concatenation
- Can be added later without architectural change due to SISOCodec abstraction

### Alternative 5: Pure SC Decoding (No List)

**Description:** Use standard successive cancellation without list extension.

**Rejected because:**
- Significant performance gap vs. ML at moderate N [1, Fig. 1]
- Cannot leverage CRC for disambiguation
- Only acceptable for very long codes (N > 2²⁰) where polarization completes

</alternatives>

---

## References

<references>

[1] I. Tal and A. Vardy, "List Decoding of Polar Codes," *IEEE Trans. Inf. Theory*, vol. 61, no. 5, pp. 2213-2226, May 2015.
    - Key sections: §I (motivation), §IV (algorithm complexity), Fig. 1-3 (performance)

[2] A. Balatsoukas-Stimming, M. B. Parizi, and A. Burg, "LLR-Based Successive Cancellation List Decoding of Polar Codes," *IEEE Trans. Signal Process.*, vol. 63, no. 19, pp. 5165-5179, Oct. 2015.
    - Key sections: §II-B (LLR update equations), §III (numerical stability)

[3] E. Arıkan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels," *IEEE Trans. Inf. Theory*, vol. 55, no. 7, pp. 3051-3073, Jul. 2009.
    - Foundation paper for polar codes

[4] K. Niu and K. Chen, "CRC-Aided Decoding of Polar Codes," *IEEE Commun. Lett.*, vol. 16, no. 10, pp. 1668-1671, Oct. 2012.
    - Key sections: §III (CA-SCL/SCS algorithms), performance vs. turbo codes

[5] G. Sarkis, P. Giard, A. Vardy, C. Thibeault, and W. J. Gross, "Fast Polar Decoders: Algorithm and Implementation," *IEEE J. Sel. Areas Commun.*, vol. 32, no. 5, pp. 946-957, May 2014.
    - Key sections: §III-IV (Rate-0, Rate-1, Rep, SPC nodes), §VIII (FPGA results)

[6] Z. Cao, X. Chen, G. Chai, K. Liang, and Y. Yuan, "Rate-Adaptive Polar-Coding-Based Reconciliation for CV-QKD at Low SNR," *Phys. Rev. Applied*, vol. 19, 044023, Apr. 2023.
    - Key sections: §I-II (QKD context), FER < 10⁻³ at SNR -4.5 dB

[7] S. M. Abbas, Y. Fan, J. Chen, and C.-Y. Tsui, "Concatenated LDPC-Polar Codes Decoding Through Belief Propagation," *Proc. IEEE Int. Symp. Circuits Syst. (ISCAS)*, 2017.
    - Key sections: §III (SISO message passing), Algorithm 1 (channel selection)

[8] F. Hua et al., "RC-LDPC-Polar Codes for Information Reconciliation in CV-QKD," *MDPI Electronics*, 2024.
    - Key sections: §2 (decoder interface), §3.2-3.3 (rate-compatible construction)

</references>

---

## Implementation Notes

<implementation_notes>

### Phase 1: Rust Foundation (ADR-0002)

Create `polar-scl` Rust crate with:
- Generic N=2^n support (compile-time or runtime)
- SIMD-optimized LLR operations (AVX-512)
- PyO3 bindings exposing `encode()`, `decode_scl()`

### Phase 2: Python Integration

1. **PolarCodec class** in `caligo.reconciliation.strategies.polar_codec`:
   - Implements `SISOCodec` protocol
   - Wraps Rust crate via PyO3
   - Handles frozen bit construction (Gaussian Approximation)

2. **LDPCCodecAdapter class** in `caligo.reconciliation.strategies.ldpc_adapter`:
   - Wraps existing `LDPCCodec`
   - Implements `SISOCodec` protocol
   - `decode_soft()`: Extract extrinsic LLRs from BP edge messages

### Phase 3: Strategy Refactor

Update `BaselineStrategy` and `BlindStrategy` to accept `SISOCodec` instead of `LDPCCodec`:
- No strategy logic changes required
- Codec selection via configuration/factory

### Verification Criteria

| Criterion | Method | Target |
|-----------|--------|--------|
| FER at design SNR | Monte Carlo (10⁶ frames) | < 10⁻⁵ with L=8 |
| Throughput | Benchmark on EPYC 7763 | > 10 Mbps/core |
| Memory | Valgrind massif | < 128 MB at L=32, N=4096 |
| Bit-exactness | Reference vector comparison | Match Tal-Vardy reference |

### Unblocked Documents

This ADR unblocks:
- **SPEC: SISO Codec Protocol** (`specs/siso-codec-protocol.md`)
- **ADR-0002: Rust Native Extension** (`adr/0002-rust-extension.md`)
- **IMPL: Phase 1 Rust Foundation** (`impl/phase1-rust-foundation.md`)

</implementation_notes>

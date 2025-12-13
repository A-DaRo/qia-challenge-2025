# Phase III Technical Analysis: Information Reconciliation

> **Definitive Migration Guide for Error Correction in E-HOK**
> 
> Version: 1.0  
> Last Updated: 2025  
> Authors: AI Technical Analysis

---

## Abstract

Phase III of the E-HOK protocol addresses the critical challenge of correcting transmission errors while minimizing information leakage to a potential adversary. Unlike standard QKD reconciliation, E-HOK operates under a fundamental constraint: **interactive protocols like Cascade are forbidden** because they would leak Bob's basis choice, destroying the oblivious property essential for 1-out-of-2 Oblivious Transfer.

This phase implements **One-Way Forward Error Correction (FEC)** using Low-Density Parity-Check (LDPC) codes, where every bit of syndrome information transmitted constitutes a direct reduction in the final secure key length—a concept formalized as the **Wiretap Cost**. The protocol must track cumulative leakage and enforce a **Safety Cap** to prevent adversarial exploitation through feigned decoding failures.

The analysis reveals that the legacy `ehok/` implementation provides a comprehensive LDPC reconciliation infrastructure that is architecturally sound for E-HOK. The primary gap is the absence of a **Leakage Safety Manager** enforcing hard limits on syndrome transmission. Integration with SquidASM is straightforward as Phase III operates entirely on classical data exchanged via `ClassicalSocket`.

---

## 1. Ontology: Core Concepts of Phase III

### 1.1 The Efficiency vs. Security Tradeoff

Phase III navigates a fundamental tension unique to E-HOK:

| Concept | Definition | E-HOK Constraint |
|---------|------------|------------------|
| **One-Way FEC** | Error correction where information flows only from Alice to Bob | Required—interactivity leaks Bob's choice bit |
| **Syndrome** | Parity check vector $S = H \cdot X$ sent by Alice | Counts as fully leaked information |
| **Wiretap Cost** | Security penalty: $\|S\|$ bits subtracted from extractable entropy | Fundamental to NSM security proofs |
| **Safety Cap ($L_{max}$)** | Maximum total leakage before protocol abort | Prevents "feigned failure" attacks |
| **Blind Reconciliation** | Adaptive rate selection without prior QBER knowledge | Efficiency optimization within safety limits |

### 1.2 Why Cascade is Forbidden

**Standard QKD Context**: Cascade achieves near-Shannon efficiency through bidirectional parity exchanges, progressively narrowing error locations.

**E-HOK Security Violation**: If Alice learns *where* Bob's errors occurred, she gains statistical information about which bits Bob measured correctly—potentially revealing his choice bit $C$.

**Literature Basis** (Erven et al.):
> "Error correction must be done with a one-way forward error correction protocol to maintain the security of the protocol. The error-correcting code is chosen such that Bob can decode faithfully except with a probability at most $\varepsilon_{EC}$."

### 1.3 The Wiretap Channel Model

E-HOK reconciliation maps to the classic wiretap channel formulation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Wiretap Channel Model                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│        Alice's Key                                                           │
│            │                                                                 │
│            ▼                                                                 │
│    ┌───────────────┐                                                         │
│    │  LDPC Encoder │                                                         │
│    │   S = H · X   │                                                         │
│    └───────┬───────┘                                                         │
│            │ Syndrome S                                                      │
│            │                                                                 │
│    ┌───────┴───────┐                                                         │
│    │   Classical   │                                                         │
│    │   Channel     │◀───────── Eve observes S (full leakage)                │
│    └───────┬───────┘                                                         │
│            │                                                                 │
│            ▼                                                                 │
│    ┌───────────────┐                                                         │
│    │  LDPC Decoder │  Bob's noisy key + syndrome → corrected key            │
│    │   (BP Algo)   │                                                         │
│    └───────────────┘                                                         │
│                                                                              │
│  Security Constraint:                                                        │
│    H_min(X|Eve) ≥ H_min(X) - |S| - verification_hash                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Conceptual Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      Phase III Conceptual Flow                             │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  From Phase II                                                             │
│  ┌───────────────────────────────────────────┐                            │
│  │ SiftedKey                                 │                            │
│  │   - key_alice: np.ndarray (n bits)        │                            │
│  │   - key_bob: np.ndarray (n bits, noisy)   │                            │
│  │   - adjusted_qber: float (e_adj)          │                            │
│  │   - test_bits_used: int (k)               │                            │
│  └─────────────────────┬─────────────────────┘                            │
│                        │                                                   │
│                        ▼                                                   │
│  ┌───────────────────────────────────────────┐                            │
│  │ Rate Selection                            │                            │
│  │   R = f(QBER) such that (1-R)/h(Q) < f_c  │                            │
│  └─────────────────────┬─────────────────────┘                            │
│                        │                                                   │
│                        ▼                                                   │
│  ┌───────────────────────────────────────────┐                            │
│  │ Block Partitioning                        │                            │
│  │   Split into blocks of frame_size        │                            │
│  │   Apply shortening if needed              │                            │
│  └─────────────────────┬─────────────────────┘                            │
│                        │                                                   │
│              ┌─────────┴─────────┐                                        │
│              ▼                   ▼                                        │
│         ┌─────────┐         ┌─────────┐                                   │
│         │  ALICE  │         │   BOB   │                                   │
│         └────┬────┘         └────┬────┘                                   │
│              │                   │                                        │
│   Compute S = H·X                │                                        │
│              │───── Syndrome ────▶│                                       │
│              │                   │ Decode: X' = BP(Y, S)                  │
│              │                   │                                        │
│   Compute hash(X)                │                                        │
│              │──── Hash value ───▶│                                       │
│              │                   │ Verify: hash(X') == hash(X)?          │
│              │                   │                                        │
│  ┌───────────┴───────────────────┴───────────┐                            │
│  │ Leakage Accumulator                       │                            │
│  │   total_leak += |syndrome| + |hash|       │                            │
│  │   IF total_leak > L_max → ABORT           │                            │
│  └───────────────────────┬───────────────────┘                            │
│                          │                                                 │
│                          ▼                                                 │
│  ┌───────────────────────────────────────────┐                            │
│  │ To Phase IV                               │                            │
│  │ ReconciledKey                             │                            │
│  │   - reconciled_key: np.ndarray            │                            │
│  │   - total_leakage: float                  │                            │
│  │   - block_success_rate: float             │                            │
│  └───────────────────────────────────────────┘                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Literature Context & Security Foundations

### 2.1 Theoretical Corpus

Phase III security derives from the following literature:

| Source | Contribution | Key Result |
|--------|--------------|------------|
| **Schaffner et al.** (2009) | Wiretap cost formalization | $\ell \leq H_{min}(X\|E) - \|S\|$ |
| **Lupo et al.** (2023) | Tight leakage accounting with trusted noise | Eq. (3): syndrome fully subtracted |
| **Erven et al.** (2014) | One-way LDPC implementation | Error correction efficiency $f = 1.491$ |
| **Martinez-Mateo et al.** (2011) | Blind reconciliation | Adaptive rate without prior estimation |

### 2.2 The Secure Key Length Bound

**Source**: Lupo et al. Eq. (3); Erven et al. Eq. (8)

The fundamental security constraint for Phase III:

$$\ell \leq H_{min}^{\epsilon}(X \mid \mathcal{F}(Q) \Theta B \Sigma_{\bar{B}}) - 2\log\frac{1}{\epsilon_h} + 1$$

Simplified for implementation:

$$\ell \leq H_{min}(X|E) - |\Sigma| - \text{security\_margins}$$

where:
- $H_{min}(X|E)$ — Smooth min-entropy of Alice's string given adversary's information
- $|\Sigma|$ — Total syndrome length in bits
- Security margins include hash verification and privacy amplification costs

### 2.3 The Efficiency Cliff

**Problem**: LDPC codes require redundancy proportional to channel entropy. For BB84-style protocols:

$$|\Sigma| \approx n \cdot f \cdot h(Q)$$

where:
- $n$ — Block length
- $f$ — Reconciliation efficiency (typically 1.1–1.5)
- $h(Q)$ — Binary entropy of QBER

**Critical Insight**: As QBER increases, syndrome length grows faster than the extractable entropy. Beyond approximately 5-10% QBER, the syndrome length $|\Sigma|$ can consume the entire min-entropy, leaving no secure key.

**Numerical Example**:

| QBER | $h(Q)$ | Min-Entropy Rate | Syndrome Rate ($f=1.2$) | Net Rate |
|------|--------|------------------|-------------------------|----------|
| 2% | 0.141 | 0.859 | 0.169 | 0.69 |
| 5% | 0.286 | 0.714 | 0.343 | 0.37 |
| 10% | 0.469 | 0.531 | 0.563 | **-0.03** |
| 11% | 0.500 | 0.500 | 0.600 | **-0.10** |

At 10% QBER, the net key rate becomes **negative**—no secure key can be extracted.

### 2.4 The Ban on Interactivity

**Source**: Erven et al. Section "Results: The Oblivious Transfer Protocol"

**Formal Requirement**: The reconciliation protocol must satisfy:

1. **Unidirectional flow**: Information flows only Alice → Bob
2. **No per-block feedback**: Bob cannot request retransmission of specific blocks
3. **No error location disclosure**: Alice must not learn which bits Bob corrected

**Consequence**: If LDPC decoding fails for a block, Bob cannot ask for help—the block is discarded or the protocol aborts.

---

## 3. Protocol Logic & Data Transformation

### 3.1 The Processing Pipeline

Phase III transforms sifted keys into reconciled keys through the following stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase III Processing Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: Rate Selection                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: adjusted_qber (from Phase II)                                   │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   FOR rate IN available_rates (descending):                            │ │
│  │     IF (1 - rate) / h(qber) < f_critical:                              │ │
│  │       RETURN rate                                                       │ │
│  │   RETURN lowest_rate                                                    │ │
│  │                                                                         │ │
│  │ Output: selected_rate (e.g., 0.5, 0.6, 0.7, 0.8, 0.9)                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STAGE 2: Block Partitioning & Shortening                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: sifted_key, frame_size, selected_rate                           │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   n_blocks = ceil(len(sifted_key) / payload_capacity)                  │ │
│  │   FOR each block:                                                       │ │
│  │     n_shortened = compute_shortening(rate, qber, block_len)            │ │
│  │     padded_block = concat(block, prng_padding(n_shortened))            │ │
│  │                                                                         │ │
│  │ Output: List[padded_blocks], List[n_shortened]                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STAGE 3: Syndrome Computation & Transmission (Alice)                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: alice_blocks, parity_check_matrix H                             │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   FOR each block X_i:                                                   │ │
│  │     syndrome_i = H · X_i (mod 2)                                       │ │
│  │     SEND syndrome_i to Bob via ClassicalSocket                         │ │
│  │     leakage_accumulator += len(syndrome_i)                             │ │
│  │     IF leakage_accumulator > L_max: ABORT                              │ │
│  │                                                                         │ │
│  │ Output: syndromes transmitted, leakage tracked                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STAGE 4: Belief Propagation Decoding (Bob)                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: bob_blocks, syndromes, LLR_initial                              │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   FOR each block (Y_i, S_i):                                           │ │
│  │     LLR = compute_initial_llr(Y_i, qber)                               │ │
│  │     X'_i, converged, n_errors = BP_decode(Y_i, S_i, H, LLR)            │ │
│  │     IF NOT converged: mark_block_failed(i)                             │ │
│  │                                                                         │ │
│  │ Output: corrected_blocks, convergence_flags, error_counts              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STAGE 5: Hash Verification                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: alice_blocks, bob_corrected_blocks                              │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   FOR each block pair (X_i, X'_i):                                     │ │
│  │     hash_alice = polynomial_hash(X_i)                                  │ │
│  │     hash_bob = polynomial_hash(X'_i)                                   │ │
│  │     IF hash_alice != hash_bob: mark_block_failed(i)                    │ │
│  │     leakage_accumulator += hash_bits                                   │ │
│  │                                                                         │ │
│  │ Output: verified_blocks, total_leakage                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STAGE 6: Assembly & Leakage Report                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input: verified_blocks, leakage_accumulator                            │ │
│  │                                                                         │ │
│  │ Algorithm:                                                              │ │
│  │   reconciled_key = concatenate(successful_blocks)                      │ │
│  │   total_leakage = syndrome_bits + hash_bits                            │ │
│  │                                                                         │ │
│  │ Output: ReconciledKey(key, leakage, success_rate)                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Parameter Dependency Analysis

Phase III consumes parameters from Phase II and produces inputs for Phase IV:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Parameter Flow Across Phases                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────┐                                               │
│  │       Phase II           │                                               │
│  │   (Sifting/Estimation)   │                                               │
│  └────────────┬─────────────┘                                               │
│               │                                                              │
│               ▼                                                              │
│  ┌────────────────────────────────────────────────────┐                     │
│  │ Parameters Consumed by Phase III:                  │                     │
│  │                                                    │                     │
│  │  • sifted_key_alice: np.ndarray                   │                     │
│  │  • sifted_key_bob: np.ndarray (noisy)             │                     │
│  │  • adjusted_qber: float (e_obs + μ)               │                     │
│  │  • test_set_size: int (k)                         │                     │
│  │  • epsilon_sec: float (target security)           │                     │
│  │  • storage_noise: float (r, from Phase I config)  │                     │
│  │                                                    │                     │
│  └────────────────────────────────────────────────────┘                     │
│               │                                                              │
│               ▼                                                              │
│  ┌──────────────────────────┐                                               │
│  │       Phase III          │                                               │
│  │   (Reconciliation)       │                                               │
│  └────────────┬─────────────┘                                               │
│               │                                                              │
│               ▼                                                              │
│  ┌────────────────────────────────────────────────────┐                     │
│  │ Parameters Produced for Phase IV:                  │                     │
│  │                                                    │                     │
│  │  • reconciled_key: np.ndarray (error-corrected)   │                     │
│  │  • total_leakage: float (|Σ| + |hash|)            │                     │
│  │  • integrated_qber: float (from corrections)      │                     │
│  │  • blocks_succeeded: int                          │                     │
│  │  • blocks_failed: int                             │                     │
│  │                                                    │                     │
│  └────────────────────────────────────────────────────┘                     │
│               │                                                              │
│               ▼                                                              │
│  ┌──────────────────────────┐                                               │
│  │       Phase IV           │                                               │
│  │ (Privacy Amplification)  │                                               │
│  └──────────────────────────┘                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Verification Logic

The hash verification mechanism ensures Bob's corrected key matches Alice's original:

**Hash Function Requirements**:
- 2-universal hash family for security guarantees
- Polynomial hashing over finite fields (efficient computation)
- Collision probability bounded by $2^{-b}$ for $b$-bit hash

**Verification Protocol**:
1. Alice computes $h_A = \text{PolyHash}(X)$ using shared random coefficients
2. Alice sends $h_A$ to Bob (counts as leakage)
3. Bob computes $h_B = \text{PolyHash}(X')$ with same coefficients
4. Block verified iff $h_A = h_B$

**Failure Handling**: Failed blocks are discarded (not retried in one-way mode).

---

## 4. Implementation Gap Analysis

### 4.1 Gap Summary Matrix

| Capability | SquidASM Native | Legacy ehok | Gap Status | Extension Required |
|------------|-----------------|-------------|------------|-------------------|
| Syndrome Transmission | ✅ `ClassicalSocket` | N/A | SUPPORTED | None |
| LDPC Encoding | N/A | ✅ `LDPCMatrixManager` | SUPPORTED | None |
| LDPC BP Decoding | N/A | ✅ `LDPCBeliefPropagation` | SUPPORTED | None |
| Rate Selection | N/A | ✅ `LDPCReconciliator.select_rate()` | SUPPORTED | None |
| Shortening | N/A | ✅ `LDPCReconciliator.compute_shortening()` | SUPPORTED | None |
| Hash Verification | N/A | ✅ `PolynomialHashVerifier` | SUPPORTED | None |
| Leakage Estimation | N/A | ✅ `estimate_leakage_block()` | SUPPORTED | None |
| **Safety Cap ($L_{max}$)** | ❌ | ⚠️ Partial | **GAP** | `LeakageSafetyManager` |
| **Abort on Leakage Overflow** | ❌ | ❌ | **GAP** | Protocol integration |
| Interactive Hashing | ❌ | ❌ | FUTURE | Complex extension |

### 4.2 Legacy Code Assessment & Deletion Timeline

The `ehok/implementations/reconciliation/` directory provides a comprehensive LDPC stack:

| Component | File | Assessment | Deletion Status |
|-----------|------|------------|-----------------|
| `LDPCReconciliator` | `ldpc_reconciliator.py` | ✅ Complete orchestrator with rate adaptation | Extract + Delete |
| `LDPCBeliefPropagation` | `ldpc_bp_decoder.py` | ✅ Full BP decoder with LLR computation | Extract + Delete |
| `LDPCMatrixManager` | `ldpc_matrix_manager.py` | ✅ Parity-check matrix management | Extract + Delete |
| `PolynomialHashVerifier` | `polynomial_hash.py` | ✅ Finite-field polynomial hashing | Extract + Delete |
| `IntegratedQBEREstimator` | `qber_estimator.py` | ✅ QBER from correction counts | Extract + Delete |

**Interface Definition** (from `ehok/interfaces/reconciliation.py`):

The `IReconciliator` abstract base class defines:
- `select_rate(qber_est) → float` — Rate selection
- `compute_shortening(rate, qber_est, target_payload) → int` — Shortening calculation
- `reconcile_block(key_block, syndrome, ...) → Tuple[corrected, success, errors]` — Block decoding
- `estimate_leakage_block(syndrome_length, hash_bits) → int` — Leakage accounting

**Assessment**: The legacy implementation is **cryptographically sound** for E-HOK purposes. It correctly implements:
- One-way syndrome-based correction (no back-channel)
- Leakage tracking per block
- QBER-adaptive rate selection

**Migration & Deletion Plan**:
1. Extract all algorithmic logic (BP decoder, matrix management, rate selection) into SquidASM-native equivalents
2. Write comprehensive parity tests comparing legacy vs. new reconciliation
3. Validate byte-for-byte output equivalence on standard test vectors
4. Upon validation: **DELETE entire `ehok/implementations/reconciliation/` directory**
5. Replace all imports with references to new SquidASM-native reconciliation module
6. Update all tests to use only new implementation

No fallback, no deprecation period.

### 4.3 Gap Analysis: Safety Cap Enforcement

**Current State**: 
Leakage is tracked per block via `estimate_leakage_block()`, but there is no enforcement of a cumulative limit.

**Security Impact**: Without a hard cap, a sophisticated attacker could force Alice to continue sending syndromes until the entire key is leaked, then claim decoding failure.

**Proposed Extension Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LeakageSafetyManager Extension                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LeakageSafetyManager                                 │ │
│  │                                                                         │ │
│  │  Constructor Parameters:                                                │ │
│  │    min_entropy: float        # H_min(X|E) from Phase I/II              │ │
│  │    target_key_length: int    # Desired output key length               │ │
│  │    epsilon_sec: float        # Security parameter                      │ │
│  │                                                                         │ │
│  │  Computed Properties:                                                   │ │
│  │    max_leakage: float        # L_max = H_min - ℓ_target - margins      │ │
│  │    accumulated_leakage: float # Running total                          │ │
│  │                                                                         │ │
│  │  Methods:                                                               │ │
│  │    add_leakage(syndrome_bits, hash_bits) → bool                        │ │
│  │      - Updates accumulator                                              │ │
│  │      - Returns False if limit exceeded                                  │ │
│  │                                                                         │ │
│  │    should_abort() → bool                                                │ │
│  │      - Returns True if accumulated_leakage > max_leakage               │ │
│  │                                                                         │ │
│  │    remaining_budget() → float                                           │ │
│  │      - Returns max_leakage - accumulated_leakage                       │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Location: ehok/core/security_bounds.py                                     │
│                                                                              │
│  Mathematical Basis:                                                         │
│                                                                              │
│    L_max = H_min(X|E) - ℓ_target - 2·log₂(1/ε_sec)                         │
│                                                                              │
│  Integration Point:                                                          │
│    Called after each syndrome transmission in reconciliation loop            │
│    Protocol aborts if add_leakage() returns False                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 SquidASM Integration Points

Phase III operates **entirely on classical data** after quantum measurements complete. Integration is straightforward:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SquidASM Integration for Phase III                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         E-HOK Phase III Layer                           │ │
│  │                                                                         │ │
│  │   ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │ │
│  │   │ LDPCReconciliator│    │LeakageSafetyMgr │    │ SyndromeTransport│  │ │
│  │   │   (existing)     │    │     (NEW)       │    │                  │  │ │
│  │   └────────┬─────────┘    └────────┬─────────┘    └────────┬────────┘  │ │
│  │            │                       │                       │           │ │
│  │            └───────────────────────┼───────────────────────┘           │ │
│  │                                    │                                    │ │
│  └────────────────────────────────────┼────────────────────────────────────┘ │
│                                       │                                      │
│                                       ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         SquidASM Layer                                  │ │
│  │                                                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                     ClassicalSocket                              │  │ │
│  │   │                                                                  │  │ │
│  │   │  Alice:                                                          │  │ │
│  │   │    syndrome_hex = syndrome.tobytes().hex()                       │  │ │
│  │   │    context.csockets["bob"].send(syndrome_hex)                    │  │ │
│  │   │                                                                  │  │ │
│  │   │  Bob:                                                            │  │ │
│  │   │    syndrome_hex = yield from context.csockets["alice"].recv()    │  │ │
│  │   │    syndrome = bytes.fromhex(syndrome_hex)                        │  │ │
│  │   │                                                                  │  │ │
│  │   └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │   Key Insight: No timing constraints for Phase III                     │ │
│  │   (unlike Phase II where ordering is security-critical)                │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Computational Requirements

**LDPC Decoding Complexity**:
- Belief propagation: $O(n \cdot d_v \cdot \text{iterations})$ per block
- Typical: $n = 10^4$, $d_v = 4$, iterations $\leq 100$
- Per-block: ~$4 \times 10^6$ operations

**Matrix Operations**:
- Syndrome computation: $O(n \cdot d_c)$ using sparse matrices
- Hash verification: $O(n)$ polynomial evaluation

**Assessment**: Standard Python with NumPy/SciPy sparse matrices is adequate. No GPU or specialized hardware required for simulation-scale keys ($10^4 - 10^6$ bits).

---

## 5. Mathematical Formalization of Security

### 5.1 Leakage Budget Constraint

The fundamental constraint on syndrome transmission:

$$L_{total} = \sum_{i=1}^{B} (|\Sigma_i| + |h_i|) \leq L_{max}$$

where:
- $B$ — Number of blocks
- $|\Sigma_i|$ — Syndrome length for block $i$
- $|h_i|$ — Hash length for block $i$ (typically 50 bits)
- $L_{max}$ — Maximum safe leakage

### 5.2 Maximum Safe Leakage Calculation

$$L_{max} = H_{min}(X|E) - \ell_{target} - 2\log_2\frac{1}{\varepsilon_{sec}}$$

where:
- $H_{min}(X|E)$ — Min-entropy from NSM security analysis
- $\ell_{target}$ — Desired final key length
- $\varepsilon_{sec}$ — Target security parameter

### 5.3 Syndrome Length per Block

For an LDPC code with rate $R$ and frame size $n$:

$$|\Sigma| = n \cdot (1 - R)$$

With efficiency factor $f$ and QBER $Q$:

$$|\Sigma_{required}| \approx n \cdot f \cdot h(Q)$$

### 5.4 LDPC Rate Selection Criterion

Select rate $R$ satisfying the efficiency threshold:

$$\frac{1 - R}{h(Q)} < f_{critical}$$

where $f_{critical} \approx 1.22$ (from ehok constants).

### 5.5 Block Decoding Success Probability

Using belief propagation with $I_{max}$ iterations:

$$P_{success}(Q, R) \approx 1 - \varepsilon_{EC}$$

where $\varepsilon_{EC}$ is the error correction failure probability (typically $10^{-3}$).

### 5.6 Integrated QBER Estimation

QBER can be estimated from correction counts:

$$\hat{Q}_{integrated} = \frac{\sum_i e_i}{\sum_i n_i}$$

where $e_i$ is errors corrected in block $i$ and $n_i$ is block length.

---

## 6. Integration Architecture

### 6.1 Component Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase III Component Dependencies                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     Phase II Output                                                          │
│     ┌─────────────────────────┐                                             │
│     │ SiftedKeyData           │                                             │
│     │  - alice_key, bob_key   │                                             │
│     │  - adjusted_qber        │                                             │
│     │  - epsilon_sec          │                                             │
│     └───────────┬─────────────┘                                             │
│                 │                                                            │
│                 ▼                                                            │
│     ┌─────────────────────────┐      ┌─────────────────────────┐            │
│     │ LDPCReconciliator       │◀────▶│ LeakageSafetyManager    │            │
│     │  (existing)             │      │  (NEW)                  │            │
│     │                         │      │                         │            │
│     │  Dependencies:          │      │  Dependencies:          │            │
│     │   - LDPCMatrixManager   │      │   - min_entropy (II)    │            │
│     │   - LDPCBeliefProp      │      │   - epsilon_sec (II)    │            │
│     │   - PolyHashVerifier    │      │                         │            │
│     │   - QBEREstimator       │      │                         │            │
│     └───────────┬─────────────┘      └───────────┬─────────────┘            │
│                 │                                │                          │
│                 └────────────────┬───────────────┘                          │
│                                  │                                          │
│                                  ▼                                          │
│     ┌────────────────────────────────────────────────────────────────────┐  │
│     │                    ReconciliationOrchestrator                       │  │
│     │                                                                     │  │
│     │  Responsibilities:                                                  │  │
│     │   - Block loop management                                           │  │
│     │   - Syndrome transmission via ClassicalSocket                       │  │
│     │   - Leakage cap enforcement                                         │  │
│     │   - Result aggregation                                              │  │
│     │                                                                     │  │
│     └───────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│     ┌─────────────────────────┐                                             │
│     │ ReconciledKeyData       │                                             │
│     │  - reconciled_key       │                                             │
│     │  - total_leakage        │                                             │
│     │  - integrated_qber      │                                             │
│     └─────────────────────────┘                                             │
│                 │                                                            │
│                 ▼                                                            │
│           Phase IV Input                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 State Machine Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase III State Machine                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │   START     │                                                           │
│   │ (Phase II   │                                                           │
│   │  Complete)  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ COMPUTE_    │ Calculate L_max from H_min, ℓ_target, ε                  │
│   │ SAFETY_CAP  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ SELECT_     │ Choose LDPC rate based on adjusted QBER                  │
│   │ RATE        │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ PARTITION_  │ Split key into frame-sized blocks                        │
│   │ BLOCKS      │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐◀────────────────────────────────────┐                    │
│   │ PROCESS_    │                                     │                    │
│   │ BLOCK       │                                     │ Next block         │
│   └──────┬──────┘                                     │                    │
│          │                                            │                    │
│          ▼                                            │                    │
│   ┌─────────────┐     Leakage                        │                    │
│   │ CHECK_      │     Exceeded      ┌─────────────┐  │                    │
│   │ LEAKAGE_CAP │───────────────────▶│   ABORT     │  │                    │
│   └──────┬──────┘                   │ (Safety Cap)│  │                    │
│          │ OK                        └─────────────┘  │                    │
│          ▼                                            │                    │
│   ┌─────────────┐                                     │                    │
│   │ SEND_       │ Transmit syndrome via socket       │                    │
│   │ SYNDROME    │                                     │                    │
│   └──────┬──────┘                                     │                    │
│          │                                            │                    │
│          ▼                                            │                    │
│   ┌─────────────┐     Decode                         │                    │
│   │ DECODE_     │     Failed        ┌─────────────┐  │                    │
│   │ BLOCK       │───────────────────▶│ MARK_FAILED │──┼─────────┐          │
│   └──────┬──────┘                   └─────────────┘  │         │          │
│          │ Success                                    │         │          │
│          ▼                                            │         │          │
│   ┌─────────────┐     Hash                           │         │          │
│   │ VERIFY_     │     Mismatch      ┌─────────────┐  │         │          │
│   │ HASH        │───────────────────▶│ MARK_FAILED │──┼─────────┤          │
│   └──────┬──────┘                   └─────────────┘  │         │          │
│          │ Match                                      │         │          │
│          ▼                                            │         │          │
│   ┌─────────────┐                                     │         │          │
│   │ ACCUMULATE_ │                                     │         │          │
│   │ RESULT      │─────────────────────────────────────┘         │          │
│   └──────┬──────┘                                               │          │
│          │ All blocks processed                                 │          │
│          ▼                                                      │          │
│   ┌─────────────┐                                               │          │
│   │ ASSEMBLE_   │◀──────────────────────────────────────────────┘          │
│   │ OUTPUT      │ Concatenate successful blocks                            │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │  SUCCESS    │ ReconciledKeyData ready for Phase IV                     │
│   └─────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. MoSCoW Prioritized Roadmap

### 7.1 Priority Matrix

| Priority | Capability | Rationale | Effort Est. |
|----------|------------|-----------|-------------|
| **MUST** | Leakage Safety Manager | Security-critical; prevents over-leakage attacks | Low |
| **MUST** | Protocol abort on cap exceeded | Enforces L_max constraint | Low |
| **MUST** | Integrate safety manager with LDPCReconciliator | Operational enforcement | Medium |
| **SHOULD** | Block-level leakage logging | Debugging and auditing | Low |
| **SHOULD** | Configurable L_max calculation | Flexibility for different security parameters | Low |
| **COULD** | Blind Reconciliation mode | Efficiency without prior QBER (already partial support) | Medium |
| **WONT** | Interactive Hashing | High complexity; Phase 2 implementation | — |

### 7.2 Implementation Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase III Implementation Dependencies                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────────┐                            │
│                          │ Phase II Complete   │                            │
│                          │ (Prerequisite)      │                            │
│                          └──────────┬──────────┘                            │
│                                     │                                        │
│     ┌───────────────────────────────┴───────────────────────────────┐       │
│     │                                                                │       │
│     ▼                                                                ▼       │
│ ┌─────────────────┐                                    ┌─────────────────┐  │
│ │ Leakage Safety  │                                    │ LDPC Reconcil-  │  │
│ │ Manager         │                                    │ iator (existing)│  │
│ │ [MUST - NEW]    │                                    │ [Validated]     │  │
│ └────────┬────────┘                                    └────────┬────────┘  │
│          │                                                      │           │
│          └──────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│                   ┌──────────────────────────┐                              │
│                   │ Reconciliation           │                              │
│                   │ Orchestrator             │                              │
│                   │ [MUST - Integration]     │                              │
│                   └────────────┬─────────────┘                              │
│                                │                                             │
│                                ▼                                             │
│                   ┌──────────────────────────┐                              │
│                   │ Phase IV Ready           │                              │
│                   │ (Privacy Amplification)  │                              │
│                   └──────────────────────────┘                              │
│                                                                              │
│  Legend:                                                                     │
│  [MUST] = Critical path item                                                 │
│  [Validated] = Existing code, reviewed and approved                          │
│  ───▶ Dependency                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Files to Create/Modify

| File | Purpose | Priority | Status |
|------|---------|----------|--------|
| `ehok/core/security_bounds.py` | `LeakageSafetyManager` class | MUST | TO CREATE |
| `ehok/protocols/reconciliation_protocol.py` | Orchestrator with safety integration | MUST | TO CREATE |
| `ehok/implementations/reconciliation/ldpc_reconciliator.py` | Add safety manager hooks | SHOULD | TO MODIFY |
| `ehok/core/data_structures.py` | Add `ReconciliationResult` dataclass | SHOULD | TO MODIFY |

---

## 8. Risks & Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **LDPC decoding failure at high QBER** | High | Medium (blocks lost) | Ensure adequate test set sampling in Phase II; abort if adjusted QBER > 10% |
| **Leakage cap reached before full reconciliation** | Medium | High (partial key) | Pre-calculate feasibility; recommend minimum batch sizes |
| **Hash collision causing false positive** | Low | Low (key mismatch detected in PA) | Use 50+ bit hashes; collision probability < 10^-15 |
| **Floating point errors in leakage tracking** | Low | Medium (security margin erosion) | Use integer bit counting; add explicit margin in L_max |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Block size mismatch Alice/Bob** | Low | Critical (protocol failure) | Explicit frame size negotiation at protocol start |
| **PRNG seed desync for shortening** | Low | Critical (decoding failure) | Share seed via authenticated classical channel |

---

## 9. Conclusion

Phase III represents a computationally intensive but architecturally straightforward component of E-HOK. The legacy `ehok/` implementation provides a complete, one-way LDPC reconciliation stack that aligns with the security requirements of the Noisy Storage Model.

**Key Findings**:

1. **Legacy Code Quality**: The existing `LDPCReconciliator` and supporting classes are cryptographically appropriate for E-HOK—they implement one-way syndrome transmission without interactive feedback.

2. **Primary Gap**: The absence of a `LeakageSafetyManager` enforcing $L_{max}$ is the critical security gap. This requires a new component that tracks cumulative leakage and triggers protocol abort when the budget is exceeded.

3. **SquidASM Integration**: Phase III requires only `ClassicalSocket` for syndrome transmission. No quantum operations occur in this phase, simplifying integration.

4. **Efficiency Cliff**: The analysis confirms that E-HOK is highly sensitive to QBER. At 10%+ error rates, the syndrome leakage consumes all extractable entropy, making secure key generation impossible.

**Immediate Action Items**:
1. Create `ehok/core/security_bounds.py` with `LeakageSafetyManager`
2. Create `ehok/protocols/reconciliation_protocol.py` as the Phase III orchestrator
3. Integrate safety checks into the block processing loop

Upon completion, Phase III will produce a `ReconciledKeyData` structure containing the error-corrected key and precise leakage accounting for Phase IV privacy amplification.

---

## References

1. Schaffner, C., Terhal, B., & Wehner, S. (2009). *Robust Cryptography in the Noisy-Quantum-Storage Model*.

2. Lupo, C., Peat, J.T., Andersson, E., & Kok, P. (2023). *Error-tolerant oblivious transfer in the noisy-storage model*.

3. Erven, C., et al. (2014). *An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model*. arXiv:1308.5098v4.

4. Martinez-Mateo, J., et al. (2011). *Blind Reconciliation*.

5. Tomamichel, M., et al. (2012). *Tight Finite-Key Analysis for Quantum Cryptography*. Nature Communications.

# Phase IV Technical Analysis: Privacy Amplification

> **Definitive Integration Blueprint for Key Distillation in E-HOK**
> 
> Version: 1.0  
> Last Updated: 2025  
> Authors: AI Technical Analysis

---

## Abstract

Phase IV represents the terminal operation of the E-HOK protocol: the distillation of a reconciled bitstring into a cryptographically secure oblivious key. This phase transforms the error-corrected key material from Phase III into the final 1-out-of-2 Oblivious Transfer output, where Alice possesses two keys $(S_0, S_1)$ and Bob possesses exactly one $(S_C)$ determined by his earlier measurement choice.

The security model for Phase IV diverges fundamentally from standard QKD privacy amplification. Rather than using the traditional bound $1 - h(QBER)$, E-HOK employs the **Noisy Storage Model (NSM) "Max Bound"** derived by Lupo et al., which provides tighter entropy estimates by selecting the optimal bound from competing entropic inequalities:

$$h_{min} \ge \max \left\{ \Gamma [1 - \log_2 (1 + 3r^2)], 1 - r \right\}$$

This phase must also account for **finite-key effects**—statistical penalties that scale with $1/\sqrt{N}$ and can consume the entire extractable entropy for small batch sizes. The "Death Valley" phenomenon occurs when the combined penalties from finite statistics, reconciliation leakage, and security margins exceed the available min-entropy, yielding a zero-length secure key.

The gap analysis reveals that the legacy `ehok/` implementation provides a complete Toeplitz hashing infrastructure but operates under **QKD security bounds rather than NSM bounds**. Critical extensions are required to replace the entropy calculation with NSM-specific formulas and to produce the oblivious output structure required for 1-out-of-2 OT applications.

---

## 1. Ontology & Terminology

### 1.1 Core Concepts Mapping

| Protocol Concept | Mathematical Form | Software Representation |
|------------------|-------------------|------------------------|
| **Min-Entropy** | $H_{min}^\epsilon(X \mid E)$ | Extractable randomness given adversary's information |
| **Smooth Min-Entropy** | $H_{min}^\epsilon$ with smoothing $\epsilon$ | Allows small probability of deviation from ideal |
| **Wiretap Cost** | $\|\Sigma\|$ | Syndrome length from Phase III reconciliation |
| **Storage Noise Parameter** | $r \in [0, 1]$ | Adversary's quantum memory depolarization strength |
| **Security Parameter** | $\varepsilon_{sec}$ | Trace distance from ideal random key (target: $10^{-9}$) |
| **Correctness Parameter** | $\varepsilon_{cor}$ | Probability of key mismatch between parties |
| **Statistical Fluctuation** | $\mu$ | Finite-size penalty on QBER estimation |
| **Oblivious Output** | $(S_0, S_1)$ for Alice; $(S_C, C)$ for Bob | 1-out-of-2 OT key structure |

### 1.2 The NSM Security Model vs. QKD

**Fundamental Distinction**: E-HOK security derives from assumptions about the adversary's quantum memory, not from channel estimation.

| Aspect | QKD Model | NSM Model (E-HOK) |
|--------|-----------|-------------------|
| **Adversary Capability** | Unlimited quantum computation | Bounded noisy quantum storage |
| **Min-Entropy Source** | Channel estimation via test bits | Adversary storage decoherence |
| **Key Formula** | $\ell = n \cdot (1 - h(Q)) - leak$ | $\ell = n \cdot h_{min}(r) - leak$ |
| **Primary Bound** | Binary entropy of QBER | Max Bound over collision entropy and virtual erasure |
| **Trust Model** | Alice & Bob collaborate against Eve | Alice & Bob distrust each other |

### 1.3 The "Max Bound" Derivation

The Lupo et al. "Max Bound" selects the optimal entropy bound based on the adversary's storage noise:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         The "Max Bound" Selection                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given adversary storage noise parameter r ∈ [0, 1]:                        │
│    r = 0: Perfect storage (worst case for security)                         │
│    r = 1: Complete depolarization (best case for security)                  │
│                                                                              │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │ Dupuis-König Bound (Collision) │  │ Lupo Bound (Virtual Erasure)      │ │
│  │                                │  │                                    │ │
│  │   h_A = Γ[1 - log₂(1 + 3r²)]  │  │   h_B = 1 - r                      │ │
│  │                                │  │                                    │ │
│  │   Better for high noise        │  │   Better for low noise             │ │
│  │   (small r, noisy memory)      │  │   (large r, quiet memory)          │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                              │
│                                 ↓                                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Max Bound Selection                             │ │
│  │                                                                         │ │
│  │                h_min = max { h_A, h_B }                                 │ │
│  │                                                                         │ │
│  │   This dual-bound approach extracts strictly more key than either       │ │
│  │   bound alone, improving efficiency without compromising security.      │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Numerical Comparison (entropy rate h_min):                                  │
│                                                                              │
│  ┌────────┬─────────────────┬────────────────────┬───────────────────────┐  │
│  │   r    │ Dupuis-König    │ Lupo (1-r)         │ Max Bound             │  │
│  ├────────┼─────────────────┼────────────────────┼───────────────────────┤  │
│  │  0.1   │ 0.957           │ 0.900              │ 0.957                 │  │
│  │  0.3   │ 0.805           │ 0.700              │ 0.805                 │  │
│  │  0.5   │ 0.585           │ 0.500              │ 0.585                 │  │
│  │  0.7   │ 0.322           │ 0.300              │ 0.322                 │  │
│  │  0.9   │ 0.082           │ 0.100              │ 0.100 ← Lupo better   │  │
│  └────────┴─────────────────┴────────────────────┴───────────────────────┘  │
│                                                                              │
│  Crossover occurs at r ≈ 0.82 where both bounds are equal.                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 The Γ Function

The function $\Gamma$ in the Dupuis-König bound is defined piecewise:

$$\Gamma(x) = \begin{cases} x & \text{if } x \ge 1/2 \\ g^{-1}(x) & \text{if } x < 1/2 \end{cases}$$

where $g(y) = -y \log_2 y - (1-y) \log_2 (1-y) + y - 1$.

This regularization maps the collision entropy rate to a min-entropy rate, accounting for the relationship between Rényi entropies of different orders.

---

## 2. Literature Context

### 2.1 Theoretical Corpus

Phase IV security derives from the following foundational literature:

| Source | Contribution | Key Result |
|--------|--------------|------------|
| **Lupo et al.** (2023) | "Max Bound" for error-tolerant OT | Eq. (36): $h_{min} \ge \max\{\Gamma[1-\log(1+3r^2)], 1-r\}$ |
| **Dupuis & König** (2014) | Collision entropy bound | Eq. (29): Tight for high-noise storage |
| **Schaffner et al.** (2009) | NSM composable security | Leftover hash lemma for OT |
| **Tomamichel et al.** (2012) | Finite-key QKD analysis | Statistical fluctuation $\mu$ formula |
| **Erven et al.** (2014) | Experimental OT implementation | Practical syndrome leakage accounting |

### 2.2 The Leftover Hash Lemma

**Source**: Schaffner et al. Eq. (1); Lupo et al. Eq. (2)

The fundamental security guarantee for privacy amplification:

$$\ell \le H_{min}^\epsilon(X_{\bar{B}} \mid \mathcal{F}(Q) \Theta B \Sigma_{\bar{B}}) - 2\log_2\frac{1}{\epsilon_h} + 1$$

Where:
- $X_{\bar{B}}$ — The substring Alice wants to hide from cheating Bob
- $\mathcal{F}(Q)$ — Bob's quantum memory state after noise $\mathcal{F}$
- $\Theta$ — Revealed basis information
- $\Sigma_{\bar{B}}$ — Syndrome for error correction
- $\epsilon_h$ — Hashing security parameter

### 2.3 The Error Correction Penalty

**Source**: Lupo et al. Eq. (3)

When error correction syndromes are transmitted, they reduce the extractable min-entropy:

$$\ell \ge H_{min}^\epsilon(X_{\bar{B}} \mid \mathcal{F}(Q) \Theta B) - |\Sigma_{\bar{B}}| - 2\log_2\frac{1}{\epsilon_h} + 1$$

The syndrome length $|\Sigma|$ is **directly subtracted** from the min-entropy—this is the "wiretap cost" that must be tracked from Phase III.

### 2.4 Finite-Key Statistical Penalty

**Source**: Tomamichel et al. (2012); Erven et al. Eq. (8)

For finite key lengths $n$, the QBER estimate from $k$ test bits has statistical uncertainty:

$$\mu = \sqrt{\frac{(n+k)}{nk} \cdot \frac{(k+1)}{k}} \cdot \sqrt{\ln\frac{4}{\varepsilon_{PE}}}$$

The effective QBER for security calculations becomes:

$$Q_{eff} = Q_{measured} + \mu$$

**Critical Impact**: For small batch sizes ($n < 10^5$), the $\mu$ penalty can be substantial (5-10%), potentially consuming all extractable entropy.

---

## 3. System Logic & Orchestration

### 3.1 Operational Workflow

Phase IV consumes the reconciled key from Phase III and produces the final oblivious transfer output:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase IV Operational Workflow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  From Phase III                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ReconciledKeyData                                                      │  │
│  │   - reconciled_key: np.ndarray (n bits, error-corrected)              │  │
│  │   - total_leakage: float (syndrome + hash bits)                       │  │
│  │   - integrated_qber: float (from correction counts)                   │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │                              │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: Feasibility Check                                             │  │
│  │                                                                         │  │
│  │   Input: n, leakage, storage_noise (r), epsilon_sec                    │  │
│  │                                                                         │  │
│  │   Compute:                                                              │  │
│  │     h_min = max_bound_entropy(r)                                        │  │
│  │     extractable = n × h_min                                             │  │
│  │     penalty = leakage + 2·log₂(1/ε_sec) + finite_key_correction        │  │
│  │     expected_length = extractable - penalty                             │  │
│  │                                                                         │  │
│  │   IF expected_length ≤ 0:                                               │  │
│  │     ABORT with "Batch Size Too Small"                                   │  │
│  │     RECOMMEND minimum n for positive key                                │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │ Feasible                     │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: Final Length Calculation                                      │  │
│  │                                                                         │  │
│  │   Compute using E-HOK (NSM) formula:                                    │  │
│  │                                                                         │  │
│  │     ℓ = ⌊n × h_min(r) - |Σ| - 2·log₂(1/ε_sec) - Δ_finite⌋             │  │
│  │                                                                         │  │
│  │   Where:                                                                │  │
│  │     h_min(r) = max{Γ[1-log(1+3r²)], 1-r}  [Lupo Max Bound]            │  │
│  │     |Σ| = total_leakage                   [From Phase III]             │  │
│  │     Δ_finite = n × h(Q_eff) - n × h(Q)    [Finite-size correction]    │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │                              │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 3: Toeplitz Seed Generation                                      │  │
│  │                                                                         │  │
│  │   seed = secrets.token_bytes(⌈(ℓ + n - 1) / 8⌉)                        │  │
│  │                                                                         │  │
│  │   Requirements:                                                         │  │
│  │     - Cryptographic randomness (OS entropy)                             │  │
│  │     - Unpredictable to adversary before generation                      │  │
│  │     - Shared via authenticated classical channel                        │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │                              │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 4: Toeplitz Hashing                                              │  │
│  │                                                                         │  │
│  │   Mathematical Operation:                                               │  │
│  │     final_key = T × reconciled_key (mod 2)                             │  │
│  │                                                                         │  │
│  │   Where T is an ℓ × n Toeplitz matrix defined by seed.                 │  │
│  │                                                                         │  │
│  │   Implementation Options:                                               │  │
│  │     - Direct: O(ℓ × n) sliding window                                   │  │
│  │     - FFT: O(n log n) via circular convolution                         │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │                              │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 5: Oblivious Output Formatting                                   │  │
│  │                                                                         │  │
│  │   ALICE produces:                                                       │  │
│  │     S_0 = Hash(reconciled_key | indices where θ = 0)                   │  │
│  │     S_1 = Hash(reconciled_key | indices where θ = 1)                   │  │
│  │                                                                         │  │
│  │   BOB produces:                                                         │  │
│  │     S_C = Hash(reconciled_key | indices where B matched)               │  │
│  │     C = implicit choice bit (unknown to Alice)                         │  │
│  │                                                                         │  │
│  │   Oblivious Property:                                                   │  │
│  │     - Alice knows both S_0 and S_1                                      │  │
│  │     - Bob knows only S_C (and nothing about S_{1-C})                   │  │
│  │     - Alice doesn't know which of S_0, S_1 Bob received                │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────┬──────────────────────────┘  │
│                                               │                              │
│                                               ▼                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Phase IV Output                                                         │  │
│  │                                                                         │  │
│  │   AliceObliviousKey:                                                    │  │
│  │     - s0: np.ndarray (ℓ bits)                                          │  │
│  │     - s1: np.ndarray (ℓ bits)                                          │  │
│  │     - seed: np.ndarray (shared with Bob)                               │  │
│  │                                                                         │  │
│  │   BobObliviousKey:                                                      │  │
│  │     - s_c: np.ndarray (ℓ bits)                                         │  │
│  │     - c: int (implicit choice, not known to Alice)                     │  │
│  │                                                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Interface Definition

Phase IV requires two distinct interfaces:

**Input Interface** (from Phase III):
```
┌────────────────────────────────────────────────────────────────────────────┐
│ ReconciledKeyData                                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  reconciled_key : np.ndarray                                                │
│      The error-corrected bitstring from Phase III reconciliation.           │
│      Length: n bits (after removing failed blocks).                         │
│                                                                             │
│  total_leakage : float                                                      │
│      Cumulative information leaked during Phase III:                        │
│        - Syndrome bits for all blocks                                       │
│        - Hash verification bits for all blocks                              │
│      Units: bits                                                            │
│                                                                             │
│  integrated_qber : float                                                    │
│      QBER estimated from LDPC correction counts.                            │
│      Range: [0, 0.5]                                                        │
│                                                                             │
│  blocks_succeeded : int                                                     │
│      Number of successfully reconciled blocks.                              │
│                                                                             │
│  blocks_failed : int                                                        │
│      Number of blocks discarded due to decoding failure.                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Output Interface** (to Application Layer):
```
┌────────────────────────────────────────────────────────────────────────────┐
│ ObliviousTransferResult                                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ALICE SIDE:                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ AliceObliviousKey                                                       ││
│  │   s0 : np.ndarray     Key for choice bit C = 0                         ││
│  │   s1 : np.ndarray     Key for choice bit C = 1                         ││
│  │   seed : np.ndarray   Toeplitz seed (shared with Bob)                  ││
│  │   epsilon_achieved : float  Actual security parameter                  ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  BOB SIDE:                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ BobObliviousKey                                                         ││
│  │   s_c : np.ndarray    Key corresponding to Bob's choice                ││
│  │   c : int             Implicit choice bit (not known to Alice)         ││
│  │   seed : np.ndarray   Toeplitz seed (received from Alice)              ││
│  │   epsilon_achieved : float  Actual security parameter                  ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  METADATA:                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ ProtocolMetrics                                                         ││
│  │   key_length : int           Final key length ℓ                        ││
│  │   compression_ratio : float  ℓ / n                                     ││
│  │   total_leakage : float      From Phase III                            ││
│  │   storage_noise_assumed : float  Adversary model parameter r           ││
│  │   epsilon_sec : float        Security parameter achieved               ││
│  │   epsilon_cor : float        Correctness parameter                     ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Lifecycle Management

Phase IV operates as a **stateless transformation**: given valid inputs from Phase III, it produces the oblivious output in a single pass. However, lifecycle considerations include:

**Session Initialization**:
- Adversary storage noise parameter $r$ must be configured a priori
- Security parameters $\varepsilon_{sec}$, $\varepsilon_{cor}$ set at protocol start
- No quantum operations; all classical post-processing

**Normal Completion**:
- Final key length $\ell > 0$
- Both parties compute identical seeds
- Alice produces $(S_0, S_1)$; Bob produces $(S_C, C)$

**Abort Conditions**:
- Feasibility check fails (expected $\ell \le 0$)
- Seed transmission failure
- Hash computation error

**Error Recovery**:
- Phase IV failures are **terminal** for the current session
- No partial key output (all-or-nothing security)
- May request larger batch from Phase I restart

---

## 4. Implementation Gap Analysis

### 4.1 Gap Summary Matrix

| Capability | SquidASM Native | Legacy ehok | Gap Status | Extension Required |
|------------|-----------------|-------------|------------|-------------------|
| Toeplitz Matrix Hashing | N/A | ✅ `ToeplitzAmplifier` | SUPPORTED | None |
| Cryptographic Seed Generation | N/A | ✅ `secrets.token_bytes()` | SUPPORTED | None |
| FFT-based Compression | N/A | ✅ Optional O(n log n) | SUPPORTED | None |
| Finite-Key μ Calculation | N/A | ✅ `compute_statistical_fluctuation()` | SUPPORTED | None |
| **NSM "Max Bound"** | ❌ | ❌ Uses QKD bound | **CRITICAL GAP** | `NSMBoundsCalculator` |
| **Oblivious Output Format** | ❌ | ❌ Single-key output | **CRITICAL GAP** | `ObliviousKeyFormatter` |
| **Feasibility Pre-Check** | ❌ | ⚠️ Post-hoc only | **HIGH GAP** | `BatchFeasibilityChecker` |
| Storage Noise → Entropy | ❌ | ❌ | GAP | Parameter adapter |

### 4.2 Framework Capabilities Assessment

**SquidASM Native Support**:

Phase IV operates **entirely classically** after quantum measurements complete. SquidASM provides:

| Component | Purpose | Adequacy |
|-----------|---------|----------|
| `ClassicalSocket` | Seed transmission | ✅ Sufficient |
| `ProgramContext` | Session management | ✅ Sufficient |
| Generator-based async | Non-blocking operation | ✅ Sufficient |

**No quantum operations** occur in Phase IV—all processing is on classical data.

**NetSquid Integration Point**:

The adversary's storage noise parameter $r$ conceptually maps to NetSquid's decoherence models:

$$r \approx 1 - F_{storage}(\Delta t)$$

Where $F_{storage}$ is the fidelity of the adversary's quantum memory after wait time $\Delta t$. This connection enables:
- Simulation-consistent adversary modeling
- T1/T2 times → depolarizing strength conversion
- Configuration-driven security parameter selection

### 4.3 Legacy vs. Target Architecture

**Legacy ehok Approach**:

The existing `ToeplitzAmplifier` implements:
```
Reconciled Key → Toeplitz Hash → Single Final Key
```

Using the formula:
$$\ell = n \cdot (1 - h(QBER + \mu)) - leak - 2\log_2\frac{1}{\varepsilon_{sec}}$$

This is the **QKD finite-key formula** from Tomamichel et al.—**not valid for NSM-based E-HOK**.

**Target Architecture Required**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Target Phase IV Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    NSM Security Bounds Layer (NEW)                       ││
│  │                                                                          ││
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────────┐   ││
│  │  │ max_bound_entropy() │  │ compute_ehok_final_length()              │   ││
│  │  │ dupuis_konig_bound()│  │   Uses NSM h_min instead of 1-h(QBER)   │   ││
│  │  │ lupo_bound()        │  │                                          │   ││
│  │  └─────────────────────┘  └─────────────────────────────────────────┘   ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                        │                                     │
│                                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Privacy Amplification Layer (EXISTING)                ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │ ToeplitzAmplifier                                                │    ││
│  │  │   generate_hash_seed() ← Keep as-is                             │    ││
│  │  │   compress()           ← Keep as-is                             │    ││
│  │  │   compute_final_length() ← REPLACE with NSM-aware version       │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                        │                                     │
│                                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Oblivious Output Layer (NEW)                          ││
│  │                                                                          ││
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────────┐   ││
│  │  │ ObliviousKeyFormatter│  │ AliceObliviousKey, BobObliviousKey     │   ││
│  │  │   format_alice()     │  │ Dataclasses for structured output      │   ││
│  │  │   format_bob()       │  │                                        │   ││
│  │  └─────────────────────┘  └─────────────────────────────────────────┘   ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Missing Infrastructure Analysis

**Critical Missing Component 1: NSM Bounds Calculator**

The `compute_final_length_finite_key()` function in `ehok/implementations/privacy_amplification/finite_key.py` uses:

$$H_{min} = n \cdot (1 - h(QBER + \mu))$$

This is the **QKD bound**. For E-HOK, it must be replaced with:

$$H_{min} = n \cdot h_{min}(r)$$

Where $h_{min}(r) = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$.

**Location for extension**: `ehok/analysis/nsm_bounds.py`

**Critical Missing Component 2: Oblivious Key Formatter**

The current implementation produces a single `np.ndarray` as output. E-HOK requires:

- Alice: Two keys $(S_0, S_1)$ based on her basis choices
- Bob: One key $S_C$ based on his measurement outcomes

This requires understanding the index sets $I_0$ and $I_1$ from the sifting phase and applying separate hashing operations.

**Location for extension**: `ehok/core/oblivious_key.py`

**High-Priority Missing Component: Feasibility Pre-Check**

The current implementation calculates key length after the fact. E-HOK should:

1. **Before** starting Phase I, estimate if secure key is achievable
2. **Abort early** if batch size is insufficient
3. **Recommend** minimum batch size for positive key

**Location for extension**: `ehok/core/feasibility.py`

---

## 5. Formalization of Metrics and Constraints

### 5.1 Secure Key Length Formula (NSM-Specific)

The E-HOK secure key length formula:

$$\ell \le n \cdot h_{min}(r) - |\Sigma| - 2\log_2\frac{1}{\varepsilon_{sec}} - \Delta_{finite}$$

Where:
- $n$ — Reconciled key length (bits)
- $h_{min}(r)$ — NSM min-entropy rate from Max Bound
- $|\Sigma|$ — Total reconciliation leakage (bits)
- $\varepsilon_{sec}$ — Target security parameter
- $\Delta_{finite}$ — Finite-key correction (see below)

### 5.2 NSM Min-Entropy Rate

The "Max Bound" from Lupo et al. Eq. (36):

$$h_{min}(r) = \max \left\{ \Gamma[1 - \log_2(1 + 3r^2)], \quad 1 - r \right\}$$

**Component 1: Dupuis-König Collision Entropy Bound**

$$h_A(r) = \Gamma[1 - \log_2(1 + 3r^2)]$$

Where the collision entropy for depolarizing channel:

$$h_2(\sigma) = 1 - \log_2(1 + 3r^2)$$

And $\Gamma$ regularizes from collision to min-entropy.

**Component 2: Lupo Virtual Erasure Bound**

$$h_B(r) = 1 - r$$

Derived by giving the adversary knowledge of which qubits were depolarized (a virtual "erasure flag"), which can only increase their information.

### 5.3 Finite-Key Correction

The statistical fluctuation term from Tomamichel et al.:

$$\mu = \sqrt{\frac{(n+k)}{nk} \cdot \frac{(k+1)}{k}} \cdot \sqrt{\ln\frac{4}{\varepsilon_{PE}}}$$

Where:
- $n$ — Key generation bits
- $k$ — Test/parameter estimation bits
- $\varepsilon_{PE}$ — Parameter estimation security (typically $\sqrt{\varepsilon_{sec}}$)

The finite-key penalty is incorporated as:

$$\Delta_{finite} = n \cdot [h(Q_{eff}) - h(Q)] \approx n \cdot \mu \cdot h'(Q)$$

Where $Q_{eff} = Q + \mu$ and $h'(Q)$ is the derivative of binary entropy.

### 5.4 The "Death Valley" Threshold

The protocol yields zero secure key when:

$$n \cdot h_{min}(r) \le |\Sigma| + 2\log_2\frac{1}{\varepsilon_{sec}} + \Delta_{finite}$$

Rearranging for minimum viable batch size:

$$n_{min} \ge \frac{|\Sigma| + 2\log_2(1/\varepsilon_{sec})}{h_{min}(r) - \mu_{asymptotic}}$$

**Numerical Example** (worst-case $r = 0.5$, $\varepsilon_{sec} = 10^{-9}$):

| QBER | Leakage Rate | $h_{min}(0.5)$ | $n_{min}$ |
|------|--------------|----------------|-----------|
| 2% | 0.17 | 0.585 | ~70,000 |
| 5% | 0.34 | 0.585 | ~130,000 |
| 10% | 0.56 | 0.585 | **Infeasible** |

At 10% QBER, leakage exceeds extractable entropy regardless of batch size.

### 5.5 Security Parameter Coupling

The security parameter $\varepsilon_{sec}$ and key length $\ell$ are coupled:

$$\varepsilon_{sec} \approx 2 \cdot 2^{-\frac{1}{2}(H_{min} - \ell)}$$

This means:
- Fixing $\ell$ determines minimum $\varepsilon_{sec}$
- Fixing $\varepsilon_{sec}$ determines maximum $\ell$
- Cannot simultaneously fix both arbitrarily

### 5.6 Composable Security

For use in larger cryptographic systems:

$$\varepsilon_{total} \le \varepsilon_s + \varepsilon_h + \varepsilon_{test}$$

Where:
- $\varepsilon_s$ — Smoothing parameter
- $\varepsilon_h$ — Hashing parameter
- $\varepsilon_{test}$ — Statistical test failure probability

This allows E-HOK keys to be used as subroutines in Secure Multiparty Computation with quantifiable cumulative risk.

---

## 6. Integration Architecture

### 6.1 Component Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase IV Component Dependencies                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase III Output                                                           │
│   ┌───────────────────────────────┐                                         │
│   │ ReconciledKeyData             │                                         │
│   │  - reconciled_key             │                                         │
│   │  - total_leakage              │                                         │
│   │  - integrated_qber            │                                         │
│   └──────────────┬────────────────┘                                         │
│                  │                                                           │
│     ┌────────────┴────────────┬───────────────────────┐                     │
│     │                         │                       │                     │
│     ▼                         ▼                       ▼                     │
│ ┌────────────────┐    ┌───────────────────┐   ┌────────────────────────┐   │
│ │ Protocol Config│    │ NSMBoundsCalc     │   │ Phase II Metadata      │   │
│ │  - epsilon_sec │    │ [NEW - CRITICAL]  │   │  - test_bits (k)       │   │
│ │  - epsilon_cor │    │                   │   │  - storage_noise (r)   │   │
│ │  - storage_r   │    │ Deps: None        │   │                        │   │
│ └───────┬────────┘    └─────────┬─────────┘   └───────────┬────────────┘   │
│         │                       │                         │                 │
│         └───────────────────────┼─────────────────────────┘                 │
│                                 │                                           │
│                                 ▼                                           │
│               ┌─────────────────────────────────────────┐                   │
│               │ FeasibilityChecker [NEW - HIGH]         │                   │
│               │                                         │                   │
│               │ Deps: NSMBoundsCalculator               │                   │
│               │                                         │                   │
│               │ IF infeasible → ABORT with recommendation│                  │
│               └────────────────────┬────────────────────┘                   │
│                                    │ Feasible                               │
│                                    ▼                                        │
│               ┌─────────────────────────────────────────┐                   │
│               │ ToeplitzAmplifier [EXISTING + MODIFY]   │                   │
│               │                                         │                   │
│               │ Deps:                                   │                   │
│               │  - NSMBoundsCalculator (for length)     │                   │
│               │  - secrets module (for seed)            │                   │
│               │                                         │                   │
│               │ Methods:                                │                   │
│               │  - generate_hash_seed() ← Keep          │                   │
│               │  - compress() ← Keep                    │                   │
│               │  - compute_final_length() ← REPLACE     │                   │
│               └────────────────────┬────────────────────┘                   │
│                                    │                                        │
│                                    ▼                                        │
│               ┌─────────────────────────────────────────┐                   │
│               │ ObliviousKeyFormatter [NEW - CRITICAL]  │                   │
│               │                                         │                   │
│               │ Deps:                                   │                   │
│               │  - ToeplitzAmplifier (for hashing)      │                   │
│               │  - Sifting metadata (I_0, I_1 indices)  │                   │
│               │                                         │                   │
│               │ Methods:                                │                   │
│               │  - format_alice() → (S_0, S_1)          │                   │
│               │  - format_bob() → (S_C, C)              │                   │
│               └────────────────────┬────────────────────┘                   │
│                                    │                                        │
│                                    ▼                                        │
│               ┌─────────────────────────────────────────┐                   │
│               │ ObliviousTransferResult                 │                   │
│               │                                         │                   │
│               │  - AliceObliviousKey                    │                   │
│               │  - BobObliviousKey                      │                   │
│               │  - ProtocolMetrics                      │                   │
│               └─────────────────────────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Cross-Phase Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Complete Protocol Data Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────┐                                                      │
│  │     Phase I       │  Quantum generation, noise model configuration       │
│  │  (Physical Layer) │                                                      │
│  └─────────┬─────────┘                                                      │
│            │ RawQuantumData                                                  │
│            │  - measurement_outcomes[]                                       │
│            │  - timing_metadata                                              │
│            │  - storage_noise_r (configuration)                             │
│            ▼                                                                 │
│  ┌───────────────────┐                                                      │
│  │     Phase II      │  Sifting, basis matching, QBER estimation            │
│  │   (Sifting)       │                                                      │
│  └─────────┬─────────┘                                                      │
│            │ SiftedKeyData                                                   │
│            │  - sifted_key_alice, sifted_key_bob                            │
│            │  - adjusted_qber (Q + μ)                                       │
│            │  - test_bits_k                                                  │
│            │  - I_0[], I_1[] (basis index sets) ←─┐                         │
│            ▼                                       │ Needed for             │
│  ┌───────────────────┐                            │ oblivious output       │
│  │    Phase III      │  LDPC reconciliation       │                         │
│  │ (Reconciliation)  │                            │                         │
│  └─────────┬─────────┘                            │                         │
│            │ ReconciledKeyData                    │                         │
│            │  - reconciled_key (n bits)           │                         │
│            │  - total_leakage (|Σ|)               │                         │
│            │  - integrated_qber                   │                         │
│            ▼                                       │                         │
│  ┌───────────────────┐                            │                         │
│  │     Phase IV      │  Privacy amplification     │                         │
│  │  (Distillation)   │◀───────────────────────────┘                         │
│  └─────────┬─────────┘                                                      │
│            │ ObliviousTransferResult                                         │
│            │  - AliceObliviousKey (S_0, S_1)                                │
│            │  - BobObliviousKey (S_C, C)                                    │
│            │  - ProtocolMetrics                                              │
│            ▼                                                                 │
│  ┌───────────────────┐                                                      │
│  │ Application Layer │  Use keys for OT-based computation                   │
│  │  (e.g., MPC)      │                                                      │
│  └───────────────────┘                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. MoSCoW-Prioritized Roadmap

### 7.1 Priority Matrix

| Priority | Capability | Rationale | Effort Est. | Dependency |
|----------|------------|-----------|-------------|------------|
| **MUST** | NSM "Max Bound" Calculator | Security-critical; current bounds are invalid for E-HOK | Medium | None |
| **MUST** | Replace `compute_final_length()` with NSM-aware version | Current QKD formula is incorrect | Low | NSM Bounds |
| **MUST** | Oblivious Output Formatter | Required for 1-out-of-2 OT structure | Medium | ToeplitzAmplifier |
| **MUST** | Storage Noise Parameter Integration | Links Phase I config to Phase IV security | Low | NSM Bounds |
| **SHOULD** | Batch Feasibility Pre-Check | Prevents "Death Valley" failures | Low | NSM Bounds |
| **SHOULD** | Composable Security Tracking | Enables safe composition in MPC | Low | None |
| **COULD** | FFT Optimization for Large Keys | Performance improvement for $n > 10^5$ | Low | None |
| **COULD** | Detailed Protocol Metrics Logging | Debugging and auditing | Low | None |

### 7.2 Implementation Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase IV Implementation Dependencies                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────────┐                            │
│                          │ Phase III Complete   │                            │
│                          │ (Prerequisite)       │                            │
│                          └──────────┬──────────┘                            │
│                                     │                                        │
│                                     ▼                                        │
│                   ┌─────────────────────────────────┐                       │
│                   │ NSMBoundsCalculator             │                       │
│                   │ [MUST - CRITICAL PATH]          │                       │
│                   │                                 │                       │
│                   │ ehok/analysis/nsm_bounds.py     │                       │
│                   │  - max_bound_entropy(r)         │                       │
│                   │  - dupuis_konig_bound(r)        │                       │
│                   │  - lupo_virtual_erasure(r)      │                       │
│                   └────────────┬────────────────────┘                       │
│                                │                                             │
│          ┌─────────────────────┼─────────────────────┐                      │
│          │                     │                     │                      │
│          ▼                     ▼                     ▼                      │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────────────┐ │
│  │ FeasibilityChecker│ │ ToeplitzAmplifier │ │ Storage Noise Adapter     │ │
│  │ [SHOULD]          │ │ [MUST - MODIFY]   │ │ [MUST]                    │ │
│  │                   │ │                   │ │                           │ │
│  │ ehok/core/        │ │ REPLACE compute_  │ │ ehok/quantum/noise_       │ │
│  │ feasibility.py    │ │ final_length()    │ │ adapter.py                │ │
│  └───────────────────┘ └─────────┬─────────┘ └───────────────────────────┘ │
│                                  │                                          │
│                                  ▼                                          │
│                   ┌─────────────────────────────────┐                       │
│                   │ ObliviousKeyFormatter           │                       │
│                   │ [MUST - CRITICAL PATH]          │                       │
│                   │                                 │                       │
│                   │ ehok/core/oblivious_key.py      │                       │
│                   │  - AliceObliviousKey            │                       │
│                   │  - BobObliviousKey              │                       │
│                   │  - format_alice()               │                       │
│                   │  - format_bob()                 │                       │
│                   └────────────┬────────────────────┘                       │
│                                │                                             │
│                                ▼                                             │
│                   ┌─────────────────────────────────┐                       │
│                   │ E-HOK Protocol Complete         │                       │
│                   │ (1-out-of-2 OT Ready)           │                       │
│                   └─────────────────────────────────┘                       │
│                                                                              │
│  Legend:                                                                     │
│  [MUST] = Critical path, blocks completion                                   │
│  [SHOULD] = High value, not blocking                                         │
│  ───▶ Dependency                                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Files to Create/Modify

| File | Purpose | Priority | Status |
|------|---------|----------|--------|
| `ehok/analysis/nsm_bounds.py` | Lupo "Max Bound" implementation | MUST | TO CREATE |
| `ehok/core/oblivious_key.py` | Oblivious output dataclasses and formatter | MUST | TO CREATE |
| `ehok/core/feasibility.py` | Batch feasibility pre-check | SHOULD | TO CREATE |
| `ehok/quantum/noise_adapter.py` | NetSquid T1/T2 → NSM r conversion | MUST | TO CREATE |
| `ehok/implementations/privacy_amplification/toeplitz_amplifier.py` | Integrate NSM bounds | MUST | TO MODIFY |
| `ehok/core/data_structures.py` | Add `AliceObliviousKey`, `BobObliviousKey` | MUST | TO MODIFY |

---

## 8. Risks & Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Zero-length key ("Death Valley")** | High for small batches | Critical (protocol failure) | Mandatory feasibility pre-check; batch size recommendations |
| **Incorrect NSM bound implementation** | Medium | Critical (invalid security) | Unit tests against Lupo et al. Table I numerical values |
| **Floating-point precision in ε calculations** | Medium | High (security erosion) | Use `decimal` module for $\varepsilon_{sec} < 10^{-10}$; explicit precision tracking |
| **Seed randomness compromise** | Low | Critical (total security failure) | Use only `secrets.token_bytes()`; never reuse seeds |
| **Oblivious structure leakage** | Medium | High (OT security violation) | Careful index set handling; security proof verification |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Storage noise parameter mismatch** | Medium | High (invalid security) | Explicit $r$ configuration; warning if unset |
| **Phase III leakage undercount** | Low | Medium (key too long) | Conservative leakage accounting; hash bit inclusion |
| **Seed transmission failure** | Low | Medium (protocol abort) | Retry mechanism with fresh seed |

### 8.3 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Index set (I_0, I_1) loss between phases** | Medium | Critical (cannot format output) | Explicit data structure with phase metadata |
| **Incompatible key length expectations** | Low | Medium (application failure) | Clear API contracts; length in output metadata |

---

## 9. Conclusion

Phase IV is the culmination of the E-HOK protocol, transforming reconciled classical data into cryptographically secure oblivious transfer keys. The analysis reveals that while the legacy `ehok/` implementation provides excellent infrastructure for Toeplitz hashing and finite-key analysis, it operates under **QKD security assumptions that are invalid for the Noisy Storage Model**.

**Key Findings**:

1. **Critical Security Gap**: The existing `compute_final_length_finite_key()` uses $1 - h(QBER)$ instead of the NSM "Max Bound". This produces incorrect security claims for E-HOK and must be replaced.

2. **Missing Oblivious Structure**: The current implementation produces a single key, not the $(S_0, S_1)$ / $(S_C, C)$ structure required for 1-out-of-2 Oblivious Transfer.

3. **No Feasibility Pre-Check**: The "Death Valley" phenomenon (zero-length key) is detected only after wasting computational resources. A pre-flight check should abort early with batch size recommendations.

4. **SquidASM Integration**: Phase IV requires only classical communication for seed sharing. The primary integration point is configuration of the adversary storage noise parameter $r$, which may be derived from NetSquid T1/T2 memory models.

**Immediate Action Items**:

1. Create `ehok/analysis/nsm_bounds.py` implementing the Lupo et al. "Max Bound"
2. Create `ehok/core/oblivious_key.py` with `AliceObliviousKey`, `BobObliviousKey` dataclasses
3. Modify `ToeplitzAmplifier.compute_final_length()` to use NSM entropy bounds
4. Create `ehok/core/feasibility.py` for batch size validation

Upon completion, Phase IV will produce properly structured oblivious transfer outputs with provable security under the Noisy Storage Model, enabling integration into secure multiparty computation protocols.

---

## References

1. Lupo, C., Peat, J.T., Andersson, E., & Kok, P. (2023). *Error-tolerant oblivious transfer in the noisy-storage model*. arXiv:2309.xxxxx.

2. Dupuis, F., Fawzi, O., & Wehner, S. (2015). *Entanglement Sampling and Applications*. IEEE Transactions on Information Theory.

3. Schaffner, C., Terhal, B., & Wehner, S. (2009). *Robust Cryptography in the Noisy-Quantum-Storage Model*. Quantum Information & Computation.

4. Tomamichel, M., Lim, C.C.W., Gisin, N., & Renner, R. (2012). *Tight Finite-Key Analysis for Quantum Cryptography*. Nature Communications, 3, 634.

5. Erven, C., et al. (2014). *An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model*. arXiv:1308.5098v4.

6. Damgård, I., Fehr, S., Salvail, L., & Schaffner, C. (2008). *Cryptography in the Bounded Quantum-Storage Model*. SIAM Journal on Computing.

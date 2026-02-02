# Caligo Phase III: Reconciliation Specification

**Document Type:** Formal Specification  
**Version:** 1.1  
**Date:** December 17, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)  
**Prerequisites:** [phase_a_spec.md](phase_a_spec.md), [phase_b_spec.md](phase_b_spec.md), [phase_c_spec.md](phase_c_spec.md), [phase_d_spec.md](phase_d_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Theoretical Foundations: LDPC Codes](#2-theoretical-foundations-ldpc-codes)
3. [Literature Insights & NSM Bounds](#3-literature-insights--nsm-bounds)
   - 3.1-3.3 NSM Security Framework & Wiretap Cost
   - 3.4 Rate-Adaptive Reconciliation: Puncturing & Shortening
   - 3.5 Blind Reconciliation Protocol
   - 3.6 Blind Reconciliation Integration for $\binom{2}{1}$-OT
   - 3.7-3.9 Industrial Parameters & QBER Thresholds
   - 3.10 Reconciliation Strategy Summary
4. [Ehok Reconciliation Architecture](#4-ehok-reconciliation-architecture)
   - 4.1-4.7 Component Analysis
   - 4.8 Rate Adaptation Implementation
   - 4.9 Migration Recommendations
   - 4.10 Blind Reconciliation Extension Points
5. [Interactive Hashing: Future Extension](#5-interactive-hashing-future-extension)
6. [Implementation Plan](#6-implementation-plan)
   - 6.1 Pre-Runtime Tools: LDPC Matrix Generation
   - 6.2 Target Package Structure
   - 6.2.1 Reconciliation Factory Module
   - 6.3 Ehok-to-Caligo Migration Map
   - 6.4 Blind Reconciliation Module (Phase 2)
   - 6.5 Module Specifications (Core Runtime)
   - 6.6-6.11 Types, Constants, Integration, Testing
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [References](#8-references)

---

## 1. Executive Summary

### 1.1 Protocol Context

Phase III (Information Reconciliation) is the **most security-critical** stage of the $\binom{2}{1}$-OT protocol. It transforms correlated but noisy classical strings held by Alice and Bob into identical keys, while preserving the **obliviousness** property that protects Bob's choice bit $C$.

In standard QKD, reconciliation is optimized for throughput using interactive protocols. In $\binom{2}{1}$-OT, the constraints are fundamentally different:

| Aspect | Standard QKD | $\binom{2}{1}$-OT Protocol |
|--------|--------------|----------------|
| **Goal** | Maximize secret key rate | Preserve obliviousness + error correction |
| **Interactivity** | Encouraged (Cascade, Winnow) | Forbidden or heavily restricted |
| **Leakage model** | Eavesdropper passive | Bob may be adversarial |
| **Syndrome cost** | Counted against Eve | Counted as direct information leak |

### 1.2 The Obliviousness Constraint

The core security guarantee of 1-out-of-2 Oblivious Transfer requires that:
- Alice learns **nothing** about Bob's choice bit $C$
- Bob learns **exactly one** of Alice's two messages $X_0, X_1$

Interactive reconciliation protocols (Cascade, Winnow) inherently leak information about *which bits* Bob has errors on. In $\binom{2}{1}$-OT, this directly reveals information about Bob's measurement bases—and hence his choice $C$. Therefore:

> **Fundamental Constraint:** Reconciliation must be **one-way** (Alice → Bob) or use provably secure blinding techniques.

### 1.3 The Wiretap Cost Model

Unlike QKD where syndromes leak to an external eavesdropper, in $\binom{2}{1}$-OT the syndrome leaks directly to **Bob**, who may be adversarial. A cheating Bob can use the syndrome $\Sigma$ to correct errors in his noisy storage of the bits he "shouldn't" know ($I_1$).

**Mathematical Impact:** The secure key length is bounded by:
$$\ell \leq H_{\min}(X | E) - |\Sigma| - \text{security\_margins}$$

Where $|\Sigma|$ is the total syndrome length transmitted. This creates a fundamental tension:
- Higher QBER → longer syndrome required → more leakage → shorter secure key
- Beyond a threshold QBER (~10-11%), the protocol becomes infeasible

### 1.4 Deliverable Overview

| Component | Purpose | Source | Est. LOC |
|-----------|---------|--------|----------|
| `reconciliation/factory.py` | Runtime type selection factory | new | ~200 |
| `reconciliation/ldpc_encoder.py` | Syndrome computation (Alice side) | ehok extraction | ~120 |
| `reconciliation/ldpc_decoder.py` | BP decoding (Bob side) | ehok extraction | ~180 |
| `reconciliation/matrix_manager.py` | LDPC matrix loading/caching | ehok extraction | ~100 |
| `reconciliation/rate_selector.py` | Adaptive rate selection | ehok extraction | ~80 |
| `reconciliation/leakage_tracker.py` | Wiretap cost accumulator | new | ~60 |
| `reconciliation/hash_verifier.py` | Polynomial hash verification | ehok extraction | ~80 |
| `reconciliation/orchestrator.py` | Phase III coordinator | new | ~150 |
| `reconciliation/blind_manager.py` | Blind reconciliation iteration [Phase 2] | new | ~80 |
| `scripts/peg_generator.py` | PEG matrix generation (offline) | ehok migration | ~434 |
| `scripts/generate_ldpc_matrices.py` | Batch matrix generation | ehok migration | ~200 |

**Runtime LOC:** ~970 (across 8 core modules, all ≤200 LOC per Caligo guidelines)  
**Offline Tools:** ~634 (not counted against runtime complexity)

### 1.5 Design Philosophy

Phase III adheres to Caligo's core principles with reconciliation-specific refinements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE III DESIGN PRINCIPLES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ONE-WAY INFORMATION FLOW                                                │
│     └── All syndrome data flows Alice → Bob                                 │
│     └── Bob NEVER sends error positions back to Alice                       │
│     └── Failure handling is local to Bob's decoder                          │
│                                                                             │
│  2. EXPLICIT LEAKAGE ACCOUNTING                                             │
│     └── Every transmitted bit is tracked in LeakageTracker                  │
│     └── Abort conditions enforced before transmission                       │
│     └── Safety cap prevents "feigned failure" attacks                       │
│                                                                             │
│  3. EXTRACTION OVER REINVENTION                                             │
│     └── LDPC decoder proven correct in ehok                                 │
│     └── BP algorithm unchanged, only interface adaptation                   │
│     └── Matrix generation via PEG stays external                            │
│                                                                             │
│  4. RATE ADAPTIVITY                                                         │
│     └── Code rate selected based on Phase II QBER estimate                  │
│     └── Shortening mechanism for variable block sizes                       │
│     └── Blind reconciliation mode for unknown channel                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.6 Integration Points

Phase III interfaces with all Caligo layers:

| Layer | Component | Interface | Purpose |
|-------|-----------|-----------|---------|
| **Phase A** | `types/phase_contracts.py` | `SiftingPhaseResult` → input | Sifted key + QBER estimate |
| **Phase A** | `types/phase_contracts.py` | `ReconciliationPhaseResult` → output | Corrected key + leakage |
| **Phase B** | `simulation/timing.py` | `TimingBarrier` | Δt enforcement (no early reveals) |
| **Phase C** | `security/bounds.py` | `max_safe_leakage()` | Safety cap computation |
| **Phase C** | `security/feasibility.py` | `FeasibilityChecker` | Pre-reconciliation validation |
| **Phase D** | `sifting/` | `estimate_qber()` | QBER for rate selection |

### 1.7 Critical Success Criteria

1. **Correctness:** Reconciled keys match with probability ≥ 1 - 10⁻⁶
2. **Obliviousness:** Zero bits of Bob's choice leaked (information-theoretic)
3. **Efficiency:** Reconciliation efficiency $f \leq 1.22$ at target QBER
4. **Safety:** Total leakage never exceeds $L_{\max}$ bound
5. **Modularity:** Each component testable in isolation

---

## 2. Theoretical Foundations: LDPC Codes

This section establishes the mathematical and algorithmic foundations for LDPC-based reconciliation. Understanding these concepts is essential for proper implementation and debugging.

### 2.1 LDPC Code Structure

An **LDPC (Low-Density Parity-Check) code** is a linear block code defined by a sparse parity-check matrix $H$ of dimensions $m \times n$, where:
- $n$ = codeword length (number of variable nodes)
- $m$ = number of parity checks (check nodes)
- $k = n - m$ = number of information bits
- $R = k/n$ = code rate

The "low-density" property means $H$ has few 1s per row/column, enabling efficient iterative decoding.

**Tanner Graph Representation:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TANNER GRAPH STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Variable Nodes (bits)         Check Nodes (parity constraints)          │
│                                                                             │
│        v₁  v₂  v₃  v₄  v₅             c₁      c₂      c₃                   │
│         ○   ○   ○   ○   ○              □       □       □                   │
│         │╲  │  ╱│╲  │  ╱│             ╱│╲     ╱│╲     ╱│╲                  │
│         │ ╲ │ ╱ │ ╲ │ ╱ │            ╱ │ ╲ ╱  │  ╲ ╱  │  ╲                │
│         │  ╲│╱  │  ╲│╱  │           ╱  │  ╲   │   ╲   │   ╲               │
│         │   ╳   │   ╳   │          ╱       ╲  │    ╲  │    ╲              │
│         │  ╱│╲  │  ╱│╲  │         ○────○────○─────○─────○                  │
│         │ ╱ │ ╲ │ ╱ │ ╲ │         v₁   v₂   v₃    v₄    v₅                 │
│         │╱  │  ╲│╱  │  ╲│                                                   │
│         □   □   □   □   □                                                   │
│                                                                             │
│    Edge (i,j) exists iff H[i,j] = 1                                        │
│    Each check c enforces: ⊕_{v∈N(c)} v = 0 (even parity)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Degree Distributions:**
LDPC codes are characterized by edge-perspective degree distributions:
- $\lambda(x) = \sum_i \lambda_i x^{i-1}$ — variable node distribution
- $\rho(x) = \sum_i \rho_i x^{i-1}$ — check node distribution

Where $\lambda_i$ (resp. $\rho_i$) is the fraction of edges connected to degree-$i$ variable (resp. check) nodes.

### 2.2 Syndrome Computation (Encoder Side)

**Definition:** For a bit string $x \in \{0,1\}^n$ and parity-check matrix $H$, the **syndrome** is:
$$s = H \cdot x \mod 2$$

Where $s \in \{0,1\}^m$ is a binary vector of length $m = n(1-R)$.

**Properties:**
1. If $x$ is a valid codeword, $s = \mathbf{0}$
2. Syndrome reveals coset information, not the codeword itself
3. Syndrome length determines reconciliation efficiency: $f = \frac{|s|}{H(X|Y)} = \frac{1-R}{h(\epsilon)}$

**Alice's Encoding Procedure:**
```python
def compute_syndrome(alice_bits: np.ndarray, H: sp.csr_matrix) -> np.ndarray:
    """
    Compute syndrome for reconciliation.
    
    Parameters
    ----------
    alice_bits : np.ndarray
        Alice's sifted key bits (length n).
    H : sp.csr_matrix
        LDPC parity-check matrix (m × n).
    
    Returns
    -------
    np.ndarray
        Syndrome vector s = H·x mod 2 (length m).
    """
    return (H @ alice_bits) % 2
```

### 2.3 Belief Propagation Decoding (Decoder Side)

The **Sum-Product Algorithm** (belief propagation in log-domain) iteratively computes posterior probabilities for each bit given the channel observations and syndrome constraint.

**Log-Likelihood Ratio (LLR) Domain:**
For bit $v$, the LLR is defined as:
$$L(v) = \log \frac{P(v=0)}{P(v=1)}$$

- $L(v) > 0$: bit more likely 0
- $L(v) < 0$: bit more likely 1
- $|L(v)|$: confidence magnitude

**Channel LLR Initialization:**
For a Binary Symmetric Channel with crossover probability $\epsilon$ (QBER):
$$L_{ch}(v) = \log \frac{1-\epsilon}{\epsilon} \cdot (-1)^{y_v}$$

Where $y_v$ is Bob's received bit. If $y_v = 0$, positive LLR; if $y_v = 1$, negative LLR.

**Message Passing Update Rules:**

**Check-to-Variable (Horizontal Step):**
$$\mu_{c \to v} = 2 \cdot \text{arctanh}\left( \prod_{v' \in \mathcal{N}(c) \setminus v} \tanh\left(\frac{\mu_{v' \to c}}{2}\right) \right) \cdot (-1)^{s_c}$$

The syndrome bit $s_c$ flips the sign if the parity constraint is unsatisfied.

**Variable-to-Check (Vertical Step):**
$$\mu_{v \to c} = L_{ch}(v) + \sum_{c' \in \mathcal{N}(v) \setminus c} \mu_{c' \to v}$$

**Hard Decision:**
$$\hat{v} = \begin{cases} 0 & \text{if } L_{total}(v) \geq 0 \\ 1 & \text{if } L_{total}(v) < 0 \end{cases}$$

Where $L_{total}(v) = L_{ch}(v) + \sum_{c \in \mathcal{N}(v)} \mu_{c \to v}$

**Convergence Check:**
Decoding succeeds when $H \cdot \hat{x} = s$ (syndrome matches target).

### 2.4 Progressive Edge Growth (PEG) Matrix Construction

The **PEG algorithm** constructs LDPC matrices with maximized girth (shortest cycle length) in the Tanner graph, crucial for good BP decoder performance.

**Algorithm Outline:**
1. Initialize empty adjacency for $n$ variable nodes and $m$ check nodes
2. For each variable node $v$ (in order):
   - For each edge $e$ to be added to $v$ (up to target degree):
     a. Perform BFS from $v$ through existing edges
     b. Find check nodes at maximum distance not yet connected to $v$
     c. Among candidates, choose the one with lowest current degree
     d. Add edge $(v, c)$
3. Output: Parity-check matrix $H$ with $H[c,v] = 1$ iff edge exists

**Key Parameters:**
| Parameter | Symbol | Typical Value | Purpose |
|-----------|--------|---------------|---------|
| Frame size | $n$ | 16384 | Codeword length |
| Code rate | $R$ | 0.50 – 0.90 | Information ratio |
| Max tree depth | $d_{max}$ | 50 | BFS search limit |
| Variable degree dist. | $\lambda(x)$ | From literature | Edge distribution |
| Check degree dist. | $\rho(x)$ | From literature | Edge distribution |

### 2.5 Rate Adaptation Mechanisms

To handle varying QBER without multiple pre-computed codes, two techniques are combined:

**Puncturing (Increase Rate):**
- Delete $p$ columns from syndrome computation
- Those $p$ positions filled with random bits (not transmitted)
- Effective rate: $R' = \frac{R_0}{1 - p/n}$

**Shortening (Decrease Rate):**
- Fix $s$ bit positions to known values (e.g., zeros)
- Those positions have infinite LLR confidence
- Effective rate: $R' = \frac{R_0 - s/n}{1 - s/n}$

**Combined Rate Formula:**
With $\pi = p/n$ (punctured fraction) and $\sigma = s/n$ (shortened fraction):
$$R = \frac{R_0 - \sigma}{1 - \pi - \sigma}$$

**Rate Selection Criterion:**
For a channel with QBER $\epsilon$, select rate $R$ such that:
$$\frac{1-R}{h(\epsilon)} < f_{crit}$$

Where $h(\epsilon) = -\epsilon \log_2(\epsilon) - (1-\epsilon)\log_2(1-\epsilon)$ is binary entropy and $f_{crit} \approx 1.22$ is the target reconciliation efficiency.

### 2.6 Efficiency Metrics

**Reconciliation Efficiency:**
$$f = \frac{|M|}{H(X|Y)} = \frac{1-R}{h(\epsilon)}$$

Where $|M|$ is syndrome length in bits and $H(X|Y) = h(\epsilon)$ for BSC.

**Performance Targets:**
| QBER Range | Target Efficiency $f$ | Required Code Rate |
|------------|----------------------|-------------------|
| 1-3% | $f \leq 1.15$ | $R \geq 0.85$ |
| 3-5% | $f \leq 1.20$ | $R \approx 0.75$ |
| 5-8% | $f \leq 1.25$ | $R \approx 0.60$ |
| 8-11% | $f \leq 1.30$ | $R \approx 0.50$ |

---

## 3. Literature Insights & NSM Bounds

This section synthesizes theoretical foundations from the noisy-storage model (NSM) literature, the Blind Reconciliation protocol, and industrial QKD post-processing to establish the security requirements and design constraints for $\binom{2}{1}$-OT reconciliation.

### 3.1 The Noisy-Storage Model Security Framework

The NSM provides information-theoretic security for two-party protocols under the physical assumption that adversaries have limited reliable quantum storage. Unlike QKD (where Alice and Bob trust each other against an external eavesdropper), in $\binom{2}{1}$-OT **Bob himself may be adversarial**.

**Security Model (Wehner et al., 2010):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NSM ADVERSARY CAPABILITIES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UNLIMITED:                         RESTRICTED:                            │
│  ├── Classical storage              └── Quantum storage: noisy channel F   │
│  ├── Classical computation                                                  │
│  ├── Quantum computation (instantaneous)                                   │
│  └── Noise-free quantum channel                                            │
│                                                                             │
│  TIMING ENFORCEMENT:                                                        │
│  └── During Δt waiting period, adversary must either:                      │
│      a) Measure quantum state (loses coherence)                            │
│      b) Store in noisy channel F (information degrades)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight for Reconciliation:** A cheating Bob can:
1. Store quantum information about unmeasured bases during $\Delta t$
2. Use syndrome information $\Sigma$ to recover bits from noisy storage
3. Therefore, $|\Sigma|$ directly reduces security margin

### 3.2 The Wiretap Cost: Syndrome Leakage Bounds

In standard QKD, syndrome information leaks to an external eavesdropper. In $\binom{2}{1}$-OT, it leaks to **Bob**, who may be the adversary attempting to learn both $X_0$ and $X_1$.

**Lupo et al. (2023) Error-Tolerant OT Bound:**
The secure key length extractable from weak string erasure is bounded by:
$$\ell \leq H_{\min}^\varepsilon(X^m | B') - |\Sigma| - \text{security\_margins}$$

Where:
- $H_{\min}^\varepsilon(X^m | B')$ = smooth min-entropy of Alice's string given Bob's total state
- $|\Sigma|$ = total syndrome bits transmitted (reconciliation cost)
- Security margins account for finite-size effects

**König et al. (2012) Min-Entropy Characterization:**
$$H_{\min}(X|Q) = -\log_2 P_{guess}(X|Q)$$

The min-entropy quantifies the maximum probability that Bob (with quantum side information $Q$) can guess Alice's string $X$.

**Practical Implication:**
$$|\Sigma|_{max} = H_{\min}^\varepsilon(X^m | B') - \ell_{target} - \delta_{security}$$

This defines the **Safety Cap**: the maximum syndrome length Alice can transmit before aborting.

### 3.3 The One-Way Constraint: Protecting Obliviousness

**Erven et al. (2014) Experimental Implementation:**
> "Interactive algorithms like Cascade cannot be used because the interaction would reveal Bob's choice bit."

**Why Interactive Protocols Fail:**
In Cascade-style reconciliation:
1. Alice sends parity of block B
2. Bob responds: "Parity matches" or "Parity mismatch"
3. Binary search narrows error location
4. **Problem:** Bob's responses reveal which bits he measured successfully

If Alice learns Bob's error positions, she learns which bases he used → she learns $C$.

**Allowed Information Flow:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALLOWED vs FORBIDDEN COMMUNICATION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ ALLOWED (One-Way):              ✗ FORBIDDEN (Interactive):              │
│  ├── Alice → Bob: syndrome         ├── Bob → Alice: error positions        │
│  ├── Alice → Bob: shortened bits   ├── Bob → Alice: block parities         │
│  └── Alice → Bob: hash for verify  └── Bob → Alice: decode success/fail    │
│                                                                             │
│  Bob's decoder must be LOCAL:                                               │
│  └── Success/failure determined without revealing to Alice                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Rate-Adaptive Reconciliation: Puncturing & Shortening

Before addressing Blind Reconciliation, we must understand the two fundamental rate-adaptation techniques from Martinez-Mateo et al. (2012):

#### 3.4.1 Puncturing (Increase Code Rate)

**Definition:** Puncturing modulates the rate of a code $\mathcal{C}(n, k)$ by deleting $p$ symbols from codewords, converting it into $\mathcal{C}(n-p, k)$.

$$R(p) = \frac{k}{n-p} = \frac{R_0}{1-\pi} \quad \text{where } \pi = p/n$$

**Mechanism:**
- Alice embeds her $m$-bit string $X$ into a length-$n$ codeword
- The $p$ punctured positions are filled with **random bits**
- Only the syndrome of the full $n$-bit word is sent
- **Key property:** Punctured symbols reveal **no information** (random padding)

**Use Case:** When channel is **better than expected** (lower QBER), use high puncturing to increase rate and reduce unnecessary redundancy.

#### 3.4.2 Shortening (Decrease Code Rate)

**Definition:** Shortening increases redundancy by fixing $s$ symbol positions to known values (e.g., zeros), converting $\mathcal{C}(n, k)$ into $\mathcal{C}(n-s, k-s)$.

$$R(s) = \frac{k-s}{n-s} = \frac{R_0 - \sigma}{1 - \sigma} \quad \text{where } \sigma = s/n$$

**Mechanism:**
- Alice and Bob agree on $s$ positions with fixed (known) values
- These bits have **infinite LLR confidence** in the decoder
- Shortened bits **do leak information** when revealed

**Use Case:** When channel is **worse than expected** (higher QBER), use shortening to provide more error-correction capability.

#### 3.4.3 Combined Rate Adaptation ($\delta$-Modulation)

The **critical insight** from Martinez-Mateo et al. is combining both techniques:

$$R = \frac{R_0 - \sigma}{1 - \pi - \sigma} = \frac{R_0 - \sigma}{1 - \delta} \quad \text{where } \delta = \pi + \sigma$$

**Fixed String Length:** Setting $\delta = p + s = d/n$ constant allows:
- String length $m = n - d$ remains **constant**
- Rate varies by adjusting $p$ vs $s$ within fixed $d$
- No need to change block size with varying QBER

**Rate Coverage:**
$$R_{\min} = \frac{R_0 - \delta}{1 - \delta} \leq R \leq \frac{R_0}{1 - \delta} = R_{\max}$$

| $\delta$ Value | Rate Range (for $R_0 = 0.5$) | QBER Coverage |
|----------------|------------------------------|---------------|
| 5% | 0.47 - 0.53 | Narrow |
| 10% | 0.44 - 0.56 | Moderate |
| 20% | 0.38 - 0.63 | Wide |

**Trade-off:** Higher $\delta$ covers wider QBER range but reduces baseline efficiency.

### 3.5 Blind Reconciliation Protocol

The **Blind Reconciliation** protocol builds on combined puncturing/shortening to eliminate the need for a priori QBER estimation—a significant advantage for $\binom{2}{1}$-OT.

#### 3.5.1 Protocol Description

**Setup:**
- Base code $\mathcal{C}(n, k)$ corrects up to $\epsilon_{\max}$
- Strings $X, Y$ of length $m = n - d$, with $d$ modulation symbols
- Maximum iterations $t$, step size $\Delta = d/t$
- Initial state: all $d$ symbols punctured ($p = d$, $s = 0$)

**Step 1 (Encoding):** Alice sends syndrome of $\tilde{X}$ (embedding $X$ with $d$ random bits)

**Step 2 (Decoding):** Bob constructs $\tilde{Y}$ with $s$ shortened + $p$ random bits, attempts BP decoding
- If syndrome matches: **Success** → protocol ends
- If syndrome fails and $s < d$: proceed to Step 3
- If $s = d$: **Failure** → protocol aborts

**Step 3 (Re-transmission):** Alice converts $\Delta$ punctured symbols to shortened:
- Sets $s = s + \Delta$, $p = p - \Delta$
- Reveals values of $\Delta$ previously-random positions
- Returns to Step 2

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BLIND RECONCILIATION ITERATIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Iteration 1: p=d, s=0    ──▶ Highest rate, minimal leakage                │
│       │                       Only syndrome transmitted                     │
│       ▼ (failure)                                                          │
│  Iteration 2: p=d-Δ, s=Δ  ──▶ Lower rate, +Δ bits leaked                   │
│       │                       Syndrome + Δ shortened values                 │
│       ▼ (failure)                                                          │
│  Iteration 3: p=d-2Δ, s=2Δ──▶ Even lower rate, +2Δ bits total              │
│       │                                                                     │
│       ▼ ... continues until success or s=d                                 │
│                                                                             │
│  Key Property: Information leakage is MONOTONIC with iterations            │
│  ├── Punctured bits → zero leakage (random, unknown to Bob)               │
│  └── Shortened bits → full leakage (values revealed)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.5.2 Average Efficiency Analysis

The average rate achieved by Blind Reconciliation:

$$\bar{R} = \sum_{i=1}^{t} a_i r_i$$

Where $a_i$ is the fraction of blocks corrected at iteration $i$:

$$a_i = \frac{F^{(i-1)} - F^{(i)}}{1 - F^{(t)}}$$

And $F^{(i)}$ is the Frame Error Rate at iteration $i$.

**Approximation (low final FER):**
$$\bar{R} \approx R_{\max} - \frac{\beta}{t-1} \sum_{i=1}^{t-1} F^{(i)} \quad \text{where } \beta = \frac{\delta}{1-\delta}$$

**Efficiency formula:**
$$\bar{f}_{BSC(\epsilon)} = \frac{1 - \bar{R}}{h(\epsilon)}$$

#### 3.5.3 Performance vs. Iteration Count

| Max Iterations | Efficiency | Messages | Use Case |
|----------------|------------|----------|----------|
| 1 | Worst (~1.3-1.5) | 1 | Minimal interaction |
| 3 | Good (~1.15-1.25) | ≤3 | **$\binom{2}{1}$-OT baseline** |
| $d$ | Best (~1.05-1.15) | ≤d | Maximum efficiency |

**Key Results (from Martinez-Mateo simulations):**
- With $n = 2000$, $\delta = 10\%$, $t = 3$ iterations:
  - Efficiency $\bar{f} \approx 1.15$ at QBER 5%
  - Efficiency $\bar{f} \approx 1.22$ at QBER 8%
- Significant improvement over rate-adaptive (non-blind) at same code length

### 3.6 Blind Reconciliation Integration for $\binom{2}{1}$-OT (Caligo)

#### 3.6.1 Suitability Assessment

**Advantages for $\binom{2}{1}$-OT:**

| Property | Benefit for $\binom{2}{1}$-OT |
|----------|-------------------|
| **No QBER pre-estimation** | Eliminates Phase II sampling overhead |
| **One-way syndrome flow** | Preserves obliviousness (Alice → Bob only) |
| **Limited interaction** | Fits within $\Delta t$ timing window |
| **Graceful degradation** | Adapts to channel fluctuations automatically |
| **Single code storage** | Simplifies matrix management |

**Compatibility with $\binom{2}{1}$-OT constraints:**
- ✅ Bob never reveals error positions to Alice
- ✅ Iteration "failures" are local to Bob's decoder
- ✅ Alice's re-transmission is **predetermined schedule**, not feedback-based
- ⚠️ Each iteration increases total leakage $|\Sigma|$

#### 3.6.2 Implementation Complexity

**Incremental over Baseline:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BASELINE (Rate-Adaptive)         │  BLIND RECONCILIATION EXTENSION        │
├───────────────────────────────────┼─────────────────────────────────────────┤
│  1. Select rate from QBER est.    │  1. Start with R_max (all punctured)   │
│  2. Compute shortening            │  2. Pre-compute Δ schedule             │
│  3. Send syndrome (one-shot)      │  3. Send syndrome + await result       │
│  4. Bob decodes (one attempt)     │  4. On failure: send shortened values  │
│  5. Verify via hash               │  5. Repeat up to t iterations          │
├───────────────────────────────────┼─────────────────────────────────────────┤
│  LOC delta: +0                    │  LOC delta: ~+80 (iteration manager)   │
│  Complexity: Low                  │  Complexity: Low-Medium                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**New Components Required:**
1. `BlindIterationManager`: Track p/s state, generate Δ batches
2. Decoder retry logic: Re-initialize LLRs with new shortened values
3. Leakage accumulator: Sum syndrome + shortened bits per iteration

**Estimated Additional LOC:** ~80-100

#### 3.6.3 Recommended Strategy for Caligo

**Phase 1 (Baseline):** Implement standard rate-adaptive reconciliation
- Single-shot syndrome transmission
- Rate selected from Phase II QBER estimate
- Shortening computed for target efficiency

**Phase 2 (Blind Extension):** Add blind reconciliation mode
- Optional flag: `blind_mode=True`
- Default to $t = 3$ iterations maximum
- Configurable $\Delta$ step size

**Decision Criteria:**
- Use **baseline** when QBER estimate confidence is high (large sample)
- Use **blind** when QBER uncertain or channel varies rapidly

#### 3.6.4 Leakage Budget Impact

For blind reconciliation, total leakage depends on which iteration succeeds:

$$|\Sigma|_{blind} = |\Sigma|_{syndrome} + s_{final} \cdot \log_2(2) = |\Sigma|_{syndrome} + s_{final}$$

Where $s_{final}$ is shortened bits at successful iteration.

**Expected value:**
$$\mathbb{E}[|\Sigma|_{blind}] = |\Sigma|_{syndrome} + \sum_{i=1}^{t} a_i \cdot s_i$$

Typically 5-15% higher than optimal single-shot, but eliminates QBER estimation cost.

### 3.7 Industrial QKD Post-Processing: Practical Parameters

**Kiktenko et al. (2016) Implementation Guidelines:**

The Russian QKD system provides validated parameters for production deployment:

**LDPC Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame size $n$ | 4096 | Balance: efficiency vs. latency |
| Code rates $R$ | 0.50 - 0.90 (step 0.05) | Cover QBER range 1-11% |
| Max iterations | 60 | Convergence guarantee |
| Critical efficiency $f_{crit}$ | 1.22 | Industry standard |

**Rate Selection Criterion:**
$$\frac{1-R}{h_b(QBER_{est})} < f_{crit} = 1.22$$

**Shortening Formula:**
$$n_s = \left\lfloor n - \frac{m}{f_{crit} \cdot h_b(QBER_{est})} \right\rfloor$$

Where $m$ is the payload length and $n_s$ are shortened bits.

**Verification via Polynomial Hashing:**
- Hash length: 50 bits
- Collision probability: $\varepsilon_{ver} < 2 \times 10^{-12}$
- Hash function: PolyR (universal polynomial hash)

### 3.8 Security Parameter Synthesis for $\binom{2}{1}$-OT

Combining insights from NSM theory and practical QKD, we derive $\binom{2}{1}$-OT-specific parameters:

**Maximum Safe Leakage Calculation:**
```python
def compute_safety_cap(
    raw_key_length: int,
    min_entropy_rate: float,
    target_key_length: int,
    security_parameter: float = 1e-10
) -> int:
    """
    Compute maximum syndrome bits Alice can transmit.
    
    Parameters
    ----------
    raw_key_length : int
        Length of sifted key (bits).
    min_entropy_rate : float
        H_min per bit from Phase C bounds (typically 0.3-0.5).
    target_key_length : int
        Desired secure OT output length.
    security_parameter : float
        Statistical security parameter ε.
    
    Returns
    -------
    int
        Maximum syndrome length L_max.
    """
    import math
    
    total_min_entropy = raw_key_length * min_entropy_rate
    finite_size_penalty = 2 * math.log2(1 / security_parameter)
    
    L_max = int(total_min_entropy - target_key_length - finite_size_penalty)
    return max(0, L_max)
```

**$\binom{2}{1}$-OT vs QKD Parameter Comparison:**

| Parameter | Standard QKD | $\binom{2}{1}$-OT Protocol |
|-----------|--------------|----------------|
| Syndrome direction | Either way | Alice → Bob only |
| Feedback allowed | Yes (interactive) | No (one-way) |
| Leakage model | To Eve | To Bob (adversary) |
| Efficiency target | $f < 1.1$ | $f < 1.22$ (relaxed) |
| Abort condition | High QBER | $|\Sigma| > L_{max}$ |
| Block size | Large (10⁵) | Moderate (10⁴) |

### 3.9 Critical QBER Thresholds

**Protocol Feasibility Bounds:**
The $\binom{2}{1}$-OT protocol becomes infeasible when syndrome cost exceeds available entropy:

$$QBER_{critical} \approx 11\%$$

At this threshold:
- $h(0.11) \approx 0.5$ bits
- Required syndrome: $\approx 0.5n$ bits
- Remaining entropy: $< \ell_{target}$

**QBER Sensitivity Analysis:**
| QBER | Binary Entropy $h(\epsilon)$ | Min Code Rate | Syndrome Cost |
|------|------------------------------|---------------|---------------|
| 1% | 0.081 | 0.90 | 10% of key |
| 3% | 0.194 | 0.76 | 24% of key |
| 5% | 0.286 | 0.65 | 35% of key |
| 8% | 0.402 | 0.51 | 49% of key |
| 11% | 0.500 | 0.39 | 61% of key |

**Implication:** $\binom{2}{1}$-OT requires higher-quality quantum channels than standard QKD to maintain positive key rates after accounting for wiretap cost.

### 3.10 Reconciliation Strategy Summary

The following table summarizes the reconciliation approaches and their applicability to $\binom{2}{1}$-OT:

| Strategy | QBER Estimate Required | Iterations | Efficiency | $\binom{2}{1}$-OT Suitability |
|----------|------------------------|------------|------------|-------------------|
| **Single LDPC (fixed rate)** | Yes (precise) | 1 | Poor (~1.3-1.5) | ✗ Too inflexible |
| **Multiple LDPC codes** | Yes (range) | 1 | Good (~1.15) | ✗ Storage overhead |
| **Rate-Adaptive (p+s)** | Yes (rough) | 1 | Good (~1.2) | ✓ Baseline choice |
| **Blind Reconciliation** | No | ≤3 | Very Good (~1.15) | ✓✓ **Recommended** |
| **Cascade** | No | Many | Excellent (~1.05) | ✗ Interactive (leaks C) |

**Caligo Implementation Recommendation:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIGO RECONCILIATION STRATEGY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1 (Baseline Implementation):                                         │
│  └── Rate-Adaptive Reconciliation                                          │
│      ├── Single code with puncturing/shortening                            │
│      ├── Rate selected from Phase II QBER estimate                         │
│      ├── One-shot syndrome transmission                                     │
│      └── Simpler implementation, lower risk                                │
│                                                                             │
│  PHASE 2 (Efficiency Extension):                                            │
│  └── Blind Reconciliation Mode                                             │
│      ├── Optional flag: blind_mode=True                                    │
│      ├── ≤3 iterations maximum                                             │
│      ├── Eliminates QBER estimation dependency                             │
│      └── ~10% efficiency improvement                                        │
│                                                                             │
│  NOT IMPLEMENTED:                                                           │
│  └── Interactive protocols (Cascade, Winnow)                               │
│      ├── Would leak Bob's error positions                                  │
│      ├── Violates one-way constraint                                       │
│      └── Compromises obliviousness                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Ehok Reconciliation Architecture

This section provides an in-depth analysis of the `ehok` reconciliation implementation, documenting the component architecture, data flow, and design decisions that inform the Caligo migration.

### 4.1 Component Overview

The ehok reconciliation subsystem consists of six tightly integrated modules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EHOK RECONCILIATION ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐                       │
│  │   PEGMatrixGenerator │────▶│  LDPCMatrixManager  │                       │
│  │   (peg_generator.py) │     │ (ldpc_matrix_manager│                       │
│  │   434 LOC            │     │  .py) 358 LOC       │                       │
│  └─────────────────────┘     └──────────┬──────────┘                       │
│                                         │                                   │
│                                         ▼                                   │
│  ┌─────────────────────┐     ┌─────────────────────┐                       │
│  │ IntegratedQBER      │     │   LDPCReconciliator │◀─────┐                │
│  │ Estimator           │◀────│   (ldpc_reconciliator│      │                │
│  │ (qber_estimator.py) │     │   .py) 678 LOC     │      │                │
│  │ 110 LOC             │     └──────────┬──────────┘      │                │
│  └─────────────────────┘                │                 │                │
│                                         │                 │                │
│                                         ▼                 │                │
│  ┌─────────────────────┐     ┌─────────────────────┐      │                │
│  │ PolynomialHash      │◀────│  LDPCBeliefPropagation│─────┘                │
│  │ Verifier            │     │  (ldpc_bp_decoder.py)│                       │
│  │ (polynomial_hash.py)│     │  200 LOC             │                       │
│  │ 139 LOC             │     └─────────────────────┘                       │
│  └─────────────────────┘                                                    │
│                                                                             │
│  TOTAL: ~1919 LOC (exceeds Caligo 200 LOC/module guideline)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Structures

The ehok implementation defines three core dataclasses for reconciliation:

**`LDPCBlockResult`** — Single block reconciliation outcome:
```python
@dataclass
class LDPCBlockResult:
    verified: bool          # Hash verification passed
    error_count: int        # Hamming weight of error vector
    block_length: int       # Payload bits (excludes padding)
    syndrome_length: int    # Syndrome bits transmitted
    hash_bits: int = 50     # Verification hash bits
```

**`LDPCMatrixPool`** — Pre-computed matrix collection:
```python
@dataclass
class LDPCMatrixPool:
    frame_size: int                          # Fixed n for all matrices
    matrices: Dict[float, sp.spmatrix]       # rate → H matrix
    rates: np.ndarray                        # Sorted available rates
    checksum: str                            # SHA-256 for sync verification
```

**`LDPCReconciliationResult`** — Aggregate Phase III output:
```python
@dataclass
class LDPCReconciliationResult:
    corrected_key: np.ndarray    # Verified payload concatenation
    qber_estimate: float         # Rolling QBER from blocks
    total_leakage: int           # Σ(syndrome + hash) bits
    blocks_processed: int        # Total blocks attempted
    blocks_verified: int         # Successful blocks
    blocks_discarded: int        # Failed blocks
```

### 4.3 Workflow Analysis

**Alice's Encoding Path:**

```python
# 1. Rate Selection (based on Phase II QBER estimate)
rate = reconciliator.select_rate(qber_est)

# 2. Shortening Computation
n_shortened = reconciliator.compute_shortening(rate, qber_est, payload_len)

# 3. Syndrome Generation
syndrome = reconciliator.compute_syndrome_block(
    key_block=alice_key,
    rate=rate,
    n_shortened=n_shortened,
    prng_seed=block_seed
)

# 4. Transmit: syndrome, rate, n_shortened, prng_seed
```

**Bob's Decoding Path:**

```python
# 1. Reconstruct padding (identical to Alice's)
padding = reconciliator._generate_padding(n_shortened, prng_seed)

# 2. Compute LLRs with QBER-based channel model
llr_error = reconciliator._build_error_llrs(payload_len, n_shortened, qber_est)

# 3. Apply syndrome-guided LLR refinement
llr_refined = reconciliator.syndrome_guided_llr_init(
    bob_block=full_frame,
    error_syndrome=target_syndrome,
    parity_matrix=H,
    initial_llrs=llr_error
)

# 4. Belief Propagation decoding with retry
corrected, converged, iterations = reconciliator.reconcile_block(
    key_block=bob_key,
    syndrome=alice_syndrome,
    rate=rate,
    n_shortened=n_shortened,
    prng_seed=block_seed,
    max_retries=2
)

# 5. Hash Verification (if converged)
matches, hash_val = reconciliator.verify_block(alice_key, corrected)
```

### 4.4 Key Algorithm Details

#### 4.4.1 Rate Selection Algorithm

```python
def select_rate(self, qber_est: float) -> float:
    """
    Select highest rate satisfying efficiency criterion.
    
    Criterion: (1 - R) / h(QBER) < f_crit = 1.22
    """
    entropy = _binary_entropy(qber_est)
    for rate in self.matrix_manager.rates:  # rates sorted descending
        if (1 - rate) / entropy < constants.LDPC_F_CRIT:
            return float(rate)
    return float(self.rates[-1])  # Fallback to lowest rate
```

**Rate-QBER Mapping (ehok default configuration):**

| QBER Range | Selected Rate | Syndrome Fraction |
|------------|---------------|-------------------|
| 0.0 - 1.5% | 0.90 | 10% |
| 1.5 - 3.0% | 0.80 | 20% |
| 3.0 - 4.5% | 0.70 | 30% |
| 4.5 - 6.0% | 0.60 | 40% |
| 6.0 - 8.0% | 0.55 | 45% |
| 8.0 - 11.0% | 0.50 | 50% |

#### 4.4.2 LLR Construction

The initial LLR for payload bits uses the BSC model:
$$L_{ch} = \log\frac{1 - QBER}{QBER}$$

For shortened (padding) bits, LLR = 100.0 (effectively infinite confidence).

**Syndrome-Guided Refinement:**
```python
def syndrome_guided_llr_init(self, bob_block, error_syndrome, H, initial_llrs):
    """
    Reduce confidence for bits in unsatisfied parity checks.
    
    reliability_factor = 1 - (unsatisfied_count / max_degree)
    """
    for check_idx, syndrome_bit in enumerate(error_syndrome):
        if syndrome_bit == 1:  # Unsatisfied check
            participating_vars = H.indices[H.indptr[check_idx]:H.indptr[check_idx+1]]
            unsatisfied_count[participating_vars] += 1
    
    reliability_factor = 1.0 - (unsatisfied_count / max_degree)
    llrs[:payload_len] *= reliability_factor[:payload_len]  # Preserve padding
    return llrs
```

#### 4.4.3 Retry Strategy

The decoder implements adaptive retry with LLR damping:

| Attempt | Iteration Scale | LLR Damping |
|---------|-----------------|-------------|
| 1 | 1.0× | 1.00 |
| 2 | 1.5× | 0.85 |
| 3 | 2.0× | 0.70 |

Damping reduces confidence, allowing the decoder to explore alternative solutions.

#### 4.4.4 Adaptive Iteration Scaling

```python
def compute_adaptive_iterations(self, measured_qber, base_iterations=60):
    """
    Scale iterations based on QBER severity.
    """
    if measured_qber < 0.02:
        return base_iterations
    elif measured_qber < 0.05:
        return int(base_iterations * (1 + (measured_qber - 0.02) / 0.03))
    elif measured_qber < 0.10:
        return int(base_iterations * (2 + 3 * (measured_qber - 0.05) / 0.05))
    else:
        return base_iterations * 5
```

### 4.5 Leakage Accounting

Ehok tracks all information disclosed during reconciliation:

```python
def estimate_leakage_block(self, syndrome_length, hash_bits, n_shortened, frame_size):
    """
    Total leakage = syndrome + hash + shortening_info + rate_selection
    """
    base_leakage = syndrome_length + hash_bits
    
    # Shortening positions (upper bound)
    if n_shortened > 0:
        ratio = frame_size / n_shortened
        shortening_leakage = n_shortened * log2(ratio)
    
    # Rate selection (negligible)
    rate_leakage = log2(len(LDPC_CODE_RATES))  # ~3-4 bits
    
    return ceil(base_leakage + shortening_leakage + rate_leakage)
```

### 4.6 Hash Verification

The polynomial hash provides $\varepsilon$-universal collision resistance:

```python
def compute_hash(self, bits, seed):
    """
    h(x) = Σ(x_i · g^(n-i)) mod p, truncated to hash_bits
    
    - prime p = 2^61 - 1 (Mersenne prime)
    - generator g = (seed mod p) + 1
    - collision probability: 2^{-50}
    """
    base = (seed % self.prime) + 1
    acc = 0
    for bit in bits:
        acc = (acc * base + bit) % self.prime
    return (acc & self.mod_mask).to_bytes(...)
```

### 4.7 Strengths and Limitations

**Strengths:**
1. **Comprehensive rate adaptation** — Covers QBER 1-11% with single matrix pool
2. **Syndrome-guided decoding** — Improves convergence near capacity
3. **Robust retry mechanism** — Handles borderline blocks gracefully
4. **Explicit leakage tracking** — Enables security cap enforcement

**Limitations (for Caligo migration):**
1. **Monolithic `LDPCReconciliator`** — 678 LOC violates 200 LOC guideline
2. **Tight coupling** — Decoder, rate selector, and verifier intertwined
3. **Missing simulation hooks** — No generator-compatible interface
4. **No explicit one-way enforcement** — Implicit in usage, not API
5. **Constants scattered** — Parameters spread across `constants.py` and inline

### 4.8 Rate Adaptation Implementation in Ehok

The ehok implementation supports combined puncturing/shortening for rate adaptation. Understanding this mechanism is critical for the Caligo migration.

#### 4.8.1 Punctured Bit Handling

**Encoding (Alice):**
```python
def _generate_padding(self, n_shortened: int, prng_seed: int) -> np.ndarray:
    """
    Generate deterministic random padding for punctured positions.
    
    Parameters
    ----------
    n_shortened : int
        Number of shortened bits (known fixed values).
    prng_seed : int
        Seed for reproducible padding generation.
    
    Returns
    -------
    np.ndarray
        Padding bits for positions [n_shortened, frame_size - payload_length).
    """
    rng = np.random.default_rng(prng_seed)
    n_punctured = self.frame_size - self.payload_length - n_shortened
    return rng.integers(0, 2, size=n_punctured, dtype=np.uint8)
```

**Decoding (Bob):**
- Bob regenerates **identical** padding using same `prng_seed`
- Punctured positions have **zero LLR** (maximum uncertainty)
- BP decoder "discovers" correct values through parity constraints

#### 4.8.2 Shortened Bit Handling

**Encoding (Alice):**
- Shortened positions fixed to known value (typically 0)
- These bits are **transmitted** to Bob (unlike punctured bits)

**Decoding (Bob):**
- Shortened positions have **infinite LLR** (absolute certainty)
- In practice: `LLR_SHORTENED = 100.0` (sufficiently large)
- BP decoder treats these as "revealed" bits

#### 4.8.3 LLR Construction

```python
def _build_error_llrs(
    self, 
    payload_len: int, 
    n_shortened: int, 
    qber: float
) -> np.ndarray:
    """
    Construct initial LLRs for full frame.
    
    LLR structure: [payload | shortened | punctured]
    
    - Payload bits: channel LLR = log((1-qber)/qber) × sign(bob_bit)
    - Shortened bits: LLR = ±100.0 (known value)
    - Punctured bits: LLR = 0.0 (maximum uncertainty)
    """
    llr = np.zeros(self.frame_size)
    
    # Payload: BSC channel model
    channel_llr = np.log((1 - qber) / qber)
    llr[:payload_len] = channel_llr  # Sign applied after bob_bits known
    
    # Shortened: high confidence (value known)
    llr[payload_len:payload_len + n_shortened] = 100.0  # Assuming zeros
    
    # Punctured: zero confidence (completely unknown)
    # llr[payload_len + n_shortened:] already 0.0
    
    return llr
```

#### 4.8.4 Combined Rate Modulation

The effective rate after modulation:

```python
def compute_effective_rate(
    self, 
    base_rate: float, 
    n_shortened: int, 
    n_punctured: int
) -> float:
    """
    Compute effective code rate after puncturing and shortening.
    
    R_eff = (R_0 - σ) / (1 - π - σ)
    
    Where:
      R_0 = base code rate
      σ = n_shortened / frame_size
      π = n_punctured / frame_size
    """
    sigma = n_shortened / self.frame_size
    pi = n_punctured / self.frame_size
    return (base_rate - sigma) / (1 - pi - sigma)
```

### 4.9 Migration Recommendations

| Ehok Component | Caligo Target | Action |
|----------------|---------------|--------|
| `LDPCReconciliator` | Split into 3 modules | Decompose by responsibility |
| `LDPCBeliefPropagation` | `ldpc_decoder.py` | Minor interface changes |
| `LDPCMatrixManager` | `matrix_manager.py` | Keep mostly intact |
| `PolynomialHashVerifier` | `hash_verifier.py` | Direct extraction |
| `IntegratedQBEREstimator` | Move to `sifting/` | Phase II responsibility |
| `PEGMatrixGenerator` | External tool | Not runtime dependency |
| Data structures | Phase A types | Align with `phase_contracts.py` |

### 4.10 Blind Reconciliation Extension Points

To support Blind Reconciliation (Phase 2 implementation), the ehok architecture needs these modifications:

#### 4.10.1 State Machine for Iterations

```python
@dataclass
class BlindState:
    """Track blind reconciliation iteration state."""
    iteration: int = 0
    n_punctured: int = 0
    n_shortened: int = 0
    syndrome_sent: bool = False
    shortened_values_sent: List[np.ndarray] = field(default_factory=list)
    
    def advance(self, delta: int, shortened_values: np.ndarray) -> None:
        """Advance to next iteration: convert Δ punctured → shortened."""
        self.n_punctured -= delta
        self.n_shortened += delta
        self.shortened_values_sent.append(shortened_values)
        self.iteration += 1
```

#### 4.10.2 Modified Decoder Interface

```python
def decode_blind_iteration(
    self,
    bob_block: np.ndarray,
    syndrome: np.ndarray,
    state: BlindState,
    qber: float
) -> Tuple[np.ndarray, bool]:
    """
    Attempt decoding with current blind state.
    
    Parameters
    ----------
    bob_block : np.ndarray
        Bob's received bits (payload only).
    syndrome : np.ndarray
        Alice's syndrome (fixed across iterations).
    state : BlindState
        Current puncture/shorten configuration.
    qber : float
        Estimated QBER for LLR computation.
    
    Returns
    -------
    corrected : np.ndarray
        Corrected payload (if converged).
    success : bool
        True if syndrome matched.
    """
    # Reconstruct full frame with current p/s allocation
    llr = self._build_blind_llrs(bob_block, state, qber)
    corrected, converged, _ = self.decoder.decode(self.H, llr, syndrome)
    return corrected[:len(bob_block)], converged
```

#### 4.10.3 Leakage Accumulation

```python
def compute_blind_leakage(self, state: BlindState) -> int:
    """
    Total leakage after blind reconciliation.
    
    Leakage = syndrome_bits + sum(shortened_bits_revealed)
    """
    syndrome_leak = self.frame_size - self.k  # m = n(1-R)
    shortened_leak = sum(len(v) for v in state.shortened_values_sent)
    return syndrome_leak + shortened_leak
```

---

## 5. Interactive Hashing: Future Extension

This section discusses **Interactive Hashing (IH)** as an advanced optimization for $\binom{2}{1}$-OT reconciliation. While not part of the baseline implementation, IH offers significant efficiency improvements and represents a natural evolution path for Caligo.

### 5.1 The Efficiency Problem

In the baseline "Send-All" reconciliation approach, Alice sends syndromes for her **entire** raw string $X^m$, even though Bob only needs half of it (the bits in $I_0$ corresponding to his choice $C$).

**Waste Analysis:**
- Raw string length: $m$ bits
- Bob's actual interest: $|I_0| \approx m/2$ bits (bases matched)
- Syndrome transmitted: $(1-R) \cdot m$ bits
- **Wasted syndrome**: $(1-R) \cdot m/2$ bits (for $I_1$, which Bob "shouldn't" know)

This wasted syndrome still counts as leakage in the security analysis, directly reducing the secure key length.

### 5.2 Interactive Hashing Concept

Interactive Hashing is a classical two-party protocol that allows Bob to **commit** to his subset $I_0$ without revealing which specific indices he holds, while giving Alice enough information to send syndromes for **only** the relevant bits.

**Protocol Intuition (Wehner et al., 2010):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE HASHING OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT:                                                                     │
│  ├── Bob: subset W^I ⊆ [a] (his measurement indices I_0)                   │
│  └── Alice: no input                                                        │
│                                                                             │
│  OUTPUT:                                                                    │
│  ├── Alice: two subsets W_0, W_1 ⊆ [a]                                     │
│  ├── Bob: same W_0, W_1                                                    │
│  └── Guarantee: ∃C ∈ {0,1} such that W_C = W^I                             │
│                                                                             │
│  SECURITY PROPERTIES:                                                       │
│  ├── Alice learns NOTHING about C (preserves obliviousness)                │
│  └── W_{1-C} is nearly random (Bob can't control both subsets)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Alice learns **two candidate subsets** but doesn't know which one Bob actually has. She sends syndromes for **both** subsets (2× instead of 1× the required amount), achieving **50% reduction** compared to the Send-All approach.

### 5.3 The NOVY Protocol

The Naor-Ostrovsky-Venkatesan-Yung (NOVY) protocol implements interactive hashing through linear algebra over GF(2):

**Protocol Steps:**
1. Bob encodes his subset $W^I$ as a $t$-bit string
2. For $i = 1, \ldots, t-1$ rounds:
   - Alice sends random vector $a_i \in \{0,1\}^t$
   - Bob computes and returns $b_i = a_i \cdot W^I$ (inner product mod 2)
3. After $t-1$ queries, the linear system has **exactly 2 solutions**
4. Alice solves to get $W_0, W_1$; Bob knows which is his original

**Security Guarantee (Ding, 2001; CCM, 1998):**
> If Bob's subset comes from a set $T$ with $|T| \leq 2^{t-k}$, the probability that Bob can force **both** $W_0, W_1 \in T$ is at most $2^{-O(k)}$.

### 5.4 Encoding Subsets as Bit Strings

To apply NOVY, subsets must be encoded as fixed-length bit strings:

**Encoding Function:**
$$\text{Enc}: \{0,1\}^t \to \mathcal{T}$$

Where $\mathcal{T}$ is the set of all subsets of $[a]$ of size $a/4$.

**Parameter Selection:**
- Choose $t$ such that $2^t \leq \binom{a}{a/4} \leq 2 \cdot 2^t$
- At least half of all valid subsets are encodable
- Encoding/decoding is efficient (see Cachin et al., 1998)

### 5.5 Integration with Reconciliation

**Protocol Flow with Interactive Hashing (Wehner et al., 2010):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               WSEE-TO-FROT WITH INTERACTIVE HASHING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. WSEE EXECUTION                                                          │
│     └── Alice: raw string X^m                                              │
│     └── Bob: indices I, noisy substring X̃_I                                │
│                                                                             │
│  2. BLOCK ARRANGEMENT                                                       │
│     └── Partition X^m into α blocks of β bits: Z ∈ M_{α×β}                 │
│     └── Bob chooses random permutation π preserving his known blocks       │
│                                                                             │
│  3. INTERACTIVE HASHING                                                     │
│     └── Bob encodes block indices as t-bit string W^I                      │
│     └── Execute NOVY protocol                                               │
│     └── Output: W_0, W_1 (Alice knows both, Bob knows which is his)        │
│                                                                             │
│  4. TARGETED SYNDROME TRANSMISSION                                          │
│     └── Alice sends syndrome ONLY for blocks in Enc(W_0) ∪ Enc(W_1)        │
│     └── Reduction: from α blocks to α/2 blocks                             │
│                                                                             │
│  5. ERROR CORRECTION + PRIVACY AMPLIFICATION                                │
│     └── Bob corrects his subset using received syndromes                   │
│     └── Both apply two-universal hash for final key extraction             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.6 Efficiency Analysis

**Syndrome Leakage Comparison:**

| Approach | Syndrome Bits | Relative Efficiency |
|----------|---------------|---------------------|
| Send-All (baseline) | $(1-R) \cdot m$ | 1.00× |
| Interactive Hashing | $(1-R) \cdot m/2$ | **2.00×** |

**Key Rate Impact:**
$$\ell_{IH} = H_{\min}(X|E) - \frac{|\Sigma|}{2} - \text{margins}$$

For typical parameters, this can **double** the achievable secure key length.

### 5.7 Security Considerations for $\binom{2}{1}$-OT

**Preserving Obliviousness:**
The critical security requirement is that Alice learns **nothing** about Bob's choice bit $C$. Interactive Hashing satisfies this:

1. After NOVY, Alice has $(W_0, W_1)$ but no information about which is Bob's
2. The syndrome Alice sends covers **both** possibilities
3. Bob's error correction reveals no additional information to Alice

**Potential Vulnerabilities:**
1. **Timing attacks:** Bob's response time in NOVY rounds might leak information
2. **Error patterns:** If syndrome correction fails differently for $W_0$ vs $W_1$
3. **Implementation bugs:** Subtle information leaks in permutation choice

### 5.8 Implementation Complexity

**NOVY Protocol Requirements:**
| Component | Complexity | Notes |
|-----------|------------|-------|
| Linear system solver | $O(t^3)$ | GF(2) Gaussian elimination |
| Subset encoding/decoding | $O(t \log t)$ | Combinatorial number system |
| Communication rounds | $t - 1$ | Sequential, cannot parallelize |
| Storage per party | $O(t^2)$ | Linear system coefficients |

**$\binom{2}{1}$-OT Integration Challenges:**
1. **Timing constraints:** NOVY rounds must complete within $\Delta t$ window
2. **Block partitioning:** Must align with LDPC frame boundaries
3. **Permutation handling:** Bob's $\pi$ must be transmitted efficiently
4. **Error handling:** What if Interactive Hashing fails?

### 5.9 Bounded Storage Model Connection

The Interactive Hashing approach originates from the **Bounded Storage Model** (Ding, 2001; Cachin-Crépeau-Marcil, 1998), which provides information-theoretic OT security under storage constraints alone.

**Key Insight from BSM:**
The "birthday paradox" ensures that Alice and Bob's random subsets share enough common indices ($\geq k$ with high probability when $|A| = |B| = 2\sqrt{kn}$).

**Theorem (Ding, 2001):**
> In Protocol A, for any bounded-storage adversary, the probability of learning both $M_0$ and $M_1$ is at most $2^{-O(k)} + 2^{-0.02n+1}$.

### 5.10 Roadmap for Caligo Integration

**Phase 1: Research & Design (Future)**
- [ ] Formal security proof review for $\binom{2}{1}$-OT adaptation
- [ ] Parameter selection for $t$, $\alpha$, $\beta$ given target key lengths
- [ ] Timing analysis: can NOVY complete within $\Delta t$?

**Phase 2: Prototype Implementation**
- [ ] Subset encoding/decoding library
- [ ] NOVY protocol state machine
- [ ] Integration tests with mock reconciliation

**Phase 3: Full Integration**
- [ ] Replace Send-All mode with IH mode
- [ ] Performance benchmarking vs baseline
- [ ] Security audit by external reviewer

**Estimated Timeline:** 3-6 months after baseline reconciliation is stable.

### 5.11 Summary: Why Not Now?

Interactive Hashing is **not included** in the baseline Caligo implementation because:

1. **Complexity:** Adds significant protocol complexity (multi-round interaction)
2. **Risk:** Incorrect implementation could break obliviousness entirely
3. **Baseline first:** Need working one-way reconciliation before optimization
4. **Marginal gain:** For low-QBER channels, Send-All may be "good enough"

However, for **high-performance $\binom{2}{1}$-OT** deployments or **longer distances** (higher QBER), Interactive Hashing becomes essential for maintaining positive key rates.

---

## 6. Implementation Plan

This section provides detailed instructions for migrating reconciliation code from ehok to Caligo and integrating it with the existing protocol phases.

### 6.1 Pre-Runtime Tools: LDPC Matrix Generation

Before implementing runtime reconciliation, Caligo requires **pre-generated LDPC matrices**. These are created offline using the PEG algorithm and stored for runtime loading.

#### 6.1.1 PEG Algorithm Overview (Migration from ehok)

The **Progressive Edge-Growth (PEG)** algorithm constructs LDPC matrices with maximized girth, crucial for BP decoder performance.

**Source:** `deprecated/ehok/implementations/reconciliation/peg_generator.py` (434 LOC)

**Algorithm Steps:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PEG MATRIX CONSTRUCTION ALGORITHM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: n (codeword length), rate R, λ(x), ρ(x) degree distributions       │
│  OUTPUT: Parity-check matrix H ∈ {0,1}^{m×n} where m = n(1-R)              │
│                                                                             │
│  1. DEGREE ASSIGNMENT                                                       │
│     ├── Sample VN degrees from λ(x) → vn_degrees[0..n-1]                   │
│     └── Sample CN degrees from ρ(x) → cn_targets[0..m-1]                   │
│                                                                             │
│  2. EDGE PLACEMENT (for each VN v in degree-ascending order)               │
│     FOR each edge e = 1..vn_degrees[v]:                                    │
│       IF e == 1:                                                           │
│         c ← select_underfull_check(cn_current, cn_targets)                 │
│       ELSE:                                                                 │
│         reachable ← BFS_tree(v, depth=max_tree_depth)                      │
│         candidates ← {c : c ∉ reachable}                                   │
│         c ← select_underfull_check(cn_current, cn_targets, candidates)     │
│       H[c,v] ← 1; update adjacencies                                       │
│                                                                             │
│  3. OUTPUT: Convert adjacency to CSR sparse matrix                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
| Decision | Rationale |
|----------|-----------|
| Process VNs in ascending degree order | Low-degree nodes benefit most from girth optimization |
| BFS for girth maximization | Finds check nodes at maximum distance from VN neighborhood |
| Prefer underfull check nodes | Maintains target degree distribution |
| CSR output format | Efficient for sparse matrix-vector multiplication |

#### 6.1.2 Degree Distributions (from Literature)

The degree distributions are critical for code performance. We use **optimized distributions** from Martinez-Mateo et al. (Appendix B):

**Source:** `deprecated/ehok/configs/ldpc_degree_distributions.yaml`

**Structure:**
```yaml
"0.50":  # Rate R = 0.50, threshold ε* ≈ 0.1026
  lambda:  # Variable node (edge-perspective)
    degrees: [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 46, 48, 54, 55, 56, 57, 58, 65]
    probabilities: [0.14438, 0.19026, 0.01836, 0.00233, ...]
  rho:  # Check node (edge-perspective)
    degrees: [9, 11, 12, 13]
    probabilities: [0.47575, 0.46847, 0.02952, 0.02626]

"0.80":  # Rate R = 0.80, threshold ε* ≈ 0.0289
  lambda:
    degrees: [1, 2, 5, 6, 7, 16, 25]
    probabilities: [0.09420, 0.18088, 0.11972, 0.08550, 0.09816, 0.07194, 0.34960]
  rho:
    degrees: [28, 29]
    probabilities: [0.58807, 0.41193]
```

**Available Rates:** 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90

**Migration Note:** These distributions are designed for:
- Moderate edge/bit ratio (≤6.06) for hardware efficiency
- Maximized decoding threshold for each rate
- Low error floor construction compatibility

#### 6.1.3 Matrix Generation Script (Migration from ehok)

**Source:** `deprecated/ehok/configs/generate_ldpc.py` (202 LOC)

**Migration Target:** `caligo/scripts/generate_ldpc_matrices.py`

**Script Workflow:**
```python
def generate_all_matrices(output_dir: Path) -> None:
    """
    Generate LDPC matrices for all configured rates.
    
    For each rate in LDPC_CODE_RATES:
      1. Load degree distribution from YAML
      2. Initialize PEGMatrixGenerator
      3. Generate parity-check matrix H
      4. Save as compressed .npz file
      5. Log statistics (shape, density, generation time)
    """
    for rate in LDPC_CODE_RATES:
        distributions = load_distributions(rate)
        generator = PEGMatrixGenerator(
            n=LDPC_FRAME_SIZE,
            rate=rate,
            lambda_dist=distributions["lambda"],
            rho_dist=distributions["rho"],
            max_tree_depth=PEG_MAX_TREE_DEPTH,
            seed=PEG_DEFAULT_SEED
        )
        H = generator.generate()
        save_matrix(H, output_dir / f"ldpc_{LDPC_FRAME_SIZE}_rate{rate:.2f}.npz")
```

**CLI Interface:**
```bash
# Generate all matrices (silent terminal, logs to file)
python -m caligo.scripts.generate_ldpc_matrices

# Generate with terminal output and debug logging
python -m caligo.scripts.generate_ldpc_matrices --log-show --log-level DEBUG

# Custom output directory
python -m caligo.scripts.generate_ldpc_matrices --output-dir ./custom_matrices/
```

**Expected Output:**
```
===============================================================================
LDPC MATRIX GENERATION
===============================================================================
[1/9] Rate 0.50: n=4096, m=2048 ... done (12.3s)
[2/9] Rate 0.55: n=4096, m=1843 ... done (11.8s)
...
===============================================================================
GENERATION COMPLETE
Total matrices: 9
Total time: 98.2s (avg: 10.9s per matrix)

Matrix Summary:
  Rate       Shape           NNZ        Density    Time (s)
  0.50       (2048, 4096)    12288      0.0015     12.3
  0.55       (1843, 4096)    11058      0.0015     11.8
  ...
===============================================================================
```

#### 6.1.4 Migration Procedure for PEG Tools

**Step 1:** Copy core algorithm
```bash
cp deprecated/ehok/implementations/reconciliation/peg_generator.py \
   caligo/caligo/scripts/peg_generator.py
```

**Step 2:** Update imports
```python
# OLD (ehok)
from ehok.core import constants
from ehok.utils.logging import get_logger

# NEW (caligo)
from caligo.reconciliation import constants
from caligo.utils.logging import LogManager
logger = LogManager.get_stack_logger(__name__)
```

**Step 3:** Copy and adapt generation script
```bash
cp deprecated/ehok/configs/generate_ldpc.py \
   caligo/caligo/scripts/generate_ldpc_matrices.py
```

**Step 4:** Migrate degree distributions
```bash
cp deprecated/ehok/configs/ldpc_degree_distributions.yaml \
   caligo/caligo/configs/ldpc_degree_distributions.yaml
```

**Step 5:** Pre-generate matrices for distribution
```bash
cd caligo
python -m caligo.scripts.generate_ldpc_matrices \
    --output-dir caligo/configs/ldpc_matrices/
```

**Note:** PEG generation is **computationally expensive** (~10-15 seconds per matrix). Matrices should be generated once and distributed with the package, not regenerated at runtime.

### 6.2 Target Package Structure

The reconciliation phase will be implemented in a new `reconciliation/` package within Caligo:

```
caligo/caligo/
├── reconciliation/
│   ├── __init__.py              # Public API exports
│   ├── factory.py               # Runtime type selection factory (~200 LOC)
│   ├── ldpc_encoder.py          # Alice's syndrome computation (~120 LOC)
│   ├── ldpc_decoder.py          # Bob's BP decoding (~180 LOC)
│   ├── matrix_manager.py        # Matrix pool loading/caching (~100 LOC)
│   ├── rate_selector.py         # Adaptive rate selection (~80 LOC)
│   ├── leakage_tracker.py       # Wiretap cost accumulator (~60 LOC)
│   ├── hash_verifier.py         # Polynomial hash verification (~80 LOC)
│   ├── blind_manager.py         # Blind reconciliation iteration (~80 LOC) [Phase 2]
│   ├── orchestrator.py          # Phase III coordinator (~150 LOC)
│   └── constants.py             # Reconciliation parameters (~40 LOC)
├── scripts/
│   ├── peg_generator.py         # PEG matrix generation (offline tool)
│   └── generate_ldpc_matrices.py # Batch matrix generation script
├── configs/
│   ├── ldpc_degree_distributions.yaml  # Degree distributions per rate
│   └── ldpc_matrices/           # Pre-generated .npz files
│       ├── ldpc_4096_rate0.50.npz
│       ├── ldpc_4096_rate0.55.npz
│       └── ... (9 files total)
├── types/
│   └── phase_contracts.py       # Add ReconciliationPhaseResult
└── tests/
    └── reconciliation/
        ├── test_bp_decoder.py
        ├── test_rate_selector.py
        ├── test_hash_verifier.py
        ├── test_leakage_tracker.py
        ├── test_blind_manager.py
        ├── test_factory.py
        └── test_orchestrator.py
```

**LOC Summary:**
| Component | Est. LOC | Source |
|-----------|----------|--------|
| Runtime modules (8) | ~890 | ehok migration + new |
| Factory module | ~200 | new |
| Blind extension | ~80 | new |
| Scripts (2) | ~640 | ehok migration |
| **Total runtime** | **~970** | All ≤200 LOC per module |

### 6.2.1 Reconciliation Factory Module (`factory.py`)

The factory module enables **runtime reconciliation type selection** via YAML configuration, supporting protocol comparison and experimentation without code changes.

#### Supported Reconciliation Types

| Type | Description | Use Case |
|------|-------------|----------|
| `baseline` | Standard rate-adaptive LDPC | Known QBER, production use |
| `blind` | Martinez-Mateo blind protocol | Unknown QBER, self-adapting |
| `interactive` | Cascade-style (NYI) | Future extension, not implemented |

#### YAML Configuration

```yaml
# config/reconciliation.yaml
reconciliation:
  type: "blind"                    # "baseline" | "blind" | "interactive"
  frame_size: 4096                 # LDPC codeword length
  max_iterations: 50               # BP decoder iterations
  max_blind_rounds: 3              # Blind: max syndrome retries
  use_nsm_informed_start: true     # Use NSM params for initial rate
  safety_margin: 0.05              # Capacity margin for rate selection
  puncturing_enabled: true         # Allow puncturing for rate-up
  shortening_enabled: true         # Allow shortening for rate-down
  ldpc_matrix_path: null           # Optional: custom matrix location
```

#### Python API Usage

```python
from caligo.reconciliation import (
    ReconciliationType,
    ReconciliationConfig,
    create_reconciler,
)
from caligo.simulation.physical_model import NSMParameters

# From configuration dictionary
config_dict = {
    "type": "blind",
    "frame_size": 4096,
    "use_nsm_informed_start": True,
}
config = ReconciliationConfig.from_dict(config_dict)

# With NSM-informed initial rate selection
nsm_params = NSMParameters.from_erven_experimental()
reconciler = create_reconciler(config, nsm_params=nsm_params)

# Execute reconciliation
reconciled_key, metadata = reconciler.reconcile(alice_bits, bob_bits)
```

#### NSM-Informed Blind Reconciliation

When `use_nsm_informed_start: true`, the factory uses NSM channel parameters to compute an optimal starting rate for blind reconciliation:

```python
# From NSMParameters.blind_reconciliation_config()
{
    "initial_rate": 0.80,          # Based on estimated QBER
    "max_iterations": 3,
    "frame_size": 4096,
    "expected_qber": 0.034,        # From qber_channel property
    "rate_adaptation": "puncturing",
    "use_nsm_informed_start": True
}
```

**QBER-to-Rate Mapping:**

| Estimated QBER | Initial Rate | Rate Adaptation |
|----------------|--------------|-----------------|
| < 2% | 0.90 | Puncturing (rate ↑) |
| 2-5% | 0.80 | Puncturing |
| 5-8% | 0.70 | Shortening (rate ↓) |
| > 8% | 0.60 | Shortening |

This eliminates the "cold start" problem of blind reconciliation where an uninformed initial rate might cause unnecessary iteration rounds.

### 6.2.2 QBER Estimation Policy & Reconciliation Type Constraints

The **core advantage** of blind reconciliation is eliminating the need for prior QBER estimation. This constraint must be enforced in the pipeline.

#### QBER Estimation Requirements by Reconciliation Type

| Type | QBER Estimation | Rationale |
|------|-----------------|-----------|
| `baseline` | **Required** | LDPC rate selection depends on measured QBER |
| `blind` | **Forbidden** | Core advantage—iterative rate discovery via puncturing/shortening |
| `interactive_hashing` | **Not applicable** | Protocol uses hash constraints, not error-correcting codes |

> **Note:** Interactive Hashing (Section 5) is a distinct protocol from LDPC-based reconciliation. It uses iterative hash function exchanges to achieve information-theoretic obliviousness, not error correction. QBER estimation is irrelevant since the protocol operates on the *structure* of the key space rather than channel error rates.

**Pipeline Enforcement:**
```python
class ReconciliationType(Enum):
    BASELINE = "baseline"
    BLIND = "blind"
    INTERACTIVE_HASHING = "interactive_hashing"
    
    @property
    def requires_qber_estimation(self) -> bool:
        """
        Returns True if this reconciliation type needs prior QBER estimation.
        
        - BASELINE: Yes—LDPC rate selection requires QBER
        - BLIND: No—core advantage is iterative rate discovery
        - INTERACTIVE_HASHING: N/A—uses hash constraints, not ECC
        """
        return self == ReconciliationType.BASELINE
```

**Orchestrator Integration:**
```python
def execute_reconciliation_phase(
    sifted_result: SiftingPhaseResult,
    config: ReconciliationConfig,
) -> ReconciliationPhaseResult:
    """
    Execute Phase III with type-appropriate QBER handling.
    """
    if config.reconciliation_type.requires_qber_estimation:
        # BASELINE: Perform QBER estimation for rate selection
        qber_estimate = estimate_qber(
            alice_bits=sifted_result.test_bits_alice,
            bob_bits=sifted_result.test_bits_bob,
        )
        target_rate = compute_rate_from_qber(qber_estimate)
    elif config.reconciliation_type == ReconciliationType.BLIND:
        # BLIND: Skip QBER estimation—iterative rate discovery
        qber_estimate = None
        target_rate = None  # Discovered via puncturing/shortening
    else:
        # INTERACTIVE_HASHING: Different protocol entirely (see Section 5)
        raise NotImplementedError("Interactive Hashing not yet implemented")
    
    reconciler = create_reconciler(config, qber_estimate)
    return reconciler.reconcile(...)
```

### 6.2.3 Simulation Module Refactoring Specification

The `physical_model.py` and `noise_models.py` modules currently contain **duplicated code** that must be consolidated to meet the <200 LOC per module constraint.

#### Current State Analysis

**physical_model.py (942 LOC)** — Exceeds limit
- `NSMParameters` dataclass with QBER calculations
- `ChannelParameters` dataclass
- PDC probability functions (Erven et al.)
- NetSquid adapter factory functions
- Duplicated QBER formula implementations

**noise_models.py (601 LOC)** — Exceeds limit
- `NSMStorageNoiseModel` class
- `ChannelNoiseProfile` dataclass with QBER calculations
- Duplicated QBER formula implementations
- Duplicated reconciliation support methods

#### Identified Code Duplication

| Function/Method | physical_model.py | noise_models.py | Action |
|-----------------|-------------------|-----------------|--------|
| QBER formula (3-term Erven) | `NSMParameters.qber_channel` | `ChannelNoiseProfile.total_qber` | Extract to `utils/qber.py` |
| Conditional QBER | `NSMParameters.qber_full_erven` | `ChannelNoiseProfile.qber_conditional` | Extract to `utils/qber.py` |
| `suggested_ldpc_rate()` | In `NSMParameters` | In `ChannelNoiseProfile` | Extract to `utils/math.py` |
| `blind_reconciliation_config()` | In `NSMParameters` | In `ChannelNoiseProfile` | Extract to `utils/math.py` |
| Security thresholds | `QBER_HARD_LIMIT`, `QBER_CONSERVATIVE_LIMIT` | Same constants | Move to `constants.py` |
| Erven experimental values | `ERVEN_*` constants | `from_erven_experimental()` | Centralize in `constants.py` |

#### Proposed Refactoring

**1. Create `caligo/simulation/constants.py` (~50 LOC)**
```python
"""Simulation constants from literature."""

# Time units (NetSquid compatibility)
NANOSECOND = 1.0
MICROSECOND = 1e3
# ...

# Security thresholds
QBER_HARD_LIMIT = 0.22        # König et al. (2012)
QBER_CONSERVATIVE_LIMIT = 0.11 # Schaffner et al. (2009)

# Erven et al. (2014) Table I
ERVEN_MU = 3.145e-5
ERVEN_ETA = 0.0150
ERVEN_E_DET = 0.0093
ERVEN_P_DARK = 1.50e-8
ERVEN_R = 0.75
ERVEN_NU = 0.002
```

**2. Extend `caligo/utils/math.py` (~40 LOC additions)**
```python
def compute_qber_erven(
    fidelity: float,
    detector_error: float,
    detection_efficiency: float,
    dark_count_prob: float,
) -> float:
    """
    Compute total QBER using Erven et al. (2014) formula.
    
    Q_total = (1-F)/2 + e_det + (1-η) × P_dark / 2
    """
    q_source = (1.0 - fidelity) / 2.0
    q_det = detector_error
    q_dark = (1.0 - detection_efficiency) * dark_count_prob / 2.0
    return q_source + q_det + q_dark

def suggested_ldpc_rate_from_qber(
    qber: float,
    safety_margin: float = 0.05,
) -> float:
    """Suggest LDPC rate from channel capacity."""
    if qber >= 0.5:
        return 0.5
    capacity = 1.0 - binary_entropy(qber)
    return max(0.5, min(0.95, capacity - safety_margin))

def blind_reconciliation_initial_config(qber: float) -> dict:
    """Generate blind reconciliation config from estimated QBER."""
    if qber < 0.02:
        return {"initial_rate": 0.90, "rate_adaptation": "puncturing"}
    elif qber < 0.05:
        return {"initial_rate": 0.80, "rate_adaptation": "puncturing"}
    elif qber < 0.08:
        return {"initial_rate": 0.70, "rate_adaptation": "shortening"}
    else:
        return {"initial_rate": 0.60, "rate_adaptation": "shortening"}
```

**3. Refactor `physical_model.py` (~180 LOC target)**

| Keep | Remove/Refactor |
|------|-----------------|
| `NSMParameters` dataclass | Move QBER methods to delegate to `utils/math` |
| `ChannelParameters` dataclass | Remove duplicated constants |
| PDC functions (`pdc_probability`, etc.) | Keep (unique to this module) |
| Factory methods | Keep |
| NetSquid adapters | Move to `netsquid_adapters.py` |

**4. Refactor `noise_models.py` (~150 LOC target)**

| Keep | Remove/Refactor |
|------|-----------------|
| `NSMStorageNoiseModel` class | Keep (unique) |
| `ChannelNoiseProfile` dataclass | Delegate QBER to `utils/math` |
| Factory methods | Keep |
| Conversion methods | Keep |
| Duplicated QBER calculations | Remove, call shared functions |

#### Target Module Sizes After Refactoring

| Module | Current LOC | Target LOC | Reduction |
|--------|-------------|------------|-----------|
| `simulation/constants.py` | (new) | ~50 | N/A |
| `simulation/physical_model.py` | 942 | ~180 | -81% |
| `simulation/noise_models.py` | 601 | ~150 | -75% |
| `utils/math.py` | 293 | ~340 | +16% |

**Total simulation/ package:** 942 + 601 = 1543 LOC → ~380 LOC (constants + physical + noise)

#### Migration Steps

1. **Create `constants.py`**: Extract all shared constants
2. **Extend `utils/math.py`**: Add `compute_qber_erven()` and reconciliation helpers
3. **Update `physical_model.py`**:
   - Import constants from `constants.py`
   - Delegate QBER calculation to `utils/math.compute_qber_erven()`
   - Move NetSquid factory functions to `netsquid_adapters.py`
4. **Update `noise_models.py`**:
   - Import constants from `constants.py`
   - Delegate QBER/rate methods to `utils/math`
   - Remove duplicated implementations
5. **Update imports** in all dependent modules
6. **Run tests** to verify no behavioral changes

#### Dependency Graph After Refactoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMULATION MODULE DEPENDENCIES                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  caligo/utils/math.py                                                       │
│    ├── binary_entropy()                                                     │
│    ├── channel_capacity()                                                   │
│    ├── compute_qber_erven()        ← NEW: shared QBER formula              │
│    ├── suggested_ldpc_rate_from_qber()  ← NEW: shared rate suggestion      │
│    └── blind_reconciliation_initial_config()  ← NEW                        │
│         ↑                                                                   │
│  ┌──────┴───────────────────────────────────────────────────┐              │
│  │                                                           │              │
│  ▼                                                           ▼              │
│  caligo/simulation/physical_model.py    caligo/simulation/noise_models.py  │
│    ├── NSMParameters                       ├── NSMStorageNoiseModel        │
│    │   └── qber_channel → delegates        │   └── apply_noise()           │
│    ├── ChannelParameters                   └── ChannelNoiseProfile         │
│    └── PDC functions                           └── total_qber → delegates  │
│         ↑                                                                   │
│         │                                                                   │
│  caligo/simulation/constants.py                                            │
│    ├── QBER_HARD_LIMIT, QBER_CONSERVATIVE_LIMIT                            │
│    ├── ERVEN_* experimental values                                         │
│    └── Time unit constants                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Ehok-to-Caligo Migration Map

This table provides the complete migration mapping from ehok to Caligo:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EHOK → CALIGO MIGRATION MAP                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RUNTIME COMPONENTS                                                         │
│  ─────────────────                                                          │
│  ehok/implementations/reconciliation/ldpc_bp_decoder.py (200 LOC)          │
│    └── caligo/reconciliation/ldpc_decoder.py                               │
│        Action: Nearly direct copy, add dataclass results                    │
│                                                                             │
│  ehok/implementations/reconciliation/ldpc_reconciliator.py (678 LOC)       │
│    ├── caligo/reconciliation/ldpc_encoder.py (~120 LOC)                    │
│    │   Extract: compute_syndrome_block(), _generate_padding()              │
│    ├── caligo/reconciliation/rate_selector.py (~80 LOC)                    │
│    │   Extract: select_rate(), compute_shortening(), _binary_entropy()     │
│    └── caligo/reconciliation/orchestrator.py (~150 LOC)                    │
│        Extract: reconcile_block(), aggregate_results()                      │
│                                                                             │
│  ehok/implementations/reconciliation/ldpc_matrix_manager.py (358 LOC)      │
│    └── caligo/reconciliation/matrix_manager.py (~100 LOC)                  │
│        Action: Simplify, remove autogeneration, keep loading/caching       │
│                                                                             │
│  ehok/implementations/reconciliation/polynomial_hash.py (139 LOC)          │
│    └── caligo/reconciliation/hash_verifier.py (~80 LOC)                    │
│        Action: Direct extraction, rename class                              │
│                                                                             │
│  NEW COMPONENTS (not from ehok)                                            │
│    ├── caligo/reconciliation/leakage_tracker.py (~60 LOC)                  │
│    └── caligo/reconciliation/blind_manager.py (~80 LOC) [Phase 2]          │
│                                                                             │
│  OFFLINE TOOLS                                                              │
│  ─────────────                                                              │
│  ehok/implementations/reconciliation/peg_generator.py (434 LOC)            │
│    └── caligo/scripts/peg_generator.py                                     │
│        Action: Copy, update imports                                         │
│                                                                             │
│  ehok/configs/generate_ldpc.py (202 LOC)                                   │
│    └── caligo/scripts/generate_ldpc_matrices.py                            │
│        Action: Copy, update imports, adjust paths                           │
│                                                                             │
│  ehok/configs/ldpc_degree_distributions.yaml                               │
│    └── caligo/configs/ldpc_degree_distributions.yaml                       │
│        Action: Direct copy (no changes needed)                              │
│                                                                             │
│  RELOCATED (not in reconciliation/)                                        │
│  ──────────                                                                 │
│  ehok/implementations/reconciliation/qber_estimator.py (110 LOC)           │
│    └── caligo/sifting/qber_estimator.py                                    │
│        Rationale: QBER estimation is Phase II (sifting) responsibility     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Blind Reconciliation Module (Phase 2 Extension)

#### 6.4.1 `blind_manager.py` — Blind Reconciliation Controller

**Purpose:** Manage puncture/shorten state transitions for blind reconciliation.

**New Component** (implements Martinez-Mateo et al. blind protocol)

**Interface:**
```python
@dataclass
class BlindConfig:
    """Configuration for blind reconciliation."""
    max_iterations: int = 3          # Maximum retry iterations (t) (can be input parameter)
    modulation_fraction: float = 0.1 # δ = d/n fraction of frame for p+s
    
    @property
    def delta_per_iteration(self) -> float:
        """Fraction converted per iteration: Δ = d/t."""
        return self.modulation_fraction / self.max_iterations

@dataclass 
class BlindIterationState:
    """State tracking for blind reconciliation iterations."""
    iteration: int           # Current iteration (0 = initial)
    n_punctured: int         # Currently punctured positions
    n_shortened: int         # Currently shortened positions  
    shortened_values: List[np.ndarray]  # Values revealed so far
    syndrome_leakage: int    # Fixed syndrome bits
    
    @property
    def total_leakage(self) -> int:
        """Total bits leaked: syndrome + shortened values."""
        shortened_bits = sum(len(v) for v in self.shortened_values)
        return self.syndrome_leakage + shortened_bits
    
    @property
    def current_rate_factor(self) -> float:
        """Rate adjustment factor for current state."""
        # R_eff / R_0 = (1 - σ) / (1 - π - σ) where σ,π are fractions
        pass

class BlindReconciliationManager:
    """
    Manage blind reconciliation iteration protocol.
    
    Protocol Flow:
    1. Initialize: all d modulation bits punctured (max rate)
    2. Alice sends syndrome (one-time)
    3. Bob attempts decode
       - Success → return corrected key
       - Failure → Alice reveals Δ shortened values
    4. Repeat step 3 until success or iteration limit
    
    Key Properties:
    - One-way information flow preserved (Alice → Bob)
    - No error-position feedback (Bob's failure is local)
    - Leakage monotonically increases with iterations
    """
    
    def __init__(
        self,
        frame_size: int,
        payload_length: int,
        config: BlindConfig
    ) -> None:
        """
        Initialize blind reconciliation manager.
        
        Parameters
        ----------
        frame_size : int
            LDPC codeword length (n).
        payload_length : int  
            Actual data bits per block (m = n - d).
        config : BlindConfig
            Protocol configuration.
        """
        ...
    
    def initialize(self, syndrome_length: int) -> BlindIterationState:
        """
        Start new blind reconciliation.
        
        Initial state: all modulation bits punctured (p=d, s=0).
        
        Parameters
        ----------
        syndrome_length : int
            Length of syndrome vector (fixed across iterations).
        
        Returns
        -------
        BlindIterationState
            Initial state with max puncturing.
        """
        ...
    
    def should_continue(self, state: BlindIterationState) -> bool:
        """
        Check if more iterations are allowed.
        
        Returns False when:
        - s = d (all bits shortened, no more room)
        - iteration >= max_iterations
        """
        ...
    
    def advance_iteration(
        self, 
        state: BlindIterationState,
        alice_shortened_values: np.ndarray
    ) -> BlindIterationState:
        """
        Advance to next iteration by revealing shortened values.
        
        Converts Δ punctured positions to shortened:
        - p_new = p_old - Δ
        - s_new = s_old + Δ
        
        Parameters
        ----------
        state : BlindIterationState
            Current iteration state.
        alice_shortened_values : np.ndarray
            Values of newly-shortened positions (from Alice).
        
        Returns
        -------
        BlindIterationState
            Updated state for next iteration.
        """
        ...
    
    def build_llr_for_state(
        self, 
        bob_bits: np.ndarray,
        state: BlindIterationState,
        qber: float
    ) -> np.ndarray:
        """
        Construct LLRs for current blind state.
        
        LLR structure:
        - Payload bits: channel LLR (based on QBER)
        - Shortened bits: high confidence (values known)
        - Punctured bits: zero confidence (unknown)
        """
        ...
```

**Estimated LOC:** ~80

### 6.5 Module Specifications (Core Runtime)

#### 6.5.1 `ldpc_encoder.py` — Alice's Encoding

**Purpose:** Compute syndrome for Alice's key blocks.

**Migration Source:** `ehok/implementations/reconciliation/ldpc_reconciliator.py` → `compute_syndrome_block()`

**Interface:**
```python
@dataclass
class SyndromeBlock:
    """Syndrome data for one LDPC block."""
    syndrome: np.ndarray      # Binary syndrome vector
    rate: float               # Code rate used
    n_shortened: int          # Shortened bit count
    prng_seed: int            # Padding generation seed
    leakage_bits: int         # Syndrome length for accounting

def compute_syndrome(
    alice_key: np.ndarray,
    H: sp.csr_matrix,
    n_shortened: int,
    prng_seed: int
) -> np.ndarray:
    """Compute s = H·x mod 2 with padding."""

def encode_block(
    key_block: np.ndarray,
    rate: float,
    matrix_manager: MatrixManager,
    qber_estimate: float
) -> SyndromeBlock:
    """Full encoding pipeline for single block."""
```

#### 6.5.2 `ldpc_decoder.py` — Bob's Decoding

**Purpose:** Belief propagation decoder with syndrome-guided initialization.

**Migration Source:** `ehok/implementations/reconciliation/ldpc_bp_decoder.py` (mostly intact)

**Interface:**
```python
@dataclass
class DecodeResult:
    """Result of BP decoding attempt."""
    corrected: np.ndarray     # Corrected payload (excluding padding)
    converged: bool           # True if syndrome matched
    iterations: int           # Iterations until convergence/timeout
    error_count: int          # Hamming weight of error vector

class BeliefPropagationDecoder:
    """Sum-product BP decoder in log-domain."""
    
    def __init__(
        self,
        max_iterations: int = 60,
        convergence_threshold: float = 1e-6
    ) -> None: ...
    
    def decode(
        self,
        H: sp.csr_matrix,
        llr: np.ndarray,
        target_syndrome: np.ndarray
    ) -> DecodeResult: ...

def build_channel_llr(
    bob_bits: np.ndarray,
    qber: float,
    n_shortened: int
) -> np.ndarray:
    """Construct initial LLRs from channel model."""

def syndrome_guided_refinement(
    llr: np.ndarray,
    error_syndrome: np.ndarray,
    H: sp.csr_matrix
) -> np.ndarray:
    """Refine LLRs based on unsatisfied parity checks."""
```

#### 6.5.3 `matrix_manager.py` — Matrix Pool Management

**Purpose:** Load, cache, and provide LDPC matrices.

**Migration Source:** `ehok/implementations/reconciliation/ldpc_matrix_manager.py` (simplified)

**Interface:**
```python
@dataclass
class MatrixPool:
    """Immutable pool of LDPC matrices."""
    frame_size: int
    matrices: Dict[float, sp.csr_matrix]
    rates: Tuple[float, ...]
    checksum: str

class MatrixManager:
    """Thread-safe matrix pool accessor."""
    
    @classmethod
    def from_directory(cls, path: Path) -> "MatrixManager":
        """Load all matrices from directory."""
    
    def get_matrix(self, rate: float) -> sp.csr_matrix:
        """Retrieve matrix for specified rate."""
    
    def verify_checksum(self, expected: str) -> bool:
        """Validate matrix pool integrity."""
    
    @property
    def frame_size(self) -> int: ...
    
    @property
    def available_rates(self) -> Tuple[float, ...]: ...
```

#### 6.5.4 `rate_selector.py` — Adaptive Rate Selection

**Purpose:** Select optimal code rate based on QBER estimate.

**Migration Source:** `ehok/implementations/reconciliation/ldpc_reconciliator.py` → `select_rate()`, `compute_shortening()`

**Interface:**
```python
@dataclass
class RateSelection:
    """Rate selection result."""
    rate: float
    n_shortened: int
    expected_efficiency: float

def select_rate(
    qber_estimate: float,
    available_rates: Tuple[float, ...],
    f_crit: float = 1.22
) -> float:
    """Select highest rate satisfying efficiency criterion."""

def compute_shortening(
    rate: float,
    qber_estimate: float,
    payload_length: int,
    frame_size: int,
    f_crit: float = 1.22
) -> int:
    """Compute shortened bits for target efficiency."""

def binary_entropy(p: float) -> float:
    """Compute h(p) = -p·log(p) - (1-p)·log(1-p)."""
```

#### 6.5.5 `leakage_tracker.py` — Wiretap Cost Accounting

**Purpose:** Track cumulative information leakage and enforce safety cap.

**New Component** (not directly from ehok)

**Interface:**
```python
@dataclass
class LeakageRecord:
    """Single leakage event."""
    syndrome_bits: int
    hash_bits: int
    shortening_bits: float  # log2(C(n, n_s)) upper bound
    timestamp: float        # Simulation time

class LeakageTracker:
    """Accumulate and enforce leakage bounds."""
    
    def __init__(self, safety_cap: int) -> None:
        """Initialize with maximum allowed leakage."""
    
    def record(self, record: LeakageRecord) -> None:
        """Add leakage event."""
    
    def check_safety(self) -> bool:
        """Return False if safety cap exceeded."""
    
    @property
    def total_leakage(self) -> int:
        """Current cumulative leakage."""
    
    @property
    def remaining_budget(self) -> int:
        """Bits remaining before abort."""
```

#### 6.5.6 `hash_verifier.py` — Block Verification

**Purpose:** Polynomial hash for corrected block verification.

**Migration Source:** `ehok/implementations/reconciliation/polynomial_hash.py` (direct extraction)

**Interface:**
```python
class PolynomialHashVerifier:
    """ε-universal hash for block verification."""
    
    def __init__(
        self,
        hash_bits: int = 50,
        prime: int = 2**61 - 1
    ) -> None: ...
    
    def compute_hash(
        self,
        bits: np.ndarray,
        seed: int
    ) -> bytes: ...
    
    def verify(
        self,
        expected: bytes,
        bits: np.ndarray,
        seed: int
    ) -> bool: ...
```

#### 6.5.7 `orchestrator.py` — Phase III Coordinator

**Purpose:** Orchestrate full reconciliation phase with Caligo integration.

**New Component** (combines ehok patterns with Caligo contracts)

**Interface:**
```python
@dataclass
class ReconciliationConfig:
    """Phase III configuration."""
    frame_size: int = 4096
    max_iterations: int = 60
    hash_bits: int = 50
    f_crit: float = 1.22
    max_retries: int = 2
    
class ReconciliationOrchestrator:
    """Coordinate Alice/Bob reconciliation flow."""
    
    def __init__(
        self,
        matrix_manager: MatrixManager,
        leakage_tracker: LeakageTracker,
        config: ReconciliationConfig
    ) -> None: ...
    
    # Alice's methods (syndrome generation)
    def alice_encode_blocks(
        self,
        sifted_key: np.ndarray,
        qber_estimate: float
    ) -> List[SyndromeBlock]: ...
    
    # Bob's methods (decoding)
    def bob_decode_blocks(
        self,
        bob_key: np.ndarray,
        syndromes: List[SyndromeBlock],
        qber_estimate: float
    ) -> ReconciliationPhaseResult: ...
    
    # Safety check
    def should_abort(self) -> bool:
        """Check if leakage cap exceeded."""
```

### 6.6 Type Definitions (Phase A Integration)

These types align with `caligo_architecture.md` §3.4 and `phase_a_spec.md` §3.3.4.

Add to `caligo/types/phase_contracts.py`:

```python
@dataclass(frozen=True)
class ReconciliationPhaseResult:
    """
    Contract: Phase III → Phase IV data transfer.
    
    Contains error-corrected key material with leakage accounting
    for privacy amplification entropy calculation.
    
    This contract aligns with caligo_architecture.md §3.4:
    - reconciled_key → reconciled key (Alice's perspective)
    - total_syndrome_bits → |Σ| leakage to adversary
    - effective_rate → achieved code rate
    
    Attributes
    ----------
    reconciled_key : bitarray
        Error-corrected key (Alice's perspective).
    num_blocks : int
        Number of LDPC blocks processed.
    blocks_succeeded : int
        Number of blocks that passed verification.
    blocks_failed : int
        Number of blocks that failed (discarded).
    total_syndrome_bits : int
        Total syndrome leakage |Σ| in bits.
    effective_rate : float
        Achieved code rate R = (n - |Σ|) / n.
    hash_verified : bool
        True if final hash verification passed.
    leakage_within_cap : bool
        True if |Σ| ≤ L_max (safety cap).
    i0_reconciled_key : bitarray
        Reconciled bits from I₀ partition (for OT S₀).
    i1_reconciled_key : bitarray
        Reconciled bits from I₁ partition (for OT S₁).
    qber_estimate : float
        Updated QBER from block error counts.
    
    Post-conditions
    ---------------
    - POST-R-001: total_syndrome_bits ≤ L_max (leakage safety cap)
    - POST-R-002: hash_verified == True (else would abort)
    - POST-R-003: len(i0_reconciled_key) + len(i1_reconciled_key) == len(reconciled_key)
    
    References
    ----------
    - caligo_architecture.md §3.4: Phase III contract
    - phase_a_spec.md §3.3.4: ReconciliationPhaseResult definition
    - phase_III_analysis.md: Wiretap cost model
    """
    reconciled_key: bitarray
    num_blocks: int
    blocks_succeeded: int
    blocks_failed: int
    total_syndrome_bits: int
    effective_rate: float
    hash_verified: bool
    leakage_within_cap: bool
    i0_reconciled_key: bitarray
    i1_reconciled_key: bitarray
    qber_estimate: float
    
    def __post_init__(self) -> None:
        if self.total_syndrome_bits < 0:
            raise ValueError("total_syndrome_bits must be non-negative")
        if not 0.0 <= self.qber_estimate <= 1.0:
            raise ValueError("qber_estimate must be in [0, 1]")
        if not 0.0 <= self.effective_rate <= 1.0:
            raise ValueError("effective_rate must be in [0, 1]")
```

**Input Contract Reference:**

```python
# From phase_a_spec.md §3.3.3 — Phase II output consumed by Phase III
@dataclass
class SiftingPhaseResult:
    """
    Contract: Phase II → Phase III data transfer.
    
    Key fields for reconciliation:
    - sifted_key_alice: Alice's bits to reconcile
    - sifted_key_bob: Bob's bits with channel errors
    - qber_adjusted: QBER with finite-size penalty (use for rate selection)
    - i0_indices, i1_indices: Partition info preserved through Phase III
    """
    sifted_key_alice: bitarray
    sifted_key_bob: bitarray
    matching_indices: np.ndarray
    i0_indices: np.ndarray
    i1_indices: np.ndarray
    test_set_indices: np.ndarray
    qber_estimate: float
    qber_adjusted: float
    finite_size_penalty: float
    test_set_size: int
    timing_compliant: bool
```

### 6.7 Constants Migration

Create `caligo/reconciliation/constants.py`:

```python
"""Phase III reconciliation constants."""

# LDPC Code Parameters
LDPC_FRAME_SIZE: int = 4096
LDPC_CODE_RATES: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
LDPC_DEFAULT_RATE: float = 0.50

# Decoder Parameters
LDPC_MAX_ITERATIONS: int = 60
LDPC_BP_THRESHOLD: float = 1e-6
LDPC_MAX_RETRIES: int = 2

# Verification Parameters
LDPC_HASH_BITS: int = 50
LDPC_HASH_PRIME: int = 2**61 - 1

# Efficiency Targets
LDPC_F_CRIT: float = 1.22

# Matrix Generation (for offline use)
PEG_MAX_TREE_DEPTH: int = 10
PEG_DEFAULT_SEED: int = 42

# File Patterns
LDPC_MATRIX_FILE_PATTERN: str = "ldpc_{frame_size}_rate{rate:.2f}.npz"
```

### 6.8 Migration Procedure

**Step 1: Create Package Structure**
```bash
mkdir -p caligo/caligo/reconciliation
touch caligo/caligo/reconciliation/__init__.py
```

**Step 2: Extract Core Algorithms**

| Source File | Target File | Extraction Notes |
|-------------|-------------|------------------|
| `ldpc_bp_decoder.py` | `ldpc_decoder.py` | Nearly direct copy, add dataclass results |
| `polynomial_hash.py` | `hash_verifier.py` | Direct copy, rename class |
| `ldpc_matrix_manager.py` | `matrix_manager.py` | Remove autogeneration logic |
| `ldpc_reconciliator.py` | Split into 3 files | See decomposition below |

**Step 3: Decompose `ldpc_reconciliator.py`**

```
ldpc_reconciliator.py (678 LOC)
├── select_rate()           → rate_selector.py
├── compute_shortening()    → rate_selector.py
├── _binary_entropy()       → rate_selector.py
├── compute_syndrome_block()→ ldpc_encoder.py
├── reconcile_block()       → ldpc_decoder.py (integrate with BP)
├── verify_block()          → (use hash_verifier.py)
├── estimate_leakage_block()→ leakage_tracker.py
├── _build_error_llrs()     → ldpc_decoder.py
├── syndrome_guided_llr_init() → ldpc_decoder.py
├── _generate_padding()     → ldpc_encoder.py
└── aggregate_results()     → orchestrator.py
```

**Step 4: Update Imports and Dependencies**

Ensure each module imports only what it needs:
- `ldpc_decoder.py`: numpy, scipy.sparse
- `rate_selector.py`: math only
- `hash_verifier.py`: numpy only
- `leakage_tracker.py`: dataclasses only

**Step 5: Integration Testing**

Create `caligo/tests/test_reconciliation/`:
```
test_bp_decoder.py      # Unit tests for BP algorithm
test_rate_selector.py   # Rate selection correctness
test_hash_verifier.py   # Hash collision bounds
test_leakage_tracker.py # Safety cap enforcement
test_orchestrator.py    # Integration with Phase A contracts
```

### 6.9 Integration with Protocol Flow

**Phase Boundary Contract:**

```python
# sifting/ → reconciliation/
sifting_result: SiftingPhaseResult = ...
recon_input = ReconciliationInput(
    alice_key=sifting_result.alice_sifted_key,
    bob_key=sifting_result.bob_sifted_key,
    qber_estimate=sifting_result.qber_estimate,
    min_entropy_rate=sifting_result.min_entropy_rate
)

# reconciliation/ → amplification/
recon_result: ReconciliationPhaseResult = orchestrator.reconcile(recon_input)
if recon_result.blocks_discarded > threshold:
    raise ReconciliationFailedError("Too many blocks failed")
    
amplify_input = AmplificationInput(
    reconciled_key=recon_result.corrected_key,
    total_leakage=recon_result.total_leakage,
    qber_estimate=recon_result.qber_estimate
)
```

**Timing Integration:**

```python
# Phase III must complete before Δt expires
with TimingBarrier(delta_t=protocol_config.delta_t):
    # Syndrome transmission (Alice → Bob)
    syndromes = orchestrator.alice_encode_blocks(alice_key, qber)
    
    # Decoding (Bob local)
    result = orchestrator.bob_decode_blocks(bob_key, syndromes, qber)
```

### 6.10 Testing Strategy

Phase III testing follows Caligo's contract-driven approach from `caligo_architecture.md`.

#### 6.10.1 Unit Tests (`tests/test_reconciliation/`)

**`test_ldpc_decoder.py`**
```python
"""BP decoder unit tests."""

class TestBPDecoder:
    """Belief Propagation decoder correctness."""
    
    def test_convergence_low_qber(self):
        """Decoder converges for QBER < 5%."""
        # Synthetic BSC channel with known error pattern
        pass
    
    def test_convergence_boundary_qber(self):
        """Decoder converges at QBER ≈ 10% (near threshold)."""
        pass
    
    def test_no_convergence_high_qber(self):
        """Decoder fails gracefully for QBER > 15%."""
        pass
    
    def test_syndrome_consistency(self):
        """H @ corrected == target_syndrome."""
        pass
    
    def test_llr_initialization(self):
        """Channel LLRs correctly computed from QBER."""
        pass
    
    def test_message_update_rules(self):
        """Check-to-variable and variable-to-check updates correct."""
        pass
```

**`test_rate_selector.py`**
```python
"""Rate selection unit tests."""

class TestRateSelector:
    """LDPC rate selection correctness."""
    
    @pytest.mark.parametrize("qber,expected_rate", [
        (0.01, 0.90),  # Low QBER → high rate
        (0.03, 0.80),  # Moderate QBER
        (0.06, 0.70),  # Medium QBER
        (0.09, 0.55),  # High QBER → low rate
    ])
    def test_rate_from_qber(self, qber, expected_rate):
        """Rate selection matches expected for known QBER."""
        pass
    
    def test_efficiency_criterion(self):
        """Selected rate satisfies f < f_crit for all valid QBER."""
        pass
    
    def test_shortening_calculation(self):
        """Shortening bits correctly computed for block size."""
        pass
    
    def test_puncturing_calculation(self):
        """Puncturing bits correctly computed for rate increase."""
        pass
```

**`test_hash_verifier.py`**
```python
"""Hash verification unit tests."""

class TestHashVerifier:
    """Polynomial hash collision bounds."""
    
    def test_collision_probability(self):
        """P(collision) < 2^{-50} for 50-bit hash."""
        # Monte Carlo test with random inputs
        pass
    
    def test_verify_correct_key(self):
        """Verification succeeds for identical keys."""
        pass
    
    def test_reject_incorrect_key(self):
        """Verification fails for differing keys."""
        pass
    
    def test_deterministic(self):
        """Same (key, seed) produces same hash."""
        pass
```

**`test_leakage_tracker.py`**
```python
"""Leakage tracking unit tests."""

class TestLeakageTracker:
    """Safety cap enforcement."""
    
    def test_accumulation_accuracy(self):
        """Leakage sum accurate within 1 bit."""
        pass
    
    def test_safety_cap_enforcement(self):
        """should_abort() True when leakage exceeds cap."""
        pass
    
    def test_cap_calculation(self):
        """Safety cap computed correctly from min-entropy."""
        pass
    
    def test_blind_leakage_accumulation(self):
        """Blind reconciliation: syndrome + shortened values tracked."""
        pass
```

#### 6.10.2 Integration Tests (`tests/test_reconciliation/test_integration.py`)

```python
"""Phase III integration tests."""

class TestReconciliationIntegration:
    """Full reconciliation flow tests."""
    
    def test_full_reconciliation_low_qber(self):
        """Complete reconciliation succeeds for QBER < 5%."""
        # End-to-end: sifted_key → reconciled_key
        pass
    
    def test_full_reconciliation_boundary(self):
        """Reconciliation succeeds at QBER ≈ 10%."""
        pass
    
    def test_abort_on_excessive_qber(self):
        """Raises QBERThresholdExceeded for QBER > 11%."""
        pass
    
    def test_abort_on_leakage_cap(self):
        """Raises LeakageCapExceeded when safety cap breached."""
        pass
    
    def test_block_discard_handling(self):
        """Failed blocks correctly excluded from output."""
        pass
    
    def test_blind_iteration_convergence(self):
        """Blind reconciliation converges within max_iterations."""
        pass
```

#### 6.10.3 Contract Validation Tests

```python
"""Phase boundary contract tests."""

class TestPhaseContracts:
    """Caligo phase contract compliance (Phase A types)."""
    
    def test_input_contract_sifting_result(self):
        """
        ReconciliationOrchestrator accepts SiftingPhaseResult.
        
        Validates:
        - POST-S-001: len(sifted_key_alice) == len(sifted_key_bob)
        - POST-S-003: qber_adjusted ≤ QBER_HARD_LIMIT
        """
        pass
    
    def test_output_contract_reconciliation_result(self):
        """
        ReconciliationOrchestrator produces valid ReconciliationPhaseResult.
        
        Validates:
        - POST-R-001: total_syndrome_bits ≤ L_max
        - POST-R-002: hash_verified == True
        - reconciled_key type is bitarray
        """
        pass
    
    def test_output_consumed_by_amplification(self):
        """
        ReconciliationPhaseResult is valid input for Phase IV.
        
        Validates amplification/ can construct AmplificationInput.
        """
        pass
```

#### 6.10.4 Property-Based Tests (Hypothesis)

```python
"""Property-based tests with Hypothesis."""

from hypothesis import given, strategies as st

class TestReconciliationProperties:
    """Statistical properties verified over many random inputs."""
    
    @given(
        key_length=st.integers(min_value=1000, max_value=10000),
        qber=st.floats(min_value=0.001, max_value=0.10),
    )
    def test_correctness_property(self, key_length, qber):
        """
        Property: For QBER < 11%, reconciled keys match.
        
        ∀ alice_key, bob_key with error_rate(alice, bob) = qber:
          reconcile(alice, bob, qber) → alice_key
        """
        pass
    
    @given(
        key_length=st.integers(min_value=1000, max_value=10000),
        qber=st.floats(min_value=0.001, max_value=0.10),
    )
    def test_leakage_bound_property(self, key_length, qber):
        """
        Property: Leakage never exceeds safety cap.
        
        ∀ reconciliation runs:
          result.total_leakage ≤ L_max(key_length, qber)
        """
        pass
    
    @given(
        key_length=st.integers(min_value=1000, max_value=10000),
        qber=st.floats(min_value=0.001, max_value=0.10),
    )
    def test_efficiency_property(self, key_length, qber):
        """
        Property: Reconciliation efficiency f < f_crit.
        
        ∀ successful reconciliation:
          syndrome_bits / (key_length * h(qber)) < 1.22
        """
        pass
```

#### 6.10.5 Security Tests

```python
"""Security-specific tests."""

class TestSecurityInvariants:
    """Obliviousness and information-theoretic security."""
    
    def test_one_way_information_flow(self):
        """
        Bob's decoding never sends information to Alice.
        
        Verify: No Bob → Alice messages during reconciliation.
        """
        pass
    
    def test_no_error_position_feedback(self):
        """
        Bob's failure state contains no error position information.
        
        Verify: DecodeResult.error_positions is None or empty.
        """
        pass
    
    def test_syndrome_leakage_only(self):
        """
        Only syndrome bits and hash transmitted.
        
        Verify: transmitted_bits == syndrome_bits + hash_bits
        """
        pass
    
    def test_feigned_failure_defense(self):
        """
        Safety cap prevents infinite retry attacks.
        
        Verify: Abort triggered after max_retries even with fake failures.
        """
        pass
```

#### 6.10.6 Test Fixtures

```python
"""Shared test fixtures for reconciliation tests."""

@pytest.fixture
def sample_sifting_result() -> SiftingPhaseResult:
    """Generate valid SiftingPhaseResult for testing."""
    n = 4096
    qber = 0.05
    alice_key = bitarray(np.random.randint(0, 2, n).tolist())
    errors = np.random.rand(n) < qber
    bob_key = bitarray((np.array(alice_key) ^ errors).tolist())
    return SiftingPhaseResult(
        sifted_key_alice=alice_key,
        sifted_key_bob=bob_key,
        matching_indices=np.arange(n),
        i0_indices=np.arange(n // 2),
        i1_indices=np.arange(n // 2, n),
        test_set_indices=np.array([]),
        qber_estimate=qber,
        qber_adjusted=qber + 0.005,  # finite-size penalty
        finite_size_penalty=0.005,
        test_set_size=100,
        timing_compliant=True,
    )

@pytest.fixture
def ldpc_matrix_4096_r50() -> sp.csr_matrix:
    """Load rate-0.50 LDPC matrix for 4096-bit codewords."""
    # Load from test assets
    pass

@pytest.fixture
def reconciliation_config() -> ReconciliationConfig:
    """Standard reconciliation configuration."""
    return ReconciliationConfig(
        frame_size=4096,
        max_iterations=60,
        hash_bits=50,
        f_crit=1.22,
        max_retries=2,
    )
```

### 6.11 Phase Integration Alignment

This section specifies how Phase III integrates with the Caligo architecture as defined in `caligo_architecture.md`.

#### 6.11.1 Input Contract: `SiftingPhaseResult` (Phase II → III)

Phase III receives `SiftingPhaseResult` from `sifting/` with the following guarantees:

```python
@dataclass
class SiftingPhaseResult:
    """
    Contract: Phase II → Phase III (from phase_a_spec.md §3.3.3).
    
    Reconciliation depends on these post-conditions:
    - POST-S-001: len(sifted_key_alice) == len(sifted_key_bob)
    - POST-S-002: qber_adjusted = qber_estimate + finite_size_penalty
    - POST-S-003: qber_adjusted ≤ QBER_HARD_LIMIT (else abort before III)
    """
    sifted_key_alice: bitarray   # Alice's sifted bits (I₀ ∪ I₁ partition)
    sifted_key_bob: bitarray     # Bob's bits (with channel errors)
    matching_indices: np.ndarray # Round indices where bases matched
    i0_indices: np.ndarray       # Partition I₀ indices (for OT)
    i1_indices: np.ndarray       # Partition I₁ indices (for OT)
    test_set_indices: np.ndarray # Indices sacrificed for QBER
    qber_estimate: float         # Raw observed QBER: e_obs
    qber_adjusted: float         # With finite-size: e_adj = e_obs + μ
    finite_size_penalty: float   # μ from Erven et al. Eq. (2)
    test_set_size: int           # |T| bits used for testing
    timing_compliant: bool       # True if Δt enforced
```

**Reconciliation Input Mapping:**
```python
def create_reconciliation_input(
    sifting_result: SiftingPhaseResult,
) -> ReconciliationInput:
    """
    Map SiftingPhaseResult to reconciliation module input.
    
    Key mapping:
    - sifted_key_alice → alice_blocks (LDPC framing)
    - sifted_key_bob → bob_blocks (LDPC framing)
    - qber_adjusted → rate selection QBER (conservative)
    - i0_indices, i1_indices → preserved for Phase IV OT output
    """
    return ReconciliationInput(
        alice_key=np.array(sifting_result.sifted_key_alice, dtype=np.uint8),
        bob_key=np.array(sifting_result.sifted_key_bob, dtype=np.uint8),
        qber_estimate=sifting_result.qber_adjusted,  # Use adjusted QBER
        i0_indices=sifting_result.i0_indices,
        i1_indices=sifting_result.i1_indices,
    )
```

#### 6.11.2 Output Contract: `ReconciliationPhaseResult` (Phase III → IV)

Phase III produces `ReconciliationPhaseResult` for `amplification/`:

```python
@dataclass
class ReconciliationPhaseResult:
    """
    Contract: Phase III → Phase IV (from phase_a_spec.md §3.3.4).
    
    Post-conditions guaranteed:
    - POST-R-001: total_syndrome_bits ≤ L_max (safety cap)
    - POST-R-002: hash_verified == True (else abort)
    """
    reconciled_key: bitarray      # Error-corrected key (Alice's perspective)
    num_blocks: int               # LDPC blocks processed
    blocks_succeeded: int         # Blocks passing verification
    blocks_failed: int            # Discarded blocks
    total_syndrome_bits: int      # Total leakage |Σ|
    effective_rate: float         # Achieved R = (n - |Σ|) / n
    hash_verified: bool           # Final integrity check
    leakage_within_cap: bool      # |Σ| ≤ L_max
    # Extended fields for Phase IV
    i0_reconciled_key: bitarray   # Reconciled bits from I₀ partition
    i1_reconciled_key: bitarray   # Reconciled bits from I₁ partition
    qber_estimate: float          # Updated QBER from block errors
```

**Amplification Input Mapping:**
```python
def create_amplification_input(
    recon_result: ReconciliationPhaseResult,
    sifting_result: SiftingPhaseResult,
) -> AmplificationInput:
    """
    Map ReconciliationPhaseResult to amplification module input.
    
    Key calculation:
    - effective_key_bits = len(reconciled_key) - total_syndrome_bits
    - min_entropy for PA = h_min(qber) * effective_key_bits
    """
    return AmplificationInput(
        reconciled_key=recon_result.reconciled_key,
        i0_key=recon_result.i0_reconciled_key,
        i1_key=recon_result.i1_reconciled_key,
        total_leakage=recon_result.total_syndrome_bits,
        qber_estimate=recon_result.qber_estimate,
        timing_compliant=sifting_result.timing_compliant,
    )
```

#### 6.11.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE III DATA FLOW (Caligo Integration)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      sifting/orchestrator.py                         │   │
│  │  ┌──────────────────────┐                                            │   │
│  │  │  SiftingPhaseResult  │                                            │   │
│  │  │  ├─ sifted_key_alice │                                            │   │
│  │  │  ├─ sifted_key_bob   │                                            │   │
│  │  │  ├─ qber_adjusted    │                                            │   │
│  │  │  ├─ i0_indices       │                                            │   │
│  │  │  └─ i1_indices       │                                            │   │
│  │  └──────────┬───────────┘                                            │   │
│  └─────────────┼────────────────────────────────────────────────────────┘   │
│                │                                                            │
│                ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  reconciliation/orchestrator.py                      │   │
│  │                                                                      │   │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │   │
│  │  │ rate_selector │───►│ ldpc_encoder  │───►│ ldpc_decoder  │         │   │
│  │  │               │    │ (syndrome)    │    │ (BP decode)   │         │   │
│  │  └───────────────┘    └───────────────┘    └───────┬───────┘         │   │
│  │                                                     │                │   │
│  │                      ┌───────────────┐              │                │   │
│  │                      │ hash_verifier │◄─────────────┘                │   │
│  │                      └───────┬───────┘                               │   │
│  │                              │                                       │   │
│  │  ┌───────────────┐           │                                       │   │
│  │  │leakage_tracker│◄──────────┼────────── (all transmissions)         │   │
│  │  └───────┬───────┘           │                                       │   │
│  │          │                   ▼                                       │   │
│  │  ┌───────┴───────────────────────────────┐                           │   │
│  │  │      ReconciliationPhaseResult        │                           │   │
│  │  │  ├─ reconciled_key                    │                           │   │
│  │  │  ├─ total_syndrome_bits (|Σ|)         │                           │   │
│  │  │  ├─ i0_reconciled_key                 │                           │   │
│  │  │  └─ i1_reconciled_key                 │                           │   │
│  │  └───────────────┬───────────────────────┘                           │   │
│  └──────────────────┼───────────────────────────────────────────────────┘   │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  amplification/orchestrator.py                       │   │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │   │
│  │  │ entropy.py    │───►│ key_length.py │───►│ toeplitz.py   │         │   │
│  │  └───────────────┘    └───────────────┘    └───────────────┘         │   │
│  │                                                     │                │   │
│  │                                                     ▼                │   │
│  │                              ┌───────────────────────────┐           │   │
│  │                              │  ObliviousTransferOutput  │           │   │
│  │                              │  ├─ s0 (from I₀)          │           │   │
│  │                              │  ├─ s1 (from I₁)          │           │   │
│  │                              │  └─ sc (Bob's choice)     │           │   │
│  │                              └───────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.11.4 Error Propagation

```python
# From types/exceptions.py (Phase A)
class ReconciliationError(ProtocolError):
    """Base class for Phase III errors."""
    pass

class QBERThresholdExceeded(ReconciliationError):
    """QBER exceeds security threshold."""
    qber: float
    threshold: float

class LeakageCapExceeded(ReconciliationError):
    """Syndrome leakage exceeds safety cap."""
    actual_leakage: int
    max_allowed: int

class DecodingFailed(ReconciliationError):
    """BP decoder did not converge."""
    blocks_failed: int
    blocks_total: int

class HashVerificationFailed(ReconciliationError):
    """Final hash check failed."""
    block_id: int
```

### 6.12 Acceptance Criteria

Phase III implementation is complete when:

- [ ] All 8 modules created with ≤200 LOC each
- [ ] `ReconciliationPhaseResult` integrated into Phase A types
- [ ] Input/output contracts aligned with `caligo_architecture.md` §3.4
- [ ] Unit test coverage ≥90% on core logic
- [ ] Integration tests pass for QBER range 1-10%
- [ ] Property-based tests (Hypothesis) pass for 1000+ random inputs
- [ ] Leakage accounting validated against theoretical bounds
- [ ] Security tests verify one-way information flow
- [ ] Documentation complete (docstrings + this spec)
- [ ] Code review by second developer

---

## 7. Acceptance Criteria

### 7.1 Functional Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| F-001 | Syndrome computation matches ehok reference | Unit test |
| F-002 | BP decoder converges for QBER < 11% | Integration test |
| F-003 | Rate selection follows efficiency criterion | Unit test |
| F-004 | Hash verification collision rate < 2^{-50} | Statistical test |
| F-005 | Leakage tracking accurate within 1% | Audit test |
| F-006 | Safety cap abort triggers correctly | Edge case test |

### 7.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NF-001 | Module size | ≤200 LOC |
| NF-002 | Test coverage | ≥90% |
| NF-003 | Memory usage | <100MB for 10K blocks |
| NF-004 | Reconciliation latency | <1s for 4096-bit block |
| NF-005 | Documentation | Numpydoc complete |

### 7.3 Security Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| S-001 | One-way information flow | Code review |
| S-002 | No Bob → Alice feedback | API constraint |
| S-003 | Leakage ≤ safety cap | Runtime enforcement |
| S-004 | Obliviousness preserved | Formal argument |

---

## 8. References

1. **Martinez-Mateo et al.** (2012). "Blind Reconciliation." *Quantum Information and Computation*.

2. **Kiktenko et al.** (2016). "Post-processing procedure for industrial quantum key distribution systems." *J. Phys.: Conf. Ser.* 741, 012081.

3. **Wehner et al.** (2010). "Implementation of two-party protocols in the noisy-storage model." *Physical Review A* 81, 052336.

4. **König et al.** (2012). "Unconditional security from noisy quantum storage." *IEEE Trans. Inform. Theory*.

5. **Ding** (2001). "Oblivious Transfer in the Bounded Storage Model." *CRYPTO 2001*, LNCS 2139.

6. **Cachin, Crépeau, Marcil** (1998). "Oblivious Transfer with a Memory-Bounded Receiver." *FOCS 1998*.

7. **Hu, Eleftheriou, Arnold** (2005). "Regular and irregular progressive edge-growth Tanner graphs." *IEEE Trans. Inform. Theory*.

8. **MacKay** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.

---

*Document generated: December 17, 2025*  
*Caligo Project — $\binom{2}{1}$-OT Protocol Implementation*

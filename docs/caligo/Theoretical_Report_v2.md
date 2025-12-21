# Theoretical Report v2: Advanced Rate-Compatible Reconciliation in Caligo

**Version:** 2.0  
**Date:** December 2025  
**Subject:** Rigorous Mathematical Framework for Baseline and Blind Reconciliation Protocols  
**Context:** $\binom{2}{1}$-Oblivious Transfer via Noisy Storage Model (NSM)

**Primary References:**
1. Elkouss, D., Martinez-Mateo, J., Lancho, D., & Martin, V. (2010). "Rate Compatible Protocol for Information Reconciliation: An Application to QKD."
2. Martinez-Mateo, J., Elkouss, D., & Martin, V. (2012). "Blind Reconciliation." *Quantum Information and Computation*, Vol. 12, No. 9&10, pp. 791-812.
3. Elkouss, D., Martinez-Mateo, J., & Martin, V. (2012). "Untainted Puncturing for Irregular Low-Density Parity-Check Codes." *IEEE Wireless Communications Letters*, Vol. 1, No. 6, pp. 585-588.

---

## 1. Introduction: The Entropy Economy Under the Noisy Storage Model

### 1.1 Problem Statement

In the Caligo protocol, Phase III (Information Reconciliation) is the critical juncture where error correction meets security. Unlike standard QKD, where syndrome information leaks to a passive eavesdropper (Eve), in our $\binom{2}{1}$-Oblivious Transfer protocol under the Noisy Storage Model (NSM), this information leaks directly to the receiver (Bob), who is a potential adversary. Under the NSM security framework, *minimizing the syndrome length* $|\Sigma|$ is paramount to maximizing the extractable secure OT output.

This report formalizes the theoretical foundations for two distinct reconciliation strategies:
- **Baseline** (Elkouss et al., 2010): Rate-compatible protocol requiring *a priori* QBER estimation.
- **Blind** (Martinez-Mateo et al., 2012): Rate-adaptive protocol without QBER pre-estimation, using iterative bit revelation.

Both strategies are unified under a single architectural foundation: a **Rate-Compatible Mother Code** utilizing **Untainted Puncturing** (Elkouss et al., 2012).

### 1.2 NSM Finite-Size Key Rate Equation

Under the Noisy Storage Model, the security of the $\binom{2}{1}$-OT protocol relies on bounding the adversary's (Bob's) information about Alice's inputs. The extractable secure key length after reconciliation is governed by the finite-size key rate equation:

$$
\ell \leq n \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{EC}} - \log_2\left(\frac{2}{\epsilon^2}\right) \right]
$$

where:
- $n$ is the number of raw bits after sifting
- $H_{\min}^{\epsilon}(X|E)$ is the smooth min-entropy of Alice's string conditioned on the adversary's quantum side information
- $\text{leak}_{\text{EC}}$ is the total leakage from error correction
- $\epsilon$ is the security parameter

The **critical constraint** is that $\text{leak}_{\text{EC}}$ must be precisely bounded. For syndrome-based reconciliation:

$$
\text{leak}_{\text{EC}} = |\Sigma| + |\text{Hash}| + |\text{Revealed}|
$$

where:
- $|\Sigma| = (1-R_{\text{eff}}) \cdot n$ is the syndrome length (in bits)
- $|\text{Hash}|$ is the verification hash length (typically 32-128 bits)
- $|\text{Revealed}|$ is the total revealed bits during blind iterations

The **reconciliation efficiency** $f$ is defined as:

$$
f = \frac{\text{leak}_{\text{EC}}}{n \cdot h(\text{QBER})}
$$

where $h(\cdot)$ is the binary entropy function. Perfect reconciliation achieves $f = 1$; practical protocols operate at $f \in [1.05, 1.2]$.

**NSM Security Requirement:** For the OT protocol to remain secure, the total leakage must satisfy:

$$
\text{leak}_{\text{EC}} < H_{\min}^{\epsilon}(X|E) - \kappa
$$

where $\kappa$ is a security margin accounting for finite-size effects and the leftover hash lemma.

---

## 2. Unified Foundation: Rate-Compatible Mother Code with Untainted Puncturing

### 2.1 Theoretical Framework for Rate Adaptation

The naive description of syndrome computation—"Alice computes $s = H_{\text{mother}} \cdot x_{\text{frame}}$"—obscures the sophisticated rate-adaptation mechanism. In the rate-compatible framework, the mother code $\mathcal{C}_{R_0}$ serves as a structural parent; the *effective code* $\mathcal{C}_{\text{eff}}$ is constructed dynamically by manipulating the input frame.

**Definition 2.1 (Effective Rate).** Given a mother code with rate $R_0$, frame size $n$, $p$ punctured bits, and $s$ shortened bits, the effective rate is:

$$
R_{\text{eff}} = \frac{R_0 - s/n}{1 - p/n - s/n} = \frac{k - s}{n - p - s}
$$

where $k = R_0 \cdot n$ is the mother code dimension.

**Definition 2.2 (Modulation Parameter).** The total modulation parameter $\delta = (p + s)/n$ determines the range of achievable rates:

$$
R_{\min} = \frac{R_0 - \delta}{1 - \delta} \leq R_{\text{eff}} \leq \frac{R_0}{1 - \delta} = R_{\max}
$$

**Theorem 2.1 (Rate Coverage).** For a mother code with $R_0 = 0.5$ and $\delta = 0.1$:
- $R_{\min} = (0.5 - 0.1)/(1 - 0.1) = 0.444$
- $R_{\max} = 0.5/(1 - 0.1) = 0.556$

This allows correction of QBER in the range $[\epsilon_{\min}, \epsilon_{\max}]$ where $\epsilon_{\max}$ corresponds to $R_{\min}$ and $\epsilon_{\min}$ corresponds to $R_{\max}$.

### 2.2 Untainted Puncturing: Formal Definition and Algorithm

Random puncturing leads to decoding failures at high effective rates due to "stopping sets"—configurations where the decoder lacks sufficient information to recover punctured symbols. The **Untainted Puncturing** algorithm (Elkouss et al., 2012) prevents this by ensuring structural properties of the punctured code.

#### 2.2.1 Graph-Theoretic Definitions

**Definition 2.3 (Neighborhood).** Let $\mathcal{N}(z)$ denote the neighborhood of node $z$ in the Tanner graph (nodes adjacent to $z$). The depth-$k$ neighborhood $\mathcal{N}^k(z)$ includes all nodes reachable by traversing at most $k$ edges.

**Definition 2.4 (k-Step Extended Recoverable Symbols).** Let $\mathcal{P}$ be the set of punctured symbols. We define:

1. **1-Step Recoverable ($\mathcal{R}_1$):** A punctured symbol $v \in \mathcal{P}$ belongs to $\mathcal{R}_1$ if $\exists c \in \mathcal{N}(v)$ such that $\forall w \in \mathcal{N}(c) \setminus \{v\}, w \notin \mathcal{P}$.
   
   *Interpretation:* There exists a check node $c$ connected to $v$ where all other connected symbols are unpunctured. This check provides full information to recover $v$ in one decoding iteration.

2. **k-Step Recoverable ($\mathcal{R}_k$, $k > 1$):** A punctured symbol $v \notin \bigcup_{i=1}^{k-1} \mathcal{R}_i$ belongs to $\mathcal{R}_k$ if $\exists c \in \mathcal{N}(v)$ and $\exists w \in \mathcal{N}(c) \setminus \{v\}$ such that $w \in \mathcal{R}_{k-1}$ and $\forall w' \in \mathcal{N}(c) \setminus \{v, w\}, w' \in \mathcal{P} \Rightarrow w' \in \bigcup_{i=1}^{k-1} \mathcal{R}_i$.

**Definition 2.5 (Survived vs. Dead Check Nodes).** Given a punctured symbol $v$:
- A check $c \in \mathcal{N}(v)$ is a **survived check node** if $\exists w \in \mathcal{N}(c) \setminus \{v\}$ such that $w \in \mathcal{R}_{k-1}$ (provides recovery information).
- A check $c \in \mathcal{N}(v)$ is a **dead check node** if it provides zero LLR to $v$ (all neighbors are unrecoverable punctured symbols).

**Definition 2.6 (Untainted Symbol).** A symbol node $v$ is **untainted** if there are no punctured symbols within $\mathcal{N}^2(v)$.

*Interpretation:* An untainted symbol has no punctured symbols among itself, its neighboring checks, or any other symbols connected to those checks. Puncturing only untainted symbols ensures that **all check nodes of a selected symbol are survived check nodes**.

#### 2.2.2 The Untainted Puncturing Algorithm

```
ALGORITHM: Untainted Puncturing Pattern Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:  Tanner graph G = (V_s ∪ V_c, E), target proportion π
OUTPUT: Puncturing pattern P ⊆ V_s

INITIALIZE:
    𝒳_∞ ← V_s                          // All symbol nodes are initially untainted
    P ← ∅                               // Empty puncturing set
    p ← 0                               // Puncture counter

WHILE 𝒳_∞ ≠ ∅ AND |P|/n < π_max DO:
    
    STEP 1 — Candidate Selection:
        Ω ← {u ∈ 𝒳_∞ : |𝒩²(u)| ≤ |𝒩²(v)| ∀v ∈ 𝒳_∞}
        // Select symbols with smallest depth-2 neighborhood
        // For regular check degree, this simplifies to lowest-degree symbols
    
    STEP 2 — Puncture Selection:
        v^(p) ← SELECT_RANDOM(Ω)        // Random tie-breaking
        P ← P ∪ {v^(p)}
    
    STEP 3 — Update Untainted Set:
        𝒳_∞ ← 𝒳_∞ \ 𝒩²(v^(p))          // Remove all symbols in depth-2 neighborhood
        p ← p + 1

END WHILE

RETURN P
```

**Key Property:** The untainted algorithm guarantees that every punctured symbol has multiple survived check nodes, enabling robust decoding even at high puncturing rates.

#### 2.2.3 Why Untainted Puncturing Prevents Stopping Sets

**Theorem 2.2 (Extended Recovery Tree Improvement).** Let $T_{v_1}, T_{v_2}$ be extended recovery trees for punctured symbols $v_1, v_2 \in \mathcal{R}_k$ that are identical except $v_2$ has an additional survived check node. Then for any binary input symmetric output memoryless channel $\mathcal{C}$:

$$
P_e(v_1) \geq P_e(v_2)
$$

where $P_e(v)$ is the extended recovery error probability.

*Proof Sketch (from Elkouss et al., 2012):* For the BEC, the proof follows from induction on erasure probabilities: adding a survived check strictly reduces the product term in the erasure recursion. For general symmetric channels, the proof uses stochastic degradation arguments and the fact that additional information cannot increase maximum likelihood decoding error.

**Corollary 2.1.** The untainted puncturing algorithm maximizes the number of survived check nodes per punctured symbol, minimizing stopping set formation.

**Practical Implication:** Our current implementation uses random puncturing via `generate_padding` in `ldpc_encoder.py`, which fails to preserve the untainted property. This explains the convergence failures observed at high rates (e.g., $R_{\text{eff}} > 0.75$) documented in `rate_selector.py`.

---

## 3. Baseline Reconciliation Protocol (Elkouss)

The Baseline protocol is an "inverse puncturing and shortening" scheme requiring *a priori* QBER estimation for single-shot rate optimization.

### 3.1 Phase II: Parameter Estimation

Before reconciliation, Alice and Bob estimate the channel crossover probability from a test subset:

$$
p^* = \frac{|\{i : x_i \neq y_i, i \in \text{TestSet}\}|}{|\text{TestSet}|}
$$

This estimate is refined with finite-size corrections to obtain the adjusted QBER $\hat{p}$.

### 3.2 Phase III: Rate Selection and Frame Construction

**Step 1: Rate Selection.** Using $\hat{p}$, Alice selects the target rate satisfying the efficiency criterion:

$$
R = 1 - f \cdot h(\hat{p})
$$

where $f$ is the target efficiency (typically $f \approx 1.1$ for practical codes).

**Step 2: Compute Modulation Parameters.** Given $\delta$ and $R$:

$$
s = \left\lceil \left( R_0 - R(1 - \delta) \right) \cdot n \right\rceil, \quad p = \lfloor \delta \cdot n \rfloor - s
$$

**Step 3: Frame Construction.** Alice constructs frame $\mathbf{x}^+ = g(\mathbf{x}, \sigma, \pi)$:
1. Payload bits $\mathbf{x}$ (length $n - \delta \cdot n$) fill the **transmission positions**
2. $p$ **punctured positions** are filled with deterministic pseudo-random padding (unknown to Bob as channel input, but position is known)
3. $s$ **shortened positions** are filled with deterministic known values (both value and position known to Bob)

**Step 4: Syndrome Computation.**

$$
\mathbf{s} = H_{\text{mother}} \cdot \mathbf{x}^+ \mod 2
$$

**Step 5: Transmission.** Alice sends $(s, p^*, \text{pattern\_index})$ to Bob.

### 3.3 Decoder Initialization (Bob)

Bob constructs his frame $\mathbf{y}^+ = g(\mathbf{y}, \sigma_{p^*}, \pi_{p^*})$ and initializes LLRs:

| Position Type | Channel LLR Value |
|---------------|-------------------|
| **Payload bits** | $\gamma_i = \ln\frac{1 - p^*}{p^*} \cdot (1 - 2y_i)$ |
| **Punctured bits** | $\gamma_i = 0$ (Erasure: no information) |
| **Shortened bits** | $\gamma_i = \pm\infty$ (Perfect knowledge) |

The decoder runs standard belief propagation against syndrome $\mathbf{s}$.

### 3.4 Leakage Analysis

**Theorem 3.1 (Baseline Leakage Bound).** For Baseline reconciliation:

$$
\text{leak}_{\text{Baseline}} = |\Sigma| + |\text{Hash}| = (1 - R_0) \cdot n + h
$$

where $h$ is the hash length. The leakage is **constant per block** and independent of the effective rate $R_{\text{eff}}$ (since we always use the mother code matrix).

---

## 4. Blind Reconciliation Protocol (Martinez-Mateo)

The Blind protocol eliminates QBER pre-estimation through iterative bit revelation.

### 4.1 The Security Argument: Shortening vs. Matrix Modification

**Critical Design Question:** Should rate adaptation modify the parity-check matrix (add rows) or modify the puncturing pattern?

**Answer:** The protocol **must use iterative shortening** (converting punctured bits to shortened bits) while retaining the original mother matrix.

**Theorem 4.1 (Syndrome Reuse Security).** In the Blind protocol, the syndrome $\mathbf{s} = H \cdot \mathbf{x}^+$ computed in iteration 1 can be reused in all subsequent iterations without leaking additional information about **unrevealed** punctured bits.

*Proof:*

1. **Information-Theoretic Argument:** Let $\mathbf{x}^+ = [\mathbf{x}_{\text{payload}}, \mathbf{x}_{\text{punct}}, \mathbf{x}_{\text{short}}]$ be the frame partition. The syndrome satisfies:

   $$
   \mathbf{s} = H_{\text{payload}} \cdot \mathbf{x}_{\text{payload}} + H_{\text{punct}} \cdot \mathbf{x}_{\text{punct}} + H_{\text{short}} \cdot \mathbf{x}_{\text{short}} \mod 2
   $$

2. **Iteration $i$ Information:** After revealing $s_i$ shortened values, Bob knows:
   - Full payload from channel: $\mathbf{y}_{\text{payload}}$
   - Revealed shortened values: $\mathbf{x}_{\text{short}}^{(i)}$
   - Syndrome: $\mathbf{s}$

3. **Unrevealed Punctured Bits:** The remaining punctured bits $\mathbf{x}_{\text{punct}}^{\text{unrev}}$ contribute to $\mathbf{s}$ via $H_{\text{punct}} \cdot \mathbf{x}_{\text{punct}}$. Since Bob has no channel observation for these bits and they were filled with values unknown to Bob, the syndrome equation provides no information about individual punctured values—only their parity combinations.

4. **Key Insight:** Each revealed shortened bit provides exactly 1 bit of information. The syndrome does not amplify this leakage because it was computed once and transmitted once.

**Corollary 4.1.** The total leakage for Blind reconciliation is:

$$
\text{leak}_{\text{Blind}} = |\Sigma| + |\text{Hash}| + |\text{Revealed}| = (1 - R_0) \cdot n + h + \sum_{i=2}^{t} \Delta_i
$$

where $\Delta_i$ is the number of bits revealed in iteration $i$.

**Theorem 4.2 (NSM Security of Blind Iteration).** Under the Noisy Storage Model, the iterative revelation in Blind reconciliation does not compromise the security of the OT protocol, provided:

1. The total leakage $\text{leak}_{\text{Blind}}$ satisfies the NSM budget constraint
2. The revelation pattern is independent of the adversary's storage attack

*Proof Sketch:*

The NSM security proof relies on bounding the adversary's (Bob's) information about Alice's inputs. After the quantum phase and storage attack, Bob's quantum side information $E$ is bounded by the noisy storage capacity.

For the $c$-th OT input (choice bit $c \in \{0,1\}$), Bob's information is:

$$
I(X_{1-c}; E, \Sigma, R) \leq I(X_{1-c}; E) + |\Sigma| + |R|
$$

where $R$ denotes the revealed bits. The first term is bounded by NSM assumptions; the second two terms are exactly tracked by our leakage accounting.

The key observation is that:
1. **Syndrome leakage is linear:** $|\Sigma| = (1-R_0) \cdot n$ regardless of iteration count
2. **Revealed bits provide parity constraints:** Each $\Delta_i$ bits revealed converts uncertainty to certainty at specific positions, which is accounted for
3. **No amplification:** The syndrome was computed once; re-use doesn't leak additional information about unrevealed positions

Therefore, $\text{leak}_{\text{Blind}} = |\Sigma| + |H| + \sum \Delta_i$ is an exact (not approximate) bound, and the NSM security proof carries through with this leakage value. ∎

### 4.2 Information Monotonicity

**Lemma 4.1.** Converting a punctured bit to a payload bit (un-puncturing) provides **no new information** to the decoder, as Bob has no channel observation for padding bits.

**Lemma 4.2.** Converting a punctured bit to a shortened bit provides **infinite information** ($\text{LLR} = \pm\infty$), enabling constraint propagation.

*Interpretation:* The only way to lower the effective code rate after syndrome transmission is to **reveal** (shorten) previously punctured values. This is the fundamental mechanism of the Blind protocol.

### 4.3 Protocol Flow

**Setup:** Fix modulation $\delta$, step size $\Delta = \lfloor \delta \cdot n / t \rfloor$, maximum iterations $t$.

**Iteration 1 (Optimistic Attempt):**
- Configuration: $p = \delta \cdot n$, $s = 0$ (all modulation bits punctured → highest rate)
- Alice sends syndrome $\mathbf{s}$ computed once
- Bob initializes LLRs with $p$ erasures and decodes

**Iteration $i \geq 2$ (On Failure):**
1. Alice selects $\Delta$ bits from punctured set
2. Alice **reveals their values** to Bob (these become shortened)
3. State update: $p \leftarrow p - \Delta$, $s \leftarrow s + \Delta$
4. Bob updates LLRs: revealed positions change from $0 \to \pm\infty$
5. Bob re-runs BP using **original syndrome** and **original matrix**

**Termination:**
- **Success:** Decoder converges and passes verification hash
- **Failure:** $p = 0$ (all bits shortened) and decoder still fails

### 4.4 Average Efficiency Analysis

The average rate achieved by Blind reconciliation depends on the FER distribution across iterations:

$$
\bar{R} = \sum_{i=1}^{t} a_i \cdot r_i
$$

where:
- $a_i = \frac{F^{(i-1)} - F^{(i)}}{1 - F^{(t)}}$ is the fraction of codewords corrected at iteration $i$
- $F^{(i)}$ is the Frame Error Rate when using adapted rate $r_i$
- $r_i = \frac{R_0 - \sigma_i}{1 - \delta}$ with $\sigma_i = (i-1) \cdot \Delta / n$

Using the Gaussian approximation for finite-length LDPC codes:

$$
F(\epsilon, N, \epsilon^*) \approx \int_{\epsilon^*}^{1} \frac{1}{\sqrt{2\pi \epsilon(1-\epsilon)/N}} \exp\left( -\frac{N(x - \epsilon)^2}{2\epsilon(1-\epsilon)} \right) dx
$$

where $\epsilon^*$ is the decoding threshold.

### 4.5 Trade-offs of Step Size $\Delta$

The step size $\Delta$ controls the granularity of rate adaptation in Blind reconciliation. This choice involves fundamental trade-offs between leakage efficiency, decoding probability, and implementation complexity.

#### 4.5.1 Quantitative Analysis

For a mother code with $R_0 = 0.5$, frame size $n$, and modulation parameter $\delta$:

| Step Size | Iterations | Avg. Revealed Bits | Hardware Complexity |
|-----------|------------|-------------------|---------------------|
| $\Delta = 1$ | $d = \delta \cdot n$ | $\mathbb{E}[\sum \Delta_i] \approx \frac{d}{2} \cdot \bar{F}$ | $d$ decoder states |
| $\Delta = d/3$ | 3 | $\mathbb{E}[\sum \Delta_i] \approx \frac{2d}{3} \cdot \bar{F}$ | 3 decoder states |
| $\Delta = d$ | 1 | $d \cdot F^{(1)}$ (if fail) | 1 decoder state |

where $\bar{F}$ is the weighted average FER across iterations.

#### 4.5.2 Leakage-Efficiency Trade-off

The average efficiency of Blind reconciliation depends on the FER distribution:

$$
\bar{f}_{\text{Blind}} = \frac{(1-R_0) \cdot n + h + \mathbb{E}[|\text{Revealed}|]}{n \cdot h(\text{QBER})}
$$

Using Equation (16) from Martinez-Mateo et al.:

$$
\bar{R} \approx R_{\max} - \frac{\beta}{t-1} \sum_{i=1}^{t-1} F^{(i)}
$$

where $\beta = \delta / (1 - \delta)$ and $F^{(i)}$ is the FER at iteration $i$.

**Key Insight:** Smaller $\Delta$ (more iterations) **always improves average efficiency** because each intermediate code has non-zero probability of successful decoding, reducing the expected revealed bits.

#### 4.5.3 Practical Recommendation

For the Caligo NSM protocol, we recommend:

- **$\Delta = d/3$** (3 iterations maximum)
- **$\delta = 0.1$** (10% modulation)

This configuration provides:
1. **Efficiency close to optimal:** Martinez-Mateo demonstrate that 3 iterations approach $\Delta=1$ performance
2. **Tractable hardware:** Only 3 decoder state copies needed
3. **Bounded latency:** Maximum 3 network round-trips per block
4. **Sufficient rate range:** $R_{\text{eff}} \in [0.444, 0.556]$ covers QBER $\in [0.02, 0.10]$

**Recommendation:** For practical implementation, $\Delta = d/3$ provides a good balance. Martinez-Mateo et al. demonstrate that **three iterations approach the efficiency of the maximally interactive case** while maintaining tractable hardware complexity.

---

## 5. Summary: Implementation Requirements

To implement these protocols within Caligo, the following architecture is required:

### 5.1 Data Structures

1. **Single Mother Matrix:** R=0.5 ACE-PEG matrix, stored as sparse CSR
2. **Pattern Library:** Dictionary of `{effective_rate: np.ndarray}` untainted puncturing patterns
3. **Compiled Adjacency:** Pre-computed row/column indices for fast syndrome computation

### 5.2 Baseline Flow

```
QBER Estimate → Lookup Effective Rate → Lookup Puncture Pattern
→ Construct Frame (Payload + Padding at pattern positions)
→ Compute Syndrome → Transmit (syndrome, pattern_id, hash)
```

### 5.3 Blind Flow

```
Select Modulation δ → Initialize p=d, s=0 → Construct Frame
→ Compute Syndrome (once) → Transmit syndrome
LOOP:
    Bob decodes with current LLRs
    IF success: DONE
    IF p=0: FAIL
    ELSE: Alice reveals Δ bits → Bob updates LLRs → RETRY
```

### 5.4 Security Invariants

1. **Leakage Monotonicity:** Total leakage must be tracked exactly and monotonically increase
2. **Syndrome Reuse:** In Blind, syndrome is computed and transmitted exactly once
3. **No Matrix Swapping:** The parity-check matrix is fixed for the entire reconciliation session
4. **Circuit Breaker:** If cumulative leakage exceeds NSM budget, abort immediately

---

## Appendix A: Notation Summary

| Symbol | Definition |
|--------|------------|
| $n$ | Frame size (codeword length) |
| $k$ | Information bits (message length) |
| $R = k/n$ | Code rate |
| $R_0$ | Mother code rate (fixed at 0.5) |
| $R_{\text{eff}}$ | Effective rate after puncturing/shortening |
| $p$ | Number of punctured bits |
| $s$ | Number of shortened bits |
| $\delta = (p+s)/n$ | Total modulation fraction |
| $\pi = p/n$ | Puncturing fraction |
| $\sigma = s/n$ | Shortening fraction |
| $h(\cdot)$ | Binary entropy function |
| $f$ | Reconciliation efficiency |
| $\mathcal{P}$ | Set of punctured symbol indices |
| $\mathcal{R}_k$ | Set of k-step recoverable symbols |
| $\mathcal{N}^k(v)$ | Depth-k neighborhood of node $v$ |
| $H_{\min}^\epsilon$ | Smooth min-entropy |

---

## Appendix B: Relevant Literature Excerpts

### From Elkouss et al. (2010) — Rate Compatible Protocol

> "Alice creates a string $\mathbf{x}^+ = g(\mathbf{x}, \sigma_{p^*}, \pi_{p^*})$ of size $n$. The function $g$ defines the $n - d$ positions are going to have the values of string $\mathbf{x}$, the $p$ positions that are going to be assigned random values, and the $s$ positions that are going to have values known by Alice and Bob."

### From Martinez-Mateo et al. (2012) — Blind Reconciliation

> "This whole procedure is done using the same base code... The only classical communication... is one message... to send the syndrome and the shortened information bits."

> "Alice can then reveal a set of the values of the previously punctured symbols... changing from punctured to shortened. This is like moving along the dotted vertical line [of the FER graph] and changing the code with $p=200, s=0$ by the code with $p=160, s=40$."

### From Elkouss et al. (2012) — Untainted Puncturing

> "A symbol node $v$ is said to be *untainted* if there are no punctured symbols within $\mathcal{N}^2(v)$."

> "The untainted algorithm is a method that chooses symbols such that all the check nodes of a selected symbol are survived check nodes."

---

*End of Theoretical Report v2*

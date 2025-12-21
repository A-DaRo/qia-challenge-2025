# Theoretical Report v2: Advanced Rate-Compatible Reconciliation in Caligo

**Version:** 2.0  
**Date:** December 2025  
**Subject:** Rigorous Mathematical Framework for Baseline and Blind Reconciliation Protocols  
**Context:** $\binom{2}{1}$-Oblivious Transfer via Noisy Storage Model (NSM)

**Primary References:**
1. Elkouss, D., Martinez-Mateo, J., Lancho, D., & Martin, V. (2010). "Rate Compatible Protocol for Information Reconciliation: An Application to QKD."  [1]
2. Martinez-Mateo, J., Elkouss, D., & Martin, V. (2012). "Blind Reconciliation." *Quantum Information and Computation*, Vol. 12, No. 9&10, pp. 791-812.  [2]
3. Elkouss, D., Martinez-Mateo, J., & Martin, V. (2012). "Untainted Puncturing for Irregular Low-Density Parity-Check Codes." *IEEE Wireless Communications Letters*, Vol. 1, No. 6, pp. 585-588.  [3]
4. Liu, J. & de Lamare, R. C. (2014). "Rate-Compatible LDPC Codes Based on Puncturing and Extension Techniques for Short Block Lengths." *arXiv:1407.5136 [cs.IT].*  [4]
5. Vellambi, B. N. & Fekri, F. (2009). "Finite-Length Rate-Compatible LDPC Codes: A Novel Puncturing Scheme." *IEEE Transactions on Communications*, Vol. 57, No. 2.  [5]
6. Kiktenko, E. et al. (2016). "Post-processing procedure for industrial quantum key distribution systems." *J. Phys.: Conf. Ser.* 741 012081.  [6]
7. Elkouss, D., Leverrier, A., Alléaume, R., & Boutros, J. J. (2009). "Efficient reconciliation protocol for discrete-variable quantum key distribution." *arXiv:0901.2140 [cs.IT].*  [7]
8. Tian, T., Jones, C., Villasenor, J. D., & Wesel, R. D. "Construction of Irregular LDPC Codes with Low Error Floors." (ACE/EMD definitions).  [8]

---

## 1. Introduction: The Entropy Economy Under the Noisy Storage Model

### 1.1 Problem Statement

In the Caligo protocol, Phase III (Information Reconciliation) is the critical juncture where error correction meets security. Unlike standard QKD, where syndrome information leaks to a passive eavesdropper (Eve), in our $\binom{2}{1}$-Oblivious Transfer protocol under the Noisy Storage Model (NSM), this information leaks directly to the receiver (Bob), who is a potential adversary. Under the NSM security framework, *minimizing the syndrome length* $|\Sigma|$ is paramount to maximizing the extractable secure OT output.

This report formalizes the theoretical foundations for two distinct reconciliation strategies:

- **Baseline** (Elkouss et al., 2010) [1]: Rate-compatible protocol requiring *a priori* QBER estimation.
- **Blind** (Martinez-Mateo et al., 2012) [2]: Rate-adaptive protocol without QBER pre-estimation, using iterative bit revelation.

Both strategies are unified under a **Hybrid Rate-Compatible Architecture**. While **Untainted Puncturing** (Elkouss et al., 2012) [3] provides strong stopping-set protection for *moderate* puncturing proportions, finite-length operation (here $n=4096$) and a wide target rate span require additional topological control at *high* effective rates. We therefore use a **two-regime puncturing strategy**:

- **Regime A (moderate rates):** strict untainted puncturing until the untainted candidate set is exhausted.
- **Regime B (high rates):** an ACE/EMD-guided intentional puncturing phase (Liu & de Lamare, 2014) [4] to continue puncturing beyond the untainted saturation point while explicitly managing short-cycle/topology effects.

This hybrid view is consistent with the broader finite-length rate-compatible LDPC puncturing literature, which repeatedly finds that “distance/topology-aware” puncturing dominates random puncturing at short block lengths (e.g., the punctured bits should be far apart in the Tanner graph to reduce degradation) [5].

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

## 2. Unified Foundation: Hybrid Puncturing for Wide-Range Compatibility

### 2.1 Theoretical Framework for Rate Adaptation

The naive description of syndrome computation—"Alice computes $s = H_{\text{mother}} \cdot x_{\text{frame}}$"—obscures the sophisticated rate-adaptation mechanism. In the rate-compatible framework, the mother code $\mathcal{C}_{R_0}$ serves as a structural parent; the *effective code* $\mathcal{C}_{\text{eff}}$ is constructed dynamically by manipulating the input frame.

To avoid ambiguity between *payload* and *coding frame* (a common source of confusion in finite-size leakage accounting), we distinguish:

- $n$: mother-code block length (parity-check matrix width).
- $k = R_0 n$: mother-code dimension.
- $d = p+s$: number of *modulation* positions (punctured+shortened positions in the frame).
- $m = n-d$: number of positions filled with the correlated raw string (payload).

This is aligned with the construction described by Elkouss et al. [1] and Martinez-Mateo et al. [2]: Alice embeds a shorter correlated string into a fixed-length LDPC frame by adding $p$ unknown (punctured) and $s$ known (shortened) symbols.

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

**Remark 2.1 (Reaching $R_{\text{eff}} \approx 0.9$ requires large puncturing).** If the design target includes very high rates (e.g., $R_{\text{eff}}=0.9$) with a mother code at $R_0=0.5$, then (even in the most optimistic case $s=0$) the required puncturing fraction is determined by

$$
R_{\text{eff}} = \frac{R_0}{1-\pi} \quad (s=0) \qquad\Rightarrow\qquad \pi = 1 - \frac{R_0}{R_{\text{eff}}}.
$$

For $R_0=0.5$ and $R_{\text{eff}}=0.9$, this yields $\pi \approx 0.444$. Consequently, any architecture that genuinely targets $0.5\to 0.9$ must operate in a *high-puncturing* finite-length regime where purely untainted puncturing cannot be expected to suffice; this motivates the hybrid puncturing strategy formalized in Section 2.2.

### 2.2 Hybrid Puncturing: From Untainted to ACE-Guided High-Rate Operation

Random puncturing leads to decoding failures at high effective rates due to finite-length artifacts (e.g., small stopping sets and harmful short-cycle interactions). The literature distinguishes random from intentional puncturing and repeatedly shows that graph-aware selection dominates random patterns at short/moderate block lengths [3]–[5].

We therefore treat puncturing as a *two-regime* problem:

- **Regime A (strictly untainted):** puncture only untainted symbols, guaranteeing strong local structural properties [3].
- **Regime B (ACE-guided):** once strict untainted candidates are depleted, continue puncturing using a cycle/topology-aware metric (ACE/EMD family) to manage short cycles/connectivity in the remaining graph [4].

#### 2.2.1 Graph-Theoretic Definitions

**Definition 2.3 (Neighborhood).** Let $\mathcal{N}(z)$ denote the neighborhood of node $z$ in the Tanner graph (nodes adjacent to $z$). The depth-$k$ neighborhood $\mathcal{N}^k(z)$ includes all nodes reachable by traversing at most $k$ edges.

**Definition 2.4 (k-Step Extended Recoverable Symbols).** Let $\mathcal{P}$ be the set of punctured symbols. We define:

1. **1-Step Recoverable ($\mathcal{R}_1$):** A punctured symbol $v \in \mathcal{P}$ belongs to $\mathcal{R}_1$ if $\exists c \in \mathcal{N}(v)$ such that $\forall w \in \mathcal{N}(c) \setminus \{v\}, w \notin \mathcal{P}$.
   
   *Interpretation:* There exists a check node $c$ connected to $v$ where all other connected symbols are unpunctured. This check provides full information to recover $v$ in one decoding iteration.

2. **k-Step Recoverable ($\mathcal{R}_k$, $k > 1$):** A punctured symbol $v \notin \bigcup_{i=1}^{k-1} \mathcal{R}_i$ belongs to $\mathcal{R}_k$ if $\exists c \in \mathcal{N}(v)$ and $\exists w \in \mathcal{N}(c) \setminus \{v\}$ such that $w \in \mathcal{R}_{k-1}$ and $\forall w' \in \mathcal{N}(c) \setminus \{v, w\}, w' \in \mathcal{P} \Rightarrow w' \in \bigcup_{i=1}^{k-1} \mathcal{R}_i$.

**Definition 2.5 (Survived vs. Dead Check Nodes).** Given a punctured symbol $v$:
- A check $c \in \mathcal{N}(v)$ is a **survived check node** if $\exists w \in \mathcal{N}(c) \setminus \{v\}$ such that $w \in \mathcal{R}_{k-1}$ (provides recovery information).
- A check $c \in \mathcal{N}(v)$ is a **dead check node** if it provides zero LLR to $v$ (all neighbors are unrecoverable punctured symbols).

**Definition 2.6 (Untainted Symbol).** A symbol node $v$ is **untainted** if there are no punctured symbols within $\mathcal{N}^2(v)$ [3].

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

#### 2.2.2b The Untainted Saturation Limit

The untainted procedure is intentionally conservative: puncturing a symbol removes (taints) its entire depth-2 neighborhood from future consideration. This structural protection implies an intrinsic *finite-length ceiling* on how many symbols can be punctured while remaining strictly untainted.

**Theorem 2.2 (Untainted Saturation; termination by depletion).** The strictly untainted algorithm terminates when the untainted candidate set is empty, i.e., when $\mathcal{X}_\infty = \emptyset$. At that point, further strict untainted puncturing is impossible by definition.

*Argument (structural, finite-length).* Each punctured symbol $v$ removes $\mathcal{N}^2(v)$ from the candidate set. For a typical irregular LDPC Tanner graph, $|\mathcal{N}^2(v)|$ is not constant but is often large compared to 1 (and grows with local degrees). Consequently, the candidate set is depleted after a moderate number of punctures even when $n$ is only a few thousand. Elkouss et al. explicitly motivate untainted puncturing as a method optimized for *moderate* puncturing proportions and short lengths [3], rather than a universal wide-range construction.

**Design Guideline 2.1 (Empirical ceiling, not a universal constant).** For finite-length irregular constructions comparable to those considered in the untainted puncturing literature, strict untainted selection commonly saturates after puncturing on the order of $\pi\sim 0.2$ of the symbols (i.e., roughly 20%), after which $\mathcal{X}_\infty$ becomes empty and strict untainted selection cannot continue. Interpreted through the optimistic $s=0$ rate formula $R_{\text{eff}}=R_0/(1-\pi)$, this corresponds to a practical ceiling around

$$
R_{\text{sat}} \approx \frac{0.5}{1-0.2} = 0.625.
$$

This heuristic ceiling matches the qualitative behavior reported across finite-length rate-compatible puncturing studies: when puncturing becomes aggressive, one must transition to a different intentional criterion that manages topological constraints (cycle structure and connectivity) rather than insisting on strict local “untaintedness” [4], [5].

#### 2.2.2c Why Strictly Untainted Helps (and why random puncturing is risky)

Untainted puncturing is best understood as a *local stopping-set avoidance heuristic*: by prohibiting punctured nodes within depth 2, it reduces the probability that a punctured node is surrounded by checks that provide no information early in belief propagation [3]. This is consistent with broader finite-length observations: puncturing bits “too close” in the Tanner graph increases performance degradation, while spacing punctures apart mitigates it [5].

#### 2.2.3 Extension to High Rates via ACE-Based Puncturing

To achieve rates $R_{\text{eff}} \in (R_{\text{sat}}, 0.9]$ with $n=4096$, we must relax the strict untainted constraint while preserving graph connectivity properties that are critical at finite length. Liu & de Lamare [4] propose puncturing strategies explicitly designed for *short/moderate block lengths* that combine (i) cycle-structure information and (ii) an extrinsic-connectivity metric (ACE/EMD family). This directly targets the failure mode that appears beyond untainted saturation: the remaining candidates are necessarily “tainted,” so selection must prioritize *which* tainted nodes are least harmful to puncture.

##### The ACE/EMD idea (connectivity-aware cycle management)

At finite length, short cycles are unavoidable and can dominate the error floor. The ACE/EMD family of metrics is designed to quantify (approximately) how well a subgraph involved in a short cycle is connected to the rest of the Tanner graph. Intuitively:

- **Low ACE / low extrinsic connectivity**: the neighborhood is “self-contained,” and harmful trapping/stopping configurations are more likely.
- **High ACE / high extrinsic connectivity**: there are more edges leaving the cycle neighborhood, improving message diversity under iterative decoding.

This is aligned with the broader finite-length puncturing principle in Vellambi & Fekri [5]: puncturing patterns should avoid creating tightly coupled punctured neighborhoods; punctured bits should be “far apart” in the Tanner graph.

##### Hybrid puncturing strategy (formalized)

We define a **Hybrid Puncturing Strategy** that combines the strengths of untainted puncturing (Regime A) and ACE/EMD-guided selection (Regime B):

1. **Phase I (Untainted / Regime A):** Run the strict untainted algorithm [3] until either:
    - the target puncturing fraction $\pi$ is met, or
    - the candidate set is exhausted ($\mathcal{X}_\infty = \emptyset$).

2. **Phase II (ACE-Based / Regime B):** If additional punctures are required to reach the target rate, rank the remaining *non-punctured* variable nodes by an ACE/EMD-informed score derived from short-cycle structure as in Liu & de Lamare [4], and puncture greedily according to that score.

    A practical (and literature-consistent) selection rule is:

    - compute, for each candidate variable node $v$, a “worst-short-cycle” ACE score over a small set of short cycle lengths (e.g., girth $g$ and nearby lengths):

      $$
      \mathrm{ACE}_{\min}(v) = \min_{\gamma \in \Gamma_{\le L}(v)} \mathrm{ACE}(\gamma),
      $$

      where $\Gamma_{\le L}(v)$ is the set of cycles through $v$ up to some small length cutoff $L$, and $\mathrm{ACE}(\gamma)$ is an extrinsic-connectivity proxy for cycle $\gamma$.

    - puncture the candidates with **largest** $\mathrm{ACE}_{\min}(v)$ first (puncture “well-connected” nodes before puncturing nodes that sit in weakly connected short-cycle neighborhoods).

    This rule captures the core intent: *if strict untaintedness is no longer possible, puncture in a way that minimizes the creation of weakly connected trapping substructures*.

3. **Rate-compatibility constraint (nesting):** To maintain rate-compatibility (higher-rate graphs as subgraphs of lower-rate graphs), the puncturing order must be fixed once and then truncated to obtain each desired rate [1], [4], [5]. The hybrid procedure naturally provides such an order: all Phase-I nodes come first, then Phase-II nodes in ranked order.

##### Why this is the right abstraction for Caligo

This hybrid strategy resolves the central inconsistency in a wide-rate report: strict untainted puncturing is *by design* a moderate-puncturing method [3], while achieving $R_{\text{eff}}\approx 0.9$ from $R_0=0.5$ necessarily requires aggressive puncturing ($\pi\approx 0.444$ in the optimistic $s=0$ case). Therefore, **a second intentional puncturing criterion is not optional**; it is required for the document’s claimed rate range.

---

## 3. Baseline Reconciliation Protocol (Elkouss)

The Baseline protocol is an **inverse puncturing and shortening** scheme performed *after* Alice and Bob already hold correlated strings. It requires a (finite-size) error-rate estimate to pick $(p,s)$ once and run a single decoding attempt at an optimized effective rate [1].

From a protocol-design perspective, Baseline trades **one sampling message** (used to estimate the channel parameter) for **one-shot decoding** (no iterative revelations as in Blind). This matters in the NSM setting because the sampling step discloses true payload bits and must be deducted from the min-entropy budget.

### 3.1 Phase II: Parameter Estimation

In the baseline protocol of Elkouss et al. [1], parameter estimation is explicitly performed by sampling disclosed bits (their Step 2–3): Bob selects a random subset of positions and sends the corresponding bits and positions to Alice; Alice compares them to her bits to estimate the crossover probability.

Before reconciliation, Alice and Bob estimate the channel crossover probability from a random test subset (size $t$):

$$
p^* = \frac{1}{t}\sum_{i \in \text{TestSet}} \mathbf{1}[x_i \neq y_i]
$$

In a strict finite-size security analysis, this estimate should be conservatively adjusted (e.g., an upper confidence bound) before selecting the coding parameters. Industrial QKD post-processing pipelines also treat the estimate conservatively and include explicit abort conditions when QBER exceeds a critical value [6].

**Leakage note:** The $t$ disclosed sample bits are *actual* key material. They must be removed from the reconciled string (and counted as leakage) exactly as noted in the QKD reconciliation literature [1].

### 3.2 Phase III: Rate Selection and Frame Construction

**Step 1: Choose modulation budget $\delta$.** Elkouss et al. define a fixed modulation budget $d = \lfloor \delta n \rfloor$ that determines the reachable effective-rate range

$$
\frac{R_0 - \delta}{1-\delta} \le R_{\text{eff}} \le \frac{R_0}{1-\delta}
$$

and emphasize the trade-off: larger $\delta$ covers a wider BER/QBER range but typically degrades efficiency [1].

**Step 2: Rate Selection.** Using the (possibly conservative) estimate $p^*$, Alice selects a target rate via the reconciliation-efficiency model

$$
R = 1 - f(p^*)\, h(p^*)
$$

where $h$ is the binary entropy function and $f(\cdot) \ge 1$ is an empirically obtained efficiency curve for the chosen code family [1]. In discrete-variable QKD, $f(p)$ is the standard figure of merit linking disclosed information to the Slepian–Wolf limit; improving $f$ directly increases the achievable secret-key rate [7].

**Step 3: Compute $(s,p)$ from $(R,\delta)$.** Writing $d = \lfloor \delta n \rfloor$, Elkouss et al. derive (their Eq. (7))

$$
s = \left\lceil \left( R_0 - R\left(1 - \frac{d}{n}\right) \right) n \right\rceil, \quad p = d - s.
$$

This is the operational “inverse puncturing and shortening” step: the channel is observed first, and the coding modulation is chosen second [1].

**Step 4: Frame construction via $g(\cdot)$.** Alice constructs a length-$n$ frame

$$
\mathbf{x}^+ = g(\mathbf{x}, \sigma, \pi)
$$

as in [1]:

1. The $n-d$ **transmission positions** are filled with the correlated string $\mathbf{x}$.
2. The $p$ **punctured positions** are filled with pseudo-random symbols generated from a synchronized PRG seed (unknown *as channel observations*; their LLR is initialized as 0).
3. The $s$ **shortened positions** are filled with PRG-derived values known to both parties (their LLR is initialized as $\pm\infty$).

Crucially, [1] assumes the **sets of positions** (transmitted/punctured/shortened) and the shortened values are reproducible from a synchronized pseudo-random generator. This eliminates extra communication overhead beyond the syndrome and the parameter estimate.

**Step 4: Syndrome Computation.**

$$
\mathbf{s} = H_{\text{mother}} \cdot \mathbf{x}^+ \mod 2
$$

**Step 5: Transmission.** In the baseline protocol description, Alice sends the syndrome of $\mathbf{x}^+$ and the estimate $p^*$ (or equivalently a code-parameter identifier derived from it) to Bob [1].

### 3.3 Decoder Initialization (Bob)

Bob constructs the corresponding frame $\mathbf{y}^+ = g(\mathbf{y}, \sigma, \pi)$ and initializes LLRs using the puncturing/shortening conventions explicitly stated in [1]: punctured symbols have $\gamma_p = 0$ and shortened symbols have $\gamma_s = \infty$.

| Position Type | Channel LLR Value |
|---------------|-------------------|
| **Payload bits** | $\gamma_i = \ln\frac{1 - p^*}{p^*} \cdot (1 - 2y_i)$ |
| **Punctured bits** | $\gamma_i = 0$ (Erasure: no information) |
| **Shortened bits** | $\gamma_i = \pm\infty$ (Perfect knowledge) |

The decoder runs standard belief propagation against syndrome $\mathbf{s}$.

### 3.4 Leakage Analysis

**Theorem 3.1 (Baseline Leakage Bound; protocol-level accounting).** For Baseline reconciliation [1], the public communication includes:

1. The syndrome (computed with the fixed mother matrix): $|\Sigma| = n-k = (1-R_0)n$.
2. The parameter-estimation disclosure: $t$ payload bits (plus their positions).
3. A verification tag (hash) of length $h$.

$$
\mathsf{leak}_{\mathsf{Baseline}} = (n-k) + t + h
$$

where $h$ is the verification-tag length. The dominant syndrome term is **constant per block** and independent of the effective rate used for decoding (because the mother matrix is fixed) [1].

**Verification necessity:** Iterative LDPC decoding can converge to an incorrect codeword that still satisfies the transmitted syndrome; industrial QKD stacks therefore add a second-stage identity check using $\varepsilon$-universal hashing [6]. For example, Kiktenko et al. report a 50-bit PolyR-type tag giving collision probability below $2\times 10^{-12}$ over $N$ blocks, explicitly motivated by the possibility of “wrong result but proper syndrome” [6].

---

## 4. Blind Reconciliation Protocol (Martinez-Mateo)

The Blind protocol eliminates *a priori* error-rate estimation by starting at the highest supported rate (all $d$ modulation symbols punctured) and then revealing previously punctured symbols in small increments until decoding succeeds or the modulation budget is exhausted [2]. Its goal is to recover much of the efficiency of finely rate-matched codes while using only a small number of interactive messages.

### 4.1 The Security Argument: Shortening vs. Matrix Modification

**Critical Design Question:** Should rate adaptation modify the parity-check matrix (add rows) or modify the puncturing pattern?

**Answer:** The protocol uses **iterative shortening** (converting punctured symbols to shortened symbols) while retaining the original base code; this is the defining mechanism in the original blind reconciliation description [2].

**Theorem 4.1 (Blind iteration leakage structure).** In the Blind protocol [2], the syndrome $\mathbf{s} = H\,\mathbf{x}^+$ is transmitted once. If decoding fails, Alice reveals values of some previously punctured symbols; the only additional leakage per extra iteration is the number of newly revealed shortened symbols (the syndrome is not retransmitted).

*Justification (as in [2], phrased for NSM accounting):*

1. **Information-Theoretic Argument:** Let $\mathbf{x}^+ = [\mathbf{x}_{\text{payload}}, \mathbf{x}_{\text{punct}}, \mathbf{x}_{\text{short}}]$ be the frame partition. The syndrome satisfies:

   $$
   \mathbf{s} = H_{\text{payload}} \cdot \mathbf{x}_{\text{payload}} + H_{\text{punct}} \cdot \mathbf{x}_{\text{punct}} + H_{\text{short}} \cdot \mathbf{x}_{\text{short}} \mod 2
   $$

2. **Iteration $i$ Information:** After revealing $s_i$ shortened values, Bob knows:
   - Full payload from channel: $\mathbf{y}_{\text{payload}}$
   - Revealed shortened values: $\mathbf{x}_{\text{short}}^{(i)}$
   - Syndrome: $\mathbf{s}$

3. **On failure:** “No extra information than the syndrome has been leaked” in the sense that punctured symbols are not sent over the channel and are treated as unknown at the decoder; revealing nothing further keeps the public transcript unchanged [2].

4. **Key insight:** Each newly shortened symbol reveals exactly one bit of information (its value). Therefore the incremental leakage is exactly the number of revealed symbols, matching the operational description in [2].

**Corollary 4.1.** The total leakage for Blind reconciliation is:

$$
\mathsf{leak}_{\mathsf{Blind}} = (n-k) + h + |\mathsf{Revealed}| = (1 - R_0)\,n + h + \sum_{i=2}^{t} \Delta_i
$$

where $\Delta_i$ is the number of symbols revealed in iteration $i$. This is the NSM-friendly form: it separates the fixed syndrome term from the strictly additive revealed-symbol term.

**Verification necessity (again):** The same “proper syndrome but wrong key” failure mode applies here; a short $\varepsilon$-universal verification tag is therefore standard practice [6].

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

The following is a direct specialization of the blind protocol description in Martinez-Mateo et al. [2] to our notation.

**Setup:** Choose a base code $\mathcal{C}(n,k)$ that can correct up to the worst-case expected error rate $\epsilon_{\max}$ (only a rough estimate is needed to pick the base code) [2]. Fix a modulation budget $d = p+s$ and define $\delta = d/n$. Let $m=n-d$ be the length of the correlated strings to reconcile.

Choose a maximum number of iterations $t$ and define the conversion step as in [2]:

$$
\Delta \triangleq d/t
$$

(In an implementation, use $\Delta = \lceil d/t \rceil$ and a final smaller step if needed so that exactly $d$ symbols are convertible in total.)

**Which symbols get revealed (random vs. fixed-by-seed)?** To match [2] while avoiding per-iteration index communication, Caligo should treat the *revelation order* as fixed at setup time:

1. Precompute an ordered list $\mathcal{P}_\text{ord}$ of the $d$ modulation indices (puncturing order), using the hybrid puncturing order from Section 2.2.
2. Optionally randomize this order by applying a pseudo-random permutation derived from a synchronized seed (so the resulting order is random-looking but deterministic and reproducible by both parties without extra messages).
3. At iteration $i$, reveal the next $\Delta$ indices from this fixed order.

This makes the revelation pattern independent of Bob's side information and *conditional only on the public iteration counter* (the only adaptivity is whether we proceed to iteration $i+1$ after a decoding failure, as in [2]).

**NSM-oriented heuristic QBER input (for simulation-driven parameterization).** Although Blind reconciliation does not require an explicit parameter-estimation message, in our NSM setting we may pass a heuristic channel estimate $\widehat{\text{QBER}}$ computed from trusted-channel parameters (`compute_qber_erven` from `utils/math.py` in the simulation noise profile). For conservative finite-key behavior, treat it analogously to the baseline's sampling estimate by using an adjusted value

$$
\widehat{\text{QBER}}_{\text{FK}} = \widehat{\text{QBER}} + \mu(\varepsilon_{\mathrm{PE}}, N)
$$

where $\mu(\cdot)$ is a confidence-radius term (same role as in Section 3).

We then use $\widehat{\text{QBER}}_{\text{FK}}$ for two *gating* decisions:

1. **Permissive starting-rate cap (small reductions only):** avoid starting from an obviously infeasible highest rate by allowing a small pre-shortening $s_1 \ge 0$ (sent alongside the syndrome in the first message) so that the initial attempt uses $p_1 = d - s_1$, $s=s_1$, instead of the absolute maximum $p=d$, $s=0$. Operationally: choose $s_1$ as the *smallest* value that brings the initial effective rate below a conservative cap implied by $\widehat{\text{QBER}}_{\text{FK}}$ (e.g., via the same efficiency model $R \approx 1 - f\,h(\widehat{\text{QBER}}_{\text{FK}})$ used in baseline selection). This is intentionally permissive: it is a small first-message adjustment (no extra interaction round), and it simply trades a few revealed bits for avoiding an avoidable first-attempt failure when the channel is clearly noisy.
2. **Restrictive iteration budget (only high QBER unlocks $t>3$):** set $t=3$ by default (so $\Delta \approx d/3$), and allow $t>3$ only when $\widehat{\text{QBER}}_{\text{FK}}$ is “high” (i.e., close to the conservative QBER threshold used to declare secure operation) so that finer-grained revelation materially improves success probability. In lower-QBER regimes, additional rounds mostly add interaction complexity and worst-case leakage without much gain.

As in [2], the choice of $\delta$ determines the reachable high-rate endpoint via $R_{\max} = R_0/(1-\delta)$ (when $s=0$). If the design target includes $R_{\text{eff}}\approx 0.9$ with $R_0=0.5$, then $\delta$ must be on the order of $0.44$ (Remark 2.1).

**Iteration 1 (Optimistic Attempt):**
- Configuration (canonical [2]): $p = d$, $s = 0$ (all $d$ modulation symbols punctured → highest rate reachable for this $\delta$)
- Configuration (NSM-gated variant, permissive cap): $p_1 = d - s_1$, $s=s_1$ with a small $s_1$ chosen from $\widehat{\text{QBER}}_{\text{FK}}$ as above
- Alice sends syndrome $\mathbf{s}$ computed once (and, if using $s_1>0$, the revealed shortened values for those $s_1$ positions)
- Bob initializes LLRs with $p$ erasures and decodes

**Iteration $i \geq 2$ (On Failure):**
Let $j=(i-2)\Delta$.

1. Alice takes the next block of indices from the predetermined order $\mathcal{P}_\text{ord}$:
    $$
    \mathcal{S}_i = \{\mathcal{P}_\text{ord}[j+1], \ldots, \mathcal{P}_\text{ord}[j+\Delta]\}
    $$
    (or fewer indices in the final step if $d$ is not divisible by $t$).
2. Alice **reveals the values** $\{x_u : u \in \mathcal{S}_i\}$ to Bob; these positions are now treated as shortened.
3. State update: $p \leftarrow p - |\mathcal{S}_i|$, $s \leftarrow s + |\mathcal{S}_i|$.
4. Bob updates LLRs for $u\in\mathcal{S}_i$: $0 \to \pm\infty$.
5. Bob re-runs BP using the **same syndrome** and **same parity-check matrix**, as in [2].

**Termination:**
- **Success:** Decoder converges and passes verification hash
- **Failure:** $p = 0$ (all bits shortened) and decoder still fails

### 4.4 Average Efficiency Analysis

Martinez-Mateo et al. derive an explicit expression for the **average rate** and hence the **average efficiency** of blind reconciliation in terms of the per-iteration FER values $F^{(i)}$ [2]. This gives the key quantitative reason the protocol works: even a small number of iterations captures a large fraction of the gain of the fully granular ($\Delta=1$) process.

The average rate achieved by Blind reconciliation depends on the FER distribution across iterations [2]:

$$
\bar{R} = \sum_{i=1}^{t} a_i \cdot r_i
$$

where:
- $a_i = \frac{F^{(i-1)} - F^{(i)}}{1 - F^{(t)}}$ is the fraction of codewords corrected at iteration $i$
- $F^{(i)}$ is the Frame Error Rate when using adapted rate $r_i$
- $r_i = \frac{R_0 - \sigma_i}{1 - \delta}$ with $\sigma_i = (i-1) \cdot \Delta / n$

In [2], the FER terms $F^{(i)}$ can be approximated analytically (Gaussian approximation) to cheaply predict performance in the waterfall region; this is useful for parameter sweeps but does not replace simulation for error-floor behavior.

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

**Key Insight (as formalized in [2]):** Smaller $\Delta$ (more iterations, finer-grained conversion of punctured to shortened symbols) improves average efficiency because each intermediate adapted code has a non-zero probability of success, lowering the expected number of revealed symbols.

#### 4.5.3 Practical Recommendation

The Blind protocol’s design parameters are inherently *application-dependent*, and the literature makes two points that are directly relevant here:

1. There is an explicit **trade-off between covered error range and efficiency** driven by $\delta$ [2] (and already in the baseline rate-compatible construction [1]).
2. With short codes, a small number of iterations (e.g., 3) can capture much of the efficiency improvement while keeping interactivity modest [2].

Two regimes are worth separating explicitly:

1. **Moderate-rate operation (small $\delta$):** If the goal is to adapt around $R_0$ over a narrow QBER band (the traditional QKD use case emphasized in [1], [2]), then small modulation fractions (e.g., $\delta=0.1$) can be attractive for implementation simplicity. This yields $R_{\text{eff}} \in [0.444, 0.556]$ for $R_0=0.5$.

2. **Wide-range operation up to $R_{\text{eff}}\approx 0.9$ (large $\delta$):** If the protocol must span a wide effective-rate range as claimed in this report, then $\delta$ must be chosen accordingly (Remark 2.1). In that case, the puncturing pattern generation must be hybrid (Section 2.2), because strict untainted puncturing will saturate well before the required puncturing proportion.

For finite-length codes, the *choice of puncturing pattern* is also material: Martinez-Mateo et al. explicitly simulate blind reconciliation with an intentional puncturing pattern “previously estimated” (their Fig. 3 caption) and later note that punctured symbols are selected according to a computed pattern designed for moderate puncturing rates [2]. This is consistent with untainted puncturing being strongest at moderate puncturing proportions [3] and with ACE/EMD metrics being effective ways to manage short-cycle connectivity at finite length [8].

In both cases, the step-size choice remains governed by the same trade-off described by Martinez-Mateo et al. [2]: smaller steps (more iterations) improve average efficiency but increase interactivity and state management. A pragmatic engineering default consistent with [2] is still:

- **$\Delta \approx d/3$** (≈3 iterations maximum)

but the appropriate $d=\delta n$ depends on whether the target includes very high rates.

---

## 5. Summary: Implementation Requirements

To implement these protocols within Caligo, the following architecture is required:

### 5.1 Data Structures

1. **Single Mother Matrix:** Fixed mother matrix (e.g., $R_0=0.5$), stored as sparse CSR
2. **Pattern Library (Hybrid):** A single *ordered* puncturing list per mother matrix, generated by the hybrid procedure:
    - Phase I: untainted order prefix [3]
    - Phase II: ACE/EMD-guided order suffix [4]
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

5. **Pattern Independence:** The puncturing/revelation *order* must be fixed independently of Bob’s behavior or side information (e.g., derived deterministically from the public code graph and a public seed). This mirrors the intent of rate-compatible puncturing as a code-design choice rather than an adaptive “oracle” [1], [2], [4].

## References

[1] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD."

[2] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Information and Computation*, Vol. 12, No. 9&10, pp. 791–812, 2012.

[3] D. Elkouss, J. Martinez-Mateo, and V. Martin, "Untainted Puncturing for Irregular Low-Density Parity-Check Codes," *IEEE Wireless Communications Letters*, Vol. 1, No. 6, pp. 585–588, 2012.

[4] J. Liu and R. C. de Lamare, "Rate-Compatible LDPC Codes Based on Puncturing and Extension Techniques for Short Block Lengths," *arXiv:1407.5136 [cs.IT]*, 2014.

[5] B. N. Vellambi and F. Fekri, "Finite-Length Rate-Compatible LDPC Codes: A Novel Puncturing Scheme," *IEEE Transactions on Communications*, Vol. 57, No. 2, 2009.

[6] E. Kiktenko, A. Trushechkin, Y. Kurochkin, and A. Fedorov, "Post-processing procedure for industrial quantum key distribution systems," *Journal of Physics: Conference Series*, 741, 012081, 2016.

[7] D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *arXiv:0901.2140 [cs.IT]*, 2009.

[8] T. Tian, C. Jones, J. D. Villasenor, and R. D. Wesel, "Construction of Irregular LDPC Codes with Low Error Floors" (EMD/ACE definitions and finite-length cycle connectivity arguments).

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

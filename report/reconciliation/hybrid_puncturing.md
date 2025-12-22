[← Return to Main Index](../index.md)

# 6.4 Rate-Compatible LDPC Codes via Hybrid Puncturing

## Introduction

Rate-compatible LDPC codes enable a single mother code to operate at multiple effective rates through **puncturing** (transmitting fewer codeword bits) and **shortening** (constraining information bits to known values). This section presents the theoretical framework for hybrid puncturing, which combines stopping-set-protected "untainted" puncturing with ACE-guided (Approximate Cycle Extrinsic) selection to achieve the wide rate coverage required for NSM protocols.

The fundamental challenge is that strict untainted puncturing *saturates* at moderate puncturing fractions, necessitating a two-regime strategy.

## Theoretical Foundation

### Rate Modulation via Puncturing

For a mother code with rate $R_0 = k/n$ and parity-check matrix $\mathbf{H} \in \{0,1\}^{m \times n}$, puncturing $p$ bits yields effective rate:

$$
R_{\text{eff}} = \frac{k}{n - p} = \frac{R_0}{1 - \pi}
$$

where $\pi = p/n$ is the puncturing fraction.

**Rate Coverage Requirement**: To span $R_{\text{eff}} \in [0.5, 0.9]$ from $R_0 = 0.5$:

$$
0.9 = \frac{0.5}{1 - \pi_{\max}} \implies \pi_{\max} = 1 - \frac{0.5}{0.9} \approx 0.444
$$

This requires puncturing **44% of the codeword**—far beyond typical untainted saturation.

### Stopping Set Theory

**Definition** (Stopping Set) [1]: A stopping set $\mathcal{S} \subseteq \{1, \ldots, n\}$ is a set of variable nodes such that every check node connected to $\mathcal{S}$ has degree $\geq 2$ in the induced subgraph.

**Significance**: If all bits in a stopping set are erased (or punctured), belief propagation cannot recover any of them—the decoder becomes "trapped."

**Theorem** (Stopping Set Failure) [1]: For binary erasure channel with erasure pattern $\mathcal{E}$, BP decoding fails if and only if $\mathcal{E}$ contains a stopping set.

### Untainted Symbol Definition

**Definition** (Untainted Symbol) [2]: A variable node $v$ is **untainted** with respect to punctured set $\mathcal{P}$ if no check node in $\mathcal{N}(v)$ connects to more than one punctured symbol.

Equivalently, $v$ is untainted if its depth-2 neighborhood contains no other punctured symbols:

$$
v \text{ is untainted} \iff \mathcal{N}^2(v) \cap \mathcal{P} = \emptyset
$$

where $\mathcal{N}^2(v) = \{u : \exists c \in \mathcal{N}(v), u \in \mathcal{N}(c)\}$.

**Property**: An untainted punctured symbol has all neighboring checks acting as **single parity checks**, providing optimal recovery redundancy.

## Two-Regime Hybrid Strategy

### Regime A: Untainted Puncturing

**Algorithm**: Greedily select untainted symbols with minimum neighborhood impact:

$$
v^* = \arg\min_{v \in \mathcal{X}_\infty} |\mathcal{N}^2(v)|
$$

where $\mathcal{X}_\infty$ is the set of currently untainted candidates.

**Update Rule**: After puncturing $v^*$:
$$
\mathcal{X}_\infty \leftarrow \mathcal{X}_\infty \setminus \mathcal{N}^2(v^*)
$$

**Saturation Theorem**: The untainted algorithm terminates when $\mathcal{X}_\infty = \emptyset$.

For irregular LDPC codes with average variable degree $\bar{d}_v \approx 3$ and check degree $\bar{d}_c \approx 6$:

$$
|\mathcal{N}^2(v)| \approx 1 + \bar{d}_v \cdot \bar{d}_c \approx 19
$$

Heuristic saturation point:
$$
\pi_{\text{sat}} \approx \frac{n}{|\mathcal{N}^2(v)|} \cdot \frac{1}{n} \approx \frac{1}{19} \approx 0.05
$$

In practice, saturation occurs at $\pi_{\text{sat}} \approx 0.15$–$0.25$ depending on code structure [2].

### Regime B: ACE-Guided Puncturing

Beyond untainted saturation, all remaining candidates are "tainted." The **ACE metric** [3] identifies which tainted symbols are safest to puncture.

**Definition** (ACE Score): For a cycle $\gamma$ through variable node $v$:

$$
\text{ACE}(\gamma) = \sum_{c \in \gamma} (d_c - 2)
$$

where $d_c$ is the check node degree. The score measures *extrinsic connectivity*—edges leaving the cycle.

**Interpretation**:
- **High ACE**: Many external edges → diverse message sources → robust decoding
- **Low ACE**: Few external edges → trapped messages → stopping set risk

**Selection Rule**: Puncture the symbol with maximum minimum ACE over short cycles:

$$
v^* = \arg\max_{v \notin \mathcal{P}} \min_{\gamma \in \Gamma_g(v)} \text{ACE}(\gamma)
$$

where $\Gamma_g(v)$ is the set of cycles through $v$ with length $\leq g + 2$ ($g$ = girth).

## Rate-Compatibility Constraint

### Nested Structure Requirement

For incremental redundancy and rate adaptation, puncturing patterns must be **nested**:

$$
\mathcal{P}(R_1) \subset \mathcal{P}(R_2) \quad \text{whenever } R_1 > R_2
$$

This ensures that lowering the rate only *adds* transmitted bits, never removes them.

**Implementation**: The hybrid algorithm generates an ordered puncturing sequence:
$$
\mathcal{L} = (v_1, v_2, \ldots, v_{p_{\max}})
$$

For target rate $R$, use prefix $\mathcal{P}(R) = \{v_1, \ldots, v_{|\mathcal{P}(R)|}\}$.

### Shortening for Rate Reduction

To achieve rates *below* $R_0$, use **shortening**: set $s$ information bits to known values (typically zero).

$$
R_{\text{eff}} = \frac{k - s}{n - s} < R_0
$$

Combined rate formula with puncturing fraction $\pi$ and shortening fraction $\sigma$:

$$
R_{\text{eff}} = \frac{R_0 - \sigma}{1 - \pi - \sigma}
$$

## Performance Analysis

### Recovery Probability Under Puncturing

For punctured symbol $v \in \mathcal{P}$, define **k-step recoverability**:

$$
v \in \mathcal{R}_k \iff v \text{ can be recovered after } k \text{ BP iterations}
$$

**Theorem** (Untainted Recovery): An untainted punctured symbol is 1-step recoverable:
$$
v \text{ untainted} \implies v \in \mathcal{R}_1
$$

**Proof**: All checks in $\mathcal{N}(v)$ connect to exactly one punctured symbol ($v$ itself), so each provides an independent parity equation for $v$. □

For ACE-punctured (tainted) symbols, recovery depends on the iterative propagation of beliefs through the Tanner graph.

### Effective Rate vs. Decoding Threshold

Puncturing increases the effective QBER threshold. For code with mother threshold $Q^*_0$:

$$
Q^*_{\text{eff}}(\pi) \approx Q^*_0 \cdot (1 - \alpha \pi)
$$

where $\alpha \in [0.5, 1]$ depends on puncturing quality. Well-chosen puncturing minimizes $\alpha$.

### Information-Theoretic Limit

The Shannon limit for the effective channel is:

$$
R_{\text{Shannon}} = 1 - h(Q) \cdot (1 - \pi)^{-1}
$$

The gap $\Delta R = R_{\text{Shannon}} - R_{\text{eff}}$ quantifies the finite-length penalty.

---

## References

[1] C. Di, D. Proietti, I. E. Telatar, T. J. Richardson, and R. L. Urbanke, "Finite-length analysis of low-density parity-check codes on the binary erasure channel," *IEEE Trans. Inf. Theory*, vol. 48, no. 6, pp. 1570–1579, 2002.

[2] J. Ha, J. Kim, and S. W. McLaughlin, "Rate-compatible puncturing of low-density parity-check codes," *IEEE Trans. Inf. Theory*, vol. 50, no. 11, pp. 2824–2836, 2004.

[3] T. Tian, C. R. Jones, J. D. Villasenor, and R. D. Wesel, "Selective avoidance of cycles in irregular LDPC code construction," *IEEE Trans. Commun.*, vol. 52, no. 8, pp. 1242–1247, 2004.

---

[← Return to Main Index](../index.md) | [Next: Leakage Accounting](./leakage_accounting.md)

[← Return to Main Index](../index.md)

# 6.3 Blind Reconciliation: Rate-Adaptive Slepian-Wolf Coding

## Information-Theoretic Foundation

### The Adaptivity Problem

Classical syndrome-based reconciliation operates at a fixed code rate $R$, requiring *a priori* estimation of the channel parameter $\epsilon$ (QBER). This creates two fundamental problems:

1. **Statistical overhead**: QBER estimation requires sacrificing $m$ bits, reducing the raw key by $m/n$.
2. **Rate mismatch**: Estimation errors lead to either:
   - $R > R_{\text{optimal}}$: Decoding failure (insufficient redundancy)
   - $R < R_{\text{optimal}}$: Excessive leakage (wasted entropy)

Martinez-Mateo et al. [1] resolved this dilemma with **Blind Reconciliation**—a protocol that discovers the effective channel capacity iteratively without explicit parameter estimation.

### The Blind Protocol Principle

Consider a mother code $\mathcal{C}(n, k)$ with parity-check matrix $H$ and $d = p + s$ modulation symbols partitioned into:
- **Punctured** ($p$): Values unknown to decoder; treated as erasures (LLR $= 0$)
- **Shortened** ($s$): Values known to decoder; treated as perfect observations (LLR $= \pm\infty$)

The effective code rate becomes:

$$
R_{\text{eff}} = \frac{R_0 - s/n}{1 - d/n} = \frac{R_0 - \sigma}{1 - \delta}
$$

where $\sigma = s/n$ and $\delta = d/n$ are the shortening and modulation fractions.

**Key insight**: The syndrome $\mathbf{s} = H \cdot \mathbf{x}$ is transmitted **once** at the protocol's start. Subsequent iterations reveal only the values of previously punctured symbols—converting them to shortened. This monotonically increases decoder knowledge without redundant communication.

---

## Protocol Specification

### Rate Coverage

For modulation parameter $\delta$ and mother rate $R_0$:

$$
R_{\min} = \frac{R_0 - \delta}{1 - \delta}, \quad R_{\max} = \frac{R_0}{1 - \delta}
$$

**Example** ($\delta = 0.44$, $R_0 = 0.5$):
$$
R_{\min} = 0.107, \quad R_{\max} = 0.893
$$

This corresponds to QBER coverage from $\approx 45\%$ to $\approx 5\%$—spanning the entire NSM-relevant regime.

### Iteration Protocol

**Setup**: Alice and Bob agree on mother code $\mathcal{C}(n, k)$, modulation $\delta$, maximum iterations $t$, and step size $\Delta = \lceil d/t \rceil$.

**Iteration 1** (optimistic):
1. Alice constructs $\mathbf{x}^+ = [\mathbf{x}_{\text{payload}}, \mathbf{x}_{\text{punct}}]$ with $p_1 = d$ punctured, $s_1 = 0$ shortened
2. Alice transmits syndrome $\mathbf{s} = H \cdot \mathbf{x}^+$ and verification hash $h$
3. Bob attempts BP decoding at rate $R_{\max}$

**Iteration $i > 1$** (fallback):
1. If decoding failed: Alice reveals $\Delta$ punctured values, converting them to shortened
2. Bob re-runs BP with enhanced LLRs at reduced rate $R_i < R_{i-1}$
3. Repeat until success or $i = t$

### Information Leakage

**Theorem (Martinez-Mateo [1])**: The total information leakage in Blind reconciliation is:

$$
\text{leak}_{\text{Blind}} = (1 - R_0) \cdot n + |h| + \sum_{i=2}^{t} \Delta_i
$$

where:
- $(1 - R_0) \cdot n$: Syndrome length (fixed, independent of effective rate)
- $|h|$: Verification hash (typically 64 bits)
- $\sum_i \Delta_i$: Revealed shortened values

**Critical property**: Unlike Cascade-style protocols, the syndrome is transmitted only once. Additional iterations reveal only $\Delta_i \ll n$ bits each.

---

## Average Efficiency Analysis

### Theoretical Framework

Let $F^{(i)}(\epsilon)$ denote the frame error rate when decoding at iteration $i$ for channel parameter $\epsilon$. The fraction of codewords corrected at iteration $i$ is:

$$
a_i = \frac{F^{(i-1)} - F^{(i)}}{1 - F^{(t)}}
$$

The average coding rate achieved is:

$$
\bar{R} = \sum_{i=1}^{t} a_i R_i
$$

**Reconciliation efficiency** (Martinez-Mateo Eq. 9):

$$
\bar{f}_{\text{BSC}(\epsilon)} = \frac{1 - \bar{R}}{h(\epsilon)}
$$

where $h(\epsilon) = -\epsilon \log_2 \epsilon - (1-\epsilon)\log_2(1-\epsilon)$ is the binary entropy.

### Efficiency vs. Cascade

For short codes ($n \approx 2 \times 10^3$), Martinez-Mateo demonstrated:

| Protocol | Iterations | Average $f$ at $\epsilon = 0.05$ |
|----------|------------|----------------------------------|
| Cascade | $\sim 20$ | 1.02 |
| Blind ($t=1$) | 1 | 1.18 |
| Blind ($t=3$) | 3 | 1.08 |

The three-iteration Blind protocol achieves near-Cascade efficiency with **three orders of magnitude fewer message exchanges**—critical for latency-sensitive applications.

---

## Security Implications for NSM

### Revelation Order Independence

**Requirement**: The order in which punctured symbols are revealed must be **independent of Bob's quantum side information**.

If revelation order were adaptive (chosen based on decoder feedback), it could constitute a side channel:

$$
I(X_{1-c}; \text{RevelationOrder}) \neq 0
$$

potentially leaking information about Alice's input $X_{1-c}$ that Bob should not learn.

**Solution**: Use a deterministic, pre-agreed revelation order derived from the hybrid puncturing pattern (see [§6.4](./hybrid_puncturing.md)). The order is public and reproducible; adaptivity is limited to the **continuation decision** (a public Boolean).

### Leakage Bounds for Finite-Size Security

The security proof requires bounding total leakage. For Blind reconciliation:

$$
\text{leak}_{\text{EC}} = n(1 - R_0) + 64 + \mathbb{E}[\text{revealed symbols}]
$$

The expected revealed symbols depend on the true QBER distribution. For well-designed codes with waterfall behavior:

$$
\mathbb{E}[\text{revealed}] \approx \delta \cdot n \cdot \Pr[\text{iteration 2+}] \ll n
$$

In the typical regime ($\epsilon \ll \epsilon_{\max}$), most blocks succeed at iteration 1, yielding leakage close to the syndrome-only baseline.

---

## Comparison with Rate-Compatible Punctured Codes

### Alternative: Fixed Puncturing Patterns

An alternative approach uses a family of codes $\{\mathcal{C}_p\}_{p=0}^{d}$ with pre-computed puncturing patterns optimized for each rate. This achieves slightly better FER waterfall curves but requires:

1. Storing $d+1$ puncturing patterns
2. Selecting pattern based on QBER estimate
3. Re-encoding if selection is incorrect

### Blind Advantage

Blind reconciliation uses a **single** mother code with **one** puncturing pattern. Rate adaptation occurs through the shortening mechanism without code selection. This simplifies implementation and eliminates estimation-dependent mode switching.

---

## References

[1] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Inf. Comput.*, vol. 12, no. 9-10, pp. 791–812, 2012.

[2] D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *Proc. IEEE Int. Symp. Inf. Theory*, pp. 1879–1883, 2009.

[3] T. Richardson and R. Urbanke, "The capacity of low-density parity-check codes under message-passing decoding," *IEEE Trans. Inf. Theory*, vol. 47, no. 2, pp. 599–618, 2001.

---

[← Return to Main Index](../index.md) | [Next: Hybrid Puncturing →](./hybrid_puncturing.md)

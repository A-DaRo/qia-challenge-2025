[← Return to Main Index](../index.md)

# 6.1 LDPC Codes for Information Reconciliation

## 6.1.1 The Source Coding Problem

### Slepian-Wolf Theorem

Information reconciliation is an instance of **source coding with side information** [1]. Alice possesses a string $X \in \{0,1\}^n$, and Bob possesses a correlated string $Y = X \oplus E$, where $E$ is the error pattern with Hamming weight $\text{wt}(E) \approx nQ$ for error rate $Q$.

**Theorem (Slepian-Wolf, 1973):** The minimum rate at which Alice must communicate to enable Bob to recover $X$ is:
$$
R_{\min} = H(X|Y) = h(Q)
$$

where $h(Q) = -Q\log_2 Q - (1-Q)\log_2(1-Q)$ is the binary entropy.

**Implications:**
- For $Q = 0.05$: $h(Q) \approx 0.286$, requiring $\geq 28.6\%$ of $n$ bits
- For $Q = 0.11$: $h(Q) \approx 0.500$, requiring $\geq 50\%$ of $n$ bits

### The Binary Symmetric Channel Model

The correlation between Alice's and Bob's strings is modeled as a **Binary Symmetric Channel (BSC)** with crossover probability $Q$:

```
                    1-Q
        X_i ────────────────── Y_i = X_i
             ╲                ╱
              ╲    Q      Q ╱
               ╲          ╱
                ╲────────╱
                    Q
        X_i ────────────────── Y_i = 1-X_i
```

**Channel Transition Probabilities:**
$$
P(Y=y|X=x) = \begin{cases}
1-Q & \text{if } y = x \\
Q & \text{if } y \neq x
\end{cases}
$$

## 6.1.2 LDPC Code Structure

### Parity-Check Matrix

An LDPC code is defined by a sparse **parity-check matrix** $H \in \{0,1\}^{m \times n}$ with:
- $n$ columns (variable nodes, codeword positions)
- $m = n - k$ rows (check nodes, parity constraints)
- Code rate $R = k/n = 1 - m/n$

**Sparsity Condition:** $H$ has constant or bounded column weight $d_v$ and row weight $d_c$:
$$
d_v \ll n, \quad d_c \ll m
$$

Typical values: $d_v \in [3, 6]$, $d_c \in [6, 20]$.

### Tanner Graph Representation

The code is equivalently represented as a **bipartite graph** $G = (V \cup C, E)$:

- **Variable nodes** $V = \{v_1, \ldots, v_n\}$: represent codeword bits
- **Check nodes** $C = \{c_1, \ldots, c_m\}$: represent parity constraints
- **Edges** $E$: $(v_j, c_i) \in E$ if $H_{ij} = 1$

**Code Definition:** $x \in \{0,1\}^n$ is a codeword if and only if:
$$
Hx = 0 \pmod{2}
$$

### Degree Distribution

The code performance is characterized by **degree distributions**:

$$
\lambda(x) = \sum_i \lambda_i x^{i-1}, \quad \rho(x) = \sum_i \rho_i x^{i-1}
$$

where $\lambda_i$ (resp. $\rho_i$) is the fraction of edges connected to degree-$i$ variable (resp. check) nodes.

**Design Rate:**
$$
R = 1 - \frac{\int_0^1 \rho(x)\,dx}{\int_0^1 \lambda(x)\,dx}
$$

## 6.1.3 Syndrome-Based Reconciliation

### Protocol Description

Rather than transmitting codewords, reconciliation uses **syndrome coding**:

1. Alice computes syndrome: $\Sigma = H \cdot X \in \{0,1\}^m$
2. Alice sends $\Sigma$ to Bob
3. Bob decodes: find $\hat{X}$ satisfying $H \cdot \hat{X} = \Sigma$ and $d_H(\hat{X}, Y)$ minimal

**Information Leakage:** The syndrome reveals exactly $m = (1-R)n$ bits of information about $X$.

### Decoding as Channel Coding

Bob's decoding problem is equivalent to decoding $Y$ over a BSC($Q$):

**Given:** Noisy observation $Y = X \oplus E$, syndrome $\Sigma = HX$

**Goal:** Find $\hat{X}$ such that $H\hat{X} = \Sigma$

**Equivalence:** This is equivalent to decoding $Y$ to find error pattern $\hat{E} = Y \oplus \hat{X}$ satisfying $H\hat{E} = HY \oplus \Sigma$.

## 6.1.4 Belief Propagation Decoding

### Log-Likelihood Ratios

The decoder operates on **log-likelihood ratios (LLRs)**:
$$
L_i = \log \frac{P(X_i = 0 | Y_i)}{P(X_i = 1 | Y_i)} = \log \frac{P(Y_i | X_i = 0)}{P(Y_i | X_i = 1)}
$$

For BSC($Q$):
$$
L_i^{(0)} = (1 - 2Y_i) \cdot \log \frac{1-Q}{Q}
$$

### Message Passing Algorithm

**Initialization:** Set variable-to-check messages $\mu_{v \to c}^{(0)} = L_v^{(0)}$.

**Check-to-Variable Update:**
$$
\mu_{c \to v}^{(t)} = 2\tanh^{-1}\left(\prod_{v' \in \mathcal{N}(c) \setminus v} \tanh\left(\frac{\mu_{v' \to c}^{(t-1)}}{2}\right)\right)
$$

**Variable-to-Check Update:**
$$
\mu_{v \to c}^{(t)} = L_v^{(0)} + \sum_{c' \in \mathcal{N}(v) \setminus c} \mu_{c' \to v}^{(t)}
$$

**Decision:**
$$
L_v^{(T)} = L_v^{(0)} + \sum_{c \in \mathcal{N}(v)} \mu_{c \to v}^{(T)}, \quad \hat{X}_v = \mathbf{1}[L_v^{(T)} < 0]
$$

### Convergence and Performance

**Density Evolution:** For infinite block length, BP threshold $Q^*$ satisfies:
$$
Q^* = \sup\{Q : \text{BP converges to zero error}\}
$$

**Finite-Length Behavior:** For finite $n$, performance degrades due to:
- **Cycles:** Short cycles in Tanner graph cause message correlation
- **Stopping Sets:** Subgraphs where BP cannot make progress

**Error Floor:** At low $Q$, residual errors from trapping sets dominate.

## 6.1.5 Rate Adaptation

### The Rate Mismatch Problem

A code designed for $Q_0$ fails when $Q > Q_0$ (under-provisioned) or wastes capacity when $Q < Q_0$ (over-provisioned).

**Solution:** Rate-compatible codes adapt a single **mother code** to varying rates.

### Puncturing (Rate Increase)

**Operation:** Delete $p$ positions from the codeword, yielding rate:
$$
R_{\text{punct}} = \frac{R_0}{1 - p/n}
$$

**Decoder Treatment:** Initialize punctured positions with $L_i = 0$ (maximum uncertainty).

### Shortening (Rate Decrease)

**Operation:** Fix $s$ positions to known values, yielding rate:
$$
R_{\text{short}} = \frac{R_0 - s/n}{1 - s/n}
$$

**Decoder Treatment:** Initialize shortened positions with $L_i = \pm\infty$ (perfect knowledge).

### Effective Rate Range

With modulation parameter $\delta = (p + s)/n$:
$$
R_{\text{eff}} \in \left[\frac{R_0 - \delta}{1 - \delta}, \frac{R_0}{1 - \delta}\right]
$$

## 6.1.6 Reconciliation Efficiency

### Definition

**Reconciliation efficiency** measures how close actual leakage is to the Slepian-Wolf bound:
$$
f = \frac{\text{leak}_{\text{EC}}}{n \cdot h(Q)} = \frac{(1-R_{\text{eff}}) \cdot n}{n \cdot h(Q)}
$$

**Ideal:** $f = 1$ (Shannon limit)
**Practical:** $f \in [1.05, 1.20]$ for well-designed LDPC codes

### Performance Summary

| QBER | $h(Q)$ | Required Rate | Typical $f$ |
|------|--------|---------------|-------------|
| 1% | 0.081 | 0.919 | 1.05 |
| 5% | 0.286 | 0.714 | 1.08 |
| 10% | 0.469 | 0.531 | 1.12 |
| 15% | 0.610 | 0.390 | 1.18 |

### Impact on Security

The reconciliation efficiency directly affects the extractable key length:
$$
\ell = n \cdot h_{\min}(r) - n \cdot h(Q) \cdot f - 2\log_2(1/\varepsilon) + 2
$$

A 10% inefficiency ($f = 1.10$) costs approximately $0.1 \cdot h(Q) \cdot n$ bits of secure key.

---

## References

[1] T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed. Wiley, 2006.

[2] D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *ISIT 2009*, arXiv:0901.2140.

[3] T. Richardson and R. Urbanke, *Modern Coding Theory*. Cambridge University Press, 2008.

---

[← Return to Main Index](../index.md) | [Next: Baseline Strategy →](./baseline_strategy.md)

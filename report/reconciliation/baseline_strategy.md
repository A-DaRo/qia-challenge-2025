[← Return to Main Index](../index.md)

# 6.2 Fixed-Rate Reconciliation Strategy

## Introduction

Information reconciliation corrects errors between Alice's string $\mathbf{x}$ and Bob's noisy observation $\mathbf{y}$, establishing the shared secret required for privacy amplification. This section analyzes the *fixed-rate* (or "baseline") strategy, where the code rate is determined *a priori* from QBER estimation, followed by a single-shot syndrome transmission.

The theoretical foundation is the Slepian-Wolf theorem [1], which establishes that lossless source coding with side information achieves the entropy limit $H(X|Y)$.

## Slepian-Wolf Problem Formulation

### Source Coding with Side Information

Alice possesses a uniformly random string $\mathbf{x} \in \{0,1\}^n$. Bob observes $\mathbf{y} = \mathbf{x} \oplus \mathbf{e}$, where $\mathbf{e} \sim \text{BSC}(Q)$ is the error pattern (binary symmetric channel with crossover probability $Q$).

**Slepian-Wolf Theorem** [1]: The minimum rate at which Alice can compress $\mathbf{x}$ such that Bob can recover it with vanishing error probability is:

$$
R_{\text{SW}} = H(X|Y) = H(\mathbf{e}) = h(Q)
$$

where $h(Q) = -Q\log_2 Q - (1-Q)\log_2(1-Q)$ is the binary entropy function.

### Syndrome Coding Realization

LDPC codes realize Slepian-Wolf compression via **syndrome coding** [2]. Given parity-check matrix $\mathbf{H} \in \{0,1\}^{m \times n}$:

1. **Alice** computes syndrome: $\mathbf{s} = \mathbf{H}\mathbf{x} \mod 2$
2. **Alice** transmits: $\mathbf{s}$ (length $m = n(1-R)$ bits)
3. **Bob** decodes: Find $\hat{\mathbf{x}}$ such that $\mathbf{H}\hat{\mathbf{x}} = \mathbf{s}$ and $d_H(\hat{\mathbf{x}}, \mathbf{y})$ is minimized

The effective compression rate is $R = 1 - m/n$, achieving the Slepian-Wolf limit when $R \to 1 - h(Q)$.

## Protocol Specification

### Phase A: Parameter Estimation

**Step 1**: Bob selects test sample $\mathcal{T} \subset \{1, \ldots, n\}$ with $|\mathcal{T}| = t$ uniformly at random.

**Step 2**: Alice and Bob publicly compare $\{(x_i, y_i) : i \in \mathcal{T}\}$.

**Step 3**: Compute empirical QBER:
$$
\hat{Q} = \frac{1}{t}\sum_{i \in \mathcal{T}} \mathbf{1}[x_i \neq y_i]
$$

**Step 4**: Apply finite-size correction (Hoeffding bound):
$$
Q^*_{\text{FK}} = \hat{Q} + \sqrt{\frac{\ln(1/\varepsilon_{\text{PE}})}{2t}}
$$

### Phase B: Rate Selection

**Step 5**: Compute target code rate from efficiency model:

$$
R_{\text{target}} = 1 - f(Q^*_{\text{FK}}) \cdot h(Q^*_{\text{FK}})
$$

where $f(Q) \geq 1$ is the reconciliation efficiency—the multiplicative overhead above the Shannon limit.

**Efficiency Model**: For practical LDPC codes [3]:

$$
f(Q) \approx \begin{cases}
1.05 - 1.10 & \text{if } Q \leq 0.05 \\
1.10 - 1.20 & \text{if } 0.05 < Q \leq 0.11 \\
\text{infeasible} & \text{if } Q > 0.11
\end{cases}
$$

The efficiency degrades near threshold due to finite code length and suboptimal degree distributions.

### Phase C: Syndrome Transmission

**Step 6**: Alice computes syndrome $\mathbf{s} = \mathbf{H}\mathbf{x}$.

**Step 7**: Alice transmits $(\mathbf{s}, R_{\text{target}}, h_{\text{verify}})$ to Bob, where $h_{\text{verify}}$ is a cryptographic hash of $\mathbf{x}$ for verification.

**Information Leakage**:
$$
\text{leak}_{\text{EC}} = |\mathbf{s}| = n(1 - R) \approx n \cdot f(Q) \cdot h(Q)
$$

### Phase D: Belief Propagation Decoding

**Step 8**: Bob initializes log-likelihood ratios (LLRs):

$$
\lambda_i = \ln\frac{P(x_i = 0 | y_i)}{P(x_i = 1 | y_i)} = (1 - 2y_i) \cdot \ln\frac{1 - Q^*_{\text{FK}}}{Q^*_{\text{FK}}}
$$

**Step 9**: Bob executes belief propagation (BP) decoding [4]:

*Variable-to-check messages:*
$$
\mu_{v \to c}^{(t)} = \lambda_v + \sum_{c' \in \mathcal{N}(v) \setminus c} \mu_{c' \to v}^{(t-1)}
$$

*Check-to-variable messages:*
$$
\mu_{c \to v}^{(t)} = 2\tanh^{-1}\left(\prod_{v' \in \mathcal{N}(c) \setminus v} \tanh\frac{\mu_{v' \to c}^{(t)}}{2}\right)
$$

**Step 10**: Termination occurs when:
- **Success**: $\mathbf{H}\hat{\mathbf{x}} = \mathbf{s}$ (syndrome match)
- **Failure**: Maximum iterations exceeded

## Theoretical Analysis

### Capacity-Achieving Behavior

For properly designed LDPC ensembles, the BP threshold $Q^*_{\text{BP}}$ approaches the Shannon limit [4]:

$$
\lim_{n \to \infty} Q^*_{\text{BP}} = Q^*_{\text{Shannon}} = h^{-1}(1 - R)
$$

**Density Evolution** [4] predicts the threshold by tracking the probability density of LLR messages through iterations.

### Finite-Length Performance

At finite block lengths, the word error rate (WER) exhibits a **waterfall** region:

$$
P_{\text{WER}}(Q) \approx \begin{cases}
\ll 10^{-6} & \text{if } Q < Q^*_{\text{BP}} - \Delta \\
\sim 1 & \text{if } Q > Q^*_{\text{BP}}
\end{cases}
$$

The transition width $\Delta$ scales as $\Delta \propto n^{-1/2}$ for random-like codes.

### Security Implications

The information leakage $\text{leak}_{\text{EC}}$ directly reduces extractable entropy:

$$
\ell \leq n \cdot h_{\min}(r) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon_{\text{sec}}) + 2
$$

**Critical observation**: Reconciliation efficiency $f > 1$ imposes a multiplicative penalty on key rate. For $Q = 0.05$:

- Shannon limit: $\text{leak}_{\text{Shannon}} = n \cdot h(0.05) \approx 0.286n$
- Practical ($f = 1.10$): $\text{leak}_{\text{practical}} \approx 0.315n$
- Overhead: $\sim 10\%$ reduction in extractable key

## Limitations of Fixed-Rate Strategy

### QBER Mismatch Risk

The fixed-rate strategy commits to $R_{\text{target}}$ based on *estimated* QBER. If the true QBER exceeds $Q^*_{\text{FK}}$:

$$
Q_{\text{true}} > Q^*_{\text{FK}} \implies P_{\text{decode fail}} \to 1
$$

This is the **catastrophic failure** mode: reconciliation fails entirely.

### Rate-Flexibility Tradeoff

Alternative approaches (blind reconciliation [5], interactive reconciliation) trade communication rounds for robustness to QBER uncertainty. See [Blind Reconciliation](./blind_strategy.md) for the adaptive alternative.

---

## References

[1] D. Slepian and J. K. Wolf, "Noiseless coding of correlated information sources," *IEEE Trans. Inf. Theory*, vol. 19, no. 4, pp. 471–480, 1973.

[2] D. J. C. MacKay, "Good error-correcting codes based on very sparse matrices," *IEEE Trans. Inf. Theory*, vol. 45, no. 2, pp. 399–431, 1999.

[3] D. Elkouss, J. Martinez-Mateo, and V. Martin, "Information reconciliation for quantum key distribution," *Quantum Inf. Comput.*, vol. 11, pp. 226–238, 2011.

[4] T. J. Richardson and R. L. Urbanke, *Modern Coding Theory*. Cambridge University Press, 2008.

[5] J. Martinez-Mateo, C. Pacher, M. Peev, A. Ciurana, and V. Martin, "Demystifying the information reconciliation protocol Cascade," *Quantum Inf. Comput.*, vol. 15, pp. 453–477, 2015.

---

[← Return to Main Index](../index.md) | [Next: Blind Reconciliation](./blind_strategy.md)

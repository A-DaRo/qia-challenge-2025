[← Return to Main Index](../index.md)

# 4.3 Statistical Sampling and Finite-Size Considerations

## Introduction

The finite-size regime of NSM protocols requires careful treatment of statistical fluctuations. When $n$ EPR pairs are generated and measured, the resulting bit strings exhibit random variations that must be bounded for security analysis. This section examines the statistical framework underlying finite-key security, connecting batched sampling strategies to the concentration inequalities that govern parameter estimation.

## Finite-Size Statistical Model

### Sample Statistics

For $n$ transmitted qubits with true QBER $Q$, the empirical error count follows:

$$
\hat{k} = \sum_{i \in \mathcal{T}} \mathbf{1}[a_i \neq b_i] \sim \text{Binomial}(|\mathcal{T}|, Q)
$$

where $\mathcal{T}$ is the test sample of size $t = |\mathcal{T}|$. The empirical QBER estimate is:

$$
\hat{Q} = \frac{\hat{k}}{t}
$$

### Concentration Bounds

The Hoeffding inequality [1] provides the finite-size confidence bound:

$$
P\left(|\hat{Q} - Q| \geq \delta\right) \leq 2\exp\left(-2t\delta^2\right)
$$

For security parameter $\varepsilon_{\text{PE}}$ (parameter estimation failure probability), the required sample size is:

$$
t \geq \frac{\ln(2/\varepsilon_{\text{PE}})}{2\delta^2}
$$

**Example**: For $\varepsilon_{\text{PE}} = 10^{-10}$ and $\delta = 0.01$ (1% precision):

$$
t \geq \frac{23.03}{0.0002} \approx 115{,}000 \text{ samples}
$$

This illustrates the fundamental tension in finite-key analysis: high-confidence parameter estimation requires substantial sample overhead.

### Conservative QBER Adjustment

The finite-key security proof requires Alice to use a *worst-case* QBER estimate [2]:

$$
Q^*_{\text{FK}} = \hat{Q} + \sqrt{\frac{\ln(1/\varepsilon_{\text{PE}})}{2t}}
$$

This one-sided Hoeffding bound ensures that the true QBER exceeds $Q^*_{\text{FK}}$ with probability at most $\varepsilon_{\text{PE}}$.

## Batched Processing

### Memory-Throughput Tradeoff

For computational efficiency, measurement records are processed in batches of size $B$. The tradeoff involves:

- **Small batches** ($B \ll n$): Lower peak memory, frequent I/O overhead
- **Large batches** ($B \sim n$): Higher memory consumption, reduced overhead

The optimal batch size balances memory constraints against processing efficiency:

$$
B_{\text{opt}} = \min\left(B_{\text{max}}, \sqrt{n \cdot \tau_{\text{IO}} / \tau_{\text{proc}}}\right)
$$

where $\tau_{\text{IO}}$ is the per-batch I/O cost and $\tau_{\text{proc}}$ is the per-sample processing cost.

### Streaming Statistics

Batched processing enables streaming computation of sufficient statistics. For QBER estimation, only the running sums are required:

$$
\text{After batch } k: \quad \hat{k}_{\text{total}} = \sum_{j=1}^{k} \hat{k}_j, \quad t_{\text{total}} = k \cdot B
$$

This permits memory-efficient processing of arbitrarily large sample sizes.

## Relation to Security Parameters

### The Sample Size–Security Tradeoff

The finite-key security bound [3] relates extractable key length to sample size:

$$
\ell \leq n \cdot h_{\min}(r) - t \cdot h(Q) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon_{\text{sec}}) + 2
$$

The test sample $t$ directly reduces the available entropy: each bit sacrificed for parameter estimation cannot contribute to the final key.

**Optimization Problem**: Choose $t$ to maximize $\ell$ subject to $\varepsilon_{\text{PE}}$ constraint:

$$
\max_t \left[ (n - t) \cdot h_{\min}(r) - \text{leak}_{\text{EC}}(Q^*_{\text{FK}}(t)) - \Delta_{\text{sec}} \right]
$$

The optimal allocation typically yields $t \sim \sqrt{n}$ for asymptotically large $n$.

### Death Valley in Finite-Size Regime

The finite-size penalties create a **Death Valley** phenomenon: for fixed $\varepsilon_{\text{sec}}$, there exists a minimum block length $n_{\text{min}}$ below which no positive key rate is achievable.

Setting $\ell = 0$ in the key length formula:

$$
n_{\text{min}} \cdot h_{\min}(r) = \text{leak}_{\text{EC}} + 2\log_2(1/\varepsilon_{\text{sec}}) - 2
$$

For $r = 0.75$, $\varepsilon_{\text{sec}} = 10^{-10}$, and $\text{leak}_{\text{EC}} \approx 0.2n$:

$$
n_{\text{min}} \approx \frac{64}{0.25 - 0.2} = 1280 \text{ bits}
$$

This threshold increases rapidly as storage noise $r$ approaches unity.

---

## References

[1] W. Hoeffding, "Probability inequalities for sums of bounded random variables," *J. Am. Stat. Assoc.*, vol. 58, no. 301, pp. 13–30, 1963.

[2] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

[3] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

---

[← Return to Main Index](../index.md) | [Next: Measurement Protocol](./measurement.md)

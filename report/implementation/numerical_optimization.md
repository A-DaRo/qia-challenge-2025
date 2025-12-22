[← Return to Main Index](../index.md)

# 9.2 Simulation Methodology

## Overview

The Caligo simulation validates theoretical NSM security bounds through discrete-event quantum network simulation. This section summarizes the computational methods enabling statistically significant exploration of the $(n, Q, r)$ parameter space.

---

## Numerical Framework

### LDPC Graph Construction

Sparse parity-check matrices are constructed using the **Progressive Edge-Growth (PEG)** algorithm [1] with **Approximate Cycle Extrinsic Message Degree (ACE)** constraints [2].

**Objective**: Maximize code girth (minimum cycle length) while avoiding low-weight near-codewords that cause error floors.

**Complexity**: $O(dn \log n)$ for degree-$d$ codes with $n$ variable nodes.

**ACE constraint** (Tian et al.):
$$
\text{ACE}(v) = \sum_{c \in \mathcal{N}(v)} \sum_{w \in \mathcal{N}(c) \setminus \{v\}} \max(0, \deg(w) - 2) \leq d_{\text{ACE}}
$$

Codes with $d_{\text{ACE}} \leq 5$ exhibit error floors below $10^{-10}$, sufficient for cryptographic applications.

### Belief Propagation Decoding

The **log-domain sum-product algorithm** computes posterior probabilities through message-passing on the factor graph [3]:

**Check-to-variable**:
$$
m_{c \to v}^{(t+1)} = 2 \tanh^{-1} \left( \prod_{w \in \mathcal{N}(c) \setminus \{v\}} \tanh\left( \frac{m_{w \to c}^{(t)}}{2} \right) \right)
$$

**Variable-to-check**:
$$
m_{v \to c}^{(t+1)} = \lambda_v + \sum_{c' \in \mathcal{N}(v) \setminus \{c\}} m_{c' \to v}^{(t)}
$$

where $\lambda_v = \log \frac{P(x_v = 0 | y_v)}{P(x_v = 1 | y_v)}$ is the channel log-likelihood ratio.

**Implementation**: Numba JIT-compiled kernels achieve $\sim 100\times$ speedup over pure Python, enabling $\sim 10^4$ decoding iterations per second on consumer hardware.

---

## Computational Optimizations

### JIT Compilation

Critical inner loops are compiled to native machine code using Numba's `@njit` decorator:

- **BP message updates**: Vectorized log-domain arithmetic
- **ACE detection**: Graph traversal with packed adjacency arrays
- **Toeplitz hashing**: FFT-based convolution for large keys

**Speedups**: $50-150\times$ over interpreted Python, approaching hand-optimized C performance.

### Memory Layout

Sparse matrices use **Compressed Sparse Row (CSR)** format:
- Contiguous memory access patterns
- Cache-friendly row traversal
- $O(nnz)$ storage vs. $O(mn)$ for dense

**Adjacency representation**: Packed integer arrays (`np.int32`) enable SIMD vectorization on modern CPUs.

---

## Statistical Significance

### Monte Carlo Simulation

Each parameter configuration $(n, Q, r)$ is evaluated through multiple independent protocol executions.

**Sample sizes**:
- Frame error rate (FER): $10^3 / \text{FER}$ trials for stable estimates
- Key rate: $\geq 100$ successful extractions for confidence intervals
- Death Valley boundary: Binary search with 20 iterations

### Confidence Intervals

For binomial FER estimation:
$$
\hat{\text{FER}} \pm z_{\alpha/2} \sqrt{\frac{\hat{\text{FER}}(1-\hat{\text{FER}})}{N}}
$$

with $z_{0.025} = 1.96$ for 95% confidence.

---

## Parameter Space Exploration

### Latin Hypercube Sampling

Initial parameter space coverage uses **Latin Hypercube Sampling** (LHS) [4]:

- Stratified sampling across $(n, Q, r, R)$
- Ensures uniform marginal distributions
- Efficient space-filling for high-dimensional exploration

### Active Learning

Subsequent sampling uses **Gaussian Process** surrogate models with **Expected Improvement** acquisition:

$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]
$$

targeting the Death Valley boundary and regime transitions.

---

## Validation Methodology

### Theoretical Bound Verification

The simulation verifies:

1. **Key length bounds**: $\ell_{\text{measured}} \leq \ell_{\text{Lupo}}$
2. **QBER thresholds**: Protocol aborts for $Q > Q_{\text{threshold}}$
3. **Min-entropy rates**: Extracted entropy $\leq n \cdot h_{\min}(r)$

### Regression Testing

Automated test suite ($\sim 450$ tests) ensures:
- Invariant preservation across refactoring
- Boundary condition handling
- Numerical stability under edge cases

---

## Computational Resources

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32 GB |
| Storage | 10 GB | 100 GB (for large sweeps) |

### Execution Time

| Task | Time per instance |
|------|-------------------|
| LDPC construction ($n = 4096$) | $\sim 0.5$ s |
| BP decoding (50 iterations) | $\sim 5$ ms |
| Full protocol execution | $\sim 100$ ms |
| Parameter sweep (1000 points) | $\sim 2$ hours |

---

## References

[1] X.-Y. Hu, E. Eleftheriou, and D. M. Arnold, "Regular and Irregular Progressive Edge-Growth Tanner Graphs," *IEEE Trans. Inf. Theory*, vol. 51, no. 1, pp. 386–398, 2005.

[2] T. Tian, C. R. Jones, J. D. Villasenor, and R. D. Wesel, "Selective Avoidance of Cycles in Irregular LDPC Code Construction," *IEEE Trans. Commun.*, vol. 52, no. 8, pp. 1242–1247, 2004.

[3] R. G. Gallager, "Low-Density Parity-Check Codes," *IRE Trans. Inf. Theory*, vol. 8, no. 1, pp. 21–28, 1962.

[4] M. D. McKay, R. J. Beckman, and W. J. Conover, "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code," *Technometrics*, vol. 21, no. 2, pp. 239–245, 1979.

---

[← Return to Main Index](../index.md)

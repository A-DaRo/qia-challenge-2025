[← Return to Main Index](../index.md)

# 7.2 Extractable Key Length: Finite-Size Security Bounds

## The Finite-Key Problem

### Asymptotic vs. Finite Regime

In the asymptotic limit ($n \to \infty$), the secret key rate approaches the Devetak-Winter bound:

$$
r_\infty = 1 - h(Q) - h(Q_{\text{storage}})
$$

However, real implementations operate with finite block lengths ($n \sim 10^3 - 10^6$), where **statistical penalties** dominate. These penalties arise from:

1. **Parameter estimation uncertainty**: QBER is estimated from finite samples
2. **Smoothing correction**: Min-entropy bounds require $\varepsilon$-smoothing
3. **Security overhead**: Achieving $\varepsilon_{\text{sec}}$-security costs entropy

The **Lupo key length formula** [1] provides tight finite-size bounds applicable to NSM protocols.

---

## Min-Entropy Framework

### Operational Definition

The **min-entropy** of $X$ given quantum side information $E$ is:

$$
H_{\min}(X | E) = -\log_2 P_g(X | E)
$$

where $P_g(X | E) = \max_{\{M_x\}} \sum_x p_X(x) \text{Tr}[M_x \rho_E^x]$ is the optimal guessing probability using quantum measurement $\{M_x\}$ on state $\rho_E^x$.

**Physical interpretation**: If $H_{\min}(X|E) = k$, the adversary's best strategy for guessing $X$ succeeds with probability at most $2^{-k}$.

### Smooth Min-Entropy

The **$\varepsilon$-smooth min-entropy** relaxes the worst-case bound by allowing the adversary's state to be $\varepsilon$-close to the actual state [2]:

$$
H_{\min}^\varepsilon(X | E) = \max_{\bar{\rho}: \frac{1}{2}\|\rho_{XE} - \bar{\rho}_{XE}\|_1 \leq \varepsilon} H_{\min}(X | \bar{E})
$$

This accounts for statistical fluctuations in parameter estimation.

---

## Min-Entropy Rate in the NSM

### Storage Channel Characterization

For depolarizing storage with parameter $r$, the per-bit min-entropy rate is bounded by [1, 3]:

$$
h_{\min}(r) = \max\left\{ h_{\text{DK}}(r), \; h_{\text{Lupo}}(r) \right\}
$$

**Dupuis-König (Collision) Bound** [3]:
$$
h_{\text{DK}}(r) = \Gamma \left[ 1 - \log_2(1 + 3r^2) \right]
$$
where $\Gamma = 1 - \log_2(2 + \sqrt{2}) \approx 0.228$.

**Lupo Virtual Erasure Bound** [1]:
$$
h_{\text{Lupo}}(r) = 1 - r
$$

### Bound Crossover

The two bounds intersect at $r^* \approx 0.25$:

| $r$ | $h_{\text{DK}}(r)$ | $h_{\text{Lupo}}(r)$ | $h_{\min}(r)$ |
|-----|--------------------|-----------------------|---------------|
| 0.1 | 0.216 | 0.9 | 0.216 |
| 0.25 | 0.166 | 0.75 | 0.166 |
| 0.5 | 0.035 | 0.5 | 0.035 |
| 0.75 | −0.087 | 0.25 | 0.25 |
| 0.9 | −0.15 | 0.1 | 0.1 |

For the Erven et al. experimental regime ($r = 0.75$), the Lupo bound dominates: $h_{\min}(0.75) = 0.25$.

---

## The Lupo Key Length Formula

### Derivation from Leftover Hash Lemma

Consider a reconciled key $X \in \{0,1\}^n$ with:
- Syndrome leakage: $|\Sigma|$ bits
- Adversary side information: $\rho_E$
- Target security: $\varepsilon_{\text{sec}}$

Applying the quantum Leftover Hash Lemma (see [§7.1](./toeplitz_hashing.md)):

$$
\varepsilon_{\text{sec}} \geq 2^{(\ell + |\Sigma|)/2 - H_{\min}^\varepsilon(X|E)/2}
$$

Solving for the extractable length $\ell$:

$$
\ell \leq H_{\min}^\varepsilon(X | E) - |\Sigma| - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2
$$

### Complete Formula

Combining with the NSM min-entropy bound:

$$
\boxed{\ell = \left\lfloor n \cdot h_{\min}(r) - n(1-R) - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2 \right\rfloor}
$$

where:
- $n$: reconciled block length
- $h_{\min}(r)$: per-bit min-entropy rate
- $R$: reconciliation code rate (syndrome leakage $= n(1-R)$)
- $\varepsilon_{\text{sec}}$: security parameter

---

## Security Penalty Analysis

### The $\Delta_{\text{sec}}$ Term

The **security penalty** is:

$$
\Delta_{\text{sec}} = 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) - 2
$$

| $\varepsilon_{\text{sec}}$ | $\Delta_{\text{sec}}$ (bits) |
|----------------------------|------------------------------|
| $10^{-6}$ | 38 |
| $10^{-8}$ | 51 |
| $10^{-10}$ | 64 |
| $10^{-12}$ | 78 |

**Physical interpretation**: This penalty quantifies the entropy cost of achieving distinguishability $\varepsilon_{\text{sec}}$ from the uniform distribution. Cryptographic applications typically require $\varepsilon_{\text{sec}} \leq 10^{-10}$.

### Finite-Size Dominance

For small blocks ($n \lesssim 10^4$), the security penalty dominates:

$$
\frac{\Delta_{\text{sec}}}{n \cdot h_{\min}(r)} \sim \frac{64}{2500} = 2.6\%
$$

For large blocks ($n \gtrsim 10^6$):

$$
\frac{\Delta_{\text{sec}}}{n \cdot h_{\min}(r)} \sim \frac{64}{2.5 \times 10^5} = 0.026\%
$$

The finite-size penalty becomes negligible in the asymptotic regime.

---

## Death Valley Phenomenon

### Definition

**Death Valley** is the parameter regime where finite-size penalties consume all extractable entropy:

$$
n \cdot h_{\min}(r) < n(1-R) + \Delta_{\text{sec}}
$$

yielding $\ell \leq 0$—no secure key can be extracted.

### Critical Threshold

Rearranging the Death Valley condition:

$$
h_{\min}(r) < 1 - R + \frac{\Delta_{\text{sec}}}{n}
$$

For fixed $(r, \varepsilon_{\text{sec}})$, there exists a **critical block length** $n^*$ below which key extraction fails:

$$
n^* = \frac{\Delta_{\text{sec}}}{h_{\min}(r) - (1 - R)}
$$

**Example** ($r = 0.75$, $R = 0.5$, $\varepsilon_{\text{sec}} = 10^{-10}$):

$$
n^* = \frac{64}{0.25 - 0.5} = \frac{64}{-0.25} < 0
$$

This indicates the code rate is too low—must increase $R$ or accept Death Valley.

For $R = 0.8$:
$$
n^* = \frac{64}{0.25 - 0.2} = \frac{64}{0.05} = 1280
$$

Blocks smaller than 1280 bits yield no key at these parameters.

### Parameter Space Mapping

The Death Valley boundary partitions the $(n, R, r)$ parameter space:

| $r$ | $h_{\min}(r)$ | Min $R$ for $n=4096$ | Min $R$ for $n=1024$ |
|-----|---------------|----------------------|----------------------|
| 0.9 | 0.10 | 0.984 | **Infeasible** |
| 0.75 | 0.25 | 0.766 | 0.937 |
| 0.5 | 0.035 | **Infeasible** | **Infeasible** |

---

## Reconciliation Efficiency Impact

### The Efficiency Factor

Define **reconciliation efficiency**:

$$
f = \frac{1 - R}{h(Q)}
$$

where $h(Q)$ is the Shannon limit for BSC($Q$). Perfect reconciliation achieves $f = 1$.

### Key Rate with Inefficiency

Incorporating $f > 1$:

$$
\ell = n \cdot h_{\min}(r) - n \cdot f \cdot h(Q) - \Delta_{\text{sec}}
$$

**Example** ($Q = 0.05$, $f = 1.08$, $r = 0.75$, $n = 10^4$):

$$
\begin{aligned}
\ell &= 10^4 \times 0.25 - 10^4 \times 1.08 \times h(0.05) - 64 \\
&= 2500 - 10^4 \times 1.08 \times 0.286 - 64 \\
&= 2500 - 3089 - 64 = -653
\end{aligned}
$$

**Death Valley**: These parameters yield negative key length.

### Escaping Death Valley

Options to achieve positive key:
1. **Increase $n$**: Dilute the $\Delta_{\text{sec}}/n$ term
2. **Reduce $f$**: Use more efficient reconciliation (longer codes)
3. **Reduce $Q$**: Improve quantum channel fidelity
4. **Increase $r$**: Accept less noisy storage (weaker security assumption)

---

## Numerical Implementation

### Algorithm

```
function ComputeKeyLength(n, R, r, Q, epsilon_sec):
    h_DK = 0.228 * (1 - log2(1 + 3*r^2))
    h_Lupo = 1 - r
    h_min = max(h_DK, h_Lupo)
    
    entropy_available = n * h_min
    syndrome_leak = n * (1 - R)
    security_penalty = 2 * log2(1/epsilon_sec) - 2
    
    raw_length = entropy_available - syndrome_leak - security_penalty
    return max(0, floor(raw_length))
```

### Sensitivity Analysis

The key length is most sensitive to:
1. **Block length $n$**: Linear dependence
2. **Storage noise $r$**: Nonlinear through $h_{\min}(r)$
3. **Code rate $R$**: Linear through $(1-R)$

Least sensitive to $\varepsilon_{\text{sec}}$ due to logarithmic dependence.

---

## References

[1] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

[2] R. Renner and R. König, "Universally Composable Privacy Amplification Against Quantum Adversaries," *TCC 2005*, LNCS 3378, pp. 407–425.

[3] F. Dupuis, O. Fawzi, and S. Wehner, "Entanglement Sampling and Applications," *IEEE Trans. Inf. Theory*, vol. 61, no. 2, pp. 1093–1112, 2015.

[4] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

---

[← Return to Main Index](../index.md) | [← Previous: Toeplitz Hashing](./toeplitz_hashing.md) | [Next: Key Derivation →](./key_derivation.md)

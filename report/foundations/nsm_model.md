[← Return to Main Index](../index.md)

# 2.1 The Noisy Storage Model

## 2.1.1 Motivation: Physical Constraints for Cryptographic Security

The impossibility results of Lo [1] and Mayers [2] establish that unconditionally secure oblivious transfer and bit commitment cannot be achieved with quantum communication alone. The Noisy Storage Model (NSM), introduced by Wehner, Schaffner, and Terhal [3] and fully developed by König, Wehner, and Wullschleger [4], circumvents this impossibility by imposing a physical constraint: **the adversary's quantum storage is subject to decoherence**.

Unlike the bounded-storage model [5], which requires explicit bounds on memory capacity, the NSM permits unbounded storage—security derives solely from the noise properties of the storage channel. This is physically realistic: all known quantum storage technologies exhibit decoherence over experimentally accessible timescales.

## 2.1.2 The Storage Channel

### Definition (Noisy Quantum Storage)

An adversary's quantum storage is modeled by a family $\{\mathcal{F}_t\}_{t \geq 0}$ of completely positive trace-preserving (CPTP) maps:
$$
\mathcal{F}_t : \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})
$$

where $\mathcal{B}(\mathcal{H})$ denotes the bounded operators on Hilbert space $\mathcal{H}$. The state $\rho$ stored at $t=0$ evolves to $\mathcal{F}_t(\rho)$ after time $t$.

### The Markovian Assumption

The noise is assumed to form a continuous one-parameter semigroup:
$$
\mathcal{F}_0 = \mathbb{1}, \qquad \mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}.
$$

**Physical Interpretation:** Noise accumulates monotonically—an adversary gains no advantage by delaying readout. Markovianity excludes memory effects where information could "revive" from the environment.

**Critical Remark:** This assumption is essential for security. A non-Markovian storage channel could in principle allow information recovery after an apparent decoherence event, invalidating the timing barrier mechanism. Section 8.4 discusses conditions under which this assumption holds.

### Protocol Enforcement

The protocol enforces a **waiting time** $\Delta t$ between quantum transmission and classical information revelation. Any quantum information retained by the adversary must pass through $\mathcal{F} = \mathcal{F}_{\Delta t}$. The adversary's actions are otherwise unrestricted: unlimited classical storage, instantaneous quantum computation, and perfect noise-free measurements are permitted.

## 2.1.3 The Depolarizing Channel

The canonical model for storage noise is the **$d$-dimensional depolarizing channel**:
$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbb{I}_d}{d}, \quad 0 \leq r \leq 1.
$$

For qubits ($d=2$):
$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbb{I}_2}{2}.
$$

**Parameters:**
- $r$: **Preservation probability**—the state survives unchanged with probability $r$
- $(1-r)$: **Depolarization probability**—the state is replaced by the maximally mixed state

**Kraus Representation:** The depolarizing channel admits the Kraus decomposition:
$$
\mathcal{N}_r(\rho) = \sum_{k=0}^{3} E_k \rho E_k^\dagger
$$
with $E_0 = \sqrt{\frac{1+3r}{4}}\mathbb{I}$, $E_1 = \sqrt{\frac{1-r}{4}}X$, $E_2 = \sqrt{\frac{1-r}{4}}Y$, $E_3 = \sqrt{\frac{1-r}{4}}Z$, where $X, Y, Z$ are the Pauli matrices.

### Classical Capacity

The **classical capacity** of $\mathcal{N}_r$ [6] is:
$$
C_{\mathcal{N}_r} = 1 - h\left(\frac{1+r}{2}\right)
$$
where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is the binary entropy.

**Asymptotic Behavior:**
- $r \to 1$ (no noise): $C_{\mathcal{N}_r} \to 1$ bit/qubit
- $r \to 0$ (full depolarization): $C_{\mathcal{N}_r} \to 0$ bits/qubit
- $r = 1/\sqrt{2}$ (critical threshold): $C_{\mathcal{N}_r} \approx 0.322$ bits/qubit

## 2.1.4 Security Condition

### Main Theorem (König, Wehner, Wullschleger [4])

Let $\mathcal{F} = \mathcal{N}^{\otimes \nu n}$ be an i.i.d. storage channel consisting of $\nu n$ independent applications of a single-qubit channel $\mathcal{N}$, where $\nu \in (0,1]$ is the **storage rate** and $n$ is the number of transmitted qubits. If the classical capacity $C_\mathcal{N}$ satisfies the **strong converse property**, then secure $\binom{2}{1}$-OT is achievable provided:
$$
\boxed{C_\mathcal{N} \cdot \nu < \frac{1}{2}}
$$

**Remarks:**
1. The strong converse property holds for the depolarizing channel [7], ensuring exponential decay of success probability for rates above capacity.
2. For $\nu = 1$ (full storage rate) and depolarizing noise, security requires $r < r_{\text{crit}}$ where $1 - h\bigl(\frac{1+r_{\text{crit}}}{2}\bigr) = 0.5$, yielding $r_{\text{crit}} \approx 0.707$.
3. The bound is tight in the sense that for $C_\mathcal{N} \cdot \nu \geq 1/2$, protocols exist that are insecure.

### The Strict Inequality: Channel vs. Storage QBER

For practical protocols, the security condition translates to a constraint on the **quantum bit error rate** (QBER). Define:
- $Q_{\text{channel}}$: QBER experienced by honest parties during quantum transmission
- $Q_{\text{storage}} = (1-r)/2$: Effective QBER introduced by storage noise

**Security Requirement [3]:**
$$
\boxed{Q_{\text{channel}} < Q_{\text{storage}}}
$$

**Interpretation:** The honest parties' quantum channel must be **strictly less noisy** than the adversary's storage. If $Q_{\text{channel}} \geq Q_{\text{storage}}$, the adversary can simulate the honest channel and gains no disadvantage from storage noise.

## 2.1.5 Optimal Adversary Strategy

### The All-or-Nothing Threshold (Wehner et al. [3])

For depolarizing storage with parameter $r$, the adversary's optimal strategy exhibits a **phase transition**:

**Case I: High Noise ($r < 1/\sqrt{2}$)**

The optimal attack is to measure immediately in the **Breidbart basis**:
$$
|0\rangle_B = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle, \quad |1\rangle_B = \sin(\pi/8)|0\rangle - \cos(\pi/8)|1\rangle.
$$

This measurement maximizes the guessing probability product $P_g(X|\sigma_+) \cdot P_g(X|\sigma_\times)$ over both BB84 bases.

**Case II: Low Noise ($r \geq 1/\sqrt{2}$)**

The optimal attack is to store the qubit unchanged and defer measurement until after classical information is revealed.

**Uncertainty Bound:** For the combined operation $\mathcal{S} = \mathcal{N} \circ P$ (partial measurement $P$ followed by noise $\mathcal{N}$):
$$
\max_\mathcal{S} \Delta(\mathcal{S}) = 
\begin{cases}
\frac{1}{2} + \frac{r}{2\sqrt{2}} & r < 1/\sqrt{2} \\
1 & r \geq 1/\sqrt{2}
\end{cases}
$$
where $\Delta(\mathcal{S})^2 = P_g(X|\mathcal{S}(\sigma_+)) \cdot P_g(X|\mathcal{S}(\sigma_\times))$.

## 2.1.6 QBER Security Thresholds

### Conservative Threshold: 11%

Schaffner [8] establishes that for individual-storage attacks with depolarizing noise:

> *"Secure oblivious transfer and secure identification can be achieved as long as the quantum bit-error rate does not exceed 11%."*

This threshold is derived from the min-entropy bound on the adversary's information, accounting for both channel errors and storage noise effects.

### Hard Limit: 22%

The theoretical maximum QBER for any NSM protocol is approximately **22%** (Lupo et al. [9]). This limit arises from:

1. **Shannon Bound:** Error correction cannot reliably correct beyond $h(Q) \to 1$
2. **Min-Entropy Exhaustion:** For $Q > 22\%$, the smooth min-entropy $H_{\min}^\varepsilon(X|E)$ becomes insufficient to cover reconciliation leakage

**Finite-Size Caveat:** For block lengths $n < 10^6$, practical limits are lower due to finite-size penalties in both parameter estimation (Hoeffding bounds) and privacy amplification ($\log_2(1/\varepsilon)$ terms).

## 2.1.7 Min-Entropy Bounds

### The Dupuis-König Bound [10]

For i.i.d. depolarizing storage, the smooth min-entropy rate is bounded by:
$$
h_{\min}(r) \geq \Gamma\bigl[1 - \log_2(1 + 3r^2)\bigr]
$$
where $\Gamma(x) = x$ for $x \geq 1/2$ and $\Gamma(x) = g^{-1}(x)$ for $x < 1/2$, with $g(y) = -y\log_2 y - (1-y)\log_2(1-y) + y - 1$.

### The Lupo Virtual Erasure Bound [9]

An alternative bound, tight for low-noise regimes:
$$
h_{\min}(r) \geq 1 - r.
$$

**Combined Bound:** The best min-entropy rate is:
$$
h_{\min}(r) = \max\left\{\Gamma\bigl[1 - \log_2(1 + 3r^2)\bigr], 1 - r\right\}.
$$

For $r \lesssim 0.25$, the Dupuis-König bound dominates; for $r \gtrsim 0.25$, the Lupo bound dominates.

## 2.1.8 Physical Justification

### Unavoidable Decoherence

Quantum storage noise is not an artificial assumption but a fundamental physical constraint:

| Platform | Dominant Noise | Typical $T_2$ |
|----------|----------------|---------------|
| Photonic qubits | Fiber loss, absorption | N/A (no storage) |
| Atomic ensembles | Spontaneous emission, collisions | 1–100 μs |
| Superconducting qubits | $T_1$, $T_2$ relaxation | 10–100 μs |
| Ion traps | Heating, motional decoherence | 1–10 s |
| NV centers | Spin-bath dephasing | 1–10 ms |

### Transfer-Induced Noise

Even with perfect quantum memories, the **encoding operation** of an unknown state into a stored qubit introduces noise [3]:

> *"The transfer of the state of a (photonic) qubit onto a different physical carrier (such as an atomic ensemble) is typically already noisy."*

In the fault-tolerant regime, encoding an unknown state into a logical qubit is not a fault-tolerant operation—**residual noise** remains from the unprotected encoding step.

---

## References

[1] H.-K. Lo, "Insecurity of quantum secure computations," *Phys. Rev. A* **56**, 1154 (1997).

[2] D. Mayers, "Unconditionally secure quantum bit commitment is impossible," *Phys. Rev. Lett.* **78**, 3414 (1997).

[3] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008).

[4] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[5] I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner, "Cryptography in the bounded quantum-storage model," *SIAM J. Comput.* **37**, 1865 (2008).

[6] C. King, "The capacity of the quantum depolarizing channel," *IEEE Trans. Inf. Theory* **49**, 221 (2003).

[7] M. M. Wilde, A. Winter, and D. Yang, "Strong converse for the classical capacity of entanglement-breaking and Hadamard channels," *Commun. Math. Phys.* **331**, 593 (2014).

[8] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus (2007).

[9] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

[10] A. Dupuis, O. Fawzi, and S. Wehner, "Entanglement Sampling and Applications," *IEEE Trans. Inf. Theory* **61**, 1093 (2015).

---

[← Return to Main Index](../index.md) | [Next: Cryptographic Primitives →](./primitives.md)

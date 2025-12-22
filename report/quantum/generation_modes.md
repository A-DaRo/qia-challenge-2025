[← Return to Main Index](../index.md)

# 4.2 Entanglement Generation: Theoretical Framework

## Introduction

The quantum phase of an NSM protocol establishes correlated measurement outcomes between Alice and Bob through distributed Bell state measurements. This section presents the theoretical framework for entanglement-based protocols, distinguishing between *sequential generation* (preserving temporal correlations) and *statistical sampling* (exploiting i.i.d. assumptions for computational efficiency).

The key physical insight is that for security analysis, the *timing* of individual generation events matters only insofar as it enforces the NSM causality constraint: Bob cannot access Alice's basis information until time $\Delta t$ has elapsed, during which any stored quantum information decoheres.

## Bell State Distribution

### Ideal Protocol

In the idealized protocol, Alice and Bob share $n$ copies of the maximally entangled Bell state:

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}}\left(|00\rangle + |11\rangle\right)
$$

Each party independently selects a measurement basis $\theta_i \in \{0, 1\}$ (computational $Z$ or Hadamard $X$) uniformly at random and measures their respective qubit. The joint statistics satisfy:

$$
P(a_i = b_i \,|\, \theta_i^A = \theta_i^B) = 1 \quad \text{(perfect correlation)}
$$

$$
P(a_i = b_i \,|\, \theta_i^A \neq \theta_i^B) = \frac{1}{2} \quad \text{(independent uniform)}
$$

### Werner State Model

In practice, imperfect sources produce Werner states [1]:

$$
\rho_F = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\frac{\mathbb{I}_4}{4}
$$

where $F \in (1/2, 1]$ is the fidelity to the target Bell state. The correlation function becomes:

$$
P(a_i \neq b_i \,|\, \theta_i^A = \theta_i^B) = \frac{1-F}{2} = Q_{\text{intrinsic}}
$$

This intrinsic error rate contributes to the measured QBER alongside detector imperfections.

## Generation Modalities

### Sequential (Discrete-Event) Generation

**Definition**: In sequential mode, EPR pairs are generated one-at-a-time within a discrete-event simulation framework. Each generation event $i \in \{1, \ldots, n\}$ occurs at simulation time $t_i$, with strict ordering $t_1 < t_2 < \cdots < t_n$.

**Physical Fidelity**: Sequential generation models:
- Network propagation delays (classical and quantum channels)
- Finite qubit coherence times during transmission
- Detector timing jitter and dead time effects

**Timing Model**: The simulation advances according to:

$$
t_{i+1} = t_i + \tau_{\text{gen}} + \tau_{\text{prop}} + \tau_{\text{meas}}
$$

where $\tau_{\text{gen}}$ is the source repetition period, $\tau_{\text{prop}}$ the channel propagation delay, and $\tau_{\text{meas}}$ the measurement duration.

**Security Enforcement**: After completing all measurements, the protocol enforces a waiting period:

$$
t_{\text{reveal}} \geq t_n + \Delta t
$$

This ensures any quantum information stored by a dishonest party experiences the full decoherence characterized by $\mathcal{F}_{\Delta t}$.

### Statistical (i.i.d.) Generation

**Definition**: Statistical generation exploits the i.i.d. assumption inherent in NSM security proofs. Rather than simulating temporal dynamics, measurement outcomes are sampled directly from the appropriate probability distributions.

**Justification**: The security proof of König et al. [2] requires only that:
1. Each transmitted qubit is independently and identically distributed
2. Basis choices are uniformly random and independent
3. The waiting time $\Delta t$ elapses before basis revelation

The *order* in which pairs are generated and the *intermediate timing* are irrelevant to the security guarantee—only the final statistics matter.

**Sampling Model**: For $n$ EPR pairs with channel fidelity $F$:

$$
\begin{aligned}
\theta_i^A, \theta_i^B &\sim \text{Bernoulli}(1/2) \quad \text{(basis selection)} \\
a_i &\sim \text{Bernoulli}(1/2) \quad \text{(Alice's outcome)} \\
e_i &\sim \text{Bernoulli}(Q) \quad \text{(error indicator)} \\
b_i &= a_i \oplus e_i \cdot \mathbf{1}[\theta_i^A = \theta_i^B] \quad \text{(Bob's outcome)}
\end{aligned}
$$

where $Q = (1-F)/2 + e_{\text{det}}$ includes both intrinsic and detector errors.

**Computational Advantage**: Statistical generation achieves $O(n)$ complexity with embarrassingly parallel execution, enabling Monte Carlo studies over $n \sim 10^6$ pairs.

## Comparison of Modalities

| Property | Sequential | Statistical |
|----------|-----------|-------------|
| **Time complexity** | $O(n \cdot \tau_{\text{sim}})$ | $O(n)$ |
| **Parallelization** | Limited (event dependencies) | Embarrassingly parallel |
| **Timing model** | Full discrete-event simulation | Abstracted (i.i.d. assumption) |
| **Security proof** | Direct correspondence to protocol | Relies on i.i.d. equivalence |
| **Use case** | Protocol validation, timing analysis | Parameter sweeps, Monte Carlo |

## Validity of the i.i.d. Assumption

The i.i.d. assumption underlying statistical generation requires scrutiny. The NSM security proof [2, Theorem 4.1] assumes:

> *"The adversary's storage consists of $\nu n$ uses of a memoryless channel $\mathcal{N}$, acting independently on each stored qubit."*

This assumption may fail if:

1. **Correlated noise**: Source fluctuations introduce temporal correlations between successive emissions
2. **Collective attacks**: The adversary performs joint operations on multiple stored qubits
3. **Non-Markovian storage**: Memory effects in the decoherence process

For depolarizing storage and individual attacks, the i.i.d. assumption is justified by the tensorization property of classical capacity:

$$
C(\mathcal{N}^{\otimes k}) = k \cdot C(\mathcal{N})
$$

However, for general storage channels, the i.i.d. equivalence remains an open theoretical question [3].

---

## References

[1] R. F. Werner, "Quantum states with Einstein-Podolsky-Rosen correlations admitting a hidden-variable model," *Phys. Rev. A*, vol. 40, pp. 4277–4281, 1989.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[3] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.*, vol. 100, 220502, 2008.

---

[← Return to Main Index](../index.md) | [Next: Batching](./batching.md)

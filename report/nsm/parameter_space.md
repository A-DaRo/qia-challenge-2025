[← Return to Main Index](../index.md)

# 8.1 NSM Parameter Space: Security Conditions and Physical Constraints

## Introduction

The Noisy Storage Model (NSM) [1] provides information-theoretic security under the physical assumption that an adversary's quantum storage undergoes decoherence during a prescribed waiting period. This section rigorously characterizes the parameter space within which security is achievable, distinguishing between fundamental limits imposed by physics and operational constraints arising from finite-size effects.

## Formal Definition of NSM

### Storage Channel Model

The adversary's quantum storage is modeled as a completely positive trace-preserving (CPTP) map:

$$
\mathcal{F}: \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})
$$

where $\mathcal{B}(\mathcal{H})$ denotes bounded operators on Hilbert space $\mathcal{H}$.

**Markovian Assumption**: The storage satisfies the semigroup property:

$$
\mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2} \quad \forall t_1, t_2 > 0
$$

This implies that delaying readout never improves the adversary's information—a crucial assumption for security.

### Security Model

**Definition** (NSM Adversary Model) [1]: The adversary possesses:
- Unlimited classical storage and computation
- Noise-free quantum operations (preparation, measurement, gates)
- Noisy quantum storage: at each waiting period $\Delta t$, stored qubits undergo $\mathcal{F} = \mathcal{F}_{\Delta t}$
- Storage capacity: can store at most $\nu n$ qubits from $n$ transmitted

**Critical constraint**: Only the quantum storage is bounded; all other resources are unlimited.

## Core NSM Parameters

### Storage Noise Parameter $r$

For the qubit depolarizing channel:

$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbb{I}}{2}
$$

The parameter $r \in [0,1]$ represents the probability that the stored state survives unchanged:
- $r = 1$: Perfect storage (no security)
- $r = 0$: Complete depolarization (maximal security)

**Physical interpretation**: For a memory with $T_1$ (relaxation) and $T_2$ (dephasing) times:

$$
r(\Delta t) \approx e^{-\Gamma \Delta t}, \quad \Gamma = \frac{1}{2}\left(\frac{1}{T_1} + \frac{1}{T_2}\right)
$$

### Storage Rate $\nu$

The fraction of transmitted qubits the adversary can store coherently:

$$
\nu = \frac{|\text{stored qubits}|}{n} \in [0, 1]
$$

For $n$ transmitted qubits, the adversary's storage acts as $\mathcal{F}^{\otimes \lfloor \nu n \rfloor}$.

### Waiting Time $\Delta t$

The protocol-enforced delay between measurement completion and basis revelation:

$$
t_{\text{reveal}} \geq t_{\text{measure}} + \Delta t
$$

During $\Delta t$, any stored quantum information decoheres according to $\mathcal{F}_{\Delta t}$.

## Security Conditions

### Fundamental Security Requirement

**Theorem** (König-Wehner-Wullschleger) [2]: Let $\mathcal{F}$ be the adversary's storage channel with classical capacity $C_\mathcal{F}$. NSM-based $\binom{2}{1}$-OT is secure if:

$$
\boxed{C_\mathcal{F} \cdot \nu < \frac{1}{2}}
$$

**Physical interpretation**: The adversary's total information gain per transmitted qubit is bounded below $1/2$ bit.

### Depolarizing Channel Capacity

For the qubit depolarizing channel $\mathcal{N}_r$, the classical capacity is [3]:

$$
C_{\mathcal{N}_r} = 1 - h\left(\frac{1+r}{2}\right) = 1 - h\left(\frac{1-r}{2}\right)
$$

where $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$ is the binary entropy.

**Asymptotic expansion** for $r$ near 1:

$$
C_{\mathcal{N}_r} \approx \frac{(1-r)^2}{2\ln 2} + O((1-r)^3)
$$

### QBER Thresholds

**Conservative Threshold** (Schaffner) [4]: For individual storage attacks:

$$
Q < Q_{\text{conservative}} = 0.11 \quad (11\%)
$$

**Hard Limit** (Lupo) [5]: For general collective attacks:

$$
Q < Q_{\text{hard}} = 0.22 \quad (22\%)
$$

**Critical insight**: The 22% threshold corresponds to the point where:

$$
Q_{\text{hard}} = \frac{1 - r_{\text{crit}}}{2} \implies r_{\text{crit}} = 0.56
$$

Beyond this, even perfect storage ($\nu = 1$) cannot extract sufficient information.

## Channel Parameter Space

### Source Imperfections

**Bell state fidelity**: For Werner state $\rho_F = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\mathbb{I}/4$:

$$
Q_{\text{source}} = \frac{1-F}{2}
$$

**Validity**: This assumes isotropic depolarization. Anisotropic noise (e.g., preferential dephasing) may exhibit different QBER-fidelity relationships.

### Detection Imperfections

The total QBER decomposes as:

$$
Q = Q_{\text{source}} + Q_{\text{detector}} + Q_{\text{dark}}
$$

where:
- $Q_{\text{detector}} = e_{\text{det}}$ (intrinsic detector error)
- $Q_{\text{dark}} = (1-\eta) P_{\text{dark}} / 2$ (dark count contribution)

### Experimental Parameter Regimes

**Erven et al. (2014)** [6]:

| Parameter | Value | Physical Origin |
|-----------|-------|-----------------|
| $r$ | 0.75 | $\Delta t = 1$ ms, room-temperature memory |
| $\nu$ | 0.002 | Limited storage capacity ($\sim 160$ qubits from $8 \times 10^7$) |
| $F$ | 0.975 | PDC source with filtering |
| $\eta$ | 0.015 | Long fiber + detector efficiency |
| $Q$ | 0.029 | Composite error rate |

**Security verification**: $C_{\mathcal{N}_{0.75}} \cdot 0.002 \approx 0.189 \times 0.002 = 0.0004 \ll 1/2$ ✓

## Finite-Size Constraints

### Minimum Block Length

The extractable key length vanishes when:

$$
n \cdot h_{\min}(r) < \text{leak}_{\text{EC}} + 2\log_2(1/\varepsilon_{\text{sec}}) - 2
$$

This defines the **Death Valley** boundary in parameter space.

### Statistical Estimation Requirements

For QBER estimation with precision $\delta$ and confidence $1 - \varepsilon_{\text{PE}}$:

$$
t \geq \frac{\ln(2/\varepsilon_{\text{PE}})}{2\delta^2}
$$

The test sample $t$ consumes bits that cannot contribute to the final key.

## Critical Assessment

### Validity of Depolarizing Model

The depolarizing channel is a **worst-case assumption** for symmetric noise. Real storage devices may exhibit:

1. **Anisotropic noise**: Dephasing-dominated ($T_2 \ll T_1$) favors Z-basis storage
2. **Non-Markovian dynamics**: Memory effects in spin bath environments
3. **Correlated errors**: Multi-qubit memories with collective decoherence

For dephasing noise $\mathcal{D}_\gamma(\rho) = (1-\gamma)\rho + \gamma Z\rho Z$:

$$
C_{\mathcal{D}_\gamma} = 1 - h(\gamma) > C_{\mathcal{N}_r} \quad \text{for same error rate}
$$

The depolarizing assumption may thus *underestimate* the adversary's capability in some physical implementations.

### Collective Attack Considerations

The security proof of [2] considers **individual attacks** where each stored qubit is processed independently. For **collective attacks** with entangled ancillas, tighter bounds apply [5].

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.*, vol. 100, 220502, 2008.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[3] C. King, "The capacity of the quantum depolarizing channel," *IEEE Trans. Inf. Theory*, vol. 49, no. 1, pp. 221–229, 2003.

[4] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus, 2007.

[5] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

[6] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

---

[← Return to Main Index](../index.md) | [Next: Noise Models](./noise_models.md)

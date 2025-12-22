[← Return to Main Index](../index.md)

# 8.2 Physical Mapping: From NSM Parameters to Decoherence Times

## The Translation Problem

The Noisy Storage Model provides an abstract security framework expressed in information-theoretic terms: storage noise parameter $r$, classical capacity $C_\mathcal{N}$, and storage rate $\nu$. Physical implementations operate at the qubit level: $T_1$ relaxation, $T_2$ dephasing, and gate fidelities. Bridging this gap requires precise mathematical mappings grounded in quantum decoherence theory.

---

## Depolarizing Channel Parametrization

### The Wehner Formalism

Following Wehner et al. [1], adversarial quantum storage is modeled as a depolarizing channel:

$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbb{I}}{2}
$$

where $r \in [0,1]$ is the **preservation probability** (storage fidelity).

**Physical interpretation**:
- $r = 1$: Perfect storage (no decoherence)
- $r = 0$: Complete depolarization (thermal equilibrium)
- $0 < r < 1$: Partial preservation during wait time $\Delta t$

The adversary cannot control $r$—it is determined by the physical storage medium and protocol timing.

### Time Evolution

For Markovian decoherence, $r$ decays exponentially with storage time:

$$
r(\Delta t) = e^{-\Gamma \Delta t}
$$

where $\Gamma$ is the **decoherence rate** determined by the storage medium's relaxation properties.

---

## $T_1$ and $T_2$ Correspondence

### Relaxation Processes

**Energy relaxation** ($T_1$): Decay of $|1\rangle$ population to $|0\rangle$

$$
\frac{d\rho_{11}}{dt} = -\frac{1}{T_1}\rho_{11}
$$

**Phase relaxation** ($T_2$): Decay of off-diagonal coherence

$$
\frac{d\rho_{01}}{dt} = -\frac{1}{T_2}\rho_{01}
$$

The fundamental constraint $T_2 \leq 2T_1$ arises because energy relaxation contributes to dephasing.

### Effective Decoherence Rate

For a qubit subject to both $T_1$ and $T_2$ processes, the effective depolarization rate is:

$$
\Gamma = \frac{1}{2}\left(\frac{1}{T_1} + \frac{1}{T_2}\right)
$$

**Derivation**: The depolarizing channel results from equal-weight Pauli errors:

$$
\mathcal{N}_r(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)
$$

The bit-flip component ($X$) arises from $T_1$; the phase-flip component ($Z$) from pure dephasing. For symmetric noise, $r = 1 - \frac{4p}{3}$ with $p$ determined by the combined rate.

### Physical Parameter Extraction

Given target storage fidelity $r$ and wait time $\Delta t$:

$$
\Gamma = -\frac{\ln r}{\Delta t}
$$

**Example** ($r = 0.75$, $\Delta t = 1$ ms):

$$
\Gamma = -\frac{\ln(0.75)}{10^6 \text{ ns}} = 2.88 \times 10^{-7} \text{ ns}^{-1}
$$

If $T_1 = T_2 \equiv T$:
$$
T = \frac{1}{\Gamma} = 3.47 \times 10^6 \text{ ns} = 3.47 \text{ ms}
$$

### Storage Platform Comparison

| Platform | Typical $T_1$ | Typical $T_2$ | $\Gamma$ | $r$ at $\Delta t = 1$ ms |
|----------|---------------|---------------|----------|--------------------------|
| Transmon (IBM) | 100 μs | 100 μs | $10^{-5}$ ns$^{-1}$ | $4.5 \times 10^{-5}$ |
| Ion trap | 1 s | 10 ms | $5 \times 10^{-8}$ ns$^{-1}$ | 0.95 |
| NV center | 10 ms | 1 ms | $5.5 \times 10^{-7}$ ns$^{-1}$ | 0.58 |
| Room temp. atomic | 100 μs | 100 μs | $10^{-5}$ ns$^{-1}$ | $4.5 \times 10^{-5}$ |

**Observation**: Current superconducting qubits ($T_2 \sim 100$ μs) are far too coherent for NSM security at $\Delta t = 1$ ms. The protocol requires either:
1. Deliberately degraded storage (adding noise)
2. Longer wait times ($\Delta t \gg T_2$)
3. Room-temperature storage with short coherence

---

## Classical Capacity Constraint

### König's Security Condition

Security requires the adversary's storage capacity to be bounded [2]:

$$
C_\mathcal{N} \cdot \nu < \frac{1}{2}
$$

where:
- $C_\mathcal{N} = 1 - H_2\left(\frac{1-r}{2}\right)$: Classical capacity of depolarizing channel
- $\nu$: Storage rate (fraction of transmitted qubits storable)

### Capacity Calculation

The classical capacity of the depolarizing channel is [3]:

$$
C_{\mathcal{N}_r} = 1 - h\left(\frac{1+r}{2}\right)
$$

where $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$ is the binary entropy.

| $r$ | $(1+r)/2$ | $h((1+r)/2)$ | $C_{\mathcal{N}_r}$ |
|-----|-----------|--------------|---------------------|
| 0.0 | 0.5 | 1.0 | 0 |
| 0.5 | 0.75 | 0.811 | 0.189 |
| 0.75 | 0.875 | 0.544 | 0.456 |
| 0.9 | 0.95 | 0.286 | 0.714 |
| 1.0 | 1.0 | 0 | 1.0 |

### Experimental Parameters (Erven et al.)

The Erven et al. [4] demonstration used:

| Parameter | Value | Physical Source |
|-----------|-------|-----------------|
| $r$ | 0.75 | Room-temperature storage, $\Delta t = 1$ ms |
| $\nu$ | 0.002 | 2 qubits stored per 1000 transmitted |
| $C_\mathcal{N}$ | 0.456 | Depolarizing capacity at $r = 0.75$ |

**Security check**:
$$
C_\mathcal{N} \cdot \nu = 0.456 \times 0.002 = 9.1 \times 10^{-4} \ll 0.5 \quad \checkmark
$$

---

## QBER-Fidelity Relationship

### Channel vs. Storage QBER

Two distinct error sources contribute to the observed QBER:

1. **Channel QBER** ($Q_{\text{ch}}$): Errors from transmission (fiber loss, detector dark counts, polarization misalignment)

2. **Storage QBER** ($Q_{\text{st}}$): Errors from adversary's imperfect storage

For security, we require:

$$
Q_{\text{ch}} < Q_{\text{st}} = \frac{1-r}{2}
$$

### Erven QBER Model

The total observed QBER is [4, Eq. 8]:

$$
Q_{\text{ch}} = \frac{1-F}{2} + e_{\text{det}} + \frac{(1-\eta)P_{\text{dark}}}{2}
$$

where:
- $F$: Bell state fidelity
- $e_{\text{det}}$: Detector misalignment error
- $\eta$: Detection efficiency
- $P_{\text{dark}}$: Dark count probability

**Example calculation** (Erven parameters):
$$
Q_{\text{ch}} = \frac{1-0.975}{2} + 0.015 + \frac{(1-0.06) \times 5 \times 10^{-6}}{2} \approx 0.0275
$$

**Security margin**:
$$
Q_{\text{st}} - Q_{\text{ch}} = 0.125 - 0.0275 = 0.0975
$$

A 9.75% margin validates the experimental security.

---

## NetSquid Parameter Mapping

### Simulation Configuration

The SquidASM simulation maps abstract NSM parameters to NetSquid primitives:

| NSM Parameter | NetSquid Equivalent | Configuration Path |
|---------------|---------------------|-------------------|
| $r$ | `depolar_rate` in `DepolarNoiseModel` | `QuantumProcessor.mem_noise_models` |
| $\Delta t$ | Event delay in `TimingBarrier` | Protocol timing |
| $F$ | `fidelity` in `QLinkConfig` | `StackNetworkConfig.link` |
| $\eta$ | `detector_efficiency` | `HeraldedDoubleClickModelParameters` |

### Noise Model Injection

Direct configuration bypasses SquidASM's simplified noise path:

1. Construct `T1T2NoiseModel(T1, T2)` from target $r$ and $\Delta t$
2. Attach to `QuantumProcessor.mem_noise_models`
3. Verify $r(\Delta t) = e^{-\Gamma \Delta t}$ through tomography

See [noise_models.md](./noise_models.md) for implementation details.

---

## Critique: Validity of Parameters

### The Markovian Assumption

NSM security proofs assume **Markovian** decoherence:

$$
\mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}
$$

**Physical validity**: This holds for:
- Weak system-environment coupling
- Memoryless environment (infinite temperature bath)
- Short correlation times ($\tau_c \ll \Delta t$)

**Failure modes**:
- Non-Markovian noise (memory effects, structured environments)
- Coherent errors (unitary rotations)
- Correlated noise across qubits

### The Independent Attack Assumption

The simulation assumes the adversary attacks each qubit independently. Collective attacks (joint measurements across multiple qubits) are not modeled.

**Justification**: König et al. [2] proved that for the depolarizing channel, individual attacks are optimal in the asymptotic limit. For finite-size analysis, this remains an open question.

### Parameter Regime Limitations

The Caligo simulation operates in:
- $r \in [0.5, 0.9]$: Intermediate noise regime
- $\Delta t \in [10^5, 10^7]$ ns: Millisecond-scale waits
- $n \in [10^3, 10^6]$: Moderate block lengths

**Extrapolation warning**: Results should not be extrapolated to:
- Ultra-low noise ($r > 0.95$): Capacity constraint may be violated
- Very short waits ($\Delta t < 100$ μs): Non-Markovian effects
- Long keys ($n > 10^8$): Asymptotic regime (different analysis)

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.*, vol. 100, 220502, 2008.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[3] C. King, "The capacity of the quantum depolarizing channel," *IEEE Trans. Inf. Theory*, vol. 49, no. 1, pp. 221–229, 2003.

[4] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

---

[← Return to Main Index](../index.md) | [Next: Noise Models →](./noise_models.md)

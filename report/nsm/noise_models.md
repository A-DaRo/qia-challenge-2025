[← Return to Main Index](../index.md)

# 8.3 Quantum Noise Models: From Theory to Simulation

## Introduction

Discrete-event quantum network simulators require precise specification of noise models at multiple levels: EPR source imperfections, channel losses, detector errors, and memory decoherence. This section examines the theoretical foundations of these noise models and their realization in the SquidASM/NetSquid simulation stack, with particular attention to the assumptions required for NSM security analysis.

## CPTP Noise Channels

### Kraus Representation

Any quantum noise channel $\mathcal{E}$ admits a Kraus decomposition:

$$
\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger, \quad \sum_k K_k^\dagger K_k = \mathbb{I}
$$

The Kraus operators $\{K_k\}$ are not unique; physical interpretation depends on the specific choice.

### Depolarizing Channel

The qubit depolarizing channel with parameter $r$:

$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbb{I}}{2}
$$

**Kraus operators**:
$$
K_0 = \sqrt{\frac{1+3r}{4}}\mathbb{I}, \quad K_1 = \sqrt{\frac{1-r}{4}}X, \quad K_2 = \sqrt{\frac{1-r}{4}}Y, \quad K_3 = \sqrt{\frac{1-r}{4}}Z
$$

**Choi matrix**:
$$
J_{\mathcal{N}_r} = \frac{1}{4}\begin{pmatrix} 1+r & 0 & 0 & 2r \\ 0 & 1-r & 0 & 0 \\ 0 & 0 & 1-r & 0 \\ 2r & 0 & 0 & 1+r \end{pmatrix}
$$

**Properties**:
- Covariant under Pauli group: $\mathcal{N}_r(U\rho U^\dagger) = U\mathcal{N}_r(\rho)U^\dagger$ for $U \in \{I, X, Y, Z\}$
- Classical capacity: $C_{\mathcal{N}_r} = 1 - h((1-r)/2)$
- Quantum capacity: $Q_{\mathcal{N}_r} = \max\{0, 1 - 2h((1-r)/2)\}$

### Dephasing Channel

Pure dephasing (phase damping) with parameter $\gamma$:

$$
\mathcal{D}_\gamma(\rho) = (1-\gamma)\rho + \gamma Z\rho Z
$$

**Kraus operators**:
$$
K_0 = \sqrt{1-\gamma}\mathbb{I}, \quad K_1 = \sqrt{\gamma}Z
$$

**Effect in computational basis**:
$$
\mathcal{D}_\gamma\begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix} = \begin{pmatrix} \rho_{00} & (1-2\gamma)\rho_{01} \\ (1-2\gamma)\rho_{10} & \rho_{11} \end{pmatrix}
$$

**Security implication**: Dephasing preserves Z-basis information while degrading X-basis. An adversary with dephasing-dominated storage should preferentially store in the Z-basis, leading to *basis-dependent* attacks not captured by the symmetric depolarizing model.

### Amplitude Damping

Models energy relaxation ($T_1$ decay):

$$
\mathcal{A}_\gamma(\rho) = K_0 \rho K_0^\dagger + K_1 \rho K_1^\dagger
$$

**Kraus operators**:
$$
K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}
$$

**Physical interpretation**: The excited state $|1\rangle$ decays to ground state $|0\rangle$ with probability $\gamma$.

## Combined T1-T2 Model

### Lindblad Master Equation

For a qubit with relaxation time $T_1$ and dephasing time $T_2$, the Lindblad equation is:

$$
\frac{d\rho}{dt} = \gamma_1 \left(\sigma_- \rho \sigma_+ - \frac{1}{2}\{\sigma_+ \sigma_-, \rho\}\right) + \frac{\gamma_\phi}{2}\left(Z\rho Z - \rho\right)
$$

where $\gamma_1 = 1/T_1$ and $\gamma_\phi = 1/T_2 - 1/(2T_1)$.

### Time-Dependent Solution

After time $t$, the density matrix elements evolve as:

$$
\rho_{00}(t) = \rho_{00}(0) + \rho_{11}(0)(1 - e^{-t/T_1})
$$
$$
\rho_{11}(t) = \rho_{11}(0) e^{-t/T_1}
$$
$$
\rho_{01}(t) = \rho_{01}(0) e^{-t/T_2}
$$

### Effective Depolarizing Approximation

For the symmetric depolarizing model to be valid, we require $T_1 \approx T_2$. The effective parameter:

$$
r(t) \approx e^{-\Gamma t}, \quad \Gamma = \frac{1}{2}\left(\frac{1}{T_1} + \frac{1}{T_2}\right)
$$

**Validity criterion**: The depolarizing approximation holds when:
$$
\left|\frac{T_2}{T_1} - 1\right| < 0.5
$$

For strongly anisotropic memories ($T_2 \ll T_1$), the dephasing channel is more appropriate.

## Simulation Implementation

### Noise Model Hierarchy

The NetSquid simulation framework implements noise through a hierarchy of models:

**Level 1 (Physical)**:
- `T1T2NoiseModel`: Lindblad evolution with $(T_1, T_2)$ parameters
- `DepolarNoiseModel`: Isotropic depolarization per timestep

**Level 2 (Link)**:
- `FibreLossModel`: Exponential attenuation in fiber
- `FixedDelayModel`: Propagation delay

**Level 3 (Protocol)**:
- EPR source fidelity $F$
- Detection efficiency $\eta$
- Dark count probability $P_{\text{dark}}$

### EPR State Preparation

The simulation prepares Werner states:

$$
\rho_F = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\frac{\mathbb{I}_4}{4}
$$

**Verification**: The state $\rho_F$ exhibits QBER:
$$
Q = \text{Tr}[(|01\rangle\langle 01| + |10\rangle\langle 10|)(\rho_F \otimes |Z\rangle\langle Z|)] = \frac{1-F}{2}
$$

### Time-Dependent vs. Time-Independent Noise

**Time-independent mode**: Noise probability $p$ applied per operation, regardless of simulated time elapsed.

**Time-dependent mode**: Noise probability scales with elapsed time:
$$
p(t) = 1 - e^{-\gamma t}
$$

For NSM analysis, time-dependent mode is essential to correctly model storage decoherence during $\Delta t$.

## Validation Against Theory

### QBER-Fidelity Consistency

For Werner state source with fidelity $F$ and no additional errors:

**Theoretical prediction**: $Q_{\text{theory}} = (1-F)/2$

**Simulation verification**: Generate $n$ EPR pairs, measure in matching bases, count disagreements.

$$
\hat{Q} = \frac{\text{disagreements}}{\text{total matching-basis pairs}}
$$

For $n \gg 1$: $|\hat{Q} - Q_{\text{theory}}| < 3\sqrt{Q(1-Q)/n}$ with 99.7% confidence.

### Storage Decoherence Verification

Store a qubit in state $|+\rangle$ for time $t$, then measure in X-basis.

**Theoretical prediction**: Error probability $p_{\text{err}}(t) = (1-r(t))/2$ for depolarizing storage.

**Simulation verification**: Repeat $n$ times, measure error rate.

## Critical Assessment

### Limitations of Implemented Models

1. **Markovian assumption**: The simulation uses memoryless noise. Real quantum memories may exhibit non-Markovian dynamics with memory kernels.

2. **Independent errors**: Noise acts independently on each qubit. Correlated errors (e.g., collective dephasing from global magnetic field fluctuations) are not modeled.

3. **Symmetric depolarization**: The default model assumes isotropic noise. Basis-dependent attacks exploiting anisotropic storage are not fully captured.

### Conservative vs. Optimistic Modeling

For **security analysis** (adversary modeling): Use conservative assumptions that maximize adversary capability.
- Assume higher storage fidelity $r$
- Assume lower storage noise rate
- Assume efficient collective attacks

For **performance estimation** (honest party modeling): Use realistic parameters matched to experimental data.

---

## References

[1] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 2010.

[2] J. Preskill, "Lecture notes on quantum computation," California Institute of Technology, 1998.

[3] T. Coopmans et al., "NetSquid, a NETwork Simulator for QUantum Information using Discrete events," *Commun. Phys.*, vol. 4, 164, 2021.

---

[← Return to Main Index](../index.md) | [Next: Timing Enforcement](./timing_enforcement.md)

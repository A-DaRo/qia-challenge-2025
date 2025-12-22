[← Return to Main Index](../index.md)

# 2.3 The SquidASM Simulation Framework

## 2.3.1 Purpose: Simulation as Theoretical Validation

The verification of finite-size security bounds in the Noisy Storage Model requires **experimental access** to quantum channels with controllable noise parameters. Physical experiments face several challenges:

1. **Parameter Control:** Quantum hardware provides limited control over noise characteristics
2. **Statistical Requirements:** Security proofs require $n > 10^5$ samples for tight bounds
3. **Reproducibility:** Stochastic quantum processes complicate systematic parameter studies

SquidASM [1] addresses these challenges by providing a **discrete-event quantum network simulator** that faithfully represents quantum state evolution, noise processes, and timing semantics. The simulation output constitutes **numerical evidence** for theoretical security claims.

**Validation Philosophy:** We do not claim that simulations replace physical experiments. Rather, simulations verify that:
- Theoretical bounds are achievable under ideal implementations
- Finite-size penalties behave as predicted asymptotically
- Protocol implementations correctly realize cryptographic definitions

## 2.3.2 Physical Foundations

### Quantum State Representation

SquidASM tracks quantum states as **density matrices**:
$$
\rho = \sum_{i,j} \rho_{ij} |i\rangle\langle j| \in \mathcal{B}(\mathcal{H})
$$

This representation is essential for:
- **Mixed States:** Noise processes produce statistical mixtures, not pure states
- **Partial Traces:** Computing reduced density matrices $\rho_A = \operatorname{Tr}_B(\rho_{AB})$
- **CPTP Maps:** Applying arbitrary completely positive trace-preserving operations

**Optimization:** For pure states, SquidASM maintains the state vector $|\psi\rangle$ until decoherence or measurement necessitates density matrix conversion.

### Noise Model Implementation

Physical noise processes are implemented as **CPTP maps** via Kraus operators:
$$
\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger, \quad \sum_k E_k^\dagger E_k = \mathbb{I}
$$

| Physical Process | Kraus Operators | Parameter |
|------------------|-----------------|-----------|
| Depolarizing | $E_0 = \sqrt{1-3p/4}\,\mathbb{I}$, $E_{1,2,3} = \sqrt{p/4}\,\sigma_{x,y,z}$ | $p \in [0,1]$ |
| Dephasing | $E_0 = \sqrt{1-\gamma}\,\mathbb{I}$, $E_1 = \sqrt{\gamma}\,|0\rangle\langle 0|$, $E_2 = \sqrt{\gamma}\,|1\rangle\langle 1|$ | $\gamma \in [0,1]$ |
| Amplitude Damping | $E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-\gamma}\end{pmatrix}$, $E_1 = \begin{pmatrix}0 & \sqrt{\gamma} \\ 0 & 0\end{pmatrix}$ | $\gamma \in [0,1]$ |

**Time Evolution:** For Markovian noise, the decoherence parameter $\gamma(t)$ evolves as:
$$
\gamma(t) = 1 - e^{-t/T_2}
$$
where $T_2$ is the dephasing time constant.

### Entanglement Generation

EPR pairs are produced with controllable **Werner state fidelity**:
$$
\rho_W(F) = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{3}\bigl(|\Phi^-\rangle\langle\Phi^-| + |\Psi^+\rangle\langle\Psi^+| + |\Psi^-\rangle\langle\Psi^-|\bigr)
$$

where $|\Phi^\pm\rangle = (|00\rangle \pm |11\rangle)/\sqrt{2}$ and $|\Psi^\pm\rangle = (|01\rangle \pm |10\rangle)/\sqrt{2}$.

**QBER from Fidelity:** For Werner states measured in the computational basis:
$$
Q = \frac{1-F}{2}
$$

This relation enables direct mapping between simulation fidelity parameters and protocol QBER.

## 2.3.3 Discrete-Event Semantics

### Timing Model

All quantum operations are **events** with explicit timestamps. The simulator maintains a priority queue of events, advancing simulation time discontinuously:

**Event Types:**
- **Quantum Gate:** Duration $\tau_{\text{gate}}$, typically $10^{-6}$ s
- **EPR Generation:** Cycle time $t_{\text{cycle}}$, typically $10^{-3}$ s
- **Measurement:** Duration $\tau_{\text{meas}}$, typically $10^{-6}$ s
- **Classical Communication:** Delay $\tau_{\text{cc}}$, configurable

**NSM Timing Enforcement:** The waiting time $\Delta t$ between quantum transmission and classical revelation is enforced via event scheduling:
$$
t_{\text{reveal}} = t_{\text{quantum}} + \Delta t
$$

Any quantum state stored by an adversary experiences noise $\mathcal{F}_{\Delta t}$ before classical information becomes available.

### Parallelism Semantics

Multiple parties execute **concurrently** in simulation time. This enables faithful representation of the adversary's strategy:
- Measurement immediately after reception ($t = t_{\text{recv}}$)
- Storage followed by deferred measurement ($t = t_{\text{recv}} + \Delta t$)
- Partial measurement with storage of unmeasured qubits

## 2.3.4 Protocol-Relevant Abstractions

### BB84 State Preparation

The four BB84 states are represented as:

| Bit $x$ | Basis $\theta$ | State | Density Matrix |
|---------|----------------|-------|----------------|
| 0 | + (Z) | $|0\rangle$ | $|0\rangle\langle 0|$ |
| 1 | + (Z) | $|1\rangle$ | $|1\rangle\langle 1|$ |
| 0 | × (X) | $|+\rangle$ | $|+\rangle\langle +|$ |
| 1 | × (X) | $|-\rangle$ | $|-\rangle\langle -|$ |

State preparation implements:
$$
|x\rangle_\theta = H^\theta X^x |0\rangle
$$
where $H$ is the Hadamard gate.

### Measurement Statistics

For a qubit $\rho$ measured in basis $\theta$:
$$
P(x | \rho, \theta) = \langle x |_\theta \rho |x\rangle_\theta = \operatorname{Tr}(M_x^\theta \rho)
$$

with POVM elements $M_x^\theta = |x\rangle_\theta \langle x|_\theta$.

**Error Probability:** When Alice prepares $|x\rangle_{\theta_A}$ and Bob measures in $\theta_B \neq \theta_A$:
$$
P(\text{error}) = \frac{1}{2}
$$

This fundamental 50% error rate for mismatched bases underlies sifting efficiency.

## 2.3.5 Validation Methodology

### Simulation as Numerical Experiment

Each simulation run constitutes a **Monte Carlo sample** from the protocol's probability distribution. Statistical properties are estimated via:
$$
\hat{Q} = \frac{1}{N}\sum_{i=1}^N Q_i, \quad \hat{\sigma}^2 = \frac{1}{N-1}\sum_{i=1}^N (Q_i - \hat{Q})^2
$$

**Confidence Intervals:** For $N$ independent runs, the standard error is:
$$
\text{SE}(\hat{Q}) = \frac{\hat{\sigma}}{\sqrt{N}}
$$

### Theoretical Predictions vs. Simulation

The validation criterion is **consistency** between:

1. **Theoretical Bound:** $H_{\min}^\varepsilon(X|E) \geq n \cdot h_{\min}(r) - O(\sqrt{n})$
2. **Simulated Rate:** $\hat{\ell} = H_{\min}^{\text{sim}}(X|E) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon)$

Agreement within statistical uncertainty validates both the theory and implementation.

### Parameter Sweeps

Systematic validation requires sweeping:
- **Block Length:** $n \in \{10^2, 10^3, 10^4, 10^5\}$ to verify finite-size scaling
- **QBER:** $Q \in [0.01, 0.15]$ to map the security threshold
- **Noise Parameter:** $r \in [0.2, 0.8]$ to verify min-entropy bounds

## 2.3.6 Limitations and Caveats

### Idealized Assumptions

SquidASM simulations assume:
1. **Perfect Classical Channels:** No errors in classical communication
2. **Synchronized Clocks:** Global simulation time without relativistic effects
3. **Markovian Noise:** Semigroup property $\mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$
4. **Known Noise Model:** Adversary's storage noise is a fixed, known CPTP map

### What Simulation Cannot Verify

Simulations cannot verify:
- **Device Independence:** Security against adversary-controlled devices
- **Side Channels:** Timing attacks, power analysis, etc.
- **Non-Markovian Effects:** Memory effects in realistic quantum storage

### Interpretation Guidelines

Simulation results should be interpreted as:
> *"Under the assumptions of the NSM framework, with depolarizing storage noise and idealized implementations, the protocol achieves $\varepsilon$-security with the claimed key rate."*

This is a theoretical validation, not an experimental demonstration.

---

## References

[1] SquidASM: Simulator for Quantum Information Distribution, https://github.com/QuTech-Delft/squidasm

[2] T. Coopmans et al., "NetSquid, a NETwork Simulator for QUantum Information using Discrete events," *Commun. Phys.* **4**, 164 (2021).

---

[← Return to Main Index](../index.md) | [Next: Protocol Literature →](./protocol_literature.md)

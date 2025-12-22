[← Return to Main Index](../index.md)

# 4.1 EPR Pair Generation

## 4.1.1 Bell States and Entanglement

### The Maximally Entangled State

An EPR (Einstein-Podolsky-Rosen) pair is a two-qubit system in a maximally entangled Bell state:
$$
|\Phi^+\rangle_{AB} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
$$

**Mathematical Properties:**

1. **Schmidt Decomposition:** The state has Schmidt rank 2 with equal coefficients:
   $$
   |\Phi^+\rangle = \frac{1}{\sqrt{2}}|0\rangle_A|0\rangle_B + \frac{1}{\sqrt{2}}|1\rangle_A|1\rangle_B
   $$

2. **Entanglement Entropy:** The reduced density matrix of either subsystem is:
   $$
   \rho_A = \operatorname{Tr}_B(|\Phi^+\rangle\langle\Phi^+|) = \frac{1}{2}\mathbb{I}_2
   $$
   yielding von Neumann entropy $S(\rho_A) = 1$ bit (maximal for a qubit).

3. **Non-local Correlations:** Measurements on $|\Phi^+\rangle$ exhibit correlations that violate Bell inequalities.

### The Complete Bell Basis

The four Bell states form an orthonormal basis for $\mathbb{C}^2 \otimes \mathbb{C}^2$:
$$
\begin{aligned}
|\Phi^\pm\rangle &= \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle) \\
|\Psi^\pm\rangle &= \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)
\end{aligned}
$$

**Relation via Local Operations:**
$$
|\Phi^-\rangle = (Z \otimes \mathbb{I})|\Phi^+\rangle, \quad
|\Psi^+\rangle = (X \otimes \mathbb{I})|\Phi^+\rangle, \quad
|\Psi^-\rangle = (XZ \otimes \mathbb{I})|\Phi^+\rangle
$$

## 4.1.2 Measurement Correlations

### Computational Basis Correlations

For $|\Phi^+\rangle$ measured in the computational (Z) basis:
$$
P(a, b | |\Phi^+\rangle, Z \otimes Z) = \frac{1}{2}\delta_{ab}
$$

**Outcomes:** Perfect correlation—Alice and Bob obtain identical bits with equal probability:
$$
P(00) = P(11) = \frac{1}{2}, \quad P(01) = P(10) = 0
$$

### Hadamard Basis Correlations

For $|\Phi^+\rangle$ measured in the X basis (Hadamard basis):
$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|++\rangle + |--\rangle)
$$

**Outcomes:** Perfect correlation in the X basis as well:
$$
P(++) = P(--) = \frac{1}{2}, \quad P(+-) = P(-+) = 0
$$

### Mismatched Basis Measurement

If Alice measures in Z and Bob in X (or vice versa):
$$
P(a, b | |\Phi^+\rangle, Z \otimes X) = \frac{1}{4} \quad \forall a, b
$$

**Consequence:** Results are **completely uncorrelated**—each outcome equally likely.

## 4.1.3 Noisy Entanglement: Werner States

### Definition

In realistic implementations, generated states are **Werner states** (isotropic noise):
$$
\rho_W(F) = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{3}\sum_{\psi \neq \Phi^+} |\psi\rangle\langle\psi|
$$

where $F \in [0,1]$ is the **fidelity** to the ideal Bell state.

**Equivalent Form:**
$$
\rho_W(F) = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{4}\mathbb{I}_4
$$

### QBER from Fidelity

For Werner state $\rho_W(F)$ measured in matching bases:
$$
Q = \frac{1-F}{2}
$$

**Derivation:** The probability of anti-correlated outcomes is:
$$
P(\text{error}) = \frac{1-F}{4} \cdot 2 = \frac{1-F}{2}
$$

**Examples:**
| Fidelity $F$ | QBER $Q$ |
|--------------|----------|
| 1.0 | 0% |
| 0.95 | 2.5% |
| 0.90 | 5% |
| 0.85 | 7.5% |
| 0.78 | 11% |

### Entanglement Threshold

Werner states are entangled if and only if $F > 1/2$:
$$
\rho_W(F) \text{ entangled} \iff F > \frac{1}{2}
$$

For $F \leq 1/2$, the state admits a local hidden variable model.

## 4.1.4 Noise Sources

### Source Imperfections

| Source | Effect | Typical Magnitude |
|--------|--------|-------------------|
| Spontaneous emission asymmetry | Mixed Bell states | $\Delta F \sim 0.01$ |
| Multi-photon emission | Reduced purity | Rate-dependent |
| Timing jitter | Mode mismatch | $\Delta F \sim 0.005$ |

### Channel Effects

| Channel | Effect | Model |
|---------|--------|-------|
| Fiber loss | Reduced rate | Exponential: $\eta = e^{-\alpha L}$ |
| Depolarization | Reduced fidelity | Werner mixing |
| Phase drift | Basis rotation | Unitary noise |

### Detector Imperfections

| Effect | Impact | Mitigation |
|--------|--------|------------|
| Dark counts | False positives | $P_{\text{dark}} \sim 10^{-6}$ |
| Detection efficiency | Reduced rate | $\eta_{\text{det}} \sim 0.9$ |
| Timing resolution | Coincidence errors | $\Delta t \sim 100$ ps |

## 4.1.5 Simulation Model

### SquidASM Entanglement Generation

The simulator models EPR generation with configurable fidelity:

**Input Parameters:**
- Target fidelity $F$
- Generation rate $r_{\text{gen}}$
- Success probability $p_{\text{success}}$

**Output:** Werner state $\rho_W(F)$ distributed to Alice and Bob.

### Timing Model

Each EPR generation attempt requires cycle time $t_{\text{cycle}}$:
$$
t_{\text{gen}}(n) = n \cdot t_{\text{cycle}}
$$

For $n = 1000$ pairs at $t_{\text{cycle}} = 1$ μs: $t_{\text{gen}} = 1$ ms.

### Statistical Validation

The simulation validates generated states by:
1. Computing measurement correlations over many samples
2. Verifying QBER matches expected $Q = (1-F)/2$
3. Checking Bell inequality violations (for $F > 1/\sqrt{2}$)

## 4.1.6 Protocol Integration

### BB84 Encoding via EPR

Instead of Alice preparing and sending BB84 states, the EPR-based protocol achieves equivalent statistics:

1. EPR source distributes $|\Phi^+\rangle$ to Alice and Bob
2. Each party measures in a randomly chosen basis ($Z$ or $X$)
3. When bases match: perfect correlation → shared random bit
4. When bases differ: uncorrelated → discard during sifting

**Equivalence to BB84:** From Bob's perspective, Alice's measurement "prepares" his qubit in a random BB84 state.

### Raw Key Generation

After $n$ rounds of EPR generation and measurement:

**Alice's Data:** $(X_A, \Theta_A)$ where $X_A \in \{0,1\}^n$, $\Theta_A \in \{+,\times\}^n$

**Bob's Data:** $(X_B, \Theta_B)$ where $X_B \in \{0,1\}^n$, $\Theta_B \in \{+,\times\}^n$

**Correlation:** For positions where $\Theta_A^{(i)} = \Theta_B^{(i)}$:
$$
\Pr[X_A^{(i)} = X_B^{(i)}] = F + \frac{1-F}{2} = \frac{1+F}{2}
$$

giving error rate $Q = (1-F)/2$.

---

[← Return to Main Index](../index.md) | [Next: Generation Modes →](./generation_modes.md)

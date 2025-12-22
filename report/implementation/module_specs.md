[← Return to Main Index](../index.md)

# Appendix B: Protocol Module Specifications

## Purpose

This appendix provides formal specifications for each protocol module in terms of input/output types and mathematical contracts. These specifications define the interface semantics that any correct implementation must satisfy.

---

## Module Specifications

### 1. EPR Generation Module

**Purpose**: Generate entangled pairs distributed between Alice and Bob.

**Input Contract**:
- $n \in \mathbb{Z}^+$: Number of EPR pairs requested
- $F \in [0.5, 1]$: Target state fidelity

**Output Contract**:
- $\{(q_A^i, q_B^i)\}_{i=1}^{n'}$: Distributed qubit pairs, $n' \leq n$
- State guarantee: $\rho_{AB} = F|\Phi^+\rangle\langle\Phi^+| + (1-F)\mathbb{I}_4/4$

**Postconditions**:
- Each pair satisfies $\text{Tr}_B[\rho_{AB}] = \mathbb{I}/2$ (maximally mixed marginals)
- Generation success probability $P_{\text{gen}} \geq 0.9$ per attempt

---

### 2. BB84 Measurement Module

**Purpose**: Perform projective measurements in randomly chosen bases.

**Input Contract**:
- $\{q^i\}$: Qubit register
- $p_Z \in [0,1]$: Probability of Z-basis selection (default $p_Z = 0.5$)

**Output Contract**:
- $\{(b^i, x^i)\}$: Measurement records
  - $b^i \in \{0, 1\}$: Basis (0 = Z, 1 = X)
  - $x^i \in \{0, 1\}$: Outcome

**Mathematical Specification**:

$$
P(x^i = 1 | \rho, b^i = 0) = \langle 1|\rho|1\rangle
$$
$$
P(x^i = 1 | \rho, b^i = 1) = \langle -|\rho|-\rangle
$$

---

### 3. Sifting Module

**Purpose**: Extract matching-basis measurement pairs.

**Input Contract**:
- $(b_A, x_A)$: Alice's measurement records, length $n$
- $(b_B, x_B)$: Bob's measurement records, length $n$

**Output Contract**:
- $(X_A, X_B)$: Sifted key strings
- $I_{\text{match}} \subseteq \{1, \ldots, n\}$: Matching indices

**Specification**:
$$
I_{\text{match}} = \{i : b_A^i = b_B^i\}
$$
$$
X_A = (x_A^i)_{i \in I_{\text{match}}}, \quad X_B = (x_B^i)_{i \in I_{\text{match}}}
$$

**Postcondition**: $|I_{\text{match}}| \approx n/2$ for uniform random bases.

---

### 4. QBER Estimation Module

**Purpose**: Estimate quantum bit error rate from sacrificed bits.

**Input Contract**:
- $(X_A, X_B)$: Sifted key strings, length $m$
- $n_{\text{sample}} \leq m$: Number of bits to sacrifice

**Output Contract**:
- $\hat{Q} \in [0, 0.5]$: QBER estimate
- $(\delta_{\text{lo}}, \delta_{\text{hi}})$: Confidence interval

**Specification**:
Sample indices $S \subseteq \{1, \ldots, m\}$ uniformly with $|S| = n_{\text{sample}}$.

$$
\hat{Q} = \frac{1}{n_{\text{sample}}} \sum_{i \in S} \mathbb{1}[X_A^i \neq X_B^i]
$$

**Confidence bound** (Hoeffding):
$$
P\left(|Q - \hat{Q}| > \delta\right) \leq 2\exp(-2n_{\text{sample}}\delta^2)
$$

---

### 5. LDPC Syndrome Computation Module

**Purpose**: Compute parity-check syndrome for error correction.

**Input Contract**:
- $X \in \{0,1\}^n$: Codeword (Bob's sifted key)
- $H \in \{0,1\}^{m \times n}$: Parity-check matrix

**Output Contract**:
- $s = HX^T \in \{0,1\}^m$: Syndrome vector

**Specification**:
$$
s_j = \bigoplus_{i : H_{ji} = 1} X_i
$$

---

### 6. Belief Propagation Decoder Module

**Purpose**: Recover Alice's key from Bob's key and syndrome.

**Input Contract**:
- $Y \in \{0,1\}^n$: Alice's noisy observation
- $s \in \{0,1\}^m$: Target syndrome
- $Q \in (0, 0.5)$: Channel error probability
- $T_{\max} \in \mathbb{Z}^+$: Maximum iterations

**Output Contract**:
- $\hat{X} \in \{0,1\}^n$: Decoded codeword
- `success: bool`: Whether $H\hat{X}^T = s$

**Specification**: Sum-product message passing on factor graph $G(H)$.

**Initialization**:
$$
\lambda_i = \log\frac{1-Q}{Q} \cdot (1 - 2Y_i)
$$

**Update equations**: See [baseline_strategy.md](../reconciliation/baseline_strategy.md).

**Termination**: Return $\hat{X}$ when $H\hat{X}^T = s$ or $t = T_{\max}$.

---

### 7. Privacy Amplification Module

**Purpose**: Extract uniform random key from partially secret input.

**Input Contract**:
- $X \in \{0,1\}^n$: Reconciled key
- $k \in \mathbb{R}^+$: Min-entropy estimate
- $\varepsilon > 0$: Security parameter

**Output Contract**:
- $K \in \{0,1\}^\ell$: Final key
- Guarantee: $\|P_{KE} - U_\ell \otimes P_E\|_1 \leq \varepsilon$

**Specification**:

$$
\ell = \lfloor k - 2\log_2(1/\varepsilon) \rfloor
$$

Key extraction via Toeplitz hashing:
$$
K = T \cdot X
$$

where $T \in \{0,1\}^{\ell \times n}$ is a Toeplitz matrix specified by $(n + \ell - 1)$ random seed bits.

---

### 8. Protocol Orchestrator Module

**Purpose**: Coordinate module execution with security checks.

**Input Contract**:
- Configuration parameters: $(n, F, R, Q_{\max}, \varepsilon)$
- Physical model parameters: $(T_1, T_2, \Delta t)$

**Output Contract**:
- `KeyOutput`: Final key $K$ and metadata, or
- `Abort`: Reason for protocol termination

**State Machine**:

| State | Transition Condition | Next State |
|-------|---------------------|------------|
| INIT | Configuration valid | QUANTUM |
| QUANTUM | EPR generation complete | SIFT |
| SIFT | Basis comparison done | ESTIMATE |
| ESTIMATE | $\hat{Q} \leq Q_{\max}$ | RECONCILE |
| ESTIMATE | $\hat{Q} > Q_{\max}$ | ABORT |
| RECONCILE | BP success | AMPLIFY |
| RECONCILE | BP failure | ABORT |
| AMPLIFY | $\ell > 0$ | OUTPUT |
| AMPLIFY | $\ell \leq 0$ | ABORT |

---

## Composition Guarantees

When modules are composed according to the protocol orchestrator:

**Theorem (Composable Security)**. If:
1. EPR source satisfies fidelity bound $F \geq F_{\min}$
2. Storage channel $\mathcal{F}$ satisfies $C_\mathcal{F} \cdot \nu < 1/2$
3. QBER $Q \leq Q_{\max}$
4. Reconciliation achieves $\epsilon_{\text{rec}} < \epsilon_{\text{target}}$

Then the final key $K$ satisfies:
$$
\|P_{KE} - U_\ell \otimes P_E\|_1 \leq \varepsilon_{\text{sec}} + \epsilon_{\text{rec}} + \delta_{\text{PE}}
$$

where $\delta_{\text{PE}}$ is the parameter estimation confidence.

---

[← Return to Main Index](../index.md) | [Next: Numerical Optimization](./numerical_optimization.md)

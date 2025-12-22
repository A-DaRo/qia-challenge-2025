[← Return to Main Index](../index.md)

# 3.1 Protocol Specification

## 3.1.1 Oblivious Transfer: Formal Statement

**Definition ($\binom{2}{1}$-Randomized Oblivious Transfer):**

A two-party protocol between Alice (sender) and Bob (receiver) producing:

| Party | Input | Output |
|-------|-------|--------|
| Alice | — | $(S_0, S_1) \in \{0,1\}^{\ell} \times \{0,1\}^{\ell}$ |
| Bob | $C \in \{0,1\}$ | $S_C \in \{0,1\}^{\ell}$ |

**Security Properties:**

An $(\varepsilon_c, \varepsilon_s, \varepsilon_r)$-secure ROT satisfies:

1. **Correctness:** Honest parties succeed with probability $\geq 1 - \varepsilon_c$
2. **Sender Security:** For any cheating Bob, there exists $C' \in \{0,1\}$ such that:
   $$
   d(S_{1-C'} | S_{C'}, \rho_B) \leq \varepsilon_s
   $$
3. **Receiver Security:** For any cheating Alice:
   $$
   I(C : \rho_A) \leq \varepsilon_r
   $$

## 3.1.2 Protocol Structure

The Caligo protocol consists of four sequential phases:

$$
\boxed{\text{Quantum} \to \text{Sifting} \to \text{Reconciliation} \to \text{Amplification}}
$$

Each phase transforms the protocol state, progressively refining raw quantum correlations into secure oblivious keys.

### Phase I: Quantum Distribution

**Objective:** Establish correlated classical strings via BB84-encoded EPR measurements.

**Protocol Steps:**

1. **EPR Generation:** Source creates $n_{\text{raw}}$ Bell states:
   $$
   |\Phi^+\rangle_{AB} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
   $$

2. **Basis Selection:** Each party independently samples:
   $$
   \theta_i^{(A)}, \theta_i^{(B)} \in_R \{+, \times\}
   $$

3. **Measurement:** Party $P \in \{A, B\}$ measures qubit $i$ in basis $\theta_i^{(P)}$:
   $$
   x_i^{(P)} = \text{outcome of } M_{\theta_i^{(P)}}
   $$

4. **Timing Enforcement:** Protocol enforces waiting time $\Delta t$ before any basis information is communicated.

**Output:** Raw strings $(X^{(A)}, \Theta^{(A)})$ and $(X^{(B)}, \Theta^{(B)})$ of length $n_{\text{raw}}$.

### Phase II: Sifting and Parameter Estimation

**Objective:** Establish common basis positions and estimate channel quality.

**Protocol Steps:**

1. **Basis Commitment:** Alice commits to her basis choices:
   $$
   C_A = H(\Theta^{(A)} \| \text{salt})
   $$

2. **Basis Exchange:** 
   - Alice sends $C_A$ to Bob
   - Bob reveals $\Theta^{(B)}$
   - Alice reveals $(\Theta^{(A)}, \text{salt})$; Bob verifies commitment

3. **Sifting:** Define index sets:
   $$
   \mathcal{I}_+ = \{i : \theta_i^{(A)} = \theta_i^{(B)} = +\}, \quad \mathcal{I}_\times = \{i : \theta_i^{(A)} = \theta_i^{(B)} = \times\}
   $$
   
   Keep only positions where bases match. Expected sifted length:
   $$
   n_{\text{sifted}} \approx \frac{n_{\text{raw}}}{2}
   $$

4. **QBER Estimation:** Sample $m$ positions for error comparison:
   $$
   \hat{Q} = \frac{1}{m}\sum_{i \in \mathcal{I}_{\text{test}}} \mathbf{1}[x_i^{(A)} \neq x_i^{(B)}]
   $$

5. **Security Check:** 
   - If $\hat{Q} > 0.22$: **Abort** (impossibility bound)
   - If $\hat{Q} > 0.11$: **Warning** (finite-size concerns)

**Output:** Sifted strings $(K_A, K_B)$ of length $n = n_{\text{sifted}} - m$, QBER estimate $\hat{Q}$.

### Phase III: Information Reconciliation

**Objective:** Correct discrepancies between $K_A$ and $K_B$.

**Problem Statement (Slepian-Wolf):** Bob knows $K_B = K_A \oplus E$ where $E$ is the error vector with $\text{wt}(E)/n \approx Q$. Alice sends syndrome information $\Sigma = H \cdot K_A$ to enable Bob's recovery of $K_A$.

**Protocol Steps:**

1. **Code Selection:** Choose LDPC code with rate $R$ satisfying:
   $$
   1 - R \geq h(Q) + \delta
   $$
   where $\delta$ is the efficiency gap.

2. **Syndrome Transmission:** Alice computes and sends:
   $$
   \Sigma = H \cdot K_A \in \mathbb{F}_2^{(1-R)n}
   $$

3. **Decoding:** Bob solves for $K_A$ given $(K_B, \Sigma)$ via belief propagation:
   $$
   \hat{K}_A = \arg\min_{K : H \cdot K = \Sigma} d_H(K, K_B)
   $$

4. **Verification:** Exchange hash values:
   $$
   h_A = \text{Hash}(K_A), \quad h_B = \text{Hash}(\hat{K}_A)
   $$
   Verify $h_A = h_B$.

**Output:** Reconciled string $K_{\text{rec}}$ of length $n$, total leakage $|\Sigma| + |\text{hash}|$.

### Phase IV: Privacy Amplification

**Objective:** Extract $\varepsilon$-secure keys from the reconciled string.

**Key Length Calculation:** The extractable length is bounded by:
$$
\ell \leq H_{\min}^\varepsilon(K_{\text{rec}} | \mathcal{F}(Q), \Theta, \Sigma) - 2\log_2(1/\varepsilon_{\text{sec}}) + 2
$$

Using the min-entropy bound and accounting for leakage:
$$
\ell = \lfloor n \cdot h_{\min}(r) - |\Sigma| - 2\log_2(1/\varepsilon_{\text{sec}}) + 2 \rfloor
$$

**Extraction:** Apply two-universal hash function:
$$
S = T \cdot K_{\text{rec}}
$$
where $T \in \{0,1\}^{\ell \times n}$ is a random Toeplitz matrix.

**Key Derivation:**
- Alice: Split $S$ into $(S_0, S_1)$ where $|S_0| = |S_1| = \ell/2$
- Bob: Derive $S_C$ from $S$ and his choice bit $C$

**Output:** Alice obtains $(S_0, S_1)$; Bob obtains $S_C$.

## 3.1.3 Timing Diagram

The protocol enforces strict temporal ordering:

```
Time →
─────────────────────────────────────────────────────────────────────

Alice:   [Prepare qubits]─────[Wait Δt]─────[Reveal bases]───[Post-proc]
              │                    │               │
              ▼                    ▼               ▼
         Generate EPR         Storage noise    Sifting begins
              │                applied to      
              │                stored qubits   
              ▼                    │
Bob:     [Measure qubits]─────[Wait Δt]─────[Compare bases]───[Post-proc]

─────────────────────────────────────────────────────────────────────
         t=0            t=Δt           t=Δt+ε         t >> Δt
```

**Critical Timing Constraint:** All quantum measurements must complete before $t = \Delta t$. Any adversary attempting to store qubits experiences noise channel $\mathcal{F}_{\Delta t}$ before receiving basis information.

## 3.1.4 Security Parameters

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Security parameter | $\varepsilon_{\text{sec}}$ | $10^{-10}$ | Adversary's distinguishing advantage |
| QBER threshold | $Q_{\text{max}}$ | $0.11$ | Maximum tolerable error rate |
| Storage noise | $r$ | $0.3$ | Depolarizing preservation probability |
| Waiting time | $\Delta t$ | $10^6$ ns | Delay before basis revelation |

**Security Inequality:** The protocol is secure if:
$$
Q_{\text{observed}} < Q_{\text{storage}} = \frac{1-r}{2}
$$

For $r = 0.3$: $Q_{\text{storage}} = 0.35$, well above the 11% operational threshold.

## 3.1.5 Data Flow Summary

| Phase | Input | Output | Information Leaked |
|-------|-------|--------|-------------------|
| Quantum | $n_{\text{raw}}$ EPR requests | $(X, \Theta)$ strings | None |
| Sifting | $(X_A, \Theta_A), (X_B, \Theta_B)$ | Sifted $(K_A, K_B)$, $\hat{Q}$ | $\Theta_A, \Theta_B$ (public) |
| Reconciliation | $(K_A, K_B)$, code $H$ | $K_{\text{rec}}$ | $|\Sigma| = n(1-R)$ bits |
| Amplification | $K_{\text{rec}}$, $\ell$ | $(S_0, S_1)$, $S_C$ | $T$ (Toeplitz seed, public) |

**Total Leakage:**
$$
\text{leak}_{\text{total}} = n(1-R) + |\text{hash}| + |\text{bases}|
$$

The bases are necessary for the protocol and do not compromise security under NSM assumptions.

---

[← Return to Main Index](../index.md) | [Next: Security Model →](./security_model.md)

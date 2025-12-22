[← Return to Main Index](../index.md)

# 2.2 Cryptographic Primitives

## 2.2.1 Oblivious Transfer: Formal Definition

**1-out-of-2 Randomized Oblivious Transfer** ($\binom{2}{1}$-ROT) is defined by the following ideal functionality:

**Ideal Functionality $\mathcal{F}_{\text{ROT}}$:**
1. Sample $S_0, S_1 \in_R \{0,1\}^\ell$ uniformly at random
2. Send $(S_0, S_1)$ to Alice
3. Receive choice bit $C \in \{0,1\}$ from Bob
4. Send $S_C$ to Bob

A protocol $\pi$ **$\varepsilon$-securely realizes** $\mathcal{F}_{\text{ROT}}$ if the joint distribution of outputs is $\varepsilon$-close to the ideal, even when one party deviates arbitrarily from the protocol.

### Security Definition (Damgård, Fehr, Salvail, Schaffner [1])

An $(\varepsilon_c, \varepsilon_s, \varepsilon_r)$-secure $\binom{2}{1}$-ROT satisfies:

**(i) Correctness:** For honest parties:
$$
\Pr[\text{Alice outputs } (S_0, S_1) \land \text{Bob outputs } S_C] \geq 1 - \varepsilon_c
$$

**(ii) Sender Security:** For honest Alice and any cheating Bob with quantum side-information $\rho_B$, there exists $C' \in \{0,1\}$ (the "committed choice") such that:
$$
d\bigl(S_{1-C'} \big| S_{C'}, \rho_B\bigr) \leq \varepsilon_s
$$

where the **nonuniformity** quantifies distinguishability from ideal randomness:
$$
d(X | \rho_E) := \frac{1}{2}\left\| \frac{\mathbb{I}_{|X|}}{|X|} \otimes \rho_E - \sum_x p_X(x) |x\rangle\langle x| \otimes \rho_E^x \right\|_1
$$

**(iii) Receiver Security:** For honest Bob with output $Y$ and any cheating Alice with state $\rho_A$, there exist strings $S'_0, S'_1$ such that:
$$
\Pr[Y = S'_C] \geq 1 - \varepsilon_r \quad \text{and} \quad I(C : S'_0, S'_1, \rho_A) = 0
$$

**Interpretation:** Sender security ensures Bob learns at most one string; receiver security ensures Alice learns nothing about which string Bob received.

### From Randomized to Chosen-Input OT

Standard OT (where Alice chooses inputs $M_0, M_1$) reduces to ROT via **one-time pad** [2]:
$$
\text{Alice sends: } (M_0 \oplus S_0, M_1 \oplus S_1)
$$
$$
\text{Bob computes: } M_C = (M_C \oplus S_C) \oplus S_C
$$

## 2.2.2 Weak String Erasure

**Definition (König, Wehner, Wullschleger [3]):** An $(\ell, m, \lambda, \varepsilon)$-Weak String Erasure (WSE) is a two-party protocol producing:

- **Alice's output:** Random string $X \in \{0,1\}^\ell$
- **Bob's output:** Index set $\mathcal{I} \subseteq [\ell]$ with $|\mathcal{I}| \approx m$ and substring $X_\mathcal{I}$

**Security Properties:**

**(i) Sender Security:** If Alice is honest, Bob's information about $X_{\bar{\mathcal{I}}}$ is bounded:
$$
H_{\min}^{\varepsilon}(X_{\bar{\mathcal{I}}} | \mathcal{I}, X_\mathcal{I}, \rho_B) \geq \lambda
$$

**(ii) Receiver Security:** If Bob is honest, Alice gains no information about $\mathcal{I}$ beyond its cardinality (within $\varepsilon$).

**Significance:** WSE is the **minimal quantum primitive** sufficient to construct OT. The NSM protocol implements WSE using BB84 states and storage noise.

### WSE from BB84 States

**Protocol:**
1. Alice prepares $n$ qubits in BB84 states: $|x_i\rangle_{\theta_i}$ where $x_i \in \{0,1\}$, $\theta_i \in \{+, \times\}$
2. Alice sends qubits to Bob
3. Bob measures in random bases $\hat{\theta}_i$
4. **Waiting period $\Delta t$** (enforced by timing protocol)
5. Alice reveals $\theta_i$
6. Bob outputs $\mathcal{I} = \{i : \hat{\theta}_i = \theta_i\}$ and $X_\mathcal{I}$

**Security Intuition:**
- Bob's measurements in wrong bases ($\hat{\theta}_i \neq \theta_i$) give random outcomes
- Stored qubits undergo noise $\mathcal{F}$, limiting deferred measurement advantage
- Bob cannot predict which bases will match, so cannot selectively store

## 2.2.3 The Finite-Size Secure Key Length

### Main Theorem (Lupo et al. [4], Equation 2)

For NSM-based ROT with $n$ transmitted qubits, the extractable key length $\ell$ satisfies:
$$
\boxed{\ell \leq H_{\min}^\varepsilon(X_{\bar{B}} | \mathcal{F}(Q), \Theta, B, \Sigma_{\bar{B}}) - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2}
$$

where:
- $X_{\bar{B}}$: Alice's bit string on positions where bases mismatched
- $\mathcal{F}(Q)$: Adversary's quantum state after storage noise
- $\Theta$: Basis choices (revealed after delay)
- $B$: Adversary's classical information
- $\Sigma_{\bar{B}}$: Syndrome leakage from error correction
- $\varepsilon_{\text{sec}}$: Security parameter

### Leakage Accounting

The min-entropy bound decomposes as:
$$
H_{\min}^\varepsilon(X | E, \Sigma) \geq H_{\min}^\varepsilon(X | E) - |\Sigma|
$$

where the **syndrome length** for an LDPC code at rate $R$ is:
$$
|\Sigma| = n(1 - R)
$$

**Total Leakage:**
$$
\text{leak}_{\text{total}} = \underbrace{n(1-R)}_{\text{syndrome}} + \underbrace{|\text{Hash}|}_{\text{verification}} + \underbrace{|\text{revealed}|}_{\text{blind rec.}}
$$

### Finite-Size Corrections

For finite block length $n$, the smooth min-entropy incurs statistical penalties:

**Parameter Estimation (Hoeffding):** Confidence interval for QBER estimate $\hat{Q}$ at confidence $1-\delta$:
$$
Q \leq \hat{Q} + \sqrt{\frac{\ln(1/\delta)}{2n_{\text{test}}}}
$$

**Privacy Amplification:** Security reduction requires:
$$
\ell \leq H_{\min}^\varepsilon - 2\log_2(1/\varepsilon_{\text{PA}})
$$

**Composable Security:** Overall security parameter:
$$
\varepsilon_{\text{total}} = \varepsilon_{\text{PE}} + \varepsilon_{\text{EC}} + \varepsilon_{\text{PA}}
$$

## 2.2.4 Smooth Min-Entropy

### Definition

For classical-quantum state $\rho_{XE}$, the **conditional min-entropy** is:
$$
H_{\min}(X|E)_\rho = -\log_2 P_{\text{guess}}(X|E)_\rho
$$

where the **guessing probability** is:
$$
P_{\text{guess}}(X|E)_\rho = \max_{\{M_x\}} \sum_x p_X(x) \operatorname{Tr}(M_x \rho_E^x)
$$

The **$\varepsilon$-smooth min-entropy** optimizes over nearby states:
$$
H_{\min}^\varepsilon(X|E)_\rho = \max_{\tilde{\rho}: \|\tilde{\rho} - \rho\|_1 \leq \varepsilon} H_{\min}(X|E)_{\tilde{\rho}}
$$

### NSM Min-Entropy Bounds

For i.i.d. depolarizing storage $\mathcal{F} = \mathcal{N}_r^{\otimes \nu n}$, König et al. [3] establish:
$$
H_{\min}^\varepsilon(X | \mathcal{F}(Q), \Theta) \geq n \cdot h_{\min}(r) - O(\sqrt{n \log(1/\varepsilon)})
$$

where the asymptotic rate $h_{\min}(r)$ is:
$$
h_{\min}(r) = \max\left\{ \Gamma\bigl[1 - \log_2(1 + 3r^2)\bigr], \, 1 - r \right\}
$$

**Regime Analysis:**
| $r$ | Dominant Bound | $h_{\min}(r)$ |
|-----|----------------|---------------|
| $0 \to 0.25$ | Dupuis-König | $\approx 0.8$ |
| $0.25 \to 0.5$ | Lupo | $0.75 \to 0.5$ |
| $0.5 \to 0.7$ | Lupo | $0.5 \to 0.3$ |

## 2.2.5 Privacy Amplification

### Two-Universal Hash Functions

A family $\mathcal{F}$ of functions $f: \{0,1\}^n \to \{0,1\}^\ell$ is **two-universal** [5] if:
$$
\forall x \neq y: \quad \Pr_{f \in_R \mathcal{F}}[f(x) = f(y)] \leq 2^{-\ell}
$$

**Toeplitz Construction:** For $T \in \{0,1\}^{\ell \times n}$ with constant diagonals:
$$
f_T(x) = T \cdot x \pmod{2}
$$

Only $n + \ell - 1$ random bits needed to specify $T$.

### Leftover Hash Lemma

**Theorem (Renner [6]):** Let $\mathcal{F}$ be two-universal, $F \in_R \mathcal{F}$. For cq-state $\rho_{XE}$:
$$
d\bigl(F(X) \big| F, E\bigr)_\rho \leq \frac{1}{2} \cdot 2^{-\frac{1}{2}(H_{\min}(X|E)_\rho - \ell)}
$$

**Corollary:** To achieve $d(F(X)|F,E) \leq \varepsilon_{\text{PA}}$, set:
$$
\ell = H_{\min}^\varepsilon(X|E) - 2\log_2(1/\varepsilon_{\text{PA}})
$$

### Practical Implementation

Caligo implements privacy amplification using:
1. **Matrix Generation:** Toeplitz matrix from shared random seed
2. **Multiplication:** $S = T \cdot X$ over $\mathbb{F}_2$
3. **Output:** $\ell$-bit secure key

**Computational Complexity:** $O(n \cdot \ell)$ via FFT-based Toeplitz multiplication.

---

## References

[1] I. B. Damgård, S. Fehr, L. Salvail, and C. Schaffner, "Cryptography in the bounded quantum-storage model," *SIAM J. Comput.* **37**, 1865 (2008).

[2] C. Crépeau, "Equivalence between two flavours of oblivious transfers," *CRYPTO '87*, LNCS **293**, 350–354 (1988).

[3] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[4] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

[5] J. L. Carter and M. N. Wegman, "Universal classes of hash functions," *J. Comput. Syst. Sci.* **18**, 143 (1979).

[6] R. Renner, "Security of Quantum Key Distribution," Ph.D. thesis, ETH Zurich (2005).

---

[← Return to Main Index](../index.md) | [Next: SquidASM Framework →](./squidasm_framework.md)

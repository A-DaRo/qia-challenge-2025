[← Return to Main Index](../index.md)

# 2.2 Cryptographic Primitives

## 2.2.1 Oblivious Transfer: Definition and Variants

**1-out-of-2 Oblivious Transfer** ($\binom{2}{1}$-OT) is a fundamental cryptographic primitive where:

- **Sender (Alice)** inputs two secret strings: $S_0, S_1 \in \{0,1\}^{\ell}$
- **Receiver (Bob)** inputs a choice bit: $C \in \{0,1\}$

### Security Definition

Following Damgård et al. [1], an $\varepsilon$-secure 1-2 **Randomized** Oblivious Transfer (ROT) satisfies:

**(i) Correctness**: If both parties are honest, Alice outputs $(S_0, S_1)$ and Bob learns $Y = S_C$, except with probability $\varepsilon_{\text{corr}}$.

**(ii) Sender Security**: If Alice is honest, for any cheating strategy of Bob resulting in state $\rho_B$, there exists $C' \in \{0,1\}$ such that:

$$
d(S_{1-C'}|S_{C'}\rho_B) \leq \varepsilon_s
$$

where $d(X|\rho_E)$ is the **nonuniformity** [2]:

$$
d(X|\rho_E) := \frac{1}{2}\left\| \frac{\mathbb{I}}{|X|} \otimes \rho_E - \sum_x P_X(x)|x\rangle\langle x| \otimes \rho_x^E \right\|_1
$$

*Intuition*: Bob gains negligible information about $S_{1-C'}$ even given quantum side information $\rho_B$.

**(iii) Receiver Security**: If Bob is honest and obtains $Y$, for any cheating Alice with state $\rho_A$, there exist $S'_0, S'_1$ such that $\Pr[Y = S'_C] \geq 1 - \varepsilon_r$ and $C$ is independent of $(S'_0, S'_1, \rho_A)$.

### From ROT to Standard OT

ROT outputs **uniformly random** strings $(S_0, S_1)$. To implement standard OT with chosen inputs $(M_0, M_1)$, Alice applies **one-time pad** [3]:

$$
\text{Send}(M_0 \oplus S_0, M_1 \oplus S_1)
$$

Bob computes $M_C = (M_C \oplus S_C) \oplus S_C$ using his received $S_C$.

## 2.2.2 Weak String Erasure (WSE)

**Definition** (König et al. [2]): In $(\ell, m, \varepsilon_s, \varepsilon_r)$-WSE:

- Alice outputs a random $\ell$-bit string $X$
- Bob outputs indices $\mathcal{I} \subseteq [\ell]$ with $|\mathcal{I}| \approx m$ and substring $X_{\mathcal{I}}$

**Security**:
1. **Sender Security**: If Alice is honest, Bob cannot distinguish $X_{\bar{\mathcal{I}}}$ from uniform within trace distance $\varepsilon_s$
2. **Receiver Security**: If Bob is honest, Alice gains no information about $\mathcal{I}$ beyond $|\mathcal{I}|$ (up to $\varepsilon_r$)

**Relation to OT**: WSE is **sufficient** to construct OT [2, Theorem 2]. The construction uses:
- Interactive hashing [4] for error correction
- Privacy amplification [5] via universal hash functions
- Classical post-processing only

## 2.2.3 The Finite-Size Key Rate Equation

For NSM-based OT, the **extractable secure key length** is bounded by [6]:

$$
\boxed{
\ell \leq n \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{EC}} - \log_2\left(\frac{2}{\epsilon^2}\right) \right]
}
$$

where:
- $n$: Block length (post-sifting raw key size)
- $H_{\min}^{\epsilon}(X|E)$: **Smooth min-entropy** of Alice's string given adversary's quantum state $E$
- $\text{leak}_{\text{EC}}$: Total information leakage during error correction
- $\epsilon$: Security parameter (failure probability)

### Leakage Components

$$
\text{leak}_{\text{EC}} = |\Sigma| + |\text{Hash}| + |\text{Revealed}|
$$

where:
- $|\Sigma| = (1-R_{\text{eff}}) \cdot n$: LDPC syndrome length
- $|\text{Hash}|$: Verification hash (typically 32-128 bits)
- $|\text{Revealed}|$: Bits revealed during blind reconciliation iterations

### Smooth Min-Entropy

The $\varepsilon$-smooth min-entropy [7] is defined as:

$$
H_{\min}^{\epsilon}(X|E)_{\rho} := \max_{\tilde{\rho}: \|\tilde{\rho} - \rho\|_1 \leq \epsilon} H_{\min}(X|E)_{\tilde{\rho}}
$$

where:

$$
H_{\min}(X|E)_{\rho} = -\log P_{\text{guess}}(X|E)_{\rho}
$$

is the conditional min-entropy, and $P_{\text{guess}}(X|E)$ is the maximum guessing probability.

**NSM Bound**: For individual-storage attacks with storage channel $\mathcal{F}$ [2]:

$$
H_{\min}^{\epsilon}(X|E) \geq -\log P_{\text{succ}}^{\mathcal{F}}(H_{\min}(X|\Theta) - \log(1/\epsilon))
$$

where $\Theta$ is the basis information and $P_{\text{succ}}^{\mathcal{F}}(R)$ is the strong-converse success probability [8].

## 2.2.4 Privacy Amplification

Privacy amplification [5] uses **universal hash functions** to compress a partially secure string into a shorter, fully secure key.

### Two-Universal Hash Functions

A class $\mathcal{F}$ of functions $f: \{0,1\}^n \to \{0,1\}^{\ell}$ is **two-universal** [9] if:

$$
\forall x \neq y, \quad \Pr_{f \in \mathcal{F}}[f(x) = f(y)] \leq 2^{-\ell}
$$

**Example**: Toeplitz matrices [10]:

$$
f_T(x) = T \cdot x
$$

where $T \in \{0,1\}^{\ell \times n}$ is a Toeplitz matrix (constant diagonals).

### Leftover Hash Lemma

**Theorem** (Renner [7, Theorem 5.5.1]): Let $\mathcal{F}$ be two-universal, $F \in \mathcal{F}$ uniformly chosen. For cq-state $\rho_{XE}$:

$$
d(F(X)|FE)_{\rho} \leq 2^{\ell/2} \cdot \frac{1}{\sqrt{P_{\text{guess}}(X|E)_{\rho}}}
$$

**Application**: If $H_{\min}^{\epsilon}(X|E) \geq \ell + 2\log(1/\varepsilon_{\text{PA}}) + k$, where $k$ is additional classical leakage, then:

$$
d(F(X)|FE) \leq \varepsilon_{\text{PA}}
$$

## 2.2.5 Universal Composability

Caligo targets **standalone security** rather than universal composability (UC) [11]. UC security would require:

1. **Simulation-based definition**: Ideal functionality $\mathcal{F}_{\text{OT}}$
2. **Simulator construction**: Simulating adversary's view in ideal world
3. **Indistinguishability**: Real and ideal executions are computationally/statistically close

**Justification for Standalone**: NSM protocols achieve standalone security with tight finite-size bounds [6]. UC-secure versions exist [12] but require additional machinery (e.g., quantum-authenticated channels, composable privacy amplification).

---

## References

[1] I. Damgård et al., "Cryptography in the Bounded-Quantum-Storage Model," *SIAM J. Comput.* **37**(6), 1865-1890 (2008).

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security From Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**(3), 1962-1984 (2012).

[3] C. E. Shannon, "Communication Theory of Secrecy Systems," *Bell Syst. Tech. J.* **28**(4), 656-715 (1949).

[4] C. H. Bennett, G. Brassard, C. Crépeau, and U. M. Maurer, "Generalized Privacy Amplification," *IEEE Trans. Inf. Theory* **41**(6), 1915-1923 (1995).

[5] C. H. Bennett et al., "Privacy Amplification by Public Discussion," *SIAM J. Comput.* **17**(2), 210-229 (1988).

[6] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight finite-key analysis for quantum cryptography," *Nat. Commun.* **3**, 634 (2012).

[7] R. Renner, "Security of Quantum Key Distribution," PhD Thesis, ETH Zurich (2005).

[8] M. M. Wilde and J. M. Renes, "Quantum Polar Codes for Arbitrary Channels," *IEEE Trans. Inf. Theory* **60**(12), 7884-7901 (2014).

[9] J. L. Carter and M. N. Wegman, "Universal Classes of Hash Functions," *J. Comput. Syst. Sci.* **18**(2), 143-154 (1979).

[10] D. R. Stinson, "Universal Hashing and Authentication Codes," *Designs, Codes and Cryptography* **4**, 369-380 (1994).

[11] R. Canetti, "Universally Composable Security: A New Paradigm for Cryptographic Protocols," *Proc. 42nd IEEE FOCS*, 136-145 (2001).

[12] S. Fehr and C. Schaffner, "Composing Quantum Protocols in a Classical Environment," in *Proc. TCC 2009*, 350-367 (2009).

---

[← Return to Main Index](../index.md) | [← Previous: NSM Model](./nsm_model.md) | [Next: SquidASM Framework →](./squidasm_framework.md)

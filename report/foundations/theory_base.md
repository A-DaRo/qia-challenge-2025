[← Return to Main Index](../index.md)

# 2.2 Theoretical Underpinnings

This chapter establishes the mathematical and theoretical foundations of the E-HOK protocol, divided into baseline and advanced implementations.

---

## 2.2.1 Baseline Protocol: Theoretical Foundations

The baseline E-HOK protocol synthesizes principles from entanglement-based quantum key distribution (QKD) with computational commitments to realize quantum oblivious key distribution [1]. This section formalizes the core theoretical constructs underpinning each protocol phase.

### Quantum State Generation and Measurement

The protocol employs **maximally entangled Bell states** as the source of quantum correlations. Alice prepares bipartite states of the form:

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
$$

where one qubit is retained locally and the other transmitted to Bob [2]. This state exhibits perfect correlations when measured in the **computational basis** $\{|0\rangle, |1\rangle\}$ and perfect anti-correlations in the **Hadamard basis** $\{|+\rangle, |-\rangle\}$, where:

$$
|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)
$$

Alice and Bob each randomly select measurement bases $a_i, \bar{a}_i \in \{0,1\}$ for $i = 1, \ldots, N$, where $0$ denotes the computational basis ($Z$) and $1$ the Hadamard basis ($X$) [1, 3]. The measurement outcome $s_i$ (Alice) and $\bar{s}_i$ (Bob) satisfy:

$$
s_i = \bar{s}_i \quad \text{if } a_i = \bar{a}_i \quad (\text{matching basis})
$$

$$
\Pr(s_i = \bar{s}_i) = \frac{1}{2} \quad \text{if } a_i \neq \bar{a}_i \quad (\text{mismatched basis})
$$

This basis-dependent correlation is the foundation of the protocol's obliviousness property: Alice knows all bits $s_i$, but remains ignorant of which subset Bob can deterministically infer [1].

### Cryptographic Commitment and Security

Before basis revelation, Bob **commits** to his measurement results $(\bar{s}, \bar{a})$ via a cryptographic hash function:

$$
H = \text{SHA-256}(\bar{s} \parallel \bar{a} \parallel r)
$$

where $r$ is a random salt [1, 4]. This commitment is **computationally binding**: under the collision-resistance assumption of SHA-256, Bob cannot alter his claimed measurements after commitment without detection (with overwhelming probability $1 - 2^{-256}$) [4]. The commitment is **unconditionally hiding** (Alice gains zero information about $\bar{s}, \bar{a}$ from $H$ alone due to the one-wayness of the hash function) [1].

**Security Rationale:** The commitment prevents a dishonest Bob from adaptively choosing his "measurement" outcomes after learning Alice's bases, which would allow him to correlate all indices (violating obliviousness) [1]. This construction relies on the **Random Oracle Model** (ROM), where the hash function is modeled as a truly random function [4, 5].

### Sifting and Error Estimation

Upon receiving Alice's basis string $a$, Bob computes:

$$
I_0 = \{i : a_i = \bar{a}_i\}, \quad I_1 = \{i : a_i \neq \bar{a}_i\}
$$

The **sifted key** corresponds to indices $I_0$ (expected size $N/2$) [2, 3]. To detect channel noise or eavesdropping, Alice selects a random test subset $T \subset I_0$ of size $m$ and requests Bob to open commitments for $i \in T$ [1]. The **Quantum Bit Error Rate (QBER)** is estimated as:

$$
e = \frac{1}{|T|} \sum_{i \in T} \mathbb{1}_{s_i \neq \bar{s}_i}
$$

where $\mathbb{1}$ is the indicator function [3]. The protocol aborts if $e > \tau$ for a predefined threshold $\tau$ (typically $0.11$) [1, 6]. The remaining sifted key has length $\ell = |I_0 \setminus T|$.

**Statistical Justification:** For large $m$, the Chernoff bound ensures $e$ approximates the true channel error rate with high confidence [6]. The disclosed set $T$ is discarded to prevent information leakage to an eavesdropper Eve [2].

### Information Reconciliation via LDPC Codes

Errors in the sifted key are corrected using **Low-Density Parity-Check (LDPC) codes** under a **reverse reconciliation** scheme (Bob's key is the reference) [7, 8]. Alice generates a parity-check matrix $H_{\text{LDPC}} \in \{0,1\}^{(n-k) \times n}$ with code rate $R = k/n$ selected based on QBER [7]:

$$
R \geq 1 - f_{\text{crit}} \cdot h_b(e)
$$

where $h_b(e) = -e \log_2 e - (1-e) \log_2(1-e)$ is the binary entropy function, and $f_{\text{crit}} \approx 1.05$-$1.2$ is the reconciliation efficiency parameter [7, 8]. Bob computes and transmits the syndrome:

$$
S_B = H_{\text{LDPC}} \cdot \bar{s}
$$

Alice computes her syndrome $S_A = H_{\text{LDPC}} \cdot s$ and uses **Belief Propagation (BP)** decoding on the syndrome difference $\Delta S = S_A \oplus S_B$ to identify the error vector $e$ such that $H_{\text{LDPC}} \cdot e = \Delta S$ [7, 8]. Corrected key: $s' = s \oplus e$.

**Verification:** Alice and Bob exchange a $\nu$-bit hash of their reconciled keys to confirm agreement (failure probability $\leq 2^{-\nu}$, typically $\nu = 50$) [7].

**Theoretical Efficiency:** LDPC codes with iterative BP decoding approach the Shannon limit for large block lengths ($n \geq 10^4$), achieving reconciliation efficiencies $f \approx 1.05$ for practical QBERs up to $11\%$ [7, 8].

### Privacy Amplification via Universal Hashing

To mitigate information leakage to Eve (from QBER and syndrome disclosure), Alice and Bob apply **privacy amplification** using a **2-universal hash function** [9, 10]. They agree on a random Toeplitz matrix $M \in \{0,1\}^{r \times \ell}$ and compute:

$$
K_{\text{final}} = M \cdot s'
$$

where $r < \ell$ is determined by the **Leftover Hash Lemma** [10]:

$$
r \leq \ell - \text{leak}_{\text{EC}} - \text{leak}_{\text{QBER}} - \log_2(1/\epsilon_{\text{sec}})
$$

Here, $\text{leak}_{\text{EC}} = n(1-R)$ (syndrome bits), $\text{leak}_{\text{QBER}} = \ell \cdot h_b(e)$ (estimated Eve's information), and $\epsilon_{\text{sec}}$ is the target security parameter (typically $10^{-10}$) [6, 10].

**Security Guarantee:** The final key $K_{\text{final}}$ is $\epsilon_{\text{sec}}$-secure, meaning Eve's distinguishing advantage for any key bit is bounded by $\epsilon_{\text{sec}}$ [10].

### The Oblivious Key Structure

The protocol outputs an **oblivious key pair** [1]:

- **Alice:** $(K_{\text{final}}, \mathbf{0})$, where $\mathbf{0}$ indicates full knowledge.
- **Bob:** $(K_{\text{final}}, x)$, where $x \in \{0,1\}^\ell$ is a **knowledge mask** with $x_i = 0$ if $i \in I_0$ (known bit) and $x_i = 1$ if $i \in I_1$ (unknown bit due to basis mismatch).

**Formal Definition [1]:** An oblivious key distribution outputs strings $(k, \emptyset)$ to Alice and $(\bar{k}, x)$ to Bob such that:

1. $k_i = \bar{k}_i$ whenever $x_i = 0$ (correlated bits).
2. $k_i$ is uniformly random and independent of $\bar{k}_i$ whenever $x_i = 1$ (uncorrelated bits).
3. Alice has no information about the distribution of $x$ prior to Bob's choice input in subsequent protocols.

This structure directly enables **oblivious transfer (OT)** primitives for secure multiparty computation [1, 11].

---

## 2.2.2 Advanced Protocols *(Planned)*

Future sections will address:

- **Blind Reconciliation:** Adaptive LDPC protocols without *a priori* QBER estimation [8].
- **Measurement-Device-Independent (MDI) Extensions:** Resistance to detector side-channels [12].
- **Noisy Storage Model (NSM):** Unconditional commitment schemes exploiting quantum memory limitations [13].

---

## References

[1] Lemus, M., Ramos, M.F., Yadav, P., Silva, N.A., Muga, N.J., Souto, A., Paunković, N., Mateus, P., and Pinto, A.N. (2020). [Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation](../literature/Generation%20and%20Distribution%20of%20Quantum%20Oblivious%20Keys%20for%20Secure%20Multiparty%20Computation.md). *arXiv:1909.11701v2*.

[2] Ilic, N. (n.d.). [The Ekert Protocol](../literature/The%20Ekert%20Protocol.md). University of Waterloo.

[3] Lemus, M., Schiansky, P., Goulão, M., Bozzio, M., Elkouss, D., Paunković, N., Mateus, P., and Walther, P. (2025). [Performance of Practical Quantum Oblivious Key Distribution](../literature/Performance%20of%20Practical%20Quantum%20Oblivious%20Key%20Distribution.md). *arXiv:2501.03973*.

[4] Halevi, S., and Micali, S. (1996). Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing. *CRYPTO '96*, 201-215. *(Referenced via [1])*.

[5] Bellare, M., and Rogaway, P. (1993). Random Oracles are Practical: A Paradigm for Designing Efficient Protocols. *ACM CCS '93*, 62-73. *(Referenced via [1])*.

[6] Scarani, V., Bechmann-Pasquinucci, H., Cerf, N.J., Dušek, M., Lütkenhaus, N., and Peev, M. (2009). The Security of Practical Quantum Key Distribution. *Rev. Mod. Phys.* 81, 1301-1350. *(Referenced via [3])*.

[7] Kiktenko, E., Trushechkin, A., Kurochkin, Y., and Fedorov, A. (2016). [Post-processing Procedure for Industrial Quantum Key Distribution Systems](../literature/Post-processing%20procedure%20for%20industrial%20quantum%20key%20distribution%20systems.md). *J. Phys.: Conf. Ser.* 741, 012081.

[8] Martinez-Mateo, J., Elkouss, D., and Martin, V. (2013). [Blind Reconciliation](../literature/Blind%20Reconciliation.md). *Quantum Inf. Comput.* 13, 0000-0000.

[9] Wegman, M.N., and Carter, J.L. (1981). New Hash Functions and Their Use in Authentication and Set Equality. *J. Comput. Syst. Sci.* 22(3), 265-279. *(Referenced via [10])*.

[10] Renner, R. (2005). Security of Quantum Key Distribution. *PhD Thesis, ETH Zurich*. *(Referenced via [3])*.

[11] Kilian, J. (1988). Founding Cryptography on Oblivious Transfer. *STOC '88*, 20-31. *(Referenced via [1])*.

[12] Lo, H.-K., Curty, M., and Qi, B. (2012). [Measurement-Device-Independent Quantum Key Distribution](../literature/High-rate%20measurement-device-independent%20quantum%20cryptography.md). *Phys. Rev. Lett.* 108, 130503.

[13] Wehner, S., Schaffner, C., and Terhal, B.M. (2008). [Cryptography from Noisy Storage](../literature/Cryptography%20from%20Noisy%20Storage.md). *Phys. Rev. Lett.* 100, 220502.

---

[← Return to Main Index](../index.md) | [Next: Baseline Protocol Architecture →](../baseline_protocol/architecture.md)

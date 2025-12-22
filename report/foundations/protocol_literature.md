[← Return to Main Index](../index.md)

# 2.4 Key Literature and Theoretical Background

This section surveys the theoretical foundations underlying the Caligo protocol, tracing the evolution from the initial NSM formulation through finite-size security analysis to practical reconciliation techniques.

## 2.4.1 Foundational Papers

### Wehner, Schaffner, Terhal (2008): Genesis of the NSM

**Reference:** S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008). [[Markdown](../../docs/literature/2008_cryptography_from_noisy_storage.md)]

This paper introduced the Noisy Storage Model and established its fundamental security guarantee.

**Key Results:**

1. **Model Definition:** The adversary's quantum storage is characterized by a depolarizing channel $\mathcal{N}_r$ with preservation probability $r$.

2. **All-or-Nothing Theorem:** The optimal adversary strategy exhibits a phase transition at $r_{\text{crit}} = 1/\sqrt{2}$:
   - For $r < 1/\sqrt{2}$: Immediate measurement in the Breidbart basis is optimal
   - For $r \geq 1/\sqrt{2}$: Storing and deferring measurement is optimal

3. **Uncertainty Bound:** For any strategy $\mathcal{S} = \mathcal{N} \circ P$ (partial measurement followed by noise):
   $$
   \Delta(\mathcal{S})^2 := P_g(X|\mathcal{S}(\sigma_+)) \cdot P_g(X|\mathcal{S}(\sigma_\times)) \leq \left(\frac{1}{2} + \frac{r}{2\sqrt{2}}\right)^2
   $$

4. **Security Proof:** For $n$ transmitted qubits and output length $\ell$:
   $$
   d(S_{1-C'} | S_{C'}, \rho_B) \leq 2^{\ell/2 - 1} \cdot (\Delta_{\max})^{n \cdot \log(4/3)/2}
   $$

**Limitation:** Assumed perfect honest operations—no channel noise or detector errors.

### König, Wehner, Wullschleger (2012): Unconditional Security

**Reference:** R. König, S. Wehner, and J. Wullschleger, "Unconditional Security From Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**(3), 1962-1984 (2012). [[Markdown](../../docs/literature/2012_unconditional_security_from_noisy_quantum_storage.md)]

This paper generalized the NSM to arbitrary noise channels and established the connection to channel capacity.

**Key Results:**

1. **General NSM:** Security for any CPTP storage channel $\mathcal{F}$ satisfying the strong converse property.

2. **Capacity Condition:** Secure OT is achievable if and only if:
   $$
   C_\mathcal{N} \cdot \nu < \frac{1}{2}
   $$
   where $C_\mathcal{N}$ is the classical capacity and $\nu$ is the storage rate.

3. **Min-Entropy Bound:** The smooth min-entropy satisfies:
   $$
   H_{\min}^\varepsilon(X | \mathcal{F}(Q), \Theta) \geq -\log_2 P_{\text{succ}}^\mathcal{F}(H_{\min}(X|\Theta) - \log(1/\varepsilon))
   $$
   where $P_{\text{succ}}^\mathcal{F}(R)$ is the strong-converse success probability at rate $R$.

4. **Weak String Erasure:** Formalized WSE as the minimal quantum primitive for OT construction.

### Schaffner, Terhal, Wehner (2009): Robust Protocols

**Reference:** C. Schaffner, B. Terhal, and S. Wehner, "Robust Cryptography in the Noisy-Quantum-Storage Model," *Quantum Inf. Comput.* **9**(11&12), 963-996 (2009).

Extended the NSM to tolerate noise in honest operations.

**Key Results:**

1. **Strict Inequality Condition:**
   $$
   Q_{\text{channel}} < Q_{\text{storage}}
   $$
   Security requires the honest channel to be strictly less noisy than the adversary's storage.

2. **11% Threshold:** For depolarizing noise, secure OT is achievable for $Q_{\text{channel}} < 0.11$ provided the strict inequality holds.

3. **Entropy Trade-off:** Characterized the relationship between Shannon entropy and min-entropy in the finite-size regime.

## 2.4.2 Finite-Size Security Analysis

### Lupo, Peat, Andersson, Kok (2023): Error-Tolerant OT

**Reference:** C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023). [[Markdown](../../docs/literature/2024_lupo_noisy_OT.md)]

This recent work provides tight finite-size bounds for NSM-based OT.

**Key Results:**

1. **Tight Entropic Bound:** For extractable key length $\ell$:
   $$
   \ell \geq H_{\min}^{\varepsilon_h}(X_{\bar{B}} | \mathcal{F}(Q), \Theta, B, \Sigma_{\bar{B}}) - 2\log_2(1/\varepsilon_h) + 1
   $$

2. **22% Hard Limit:** The absolute maximum QBER for any NSM protocol is approximately 22%, arising from the Shannon bound on error correction.

3. **Syndrome Leakage:** Explicit accounting for reconciliation syndrome leakage:
   $$
   H_{\min}^\varepsilon(X | E, \Sigma) \geq H_{\min}^\varepsilon(X | E) - |\Sigma|
   $$

4. **Trusted vs. Untrusted Noise:** Framework for distinguishing channel noise (trusted) from storage noise (adversarial).

**Implication for Caligo:** Minimizing syndrome length $|\Sigma|$ is critical for achieving positive key rates in the finite-size regime.

### The Death Valley Phenomenon

The finite-size regime exhibits a critical phenomenon we term **Death Valley**:

**Definition:** Death Valley is the range of block lengths $n$ where:
- QBER is below the asymptotic threshold $Q < 0.11$
- Yet $\ell(n, Q, \varepsilon) \leq 0$ due to finite-size penalties

**Mathematical Origin:** The extractable length scales as:
$$
\ell(n) = n \cdot h_{\min}(r) - (1-R) \cdot n - 2\log_2(1/\varepsilon) - O(\sqrt{n})
$$

For small $n$, the $O(\sqrt{n})$ and $\log(1/\varepsilon)$ terms dominate, yielding $\ell < 0$.

**Critical Block Length:** The minimum $n$ for positive key rate is approximately:
$$
n_{\min} \approx \frac{4\log_2^2(1/\varepsilon)}{(h_{\min}(r) - (1-R) - h(Q))^2}
$$

## 2.4.3 Information Reconciliation Theory

### Elkouss et al. (2009): LDPC for QKD

**Reference:** D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *ISIT 2009*, arXiv:0901.2140 (2009).

**Key Contribution:** Replaced interactive Cascade reconciliation with one-way LDPC syndrome transmission.

**Reconciliation Efficiency:** Defined as:
$$
f = \frac{\text{leak}_{\text{EC}}}{n \cdot h(Q)} = \frac{(1-R) \cdot n}{n \cdot h(Q)} = \frac{1-R}{h(Q)}
$$

where $R$ is the LDPC code rate and $h(Q)$ is the binary entropy at error rate $Q$.

**Performance:** For $Q = 0.05$, LDPC achieves $f \approx 1.05$ versus Cascade's $f \approx 1.10$.

### Martinez-Mateo et al. (2012): Blind Reconciliation

**Reference:** J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation" (2012). [[Markdown](../../docs/literature/2012_blind_reconciliation.md)]

**Key Innovation:** Rate-adaptive reconciliation without explicit QBER estimation:

**Algorithm:**
1. Initialize with optimistic rate estimate
2. Iterate: transmit syndrome → attempt decoding → reveal additional bits if needed
3. Converge when decoding succeeds

**Security Implication:** Each revealed bit reduces extractable entropy by 1 bit:
$$
H_{\min}^{\text{post-EC}} = H_{\min}^{\text{pre-EC}} - |\Sigma| - |\text{revealed}|
$$

### Elkouss et al. (2010): Rate-Compatible Codes

**Reference:** D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD" (2010).

**Key Techniques:**

1. **Puncturing:** Remove $p$ parity bits to increase rate:
   $$
   R_{\text{punctured}} = \frac{R_0}{1 - \pi}, \quad \pi = p/n
   $$

2. **Shortening:** Fix $s$ information bits to decrease rate:
   $$
   R_{\text{shortened}} = \frac{R_0 - \sigma}{1 - \sigma - \pi}, \quad \sigma = s/n
   $$

**Application:** Single LDPC code family serves range of QBER values via puncturing/shortening.

## 2.4.4 Theoretical Bounds and Limits

### Shannon Limit for Reconciliation

The Slepian-Wolf theorem establishes the minimum communication for source coding with side information:
$$
R_{\text{min}} = H(X|Y) = h(Q)
$$

**Reconciliation Efficiency:** Achieving $f \to 1$ requires approaching the Shannon limit.

### Security-Efficiency Trade-off

There exists a fundamental tension:

| Objective | Requires | Consequence |
|-----------|----------|-------------|
| High security | Low $\ell$ | Large $\varepsilon$ margin |
| High efficiency | High $R$ | More syndrome leakage |
| Practical operation | Moderate $n$ | Finite-size penalties |

**Optimal Operating Point:** For given $Q$ and target $\varepsilon$, there exists an optimal $(n, R)$ pair maximizing throughput.

### Asymptotic Key Rate

In the limit $n \to \infty$, the key rate approaches:
$$
r_\infty = h_{\min}(r) - h(Q)
$$

For $Q = 0.05$ and $r = 0.3$:
$$
r_\infty = (1 - 0.3) - h(0.05) \approx 0.7 - 0.286 \approx 0.414 \text{ bits/qubit}
$$

### Finite-Size Key Rate

For finite $n$, the practical key rate is:
$$
r(n) = h_{\min}(r) - h(Q) \cdot f - \frac{2\log_2(1/\varepsilon)}{n} - O(n^{-1/2})
$$

The $O(n^{-1/2})$ term from parameter estimation dominates for $n < 10^4$.

---

## Summary Table

| Paper | Year | Key Contribution | Relevance to Caligo |
|-------|------|------------------|---------------------|
| Wehner et al. | 2008 | NSM definition, all-or-nothing theorem | Foundational security model |
| König et al. | 2012 | General NSM, capacity condition | Min-entropy bounds |
| Schaffner et al. | 2009 | Robust protocols, 11% threshold | Practical QBER limits |
| Lupo et al. | 2023 | Tight finite-size bounds, 22% limit | Key length formula |
| Elkouss et al. | 2009 | LDPC reconciliation | Efficient error correction |
| Martinez-Mateo et al. | 2012 | Blind reconciliation | Rate-adaptive protocol |

---

[← Return to Main Index](../index.md) | [Next: Protocol Architecture →](../architecture/protocol_overview.md)

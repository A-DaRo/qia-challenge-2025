[← Return to Main Index](../index.md)

# 2.4 Protocol Literature Review

This section surveys the theoretical foundations underlying Caligo's implementation, organized chronologically to trace the evolution of ideas from initial NSM proposals through reconciliation techniques to the E-HOK protocol.

## 2.4.1 Noisy Storage Model Foundations (2008-2012)

### Wehner, Schaffner, Terhal (2008): The NSM Birth

**Reference**: S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008).

**Key Contributions**:
1. **Model Definition**: Introduced depolarizing channel $\mathcal{N}_r(\rho) = r\rho + (1-r)\mathbb{I}/2$ for quantum storage
2. **All-or-Nothing Theorem**: Proved $r_{\text{crit}} = 1/\sqrt{2}$ threshold for optimal adversary strategy
3. **Protocol 1**: Simple OT using BB84 states with individual-storage attack analysis

**Security Proof Sketch**:
- Adversary Bob stores qubits undergoing noise $\mathcal{N}_r$ for time $\Delta t$
- Uncertainty relation: $P_g(X|\mathcal{N}_r(\sigma_+)) \cdot P_g(X|\mathcal{N}_r(\sigma_\times)) \leq \Delta(S)^2$
- For depolarizing: $\Delta_{\max} = \frac{1}{2} + \frac{r}{2\sqrt{2}}$ (Breidbart basis measurement)
- Leftover hash lemma bounds nonuniformity: $d(S_{\bar{C}}|S_{C'}\rho_B) \leq 2^{\ell/2 - 1}(\Delta_{\max})^{n\log(4/3)/2}$

**Limitation**: Assumed **perfect operations** for honest parties (no channel noise, detector errors).

### König, Wehner, Wullschleger (2012): Unconditional Security

**Reference**: R. König, S. Wehner, and J. Wullschleger, "Unconditional Security From Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**(3), 1962-1984 (2012).

**Advances**:
1. **General NSM**: Arbitrary noise $\mathcal{F}$ (not just depolarizing), unbounded storage
2. **Strong Converse**: Used channel capacity $C_{\mathcal{F}}$ and error exponent $\gamma(R)$:
   $$H_{\min}^{\epsilon}(X|\mathcal{F}(Q)\Theta) \geq -\log P_{\text{succ}}^{\mathcal{F}}(H_{\min}(X|\Theta) - \log(1/\epsilon))$$
3. **Weak String Erasure**: Introduced WSE primitive as OT building block
4. **Min-Entropy Bounds**: Rigorous smooth min-entropy analysis for finite-size security

**Weak String Erasure Protocol**:
- Alice sends $n$ BB84 states; Bob measures in random basis $C$
- After delay $\Delta t$, Alice reveals bases $\Theta$
- Bob learns $X_{I_C}$ (matched bases), erased on $X_{I_{\bar{C}}}$ (mismatched)

**From WSE to OT**:
1. Interactive hashing for error correction
2. Privacy amplification with two-universal hash $F_C, F_{\bar{C}}$
3. Alice outputs $(S_0, S_1) = (F_+(X|_{I_+}), F_\times(X|_{I_\times}))$

### Schaffner, Terhal, Wehner (2009): Robust NSM

**Reference**: C. Schaffner, B. Terhal, and S. Wehner, "Robust Cryptography in the Noisy-Quantum-Storage Model," *Quantum Inf. Comput.* **9**(11&12), 963-996 (2009).

**Breakthrough**: Extended to **noisy honest parties**:

$$
Q_{\text{channel}} < Q_{\text{storage}} \quad (\text{strictly less condition})
$$

**11% Threshold**: For depolarizing storage and channel noise, secure OT achievable if $Q_{\text{channel}} < 0.11$ and $Q_{\text{channel}} < Q_{\text{storage}}$.

**Entropy Trade-off**:
- Conditional Shannon entropy $H(X|\mathcal{F}(Q)\Theta B)$ bounds adversary uncertainty
- Smooth min-entropy related via $H_{\min}^{\epsilon}(X|E) \geq H(X|E) - \Delta$ for small $\Delta$

**Practical Implications**:
- Tolerates realistic detector inefficiencies ($\eta \sim 10\%$ at telecom, $\sim 70\%$ visible)
- Enables identification protocols (reusable passwords)

## 2.4.2 Practical Implementation Studies (2010-2023)

### Wehner et al. (2010): Device Characterization

**Reference**: S. Wehner, M. Curty, C. Schaffner, and H.-K. Lo, "Implementation of two-party protocols in the noisy-storage model," *Phys. Rev. A* **81**, 052336 (2010).

**Focus**: Translating NSM theory to **experimental parameters**:

| Parameter | Definition | Impact on Security |
|-----------|------------|-------------------|
| $P_{\text{src}}^1$ | Single-photon emission probability | Bounds dishonest Bob's multi-photon attacks |
| $P_{\text{B,click}}^{h\|1}$ | Honest Bob detection given 1 photon | Sets achievable raw key rate |
| $P_{\text{B,err}}^h$ | Honest Bob bit-flip error | Determines reconciliation efficiency requirement |
| $P_{\text{dark}}$ | Dark count rate | Contributes to channel QBER |

**Weak Coherent Pulses**: For Poissonian source with mean photon number $\mu$:

$$
P_{\text{src}}^k = \frac{\mu^k e^{-\mu}}{k!}
$$

**Decoy States**: Proposed using multiple intensities $\{\mu_s\}$ to bound multi-photon contributions.

**Loss Handling**: Introduced **erasure channel** model:
- Bob confirms photon receipt (avoiding positional ambiguity)
- Only detected photons contribute to raw key

### Lupo et al. (2023): Error-Tolerant OT

**Reference**: C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," *arXiv:2309.xxxxx* (2023).

**Contributions**:
1. **Tight Entropic Bounds**: Improved uncertainty relations for correlated noise
2. **22% Hard Limit**: Demonstrated $Q_{\text{total}} < 0.22$ as absolute maximum for unbounded noisy storage
3. **Trusted vs. Untrusted Noise**: Explicit trade-off between channel and storage noise

**Security Bound** (Equation 2):

$$
\ell \geq H_{\min}^{\epsilon_h}(X_{\bar{B}}|\mathcal{F}(Q)\Theta B\Sigma_{\bar{B}}) - 2\log(1/\epsilon_h) + 1
$$

**Syndrome Leakage**: Reconciliation syndrome $\Sigma_{\bar{B}}$ reduces extractable length:

$$
H_{\min}^{\epsilon_h}(X_{\bar{B}}|\mathcal{F}(Q)\Theta B\Sigma_{\bar{B}}) \geq H_{\min}^{\epsilon_h}(X_{\bar{B}}|\mathcal{F}(Q)\Theta B) - |\Sigma_{\bar{B}}|
$$

**Implication for Caligo**: Minimizing $|\Sigma_{\bar{B}}|$ is **critical** for secure key rate.

## 2.4.3 Information Reconciliation (2009-2012)

### Elkouss et al. (2009): LDPC for QKD

**Reference**: D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *arXiv:0901.2140* (2009).

**Innovation**: Replaced interactive Cascade with **one-way LDPC syndrome transmission**.

**Reconciliation Efficiency**:

$$
f = \frac{\text{leak}_{\text{EC}}}{n \cdot h(Q)} = \frac{(1-R)n}{n \cdot h(Q)} = \frac{1-R}{h(Q)}
$$

**Performance**: For $Q = 0.05$:
- Cascade: $f \approx 1.10$
- LDPC (optimized): $f \approx 1.05$

**Degree Distribution Optimization**: Used Differential Evolution to find $\lambda(x), \rho(x)$ achieving thresholds near Shannon limit.

### Elkouss et al. (2010): Rate-Compatible Protocol

**Reference**: D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD" (2010).

**Problem**: Single LDPC code only efficient near its design QBER.

**Solution**: **Puncturing and Shortening**:

- **Puncturing**: Remove $p$ parity bits → increase rate $R_0 \to R_0/(1-\pi)$ where $\pi = p/n$
- **Shortening**: Fix $s$ information bits → decrease rate $R_0 \to (R_0 - \sigma)/(1-\sigma-\pi)$ where $\sigma = s/n$

**Effective Rate**:

$$
R_{\text{eff}} = \frac{R_0 - \sigma}{1 - \pi - \sigma}
$$

**Rate Adaptation**: By varying $(\pi, \sigma)$ while keeping $\delta = \pi + \sigma$ fixed, a **single mother code** covers QBER range $[Q_{\min}, Q_{\max}]$.

### Elkouss et al. (2012): Untainted Puncturing

**Reference**: D. Elkouss, J. Martinez-Mateo, and V. Martin, "Untainted Puncturing for Irregular Low-Density Parity-Check Codes," *IEEE Wireless Commun. Lett.* **1**(6), 585-588 (2012).

**Stopping Set Problem**: Random puncturing can create **stopping sets** (variable nodes disconnected from parity checks) → decoding failure.

**Untainted Definition**: A puncturing pattern is **untainted** if every check node remains connected to $\geq 2$ non-punctured variable nodes.

**Algorithm**:
1. Compute check-node degrees: $d_c[j] = $ # non-punctured neighbors of check $j$
2. Select variable $i$ for puncturing only if $\min_{j \in N(i)} d_c[j] > 2$
3. Update $d_c$ and repeat

**Performance**: Untainted puncturing achieves Frame Error Rate (FER) $\sim 10^{-3}$ at 95% of capacity vs. 85% for random puncturing.

**Saturation**: For finite-length codes, untainted candidates exhaust at moderate rates ($R_{\text{eff}} \sim 0.6$).

### Martinez-Mateo et al. (2012): Blind Reconciliation

**Reference**: J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Inf. Comput.* **12**(9&10), 791-812 (2012).

**Motivation**: Rate-compatible methods require **a priori QBER estimation**, which:
1. Consumes raw key for sampling
2. Reveals information to adversary
3. Introduces statistical error

**Blind Protocol**:
1. Alice sends syndrome for **maximum rate** $R_{\max}$ (all $d$ symbols punctured)
2. If Bob's decoding fails, Alice reveals $\Delta$ punctured bits (converts to shortened)
3. Repeat until success or all $d$ symbols shortened

**Iterations**: Typically $t = 3$ iterations sufficient for efficiency $f \in [1.05, 1.15]$.

**Average Efficiency** (Equation 9):

$$
\bar{f} = \frac{1 - \sum_{i=1}^t a_i r_i}{h(Q)}
$$

where $a_i = (F^{(i-1)} - F^{(i)})/(1 - F^{(t)})$ is fraction corrected in iteration $i$, and $F^{(i)}$ is Frame Error Rate.

**Advantage**: No QBER estimation leakage; adapts automatically to channel.

## 2.4.4 Oblivious Key Distribution (2020)

### Lemus et al. (2020): E-HOK Protocol

**Reference**: M. Lemus et al., "Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation," *arXiv:1909.11701v2* (2020).

**Contribution**: **Hybrid quantum-classical** approach to OT:

1. **Quantum Phase**: EPR-based key distribution (similar to BB84)
2. **Classical Commitment**: Hash-based commitments instead of quantum memory assumption
3. **Practical Focus**: Targets 2-party and multi-party secure computation

**Protocol $\pi_{\text{QOT}}$** (Figure 2 in paper):

1. Alice prepares $|(s_i, a_i)\rangle$ states ($n+m$ qubits)
   - $(s_i, a_i) \in \{0,1\}^2$: bit value and basis choice
   - $|(0,0)\rangle = |0\rangle$, $|(0,1)\rangle = |+\rangle$, etc.

2. Bob measures in random basis $\bar{a} \in \{0,1\}^{n+m}$

3. Bob commits measurement bases and outcomes using hash $h$

4. Alice challenges $m$ random positions

5. Bob opens challenged commitments; Alice verifies correlations

6. Alice reveals basis choices $a$ for remaining $n$ positions

7. Bob partitions into matched ($I_0$) and mismatched ($I_1$) bases

8. Bob sends $(I_c, I_{1-c})$ where $c$ is his choice bit

9. Alice encrypts: $\tilde{b}_0 = b_0 \oplus f(s_{I_0})$, $\tilde{b}_1 = b_1 \oplus f(s_{I_1})$

10. Bob decrypts: $b_c = \tilde{b}_c \oplus f(\bar{s}_{I_c})$

**Security Analysis**:
- **Computational**: Based on collision-resistant hash (not information-theoretic)
- **Random Oracle Model**: Hash modeled as truly random function
- **Efficiency**: $O(n)$ qubit transmissions for $O(\log n)$-bit OT

**Relevance to Caligo**: Demonstrates **feasibility** of OT with current QKD hardware. Caligo extends to NSM setting for information-theoretic security.

## 2.4.5 Synthesis for Caligo

### Design Choices Informed by Literature

| Decision | Justification | Reference |
|----------|---------------|-----------|
| **NSM over BSM** | Realistic noise assumption | König et al. (2012) |
| **11% QBER Threshold** | Proven secure for individual attacks | Schaffner et al. (2009) |
| **LDPC Reconciliation** | Non-interactive, efficient | Elkouss et al. (2009) |
| **Hybrid Puncturing** | Untainted + ACE-guided for finite-length | Elkouss et al. (2012) |
| **Blind Strategy** | No QBER estimation leakage | Martinez-Mateo et al. (2012) |
| **EPR Pairs** | Native to SquidASM, matches E-HOK | Lemus et al. (2020) |
| **Toeplitz Hashing** | Two-universal, hardware-efficient | König et al. (2012) |

### Open Research Questions

1. **Coherent Attacks**: Security beyond individual-storage attacks [König et al., Conclusion]
2. **Non-Markovian Noise**: Time-correlated decoherence models [Lupo et al., Section VII]
3. **Finite-Length Optimization**: Tighter bounds for $n < 10^4$ [Martinez-Mateo et al., Section 5]
4. **Composable Security**: UC-secure NSM protocols [Fehr & Schaffner (2009)]
5. **Multi-Party NSM**: Extending OT to $n > 2$ parties under NSM [König & Terhal (2008)]

---

## References

(Complete bibliography available in [Bibliography](../references/bibliography.md))

---

[← Return to Main Index](../index.md) | [← Previous: SquidASM Framework](./squidasm_framework.md) | [Next Chapter: Protocol Architecture →](../architecture/protocol_overview.md)

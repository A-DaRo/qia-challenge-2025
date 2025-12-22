[← Return to Main Index](../index.md)

# 3.2 Security Model

## 3.2.1 Adversary Model

The NSM security proof holds against an adversary with the following capabilities:

**Unlimited Resources:**
- Unbounded classical computation and storage
- Perfect quantum operations (unitaries, measurements)
- Instantaneous classical communication

**Limited Resource:**
- Quantum storage subject to noise channel $\mathcal{F}$

**Adversary Classes:**

| Class | Description | Relevant Bound |
|-------|-------------|----------------|
| Individual Storage | Eve stores and attacks qubits independently | 11% threshold |
| Collective Storage | Eve applies joint operations on stored qubits | 22% hard limit |
| Coherent Attack | Eve entangles stored qubits with ancilla | Open problem |

Caligo's security analysis assumes **individual storage attacks**, corresponding to the conservative 11% threshold.

## 3.2.2 Honest-but-Curious vs. Malicious

**Honest-but-Curious (Semi-Honest):**
- Parties follow the protocol specification
- Adversary (corrupted party) attempts to learn extra information

**Malicious:**
- Corrupted party may deviate arbitrarily from protocol
- Must account for active attacks

**Caligo's Security Level:** The current implementation provides security against **honest-but-curious** adversaries. Malicious security would require additional mechanisms (e.g., commitment schemes, zero-knowledge proofs) not currently implemented.

## 3.2.3 The Security Condition

### Channel vs. Storage Noise

**Theorem (Schaffner, Terhal, Wehner [1]):** An NSM protocol achieves $\varepsilon$-security if:
$$
Q_{\text{channel}} < Q_{\text{storage}}
$$

where:
- $Q_{\text{channel}}$: QBER experienced by honest parties
- $Q_{\text{storage}} = (1-r)/2$: Effective QBER from storage noise

**Interpretation:** The honest channel must be **strictly less noisy** than the adversary's storage. If equality holds, the adversary can simulate the honest channel perfectly.

### Capacity Condition (General NSM)

For arbitrary storage channel $\mathcal{F}$, König et al. [2] establish:
$$
C_\mathcal{F} \cdot \nu < \frac{1}{2}
$$

For depolarizing noise:
$$
C_{\mathcal{N}_r} = 1 - h\left(\frac{1+r}{2}\right)
$$

With storage rate $\nu = 1$, security requires:
$$
1 - h\left(\frac{1+r}{2}\right) < \frac{1}{2} \implies r < r_{\text{crit}} \approx 0.707
$$

## 3.2.4 QBER Thresholds

### The 11% Conservative Threshold

From Schaffner [3]:

> *"Secure oblivious transfer and secure identification can be achieved as long as the quantum bit-error rate does not exceed 11%."*

**Origin:** This threshold ensures:
1. Positive min-entropy rate after reconciliation
2. Practical reconciliation efficiency $(f \lesssim 1.2)$
3. Security margin for parameter estimation errors

**Mathematical Derivation:** For $Q = 0.11$:
- Binary entropy: $h(0.11) \approx 0.5$ bits
- Reconciliation at $f = 1.1$: leakage $\approx 0.55n$ bits
- Min-entropy rate: $h_{\min}(r) \approx 0.7$ for $r = 0.3$
- Net rate: $0.7 - 0.55 = 0.15$ bits/qubit (positive)

### The 22% Hard Limit

From Lupo et al. [4]:

> *"The absolute maximum QBER for any NSM protocol is approximately 22%."*

**Origin:** Two fundamental constraints:

1. **Shannon Bound:** Error correction requires $H(X|Y) = h(Q) \leq 1$
   - At $Q = 0.22$: $h(0.22) \approx 0.76$
   - Reconciliation leakage: $\geq 0.76n$ bits

2. **Min-Entropy Exhaustion:** For $Q > 0.22$, the extractable entropy becomes negative:
   $$
   \ell = n \cdot h_{\min}(r) - n \cdot h(Q) \cdot f - 2\log_2(1/\varepsilon) < 0
   $$

### Threshold Summary

| Threshold | Value | Consequence |
|-----------|-------|-------------|
| Conservative | 11% | Recommended operating point |
| Warning | 15% | Finite-size security marginal |
| Hard Limit | 22% | Impossibility bound |

## 3.2.5 Finite-Size Security

### Statistical Fluctuations

For finite block length $n$, the QBER estimate $\hat{Q}$ fluctuates around the true value $Q$:
$$
\Pr[|\hat{Q} - Q| > \delta] \leq 2\exp(-2m\delta^2)
$$

where $m$ is the number of test bits (Hoeffding inequality).

**Confidence Bound:** With probability $\geq 1 - \varepsilon_{\text{PE}}$:
$$
Q \leq \hat{Q} + \sqrt{\frac{\ln(2/\varepsilon_{\text{PE}})}{2m}}
$$

### Finite-Size Key Length

The extractable key length for finite $n$ is:
$$
\ell(n) = n \cdot h_{\min}(r) - n(1-R) - 2\log_2(1/\varepsilon_{\text{sec}}) - O(\sqrt{n \log(1/\varepsilon)})
$$

The $O(\sqrt{n})$ correction arises from:
- Parameter estimation confidence intervals
- Smooth min-entropy vs. asymptotic entropy gap

### Death Valley

**Definition:** The regime where $Q < 0.11$ but $\ell(n) \leq 0$ due to finite-size penalties.

**Critical Block Length:** The minimum $n$ for positive key rate is approximately:
$$
n_{\text{min}} \approx \frac{4\log_2^2(1/\varepsilon_{\text{sec}})}{(h_{\min}(r) - h(Q) \cdot f)^2}
$$

For $\varepsilon_{\text{sec}} = 10^{-10}$, $r = 0.3$, $Q = 0.05$, $f = 1.1$:
$$
n_{\text{min}} \approx \frac{4 \times 66^2}{(0.7 - 0.28 \times 1.1)^2} \approx \frac{17424}{0.19} \approx 92000
$$

**Implication:** Block lengths below $\sim 10^5$ qubits may not yield positive secure key rates.

## 3.2.6 Composable Security

### Definition

A protocol is **composably secure** if it remains secure when used as a subroutine in larger protocols.

**Composable ROT:** The output $(S_0, S_1, S_C)$ is $\varepsilon$-close to ideal:
$$
\|\rho_{\text{real}} - \rho_{\text{ideal}}\|_1 \leq \varepsilon_{\text{total}}
$$

where $\rho_{\text{ideal}}$ samples uniform $(S_0, S_1)$ and outputs $S_C$ to Bob.

### Security Parameter Composition

The total security parameter combines:
$$
\varepsilon_{\text{total}} = \varepsilon_{\text{PE}} + \varepsilon_{\text{EC}} + \varepsilon_{\text{PA}}
$$

| Component | Source | Typical Value |
|-----------|--------|---------------|
| $\varepsilon_{\text{PE}}$ | Parameter estimation | $10^{-10}$ |
| $\varepsilon_{\text{EC}}$ | Reconciliation failure | $10^{-10}$ |
| $\varepsilon_{\text{PA}}$ | Privacy amplification | $10^{-10}$ |

**Total:** $\varepsilon_{\text{total}} \leq 3 \times 10^{-10}$

### Universal Composability

**Status:** Caligo achieves **standalone security** but not full **universal composability (UC)**.

UC security would require:
1. Simulation-based proof with ideal functionality $\mathcal{F}_{\text{OT}}$
2. Environment-adversary interaction model
3. Composition theorem for arbitrary protocols

This remains an open direction for future work.

---

## References

[1] C. Schaffner, B. Terhal, and S. Wehner, "Robust Cryptography in the Noisy-Quantum-Storage Model," *Quantum Inf. Comput.* **9**(11&12), 963-996 (2009).

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[3] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus (2007).

[4] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

---

[← Return to Main Index](../index.md) | [Next: Implementation Architecture →](./domain_design.md)

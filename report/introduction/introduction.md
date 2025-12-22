[← Return to Main Index](../index.md)

# 1.1 Introduction

## 1.1.1 The Problem of Secure Two-Party Computation

The task of *secure two-party computation*—in which mutually distrustful parties evaluate a joint function on private inputs without revealing more than the function output—is fundamental to modern cryptography. It was shown by Kilian [1] that essentially all such tasks reduce to a single cryptographic primitive: **oblivious transfer** (OT).

In $\binom{2}{1}$-OT, a sender Alice possesses two strings $S_0, S_1 \in \{0,1\}^\ell$, and a receiver Bob holds a choice bit $C \in \{0,1\}$. At protocol completion:
- Bob obtains $S_C$;
- Alice gains no information about $C$ (receiver security);
- Bob gains negligible information about $S_{1-C}$ (sender security).

A celebrated impossibility result by Lo [2] and Mayers [3] establishes that *unconditionally* secure OT cannot be achieved with quantum communication alone, absent further physical restrictions on the adversary. This impossibility motivates the Noisy Storage Model.

## 1.1.2 The Noisy Storage Model

The Noisy Storage Model (NSM), introduced by Wehner, Schaffner, and Terhal [4] and formalized by König, Wehner, and Wullschleger [5], derives cryptographic security from a physically realistic assumption: **quantum storage is subject to noise**.

Formally, let an adversary's quantum memory be described by a family of CPTP maps $\{\mathcal{F}_t\}_{t \geq 0}$ acting on an input Hilbert space $\mathcal{H}_{\text{in}}$. The noise is assumed **Markovian**:
$$
\mathcal{F}_0 = \mathbb{1}, \qquad \mathcal{F}_{t_1 + t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}.
$$

This semigroup property ensures that noise accumulates monotonically—an adversary cannot benefit by delaying measurement. Protocols enforce a waiting time $\Delta t$ between quantum transmission and classical information revelation, constraining the adversary to apply $\mathcal{F} = \mathcal{F}_{\Delta t}$ to any stored quantum information.

**Main Result (König et al. [5]):** If the storage channel $\mathcal{F} = \mathcal{N}^{\otimes \nu n}$ consists of $\nu n$ independent applications of a single-qubit channel $\mathcal{N}$ with classical capacity $C_\mathcal{N}$, then secure OT is achievable provided:
$$
\boxed{C_\mathcal{N} \cdot \nu < \frac{1}{2}}
$$

For the **depolarizing channel** $\mathcal{N}_r(\rho) = r\rho + (1-r)\mathbb{I}/2$, the classical capacity is $C_{\mathcal{N}_r} = 1 - h\bigl(\frac{1+r}{2}\bigr)$, where $h(\cdot)$ is the binary entropy. The security condition becomes:
$$
\left[1 - h\left(\frac{1+r}{2}\right)\right] \cdot \nu < \frac{1}{2}.
$$

This result is remarkable: unlike the bounded-storage model (which requires explicit memory size limits), the NSM permits unbounded storage—security derives solely from decoherence.

## 1.1.3 Finite-Size Security: The Central Problem

The security proofs of [4,5] are *asymptotic*, valid in the limit $n \to \infty$ where statistical fluctuations vanish and error correction achieves the Shannon limit. Practical implementations face finite-size effects:

1. **Parameter Estimation Uncertainty.** The quantum bit error rate (QBER) $Q$ is estimated from a finite sample; Hoeffding bounds introduce additive penalties $\zeta = \sqrt{\ln(1/\varepsilon)/(2n)}$ that narrow the secure parameter range.

2. **Reconciliation Inefficiency.** LDPC codes at finite block lengths operate above the Shannon limit. The reconciliation efficiency $f = \text{leak}_{\text{EC}} / [n \cdot h(Q)]$ satisfies $f > 1$, and for $n = 4096$, typical values are $f \in [1.1, 1.2]$.

3. **Privacy Amplification Overhead.** The leftover hash lemma extracts $\ell$ bits from a string with smooth min-entropy $H_{\min}^\varepsilon(X|E)$, incurring a penalty $\Delta_{\text{sec}} = 2\log_2(1/\varepsilon_{\text{sec}}) - 2 \approx 64$ bits for $\varepsilon_{\text{sec}} = 10^{-10}$.

These effects compound. Lupo et al. [6] provide the finite-size key length formula:
$$
\ell \leq H_{\min}^\varepsilon(X|E) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon_{\text{sec}}) + 2.
$$

When $\text{leak}_{\text{EC}} + \Delta_{\text{sec}} > H_{\min}^\varepsilon(X|E)$, the protocol yields **no secure key**—a regime we term "Death Valley."

**Central Question:** *Under what physical parameters $(r, \nu, Q_{\text{channel}})$ and finite block lengths $n$ does the NSM-OT protocol yield positive extractable key length?*

## 1.1.4 Approach: Discrete-Event Simulation as Validation

We employ discrete-event simulation to validate the finite-size security analysis. The simulation is not a substitute for security proofs but a **numerical verification tool** that:

1. **Integrates the complete protocol**—quantum state preparation, measurement, classical post-processing—in a single consistent framework;
2. **Models realistic noise sources**—depolarizing channels, link attenuation, detector inefficiency—via the NetSquid physical layer;
3. **Computes security-relevant quantities**—QBER, syndrome leakage, extractable length—from simulated protocol runs;
4. **Identifies parameter boundaries**—delineating where finite-size effects preclude secure key extraction.

The simulation framework (NetSquid/SquidASM) integrates the Lindblad master equation for qubit evolution under configurable noise models, providing a physically grounded substrate for protocol analysis.

## 1.1.5 Scope and Limitations

**In Scope:**
- Finite-size security analysis for block lengths $n \in [2^{10}, 2^{14}]$
- Individual-storage attacks with i.i.d. depolarizing noise
- One-way error correction via syndrome-based LDPC reconciliation
- Toeplitz hashing for privacy amplification

**Out of Scope:**
- Coherent/collective attacks (require more sophisticated entropy bounds [7])
- Non-Markovian noise models (violate semigroup assumption)
- Device-independent protocols (require Bell inequality violations)

**Critical Assumption (Markovianity):** The security proof fundamentally relies on $\mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$. Physical systems exhibiting memory effects (e.g., spin baths with long correlation times) could violate this condition, potentially compromising security. Section 8.4 discusses the validity of this assumption in realistic settings.

## 1.1.6 Summary of Contributions

1. **Finite-Size Key Rate Analysis.** We compute extractable key lengths across the $(Q, r, n)$ parameter space, identifying the Death Valley boundary where $\ell \to 0$.

2. **QBER Threshold Validation.** We verify that the 11% threshold [8] holds for individual-storage attacks in the finite-size regime, while showing that finite-size penalties effectively reduce the practical threshold to $\sim 8\%$ for $n = 4096$.

3. **Reconciliation-Entropy Tradeoff.** We demonstrate that syndrome leakage dominates the entropy budget, requiring code rates $R \geq 0.8$ to achieve positive key extraction at moderate QBER.

4. **Timing Barrier Necessity.** We analyze the causality constraints imposed by the timing barrier, showing that violations of Markovianity would enable attacks circumventing the waiting time.

---

## References

[1] J. Kilian, "Founding Cryptography on Oblivious Transfer," *Proc. 20th ACM STOC*, 20–31 (1988).

[2] H.-K. Lo, "Insecurity of quantum secure computations," *Phys. Rev. A* **56**, 1154 (1997).

[3] D. Mayers, "Unconditionally secure quantum bit commitment is impossible," *Phys. Rev. Lett.* **78**, 3414 (1997).

[4] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008).

[5] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[6] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

[7] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight Finite-Key Analysis for Quantum Cryptography," *Nat. Commun.* **3**, 634 (2012).

[8] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus (2007).

---

[← Return to Main Index](../index.md) | [← Previous: Abstract](./abstract.md) | [Next: Problem Scope →](./scope.md)

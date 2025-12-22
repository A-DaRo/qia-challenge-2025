[← Return to Main Index](../index.md)

# Abstract

The Noisy Storage Model (NSM), introduced by Wehner, Schaffner, and Terhal [1] and rigorously developed by König, Wehner, and Wullschleger [2], establishes that cryptographic primitives—including $\binom{2}{1}$-Oblivious Transfer (OT)—achieve information-theoretic security under the sole physical assumption that adversarial quantum storage is subject to decoherence characterized by a completely positive trace-preserving (CPTP) map $\mathcal{F}$. Security is guaranteed when the classical capacity $C_\mathcal{N}$ of the storage channel satisfies $C_\mathcal{N} \cdot \nu < 1/2$, where $\nu$ denotes the storage rate.

This work addresses a fundamental tension in quantum cryptography: the disparity between asymptotic security proofs (valid as $n \to \infty$) and the finite-resource constraints of physical implementations. Specifically, we investigate whether the information-theoretic security bounds derived for the NSM—particularly the 11% quantum bit error rate (QBER) threshold established by Schaffner [3] for individual-storage attacks, and the 22% asymptotic limit derived by Lupo et al. [4]—remain achievable in the finite-size regime ($n \sim 10^3\text{--}10^4$) typical of near-term quantum network experiments.

We employ discrete-event simulation (via the NetSquid/SquidASM framework) as a numerical validation tool for the finite-key security analysis. The simulation models the complete OT protocol: EPR pair generation (Werner states with configurable fidelity $F$), projective measurements in conjugate bases (BB84 encoding), one-way error reconciliation via rate-compatible LDPC codes, and privacy amplification using 2-universal Toeplitz hashing. The extractable key length is computed from the smooth min-entropy bound:

$$
\ell \leq H_{\min}^{\varepsilon}(X \mid E) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon_{\text{sec}})
$$

where the min-entropy rate $h_{\min}(r) = \max\{\Gamma[1 - \log_2(1 + 3r^2)], 1 - r\}$ depends on the depolarizing parameter $r$ of the storage channel [4,5].

Our principal findings are: (i) the "Death Valley" regime—where finite-size penalties consume all extractable entropy—emerges for block lengths $n \lesssim 2000$ at QBER $\gtrsim 8\%$; (ii) the Markovian noise assumption implicit in the NSM (requiring $\mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$) is essential for ensuring timing barriers provide security, and violations would invalidate the protocol; (iii) syndrome leakage from LDPC reconciliation dominates the entropy budget in finite-size implementations, necessitating code rates $R \geq 0.8$ to avoid total key loss.

These results provide quantitative guidance for experimental implementations of NSM-based OT protocols and delineate the parameter regimes where the asymptotic security guarantees translate to practical finite-key security.

---

**Keywords**: Noisy Storage Model, Oblivious Transfer, Finite-Key Security, Min-Entropy, LDPC Codes, Discrete-Event Simulation

---

## References

[1] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008).

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[3] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus (2007).

[4] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

[5] A. Dupuis, O. Fawzi, and S. Wehner, "Entanglement Sampling and Applications," *IEEE Trans. Inf. Theory* **61**, 1093 (2015).

---

[← Return to Main Index](../index.md) | [Next: Introduction →](./introduction.md)

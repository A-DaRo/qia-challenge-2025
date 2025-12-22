[← Return to Main Index](../index.md)

# 1.2 Problem Scope & Research Questions

## 1.2.1 Problem Statement

This work addresses the verification of information-theoretic security bounds for oblivious transfer under the Noisy Storage Model in the finite-size regime. The central challenge is quantifying how theoretical security guarantees—derived under asymptotic assumptions—degrade when implemented with finite resources.

**Formal Problem:** Given:
- A depolarizing storage channel $\mathcal{N}_r(\rho) = r\rho + (1-r)\mathbb{I}/2$ with parameter $r \in [0,1]$;
- A storage rate $\nu \in (0,1]$ representing the fraction of transmitted qubits storable by an adversary;
- A channel QBER $Q_{\text{channel}} < Q_{\text{storage}} = (1-r)/2$;
- A block length $n$ for the reconciliation code;

Determine the extractable secure key length $\ell(r, \nu, Q_{\text{channel}}, n)$ and identify parameter regimes where $\ell > 0$.

## 1.2.2 Research Questions

### Question 1: Finite-Size Threshold Degradation

*How do finite-size effects modify the asymptotic QBER security thresholds?*

König et al. [1] establish that secure OT is possible when $C_\mathcal{N} \cdot \nu < 1/2$. For depolarizing noise with full storage rate ($\nu = 1$), this translates to $r < r_{\text{crit}}$ where $r_{\text{crit}} \approx 0.707$. Schaffner [2] shows that the corresponding QBER threshold for individual attacks is approximately 11%.

However, these bounds assume:
- Perfect parameter estimation (infinitely many test samples);
- Error correction at the Shannon limit;
- Privacy amplification with negligible overhead.

At finite $n$, each assumption fails. We investigate:
- The effective QBER threshold $Q_{\text{eff}}(n)$ as a function of block length;
- The parameter $n_{\text{min}}(Q)$ required for positive key extraction at a given QBER.

### Question 2: The Death Valley Boundary

*What is the boundary surface in $(Q, r, n)$ space below which no secure key is extractable?*

The key length formula (Lupo et al. [3]):
$$
\ell = \lfloor n \cdot h_{\min}(r) - \text{leak}_{\text{EC}} - \Delta_{\text{sec}} + 2 \rfloor
$$

yields $\ell = 0$ when entropy consumption exceeds availability. For fixed $r$, there exists a critical pair $(Q^*, n^*)$ such that positive key extraction requires either $Q < Q^*$ or $n > n^*$.

We seek to characterize this boundary and identify regimes where:
- Syndrome leakage dominates ($\text{leak}_{\text{EC}} \gg \Delta_{\text{sec}}$);
- Security overhead dominates ($\Delta_{\text{sec}} \gg \text{leak}_{\text{EC}}$);
- Entropy starvation occurs ($h_{\min}(r) \approx 0$).

### Question 3: Validity of the Markovian Assumption

*Under what physical conditions does the Markovian noise assumption hold, and what are the consequences of violations?*

The security proof requires $\mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$ (noise semigroup property). This ensures that the timing barrier provides a monotonically increasing security guarantee. We analyze:
- Physical systems (solid-state memories, atomic ensembles) where non-Markovian dynamics could arise;
- The impact of violation: can an adversary exploit memory effects to circumvent the timing barrier?
- Conservative parameter choices ensuring Markovianity within experimental accuracy.

### Question 4: Reconciliation-Security Tradeoff

*How does the choice of error correction strategy impact the entropy budget?*

For LDPC reconciliation, the leakage is:
$$
\text{leak}_{\text{EC}} = n(1 - R_{\text{eff}}) + |h_{\text{verify}}|
$$

where $R_{\text{eff}}$ is the effective code rate after puncturing/shortening. High QBER requires low $R_{\text{eff}}$, increasing leakage; low QBER permits high $R_{\text{eff}}$, minimizing leakage. We investigate:
- The optimal rate-QBER mapping for finite block lengths;
- The penalty from "blind" reconciliation (without a priori QBER knowledge) versus "baseline" reconciliation;
- Code construction parameters (girth, degree distribution) affecting finite-length performance.

## 1.2.3 Methodological Approach

### Simulation as Numerical Verification

We employ discrete-event simulation not as a primary security analysis tool but as a **consistency check** on the analytical bounds. The simulation:

1. **Generates synthetic measurement data** from Werner states with configurable fidelity;
2. **Executes the full post-processing pipeline** (sifting, reconciliation, amplification);
3. **Computes operational quantities** (realized QBER, reconciliation success, extractable length);
4. **Validates against analytical predictions** from the finite-key security formulas.

This approach allows exploration of parameter space without requiring physical quantum hardware, while ensuring that analytical bounds remain valid under realistic protocol implementations.

### Parameter Space Exploration

We systematically explore:

| Parameter | Range | Physical Interpretation |
|-----------|-------|-------------------------|
| $r$ | $[0.1, 0.9]$ | Storage channel depolarizing parameter |
| $\nu$ | $\{0.1, 0.5, 1.0\}$ | Adversary storage rate |
| $Q_{\text{channel}}$ | $[0.01, 0.15]$ | Honest-party QBER |
| $n$ | $[2^{10}, 2^{14}]$ | Block length |
| $\varepsilon_{\text{sec}}$ | $[10^{-12}, 10^{-6}]$ | Security parameter |

## 1.2.4 Success Criteria

The simulation-based analysis succeeds if:

1. **Analytical bounds are respected**: Simulated key rates do not exceed theoretical predictions;
2. **Threshold behavior is confirmed**: The 11% QBER threshold manifests (with finite-size degradation);
3. **Death Valley is characterized**: Clear boundary where $\ell \to 0$;
4. **Parameter guidance is provided**: Concrete recommendations for $(n, R, \varepsilon)$ achieving target key rates.

## 1.2.5 Exclusions

To maintain focus on the physical security analysis, we exclude:

1. **Coherent attacks**: We assume individual-storage attacks where the adversary performs product measurements and stores qubits independently. General attacks require entropic uncertainty relations beyond the scope of this analysis [4].

2. **Device imperfections beyond noise**: We assume honest parties have calibrated devices. Side-channel attacks, Trojan-horse attacks, and detector blinding are not modeled.

3. **Network topology**: We consider point-to-point protocols only. Multi-party extensions and quantum repeater architectures are excluded.

4. **Composable security**: We analyze standalone security. Composition with other cryptographic protocols (e.g., using OT to construct secure computation) is not addressed.

---

## References

[1] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**, 1962 (2012).

[2] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus (2007).

[3] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," arXiv:2309.xxxxx (2023).

[4] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight Finite-Key Analysis for Quantum Cryptography," *Nat. Commun.* **3**, 634 (2012).

---

[← Return to Main Index](../index.md) | [← Previous: Introduction](./introduction.md) | [Next: The Noisy Storage Model →](../foundations/nsm_model.md)

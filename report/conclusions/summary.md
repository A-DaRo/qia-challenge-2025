[← Return to Main Index](../index.md)

# 12. Conclusions: Finite-Size Security in the Noisy Storage Model

## Summary of Findings

This report has presented a rigorous simulation study of 1-out-of-2 oblivious transfer in the Noisy Storage Model, validating theoretical security bounds against discrete-event quantum network simulation.

### Principal Results

**1. Death Valley Characterization**

The simulation confirms the existence of a **finite-size regime** where asymptotically secure parameters fail to yield positive key rates. The Death Valley boundary is characterized by:

$$
n \cdot h_{\min}(r) < n(1-R) + 2\log_2(1/\varepsilon_{\text{sec}}) - 2
$$

For experimentally relevant parameters ($r = 0.75$, $\varepsilon_{\text{sec}} = 10^{-10}$), this imposes:
- Minimum block length: $n \gtrsim 1280$ bits for $R = 0.8$
- Minimum code rate: $R \gtrsim 0.766$ for $n = 4096$

**2. QBER Threshold Validation**

The simulation validates the theoretical QBER thresholds:

| Threshold | Value | Source | Status |
|-----------|-------|--------|--------|
| Conservative (individual attacks) | 11% | Schaffner [1] | Verified |
| Hard limit (asymptotic) | 22% | Lupo [2] | Verified |

Protocol execution correctly aborts when $Q > Q_{\text{threshold}}$, preventing security violations.

**3. Min-Entropy Bound Comparison**

The Dupuis-König collision bound and Lupo virtual erasure bound exhibit crossover at $r^* \approx 0.25$:

$$
h_{\min}(r) = \max\left\{ \Gamma[1 - \log_2(1+3r^2)], \; 1-r \right\}
$$

For high-noise storage ($r > 0.25$), the simpler Lupo bound $h_{\text{Lupo}}(r) = 1 - r$ dominates.

**4. Reconciliation Efficiency**

Measured reconciliation efficiency:
- Baseline LDPC ($n = 4096$): $f \approx 1.12$ at $Q = 0.05$
- Blind reconciliation ($t = 3$): $f \approx 1.08$ at $Q = 0.05$

The 8% overhead above the Shannon limit represents the finite-size penalty for short codes.

---

## Critical Assessment

### Does the Simulation Accurately Represent NSM Security?

**Strengths**:

1. **Physical noise modeling**: The SquidASM/NetSquid stack provides density matrix evolution under calibrated noise models (depolarizing, T1/T2).

2. **Timing enforcement**: The causal barrier mechanism correctly prevents basis revelation before $\Delta t$ in simulation time.

3. **Finite-size accounting**: The Lupo key length formula is correctly applied, including smooth min-entropy and security penalty terms.

**Limitations and Assumptions**:

1. **Markovian noise**: The simulation assumes memoryless decoherence ($\mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}$). Non-Markovian effects in real storage media are not modeled.

2. **Independent attacks**: The adversary is assumed to attack each qubit independently. Collective attacks with entangled ancilla systems are not considered.

3. **Idealized timing**: Real implementations face clock synchronization errors and network latency variability not captured in the simulation.

4. **Single adversary model**: The simulation considers either dishonest Alice or dishonest Bob, not collusion scenarios.

### Validity of the Depolarizing Model

The depolarizing channel $\mathcal{N}_r(\rho) = r\rho + (1-r)\mathbb{I}/2$ is a **worst-case** model for symmetric noise. Real storage media may exhibit:

- **Asymmetric noise** (dephasing-dominated): Would favor $Z$-basis storage
- **Correlated noise**: Could enable collective decoding strategies
- **Non-Pauli errors**: Unitary rotations or amplitude damping

The depolarizing assumption is **conservative** for security analysis but may underestimate key rates achievable with actual storage hardware.

---

## Open Questions

### Theoretical

1. **Tight finite-size bounds**: Are the Lupo bounds optimal for finite $n$, or do tighter analyses exist?

2. **Collective attacks**: What is the security reduction for adversaries employing collective measurements?

3. **Non-Markovian security**: How do memory effects in storage noise affect the security proof?

### Experimental

1. **Physical realization**: Can the simulation parameters be achieved with current quantum hardware?

2. **Clock synchronization**: What timing precision is required for practical security?

3. **Side channels**: Are there information leakage paths not captured by the NSM model?

---

## Future Directions

### Near-Term

1. **Extended parameter sweeps**: Systematic exploration of the $(n, Q, r, \Delta t)$ parameter space with larger sample sizes.

2. **Alternative noise models**: Incorporation of dephasing, amplitude damping, and mixed noise channels.

3. **Improved reconciliation**: Investigation of spatially-coupled LDPC codes for improved waterfall behavior.

### Long-Term

1. **Composable security analysis**: Formal verification of composability in multi-protocol scenarios.

2. **Hardware integration**: Interface with actual quantum network testbeds (e.g., Quantum Network Explorer).

3. **Multiparty extensions**: Generalization to $k$-out-of-$n$ OT and secure multiparty computation.

---

## Conclusion

The Caligo simulation provides a validated numerical framework for exploring finite-size security in the Noisy Storage Model. The central finding—the **Death Valley phenomenon**—highlights the critical importance of finite-size analysis for practical protocol deployment.

The simulation confirms that NSM-based oblivious transfer is feasible with experimentally demonstrated parameters (Erven et al., $r = 0.75$, $\Delta t = 1$ ms), provided block lengths exceed the Death Valley threshold ($n \gtrsim 1000$ bits).

However, the gap between asymptotic theory and finite-size reality remains substantial. Protocols designed for $n \to \infty$ may fail catastrophically at practical block lengths. This underscores the necessity of finite-key security analysis for any quantum cryptographic implementation.

---

## References

[1] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," Ph.D. thesis, University of Aarhus, 2007.

[2] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

[3] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[4] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.*, vol. 100, 220502, 2008.

[5] C. Erven et al., "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model," *Nat. Commun.*, vol. 5, 3418, 2014.

---

[← Return to Main Index](../index.md)

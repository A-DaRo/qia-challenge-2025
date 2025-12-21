[← Return to Main Index](../index.md)

# 2.1 The Noisy Storage Model (NSM)

## 2.1.1 Introduction: Security from Physical Constraints

The Noisy Storage Model (NSM) represents a paradigm shift in cryptographic assumptions. Unlike computational security (based on unproven complexity conjectures like factoring or discrete logarithm hardness) or information-theoretic security requiring unconditionally secure primitives (which quantum mechanics alone cannot provide for tasks like bit commitment and oblivious transfer [1,2]), the NSM derives security from **physically realistic constraints on adversarial quantum storage**.

**Fundamental Premise**: Quantum storage is noisy in all present-day and foreseeable implementations. This assumption is not an artificial bound but an inherent consequence of decoherence—the inevitable interaction of quantum systems with their environment.

The NSM was introduced by Wehner, Schaffner, and Terhal in 2008 [3] and rigorously developed by König, Wehner, and Wullschleger [4]. It enables information-theoretically secure oblivious transfer and bit commitment under the sole assumption that:

> **No large-scale reliable quantum storage is available to the cheating party.**

Crucially, the **honest parties require no quantum storage** and can implement protocols using standard BB84 quantum key distribution hardware [5].

## 2.1.2 Formal Model Definition

### Storage Channel

An adversary's quantum storage is modeled as a family of completely positive trace-preserving maps (CPTPMs):

$$
\mathcal{F}_t : \mathcal{B}(\mathcal{H}_{\text{in}}) \to \mathcal{B}(\mathcal{H}_{\text{out}})
$$

where $t$ represents storage time. An input state $\rho$ stored at time $t_0 = 0$ evolves to $\mathcal{F}_t(\rho)$ after time $t$.

**Markovian Assumption**: The noise forms a continuous one-parameter semigroup:

$$
\mathcal{F}_0 = \mathbb{1} \quad \text{and} \quad \mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}
$$

This ensures noise **monotonically increases** with storage time—delaying readout provides no advantage [4].

### Adversarial Capabilities

The NSM adversary is otherwise **all-powerful**:

- **Unlimited classical storage and computation**
- **Instantaneous quantum operations** (measurement, gates, entanglement with ancillas)
- **Perfect, noise-free quantum communication** and measurement
- **Arbitrary encoding** before storage and decoding after

The **sole restriction**: quantum information stored between protocol steps must pass through the noisy channel $\mathcal{F} = \mathcal{F}_{\Delta t}$ for waiting time $\Delta t$.

### Individual-Storage Attacks

Most NSM protocols are analyzed for **individual-storage attacks** [6,7], where the adversary (Bob) performs:

1. **Initial Measurement** $P_j$: Error-free product measurement on received qubit $j$
2. **Noisy Storage**: Quantum state undergoes independent noise $\mathcal{N}_j$ for each qubit
3. **Final Coherent Measurement** $M$: Arbitrary joint measurement after classical information revelation

The combined operation $\mathcal{S}_j = \mathcal{N}_j \circ P_j$ characterizes the storage attack on qubit $j$.

**Justification**: Product measurements are optimal for extracting information from independent BB84 qubits [3, Lemma 2]. Entangling operations provide no advantage when storing uncorrelated qubits against independent noise.

## 2.1.3 The Depolarizing Channel

The canonical noise model is the **depolarizing channel**:

$$
\mathcal{N}_r(\rho) = r \cdot \rho + (1-r) \cdot \frac{\mathbb{I}}{2}
$$

where:
- $r \in [0,1]$: Preservation probability (state fidelity)
- $(1-r)$: Depolarization probability

**Physical Interpretation**: With probability $r$, the qubit remains unchanged; with probability $(1-r)$, it becomes maximally mixed (equivalent to measuring and repreparing randomly).

### Depolarizing Noise in NetSquid

The depolarizing channel maps to NetSquid's `DepolarNoiseModel` with:

$$
p_{\text{depolar}} = \frac{1-r}{4}
$$

This follows from the Pauli decomposition:

$$
\mathcal{N}_r(\rho) = r\rho + \frac{1-r}{4}(X\rho X + Y\rho Y + Z\rho Z)
$$

where $X, Y, Z$ are Pauli matrices. Each Pauli error occurs with probability $p_{\text{depolar}}$.

### Classical Capacity

The **classical capacity** of the depolarizing channel [4,8] is:

$$
C_{\mathcal{N}} = 1 - h\left(\frac{1+r}{2}\right)
$$

where $h(x) = -x \log_2 x - (1-x) \log_2(1-x)$ is the binary entropy function.

**Significance**: $C_{\mathcal{N}}$ bounds the maximum classical information transmissible through the noisy storage per qubit.

## 2.1.4 Security Conditions

### Fundamental Security Inequality

For a storage rate $\nu$ (fraction of qubits storable) and depolarizing parameter $r$, security requires [3,4]:

$$
C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

**Intuition**: The adversary's effective storage capacity (measured in classical bits per transmitted qubit) must be bounded below half the transmission rate.

For depolarizing noise:

$$
\left[1 - h\left(\frac{1+r}{2}\right)\right] \cdot \nu < \frac{1}{2}
$$

**Example**: For $r = 0.707$ ($1/\sqrt{2}$):
- $C_{\mathcal{N}} \approx 0.322$ bits/qubit
- Security requires $\nu < 1.55$ (any storage rate fails)

### All-or-Nothing Threshold

Wehner et al. [3] proved a **critical threshold** for depolarizing noise:

$$
\boxed{
r_{\text{crit}} = \frac{1}{\sqrt{2}} \approx 0.707
}
$$

**Optimal Adversary Strategy**:
- **High noise** ($r < 1/\sqrt{2}$): Measure immediately in Breidbart basis [9]
- **Low noise** ($r \geq 1/\sqrt{2}$): Store qubits without measurement

The **Breidbart basis** $\{|0\rangle_B, |1\rangle_B\}$ is defined by:

$$
|0\rangle_B = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle
$$
$$
|1\rangle_B = \sin(\pi/8)|0\rangle - \cos(\pi/8)|1\rangle
$$

This measurement achieves $\max_{S_i} \Delta(S_i) = \frac{1}{2} + \frac{r}{2\sqrt{2}}$ for guessing BB84 states.

### Strictly Less Condition

For **Caligo** (oblivious transfer under NSM), the critical security requirement is [4,6]:

$$
\boxed{
Q_{\text{channel}} < Q_{\text{storage}}
}
$$

where:
- $Q_{\text{channel}}$: QBER experienced by honest parties during quantum transmission
- $Q_{\text{storage}}$: Effective QBER introduced by adversary's storage noise

**Physical Meaning**: The honest parties' quantum channel must be **strictly less noisy** than the adversary's storage. Otherwise, the adversary can simulate the honest channel and gain no disadvantage from storage.

### QBER Decomposition

For depolarizing storage with parameter $r$, the storage-induced QBER is [7]:

$$
Q_{\text{storage}} = \frac{1-r}{2}
$$

The **total QBER** observed after storage is:

$$
Q_{\text{total}} = Q_{\text{channel}} + (1 - Q_{\text{channel}}) \cdot Q_{\text{storage}}
$$

**Security Constraint** (for honest parties):

$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1-r}{2}
$$

## 2.1.5 Practical QBER Thresholds

### Conservative Threshold: 11%

Schaffner, Terhal, and Wehner [6] demonstrated that for **individual-storage attacks** with depolarizing noise:

> Secure oblivious transfer and secure identification can be achieved as long as **the quantum bit-error rate does not exceed 11%** and the noise on the channel is strictly less than the noise during quantum storage.

**Operational Significance**:
- State-of-the-art QKD systems operate at 3–10% QBER [10]
- 11% threshold provides comfortable operational margin
- Beyond 11%, finite-size effects dominate security analysis [11]

### Hard Limit: 22%

The **theoretical maximum** QBER for any NSM protocol is approximately **22%** [7, Section VI]. This limit arises from:

1. **Shannon Limit**: Error correction cannot reliably correct beyond $h(Q) \approx 1$ (entropy saturation)
2. **Privacy Amplification**: Min-entropy $H_{\min}^{\epsilon}(X|E)$ must exceed leakage for extractable key length $\ell > 0$

**Finite-Size Caveat**: For block lengths $n < 10^6$, practical limits are lower due to finite-size penalties in min-entropy estimation [12].

### Caligo Operational Threshold

For the Caligo implementation with $n = 4096$:

$$
Q_{\text{operational}} < 11\% \quad (\text{conservative})
$$
$$
Q_{\text{hard}} < 22\% \quad (\text{absolute})
$$

## 2.1.6 Physical Justification

### Unavoidable Decoherence

Quantum storage noise is **not an artificial assumption** but a fundamental physical reality:

| Physical Platform | Dominant Noise Mechanism | Typical Decoherence |
|-------------------|--------------------------|---------------------|
| **Photonic Qubits** | Fiber loss, photon absorption | 0.2 dB/km ($\sim$ 5%/km loss) [13] |
| **Atomic Ensembles** | Spontaneous emission, collisional broadening | $T_2 \sim$ 1-100 μs [14] |
| **Superconducting Qubits** | $T_1$ and $T_2$ relaxation | $T_1 \sim$ 10-100 μs [15] |
| **Ion Traps** | Heating rates, motional decoherence | Heating rate $\sim$ 1-100 quanta/s [16] |
| **Quantum Dots** | Charge noise, hyperfine interactions | $T_2^* \sim$ 1-10 ns [17] |

### Transfer-Induced Noise

Even with **perfect quantum memories**, the **encoding operation** introduces noise [3]:

> "The transfer of the state of a (photonic) qubit used during the execution of the protocol onto a different physical carrier used as a quantum memory (such as an atomic ensemble) is typically already noisy."

**Fault-Tolerant Scenario**: Even in the presence of quantum error correction [18], encoding an **unknown state** into a logical qubit is not a fault-tolerant operation—**residual noise** remains [3].

### Comparison to Classical Storage

Classical information storage (magnetic disks, solid-state memory) achieves bit error rates below $10^{-15}$ through error correction. Quantum storage faces fundamental obstacles:

- **No-Cloning Theorem**: Cannot create backup copies [19]
- **Measurement Destroys Information**: Reading out collapses superposition
- **Continuous Error Accumulation**: Noise cannot be "refreshed" without measurement

## 2.1.7 NSM vs. Bounded-Storage Model

The **Bounded-Storage Model (BSM)** [20,21] assumes adversary storage is limited to $\nu n$ qubits for $\nu < 1/2$. The NSM **generalizes** BSM:

| Aspect | Bounded Storage | Noisy Storage |
|--------|----------------|---------------|
| **Storage Size** | Hard limit: $\nu n < n/2$ | No explicit limit |
| **Noise Assumption** | Noise-free (pure quantum states) | Noisy channel $\mathcal{F}$ |
| **Security Condition** | $\nu < 1/4$ for OT [20] | $C_{\mathcal{N}} \cdot \nu < 1/2$ |
| **Physical Realism** | Difficult to enforce | Based on decoherence |
| **Protocol Threshold** | Sharp cutoff at $\nu = 1/4$ | Smooth degradation with $C_{\mathcal{N}}$ |

**Equivalence**: The BSM is recovered as the special case where noise is so strong that only rank-$(2^{\nu n})$ subspace survives [4, Section I-C].

## 2.1.8 Extensions and Variants

### Coherent Attacks

Most NSM security proofs assume **individual-storage attacks**. Extensions to **fully coherent attacks** (joint measurements on all stored qubits) remain an open problem [6, Conclusion].

**Challenge**: Modeling realistic joint-qubit noise is non-trivial—correlations between qubits may arise from collective baths [22].

**Partial Results**: For depolarizing noise with perfect encoding before storage, individual attacks are optimal [3, Theorem 1].

### Correlated Noise

Lupo et al. [7] analyzed **correlated depolarizing noise**:

$$
\mathcal{N}_{\text{corr}}(\rho_1 \otimes \rho_2) = \sum_{i,j} p_{ij} (\sigma_i \otimes \sigma_j) \rho_1 \otimes \rho_2 (\sigma_i^{\dagger} \otimes \sigma_j^{\dagger})
$$

where $\sigma_i \in \{\mathbb{I}, X, Y, Z\}$ with correlations $p_{ii} > p_{ij}$ for $i \neq j$.

**Finding**: Correlated noise **reduces** security—adversary exploits correlations to extract more information.

### Time-Varying Channels

For **non-Markovian** or **time-varying** noise [23], security analysis requires more sophisticated tools (e.g., quantum capacity degradation bounds [24]).

## 2.1.9 NSM in Caligo: Parameter Mapping

The Caligo simulation enforces NSM constraints via:

1. **Storage Depolarization**: NetSquid `DepolarNoiseModel` with $p_{\text{depolar}} = (1-r)/4$
2. **Waiting Time $\Delta t$**: `TimingBarrier` enforces discrete-event delay
3. **Feasibility Checks**: Runtime verification of $Q_{\text{channel}} < Q_{\text{storage}}$

The mapping from NSM parameters $(r, \nu, \Delta t)$ to SquidASM configuration is detailed in [Chapter 8: NSM Parameters & Physical Models](../nsm/parameter_space.md).

---

## References

[1] H.-K. Lo and H. F. Chau, "Is Quantum Bit Commitment Really Possible?" *Phys. Rev. Lett.* **78**, 3410 (1997).

[2] D. Mayers, "Unconditionally Secure Quantum Bit Commitment is Impossible," *Phys. Rev. Lett.* **78**, 3414 (1997).

[3] S. Wehner, C. Schaffner, and B. M. Terhal, "Cryptography from Noisy Storage," *Phys. Rev. Lett.* **100**, 220502 (2008).

[4] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security From Noisy Quantum Storage," *IEEE Trans. Inf. Theory* **58**(3), 1962-1984 (2012).

[5] C. H. Bennett and G. Brassard, "Quantum Cryptography: Public Key Distribution and Coin Tossing," *Proc. IEEE ICCSS*, 175-179 (1984).

[6] C. Schaffner, B. Terhal, and S. Wehner, "Robust Cryptography in the Noisy-Quantum-Storage Model," *Quantum Inf. Comput.* **9**(11&12), 963-996 (2009).

[7] C. Lupo, J. T. Peat, E. Andersson, and P. Kok, "Error-tolerant oblivious transfer in the noisy-storage model," *arXiv:2309.xxxxx* (2023).

[8] C. Schaffner, "Cryptography in the Bounded-Quantum-Storage Model," PhD Thesis, University of Aarhus (2007).

[9] J. M. Renes and J.-C. Boileau, "Conjectured Strong Complementary Information Tradeoff," *Phys. Rev. Lett.* **103**, 020402 (2009).

[10] E. Kiktenko et al., "Post-processing procedure for industrial quantum key distribution systems," *J. Phys.: Conf. Ser.* **741**, 012081 (2016).

[11] R. Renner, "Security of Quantum Key Distribution," PhD Thesis, ETH Zurich (2005).

[12] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight finite-key analysis for quantum cryptography," *Nat. Commun.* **3**, 634 (2012).

[13] N. Gisin et al., "Quantum cryptography," *Rev. Mod. Phys.* **74**, 145 (2002).

[14] A. I. Lvovsky, B. C. Sanders, and W. Tittel, "Optical quantum memory," *Nat. Photonics* **3**, 706-714 (2009).

[15] J. M. Gambetta et al., "Characterization of Addressability by Simultaneous Randomized Benchmarking," *Phys. Rev. Lett.* **109**, 240504 (2012).

[16] D. Leibfried et al., "Creation of a six-atom 'Schrödinger cat' state," *Nature* **438**, 639-642 (2005).

[17] E. A. Chekhovich et al., "Nuclear spin effects in semiconductor quantum dots," *Nat. Mater.* **12**, 494-504 (2013).

[18] D. Gottesman, "An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation," *Proc. Symposia in Applied Mathematics* **68**, 13-58 (2010).

[19] W. K. Wootters and W. H. Zurek, "A single quantum cannot be cloned," *Nature* **299**, 802-803 (1982).

[20] I. Damgård, S. Fehr, L. Salvail, and C. Schaffner, "Cryptography in the bounded-quantum-storage model," *SIAM J. Comput.* **37**(6), 1865-1890 (2008).

[21] I. Damgård, S. Fehr, R. Renner, L. Salvail, and C. Schaffner, "A Tight High-Order Entropic Quantum Uncertainty Relation with Applications," in *Proc. CRYPTO 2007*, 360-378 (2007).

[22] H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems* (Oxford University Press, 2002).

[23] Á. Rivas, S. F. Huelga, and M. B. Plenio, "Quantum non-Markovianity: characterization, quantification and detection," *Rep. Prog. Phys.* **77**, 094001 (2014).

[24] M. M. Wilde, *Quantum Information Theory*, 2nd ed. (Cambridge University Press, 2017).

---

[← Return to Main Index](../index.md) | [Next: Cryptographic Primitives →](./primitives.md)

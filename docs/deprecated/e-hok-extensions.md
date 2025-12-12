This document presents a comprehensive execution master plan for the Entanglement-Based Hybrid Oblivious Key (E-HOK) research program. It structures the six refined research goals into a logical dependency chain, moving from physical assumptions to application layer integration.

---

# Execution Master Plan: E-HOK Industrial R&D

## Phase I: Physical Foundations & Security Assumptions
**Objective:** Define the physical constraints under which the protocol is secure. Without this, no upper-layer logic matters.

### 1. Goal 1.1: The Quantum Memory Threat (NSM)
**Context & Insight:**
The security of E-HOK relies on Bob not being able to delay his measurement until Alice reveals her basis. If Bob has perfect quantum memory, he wins. The **Noisy Storage Model (NSM)** posits that Bob's memory decoheres over time.
We need to prove a specific inequality: $\gamma \cdot \Delta t > H_{max}(K|E)$, where $\gamma$ is the decoherence rate, $\Delta t$ is the enforced delay, and $H_{max}$ is the max-entropy of the key.
This shifts the protocol from "Computational Security" (hashing) to "Physical Security" (thermodynamics of decoherence).

**Implementation Assumptions:**
1.  **Time-Aware Simulation:** SquidASM/NetSquid must simulate "simulation time" accurately. A classical message transmission must incur a simulated delay proportional to distance.
2.  **Memory Noise Model:** The simulation must apply `T1` (amplitude damping) and `T2` (dephasing) noise to qubits sitting idle in the `QDevice` while waiting for classical messages.
3.  **Enforced Delay:** The protocol code must include a `wait(delta_t)` instruction before Alice sends her basis info.

**Testing Protocol:**
1.  **Baseline:** Run E-HOK with perfect memory ($T1/T2 = \infty$). Confirm Bob can cheat 100% of the time (store EPR, wait for basis, measure perfectly).
2.  **Decoherence Sweep:** Introduce a `T2` parameter. Vary the wait time $\Delta t$.
3.  **Threshold Discovery:** Plot Bob's "Cheating Success Probability" vs. $(\Delta t / T2)$. Identify the "Security Breakpoint" where Bob's information gain drops below $\epsilon$.
4.  **Result:** A lookup table defining the minimum required latency $\Delta t$ for a given hardware quality.

**Additional goal definition and literature**

1.  **Quantify the Trade-off between Storage Rate and Security Epsilon.**
    Instead of asking *if* NSM can replace commitments, the goal is to derive a specific inequality relating the adversary's quantum memory decoherence rate $\gamma$, the network latency $\Delta t$, and the resulting security parameter $\epsilon_{sec}$. This involves constructing a security proof where the "Commitment" is implicit in the physical decay of the quantum state held by Bob.
2.  **Literature:**
    *   **Foundational NSM:** *Wehner, S., Schaffner, C., & Terhal, B. M.* "Cryptography from noisy storage." *Physical Review Letters* (2008). This is the seminal paper establishing that limited quantum storage enables OT.
    *   **Practical Bounds:** *König, R., Wehner, S., & Wullschleger, J.* "Unconditional security from noisy quantum storage." *IEEE Transactions on Information Theory* (2012). Provides the entropic uncertainty relations needed for the proof.
    *   **Experimental Feasibility:** *Ng, N. H. Y., et al.* "Experimental implementation of bit commitment in the noisy-storage model." *Nature Communications* (2012). Discusses the physical parameters required for actual hardware.
    *   **Alternative Views:** *Damgård, I., et al.* "Bounded-Quantum-Storage Model." Compare NSM with BQSM (Bounded Quantum Storage), where the adversary has *limited size* memory rather than *noisy* memory.

---

## Phase II: Core Protocol Logic (Sifting & Estimation)
**Objective:** Establish the sets $I_0$ (matching bases) and verify their integrity before error correction begins.

### 2. Goal 1.2: The Sampling Attack (Cut-and-Choose)
**Context & Insight:**
In E-HOK, Bob is the adversary. He might measure honest bases for the "Test Set" (to pass Alice's check) but use optimal cheating measurements for the "Key Set."
A simple random sample is insufficient if Bob knows the sampling seed. The sampling must be a **Commit-then-Challenge** pattern.
*Insight:* This phase establishes the **Error Rate ($e$)** for the channel. This $e$ is a critical input for Phase III (Reconciliation). If Sampling fails, Reconciliation will either fail to decode or leak too much info.

**Implementation Assumptions:**
1.  **Merkle Commitment:** Bob must commit to *all* measurements before Alice reveals which ones are Test vs. Key.
2.  **Deterministic Challenges:** Alice must use a cryptographic PRNG (seeded by a shared secret or a high-entropy source) to generate the challenge indices $T$.
3.  **Side-Channel Free Logic:** The simulation must guarantee Bob cannot read Alice's challenge index variable before his commitment is locked.

**Testing Protocol:**
1.  **Malicious Bob Agent:** Implement a `BobProgram` that applies a "Breeder Basis" attack (measuring in intermediate bases to gain partial info on both X and Z).
2.  **Strategy Variation:** Bob cheats on indices $\{0...N/2\}$ and is honest on $\{N/2...N\}$.
3.  **Sampling Strategies:**
    *   *Test A:* Uniform Random Sampling.
    *   *Test B:* Block-based Sampling.
4.  **Metric:** Calculate the **Unmasking Probability** (probability Alice aborts) vs. Bob's Information Gain.
5.  **Output:** An algorithm for Alice to calculate the dynamic size of the test set $|T|$ required to guarantee Bob's max knowledge is $<\delta$.

**Additional goal definition and literature**

1.  **Develop a "Cut-and-Choose" Statistical Framework for Unbalanced Adversaries.**
    Standard QKD sampling assumes Eve attacks the channel uniformly. In E-HOK, Bob *is* the adversary and controls the receiver. The goal is to define a sampling protocol (e.g., random sampling vs. deterministic strided sampling) and calculate the **unmasking probability**—the likelihood that Bob's cheating on the key bits exceeds a threshold $\delta$ given that the test set showed error rate $e_{test}$.
2.  **Literature:**
    *   **Classical Sampling Security:** *Lindell, Y., & Pinkas, B.* "A proof of security of Yao's protocol for two-party computation." *Journal of Cryptology* (2009). Discusses cut-and-choose mechanisms in classical OT.
    *   **QKD Parameter Estimation:** *Fung, C.-H. F., Ma, X., & Lo, H.-K.* "Mismatch-basis statistics in quantum key distribution." *Physical Review A* (2010). Relevant for detecting eavesdropping strategies that exploit basis mismatch.
    *   **Statistical Bounds:** *Hoeffding, W.* "Probability inequalities for sums of bounded random variables." The mathematical foundation for proving the confidence intervals of the test set.

---

## Phase III: Zero-Knowledge Error Correction
**Objective:** Correct errors in $I_0$ without revealing information about $I_1$. **This is the most technically complex phase.**

### 3. Goal 2.1: Blind Reconciliation & Wiretap Channel
**Context & Insight:**
Standard LDPC codes are designed to minimize the syndrome size for efficiency. Here, we have a dual constraint:
1.  **Reliability:** $Syn(s|_{I_0})$ must correct $s|_{I_0}$.
2.  **Secrecy:** $Syn(s|_{I_0})$ must be uncorrelated with $s|_{I_1}$.
This is a **Wiretap Channel** problem. If the parity check matrix $H$ has rows that sum bits from both $I_0$ and $I_1$, the syndrome bit leaks linear combinations of the unknown set to Bob.
*Dependency:* This requires the Error Rate $e$ from Phase II to select the correct Code Rate.

**Implementation Assumptions:**
1.  **Matrix Construction:** Requires an external library (or pre-computed matrices) to generate **Multi-Edge Type (MET) LDPC** matrices.
2.  **Sifting-Awareness:** The reconciliation code must accept the sifting mask $x$ as input.
3.  **GF(2) Algebra:** Efficient sparse matrix multiplication in Python/Numpy.

**Testing Protocol:**
1.  **Graph Analysis:** Visualize the Tanner Graph of the LDPC matrix. Verify that variable nodes in $I_1$ are not connected to check nodes that provide high information about $I_0$.
2.  **Information Leakage Test:**
    *   Generate random strings $s$.
    *   Compute Syndrome $S = H \cdot s$.
    *   Give Bob $S$ and $s|_{I_0}$ (perfect knowledge of known bits).
    *   Ask Bob (solver) to guess $s|_{I_1}$.
    *   The conditional entropy $H(s|_{I_1} | S, s|_{I_0})$ must remain maximal.
3.  **Throughput Test:** Measure the decoding speed (bits/sec) vs. Leakage.

**Additional goal definition and literature**

1.  **Construct a "Dual-Graph" MET-LDPC Code.**
    The objective is to algorithmically generate a Parity Check Matrix $H$ that satisfies two conditions simultaneously: (1) it corrects errors on the subgraph defined by $I_0$ (matching bases) efficiently, and (2) the projection of $H$ onto the subgraph $I_1$ (mismatched bases) yields a syndrome that is statistically independent of the bits in $I_1$. This effectively models the reconciliation step as a **Wiretap Channel** where the "wiretapper" (Bob) has partial knowledge.
2.  **Literature:**
    *   **Foundational Wiretap:** *Wyner, A. D.* "The wire-tap channel." *Bell System Technical Journal* (1975). The theoretical basis for Secrecy Capacity.
    *   **Blind Reconciliation:** *Elkouss, D., & Stojanovic, A.* "Blind Reconciliation." (Search for works combining Slepian-Wolf coding with privacy amplification).
    *   **Constructive LDPC:** *Richardson, T. J., Shokrollahi, M. A., & Urbanke, R. L.* "Design of capacity-approaching irregular low-density parity-check codes." *IEEE Transactions on Information Theory* (2001).
    *   **Zero-Leakage Coding:** *Ye, C., & Narayanan, K. R.* "Secret Key Generation from Vector Gaussian Sources." Discusses coding schemes that explicitly minimize leakage to an eavesdropper.

---

## Phase IV: Statistical Rigor & Composability
**Objective:** Move from "Average Case" simulations to "Worst Case" guarantees required for certification.

### 4. Goal 4.1: Finite-Key Analysis
**Context & Insight:**
In simulation, we might run $N=10^5$ bits. Asymptotic theories ($N \to \infty$) assume error distributions are perfectly Gaussian. In finite usage, "lucky" fluctuations happen.
We must calculate the **Finite-Key Penalty ($\Delta$)**. The final key length is $\ell = N(1 - h(e)) - leak_{EC} - \Delta(N, \epsilon)$.
*Dependency:* Requires the Reconciliation Leakage value ($leak_{EC}$) from Phase III.

**Implementation Assumptions:**
1.  **Raw Bit Access:** The simulation must log raw bitstreams, not just summary statistics.
2.  **Renner Bounds:** Implementation of the smooth min-entropy formulas (e.g., using `scipy.stats`).

**Testing Protocol:**
1.  **Monte Carlo Simulation:** Run the E-HOK pipeline 10,000 times with $N=10^4$.
2.  **Failure Analysis:** Count how many times Bob possesses more information than the theoretical $\epsilon$ predicts.
3.  **Calibration:** Adjust the penalty parameter $\Delta$ until the empirical failure rate matches the target $\epsilon_{sec}$ (e.g., $10^{-10}$).

**Additional goal definition and literature**

1.  **Derive One-Sided Smooth Min-Entropy Bounds.**
    Standard finite-key analysis (Tomamichel) ensures *Eve* knows nothing. In E-HOK, we need a bound ensuring *Bob* knows nothing about $I_1$. The goal is to calculate the **penalty term** $\Delta$ that must be subtracted from the raw key length $N$ to account for statistical fluctuations in Bob's memory, such that $H_{min}^\epsilon(K_{I_1} | Bob) \approx \text{length}(K_{I_1})$.
2.  **Literature:**
    *   **The Bible of Finite-Key:** *Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R.* "Tight Finite-Key Analysis for Quantum Cryptography." *Nature Communications* (2012).
    *   **Smooth Min-Entropy:** *Renner, R., & Wolf, S.* "Smooth Renyi Entropy and Applications." *ISIT* (2004).
    *   **Oblivious Transfer Specifics:** *Damgård, I., Fehr, S., Lunghi, L., Salvail, L., & Schaffner, C.* "Improving the Security of Quantum Protocols via Commit-and-Open." *Crypto* (2014). Addresses finite-key effects specifically in the context of commitment and OT.

### 5. Goal 3.1: UC Composability (OT Extension)
**Context & Insight:**
E-HOK is useless on its own; it seeds OT Extension. **Universal Composability (UC)** asks: "Is the key generated by E-HOK indistinguishable from a random oracle to the OT Extension protocol?"
If E-HOK has subtle correlations (e.g., adjacent bits are correlated due to laser memory), the OT Extension might break.

**Implementation Assumptions:**
1.  **Interface Mock:** Create a wrapper that exposes E-HOK as a "Base OT Function."
2.  **IKNP03 Simulation:** Implement a lightweight version of the IKNP03 OT Extension protocol in Python.

**Testing Protocol:**
1.  **End-to-End Run:**
    *   Step A: Run E-HOK to get 128 Base OTs.
    *   Step B: Use these 128 OTs to generate $10^6$ Extended OTs via IKNP03.
2.  **Distinguisher Test:** Compare the statistical distribution of the $10^6$ Extended OTs against a control group generated from a true random seed.
3.  **Success Metric:** The Kolmogorov-Smirnov distance between the two distributions should be $<\epsilon$.

**Additional goal definition and literature**

1.  **Establish Universal Composability (UC) Bounds for Hybrid OT.**
    The goal is to formally prove that replacing the "Base OTs" in protocols like IKNP03 with "Quantum-generated OTs" maintains the security of the superstructure. Specifically, determine the required **Key Refresh Rate**: how many quantum OTs must be generated per second to sustain a specific classical output rate (e.g., 1 Gbit/s) without violating the $\epsilon_{UC}$ security parameter.
2.  **Literature:**
    *   **The Standard:** *Ishai, Y., Kilian, J., Nissim, K., & Petrank, E.* "Extending Oblivious Transfers Efficiently." (IKNP03). *Crypto* (2003).
    *   **Silent OT:** *Boyle, E., et al.* "Efficient Two-Round OT Extension with Silent Preprocessing." *CCS* (2019). This is critical for quantum links because it moves the communication burden to local computation, fitting the "low bandwidth, high latency" profile of quantum links.
    *   **UC Framework:** *Canetti, R.* "Universally Composable Security: A New Paradigm for Cryptographic Protocols." *FOCS* (2001).
    *   **Quantum-Classical Hybrid:** *Unruh, D.* "Universally Composable Quantum Multi-Party Computation." *Eurocrypt* (2010). Addresses the specific issues of composing quantum subroutines with classical protocols.

---

## Phase V: Advanced Architecture (Optional/Future)

### 6. Goal 4.2: MDI Architecture
**Context & Insight:**
To remove detector hacking risks, we introduce a central node (Charlie). Alice and Bob send states to Charlie.
This changes the SquidASM topology from 2-node to 3-node (Star topology).

**Implementation Assumptions:**
1.  **NetSquid Topology:** `network_config.yaml` must define 3 stacks.
2.  **BSM Simulation:** Charlie needs a Quantum Processor capable of Bell State Measurements.

**Testing Protocol:**
1.  **Correlation Check:** Verify that Alice and Bob can infer each other's bits based on Charlie's BSM announcement.
2.  **Untrusted Charlie:** Implement a malicious Charlie who reports false BSM results. Verify that Alice and Bob detect the error during the Sampling Phase (Phase II).

**Additional goal definition and literature**

1.  **Architect an MDI-E-HOK Protocol.**
    Design a variation of the protocol where Alice and Bob both transmit to an untrusted Charlie. The goal is to prove that if Charlie performs a Bell State Measurement (BSM) and announces the result, he cannot learn the Oblivious Key, nor can he force Alice and Bob into a state where Bob learns Alice's $I_1$ bits. This decouples security from detector vulnerabilities (blinding attacks).
2.  **Literature:**
    *   **Seminal MDI-QKD:** *Lo, H.-K., Curty, M., & Qi, B.* "Measurement-Device-Independent Quantum Key Distribution." *Physical Review Letters* (2012).
    *   **Trojan Horse Defense:** *Lucamarini, M., et al.* "Practical Security Bounds Against the Trojan-Horse Attack in Quantum Key Distribution." *Physical Review X* (2015).
    *   **MDI-QT (Quantum Transfer):** Look for literature on "Measurement-Device-Independent Quantum Oblivious Transfer." (This is a cutting-edge/sparse field, representing a high-value research gap).
    *   **Untrusted Relays:** *Pirandola, S., et al.* "High-rate measurement-device-independent quantum cryptography." *Nature Photonics* (2015).

---


## Summary of Execution Order

1.  **Phase I (NSM):** Build the physical simulator parameters (Decoherence).
    *   *Output:* "Hardware Requirements Spec" (min $T2$, max latency).
2.  **Phase II (Sampling):** Build the Test/verify logic.
    *   *Output:* "Protocol Logic" (Commit/Challenge flow).
3.  **Phase III (Reconciliation):** Build the MET-LDPC codec.
    *   *Output:* "Encoder/Decoder Module" (Zero-knowledge guarantee).
4.  **Phase IV (Finite-Key):** Run massive statistical validation.
    *   *Output:* "Safety Margin Parameters" (Lookup table for $\Delta$).
5.  **Phase V (Composability):** Integrate with IKNP03.
    *   *Output:* "Application SDK" (OT Extension interface).
# Phase II: Sifting & Estimation

## 1. Executive Summary
Phase II is the "gatekeeper" of the E-HOK protocol. Its objective is to filter the raw quantum data generated in Phase I to distinguish between **passive channel loss** (natural erasures) and **active adversarial filtering** (where a cheating Bob claims to have lost qubits he failed to store successfully).

This phase implements a "Commit-then-Reveal" logic rooted in the **Weak String Erasure (WSE)** primitive. It must strictly enforce a temporal sequence where Bob commits to his detection events *before* gaining the information necessary to evaluate them. Furthermore, this phase moves beyond asymptotic assumptions, applying finite-key statistical bounds to guarantee that the estimated error rate ($e_{obs}$) safely bounds the worst-case error rate ($e_{max}$) on the remaining key.

---

## 2. Key Theoretical and Mathematical Insights

### A. The "Missing Rounds" Constraint (Handling Erasures)
In real-world implementations, photons are lost. A cheating Bob could exploit this by selectively reporting "loss" only on qubits where his noisy memory failed, effectively post-selecting a lower-noise sub-key.
*   **Theoretical Basis:** *Schaffner et al.* define the protocol for "Weak String Erasure with Errors." A critical security requirement is that Bob must report the set of indices $\mathcal{M}$ (missing rounds) *before* he knows the basis information.
*   **Mathematical Insight:** Alice must validate that the number of reported missing rounds aligns with the expected channel transmittance parameters established in Phase I. This is validated using **Chernoff’s inequality** to define a secure interval.
    *   The number of valid rounds $S$ must satisfy:
        $$ \text{Prob}[|S - P_{expected} \cdot M| \ge \zeta \cdot M] < \varepsilon $$
        Where $\zeta = \sqrt{\ln(2/\varepsilon)/(2M)}$ defines the statistical tolerance interval.
        See Eq. (1) and (2) in *Schaffner et al.* [2].

### B. Statistical Confidence in Finite Sets (The $\mu$ Parameter)
The protocol cannot assume the error rate observed on the test set is identical to the error rate of the secret key.
*   **Theoretical Basis:** *Erven et al.* apply the **Smooth Min-Entropy** framework to account for finite-size effects. They derive a penalty term $\mu$ that represents the statistical fluctuation probable between the sample set (size $k$) and the remaining key (size $n$).
*   **Mathematical Insight:** The protocol must not use the raw observed error rate $Q_{tol}$ for privacy amplification. Instead, it must use an adjusted rate $Q_{tol} + \mu$:
    $$ \mu := \sqrt{\frac{n + k}{n k} \frac{k + 1}{k}} \ln \frac{4}{\varepsilon_{sec}} $$
    This term $\mu$ must be added to the channel error when calculating the secure key length $\ell$.
    See Theorem 2 and Eq. (2) in *Erven et al.* [6].

### C. Decoy State Statistics
To protect against Photon-Number-Splitting (PNS) attacks without discarding all multi-photon events, the protocol relies on modulating source intensity.
*   **Theoretical Basis:** *Schaffner et al.* (Protocol 2) and *Erven et al.* describe using decoy states to place a tighter bound on the parameter $r^{(1)}$ (the fraction of single-photon rounds Bob reports as missing).
*   **Insight:** By comparing yield statistics ($Q_{yield}$) between Signal ($\mu$) and Decoy ($\nu$) states, Alice can mathematically prove if Bob is selectively suppressing single-photon events to hide his storage inefficiencies.

---

## 3. Key Implementation Requirements

To satisfy the theoretical security definitions, the implementation must adhere to a strict temporal order and utilize dynamic statistical bounding.

### Requirement 1: The "Sandwich" Protocol Flow (Strict Ordering)
The security of E-HOK collapses if Bob learns the bases before committing to his detection events. The implementation must enforce this specific sequence:
1.  **Quantum Transmission:** Alice sends qubits; Bob measures or stores.
2.  **Reporting Phase:** Bob sends the set of indices $\mathcal{M}$ (Missing Rounds) to Alice.
    *   *Check:* Alice verifies $|\mathcal{M}|$ against expected loss (Insight A). If out of bounds $\rightarrow$ **Abort**.
3.  **Wait/Commitment:** The physical wait time $\Delta t$ elapses (Phase I requirement).
4.  **Basis Reveal:** Alice sends basis string $a$.
5.  **Sifting:** Bob computes $I_0$ (match) and $I_1$ (mismatch).
6.  **Sampling/Challenge:** Alice requests a random subset $k$ from $I_0$. Bob reveals values.
    *   *Check:* Alice calculates QBER on $k$.

### Requirement 2: Dynamic "Pre-Processing" Calculation
The software stack must compute the secure key parameters *dynamically* based on the actual number of photons received, not theoretical maximums.
*   **Action:** Upon receiving Bob's report $\mathcal{M}$, the system calculates the actual block sizes $n$ (raw key) and $k$ (parameter estimation bits).
*   **Logic:**
    1.  Compute the statistical penalty $\mu$ using the formula from *Erven et al.* (Insight B).
    2.  Update the maximum tolerable QBER ($Q_{tol}$) using this $\mu$.
    3.  If the observed QBER ($e_{obs} + \mu$) > Hard Limit (22%), **Abort**.

### Requirement 3: Decoy State Validator (Optional but Recommended)
To implement the "Robust" variant of E-HOK:
*   **Action:** The quantum generator in Phase I must interleave pulses with intensity $\mu$ (signal) and $\nu$ (decoy).
*   **Logic:** In Phase II, Alice must separate statistics for $\mu$-pulses and $\nu$-pulses. She must verify that the yield $Y_\mu$ and $Y_\nu$ follow the expected physical distribution. If $Y_\nu$ is anomalously low compared to $Y_\mu$, it indicates Bob is blocking multi-photon pulses to mask a PNS attack.

---

## 4. Strengths and Critical Aspects

### Strengths
1.  **Finite-Size Security:** By adopting the analysis from *Erven et al.*, the protocol provides a concrete security parameter $\varepsilon_{sec}$ (e.g., $10^{-10}$) for real-world block sizes ($N \approx 10^5 - 10^7$), rather than relying on asymptotic proofs that only hold for $N \to \infty$.
2.  **Active Cheating Detection:** The "Missing Rounds" reporting mechanism explicitly prevents the most common storage-attack strategy (where Bob blames his poor memory coherence on channel loss).
3.  **Modularity:** The Sifting phase effectively decouples the physical transmission errors from the adversarial storage errors, allowing the protocol to handle trusted device noise up to specific limits defined in *Lupo et al.* [3].

### Critical Aspects (Failure Modes)
1.  **The "A Priori" Dilemma:** The validation of Missing Rounds (Insight A) requires Alice to know the channel's transmission probability $P_{trans}$ *before* the protocol starts.
    *   *Risk:* If the real channel fluctuates (e.g., a fiber is bent or atmospheric turbulence changes), honest Bob's yield will drop. Alice's Chernoff bound check will fail, causing a false positive abort. The protocol requires highly stable channels or real-time calibration.
2.  **Sample Size Sensitivity:** The penalty term $\mu$ scales inversely with the square root of the sample size $\sqrt{k}$. If the test set $k$ is too small (to save bits for the key), $\mu$ becomes large, drastically reducing the tolerable QBER and potentially rendering the key insecure.
3.  **Side-Channel in Reporting:** If the classical channel used for reporting $\mathcal{M}$ has significant latency, and Alice sends bases *asynchronously* based on a local clock, a race condition could occur where Bob receives bases before his report is fully registered, allowing him to cheat. Acknowledgments must be synchronous.

---

## 5. References
*   [1] `Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation.md` (Lemus et al., 2020)
*   [2] `ROBUST CRYPTOGRAPHY IN THE NOISY-QUANTUM-STORAGE MODEL.md` (Schaffner et al., 2009)
*   [3] `Error-tolerant oblivious transfer in the noisy-storage model.md` (Lupo et al., 2023)
*   [6] `An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model.md` (Erven et al., 2014)
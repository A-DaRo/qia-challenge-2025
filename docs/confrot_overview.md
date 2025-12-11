
This is a high-level confrontation of the three foundational texts for the E-HOK project.

*   **[König 2012]** provides the **Physical Law** (Noisy Storage) that enables OT.
*   **[Lemus 2025]** provides the **Protocol Architecture** (Commit-Verify) and experimental reality check.
*   **[Tomamichel 2012]** provides the **Statistical Ruler** (Finite-Key Analysis) to measure security.

The friction between these three defines the valid operating range of E-HOK.

---

# 1. In-Depth Confrontation: The Trilemma

The central conflict lies in the **Source of Security** and the **Tolerance to Error**.

### A. The Adversary Model Clash (Phase I)
*   **König [3]:** Assumes the adversary (Bob) has **unbounded computational power** but **imperfect quantum storage**.
    *   *Math:* Security holds if $C_N \cdot \nu < 1/2$.
    *   *Promise:* Everlasting Security. Even if P=NP, Bob cannot decrypt past OTs because the quantum state has decohered.
*   **Lemus [12]:** Assumes the adversary has **perfect quantum storage** (potentially) but **bounded computational power**.
    *   *Math:* Security holds if $H$ is collision-resistant (Computational).
    *   *Critique:* Lemus explicitly rejects König’s premise, arguing that technological advances in quantum memory will inevitably break NSM-based protocols.
*   **Tomamichel [7]:** Agnostic to the adversary's nature (Eve), but focuses on **Information Theoretic** security derived from the Uncertainty Principle.
    *   *Role:* Provides the bridging math. Whether the entropy comes from Storage Noise (König) or Computational Hashing (Lemus), Tomamichel provides the calculus to distill the final key.

### B. The Error Rate (QBER) Bottleneck (Phase IV)
*   **Tomamichel [7]:** In standard QKD, positive keys are possible up to $Q \approx 11\%$ (for $N \to \infty$).
*   **König [3]:** Security depends on the *gap* between Channel Capacity ($C_N$) and Storage Rate. If storage noise is high ($r \ll 1$), the protocol can tolerate moderate transmission errors.
*   **Lemus [12]:** The specific **BBCS92** logic collapses rapidly under noise.
    *   *Math:* Experimental threshold is **$Q_{max} \approx 1.14\%$**. Theoretical max is $\approx 2.8\%$.
    *   *Confrontation:* If E-HOK uses the protocol flow from Lemus but the channel noise from a standard Tomamichel QKD simulation ($Q \approx 5\%$), the system will **fail to generate any key**.

### C. The Finite-Size Penalty (Phase IV)
*   **König [3]:** Provides asymptotic bounds ($N \to \infty$) for existence proofs. Shows $\ell \approx -\log P_{succ}$.
*   **Lemus [12]:** Adapts bounds for finite size but relies on specific experimental parameters ($\delta_1, \delta_2$).
*   **Tomamichel [7]:** Provides the **exact penalty term**.
    *   *Math:* $\ell \le n [q - h(Q + \mu)]$.
    *   *Synthesis:* E-HOK must apply Tomamichel's $\mu$ (sampling penalty) to König's $C_N$ (storage capacity).

---

# 2. Strategic Integration into E-HOK Specifications

The E-HOK architecture must synthesize these three papers. One cannot simply "pick one"; they must be layered.

## Phase I: Physical Foundations
**Objective:** Define the Physical Security Guarantee.

*   **Retain from [König 2012]:** The **Noisy Storage Inequality**.
    *   **Requirement:** E-HOK must enforce the "Wait Time" $\Delta t$.
    *   **Math:** The security condition is **$C_N(\Delta t) \cdot \nu < 1/2$**.
    *   **Strength Achieved:** **Everlasting Security** (Information Theoretic) against quantum memory attacks.
*   **Retain from [Lemus 2025]:** The **Hybrid Fallback**.
    *   **Logic:** If the "Decoherence Sweep" reveals Bob's memory is too good (violating König), the system must degrade gracefully to Computational Security (using Commitments) rather than failing.
    *   **Strength Achieved:** **Forward Security** (Computational).

## Phase II: Core Protocol Logic (Sifting)
**Objective:** Prevent Bob from cheating during measurement.

*   **Retain from [Lemus 2025]:** The **Commit-Verify Pattern**.
    *   *Critical:* König’s paper assumes a generic reduction to coding. Lemus provides the actual steps: Bob measures $\to$ Bob Commits (Hash) $\to$ Alice Challenges $\to$ Bob Opens.
    *   *Action:* E-HOK must implement the **BLAKE3-based commitment** described in Lemus. Without this, König’s physical assumptions are useless because Bob can cheat classically by changing his reported basis.
*   **Retain from [Tomamichel 2012]:** **Asymmetric Basis Choice**.
    *   *Optimization:* Lemus uses symmetric (50/50) bases. Tomamichel proves this is inefficient. E-HOK should use $p_X \approx 0.9$ (for Key) and $p_Z \approx 0.1$ (for Estimation) to maximize yield, applying Tomamichel’s sifting logic to the Lemus protocol structure.

## Phase III: Reconciliation
**Objective:** Correct errors without leaking the Oblivious Key.

*   **Retain from [König 2012]:** The **Wiretap Constraint**.
    *   *Logic:* Reconciliation must be treated as a Wiretap Channel where Bob's view of the *other* key ($I_1$) is the wiretapper's view.
*   **Retain from [Lemus 2025]:** **Verifiability**.
    *   *Requirement:* Bob must prove he successfully reconciled $I_0$ (via a hash of the syndrome/key) before Alice proceeds. This prevents "Selective Failure" attacks where Bob aborts only when he has low information.

## Phase IV: Statistical Rigor
**Objective:** Calculate the exact safe key length $\ell$.

*   **Retain from [Tomamichel 2012]:** The **Master Equation**.
    *   E-HOK must use the Tomamichel formula for $\ell$, but substitute the entropic source.
    *   **Synthesized Formula:**
        $$ \ell \le n \cdot \underbrace{[ -\log P_{succ}^{\text{NSM}}(n) ]}_{\text{König (Storage Capacity)}} - \underbrace{\text{leak}_{EC}}_{\text{Martinez-Mateo}} - \underbrace{n \cdot h(Q + \mu) + \Delta_{finite}}_{\text{Tomamichel (Sampling Penalty)}} $$
*   **Retain from [Lemus 2025]:** The **Multi-Photon Penalty**.
    *   *Correction:* Tomamichel assumes single photons ($q=1$). Real lasers are Poissonian.
    *   *Math:* E-HOK must subtract the term **$P_{multi}(1-\alpha)N_0$** (from Lemus Eq. 38) from the final key length to account for multi-photon side channels.
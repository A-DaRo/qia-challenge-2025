# Phase III: Information Reconciliation

## 1. Executive Summary
Phase III is the most perilous stage for the protocol's efficiency. Its goal is to correct errors in the honest key (the bits Bob actually measured, $I_0$) while minimizing the information leaked to a cheating Bob about the bits he missed ($I_1$).

Unlike standard QKD, where interactivity is encouraged to optimize efficiency, E-HOK operates under a strict constraint: **Interactive protocols like Cascade are forbidden** in their standard form because they leak Bob's basis choices. The protocol must instead rely on **One-Way Forward Error Correction (FEC)** or specialized "Blind" protocols. Furthermore, every bit of error-correction information (syndrome) sent by Alice must be treated as a direct reduction in the final secure key length, a concept known as the **Wiretap Cost**.

---

## 2. Key Theoretical and Mathematical Insights

### A. The Ban on Interactivity (Protecting Obliviousness)
Standard reconciliation protocols involve bidirectional discussion (e.g., "Do we match on this block?"). In E-HOK, if Alice learns where Bob's errors are, she might deduce his basis choice $C$.
*   **Theoretical Basis:** *Erven et al.* explicitly state that interactive algorithms like Cascade cannot be used because the interaction would reveal Bob's choice bit.
*   **Constraint:** The reconciliation must be **One-Way** from Alice to Bob (or use a blinded interactive method).
*   **Impact:** If Bob fails to decode a block using the provided syndrome, he cannot ask for specific help on that block without risking the security of his choice bit.

### B. The "Wiretap Cost" (Syndrome Leakage Penalty)
Since Alice does not know which bits Bob holds ($I_0$) and which he missed ($I_1$), she must send syndromes covering the entire raw string (or use Interactive Hashing to narrow it down). A cheating Bob can use these syndromes to correct errors in his *noisy storage* of the unknown bits.
*   **Theoretical Basis:** *Schaffner et al.* and *Lupo et al.* formalize this leakage. The length of the syndrome $|\Sigma|$ must be subtracted from the min-entropy of the raw key to determine the final secure length.
*   **Mathematical Insight:** The secure key length $\ell$ is bounded by:
    $$ \ell \le H_{min}(X|E) - |\Sigma| - \text{security\_margins} $$
    See Eq. (3) in *Lupo et al.* [3] and Eq. (8) in *Erven et al.* [6].
    This implies that if the error rate is high (requiring a long syndrome), the term $|\Sigma|$ dominates, and the secure key length $\ell$ drops to zero.

### C. Blind Reconciliation (Efficiency via Adaptivity)
To mitigate the inefficiency of fixed-rate codes, the protocol can use "Blind Reconciliation," where the code rate adapts to the channel without prior estimation.
*   **Theoretical Basis:** *Martinez-Mateo et al.* propose a protocol where Alice sends a minimal syndrome first. If decoding fails, she sends incremental parity bits (puncturing).
*   **Mathematical Insight:** The efficiency of this method is high, but in the context of E-HOK, the total length of all sent increments must be tracked.
    $$ |\Sigma_{total}| = \sum |\text{increment}_i| $$
    This total length is what must be subtracted from the key entropy (Insight B).

---

## 3. Key Implementation Requirements

The implementation must balance the need for error correction with the strict requirement to cap information leakage.

### Requirement 1: One-Way LDPC Implementation
The baseline implementation must use **Low-Density Parity-Check (LDPC)** codes in a one-way configuration.
*   **Action:**
    1.  Alice computes the syndrome $S = H \cdot X$ for her entire string $X$.
    2.  Alice sends $S$ to Bob.
    3.  Bob uses his basis matching information ($I_0$) to identify which parts of $X$ he should have. He attempts to decode *only* those parts using $S$.
*   **Constraint:** Alice must *not* wait for Bob's confirmation for individual blocks. The flow is unidirectional.

### Requirement 2: The "Safety Cap" (Leakage Accumulator)
To prevent a "feigned failure" attack where a cheating Bob forces Alice to reveal the whole key by pretending he cannot decode, the protocol must enforce a hard limit on syndrome length.
*   **Action:**
    1.  Calculate the **Maximum Safe Leakage** $L_{max}$ using the parameters from Phase I/II:
        $$ L_{max} = H_{min}(X|E) - \ell_{target} $$
    2.  If using an iterative scheme (Blind Reconciliation), maintain a counter `CurrentSyndromeLength`.
    3.  **Abort Condition:** If `CurrentSyndromeLength` exceeds $L_{max}$, Alice must stop sending parity bits immediately. The protocol aborts or discards that block.

### Requirement 3: "Send-All" vs. "Interactive Hashing" Selector
The system should support two modes for handling the "unknown set" problem:
*   **Mode A (Baseline - "Send-All"):** Alice sends syndromes for the entire raw string $X$.
    *   *Pros:* Simple to implement.
    *   *Cons:* Highly inefficient. Half the syndrome bits are "wasted" on $I_1$ (the bits Bob shouldn't know), but they still count as leakage.
*   **Mode B (Advanced - Interactive Hashing):** Bob uses **Interactive Hashing** (as described in *Wehner et al.* [4]) to commit to his set $I_0$ without revealing it fully. Alice learns two possible sets $\{W_0, W_1\}$ and sends syndromes for only those two.
    *   *Pros:* Reduces leakage significantly.
    *   *Cons:* Requires complex, multi-round interaction. Recommended only for high-performance variants.

---

## 4. Strengths and Critical Aspects

### Strengths
1.  **Information-Theoretic Security:** By treating the syndrome as fully leaked information (subtracting $|\Sigma|$ from the key), the protocol maintains security even if the adversary has infinite computing power to analyze the syndrome.
2.  **Adaptability:** The inclusion of **Blind Reconciliation** (*Martinez-Mateo et al.*) allows the protocol to function efficiently across a range of error rates without needing precise *a priori* channel estimation (provided the "Safety Cap" is respected).
3.  **Hardware Friendliness:** LDPC codes are highly parallelizable and suitable for FPGA implementation, addressing the scalability goals of E-HOK.

### Critical Aspects (Failure Modes)
1.  **The Efficiency Cliff:** If the Quantum Bit Error Rate (QBER) is high (> 5-10%), the size of the syndrome $|\Sigma|$ required for LDPC decoding grows rapidly. Since $|\Sigma|$ is subtracted from the secure key, the effective key rate $\ell$ hits zero much faster than in QKD. This makes E-HOK very sensitive to channel noise.
2.  **Interactive Hashing Complexity:** While Interactive Hashing improves the theoretical rate, implementing it securely is difficult. A flawed implementation could leak Bob's choice bit $C$, completely breaking the "Oblivious" property.
3.  **Finite Block Size Effects:** LDPC codes require large block sizes ($10^4 - 10^5$ bits) to reach Shannon capacity. If the E-HOK experiment is run with short bursts of photons, the error correction will be inefficient (large $|\Sigma|$), killing the secure key rate.

---

## 5. References
*   [2] `ROBUST CRYPTOGRAPHY IN THE NOISY-QUANTUM-STORAGE MODEL.md` (Schaffner et al., 2009)
*   [3] `Error-tolerant oblivious transfer in the noisy-storage model.md` (Lupo et al., 2023)
*   [4] `Unconditional-security-from-noisy-quantum-storage.md` (König et al., 2012)
*   [6] `An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model.md` (Erven et al., 2014)
*   [7] `Blind Reconciliation.md` (Martinez-Mateo et al., 2011)
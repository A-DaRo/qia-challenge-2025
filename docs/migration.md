# Migration Concepts: From Standard QKD to E-HOK
**Transition Guide for SquidASM Implementation**

This document outlines the fundamental conceptual shifts required when migrating from the "Standard QKD" (BB84/Cascade) hackathon challenge to the "Entanglement-based Hybrid Oblivious Key" (E-HOK) research task.

---

## 1. The Trust Model Shift
**From Mutual Cooperation to Mutual Distrust**

*   **Old Concept (QKD):** Alice and Bob are trusted partners. They collaborate to eliminate an external eavesdropper (Eve). All leakage between Alice and Bob (e.g., during Cascade) is acceptable as long as it is quantified and subtracted.
*   **New Concept (E-HOK):** Alice and Bob are **adversaries**.
    *   **Alice's Goal:** Prevent Bob from learning bits where bases mismatched ($I_1$).
    *   **Bob's Goal:** Prevent Alice from knowing which bases he chose ($x$).
    *   **Implication:** Leakage is no longer just "lost key rate"; it is a **security breach**. Interactive protocols like Cascade are dangerous because Bob can craft parity requests to probe Alice's secret bits.

## 2. The Physical Layer Shift
**From Prepare-and-Measure to Entanglement-based**

*   **Old Concept (P&M):** Alice generates random bits in Python (`random.randint`) and encodes them into qubits (`q.H()`, `q.X()`).
*   **New Concept (E-HOK):** Randomness is **intrinsic to the quantum measurement**.
    *   Alice requests `EPRSocket.create_keep()`.
    *   Alice measures her half. The *outcome* of this measurement becomes her bit string $s$.
    *   **Why:** This maps directly to the BBM92 protocol structure, which is the entanglement equivalent of the HOK protocol, avoiding the overhead of simulating teleportation.

## 3. The Reconciliation Shift
**From Interactive Parity to Blind Syndrome**

*   **Old Concept (Cascade):** Bidirectional, many round-trips. Alice and Bob "chat" to fix errors.
*   **New Concept (LDPC):** Unidirectional. Alice sends a syndrome $Syn = H \cdot s|_{I_0}$.
    *   **Sifting-Aware:** Reconciliation must happen *strictly* on the subset of bits where bases match ($I_0$). Attempting to reconcile the whole string before sifting (to hide the basis info) risks leaking information about the unknown set ($I_1$) via the syndrome.
    *   **Efficiency:** We move from CPU-heavy recursion (Cascade) to Matrix-vector operations (LDPC).

## 4. The Classical Security Shift
**From Authentication to Commitment**

*   **Old Concept (Authenticated Channel):** Prevents Man-in-the-Middle (Eve). It ensures Alice is talking to Bob.
*   **New Concept (Commitment Scheme):** Prevents **Bob** from cheating.
    *   **The Problem:** Bob must not wait to hear Alice's bases before deciding his own measurement claims.
    *   **The Solution:** Bob must **Commit** to his measurement outcomes (via Hashing/Merkle Trees) *before* Alice reveals her basis string. The `AuthenticatedSocket` is insufficient; the protocol flow must enforce: `Measure` $\to$ `Commit` $\to$ `Reveal Bases` $\to$ `Open Commitment`.

## Summary of Component Migration

| Component | Standard QKD Implementation | E-HOK Implementation |
| :--- | :--- | :--- |
| **Random Source** | Python RNG (`secrets` module) | Quantum Measurement (`q.measure()`) |
| **Topology** | Alice prepares, Bob measures | NetworkStack generates EPR, both measure |
| **Error Correction** | Cascade (Interactive) | LDPC (One-way Syndrome) |
| **Privacy Amp.** | Toeplitz (Remove Eve) | Toeplitz/Trevisan (Remove Bob's partial info) |
| **Classical Flow** | Auth $\to$ Sift $\to$ Reconcile | Measure $\to$ **Commit** $\to$ Sift $\to$ Reconcile |

## Critical Implementation Warning
**Do not reuse the Cascade Reconciliator.**
While it is tempting to reuse the existing Cascade module for the $I_0$ subset, its interactive nature makes it vulnerable to timing attacks and side-channel leakage in an adversarial setting. The migration to LDPC is not just for efficiency; it is a security requirement for Oblivious Transfer.
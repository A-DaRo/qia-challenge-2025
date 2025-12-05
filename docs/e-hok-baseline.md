# Baseline E-HOK: Implementation Plan & Requirements

This document defines the **Minimum Viable Protocol (MVP)** for the Entanglement-based Hybrid Oblivious Key (E-HOK) system. It establishes the foundational software architecture and functional requirements necessary to support the advanced "Industrial R&D Master Plan."

**Objective:** To implement a functionally complete E-HOK pipeline in SquidASM that successfully generates Oblivious Keys under ideal or simple noise conditions, featuring modular interfaces that allow for the "hot-swapping" of advanced components (e.g., MET-LDPC, NSM) in future research phases.

---

## 1. Architectural Requirements

The Baseline E-HOK must be built on a modular "Manager/Worker" architecture to separate high-level protocol logic from low-level quantum operations.

### 1.1. Network Topology (SquidASM)
*   **Nodes:** Two nodes (Alice, Bob) connected via:
    *   **Quantum Link:** Fidelity > 0.95 (Depolarizing noise).
    *   **Classical Link:** Authenticated, reliable channel.
*   **Hardware Config (`QDevice`):**
    *   Minimum 5 qubits per node.
    *   **Must-Have:** A **Batching Manager** to handle streaming operations. The system must process $N=10,000$ bits using a 5-qubit memory by cycling through create/measure loops.

### 1.2. Modular Interfaces (Abstract Base Classes)
To support future R&D, the implementation **must** define strict interfaces:
*   `ICommitmentScheme`: Defines `commit(data)` and `open(proof)`.
    *   *Baseline implementation:* SHA-256 Hash.
*   `IReconciliator`: Defines `reconcile(sifted_key, error_rate)`.
    *   *Baseline implementation:* Standard LDPC (using `scipy.sparse`).
*   `IPrivacyAmplifier`: Defines `compress(key, seed)`.
    *   *Baseline implementation:* Standard Toeplitz (Numpy matrix multiplication).

---

## 2. Functional Phases (The "Must-Have" Workflow)

The protocol execution follows a strict sequential order to maintain security.

### Phase I: Quantum Generation (The Raw Material)
**Requirement:** Generate raw correlated bitstrings where Alice knows $s$ and Bob knows $\bar{s}$.
*   **Mechanism:** Entanglement-based (EPR).
*   **Operation:**
    1.  Alice and Bob negotiate a batch size (e.g., 100).
    2.  They execute `create_keep` (or `create_measure`) for that batch.
    3.  **Local Randomness:** Both parties independently choose basis $a_i, \bar{a}_i \in \{0, 1\}$ (Z/X).
    4.  **Measurement:** Immediate measurement.
    5.  **Buffering:** Results are stored in a local classical buffer `List[Tuple[Bit, Basis]]`.
*   **Output:** Raw classical arrays $s, a$ (Alice) and $\bar{s}, \bar{a}$ (Bob).

### Phase II: Commitment (The Trust Anchor)
**Requirement:** Bob must irrevocably bind himself to his measurement outcomes *before* learning Alice's bases.
*   **Logic:**
    1.  Bob concatenates his measurements and basis choices.
    2.  Bob computes $H = \text{SHA256}(\bar{s} || \bar{a})$.
    3.  Bob sends $H$ to Alice via `AuthenticatedSocket`.
    4.  **Blocking:** Alice's program **must pause** and wait for this message before proceeding.

### Phase III: Sifting & Sampling (The Verification)
**Requirement:** Reveal bases, identify matching indices ($I_0$), and estimate Error Rate ($e$).
*   **Logic:**
    1.  Alice sends her basis string $a$.
    2.  Bob identifies $I_0 = \{i : a_i = \bar{a}_i\}$ and $I_1 = \{i : a_i \neq \bar{a}_i\}$.
    3.  **Sampling:**
        *   Alice deterministically generates a random subset $T \subset I_0$ (Test Set) using a shared seed or transmitted list.
        *   Alice requests Bob to open the values in $T$.
    4.  **Verification:**
        *   Bob sends $\bar{s}|_T$ and the de-commitment parameters (salt/nonce).
        *   Alice verifies the values match the Commitment $H$.
        *   Alice calculates QBER on $T$. If QBER > Threshold (e.g., 11%), **ABORT**.

### Phase IV: Information Reconciliation (The Correction)
**Requirement:** Correct errors in $I_0$ using a non-interactive (or minimal) syndrome.
*   **Logic:**
    1.  Alice generates a Parity Check Matrix $M$ (loaded from file or generated).
    2.  Alice computes Syndrome $S = M \cdot s|_{I_0}$.
    3.  Alice sends $S$ to Bob.
    4.  Bob decodes using Belief Propagation to obtain corrected string $s'|_{I_0}$.
    5.  **Hash Check:** A simple hash of the corrected key is exchanged to confirm success.

### Phase V: Privacy Amplification (The Distillation)
**Requirement:** Reduce the key length to account for information leaked to Eve (QBER) and to Bob (Reconciliation).
*   **Logic:**
    1.  Alice generates a random seed for a Toeplitz matrix.
    2.  Both parties apply the matrix to their keys.
    3.  **Output:** Alice outputs $(K_{final}, \text{None})$. Bob outputs $(K_{final}, x)$, where $x$ is the mask indicating which bits he originally knew vs. unknown.

---

## 3. Data Structures & State Management

The baseline must implement specific data structures to handle the "Oblivious" nature of the key.

### 3.1. The `ObliviousKey` Object
Instead of just returning a byte array, the protocol must return a structured object:
```python
@dataclass
class ObliviousKey:
    key_value: np.ndarray      # The final bitstring
    knowledge_mask: np.ndarray # Array of 0s and 1s (0=Known, 1=Unknown)
    security_param: float      # The estimated epsilon
```
*   **Alice:** `knowledge_mask` is all 0s (she knows everything).
*   **Bob:** `knowledge_mask` maps to the $I_1$ set.

### 3.2. Simulation State Safety
To prevent "Time-Travel" debugging issues (where a user accidentally inspects a future state):
*   **Encapsulation:** The raw quantum results must be private members of the Protocol class, accessible only via the `commit` and `reveal` methods.

---

## 4. Testing the Baseline

The baseline is considered complete when the following tests pass:

1.  **The "Honest Execution" Test:**
    *   Noise = 0%.
    *   Result: Alice and Bob keys match perfectly. Bob's `knowledge_mask` accurately reflects the ~50% basis mismatch.
2.  **The "Noise Tolerance" Test:**
    *   Noise = 5% (QBER).
    *   Result: LDPC Reconciliation succeeds. Keys match.
3.  **The "Commitment Ordering" Test:**
    *   Modify Alice to send bases *before* receiving commitment.
    *   Result: Bob's program (or the Protocol Manager) throws a `ProtocolError` or `SecurityException`.

---

## 5. Summary of Baseline vs. R&D

| Feature | **Baseline E-HOK (This Plan)** | **Industrial R&D Target** |
| :--- | :--- | :--- |
| **Commitment** | SHA-256 Hash | Noisy Storage (Wait-time) |
| **Reconciliation** | Standard LDPC (scipy) | Blind / MET-LDPC (C++) |
| **Sampling** | Random Subset | Cut-and-Choose / Block |
| **Security** | Computational (Hash) | Physical / Unconditional |
| **Output** | Python Dataclass | OT Extension Seed Interface |

This Baseline plan provides the stable, modular scaffolding required to incrementally implement the advanced theoretical concepts defined in the Master Plan without refactoring the entire simulation stack.
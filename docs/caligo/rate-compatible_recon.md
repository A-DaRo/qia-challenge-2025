# Theoretical Report: Advanced Rate-Compatible Reconciliation in Caligo

**Date:** January 2025
**Subject:** Architectural Definition of Baseline and Blind Reconciliation Protocols
**Context:** $\binom{2}{1}$-Oblivious Transfer via Noisy Storage Model (NSM)
**References:**
1.  Elkouss et al., "Rate Compatible Protocol for Information Reconciliation: An application to QKD" (2010).
2.  Martinez-Mateo et al., "Blind Reconciliation" (2012).
3.  Elkouss et al., "Untainted Puncturing for Irregular Low-Density Parity-Check Codes" (2012).

---

## 1. Introduction: The Entropy Economy

In the Caligo protocol, Phase III (Information Reconciliation) is the critical juncture where error correction meets security. Unlike standard QKD, where syndrome information leaks to a passive eavesdropper, in the Caligo OT protocol, this information leaks to the receiver (Bob), who is a potential adversary. Under the Noisy Storage Model (NSM), minimizing the syndrome length $|\Sigma|$ is paramount to maximizing the extractable secure key length.

This report defines the theoretical implementation of two distinct reconciliation strategies: **Baseline** (Elkouss) and **Blind** (Martinez-Mateo). Both strategies are unified under a single architectural foundation: a **Rate-Compatible Mother Code** utilizing **Untainted Puncturing**.

---

## 2. The Unified Foundation: Rate-Compatible Mother Code

### 2.1. Deconstructing the "Syndrome Computation" Simplification
A naive description of the protocol states: *"Alice computes the syndrome $s = H_{mother} \cdot x_{frame}$ using the robust R=0.5 mother code matrix."* While arithmetically true, this is a theoretical simplification that obscures the rate-adaptation mechanism.

In the rate-compatible framework defined by Elkouss et al. (2010), the mother code $\mathcal{C}_{R_0}$ serves as a structural parent. The *effective code* $\mathcal{C}_{eff}$ used for a specific block is constructed dynamically by manipulating the input frame $x_{frame}$.

As defined in **"Rate Compatible Protocol for Information Reconciliation" (Elkouss et al., 2010)**:
> "The protocol assumes the existence of a shared pool of codes of length $n$, adjusted for different rates... Depending on the range of crossover probabilities to be corrected, a parameter $\delta$ is chosen to set the proportion of bits to be either shortened ($\sigma$) or punctured ($\pi$)."

Therefore, the syndrome computation is actually a mapping function. Alice maps her payload $\mathbf{x}$ (length $k'$) into a frame $\mathbf{x}^+$ (length $n$) using an embedding function $g$:
$$ \mathbf{x}^+ = g(\mathbf{x}, \sigma, \pi) $$
The syndrome $s(\mathbf{x}^+)$ is then computed against the mother matrix. The decoder does not see the mother code; it sees a subgraph defined by the boundary conditions imposed by $\pi$ (erased nodes) and $\sigma$ (clamped nodes).

### 2.2. The Necessity of Untainted Puncturing
To achieve high effective rates (e.g., $R > 0.8$) using a Rate 0.5 mother code, a significant portion of the frame must be punctured. Random puncturing leads to decoding failures due to "tainted" nodes.

We adopt the **Untainted Puncturing** algorithm. As defined in **"Untainted Puncturing for Irregular Low-Density Parity-Check Codes" (Elkouss et al., 2012)**:
> "A symbol node $v$ is said to be *untainted* if there are no punctured symbols within $\mathcal{N}^2(v)$."

This ensures that *"all the check nodes of a selected symbol are survived check nodes,"* preserving the message-passing capability of the decoder even at high puncturing rates.

---

## 3. Baseline Reconciliation (Elkouss Protocol)

The Baseline protocol is an "inverse puncturing and shortening" scheme. It requires an *a priori* estimate of the channel error rate (QBER) to optimize the code rate in a single shot.

### 3.1. Phase II: Estimation
Before reconciliation, Alice and Bob must perform parameter estimation.
> "Alice extracts $m(\mathbf{x})$ and estimates the crossover probability: $p^* = \frac{m(\mathbf{x}) + m(\mathbf{y})}{t}$" (Elkouss et al., 2010).

### 3.2. Phase III: Rate Selection and Encoding
Using $p^*$, Alice determines the optimal efficiency $f(p^*)$ and the target rate $R$. She then calculates the required number of shortened ($s$) and punctured ($p$) bits.

**Exact Protocol Flow (Elkouss et al., 2010):**
1.  **Frame Construction:**
    > "Alice creates a string $\mathbf{x}^+ = g(\mathbf{x}, \sigma_{p^*}, \pi_{p^*})$ of size $n$."
    The function $g$ places the payload bits into the $n-d$ transmission positions. It fills the $p$ punctured positions with random padding (unknown to Bob) and the $s$ shortened positions with deterministic values (known to Bob).
2.  **Syndrome Transmission:**
    > "She then sends $s(\mathbf{x}^+)$, the syndrome of $\mathbf{x}^+$, to Bob as well as the estimated crossover probability $p^*$."
    Note: In Caligo, $p^*$ is used to derive the puncturing pattern index, which is sent as metadata.

### 3.3. Decoding
Bob replicates the frame construction $g$ using the received metadata.
> "Bob can reproduce Alice’s estimation of the optimal rate $R$, the positions of the $p$ punctured bits, and the positions and values of the $s$ shortened bits, and then he creates the corresponding string $\mathbf{y}^+$." (Elkouss et al., 2010).

For the decoder initialization:
*   **Punctured bits:** $LLR = 0$ (Erasure).
*   **Shortened bits:** $LLR = \pm \infty$ (Perfect knowledge).
*   **Payload bits:** $LLR = \ln((1-p^*)/p^*)$ (Channel output).

---

## 4. Blind Reconciliation (Martinez-Mateo Protocol)

The Blind protocol eliminates the need for QBER estimation by using an iterative "reveal" mechanism.

### 4.1. The Argument: Shortening vs. Matrix Modification
A critical design decision is how to lower the code rate when the initial high-rate attempt fails.
**Question:** Should the protocol modify the matrix (e.g., add rows) or modify the puncturing pattern?
**Answer:** The protocol **must use iterative shortening** (converting punctured bits to shortened bits) while retaining the original mother matrix.

**Theoretical Justification:**
1.  **Syndrome Reuse:** The defining efficiency of the Blind protocol is that the large syndrome block sent in the first iteration is never discarded.
    > "This whole procedure is done using the same base code... The only classical communication... is one message... to send the syndrome and the shortened information bits." (**"Blind Reconciliation"**, Martinez-Mateo et al., 2012).
    Modifying the matrix structure would invalidate the syndrome $s = H \cdot x$, forcing a re-transmission of parity bits and increasing leakage drastically.

2.  **Information Monotonicity:**
    Simply "un-puncturing" a bit (treating it as payload) provides no new information to the decoder, as Bob has no channel observation for that padding bit. However, "shortening" a bit (revealing its value) provides infinite information.
    > "Alice can then reveal a set of the values of the previously punctured symbols... changing from punctured to shortened." (Martinez-Mateo et al., 2012).

3.  **Traversal of the FER Curve:**
    By converting punctured bits to shortened bits, we mathematically traverse the FER curve of the code family.
    > "This is like moving along the dotted vertical line [of the FER graph] and changing the code with $p=200, s=0$ by the code with $p=160, s=40$." (Martinez-Mateo et al., 2012).

### 4.2. Protocol Flow
**Initialization:**
Alice and Bob set a modulation parameter $d$ (total bits to be manipulated) and a step size $\Delta$.

**Iteration 1 (The Optimistic Attempt):**
*   **Configuration:** $p=d, s=0$. All modulation bits are punctured.
*   **Encoding:** Alice embeds payload into $n-d$ bits. She fills $d$ bits with random padding.
    > "Alice sends the syndrome in $\mathcal{C}$ of a word $\tilde{X}$ consisting on embedding $X$ in a length $n$ string and filling the remaining $d$ positions with random symbols." (Martinez-Mateo et al., 2012).
*   **Decoding:** Bob attempts to decode with $d$ erasures.

**Subsequent Iterations (On Failure):**
*   **Mechanism:** Alice selects $\Delta$ bits from the punctured set. She reveals their values to Bob.
*   **State Update:** $p = p - \Delta$, $s = s + \Delta$.
*   **Re-Decoding:** Bob updates the LLRs for the $\Delta$ bits from $0$ to $\pm \infty$. He re-runs the BP decoder using the **original syndrome** and the **original matrix**.
    > "If $d=s$ the protocol fails, else Alice sets $s = s + \Delta$, reveals Bob $\Delta$ symbols and they return to Step 2 and perform a new iteration." (Martinez-Mateo et al., 2012).

---

## 5. Summary of Implementation Requirements

To implement these protocols within Caligo, the following architecture is required:

1.  **Single Mother Matrix:** A robust R=0.5 matrix constructed via ACE-PEG.
2.  **Pattern Library:** A set of **Untainted Puncturing** patterns generated offline.
3.  **Baseline Path:**
    *   Estimate QBER $\to$ Lookup Pattern $\to$ Construct Frame (Payload + Padding) $\to$ Compute Syndrome.
4.  **Blind Path:**
    *   Select Modulation $d$ $\to$ Construct Frame (Payload + Padding) $\to$ Compute Syndrome.
    *   Loop: If Fail $\to$ Reveal $\Delta$ padding bits (convert to Shortened) $\to$ Bob updates LLRs $\to$ Retry.

This architecture ensures strict adherence to the theoretical bounds provided by Elkouss and Martinez-Mateo while optimizing for the unique security constraints of the NSM.

---

# Implementation Report: Advanced Rate-Compatible Reconciliation in Caligo

**Date:** January 2025
**Module:** `caligo.reconciliation`
**Based on:** Theoretical Report (Elkouss/Martinez-Mateo)

---

## 1. Executive Summary

This report details the implementation plan to transition the Caligo project from a multi-matrix reconciliation architecture to a **Unified Mother Code Architecture**. This transition enables two distinct, rate-compatible protocols—**Baseline** (Elkouss) and **Blind** (Martinez-Mateo)—to operate on a single, optimized R=0.5 ACE-PEG matrix.

The implementation addresses critical security requirements under the Noisy Storage Model (NSM) by strictly accounting for information leakage. As noted in the Theoretical Report, *"inefficiency is indistinguishable from insecurity"*; therefore, the codebase must enforce precise syndrome reuse and monotonic leakage tracking.

---

## 2. Architectural Analysis & Refactoring

### 2.1. Identified Code Smells & Architectural Flaws

1.  **Ambiguous Matrix Management (`matrix_manager.py`):**
    *   *Current State:* Loads multiple large matrices for different rates.
    *   *Flaw:* Contradicts the rate-compatible theory. High-rate matrices generated via standard PEG suffer from constraint sparsity.
    *   *Correction:* Refactor to `MotherCodeManager`. It must enforce the loading of exactly *one* mother matrix and a library of *untainted puncturing patterns*.

2.  **Conflated Orchestration Logic:**
    *   *Current State:* `reconciliation/orchestrator.py` handles block partitioning, rate selection, and decoding logic in a monolithic class. Further there is `protocol/orchestrator.py` (SquidASM simulation runner) and `reconciliation/orchestrator.py` (Reconciliation logic)
    *   *Flaw:* It cannot cleanly support the iterative, stateful nature of the Blind protocol without becoming unmaintainable.  Naming confusion and unclear boundaries.
    *   *Correction:* Implement the **Strategy Pattern**. The orchestrator becomes a context manager that delegates execution to `BaselineStrategy` or `BlindStrategy`. Rename `reconciliation/orchestrator.py` to `reconciliation/session.py` to represent the active state of a reconciliation session.

3.  **Logic Leaking into Roles (`alice.py`, `bob.py`):**
    *   *Issue:* The high-level role classes currently contain low-level reconciliation logic (e.g., slicing arrays, manually calling the matrix manager, handling specific message types).
    *   *Risk:* This violates the Single Responsibility Principle. Adding a new protocol requires modifying the core role classes, increasing the risk of regression.
    *   *Fix:* Delegate reconciliation logic to the **Strategy Pattern** managed by the `reconciliation` package.

4.  **Insecure Padding Generation (`ldpc_encoder.py`):**
    *   *Current State:* Uses `generate_padding` with random integers.
    *   *Flaw:* This constitutes *random puncturing*, which creates "tainted" nodes.
    *   *Correction:* Replace with `apply_puncture_pattern`, which uses deterministic patterns generated offline to ensure *"all the check nodes of a selected symbol are survived check nodes"* (Elkouss et al., 2012).

5.  **Implicit "Blind" State:**
    *   *Issue:* Blind reconciliation requires state persistence across network round-trips (tracking which bits were punctured vs. shortened). Currently, this logic is scattered.
    *   *Fix:* Encapsulate state within a `BlindReconciliationStrategy` class.

### 2.2. New Architecture: The Strategy Pattern

We introduce an abstract base class to encapsulate the protocol logic, separating it from the low-level crypto/network primitives. This also allows the protocol to switch between "Baseline" and "Blind" at runtime based on the YAML configuration.

```python
# caligo/reconciliation/strategies/base.py

class ReconciliationStrategy(ABC):
    """
    Abstract base class for reconciliation protocols.
    """
    def __init__(self, mother_code_manager: MotherCodeManager, leakage_tracker: LeakageTracker):
        self.manager = mother_code_manager
        self.tracker = leakage_tracker

    @abstractmethod
    def reconcile_alice(
        self, context: ProtocolContext, sifted_key: np.ndarray, **kwargs
    ) -> Generator[Message, None, ReconciliationResult]:
        """Execute Alice's side of the protocol."""
        pass

    @abstractmethod
    def reconcile_bob(
        self, context: ProtocolContext, sifted_key: np.ndarray, **kwargs
    ) -> Generator[Message, None, ReconciliationResult]:
        """Execute Bob's side of the protocol."""
        pass
```

### 2.3. Configuration Structure (To be integrated with the other example configurations) (`../configs/recon_blind_example.yaml`)

The configuration file drives the factory.

```yaml
reconciliation:
  # Strategy selection: "baseline" or "blind"
  method: "blind"
  
  # Mother Code Configuration
  matrix_path: "configs/ldpc_matrices/ldpc_ace_peg/ldpc_4096_rate0.50.npz"
  
  # Puncturing Patterns Directory
  pattern_dir: "configs/ldpc_matrices/puncture_patterns_ace_peg/ldpc_4096"

  # Baseline Specific
  qber_estimation_fraction: 0.1
  
  # Blind Specific
  blind_modulation_fraction: 0.10  # (d/n)
  blind_step_size: 128             # Delta
  blind_max_iterations: 3
```

---

## 3. Component-Level Implementation

**Goal:** Ensure the low-level tools support the "Mother Code + Untainted Puncturing" theory.

### 3.1. Data Layer: Mother Code & Patterns

**File:** `caligo/reconciliation/matrix_manager.py` (Refactored)

The manager must provide access to the ACE-PEG matrix and the patterns.

*   **Change:**
    *   Remove logic that scans for multiple `.npz` files.
    *   Implement `load_mother_code(path)` which loads the single ACE-PEG R=0.5 matrix.
    *   Implement `load_patterns(dir)` which loads `.npy` files into a dictionary `{target_rate: pattern_array}`.
    *   Ensure `get_matrix()` always returns the mother matrix.

*   **Input Data:**
    *   Matrix: `configs/ldpc_matrices/ldpc_ace_peg/ldpc_4096_rate0.50.npz`
    *   Patterns: `configs/ldpc_matrices/puncture_patterns_ace_peg/ldpc_4096/*.npy`
*   **Key Methods:**
    *   `get_mother_matrix()`: Returns the sparse CSR matrix.
    *   `get_pattern(effective_rate: float)`: Returns the `np.ndarray` of indices to puncture to achieve the target rate.

### 3.2. Encoder: Untainted Puncturing

**File:** `caligo/reconciliation/ldpc_encoder.py`

The encoder must implement the embedding function $g(\mathbf{x}, \sigma, \pi)$ described in the theoretical report.

*   **Change:**
    *   Remove `generate_padding` (random padding).
    *   Strictly implement `apply_puncture_pattern(payload, pattern shortened_indices)`.
    *   **Crucial:** Ensure payload bits map to *non-punctured* indices of the mother code frame.

*   **Logic:**
    1.  Accept `payload`, `puncture_pattern` (indices), and `shortened_indices` (optional).
    2.  Create a frame of zeros (size $n$).
    3.  Map `payload` bits to indices $i \notin (\pi \cup \sigma)$.
    4.  Map Random Padding to indices $j \in \pi$.
    5.  Map Known Values to indices $k \in \sigma$.
    6.  Compute Syndrome: $s = H_{mother} \cdot \text{frame}$.

### 3.3. Decoder: Three-State LLRs

**File:** `caligo/reconciliation/ldpc_decoder.py`

The `build_channel_llr` function is the critical interface for the Blind protocol's iterative shortening. It must support three distinct belief states.

*   **Change:**
    *   Update `build_channel_llr`. It must accept two masks:
        1.  `punctured_mask`: Sets LLR to `0` (Unknown).
        2.  `shortened_mask`: Sets LLR to `±∞` (Known).
    *   This distinction is vital for the Blind protocol.

```python
def build_channel_llr(
    received_bits: np.ndarray, 
    qber: float, 
    punctured_mask: np.ndarray, 
    shortened_mask: np.ndarray,
    shortened_values: np.ndarray
) -> np.ndarray:
    # 1. Payload Bits (Channel Information)
    llr = compute_log_likelihood(received_bits, qber)
    
    # 2. Punctured Bits (Erasure / Padding)
    # "Bob has no information about bit i. His LLR remains 0."
    llr[punctured_mask] = 0.0
    
    # 3. Shortened Bits (Perfect Knowledge)
    # "She reveals the value of bit i. Bob sets LLR = +/- infinity."
    llr[shortened_mask] = np.where(shortened_values == 0, 1000.0, -1000.0)
    
    return llr
```

### 3.4. Fix `RateSelector`
*   **Action:** Update `caligo/reconciliation/rate_selector.py`.
*   **Change:** It currently selects physical rates. It must now calculate **Effective Rates**.
    *   Input: `QBER`.
    *   Logic: Find target rate $R_{eff} < 1 - h(QBER)$.
    *   Output: The specific `puncture_pattern` required to transform the R=0.5 mother code into $R_{eff}$.

---

## 4. Protocol Logic Implementation

### 4.1. Baseline Strategy (Elkouss)

**Configuration:**
```yaml
reconciliation:
  method: "baseline"
  qber_estimation_fraction: 0.1
```

**Implementation Logic:**

1.  **Estimation:** Alice/Bob run the existing QBER estimation routine (sacrificing ~10% of bits).
2.  **Rate Selection (Alice):**
    *   Take `QBER` (passed from Sifting phase).
    *   Calculate $f(p^*) = \frac{1-R}{h(p^*)}$.
    *   Select target effective rate $R_{eff}$. 
    *   Retrieve pattern $\pi$ from `MotherCodeManager`.
3.  **Encoding (Alice):**
    *   Call `ldpc_encoder.encode(payload, pattern=pi)`.
    *   **Leakage:** Record `len(syndrome) + len(hash)`.
    *   Send `Message(type=SYNDROME, payload=s, metadata={rate: R_eff})`.
4.  **Decoding (Bob):**
    *   Receive $s$ and $R_{eff}$.
    *   Retrieve pattern $\pi$.
    *   Call `ldpc_decoder.decode` with `punctured_mask=pi`.

### 4.2. Blind Strategy (Martinez-Mateo)

**Configuration:**
```yaml
reconciliation:
  method: "blind"
  blind_modulation_fraction: 0.10  # d/n
  blind_step_size: 128             # Delta
```

**Implementation Logic:**

1.  **Setup:**
    *   No QBER. Use NSM physical heuristic to select highest feasible rate
    *   Identify modulation set $D$ (last $d$ indices of the frame).
    *   Initialize state: `punctured_indices = D`, `shortened_indices = []`.

2.  **Iteration 1 (Optimistic):**
    *   **Alice:**
        *   Construct frame treating $D$ as punctured (Value 0 in frame, but logically "unknown").
        *   Compute $s = H_{mother} \cdot \text{frame}$.
        *   **Leakage:** Record `len(syndrome) + len(hash)`.
        *   Send `Message(type=SYNDROME, payload=s, metadata={blind_iter: 1})`.
    *   **Bob:**
        *   Init LLRs with `punctured_mask=D`.
        *   Decode. If success, send `ACK`. If fail, send `NACK`.

3.  **Iteration $k$ (Correction):**
    *   **Alice:**
        *   Receive `NACK`.
        *   Select $\Delta$ indices from `punctured_indices`.
        *   Move them to `shortened_indices`.
        *   **Leakage:** Add $\Delta$ to tracker.
        *   Send `Message(type=REVEAL, indices=..., values=...)`.
    *   **Bob:**
        *   Receive `REVEAL`.
        *   Update LLRs: Set $\Delta$ indices to $\pm \infty$.
        *   **Re-run Decoder:** Use **same** syndrome $s$ and **same** matrix $H_{mother}$.
        *   *Theoretical Note:* This implements the "traversal of the FER curve" described by Martinez-Mateo without invalidating the syndrome.

---

## 5. High-Level Integration

**Goal:** Wire the strategies into the main application.

### 5.1. The Factory (`caligo/reconciliation/factory.py`)
Update the factory to read the YAML config and instantiate the correct class.

```python
def create_reconciler(config: Config, matrix_manager: MatrixManager) -> ReconciliationStrategy:
    if config.reconciliation.method == "baseline":
        return BaselineStrategy(matrix_manager, config)
    elif config.reconciliation.method == "blind":
        return BlindStrategy(matrix_manager, config)
    else:
        raise ValueError("Unknown reconciliation method")
```

### 5.2. Updating `alice.py` and `bob.py`

Refactor the `_phase3_reconcile` methods to be agnostic of the specific protocol details.

**Example (Alice):**
```python
# caligo/protocol/alice.py

def _phase3_reconcile(self, context, sifted_key, qber):
    # Factory creates the specific strategy based on config
    reconciler = ReconciliationFactory.create("Alice", self.params)
    
    # Run the generator
    # The strategy handles the specific message loop (Syndrome vs Reveal)
    result = yield from reconciler.reconcile(
        context=context, 
        socket=self._ordered_socket,
        key=sifted_key,
        qber_hint=qber
    )
    
    return result
```

---

## 6. Robustness & Correctness Checklist

To ensure the implementation meets the theoretical requirements:

1.  **Syndrome Invariance (Blind):** Verify in tests that for Blind reconciliation, the syndrome `s` is computed *once* and never changes across iterations. Only the auxiliary `REVEAL` messages change.
2.  **Untainted Property:** Ensure the `generate_puncture_patterns.py` script (in `scripts/`) is used to generate the `.npy` files, and that `ldpc_encoder` crashes if it tries to encode without a valid pattern for the requested rate.
3.  **Leakage Accounting:**
    *   **Baseline:** Leakage = `len(syndrome) + len(hash)`.
    *   **Blind:** Leakage = `len(syndrome) + len(hash) + (iterations * delta)`.
    *   The `LeakageTracker` must be updated to accept dynamic leakage additions from the Blind strategy.

## 7. Deployment & Migration

### 7.1. Integration Steps
1.  **Refactor Managers:** Replace `MatrixManager` with `MotherCodeManager`.
2.  **Implement Strategies:** Create `reconciliation/strategies/baseline.py` and `blind.py`.
3.  **Update Roles:** Modify `alice.py` and `bob.py` to use `ReconciliationFactory`.
4.  **Config Update:** Switch `config.yaml` to point to the new ACE-PEG matrix.

### 7.2. Verification Checklist
*   [ ] **Syndrome Invariance:** Verify Blind protocol does not re-compute syndrome in Iteration 2.
*   [ ] **Untainted Property:** Ensure encoder throws error if pattern is not found (preventing fallback to random padding).
*   [ ] **Leakage Accounting:** Verify `LeakageTracker` accurately sums syndrome + hash + revealed bits.
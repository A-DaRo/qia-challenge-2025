# Sprint 3 Implementation Specification — Output & Integration (Phase IV + E2E)

> **Sprint**: 3
>
> **Theme**: Output & Integration — NSM-correct privacy amplification, oblivious OT output formatting, and end-to-end orchestration + validation.
>
> **Scope Constraint**: This document specifies *what to build* (interfaces, dataclass contracts, mathematical checks, abort triggers, and validation suites). It intentionally contains **no implementation code**.

---

## 1. Sprint 3 Executive Summary

### Goal
Complete the E-HOK pipeline by delivering:

1. **NSM-correct privacy amplification length selection** (replace QKD-derived length formula with NSM “Max Bound” + leftover hash lemma accounting).
2. **Oblivious output formatting** for 1-out-of-2 OT:
   - Alice outputs $(S_0, S_1)$.
   - Bob outputs exactly one $S_C$ and retains choice bit $C$.
3. **End-to-end protocol orchestration** with SquidASM compliance and traceable abort reasons.
4. **E2E adversarial + statistical validation** anchored to the roadmap success criteria.

### Reference Map (Roadmap §4.4 → Phase IV analysis → Squid assessment)
Derived strictly from:

- Roadmap Sprint 3 tasks in [docs/implementation plan/master_roadmap.md](master_roadmap.md#L430-L467):
  - **TOEPLITZ-MODIFY-001**
  - **OBLIV-FORMAT-001**
  - **STORAGE-LINK-001**
  - **E2E-PIPELINE-001**
  - **E2E-ADVERSARIAL-001**
  - **E2E-STATISTICAL-001**
- Phase IV analysis (privacy amplification workflow + interfaces + leftover hash lemma + NSM bound):
  - [docs/implementation plan/phase_IV_analysis.md](phase_IV_analysis.md)
- SquidASM assessment (explicit gap statements + integration constraints):
  - [docs/implementation plan/squid_assesment.md](squid_assesment.md)

### Sprint 3 Deliverable Contract (What “done” means)
Sprint 3 is complete when a single end-to-end run produces **structured OT output** and the validation suite demonstrates:

- **Correctness**: Alice/Bob derived keys are consistent where required.
- **Security accounting correctness**: Phase IV length calculation uses NSM Max Bound and subtracts the Phase III wiretap cost $|\Sigma|$.
- **Flow safety**: Δt enforcement + ordered commit-then-reveal semantics remain intact in the full pipeline.
- **Traceability**: Each abort outcome maps to a specific invariant violation and a test.

---

## 2. System Boundary Contracts (Phase III → Phase IV → Application)

Sprint 3 is primarily about making Phase IV *computable, correct, and consumable* by an application.

### 2.1 Input Contract: `ReconciledKeyData` (from Phase III)
Source: Phase IV analysis “Input Interface” box.

**Dataclass fields required**:

- `reconciled_key : np.ndarray`
  - Error-corrected bitstring.
  - Length: $n$ bits.
- `total_leakage_bits : int`
  - Total Phase III public leakage (syndrome bits + verification/hash bits).
  - Units: bits.
- `integrated_qber : float`
  - QBER estimate produced by Phase III (e.g., from LDPC decoding/correction counts).
  - Range: $[0, 0.5]$.
- `blocks_succeeded : int`
- `blocks_failed : int`

**Invariants**:
- `total_leakage_bits >= 0`.
- `reconciled_key.dtype == np.uint8` and is bit-valued.
- `0 <= integrated_qber <= 0.5`.

### 2.2 Phase IV Output Contract: `ObliviousTransferResult`
Source: Phase IV analysis “Output Interface” box and Roadmap “Oblivious Output Structure”.

Sprint 3 MUST define a structured output type that separates Alice’s and Bob’s outputs.

**Dataclasses required (Phase IV analysis)**:

- `AliceObliviousKey`
  - `s0 : np.ndarray` (ℓ bits)
  - `s1 : np.ndarray` (ℓ bits)
  - `seed : np.ndarray` (Toeplitz seed shared with Bob)
  - `epsilon_achieved : float`
- `BobObliviousKey`
  - `s_c : np.ndarray` (ℓ bits)
  - `c : int` (implicit choice bit; NOT sent to Alice)
  - `seed : np.ndarray` (received from Alice)
  - `epsilon_achieved : float`
- `ProtocolMetrics`
  - `key_length : int` (ℓ)
  - `compression_ratio : float` (ℓ / n)
  - `total_leakage_bits : int`
  - `storage_noise_assumed : float` (r)
  - `epsilon_sec : float`
  - `epsilon_cor : float`

**Hard requirement (OT contract)**:
- Alice’s side MUST expose $(S_0,S_1)$.
- Bob’s side MUST expose exactly one key $S_C$ and the bit $C$.
- Alice MUST NOT learn $C$ (this must be enforced at the protocol interface level: no message or return value to Alice includes $C$).

---

## 3. Component Spec: TOEPLITZ-MODIFY-001 (NSM-Correct Privacy Amplification)

**Roadmap task**: TOEPLITZ-MODIFY-001 in [master_roadmap.md](master_roadmap.md#L436-L441).

### 3.1 Scope
Replace the final-length calculation used by Toeplitz privacy amplification so that it is **NSM-correct**.

The repository already contains Toeplitz hashing infrastructure:
- `ehok/implementations/privacy_amplification/toeplitz_amplifier.py`
- `ehok/implementations/privacy_amplification/finite_key.py`

Per the Squid assessment, Toeplitz hashing and finite-key μ computation exist but the **entropy bound is QKD-based and invalid** for NSM E-HOK.

### 3.2 Required Separation of Concerns
- **Pure math**:
  - NSM Max Bound and Γ function are specified in Sprint 1 and live in `ehok/analysis/nsm_bounds.py`.
  - Phase IV length selection MUST use that module (not embed NetSquid/SquidASM logic).
- **System logic** (seed transmission, ordered sockets, orchestration): MUST NOT be added inside `ehok/analysis/nsm_bounds.py`.

### 3.3 Correct Security Logic (Leftover Hash Lemma + Wiretap Cost)
Source: Phase IV analysis “Leftover Hash Lemma” and “Error Correction Penalty”.

Sprint 3 MUST ensure Phase IV length selection respects:

1) Privacy amplification bound:

$$\ell \le H_{min}^\epsilon(X_{\bar{B}} \mid \mathcal{F}(Q)\,\Theta\, B\, \Sigma_{\bar{B}}) - 2\log_2\frac{1}{\epsilon_h} + 1$$

2) Syndromes subtract directly (“wiretap cost”):

$$\ell \ge H_{min}^\epsilon(X_{\bar{B}} \mid \mathcal{F}(Q)\,\Theta\, B) - |\Sigma_{\bar{B}}| - 2\log_2\frac{1}{\epsilon_h} + 1$$

**Hard requirement**: $|\Sigma|$ is provided by Phase III (`total_leakage_bits`) and MUST be subtracted directly.

### 3.4 E-HOK (NSM) Final Length Formula
Source: Phase IV analysis “Operational Workflow” and “Secure Key Length Formula (NSM-Specific)”.

Sprint 3 MUST compute output length using the NSM min-entropy rate $h_{min}(r)$:

$$\ell \le n\cdot h_{min}(r) - |\Sigma| - 2\log_2\frac{1}{\varepsilon_{sec}} - \Delta_{finite}$$

Where:
- $n$ is the reconciled input length.
- $h_{min}(r)$ is from the NSM “Max Bound” (Sprint 1 output).
- $|\Sigma|$ is total reconciliation leakage in bits.
- $\varepsilon_{sec}$ is the security parameter.
- $\Delta_{finite}$ captures finite-size penalties (Phase IV analysis notes it can dominate in the “Death Valley” regime).

**Explicit forbidden behavior**:
- Phase IV length selection MUST NOT use the QKD asymptotic rate $1-h(Q)$ as the min-entropy source.

### 3.5 Toeplitz Seed Requirements
Source: Phase IV analysis “Toeplitz Seed Generation” stage and Squid assessment Toeplitz section.

- Seed generation MUST use cryptographically strong randomness (`secrets.token_bytes()` in current Toeplitz implementation).
- Seed MUST NOT be reused across sessions.
- Seed length must remain consistent with Toeplitz hashing: seed bits length must be $(\ell + n - 1)$.

### 3.6 Feasibility Pre-Check (“Death Valley” Guard)
Source: Phase IV analysis “Feasibility Check” stage and Squid assessment Feasibility gap.

Sprint 3 MUST treat feasibility as a *hard gate* before committing to Phase IV output:

- Compute `expected_length = n*h_min(r) - penalty`.
- If `expected_length <= 0`, abort with **Batch Size Too Small** (or equivalent abort code defined in the project taxonomy), and provide a recommendation for a larger batch size.

**No partial output**: If Phase IV aborts, neither party may receive a partial key.

---

## 4. Component Spec: OBLIV-FORMAT-001 (ObliviousKeyFormatter)

**Roadmap task**: OBLIV-FORMAT-001 in [master_roadmap.md](master_roadmap.md#L436-L448).

### 4.1 Rationale
The current codebase contains a legacy single-key output (`ObliviousKey` with a knowledge mask). Sprint 3 MUST produce **OT-structured output** $(S_0,S_1)$ and $(S_C,C)$.

### 4.2 Inputs Required to Format Oblivious Output
Source: Phase IV analysis “STAGE 5: Oblivious Output Formatting”.

Sprint 3 MUST ensure the pipeline preserves enough metadata to create the correct key partitions.

**Formatter inputs MUST include** (split by party):

- Common / shared:
  - `output_length_bits : int` (ℓ)
  - `seed : np.ndarray` (Toeplitz seed; generated once and shared with Bob)
- Alice-side formatting MUST have:
  - `reconciled_key : np.ndarray` (the Phase III reconciled key)
  - `key_round_indices : np.ndarray` mapping each reconciled bit to its original round index in $\{0,\dots,M-1\}$
  - `bases_alice : np.ndarray` (length $M$, basis choices per round)
- Bob-side formatting MUST have:
  - `reconciled_key : np.ndarray`
  - `key_round_indices : np.ndarray`
  - `bases_alice : np.ndarray` (received during basis reveal)
  - `bases_bob : np.ndarray` (local)

**Reason**: The Phase IV analysis indicates the oblivious structure is derived from basis-dependent index sets (θ partitions and match/mismatch partitions).

### 4.3 Oblivious Partition Semantics
Source: Phase IV analysis STAGE 5:

- Alice MUST compute:
  - $S_0 = \mathrm{Hash}(X\mid \{i: \theta_i = 0\})$
  - $S_1 = \mathrm{Hash}(X\mid \{i: \theta_i = 1\})$
- Bob MUST compute:
  - $S_C = \mathrm{Hash}(X\mid \{i: \text{Bob matched Alice}\})$
  - $C$ is implicit and not known to Alice.

**Minimal allowed interpretation (Sprint 3 contract)**:
- $C$ is derived from Bob’s basis choice rule in the pipeline (must be documented in the formatter docstring and reflected in tests).
- Alice’s side must not require knowledge of `bases_bob`.
- Bob’s side must not receive both $S_0$ and $S_1$.

**Seed compatibility requirement (single-seed design)**

The Phase IV analysis output interface contains a single Toeplitz seed shared between Alice and Bob. To keep Toeplitz hashing well-defined while still producing multiple outputs, Sprint 3 MUST define oblivious partitioning in a way that preserves a fixed input length $n$.

The simplest compliant construction is a **masking-based partition**:

- Define $x \in \{0,1\}^n$ as the reconciled key bits.
- For any subset of indices $J \subseteq \{0,\dots,n-1\}$, define $x^{(J)}$ by:
  - $x^{(J)}_i = x_i$ if $i \in J$, else $0$.

Then compute:

- $S_0 = T(\text{seed}) \cdot x^{(J_0)} \bmod 2$
- $S_1 = T(\text{seed}) \cdot x^{(J_1)} \bmod 2$
- $S_C = T(\text{seed}) \cdot x^{(J_C)} \bmod 2$

where $T(\text{seed})$ is the Toeplitz hash function for output length ℓ and input length $n$, and the $J_\bullet$ sets are defined from the relevant basis-derived partitions.

### 4.4 Output Dataclasses and Invariants
The dataclasses in Section 2.2 MUST be implemented with strict validation:

- All key arrays are `np.uint8` and bit-valued.
- `len(s0) == len(s1) == len(s_c) == ℓ`.
- `seed` must be non-empty if `ℓ > 0`.
- `c` must be in `{0,1}`.

### 4.5 Required Failure Modes
The formatter MUST fail fast (abort Phase IV) if:

- `output_length_bits <= 0`.
- Any input array lengths are inconsistent with `key_round_indices` mapping.
- `seed` length is incompatible with Toeplitz hashing of the intended input length.

---

## 5. Component Spec: STORAGE-LINK-001 (Storage Noise r Wiring)

**Roadmap task**: STORAGE-LINK-001 in [master_roadmap.md](master_roadmap.md#L436-L441).

### 5.1 Requirement
The adversary storage noise parameter $r$ MUST be configured before the run and must flow to Phase IV calculations without implicit defaults.

### 5.2 Parameter Flow Contract
Source: Phase IV analysis lifecycle management and Squid assessment “NetSquid Integration Point”.

Sprint 3 MUST guarantee:

- The same configured/derived `storage_noise_r` is visible to:
  - Feasibility checks (pre-flight and Phase IV feasibility).
  - Phase IV final-length selection.
  - Phase IV output metrics.

### 5.3 Optional NetSquid Link (Simulation-Consistent r)
Phase IV analysis and Squid assessment describe mapping storage noise to NetSquid memory models:

$$r \approx 1 - F_{storage}(\Delta t)$$

Sprint 3 MUST document (even if not implemented in this sprint) whether $r$ is:

- **Configured directly** in `ProtocolConfig`, OR
- **Derived** from NetSquid memory parameters (T1/T2) + Δt via an adapter.

If derived, the adapter MUST be isolated from pure math modules (no NetSquid imports in `ehok/analysis/nsm_bounds.py`).

---

## 6. Component Spec: E2E-PIPELINE-001 (End-to-End Orchestrator)

**Roadmap task**: E2E-PIPELINE-001 in [master_roadmap.md](master_roadmap.md#L453-L466).

### 6.1 Orchestrator Responsibility
The orchestrator MUST produce a single protocol run that:

- Enforces ordering and Δt constraints from earlier sprints.
- Produces Phase IV OT-structured output.
- Exposes a traceable transcript (diagnostics/metrics sufficient for E2E tests).

### 6.2 SquidASM Compliance Requirements (Non-Negotiable)
Derived from Squid assessment “Gap Analysis” and “ClassicalSocket” constraints.

Sprint 3 MUST validate compliance with these constraints:

- No use of blocking wall-clock waits (no `time.sleep` in protocol execution).
- Timing enforcement uses simulation-time constructs consistent with SquidASM/NetSquid (e.g., generator-based yielding and simulation time checks).
- Security-critical commit-then-reveal ordering uses an ordered messaging layer (from Sprint 2) rather than raw `ClassicalSocket`.
- Phase IV is purely classical (no qubit operations after the quantum phase completes).

### 6.3 Required Pipeline Phases and Phase Boundaries
Sprint 3 orchestrator MUST enforce the following high-level order:

1. Quantum measurement phase (Phase I)
2. Missing rounds commitment + ordered acknowledgment (Phase II)
3. Δt barrier (Sprint 1 timing enforcer)
4. Basis reveal + sifting + sampling + QBER estimation (Phase II)
5. LDPC reconciliation with leakage tracking and safety cap enforcement (Phase III)
6. Phase IV feasibility check (Death Valley gate)
7. Phase IV NSM-correct privacy amplification and Toeplitz seed sharing
8. Oblivious output formatting and return of `ObliviousTransferResult`

**Hard invariant (from Sprint 2 + Squid assessment)**:
- Bob’s detection report must be received and ACKed before Alice reveals any basis information.

---

## 7. Validation Suite Specification

Sprint 3 MUST introduce validation that makes it impossible to regress to QKD bounds or single-key outputs.

### 7.1 Unit Tests (Component Level)

#### 7.1.1 NSM length-selection correctness
Tests MUST assert that Phase IV length selection:

- Uses `h_min(r)` from NSM Max Bound (Sprint 1 output).
- Subtracts `total_leakage_bits` exactly.
- Includes a hashing security cost term consistent with the leftover hash lemma form ($-2\log_2(1/\epsilon_h) + 1$) or the project-configured equivalent.
- Produces `0` (not negative) when infeasible.

#### 7.1.2 Seed invariants
Tests MUST verify:

- Seed length equals `(output_length_bits + input_length_bits - 1)` in bits.
- Seed changes between sessions (no reuse) when run twice.

#### 7.1.3 Oblivious formatting invariants
Tests MUST verify:

- Alice obtains both `s0` and `s1` of equal length.
- Bob obtains `s_c` plus `c`.
- Alice does not receive `c` in any output structure.

### 7.2 Integration Tests (Phase III → Phase IV Boundary)

Tests MUST verify that Phase III output passes the complete Phase IV input contract:

- `ReconciledKeyData.total_leakage_bits` is populated and equals the tracked wiretap cost.
- The Phase IV feasibility pre-check runs before hashing.

### 7.3 E2E-ADVERSARIAL-001 (Adversarial Condition Tests)

Sprint 3 MUST include tests that force the correct aborts under:

- **Ordering attack**: Basis reveal before detection report ACK.
- **Leakage amplification attack**: Reconciliation attempts that exceed $L_{max}$.
- **Detection anomaly**: Missing/detected rounds inconsistent with Hoeffding bound.
- **High QBER**: Adjusted-QBER abort threshold (Sprint 2 invariant).
- **Seed transmission failure**: Seed mismatch / missing seed causes terminal abort with no key output.

### 7.4 E2E-STATISTICAL-001 (Statistical Validation Suite)

The statistical suite MUST validate the roadmap completion criterion and the Phase IV analysis “Death Valley” effect:

- For a “reasonable” regime (roadmap example: $Q=5\%$, $n=10{,}000$):
  - E2E run produces a **positive** key length.
- For smaller batch sizes (below feasibility threshold):
  - Phase IV aborts early and recommends a larger batch size.
- Key length should be monotone non-decreasing with $n$ for fixed parameters (qualitative check).

---

## 8. Risks and Mandatory Mitigations (Sprint 3)

### 8.1 Risk: Regression to QKD bounds
**Mitigation**: Unit tests that fail if any path uses $1-h(Q)$ as the entropy rate in Phase IV.

### 8.2 Risk: Single-key output persists
**Mitigation**: Strict output contract tests requiring $(S_0,S_1)$ and $(S_C,C)$.

### 8.3 Risk: SquidASM ordering/timing gaps reappear under E2E
**Mitigation**: E2E adversarial tests that intentionally reorder messages / delay ACKs and require abort.

---

## 9. Definition of Done (Sprint 3)

Sprint 3 is “done” when all items below are true:

1. TOEPLITZ-MODIFY-001: Phase IV length selection uses NSM Max Bound and wiretap cost subtraction.
2. OBLIV-FORMAT-001: OT output is structured and validated.
3. STORAGE-LINK-001: `storage_noise_r` is explicitly configured/derived and visible in Phase IV metrics.
4. E2E-PIPELINE-001: A full run executes through Phase IV and returns an `ObliviousTransferResult`.
5. E2E-ADVERSARIAL-001: Adversarial tests produce correct aborts.
6. E2E-STATISTICAL-001: Statistical tests demonstrate a positive key in the roadmap target regime and detect “Death Valley” infeasibility early.

---

## 10. File Targets (Sprint 3)

Roadmap targets (with repository path reality):

- **TOEPLITZ-MODIFY-001**
  - Roadmap: `ehok/implementations/privacy_amplification.py`
  - Repository: `ehok/implementations/privacy_amplification/toeplitz_amplifier.py` (modify `compute_final_length` semantics to be NSM-correct)
- **OBLIV-FORMAT-001**
  - Create: `ehok/core/oblivious_key.py`
- **STORAGE-LINK-001**
  - Modify: `ehok/core/feasibility.py` (ensure r flows into Phase IV decisions)
- **E2E-PIPELINE-001**
  - Create: `ehok/protocols/ehok_protocol.py`
- **E2E-ADVERSARIAL-001**
  - Create: `ehok/tests/test_adversarial.py`
- **E2E-STATISTICAL-001**
  - Create: `ehok/tests/test_statistical.py`

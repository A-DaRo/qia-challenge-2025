# Sprint 2 Implementation Specification — Protocol Layer (Phases II & III)

> **Sprint**: 2 (Days 11–20)
> 
> **Theme**: Protocol Layer — Ordered messaging, detection validation, finite-size statistical guardrails, and reconciliation leakage safety.
> 
> **Scope Constraint**: This document specifies *what to build* (state machines, dataclass payloads, mathematical checks, abort triggers, and component interactions). It intentionally contains **no implementation code**.

---

## 1. Sprint 2 Executive Summary

### Goal
Implement the **Classical Protocol Layer** that converts Phase I raw quantum events into Phase III error-corrected keys while enforcing:

- **Asynchronous order enforcement** (Commit-then-Reveal / “Wait for ACK” semantics).
- **Statistical guardrails** (Hoeffding/Chernoff validation of detection counts and finite-size adjustment of QBER).
- **Leakage budgeting** (hard cap on syndrome/verification leakage and abort on overrun).

Sprint 2 delivers the protocol-critical “messy middle”: message ordering + real-time statistics + one-way reconciliation safety.

### The “Handshake” (Sprint 1 → Sprint 2)
Sprint 1 introduces **TimingEnforcer (Δt barrier)** and feasibility gates. Sprint 2 must integrate with that ordering as follows:

1. **Bob commits detection report** (missing/detected rounds) and it is **ACKed** over an ordered socket.
2. **Δt must elapse** (TimingEnforcer) before Alice sends basis information.
3. Only then does Phase II sifting/estimation proceed (indices, test subset, QBER+μ).

**Security handshake invariant** (from Phase II analysis): Bob’s detection report must be received and acknowledged **before** Alice reveals basis information.

### Reference Map (Roadmap §4.3 → Phase Analyses)
Derived strictly from [docs/implementation plan/master_roadmap.md](master_roadmap.md#L401-L447) §4.3 and the phase analyses:

- **TASK-ORDERED-MSG-001** (roadmap: ORDERED-MSG-001) → Phase II “Sandwich” flow + Gap Analysis (ordered acknowledgment) and Phase II state machine.
- **TASK-DETECT-VALID-001** (roadmap: DETECT-VALID-001) → Phase II §2.2 Missing Rounds Constraint (Chernoff/Hoeffding).
- **TASK-FINITE-SIZE-001** (roadmap: FINITE-SIZE-001) → Phase II §2.3 Finite-Size Statistical Penalty $\mu$; literature Eq. (2) in Scarani-style finite-key analysis.
- **TASK-QBER-ADJUST-001** (roadmap: QBER-ADJUST-001) → Phase II state machine + hard abort semantics for adjusted QBER.
- **TASK-LEAKAGE-MGR-001** (roadmap: LEAKAGE-MGR-001) → Phase III wiretap cost + safety cap $L_{max}$ gap analysis.
- **TASK-LDPC-INTEGRATE-001** (roadmap: LDPC-INTEGRATE-001) → Phase III pipeline stages 3–6 and leakage tracking requirements.

---

## 2. Component Specification: Ordered Messaging & Sifting

**Context**: TASK-ORDERED-MSG-001.

### 2.1 Protocol Logic: `OrderedProtocolSocket` State Machine

**Problem** (Phase II): SquidASM `ClassicalSocket` has send/recv but does not guarantee causal ordering between “report received” and “bases revealed”. Network latency can reorder causally critical events.

**Design Objective**: Wrap a `ClassicalSocket` with a minimal ordering layer that provides:

- **Send-with-ACK**: sender blocks until an ACK for that message is observed.
- **Receive-and-ACK**: receiver automatically sends ACK upon accepting a message.
- **Monotonic sequencing**: a per-session sequence number so ACKs are unambiguous.

**Required Invariant (security-critical)**
- Alice MUST NOT send `BasisReveal` unless she has successfully completed `send_with_ack(DetectionReport)` in the opposite direction (i.e., Bob’s detection report has been received and acknowledged).

### 2.2 Message Envelope (Minimal, Ordering-Only)
All ordered messages MUST be carried in an envelope with:

- `session_id`: opaque identifier for a single protocol execution.
- `seq`: monotonically increasing integer per direction per session.
- `msg_type`: enum/string discriminator (e.g., `DETECTION_REPORT`, `BASIS_REVEAL`, `ACK`).
- `payload`: the dataclass payload (see below).

**ACK payload** MUST contain:
- `ack_seq`: the `seq` being acknowledged.
- `ack_msg_type`: msg_type being acknowledged (defensive against ambiguity).

**Replay/duplication handling** (simplest safe interpretation)
- If a receiver observes a duplicate `(session_id, seq)` already processed, it MUST re-send the corresponding ACK and MUST NOT re-apply the payload.
- If a receiver observes a future `seq` > expected next value, it MUST buffer or reject. For Sprint 2, the simplest allowed behavior is: **reject and abort** with a protocol-violation error (since out-of-order delivery indicates the ordering layer is not functioning as required).

### 2.3 `OrderedProtocolSocket` States
This is the transport state machine (not the whole Phase II protocol):

- **IDLE**: ready to send or receive.
- **SENT_WAIT_ACK(seq)**: message sent; waiting for matching ACK.
- **RECV_PROCESSING(seq)**: message received; validating and emitting ACK.
- **VIOLATION/ABORT**: unrecoverable ordering failure.

**Timeout behavior**
- `send_with_ack(..., timeout)` MUST either:
  - complete upon receiving a matching ACK, OR
  - transition to `VIOLATION/ABORT` when timeout expires.

Rationale: Phase II analysis explicitly calls timeouts “ProtocolViolation on timeout” for ordered acknowledgment.

### 2.4 Sifting Integration Point

The ordered socket is used to enforce the Phase II causal chain:

1. Bob → Alice: `DetectionReport` (ordered receive on Alice).
2. Alice → Bob: `ACK` for detection report (automatic from ordered receive).
3. Alice waits **Δt** (TimingEnforcer from Sprint 1).
4. Alice → Bob: `BasisReveal` (may use ordered send; MUST NOT precede step 2).

After basis reveal:
- Bob computes and sends index lists (e.g., $I_0, I_1$) and QBER test data using normal or ordered messaging (ordering is no longer security-critical in the same way as the commit-then-reveal step, but still recommended to keep implementation deterministic).

### 2.5 Data Structures: Payload Contracts

#### 2.5.1 `DetectionReport` payload
Derived from Phase II analysis “DetectionReport (Dataclass)” box.

Required fields:
- `total_rounds: int` ($M$)
- `detected_indices: list[int]`
- `missing_indices: list[int]`

Required invariants:
- `len(detected_indices) + len(missing_indices) == total_rounds`
- `detected_indices ∩ missing_indices = ∅`
- All indices are within `[0, total_rounds-1]`

Derived properties (for validation logic):
- `S := len(detected_indices)`
- `detection_rate := S / M`

#### 2.5.2 `BasisReveal` payload
Phase II uses basis string $\alpha^m$; Sprint 2 must define a payload representation.

Required fields:
- `total_rounds: int` ($M$)
- `bases: list[int]` of length $M$ where each element encodes Alice’s basis choice for that round.

Encoding constraints:
- Basis encoding MUST be stable and explicit (e.g., `0 := Z`, `1 := X`), and MUST be documented in the class docstring.

Required invariants:
- `len(bases) == total_rounds`

---

## 3. Component Specification: Statistical Validation

**Context**: TASK-DETECT-VALID-001, TASK-FINITE-SIZE-001, TASK-QBER-ADJUST-001.

### 3.1 Detection Validator (Chernoff/Hoeffding Guardrail)

**Purpose**: Detect post-selection attacks where Bob claims “missing rounds” strategically.

**Inputs**
- `DetectionReport` from Bob.
- `P_expected` (expected detection probability / transmittance) calibrated from Phase I (channel + device characterization).
- `\varepsilon` (failure probability budget for this check). Phase II analysis uses $\varepsilon$ and uses Hoeffding-style form.

**Test statistic**
Let:
- $M$ = `total_rounds`
- $S$ = number of detected rounds = `len(detected_indices)`

Alice MUST validate:

$$\Pr\left[\lvert S - P_{expected} \cdot M\rvert \ge \zeta \cdot M\right] < \varepsilon$$

with tolerance (Phase II):

$$\zeta = \sqrt{\frac{\ln(2/\varepsilon)}{2M}}$$

**Acceptance interval**

$$S \in \left[(P_{expected}-\zeta)M,\, (P_{expected}+\zeta)M\right]$$

**Abort trigger**
- If $S$ is outside the interval, trigger an abort classified as **Detection Anomaly**.

**Diagnostics requirement**
The validator MUST return a diagnostic message containing at minimum:
- observed $S$ and $M$,
- configured $P_{expected}$,
- computed $\zeta$,
- computed bounds.

### 3.2 Finite-Size Analysis: Statistical Penalty $\mu$

**Purpose**: Convert observed test-set QBER into a conservative (worst-case) bound used for security decisions.

**Inputs**
- $k$: test set size.
- $n$: remaining key size.
- $\varepsilon_{sec}$: security parameter used for finite-size correction.

**Penalty formula**
Phase II analysis specifies:

$$\mu := \sqrt{\frac{n + k}{nk} \cdot \frac{k + 1}{k}} \cdot \ln\frac{4}{\varepsilon_{sec}}$$

This matches the literature finite-key expression (see [docs/literature/Tight Finite-Key Analysis for Quantum Cryptography.md](../literature/Tight%20Finite-Key%20Analysis%20for%20Quantum%20Cryptography.md#L254) Eq. (2)).

**Output**
- $\mu$ as a real-valued penalty to be added to observed QBER.

**Domain constraints (abort/invalid-input)**
- Require $n > 0$ and $k > 0$.
- Require $\varepsilon_{sec} \in (0,1)$.

### 3.3 QBER Adjustment and Abort Logic

Let:
- $e_{obs}$ be the observed QBER on the test set.

Compute:

$$e_{adj} = e_{obs} + \mu$$

**Hard abort trigger (Sprint 1 compatibility + Phase II state machine)**
- If $e_{adj} > QBER_{limit}$, abort.

Where $QBER_{limit}$ MUST be configured as the Phase II “hard limit” (from Phase II and roadmap context):
- Default: $QBER_{limit} = 0.22$.

**Required condition (explicit from user mandate)**

$$\textbf{If } QBER_{obs} + \mu > QBER_{limit} \textbf{, trigger Abort.}$$

**Warning trigger (optional but recommended; matches Phase II narrative)**
- If $e_{adj} > 0.11$ and $\le 0.22$, raise a warning status (continue but record degraded regime).

---

## 4. Component Specification: Reconciliation & Leakage

**Context**: TASK-LEAKAGE-MGR-001, TASK-LDPC-INTEGRATE-001.

### 4.1 Leakage Budget Model

Phase III treats every transmitted bit of:
- LDPC syndrome, and
- verification hash/tag

as **fully leaked** information. This is the “Wiretap Cost” $|\Sigma|$ which MUST be carried forward into Phase IV.

### 4.2 `LeakageSafetyManager` Contract

**Purpose**: Enforce a hard leakage cap to prevent “feigned failure” attacks where a dishonest party forces repeated leakage until security collapses.

**Inputs**
- `L_max`: maximum allowed total leakage (bits). This MAY be derived from configured target security margins; the key property is that it is a hard cap.
- Per-block leakage events:
  - `syndrome_bits` (integer)
  - `hash_bits` (integer)

**State**
- `total_syndrome_bits`
- `total_hash_bits`
- `total_leakage_bits = total_syndrome_bits + total_hash_bits`

**Hard abort trigger**
- If `total_leakage_bits > L_max` → abort classified as **Leakage Cap Exceeded**.

**Output required for Phase IV**
- `wiretap_cost_bits := total_leakage_bits` (or, if Phase IV distinguishes, provide both totals plus sum).

### 4.3 Integration: `LDPCReconciliator` → Leakage Manager

Phase III pipeline stages (from Phase III analysis stages 3–6) require the following interaction pattern:

1. For each LDPC block, Alice sends syndrome $S_i$.
2. Immediately after sending, the protocol MUST call LeakageSafetyManager accounting with `syndrome_bits = len(S_i)`.
3. During hash verification, Alice sends a hash/tag for each verified block.
4. LeakageSafetyManager MUST also account for `hash_bits`.

**Reporting back**
`LDPCReconciliator` (or the protocol orchestrator wrapping it) MUST produce a per-block report containing:
- `block_index`
- `syndrome_bits`
- `hash_bits`
- `decode_converged` (bool)
- `hash_verified` (bool)

LeakageSafetyManager MUST be updated from this per-block report (single responsibility: the manager owns cumulative accounting and abort decision).

### 4.4 Wiretap Cost Definition to Pass to Phase IV

Sprint 2 must standardize the Phase III → IV interface value:

- **Wiretap Cost**: $|\Sigma| := \sum_i |S_i| + \sum_i |h_i|$ (in bits), where $S_i$ is syndrome and $h_i$ is verification hash/tag.

This quantity MUST be:
- logged into metrics,
- returned in the Phase III output dataclass, and
- consumed by Phase IV min-entropy/key-length computation as a subtraction term.

---

## 5. Testing & Quality Assurance Plan

All tests are specified test-first; implementations must satisfy these before legacy removal.

### 5.1 Network Simulation Tests (Ordered Socket)

**Goal**: Prove the “Wait for ACK” invariant under asynchronous delivery.

Test categories:
- **Artificial delay injection**: Delay ACK delivery so sender blocks; ensure basis reveal is not sent.
- **Out-of-order packet injection**: Deliver ACKs out of order or deliver two payloads with swapped `seq`; ensure the system aborts with a protocol violation.
- **Duplicate delivery**: Deliver the same `(session_id, seq)` twice; ensure idempotent handling (no double-processing; ACK re-sent).

Success criteria:
- A basis reveal message is never observed by Bob unless Alice has received the ACK for Bob’s detection report.

### 5.2 Statistical Injection Tests (Detection Validator)

**Goal**: Ensure detection anomalies are caught with the specified bound.

Test cases:
- Construct `DetectionReport` with $S$ inside $[(P_{expected}-\zeta)M,(P_{expected}+\zeta)M]$ → validation passes.
- Construct `DetectionReport` with $S$ just outside the interval → abort.

Acceptance criteria:
- Computed $\zeta$ matches the Phase II formula.
- Diagnostic message includes $S$, $M$, $P_{expected}$, $\zeta$, and interval.

### 5.3 Finite-Size / QBER Adjustment Tests

**Goal**: Ensure the penalty $\mu$ and abort rule are implemented exactly.

Test cases:
- Verify $\mu$ matches the Scarani-style expression (Eq. (2)) for fixed $(n,k,\varepsilon_{sec})$.
- Verify abort condition: if $e_{obs}+\mu > 0.22$ → abort.
- Verify boundary behavior: if $e_{obs}+\mu = 0.22$ → do not abort (strictly “>” unless Phase II analysis specifies otherwise).

### 5.4 Reconciliation Failure / Leakage Safety Tests

**Goal**: Ensure leakage is treated as a strict budget.

Test cases:
- Force large syndromes (or repeated block attempts at the protocol level) such that `total_leakage_bits > L_max` → abort.
- Force decode failures: ensure one-way constraint holds (no interactive “retry” messages), and failed blocks are either discarded or cause abort according to the chosen policy.

Acceptance criteria:
- Wiretap cost returned equals the sum of all syndrome + hash bits actually transmitted.

---

## 6. Risks & Definition of Done

### Specific Risks (Roadmap-aligned)

- **LDPC efficiency cliff** (roadmap RISK-002): syndrome rate grows rapidly with QBER; net key rate can become negative around moderate QBER.
  - Mitigation in Sprint 2: enforce leakage cap + emit metrics per block; do not silently continue leaking.

- **Finite-key “Death Valley”** (roadmap RISK-003): finite-size penalties and leakage can drive effective key length to zero.
  - Mitigation in Sprint 2: fail-fast based on adjusted QBER and leakage budget; integration tests must cover this regime.

- **Ordering race conditions**: asynchronous network behavior can break commit-then-reveal if ACK ordering is not enforced.
  - Mitigation: OrderedProtocolSocket + explicit protocol invariant tests.

### Definition of Done (Sprint 2 closure conditions)

Sprint 2 is complete when all of the following are true:

1. **Ordered Messaging**: `OrderedProtocolSocket` enforces “detection report ACK before basis reveal” in simulation tests.
2. **Detection Validation**: Hoeffding/Chernoff-based validator aborts on detection anomalies with $\zeta=\sqrt{\ln(2/\varepsilon)/(2M)}$.
3. **Finite-size QBER**: $\mu$ is computed per the specified formula; abort triggers on $e_{obs}+\mu>0.22$.
4. **Leakage Safety**: LeakageSafetyManager enforces $L_{max}$; wiretap cost $|\Sigma|$ is produced for Phase IV.
5. **Integration tests**: A Phase I → II → III integration test produces:
   - a sifted key,
   - an adjusted QBER,
   - reconciliation output with explicit `wiretap_cost_bits`,
   - and no basis reveal before detection ACK.

---

### Appendix: Primary Source Links

- [Roadmap Sprint 2 tasks](master_roadmap.md#L401-L447)
- [Phase II analysis (ordered messaging, Chernoff/Hoeffding, $\mu$)](phase_II_analysis.md)
- [Phase III analysis (LDPC, wiretap cost, safety cap)](phase_III_analysis.md)
- [Finite-key penalty expression (Scarani-style Eq. (2))](../literature/Tight%20Finite-Key%20Analysis%20for%20Quantum%20Cryptography.md#L254)

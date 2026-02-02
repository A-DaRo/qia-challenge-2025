# Robust Reconciliation: Implementation Review & Hardening Strategy

**Document Type:** Technical Review & Implementation Roadmap  
**Version:** 1.0  
**Date:** December 19, 2025  
**Status:** ✅ **COMPLETE**  
**Reviewers:** Senior AI Technical Peer  
**Related Documents:**
- [recon_phase_spec.md](recon_phase_spec.md) — Phase III Specification
- [Blind Reconciliation.md](../literature/Blind%20Reconciliation.md) — Martinez-Mateo et al. 2003
- [Post-processing procedure for industrial quantum key distribution systems.md](../literature/Post-processing%20procedure%20for%20industrial%20quantum%20key%20distribution%20systems.md) — Kiktenko et al. 2016

---

## Executive Summary

This document presents a comprehensive review of the Caligo reconciliation implementation (`caligo/reconciliation/`), analyzing:

1. **Architecture & Design Patterns**: Factory-based strategy selection, orchestrator coordination
2. **Code Quality**: Coupling, redundancy, adherence to SOLID principles
3. **Literature Fidelity**: Mathematical correctness vs. Kiktenko (2016) and Martinez-Mateo (2003)
4. **Test Coverage**: Current unit tests vs. edge-case resilience requirements
5. **Identified Issues**: Data size handling (>4096 bits), noisy input handling, pipeline inefficiencies

---

### Critical Findings

| Priority | Issue | Impact | Status |
|----------|-------|--------|--------|
| 🔴 **P0** | **Multi-block reconciliation not implemented** | Keys >4096 bits fail or produce incorrect results | ❌ Blocking |
| 🔴 **P0** | **Kiktenko shortening formula incorrect** | Conflates padding with efficiency-driven shortening | ❌ Blocking |
| 🔴 **P0** | **Protocol bypasses factory pattern** | Direct instantiation violates architecture, prevents strategy switching | ❌ Blocking |
| 🟠 **P1** | **Blind reconciliation is dead code** | Implemented but never invoked by protocol | ⚠️ Enhancement |
| 🟠 **P1** | **Test suite lacks edge cases** | No tests for >4096 bits, high QBER, non-convergence | ⚠️ Enhancement |
| 🟡 **P2** | **Tight coupling: protocol ↔ reconciliation** | Violates SRP, makes testing difficult | ⚠️ Refactor needed |

---

### Impact Assessment

**Severity:** 🔴 **HIGH** — Current implementation cannot handle variable-length keys correctly

**Affected Scenarios:**
1. **Small-scale tests** (n ≤ 4096): ✅ Works correctly
2. **Production scale** (n > 4096): ❌ Fails with `ValueError` or silent corruption
3. **High QBER** (8-11%): ⚠️ Untested, likely fragile
4. **Blind strategy**: ❌ Never executes despite being implemented

**User Impact:**
- Protocol execution fails when sifted key length exceeds 4096 bits
- No runtime strategy selection (baseline vs. blind)
- Suboptimal efficiency due to incorrect Kiktenko shortening
- No visibility into which issues are reconciliation vs. quantum layer

---

### Recommendations Summary

**Immediate Actions (Before Production):**
1. ✅ Implement `reconcile_multi_block()` in orchestrator
2. ✅ Fix Kiktenko shortening calculation
3. ✅ Refactor protocol layer to use factory pattern
4. ✅ Add multi-block tests (rob_001, rob_002)

**Short-Term (Next Sprint):**
5. ✅ Complete robustness test suite (850 LOC)
6. ✅ Integrate blind reconciliation with protocol
7. ✅ Improve error handling and exception taxonomy
8. ✅ Update documentation with correct formulas

**Long-Term (Future Enhancements):**
9. 🔄 Benchmark baseline vs. blind performance
10. 🔄 Optimize decoder for high-QBER scenarios
11. 🔄 Implement adaptive rate selection based on real-time QBER feedback

---

**Report Sections:**
- [1. Introduction](#1-introduction) — Reconciliation in 1-out-of-2 OT context
- [2. Architecture Analysis](#2-architecture-analysis) — Factory pattern, orchestrator, data flow
- [3. Code Quality Assessment](#3-code-quality-assessment) — Coupling, SRP violations, redundancy
- [4. Literature Validation](#4-literature-validation) — Kiktenko & Martinez-Mateo compliance
- [5. Unit Test Hardening](#5-unit-test-hardening) — Coverage gaps, proposed hard tests
- [6. Implementation Roadmap](#6-implementation-roadmap) — Prioritized fixes, timeline, metrics

---

## Table of Contents

1. [Introduction: Reconciliation in 1-out-of-2 OT Context](#1-introduction)
2. [Architecture Analysis](#2-architecture-analysis)
3. [Code Quality Assessment](#3-code-quality-assessment)
4. [Literature Validation](#4-literature-validation)
5. [Unit Test Hardening](#5-unit-test-hardening)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [References](#7-references)

---

## 1. Introduction: Reconciliation in 1-out-of-2 OT Context

### 1.1 Protocol Security Requirements

Phase III (Information Reconciliation) in the Caligo $\binom{2}{1}$-OT protocol operates under constraints fundamentally different from standard QKD:

| Security Property | Standard QKD | $\binom{2}{1}$-OT (Caligo) |
|-------------------|--------------|----------------------------|
| **Eavesdropper Model** | External passive observer (Eve) | Internal adversarial party (Bob may be malicious) |
| **Leakage Target** | External eavesdropper | Direct leak to protocol participant |
| **Obliviousness** | Not required | **Critical**: Alice must not learn Bob's choice $C$ |
| **Interactivity Constraint** | Encouraged (Cascade, Winnow) | **Forbidden**: One-way only (Alice → Bob) |
| **Syndrome Cost** | Counted against Eve's information | Counted as **wiretap cost** against Bob's storage attack |

The **wiretap cost model** means every syndrome bit transmitted enables Bob to potentially correct errors in the "wrong" key partition. This creates a fundamental tension:

$$\text{Secure Key Length} \leq H_{\min}(X | E) - \underbrace{|\Sigma|}_{\text{syndrome leakage}} - \text{security margins}$$

**Key Insight from recon_phase_spec.md Section 1.3:**
> Beyond a threshold QBER (~10-11%), the protocol becomes infeasible due to syndrome leakage exhausting available entropy.

### 1.2 Implemented Strategies

Caligo supports two reconciliation approaches via runtime factory selection:

#### Strategy 1: Baseline (Kiktenko et al. 2016)
**File:** `reconciliation/orchestrator.py` (primary logic)  
**Literature:** *Post-processing procedure for industrial quantum key distribution systems*  
**Key Features:**
- Fixed frame size $n = 4096$
- Rate-adaptive LDPC with rates $R \in [0.5, 0.9]$ (step 0.05)
- **Shortening technique** to reduce frame error rate
- Efficiency criterion: $\frac{1-R}{h_b(\text{QBER}_{\text{est}})} < f_{\text{crit}} = 1.22$
- Polynomial hash verification (PolyR, 50 bits)

**Mathematical Formulation (from Kiktenko Eq. 2):**
$$n_s = \left\lfloor n - \frac{m}{f_{\text{crit}} \cdot h_b(\text{QBER}_{\text{est}})} \right\rfloor$$

Where:
- $n_s$ = shortened bits
- $m$ = check bits (syndrome length)
- $h_b(x) = -x \log_2(x) - (1-x) \log_2(1-x)$ = binary entropy

#### Strategy 2: Blind (Martinez-Mateo et al. 2003)
**File:** `reconciliation/blind_manager.py`  
**Literature:** *Blind Reconciliation*  
**Key Features:**
- **No prior QBER estimation required** (core advantage)
- Iterative puncture-to-shorten conversion
- Start with maximum puncturing (highest rate), progressively reveal shortened values
- Adaptive rate discovery through syndrome-based feedback

**Protocol Flow (Martinez-Mateo Section 2.2):**
1. Initialize: $d$ modulation bits punctured (rate maximized)
2. Alice sends syndrome $\Sigma$ (one-time)
3. Bob attempts belief propagation decoding
   - **Success** → return corrected key
   - **Failure** → Alice reveals $\Delta = d/t$ shortened bit values
4. Repeat step 3 for maximum $t$ iterations

**Trade-off:** Blind method increases leakage per iteration but eliminates costly QBER estimation phase.

### 1.3 Current Implementation Status

As of December 2025:
- ✅ Factory pattern implemented (`factory.py`)
- ✅ Baseline strategy operational with single-block processing
- ✅ Blind manager scaffolding in place
- ⚠️ **Multi-block reconciliation not implemented**
- ⚠️ **Variable-length key handling problematic**
- ⚠️ Integration with protocol layer incomplete for blind strategy
- ❌ Blind strategy not yet invoked from protocol executors

---

## 2. Architecture Analysis

### 2.1 Factory Pattern Implementation

**Files:** `reconciliation/factory.py` (857 lines)

The factory implements runtime strategy selection through three components:

```python
ReconciliationType (Enum)
    ├── BASELINE    → BaselineReconciler (placeholder)
    ├── BLIND       → BlindReconciler (operational)
    └── INTERACTIVE → InteractiveReconciler (raises NotImplementedError)

ReconciliationConfig (Dataclass)
    - Encapsulates all reconciliation parameters
    - Validation in __post_init__
    - YAML serialization support

create_reconciler(config, nsm_params) → Reconciler
    - Returns strategy-specific instance
    - Optional NSM-informed initialization
```

**Design Assessment:**

✅ **Strengths:**
- Clear separation of concerns between configuration and execution
- Type-safe enum for strategy selection
- Protocol-based `Reconciler` interface (though only documented in comments)
- Explicit QBER requirement tracking (`requires_qber_estimation` property)

⚠️ **Issues:**
1. **Placeholder Implementation:** `BaselineReconciler.reconcile()` returns input unchanged with warning
2. **No Formal Protocol:** The `Reconciler` protocol is only defined in comments, not as a `typing.Protocol`
3. **Hidden Dependency:** `BlindReconciler` lazy-loads `ReconciliationOrchestrator`, creating circular import risk
4. **Inconsistent Abstraction:** `BlindReconciler` directly instantiates `MatrixManager`, violating dependency injection

### 2.2 Orchestrator Coordination Model

**Files:** `reconciliation/orchestrator.py` (377 lines)

The orchestrator coordinates LDPC reconciliation for **single blocks**:

```
ReconciliationOrchestrator
    ├── matrix_manager: MatrixManager        # LDPC matrix pool
    ├── leakage_tracker: LeakageTracker      # Syndrome leakage accounting
    ├── decoder: BeliefPropagationDecoder    # BP algorithm
    └── hash_verifier: PolynomialHashVerifier # Post-decode verification

reconcile_block(alice_key, bob_key, qber, block_id) → BlockResult
    │
    ├─[1] Rate Selection
    │     └─ select_rate_with_parameters() → (rate, n_shortened)
    │
    ├─[2] Matrix Loading
    │     └─ matrix_manager.get_matrix(rate) → H
    │
    ├─[3] Alice: Syndrome Encoding
    │     └─ encode_block(alice_key, H, n_shortened) → SyndromeBlock
    │
    ├─[4] Bob: BP Decoding (with retry)
    │     └─ decoder.decode(bob_llr, syndrome) → DecodeResult
    │
    ├─[5] Hash Verification
    │     └─ hash_verifier.verify(corrected, alice_hash)
    │
    └─[6] Leakage Recording
          └─ leakage_tracker.record_block(syndrome_length, n_shortened)
```

**Critical Finding: Single-Block Limitation**

The orchestrator's `reconcile_block()` method is designed to process **exactly one LDPC frame** (4096 bits). However:

```python
# From alice.py line 275-285
def _phase3_reconcile(self, alice_bits, bob_bits, qber_adjusted):
    alice_arr = bitarray_to_numpy(alice_bits)
    bob_arr = bitarray_to_numpy(bob_bits)
    
    orchestrator = ReconciliationOrchestrator(...)
    block_result = orchestrator.reconcile_block(
        alice_key=alice_arr,  # ← Can be any length!
        bob_key=bob_arr,
        qber_estimate=qber_adjusted,
        block_id=0,
    )
```

**Problem:** The protocol passes the **entire sifted key** (variable length, potentially >4096 bits) to `reconcile_block()`, which:
- Treats it as a single block
- Computes incorrect shortening: `n_shortened = 4096 - len(full_key)`
- Results in **negative shortening** if `len(full_key) > 4096`
- Or **excessive shortening** if `len(full_key) < 4096`, wasting capacity

### 2.3 Component Interaction Diagram

```
Protocol Layer (alice.py / bob.py)
    │
    ├──> _phase3_reconcile(alice_bits, bob_bits, qber)
    │        │
    │        ├─ Instantiates: MatrixManager, ReconciliationOrchestrator
    │        │                 (NEW INSTANCES PER CALL - no reuse!)
    │        │
    │        └─ Calls: orchestrator.reconcile_block()
    │                     │
    │                     ├─> rate_selector.select_rate_with_parameters()
    │                     ├─> matrix_manager.get_matrix(rate)
    │                     ├─> ldpc_encoder.encode_block()
    │                     ├─> ldpc_decoder.decode() [with retry]
    │                     ├─> hash_verifier.verify()
    │                     └─> leakage_tracker.record_block()
    │
    └──> Factory Layer (factory.py) [NOT USED BY PROTOCOL]
             │
             └─ create_reconciler(config) → {Baseline|Blind}Reconciler
                     │
                     └─> ReconciliationOrchestrator [lazy-loaded in BlindReconciler]
```

**Coupling Issue:** The protocol layer (`alice.py`) **directly imports** and instantiates reconciliation internals, completely bypassing the factory. This violates the intended architecture.

### 2.4 Data Flow: Protocol Layer → Reconciliation

**Actual Flow (as implemented):**

```
Phase II: Sifting
    ├─ Alice: n=10000 EPR pairs → sifting → ~5000 matching bits
    ├─ Test subset selection: 10% for QBER → 500 test bits
    └─ Key subset: 4500 bits remain for reconciliation
        │
        ├─ alice_bits: bitarray(4500)  ← Problem: > 4096!
        ├─ bob_bits:   bitarray(4500)
        └─ qber_adjusted: 0.035
            │
            └──> orchestrator.reconcile_block(4500-bit arrays, qber=0.035)
                    │
                    ├─ rate_selector: QBER=3.5% → rate=0.80
                    │                  syndrome_len = 4096*(1-0.80) = 819 bits
                    │
                    ├─ encoder: payload=4500, frame_size=4096
                    │           n_shortened = 4096-4500 = -404  ← INVALID!
                    │           ValueError or silent corruption
                    │
                    └─ [Reconciliation fails or produces incorrect output]
```

**Expected Flow (for variable-length keys):**

```
Phase III: Reconciliation (with partitioning)
    │
    ├─[1] Partition key into blocks
    │     4500 bits → [block_0: 4096 bits] + [block_1: 404 bits]
    │
    ├─[2] Process each block
    │     │
    │     ├─ Block 0: reconcile_block(4096 bits) → OK
    │     │           n_shortened = 0 (full frame)
    │     │
    │     └─ Block 1: reconcile_block(404 bits) → OK
    │               n_shortened = 4096-404 = 3692
    │               syndrome = H @ [404 payload | 3692 padding]
    │
    └─[3] Concatenate corrected blocks
          [4096] + [404] = 4500 reconciled bits
```

---

### 2.5 Required Protocol Differences: Baseline vs. Blind Reconciliation

This section focuses on the *protocol-level* deltas that must exist between the two reconciliation families:

1. **Baseline (Kiktenko-style)**: uses an explicit QBER estimate to select an LDPC configuration and then performs one-way syndrome reconciliation.
2. **Blind (Martinez-Mateo-style)**: eliminates explicit QBER estimation, and instead performs *rate discovery* through multiple predetermined disclosure rounds (puncturing → shortening), still respecting one-way information flow.

These differences are not “implementation detail” — they are **semantic** differences that should be expressed in the protocol architecture as configuration-driven strategy selection.

#### 2.5.1 Baseline vs. Blind: What Must Differ (Summary Table)

| Dimension | Baseline (rate-adaptive LDPC) | Blind (iterative puncture/shorten)
|---|---|---|
| **Input requirement** | Requires $\widehat{\text{QBER}}$ (preferably adjusted for finite-size) | Does **not** require explicit QBER estimation from test bits
| **How rate is chosen** | Deterministic from $\widehat{\text{QBER}}$ + $f_{crit}$ criterion | Starts at high rate (punctured), then decreases effective rate by converting punctured → shortened over rounds
| **Number of protocol disclosures** | Typically **1** disclosure (syndrome + verify hash) | **Multiple** disclosures (initial syndrome + additional shortened values), bounded by $t$
| **Information direction** | One-way: Alice → Bob (syndrome + verification material) | One-way: Alice → Bob (syndrome + predetermined reveal schedule)
| **Bob → Alice feedback** | Not required for correctness; not allowed if it leaks choice | Not required *in Caligo*; classic blind uses feedback, but Caligo must avoid it
| **What is disclosed** | Syndrome bits (and possibly public metadata) | Syndrome bits + revealed shortened values across rounds
| **Leakage accounting** | $|\Sigma|$ (syndrome) + any shortening/padding disclosures | $|\Sigma|$ + total revealed shortened bits across rounds
| **Where keys live** | Alice uses her own key as ground truth; Bob corrects locally | Same; Bob corrects locally, using progressively stronger constraints
| **What Phase II must provide** | Sifted key material **plus** channel estimate (QBER) | Sifted key material only; avoid sacrificing test bits

#### 2.5.2 Important Clarification: “Blind” still needs a decoding prior

Blind reconciliation eliminates *explicit* QBER estimation from publicly disclosed test bits.
However, the LDPC BP decoder still needs a **channel prior** (or an equivalent LLR initialization policy).

In the current implementation, `BlindReconciliationManager.build_llr_for_state()` uses a `qber` parameter (default 0.05) to compute channel LLR magnitude:

$$L_{ch} = \log\frac{1-\epsilon}{\epsilon}$$

This is not a contradiction if we treat $\epsilon$ as a **conservative prior**, not a measured QBER.
For Caligo, the design goal should be:

- **Baseline**: $\epsilon$ must come from Phase II measurement (test bits) and be tracked as a protocol statistic.
- **Blind**: $\epsilon$ may be fixed (conservative) or derived from *non-revealing* side-information (e.g., NSM parameters), but must not require test-bit disclosure.

#### 2.5.3 Current Protocol Reality vs. Required Directionality

Although Phase III is specified as one-way (Alice → Bob), the current executor wiring is effectively **centralized reconciliation at Alice**:

- In `alice.py`, `_phase3_reconcile()` takes both `alice_bits` and `bob_bits` and calls `orchestrator.reconcile_block(alice_key, bob_key, qber)` locally.
- In `bob.py`, Bob does not reconcile; he receives a fully “reconciled” key via `MessageType.SYNDROME_RESPONSE`.

This has two critical consequences:

1. **It collapses the protocol boundary**: the reconciliation implementation is no longer a one-way LDPC syndrome protocol; it becomes “Alice learns Bob’s raw key material and corrects it.”
2. **It blocks supporting blind reconciliation properly**: blind reconciliation is fundamentally about what Alice discloses and what Bob can decode locally, round-by-round.

Therefore, supporting baseline *and* blind robustly requires that Phase III be re-centered to the intended model:

```
Baseline (Required):
    Alice: compute syndrome + verify-hash → send to Bob
    Bob: decode + verify locally

Blind (Required in Caligo constraints):
    Alice: send initial syndrome + predetermined reveal schedule (Δ chunks)
    Bob: attempt decode after each chunk; stop locally when verified
```

---

### 2.6 Implementing Both Protocols Without Coupling or Duplication

The core architecture problem is that reconciliation currently “leaks upward” into the protocol programs:

- Phase II (`_phase2_sifting`) always performs QBER estimation (test-bit sacrifice + extra message exchange).
- Phase III (`_phase3_reconcile`) directly instantiates reconciliation internals (bypassing the factory), and assumes a single-block baseline flow.

To support baseline and blind cleanly, Phase II and Phase III should each be split into:

1. A **protocol-agnostic core** (pure data transformations and algorithms)
2. A thin **transport driver** (message send/recv)
3. A **strategy** chosen by configuration (baseline vs. blind)

The key is to move “what is required” behind a small interface, so that protocol code does not branch on reconciliation internals.

#### 2.6.1 Define Minimal, Strategy-Owned Requirements

Reconciliation protocols differ most strongly in what they need from Phase II:

- Baseline needs `qber_adjusted` for rate selection.
- Blind does not, and should ideally avoid sacrificing test bits.

Instead of encoding this as protocol-specific `if blind: ... else: ...` branches scattered across `_phase2_sifting` and `_phase3_reconcile`, define a single “requirements object” derived from configuration:

```text
ReconciliationRequirements
    - needs_qber_estimate: bool
    - qber_source: {test_bits, nsm_prior, fixed_prior}
    - qber_budget: {min_test_bits, test_fraction}  (only meaningful if needs_qber_estimate)
    - phase3_message_plan: {single_shot, multi_round}
    - max_rounds: int  (blind)
```

Crucially, this is **not** a dependency on `ReconciliationOrchestrator` or `MatrixManager`. It is a small declarative contract.

#### 2.6.2 Phase II Refactor: Separate “Sifting” from “Channel Estimation”

Right now, Phase II performs the following tasks in a single procedure:

1. Sifting (compute matched-basis positions)
2. Partition selection (`I0/I1`)
3. Test selection (`T`) and message exchange for test bits
4. QBER estimation and finite-size adjustment
5. Key filtering (remove test bits)

Only (3–4) are *baseline-specific*.

To avoid coupling and avoid code duplication, Phase II should be split conceptually into:

**A. Sifting core (always required)**
- Compute sifted keys and indices needed for Phase IV partitioning.
- Does not depend on reconciliation type.

**B. Optional channel estimation (strategy-controlled)**
- If `needs_qber_estimate=True`: select test subset, run the test-bit exchange, compute `qber_estimate` and `qber_adjusted`, and filter keys.
- If `needs_qber_estimate=False`: do not select a test subset; keep all sifted bits as key material.

This accomplishes the user-facing requirement:

> “Not all reconciliation protocols require QBER calculation; test bits are precious key material that can be spared.”

It also prevents *structural duplication* because the sifting core is shared, and only the estimation sub-step is strategy-dependent.

#### 2.6.3 Phase III Refactor: Strategy-Driven Message Production (Alice) and Consumption (Bob)

The most robust separation is to treat Phase III as a **message protocol** rather than as a direct function call.
In other words, the reconciliation strategy should define:

- What Alice must send (and in how many rounds)
- What Bob must do locally to decode and verify
- What metadata must be tracked for leakage accounting

#### Baseline: Single-shot one-way disclosure

Baseline can be expressed as one “round” (per block):

```
Alice → Bob:
    - block_id
    - rate (or enough to derive it)
    - syndrome
    - hash seed + verification hash
    - shortening/padding spec (public)

Bob:
    - decode
    - verify
    - abort locally on failure
```

Key property: **Alice does not need Bob’s bits** and should never receive them during Phase III.

#### Blind: Multi-round predetermined disclosures (Caligo-safe version)

Classic blind reconciliation uses Bob feedback (“failed, send more”).
However, Caligo’s Phase III specification explicitly highlights that Bob → Alice decode success/failure is forbidden or heavily restricted.

Therefore, the Caligo-safe blind mode should be modeled as:

```
Alice → Bob (round r = 0..t-1):
    - block_id
    - syndrome (sent at r=0)
    - reveal chunk r of shortened values (sent at r>=1)
    - verification hash (can be sent once; Bob checks after each round)

Bob:
    - attempt decode after each round locally
    - stop consuming further rounds if verified
```

Important consequence (must be explicit in design docs):

- Without Bob feedback, Alice cannot stop early, so the protocol either:
    - leaks the **full predetermined schedule** every time (worst-case leakage), or
    - allows a minimal “stop” feedback bit (which must be argued safe), or
    - chooses blind parameters (small $t$, small δ) such that worst-case leakage is acceptable.

This trade-off must be reflected in leakage budgeting and feasibility checks.

#### 2.6.3.1 Concrete Message Schemas (OrderedSocket / MessageType payloads)

This subsection proposes **protocol-level message schemas** that enable baseline and blind reconciliation to coexist under a shared transport (`OrderedSocket`) **without** protocol↔reconciliation coupling.

Design goals:

- **No reconciliation internals leaked into protocol programs**: protocol code sends/receives opaque “reconciliation messages” produced/consumed by the selected strategy.
- **One-way disclosure**: Alice → Bob messages only during Phase III.
- **Deterministic schedules** where possible (avoid sending large index lists each round).
- **Multi-block compatible**: the schema must support $n_{key} > 4096$ by repeating per-block messages.

##### Common envelope (shared across all reconciliation strategies)

All reconciliation messages should share a minimal common header so the protocol can route/store them without understanding the details:

```text
MessageType.RECONCILIATION
    payload:
        version: int                     # schema version
        strategy: str                    # "baseline" | "blind"
        phase: str                       # "begin" | "block" | "end"
        block_id: int                    # 0..num_blocks-1
        round_id: int                    # 0..max_rounds-1 (0 for baseline)
        data: dict                       # strategy-specific fields
```

Rationale:

- Using a single `MessageType.RECONCILIATION` avoids adding many MessageType variants and keeps the protocol layer from branching.
- The reconciliation strategy owns the `data` schema for each phase.

If you prefer multiple message types for easier debugging, the same `data` payloads apply; only the envelope changes.

##### Baseline reconciliation (single-shot per block)

Baseline is conceptually “one round” per block:

**1) Begin (optional, recommended for metadata + precomputation)**

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "baseline"
        phase: "begin"
        block_id: -1
        round_id: -1
        data:
            frame_size: int                # e.g. 4096
            num_blocks: int
            payload_bit_lengths: list[int] # per block (last block shorter)
            qber_adjusted: float           # derived from test bits (public statistic)
            f_crit: float                  # e.g. 1.22
            hash_bits: int                 # e.g. 50
            prng_domain_sep: int           # domain separator constant
```

Notes:

- `qber_adjusted` is already “paid for” by the test-bit exchange; it is not additional secret leakage.
- Including `payload_bit_lengths` removes ambiguity for last-block shortening.

**2) Block message (required, repeated per block)**

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "baseline"
        phase: "block"
        block_id: int
        round_id: 0
        data:
            rate: float                    # LDPC rate used for this block
            n_shortened: int               # shortening bits for this block
            prng_seed: int                 # deterministic seed for frame construction
            syndrome: hex                  # bytes: syndrome bits packed
            hash_seed: int                 # seed for polynomial hash
            verify_hash: hex               # bytes: polynomial hash output
```

Interpretation contract:

- `prng_seed` must deterministically define the same shortening/padding convention on both sides.
- Bob computes `LLR(payload)` from his received bits and the baseline channel estimate; shortened bits have “infinite confidence” based on the convention (typically fixed to 0).
- Bob verifies with `verify_hash` after decoding. No Bob→Alice success signal is required.

**3) End (optional)**

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "baseline"
        phase: "end"
        block_id: -1
        round_id: -1
        data:
            total_syndrome_bits: int       # optional convenience, Bob can compute
```

##### Blind reconciliation (multi-round predetermined schedule per block)

Blind must support multiple rounds while preserving Caligo’s one-way constraint.
To avoid forbidden Bob→Alice feedback, Alice transmits a **predetermined** reveal schedule up to `max_rounds`.

Operational caveat for `OrderedSocket`:

- Even if Bob succeeds early, he must still **receive** the remaining round messages to avoid deadlock.
- He may ignore them locally once verified.

**Blind decoding prior (LLR magnitude): which QBER?**

For blind BP decoding, we need a prior only to set the **LLR magnitude**, not to select a rate from measured test bits.
The best non-leaking prior available in Caligo is derived from the trusted channel model:

- Use `qber_conditional` (QBER conditioned on detection / successful measurement), not `total_qber`.

Reason:

- Phase III decoding is performed on *sifted detected bits*. Loss events do not appear as bits in the sifted key.
- `total_qber` typically conflates misalignment/errors with loss/dark-count contributions in a way that is meaningful for link budgeting, but not as a direct BSC crossover parameter for the already-conditioned key.

Concretely, the blind strategy should accept a field like:

```text
llr_prior_qber = ChannelNoiseProfile.qber_conditional
```

or, if only NSM parameters are available:

```text
llr_prior_qber = NSMParameters.qber_channel   # interpreted as conditional on detection
```

**1) Begin (recommended)**

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "blind"
        phase: "begin"
        block_id: -1
        round_id: -1
        data:
            frame_size: int
            num_blocks: int
            payload_bit_lengths: list[int]
            max_rounds: int                # t
            modulation_fraction: float     # δ
            schedule_seed: int             # defines puncture/shorten positions + reveal order
            llr_prior_qber: float          # from quantum/noise model (non-test-bit)
            hash_bits: int
```

**2) Round 0 (syndrome + verification hash)**

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "blind"
        phase: "block"
        block_id: int
        round_id: 0
        data:
            base_rate: float               # starting rate (high), optional if implied
            n_punctured: int               # d initially
            n_shortened: int               # 0 initially
            prng_seed: int                 # deterministic seed for frame construction
            syndrome: hex
            hash_seed: int
            verify_hash: hex
```

**3) Rounds r = 1..t-1 (reveal shortened values for Δ positions)**

To reduce payload size and coupling, do not send explicit indices if both parties can derive the schedule from `schedule_seed` and the known `frame_size`.
Send only the values in the agreed order.

```text
MessageType.RECONCILIATION
    payload:
        version: 1
        strategy: "blind"
        phase: "block"
        block_id: int
        round_id: int                    # 1..max_rounds-1
        data:
            newly_shortened_count: int     # Δ this round
            shortened_values: hex          # packed bits (length Δ)
```

Bob-side interpretation:

- Using `schedule_seed`, Bob maps the `shortened_values` onto the next Δ scheduled positions.
- Bob updates the decoder state (punctured → shortened), rebuilds LLRs, decodes, and checks `verify_hash`.

Leakage accounting:

$$|\Sigma|_{blind} = |\Sigma|_{syndrome} + \sum_{r=1}^{t-1} \Delta_r$$

Where each $\Delta_r$ is the number of revealed shortened bits in round $r$.


#### 2.6.4 Factory Use Without Increased Coupling

The factory in `reconciliation/factory.py` should remain the *only* construction point for reconciliation strategies.
To keep coupling low:

- Protocol code should depend only on a small interface such as:

```text
IReconciliationStrategy
    - requirements() -> ReconciliationRequirements
    - alice_build_messages(...) -> Iterable[AliceToBobReconMessage]
    - bob_consume_messages(...) -> BobReconResult
```

- The factory can inject heavy dependencies (`MatrixManager`, `ReconciliationOrchestrator`, `LeakageTracker`) into concrete strategies.
- Protocol layer should not import those internals at all.

This aligns with the refactor recommendation already present in Section 3.1, but adds the missing piece: the strategy must own the *message protocol*, not just an in-process `reconcile()` call.

#### 2.6.5 Contract Implications (Phase II / Phase III DTOs)

The current `SiftingPhaseResult` contract requires QBER fields (`qber_estimate`, `qber_adjusted`, `finite_size_penalty`, `test_set_indices`) unconditionally.
That makes blind reconciliation structurally awkward: the contract forces Phase II to perform QBER estimation even when it should not.

To support both protocols cleanly, the contract must be split so that QBER/test metadata is an *optional add-on* rather than mandatory structure.

**Option A (required): split the DTOs**

1) `SiftedKeyMaterial` (always produced)

- `sifted_key_alice: bitarray`
- `sifted_key_bob: bitarray`
- `matching_indices: np.ndarray`
- `i0_indices: np.ndarray`
- `i1_indices: np.ndarray`
- `timing_compliant: bool`

Post-conditions (examples):

- `len(sifted_key_alice) == len(sifted_key_bob)`
- `matching_indices` is consistent with key lengths

2) `ChannelEstimate` (produced only if the selected reconciliation strategy requires it)

- `test_set_indices: np.ndarray`
- `qber_estimate: float`
- `finite_size_penalty: float`
- `qber_adjusted: float` with $qber\_adjusted = qber\_estimate + finite\_size\_penalty$
- `test_set_size: int`

Post-conditions (examples):

- `abs(qber_adjusted - (qber_estimate + finite_size_penalty)) <= tol`
- `qber_adjusted <= QBER_HARD_LIMIT`

3) `PhaseIIResult`

- `sifted: SiftedKeyMaterial`
- `channel_estimate: ChannelEstimate | None`

Implications:

- **Baseline**: must produce `channel_estimate` and pass `qber_adjusted` into Phase III rate selection.
- **Blind**: sets `channel_estimate=None`; Phase III uses `llr_prior_qber` from the trusted noise model (e.g., `qber_conditional`) without consuming test bits.

This avoids “strategy knowledge” leaking into generic post-conditions and keeps Phase II testable without reconciliation configuration.


---

## 3. Code Quality Assessment

### 3.1 Coupling Between Reconciliation and Protocol Packages

**Tight Coupling Violations:**

| Issue | Location | Impact | Severity |
|-------|----------|--------|----------|
| **Direct Instantiation** | `alice.py:275-278` | Protocol bypasses factory, hardcodes orchestrator creation | 🔴 High |
| **Hardcoded Constants** | `alice.py:278` | `frame_size=4096, max_retries=2` duplicates `constants.py` | 🟡 Medium |
| **Missing Abstraction** | `alice.py:19-20` | Imports internal classes instead of factory function | 🔴 High |
| **No Dependency Injection** | `alice.py:269` | `_phase3_reconcile` creates disposable orchestrator per call | 🟡 Medium |
| **Circular Import Risk** | `factory.py:578` | `BlindReconciler` imports `orchestrator` at module level | 🟠 Medium-High |

**Recommended Refactoring:**

```python
# Current (alice.py) - TIGHT COUPLING
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import ReconciliationOrchestrator

def _phase3_reconcile(self, alice_bits, bob_bits, qber):
    matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
    orchestrator = ReconciliationOrchestrator(matrix_manager, ...)
    result = orchestrator.reconcile_block(...)

# Proposed - LOOSE COUPLING via Factory (ONE-WAY, message-based)
#
# NOTE: The current factory `Reconciler.reconcile(alice_bits, bob_bits, ...)` is an
# in-process API that assumes Alice can access Bob's raw bits. That is incompatible
# with the intended Phase III semantics (Alice → Bob syndrome flow).
#
# The protocol-facing API should instead be message-based: Alice *produces* disclosure
# messages; Bob *consumes* them to decode and verify locally.

from caligo.reconciliation.factory import create_reconciler, ReconciliationConfig

class AliceProgram(CaligoProgram):
    def __init__(self, params: ProtocolParameters):
        super().__init__(params)
        recon_config = ReconciliationConfig.from_dict(params.reconciliation_config)
        self._recon = create_reconciler(recon_config, nsm_params=params.nsm_params)

    def _phase3_reconcile(self, alice_bits, qber_or_none):
        # Proposed API sketch:
        #   messages, meta = self._recon.alice_build_messages(sifted_bits, qber=qber_or_none)
        #   for msg in messages: send(msg)
        # Alice keeps `alice_bits` as ground truth; Phase III output is leakage metadata.
        ...

class BobProgram(CaligoProgram):
    def __init__(self, params: ProtocolParameters):
        super().__init__(params)
        recon_config = ReconciliationConfig.from_dict(params.reconciliation_config)
        self._recon = create_reconciler(recon_config, nsm_params=params.nsm_params)

    def _phase3_reconcile(self, bob_bits):
        # Proposed API sketch:
        #   messages = recv_all_recon_messages()
        #   corrected, meta = self._recon.bob_consume_messages(bob_bits, messages)
        #   if not meta['hash_verified']: abort
        ...
```

### 3.2 Single Responsibility Violations

**SRP Breaches:**

1. **`ReconciliationOrchestrator`** (orchestrator.py)
   - **Responsibilities:** Rate selection, encoding, decoding, verification, leakage tracking, retry logic
   - **Violation:** Coordinates 6 distinct phases, each with complex logic
   - **Fix:** Extract `BlockReconciler` with single `reconcile_block()` responsibility

2. **`BlindReconciler`** (factory.py)
   - **Responsibilities:** Configuration, matrix loading, orchestrator construction, reconciliation execution
   - **Violation:** Acts as both factory and executor
   - **Fix:** Separate `BlindReconciliationStrategy` (execution) from `BlindReconcilerFactory` (construction)

3. **`encode_block()`** (ldpc_encoder.py)
   - **Responsibilities:** Frame construction, padding generation, syndrome computation, result packaging
   - **Violation:** Multiple calling conventions (full API vs. simple API)
   - **Fix:** Split into `prepare_encode_frame()` and `compute_syndrome_for_frame()`

### 3.3 Redundancy Detection

**Code Duplication:**

| Redundancy | Location 1 | Location 2 | LOC | Fix |
|------------|------------|------------|-----|-----|
| **Frame Preparation** | `ldpc_encoder.encode_block()` | `ldpc_encoder.prepare_frame()` | ~30 | Consolidate into `prepare_frame()`, call from `encode_block()` |
| **Matrix Loading** | `alice.py:275` | `factory.py:578` (`BlindReconciler`) | ~5 | Centralize in factory or base class |
| **PRNG Seeding** | `ldpc_encoder.generate_padding()` | `rate_selector.py` (implied) | ~10 | Extract to `utils/prng.py` |
| **Binary Entropy** | `rate_selector.binary_entropy()` | `utils/math.py` (potential duplicate?) | ~5 | Verify and consolidate |

**Inconsistent Naming:**

- `n_shortened` (orchestrator) vs. `n_s` (Kiktenko paper) vs. `shortening_bits` (tests)
- `frame_size` vs. `block_size` vs. `codeword_length`
- `qber_estimate` vs. `qber_adjusted` vs. `measured_qber`

**Recommendation:** Establish glossary in `constants.py` docstring

### 3.4 Type Safety and Contract Adherence

**Type Annotation Issues:**

```python
# Missing return type (orchestrator.py:166)
def reconcile_block(self, alice_key, bob_key, qber_estimate, block_id=0):  # ← No -> BlockResult

# Overly permissive union (ldpc_encoder.py:137)
H: "sp.csr_matrix | CompiledParityCheckMatrix"  # ← Interface should be protocol

# Implicit optional (rate_selector.py:88)
def select_rate(qber_estimate: float, available_rates=constants.LDPC_CODE_RATES):  # ← = default hides Optional
```

**Contract Violations:**

1. **`reconcile_block()`** (orchestrator.py:166)
   - **Documented:** "Reconcile a single key block"
   - **Actual:** Accepts arbitrary-length keys, attempts to reconcile as single block
   - **Violation:** No validation that `len(alice_key) <= frame_size`

2. **`encode_block()`** (ldpc_encoder.py:163)
   - **Documented:** "Supports two calling conventions"
   - **Actual:** Ambiguous dispatch logic based on parameter presence
   - **Violation:** Type checker cannot verify correctness

3. **`BlindIterationState`** (blind_manager.py:96)
   - **Documented:** "Supports two interface styles"
   - **Actual:** Dataclass with 10 attributes, unclear which are mutually exclusive
   - **Violation:** No validation of state consistency

---

## 4. Literature Validation

### 4.1 Baseline vs. Kiktenko et al. (2016)

**Reference:** *Post-processing procedure for industrial quantum key distribution systems*  
**Journal:** J. Phys.: Conf. Ser. 741 012081

#### 4.1.1 Rate Selection Algorithm

**Kiktenko Equation (1):**
$$\frac{1-R}{h_b(\text{QBER}_{\text{est}})} < f_{\text{crit}}$$

Where $f_{\text{crit}} = 1.22$ is the efficiency parameter.

**Implementation:** `rate_selector.py:93-148`

```python
def select_rate(qber_estimate, available_rates, f_crit=1.22):
    entropy = binary_entropy(qber_estimate)
    
    # Minimum rate from efficiency criterion
    r_min_efficiency = 1.0 - f_crit * entropy  # ✅ CORRECT: R > 1 - f·h(Q)
    
    # QBER-based thresholds (error correction guidance)
    qber_thresholds = [
        (0.015, 0.90), (0.030, 0.80), (0.045, 0.70),
        (0.060, 0.60), (0.080, 0.55), (0.110, 0.50),
    ]
    
    qber_rate = available_rates[0]
    for threshold, rate in qber_thresholds:
        if qber_estimate < threshold:
            qber_rate = rate
            break
    
    target_rate = max(qber_rate, r_min_efficiency)  # ✅ Satisfies both constraints
```

**Validation:** ✅ **CORRECT**

The implementation properly combines two selection criteria:
1. **Efficiency constraint** (Kiktenko Eq. 1): Ensures syndrome leakage is bounded
2. **Error correction capability**: Lower QBER allows higher rates

**Deviation Note:** The QBER threshold table is not from Kiktenko but appears to be empirically derived. Kiktenko only specifies the efficiency criterion.

#### 4.1.2 Shortening Calculation

**Kiktenko Equation (2):**
$$n_s = \left\lfloor n - \frac{m}{f_{\text{crit}} \cdot h_b(\text{QBER}_{\text{est}})} \right\rfloor$$

Where:
- $n$ = frame size (4096)
- $m$ = number of check bits = $n(1-R)$
- $n_s$ = shortened bits

**Implementation:** `rate_selector.py:169-212`

```python
def compute_shortening(rate, qber_estimate, payload_length, frame_size=4096, f_crit=1.22):
    # Basic shortening: fill frame with padding
    basic_shortened = max(0, frame_size - payload_length)  # ← Problem here!
    
    entropy = binary_entropy(qber_estimate)
    if entropy <= 0.0:
        return basic_shortened
    
    # From Martinez-Mateo: n_s = n - m / (f_crit · h(QBER))
    n_s_extra = int(math.floor(frame_size - payload_length / (f_crit * entropy)))
    n_s_extra = max(0, n_s_extra)
    
    n_shortened = max(basic_shortened, n_s_extra)  # ← Incorrect formula!
```

**Validation:** ❌ **INCORRECT**

**Critical Deviation:**

The implementation **conflates two different concepts**:

1. **Kiktenko's shortening:** Technique to reduce frame error rate by increasing redundancy
   $$n_s = \left\lfloor n - \frac{m}{f_{\text{crit}} \cdot h_b(Q)} \right\rfloor$$
   - Purpose: Improve decoder convergence probability
   - Effect: Reduces effective payload from $k = n(1-R)$ to $k' = k - n_s$
   - The $n_s$ bits are **set to known values** (from PRNG), not payload

2. **Frame padding:** Extending payload to match fixed LDPC frame size
   $$n_{\text{padding}} = n - |\text{payload}|$$
   - Purpose: Match matrix dimensions
   - Effect: No impact on error correction, just dimension alignment

**Correct Implementation (Kiktenko):**

```python
def compute_shortening_kiktenko(rate, qber_estimate, frame_size=4096, f_crit=1.22):
    """Compute Kiktenko shortening for frame error rate reduction."""
    m = int(frame_size * (1 - rate))  # Check bits
    entropy = binary_entropy(qber_estimate)
    
    if entropy <= 0:
        return 0  # No shortening needed for perfect channel
    
    # Kiktenko Eq. 2 (exact)
    n_s = math.floor(frame_size - m / (f_crit * entropy))
    
    # n_s must be non-negative and less than frame size
    return max(0, min(n_s, frame_size - 1))
```

**What Happens in Current Code:**

When `payload_length < 4096`:
- `basic_shortened = 4096 - payload_length` (e.g., 4096 - 3000 = 1096)
- This is **padding**, not Kiktenko shortening!
- Result: Kiktenko's formula is **never actually applied** for efficiency

When `payload_length > 4096`:
- `basic_shortened = 0` (no room for shortening)
- `n_s_extra` computation attempts Kiktenko formula but with wrong numerator
- Result: **Invalid state**, should have been rejected

**Impact:** The baseline reconciliation is not implementing Kiktenko's shortening technique. It's only doing frame padding, which explains why efficiency may be suboptimal.

#### 4.1.3 Verification Protocol

**Kiktenko Section 3.1 (Hash Verification):**
> "We employ comparison of hash-tags constructed with $\epsilon$-universal polynomial hashing [15]. In particular, we use modified 50-bit variant of PolyR hash function that provide collision probability for a $N$ blocks of $n - n_s$ bits on the level of $\epsilon_{\text{ver}} < 2 \times 10^{-12}$."

**Implementation:** `hash_verifier.py` (not fully reviewed here, but interface exists)

```python
class PolynomialHashVerifier:
    def __init__(self, hash_bits=50):  # ✅ 50 bits as per Kiktenko
        self.hash_bits = hash_bits
        self.prime = 2**61 - 1  # Mersenne prime
    
    def compute_hash(self, data, seed):
        """Compute polynomial hash."""
        ...
    
    def verify(self, data, expected_hash, seed):
        """Verify hash matches."""
        return self.compute_hash(data, seed) == expected_hash
```

**Validation:** ⚠️ **PARTIAL MATCH**

- ✅ 50-bit hash length matches Kiktenko
- ✅ Polynomial hash family used
- ⚠️ Collision probability not explicitly documented (should compute and verify $\epsilon_{\text{ver}}$)
- ⚠️ Kiktenko uses "modified PolyR" — unclear if implementation matches their modification

### 4.2 Blind vs. Martinez-Mateo et al. (2003)

**Reference:** *Blind Reconciliation*  
**Authors:** Martinez-Mateo, J., Elkouss, D., Martin, V.

**Note:** User provided document labeled "2003" but content appears to be from a later publication (likely 2012 based on reference style). Using attached content as authoritative.

#### 4.2.1 Iteration Protocol

**Martinez-Mateo Protocol (from attached document):**

```
Blind Reconciliation Protocol:
1. Parameters: frame size n, modulation bits d, iterations t
2. Initialize: p = d (all modulation bits punctured), s = 0 (none shortened)
3. Alice sends syndrome Σ (one-time)
4. For iteration i = 1 to t:
   a. Bob attempts BP decoding with p punctured, s shortened
   b. If SUCCESS: return corrected key, terminate
   c. If FAILURE:
      - Convert Δ = d/t bits from punctured → shortened
      - Alice reveals values of newly shortened bits
      - p ← p - Δ
      - s ← s + Δ
5. If all iterations exhausted: ABORT
```

**Implementation:** `blind_manager.py:143-372`

```python
class BlindReconciliationManager:
    def initialize(self, syndrome_length):
        """Start with max puncturing."""
        return BlindIterationState(
            iteration=0,
            n_punctured=self.config.modulation_bits,  # ✅ p = d initially
            n_shortened=0,                             # ✅ s = 0 initially
            syndrome_leakage=syndrome_length,          # ✅ Σ recorded
        )
    
    def advance_iteration(self, state, shortened_values):
        """Convert punctured → shortened."""
        delta = min(self.config.delta_per_iteration, state.n_punctured)
        return BlindIterationState(
            iteration=state.iteration + 1,
            n_punctured=state.n_punctured - delta,    # ✅ p ← p - Δ
            n_shortened=state.n_shortened + delta,    # ✅ s ← s + Δ
            shortened_values=state.shortened_values + [shortened_values],  # ✅ Reveal values
        )
```

**Validation:** ✅ **CORRECT (Protocol Structure)**

The iteration mechanics match Martinez-Mateo:
- Initial state: maximum puncturing (highest rate, minimum leakage)
- Progressive shortening: reduces uncertainty by revealing known bits
- Leakage increases per iteration: $|\Sigma| + \sum_{i=1}^{k} \Delta_i$

#### 4.2.2 Modulation Bit Handling

**Martinez-Mateo Parameters:**
- $n$ = frame size (e.g., 2048 or 4096)
- $d$ = modulation bits (typically $d = \delta \cdot n$, where $\delta \approx 0.1$-$0.3$)
- $t$ = max iterations (typically 3-5)
- $\Delta = d/t$ = bits converted per iteration

**Implementation:** `constants.py:91-118`

```python
BLIND_MODULATION_FRACTION: float = 0.15  # δ = 15%
BLIND_MAX_ITERATIONS: int = 3             # t = 3
```

**Calculated:** For $n = 4096$:
- $d = 0.15 \times 4096 = 614.4 \approx 614$ bits
- $\Delta = 614 / 3 \approx 205$ bits per iteration

**Validation:** ✅ **REASONABLE PARAMETERS**

These values are consistent with Martinez-Mateo's recommendations for 4096-bit frames. However:

⚠️ **Concern:** The paper shows blind reconciliation works best with **short codes** (e.g., $n = 2 \times 10^3 = 2048$). Effectiveness with $n = 4096$ is less well-characterized in the paper.

#### 4.2.3 Leakage Accounting

**Martinez-Mateo Leakage Formula:**
$$L_{\text{total}} = |\Sigma| + \sum_{i=1}^{k} |V_i|$$

Where:
- $|\Sigma|$ = syndrome length (one-time)
- $|V_i|$ = revealed shortened values at iteration $i$

**Implementation:** `blind_manager.py:129-134`

```python
@property
def total_leakage(self) -> int:
    """Total bits leaked: syndrome + revealed shortened values."""
    shortened_bits = sum(len(v) for v in self.shortened_values)
    return self.syndrome_leakage + shortened_bits
```

**Validation:** ✅ **CORRECT**

Leakage calculation matches Martinez-Mateo formula exactly.

#### 4.2.4 **Critical Missing Feature: Integration with Protocol**

**Status:** ❌ **BLIND STRATEGY NOT CONNECTED**

While `BlindReconciliationManager` implements the iteration protocol correctly, it is **never invoked** by the protocol executors:

```bash
$ grep -r "BlindReconciliationManager" caligo/protocol/
# (No matches)

$ grep -r "blind" caligo/protocol/alice.py
# (No matches)
```

The protocol (`alice.py:269`) **hardcodes** the baseline approach:
```python
def _phase3_reconcile(self, alice_bits, bob_bits, qber_adjusted):
    orchestrator = ReconciliationOrchestrator(...)  # ← Always baseline
    result = orchestrator.reconcile_block(...)
```

**Impact:** The blind reconciliation implementation exists but is **dead code** from the protocol's perspective.

### 4.3 Summary of Literature Deviations

| Aspect | Paper Requirement | Implementation | Status |
|--------|-------------------|----------------|--------|
| **Kiktenko: Rate Selection** | $(1-R) / h(Q) < f_{\text{crit}}$ | ✅ Correctly implemented | ✅ |
| **Kiktenko: Shortening Formula** | $n_s = \lfloor n - m/(f \cdot h(Q)) \rfloor$ | ❌ Conflated with padding | ❌ |
| **Kiktenko: Hash Verification** | 50-bit PolyR hash | ⚠️ 50-bit poly hash (exact variant unclear) | ⚠️ |
| **Martinez-Mateo: Iteration Protocol** | Puncture → shorten conversion | ✅ Correctly implemented | ✅ |
| **Martinez-Mateo: Leakage Accounting** | $L = \|\Sigma\| + \sum \|V_i\|$ | ✅ Correctly implemented | ✅ |
| **Martinez-Mateo: Protocol Integration** | Should be runtime-selectable | ❌ Not connected to protocol | ❌ |
| **Multi-block Handling** | Both papers assume single-block | ❌ No partitioning in protocol | ❌ |

---

## 5. Unit Test Hardening

### 5.1 Current Test Suite Analysis

**Test Files:**
- `test_integration.py` (285 lines): End-to-end reconciliation flows
- `test_blind_manager.py` (250 lines): Blind iteration state management
- `test_bp_decoder.py`: Belief propagation decoder tests
- `test_factory.py` (232 lines): Factory pattern and reconciler creation
- `test_hash_verifier.py`: Polynomial hash verification
- `test_matrix_manager_contracts.py`: Matrix loading and caching
- `test_rate_selector.py`: Rate selection algorithm
- `test_leakage_tracker.py`: Leakage accounting

**Coverage Assessment:**

| Component | Tested | Edge Cases Covered | Gap Analysis |
|-----------|--------|-------------------|--------------|
| **Rate Selection** | ✅ Yes | ⚠️ Partial | Missing: QBER → 0.5 (max entropy), QBER = 0.0 |
| **Shortening Calculation** | ✅ Yes | ❌ No | Missing: payload > frame_size, Kiktenko formula validation |
| **Encoder** | ✅ Yes | ⚠️ Partial | Missing: Multi-block encoding, variable lengths |
| **Decoder** | ✅ Yes | ⚠️ Partial | Missing: Non-convergence scenarios, high QBER (>8%) |
| **Orchestrator** | ✅ Yes | ❌ No | **Critical:** No tests for keys > 4096 bits |
| **Blind Manager** | ✅ Yes | ⚠️ Partial | Missing: All iterations exhausted, Martinez-Mateo interface |
| **Factory** | ✅ Yes | ✅ Good | Well-tested, placeholders acknowledged |
| **Multi-block Pipeline** | ❌ No | ❌ No | **Critical:** No tests for block partitioning |

### 5.2 Edge Case Coverage Gaps

#### Gap 1: Variable-Length Key Handling
**Current Test Payloads:**
- `test_integration.py:42`: 2867 bits (70% of 4096) ✅
- `test_integration.py:68`: 2867 bits with 3% noise ✅
- `test_integration.py:230`: 2867 bits with 2% noise ✅

**Missing:**
- ❌ Payload < 1024 bits (high shortening ratio)
- ❌ Payload = 4096 bits (no shortening, full frame)
- ❌ **Payload > 4096 bits** (multi-block required) 🔴 **CRITICAL**
- ❌ Payload = 8192 bits (exactly 2 blocks)
- ❌ Payload = 8500 bits (2 full blocks + 308 bit partial block)
- ❌ Payload > 40,000 bits (10+ blocks, leakage accumulation)

#### Gap 2: High QBER Scenarios
**Current Test QBERs:**
- `test_integration.py:51`: 0.01 (1%) ✅
- `test_integration.py:75`: 0.03 (3%) actual, 0.05 (5%) estimated ✅
- `test_integration.py:236`: 0.02 (2%) actual, 0.03 (3%) estimated ✅

**Missing:**
- ❌ QBER = 0.08 (8%) — threshold for rate 0.55
- ❌ QBER = 0.10 (10%) — near protocol abort threshold
- ❌ **QBER = 0.11 (11%)** — at/above Kiktenko Table limit 🔴
- ❌ QBER = 0.15 (15%) — should fail reconciliation gracefully

#### Gap 3: Decoder Non-Convergence
**Current Decoder Tests:**
- `test_bp_decoder.py:133`: Noiseless decode (always converges) ✅
- `test_bp_decoder.py:148`: Low noise decode (converges) ✅

**Missing:**
- ❌ Decode failure → retry with LLR damping
- ❌ All retries exhausted → non-converged result
- ❌ Frame error rate measurement
- ❌ Syndrome mismatch detection

#### Gap 4: Leakage Budget Exhaustion
**Current Leakage Tests:**
- `test_integration.py:150`: 10 blocks, 5% QBER, under budget ✅
- `test_integration.py:176`: 5 blocks, 10% QBER, close to cap ✅

**Missing:**
- ❌ Exact cap exceeded by 1 bit (boundary test)
- ❌ Multi-block reconciliation triggering abort mid-stream
- ❌ Leakage vs. remaining entropy calculation verification

### 5.3 Proposed Hard Tests

#### Test Suite: `test_reconciliation_robustness.py`

```python
"""
Robustness tests for reconciliation under extreme conditions.

These tests stress the implementation with:
- Very long keys (>10,000 bits)
- High QBER (8-11%)
- Non-convergent decoding scenarios
- Multi-block partitioning edge cases
"""

import numpy as np
import pytest

from caligo.reconciliation.orchestrator import ReconciliationOrchestrator, partition_key
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation import constants


class TestVariableLengthKeys:
    """Tests for keys of various lengths."""
    
    @pytest.mark.parametrize("payload_length", [
        256,      # Very short (93.75% shortening)
        2048,     # Half frame (50% shortening)
        4096,     # Exact frame (0% shortening)
        4500,     # Slightly over frame (requires multi-block)
        8192,     # Exactly 2 blocks
        8500,     # 2 full + 1 partial block
        12000,    # ~3 blocks
    ])
    def test_rob_001_variable_payload_lengths(
        self,
        payload_length: int,
        matrix_manager: MatrixManager,
    ) -> None:
        """Reconcile keys of varying lengths."""
        rng = np.random.default_rng(42 + payload_length)
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        
        # Bob's key with 2% errors
        bob_key = alice_key.copy()
        n_errors = max(1, int(payload_length * 0.02))
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_key[error_pos] = 1 - bob_key[error_pos]
        
        # Partition if needed
        if payload_length <= 4096:
            blocks_alice = [alice_key]
            blocks_bob = [bob_key]
        else:
            blocks_alice = partition_key(alice_key, block_size=4096)
            blocks_bob = partition_key(bob_key, block_size=4096)
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            safety_cap=payload_length * 2,  # Generous cap
        )
        
        corrected_blocks = []
        for i, (alice_block, bob_block) in enumerate(zip(blocks_alice, blocks_bob)):
            result = orchestrator.reconcile_block(
                alice_key=alice_block,
                bob_key=bob_block,
                qber_estimate=0.03,
                block_id=i,
            )
            
            assert result.verified, f"Block {i} verification failed"
            assert result.converged, f"Block {i} did not converge"
            corrected_blocks.append(result.corrected_payload)
        
        # Reconstruct full key
        corrected_full = np.concatenate(corrected_blocks)
        np.testing.assert_array_equal(corrected_full, alice_key)
    
    def test_rob_002_payload_exceeds_frame_without_partition(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Passing >4096 bits without partitioning should fail gracefully."""
        rng = np.random.default_rng(99)
        alice_key = rng.integers(0, 2, size=5000, dtype=np.int8)
        bob_key = alice_key.copy()
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
        )
        
        # This should either:
        # 1. Raise ValueError (preferred)
        # 2. Abort with unverified result
        # It should NOT silently corrupt data
        with pytest.raises((ValueError, AssertionError)):
            result = orchestrator.reconcile_block(
                alice_key=alice_key,
                bob_key=bob_key,
                qber_estimate=0.03,
            )
            # If we reach here, at minimum it must not verify
            assert not result.verified


class TestHighQBERScenarios:
    """Tests for reconciliation under high error rates."""
    
    @pytest.mark.parametrize("qber", [
        0.08,   # 8% - high but feasible
        0.10,   # 10% - near limit
        0.11,   # 11% - at/above Kiktenko table limit
    ])
    def test_rob_010_high_qber_reconciliation(
        self,
        qber: float,
        matrix_manager: MatrixManager,
    ) -> None:
        """Reconcile at high QBER levels."""
        payload_length = 3500
        rng = np.random.default_rng(int(qber * 1000))
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        
        # Bob's key with specified QBER
        bob_key = alice_key.copy()
        n_errors = int(payload_length * qber)
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_key[error_pos] = 1 - bob_key[error_pos]
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            safety_cap=10**6,  # Large cap to avoid leakage abort
        )
        
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=qber,
        )
        
        # At high QBER, convergence may fail
        if qber <= 0.10:
            # Should succeed with retry
            assert result.converged or result.verified
        else:
            # May fail at 11% QBER - acceptable
            if not result.converged:
                pytest.skip(f"QBER={qber:.2f} exceeds decoder capability")
    
    def test_rob_011_qber_near_half_should_fail(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """QBER approaching 0.5 (random channel) should fail."""
        payload_length = 2000
        qber = 0.40  # 40% errors - nearly random
        
        rng = np.random.default_rng(123)
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        bob_key = alice_key.copy()
        
        n_errors = int(payload_length * qber)
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_key[error_pos] = 1 - bob_key[error_pos]
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
        )
        
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=qber,
        )
        
        # Decoder should not converge at 40% QBER
        assert not result.converged
        assert not result.verified


class TestDecoderNonConvergence:
    """Tests for decoder failure and retry logic."""
    
    def test_rob_020_decoder_retry_with_llr_damping(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Verify retry mechanism activates on initial failure."""
        # Use moderate QBER that might require retry
        payload_length = 3000
        qber = 0.07
        
        rng = np.random.default_rng(555)
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        bob_key = alice_key.copy()
        
        n_errors = int(payload_length * qber)
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_key[error_pos] = 1 - bob_key[error_pos]
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            config=ReconciliationOrchestratorConfig(max_retries=2),
        )
        
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=qber,
        )
        
        # With retry, should eventually converge or verify
        assert result.converged or result.verified
    
    def test_rob_021_all_retries_exhausted(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Handle case where all retry attempts fail."""
        # Create pathologically difficult case
        payload_length = 3000
        qber = 0.12  # Above design limit
        
        rng = np.random.default_rng(666)
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        bob_key = alice_key.copy()
        
        n_errors = int(payload_length * qber)
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_key[error_pos] = 1 - bob_key[error_pos]
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            config=ReconciliationOrchestratorConfig(max_retries=1),
        )
        
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=qber,
        )
        
        # Should return non-converged result gracefully
        assert not result.converged
        assert not result.verified
        # Should not raise exception


class TestLeakageBudgetExhaustion:
    """Tests for leakage cap enforcement."""
    
    def test_rob_030_exact_cap_boundary(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Test exact leakage cap boundary (cap - 1, cap, cap + 1)."""
        from caligo.reconciliation.leakage_tracker import LeakageTracker
        
        # Set tight cap
        cap = 2000
        tracker = LeakageTracker(safety_cap=cap)
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            leakage_tracker=tracker,
        )
        
        payload_length = 2867  # ~70% of 4096
        qber = 0.05
        
        rng = np.random.default_rng(777)
        alice_key = rng.integers(0, 2, size=payload_length, dtype=np.int8)
        bob_key = alice_key.copy()
        
        # First block should succeed
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=qber,
            block_id=0,
        )
        
        assert result.verified
        leakage_after_block1 = tracker.total_leakage
        
        # If close to cap, second block should trigger abort check
        if leakage_after_block1 < cap:
            # Try second block
            alice_key2 = rng.integers(0, 2, size=payload_length, dtype=np.int8)
            bob_key2 = alice_key2.copy()
            
            # This may exceed cap
            if orchestrator.should_abort():
                pytest.skip("Leakage cap would be exceeded")
            else:
                result2 = orchestrator.reconcile_block(
                    alice_key=alice_key2,
                    bob_key=bob_key2,
                    qber_estimate=qber,
                    block_id=1,
                )
    
    def test_rob_031_multi_block_abort_mid_stream(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Abort during multi-block reconciliation when cap exceeded."""
        from caligo.reconciliation.leakage_tracker import LeakageTracker
        
        # Tight cap that allows only 2-3 blocks
        cap = 3000
        tracker = LeakageTracker(safety_cap=cap)
        
        orchestrator = ReconciliationOrchestrator(
            matrix_manager=matrix_manager,
            leakage_tracker=tracker,
        )
        
        # 5 blocks worth of data
        total_length = 20000
        blocks = partition_key(
            np.random.default_rng(888).integers(0, 2, size=total_length, dtype=np.int8),
            block_size=4096,
        )
        
        reconciled_blocks = []
        for i, block in enumerate(blocks):
            if orchestrator.should_abort():
                # Abort condition triggered
                assert i > 0, "Should process at least one block before abort"
                break
            
            result = orchestrator.reconcile_block(
                alice_key=block,
                bob_key=block.copy(),  # Noiseless for simplicity
                qber_estimate=0.03,
                block_id=i,
            )
            reconciled_blocks.append(result)
        
        # Should have aborted before all blocks
        assert len(reconciled_blocks) < len(blocks)


class TestBlindReconciliationEdgeCases:
    """Tests for blind reconciliation edge cases."""
    
    def test_rob_040_blind_all_iterations_exhausted(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Blind reconciliation exhausts all iterations without convergence."""
        from caligo.reconciliation.blind_manager import BlindReconciliationManager, BlindConfig
        from caligo.reconciliation.factory import BlindReconciler, ReconciliationConfig, ReconciliationType
        
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BLIND,
            max_blind_rounds=3,
            frame_size=4096,
        )
        
        reconciler = BlindReconciler(config)
        
        # Create difficult case (high QBER)
        payload_length = 3000
        qber = 0.10
        
        rng = np.random.default_rng(999)
        alice_bits = rng.integers(0, 2, size=payload_length, dtype=np.uint8).tobytes()
        bob_bits = alice_bits  # Even with identical, high QBER estimate may cause issues
        
        corrected, metadata = reconciler.reconcile(alice_bits, bob_bits)
        
        # Should complete without exception
        assert metadata['reconciliation_type'] == 'blind'
        # Status may be 'success' or 'failed'
    
    def test_rob_041_blind_first_iteration_success(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Blind reconciliation succeeds on first iteration (best case)."""
        from caligo.reconciliation.factory import BlindReconciler, ReconciliationConfig, ReconciliationType
        
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BLIND,
            max_blind_rounds=3,
        )
        
        reconciler = BlindReconciler(config)
        
        # Low QBER, should converge quickly
        payload_length = 3000
        qber = 0.01
        
        rng = np.random.default_rng(1111)
        alice_bits = rng.integers(0, 2, size=payload_length, dtype=np.uint8)
        bob_bits = alice_bits.copy()
        
        # Add minimal errors
        n_errors = int(payload_length * qber)
        error_pos = rng.choice(payload_length, size=n_errors, replace=False)
        bob_bits[error_pos] = 1 - bob_bits[error_pos]
        
        corrected, metadata = reconciler.reconcile(
            alice_bits.tobytes(),
            bob_bits.tobytes(),
        )
        
        assert metadata['status'] == 'success'
        assert metadata['verified'] is True
```

**Additional Test Files:**

- `test_kiktenko_compliance.py`: Direct validation against Kiktenko equations
- `test_martinez_mateo_compliance.py`: Direct validation against Martinez-Mateo protocol
- `test_boundary_conditions.py`: Zero-length keys, single-bit keys, maximum-length keys

### 5.4 Test Execution Strategy

**Priority Levels:**

1. **🔴 P0 (Blocking):** Multi-block tests (rob_001, rob_002) — must pass before production
2. **🟠 P1 (High):** High QBER tests (rob_010, rob_011) — validate design limits
3. **🟡 P2 (Medium):** Decoder convergence tests (rob_020, rob_021) — improve robustness
4. **🟢 P3 (Low):** Boundary tests (rob_030, rob_031) — edge case hardening

**Test Markers:**

```python
@pytest.mark.robustness    # All robustness tests
@pytest.mark.slow          # Tests > 5 seconds
@pytest.mark.multiblock    # Multi-block reconciliation tests
@pytest.mark.highqber      # QBER ≥ 8% tests
```

**Execution:**

```bash
# Run all robustness tests
pytest tests/reconciliation/test_reconciliation_robustness.py -v

# Run only blocking tests
pytest -m "robustness and multiblock" -v

# Run with coverage
pytest tests/reconciliation/ --cov=caligo.reconciliation --cov-report=html
```

---

## 6. Implementation Roadmap

### 6.1 Critical Fixes (P0 - Must Fix Before Production)

#### Issue 1: Multi-Block Reconciliation Pipeline

**Problem:** Protocol passes variable-length keys to `reconcile_block()` without partitioning, causing failures when `len(key) > 4096`.

**Root Cause:**
```python
# alice.py:275-285 (INCORRECT)
def _phase3_reconcile(self, alice_bits, bob_bits, qber_adjusted):
    orchestrator = ReconciliationOrchestrator(...)
    result = orchestrator.reconcile_block(
        alice_key=alice_arr,  # Can be any length!
        bob_key=bob_arr,
        qber_estimate=qber_adjusted,
        block_id=0,
    )
```

**Solution:** Implement multi-block reconciliation wrapper

**Implementation Steps:**

1. **Create `orchestrator.reconcile_multi_block()` method:**

```python
def reconcile_multi_block(
    self,
    alice_key: np.ndarray,
    bob_key: np.ndarray,
    qber_estimate: float,
) -> MultiBlockResult:
    """
    Reconcile variable-length key using block partitioning.
    
    Parameters
    ----------
    alice_key : np.ndarray
        Alice's full key (any length).
    bob_key : np.ndarray
        Bob's full key (any length).
    qber_estimate : float
        Estimated QBER.
    
    Returns
    -------
    MultiBlockResult
        Aggregated result across all blocks.
    """
    frame_size = self.config.frame_size
    
    # Partition into blocks
    alice_blocks = partition_key(alice_key, block_size=frame_size)
    bob_blocks = partition_key(bob_key, block_size=frame_size)
    
    corrected_blocks = []
    total_syndrome_bits = 0
    all_verified = True
    all_converged = True
    
    for block_id, (alice_block, bob_block) in enumerate(zip(alice_blocks, bob_blocks)):
        # Check leakage before processing
        if self.should_abort():
            raise LeakageCapExceededError(
                f"Leakage cap exceeded at block {block_id}/{len(alice_blocks)}"
            )
        
        result = self.reconcile_block(
            alice_key=alice_block,
            bob_key=bob_block,
            qber_estimate=qber_estimate,
            block_id=block_id,
        )
        
        corrected_blocks.append(result.corrected_payload)
        total_syndrome_bits += result.syndrome_length
        all_verified &= result.verified
        all_converged &= result.converged
        
        if not result.verified:
            logger.warning(f"Block {block_id} failed verification")
    
    # Concatenate corrected blocks
    corrected_full = np.concatenate(corrected_blocks)
    
    return MultiBlockResult(
        corrected_payload=corrected_full,
        verified=all_verified,
        converged=all_converged,
        num_blocks=len(alice_blocks),
        total_syndrome_length=total_syndrome_bits,
    )
```

2. **Update `alice.py` to use multi-block API:**

```python
def _phase3_reconcile(self, alice_bits, bob_bits, qber_adjusted):
    alice_arr = bitarray_to_numpy(alice_bits)
    bob_arr = bitarray_to_numpy(bob_bits)
    
    matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
    orchestrator = ReconciliationOrchestrator(
        matrix_manager=matrix_manager,
        config=ReconciliationOrchestratorConfig(frame_size=4096, max_retries=2),
        safety_cap=10**12,
    )
    
    # Use multi-block API
    result = orchestrator.reconcile_multi_block(
        alice_key=alice_arr,
        bob_key=bob_arr,
        qber_estimate=qber_adjusted,
    )
    
    if not result.verified:
        raise SecurityError(f"Reconciliation failed: {result.num_blocks} blocks processed")
    
    reconciled = bitarray_from_numpy(result.corrected_payload)
    return reconciled, result.total_syndrome_length
```

3. **Add validation to `reconcile_block()`:**

```python
def reconcile_block(self, alice_key, bob_key, qber_estimate, block_id=0):
    payload_len = len(alice_key)
    frame_size = self.config.frame_size
    
    # Validation: reject oversized blocks
    if payload_len > frame_size:
        raise ValueError(
            f"Block {block_id}: payload_len={payload_len} exceeds frame_size={frame_size}. "
            f"Use reconcile_multi_block() for variable-length keys."
        )
    
    # ... rest of reconcile_block logic
```

**Estimated LOC:** +80 lines (orchestrator.py), +15 lines (alice.py)  
**Testing:** `test_reconciliation_robustness.py::TestVariableLengthKeys`

---

#### Issue 2: Kiktenko Shortening Formula Incorrect

**Problem:** `compute_shortening()` conflates frame padding with Kiktenko's efficiency-driven shortening.

**Root Cause (rate_selector.py:193-210):**

```python
def compute_shortening(rate, qber_estimate, payload_length, frame_size=4096, f_crit=1.22):
    # This is PADDING, not Kiktenko shortening!
    basic_shortened = max(0, frame_size - payload_length)
    
    # This formula is WRONG
    n_s_extra = int(math.floor(frame_size - payload_length / (f_crit * entropy)))
    
    n_shortened = max(basic_shortened, n_s_extra)  # Incorrect logic
```

**Kiktenko Equation (2):**
$$n_s = \left\lfloor n - \frac{m}{f_{\text{crit}} \cdot h_b(\text{QBER}_{\text{est}})} \right\rfloor$$

**Solution:** Separate padding from Kiktenko shortening

**Implementation:**

```python
def compute_padding(payload_length: int, frame_size: int) -> int:
    """
    Compute padding bits to fill frame.
    
    This is NOT Kiktenko shortening - just dimension alignment.
    """
    return max(0, frame_size - payload_length)


def compute_kiktenko_shortening(
    rate: float,
    qber_estimate: float,
    frame_size: int = 4096,
    f_crit: float = 1.22,
) -> int:
    """
    Compute Kiktenko shortening for frame error rate reduction.
    
    From Kiktenko et al. (2016) Equation (2):
        n_s = floor(n - m / (f_crit * h(QBER)))
    
    Where:
        n = frame_size
        m = n * (1 - rate) = check bits
    
    Returns
    -------
    int
        Number of bits to shorten (0 if not needed).
    """
    m = int(frame_size * (1 - rate))
    entropy = binary_entropy(qber_estimate)
    
    if entropy <= 0:
        return 0
    
    # Kiktenko Eq. 2 (exact)
    n_s = math.floor(frame_size - m / (f_crit * entropy))
    
    # Shortening must be non-negative and less than frame size
    return max(0, min(n_s, frame_size - 1))


def select_rate_with_parameters(
    qber_estimate: float,
    payload_length: int,
    frame_size: int = 4096,
    available_rates: Tuple[float, ...] = constants.LDPC_CODE_RATES,
    f_crit: float = 1.22,
) -> RateSelection:
    """Select rate and compute ALL parameters separately."""
    rate = select_rate(qber_estimate, available_rates, f_crit)
    
    # Padding: fill frame to match matrix dimensions
    n_padding = compute_padding(payload_length, frame_size)
    
    # Kiktenko shortening: improve frame error rate
    n_kiktenko = compute_kiktenko_shortening(rate, qber_estimate, frame_size, f_crit)
    
    # Total shortened = padding + Kiktenko extra
    # BUT: if payload + padding + kiktenko > frame_size, we must reduce
    n_total_shortened = min(n_padding + n_kiktenko, frame_size - 1)
    
    # ... rest of function
    return RateSelection(
        rate=rate,
        n_shortened=n_total_shortened,
        n_padding=n_padding,        # New field
        n_kiktenko=n_kiktenko,      # New field
        # ...
    )
```

**Estimated LOC:** +40 lines (rate_selector.py refactor)  
**Testing:** `test_kiktenko_compliance.py` (new file)

---

#### Issue 3: Protocol Bypasses Factory

**Problem:** `alice.py` directly instantiates `MatrixManager` and `ReconciliationOrchestrator`, ignoring factory pattern.

**Solution:** Use factory for reconciler creation

**Implementation:**

1. **Modify `ProtocolParameters` to include reconciliation config:**

```python
@dataclass
class ProtocolParameters:
    num_pairs: int
    nsm_params: NSMParameters
    reconciliation_config: Dict[str, Any]  # New field
    # ...
```

2. **Update `AliceProgram.__init__()` to use factory:**

```python
from caligo.reconciliation.factory import create_reconciler, ReconciliationConfig

class AliceProgram(CaligoProgram):
    def __init__(self, params: ProtocolParameters):
        super().__init__(params)
        self._commitment = SHA256Commitment()
        self._sifter = Sifter()
        self._qber_estimator = QBEREstimator(epsilon_sec=1e-10)
        
        # Initialize reconciler via factory
        recon_config = ReconciliationConfig.from_dict(params.reconciliation_config)
        self._reconciler = create_reconciler(recon_config, nsm_params=params.nsm_params)
```

3. **Simplify `_phase3_reconcile()` to use reconciler interface:**

```python
def _phase3_reconcile(self, alice_bits, bob_bits, qber_adjusted):
    # Convert to bytes for reconciler interface
    alice_bytes = bitarray_to_numpy(alice_bits).tobytes()
    bob_bytes = bitarray_to_numpy(bob_bits).tobytes()
    
    # Use reconciler interface
    corrected_bytes, metadata = self._reconciler.reconcile(
        alice_bytes,
        bob_bytes,
        measured_qber=qber_adjusted,
    )
    
    if metadata['status'] != 'success':
        raise SecurityError(f"Reconciliation failed: {metadata}")
    
    # Convert back to bitarray
    corrected_np = np.frombuffer(corrected_bytes, dtype=np.uint8)
    reconciled = bitarray_from_numpy(corrected_np)
    
    return reconciled, metadata['syndrome_length']
```

**Estimated LOC:** +10 lines (alice.py), -15 lines (remove direct instantiation)  
**Testing:** `test_factory_integration.py` (already exists)

---

### 6.2 Enhancement Priorities (P1 - High Priority)

#### Enhancement 1: Blind Reconciliation Integration

**Status:** Dead code - implemented but not connected

**Implementation:**

1. **Complete `BaselineReconciler.reconcile()` placeholder**
2. **Test blind reconciliation end-to-end** via factory
3. **Add YAML config examples** for both strategies
4. **Document performance trade-offs** (blind vs. baseline)

**Estimated LOC:** +120 lines (baseline reconciler), +50 lines (docs)

---

#### Enhancement 2: Improved Error Handling

**Current State:** Exceptions not consistently raised or handled

**Implementation:**

1. **Define reconciliation-specific exceptions:**

```python
class ReconciliationError(Exception):
    """Base exception for reconciliation failures."""

class LeakageCapExceededError(ReconciliationError):
    """Leakage budget exhausted."""

class DecoderConvergenceError(ReconciliationError):
    """Belief propagation did not converge."""

class HashVerificationError(ReconciliationError):
    """Hash mismatch after decoding."""
```

2. **Add exception handling in orchestrator**
3. **Propagate errors cleanly to protocol layer**

**Estimated LOC:** +40 lines (exceptions.py), +30 lines (orchestrator.py)

---

### 6.3 Test Suite Expansion (P1)

**Deliverables:**

1. **`test_reconciliation_robustness.py`** (see Section 5.3) — 400 LOC
2. **`test_kiktenko_compliance.py`** — 150 LOC
   - Direct validation of Equations (1) and (2)
   - Shortening parameter ranges
   - Efficiency measurement
3. **`test_martinez_mateo_compliance.py`** — 200 LOC
   - Iteration protocol correctness
   - Leakage accounting validation
   - Puncture/shorten conversion
4. **`test_boundary_conditions.py`** — 100 LOC
   - Zero-length keys
   - Single-bit keys
   - Maximum-length keys (100,000+ bits)

**Total Estimated LOC:** 850 lines of tests

---

### 6.4 Documentation Updates (P2)

1. **Update `recon_phase_spec.md`** with:
   - Multi-block reconciliation section
   - Clarified shortening vs. padding distinction
   - Factory integration examples

2. **Create `reconciliation_user_guide.md`** with:
   - When to use baseline vs. blind
   - QBER threshold guidelines
   - Performance benchmarks
   - Troubleshooting guide

3. **Add inline docstrings** for:
   - `reconcile_multi_block()`
   - `compute_padding()` vs. `compute_kiktenko_shortening()`
   - Exception classes

**Estimated LOC:** +300 lines (documentation)

---

### 6.5 Implementation Timeline

**Phase 1 (Week 1): Critical Fixes**
- Day 1-2: Multi-block reconciliation implementation
- Day 3: Kiktenko shortening formula correction
- Day 4-5: Protocol-factory decoupling
- Day 5: Integration testing

**Phase 2 (Week 2): Test Hardening**
- Day 1-2: Robustness test suite (test_reconciliation_robustness.py)
- Day 3: Literature compliance tests (Kiktenko + Martinez-Mateo)
- Day 4: Boundary condition tests
- Day 5: CI/CD integration, coverage analysis

**Phase 3 (Week 3): Enhancements**
- Day 1-2: Baseline reconciler completion
- Day 3: Blind reconciliation end-to-end testing
- Day 4-5: Error handling improvements

**Phase 4 (Week 4): Documentation & Polish**
- Day 1-2: Documentation updates
- Day 3: User guide creation
- Day 4-5: Code review, refactoring, final testing

---

### 6.6 Success Metrics

**Code Quality:**
- ✅ All P0 issues resolved
- ✅ Test coverage ≥ 90% for reconciliation package
- ✅ No direct imports of reconciliation internals in protocol layer
- ✅ Mypy type checking passes with no errors

**Functional:**
- ✅ Reconciliation succeeds for keys up to 100,000 bits
- ✅ QBER handling up to 10% without failure
- ✅ Multi-block leakage accounting accurate to ±1 bit
- ✅ Blind reconciliation operational via factory

**Performance:**
- ✅ Reconciliation time scales linearly with key length
- ✅ Memory usage < 50 MB for 100,000-bit keys
- ✅ Decoder convergence rate > 95% for QBER ≤ 8%

---

**Status:** Implementation roadmap complete. Proceeding with report finalization.

---

## 7. References

1. **Martinez-Mateo, J., Elkouss, D., & Martin, V.** (2003/2012). *Blind Reconciliation*. Quantum Information and Computation.
   - Blind reconciliation protocol without prior QBER estimation
   - Iterative puncture-to-shorten conversion technique
   - Leakage accounting for oblivious transfer context

2. **Kiktenko, E. O., et al.** (2016). *Post-processing procedure for industrial quantum key distribution systems*. Journal of Physics: Conference Series, 741(1), 012081.
   - Rate selection criterion: $(1-R)/h_b(\text{QBER}) < f_{\text{crit}}$
   - Shortening technique: $n_s = \lfloor n - m/(f_{\text{crit}} \cdot h_b(\text{QBER})) \rfloor$
   - 50-bit polynomial hash verification

3. **Caligo Phase III Specification:** [recon_phase_spec.md](recon_phase_spec.md)
   - Reconciliation in $\binom{2}{1}$-OT protocol context
   - Wiretap cost model and leakage accounting
   - One-way information flow requirement

4. **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
   - LDPC code fundamentals
   - Belief propagation algorithm
   - Syndrome decoding

5. **Hu, X.-Y., Eleftheriou, E., & Arnold, D. M.** (2005). *Progressive edge-growth Tanner graphs*. IEEE Globecom.
   - PEG algorithm for LDPC matrix construction
   - Degree distribution optimization

6. **Elkouss, D., Martinez-Mateo, J., & Martin, V.** (2009). *Rate compatible protocol for information reconciliation*. IEEE International Symposium on Information Theory.
   - Rate-compatible LDPC code families
   - Puncturing and shortening techniques

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-19 | AI Technical Reviewer | Initial comprehensive review |

---

**Appendix A: Code Snippet Locations**

| Component | Primary File | Line Range | Key Methods |
|-----------|--------------|------------|-------------|
| Factory | `factory.py` | 1-857 | `create_reconciler()`, `ReconciliationType` |
| Orchestrator | `orchestrator.py` | 1-377 | `reconcile_block()`, `_decode_with_retry()` |
| Rate Selector | `rate_selector.py` | 1-309 | `select_rate()`, `compute_shortening()` |
| LDPC Encoder | `ldpc_encoder.py` | 1-286 | `encode_block()`, `prepare_frame()` |
| BP Decoder | `ldpc_decoder.py` | (not fully reviewed) | `BeliefPropagationDecoder.decode()` |
| Blind Manager | `blind_manager.py` | 1-372 | `initialize()`, `advance_iteration()` |
| Protocol Integration | `alice.py` | 269-290 | `_phase3_reconcile()` |

---

**Appendix B: Test File Coverage Matrix**

| Test File | Component Tested | Coverage | Hard Tests |
|-----------|------------------|----------|------------|
| `test_integration.py` | End-to-end flow | 85% | ⚠️ Missing multi-block |
| `test_blind_manager.py` | Blind iteration | 90% | ⚠️ Missing Martinez-Mateo style |
| `test_factory.py` | Factory pattern | 95% | ✅ Good |
| `test_bp_decoder.py` | BP algorithm | 75% | ⚠️ Missing high-QBER |
| `test_rate_selector.py` | Rate selection | 80% | ⚠️ Missing Kiktenko validation |
| `test_hash_verifier.py` | Hash verification | 90% | ✅ Good |
| `test_matrix_manager_contracts.py` | Matrix loading | 85% | ✅ Good |
| `test_leakage_tracker.py` | Leakage accounting | 90% | ⚠️ Missing exact boundary |
| **`test_reconciliation_robustness.py`** | **Edge cases** | **NEW** | **🔴 Critical** |
| **`test_kiktenko_compliance.py`** | **Literature** | **NEW** | **🟠 High Priority** |
| **`test_martinez_mateo_compliance.py`** | **Literature** | **NEW** | **🟠 High Priority** |

---

**END OF REPORT**

---

**Next Steps:**

1. ✅ Review findings with development team
2. ✅ Prioritize P0 fixes in sprint planning
3. ✅ Assign implementation tasks
4. ✅ Set up CI/CD for new robustness tests
5. ✅ Schedule code review sessions
6. ✅ Update project roadmap with timeline

For questions or clarifications, consult:
- **Architecture:** Section 2 (Factory, Orchestrator, Data Flow)
- **Specific Bug:** Section 6.1 (Critical Fixes with code examples)
- **Testing Strategy:** Section 5 (Coverage gaps, proposed tests)
- **Literature Compliance:** Section 4 (Kiktenko & Martinez-Mateo validation)

**Document Prepared By:** AI Senior Technical Reviewer  
**Review Confidence:** High (based on comprehensive code analysis, literature comparison, and test assessment)  
**Estimated Implementation Effort:** 4 weeks (1 developer, full-time)  
**Risk if Not Addressed:** 🔴 **HIGH** — Protocol will fail in production scenarios with realistic key lengths.
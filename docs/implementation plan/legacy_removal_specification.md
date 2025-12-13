# Legacy Removal & Hardening Sprint — Implementation Specification

> **Sprint**: Legacy Removal & Hardening (Post Sprint 3)
>
> **Document ID**: Legacy-Removal / E-HOK-on-SquidASM
>
> **Persona**: Lead Code Quality Engineer and Maintenance Specialist
>
> **Scope Constraint**: This document specifies *what to refactor, what to delete, and how to prove it is safe*, derived strictly from the Aggressive Legacy Removal methodology in the master roadmap and the Phase I–IV analyses. It contains **no implementation code**.

---

## 1. Executive Summary

### Goal
Achieve a clean, NSM-compliant E-HOK codebase where **sound algorithmic logic is preserved and refactored** to integrate with SquidASM interfaces, while **QKD-derived security models and incompatible orchestration layers are removed**.

### Critical Architectural Clarification

> **⚠️ CORRECTION OF MISCONCEPTION**: SquidASM is an **application-layer simulation interface**—it does NOT provide cryptographic primitives such as LDPC encoders, basis matching algorithms, or Toeplitz hashers. The mathematical implementations in `ehok/implementations/` are **sound and must be preserved**.

**Definition of "SquidASM-Native"**: Throughout this document, "SquidASM-native" means **E-HOK code that imports and uses SquidASM interfaces** (e.g., `ClassicalSocket`, `OrderedProtocolSocket`, application protocols). It does **NOT** mean "code provided by the SquidASM library." SquidASM provides the communication and simulation infrastructure; E-HOK provides the cryptographic algorithms.

### The Algorithm vs. Orchestration Distinction

| Category | Definition | Action |
|----------|------------|--------|
| **Algorithms (KEEP/REFACTOR)** | Mathematical logic that is cryptographically sound and framework-agnostic (LDPC encode/decode, basis matching, Toeplitz matrix multiplication, Chernoff bounds) | Preserve, wrap with SquidASM socket integration, add NSM-aware security managers |
| **Orchestration/Security Logic (DELETE/REPLACE)** | Code that couples algorithms to incorrect security models or incompatible communication patterns (QKD entropy bounds, synchronous runners, code ignoring ordered acknowledgments) | Delete after replacement is validated |

### Scope
This sprint targets **orchestration wrappers and security formulas** that are incompatible with NSM semantics:

- Phase II: Synchronous sifting orchestration replaced with `OrderedProtocolSocket` integration; the **basis matching math is preserved**.
- Phase III: QKD-bound leakage calculations replaced with NSM `LeakageSafetyManager`; the **LDPC infrastructure is architecturally sound and preserved** (per Phase III analysis).
- Phase IV: QKD finite-key formula ($1 - h(Q)$) replaced with NSM Max Bound; the **Toeplitz hashing mechanics are preserved**.
- Phase I: Legacy configuration compatibility pathways removed.

### Success Metric
The sprint is successful only if ALL of the following are true:

1. **Preserved Algorithmic Core**: The mathematical logic for LDPC encoding/decoding, basis identification, Toeplitz hashing, and statistical validation remains intact and is exercised by tests.
2. **NSM Security Supremacy**: No codepath computes final key length using QKD entropy-rate proofs (e.g., $1-h(Q)$); all security calculations use NSM Max Bound ($h_{min}(r) = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$).
3. **SquidASM Integration Complete**: All network communication uses SquidASM sockets (`ClassicalSocket`, `OrderedProtocolSocket`) with proper asynchronous patterns.
4. **Green build constraint**: Refactoring does not break CI; tests exercise both preserved algorithms and new integration layers.
5. **Static rigor**: The repository passes strict type checking and lint gates specified in Section 4.

---

## 2. The Refactoring & Deletion Manifest (Strategy)

### 2.1 Core Principle: Algorithm Preservation, Orchestration Replacement

> **Reference**: Phase III analysis states the legacy LDPC infrastructure is "architecturally sound for E-HOK." Phase IV analysis states Toeplitz hashing mechanics should be "extracted from legacy." The mathematical logic is NOT the problem—the **security model wrappers** are.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALGORITHM vs. ORCHESTRATION DISTINCTION                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ALGORITHMS (PRESERVE & REFACTOR)                                        ││
│  │                                                                          ││
│  │  Mathematical operations that are framework-agnostic and                 ││
│  │  cryptographically sound. These work the same whether called             ││
│  │  synchronously or asynchronously.                                        ││
│  │                                                                          ││
│  │  Examples:                                                               ││
│  │   • LDPC syndrome computation: S = H · X (mod 2)                        ││
│  │   • Belief Propagation decoding iterations                              ││
│  │   • Basis matching: identify_matching_bases(α, β) → I₀, I₁             ││
│  │   • Toeplitz matrix multiplication: T × key (mod 2)                     ││
│  │   • Chernoff/Hoeffding bound calculations                               ││
│  │   • QBER computation from test subset                                    ││
│  │   • Polynomial hash verification                                         ││
│  │                                                                          ││
│  │  ACTION: Keep the math. Wrap with SquidASM socket integration.          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ORCHESTRATION & SECURITY LOGIC (DELETE & REPLACE)                       ││
│  │                                                                          ││
│  │  Code that couples algorithms to incorrect security models or           ││
│  │  communication patterns incompatible with NSM/SquidASM.                  ││
│  │                                                                          ││
│  │  Examples:                                                               ││
│  │   • Final key length using QKD bound: ℓ = n(1-h(Q)) - leak              ││
│  │   • Synchronous "send-then-wait" patterns (vs. OrderedProtocolSocket)  ││
│  │   • Functions that ignore Δt timing enforcement                         ││
│  │   • Code producing single-key output (vs. OT (S₀,S₁)/(S_C,C) format)   ││
│  │   • Leakage tracking without L_max hard cap enforcement                 ││
│  │   • QBER threshold checks without finite-size penalty μ                 ││
│  │                                                                          ││
│  │  ACTION: Delete after NSM-compliant replacement is validated.           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Identification Strategy (How to Classify Code)

Derived from:
- Master roadmap "Aggressive Legacy Removal" and TDM protocol: `WRAP → REPLICATE → VALIDATE → DELETE` (no rollback).
- Phase I "Aggressive Legacy Removal Strategy": "without maintaining backward compatibility."

#### 2.2.1 Classification Rules

**Code is ALGORITHMIC (KEEP) if:**
1. It performs pure mathematical computation with no network I/O
2. It operates identically whether called sync or async
3. The Phase analyses describe it as "architecturally sound," "valid math," or "extract mechanics from"
4. It computes parity, syndrome, hash, entropy, or statistical bounds

**Code is ORCHESTRATION/SECURITY LOGIC (DELETE) if:**
1. It computes final key length using QKD entropy bounds ($1-h(Q)$)
2. It performs network communication without using SquidASM sockets
3. It ignores ordered acknowledgment requirements
4. It produces outputs incompatible with OT structure
5. It provides backward-compatibility fallbacks to non-NSM behavior

#### 2.2.2 Mandatory Audit Procedure (Before Any Deletion)

For each candidate module:

1. **Separate Algorithm from Orchestration**: Identify which functions are pure math vs. which couple to I/O or security models.
2. **Extract Algorithmic Core**: If a file mixes both, extract the algorithmic functions to a preserved module before deleting the orchestration wrapper.
3. **Locate All Importers**: Production code + tests.
4. **Confirm Parity Tests**: Ensure the refactored code produces identical outputs for algorithmic operations.
5. **Delete Only After Validation**: Only then delete the orchestration layer and rewire imports.

This procedure is the "Test-Driven Migration Protocol" from the master roadmap: write tests against the interface, parity-test legacy vs new, then delete legacy and update imports.

---

### 2.3 Phase-by-Phase Refactoring Plan

This section **corrects** the misconception that entire modules should be deleted. Each phase preserves algorithmic logic while replacing orchestration/security wrappers.

#### 2.3.1 Phase II — Sifting & Estimation

**Source**: Phase II analysis confirms basis matching logic (`identify_matching_bases`) is "valid math."

**File**: `ehok/core/sifting.py`

| Component | Classification | Action |
|-----------|---------------|--------|
| `identify_matching_bases(α, β)` | **ALGORITHM** | **KEEP** — Pure index set computation |
| `compute_qber(key_a, key_b, indices)` | **ALGORITHM** | **KEEP** — Statistical calculation |
| `select_test_subset(indices, k)` | **ALGORITHM** | **KEEP** — Random sampling |
| `compute_finite_size_penalty(n, k, ε)` | **ALGORITHM** | **KEEP** — Hoeffding bound |
| `run_sifting_protocol()` (if synchronous) | **ORCHESTRATION** | **REFACTOR** — Wrap with `OrderedProtocolSocket` |
| QBER threshold abort without $e_{adj}$ | **SECURITY LOGIC** | **DELETE** — Replace with adjusted-QBER check |

**Refactoring Requirements**:
- Preserve all pure functions that compute $I_0$, $I_1$, QBER, and $\mu$
- Create `SiftingProtocol` class that uses `OrderedProtocolSocket` for message ordering
- Integrate Chernoff/Hoeffding validation for missing rounds
- Use $e_{adj} = e_{obs} + \mu$ for all threshold comparisons

**Deletion Gate**: Delete synchronous orchestration wrapper only after async protocol passes parity tests on algorithmic outputs.

#### 2.3.2 Phase III — Information Reconciliation

**Source**: Phase III analysis explicitly states: "The legacy `ehok/` implementation provides a comprehensive LDPC reconciliation infrastructure that is **architecturally sound** for E-HOK."

**Directory**: `ehok/implementations/reconciliation/`

| Component | Classification | Action |
|-----------|---------------|--------|
| `LDPCMatrixManager` | **ALGORITHM** | **KEEP** — Parity-check matrix operations |
| `LDPCBeliefPropagation` (decoder) | **ALGORITHM** | **KEEP** — BP iteration logic |
| `compute_syndrome(H, X)` | **ALGORITHM** | **KEEP** — Matrix multiplication mod 2 |
| `compute_llr(Y, qber)` | **ALGORITHM** | **KEEP** — Log-likelihood ratio |
| `PolyHashVerifier` | **ALGORITHM** | **KEEP** — Polynomial hashing |
| `select_ldpc_rate(qber)` | **ALGORITHM** | **KEEP** — Rate selection logic |
| Orchestration loop (send syndrome, receive, verify) | **ORCHESTRATION** | **REFACTOR** — Use `ClassicalSocket` async |
| Leakage tracking without $L_{max}$ cap | **SECURITY LOGIC** | **DELETE** — Replace with `LeakageSafetyManager` |

**Refactoring Requirements**:
- **DO NOT DELETE** the LDPC encoder/decoder, matrix manager, or hash verifier
- Create `LeakageSafetyManager` that enforces $L_{max}$ hard cap (Phase III gap)
- Create `ReconciliationProtocol` that wraps algorithmic calls with SquidASM socket I/O
- Integrate abort-on-leakage-exceeded logic

**Deletion Gate**: Delete only the orchestration wrapper that lacks leakage enforcement. Preserve all algorithmic components.

> **⚠️ CRITICAL**: The Phase III analysis gap is the **absence of a LeakageSafetyManager**, not a flaw in the LDPC stack itself. Wrap the existing LDPC infrastructure; do not delete it.

#### 2.3.3 Phase IV — Privacy Amplification

**Source**: Phase IV analysis states Toeplitz hashing mechanics should be "extracted from legacy." The gap is the **security formula**, not the hash function.

**Files**: `ehok/implementations/privacy_amplification/`

| Component | Classification | Action |
|-----------|---------------|--------|
| Toeplitz matrix construction | **ALGORITHM** | **KEEP** — Matrix generation from seed |
| Toeplitz hash: `T × key (mod 2)` | **ALGORITHM** | **KEEP** — Matrix-vector multiplication |
| Seed generation (`secrets.token_bytes`) | **ALGORITHM** | **KEEP** — Cryptographic randomness |
| `compute_final_length_finite_key()` using $1-h(Q)$ | **SECURITY LOGIC** | **DELETE** — Invalid for NSM |
| Single-key output format | **ORCHESTRATION** | **DELETE** — OT requires $(S_0,S_1)$, $(S_C,C)$ |

**Refactoring Requirements**:
- **Extract and KEEP** Toeplitz hashing mechanics (seed generation, matrix multiplication)
- Create `NSMBoundsCalculator` implementing Lupo et al. Max Bound:
  $$h_{min}(r) = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$$
- Replace `compute_final_length()` to use NSM bounds instead of QKD bounds
- Create `ObliviousKeyFormatter` producing OT-structured output

**Deletion Gate**: Delete the QKD-bound key length function and single-key output code. Preserve Toeplitz hashing mechanics.

#### 2.3.4 Phase I — Configuration

**Source**: Phase I analysis "Configuration Transition (Breaking Change)".

| Component | Classification | Action |
|-----------|---------------|--------|
| NSM-aware configuration schema | **KEEP** | Required for NSM security |
| Legacy format acceptance/fallback | **ORCHESTRATION** | **DELETE** — No backward compatibility |
| Configuration validation | **ALGORITHM** | **KEEP** — Schema validation logic |

**Refactoring Requirements**:
- Single mandatory configuration format with NSM parameters ($r$, $\Delta t$, $\varepsilon_{sec}$)
- Hard error on legacy configuration keys
- No silent defaulting for critical security parameters

---

### 2.4 Configuration Cleanup (Remove QKD-Only Parameters)

Derived from:
- Phase I "Configuration Transition (Breaking Change)": legacy configs are no longer supported.
- Phase IV "NSM Security Model vs QKD": E-HOK security is not QKD; Phase IV length selection must not use $1-h(Q)$ bounds.

#### 2.4.1 Required Outcome
The final configuration surface MUST:

- Expose **only parameters that are required for NSM E-HOK execution**, including:
  - NSM/noise parameters (at minimum $\mu$, $\eta$, and storage noise parameter $r$; Phase I provides a representative NSM-aware schema).
  - The timing barrier parameter $\Delta t$.
  - Security parameters used by the NSM-bound security accounting (e.g., $\varepsilon_{sec}$).

- Remove or hard-error on any configuration values that are only meaningful in QKD security proofs (e.g., parameters that exist solely to support $1-h(Q)$ final-length logic, or to select a "QKD finite-key" bound for Phase IV).

#### 2.4.2 Migration Rule (Breaking Change, No Fallback)
- Any legacy config schema acceptance MUST be deleted (Phase I explicitly labels it "DELETED: Legacy format is no longer accepted").
- No silent defaulting: parameters must be explicit (e.g., storage noise parameter must be explicitly set).

---

## 3. Test Suite Refactoring

### 3.1 Import Migration (Tests Must Point to Refactored Modules)
Derived from:
- Master roadmap TDM protocol steps 2, 4, 6, 7, 8.
- Phase II, III, IV refactoring plans requiring tests to exercise both algorithms and new integration layers.

#### 3.1.1 Test Refactor Rules
1. **Algorithmic tests preserved**: Tests for pure mathematical functions (LDPC, Toeplitz, sifting math) must continue to pass unchanged.
2. **Orchestration tests updated**: Tests for protocol flow must use SquidASM-native implementations.
3. **Security model tests added**: New tests must verify NSM bounds are used (not QKD bounds).
4. **Interface-first testing**: Tests should prefer importing phase interfaces/contracts rather than concrete implementations.

### 3.2 Logic Adaptation (Eliminate Legacy Assumptions)
Derived from:
- Phase I timing barrier requirement (abort if basis revealed before $\Delta t$).
- Phase II ordering requirements and the sifting deletion plan.

#### 3.2.1 Mandatory Test Updates
- Any test that assumes "instantaneous" ordering across messages must be rewritten to respect:
  - Ordered acknowledgment enforcement, and
  - The $\Delta t$ barrier.

If a test's premise is *fundamentally incompatible* with NSM semantics (e.g., it asserts behavior that would violate the timing barrier), it must be deleted or replaced with an NSM-correct version.

### 3.3 Parity Verification (Algorithm Equivalence)
Derived from the roadmap's parity-test gate and Phase II–IV refactoring requirements.

- Before modifying any algorithmic component, create a parity suite that demonstrates:
  - Matching behavior on deterministic inputs (byte-for-byte where feasible).
  - Equivalent scenario coverage.

- After refactoring, verify:
  - The algorithmic test suite produces identical outputs.
  - The integration test suite exercises the new SquidASM socket paths.

---

## 4. Static Analysis Enforcement

### 4.1 Mandatory Standards (Quality Gate Commands)
The build is considered FAILING unless the following exact commands pass:

- **Type Checking**:
  - `mypy --strict ehok/ --ignore-missing-imports`

- **Linting**:
  - `flake8 ehok/ --count --select=E9,F63,F7,F82 --show-source --statistics && flake8 ehok/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics`

### 4.2 Type Annotation Requirements
- Every function signature in retained code must have full type hints.
- `Any` is forbidden unless explicitly justified with a written rationale and localized to the narrowest possible scope.

### 4.3 Ambiguity Elimination Rules
- No "deprecated" codepaths may remain in production paths.
- No "fallback to legacy formula" behavior is permitted.
- Any conditional logic selecting between NSM and QKD security models is forbidden: **NSM is the only model**.

---

## 5. Verification & Definition of Done

### 5.1 The "Grep" Test (Zero Tolerance for QKD Security Logic)
Derived from:
- Phase I checkpoint CP-PHI-005 ("all references to legacy code removed") using code review + grep.

The sprint FAILS if any of the following remain in **security-critical paths** after the refactoring PR(s):

- QKD-derived security-bound markers for Phase IV key-length selection:
  - `1 - h(` used as a security bound (entropy-rate style)
  - `binary_entropy(` used as the *primary* security formula (not as a generic utility)
  - References to Tomamichel finite-key as the *primary* security basis for E-HOK Phase IV length selection

- OT output violations:
  - Any public API that returns only a single final key for OT output formatting (Phase IV requires $(S_0,S_1)$ and $(S_C,C)$ structured outputs).

### 5.2 Algorithm Preservation Verification

The sprint FAILS if:
- LDPC encoding/decoding tests fail after refactoring
- Toeplitz hashing produces different outputs for identical inputs
- Basis matching produces different index sets for identical inputs
- Statistical bound calculations produce different values

### 5.3 CI Validation (Fresh Clone)
Derived from the master roadmap quality gates and "Green build constraint".

- CI must pass on a clean checkout with:
  - Type check (mypy strict)
  - Lint gate (flake8)
  - Unit and integration tests (per repository CI definition)

### 5.4 Documentation Update
Derived from Phase II/III/IV refactoring plans.

- Update documentation and docstrings to:
  - Clarify that SquidASM provides communication infrastructure, not cryptographic algorithms
  - Reference new module organization if any files are restructured
  - Remove any statement suggesting QKD security bounds are acceptable substitutes for NSM

### 5.5 Definition of Done (Hard Gate)
This sprint is DONE only when:

1. **Algorithms Preserved**: All Phase-II/III/IV algorithmic components (LDPC, Toeplitz, sifting math, statistical bounds) pass their unit tests unchanged.
2. **Orchestration Refactored**: Protocol orchestration uses SquidASM sockets with proper async patterns.
3. **Security Logic Replaced**: No codepaths remain that implement QKD-derived final-length selection; all use NSM Max Bound.
4. **OT Output Correct**: Phase IV produces $(S_0, S_1)$ for Alice and $(S_C, C)$ for Bob.
5. **Static Checks Pass**: Using the exact commands in Section 4.
6. **Grep Test Passes**: No QKD security formulas in security-critical paths per Section 5.1.

---

## Appendix A: Summary of What to KEEP vs. DELETE

### A.1 KEEP (Algorithmic Core)

| Module | Component | Reason |
|--------|-----------|--------|
| `ehok/implementations/reconciliation/` | LDPC matrix operations | Phase III: "architecturally sound" |
| `ehok/implementations/reconciliation/` | Belief Propagation decoder | Phase III: "architecturally sound" |
| `ehok/implementations/reconciliation/` | Polynomial hash verification | Valid cryptographic primitive |
| `ehok/implementations/privacy_amplification/` | Toeplitz matrix multiplication | Phase IV: "extract mechanics" |
| `ehok/implementations/privacy_amplification/` | Seed generation | Valid cryptographic primitive |
| `ehok/core/sifting.py` | Basis matching functions | Phase II: "valid math" |
| `ehok/core/sifting.py` | QBER computation | Statistical calculation |
| `ehok/core/sifting.py` | Chernoff/Hoeffding bounds | Statistical calculation |

### A.2 DELETE (Orchestration/Security Logic)

| Target | Reason |
|--------|--------|
| `compute_final_length_finite_key()` using $1-h(Q)$ | Invalid QKD bound for NSM |
| Single-key output format | OT requires $(S_0,S_1)$, $(S_C,C)$ |
| Synchronous protocol runners without SquidASM sockets | Incompatible with framework |
| Leakage tracking without $L_{max}$ enforcement | Security gap |
| QBER threshold without finite-size penalty | Security gap |
| Legacy configuration acceptance | Breaking change by design |

---

## References

1. Phase III Analysis: "The legacy `ehok/` implementation provides a comprehensive LDPC reconciliation infrastructure that is **architecturally sound** for E-HOK."

2. Phase IV Analysis: "The gap analysis reveals that the legacy `ehok/` implementation provides a complete Toeplitz hashing infrastructure but operates under **QKD security bounds rather than NSM bounds**."

3. Lupo et al. (2023): Max Bound formula $h_{min}(r) = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$

4. Master Roadmap: TDM protocol `WRAP → REPLICATE → VALIDATE → DELETE`

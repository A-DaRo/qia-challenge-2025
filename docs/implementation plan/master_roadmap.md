# Master Implementation Roadmap: E-HOK on SquidASM

> **Version**: 1.0.0  
> **Classification**: Chief Protocol Architect / Technical Program Manager  
> **Scope**: Executive-level synthesis of Phases I–IV deep-dive analyses  
> **Last Updated**: Session timestamp  

---

## Document Purpose and Philosophy

This document is the **operational master plan** for implementing E-HOK (Entanglement-based Honest Oblivious Key) within the SquidASM simulation framework. It synthesizes the four phase-specific technical analyses into a unified, actionable project roadmap.

**Guiding Principles**:

1. **Synthesize, Not Repeat**: This document hyperlinks to phase analyses for mathematical proofs. It operationalizes findings into tasks.
2. **Aggressive Legacy Removal**: Legacy `ehok` components are wrapped, replicated in SquidASM context, validated for parity, then **permanently deleted**. No rollback option.
3. **Test-Driven Migration (TDM)**: Every component has validation tests defined *before* migration. Green tests gatekeep the critical path and must pass before legacy code removal.
4. **No Code Here**: This document defines *what* to build and *when*—not *how* to implement.

---

## Table of Contents

1. [Strategic Overview](#1-strategic-overview)
2. [Quality Assurance & Validation Framework](#2-quality-assurance--validation-framework)
3. [Master Dependency Graph](#3-master-dependency-graph)
4. [Iterative MoSCoW Roadmap](#4-iterative-moscow-roadmap)
5. [Risk Management & Contingencies](#5-risk-management--contingencies)
6. [Operational Metrics & Success Criteria](#6-operational-metrics--success-criteria)
7. [Appendix: Reference Links](#appendix-reference-links)

---

## 1. Strategic Overview

### 1.1 Project Timeline

| Sprint | Duration | Theme | Deliverables |
|--------|----------|-------|--------------|
| **Sprint 0** | Days 1–3 | Foundation | CI scaffold, test infrastructure, dataclass contracts |
| **Sprint 1** | Days 4–10 | Security Core | NSM bounds calculator, timing enforcement, feasibility pre-check |
| **Sprint 2** | Days 11–20 | Protocol Layer | Ordered messaging, detection validation, LDPC safety cap |
| **Sprint 3** | Days 21–30 | Output & Integration | Oblivious key formatter, end-to-end pipeline, validation suite |

**Total Duration**: ~6 weeks (including buffer for unforeseen blockers)

### 1.2 Migration Methodology: Aggressive Legacy Removal

The **Aggressive Removal** strategy replaces legacy code component-by-component without maintaining dual implementations or rollback options:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Aggressive Legacy Removal Pattern                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 1: WRAP                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Create abstraction layer (interfaces) around legacy components      │   │
│   │ Legacy code unchanged; new code programs to interface               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│   Phase 2: REPLICATE                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Implement SquidASM-native versions satisfying the same interface    │   │
│   │ New implementation coexists temporarily during validation            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼                                             │
│   Phase 3: VALIDATE                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Run both implementations in parallel; compare outputs               │   │
│   │ Test suite must show byte-for-byte parity on deterministic runs     │   │
│   │ Validation GATES legacy code deletion—no removal without proof      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼ Parity Tests Pass                          │
│   Phase 4: DELETE (No Rollback)                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ PERMANENTLY remove legacy code from repository                      │   │
│   │ Update all internal imports to reference SquidASM-native version    │   │
│   │ No deprecation period, no feature flag fallback                     │   │
│   │ Version bump reflects breaking change                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Differences from Strangler Fig**:
- **No Feature Flags**: New implementation becomes the only path
- **No Long-term Dual Maintenance**: Eliminates technical debt from maintaining two code paths
- **Faster Iteration**: Validation gates immediate deletion, preventing stale code accumulation
- **Clear Responsibility**: Each developer knows exactly which implementation is in use

### 1.3 Global Definitions

| Term | Definition | Reference |
|------|------------|-----------|
| **NSM** | Noisy Storage Model — security derived from adversary's storage decoherence, not computational limits | [Lupo 2023] §2 |
| **Δt** | Mandatory wait time between Bob's receipt acknowledgment and Alice's basis reveal; NSM cornerstone | [Phase I Analysis](phase_I_analysis.md#44-timing-primitive-τ) |
| **Max Bound** | $h_{min} = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$; replaces QKD entropy formula | [Phase IV Analysis](phase_IV_analysis.md#31-nsm-max-bound) |
| **Oblivious Output** | Alice: $(S_0, S_1)$; Bob: $(S_C, C)$ — 1-out-of-2 OT structure | [Phase IV Analysis](phase_IV_analysis.md#45-oblivious-output-formatting) |
| **Wiretap Cost** | Syndrome length $|\Sigma|$ subtracted from min-entropy during reconciliation | [Phase III Analysis](phase_III_analysis.md#32-wiretap-channel-formalization) |
| **QBER** | Quantum Bit Error Rate — noise measured during Phase II sifting | [Phase II Analysis](phase_II_analysis.md#42-qber-estimation) |

### 1.4 Success Definition

The project is **complete** when:

1. End-to-end simulation produces valid oblivious keys with $\varepsilon_{sec} \leq 10^{-6}$
2. All MUST items pass validation gates
3. Security bounds use NSM "Max Bound" (not QKD formulas)
4. Timing enforcement prevents basis reveal before $\Delta t$ elapsed
5. Integration tests run in CI without manual intervention

---

## 2. Quality Assurance & Validation Framework

### 2.1 Testing Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Testing Pyramid                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌───────────────┐                                 │
│                           │   End-to-End  │  ← Full protocol simulation     │
│                           │   (E2E)       │  ← Run weekly / before release  │
│                           └───────────────┘                                 │
│                          ╱                 ╲                                │
│                         ╱                   ╲                               │
│                ┌───────────────┐     ┌───────────────┐                      │
│                │  Integration  │     │  Integration  │  ← Phase boundaries  │
│                │  (Phase I→II) │     │  (III→IV)     │  ← Run daily in CI   │
│                └───────────────┘     └───────────────┘                      │
│               ╱                                       ╲                     │
│              ╱                                         ╲                    │
│     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│     │    Unit       │  │    Unit       │  │    Unit       │  ← Components  │
│     │ (NSMBounds)   │  │ (LeakageMgr)  │  │ (Toeplitz)    │  ← Run on push │
│     └───────────────┘  └───────────────┘  └───────────────┘                │
│    ╱                                                       ╲               │
│   ╱                                                         ╲              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Static Analysis + Type Checking                    │ │
│  │                     (mypy, pylint, Pylance)                           │ │
│  │                     Run on every push                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Validation Categories

| Category | Purpose | Trigger | Failure Action |
|----------|---------|---------|----------------|
| **Structural** | Verify dataclass contracts between phases | Pre-commit hook | Block commit |
| **Mathematical** | Validate security formulas against literature | Unit test | Block merge |
| **Behavioral** | Confirm protocol flow and abort conditions | Integration test | Flag for review |
| **Statistical** | Verify error rates match theoretical bounds | E2E test | Re-run with higher $n$ |

### 2.3 CI/CD Gate Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CI Pipeline Stages                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: Quality Gate (Every Push)                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ ✓ Static type checking (mypy --strict)                              │   │
│   │ ✓ Linting (flake8, pylint)                                          │   │
│   │ ✓ Docstring validation (pydocstyle --convention=numpy)              │   │
│   │ ✓ Unit tests (pytest -m unit)                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼ Pass                                        │
│   STAGE 2: Integration Gate (Every PR)                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ ✓ Integration tests (pytest -m integration)                         │   │
│   │ ✓ Coverage threshold (≥80%)                                         │   │
│   │ ✓ Security formula validation tests                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                             │
│                                ▼ Pass                                        │
│   STAGE 3: E2E Gate (Pre-Merge to Main)                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ ✓ Full protocol simulation (n=10000)                                │   │
│   │ ✓ QBER within expected bounds                                       │   │
│   │ ✓ Positive secure key rate                                          │   │
│   │ ✓ Abort codes triggered under adversarial conditions                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Test-Driven Migration Protocol

For each legacy component being migrated:

1. **Define Interface**: Create abstract base class with expected signature
2. **Write Tests Against Interface**: Tests reference interface, not implementation
3. **Implement New Component**: SquidASM-native version satisfies interface
4. **Parity Test**: Run both implementations with fixed seed; assert identical output
5. **Validation Gate**: Parity test must pass with ≥99% coverage overlap before proceeding
6. **Delete Legacy**: Remove legacy implementation file and all references from codebase
7. **Update Imports**: Redirect all internal and test imports to new SquidASM-native version
8. **Verification**: Confirm test suite runs without fallback code paths

**No Rollback Option**: Once legacy code is deleted, there is no mechanism to restore it. This enforces forward momentum and prevents accumulation of duplicated logic.

---

## 3. Master Dependency Graph

### 3.1 Cross-Phase Critical Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Critical Path DAG                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SPRINT 0: Foundation                                                       │
│   ══════════════════                                                        │
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │  TASK-INFRA-001: CI/CD Pipeline Setup                             │     │
│   │  TASK-INFRA-002: Dataclass Contracts (Phase I/O Boundaries)       │     │
│   │  TASK-INFRA-003: Logging Infrastructure (LogManager)              │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                 │                                            │
│   ══════════════════════════════╪════════════════════════════════════════   │
│                                 │                                            │
│   SPRINT 1: Security Core       ▼                                            │
│   ═══════════════════════════════                                           │
│                                                                              │
│   ┌──────────────────────────┐                                              │
│   │ TASK-NSM-001             │  NSMBoundsCalculator                         │
│   │ [CRITICAL - BLOCKS ALL]  │  max_bound_entropy(r), dupuis_konig()        │
│   └───────────┬──────────────┘                                              │
│               │                                                              │
│       ┌───────┴───────┬───────────────────────────────┐                     │
│       │               │                               │                     │
│       ▼               ▼                               ▼                     │
│   ┌──────────┐   ┌──────────────┐           ┌────────────────────┐         │
│   │TASK-FEAS │   │TASK-TIMING   │           │TASK-NOISE-ADAPTER  │         │
│   │-001      │   │-001          │           │-001                │         │
│   │Preflight │   │TimingEnforcer│           │PhysicalModelAdapter│         │
│   │Feasibility│  │(Δt barrier)  │           │(μ,η,e_det→NetSquid)│         │
│   └─────┬────┘   └──────┬───────┘           └─────────┬──────────┘         │
│         │               │                             │                     │
│   ══════╪═══════════════╪═════════════════════════════╪══════════════════   │
│         │               │                             │                     │
│   SPRINT 2: Protocol Layer                                                   │
│   ════════════════════════                                                  │
│         │               │                             │                     │
│         ▼               ▼                             ▼                     │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                     Phase I Integration Point                     │     │
│   │                     (Quantum Generation Ready)                    │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐     │
│   │TASK-ORDERED  │          │TASK-DETECT   │          │TASK-FINITE   │     │
│   │-MSG-001      │          │-VALID-001    │          │-SIZE-001     │     │
│   │OrderedSocket │          │ChernoffValid │          │FiniteSizePen │     │
│   │(ACK ordering)│          │(detection OK)│          │(μ penalty)   │     │
│   └──────┬───────┘          └──────┬───────┘          └──────┬───────┘     │
│          │                         │                         │              │
│          └─────────────────────────┼─────────────────────────┘              │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                     Phase II Integration Point                    │     │
│   │                     (Sifting & Estimation Ready)                  │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                         │
│          ┌─────────────────────────┴─────────────────────────┐              │
│          │                                                   │              │
│          ▼                                                   ▼              │
│   ┌──────────────┐                                   ┌──────────────┐      │
│   │TASK-LDPC     │                                   │TASK-LEAKAGE  │      │
│   │-SAFETY-001   │                                   │-MGR-001      │      │
│   │LDPCReconcil  │◀─────────────────────────────────▶│LeakageSafety │      │
│   │(existing OK) │                                   │Manager (NEW) │      │
│   └──────┬───────┘                                   └──────┬───────┘      │
│          │                                                  │               │
│          └──────────────────────────┬───────────────────────┘               │
│                                     │                                        │
│                                     ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    Phase III Integration Point                    │     │
│   │                    (Reconciliation Ready)                         │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                         │
│   ═════════════════════════════════╪═════════════════════════════════════   │
│                                    │                                         │
│   SPRINT 3: Output & Integration   │                                         │
│   ══════════════════════════════════                                        │
│                                    │                                         │
│          ┌─────────────────────────┴─────────────────────────┐              │
│          │                                                   │              │
│          ▼                                                   ▼              │
│   ┌──────────────┐                                   ┌──────────────┐      │
│   │TASK-TOEPLITZ │                                   │TASK-OBLIV    │      │
│   │-MODIFY-001   │                                   │-FORMAT-001   │      │
│   │Replace       │──────────────────────────────────▶│ObliviousKey  │      │
│   │compute_final │                                   │Formatter     │      │
│   │_length()     │                                   │(S_0,S_1)/(S_C,C)│   │
│   └──────┬───────┘                                   └──────┬───────┘      │
│          │                                                  │               │
│          └──────────────────────────┬───────────────────────┘               │
│                                     │                                        │
│                                     ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    Phase IV Integration Point                     │     │
│   │                    (Oblivious Output Ready)                       │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    TASK-E2E-001                                   │     │
│   │                    End-to-End Integration Suite                   │     │
│   │                    [VALIDATION MILESTONE]                         │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Blocking Dependencies Summary

| Blocked Task | Depends On | Rationale |
|--------------|------------|-----------|
| All security calculations | **TASK-NSM-001** | NSM bounds are prerequisite for all length calculations |
| Quantum generation | **TASK-TIMING-001** | Cannot proceed without Δt enforcement primitive |
| Phase II sifting | **TASK-FEAS-001** | Must validate feasibility before consuming resources |
| Reconciliation | Phase II validated | Cannot reconcile without sifted key material |
| Privacy amplification | **TASK-LEAKAGE-MGR-001** | Must track wiretap cost from Phase III |
| Oblivious output | **TASK-OBLIV-FORMAT-001** | E-HOK requires $(S_0, S_1)$, $(S_C, C)$ structure |

---

## 4. Iterative MoSCoW Roadmap

### 4.1 Sprint 0: Foundation (Days 1–3)

**Objective**: Establish infrastructure, contracts, and development workflow.

| ID | Task | Priority | Status | Validation |
|----|------|----------|--------|------------|
| INFRA-001 | Configure CI pipeline (GitHub Actions) | MUST | TO DO | Pipeline runs on push |
| INFRA-002 | Define phase I/O dataclasses | MUST | TO DO | Type checker passes |
| INFRA-003 | Establish logging infrastructure | MUST | TO DO | LogManager functional |
| INFRA-004 | Create test fixtures for deterministic runs | SHOULD | TO DO | Seeded tests produce identical output |

**Deliverables**:
- `ehok/core/data_structures.py` — Phase boundary dataclasses
- `.github/workflows/ci.yml` — CI pipeline configuration
- `ehok/utils/logging.py` — LogManager (if not already complete)

---

### 4.2 Sprint 1: Security Core (Days 4–10)

**Objective**: Implement the NSM security foundation that all other phases depend on.

#### 4.2.1 MUST Items

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| NSM-001 | Implement `NSMBoundsCalculator` with Max Bound | `ehok/analysis/nsm_bounds.py` | Unit tests against [Lupo 2023] Table 1 |
| TIMING-001 | Implement `TimingEnforcer` for Δt barrier | `ehok/core/timing.py` | Simulation trace shows barrier markers |
| FEAS-001 | Implement pre-flight feasibility check | `ehok/core/feasibility.py` | Abort triggered at Q > 22% |
| NOISE-PARAMS-001 | Expose NSM params in config schema | `ehok/configs/protocol_config.py` | Schema accepts μ, η, e_det |

**NSMBoundsCalculator Contract** (see [Phase IV Analysis](phase_IV_analysis.md#31-nsm-max-bound)):

```
Input:  storage_noise_r: float ∈ [0, 0.5]
        adjusted_qber: float ∈ [0, 0.22]
        total_leakage: int (syndrome bits)
        epsilon_sec: float (e.g., 1e-6)
        
Output: max_secure_key_length: int
        min_entropy_per_bit: float
        feasibility_status: FeasibilityResult
```

#### 4.2.2 SHOULD Items

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| NOISE-ADAPTER-001 | Physical model adapter (μ,η → NetSquid) | `ehok/quantum/noise_adapter.py` | Conversion matches [Erven 2014] |
| TRANSCRIPT-001 | Protocol transcript object | `ehok/protocols/transcript.py` | Transcript auditable for ordering |

---

### 4.3 Sprint 2: Protocol Layer (Days 11–20)

**Objective**: Implement classical protocol components: ordered messaging, detection validation, sifting, and reconciliation safety.

#### 4.3.1 Phase I → II Transition Components

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| ORDERED-MSG-001 | `OrderedProtocolSocket` wrapper | `ehok/protocols/ordered_messaging.py` | ACK ordering enforced |
| DETECT-VALID-001 | `DetectionValidator` with Chernoff | `ehok/quantum/detection.py` | Abort on detection anomaly |
| FINITE-SIZE-001 | Finite-size penalty calculator | `ehok/analysis/statistics.py` | μ matches [Scarani 2009] formula |
| QBER-ADJUST-001 | Update abort threshold to use adjusted QBER | `ehok/core/sifting.py` | Tests use e_obs + μ |

**Ordered Messaging Invariant**:
> Bob's detection report must be received and acknowledged before Alice reveals any basis information.

#### 4.3.2 Phase III Components

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| LEAKAGE-MGR-001 | `LeakageSafetyManager` enforcing L_max | `ehok/core/security_bounds.py` | Abort on cap exceeded |
| LDPC-INTEGRATE-001 | Integrate safety manager with `LDPCReconciliator` | `ehok/implementations/reconciliation.py` | Leakage tracked per-block |
| RECON-PROTOCOL-001 | `ReconciliationOrchestrator` with SquidASM sockets | `ehok/protocols/reconciliation_protocol.py` | End-to-end block correction |

**Wiretap Cost Invariant** (see [Phase III Analysis](phase_III_analysis.md#32-wiretap-channel-formalization)):
> Total syndrome leakage |Σ| must be tracked and passed to Phase IV for min-entropy deduction.

---

### 4.4 Sprint 3: Output & Integration (Days 21–30)

**Objective**: Complete privacy amplification, oblivious output formatting, and end-to-end validation.

#### 4.4.1 Phase IV Components

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| TOEPLITZ-MODIFY-001 | Replace `compute_final_length()` with NSM version | `ehok/implementations/privacy_amplification.py` | Uses Max Bound, not QKD |
| OBLIV-FORMAT-001 | `ObliviousKeyFormatter` for OT output | `ehok/core/oblivious_key.py` | Alice: (S_0, S_1); Bob: (S_C, C) |
| STORAGE-LINK-001 | Link Phase I storage noise config to Phase IV | `ehok/core/feasibility.py` | Parameter flows through pipeline |

**Oblivious Output Structure**:
```
Alice Output:
  - S_0: key if Bob's choice bit C=0
  - S_1: key if Bob's choice bit C=1
  
Bob Output:
  - S_C: S_{Bob's choice bit}
  - C: Bob's choice bit
```

#### 4.4.2 Integration & Validation

| ID | Task | File(s) | Validation |
|----|------|---------|------------|
| E2E-PIPELINE-001 | End-to-end protocol orchestrator | `ehok/protocols/ehok_protocol.py` | Full run produces valid keys |
| E2E-ADVERSARIAL-001 | Adversarial condition tests | `ehok/tests/test_adversarial.py` | Correct aborts under attack |
| E2E-STATISTICAL-001 | Statistical validation suite | `ehok/tests/test_statistical.py` | Key rate matches theory |

---

### 4.5 Files To Create Summary

| Phase | File | Purpose | Priority |
|-------|------|---------|----------|
| Core | `ehok/analysis/nsm_bounds.py` | NSM Max Bound calculator | MUST |
| Core | `ehok/core/timing.py` | TimingEnforcer (Δt barrier) | MUST |
| Core | `ehok/core/feasibility.py` | Pre-flight feasibility check | MUST |
| Core | `ehok/core/oblivious_key.py` | Oblivious output formatter | MUST |
| Core | `ehok/core/security_bounds.py` | LeakageSafetyManager | MUST |
| I/II | `ehok/protocols/ordered_messaging.py` | OrderedProtocolSocket | MUST |
| II | `ehok/quantum/detection.py` | DetectionValidator, DetectionReport | SHOULD |
| II | `ehok/analysis/statistics.py` | Finite-size penalty, adjusted QBER | MUST |
| III | `ehok/protocols/reconciliation_protocol.py` | ReconciliationOrchestrator | MUST |
| IV | `ehok/core/oblivious_key.py` | ObliviousKeyFormatter | MUST |
| Int | `ehok/protocols/ehok_protocol.py` | End-to-end orchestrator | MUST |

---

## 5. Risk Management & Contingencies

### 5.1 Risk Registry

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| RISK-001 | NetSquid timing API incompatible with Δt enforcement | Medium | Critical | Investigate `ns.sim_time()` access early in Sprint 1 |
| RISK-002 | LDPC efficiency cliff at moderate QBER (>10%) | High | High | Implement adaptive rate selection; pre-flight feasibility check |
| RISK-003 | Finite-key "Death Valley" causes zero key rate | Medium | High | Batch feasibility pre-check before quantum phase |
| RISK-004 | Legacy code assumptions break under SquidASM | Medium | Medium | Comprehensive parity tests before deletion + early detection via CI |
| RISK-005 | Oblivious output structure incompatible with MPC use case | Low | High | Validate output format with application layer early |

### 5.2 Contingency Protocols

**RISK-001 Contingency**: If `ns.sim_time()` is not accessible from SquidASM programs:
1. Implement timing enforcement at NetSquid level via custom `Protocol` subclass
2. Document architectural divergence from pure SquidASM approach
3. Create issue for SquidASM maintainers requesting timing primitive

**RISK-003 Contingency**: If minimum batch size exceeds practical limits:
1. Research super-batch accumulation across multiple sessions
2. Consider relaxed security parameter (ε = 10⁻³ instead of 10⁻⁶)
3. Document trade-off in protocol configuration

### 5.3 Decision Log Template

| Date | Decision | Rationale | Alternatives Considered | Owner |
|------|----------|-----------|-------------------------|-------|
| — | — | — | — | — |

---

## 6. Operational Metrics & Success Criteria

### 6.1 Key Performance Indicators (KPIs)

| KPI | Symbol | Target | Measurement Method |
|-----|--------|--------|-------------------|
| **Secure Key Rate** | $R_{sk}$ | $> 0$ bits/round | $R_{sk} = \frac{\ell}{n} \cdot P_{sift}$ where $\ell$ is final key length, $n$ is raw pairs |
| **Composable Security** | $\varepsilon_{tot}$ | $\leq 10^{-6}$ | $\varepsilon_{tot} = \varepsilon_{cor} + \varepsilon_{sec}$ from finite-key analysis |
| **QBER Observed** | $Q_{obs}$ | $< 11\%$ warning, $< 22\%$ hard limit | Sample $k$ bits during Phase II |
| **Min-Entropy per Bit** | $h_{min}$ | $> 0$ | From NSM Max Bound: $h_{min} = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$ |
| **Reconciliation Efficiency** | $f_{EC}$ | $\geq 0.95$ | $f_{EC} = \frac{H(Q)}{|\Sigma|/n}$ where $H(Q)$ is binary entropy |

### 6.2 Rate Formulas

**Asymptotic Secure Key Rate** (ideal limit):
$$R_{sk}^{asymp} = P_{sift} \cdot \left[ h_{min}(Q, r) - f_{EC} \cdot H(Q) \right]$$

**Finite-Key Secure Key Rate** (practical):
$$R_{sk}^{finite} = P_{sift} \cdot \left[ h_{min}(Q + \mu, r) - f_{EC} \cdot H(Q) - \Delta_n(\varepsilon_{sec}) \right]$$

where:
- $P_{sift} \approx 0.5$ (matching basis probability)
- $\mu = \sqrt{\frac{\ln(2/\varepsilon_{PE})}{2k}}$ (finite-size statistical penalty)
- $\Delta_n(\varepsilon) = \frac{2\log_2(1/\varepsilon)}{n}$ (privacy amplification penalty)

### 6.3 Validation Checkpoints

| Milestone | Validation Test | Pass Criterion |
|-----------|-----------------|----------------|
| Sprint 1 Complete | NSM bounds unit tests | Output matches [Lupo 2023] Table 1 within 0.1% |
| Sprint 2 Complete | Phase I→II integration | Sifted key generated without abort at Q=5% |
| Sprint 3 Complete | Full E2E simulation | Positive key rate at Q=5%, n=10,000 |
| Release Ready | Adversarial suite | All abort conditions triggered correctly |

### 6.4 Abort Code Taxonomy

| Code | Phase | Condition | Response |
|------|-------|-----------|----------|
| `ABORT-I-FEAS-001` | I | Pre-flight: Q_expected > 22% | Terminate before EPR generation |
| `ABORT-II-DETECT-001` | II | Detection rate Chernoff violation | Terminate; possible active attack |
| `ABORT-II-QBER-001` | II | Adjusted QBER > 22% | Terminate; channel too noisy |
| `ABORT-III-LEAKAGE-001` | III | Syndrome leakage exceeds L_max | Terminate; security compromised |
| `ABORT-IV-FEAS-001` | IV | Min-entropy insufficient for key length | Terminate; Death Valley |

---

## Appendix: Reference Links

### Phase Analysis Documents

- [Phase I Analysis: Quantum Generation](phase_I_analysis.md) — Physical layer, timing, feasibility
- [Phase II Analysis: Sifting & Estimation](phase_II_analysis.md) — Detection, basis matching, QBER
- [Phase III Analysis: Information Reconciliation](phase_III_analysis.md) — LDPC, wiretap cost, safety cap
- [Phase IV Analysis: Privacy Amplification](phase_IV_analysis.md) — NSM bounds, oblivious output

### Project Infrastructure

| Resource | Purpose |
|----------|---------|
| `ehok/` | Implementation workspace |
| `squidasm/` | Framework source (read-only reference) |
| `qia/lib/python3.10/site-packages/netqasm/` | SDK reference |

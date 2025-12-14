# Remediation Integration and Refactoring Guide

> **Document ID**: REF-001 / E-HOK-on-SquidASM  
> **Classification**: Lead Software Architect & Code Refactoring Specialist  
> **Date**: 2025-12-14  
> **References**: `remediation_specification.md`, `system_test_report.md`, `master_roadmap.md`  
> **Status**: Active Refactoring Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architectural Critique](#2-high-level-architectural-critique)
3. [Core Components Analysis](#3-core-components-analysis)
4. [Physical Layer Analysis](#4-physical-layer-analysis)
5. [Protocol Components Analysis](#5-protocol-components-analysis)
6. [Implementation Layer Analysis](#6-implementation-layer-analysis)
7. [Utilities & Interfaces Analysis](#7-utilities--interfaces-analysis)
8. [Cleanup Manifest](#8-cleanup-manifest)
9. [Master Refactoring Plan](#9-master-refactoring-plan)

---

## 1. Executive Summary

### 1.1 Purpose

This document bridges the **Remediation Specification** (defining *what* to build) with the **existing codebase reality** (defining *where* and *how* to integrate). The objective is to ensure new components are **organically woven** into the architecture rather than bolted on as afterthoughts.

### 1.2 Architectural Health Summary

| Layer | Current State | Primary Issues | Remediation Impact |
|-------|---------------|----------------|-------------------|
| **ehok/protocols/** | ⚠️ Functional | Raw socket usage, missing timing enforcement | OrderedProtocolSocket, TimingEnforcer injection |
| **ehok/core/** | ✅ Solid | Minor API gaps (batch_size in FeasibilityInputs) | API alignment, tighter NSM integration |
| **ehok/quantum/** | ⚠️ Partial | Missing PhysicalModelAdapter class | Critical gap closure |
| **ehok/analysis/** | ✅ Complete | None identified | Stable reference point |
| **ehok/implementations/** | ✅ Functional | Leakage tracking external | LeakageSafetyManager hookup |
| **ehok/configs/** | ⚠️ Scattered | Dual config schemas (config.py vs protocol_config.py) | Consolidation required |

### 1.3 Key Integration Patterns Required

1. **Physical Model Adapter**: Bridge NSM theory → SquidASM simulation
2. **Ordered Messaging**: Replace raw `csocket` usage with `OrderedProtocolSocket`
3. **Timing Enforcement**: Inject `TimingEnforcer` into protocol control flow
4. **Leakage Tracking**: Wire `LeakageSafetyManager` into reconciliation loop
5. **Configuration Unification**: Consolidate scattered config schemas

---

## 2. High-Level Architectural Critique

### 2.1 Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          E-HOK Architecture Overview                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐   │
│  │   protocols/    │──────▶│      core/      │◀──────│  implementations/│   │
│  │  alice.py       │       │  feasibility.py │       │  reconciliation/ │   │
│  │  bob.py         │       │  timing.py      │       │  privacy_amp/    │   │
│  │  base.py        │       │  sifting.py     │       │  commitment/     │   │
│  └────────┬────────┘       └────────┬────────┘       └─────────────────┘   │
│           │                         │                                        │
│           │  Direct calls           │  NSM Math                             │
│           ▼                         ▼                                        │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐   │
│  │    quantum/     │       │    analysis/    │       │    configs/      │   │
│  │  runner.py      │       │  nsm_bounds.py  │       │  protocol_*.py   │   │
│  │  noise_adapter  │       │  metrics.py     │       │  config.py       │   │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘   │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════   │
│                           SquidASM / NetSquid Layer                          │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Global Design Flaws Identified

#### Flaw 1: **Configuration Schema Duplication** (Scattershot Configuration)

**Evidence:**
- `ehok/core/config.py`: Defines `QuantumConfig`, `SecurityConfig`, `ReconciliationConfig`, `NSMConfig`, `PrivacyAmplificationConfig`, `ProtocolConfig`
- `ehok/configs/protocol_config.py`: Defines `PhysicalParameters`, `NSMSecurityParameters`, `ProtocolParameters`, `ProtocolConfig`

**Problem:** Two parallel configuration hierarchies with overlapping concerns:
- `core/config.py::NSMConfig` vs `configs/protocol_config.py::NSMSecurityParameters`
- Different naming conventions and validation approaches
- No clear ownership boundary

**Impact:** Protocols must juggle two config objects, leading to confusion about which `ProtocolConfig` to use.

**Remediation:** See Section 8.1 for consolidation plan.

---

#### Flaw 2: **Raw Socket Usage in Protocols** (Tight Coupling)

**Evidence** (`ehok/protocols/alice.py` lines 106-149):
```python
# Direct raw socket access
commitment_msg = yield from self.context.csockets[self.PEER_NAME].recv()
# ...
self.context.csockets[self.PEER_NAME].send(bases_msg)
```

**Problem:** 
- No ordering enforcement on message exchange
- No automatic ACK tracking
- Timing barrier bypass possible
- `OrderedProtocolSocket` exists but is not used in actual protocol flows

**Code Smell:** **Hardcoded Wiring** – Protocols directly access `context.csockets[]` rather than injecting a socket abstraction.

**Remediation:** Replace all raw `csockets[]` access with `OrderedProtocolSocket` wrapper. See Section 5.3.

---

#### Flaw 3: **Missing Timing Enforcement in Control Flow**

**Evidence:** `TimingEnforcer` class exists (`ehok/core/timing.py`) with full implementation, but:
- `AliceBaselineEHOK._phase3_sifting_sampling()` sends bases without timing check
- `BobBaselineEHOK._phase3_sifting_sampling()` receives bases without verifying Δt elapsed
- No integration point between `TimingEnforcer` and `EHOKRole.run()`

**Code Smell:** **Orphan Implementation** – `TimingEnforcer` is complete but unused.

**Remediation:** Inject timing checks at protocol phase boundaries. See Section 5.4.

---

#### Flaw 4: **Physical Model Adapter Gap** (Theory-Simulation Disconnect)

**Evidence** (`ehok/quantum/noise_adapter.py`):
- `SimulatorNoiseParams` dataclass ✅
- `physical_to_simulator()` function ✅
- **Missing:** `PhysicalModelAdapter` class
- **Missing:** `estimate_storage_noise_from_netsquid()` function

**Problem:** No code path to:
1. Translate `PhysicalParameters` → `StackNetworkConfig`
2. Extract NSM storage noise `r` from NetSquid T1/T2 parameters
3. Bridge `TimingEnforcer` with simulation time queries

**Impact:** System test failures SYS-INT-NOISE-001, SYS-INT-NOISE-002.

**Remediation:** Implement `PhysicalModelAdapter` per `remediation_specification.md` Section 2.

---

#### Flaw 5: **Lack of Dependency Injection**

**Evidence** (`ehok/protocols/base.py` lines 48-59):
```python
def _build_strategies(self) -> None:
    self.commitment_scheme = factories.build_commitment_scheme(self.config)
    self.sampling_strategy = factories.build_sampling_strategy(self.config)
    self.noise_estimator = factories.build_noise_estimator(self.config)
```

**Problem:** Strategy creation is hardcoded to factory module. No ability to:
- Inject mock implementations for testing
- Override strategies without modifying factories
- Support runtime strategy switching

**Pattern Recommendation:** Introduce optional constructor injection:
```python
def __init__(
    self,
    config: ProtocolConfig | None = None,
    ordered_socket: OrderedProtocolSocket | None = None,  # NEW
    timing_enforcer: TimingEnforcer | None = None,        # NEW
    **kwargs
):
```

---

### 2.3 Circular Dependency Analysis

**Finding:** No circular dependencies detected at module level.

Dependency flow is correctly unidirectional:
```
protocols → core → analysis
protocols → implementations → interfaces
protocols → quantum
```

`implementations/` correctly depends on `interfaces/` (abstractions), not concrete protocols.

---

## 3. Core Components Analysis

### 3.1 `ehok/core/config.py` – Runtime Configuration

**Status:** ⚠️ Needs Consolidation

**Code Smell: Primitive Obsession**

The `ProtocolConfig` class aggregates multiple sub-configs but passes them through many layers:
```python
@dataclass
class ProtocolConfig:
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    privacy_amplification: PrivacyAmplificationConfig = field(...)
    nsm: NSMConfig = field(default_factory=NSMConfig)
    sampling_seed: int | None = None
```

**Integration Points:**
1. `PhysicalModelAdapter` should accept `PhysicalParameters` (from `configs/protocol_config.py`)
2. `TimingEnforcer` should accept `NSMConfig.delta_t_ns`
3. `FeasibilityChecker` should integrate both config schemas

**Refactoring Required:**
- Add `@classmethod from_physical_parameters()` to allow construction from the other config schema
- Consider deprecating one schema in favor of the other

---

### 3.2 `ehok/core/feasibility.py` – Pre-Flight Gate

**Status:** ✅ Well-Designed, Minor API Gap

**Strengths:**
- Clean `FeasibilityInputs` dataclass
- Comprehensive abort code taxonomy
- Proper integration with `NSMBoundsCalculator`

**API Gap (from `remediation_specification.md` Section 4.1):**

`FeasibilityInputs` is missing `batch_size` parameter for "Death Valley" detection in small-batch scenarios.

**Current:**
```python
@dataclass(frozen=True)
class FeasibilityInputs:
    expected_qber: float
    storage_noise_r: float
    storage_rate_nu: float
    epsilon_sec: float
    n_target_sifted_bits: int
    expected_leakage_bits: int
```

**Required Addition:**
```python
    batch_size: int = 0  # 0 = full-session mode, >0 = per-batch feasibility
```

---

### 3.3 `ehok/core/timing.py` – Timing Barrier Enforcer

**Status:** ✅ Complete Implementation, Unused

**Architecture:**
- `TimingConfig` dataclass with `delta_t_ns`
- `TimingState` enum: `UNINITIALIZED → COMMIT_RECEIVED → BARRIER_SATISFIED → BASIS_REVEALED`
- `TimingEnforcer` class with clean state machine

**Attribute Naming Note:**
The remediation spec mentions `_commit_ack_time_ns` – this matches the actual implementation (line 224).

**Integration Point:**
The `TimingEnforcer` must be injected into `EHOKRole` and called:
1. After Bob's commitment is acknowledged: `enforcer.mark_commit_received(sim_time_ns=...)`
2. Before Alice reveals bases: `enforcer.mark_basis_reveal_attempt(sim_time_ns=...)`

**Current Gap:** No protocol code calls these methods. The integration layer is missing.

---

### 3.4 `ehok/core/sifting.py` – Basis Sifting Logic

**Status:** ✅ Stateless, Clean

**Design:** Uses static methods within `SiftingManager` class – effectively a namespace for pure functions.

**No Code Smells Detected.**

---

### 3.5 `ehok/analysis/nsm_bounds.py` – NSM Security Math

**Status:** ✅ Complete, Reference Implementation

**Strengths:**
- Pure Python, no simulator dependencies
- Comprehensive docstrings with mathematical references
- `gamma_function()`, `max_bound_entropy_rate()`, `collision_entropy_rate()` all verified
- `NSMBoundsCalculator` provides stateful composition

**Test Verification (from `system_test_report.md`):**
```
max_bound_entropy_rate(0.3) = 0.7  # Matches Lupo et al. Eq. (36) ✅
```

**No Changes Required.** This module serves as the stable foundation for remediation.

---

### 3.6 `ehok/configs/protocol_config.py` – Physical Parameters

**Status:** ⚠️ Parallel Schema (See Flaw 1)

**Components:**
- `PhysicalParameters`: μ, η, e_det, P_dark (Erven defaults)
- `NSMSecurityParameters`: r, ν, Δt
- `ProtocolParameters`: ε_sec, ε_cor

**Key Difference from `core/config.py`:**
This schema is **physics-focused** (Erven et al. parameters), while `core/config.py` is **protocol-execution-focused** (batch sizes, thresholds).

**Integration Point for PhysicalModelAdapter:**
`PhysicalModelAdapter` should consume `PhysicalParameters` and produce `DepolariseLinkConfig`.

---

## 4. Physical Layer Analysis

### 4.1 `ehok/quantum/noise_adapter.py` – Current State

**Status:** ❌ Critical Gap

**What Exists:**
- `SimulatorNoiseParams` dataclass (lines 47-88) ✅
- `physical_to_simulator(PhysicalParameters) → SimulatorNoiseParams` (lines 102-175) ✅
- `estimate_qber_from_physical()` (lines 178-232) ✅
- `estimate_sifted_rate()` (lines 235-265) ✅
- `validate_physical_params_for_simulation()` (lines 273-315) ✅

**What's Missing:**
1. `PhysicalModelAdapter` class
2. `estimate_storage_noise_from_netsquid(T1, T2, delta_t) → r`
3. `to_squidasm_link_config() → DepolariseLinkConfig`

**Injection Point:**
```python
# File: ehok/quantum/noise_adapter.py
# Location: After line 340 (end of file)

# === NEW SECTION: Physical Model Adapter ===

class PhysicalModelAdapter:
    """
    Adapter bridging NSM physical parameters to SquidASM simulation config.
    
    Implements TASK-NOISE-ADAPTER-002 from remediation_specification.md.
    """
    # See remediation_specification.md Section 2.3-2.4 for full spec
```

---

### 4.2 `ehok/quantum/runner.py` – Quantum Phase Execution

**Status:** ✅ Clean Abstraction

**Architecture:**
- `QuantumPhaseResult` dataclass for outputs
- `QuantumPhaseRunner` encapsulates EPR generation and measurement
- Uses `BatchingManager` and `EPRGenerator` for batched execution

**Code Smell Check:** No raw NetSquid calls leaking into business logic. Clean separation.

**Integration Point:**
`QuantumPhaseRunner` could optionally accept a `PhysicalModelAdapter` to:
1. Validate physical parameters match network config
2. Pre-compute expected QBER for feasibility pre-check

---

### 4.3 `ehok/quantum/batching_manager.py` – Batch Control

**Status:** ✅ Single Responsibility

No changes required.

---

## 5. Protocol Components Analysis

### 5.1 `ehok/protocols/base.py` – Template Pattern

**Status:** ⚠️ Needs Injection Points

**Current Design:**
```python
class EHOKRole(Program, ABC):
    def run(self, context) -> Generator[...]:
        self.context = context
        self._build_strategies()
        quantum_runner = self._build_quantum_runner(context)
        quantum_result = yield from self._phase1_quantum(quantum_runner)
        result = yield from self._execute_remaining_phases(quantum_result)
        return result
```

**Design Pattern:** Template Method (correct choice)

**Code Smells:**
1. **Feature Envy**: `run()` directly accesses `context.csockets[]`, `context.connection`
2. **Missing Injection**: No hook for `OrderedProtocolSocket` or `TimingEnforcer`

**Refactoring Plan:**

```python
class EHOKRole(Program, ABC):
    def __init__(
        self,
        config: ProtocolConfig | None = None,
        ordered_socket_factory: Callable | None = None,  # NEW
        timing_enforcer: TimingEnforcer | None = None,    # NEW
        total_pairs: int | None = None,
        **_: Any
    ):
        self.config = config or ProtocolConfig.baseline()
        self._ordered_socket_factory = ordered_socket_factory
        self._timing_enforcer = timing_enforcer
        # ... existing init ...
    
    def _setup_ordered_socket(self) -> OrderedProtocolSocket:
        """Lazy initialization of ordered socket wrapper."""
        if self._ordered_socket is None:
            raw_socket = self.context.csockets[self.PEER_NAME]
            self._ordered_socket = OrderedProtocolSocket(
                # session_id derived from protocol context
            )
        return self._ordered_socket
```

---

### 5.2 `ehok/protocols/alice.py` – Alice Role

**Status:** ⚠️ Raw Socket Usage

**Phase Flow Analysis:**

| Phase | Method | Socket Usage | Timing Check |
|-------|--------|--------------|--------------|
| 2 | `_phase2_receive_commitment()` | `csocket.recv()` direct | None |
| 3 | `_phase3_sifting_sampling()` | `csocket.send()` direct | **MISSING: Should verify Δt before basis send** |
| 4 | `_phase4_reconciliation()` | `csocket.recv/send()` | N/A (post-basis) |
| 5 | `_phase5_privacy_amplification()` | `csocket.send()` | N/A |

**Critical Integration Point (Phase 3, line 133):**
```python
# CURRENT (INSECURE):
bases_msg = bases_alice.tobytes().hex()
self.context.csockets[self.PEER_NAME].send(bases_msg)

# REQUIRED (WITH TIMING ENFORCEMENT):
if self._timing_enforcer:
    sim_time = ns.sim_time()
    self._timing_enforcer.mark_basis_reveal_attempt(sim_time_ns=sim_time)

envelope = self._ordered_socket.create_envelope(
    MessageType.BASIS_REVEAL,
    BasisReveal(total_rounds=len(bases_alice), bases=bases_alice.tolist()).to_dict()
)
yield from self._send_with_ack(envelope)
```

---

### 5.3 `ehok/protocols/bob.py` – Bob Role

**Status:** ⚠️ Raw Socket Usage, Missing Timing Mark

**Phase Flow Analysis:**

| Phase | Method | Socket Usage | Timing Check |
|-------|--------|--------------|--------------|
| 2 | `_phase2_send_commitment()` | `csocket.send()` direct | **MISSING: Should mark commit time** |
| 3 | `_phase3_sifting_sampling()` | `csocket.recv()` direct | N/A (receives basis) |
| 4 | `_phase4_reconciliation()` | `csocket.recv/send()` | N/A |
| 5 | `_phase5_privacy_amplification()` | `csocket.recv()` | N/A |

**Critical Integration Point (Phase 2, after line 95):**
```python
# After sending commitment:
self.context.csockets[self.PEER_NAME].send(commitment.hex())
yield from self.context.connection.flush()

# ADD: Mark timing barrier start
if self._timing_enforcer:
    sim_time = ns.sim_time()
    self._timing_enforcer.mark_commit_received(sim_time_ns=sim_time)
```

---

### 5.4 `ehok/protocols/ordered_messaging.py` – Message Ordering

**Status:** ⚠️ Incomplete Generator Methods

**What Exists (lines 1-590):**
- `MessageEnvelope` dataclass ✅
- `OrderedProtocolSocket` class with:
  - `create_envelope()` ✅
  - `mark_sent()` ✅
  - `process_received()` ✅
  - `mark_timeout()` ✅
- `DetectionReport`, `BasisReveal` payloads ✅

**What's Missing (per `remediation_specification.md` Section 3):**
1. `send_with_ack()` generator method
2. `recv_and_ack()` generator method
3. `ProtocolViolation` exception class

**Injection Point (after line 600):**
```python
# === NEW SECTION: Generator-Based Async Methods ===

class ProtocolViolation(Exception):
    """Raised when protocol invariants are violated."""
    pass

class SquidASMOrderedSocket(OrderedProtocolSocket):
    """
    SquidASM-integrated version with generator-based async.
    
    Extends base OrderedProtocolSocket with yield-from compatible
    send_with_ack() and recv_and_ack() methods.
    """
    
    def __init__(self, raw_socket, session_id: str | None = None):
        super().__init__(session_id)
        self._raw_socket = raw_socket
    
    def send_with_ack(
        self,
        envelope: MessageEnvelope,
        timeout_ns: int = DEFAULT_ACK_TIMEOUT_NS
    ) -> Generator[EventExpression, None, None]:
        """
        Send message and block until ACK received.
        
        Yields
        ------
        EventExpression
            SquidASM event to wait for.
        """
        # Send the message
        yield from self._raw_socket.send(envelope.to_json())
        self.mark_sent(envelope)
        
        # Wait for ACK
        start_time = ns.sim_time()
        while True:
            # Check timeout
            if ns.sim_time() - start_time > timeout_ns:
                self.mark_timeout()  # Raises AckTimeoutError
            
            response = yield from self._raw_socket.recv()
            ack_envelope = MessageEnvelope.from_json(response)
            result = self.process_received(ack_envelope)
            
            if result is None:  # Was an ACK, we're done
                return
```

---

### 5.5 `ehok/protocols/leakage_manager.py` – Leakage Tracking

**Status:** ✅ Complete, Needs Wiring

**Integration Point:**
`LeakageSafetyManager` is correctly implemented but not called from reconciliation loop.

**Wiring Location:** `alice.py::_phase4_reconciliation()` and `bob.py::_phase4_reconciliation()`

```python
# In AliceBaselineEHOK._phase4_reconciliation():
# After receiving syndrome from Bob, account leakage:
if self._leakage_manager:
    report = BlockReconciliationReport(
        block_index=0,
        syndrome_bits=len(syndrome) * 8,
        hash_bits=self.reconciliator.hash_verifier.hash_length_bits,
        decode_converged=converged,
        hash_verified=verified,
        iterations=iterations
    )
    self._leakage_manager.account_block(report)
    if self._leakage_manager.is_cap_exceeded:
        raise LeakageCapExceededError(ABORT_CODE_LEAKAGE_CAP_EXCEEDED)
```

---

### 5.6 `ehok/protocols/statistical_validation.py` – Phase II Validation

**Status:** ✅ Complete

**Components:**
- `DetectionValidator` (Hoeffding bounds) ✅
- `FiniteSizePenaltyCalculator` ✅
- `QBERAdjuster` ✅

**Minor API Note (from `remediation_specification.md` Section 4.2):**
Test expects `DetectionValidator(expected_detection_prob=...)` but constructor signature should be verified:

```python
# Line ~200 onwards shows:
class DetectionValidator:
    # Check constructor signature matches test expectation
```

---

## 6. Implementation Layer Analysis

### 6.1 `ehok/implementations/factories.py` – Strategy Factory

**Status:** ✅ Single Responsibility

**Current Design:**
```python
def build_reconciliator(config, parity_check_matrix=None) -> IReconciliator:
    # Returns LDPCReconciliator
    
def build_privacy_amplifier(config) -> IPrivacyAmplifier:
    # Returns ToeplitzAmplifier
```

**Integration Point:**
Factories should remain lean. New components (PhysicalModelAdapter, TimingEnforcer) should be constructed at protocol level, not via factory.

---

### 6.2 `ehok/implementations/reconciliation/` – LDPC Pipeline

**Status:** ✅ Modular, Clean

**Components:**
- `LDPCMatrixManager`: Pool management
- `LDPCBeliefPropagation`: Decoder
- `LDPCReconciliator`: Orchestration

**Leakage Hook Point:**
`LDPCReconciliator.reconcile_block()` returns `(corrected, converged, error_count)`.

The caller (`AliceBaselineEHOK`) should create `BlockReconciliationReport` and feed to `LeakageSafetyManager`.

---

### 6.3 `ehok/implementations/privacy_amplification/` – Key Compression

**Status:** ✅ Dual-Mode Design

**Components:**
- `ToeplitzAmplifier`: Core compression
- `finite_key.py`: QKD-style bounds (legacy?)
- `nsm_privacy_amplifier.py`: NSM-correct bounds ✅

**Potential Redundancy:**
`finite_key.py` appears to implement **QKD finite-key bounds**, while `nsm_privacy_amplifier.py` implements **NSM bounds**. 

If QKD mode is not intended, `finite_key.py` may be legacy code. Verify usage before deprecation. Proceed with removal.

---

### 6.4 `ehok/implementations/commitment/` – Commitment Schemes

**Status:** ✅ Interface-Compliant

**Components:**
- `SHA256Commitment`: Production implementation
- `MerkleCommitment`: Advanced (subset proofs)

Both implement `ICommitmentScheme`. No changes required.

---

## 7. Utilities & Interfaces Analysis

### 7.1 `ehok/utils/logging.py` – Logging Infrastructure

**Status:** ✅ Clean

**Features:**
- `get_logger()`: Single entry point
- `setup_script_logging()`: Dual file/terminal output
- SquidASM `LogManager` fallback for non-simulation environments

No changes required.

---

### 7.2 `ehok/utils/classical_sockets.py` – Socket Helpers

**Status:** 🔍 Verify Usage

Check if any utility functions here should be integrated with `OrderedProtocolSocket`.

---

### 7.3 `ehok/interfaces/` – Abstract Contracts

**Status:** ✅ Complete Set

**Interfaces Defined:**
- `ICommitmentScheme` ✅
- `IReconciliator` ✅
- `IPrivacyAmplifier` ✅
- `ISamplingStrategy` ✅
- `INoiseEstimator` ✅

**Missing Interface:**
Add `IOrderedSocket` interface for `OrderedProtocolSocket` abstraction.

---

## 8. Cleanup Manifest

### 8.1 Configuration Consolidation

**Action:** Deprecate `ehok/configs/protocol_config.py` in favor of `ehok/core/config.py`

**Reason:** 
- `core/config.py` is used by protocols
- `configs/protocol_config.py` is used by `noise_adapter.py`
- Merge `PhysicalParameters` into `core/config.py`

**Migration Steps:**
1. Add `PhysicalConfig` to `core/config.py` (copy from `protocol_config.py`)
2. Add `@deprecated` decorator to `configs/protocol_config.py::PhysicalParameters`
3. Update `noise_adapter.py` imports
4. Update tests
5. Remove `configs/protocol_config.py` after deprecation period

---

### 8.2 Potential Dead Code Candidates

| File | Candidate | Reason | Action |
|------|-----------|--------|--------|
| `implementations/privacy_amplification/finite_key.py` | Entire module? | QKD bounds, NSM not QKD | Verify usage, remove if unused |
| `core/constants.py` | Duplicate constants | Some values duplicated in config defaults | Audit and consolidate |

---

### 8.3 Type Safety Improvements

**Files with `Any` Usage:**
- `protocols/base.py`: `Dict[str, Any]` return type
- `protocols/alice.py`: Multiple `Any` in type hints

**Recommendation:** Replace with typed `ProtocolResult` dataclass.

---

## 9. Master Refactoring Plan

### Phase 0: Preparation (Non-Breaking)

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 0.1 | Add `ProtocolViolation` exception | `ordered_messaging.py` | None |
| 0.2 | Add `batch_size` to `FeasibilityInputs` | `feasibility.py` | None |
| 0.3 | Create `IOrderedSocket` interface | `interfaces/ordered_socket.py` | None |

### Phase 1: Physical Model Adapter (Critical Gap)

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 1.1 | Implement `PhysicalModelAdapter` class | `quantum/noise_adapter.py` | 0.* |
| 1.2 | Implement `estimate_storage_noise_from_netsquid()` | `quantum/noise_adapter.py` | 1.1 |
| 1.3 | Implement `to_squidasm_link_config()` | `quantum/noise_adapter.py` | 1.1 |
| 1.4 | Add unit tests | `tests/test_sprint1_config_adapter.py` | 1.3 |

### Phase 2: Ordered Messaging Integration

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 2.1 | Implement `SquidASMOrderedSocket` subclass | `ordered_messaging.py` | 0.1 |
| 2.2 | Implement `send_with_ack()` generator | `ordered_messaging.py` | 2.1 |
| 2.3 | Implement `recv_and_ack()` generator | `ordered_messaging.py` | 2.1 |
| 2.4 | Add generator integration tests | `tests/test_sprint2_ordered_messaging.py` | 2.3 |

### Phase 3: Protocol Rewiring

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 3.1 | Add `_ordered_socket` to `EHOKRole.__init__` | `base.py` | 2.* |
| 3.2 | Add `_timing_enforcer` to `EHOKRole.__init__` | `base.py` | None |
| 3.3 | Refactor `AliceBaselineEHOK._phase2_*` to use ordered socket | `alice.py` | 3.1 |
| 3.4 | Refactor `AliceBaselineEHOK._phase3_*` with timing check | `alice.py` | 3.2 |
| 3.5 | Refactor `BobBaselineEHOK._phase2_*` with timing mark | `bob.py` | 3.2 |
| 3.6 | Refactor `BobBaselineEHOK._phase3_*` to use ordered socket | `bob.py` | 3.1 |
| 3.7 | Wire `LeakageSafetyManager` into Phase 4 | `alice.py`, `bob.py` | None |

### Phase 4: Configuration Consolidation

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 4.1 | Merge `PhysicalParameters` into `core/config.py` | `core/config.py` | None |
| 4.2 | Add `@deprecated` decorator | `configs/protocol_config.py` | 4.1 |
| 4.3 | Update all imports | Multiple | 4.2 |
| 4.4 | Remove deprecated module | `configs/protocol_config.py` | After deprecation period |

### Phase 5: Cleanup & Verification

| Step | Task | Files | Dependency |
|------|------|-------|------------|
| 5.1 | Audit `finite_key.py` usage | - | None |
| 5.2 | Remove dead code (if confirmed) | TBD | 5.1 |
| 5.3 | Full system test suite | `tests/system/` | All |
| 5.4 | Update documentation | `docs/` | All |

---

## Appendix A: Detailed Code Injection Points

### A.1 `ehok/quantum/noise_adapter.py` – After Line 340

```python
# =============================================================================
# Physical Model Adapter (TASK-NOISE-ADAPTER-002)
# =============================================================================

@dataclass(frozen=True)
class AdapterOutput:
    """Output from PhysicalModelAdapter translation."""
    
    link_fidelity: float
    prob_success: float
    t_cycle_ns: float
    storage_noise_r: float
    expected_qber: float


class PhysicalModelAdapter:
    """
    Adapter bridging NSM physical parameters to SquidASM simulation config.
    
    This class implements the critical translation layer between:
    1. Physical device parameters (μ, η, e_det) → SquidASM link config
    2. NetSquid memory parameters (T1, T2) → NSM storage noise r
    
    References
    ----------
    - remediation_specification.md Section 2.3
    - Erven et al. (2014) Table I
    """
    
    def __init__(self, physical_params: PhysicalParameters):
        self._params = physical_params
        self._sim_params = physical_to_simulator(physical_params)
    
    def to_squidasm_link_config(self) -> "DepolariseLinkConfig":
        """
        Create SquidASM link configuration from physical parameters.
        
        Returns
        -------
        DepolariseLinkConfig
            Configuration for squidasm.run.stack.config
        """
        from squidasm.run.stack.config import DepolariseLinkConfig
        
        return DepolariseLinkConfig(
            fidelity=self._sim_params.link_fidelity,
            prob_success=self._sim_params.expected_detection_prob,
            t_cycle=1e6,  # 1ms default cycle time
        )
    
    @staticmethod
    def estimate_storage_noise_r(
        T1_ns: float,
        T2_ns: float,
        delta_t_ns: float
    ) -> float:
        """
        Estimate NSM storage noise parameter r from T1/T2 decoherence.
        
        For depolarizing noise, the retention probability after time t is:
            r(t) = (1 + 3·e^{-t/T1}·e^{-t/T2}) / 4
        
        Simplified for T1 >> T2 (common regime):
            r(Δt) ≈ e^{-Δt/T2}
        """
        if T2_ns <= 0:
            return 0.0  # Complete decoherence
        
        import math
        r = math.exp(-delta_t_ns / T2_ns)
        return max(0.0, min(1.0, r))


def estimate_storage_noise_from_netsquid(
    T1_ns: float,
    T2_ns: float,
    delta_t_ns: float
) -> float:
    """
    Standalone function for T1/T2 → r conversion.
    
    See PhysicalModelAdapter.estimate_storage_noise_r for details.
    """
    return PhysicalModelAdapter.estimate_storage_noise_r(T1_ns, T2_ns, delta_t_ns)
```

### A.2 `ehok/protocols/base.py` – Constructor Modification

```python
# Replace __init__ (lines 28-42) with:

def __init__(
    self,
    config: ProtocolConfig | None = None,
    total_pairs: int | None = None,
    timing_enforcer: TimingEnforcer | None = None,  # NEW
    leakage_manager: LeakageSafetyManager | None = None,  # NEW
    **_: Any
):
    self.config = config or ProtocolConfig.baseline()
    if total_pairs is not None:
        self.config.quantum.total_pairs = total_pairs
    
    self.sifting_manager = SiftingManager()
    self._timing_enforcer = timing_enforcer  # NEW
    self._leakage_manager = leakage_manager  # NEW
    self._ordered_socket: OrderedProtocolSocket | None = None  # NEW
    
    # Strategy placeholders; built lazily in run()
    self.commitment_scheme = None
    self.reconciliator = None
    self.privacy_amplifier = None
    self.sampling_strategy = None
    self.noise_estimator = None
```

---

## Appendix B: Test Coverage Requirements

Each refactoring step must maintain or improve test coverage:

| Component | Current Tests | Required New Tests |
|-----------|---------------|-------------------|
| `PhysicalModelAdapter` | None | `test_physical_model_adapter.py` |
| `OrderedProtocolSocket` generator methods | Partial | `test_ordered_socket_generators.py` |
| Timing integration | `test_sprint1_timing_feasibility.py` | `test_timing_protocol_integration.py` |
| Leakage wiring | `test_sprint2_leakage_manager.py` | `test_leakage_protocol_integration.py` |

---

## Appendix C: Deprecation Template

```python
import warnings
from functools import wraps

def deprecated(message: str):
    """Decorator to mark functions/classes as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage:
@deprecated("Use ehok.core.config.PhysicalConfig instead")
class PhysicalParameters:
    ...
```

---

**Document End**

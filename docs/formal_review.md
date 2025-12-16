# Formal Project Review: `ehok` Package

**Auditor:** External Code Auditor (Black/White Box Analysis)  
**Date:** December 16, 2025  
**Package:** `ehok` (E-HOK: Entangled Hybrid Oblivious Key Distribution)  
**Version:** 0.1.0

---

## Executive Summary

The `ehok` package implements the **E-HOK (Entangled Hybrid Oblivious Key Distribution)** protocol—a quantum cryptographic primitive for generating oblivious keys suitable for secure multiparty computation. The implementation is designed to run on the **SquidASM** quantum network simulator and targets the **Noisy Storage Model (NSM)** security framework.

The codebase demonstrates mature software engineering practices: clean separation of concerns, extensive use of design patterns, comprehensive data validation, and rigorous security-focused architecture. However, the complexity inherent in mapping a theoretical cryptographic protocol to simulation code has introduced some architectural debt that warrants attention.

**Key Findings:**
- ✅ Well-structured package with clear domain boundaries
- ✅ Comprehensive use of Strategy, Factory, and Template Method patterns
- ✅ Rigorous dataclass contracts with runtime validation
- ⚠️ Deprecated module (`configs/protocol_config.py`) still present
- ⚠️ Some tight coupling between protocol roles and infrastructure
- ⚠️ Incomplete integration of some Sprint 2 components (ordered messaging not fully wired)

---

## 1. High-Level Structural Map

### 1.1 Package Hierarchy

```
ehok/
├── __init__.py              # Public API exports
├── core/                    # Domain primitives & configuration
├── interfaces/              # Abstract base classes (Strategy interfaces)
├── implementations/         # Concrete algorithm implementations
├── protocols/               # SquidASM Program roles (Alice/Bob)
├── quantum/                 # Quantum operations & batching
├── analysis/                # NSM bounds & metrics calculation
├── configs/                 # Network YAML + LDPC matrices
├── utils/                   # Logging infrastructure
├── examples/                # Runnable demonstration scripts
└── tests/                   # Comprehensive test suite
```

### 1.2 Module Responsibilities

| Sub-package | Responsibility |
|-------------|----------------|
| **`core/`** | Foundational domain model: data structures, exceptions, constants, configuration schemas, sifting logic, timing enforcement, feasibility checking, and OT output formatting. |
| **`interfaces/`** | Abstract base classes defining pluggable strategy contracts: `ICommitmentScheme`, `IReconciliator`, `IPrivacyAmplifier`, `ISamplingStrategy`, `INoiseEstimator`. |
| **`implementations/`** | Concrete implementations of interface contracts organized by domain (commitment, reconciliation, privacy amplification, sampling, noise estimation). |
| **`protocols/`** | SquidASM `Program` subclasses implementing Alice and Bob roles, plus protocol infrastructure (ordered messaging, leakage management, statistical validation). |
| **`quantum/`** | Quantum-layer operations: EPR generation, batching, basis selection, measurement buffering, noise adaptation. |
| **`analysis/`** | Pure-Python NSM security bound calculations, channel capacity functions, and derived metrics. |
| **`configs/`** | Static configuration assets: network YAML files, LDPC degree distributions, pre-generated parity-check matrices. |
| **`utils/`** | Cross-cutting concerns: logging infrastructure with SquidASM integration. |
| **`examples/`** | Entry-point scripts demonstrating protocol execution. |
| **`tests/`** | Unit, integration, and system tests organized by sprint. |

---

## 2. Architectural Analysis

### 2.1 Design Patterns Identified

#### 2.1.1 Strategy Pattern
**Evidence:** The `interfaces/` package defines five abstract strategy interfaces:

```python
# interfaces/commitment.py
class ICommitmentScheme(ABC):
    @abstractmethod
    def commit(self, data: np.ndarray) -> Tuple[bytes, Any]: ...
    @abstractmethod
    def verify(self, commitment: bytes, data: np.ndarray, decommitment_info: Any) -> bool: ...
```

Concrete implementations:
- `SHA256Commitment` (simple hash-based)
- `MerkleCommitment` (tree-based subset opening)

**Impact:** Enables hot-swapping of cryptographic primitives without modifying protocol logic.

#### 2.1.2 Factory Pattern
**Evidence:** `implementations/factories.py` centralizes object construction:

```python
def build_commitment_scheme(config: ProtocolConfig) -> ICommitmentScheme:
    return SHA256Commitment()

def build_reconciliator(config: ProtocolConfig, ...) -> IReconciliator:
    manager = LDPCMatrixManager.from_directory(...)
    decoder = LDPCBeliefPropagation(...)
    return LDPCReconciliator(manager, bp_decoder=decoder)
```

**Impact:** Decouples construction logic from usage; supports configuration-driven assembly.

#### 2.1.3 Template Method Pattern
**Evidence:** `protocols/base.py` defines `EHOKRole`, an abstract base implementing the protocol skeleton:

```python
class EHOKRole(Program, ABC):
    def run(self, context) -> Generator[...]:
        # Template method: fixed structure
        quantum_result = yield from self._phase1_quantum(quantum_runner)
        result = yield from self._execute_remaining_phases(quantum_result)
        return result

    @abstractmethod
    def _execute_remaining_phases(self, quantum_result) -> Generator[...]:
        # Hook for subclasses
        raise NotImplementedError
```

**Impact:** Enforces consistent protocol structure while allowing role-specific customization.

#### 2.1.4 Dataclass Contracts (Design-by-Contract)
**Evidence:** Phase boundary contracts in `core/data_structures.py`:

| Contract | Phase Boundary |
|----------|----------------|
| `QuantumPhaseOutput` | Phase I → Phase II |
| `SiftedKeyData` | Phase II → Phase III |
| `ReconciledKeyData` | Phase III → Phase IV |
| `ObliviousTransferOutput` | Phase IV output |

Each dataclass includes `__post_init__` validation:

```python
@dataclass
class QuantumPhaseOutput:
    def __post_init__(self) -> None:
        # POST-PHI-001: Array length consistency
        for name, arr in arrays:
            if len(arr) != self.n_pairs:
                raise ValueError(...)
```

**Impact:** Runtime enforcement of protocol invariants; catches integration bugs early.

#### 2.1.5 State Machine Pattern
**Evidence:** `protocols/ordered_messaging.py` implements socket state tracking:

```python
class SocketState(Enum):
    IDLE = auto()
    SENT_WAIT_ACK = auto()
    RECV_PROCESSING = auto()
    VIOLATION = auto()  # Terminal state
```

**Impact:** Enforces commit-then-reveal semantics critical for NSM security.

---

### 2.2 Execution Workflow

#### 2.2.1 Entry Points

**Primary Entry Point:** `ehok/examples/run_baseline.py`
```python
def run_ehok_baseline(num_pairs, network_config_path, logger):
    alice_program = AliceEHOKProgram(total_pairs=num_pairs)
    bob_program = BobEHOKProgram(total_pairs=num_pairs)
    alice_results, bob_results = run(
        config=network_cfg,
        programs={"alice": alice_program, "bob": bob_program},
        num_times=1
    )
```

The protocol is invoked via SquidASM's `run()` function, which orchestrates the discrete-event simulation.

#### 2.2.2 Protocol Phase Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         E-HOK PROTOCOL PHASES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase I: Quantum Generation                                            │
│  ├── EPR pair creation via SquidASM EPRSocket                          │
│  ├── Random basis selection (Z/X) per qubit                            │
│  ├── Measurement and result buffering                                  │
│  └── Output: QuantumPhaseOutput                                        │
│                                                                         │
│  Phase II: Commitment & Sifting                                         │
│  ├── Bob commits to outcomes/bases (SHA256)                            │
│  ├── Alice reveals her bases (after Δt timing barrier)                 │
│  ├── Basis matching → sifted key indices                               │
│  ├── Test set selection for QBER estimation                            │
│  ├── Commitment verification                                           │
│  └── Output: SiftedKeyData                                             │
│                                                                         │
│  Phase III: Information Reconciliation                                  │
│  ├── LDPC matrix pool synchronization (checksum exchange)              │
│  ├── Rate selection based on QBER                                      │
│  ├── Syndrome computation and transmission (Bob → Alice)               │
│  ├── Belief propagation decoding (Alice)                               │
│  ├── Hash verification                                                 │
│  ├── Leakage accounting (wiretap cost)                                 │
│  └── Output: ReconciledKeyData                                         │
│                                                                         │
│  Phase IV: Privacy Amplification                                        │
│  ├── NSM max-bound entropy calculation                                 │
│  ├── Secure key length determination                                   │
│  ├── Toeplitz seed generation and sharing                              │
│  ├── Key compression via Toeplitz hashing                              │
│  └── Output: ObliviousTransferOutput                                   │
│                                                                         │
│  Final Output:                                                          │
│  ├── Alice: (S_0, S_1) — both candidate keys                           │
│  └── Bob: S_C — chosen key (C ∈ {0,1})                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 Control Flow Tracing

1. **Initialization:** `EHOKRole.run()` builds strategies via factories
2. **Quantum Phase:** `QuantumPhaseRunner.run()` yields EPR batches
3. **Classical Phases:** Role-specific `_execute_remaining_phases()` orchestrates sifting, reconciliation, and amplification
4. **Result Aggregation:** `_result_success()` / `_result_abort()` constructs final output dictionary

---

### 2.3 Data Flow

```
Configuration (ProtocolConfig)
         │
         ▼
┌────────────────────┐
│  Phase I: Quantum  │
│  - EPR generation  │
│  - Measurement     │
└────────┬───────────┘
         │ QuantumPhaseOutput
         ▼
┌────────────────────┐
│  Phase II: Sifting │
│  - Commitment      │
│  - Basis matching  │
│  - QBER estimation │
└────────┬───────────┘
         │ SiftedKeyData
         ▼
┌────────────────────────┐
│  Phase III: Reconcile  │
│  - LDPC decoding       │
│  - Leakage tracking    │
└────────┬───────────────┘
         │ ReconciledKeyData
         ▼
┌────────────────────────────┐
│  Phase IV: Privacy Amp.    │
│  - NSM entropy bounds      │
│  - Toeplitz compression    │
└────────┬───────────────────┘
         │ ObliviousTransferOutput
         ▼
┌─────────────────────────┐
│  Final: ObliviousKey    │
│  - Alice: (S_0, S_1)    │
│  - Bob: S_C, choice bit │
└─────────────────────────┘
```

---

## 3. Entry Points & Public API

### 3.1 Public API (from `ehok/__init__.py`)

```python
# Data Structures
from ehok import ObliviousKey, MeasurementRecord, ProtocolResult, ExecutionMetrics
from ehok import ProtocolConfig

# Exceptions
from ehok import (
    EHOKException, SecurityException, ProtocolError,
    QBERTooHighError, ReconciliationFailedError, CommitmentVerificationError
)

# Constants
from ehok import QBER_THRESHOLD, TARGET_EPSILON_SEC, TEST_SET_FRACTION, TOTAL_EPR_PAIRS, BATCH_SIZE
```

### 3.2 Intended Usage Pattern

```python
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

from ehok.protocols.alice import AliceEHOKProgram
from ehok.protocols.bob import BobEHOKProgram
from ehok.core.config import ProtocolConfig

# 1. Load network configuration
network_cfg = StackNetworkConfig.from_file("network_baseline.yaml")

# 2. Create protocol programs
config = ProtocolConfig.baseline()
alice = AliceEHOKProgram(config=config)
bob = BobEHOKProgram(config=config)

# 3. Execute simulation
alice_results, bob_results = run(
    config=network_cfg,
    programs={"alice": alice, "bob": bob},
    num_times=1
)

# 4. Access results
alice_key = alice_results[0]["oblivious_key"]  # ObliviousKey
bob_key = bob_results[0]["oblivious_key"]
```

### 3.3 Output Structure

The protocol produces dictionaries with the following schema:

**Success Case:**
```python
{
    "success": True,
    "oblivious_key": ObliviousKey(...),  # Final key material
    "qber": 0.023,                        # Measured QBER
    "raw_count": 10000,                   # EPR pairs generated
    "sifted_count": 4891,                 # After basis matching
    "test_count": 489,                    # Used for QBER estimation
    "final_count": 1024,                  # Privacy-amplified key length
    "role": "alice",
    "measurement_records": [...]
}
```

**Abort Case:**
```python
{
    "success": False,
    "abort_reason": "QBER_TOO_HIGH",
    "oblivious_key": None,
    "qber": 0.15,
    ...
}
```

---

## 4. Critical Review & Recommendations

### 4.1 Code Smells

#### 4.1.1 Deprecated Module Still Present

**Location:** `configs/protocol_config.py`

**Issue:** The module header declares deprecation:
```python
warnings.warn(
    "ehok.configs.protocol_config is deprecated. "
    "Use ehok.core.config instead for PhysicalParameters and configuration classes.",
    DeprecationWarning,
    stacklevel=2
)
```

Yet the module remains in the codebase with ~395 lines of code.

**Recommendation:** Remove `configs/protocol_config.py` and update any remaining imports to use `core/config.py`. This eliminates confusion and prevents accidental usage.

---

#### 4.1.2 Incomplete Ordered Messaging Integration

**Location:** `protocols/alice.py`, `protocols/bob.py`

**Issue:** The `OrderedProtocolSocket` infrastructure exists in `protocols/ordered_messaging.py` (1022 lines) with sophisticated state machine and ACK semantics, but it is only partially integrated into the protocol roles:

```python
# In EHOKRole.__init__
self._ordered_socket = ordered_socket

# But actual phase communication uses raw classical sockets:
yield from self.context.csockets[self.PEER_NAME].recv()
```

The ordered messaging infrastructure enforces commit-then-reveal semantics, which is security-critical per NSM requirements, but the current implementation bypasses it.

**Recommendation:** Either:
1. Complete the integration of `OrderedProtocolSocket` for all inter-party communication, or
2. Document why the raw socket approach is acceptable for the current security model

---

#### 4.1.3 Tight Coupling in Protocol Configuration

**Location:** `core/config.py`

**Issue:** `ProtocolConfig` has grown to encompass many sub-configs:
```python
@dataclass
class ProtocolConfig:
    quantum: QuantumConfig
    security: SecurityConfig
    reconciliation: ReconciliationConfig
    nsm: NSMConfig
    privacy_amplification: PrivacyAmplificationConfig
    sampling_seed: Optional[int]
```

Some of these are accessed through nested attribute chains in protocol code:
```python
self.config.privacy_amplification.target_epsilon_sec
self.config.nsm.storage_noise_r
```

**Recommendation:** Consider introducing a configuration facade or accessor methods to flatten common access patterns and reduce coupling.

---

#### 4.1.4 Mixed Concerns in Data Structures

**Location:** `core/data_structures.py` (1077 lines)

**Issue:** This module contains both:
- Pure data contracts (`QuantumPhaseOutput`, `SiftedKeyData`)
- Behavioral classes (`LDPCBlockResult`, `LDPCReconciliationResult`)
- Output formatters (`ObliviousTransferOutput`)

The file's length makes navigation difficult.

**Recommendation:** Consider splitting into:
- `core/contracts/phase_contracts.py` — Phase boundary dataclasses
- `core/contracts/ldpc_contracts.py` — LDPC-specific result structures
- `core/contracts/output_contracts.py` — Final output structures

---

#### 4.1.5 Hardcoded Default in Privacy Amplification

**Location:** `protocols/alice.py`, lines 285-288; `protocols/bob.py`, lines 225-228

**Issue:**
```python
storage_noise_r = getattr(self.config, 'nsm', None)
if storage_noise_r is not None:
    storage_noise_r = self.config.nsm.storage_noise_r
else:
    storage_noise_r = 0.75  # Default
```

The magic number `0.75` is duplicated and should come from constants.

**Recommendation:** Define `DEFAULT_STORAGE_NOISE_R` in `constants.py` and reference it.

---

### 4.2 Redundancy Analysis

#### 4.2.1 Dual Configuration Modules

**Files:**
- `configs/protocol_config.py` (deprecated, 395 lines)
- `core/config.py` (canonical, 301 lines)

**Issue:** Both define configuration schemas with significant overlap.

**Recommendation:** Delete `configs/protocol_config.py` entirely after verifying no external dependencies.

---

#### 4.2.2 Duplicate Entropy Functions

**Locations:**
- `analysis/nsm_bounds.py` → `_binary_entropy()`, `binary_entropy()`
- `implementations/reconciliation/ldpc_reconciliator.py` → `_binary_entropy()`
- `implementations/privacy_amplification/finite_key.py` → `binary_entropy()`

**Issue:** The binary entropy function is implemented at least three times with identical logic.

**Recommendation:** Consolidate into a single utility function in `core/` or `analysis/` and import elsewhere.

---

#### 4.2.3 LDPC Constants Scattered

**Locations:**
- `core/constants.py` — `LDPC_FRAME_SIZE`, `LDPC_CODE_RATES`, etc.
- `implementations/reconciliation/ldpc_*.py` — Local constant references

**Issue:** Some LDPC parameters are defined centrally, but understanding the full LDPC configuration requires reading multiple files.

**Recommendation:** Create a dedicated `core/ldpc_constants.py` or a `LDPC_CONFIG` namespace in `constants.py` for discoverability.

---

### 4.3 Improvement Proposals

#### 4.3.1 Add Protocol Result Typing

**Current State:**
```python
def _execute_remaining_phases(...) -> Generator[EventExpression, None, Dict[str, Any]]:
```

**Proposal:** Define a `ProtocolResultDict` TypedDict:
```python
from typing import TypedDict

class ProtocolResultDict(TypedDict):
    success: bool
    oblivious_key: Optional[ObliviousKey]
    qber: float
    raw_count: int
    sifted_count: int
    test_count: int
    final_count: int
    role: str
    measurement_records: List[MeasurementRecord]
```

---

#### 4.3.2 Extract Protocol Orchestration

**Current State:** `AliceBaselineEHOK` and `BobBaselineEHOK` mix:
- SquidASM integration (generator protocol)
- Phase orchestration
- Strategy invocation
- Result construction

**Proposal:** Introduce a `ProtocolOrchestrator` that handles phase sequencing, with roles delegating to it:
```python
class ProtocolOrchestrator:
    def execute_phases(self, quantum_result, strategies, context) -> ProtocolResult:
        sifted = self._phase2_sift(...)
        reconciled = self._phase3_reconcile(...)
        amplified = self._phase4_amplify(...)
        return self._construct_result(...)
```

---

#### 4.3.3 Add Integration Test for Ordered Messaging

**Gap Identified:** While `test_sprint2_ordered_messaging.py` exists, there's no integration test demonstrating the full commit-then-reveal flow with timing barrier enforcement.

**Proposal:** Add `test_ordered_messaging_integration.py` that:
1. Creates a simulated timing enforcer
2. Verifies that basis reveal is blocked before Δt
3. Confirms proper ACK sequencing

---

#### 4.3.4 Documentation of Abort Code Taxonomy

**Current State:** Abort codes are scattered across modules:
- `ABORT-I-FEAS-*` in `core/feasibility.py`
- `ABORT-II-*` in `protocols/statistical_validation.py`
- `ABORT-III-*` in `protocols/leakage_manager.py`

**Proposal:** Create `core/abort_codes.py` consolidating all abort codes with docstrings explaining the security implications of each.

---

### 4.4 Security Observations

#### 4.4.1 Timing Barrier Opt-In Design

**Observation:** The NSM timing barrier (`TimingEnforcer`) is **not** created by default:
```python
# In EHOKRole._setup_injected_dependencies()
# TimingEnforcer - NOT created by default (opt-in for security)
# Inject via constructor to enable NSM timing barrier enforcement
```

**Assessment:** This is intentional to allow unit tests to run without timing constraints. However, production deployments **must** inject a `TimingEnforcer` for security compliance.

**Recommendation:** Add a configuration flag `enforce_timing_barrier: bool = False` with a prominent warning in documentation that production deployments should set this to `True`.

---

#### 4.4.2 Commitment Scheme Weakness

**Observation:** `SHA256Commitment` does not support efficient subset opening:
```python
def open_subset(self, indices, data, decommitment_info):
    # SHA256: Opens entire data (subset opening not optimized)
```

**Assessment:** For the current protocol, this is acceptable. However, if the test set fraction increases significantly, a Merkle-based commitment would provide logarithmic proof sizes.

**Recommendation:** The existing `merkle_commitment.py` should be validated and made the default for production scenarios.

---

## 5. Test Coverage Assessment

### 5.1 Test Organization

```
tests/
├── test_commitment.py          # Unit: commitment schemes
├── test_contracts.py           # Unit: dataclass validation
├── test_foundation.py          # Unit: core utilities
├── test_integration.py         # Integration: multi-phase flows
├── test_ldpc_*.py              # Unit/Integration: LDPC components
├── test_privacy_amplification.py
├── test_quantum.py             # Unit: quantum operations
├── test_reconciliation_integration.py
├── test_sifting.py
├── test_sprint*_*.py           # Sprint-organized tests
└── test_system.py              # End-to-end system tests
```

### 5.2 Coverage Gaps

Based on file naming and structure, the following areas may lack comprehensive coverage:

1. **Ordered Messaging Edge Cases:** ACK timeout, duplicate detection, out-of-order handling
2. **Feasibility Checker Boundaries:** Death Valley edge cases
3. **NSM Bounds Numerical Stability:** Edge cases where r → 0 or r → 1

---

## 6. Conclusion

The `ehok` package represents a sophisticated implementation of the E-HOK quantum cryptographic protocol. The architecture demonstrates strong software engineering discipline with clear pattern usage, comprehensive validation, and security-conscious design.

**Strengths:**
- Clean separation of concerns via Strategy pattern
- Rigorous dataclass contracts with runtime validation
- Comprehensive abort code taxonomy for security failures
- Well-organized test structure following sprint methodology

**Areas for Improvement:**
- Remove deprecated `configs/protocol_config.py`
- Complete `OrderedProtocolSocket` integration or document its partial status
- Consolidate duplicate utility functions (binary entropy)
- Add configuration flag for mandatory timing barrier in production

**Overall Assessment:** The codebase is in good health for a pre-1.0 implementation. Addressing the identified redundancies and completing the ordered messaging integration would significantly improve maintainability and security posture.

---

*End of Formal Review*

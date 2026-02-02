# Caligo: $\binom{2}{1}$-OT Protocol Implementation Architecture

**Author:** Lead Software Architect  
**Date:** December 16, 2025  
**Version:** 1.0  
**Objective:** Design a clean-slate, domain-driven $\binom{2}{1}$-OT protocol implementation with native SquidASM/NetSquid integration.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Lessons Learned from ehok](#2-lessons-learned-from-ehok)
3. [Domain Model Analysis](#3-domain-model-analysis)
4. [Simulation Layer Requirements](#4-simulation-layer-requirements)
5. [Package Architecture](#5-package-architecture)
6. [Module Specifications](#6-module-specifications)
7. [SquidASM/NetSquid Bridge Layer](#7-squidasmnetqasm-bridge-layer)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Project Genesis

**Caligo** (Latin: "fog/mist" — evoking the obscured nature of oblivious transfer) is a ground-up reimplementation of the $\binom{2}{1}$-OT (Efficient High-dimensional Oblivious Key) protocol. Rather than refactoring the existing `ehok` codebase, we choose a clean-slate approach for the following reasons:

| Rationale | Justification |
|-----------|---------------|
| **Technical debt elimination** | ehok suffers from architecture astronaut syndrome with 6+ abstraction layers |
| **Simulation-first design** | ehok treats SquidASM as an afterthought; caligo designs around it |
| **Domain alignment** | Package structure will mirror $\binom{2}{1}$-OT's 4-phase protocol from day one |
| **Integration by design** | NetSquid noise models, timing, and discrete-event semantics are first-class citizens |

### 1.2 Core Design Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CALIGO DESIGN PRINCIPLES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SIMULATION-NATIVE                                                       │
│     └── Every component understands NetSquid's discrete-event model         │
│     └── Timing is explicit, not an afterthought                             │
│     └── Noise models are parameterizable from protocol configuration        │
│                                                                             │
│  2. DOMAIN-DRIVEN STRUCTURE                                                 │
│     └── Package names reflect $\binom{2}{1}$-OT phases, not software patterns           │
│     └── A physicist should recognize the directory structure                │
│                                                                             │
│  3. LEAN MODULES (≤200 LOC)                                                 │
│     └── Single Responsibility enforced by size constraint                   │
│     └── No god-classes, no monolithic data_structures.py                    │
│                                                                             │
│  4. NO SPECULATIVE ABSTRACTION                                              │
│     └── Interfaces only when multiple implementations exist                 │
│     └── Direct imports over factory indirection                             │
│                                                                             │
│  5. EXPLICIT OVER IMPLICIT                                                  │
│     └── Configuration as code, not hidden YAML                              │
│     └── Clear data flow between phases                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Differentiators from ehok

| Aspect | ehok (Current) | caligo (Target) |
|--------|---------------|-----------------|
| **SquidASM integration** | Wrapper-based, brittle | Native, simulation-aware |
| **Timing enforcement** | Missing Δt primitives | Built-in `TimingBarrier` |
| **Noise models** | Hardcoded assumptions | Configurable via `PhysicalModel` |
| **Package structure** | `core/`, `implementations/` | `quantum/`, `sifting/`, `reconciliation/`, `amplification/` |
| **Module size** | Up to 1076 LOC | Max 200 LOC |
| **Interfaces** | 5+ unused abstractions | Only where needed |
| **LDPC matrices** | Bundled in configs/ | Generated asset, external storage |
| **Test coverage** | Partial, fragile | Phase-contract-driven |

### 1.4 Target Metrics

| Metric | Target |
|--------|--------|
| Maximum module LOC | 200 |
| Package count | 10 (domain-aligned) |
| Total estimated LOC | ~2,500 |
| SquidASM version compatibility | 0.12+ |
| NetSquid version compatibility | 1.1+ |
| Test coverage | ≥90% on core logic |

---

## 2. Lessons Learned from ehok

### 2.1 Anti-Patterns to Avoid

#### 2.1.1 Architecture Astronaut Syndrome

ehok's `interfaces/` package defines 5 abstract interfaces with single implementations:

```
interfaces/
├── commitment_interface.py      → Only SHA256Commitment exists
├── reconciliation_interface.py  → Only LDPCReconciliator exists  
├── privacy_amplification_interface.py → Only ToeplitzAmplifier exists
├── sampling_interface.py        → Only RandomSampling exists
└── noise_estimator_interface.py → UNUSED entirely
```

**Caligo approach:** No interface until second implementation is needed.

#### 2.1.2 Monolithic Data Structures

ehok's `core/data_structures.py` (1076 LOC) contains:
- 8 enums (protocol phases, abort reasons, warning codes)
- 12 dataclasses (phase outputs, LDPC results, OT outputs)
- Mixed concerns (protocol flow + LDPC-specific + output formatting)

**Caligo approach:** Domain-specific types live in their phase package.

#### 2.1.3 Missing Simulation Integration

ehok treats SquidASM as a black box:
- No access to `ns.sim_time()` for Δt enforcement
- No parameterization of NetSquid noise models
- No understanding of generator-based async model

**Caligo approach:** Dedicated `simulation/` package bridges the gap.

### 2.2 What to Preserve from ehok

| Component | Rationale | Integration Strategy |
|-----------|-----------|---------------------|
| SHA256 commitment logic | Cryptographically sound | Extract to `sifting/commitment.py` |
| LDPC BP decoder core | Well-tested algorithm | Extract to `reconciliation/decoder.py` |
| Toeplitz hashing | Standard PA approach | Extract to `amplification/toeplitz.py` |
| NSM entropy bounds math | Protocol-critical | Extract to `security/bounds.py` |
| Logging infrastructure | Working observability | Adapt to `utils/logging.py` |

---

## 3. Domain Model Analysis

### 3.1 $\binom{2}{1}$-OT Protocol Phases

The $\binom{2}{1}$-OT protocol implements **1-out-of-2 Oblivious Transfer** using the Noisy Storage Model (NSM). Security derives from the assumption that an adversary's quantum memory decoheres faster than the protocol's timing constraints allow exploitation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    $\binom{2}{1}$-OT PROTOCOL: TEMPORAL FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIME ═══════════════════════════════════════════════════════════════════►  │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────────┐   │
│  │   PHASE I    │   │   PHASE II   │   │ PHASE III │   │   PHASE IV     │   │
│  │   Quantum    │──►│   Sifting    │──►│   Recon-  │──►│   Privacy      │   │
│  │  Generation  │   │  & Estimate  │   │  ciliation│   │ Amplification  │   │
│  └──────┬───────┘   └──────┬───────┘   └─────┬─────┘   └───────┬────────┘   │
│         │                  │                 │                  │           │
│         ▼                  ▼                 ▼                  ▼           │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────────┐   │
│  │ EPR pairs    │   │ Sifted keys  │   │ Reconciled│   │ OT Output:     │   │
│  │ Measurements │   │ QBER estimate│   │ key blocks│   │ S₀, S₁, Sᴄ     │   │
│  │ Timing marks │   │ Test results │   │ Leakage Σ │   │                │   │
│  └──────────────┘   └──────────────┘   └───────────┘   └────────────────┘   │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  CRITICAL TIMING: Δt wait between Phase I completion and Phase II reveal    │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase I: Quantum Generation

**Domain Concepts:**
- EPR pair generation and distribution
- Basis selection (Z/X for BB84-style measurement)
- Measurement outcomes buffering
- Memory-constrained batching

**Simulation Requirements:**

| Requirement | SquidASM/NetSquid Component | Gap Status |
|-------------|----------------------------|------------|
| EPR generation | `EPRSocket.create_keep()` / `recv_keep()` | ✅ Native |
| Basis selection | `QubitMeasureBasis.X/Z` | ✅ Native |
| Noise modeling | `DepolarNoiseModel`, `T1T2NoiseModel` | ⚠️ Needs adapter |
| Source quality (μ) | Not exposed | ❌ Extension needed |
| Detection efficiency (η) | Not exposed | ❌ Extension needed |
| Timing marks | `ns.sim_time()` | ⚠️ Manual integration |

**Phase Contract (Output):**
```python
@dataclass
class QuantumPhaseResult:
    """Contract: Phase I → Phase II data transfer."""
    measurement_outcomes: list[int]      # 0/1 outcomes
    basis_choices: list[int]             # 0=Z, 1=X
    round_ids: list[int]                 # For missing round tracking
    generation_timestamp: float          # ns.sim_time() at completion
    num_pairs_requested: int
    num_pairs_generated: int             # May differ due to losses
```

### 3.3 Phase II: Sifting & Estimation

**Domain Concepts:**
- Commitment scheme (SHA256)
- Detection report exchange
- Δt timing barrier (NSM security)
- Basis reveal and sifting
- QBER estimation on test subset
- Finite-size penalty (μ) calculation

**Simulation Requirements:**

| Requirement | SquidASM/NetSquid Component | Gap Status |
|-------------|----------------------------|------------|
| Classical messaging | `ClassicalSocket.send()/recv()` | ✅ Native |
| Ordered acknowledgments | Not enforced | ❌ Extension needed |
| Δt timing enforcement | No primitive | ❌ Critical extension |
| Detection events | Implicit in EPR success/fail | ⚠️ Needs wrapper |

**Critical: The "Sandwich" Protocol**

```
Alice                                    Bob
  │                                        │
  │─────── EPR Pairs (Quantum) ──────────► │  ← Phase I
  │                                        │
  │◄────── Detection Report ────────────── │  
  │                                        │
  │         ╔═══════════════════╗          │
  │         ║   Δt WAIT TIME    ║          │  ← NSM Security
  │         ║  (Adversary's     ║          │
  │         ║   storage decays) ║          │
  │         ╚═══════════════════╝          │
  │                                        │
  │─────── Basis Reveal ──────────────────►│  ← Phase II continues
  │◄────── Bob's Bases ─────────────────── │
  │                                        │
  │         [Sifting & QBER Est.]          │
```

**Phase Contract (Output):**
```python
@dataclass
class SiftingPhaseResult:
    """Contract: Phase II → Phase III data transfer."""
    sifted_key_alice: bitarray            # Alice's sifted bits
    sifted_key_bob: bitarray              # Bob's sifted bits  
    matching_indices: list[int]           # Indices where bases matched
    qber_estimate: float                  # Estimated QBER
    qber_confidence: float                # Statistical confidence
    finite_size_penalty: float            # μ term
    test_set_indices: list[int]           # Sacrificed for estimation
    timing_compliant: bool                # Δt was enforced
```

### 3.4 Phase III: Information Reconciliation

**Domain Concepts:**
- LDPC code selection (rate based on QBER)
- Syndrome computation
- Belief Propagation (BP) decoding
- Hash verification
- Leakage tracking (|Σ| for entropy calculation)

**Simulation Requirements:**

| Requirement | SquidASM/NetSquid Component | Gap Status |
|-------------|----------------------------|------------|
| Classical messaging | `ClassicalSocket` | ✅ Native |
| LDPC matrices | External asset | ✅ Utility (not simulation) |
| BP decoding | Pure computation | ✅ No simulation needed |

**Phase Contract (Output):**
```python
@dataclass
class ReconciliationPhaseResult:
    """Contract: Phase III → Phase IV data transfer."""
    reconciled_key: bitarray              # Error-corrected key
    num_blocks: int                       # LDPC blocks processed
    blocks_succeeded: int                 # Blocks that passed verification
    total_syndrome_bits: int              # |Σ| leakage to adversary
    effective_rate: float                 # Achieved code rate
    hash_verified: bool                   # Final integrity check
```

### 3.5 Phase IV: Privacy Amplification

**Domain Concepts:**
- NSM entropy bound calculation (h_min)
- Secure key length determination
- Toeplitz matrix hashing
- OT output formatting (S₀, S₁, Sᴄ)

**Simulation Requirements:**

| Requirement | SquidASM/NetSquid Component | Gap Status |
|-------------|----------------------------|------------|
| Entropy calculation | Pure math | ✅ No simulation needed |
| Toeplitz hashing | Pure computation | ✅ No simulation needed |
| Random seed exchange | `ClassicalSocket` | ✅ Native |

**Phase Contract (Output):**
```python
@dataclass
class ObliviousTransferOutput:
    """Final protocol output: 1-out-of-2 OT keys."""
    s0: bitarray                          # Key for choice bit 0
    s1: bitarray                          # Key for choice bit 1
    sc: bitarray                          # Key Bob receives (based on choice)
    key_length: int                       # Secure output length
    security_parameter: float             # ε_sec achieved
    entropy_consumed: float               # h_min used
```

### 3.6 Cross-Cutting Concerns

| Concern | Responsibility | Package Location |
|---------|---------------|------------------|
| NSM security bounds | Entropy calculations, feasibility | `security/` |
| Timing enforcement | Δt barrier, simulation time | `simulation/` |
| Classical messaging | Ordered protocol sockets | `connection/` |
| Physical model | Noise adaptation, QBER bounds | `simulation/` |
| LDPC utilities | Matrix generation, management | `ldpc/` |
| Logging | Structured observability | `utils/` |

---

## 4. Simulation Layer Requirements

### 4.1 SquidASM Execution Model

Caligo must deeply understand SquidASM's generator-based execution:

```python
# SquidASM programs are generators that yield control
class AliceProgram(Program):
    def run(self, context: ProgramContext):
        # Yields pause execution until EPR is ready
        epr_socket = context.epr_sockets["bob"]
        qubit = yield from epr_socket.create_keep()[0]
        
        # Each yield allows NetSquid's discrete-event 
        # simulation to advance
        m = qubit.measure()
        yield from context.connection.flush()
```

**Key Implications for Caligo:**

1. **No blocking calls** - Everything must be generator-compatible
2. **Explicit time awareness** - Use `ns.sim_time()` for timestamps
3. **Batched operations** - Minimize yields for performance
4. **State machine design** - Protocol phases as explicit states

### 4.2 NetSquid Noise Model Integration

The critical gap in ehok is the lack of parameterizable noise models. Caligo introduces a `PhysicalModel` abstraction that bridges NSM parameters to NetSquid:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHYSICAL MODEL BRIDGE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NSM Parameters                    NetSquid Models                          │
│  ┌────────────────┐               ┌────────────────────────┐                │
│  │ Source Quality │               │ DepolarNoiseModel      │                │
│  │      (μ)       │──────────────►│  - depolar_rate        │                │
│  └────────────────┘               └────────────────────────┘                │
│                                                                             │
│  ┌────────────────┐               ┌────────────────────────┐                │
│  │ Detection Eff. │               │ FibreLossModel         │                │
│  │      (η)       │──────────────►│  - p_loss_length       │                │
│  └────────────────┘               │  - p_loss_init         │                │
│                                   └────────────────────────┘                │
│  ┌────────────────┐               ┌────────────────────────┐                │
│  │ Intrinsic Err. │               │ T1T2NoiseModel         │                │
│  │    (e_det)     │──────────────►│  - T1, T2 times        │                │
│  └────────────────┘               └────────────────────────┘                │
│                                                                             │
│  ┌────────────────┐               ┌────────────────────────┐                │
│  │ Storage Noise  │               │ DepolarNoiseModel      │                │
│  │      (r)       │──────────────►│  (on memory qubits)    │                │
│  └────────────────┘               └────────────────────────┘                │
│                                                                             │
│  ══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│  QBER_total = f(μ, η, e_det) must satisfy:                                  │
│    - QBER_total < 22% (hard limit - security impossible beyond)             │
│    - QBER_total < 11% (conservative - recommended operating point)          │
│                                                                             │
│  NSM Security requires: Trusted_noise < Untrusted_storage_noise             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Timing Enforcement Architecture

The Δt timing barrier is **security-critical**. Caligo implements this via a `TimingBarrier` that integrates with NetSquid's simulation clock:

```python
# Conceptual design - actual implementation in simulation/timing.py
class TimingBarrier:
    """
    Enforces NSM timing constraints.
    
    The barrier ensures that basis revelation occurs only after
    Δt has elapsed since quantum transmission completion.
    """
    
    def __init__(self, delta_t_ns: float):
        self.delta_t = delta_t_ns
        self._transmission_complete_time: Optional[float] = None
    
    def mark_transmission_complete(self) -> None:
        """Record when quantum phase completes."""
        self._transmission_complete_time = ns.sim_time()
    
    def can_reveal_bases(self) -> bool:
        """Check if Δt has elapsed."""
        if self._transmission_complete_time is None:
            return False
        elapsed = ns.sim_time() - self._transmission_complete_time
        return elapsed >= self.delta_t
    
    def wait_until_safe(self) -> Generator:
        """Yield until Δt has elapsed (simulation-compatible)."""
        while not self.can_reveal_bases():
            # This integrates with NetSquid's event scheduling
            yield from self._schedule_wait()
```

### 4.4 Required SquidASM Extensions

Based on the gap analysis, Caligo requires these extensions:

| Extension | Purpose | Priority |
|-----------|---------|----------|
| `PhysicalModelAdapter` | NSM params → NetSquid noise models | CRITICAL |
| `TimingBarrier` | Δt enforcement with sim_time() | CRITICAL |
| `DetectionEventTracker` | Track qubit losses for validation | HIGH |
| `OrderedProtocolSocket` | Enforce message ordering with ACKs | HIGH |
| `QBEREstimator` | Real-time QBER from measurement data | MEDIUM |

### 4.5 Simulation Configuration

Caligo externalizes all simulation parameters:

```python
@dataclass
class SimulationConfig:
    """Complete simulation environment specification."""
    
    # Network topology
    num_nodes: int = 2
    link_distance_km: float = 10.0
    
    # Physical channel
    source_quality_mu: float = 0.98          # μ: source fidelity
    detection_efficiency_eta: float = 0.85   # η: detector efficiency  
    intrinsic_error_e_det: float = 0.01      # e_det: detector error
    dark_count_rate_hz: float = 100.0        # Dark counts per second
    
    # Timing
    delta_t_ns: float = 1_000_000.0          # Δt: 1ms default
    state_delay_ns: float = 1000.0           # EPR delivery delay
    
    # Memory (for NSM)
    memory_t1_ns: float = 10_000_000.0       # T1: 10ms
    memory_t2_ns: float = 1_000_000.0        # T2: 1ms
    adversary_storage_noise_r: float = 0.1   # r: storage noise rate
    
    # Security
    epsilon_security: float = 1e-10          # ε_sec
    qber_hard_limit: float = 0.22            # Abort threshold
    qber_conservative_limit: float = 0.11    # Warning threshold
```

---

## 5. Package Architecture

### 5.1 Directory Structure

```
caligo/
├── __init__.py                          # Public API exports (< 50 LOC)
│
├── types/                               # Domain primitives & contracts
│   ├── __init__.py                      # Type re-exports
│   ├── keys.py                          # ObliviousKey, AliceKey, BobKey (< 100 LOC)
│   ├── measurements.py                  # MeasurementRecord, RoundResult (< 100 LOC)
│   ├── phase_contracts.py               # Phase I→IV boundary dataclasses (< 150 LOC)
│   └── exceptions.py                    # Exception hierarchy + enums (< 100 LOC)
│
├── simulation/                          # SquidASM/NetSquid bridge layer [NEW]
│   ├── __init__.py                      # Simulation API
│   ├── physical_model.py                # PhysicalModelAdapter: μ,η,e_det → NetSquid (< 180 LOC)
│   ├── timing.py                        # TimingBarrier, Δt enforcement (< 150 LOC)
│   ├── network_builder.py               # Build SquidASM network from config (< 180 LOC)
│   ├── detection.py                     # DetectionEventTracker (< 120 LOC)
│   └── noise_models.py                  # Custom noise model wrappers (< 150 LOC)
│
├── security/                            # NSM security layer (formerly analysis/)
│   ├── __init__.py
│   ├── bounds.py                        # Max Bound, Γ, entropy rate (< 200 LOC)
│   ├── feasibility.py                   # Pre-flight checker (< 150 LOC)
│   └── parameters.py                    # NSM config: r, ν, ε_sec (< 100 LOC)
│
├── quantum/                             # Phase I: Quantum Generation
│   ├── __init__.py
│   ├── epr.py                           # EPR generation via EPRSocket (< 150 LOC)
│   ├── basis.py                         # Basis selection strategies (< 80 LOC)
│   ├── measurement.py                   # Measurement buffering & storage (< 100 LOC)
│   └── batching.py                      # Memory-constrained batch manager (< 120 LOC)
│
├── sifting/                             # Phase II: Sifting & Estimation
│   ├── __init__.py
│   ├── commitment.py                    # SHA256 commitment scheme (< 120 LOC)
│   ├── sifter.py                        # I₀/I₁ partitioning, key extraction (< 150 LOC)
│   ├── qber.py                          # QBER estimation + μ penalty (< 120 LOC)
│   └── detection_validator.py           # Chernoff bounds, loss validation (< 150 LOC)
│
├── reconciliation/                      # Phase III: Information Reconciliation
│   ├── __init__.py
│   ├── syndrome.py                      # Syndrome computation (< 100 LOC)
│   ├── decoder.py                       # BP decoder core (< 180 LOC)
│   ├── verifier.py                      # Hash verification (< 100 LOC)
│   └── leakage.py                       # Wiretap cost |Σ| tracking (< 120 LOC)
│
├── amplification/                       # Phase IV: Privacy Amplification
│   ├── __init__.py
│   ├── entropy.py                       # h_min calculation from NSM (< 120 LOC)
│   ├── key_length.py                    # Secure length determination (< 100 LOC)
│   ├── toeplitz.py                      # Toeplitz matrix hashing (< 150 LOC)
│   └── formatter.py                     # OT output: S₀, S₁, Sᴄ (< 100 LOC)
│
├── ldpc/                                # LDPC code utilities
│   ├── __init__.py
│   ├── generator.py                     # PEG matrix generation (< 200 LOC)
│   ├── manager.py                       # Matrix loading, caching (< 150 LOC)
│   └── distributions.py                 # Degree distributions (< 80 LOC)
│
├── connection/                          # Classical communication layer
│   ├── __init__.py
│   ├── socket_wrapper.py                # SquidASM socket abstraction (< 100 LOC)
│   ├── messaging.py                     # Message envelope, serialization (< 150 LOC)
│   └── ordered_protocol.py              # Commit-then-reveal state machine (< 150 LOC)
│
├── protocol/                            # Protocol orchestration
│   ├── __init__.py
│   ├── config.py                        # Unified CaligoConfig (< 150 LOC)
│   ├── orchestrator.py                  # Phase sequencing logic (< 200 LOC)
│   ├── alice.py                         # Alice's SquidASM Program (< 180 LOC)
│   └── bob.py                           # Bob's SquidASM Program (< 180 LOC)
│
├── utils/                               # Cross-cutting utilities
│   ├── __init__.py
│   ├── logging.py                       # Structured logging (< 100 LOC)
│   ├── math.py                          # Binary entropy, shared functions (< 80 LOC)
│   └── bitarray_utils.py                # Bitarray helpers (< 80 LOC)
│
└── tests/                               # Test suite (not counted in LOC)
    ├── conftest.py
    ├── test_simulation/                 # Simulation layer tests
    ├── test_quantum/                    # Phase I tests
    ├── test_sifting/                    # Phase II tests
    ├── test_reconciliation/             # Phase III tests
    ├── test_amplification/              # Phase IV tests
    ├── test_security/                   # NSM bounds tests
    └── test_integration/                # End-to-end protocol tests
```

### 5.2 Package Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CALIGO DEPENDENCY ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌──────────────┐                               │
│                              │   protocol/  │                               │
│                              │ orchestrator │                               │
│                              │ alice, bob   │                               │
│                              └──────┬───────┘                               │
│                                     │                                       │
│              ┌──────────────────────┼──────────────────────┐                │
│              │                      │                      │                │
│              ▼                      ▼                      ▼                │
│       ┌──────────┐          ┌──────────────┐        ┌────────────┐          │
│       │ quantum/ │          │   sifting/   │        │reconcilia- │          │
│       │ Phase I  │────┐     │   Phase II   │───┐    │   tion/    │          │
│       └────┬─────┘    │     └──────┬───────┘   │    │ Phase III  │          │
│            │          │            │           │    └─────┬──────┘          │
│            │          │            │           │          │                 │
│            │          │            ▼           │          ▼                 │
│            │          │     ┌──────────────┐   │   ┌────────────┐           │
│            │          └────►│  security/   │◄──┘   │amplifica-  │           │
│            │                │  NSM bounds  │◄──────│   tion/    │           │
│            │                │  feasibility │       │ Phase IV   │           │
│            │                └──────┬───────┘       └─────┬──────┘           │
│            │                       │                     │                  │
│            ▼                       ▼                     │                  │
│     ┌─────────────┐         ┌──────────────┐             │                  │
│     │ simulation/ │◄────────│ connection/  │◄────────────┘                  │
│     │ (BRIDGE)    │         │  messaging   │                                │
│     └──────┬──────┘         └──────────────┘                                │
│            │                                                                │
│  ══════════╪════════════════════════════════════════════════════════════    │
│  EXTERNAL  │  DEPENDENCIES                                                  │
│  ══════════╪════════════════════════════════════════════════════════════    │
│            │                                                                │
│            ▼                                                                │
│     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│     │  SquidASM   │      │   NetQASM   │      │   NetSquid  │               │
│     │  (0.12+)    │─────►│   (0.15+)   │─────►│   (1.1+)    │               │
│     └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  SHARED UTILITIES (no external deps except stdlib)                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│            ▲                                                                │
│     ┌──────┴──────┐      ┌─────────────┐      ┌─────────────┐               │
│     │   types/    │      │    ldpc/    │      │   utils/    │               │
│     │  contracts  │      │  matrices   │      │   logging   │               │
│     └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Import Rules

To maintain clean architecture, Caligo enforces strict import rules:

| Source Package | Allowed Imports | Forbidden Imports |
|----------------|-----------------|-------------------|
| `types/` | `utils/`, stdlib | All other caligo packages |
| `utils/` | stdlib only | All caligo packages |
| `simulation/` | `types/`, `utils/`, SquidASM, NetSquid | Protocol packages |
| `security/` | `types/`, `utils/`, `simulation/` | Protocol packages |
| `ldpc/` | `types/`, `utils/` | All other caligo packages |
| `quantum/` | `types/`, `utils/`, `simulation/`, `security/` | Other phases |
| `sifting/` | `types/`, `utils/`, `simulation/`, `security/`, `connection/` | Other phases |
| `reconciliation/` | `types/`, `utils/`, `ldpc/`, `connection/` | Other phases |
| `amplification/` | `types/`, `utils/`, `security/`, `connection/` | Other phases |
| `connection/` | `types/`, `utils/`, `simulation/` | Phase packages |
| `protocol/` | **All packages** | None (orchestration layer) |

### 5.4 LOC Budget Summary

| Package | Modules | Total LOC Budget |
|---------|---------|-----------------|
| `types/` | 4 | 450 |
| `simulation/` | 5 | 780 |
| `security/` | 3 | 450 |
| `quantum/` | 4 | 450 |
| `sifting/` | 4 | 540 |
| `reconciliation/` | 4 | 500 |
| `amplification/` | 4 | 470 |
| `ldpc/` | 3 | 430 |
| `connection/` | 3 | 400 |
| `protocol/` | 4 | 710 |
| `utils/` | 3 | 260 |
| **TOTAL** | **41** | **~5,440** |

*Note: Budget includes generous margins. Actual implementation target is ~3,500 LOC.*

---

## 6. Module Specifications

### 6.1 Simulation Layer (`simulation/`)

This package is **Caligo's core innovation** — the bridge between $\binom{2}{1}$-OT's domain logic and SquidASM/NetSquid's simulation infrastructure.

#### 6.1.1 `physical_model.py` (< 180 LOC)

**Purpose:** Translate NSM physical parameters to NetSquid noise models.

```python
@dataclass
class PhysicalChannelParameters:
    """NSM-compatible physical channel specification."""
    source_quality_mu: float = 0.98       # μ ∈ [0,1]: EPR source fidelity
    detection_efficiency_eta: float = 0.85 # η ∈ [0,1]: detector efficiency
    intrinsic_error_e_det: float = 0.01   # e_det ∈ [0,0.5]: detector error
    dark_count_rate_hz: float = 100.0     # Dark counts per second
    fiber_length_km: float = 10.0         # Channel length
    fiber_loss_db_per_km: float = 0.2     # Attenuation

class PhysicalModelAdapter:
    """
    Converts NSM parameters to NetSquid noise configurations.
    
    The adapter performs the following mappings:
    
    μ (source quality) → DepolarNoiseModel.depolar_rate
        Using: depolar_rate = (1 - μ) * 4/3
        
    η (detection efficiency) → FibreLossModel.p_loss_init
        Using: p_loss_init = 1 - η
        
    e_det (intrinsic error) → Additional DepolarNoiseModel
        Applied after measurement
    """
    
    def to_link_config(self) -> LinkConfig:
        """Generate SquidASM LinkConfig from physical parameters."""
        
    def to_netsquid_noise_models(self) -> list[QuantumErrorModel]:
        """Generate NetSquid noise model chain."""
        
    def calculate_expected_qber(self) -> float:
        """
        Compute expected QBER from physical parameters.
        
        QBER_total ≈ (1-μ)/2 + e_det + (dark_count contribution)
        """
```

**NetSquid Integration Points:**
- `netsquid.components.models.qerrormodels.DepolarNoiseModel`
- `netsquid.components.models.qerrormodels.FibreLossModel`
- `netsquid.components.models.qerrormodels.T1T2NoiseModel`

#### 6.1.2 `timing.py` (< 150 LOC)

**Purpose:** Enforce NSM timing constraints using NetSquid's discrete-event simulation.

```python
class TimingBarrier:
    """
    Enforces the critical Δt wait time for NSM security.
    
    Security Requirement:
    -------------------
    Adversary's quantum storage must decohere during Δt.
    Basis revelation MUST NOT occur before Δt has elapsed
    since quantum transmission completed.
    
    Implementation:
    --------------
    Uses ns.sim_time() to track simulation time and enforces
    the timing constraint through generator-compatible waits.
    """
    
    def __init__(self, delta_t_ns: float):
        """
        Parameters
        ----------
        delta_t_ns : float
            Minimum wait time in nanoseconds between quantum
            transmission and basis revelation.
        """
        
    def mark_quantum_complete(self) -> None:
        """Record timestamp when quantum phase completes."""
        
    def wait_for_safety(self) -> Generator[Any, None, None]:
        """
        Generator that yields until Δt has elapsed.
        
        Compatible with SquidASM's generator-based execution model.
        Integrates with NetSquid's event scheduler.
        """
        
    def assert_timing_compliant(self) -> None:
        """Raise TimingViolationError if Δt not satisfied."""

class TimingViolationError(SecurityError):
    """Raised when NSM timing constraints are violated."""
```

**NetSquid Integration Points:**
- `netsquid.sim_time()` for current simulation time
- `pydynaa` event scheduling for wait implementation

#### 6.1.3 `network_builder.py` (< 180 LOC)

**Purpose:** Construct SquidASM network configurations from Caligo's config.

```python
class NetworkBuilder:
    """
    Builds SquidASM network topology from CaligoConfig.
    
    Translates high-level protocol configuration into the
    SquidASM NetworkConfig YAML format and programmatically
    constructs network components.
    """
    
    def __init__(self, config: CaligoConfig):
        """Initialize builder with protocol configuration."""
        
    def build_network_config(self) -> dict:
        """
        Generate SquidASM-compatible network configuration.
        
        Returns
        -------
        dict
            Configuration dict matching SquidASM's network.yaml format:
            {
                'stacks': [...],
                'links': [...],
                'noise_type': ...,
                'fidelity': ...
            }
        """
        
    def create_network(self) -> SquidASMNetwork:
        """
        Programmatically create SquidASM network instance.
        
        Alternative to YAML loading for tighter integration.
        """
        
    def inject_noise_models(self, network: SquidASMNetwork) -> None:
        """
        Post-creation injection of custom noise models.
        
        Used when standard SquidASM noise types are insufficient.
        """
```

**SquidASM Integration Points:**
- `squidasm.sim.network.network.SquidASMNetwork`
- `squidasm.sim.stack.stack.StackNetworkConfig`
- `squidasm.run.multithread.run()`

#### 6.1.4 `detection.py` (< 120 LOC)

**Purpose:** Track qubit detection events for validation and statistics.

```python
@dataclass
class DetectionEvent:
    """Single detection event record."""
    round_id: int
    detected: bool
    timestamp_ns: float
    basis: int  # 0=Z, 1=X

class DetectionEventTracker:
    """
    Tracks detection events for missing round validation.
    
    Monitors EPR generation success/failure and validates
    that reported losses match expected channel transmittance.
    """
    
    def record_event(self, round_id: int, detected: bool) -> None:
        """Record a detection event."""
        
    def get_detection_rate(self) -> float:
        """Calculate observed detection rate."""
        
    def validate_against_expected(
        self, 
        expected_transmittance: float,
        confidence: float = 0.99
    ) -> ValidationResult:
        """
        Validate observed vs expected detection rate.
        
        Uses Chernoff bounds to check for anomalies.
        """
```

#### 6.1.5 `noise_models.py` (< 150 LOC)

**Purpose:** Custom NetSquid noise models for NSM-specific requirements.

```python
class AdversaryStorageNoiseModel(QuantumErrorModel):
    """
    Models adversary's quantum memory decoherence.
    
    In the NSM, security derives from the adversary's storage
    being noisy. This model represents Eve's memory with
    configurable noise rate r.
    
    Used for security analysis, not direct simulation of
    honest parties.
    """
    
    def __init__(self, storage_noise_rate: float):
        """
        Parameters
        ----------
        storage_noise_rate : float
            Rate r at which adversary's storage decoheres.
            Higher r = more security margin.
        """

class CompositeChannelNoise(QuantumErrorModel):
    """
    Combines multiple noise sources into single model.
    
    Aggregates source imperfection, fiber loss, and detector
    noise into a unified error model for simplified application.
    """
```

---

### 6.2 Security Layer (`security/`)

#### 6.2.1 `bounds.py` (< 200 LOC)

**Purpose:** NSM entropy calculations and security bounds.

```python
def gamma_function(r: float) -> float:
    """
    Compute Γ(r) for NSM security bound.
    
    Γ(r) characterizes the adversary's storage quality.
    
    Parameters
    ----------
    r : float
        Storage noise rate ∈ [0, 1]
        
    Returns
    -------
    float
        Γ(r) value used in entropy calculations
    """

def max_bound_entropy_rate(
    qber: float,
    r: float,
    nu: float,
    n: int
) -> float:
    """
    Calculate maximum extractable entropy rate per qubit.
    
    This is the core NSM bound determining secure key length.
    
    Parameters
    ----------
    qber : float
        Observed quantum bit error rate
    r : float
        Adversary storage noise rate
    nu : float
        Security parameter ν
    n : int
        Number of qubits
        
    Returns
    -------
    float
        Entropy rate h_min(r) in bits per qubit
    """

def channel_capacity(qber: float) -> float:
    """Binary symmetric channel capacity: 1 - h(qber)"""

class NSMBoundsCalculator:
    """
    Comprehensive NSM security bound calculations.
    
    Encapsulates all entropy and security parameter computations
    needed for Protocol Phase IV.
    """
    
    def calculate_secure_key_length(
        self,
        reconciled_bits: int,
        syndrome_leakage: int,
        qber: float
    ) -> int:
        """Determine secure output key length."""
```

#### 6.2.2 `feasibility.py` (< 150 LOC)

**Purpose:** Pre-flight protocol feasibility checks.

```python
class FeasibilityResult(Enum):
    """Pre-flight check outcomes."""
    FEASIBLE = "feasible"
    WARNING_HIGH_QBER = "warning_high_qber"
    ABORT_QBER_EXCEEDED = "abort_qber_exceeded"
    ABORT_NSM_VIOLATED = "abort_nsm_violated"

class FeasibilityChecker:
    """
    Pre-flight feasibility validation for $\binom{2}{1}$-OT execution.
    
    Checks that physical parameters support secure protocol
    execution before committing resources.
    """
    
    def check(
        self,
        physical_params: PhysicalChannelParameters,
        security_params: SecurityParameters
    ) -> tuple[FeasibilityResult, str]:
        """
        Perform comprehensive feasibility check.
        
        Validates:
        1. Expected QBER < 22% (hard limit)
        2. Expected QBER < 11% (conservative)
        3. Trusted noise < Adversary storage noise (NSM)
        
        Returns
        -------
        tuple[FeasibilityResult, str]
            Result enum and human-readable explanation
        """
```

---

### 6.3 Protocol Phase Packages

#### 6.3.1 `quantum/epr.py` (< 150 LOC)

```python
class EPRGenerator:
    """
    Manages EPR pair generation via SquidASM.
    
    Wraps EPRSocket operations with timing marks and
    detection tracking for Caligo requirements.
    """
    
    def __init__(
        self,
        epr_socket: EPRSocket,
        timing_barrier: TimingBarrier,
        detection_tracker: DetectionEventTracker
    ):
        """Initialize with SquidASM socket and tracking."""
        
    def generate_batch(
        self,
        num_pairs: int,
        basis_choices: list[int]
    ) -> Generator[Any, None, QuantumPhaseResult]:
        """
        Generate EPR pairs with measurements.
        
        Generator-compatible for SquidASM execution model.
        
        Yields
        ------
        Control back to SquidASM for async operations
        
        Returns (on completion)
        -------
        QuantumPhaseResult
            Phase I output contract
        """
```

#### 6.3.2 `sifting/commitment.py` (< 120 LOC)

```python
class SHA256Commitment:
    """
    SHA256-based commitment scheme for basis hiding.
    
    Used in Phase II to commit to detection reports
    before basis revelation.
    """
    
    def commit(self, data: bytes) -> tuple[bytes, bytes]:
        """
        Generate commitment.
        
        Parameters
        ----------
        data : bytes
            Data to commit to
            
        Returns
        -------
        tuple[bytes, bytes]
            (commitment, opening) pair
        """
        
    def verify(
        self,
        commitment: bytes,
        opening: bytes,
        data: bytes
    ) -> bool:
        """Verify commitment opens to claimed data."""
```

#### 6.3.3 `reconciliation/decoder.py` (< 180 LOC)

```python
class BeliefPropagationDecoder:
    """
    LDPC Belief Propagation decoder for error correction.
    
    Implements min-sum BP variant for efficient decoding.
    """
    
    def __init__(
        self,
        parity_check_matrix: np.ndarray,
        max_iterations: int = 100
    ):
        """Initialize decoder with LDPC matrix."""
        
    def decode(
        self,
        received: np.ndarray,
        syndrome: np.ndarray,
        channel_llr: np.ndarray
    ) -> tuple[np.ndarray, bool, int]:
        """
        Perform BP decoding.
        
        Parameters
        ----------
        received : np.ndarray
            Bob's received bits
        syndrome : np.ndarray
            Syndrome from Alice
        channel_llr : np.ndarray
            Initial LLR from channel (based on QBER)
            
        Returns
        -------
        tuple[np.ndarray, bool, int]
            (decoded_bits, success, iterations_used)
        """
```

#### 6.3.4 `amplification/toeplitz.py` (< 150 LOC)

```python
class ToeplitzHasher:
    """
    Toeplitz matrix-based universal hashing for privacy amplification.
    
    Implements efficient hashing using FFT-based multiplication.
    """
    
    def __init__(self, seed: bytes, input_length: int, output_length: int):
        """
        Initialize hasher with random seed.
        
        Parameters
        ----------
        seed : bytes
            Random seed for Toeplitz matrix generation
        input_length : int
            Length of input key in bits
        output_length : int
            Desired output length (secure key length)
        """
        
    def hash(self, input_key: bitarray) -> bitarray:
        """
        Apply Toeplitz hash to input.
        
        Returns
        -------
        bitarray
            Privacy-amplified key of output_length bits
        """
```

---

### 6.4 Connection Layer (`connection/`)

#### 6.4.1 `ordered_protocol.py` (< 150 LOC)

```python
class ProtocolPhase(Enum):
    """State machine phases for ordered messaging."""
    INIT = "init"
    DETECTION_REPORT = "detection_report"
    WAITING_DELTA_T = "waiting_delta_t"
    BASIS_REVEAL = "basis_reveal"
    SIFTING = "sifting"
    RECONCILIATION = "reconciliation"
    AMPLIFICATION = "amplification"
    COMPLETE = "complete"

class OrderedProtocolSocket:
    """
    Enforces strict message ordering for $\binom{2}{1}$-OT protocol.
    
    Prevents race conditions in the "Sandwich" protocol flow
    by requiring acknowledgments before state transitions.
    
    State Machine:
    -------------
    INIT → DETECTION_REPORT → WAITING_DELTA_T → BASIS_REVEAL → ...
    
    Each transition requires explicit ACK from peer.
    """
    
    def __init__(self, csocket: ClassicalSocket, role: str):
        """
        Parameters
        ----------
        csocket : ClassicalSocket
            Underlying SquidASM classical socket
        role : str
            "alice" or "bob"
        """
        
    def send_with_ack(
        self,
        message: bytes,
        expected_phase: ProtocolPhase
    ) -> Generator:
        """Send message and wait for acknowledgment."""
        
    def transition_to(self, phase: ProtocolPhase) -> None:
        """Explicit phase transition with validation."""
```

---

## 7. SquidASM/NetSquid Bridge Layer

This section provides in-depth technical specifications for integrating Caligo with the quantum network simulation stack.

### 7.1 Framework Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION FRAMEWORK STACK                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                           CALIGO                                   │     │
│  │  Protocol logic, domain types, security analysis                   │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          SquidASM (0.12+)                          │     │
│  │  • Program abstraction (Alice/Bob programs)                        │     │
│  │  • EPRSocket, ClassicalSocket                                      │     │
│  │  • ProgramContext for resource access                              │     │
│  │  • Multithread runner                                              │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          NetQASM (0.15+)                           │     │
│  │  • SDK: Qubit, QubitMeasureBasis                                   │     │
│  │  • EprMeasBasis, EprMeasureResult                                  │     │
│  │  • Instruction compilation                                         │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          NetSquid (1.1+)                           │     │
│  │  • Discrete-event simulation core                                  │     │
│  │  • Quantum error models (DepolarNoiseModel, T1T2NoiseModel, etc.)  │     │
│  │  • Components: QuantumProcessor, QuantumMemory                     │     │
│  │  • ns.sim_time(), pydynaa event scheduling                         │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                       netsquid_magic (0.7+)                        │     │
│  │  • MagicDistributor: EPR pair distribution                         │     │
│  │  • State samplers: Perfect, Depolarise, LinearDepolarise           │     │
│  │  • Link layer protocols                                            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Critical Integration Points

#### 7.2.1 EPR Generation Integration

**SquidASM Interface:**
```python
# From squidasm/squidasm/sim/stack/program.py
class ProgramContext:
    epr_sockets: Dict[str, EPRSocket]

# From squidasm/squidasm/sim/stack/egp.py  
class EPRSocket:
    def create_keep(
        self,
        number: int = 1,
        post_routine: Optional[Callable] = None,
        sequential: bool = False,
        tp: EPRType = EPRType.K
    ) -> Generator[EventExpression, None, List[Qubit]]
    
    def recv_keep(
        self,
        number: int = 1,
        post_routine: Optional[Callable] = None,
        sequential: bool = False
    ) -> Generator[EventExpression, None, List[Qubit]]
```

**Caligo Wrapper Strategy:**
```python
# caligo/quantum/epr.py

class EPRGeneratorConfig:
    """Configuration for EPR generation behavior."""
    batch_size: int = 100              # Pairs per batch (memory limit)
    sequential_mode: bool = False       # True for debugging
    post_measurement: bool = True       # Measure immediately

class CaligoEPRGenerator:
    """
    Caligo's EPR generation wrapper.
    
    Adds:
    - Timing marks for Δt enforcement
    - Detection tracking for validation
    - Basis selection integration
    - Memory-aware batching
    """
    
    def __init__(
        self,
        epr_socket: EPRSocket,
        config: EPRGeneratorConfig,
        timing: TimingBarrier,
        tracker: DetectionEventTracker
    ):
        self._socket = epr_socket
        self._config = config
        self._timing = timing
        self._tracker = tracker
        
    def generate_measured_pairs(
        self,
        num_pairs: int,
        basis_selector: BasisSelector
    ) -> Generator[EventExpression, None, QuantumPhaseResult]:
        """
        Generate EPR pairs with immediate measurement.
        
        This is the primary generation mode for $\binom{2}{1}$-OT where
        Alice and Bob measure immediately in random bases.
        
        Yields
        ------
        EventExpression
            SquidASM event expressions for async coordination
            
        Returns
        -------
        QuantumPhaseResult
            Phase I contract with all measurements and metadata
        """
        outcomes = []
        bases = []
        round_ids = []
        
        for batch_start in range(0, num_pairs, self._config.batch_size):
            batch_size = min(self._config.batch_size, num_pairs - batch_start)
            batch_bases = basis_selector.select_batch(batch_size)
            
            # Yield to SquidASM for EPR generation
            qubits = yield from self._socket.create_keep(number=batch_size)
            
            for i, qubit in enumerate(qubits):
                basis = batch_bases[i]
                outcome = qubit.measure(
                    basis=QubitMeasureBasis.X if basis == 1 else QubitMeasureBasis.Z
                )
                outcomes.append(int(outcome))
                bases.append(basis)
                round_ids.append(batch_start + i)
                
                self._tracker.record_event(
                    round_id=batch_start + i,
                    detected=True,
                    timestamp=ns.sim_time()
                )
        
        # Mark quantum phase complete for timing
        self._timing.mark_quantum_complete()
        
        return QuantumPhaseResult(
            measurement_outcomes=outcomes,
            basis_choices=bases,
            round_ids=round_ids,
            generation_timestamp=ns.sim_time(),
            num_pairs_requested=num_pairs,
            num_pairs_generated=len(outcomes)
        )
```

#### 7.2.2 Noise Model Integration

**NetSquid Native Models (from netsquid.components.models.qerrormodels):**

| Model | Parameters | $\binom{2}{1}$-OT Usage |
|-------|------------|-------------|
| `DepolarNoiseModel` | `depolar_rate`, `time_independent` | Source imperfection (μ) |
| `DephaseNoiseModel` | `dephase_rate`, `time_independent` | Phase errors |
| `T1T2NoiseModel` | `T1`, `T2` | Memory decoherence |
| `FibreLossModel` | `p_loss_length`, `p_loss_init` | Detection efficiency (η) |

**Caligo Adapter Implementation:**
```python
# caligo/simulation/physical_model.py

from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    DephaseNoiseModel, 
    T1T2NoiseModel,
    FibreLossModel
)

class PhysicalModelAdapter:
    """
    Maps NSM parameters to NetSquid noise models.
    
    The Noisy Storage Model uses physical parameters that don't
    directly correspond to NetSquid's noise model parameters.
    This adapter performs the necessary translations.
    """
    
    def __init__(self, params: PhysicalChannelParameters):
        self._params = params
        
    def create_source_noise_model(self) -> DepolarNoiseModel:
        """
        Create noise model for imperfect EPR source.
        
        Source Quality μ Translation:
        ----------------------------
        A source with quality μ produces Werner states:
            ρ = μ|Φ+⟩⟨Φ+| + (1-μ)I/4
            
        This corresponds to a depolarizing channel with:
            depolar_rate = (1 - μ) * 4/3
            
        For μ = 1.0: Perfect source, no depolarization
        For μ = 0.5: Maximally mixed, depolar_rate = 2/3
        """
        if self._params.source_quality_mu >= 1.0:
            return None  # No noise needed
            
        depolar_rate = (1.0 - self._params.source_quality_mu) * 4.0 / 3.0
        return DepolarNoiseModel(
            depolar_rate=depolar_rate,
            time_independent=True  # Applied once at generation
        )
    
    def create_fiber_loss_model(self) -> FibreLossModel:
        """
        Create fiber loss model for detection efficiency.
        
        Detection Efficiency η Translation:
        ----------------------------------
        η combines fiber transmission and detector efficiency.
        
        Total loss = fiber_loss * (1 - η_detector)
        
        Where fiber_loss depends on length and attenuation.
        """
        # Fiber transmission probability
        fiber_transmission = 10 ** (
            -self._params.fiber_loss_db_per_km * 
            self._params.fiber_length_km / 10.0
        )
        
        # Combined with detector efficiency
        total_transmission = fiber_transmission * self._params.detection_efficiency_eta
        
        return FibreLossModel(
            p_loss_init=1.0 - total_transmission,
            p_loss_length=0  # Already accounted for
        )
    
    def create_memory_noise_model(
        self,
        t1_ns: float,
        t2_ns: float
    ) -> T1T2NoiseModel:
        """
        Create memory decoherence model.
        
        For honest parties, this models realistic quantum memory.
        T1: Amplitude damping time (energy relaxation)
        T2: Dephasing time (T2 ≤ 2*T1)
        """
        return T1T2NoiseModel(T1=t1_ns, T2=t2_ns)
    
    def calculate_expected_qber(self) -> float:
        """
        Calculate expected QBER from all physical parameters.
        
        QBER Components:
        ---------------
        1. Source imperfection: Q_source ≈ (1-μ)/2
        2. Detector error: Q_det = e_det  
        3. Dark counts: Q_dark ≈ p_dark / (p_dark + p_signal)
        
        Total (approximation for low noise):
            Q_total ≈ Q_source + Q_det + Q_dark
        """
        q_source = (1.0 - self._params.source_quality_mu) / 2.0
        q_det = self._params.intrinsic_error_e_det
        
        # Dark count contribution (simplified)
        # Assumes 1 MHz detection rate for normalization
        detection_rate = 1e6  # Hz
        p_dark = self._params.dark_count_rate_hz / detection_rate
        q_dark = p_dark / (p_dark + self._params.detection_efficiency_eta)
        
        return q_source + q_det + q_dark
```

#### 7.2.3 Timing Integration

**NetSquid Time Access:**
```python
# Direct simulation time access
import netsquid as ns
current_time = ns.sim_time()  # Returns float in nanoseconds
```

**Caligo TimingBarrier Implementation:**
```python
# caligo/simulation/timing.py

import netsquid as ns
from typing import Generator, Any

class TimingBarrier:
    """
    NSM timing enforcement integrated with NetSquid simulation.
    
    Critical Security Property:
    -------------------------
    The adversary's quantum storage noise rate r means their
    stored qubits decohere as: ρ(t) → ρ_decohered after time Δt.
    
    We MUST ensure Δt has passed (in simulation time) before
    basis revelation, otherwise NSM security assumptions fail.
    """
    
    def __init__(self, delta_t_ns: float):
        self._delta_t = delta_t_ns
        self._quantum_complete_time: Optional[float] = None
        self._timing_validated: bool = False
        
    def mark_quantum_complete(self) -> None:
        """
        Record when quantum transmission phase completes.
        
        Called after all EPR pairs are generated and measured
        but BEFORE any basis information is exchanged.
        """
        self._quantum_complete_time = ns.sim_time()
        
    def elapsed_time(self) -> float:
        """Time elapsed since quantum phase completion."""
        if self._quantum_complete_time is None:
            return 0.0
        return ns.sim_time() - self._quantum_complete_time
    
    def is_safe_to_reveal(self) -> bool:
        """Check if Δt has elapsed."""
        return self.elapsed_time() >= self._delta_t
    
    def wait_for_safety(self) -> Generator[Any, None, None]:
        """
        Yield until Δt has elapsed.
        
        This integrates with SquidASM's generator model.
        In practice, the classical socket delays naturally
        introduce time, but this ensures the minimum.
        
        Implementation Note:
        -------------------
        NetSquid advances simulation time when events are scheduled.
        We schedule a "wake up" event at the required time and yield.
        """
        if self._quantum_complete_time is None:
            raise TimingViolationError(
                "Cannot wait for safety: quantum phase not marked complete"
            )
            
        remaining_ns = self._delta_t - self.elapsed_time()
        
        if remaining_ns > 0:
            # Schedule wake-up event
            # This uses pydynaa's event scheduling under the hood
            yield from self._schedule_delay(remaining_ns)
            
        self._timing_validated = True
    
    def assert_compliance(self) -> None:
        """Raise error if timing constraint not satisfied."""
        if not self.is_safe_to_reveal():
            raise TimingViolationError(
                f"Δt not satisfied: elapsed={self.elapsed_time():.0f}ns, "
                f"required={self._delta_t:.0f}ns"
            )
    
    def _schedule_delay(self, delay_ns: float) -> Generator:
        """
        Schedule a delay in the NetSquid simulation.
        
        Implementation depends on execution context:
        - In SquidASM: yield from sleep-equivalent
        - In standalone: ns.sim_run(duration=delay_ns)
        """
        # Placeholder - actual implementation uses pydynaa
        # or SquidASM's scheduling primitives
        pass
```

### 7.3 Network Configuration Generation

**SquidASM Network YAML Structure:**
```yaml
# Standard SquidASM network configuration format
stacks:
  - name: "Alice"
    qdevice_typ: "generic"
    qdevice_cfg:
      num_qubits: 10
  - name: "Bob"
    qdevice_typ: "generic"
    qdevice_cfg:
      num_qubits: 10

links:
  - stack1: "Alice"
    stack2: "Bob"
    typ: "depolarise"
    cfg:
      fidelity: 0.95
      prob_success: 0.9
      t_cycle: 1000
```

**Caligo Configuration Generator:**
```python
# caligo/simulation/network_builder.py

class NetworkBuilder:
    """
    Generates SquidASM network configurations from Caligo parameters.
    """
    
    def __init__(self, config: CaligoConfig):
        self._config = config
        self._physical = PhysicalModelAdapter(config.physical)
        
    def generate_config_dict(self) -> dict:
        """
        Generate SquidASM-compatible network configuration.
        
        Maps:
        - source_quality_mu → fidelity
        - detection_efficiency_eta → prob_success
        - Physical timing → t_cycle
        """
        # Calculate fidelity from source quality
        # Fidelity in SquidASM is F = (1 + 3*μ_werner) / 4
        # For our source quality μ (Werner parameter):
        fidelity = (1.0 + 3.0 * self._config.physical.source_quality_mu) / 4.0
        
        # prob_success combines detection efficiency and fiber transmission
        fiber_trans = 10 ** (
            -self._config.physical.fiber_loss_db_per_km *
            self._config.physical.fiber_length_km / 10.0
        )
        prob_success = fiber_trans * self._config.physical.detection_efficiency_eta
        
        return {
            "stacks": [
                {
                    "name": "Alice",
                    "qdevice_typ": "generic",
                    "qdevice_cfg": {
                        "num_qubits": self._config.quantum.max_qubits_per_node,
                        "T1": self._config.physical.memory_t1_ns,
                        "T2": self._config.physical.memory_t2_ns,
                    }
                },
                {
                    "name": "Bob",
                    "qdevice_typ": "generic",
                    "qdevice_cfg": {
                        "num_qubits": self._config.quantum.max_qubits_per_node,
                        "T1": self._config.physical.memory_t1_ns,
                        "T2": self._config.physical.memory_t2_ns,
                    }
                }
            ],
            "links": [
                {
                    "stack1": "Alice",
                    "stack2": "Bob",
                    "typ": "depolarise",
                    "cfg": {
                        "fidelity": fidelity,
                        "prob_success": prob_success,
                        "t_cycle": self._config.physical.state_delay_ns,
                    }
                }
            ]
        }
    
    def build_and_run(
        self,
        alice_program: Type[Program],
        bob_program: Type[Program],
        alice_input: Any,
        bob_input: Any
    ) -> list[dict]:
        """
        Build network and execute programs.
        
        Returns results from both Alice and Bob.
        """
        from squidasm.run.stack.run import run
        
        config_dict = self.generate_config_dict()
        cfg = StackNetworkConfig.from_dict(config_dict)
        
        return run(
            config=cfg,
            programs={"Alice": alice_program, "Bob": bob_program},
            inputs={"Alice": alice_input, "Bob": bob_input}
        )
```

### 7.4 Generator-Based Execution Patterns

SquidASM uses Python generators for asynchronous quantum operations. Caligo must follow these patterns:

```python
# Pattern 1: Sequential quantum operations
def sequential_generation(context: ProgramContext, n: int):
    """Generate n EPR pairs sequentially."""
    epr = context.epr_sockets["peer"]
    results = []
    
    for i in range(n):
        qubit = yield from epr.create_keep()[0]  # Yields for each pair
        m = qubit.measure()
        yield from context.connection.flush()   # Sync measurement
        results.append(int(m))
        
    return results

# Pattern 2: Batched quantum operations (preferred for performance)
def batched_generation(context: ProgramContext, n: int, batch_size: int):
    """Generate n EPR pairs in batches."""
    epr = context.epr_sockets["peer"]
    results = []
    
    for start in range(0, n, batch_size):
        batch_n = min(batch_size, n - start)
        qubits = yield from epr.create_keep(number=batch_n)  # Single yield
        
        for qubit in qubits:
            m = qubit.measure()
            results.append(int(m))
            
        yield from context.connection.flush()  # Sync batch
        
    return results

# Pattern 3: Interleaved quantum and classical ($\binom{2}{1}$-OT pattern)
def ehok_phase1(context: ProgramContext, config: QuantumConfig):
    """
    $\binom{2}{1}$-OT Phase I with timing integration.
    
    Key requirements:
    1. Generate all EPR pairs
    2. Mark timing for Δt
    3. Do NOT send classical until Δt passes
    """
    epr = context.epr_sockets["bob"]
    csocket = context.csockets["bob"]
    timing = TimingBarrier(config.delta_t_ns)
    
    # Quantum phase
    measurements = []
    bases = []
    
    for batch in generate_batches(config.num_pairs, config.batch_size):
        qubits = yield from epr.create_keep(number=len(batch.basis_choices))
        
        for qubit, basis in zip(qubits, batch.basis_choices):
            m = qubit.measure(
                basis=QubitMeasureBasis.X if basis else QubitMeasureBasis.Z
            )
            measurements.append(int(m))
            bases.append(basis)
            
        yield from context.connection.flush()
    
    # CRITICAL: Mark quantum complete for timing
    timing.mark_quantum_complete()
    
    # Wait for Δt before any classical communication
    yield from timing.wait_for_safety()
    
    # Now safe to proceed with Phase II
    return QuantumPhaseResult(
        measurement_outcomes=measurements,
        basis_choices=bases,
        # ... other fields
    )
```

### 7.5 Error Handling Integration

```python
# caligo/types/exceptions.py

class CaligoError(Exception):
    """Base exception for all Caligo errors."""
    pass

class SimulationError(CaligoError):
    """Errors related to SquidASM/NetSquid simulation."""
    pass

class TimingViolationError(SimulationError):
    """NSM timing constraint violated."""
    pass

class SecurityError(CaligoError):
    """Security-critical error - protocol must abort."""
    pass

class QBERThresholdExceeded(SecurityError):
    """QBER exceeds secure threshold."""
    def __init__(self, observed: float, threshold: float):
        self.observed = observed
        self.threshold = threshold
        super().__init__(
            f"QBER {observed:.2%} exceeds threshold {threshold:.2%}"
        )

class NSMViolationError(SecurityError):
    """NSM assumptions not satisfied."""
    pass
```

---

## 8. Implementation Roadmap

### 8.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CALIGO IMPLEMENTATION PHASES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE A: Foundation (Week 1-2)                                             │
│  ├── types/ package: domain primitives                                      │
│  ├── utils/ package: logging, math helpers                                  │
│  └── Project scaffolding: pyproject.toml, testing infrastructure            │
│                                                                             │
│  PHASE B: Simulation Layer (Week 2-3)                                       │
│  ├── simulation/physical_model.py: NSM → NetSquid mapping                   │
│  ├── simulation/timing.py: TimingBarrier with ns.sim_time()                 │
│  ├── simulation/network_builder.py: Config generation                       │
│  └── Integration tests with SquidASM                                        │
│                                                                             │
│  PHASE C: Security Layer (Week 3-4)                                         │
│  ├── security/bounds.py: NSM entropy calculations                           │
│  ├── security/feasibility.py: Pre-flight checks                             │
│  └── Unit tests for all security bounds                                     │
│                                                                             │
│  PHASE D: Protocol Phases (Week 4-6)                                        │
│  ├── quantum/: Phase I implementation                                       │
│  ├── sifting/: Phase II implementation                                      │
│  ├── reconciliation/: Phase III (extract from ehok)                         │
│  ├── amplification/: Phase IV (extract from ehok)                           │
│  └── Phase contract tests                                                   │
│                                                                             │
│  PHASE E: Orchestration (Week 6-7)                                          │
│  ├── connection/: Ordered messaging                                         │
│  ├── protocol/alice.py, bob.py: SquidASM programs                           │
│  ├── protocol/orchestrator.py: Phase sequencing                             │
│  └── End-to-end integration tests                                           │
│                                                                             │
│  PHASE F: Validation (Week 7-8)                                             │
│  ├── Full caligo protocol runs                                               │
│  ├── Performance benchmarking                                               │
│  ├── Security validation against NSM bounds                                 │
│  └── Documentation and examples                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Detailed Task Breakdown

#### Phase A: Foundation

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| A1 | Project scaffolding (pyproject.toml, structure) | 2h | None |
| A2 | `types/keys.py`: Key dataclasses | 3h | A1 |
| A3 | `types/measurements.py`: Measurement records | 2h | A1 |
| A4 | `types/phase_contracts.py`: Phase I-IV contracts | 4h | A2, A3 |
| A5 | `types/exceptions.py`: Exception hierarchy | 2h | A1 |
| A6 | `utils/logging.py`: Structured logging | 3h | A1 |
| A7 | `utils/math.py`: Binary entropy, etc. | 2h | A1 |
| A8 | Test infrastructure: conftest.py, fixtures | 4h | A1-A7 |

#### Phase B: Simulation Layer

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| B1 | `simulation/physical_model.py` | 8h | A (all) |
| B2 | `simulation/timing.py`: TimingBarrier | 6h | A5 |
| B3 | `simulation/detection.py`: Event tracking | 4h | A3 |
| B4 | `simulation/network_builder.py` | 6h | B1 |
| B5 | `simulation/noise_models.py`: Custom models | 4h | B1 |
| B6 | Integration tests: SquidASM execution | 8h | B1-B5 |

#### Phase C: Security Layer

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| C1 | `security/bounds.py`: NSM math | 8h | A7 |
| C2 | `security/feasibility.py`: Checker | 4h | C1, B1 |
| C3 | `security/parameters.py`: Config | 2h | A1 |
| C4 | Unit tests: All bounds validated | 6h | C1-C3 |

#### Phase D: Protocol Phases

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| D1 | `quantum/epr.py`: EPR generation | 6h | B (all) |
| D2 | `quantum/basis.py`: Basis selection | 2h | A3 |
| D3 | `quantum/measurement.py`: Buffering | 3h | A3 |
| D4 | `quantum/batching.py`: Memory management | 4h | D1 |
| D5 | `sifting/commitment.py`: SHA256 | 3h | A5 |
| D6 | `sifting/sifter.py`: Key extraction | 4h | A4 |
| D7 | `sifting/qber.py`: Estimation | 4h | C1 |
| D8 | `sifting/detection_validator.py`: Chernoff | 4h | B3 |
| D9 | Extract LDPC decoder from ehok | 4h | None |
| D10 | `reconciliation/` package | 8h | D9 |
| D11 | Extract Toeplitz from ehok | 2h | None |
| D12 | `amplification/` package | 6h | D11, C1 |

#### Phase E: Orchestration

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| E1 | `connection/socket_wrapper.py` | 3h | B2 |
| E2 | `connection/messaging.py`: Envelopes | 4h | A4 |
| E3 | `connection/ordered_protocol.py`: State machine | 6h | E1, E2 |
| E4 | `protocol/config.py`: CaligoConfig | 4h | B1, C3 |
| E5 | `protocol/alice.py`: Alice program | 8h | D (all), E3 |
| E6 | `protocol/bob.py`: Bob program | 8h | D (all), E3 |
| E7 | `protocol/orchestrator.py`: Sequencing | 6h | E4-E6 |
| E8 | End-to-end tests | 10h | E (all) |

#### Phase F: Validation

| Task | Deliverable | Est. Hours | Dependencies |
|------|-------------|------------|--------------|
| F1 | Full protocol runs with various configs | 8h | E (all) |
| F2 | Performance benchmarking | 4h | F1 |
| F3 | Security validation tests | 6h | F1, C4 |
| F4 | Documentation: README, API docs | 6h | F1-F3 |
| F5 | Example scripts | 4h | F4 |

### 8.3 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SquidASM API changes | HIGH | Pin version, add compatibility layer |
| Timing enforcement complexity | HIGH | Early prototype in Phase B |
| NSM bounds validation | MEDIUM | Literature review, test against known results |
| LDPC extraction from ehok | LOW | Well-tested, isolated component |
| Performance (batch sizes) | MEDIUM | Configurable, benchmark early |

### 8.4 Success Criteria

1. **Functional**: Complete $\binom{2}{1}$-OT execution produces valid S₀, S₁, Sᴄ keys
2. **Secure**: QBER thresholds enforced, Δt timing verified
3. **Integrated**: Runs on SquidASM 0.12+ without patches
4. **Maintainable**: All modules ≤ 200 LOC, 90%+ test coverage
5. **Documented**: API docs, examples, architecture diagram

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Lead Architect | Initial complete architecture document |


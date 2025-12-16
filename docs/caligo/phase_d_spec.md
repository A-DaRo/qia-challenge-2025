# Caligo Phase D: Protocol Phases Specification

**Document Type:** Formal Specification  
**Version:** 1.0  
**Date:** December 17, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)  
**Prerequisites:** [phase_a_spec.md](phase_a_spec.md), [phase_b_spec.md](phase_b_spec.md), [phase_c_spec.md](phase_c_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope & Deliverables](#2-scope--deliverables)
3. [Theoretical Workflow Overview](#3-theoretical-workflow-overview)
4. [Package: `quantum/` — Phase I Implementation](#4-package-quantum--phase-i-implementation)
5. [Package: `sifting/` — Phase II Implementation](#5-package-sifting--phase-ii-implementation)
6. [Package: `amplification/` — Phase IV Implementation](#6-package-amplification--phase-iv-implementation)
7. [Phase Contract Tests](#7-phase-contract-tests)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

## 1. Executive Summary

**Phase D** implements the core protocol phases that constitute the E-HOK workflow. This phase translates theoretical cryptographic primitives into executable simulation components that integrate seamlessly with Phases A-C (Foundation, Simulation, Security layers).

### 1.1 Deliverable Overview

| Package | Protocol Phase | Purpose | Est. LOC |
|---------|----------------|---------|----------|
| `quantum/` | Phase I | EPR generation, basis selection, measurement | ~450 |
| `sifting/` | Phase II | Commit-reveal, basis matching, QBER estimation | ~540 |
| `amplification/` | Phase IV | Entropy calculation, privacy amplification, OT output | ~470 |

**Note:** Phase III (Reconciliation) is handled separately as it involves LDPC codes extracted from ehok with minimal modification.

### 1.2 Design Philosophy

Phase D adheres to the Caligo design principles established in the architecture document:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE D DESIGN PRINCIPLES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DOMAIN-DRIVEN MODULES                                                   │
│     └── Package names reflect E-HOK phases (quantum/, sifting/, etc.)       │
│     └── Module names reflect cryptographic primitives                       │
│                                                                             │
│  2. SIMULATION-NATIVE EXECUTION                                             │
│     └── All quantum operations are generator-compatible                     │
│     └── Timing barriers integrate with ns.sim_time()                        │
│     └── Detection events flow through Phase B infrastructure                │
│                                                                             │
│  3. CONTRACT-DRIVEN BOUNDARIES                                              │
│     └── Each phase produces typed output contracts (Phase A types)          │
│     └── Contracts are validated at phase transitions                        │
│     └── Security checks gate progression (Phase C integration)              │
│                                                                             │
│  4. EXTRACTION FROM ehok                                                    │
│     └── Proven algorithms extracted, not reinvented                         │
│     └── Toeplitz hashing, QBER estimation from ehok                         │
│     └── Refactored for Caligo's module size constraint (≤200 LOC)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Integration Points

Phase D bridges all preceding phases:

| Dependency | Component Used | Purpose |
|------------|----------------|---------|
| **Phase A** | `types/phase_contracts.py` | `QuantumPhaseResult`, `SiftingPhaseResult`, `AmplificationResult` |
| **Phase A** | `types/exceptions.py` | `QBERThresholdExceeded`, `SecurityError` |
| **Phase B** | `simulation/timing.py` | `TimingBarrier` for Δt enforcement |
| **Phase B** | `simulation/detection.py` | `DetectionEventTracker` |
| **Phase C** | `security/bounds.py` | `max_bound_entropy()`, `gamma_function()` |
| **Phase C** | `security/feasibility.py` | `FeasibilityChecker` for pre-flight validation |

### 1.4 Critical Protocol Flow

The 1-out-of-2 OT protocol enforces a strict temporal ordering that Phase D must implement:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OT TEMPORAL FLOW (Phase D Scope)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIME ════════════════════════════════════════════════════════════════════► │
│                                                                             │
│       PHASE I (quantum/)              PHASE II (sifting/)                   │
│  ┌─────────────────────────┐     ┌───────────────────────────────┐          │
│  │ 1. EPR Generation       │     │ 4. Bob: Commit to detections  │          │
│  │ 2. Basis Selection      │────►│ 5. TIMING BARRIER (Δt wait)   │          │
│  │ 3. Measurement          │     │ 6. Alice: Reveal bases        │          │
│  └─────────────────────────┘     │ 7. Sifting (I₀/I₁ partition)  │          │
│                                  │ 8. Test sampling & QBER       │          │
│                                  └───────────────────────────────┘          │
│                                                   │                         │
│                                                   ▼                         │
│                                         [Phase III: Reconciliation]         │
│                                                   │                         │
│                                                   ▼                         │
│                                  ┌───────────────────────────────┐          │
│                                  │    PHASE IV (amplification/)  │          │
│                                  │ 9. Entropy calculation        │          │
│                                  │ 10. Key length determination  │          │
│                                  │ 11. Toeplitz hashing          │          │
│                                  │ 12. OT output formatting      │          │
│                                  └───────────────────────────────┘          │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  SECURITY INVARIANT: Steps 4-6 enforce NSM temporal constraint              │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Scope & Deliverables

### 2.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `caligo/quantum/epr.py` | EPR pair generation via SquidASM |
| `caligo/quantum/basis.py` | Random basis selection |
| `caligo/quantum/measurement.py` | Measurement buffering |
| `caligo/quantum/batching.py` | Memory-constrained batch management |
| `caligo/sifting/commitment.py` | SHA256 commitment scheme |
| `caligo/sifting/sifter.py` | I₀/I₁ partitioning, basis matching |
| `caligo/sifting/qber.py` | QBER estimation with μ penalty |
| `caligo/sifting/detection_validator.py` | Chernoff bounds for loss validation |
| `caligo/amplification/entropy.py` | NSM entropy calculation |
| `caligo/amplification/key_length.py` | Secure key length determination |
| `caligo/amplification/toeplitz.py` | Toeplitz matrix hashing |
| `caligo/amplification/formatter.py` | OT output: S₀, S₁, Sᴄ |
| Phase contract tests | Boundary validation tests |

### 2.2 Out of Scope

| Item | Rationale |
|------|-----------|
| `reconciliation/` package | Separate extraction effort; LDPC complexity |
| SquidASM program orchestration | Phase E (`protocol/alice.py`, `protocol/bob.py`) |
| Network topology configuration | Phase B (`simulation/network_builder.py`) |

### 2.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Phase A-C | — | Foundation types, simulation, security |
| SquidASM | ≥0.12 | `EPRSocket`, `QubitMeasureBasis` |
| NetQASM | ≥0.15 | Quantum operations |
| NumPy | ≥1.21 | Array operations |
| SciPy | ≥1.7 | FFT for Toeplitz |

---

## 3. Theoretical Workflow Overview

### 3.1 The E-HOK Security Model

E-HOK implements **1-out-of-2 Oblivious Transfer (OT)** using the Noisy Storage Model (NSM). The fundamental security assumption is:

> **NSM Assumption:** An adversary's quantum memory decoheres faster than the protocol's timing constraints allow exploitation.

This differs from QKD where Alice and Bob trust each other against an external eavesdropper Eve. In E-HOK:

- **Alice** holds two messages (keys) $S_0$ and $S_1$
- **Bob** has a choice bit $C \in \{0, 1\}$
- **Goal:** Bob learns $S_C$ while Alice learns nothing about $C$, and Bob learns nothing about $S_{1-C}$

**Source:** Lemus et al. (2020), Section 3; König et al. (2012), Definition I.1

### 3.2 The "Strictly Less" Security Condition

**Source:** Schaffner et al. (2009), Corollary 7; Lupo et al. (2023), Section VI

For security to hold:

$$
h(P_{error}) < h_{min}(r)
$$

Where:
- $P_{error}$ — Trusted noise (channel + device errors)
- $h_{min}(r)$ — Min-entropy from adversary's storage decoherence
- $r$ — Adversary's storage noise parameter

**Implementation Impact:** Phase D modules must track and validate this condition at every stage.

### 3.3 The Timing Barrier Requirement

**Source:** König et al. (2012), Theorem I.1; Schaffner et al. (2009), Protocol 2

The protocol enforces temporal ordering:

1. **Quantum transmission** completes
2. **Bob commits** to detection report (without knowing bases)
3. **Wait time Δt** elapses (storage decoherence)
4. **Alice reveals** basis choices

If step 4 occurs before step 3, a dishonest Bob can perform a coherent attack.

**Implementation:** `sifting/` package integrates with `TimingBarrier` from Phase B.

---

## 4. Package: `quantum/` — Phase I Implementation

Phase I generates the correlated raw material via quantum entanglement. This package implements EPR pair distribution, basis selection, and measurement buffering.

### 4.1 Theoretical Foundation

**Source:** König et al. (2012), Section II; Lemus et al. (2020), Protocol Description

Phase I implements the **quantum transmission** step of the WSE (Weak String Erasure) primitive:

1. **Alice** and **Bob** share EPR pairs $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
2. Each party independently selects a random basis $a_i, b_i \in \{Z, X\}$
3. Each party measures their qubit in the chosen basis, obtaining outcomes $s_i, \bar{s}_i$

**Correlation Property:** When $a_i = b_i$, the outcomes are perfectly correlated: $s_i = \bar{s}_i$ (in the absence of noise).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EPR MEASUREMENT CORRELATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        ALICE                                   BOB                          │
│     ┌─────────┐                           ┌─────────┐                       │
│     │  a_i    │  Basis choice             │   b_i   │  Basis choice         │
│     │ (Z/X)   │                           │  (Z/X)  │                       │
│     └────┬────┘                           └────┬────┘                       │
│          │                                     │                            │
│          ▼                                     ▼                            │
│     ┌─────────┐      EPR Pair             ┌─────────┐                       │
│     │ Measure │◄────|Φ⁺⟩─────────────────►│ Measure  │                      │
│     │ qubit A │                           │ qubit B │                       │
│     └────┬────┘                           └────┬────┘                       │
│          │                                     │                            │
│          ▼                                     ▼                            │
│     ┌─────────┐                           ┌─────────┐                       │
│     │  s_i    │  Outcome                  │  s̄_i    │  Outcome              │
│     │ (0/1)   │                           │  (0/1)  │                       │
│     └─────────┘                           └─────────┘                       │
│                                                                             │
│   If a_i = b_i:  s_i = s̄_i  (perfect correlation)                           │
│   If a_i ≠ b_i:  s_i, s̄_i   (independent, random)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module: `epr.py` (~150 LOC)

**Purpose:** EPR pair generation via SquidASM's `EPRSocket` interface.

#### 4.2.1 Class: `EPRGenerator`

```python
class EPRGenerator:
    """
    Generate EPR pairs using SquidASM EPRSocket.
    
    This class wraps SquidASM's EPR socket operations with Caligo's
    timing infrastructure for NSM compliance.
    
    Attributes
    ----------
    epr_socket : EPRSocket
        SquidASM EPR socket for entanglement generation.
    role : Literal["alice", "bob"]
        Party role determining create/receive behavior.
    timing_barrier : TimingBarrier
        Phase B timing infrastructure for Δt enforcement.
    detection_tracker : DetectionEventTracker
        Phase B detection tracking for loss validation.
    
    SquidASM Integration
    --------------------
    - Alice calls `epr_socket.create_keep(number=n)` → receives n Qubit objects
    - Bob calls `epr_socket.recv_keep(number=n)` → receives n Qubit objects
    - Both parties retain qubits in quantum memory for basis-dependent measurement
    
    References
    ----------
    - SquidASM EPRSocket API: squidasm/sim/stack/common.py
    - NetQASM Qubit API: netqasm/sdk/qubit.py
    """
    
    def __init__(
        self,
        epr_socket: "EPRSocket",
        role: Literal["alice", "bob"],
        timing_barrier: "TimingBarrier",
        detection_tracker: "DetectionEventTracker"
    ):
        """
        Initialize EPR generator.
        
        Parameters
        ----------
        epr_socket : EPRSocket
            SquidASM socket for EPR pair distribution.
        role : Literal["alice", "bob"]
            Party role. Alice creates, Bob receives.
        timing_barrier : TimingBarrier
            For marking quantum phase completion time.
        detection_tracker : DetectionEventTracker
            For recording detection events.
        """
```

#### 4.2.2 Method: `generate_batch()`

```python
def generate_batch(
    self,
    batch_size: int
) -> Generator[EventExpression, None, list["Qubit"]]:
    """
    Generate a batch of EPR pairs.
    
    Generator-compatible method for SquidASM's async execution model.
    
    Parameters
    ----------
    batch_size : int
        Number of EPR pairs to generate in this batch.
    
    Yields
    ------
    EventExpression
        Control back to SquidASM scheduler for quantum operations.
    
    Returns
    -------
    list[Qubit]
        List of local qubit handles from EPR pairs.
    
    SquidASM Execution Model
    ------------------------
    This method is a generator that yields control to SquidASM:
    
        qubits = yield from epr_generator.generate_batch(100)
    
    The `yield from` passes control to the SquidASM scheduler,
    which executes the quantum operations via NetSquid simulation.
    
    Role-Dependent Behavior
    -----------------------
    - Alice (creator): `epr_socket.create_keep(number=batch_size)`
    - Bob (receiver): `epr_socket.recv_keep(number=batch_size)`
    
    Notes
    -----
    After this method returns, qubits are in quantum memory but
    NOT yet measured. The caller must subsequently measure with
    `measure_batch()`.
    """
```

#### 4.2.3 Integration with Phase B

```python
def mark_quantum_phase_complete(self) -> None:
    """
    Record quantum transmission completion timestamp.
    
    CRITICAL: This method MUST be called after all EPR pairs
    are generated and measured. It marks the start of the Δt
    wait period for NSM security.
    
    Integration
    -----------
    Calls `self.timing_barrier.mark_quantum_complete()` which
    records `ns.sim_time()` as the reference point for the
    timing constraint.
    
    Security Requirement
    --------------------
    Basis revelation (Phase II) MUST NOT occur until Δt has
    elapsed after this timestamp.
    """
```

### 4.3 Module: `basis.py` (~80 LOC)

**Purpose:** Cryptographically secure random basis selection.

#### 4.3.1 Class: `BasisSelector`

```python
class BasisSelector:
    """
    Generate random measurement bases for quantum measurements.
    
    Uses cryptographically secure randomness for basis selection.
    The randomness of basis choices is CRITICAL for E-HOK security:
    predictable bases would allow an adversary to gain full
    information about the key.
    
    Basis Encoding
    --------------
    - 0: Z-basis (computational basis, |0⟩/|1⟩)
    - 1: X-basis (Hadamard basis, |+⟩/|-⟩)
    
    Security Requirement
    --------------------
    P(basis[i] = 0) = P(basis[i] = 1) = 0.5
    Basis choices must be independent and uniformly random.
    
    References
    ----------
    - BB84 basis selection: Bennett & Brassard (1984)
    - E-HOK basis usage: Lemus et al. (2020), Section 3
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize basis selector.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility (TESTING ONLY).
            In production, MUST be None for true randomness.
        
        Implementation Note
        -------------------
        Uses numpy.random.default_rng() with PCG64 generator.
        For production cryptographic use, consider secrets module.
        """
    
    def generate_bases(self, count: int) -> np.ndarray:
        """
        Generate random basis choices.
        
        Parameters
        ----------
        count : int
            Number of basis choices to generate.
        
        Returns
        -------
        bases : np.ndarray
            Array of basis choices, shape (count,), dtype uint8.
            Values are 0 (Z-basis) or 1 (X-basis).
        
        Mathematical Definition
        -----------------------
        For each i ∈ [0, count):
            bases[i] ← Uniform({0, 1})
        """
```

### 4.4 Module: `measurement.py` (~100 LOC)

**Purpose:** Measurement execution and result buffering.

#### 4.4.1 Class: `MeasurementExecutor`

```python
class MeasurementExecutor:
    """
    Execute measurements on qubits with basis-dependent rotations.
    
    NetQASM Measurement Model
    -------------------------
    NetQASM's `Qubit.measure()` always measures in the Z-basis.
    To measure in the X-basis, we first apply a Hadamard gate:
    
        Z-basis: measure directly
        X-basis: apply H, then measure
    
    The measurement result is a NetQASM Future that resolves
    after `connection.flush()` is called.
    """
    
    def measure_batch(
        self,
        qubits: list["Qubit"],
        bases: np.ndarray
    ) -> list["Future"]:
        """
        Measure qubits in specified bases.
        
        Parameters
        ----------
        qubits : list[Qubit]
            Qubits to measure.
        bases : np.ndarray
            Basis for each qubit (0=Z, 1=X).
        
        Returns
        -------
        futures : list[Future]
            Measurement outcome futures. Values available after flush().
        
        Implementation
        --------------
        For each (qubit, basis) pair:
            if basis == 1:  # X-basis
                qubit.H()   # Apply Hadamard
            outcome = qubit.measure()
            futures.append(outcome)
        
        NetQASM Gate Application
        ------------------------
        The H gate transforms:
            |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2
            |1⟩ → |-⟩ = (|0⟩ - |1⟩)/√2
        
        So measuring in Z after H is equivalent to measuring in X.
        """
```

#### 4.4.2 Class: `MeasurementBuffer`

```python
class MeasurementBuffer:
    """
    Buffer for storing measurement results across batches.
    
    Accumulates measurement records from multiple batches until
    the full quantum phase is complete.
    
    Attributes
    ----------
    records : list[MeasurementRecord]
        Accumulated measurement records.
    
    Thread Safety
    -------------
    This class is NOT thread-safe. In Caligo's simulation model,
    each party has its own buffer instance.
    """
    
    def add_batch(
        self,
        outcomes: np.ndarray,
        bases: np.ndarray,
        timestamps: np.ndarray
    ) -> None:
        """
        Add a batch of measurement results.
        
        Parameters
        ----------
        outcomes : np.ndarray
            Measurement outcomes (0 or 1), shape (batch_size,).
        bases : np.ndarray
            Measurement bases (0=Z, 1=X), shape (batch_size,).
        timestamps : np.ndarray
            Simulation timestamps (ns), shape (batch_size,).
        
        Validation
        ----------
        Arrays must have identical lengths.
        Outcomes must be binary (0 or 1).
        Bases must be binary (0 or 1).
        """
    
    def to_phase_result(self) -> "QuantumPhaseResult":
        """
        Convert buffer contents to Phase I output contract.
        
        Returns
        -------
        QuantumPhaseResult
            Phase A type for Phase I → Phase II boundary.
        """
```

### 4.5 Module: `batching.py` (~120 LOC)

**Purpose:** Memory-constrained batch management for large EPR counts.

#### 4.5.1 The Batching Problem

**Source:** ehok/quantum/batching_manager.py; SquidASM memory constraints

Quantum simulators have memory constraints:
- NetSquid node memory is finite (typically 1-100 qubits)
- Large protocols require 10⁴ - 10⁶ EPR pairs
- Solution: Process in batches, measure before generating next batch

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BATCHING WORKFLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Total pairs: N = 100,000                                                   │
│  Batch size:  B = 100                                                       │
│  Num batches: ⌈N/B⌉ = 1,000                                                 │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ for batch_idx in range(num_batches):                                 │   │
│  │     qubits = yield from epr_generator.generate_batch(batch_size)     │   │
│  │     futures = measurement_executor.measure_batch(qubits, bases)      │   │
│  │     yield from connection.flush()  # Execute quantum ops             │   │
│  │     outcomes = extract_outcomes(futures)                             │   │
│  │     buffer.add_batch(outcomes, bases, timestamps)                    │   │
│  │     # Qubits now measured → memory freed for next batch              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.5.2 Class: `BatchingManager`

```python
class BatchingManager:
    """
    Manage streaming EPR generation with memory constraints.
    
    Coordinates batch-by-batch EPR pair generation to overcome
    quantum memory limitations in NetSquid simulation.
    
    Attributes
    ----------
    total_pairs : int
        Total number of EPR pairs to generate.
    batch_size : int
        Maximum pairs per batch (memory constraint).
    num_batches : int
        Number of batches needed: ⌈total_pairs / batch_size⌉.
    
    Configuration
    -------------
    Typical values from ehok:
    - total_pairs: 10,000 - 1,000,000
    - batch_size: 100 - 1,000 (depends on NetSquid memory config)
    """
    
    def __init__(
        self,
        total_pairs: int,
        batch_size: int
    ):
        """
        Initialize batching manager.
        
        Parameters
        ----------
        total_pairs : int
            Total EPR pairs for the protocol.
        batch_size : int
            Maximum pairs per batch.
        
        Raises
        ------
        ValueError
            If total_pairs < 1 or batch_size < 1.
        """
    
    def compute_batch_sizes(self) -> list[int]:
        """
        Compute the size of each batch.
        
        Returns
        -------
        sizes : list[int]
            Size of each batch. Last batch may be smaller if
            total_pairs is not divisible by batch_size.
        
        Example
        -------
        >>> mgr = BatchingManager(total_pairs=250, batch_size=100)
        >>> mgr.compute_batch_sizes()
        [100, 100, 50]
        """
```

### 4.6 Phase I Output Contract

**Defined in:** `caligo/types/phase_contracts.py` (Phase A)

```python
@dataclass(frozen=True)
class QuantumPhaseResult:
    """
    Contract: Phase I → Phase II data transfer.
    
    Contains all data produced by the quantum generation phase
    needed for subsequent sifting and estimation.
    
    Attributes
    ----------
    measurement_outcomes : np.ndarray
        Measurement outcomes, shape (n,), dtype uint8.
        Values are 0 or 1.
    basis_choices : np.ndarray
        Basis choices, shape (n,), dtype uint8.
        Values are 0 (Z) or 1 (X).
    timestamps : np.ndarray
        Simulation timestamps (ns), shape (n,), dtype float64.
    detection_events : np.ndarray
        Detection success flags, shape (n,), dtype bool.
        True if EPR pair was successfully distributed.
    quantum_complete_time : float
        Simulation time when quantum phase completed (ns).
        Used as reference for timing barrier.
    
    Invariants
    ----------
    - All arrays have identical length n
    - measurement_outcomes[i] ∈ {0, 1}
    - basis_choices[i] ∈ {0, 1}
    - quantum_complete_time > 0
    
    Usage
    -----
    This contract is produced by quantum/ and consumed by sifting/.
    The timing barrier uses quantum_complete_time to enforce Δt.
    """
```

---

## 5. Package: `sifting/` — Phase II Implementation

Phase II is the "gatekeeper" that filters raw quantum data, implements the commit-reveal protocol, and estimates the channel error rate. This package enforces the critical temporal ordering required for NSM security.

### 5.1 Theoretical Foundation

**Source:** Schaffner et al. (2009), Protocol 2; Erven et al. (2014), Section III

Phase II implements three critical security mechanisms:

1. **Commit-then-Reveal:** Bob commits to detection events BEFORE learning Alice's bases
2. **Missing Rounds Validation:** Validate that Bob's reported losses match expected channel behavior
3. **QBER Estimation:** Measure error rate with finite-size statistical corrections

#### 5.1.1 The "Sandwich" Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE II: COMMIT-REVEAL PROTOCOL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ALICE                           TIMING                              BOB    │
│  ──────                                                             ─────   │
│                                                                             │
│  [Phase I complete]                                     [Phase I complete]  │
│         │                                                      │            │
│         │           ◄────── Commit(𝓜) ───────                 │            │
│         │           Bob sends detection report                 │            │
│         │           (which rounds he detected)                 │            │
│         ▼                                                      │            │
│  ┌──────────────┐                                              │            │
│  │ VALIDATE 𝓜  │  Chernoff bounds check                       │            │
│  │ against η    │  (expected transmittance)                    │            │
│  └──────────────┘                                              │            │
│         │                                                      │            │
│         │           ═══════════════════════════                │            │
│         │           ║   TIMING BARRIER Δt    ║                 │            │
│         │           ║   (NSM security wait)  ║                 │            │
│         │           ═══════════════════════════                │            │
│         │                                                      │            │
│         │           ─────── Reveal(a) ──────►                  │            │
│         │           Alice sends basis string                   │            │
│         │                                                      ▼            │
│         │                                              ┌──────────────┐     │
│         │                                              │   SIFTING    │     │
│         │                                              │ Compute I₀,I₁│     │
│         │                                              └──────────────┘     │
│         │                                                      │            │
│         │           ◄───── Test values ──────                  │            │
│         │           Bob reveals subset for QBER                │            │
│         ▼                                                      │            │
│  ┌──────────────┐                                              │            │
│  │ ESTIMATE QBER│  With finite-size penalty μ                  │            │
│  │ CHECK < 22%  │                                              │            │
│  └──────────────┘                                              │            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Critical Security Invariant:** Bob MUST commit to $\mathcal{M}$ (detection events) BEFORE receiving basis string $a$. The timing barrier Δt ensures adversarial quantum storage has decohered.

#### 5.1.2 The Missing Rounds Constraint

**Source:** Schaffner et al. (2009), Section 4.2; Erven et al. (2014), Eq. (1)

A cheating Bob could claim "loss" only on rounds where his noisy storage failed, post-selecting a lower-noise key. Alice validates using **Hoeffding's inequality**:

$$
\Pr[|S - P_{expected} \cdot M| \geq \zeta \cdot M] < \varepsilon
$$

Where:
- $S$ — Number of detected rounds (reported by Bob)
- $M$ — Total rounds
- $P_{expected}$ — Expected detection probability from channel calibration
- $\zeta = \sqrt{\ln(2/\varepsilon) / (2M)}$ — Statistical tolerance

#### 5.1.3 The Finite-Size Penalty μ

**Source:** Erven et al. (2014), Eq. (2); Tomamichel et al. (2012)

The observed QBER on test set $k$ may differ from the true error rate on key set $n$. The penalty μ bounds this statistical fluctuation:

$$
\mu = \sqrt{\frac{n + k}{n \cdot k} \cdot \frac{k + 1}{k}} \cdot \ln\frac{4}{\varepsilon_{sec}}
$$

The adjusted QBER for security calculations is: $Q_{adj} = Q_{obs} + \mu$

### 5.2 Module: `commitment.py` (~120 LOC)

**Purpose:** SHA256-based commitment scheme for detection report hiding.

#### 5.2.1 Class: `SHA256Commitment`

```python
class SHA256Commitment:
    """
    SHA256-based cryptographic commitment scheme.
    
    Used in Phase II to commit Bob to his detection report before
    Alice reveals the basis string. This enforces the temporal
    ordering critical for NSM security.
    
    Commitment Scheme
    -----------------
    Commit: C = SHA256(salt || data)
    Open:   Reveal (data, salt), verify C = SHA256(salt || data)
    
    Security Properties
    -------------------
    - Hiding: C reveals nothing about data (hash preimage resistance)
    - Binding: Cannot find data' ≠ data with same C (collision resistance)
    
    Trade-offs
    ----------
    - Compact commitment: 32 bytes regardless of data size
    - Opening requires full data revelation
    
    References
    ----------
    - Lemus et al. (2020): Hybrid commitment approach
    """
    
    def __init__(self, salt_length: int = 32):
        """
        Initialize commitment scheme.
        
        Parameters
        ----------
        salt_length : int
            Length of random salt in bytes. Default: 32.
            Longer salts provide additional security margin.
        """
    
    def commit(self, data: np.ndarray) -> tuple[bytes, bytes]:
        """
        Generate commitment to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to commit to (detection report as bitstring).
        
        Returns
        -------
        commitment : bytes
            32-byte SHA256 hash C = H(salt || data).
        salt : bytes
            Random salt used (decommitment info).
        
        Security Note
        -------------
        Salt MUST be generated using cryptographically secure
        randomness (secrets.token_bytes).
        """
    
    def verify(
        self,
        commitment: bytes,
        data: np.ndarray,
        salt: bytes
    ) -> bool:
        """
        Verify that data matches commitment.
        
        Parameters
        ----------
        commitment : bytes
            The commitment hash to verify against.
        data : np.ndarray
            Claimed data.
        salt : bytes
            Claimed salt from commitment phase.
        
        Returns
        -------
        valid : bool
            True if SHA256(salt || data) == commitment.
        
        Raises
        ------
        CommitmentVerificationError
            If verification fails (optional, can return False).
        """
```

### 5.3 Module: `sifter.py` (~150 LOC)

**Purpose:** Basis matching and I₀/I₁ partition computation.

#### 5.3.1 The Sifting Process

After Alice reveals bases, Bob computes the key partition:

- **I₀ (Matching bases):** Indices where $a_i = b_i$ → Sifted key candidates
- **I₁ (Mismatching bases):** Indices where $a_i \neq b_i$ → Used for oblivious transfer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIFTING PARTITION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Alice's bases:  a = [ Z  X  Z  X  Z  X  Z  X  Z  X ]                       │
│  Bob's bases:    b = [ Z  Z  X  X  Z  X  Z  Z  Z  X ]                       │
│                        │  │  │  │  │  │  │  │  │  │                         │
│  Match status:         ✓  ✗  ✗  ✓  ✓  ✓  ✓  ✗  ✓  ✓                     │
│                                                                             │
│  I₀ (match):     {0, 3, 4, 5, 6, 8, 9}  →  Sifted key                       │
│  I₁ (mismatch):  {1, 2, 7}              →  Random bits (oblivious)          │
│                                                                             │
│  Expected: |I₀| ≈ |I₁| ≈ n/2 (uniform random bases)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Class: `Sifter`

```python
class Sifter:
    """
    Perform basis sifting to partition indices into I₀ and I₁.
    
    This class implements the core sifting logic that identifies
    matching and mismatching basis indices after basis revelation.
    
    OT Structure
    ------------
    In 1-out-of-2 OT:
    - I₀ indices: Bob measured in correct basis → knows these bits
    - I₁ indices: Bob measured in wrong basis → random values
    
    Alice generates keys from BOTH partitions:
        S₀ = Hash(bits at I₀ where a_i = 0)
        S₁ = Hash(bits at I₀ where a_i = 1)
    
    Bob obtains ONE key determined by his choice bit C (implicit
    in his basis selection pattern).
    """
    
    @staticmethod
    def identify_matching_bases(
        bases_alice: np.ndarray,
        bases_bob: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Partition indices by basis match status.
        
        Parameters
        ----------
        bases_alice : np.ndarray
            Alice's basis choices, shape (n,).
        bases_bob : np.ndarray
            Bob's basis choices, shape (n,).
        
        Returns
        -------
        I_0 : np.ndarray
            Indices where bases match.
        I_1 : np.ndarray
            Indices where bases mismatch.
        
        Mathematical Definition
        -----------------------
        I₀ = {i : a_i = b_i}
        I₁ = {i : a_i ≠ b_i}
        
        Note: I₀ ∪ I₁ = {0, ..., n-1} and I₀ ∩ I₁ = ∅
        """
    
    @staticmethod
    def select_test_set(
        I_0: np.ndarray,
        fraction: float = 0.1,
        seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select random subset of I₀ for QBER estimation.
        
        Parameters
        ----------
        I_0 : np.ndarray
            Indices where bases match.
        fraction : float
            Fraction of I₀ to use for testing. Default: 0.1 (10%).
        seed : int, optional
            Random seed. If None, derives deterministic seed from I₀
            to ensure both parties select same test set.
        
        Returns
        -------
        test_set : np.ndarray
            Indices selected for QBER testing (T ⊂ I₀).
        key_set : np.ndarray
            Remaining indices for key extraction (I₀ \\ T).
        
        Security Consideration
        ----------------------
        Test set selection must be:
        1. Random (prevents adversary from targeting test bits)
        2. Agreed upon by both parties (deterministic from shared data)
        """
```

### 5.4 Module: `qber.py` (~120 LOC)

**Purpose:** QBER estimation with finite-size statistical corrections.

#### 5.4.1 Class: `QBEREstimator`

```python
class QBEREstimator:
    """
    Estimate Quantum Bit Error Rate with finite-size corrections.
    
    This class computes the observed QBER and applies the statistical
    penalty μ to obtain a security-conservative estimate.
    
    QBER Definition
    ---------------
    QBER = (# of errors in test set) / (size of test set)
    
    Where an error occurs when Alice's outcome ≠ Bob's outcome
    on a matching-basis index.
    
    Finite-Size Correction
    ----------------------
    The observed QBER on test set k may differ from the true error
    rate on key set n. We use:
    
        Q_adjusted = Q_observed + μ
    
    where μ is the statistical penalty from FiniteSizePenalty.
    
    References
    ----------
    - Erven et al. (2014), Eq. (2)
    - Tomamichel et al. (2012): Tight finite-key analysis
    """
    
    def __init__(self, epsilon_sec: float = 1e-10):
        """
        Initialize QBER estimator.
        
        Parameters
        ----------
        epsilon_sec : float
            Security parameter for finite-size penalty.
        """
    
    def estimate(
        self,
        outcomes_alice: np.ndarray,
        outcomes_bob: np.ndarray,
        test_indices: np.ndarray
    ) -> float:
        """
        Estimate raw QBER on test set.
        
        Parameters
        ----------
        outcomes_alice : np.ndarray
            Alice's measurement outcomes.
        outcomes_bob : np.ndarray
            Bob's measurement outcomes.
        test_indices : np.ndarray
            Indices to use for estimation (test set T).
        
        Returns
        -------
        qber : float
            Raw observed QBER ∈ [0, 1].
        """
    
    def compute_penalty_mu(
        self,
        test_size_k: int,
        key_size_n: int
    ) -> float:
        """
        Compute finite-size statistical penalty μ.
        
        Formula (Erven et al., 2014):
        
            μ = √[(n+k)/(n·k) · (k+1)/k] · ln(4/ε_sec)
        
        Parameters
        ----------
        test_size_k : int
            Size of test set.
        key_size_n : int
            Size of key set.
        
        Returns
        -------
        mu : float
            Statistical penalty to add to observed QBER.
        
        Notes
        -----
        μ scales as O(1/√k), so larger test sets give smaller penalties
        but leave fewer bits for the final key.
        """
    
    def check_threshold(
        self,
        adjusted_qber: float,
        warning_threshold: float = 0.11,
        hard_limit: float = 0.22
    ) -> "QBERValidationResult":
        """
        Check QBER against security thresholds.
        
        Parameters
        ----------
        adjusted_qber : float
            QBER with finite-size penalty applied.
        warning_threshold : float
            Conservative threshold (Schaffner Corollary 7). Default: 11%.
        hard_limit : float
            Absolute limit (Lupo Section VI). Default: 22%.
        
        Returns
        -------
        result : QBERValidationResult
            Contains status (PASSED/WARNING/FAILED) and diagnostic info.
        
        Abort Logic
        -----------
        - adjusted_qber > hard_limit: ABORT (security impossible)
        - adjusted_qber > warning_threshold: WARNING (marginal security)
        - otherwise: PASSED
        """
```

### 5.5 Module: `detection_validator.py` (~150 LOC)

**Purpose:** Validate Bob's detection report against expected channel behavior.

#### 5.5.1 Class: `DetectionValidator`

```python
class DetectionValidator:
    """
    Validate detection reports using Hoeffding bounds.
    
    This class implements the statistical test to detect post-selection
    attacks where Bob claims "missing rounds" strategically to hide
    storage decoherence.
    
    Attack Scenario
    ---------------
    Cheating Bob stores qubits but some decohere. He could:
    1. Claim those rounds as "missing" (blamed on channel loss)
    2. Post-select only rounds where storage succeeded
    3. Effectively reduce his apparent storage noise
    
    Defense
    -------
    Alice validates that reported detections S satisfy:
        |S - P_expected · M| < ζ · M
    
    where ζ = √(ln(2/ε) / (2M)) from Hoeffding's inequality.
    
    Attributes
    ----------
    expected_detection_prob : float
        Expected detection probability from channel calibration.
    failure_probability : float
        Failure probability ε for the statistical test.
    
    References
    ----------
    - Schaffner et al. (2009), Section 4.2
    """
    
    def __init__(
        self,
        expected_detection_prob: float,
        failure_probability: float = 1e-10
    ):
        """
        Initialize detection validator.
        
        Parameters
        ----------
        expected_detection_prob : float
            Expected P_det from channel characterization. Must be in (0, 1).
        failure_probability : float
            Failure probability budget ε. Default: 1e-10.
        """
    
    def compute_tolerance(self, total_rounds: int) -> float:
        """
        Compute Hoeffding tolerance ζ.
        
        Formula: ζ = √(ln(2/ε) / (2M))
        
        Parameters
        ----------
        total_rounds : int
            Total number of rounds M.
        
        Returns
        -------
        zeta : float
            Statistical tolerance ζ.
        """
    
    def compute_acceptance_interval(
        self,
        total_rounds: int
    ) -> tuple[int, int]:
        """
        Compute acceptance interval for detection count.
        
        Parameters
        ----------
        total_rounds : int
            Total number of rounds M.
        
        Returns
        -------
        lower : int
            Minimum acceptable detections: ⌈(P - ζ) · M⌉
        upper : int
            Maximum acceptable detections: ⌊(P + ζ) · M⌋
        """
    
    def validate(
        self,
        detected_count: int,
        total_rounds: int
    ) -> "DetectionValidationResult":
        """
        Validate detection report.
        
        Parameters
        ----------
        detected_count : int
            Number of rounds Bob reports as detected (S).
        total_rounds : int
            Total number of rounds (M).
        
        Returns
        -------
        result : DetectionValidationResult
            Contains status and diagnostic information.
        
        Abort Logic
        -----------
        If S falls outside [lower, upper]: ABORT with code ABORT-II-DET-001
        """
```

### 5.6 Phase II Output Contract

**Defined in:** `caligo/types/phase_contracts.py` (Phase A)

```python
@dataclass(frozen=True)
class SiftingPhaseResult:
    """
    Contract: Phase II → Phase III data transfer.
    
    Contains all data produced by sifting phase needed for
    subsequent reconciliation and amplification.
    
    Attributes
    ----------
    I_0 : np.ndarray
        Indices where bases matched (sifted key candidates).
    I_1 : np.ndarray
        Indices where bases mismatched (oblivious partition).
    test_indices : np.ndarray
        Indices used for QBER estimation (T ⊂ I₀).
    key_indices : np.ndarray
        Indices for key extraction (I₀ \\ T).
    observed_qber : float
        Raw QBER observed on test set.
    adjusted_qber : float
        QBER with finite-size penalty μ applied.
    penalty_mu : float
        The μ value used for adjustment.
    detection_validation_passed : bool
        Whether detection report passed Hoeffding check.
    timing_barrier_satisfied : bool
        Whether Δt elapsed before basis revelation.
    
    Invariants
    ----------
    - test_indices ⊂ I_0
    - key_indices = I_0 \\ test_indices
    - adjusted_qber = observed_qber + penalty_mu
    - adjusted_qber < 0.22 (otherwise abort)
    """
```

---

## 6. Package: `amplification/` — Phase IV Implementation

Phase IV is the "distillation" process that compresses the reconciled key into a final secret key, eliminating residual adversarial information. This package implements the NSM-specific entropy bounds and Toeplitz hashing.

### 6.1 Theoretical Foundation

**Source:** Lupo et al. (2023), Section V; Schaffner et al. (2009), Theorem 6

Privacy amplification extracts a shorter, secure key from a longer, partially compromised key:

$$
\ell = n \cdot h_{min}(r) - |\Sigma| - 2\log_2(1/\varepsilon_{sec}) - \Delta_{finite}
$$

Where:
- $n$ — Reconciled key length
- $h_{min}(r)$ — Min-entropy rate from NSM bounds (bits per raw bit)
- $|\Sigma|$ — Information leaked during reconciliation (wiretap cost)
- $\varepsilon_{sec}$ — Target security parameter
- $\Delta_{finite}$ — Finite-key statistical correction

#### 6.1.1 The "Max Bound" (Optimal Entropy)

**Source:** Lupo et al. (2023), Eq. (36)

The secure bit rate uses the maximum of two bounds:

$$
h_{min}(r) \geq \max\left\{ \Gamma[1 - \log_2(1 + 3r^2)], 1 - r \right\}
$$

- **Collision entropy bound (Dupuis-König):** $\Gamma[1 - \log_2(1 + 3r^2)]$
  - Better for high-noise storage (small $r$)
- **Virtual erasure bound (Lupo):** $1 - r$
  - Better for low-noise storage (large $r$)
- **Crossover point:** $r \approx 0.82$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NSM ENTROPY BOUNDS COMPARISON                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  h_min                                                                      │
│    1.0 ┤                                                                    │
│        │ ╲                                                                  │
│    0.8 ┤  ╲    Collision bound (Γ[1-log(1+3r²)])                            │
│        │   ╲                                                                │
│    0.6 ┤    ╲                                                               │
│        │     ╲────╮                                                         │
│    0.4 ┤          ╲   Virtual erasure bound (1-r)                           │
│        │           ╲                                                        │
│    0.2 ┤            ╲─────────                                              │
│        │                     ╲                                              │
│    0.0 ┤──────────────────────┴─────────►  r (storage noise)                │
│        0.0      0.25     0.5      0.75     1.0                              │
│                              ↑                                              │
│                         r ≈ 0.82 (crossover)                                │
│                                                                             │
│  Max Bound: Select the HIGHER curve at each r value                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.1.2 The Γ Function

**Source:** Lupo et al. (2023), Eq. (24)-(25)

The Γ function regularizes collision entropy to min-entropy:

$$
\Gamma(x) = \begin{cases}
x & \text{if } x \geq 1/2 \\
g^{-1}(x) & \text{if } x < 1/2
\end{cases}
$$

Where $g(y) = -y\log_2(y) - (1-y)\log_2(1-y) + y - 1 = h(y) + y - 1$

**Implementation:** Requires numerical root-finding for $g^{-1}(x)$ via Newton-Raphson.

#### 6.1.3 The OT Output Structure

**Source:** Lemus et al. (2020), Section 3; König et al. (2012)

After privacy amplification, the protocol produces the 1-out-of-2 OT structure:

- **Alice's output:** Two keys $S_0$ and $S_1$
- **Bob's output:** One key $S_C$ and choice bit $C$

The security guarantees:
- **Sender privacy:** Bob cannot learn $S_{1-C}$
- **Receiver privacy:** Alice cannot learn $C$
- **Correctness:** $S_C = $ Alice's $S_C$

### 6.2 Module: `entropy.py` (~120 LOC)

**Purpose:** NSM entropy bound calculations from Phase C security layer.

#### 6.2.1 Class: `NSMEntropyCalculator`

```python
class NSMEntropyCalculator:
    """
    Calculate min-entropy rate using NSM bounds.
    
    This class wraps the Phase C security bounds for use in
    privacy amplification key length calculation.
    
    Bounds Implemented
    ------------------
    1. Collision entropy bound: Γ[1 - log₂(1 + 3r²)]
    2. Virtual erasure bound: 1 - r
    3. Max Bound: max of (1) and (2)
    
    Attributes
    ----------
    storage_noise_r : float
        Adversary's storage noise parameter r ∈ [0, 1].
    
    References
    ----------
    - Lupo et al. (2023), Eq. (36)
    - Phase C specification: security/bounds.py
    """
    
    def __init__(self, storage_noise_r: float):
        """
        Initialize entropy calculator.
        
        Parameters
        ----------
        storage_noise_r : float
            Adversary's storage noise parameter.
            r = 0: Complete decoherence (maximum security)
            r = 1: Perfect storage (no security)
        
        Raises
        ------
        ValueError
            If r not in [0, 1].
        """
    
    def collision_entropy_rate(self) -> float:
        """
        Compute collision entropy rate h₂(σ).
        
        Formula: h₂ = 1 - log₂(1 + 3r²)
        
        Returns
        -------
        h2 : float
            Collision entropy rate.
        
        References
        ----------
        Lupo et al. (2023), Eq. (27)
        """
    
    def dupuis_konig_bound(self) -> float:
        """
        Compute Dupuis-König min-entropy bound.
        
        Formula: h_A = Γ[1 - log₂(1 + 3r²)]
        
        Returns
        -------
        h_A : float
            Min-entropy rate from collision entropy.
        
        Notes
        -----
        Better for high-noise storage (r small).
        """
    
    def virtual_erasure_bound(self) -> float:
        """
        Compute Lupo virtual erasure bound.
        
        Formula: h_B = 1 - r
        
        Returns
        -------
        h_B : float
            Min-entropy rate from virtual erasure.
        
        Notes
        -----
        Better for low-noise storage (r large).
        Physical intuition: (1-r) fraction of qubits are
        fully depolarized, contributing 1 bit of entropy each.
        """
    
    def max_bound_entropy_rate(self) -> float:
        """
        Compute optimal min-entropy rate (Max Bound).
        
        Formula: h_min = max{ Γ[1-log(1+3r²)], 1-r }
        
        Returns
        -------
        h_min : float
            Optimal min-entropy rate.
        bound_used : str
            Which bound dominated: "dupuis_konig" or "virtual_erasure"
        
        References
        ----------
        Lupo et al. (2023), Eq. (36)
        """
```

### 6.3 Module: `key_length.py` (~100 LOC)

**Purpose:** Determine secure key length from entropy and leakage.

#### 6.3.1 Class: `SecureKeyLengthCalculator`

```python
class SecureKeyLengthCalculator:
    """
    Calculate secure key length for privacy amplification.
    
    Implements the finite-key formula accounting for:
    - NSM min-entropy from storage decoherence
    - Wiretap cost (reconciliation leakage)
    - Security parameter penalty
    - Finite-key statistical corrections
    
    Formula
    -------
    ℓ = ⌊n · h_min(r) - |Σ| - 2·log₂(1/ε_sec) - Δ_finite⌋
    
    Where:
    - n: Reconciled key length
    - h_min(r): Min-entropy rate from Max Bound
    - |Σ|: Syndrome leakage (wiretap cost)
    - ε_sec: Security parameter
    - Δ_finite: Finite-key correction
    
    The "Death Valley" Problem
    --------------------------
    For small n, the penalties may exceed the entropy, yielding ℓ ≤ 0.
    In this case, secure key extraction is IMPOSSIBLE and the
    protocol must abort or accumulate more raw bits.
    
    References
    ----------
    - Erven et al. (2014), Eq. (8)
    - Phase IV analysis: Implementation Plan
    """
    
    def __init__(
        self,
        storage_noise_r: float,
        epsilon_sec: float = 1e-10,
        epsilon_cor: float = 1e-15
    ):
        """
        Initialize key length calculator.
        
        Parameters
        ----------
        storage_noise_r : float
            Adversary's storage noise parameter.
        epsilon_sec : float
            Security parameter. Default: 1e-10.
        epsilon_cor : float
            Correctness parameter. Default: 1e-15.
        """
    
    def compute_extractable_bits(
        self,
        reconciled_length: int
    ) -> int:
        """
        Compute raw extractable entropy (before deductions).
        
        Formula: n · h_min(r)
        
        Parameters
        ----------
        reconciled_length : int
            Number of reconciled bits n.
        
        Returns
        -------
        bits : int
            Total extractable entropy bits (floor).
        """
    
    def compute_security_penalty(self) -> float:
        """
        Compute security parameter deduction.
        
        Formula: 2 · log₂(1/ε_sec)
        
        Returns
        -------
        penalty : float
            Bits to subtract for security margin.
        
        Example
        -------
        For ε_sec = 1e-10: penalty ≈ 66.4 bits
        """
    
    def compute_final_length(
        self,
        reconciled_length: int,
        syndrome_leakage: int,
        finite_key_penalty: float = 0.0
    ) -> int:
        """
        Compute final secure key length.
        
        Parameters
        ----------
        reconciled_length : int
            Number of reconciled bits n.
        syndrome_leakage : int
            Bits leaked during reconciliation |Σ|.
        finite_key_penalty : float
            Additional finite-key correction Δ_finite.
        
        Returns
        -------
        length : int
            Secure key length ℓ. Returns 0 if computation yields ≤ 0.
        
        Raises
        ------
        SecurityError
            If ℓ ≤ 0 (Death Valley scenario) and strict mode enabled.
        """
    
    def check_feasibility(
        self,
        reconciled_length: int,
        syndrome_leakage: int
    ) -> "FeasibilityResult":
        """
        Check if positive key extraction is possible.
        
        Returns
        -------
        result : FeasibilityResult
            FEASIBLE if ℓ > 0, INFEASIBLE_* otherwise.
        """
```

### 6.4 Module: `toeplitz.py` (~150 LOC)

**Purpose:** Toeplitz matrix-based universal hashing.

#### 6.4.1 Toeplitz Hashing Background

**Source:** Carter & Wegman (1979); Tomamichel et al. (2012)

Toeplitz matrices form a **2-universal hash family**. A Toeplitz matrix T is defined by a single row and column:

```
T = | t₀   t_{-1}  t_{-2}  ...  t_{1-n} |
    | t₁   t₀      t_{-1}  ...  t_{2-n} |
    | t₂   t₁      t₀      ...  t_{3-n} |
    | ...                               |
    | t_{m-1}  ...         ...  t_{m-n} |
```

**Leftover Hash Lemma:** If $\ell \leq H_{min}^{\varepsilon}(X|E) - 2\log(1/\varepsilon_h)$, then hashing with Toeplitz produces key $\varepsilon_h$-close to uniform.

#### 6.4.2 Class: `ToeplitzHasher`

```python
class ToeplitzHasher:
    """
    Toeplitz matrix-based universal hashing for privacy amplification.
    
    This class implements privacy amplification using Toeplitz matrices,
    which form a 2-universal hash family with efficient FFT-based
    multiplication.
    
    Construction
    ------------
    A Toeplitz matrix T of size m×n is defined by a seed vector
    s of length (m + n - 1):
    
        T[i,j] = s[i - j + n - 1]
    
    The hash output is: y = T @ x mod 2
    
    Efficiency
    ----------
    Direct multiplication: O(m·n)
    FFT-based (large keys): O(n log n)
    
    Attributes
    ----------
    seed : np.ndarray
        Random seed defining the Toeplitz matrix.
    input_length : int
        Expected input key length (n).
    output_length : int
        Desired output key length (m).
    use_fft : bool
        Whether to use FFT-based multiplication for large keys.
    
    References
    ----------
    - Carter & Wegman (1979): Universal hash families
    """
    
    def __init__(
        self,
        input_length: int,
        output_length: int,
        seed: Optional[bytes] = None,
        use_fft: bool = False,
        fft_threshold: int = 10000
    ):
        """
        Initialize Toeplitz hasher.
        
        Parameters
        ----------
        input_length : int
            Length of input key (n).
        output_length : int
            Desired output length (m). Must be ≤ input_length.
        seed : bytes, optional
            Random seed. If None, generates cryptographically secure seed.
        use_fft : bool
            Use FFT multiplication for large keys. Default: False.
        fft_threshold : int
            Key length above which FFT is used. Default: 10000.
        
        Raises
        ------
        ValueError
            If output_length > input_length.
        """
    
    def generate_seed(self) -> np.ndarray:
        """
        Generate cryptographically secure random seed.
        
        Returns
        -------
        seed : np.ndarray
            Random bitstring of length (m + n - 1).
        
        Security Note
        -------------
        Uses secrets.token_bytes() for cryptographic randomness.
        The seed MUST be transmitted to Bob over authenticated channel
        AFTER the timing barrier.
        """
    
    def hash(self, input_key: np.ndarray) -> np.ndarray:
        """
        Apply Toeplitz hash to input key.
        
        Parameters
        ----------
        input_key : np.ndarray
            Input key bitstring of length n.
        
        Returns
        -------
        output_key : np.ndarray
            Hashed key of length m.
        
        Algorithm
        ---------
        output[i] = Σⱼ seed[i + n - 1 - j] · input[j] mod 2
        
        Implementation selects between direct and FFT methods
        based on key size.
        """
    
    def _hash_direct(self, input_key: np.ndarray) -> np.ndarray:
        """
        Direct O(m·n) Toeplitz-vector multiplication.
        
        Uses sliding window approach without constructing full matrix.
        """
    
    def _hash_fft(self, input_key: np.ndarray) -> np.ndarray:
        """
        FFT-based O(n log n) Toeplitz-vector multiplication.
        
        Embeds Toeplitz multiply in circulant multiply via FFT.
        Works over integers, takes mod 2 at end.
        """
```

### 6.5 Module: `formatter.py` (~100 LOC)

**Purpose:** Format privacy-amplified keys into OT output structure.

#### 6.5.1 Class: `OTOutputFormatter`

```python
class OTOutputFormatter:
    """
    Format privacy amplification output as 1-out-of-2 OT structure.
    
    Transforms the hashed reconciled key into the final OT output
    where Alice obtains (S₀, S₁) and Bob obtains (S_C, C).
    
    OT Partition
    ------------
    The partition is based on Alice's basis choices:
    
    - S₀: Hash of reconciled bits where Alice used basis 0 (Z)
    - S₁: Hash of reconciled bits where Alice used basis 1 (X)
    
    Bob obtains S_C where C is determined by his measurement choices.
    His "implicit choice bit" is the majority basis in his matching set.
    
    Security Guarantee
    ------------------
    After privacy amplification:
    - Bob cannot compute S_{1-C} (bounded by NSM entropy loss)
    - Alice cannot determine C (basis choices are random)
    """
    
    @staticmethod
    def compute_alice_keys(
        reconciled_key: np.ndarray,
        alice_bases: np.ndarray,
        key_indices: np.ndarray,
        hasher: ToeplitzHasher
    ) -> "AliceObliviousKey":
        """
        Compute Alice's OT output (both keys).
        
        Parameters
        ----------
        reconciled_key : np.ndarray
            Error-corrected key bits.
        alice_bases : np.ndarray
            Alice's basis choices (0=Z, 1=X).
        key_indices : np.ndarray
            Indices in the sifted key (post-test-sampling).
        hasher : ToeplitzHasher
            Initialized hasher for privacy amplification.
        
        Returns
        -------
        alice_keys : AliceObliviousKey
            Dataclass containing S₀, S₁, and metadata.
        
        Algorithm
        ---------
        1. Partition key_indices by alice_bases[i]:
           J₀ = {i ∈ key_indices : alice_bases[i] = 0}
           J₁ = {i ∈ key_indices : alice_bases[i] = 1}
        2. Extract bits: x₀ = reconciled_key[J₀], x₁ = reconciled_key[J₁]
        3. Hash: S₀ = hasher.hash(x₀), S₁ = hasher.hash(x₁)
        """
    
    @staticmethod
    def compute_bob_key(
        reconciled_key: np.ndarray,
        bob_bases: np.ndarray,
        alice_bases: np.ndarray,
        key_indices: np.ndarray,
        hasher: ToeplitzHasher
    ) -> "BobObliviousKey":
        """
        Compute Bob's OT output (single key and choice bit).
        
        Parameters
        ----------
        reconciled_key : np.ndarray
            Error-corrected key bits.
        bob_bases : np.ndarray
            Bob's basis choices.
        alice_bases : np.ndarray
            Alice's revealed basis choices.
        key_indices : np.ndarray
            Indices in the sifted key.
        hasher : ToeplitzHasher
            Initialized hasher.
        
        Returns
        -------
        bob_key : BobObliviousKey
            Dataclass containing S_C, choice bit C, and metadata.
        
        Choice Bit Determination
        ------------------------
        Bob's implicit choice C is determined by which of Alice's
        basis values (0 or 1) appears in Bob's matching indices.
        
        In practice, C = alice_bases[key_indices[0]] for the
        indices where Bob measured correctly.
        """
```

### 6.6 Phase IV Output Contract

**Defined in:** `caligo/types/phase_contracts.py` (Phase A)

```python
@dataclass(frozen=True)
class AmplificationResult:
    """
    Contract: Phase IV → Protocol completion.
    
    Contains the final OT output and comprehensive metrics.
    
    Attributes
    ----------
    alice_keys : AliceObliviousKey
        Alice's output: S₀, S₁, and metadata.
    bob_key : BobObliviousKey
        Bob's output: S_C, choice C, and metadata.
    metrics : ProtocolMetrics
        Comprehensive protocol execution statistics.
    
    Security Properties
    -------------------
    - len(alice_keys.key_0) == len(alice_keys.key_1) == final_length
    - len(bob_key.key_c) == final_length
    - bob_key.key_c == alice_keys.key_{bob_key.choice_bit}
    - Security parameter ε_sec bounds adversarial advantage
    
    Verification
    ------------
    For testing/debugging, correctness can be verified:
        assert np.array_equal(
            bob_key.key_c,
            alice_keys.key_0 if bob_key.choice_bit == 0 else alice_keys.key_1
        )
    """
```

---

## 7. Phase Contract Tests

This section specifies the contract-based testing strategy for Phase D modules. Tests validate both individual module correctness and inter-phase integration.

### 7.1 Testing Philosophy

**Contract-First Testing:**
1. Each module exposes a well-defined interface (input/output contract)
2. Tests verify contracts independent of implementation details
3. Boundary conditions and error cases receive explicit coverage
4. Property-based testing for mathematical invariants

### 7.2 Phase I Contract Tests (`quantum/`)

```python
# tests/caligo/phases/test_quantum_contracts.py

class TestEPRGeneratorContract:
    """Contract tests for EPR generation."""
    
    def test_batch_size_honored(self, epr_generator, batch_size):
        """Generator yields exactly batch_size EPR pairs per batch."""
        result = epr_generator.generate_batch(batch_size)
        assert len(result.qubits) == batch_size
        assert len(result.creation_timestamps) == batch_size
    
    def test_measurement_outcomes_binary(self, measurement_buffer):
        """All measurement outcomes are 0 or 1."""
        outcomes = measurement_buffer.get_outcomes()
        assert all(o in (0, 1) for o in outcomes)
    
    def test_basis_selection_uniform_distribution(self, basis_selector, n=10000):
        """Basis selection is approximately uniform over large sample."""
        bases = basis_selector.select_batch(n)
        z_count = np.sum(bases == 0)
        # Allow ±5% deviation from 50%
        assert 0.45 * n <= z_count <= 0.55 * n

class TestBasisCorrelationContract:
    """Verify basis-outcome correlation for quantum correctness."""
    
    def test_same_basis_agreement(self, alice_results, bob_results):
        """When bases match, outcomes should match (QBER bound)."""
        same_basis = alice_results.bases == bob_results.bases
        matching = alice_results.outcomes[same_basis] == bob_results.outcomes[same_basis]
        agreement_rate = np.mean(matching)
        # Must exceed (1 - QBER_HARD_LIMIT)
        assert agreement_rate >= 0.78  # 1 - 0.22
```

### 7.3 Phase II Contract Tests (`sifting/`)

```python
# tests/caligo/phases/test_sifting_contracts.py

class TestCommitmentContract:
    """Contract tests for commit-reveal protocol."""
    
    def test_commitment_binding(self, committer, bases):
        """Cannot open commitment to different value."""
        commitment = committer.commit(bases)
        
        # Must open to original
        assert committer.verify(commitment, bases)
        
        # Cannot open to modified
        modified = bases.copy()
        modified[0] = 1 - modified[0]
        assert not committer.verify(commitment, modified)
    
    def test_commitment_hiding(self, committer, bases1, bases2):
        """Commitments reveal nothing about committed value."""
        # Different bases produce different-looking commitments
        c1 = committer.commit(bases1)
        c2 = committer.commit(bases2)
        # Commitments are random (cannot distinguish statistically)
        assert c1 != c2  # Probabilistically

class TestDetectionValidatorContract:
    """Contract tests for detection statistics validation."""
    
    def test_hoeffding_bound_respected(self, validator, detection_stats):
        """Validation uses proper Hoeffding tolerance."""
        # ζ = √(ln(2/ε)/(2M))
        M = detection_stats.sample_count
        epsilon = validator.epsilon_det
        expected_tolerance = np.sqrt(np.log(2/epsilon) / (2*M))
        
        result = validator.validate(detection_stats)
        assert abs(result.tolerance - expected_tolerance) < 1e-10
    
    def test_abort_on_high_discrepancy(self, validator):
        """Validator aborts when deviation exceeds tolerance."""
        bad_stats = DetectionStats(
            alice_detection_rate=0.90,
            bob_detection_rate=0.60,  # Huge discrepancy
            sample_count=1000
        )
        with pytest.raises(SecurityError):
            validator.validate(bad_stats)

class TestQBERContract:
    """Contract tests for QBER estimation."""
    
    def test_qber_range(self, qber_estimator, test_indices, outcomes):
        """QBER must be in [0, 1]."""
        qber = qber_estimator.estimate(test_indices, outcomes)
        assert 0.0 <= qber.value <= 1.0
    
    def test_abort_threshold(self, qber_estimator):
        """Estimator aborts when QBER exceeds threshold."""
        # Construct high-error scenario
        alice_outcomes = np.zeros(100, dtype=int)
        bob_outcomes = np.ones(100, dtype=int)  # 100% error
        
        with pytest.raises(SecurityError, match="QBER.*exceeds"):
            qber_estimator.validate_security(
                alice_outcomes, bob_outcomes
            )
    
    def test_finite_size_penalty_applied(self, qber_estimator):
        """μ penalty is applied for finite-key security."""
        result = qber_estimator.estimate_with_correction(
            n=1000, k=500, epsilon_sec=1e-10
        )
        # μ = √[(n+k)/(n·k)·(k+1)/k]·ln(4/ε_sec)
        expected_mu = np.sqrt(
            (1000 + 500) / (1000 * 500) * 501 / 500
        ) * np.log(4 / 1e-10)
        assert abs(result.mu_penalty - expected_mu) < 1e-6
```

### 7.4 Phase IV Contract Tests (`amplification/`)

```python
# tests/caligo/phases/test_amplification_contracts.py

class TestEntropyCalculatorContract:
    """Contract tests for NSM entropy bounds."""
    
    @pytest.mark.parametrize("r", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_entropy_in_valid_range(self, entropy_calc, r):
        """Entropy rate must be in [0, 1]."""
        entropy_calc.storage_noise_r = r
        h_min = entropy_calc.max_bound_entropy_rate()
        assert 0.0 <= h_min <= 1.0
    
    def test_max_bound_is_maximum(self, entropy_calc):
        """Max bound equals max of both constituent bounds."""
        h_dk = entropy_calc.dupuis_konig_bound()
        h_ve = entropy_calc.virtual_erasure_bound()
        h_max = entropy_calc.max_bound_entropy_rate()
        assert h_max == max(h_dk, h_ve)
    
    def test_crossover_point(self):
        """Bounds cross near r ≈ 0.82."""
        calc_low = NSMEntropyCalculator(storage_noise_r=0.80)
        calc_high = NSMEntropyCalculator(storage_noise_r=0.84)
        
        # At r=0.80, Dupuis-König should dominate
        # At r=0.84, virtual erasure should dominate
        _, bound_low = calc_low.max_bound_entropy_rate()
        _, bound_high = calc_high.max_bound_entropy_rate()
        
        assert bound_low == "dupuis_konig"
        assert bound_high == "virtual_erasure"

class TestKeyLengthContract:
    """Contract tests for secure key length calculation."""
    
    def test_positive_key_requires_sufficient_input(self, key_calc):
        """Short inputs produce zero-length keys (Death Valley)."""
        # Very short key with high leakage
        result = key_calc.compute_final_length(
            reconciled_length=100,
            syndrome_leakage=50
        )
        # Security penalty alone is ~66 bits for ε=1e-10
        assert result == 0  # Death Valley
    
    def test_key_length_monotonic_in_input(self, key_calc):
        """Longer input yields longer (or equal) output."""
        lengths = []
        for n in [1000, 2000, 3000, 4000]:
            ell = key_calc.compute_final_length(
                reconciled_length=n,
                syndrome_leakage=int(0.1 * n)  # 10% leakage
            )
            lengths.append(ell)
        
        assert lengths == sorted(lengths)  # Monotonic

class TestToeplitzContract:
    """Contract tests for Toeplitz hashing."""
    
    def test_output_length_exact(self, hasher):
        """Output has exactly requested length."""
        input_key = np.random.randint(0, 2, hasher.input_length)
        output = hasher.hash(input_key)
        assert len(output) == hasher.output_length
    
    def test_output_binary(self, hasher):
        """Output bits are 0 or 1."""
        input_key = np.random.randint(0, 2, hasher.input_length)
        output = hasher.hash(input_key)
        assert all(b in (0, 1) for b in output)
    
    def test_different_seeds_different_outputs(self, input_length, output_length):
        """Different seeds produce different hashes (probabilistically)."""
        key = np.random.randint(0, 2, input_length)
        
        hasher1 = ToeplitzHasher(input_length, output_length, seed=secrets.token_bytes(32))
        hasher2 = ToeplitzHasher(input_length, output_length, seed=secrets.token_bytes(32))
        
        out1 = hasher1.hash(key)
        out2 = hasher2.hash(key)
        
        # Should differ with overwhelming probability
        assert not np.array_equal(out1, out2)
    
    def test_fft_direct_equivalence(self, input_length, output_length):
        """FFT and direct methods produce identical results."""
        seed = secrets.token_bytes((input_length + output_length - 1 + 7) // 8)
        key = np.random.randint(0, 2, input_length)
        
        direct = ToeplitzHasher(input_length, output_length, seed, use_fft=False)
        fft = ToeplitzHasher(input_length, output_length, seed, use_fft=True)
        
        assert np.array_equal(direct.hash(key), fft.hash(key))

class TestOTFormatterContract:
    """Contract tests for OT output structure."""
    
    def test_alice_produces_two_keys(self, formatter, reconciled_data):
        """Alice's output contains exactly two keys."""
        alice_keys = formatter.compute_alice_keys(**reconciled_data)
        assert alice_keys.key_0 is not None
        assert alice_keys.key_1 is not None
    
    def test_bob_produces_one_key(self, formatter, reconciled_data):
        """Bob's output contains exactly one key and choice bit."""
        bob_key = formatter.compute_bob_key(**reconciled_data)
        assert bob_key.key_c is not None
        assert bob_key.choice_bit in (0, 1)
    
    def test_correctness_bob_gets_right_key(self, formatter, reconciled_data):
        """Bob's key equals the appropriate Alice key."""
        alice_keys = formatter.compute_alice_keys(**reconciled_data)
        bob_key = formatter.compute_bob_key(**reconciled_data)
        
        expected = alice_keys.key_0 if bob_key.choice_bit == 0 else alice_keys.key_1
        assert np.array_equal(bob_key.key_c, expected)
```

### 7.5 Cross-Phase Integration Tests

```python
# tests/caligo/phases/test_phase_integration.py

class TestQuantumToSiftingIntegration:
    """Test Phase I → Phase II data flow."""
    
    def test_measurement_buffer_feeds_sifting(
        self, quantum_runner, sifting_manager
    ):
        """Measurement buffer output matches sifting input contract."""
        # Phase I output
        quantum_result = quantum_runner.run(batch_count=10)
        
        # Phase II input
        sifting_input = SiftingInput(
            alice_bases=quantum_result.alice_bases,
            alice_outcomes=quantum_result.alice_outcomes,
            bob_bases=quantum_result.bob_bases,
            bob_outcomes=quantum_result.bob_outcomes
        )
        
        # Should process without type/shape errors
        sifting_result = sifting_manager.process(sifting_input)
        assert sifting_result is not None

class TestSiftingToAmplificationIntegration:
    """Test Phase II → Phase IV data flow."""
    
    def test_sifting_output_feeds_amplification(
        self, sifting_result, amplification_runner
    ):
        """Sifting output matches amplification input contract."""
        # Phase IV input from Phase II output
        amp_input = AmplificationInput(
            reconciled_key=sifting_result.key_bits,
            alice_bases=sifting_result.alice_bases,
            key_indices=sifting_result.key_indices,
            qber_estimate=sifting_result.qber,
            syndrome_leakage=sifting_result.leakage_bits
        )
        
        # Should process without errors
        final_result = amplification_runner.run(amp_input)
        assert final_result.alice_keys is not None
        assert final_result.bob_key is not None
```

---

## 8. Acceptance Criteria

### 8.1 Module Completeness

| Module | Required Classes | Required Methods | LOC Target |
|--------|------------------|------------------|------------|
| `quantum/epr.py` | EPRGenerator | generate_batch, generate_async | ≤150 |
| `quantum/basis.py` | BasisSelector | select_single, select_batch | ≤80 |
| `quantum/measurement.py` | MeasurementBuffer | add_outcome, get_batch | ≤100 |
| `quantum/batching.py` | BatchingManager | configure, start_batch | ≤120 |
| `sifting/commitment.py` | CommitmentScheme | commit, verify | ≤100 |
| `sifting/sifter.py` | SiftingManager | compute_sifted_key | ≤100 |
| `sifting/qber.py` | QBEREstimator | estimate, validate | ≤100 |
| `sifting/detection_validator.py` | DetectionValidator | validate_statistics | ≤80 |
| `amplification/entropy.py` | NSMEntropyCalculator | max_bound_entropy_rate | ≤120 |
| `amplification/key_length.py` | SecureKeyLengthCalculator | compute_final_length | ≤100 |
| `amplification/toeplitz.py` | ToeplitzHasher | hash, generate_seed | ≤150 |
| `amplification/formatter.py` | OTOutputFormatter | compute_alice_keys, compute_bob_key | ≤100 |

### 8.2 Test Coverage Requirements

| Category | Minimum Coverage | Notes |
|----------|------------------|-------|
| Unit tests | 90% line coverage | Per module |
| Contract tests | 100% of public interfaces | All public methods |
| Boundary tests | All documented edge cases | QBER limits, Death Valley, etc. |
| Integration tests | All phase transitions | I→II, II→IV |
| Property tests | Mathematical invariants | Entropy bounds, hash properties |

### 8.3 Performance Targets

| Operation | Target | Constraint |
|-----------|--------|------------|
| Basis selection (1000 bits) | <1 ms | Single-threaded |
| Commitment generation | <10 ms | SHA-256 |
| QBER estimation | <5 ms | 10,000 sample positions |
| Toeplitz hash (10,000 bits) | <50 ms | Direct method |
| Toeplitz hash (100,000 bits) | <500 ms | FFT method |

### 8.4 Security Requirements

| Requirement | Validation |
|-------------|------------|
| Cryptographic RNG | All basis selection uses `secrets` module |
| Commitment binding | Hash collision probability < 2^{-256} |
| QBER threshold | Hard abort at ≥22% (Lupo limit) |
| Warning threshold | Log warning at ≥11% (Schaffner limit) |
| Security parameter | Default ε_sec = 10^{-10} |
| Timing barrier | No early reveal of basis choices |

---

## 9. References

### 9.1 Primary Literature

1. **Lupo, C., et al.** (2023). "Quantum and device-independent unconditionally secure digital signatures."  
   *Key contributions:* Max Bound entropy formula (Eq. 36), virtual erasure bound, QBER limit 22%.

2. **Schaffner, C., et al.** (2009). "Simple protocols for oblivious transfer and secure identification in the noisy-quantum-storage model."  
   *Key contributions:* NSM security model, Γ function, QBER limit 11%.

3. **König, R., et al.** (2012). "Unconditional security from noisy quantum storage."  
   *Key contributions:* Min-entropy smoothing, Dupuis-König bound.

4. **Erven, C., et al.** (2014). "An experimental implementation of oblivious transfer in the noisy storage model."  
   *Key contributions:* Finite-key analysis, μ penalty formula, practical implementation.

5. **Lemus, M., et al.** (2020). "Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation."  
   *Key contributions:* OT output structure, key partitioning by basis.

### 9.2 Technical Standards

6. **Carter, J.L., & Wegman, M.N.** (1979). "Universal classes of hash functions."  
   *Key contributions:* 2-universal hash families, Toeplitz matrices.

7. **NIST FIPS 180-4.** Secure Hash Standard (SHA-256).  
   *Key contributions:* Commitment scheme cryptographic foundation.

### 9.3 Internal Documentation

8. **Phase A Specification** (`phase_a_spec.md`): Type system and contracts
9. **Phase B Specification** (`phase_b_spec.md`): Communication layer
10. **Phase C Specification** (`phase_c_spec.md`): Security layer
11. **Phase I Implementation Plan** (`phase_I.md`): Quantum measurement details
12. **Phase II Implementation Plan** (`phase_II.md`): Sifting workflow
13. **Phase IV Implementation Plan** (`phase_IV.md`): Privacy amplification

### 9.4 Source Code References

| ehok Module | Caligo Equivalent | Key Insights |
|-------------|-------------------|--------------|
| `ehok/quantum/batching_manager.py` | `quantum/batching.py` | Batch lifecycle, EPR generation |
| `ehok/quantum/basis_selection.py` | `quantum/basis.py` | Uniform random selection |
| `ehok/core/sifting.py` | `sifting/sifter.py` | Basis matching, key extraction |
| `ehok/implementations/commitment/sha256_commitment.py` | `sifting/commitment.py` | Binding/hiding properties |
| `ehok/protocols/statistical_validation.py` | `sifting/detection_validator.py` | Hoeffding bounds, μ penalty |
| `ehok/analysis/nsm_bounds.py` | `amplification/entropy.py` | Γ function, Max Bound |
| `ehok/implementations/privacy_amplification/toeplitz_amplifier.py` | `amplification/toeplitz.py` | Direct and FFT methods |
| `ehok/core/oblivious_formatter.py` | `amplification/formatter.py` | OT structure formatting |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-01-XX | Caligo Team | Initial draft - Sections 1-5 |
| 0.2 | 2025-01-XX | Caligo Team | Added amplification/ (Section 6) |
| 1.0 | 2025-01-XX | Caligo Team | Complete specification |


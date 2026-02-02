# Caligo Phase A: Foundation Specification

**Document Type:** Formal Specification  
**Version:** 1.0  
**Date:** December 16, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope & Deliverables](#2-scope--deliverables)
3. [Package: `types/`](#3-package-types)
4. [Package: `utils/`](#4-package-utils)
5. [Project Scaffolding](#5-project-scaffolding)
6. [Numerical Constants & Baseline Values](#6-numerical-constants--baseline-values)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

## 1. Executive Summary

**Phase A** establishes the foundational infrastructure for the Caligo project:

| Component | Purpose | Est. LOC |
|-----------|---------|----------|
| `types/` | Domain primitives, phase contracts, exceptions | ~450 |
| `utils/` | Logging, mathematical functions, bitarray helpers | ~260 |
| Scaffolding | `pyproject.toml`, test infrastructure, CI configuration | N/A |

**Critical Design Principle:** All types defined in Phase A serve as the canonical vocabulary for the entire Caligo codebase. These dataclasses form the **phase boundary contracts** that enable:

1. Type-safe data flow between protocol phases
2. Runtime validation via Design-by-Contract
3. Deterministic parity testing with reference implementations

---

## 2. Scope & Deliverables

### 2.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `caligo/types/keys.py` | Oblivious key representations |
| `caligo/types/measurements.py` | Measurement records and round results |
| `caligo/types/phase_contracts.py` | Phase I→IV boundary dataclasses |
| `caligo/types/exceptions.py` | Exception hierarchy and protocol enums |
| `caligo/utils/logging.py` | Structured logging with SquidASM compatibility |
| `caligo/utils/math.py` | Binary entropy and shared mathematical functions |
| `caligo/utils/bitarray_utils.py` | Bitarray manipulation helpers |
| `pyproject.toml` | Package configuration and dependencies |
| `tests/conftest.py` | Pytest fixtures and test infrastructure |

### 2.2 Out of Scope

- Simulation layer (`simulation/`) — Phase B
- Security bounds (`security/`) — Phase C
- Protocol phase implementations — Phase D
- SquidASM program definitions — Phase E

### 2.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | ≥3.10 | Type hints, dataclasses |
| NumPy | ≥1.21 | Array operations |
| bitarray | ≥2.6 | Efficient bit manipulation |
| pytest | ≥7.0 | Testing framework |
| SquidASM | ≥0.12 | LogManager integration (optional) |

---

## 3. Package: `types/`

### 3.1 Module: `keys.py` (< 100 LOC)

**Purpose:** Define the canonical representation of oblivious keys — the final output of the $\binom{2}{1}$-OT protocol.

#### 3.1.1 `ObliviousKey` (Base Dataclass)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ObliviousKey Specification                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  An ObliviousKey represents a cryptographic key extracted via the $\binom{2}{1}$-OT     │
│  protocol. The key is "oblivious" because one party (Bob) learns only       │
│  one of two possible keys, while the other party (Alice) doesn't know       │
│  which key Bob obtained.                                                    │
│                                                                             │
│  Attributes:                                                                │
│  ┌─────────────────┬────────────────┬─────────────────────────────────────┐│
│  │ Name            │ Type           │ Description                         ││
│  ├─────────────────┼────────────────┼─────────────────────────────────────┤│
│  │ bits            │ bitarray       │ The key bits                        ││
│  │ length          │ int            │ Key length in bits (≥1)             ││
│  │ security_param  │ float          │ ε_sec achieved (default: 1e-10)     ││
│  │ creation_time   │ float          │ Simulation timestamp (ns)           ││
│  └─────────────────┴────────────────┴─────────────────────────────────────┘│
│                                                                             │
│  Invariants:                                                                │
│  • INV-KEY-001: len(bits) == length                                        │
│  • INV-KEY-002: security_param ∈ (0, 1)                                    │
│  • INV-KEY-003: creation_time ≥ 0                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 `AliceObliviousKey` (Specialized)

```python
@dataclass(frozen=True)
class AliceObliviousKey:
    """
    Alice's output from the $\binom{2}{1}$-OT protocol: two keys S₀ and S₁.
    
    Alice possesses both keys but does not know which one Bob received.
    This is the "sender" output in 1-out-of-2 Oblivious Transfer.
    
    Attributes
    ----------
    s0 : bitarray
        Key corresponding to choice bit 0.
    s1 : bitarray  
        Key corresponding to choice bit 1.
    key_length : int
        Length of each key in bits (both must be equal).
    security_parameter : float
        ε_sec achieved, typically 10^{-10} per Erven et al.
    entropy_consumed : float
        Total min-entropy consumed in privacy amplification.
    
    Invariants
    ----------
    - INV-ALICE-001: len(s0) == len(s1) == key_length
    - INV-ALICE-002: security_parameter ∈ (0, 1)
    
    References
    ----------
    - Erven et al. (2014), Section "Results: The Oblivious Transfer Protocol"
    """
```

#### 3.1.3 `BobObliviousKey` (Specialized)

```python
@dataclass(frozen=True)
class BobObliviousKey:
    """
    Bob's output from the $\binom{2}{1}$-OT protocol: one key Sᴄ and his choice bit C.
    
    Bob receives exactly one of Alice's two keys, determined by his choice
    bit C. He cannot learn anything about the other key S_{1-C}.
    
    Attributes
    ----------
    sc : bitarray
        The key Bob received (either S₀ or S₁).
    choice_bit : int
        Bob's choice bit C ∈ {0, 1}.
    key_length : int
        Length of the received key in bits.
    security_parameter : float
        ε_sec achieved.
    
    Invariants
    ----------
    - INV-BOB-001: len(sc) == key_length
    - INV-BOB-002: choice_bit ∈ {0, 1}
    
    References
    ----------
    - Schaffner et al. (2009), Definition 1: "ε-secure 1-2 ROT"
    """
```

---

### 3.2 Module: `measurements.py` (< 100 LOC)

**Purpose:** Define records for quantum measurement outcomes and detection events.

#### 3.2.1 `MeasurementRecord`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MeasurementRecord Specification                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Records a single quantum measurement event from EPR pair generation.       │
│                                                                             │
│  Attributes:                                                                │
│  ┌─────────────────┬────────────────┬─────────────────────────────────────┐│
│  │ Name            │ Type           │ Description                         ││
│  ├─────────────────┼────────────────┼─────────────────────────────────────┤│
│  │ round_id        │ int            │ Unique identifier for this round    ││
│  │ outcome         │ int            │ Measurement result: 0 or 1          ││
│  │ basis           │ int            │ Basis used: 0 (Z) or 1 (X)          ││
│  │ timestamp_ns    │ float          │ Simulation time when measured       ││
│  │ detected        │ bool           │ True if photon was detected         ││
│  └─────────────────┴────────────────┴─────────────────────────────────────┘│
│                                                                             │
│  Invariants:                                                                │
│  • INV-MEAS-001: outcome ∈ {0, 1}                                          │
│  • INV-MEAS-002: basis ∈ {0, 1}                                            │
│  • INV-MEAS-003: round_id ≥ 0                                              │
│  • INV-MEAS-004: timestamp_ns ≥ 0                                          │
│                                                                             │
│  Domain Mapping (BB84):                                                     │
│  • basis=0 (Z): Computational basis {|0⟩, |1⟩}                             │
│  • basis=1 (X): Hadamard basis {|+⟩, |-⟩}                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 `RoundResult`

```python
@dataclass
class RoundResult:
    """
    Combined result for a single EPR pair round (both Alice and Bob).
    
    Aggregates measurement outcomes from both parties for a single
    EPR pair, enabling sifting and QBER calculation.
    
    Attributes
    ----------
    round_id : int
        Unique round identifier.
    alice_outcome : int
        Alice's measurement outcome (0 or 1).
    bob_outcome : int
        Bob's measurement outcome (0 or 1).
    alice_basis : int
        Alice's basis choice (0=Z, 1=X).
    bob_basis : int
        Bob's basis choice (0=Z, 1=X).
    bases_match : bool
        True if alice_basis == bob_basis.
    outcomes_match : bool
        True if alice_outcome == bob_outcome (meaningful only when bases_match).
    alice_detected : bool
        True if Alice detected a photon.
    bob_detected : bool
        True if Bob detected a photon.
    
    Derived Properties
    ------------------
    is_valid : bool
        True if both parties detected photons.
    contributes_to_sifted_key : bool
        True if valid AND bases match.
    has_error : bool
        True if bases match but outcomes differ (contributes to QBER).
    """
```

#### 3.2.3 `DetectionEvent`

```python
@dataclass
class DetectionEvent:
    """
    Single detection event for missing rounds validation.
    
    Used to track which rounds resulted in successful photon detection
    for Chernoff-bound validation against expected channel transmittance.
    
    Attributes
    ----------
    round_id : int
        Round identifier.
    detected : bool
        True if photon was detected.
    timestamp_ns : float
        Detection timestamp in nanoseconds.
    
    References
    ----------
    - Erven et al. (2014): "Alice checks if number of photons measured
      by Bob falls within acceptable interval for security"
    """
```

---

### 3.3 Module: `phase_contracts.py` (< 150 LOC)

**Purpose:** Define the canonical data contracts at each protocol phase boundary.

#### 3.3.1 Contract Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     $\binom{2}{1}$-OT Phase Boundary Contracts                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase I (Quantum) ──► QuantumPhaseResult ──► Phase II (Sifting)           │
│                                                                             │
│  Phase II (Sifting) ──► SiftingPhaseResult ──► Phase III (Reconciliation)  │
│                                                                             │
│  Phase III (Reconciliation) ──► ReconciliationPhaseResult ──► Phase IV     │
│                                                                             │
│  Phase IV (Amplification) ──► ObliviousTransferOutput ──► Application      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 `QuantumPhaseResult`

```python
@dataclass
class QuantumPhaseResult:
    """
    Contract: Phase I → Phase II data transfer.
    
    Contains all quantum measurement data and metadata required for
    the sifting and estimation phase. Enforces NSM timing invariants.
    
    Attributes
    ----------
    measurement_outcomes : np.ndarray
        Array of measurement outcomes (0/1), shape (n_pairs,), dtype uint8.
    basis_choices : np.ndarray
        Array of basis choices (0=Z, 1=X), shape (n_pairs,), dtype uint8.
    round_ids : np.ndarray
        Array of round identifiers, shape (n_pairs,), dtype int64.
    generation_timestamp : float
        Simulation time (ns) when quantum phase completed.
    num_pairs_requested : int
        Number of EPR pairs requested.
    num_pairs_generated : int
        Number of EPR pairs actually generated (may differ due to losses).
    detection_events : List[DetectionEvent]
        Detection event records for validation.
    timing_barrier_marked : bool
        True if TimingBarrier.mark_quantum_complete() was called.
    
    Pre-conditions (caller must ensure)
    ------------------------------------
    - num_pairs_requested > 0
    - Quantum channel is calibrated
    
    Post-conditions (this contract guarantees)
    ------------------------------------------
    - POST-Q-001: len(measurement_outcomes) == num_pairs_generated
    - POST-Q-002: len(basis_choices) == num_pairs_generated
    - POST-Q-003: All outcomes ∈ {0, 1}
    - POST-Q-004: All bases ∈ {0, 1}
    - POST-Q-005: timing_barrier_marked == True (for NSM security)
    
    References
    ----------
    - phase_I_analysis.md Section 7.2
    """
```

#### 3.3.3 `SiftingPhaseResult`

```python
@dataclass
class SiftingPhaseResult:
    """
    Contract: Phase II → Phase III data transfer.
    
    Contains sifted key material with QBER estimates and statistical
    penalties accounting for finite-size effects.
    
    Attributes
    ----------
    sifted_key_alice : bitarray
        Alice's sifted key bits (matching basis positions).
    sifted_key_bob : bitarray
        Bob's sifted key bits (may contain errors).
    matching_indices : np.ndarray
        Original round indices where bases matched, dtype int64.
    i0_indices : np.ndarray
        Indices for I₀ partition (Alice's random subset).
    i1_indices : np.ndarray
        Indices for I₁ partition (complement of I₀).
    test_set_indices : np.ndarray
        Indices sacrificed for QBER estimation.
    qber_estimate : float
        Observed QBER on test set: e_obs.
    qber_adjusted : float
        QBER with finite-size penalty: e_adj = e_obs + μ.
    finite_size_penalty : float
        Statistical penalty μ from Erven et al. Eq. (2).
    test_set_size : int
        Number of bits used for testing |T|.
    timing_compliant : bool
        True if Δt was properly enforced before basis revelation.
    
    Post-conditions
    ---------------
    - POST-S-001: len(sifted_key_alice) == len(sifted_key_bob)
    - POST-S-002: qber_adjusted = qber_estimate + finite_size_penalty
    - POST-S-003: qber_adjusted ≤ QBER_HARD_LIMIT (else would abort)
    
    References
    ----------
    - phase_II_analysis.md Section 1.1
    - Erven et al. (2014) Theorem 2, Eq. (2) for μ calculation
    """
```

#### 3.3.4 `ReconciliationPhaseResult`

```python
@dataclass
class ReconciliationPhaseResult:
    """
    Contract: Phase III → Phase IV data transfer.
    
    Contains error-corrected key material with leakage accounting
    for privacy amplification entropy calculation.
    
    Attributes
    ----------
    reconciled_key : bitarray
        Error-corrected key (Alice's perspective).
    num_blocks : int
        Number of LDPC blocks processed.
    blocks_succeeded : int
        Number of blocks that passed verification.
    blocks_failed : int
        Number of blocks that failed (discarded).
    total_syndrome_bits : int
        Total syndrome leakage |Σ| in bits.
    effective_rate : float
        Achieved code rate R = (n - |Σ|) / n.
    hash_verified : bool
        True if final hash verification passed.
    leakage_within_cap : bool
        True if |Σ| ≤ L_max (safety cap).
    
    Post-conditions
    ---------------
    - POST-R-001: total_syndrome_bits ≤ L_max (leakage safety cap)
    - POST-R-002: hash_verified == True (else would abort)
    
    References
    ----------
    - phase_III_analysis.md Section 1.1 (Wiretap Cost)
    - Schaffner et al. (2009): "length of syndrome must be subtracted"
    """
```

#### 3.3.5 `AmplificationPhaseResult`

```python
@dataclass
class AmplificationPhaseResult:
    """
    Contract: Phase IV → Final protocol output (role-specific).
    
    Contains the privacy-amplified output for a single party,
    plus diagnostic metrics. This is the role-specific view;
    the aggregate protocol output is ObliviousTransferOutput.
    
    Attributes
    ----------
    oblivious_key : Union[AliceObliviousKey, BobObliviousKey]
        Role-dependent output key(s).
    qber : float
        Final adjusted QBER used for security calculations.
    key_length : int
        Length of extracted key(s) in bits.
    entropy_consumed : float
        Total min-entropy consumed (h_min * n - leakage).
    entropy_rate : float
        Efficiency: key_length / raw_bits.
    metrics : Dict[str, Any]
        Diagnostic data (timing, block stats, etc.).
    
    Post-conditions
    ---------------
    - POST-AMP-001: key_length > 0 (else abort before reaching here)
    - POST-AMP-002: entropy_consumed >= key_length + 2*log(1/ε_sec)
    
    References
    ----------
    - Lupo et al. (2023), Eq. (43): Key length formula
    - Erven et al. (2014): Experimental validation
    """
```

#### 3.3.6 `ObliviousTransferOutput`

```python
@dataclass
class ObliviousTransferOutput:
    """
    Final protocol output: 1-out-of-2 OT keys.
    
    The terminal output of the $\binom{2}{1}$-OT protocol, containing:
    - For Alice: Two keys (S₀, S₁)
    - For Bob: One key (Sᴄ) and his choice bit C
    
    Attributes
    ----------
    alice_key : AliceObliviousKey
        Alice's output containing S₀ and S₁.
    bob_key : BobObliviousKey
        Bob's output containing Sᴄ and C.
    protocol_succeeded : bool
        True if protocol completed without abort.
    total_rounds : int
        Total EPR pairs used in the protocol.
    final_key_length : int
        Length of extracted keys in bits.
    security_parameter : float
        ε_sec achieved (trace distance from ideal).
    entropy_rate : float
        Bits of key per input bit: ℓ / n.
    
    Post-conditions
    ---------------
    - POST-OT-001: len(alice_key.s0) == len(alice_key.s1) == final_key_length
    - POST-OT-002: len(bob_key.sc) == final_key_length
    - POST-OT-003: bob_key.sc == alice_key.s0 if bob_key.choice_bit == 0
                   bob_key.sc == alice_key.s1 if bob_key.choice_bit == 1
    
    References
    ----------
    - Schaffner et al. (2009) Definition 1
    - Erven et al. (2014) "Results" section
    """
```

---

### 3.4 Module: `exceptions.py` (< 100 LOC)

**Purpose:** Define the exception hierarchy and protocol state enumerations.

#### 3.4.1 Exception Hierarchy

```
CaligoError (Base)
├── SimulationError
│   ├── TimingViolationError      # Δt not satisfied
│   ├── NetworkConfigError        # SquidASM network setup failed
│   └── EPRGenerationError        # EPR pair generation failed
├── SecurityError
│   ├── QBERThresholdExceeded     # QBER > threshold
│   ├── NSMViolationError         # NSM assumptions violated
│   ├── FeasibilityError          # Pre-flight check failed
│   ├── EntropyDepletedError      # No extractable entropy
│   └── CommitmentVerificationError # Commitment hash mismatch
├── ProtocolError
│   ├── PhaseOrderViolation       # Phases executed out of order
│   ├── ContractViolation         # Phase contract invariant failed
│   └── ReconciliationError       # LDPC decoding failed
├── ConnectionError
│   ├── OrderingViolationError    # Message ordering violated
│   ├── AckTimeoutError           # ACK not received in time
│   ├── SessionMismatchError      # Session ID mismatch
│   └── OutOfOrderError           # Sequence number out of order
└── ConfigurationError
    ├── InvalidParameterError     # Configuration parameter out of range
    └── MissingConfigError        # Required configuration not provided
```

#### 3.4.2 `ProtocolPhase` Enum

```python
class ProtocolPhase(Enum):
    """
    Enumeration of $\binom{2}{1}$-OT protocol phases.
    
    Used for state machine tracking and transcript annotation.
    Phase transitions must follow the order:
    INIT → QUANTUM → SIFTING → RECONCILIATION → AMPLIFICATION → COMPLETED
    
    Any phase may transition to ABORTED.
    """
    INIT = "init"
    QUANTUM = "quantum"              # Phase I
    SIFTING = "sifting"              # Phase II
    RECONCILIATION = "reconciliation" # Phase III
    AMPLIFICATION = "amplification"  # Phase IV
    COMPLETED = "completed"
    ABORTED = "aborted"
```

#### 3.4.3 `AbortReason` Enum

```python
class AbortReason(Enum):
    """
    Enumeration of protocol abort reasons with diagnostic codes.
    
    Code taxonomy:
    - ABORT-I-*: Phase I (Quantum) abort conditions
    - ABORT-II-*: Phase II (Sifting) abort conditions
    - ABORT-III-*: Phase III (Reconciliation) abort conditions
    - ABORT-IV-*: Phase IV (Amplification) abort conditions
    """
    # Phase I
    FEASIBILITY_HARD_LIMIT = "ABORT-I-FEAS-001"   # Q_total > 22%
    TIMING_VIOLATION = "ABORT-I-TIMING-001"       # Basis revealed before Δt
    
    # Phase II
    DETECTION_ANOMALY = "ABORT-II-DET-001"        # Chernoff bound violated
    QBER_HARD_LIMIT = "ABORT-II-QBER-001"         # Adjusted QBER > 22%
    MISSING_ROUNDS_INVALID = "ABORT-II-MISS-001"  # Invalid loss report
    
    # Phase III
    LEAKAGE_CAP_EXCEEDED = "ABORT-III-LEAK-001"   # |Σ| > L_max
    RECONCILIATION_FAILED = "ABORT-III-REC-001"   # Decoder failure
    VERIFICATION_FAILED = "ABORT-III-VER-001"     # Hash mismatch
    
    # Phase IV
    ENTROPY_DEPLETED = "ABORT-IV-ENT-001"         # h_min ≤ 0
    KEY_LENGTH_ZERO = "ABORT-IV-LEN-001"          # ℓ = 0
```

---

## 4. Package: `utils/`

### 4.1 Module: `logging.py` (< 100 LOC)

**Purpose:** Provide structured logging compatible with SquidASM's LogManager.

#### 4.1.1 Requirements

| Requirement | Description |
|-------------|-------------|
| UTIL-LOG-001 | All Caligo modules obtain loggers through a single utility function |
| UTIL-LOG-002 | Graceful fallback when SquidASM is not available (unit test mode) |
| UTIL-LOG-003 | Idempotent setup (no handler duplication on repeated calls) |
| UTIL-LOG-004 | Configurable file and terminal output |
| UTIL-LOG-005 | Thread-safe logger configuration |

#### 4.1.2 API Specification

```python
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with SquidASM compatibility.
    
    Parameters
    ----------
    name : str
        Hierarchical logger name (e.g., "caligo.quantum.epr").
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Notes
    -----
    Falls back to standard logging.getLogger() when SquidASM is
    not available, enabling use in non-simulation unit tests.
    """

def setup_script_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    show_terminal: bool = False
) -> logging.Logger:
    """
    Configure logging for standalone scripts.
    
    Sets up dual-output logging:
    - File output: ALWAYS enabled, named <script_name>.log
    - Terminal output: Controlled by show_terminal flag
    
    This function is idempotent.
    """
```

---

### 4.2 Module: `math.py` (< 80 LOC)

**Purpose:** Provide mathematical functions used throughout the protocol.

#### 4.2.1 `binary_entropy`

```python
def binary_entropy(p: float) -> float:
    """
    Compute the binary entropy function h(p).
    
    h(p) = -p·log₂(p) - (1-p)·log₂(1-p)
    
    Parameters
    ----------
    p : float
        Probability value in [0, 1].
    
    Returns
    -------
    float
        Binary entropy in bits.
    
    Edge Cases
    ----------
    - h(0) = 0 (by convention: 0·log(0) = 0)
    - h(1) = 0
    - h(0.5) = 1 (maximum entropy)
    
    References
    ----------
    - Schaffner et al. (2009) Section 3.2: "binary-entropy function"
    """
```

#### 4.2.2 `channel_capacity`

```python
def channel_capacity(qber: float) -> float:
    """
    Compute binary symmetric channel capacity.
    
    C(QBER) = 1 - h(QBER)
    
    Parameters
    ----------
    qber : float
        Quantum bit error rate in [0, 0.5].
    
    Returns
    -------
    float
        Channel capacity in bits per symbol.
    
    References
    ----------
    - Shannon's noisy channel coding theorem
    """
```

#### 4.2.3 `finite_size_penalty`

```python
def finite_size_penalty(
    n: int,
    k: int,
    epsilon_sec: float = 1e-10
) -> float:
    """
    Compute finite-size statistical penalty μ.
    
    μ = √((n + k)/(n·k) · (k + 1)/k) · ln(4/ε_sec)
    
    Parameters
    ----------
    n : int
        Size of remaining key (after test set removal).
    k : int
        Size of test set.
    epsilon_sec : float
        Security parameter (default: 10^{-10}).
    
    Returns
    -------
    float
        Statistical penalty μ to add to observed QBER.
    
    Pre-conditions
    --------------
    - n > 0
    - k > 0
    - epsilon_sec ∈ (0, 1)
    
    References
    ----------
    - Erven et al. (2014) Theorem 2, Eq. (2)
    - phase_II.md Section 2.B
    """
```

#### 4.2.4 `gamma_function`

```python
def gamma_function(r: float) -> float:
    """
    Compute Γ(r) for NSM security bound.
    
    For depolarizing storage with noise parameter r:
    Γ(r) characterizes the adversary's storage quality.
    
    Γ(r) = 1 - log₂(1 + 3r²) for depolarizing channel
    
    Parameters
    ----------
    r : float
        Storage noise parameter in [0, 1].
        r = 0: Perfect storage (worst case)
        r = 1: Complete depolarization (best case)
    
    Returns
    -------
    float
        Γ(r) value used in entropy bounds.
    
    References
    ----------
    - Lupo et al. (2020) "Max Bound" derivation
    - phase_IV.md Section 2.A
    """
```

---

### 4.3 Module: `bitarray_utils.py` (< 80 LOC)

**Purpose:** Provide bitarray manipulation helpers.

#### 4.3.1 Core Functions

```python
def xor_bitarrays(a: bitarray, b: bitarray) -> bitarray:
    """XOR two bitarrays of equal length."""

def bitarray_to_bytes(bits: bitarray) -> bytes:
    """Convert bitarray to bytes (big-endian)."""

def bytes_to_bitarray(data: bytes) -> bitarray:
    """Convert bytes to bitarray."""

def random_bitarray(length: int) -> bitarray:
    """Generate cryptographically secure random bitarray."""

def hamming_distance(a: bitarray, b: bitarray) -> int:
    """Compute Hamming distance between two bitarrays."""

def slice_bitarray(bits: bitarray, indices: np.ndarray) -> bitarray:
    """Extract bits at specified indices."""
```

---

## 5. Project Scaffolding

### 5.1 `pyproject.toml` Specification

```toml
[project]
name = "caligo"
version = "0.1.0"
description = "1-out-of-2 OT Protocol Implementation with Native SquidASM Integration"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Alessandro Da Ros"}
]
keywords = ["quantum", "cryptography", "oblivious-transfer", "squidasm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Security :: Cryptography",
    "Programming Language :: Python :: 3.10",
]

dependencies = [
    "numpy>=1.21",
    "bitarray>=2.6",
    "scipy>=1.7",
]

[project.optional-dependencies]
simulation = [
    "squidasm>=0.12",
    "netqasm>=0.15",
    "netsquid>=1.1",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "black>=23.0",
    "ruff>=0.1",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "D"]
```

### 5.2 Directory Structure (Phase A)

```
caligo/
├── __init__.py                # Package exports
├── types/
│   ├── __init__.py           # Type re-exports
│   ├── keys.py               # Key dataclasses
│   ├── measurements.py       # Measurement records
│   ├── phase_contracts.py    # Phase boundary contracts
│   └── exceptions.py         # Exceptions + enums
├── utils/
│   ├── __init__.py           # Utility exports
│   ├── logging.py            # Logging infrastructure
│   ├── math.py               # Mathematical functions
│   └── bitarray_utils.py     # Bitarray helpers
└── tests/
    ├── conftest.py           # Shared fixtures
    ├── test_types/
    │   ├── test_keys.py
    │   ├── test_measurements.py
    │   ├── test_phase_contracts.py
    │   └── test_exceptions.py
    └── test_utils/
        ├── test_logging.py
        ├── test_math.py
        └── test_bitarray_utils.py
```

---

## 6. Numerical Constants & Baseline Values

### 6.1 Security Thresholds

These values are derived from the literature and represent hard constraints:

| Constant | Value | Source | Description |
|----------|-------|--------|-------------|
| `QBER_HARD_LIMIT` | 0.22 (22%) | König et al. (2012) | Maximum QBER for security possibility |
| `QBER_CONSERVATIVE_LIMIT` | 0.11 (11%) | Schaffner et al. (2009), Corollary 7 | Recommended operational threshold |
| `EPSILON_SECURITY_DEFAULT` | 1e-10 | Erven et al. (2014), Table I | Default security parameter ε_sec |
| `EPSILON_CORRECTNESS_DEFAULT` | 2.5e-7 | Erven et al. (2014), Table I | Default correctness parameter |

**Literature Justification:**

> "We are now able to show that [...] 1-2 oblivious transfer and secure identification can be achieved in the noisy-storage model with depolarizing storage noise, as long as the quantum bit-error rate of the channel does not exceed **11%** and the noise on the channel is strictly less than the noise during quantum storage. This is **optimal for the protocol considered**."
> — Schaffner et al. (2009), Section 2

### 6.2 NSM Storage Parameters

| Constant | Default Value | Range | Description |
|----------|---------------|-------|-------------|
| `STORAGE_NOISE_R_DEFAULT` | 0.75 | [0, 1] | Adversary storage noise rate |
| `STORAGE_RATE_NU_DEFAULT` | 0.002 | [0, 1] | Fraction of qubits storable |
| `STORAGE_DIMENSION_D` | 2 | {2} | Qubit dimension |

**Literature Reference (Erven et al. 2014, Table I):**
```
Adversary's Memory Limitations    Value
d                                 2
r                                 0.75
ν                                 0.002
```

### 6.3 Physical Channel Parameters

| Constant | Typical Value | Description |
|----------|---------------|-------------|
| `SOURCE_QUALITY_MU_DEFAULT` | 0.98 | EPR source fidelity |
| `DETECTION_EFFICIENCY_ETA_DEFAULT` | 0.85 | Combined detector efficiency |
| `INTRINSIC_ERROR_E_DET_DEFAULT` | 0.01 | Detector error rate |
| `DARK_COUNT_RATE_HZ_DEFAULT` | 100.0 | Dark counts per second |
| `FIBER_LOSS_DB_PER_KM` | 0.2 | Fiber attenuation |

### 6.4 Timing Parameters

| Constant | Default Value | Description |
|----------|---------------|-------------|
| `DELTA_T_NS_DEFAULT` | 1_000_000 | NSM wait time Δt (1 ms) |
| `STATE_DELAY_NS_DEFAULT` | 1_000 | EPR delivery delay |
| `MEMORY_T1_NS_DEFAULT` | 10_000_000 | T1 relaxation time (10 ms) |
| `MEMORY_T2_NS_DEFAULT` | 1_000_000 | T2 dephasing time (1 ms) |

### 6.5 Error Correction Parameters

| Constant | Default Value | Description |
|----------|---------------|-------------|
| `EC_EFFICIENCY_F_DEFAULT` | 1.16 | LDPC efficiency factor f |
| `EC_FAILURE_PROB_DEFAULT` | 3.09e-3 | ε_EC from Erven et al. |
| `BP_MAX_ITERATIONS` | 100 | Belief propagation iterations |

**Literature Reference (Erven et al. 2014):**
```
ε_EC                              3.09 × 10^{-3}
f                                 1.491
```

### 6.6 Batch Size Constraints

| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_SIFTED_KEY_LENGTH` | 10_000 | Minimum for finite-key security |
| `RECOMMENDED_BATCH_SIZE` | 100_000 | Recommended per-round generation |
| `MAX_QUBITS_PER_NODE` | 10 | SquidASM memory constraint |

**Finite-Key Warning (Lupo et al. 2020):**
> "For small $N$ (e.g., $< 10^5$), the $\Delta$ penalty can consume the entire key, making secure OT impossible."

---

## 7. Testing Strategy

### 7.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Validate individual dataclasses and functions | `tests/test_types/`, `tests/test_utils/` |
| Contract Tests | Verify phase boundary invariants | `tests/test_types/test_phase_contracts.py` |
| Property Tests | Fuzz test edge cases | Integrated with unit tests |

### 7.2 `conftest.py` Fixtures

```python
# tests/conftest.py

import pytest
import numpy as np
from bitarray import bitarray

@pytest.fixture
def sample_quantum_phase_result():
    """Generate a valid QuantumPhaseResult for testing."""
    n = 1000
    return QuantumPhaseResult(
        measurement_outcomes=np.random.randint(0, 2, n, dtype=np.uint8),
        basis_choices=np.random.randint(0, 2, n, dtype=np.uint8),
        round_ids=np.arange(n, dtype=np.int64),
        generation_timestamp=1_000_000.0,
        num_pairs_requested=n,
        num_pairs_generated=n,
        detection_events=[],
        timing_barrier_marked=True,
    )

@pytest.fixture
def sample_sifting_result():
    """Generate a valid SiftingPhaseResult for testing."""
    n = 500
    key = bitarray(np.random.randint(0, 2, n).tolist())
    return SiftingPhaseResult(
        sifted_key_alice=key,
        sifted_key_bob=key.copy(),  # Perfect correlation for test
        matching_indices=np.arange(n, dtype=np.int64),
        i0_indices=np.arange(0, n//2, dtype=np.int64),
        i1_indices=np.arange(n//2, n, dtype=np.int64),
        test_set_indices=np.arange(0, n//10, dtype=np.int64),
        qber_estimate=0.05,
        qber_adjusted=0.06,
        finite_size_penalty=0.01,
        test_set_size=n//10,
        timing_compliant=True,
    )

@pytest.fixture
def security_params():
    """Standard security parameters for testing."""
    return {
        'epsilon_sec': 1e-10,
        'qber_hard_limit': 0.22,
        'qber_conservative': 0.11,
        'storage_noise_r': 0.75,
    }
```

### 7.3 Test Examples

```python
# tests/test_types/test_keys.py

def test_alice_key_invariant_length_match():
    """INV-ALICE-001: s0 and s1 must have equal length."""
    s0 = bitarray('10101010')
    s1 = bitarray('01010101')
    key = AliceObliviousKey(
        s0=s0, s1=s1, key_length=8,
        security_parameter=1e-10, entropy_consumed=4.0
    )
    assert len(key.s0) == len(key.s1) == key.key_length

def test_alice_key_rejects_mismatched_lengths():
    """INV-ALICE-001 violation should raise."""
    s0 = bitarray('1010')
    s1 = bitarray('01010101')  # Different length
    with pytest.raises(ContractViolation):
        AliceObliviousKey(s0=s0, s1=s1, key_length=4, ...)

# tests/test_utils/test_math.py

def test_binary_entropy_bounds():
    """h(p) ∈ [0, 1] for p ∈ [0, 1]."""
    for p in np.linspace(0, 1, 100):
        h = binary_entropy(p)
        assert 0 <= h <= 1

def test_binary_entropy_maximum_at_half():
    """h(0.5) = 1 (maximum entropy)."""
    assert abs(binary_entropy(0.5) - 1.0) < 1e-10

def test_finite_size_penalty_matches_literature():
    """Verify μ calculation against Erven et al. reference values."""
    # From Erven et al. (2014) experimental setup
    n = 8e7  # Total rounds
    k = int(n * 0.1)  # 10% test fraction
    epsilon = 2.5e-7
    
    mu = finite_size_penalty(int(n - k), k, epsilon)
    # Expected: small penalty for large n
    assert mu < 0.01
```

---

## 8. Acceptance Criteria

### 8.1 Functional Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-A-001 | All dataclasses validate invariants in `__post_init__` | Unit tests |
| AC-A-002 | Invalid data raises appropriate exception type | Unit tests |
| AC-A-003 | Logging works with and without SquidASM | Unit tests |
| AC-A-004 | Mathematical functions match literature formulas | Unit tests with reference values |
| AC-A-005 | Type hints pass mypy strict mode | CI check |

### 8.2 Quality Criteria

| ID | Criterion | Target |
|----|-----------|--------|
| AC-A-006 | Test coverage for `types/` | ≥95% |
| AC-A-007 | Test coverage for `utils/` | ≥95% |
| AC-A-008 | All modules ≤ 200 LOC | Static analysis |
| AC-A-009 | Numpydoc format for all public APIs | Linter check |
| AC-A-010 | No `print()` statements | Grep check |

### 8.3 Documentation Criteria

| ID | Criterion |
|----|-----------|
| AC-A-011 | All public classes have docstrings with Parameters/Returns/References |
| AC-A-012 | All constants have inline comments citing literature source |
| AC-A-013 | README.md documents installation and basic usage |

---

## 9. References

### 9.1 Primary Literature

| Citation | Title | Usage in Phase A |
|----------|-------|------------------|
| Erven et al. (2014) | "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model" | Security parameters, μ calculation, experimental baselines |
| Schaffner et al. (2009) | "Robust Cryptography in the Noisy-Quantum-Storage Model" | QBER threshold (11%), protocol definition |
| König et al. (2012) | "Unconditional Security from Noisy Quantum Storage" | NSM theoretical foundation, timing semantics |
| Lupo et al. (2020) | "Performance of Practical Quantum Oblivious Key Distribution" | Max Bound, finite-key analysis |

### 9.2 Internal Documents

| Document | Relevance |
|----------|-----------|
| [caligo_architecture.md](caligo_architecture.md) | Parent architecture document |
| [phase_I_analysis.md](../implementation%20plan/phase_I_analysis.md) | Timing invariants, POST-PHI conditions |
| [phase_II_analysis.md](../implementation%20plan/phase_II_analysis.md) | Sifting contracts, μ penalty |
| [phase_III_analysis.md](../implementation%20plan/phase_III_analysis.md) | Reconciliation contracts, wiretap cost |
| [phase_IV_analysis.md](../implementation%20plan/phase_IV_analysis.md) | Amplification contracts, Max Bound |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Lead Architect | Initial specification |

# Caligo Phase E: Orchestration Layer Specification

**Document Type:** Formal Specification  
**Version:** 1.0  
**Date:** December 17, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)  
**Prerequisites:** [phase_a_spec.md](phase_a_spec.md), [phase_b_spec.md](phase_b_spec.md), [phase_c_spec.md](phase_c_spec.md), [phase_d_spec.md](phase_d_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope & Deliverables](#2-scope--deliverables)
3. [SquidASM Integration Foundations](#3-squidasm-integration-foundations)
4. [Package: `connection/` — Ordered Messaging](#4-package-connection--ordered-messaging)
5. [Package: `protocol/` — SquidASM Programs](#5-package-protocol--squidasm-programs)
6. [End-to-End Integration Tests](#6-end-to-end-integration-tests)
7. [Acceptance Criteria](#7-acceptance-criteria)
8. [References](#8-references)

---

## 1. Executive Summary

**Phase E** implements the orchestration layer that binds all Caligo components into executable SquidASM programs. This phase is the **integration keystone** — it transforms the modular components from Phases A-D into coherent protocol executions within NetSquid's discrete-event simulation framework.

### 1.1 Deliverable Overview

| Package | Purpose | Est. LOC |
|---------|---------|----------|
| `connection/` | Ordered messaging with commit-then-reveal enforcement | ~300 |
| `protocol/` | SquidASM program implementations (Alice, Bob, Orchestrator) | ~450 |
| Integration Tests | End-to-end protocol validation | ~400 |

### 1.2 Design Philosophy

Phase E adheres to Caligo's simulation-native design principles while introducing orchestration-specific patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE E DESIGN PRINCIPLES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SQUIDASM-NATIVE PROGRAMS                                                │
│     └── Extend squidasm.sim.stack.program.Program base class                │
│     └── Implement run() as generator yielding EventExpressions              │
│     └── Use ProgramContext for csockets, epr_sockets, connection access     │
│                                                                             │
│  2. GENERATOR-BASED EXECUTION MODEL                                         │
│     └── All I/O operations (quantum and classical) yield control            │
│     └── NetSquid scheduler resumes execution at appropriate sim_time        │
│     └── No blocking calls — discrete-event semantics throughout             │
│                                                                             │
│  3. SECURITY-FIRST MESSAGING                                                │
│     └── Ordered message delivery with ACK enforcement                       │
│     └── Commit-then-reveal semantics are non-negotiable                     │
│     └── Timing barrier integration with Phase B TimingBarrier               │
│                                                                             │
│  4. PHASE CONTRACT ENFORCEMENT                                              │
│     └── Phase transitions validated via Phase A contracts                   │
│     └── Security checks gate progression (Phase C FeasibilityChecker)       │
│     └── Abort on constraint violations with structured error codes          │
│                                                                             │
│  5. MINIMAL COUPLING TO ehok                                                │
│     └── Extract proven patterns, not copy-paste                             │
│     └── Refactor for Caligo's ≤200 LOC module constraint                    │
│     └── Replace ehok's factory indirection with direct imports              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Integration Points

Phase E integrates all preceding phases into a cohesive execution framework:

| Phase | Components Used | Purpose in Phase E |
|-------|-----------------|-------------------|
| **Phase A** | `types/phase_contracts.py` | `QuantumPhaseResult`, `SiftingPhaseResult`, `AmplificationPhaseResult` |
| **Phase A** | `types/exceptions.py` | `SecurityError`, `OrderingViolationError`, `TimingViolationError` |
| **Phase A** | `types/keys.py` | `AliceObliviousKey`, `BobObliviousKey` |
| **Phase B** | `simulation/timing.py` | `TimingBarrier` for NSM Δt enforcement |
| **Phase B** | `simulation/physical_model.py` | `PhysicalChannelParameters` for feasibility |
| **Phase C** | `security/feasibility.py` | Pre-flight `FeasibilityChecker` |
| **Phase C** | `security/bounds.py` | Runtime QBER validation |
| **Phase D** | `quantum/*` | EPR generation, basis selection, measurement |
| **Phase D** | `sifting/*` | Commitment, sifting, QBER estimation |
| **Phase D** | `amplification/*` | Privacy amplification, OT formatting |

### 1.4 SquidASM Execution Model

Understanding SquidASM's execution model is essential for Phase E implementation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SQUIDASM PROGRAM EXECUTION FLOW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  ProgramContext  │   Provided by SquidASM runtime to run()               │
│  ├──────────────────┤                                                       │
│  │ .connection      │ ─► BaseNetQASMConnection: quantum instruction commit  │
│  │ .csockets        │ ─► Dict[str, Socket]: classical messaging             │
│  │ .epr_sockets     │ ─► Dict[str, EPRSocket]: EPR pair generation          │
│  │ .app_id          │ ─► int: unique application identifier                 │
│  └──────────────────┘                                                       │
│                                                                             │
│  GENERATOR PROTOCOL                                                         │
│  ──────────────────                                                         │
│  def run(self, context: ProgramContext) -> Generator[EventExpression, ...]: │
│      # 1. Classical send (non-blocking)                                     │
│      context.csockets["peer"].send(msg)                                     │
│                                                                             │
│      # 2. Classical receive (yields control until message arrives)          │
│      response = yield from context.csockets["peer"].recv()                  │
│                                                                             │
│      # 3. EPR generation (yields control during quantum operations)         │
│      epr_pairs = yield from context.epr_sockets["peer"].create_keep(n=100)  │
│                                                                             │
│      # 4. Flush connection (yields control until quantum ops complete)      │
│      yield from context.connection.flush()                                  │
│                                                                             │
│      return {"result": final_key}  # Dict returned to simulation runner     │
│                                                                             │
│  KEY INSIGHT: Every I/O operation uses `yield from` to allow NetSquid's     │
│  discrete-event scheduler to advance simulation time appropriately.         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Critical Protocol Flow (Phase E Scope)

Phase E orchestrates the complete E-HOK protocol execution:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    E-HOK ORCHESTRATION FLOW (Phase E)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ALICE                                          BOB                         │
│  ═════                                          ═══                         │
│                                                                             │
│  ┌─── Phase I (quantum/) ────────────────────────────────────────────┐      │
│  │ EPR Generation + Basis Selection + Measurement (Parallel)         │      │
│  │ ────────────────────────────────────────────────────────────────  │      │
│  │ [A] generate_bases()         [B] generate_bases()                 │      │
│  │ [A] yield from epr_socket.create_keep(n)                          │      │
│  │                              [B] yield from epr_socket.recv_keep() │      │
│  │ [A] measure_qubits()         [B] measure_qubits()                 │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                            │                                                │
│                            ▼                                                │
│  ┌─── Phase II (connection/ + sifting/) ─────────────────────────────┐      │
│  │                                                                   │      │
│  │      ┌──────────────────────────────────────────┐                 │      │
│  │      │ COMMIT-THEN-REVEAL (Security Critical)   │                 │      │
│  │      └──────────────────────────────────────────┘                 │      │
│  │                                                                   │      │
│  │ [B] commitment = SHA256(outcomes || bases || salt)                │      │
│  │ [B] send_with_ack(commitment)  ──────────►  [A] recv_and_ack()    │      │
│  │                                                                   │      │
│  │      ════════════════════════════════════════════════════════     │      │
│  │      TIMING BARRIER: Alice waits Δt (Phase B TimingBarrier)       │      │
│  │      ════════════════════════════════════════════════════════     │      │
│  │                                                                   │      │
│  │ [A] send_with_ack(bases)       ──────────►  [B] recv_and_ack()    │      │
│  │ [B] verify_commitment()                                           │      │
│  │ [A,B] sift_keys() + test_sampling() + estimate_qber()             │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                            │                                                │
│                            ▼                                                │
│  ┌─── Phase III (reconciliation/) ───────────────────────────────────┐      │
│  │ LDPC-based error correction (existing ehok implementation)        │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                            │                                                │
│                            ▼                                                │
│  ┌─── Phase IV (amplification/) ─────────────────────────────────────┐      │
│  │ [A] compute_entropy_bound()                                       │      │
│  │ [A] compute_key_length()                                          │      │
│  │ [A] generate_toeplitz_seed()                                      │      │
│  │ [A] send(toeplitz_seed)        ──────────►  [B] recv(seed)        │      │
│  │ [A] S_0, S_1 = hash(key_by_basis)                                 │      │
│  │                                [B] S_C = hash(key_for_choice)     │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  OUTPUT: Alice(S_0, S_1) | Bob(S_C, choice_bit)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.6 Key Insights from ehok Analysis

Analysis of the ehok codebase reveals patterns to adopt and anti-patterns to avoid:

| ehok Component | Insight | Caligo Approach |
|----------------|---------|-----------------|
| `protocols/base.py` | Template method pattern for phase sequencing | Adopt: `CaligoProgram` base class |
| `protocols/ordered_messaging.py` | 1022 LOC with comprehensive state machine | Simplify: Split into envelope + socket modules |
| `protocols/alice.py`, `bob.py` | Role-specific phase implementations | Adopt: Symmetric Alice/Bob programs |
| Factory indirection | 5 factories for single implementations | Eliminate: Direct imports |
| `EHOKRole.meta` property | ProgramMeta declaration for SquidASM | Adopt: Required for SquidASM integration |
| `_execute_remaining_phases()` | Generator-based phase orchestration | Adopt: Core execution pattern |

---

## 2. Scope & Deliverables

### 2.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `caligo/connection/envelope.py` | Message envelope with sequence tracking |
| `caligo/connection/ordered_socket.py` | ACK-enforced ordered messaging |
| `caligo/connection/__init__.py` | Package exports |
| `caligo/protocol/alice.py` | Alice SquidASM program |
| `caligo/protocol/bob.py` | Bob SquidASM program |
| `caligo/protocol/orchestrator.py` | Phase sequencing and validation |
| `caligo/protocol/__init__.py` | Package exports |
| `tests/e2e/` | End-to-end integration tests |

### 2.2 Out of Scope

- Phase D protocol implementations (imported from `caligo.quantum`, `caligo.sifting`, etc.)
- Network topology configuration (Phase B `network_builder.py`)
- Security bound calculations (Phase C `security/bounds.py`)
- Type definitions (Phase A)

### 2.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| SquidASM | ≥0.12 | `Program`, `ProgramContext`, `ProgramMeta` |
| NetQASM | ≥0.10 | `EPRSocket`, `Socket`, `BaseNetQASMConnection` |
| pydynaa | ≥0.1 | `EventExpression` for generator yields |
| Phase A-D | — | All Caligo foundation packages |

---

## 3. SquidASM Integration Foundations

### 3.1 Program Interface Contract

All Caligo protocol programs must implement the SquidASM `Program` interface:

```python
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

class CaligoProgram(Program):
    """
    Abstract base for Caligo protocol programs.
    
    Subclasses must implement:
    - meta property: Declare classical and EPR socket requirements
    - run() generator: Execute protocol phases yielding at I/O points
    """
    
    @property
    def meta(self) -> ProgramMeta:
        """
        Declare program resource requirements.
        
        Returns
        -------
        ProgramMeta
            name: str — Program identifier
            csockets: List[str] — Peer names for classical communication
            epr_sockets: List[str] — Peer names for EPR distribution
            max_qubits: int — Maximum qubits needed simultaneously
        """
        raise NotImplementedError
    
    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        """
        Execute protocol as a generator.
        
        Parameters
        ----------
        context : ProgramContext
            SquidASM-provided execution context with:
            - connection: NetQASM connection for quantum operations
            - csockets: Dict[peer_name, ClassicalSocket]
            - epr_sockets: Dict[peer_name, EPRSocket]
            - app_id: Unique application identifier
        
        Yields
        ------
        EventExpression
            Control yielded to NetSquid scheduler during I/O.
        
        Returns
        -------
        Dict[str, Any]
            Protocol outputs (keys, metrics, abort info).
        """
        raise NotImplementedError
```

### 3.2 ClassicalSocket API

SquidASM's `ClassicalSocket` provides the foundation for all classical communication:

| Method | Signature | Behavior |
|--------|-----------|----------|
| `send` | `send(msg: str) -> None` | Non-blocking string send |
| `recv` | `recv() -> Generator[..., str]` | Yields until message received |
| `send_int` | `send_int(value: int) -> None` | Send integer as string |
| `recv_int` | `recv_int() -> Generator[..., int]` | Receive and parse integer |
| `send_float` | `send_float(value: float) -> None` | Send float as string |
| `recv_float` | `recv_float() -> Generator[..., float]` | Receive and parse float |

**Critical Pattern:** All receive operations are generators requiring `yield from`:

```python
# CORRECT: Generator delegation
message = yield from context.csockets["peer"].recv()

# INCORRECT: Returns generator object, not result
message = context.csockets["peer"].recv()  # Bug!
```

### 3.3 Connection Flush Pattern

Quantum operations are batched and must be explicitly flushed:

```python
# Generate EPR pairs (queued)
epr_pairs = yield from context.epr_sockets["peer"].create_keep(num=100)

# Apply gates and measurements (queued)
for qubit in epr_pairs:
    qubit.H()  # Hadamard in X basis
    m = qubit.measure()

# Commit all queued operations to NetSquid simulation
yield from context.connection.flush()

# Now measurement results are available
outcomes = [int(m) for m in measurements]
```

---

## 4. Package: `connection/` — Ordered Messaging

### 4.1 Theoretical Foundation

**Source:** König et al. (2012), Section 3; ehok Sprint 2 Specification

The commit-then-reveal ordering is **fundamental to NSM security**. If Bob receives Alice's basis information before his detection report is acknowledged, he can:

1. Check his quantum storage for each basis
2. Claim "missing" only for rounds where storage failed
3. Effectively post-select a lower-noise sub-key, breaking security

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMMIT-THEN-REVEAL SECURITY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ATTACK SCENARIO (Without Ordering Enforcement)                             │
│  ───────────────────────────────────────────────                            │
│                                                                             │
│  1. Bob receives Alice's bases EARLY                                        │
│  2. Bob checks quantum storage for each basis                               │
│  3. For failed storage → claim "no detection" (post-select)                 │
│  4. Result: Bob's key has artificially LOW noise                            │
│  5. Security guarantee BROKEN: C_N · ν < 1/2 no longer holds                │
│                                                                             │
│  DEFENSE (Ordered Messaging with ACK)                                       │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  1. Bob COMMITS to detection report (hash)                                  │
│  2. Bob sends commitment → Alice ACKs                                       │
│  3. TIMING BARRIER: Alice waits Δt                                          │
│  4. Alice reveals bases ONLY AFTER ACK received                             │
│  5. Bob cannot change detection claims post-hoc                             │
│                                                                             │
│  The ACK ensures causal ordering in the discrete-event simulation.          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module: `envelope.py` (~80 LOC)

**Purpose:** Message envelope with sequence tracking and type discrimination.

#### 4.2.1 `MessageType` Enum

```python
class MessageType(Enum):
    """
    Protocol message type discriminators.
    
    These identify message semantics for ordered delivery validation.
    
    Phase II Messages
    -----------------
    DETECTION_COMMITMENT : Bob's commitment to detection report
    BASIS_REVEAL : Alice's basis choices after timing barrier
    COMMITMENT_OPENING : Bob's decommitment (salt + original data)
    
    Sifting Messages
    ----------------
    INDEX_LISTS : Sifted index sets (I_0, I_1, test_set, key_set)
    TEST_OUTCOMES : Test set measurement outcomes for QBER
    
    Reconciliation Messages
    -----------------------
    SYNDROME : Alice's LDPC syndrome for error correction
    SYNDROME_RESPONSE : Bob's parity check results
    
    Amplification Messages
    ----------------------
    TOEPLITZ_SEED : Random seed for privacy amplification hash
    
    Control Messages
    ----------------
    ACK : Acknowledgment for ordered delivery
    ABORT : Protocol abort with reason code
    """
    
    # Phase II
    DETECTION_COMMITMENT = "DETECTION_COMMITMENT"
    BASIS_REVEAL = "BASIS_REVEAL"
    COMMITMENT_OPENING = "COMMITMENT_OPENING"
    
    # Sifting
    INDEX_LISTS = "INDEX_LISTS"
    TEST_OUTCOMES = "TEST_OUTCOMES"
    
    # Reconciliation
    SYNDROME = "SYNDROME"
    SYNDROME_RESPONSE = "SYNDROME_RESPONSE"
    
    # Amplification
    TOEPLITZ_SEED = "TOEPLITZ_SEED"
    
    # Control
    ACK = "ACK"
    ABORT = "ABORT"
```

#### 4.2.2 `MessageEnvelope` Dataclass

```python
@dataclass(frozen=True)
class MessageEnvelope:
    """
    Envelope for ordered protocol messages.
    
    All ordered messages are wrapped in this envelope to enable
    sequence tracking, session binding, and acknowledgment.
    
    Attributes
    ----------
    session_id : str
        Opaque identifier binding messages to a single protocol run.
        Generated as UUID at session start.
    seq : int
        Monotonically increasing sequence number per direction.
        Starts at 0 for each party, increments per sent message.
    msg_type : MessageType
        Discriminator identifying message semantics.
    payload : Dict[str, Any]
        The message payload (serialized as JSON).
    
    Serialization
    -------------
    Uses JSON for transport over ClassicalSocket. Binary payloads
    (numpy arrays) are hex-encoded before inclusion.
    
    Example
    -------
    >>> envelope = MessageEnvelope(
    ...     session_id="abc123",
    ...     seq=0,
    ...     msg_type=MessageType.DETECTION_COMMITMENT,
    ...     payload={"commitment": "a1b2c3...", "round_count": 10000}
    ... )
    >>> json_str = envelope.to_json()
    >>> recovered = MessageEnvelope.from_json(json_str)
    """
    
    session_id: str
    seq: int
    msg_type: MessageType
    payload: Dict[str, Any]
    
    def to_json(self) -> str:
        """
        Serialize envelope to JSON string.
        
        Returns
        -------
        str
            JSON-encoded envelope for ClassicalSocket transmission.
        """
    
    @classmethod
    def from_json(cls, data: str) -> "MessageEnvelope":
        """
        Deserialize envelope from JSON string.
        
        Parameters
        ----------
        data : str
            JSON-encoded envelope.
        
        Returns
        -------
        MessageEnvelope
            Reconstructed envelope.
        
        Raises
        ------
        ValueError
            If JSON is malformed or missing required fields.
        """
```

#### 4.2.3 `AckPayload` Dataclass

```python
@dataclass(frozen=True)
class AckPayload:
    """
    Acknowledgment payload for ordered messages.
    
    Attributes
    ----------
    ack_seq : int
        Sequence number of the message being acknowledged.
    ack_msg_type : MessageType
        Type of message being acknowledged (defensive validation).
    
    Notes
    -----
    Including msg_type in ACK provides defense-in-depth against
    sequence number collision attacks.
    """
    
    ack_seq: int
    ack_msg_type: MessageType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for envelope payload."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AckPayload":
        """Reconstruct from dictionary."""
```

### 4.3 Module: `ordered_socket.py` (~150 LOC)

**Purpose:** Socket wrapper enforcing ordered delivery with ACK.

#### 4.3.1 `SocketState` Enum

```python
class SocketState(Enum):
    """
    State machine states for OrderedSocket.
    
    State Transitions
    -----------------
    IDLE → SENT_WAIT_ACK : After send_with_ack() sends message
    SENT_WAIT_ACK → IDLE : After valid ACK received
    SENT_WAIT_ACK → VIOLATION : On ACK timeout or invalid ACK
    IDLE → VIOLATION : On session mismatch or sequence error
    VIOLATION : Terminal state (no recovery)
    
    Notes
    -----
    The VIOLATION state is terminal. Once entered, the socket cannot
    be used for further communication. This enforces fail-fast semantics
    for security-critical ordering violations.
    """
    
    IDLE = auto()
    SENT_WAIT_ACK = auto()
    VIOLATION = auto()
```

#### 4.3.2 State Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORDERED SOCKET STATE MACHINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌──────────────┐                                     │
│                        │     IDLE     │◄─────────────────────────┐          │
│                        └──────┬───────┘                          │          │
│                               │                                  │          │
│                    send_with_ack()                      valid ACK received  │
│                               │                                  │          │
│                               ▼                                  │          │
│                   ┌───────────────────────┐                      │          │
│                   │    SENT_WAIT_ACK      │──────────────────────┘          │
│                   └───────────┬───────────┘                                 │
│                               │                                             │
│              timeout / invalid ACK / session mismatch                       │
│                               │                                             │
│                               ▼                                             │
│                   ┌───────────────────────┐                                 │
│                   │      VIOLATION        │  (Terminal - No Recovery)       │
│                   └───────────────────────┘                                 │
│                                                                             │
│  On VIOLATION: Raise OrderingViolationError, abort protocol                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.3.3 `OrderedSocket` Class

```python
class OrderedSocket:
    """
    Socket wrapper enforcing ordered message delivery with acknowledgments.
    
    This class provides send-with-ACK and receive-and-ACK semantics to
    enforce the commit-then-reveal ordering required for NSM security.
    
    Usage Pattern
    -------------
    The ordered socket wraps a SquidASM ClassicalSocket and provides
    generator-based methods compatible with the SquidASM execution model:
    
    # In Alice's run() generator:
    ordered = OrderedSocket(session_id="...")
    
    # Send with ACK (blocks until ACK received)
    yield from ordered.send_with_ack(
        socket=context.csockets["bob"],
        msg_type=MessageType.BASIS_REVEAL,
        payload={"bases": bases.tobytes().hex()}
    )
    
    # Receive and auto-ACK
    envelope = yield from ordered.recv_and_ack(
        socket=context.csockets["bob"]
    )
    commitment = bytes.fromhex(envelope.payload["commitment"])
    
    Thread Safety
    -------------
    NOT thread-safe. Each protocol execution should use its own instance.
    
    Attributes
    ----------
    session_id : str
        Unique identifier for this protocol session.
    state : SocketState
        Current state machine state.
    send_seq : int
        Next sequence number for outgoing messages.
    recv_seq : int
        Next expected sequence number for incoming messages.
    
    References
    ----------
    - ehok/protocols/ordered_messaging.py
    - König et al. (2012): Commit-then-reveal semantics
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize ordered socket.
        
        Parameters
        ----------
        session_id : str, optional
            Session identifier. If None, generates UUID.
        """
    
    def send_with_ack(
        self,
        socket: "ClassicalSocket",
        msg_type: MessageType,
        payload: Dict[str, Any],
        timeout_ns: int = 5_000_000_000
    ) -> Generator[EventExpression, None, None]:
        """
        Send message and block until ACK received.
        
        This is the primary method for enforcing ordered delivery.
        The sender cannot proceed until the receiver acknowledges.
        
        Parameters
        ----------
        socket : ClassicalSocket
            SquidASM classical socket.
        msg_type : MessageType
            Type of message being sent.
        payload : Dict[str, Any]
            Message payload.
        timeout_ns : int
            Maximum nanoseconds to wait for ACK. Default: 5 seconds.
        
        Yields
        ------
        EventExpression
            Control yielded to NetSquid during socket I/O.
        
        Raises
        ------
        AckTimeoutError
            If ACK not received within timeout.
        OrderingViolationError
            If socket not in IDLE state, or ACK validation fails.
        
        Security Note
        -------------
        This method enforces commit-then-reveal by blocking until
        the receiver has acknowledged. The sender's subsequent
        actions (e.g., revealing bases) are causally ordered after
        the receiver's commitment.
        """
    
    def recv_and_ack(
        self,
        socket: "ClassicalSocket"
    ) -> Generator[EventExpression, None, MessageEnvelope]:
        """
        Receive message and automatically send ACK.
        
        Parameters
        ----------
        socket : ClassicalSocket
            SquidASM classical socket.
        
        Yields
        ------
        EventExpression
            Control yielded to NetSquid during socket I/O.
        
        Returns
        -------
        MessageEnvelope
            The received (and acknowledged) message.
        
        Raises
        ------
        OutOfOrderError
            If message sequence doesn't match expected.
        OrderingViolationError
            If session mismatch or socket in VIOLATION state.
        """
    
    def create_envelope(
        self,
        msg_type: MessageType,
        payload: Dict[str, Any]
    ) -> MessageEnvelope:
        """
        Create envelope with current sequence number.
        
        For direct socket operations when ACK is not required
        (e.g., final message in a phase).
        """
```

### 4.4 Module: `exceptions.py` (~40 LOC)

**Purpose:** Connection-specific exception hierarchy.

```python
class ConnectionError(CaligoError):
    """Base exception for connection layer errors."""
    pass

class OrderingViolationError(ConnectionError):
    """
    Message ordering constraint violated.
    
    This is a security-critical error indicating that the
    commit-then-reveal semantics cannot be guaranteed.
    Protocol must abort immediately.
    
    Abort Code: ABORT-CONN-ORDER-001
    """
    pass

class AckTimeoutError(ConnectionError):
    """
    ACK not received within timeout period.
    
    The sender cannot verify that the receiver has committed
    to their state before proceeding. Protocol must abort.
    
    Abort Code: ABORT-CONN-TIMEOUT-001
    """
    pass

class SessionMismatchError(ConnectionError):
    """
    Message received from different session.
    
    Indicates either a bug or an active attack attempting
    to inject messages from a different protocol run.
    
    Abort Code: ABORT-CONN-SESSION-001
    """
    pass

class OutOfOrderError(ConnectionError):
    """
    Message sequence number out of expected order.
    
    Strict ordering is enforced — out-of-order messages
    are rejected and the protocol aborts.
    
    Abort Code: ABORT-CONN-SEQ-001
    """
    pass
```

### 4.5 Connection Package Contract

```python
# caligo/connection/__init__.py

"""
Connection package for ordered protocol messaging.

This package implements the commit-then-reveal semantics required
for NSM security. All protocol messages with ordering requirements
should use OrderedSocket rather than raw ClassicalSocket.

Exports
-------
MessageType : Enum
    Protocol message type discriminators.
MessageEnvelope : dataclass
    Message wrapper with sequence tracking.
AckPayload : dataclass
    Acknowledgment payload structure.
OrderedSocket : class
    ACK-enforced ordered messaging.

Exceptions
----------
OrderingViolationError : exception
    Message ordering violated (abort).
AckTimeoutError : exception
    ACK timeout (abort).
SessionMismatchError : exception
    Session ID mismatch (abort).
OutOfOrderError : exception
    Sequence number error (abort).
"""

from caligo.connection.envelope import (
    MessageType,
    MessageEnvelope,
    AckPayload,
)
from caligo.connection.ordered_socket import (
    SocketState,
    OrderedSocket,
)
from caligo.connection.exceptions import (
    ConnectionError,
    OrderingViolationError,
    AckTimeoutError,
    SessionMismatchError,
    OutOfOrderError,
)

__all__ = [
    "MessageType",
    "MessageEnvelope",
    "AckPayload",
    "SocketState",
    "OrderedSocket",
    "ConnectionError",
    "OrderingViolationError",
    "AckTimeoutError",
    "SessionMismatchError",
    "OutOfOrderError",
]
```

---

## 5. Package: `protocol/` — SquidASM Programs

### 5.1 Design Overview

The protocol package implements the complete E-HOK workflow as SquidASM-compatible programs. The design uses a **template method pattern** where a base class defines the phase sequencing, and role-specific subclasses (Alice, Bob) implement phase behaviors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROTOCOL PACKAGE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      ┌─────────────────────────┐                            │
│                      │  squidasm.Program       │  (SquidASM interface)      │
│                      └───────────┬─────────────┘                            │
│                                  │                                          │
│                                  ▼                                          │
│                      ┌─────────────────────────┐                            │
│                      │    CaligoProgram        │  (Template base class)     │
│                      │  ──────────────────     │                            │
│                      │  + meta property        │                            │
│                      │  + run() generator      │                            │
│                      │  + _phase1_quantum()    │                            │
│                      │  + _phase2_sifting()    │   (abstract)               │
│                      │  + _phase3_recon()      │   (abstract)               │
│                      │  + _phase4_amplify()    │   (abstract)               │
│                      └───────────┬─────────────┘                            │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              │                                       │                      │
│              ▼                                       ▼                      │
│  ┌─────────────────────────┐         ┌─────────────────────────┐            │
│  │    AliceProgram         │         │     BobProgram          │            │
│  │  ──────────────────     │         │  ──────────────────     │            │
│  │  PEER = "bob"           │         │  PEER = "alice"         │            │
│  │  + _phase2_sifting()    │         │  + _phase2_sifting()    │            │
│  │    (receive commitment) │         │    (send commitment)    │            │
│  │  + _phase4_amplify()    │         │  + _phase4_amplify()    │            │
│  │    (produce S_0, S_1)   │         │    (produce S_C)        │            │
│  └─────────────────────────┘         └─────────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Module: `base.py` (~180 LOC)

**Purpose:** Template base class defining phase orchestration.

#### 5.2.1 `CaligoProgram` Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from pydynaa import EventExpression
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

from caligo.connection import OrderedSocket
from caligo.simulation.timing import TimingBarrier
from caligo.types.phase_contracts import (
    QuantumPhaseResult,
    SiftingPhaseResult,
    ReconciliationPhaseResult,
    AmplificationPhaseResult,
)
from caligo.utils.logging import get_logger


class CaligoProgram(Program, ABC):
    """
    Template base class for Caligo E-HOK protocol programs.
    
    This class implements the Template Method pattern, defining the
    high-level protocol flow while delegating role-specific behavior
    to Alice/Bob subclasses.
    
    Phase Sequencing
    ----------------
    The run() method orchestrates the four protocol phases:
    
    1. Phase I (Quantum): EPR generation, basis selection, measurement
       - Symmetric: Both Alice and Bob execute the same logic
       - Implemented in base class
    
    2. Phase II (Sifting): Commitment, basis reveal, key extraction
       - Asymmetric: Alice receives commitment, Bob sends commitment
       - Implemented in subclasses
    
    3. Phase III (Reconciliation): LDPC error correction
       - Asymmetric: Alice decodes, Bob sends syndrome
       - Implemented in subclasses
    
    4. Phase IV (Amplification): Privacy amplification, OT output
       - Asymmetric: Alice produces (S_0, S_1), Bob produces (S_C, C)
       - Implemented in subclasses
    
    Dependency Injection
    --------------------
    Security components are injected to enable/disable enforcement:
    
    - OrderedSocket: Enforces commit-then-reveal message ordering
    - TimingBarrier: Enforces NSM timing constraint Δt
    
    By default, these are NOT created automatically. Tests can run
    without timing constraints, while production code injects
    fully-configured enforcers.
    
    Attributes
    ----------
    PEER : str
        Name of peer node (subclass must define).
    ROLE : str
        This node's role ("alice" or "bob").
    config : ProtocolConfig
        Protocol configuration parameters.
    ordered_socket : OrderedSocket, optional
        Injected ordered messaging socket.
    timing_barrier : TimingBarrier, optional
        Injected NSM timing enforcer.
    
    References
    ----------
    - ehok/protocols/base.py: EHOKRole template class
    - SquidASM Program interface
    """
    
    PEER: str  # Subclass must define
    ROLE: str  # Subclass must define
    
    def __init__(
        self,
        config: "ProtocolConfig",
        ordered_socket: Optional[OrderedSocket] = None,
        timing_barrier: Optional[TimingBarrier] = None,
    ):
        """
        Initialize Caligo program.
        
        Parameters
        ----------
        config : ProtocolConfig
            Protocol configuration.
        ordered_socket : OrderedSocket, optional
            Pre-configured ordered socket. If None, creates default.
        timing_barrier : TimingBarrier, optional
            Pre-configured timing enforcer. If None, timing is NOT enforced.
        """
        self.config = config
        self._ordered_socket = ordered_socket
        self._timing_barrier = timing_barrier
        self._context: Optional[ProgramContext] = None
    
    @property
    def meta(self) -> ProgramMeta:
        """
        Declare program resource requirements.
        
        Returns
        -------
        ProgramMeta
            SquidASM program metadata.
        """
        return ProgramMeta(
            name=f"caligo_{self.ROLE}",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self.config.quantum.max_qubits,
        )
    
    @property
    def context(self) -> ProgramContext:
        """Access program context (available after run() starts)."""
        if self._context is None:
            raise RuntimeError("Context not available before run()")
        return self._context
    
    @property
    def csocket(self) -> "ClassicalSocket":
        """Shorthand for peer classical socket."""
        return self.context.csockets[self.PEER]
    
    @property
    def epr_socket(self) -> "EPRSocket":
        """Shorthand for peer EPR socket."""
        return self.context.epr_sockets[self.PEER]
    
    @property
    def connection(self) -> "BaseNetQASMConnection":
        """Shorthand for NetQASM connection."""
        return self.context.connection
    
    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        """
        Execute E-HOK protocol as SquidASM generator.
        
        This is the main entry point called by SquidASM runtime.
        
        Parameters
        ----------
        context : ProgramContext
            SquidASM-provided execution context.
        
        Yields
        ------
        EventExpression
            Control yielded to NetSquid during I/O operations.
        
        Returns
        -------
        Dict[str, Any]
            Protocol results including:
            - success: bool
            - oblivious_key: Alice/BobObliviousKey or None
            - qber: float
            - metrics: Dict[str, Any]
            - abort_reason: str (if success=False)
        """
        self._context = context
        self._setup_dependencies()
        logger = get_logger(f"protocol.{self.ROLE}")
        logger.info("Starting %s E-HOK protocol", self.ROLE)
        
        try:
            # Phase I: Quantum (symmetric)
            quantum_result = yield from self._phase1_quantum()
            
            # Phase II: Sifting (asymmetric)
            sifting_result = yield from self._phase2_sifting(quantum_result)
            
            # Phase III: Reconciliation (asymmetric)
            recon_result = yield from self._phase3_reconciliation(sifting_result)
            
            # Phase IV: Amplification (asymmetric)
            final_result = yield from self._phase4_amplification(recon_result)
            
            return self._format_success(final_result)
            
        except SecurityError as e:
            logger.error("Protocol aborted: %s", e)
            return self._format_abort(str(e))
    
    def _setup_dependencies(self) -> None:
        """Initialize injected dependencies after context available."""
        if self._ordered_socket is None:
            self._ordered_socket = OrderedSocket()
    
    def _phase1_quantum(
        self
    ) -> Generator[EventExpression, None, QuantumPhaseResult]:
        """
        Execute Phase I: Quantum generation and measurement.
        
        This phase is symmetric — both Alice and Bob execute
        identical logic, differing only in EPR socket role
        (create_keep vs recv_keep).
        
        Yields
        ------
        EventExpression
            During EPR generation and measurement flush.
        
        Returns
        -------
        QuantumPhaseResult
            Outcomes, bases, timestamps for all measured qubits.
        """
        # Implementation delegates to Phase D quantum/ package
        from caligo.quantum import QuantumPhaseRunner
        
        runner = QuantumPhaseRunner(
            context=self.context,
            peer_name=self.PEER,
            role=self.ROLE,
            config=self.config,
        )
        return (yield from runner.run())
    
    @abstractmethod
    def _phase2_sifting(
        self, quantum_result: QuantumPhaseResult
    ) -> Generator[EventExpression, None, SiftingPhaseResult]:
        """
        Execute Phase II: Commitment, sifting, QBER estimation.
        
        Alice: Receive commitment → timing barrier → reveal bases
        Bob: Send commitment → receive bases → verify
        
        Must be implemented by Alice/Bob subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _phase3_reconciliation(
        self, sifting_result: SiftingPhaseResult
    ) -> Generator[EventExpression, None, ReconciliationPhaseResult]:
        """
        Execute Phase III: LDPC error correction.
        
        Alice: Receive syndrome → decode → verify hash
        Bob: Compute syndrome → send → receive verification
        
        Must be implemented by Alice/Bob subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _phase4_amplification(
        self, recon_result: ReconciliationPhaseResult
    ) -> Generator[EventExpression, None, AmplificationPhaseResult]:
        """
        Execute Phase IV: Privacy amplification.
        
        Alice: Compute key length → generate seed → hash to (S_0, S_1)
        Bob: Receive seed → hash to S_C
        
        Must be implemented by Alice/Bob subclasses.
        """
        raise NotImplementedError
    
    def _format_success(
        self, result: AmplificationPhaseResult
    ) -> Dict[str, Any]:
        """Format successful protocol completion."""
        return {
            "success": True,
            "oblivious_key": result.oblivious_key,
            "qber": result.qber,
            "final_length": result.key_length,
            "metrics": result.metrics,
            "role": self.ROLE,
        }
    
    def _format_abort(self, reason: str) -> Dict[str, Any]:
        """Format protocol abort."""
        return {
            "success": False,
            "abort_reason": reason,
            "oblivious_key": None,
            "role": self.ROLE,
        }
```

### 5.3 Module: `alice.py` (~150 LOC)

**Purpose:** Alice's E-HOK protocol implementation.

#### 5.3.1 Alice's Role in E-HOK

Alice has specific responsibilities in each phase:

| Phase | Alice's Actions |
|-------|-----------------|
| Phase I | Create EPR pairs (`create_keep`) |
| Phase II | Receive Bob's commitment → Wait Δt → Reveal bases |
| Phase III | Receive syndrome → LDPC decode → Send verification |
| Phase IV | Compute key length → Generate seed → Produce (S_0, S_1) |

#### 5.3.2 `AliceProgram` Class

```python
class AliceProgram(CaligoProgram):
    """
    Alice's E-HOK protocol implementation.
    
    Alice is the "sender" in the 1-out-of-2 OT terminology. She:
    - Creates EPR pairs (initiator role)
    - Receives Bob's commitment before revealing bases
    - Produces TWO keys: S_0 and S_1
    - Does NOT learn which key Bob obtained
    
    Security Properties
    -------------------
    - Receiver Privacy: Alice cannot determine Bob's choice bit C
    - Sender Privacy: Ensured by NSM — Bob cannot learn S_{1-C}
    
    Timing Barrier
    --------------
    Alice MUST wait Δt after receiving Bob's commitment before
    revealing her bases. This is the critical NSM security mechanism.
    The TimingBarrier enforces: t_reveal - t_commit ≥ Δt
    
    Attributes
    ----------
    PEER : str = "bob"
    ROLE : str = "alice"
    """
    
    PEER = "bob"
    ROLE = "alice"
    
    def _phase2_sifting(
        self, quantum_result: QuantumPhaseResult
    ) -> Generator[EventExpression, None, SiftingPhaseResult]:
        """
        Alice's Phase II: Receive commitment, wait, reveal bases.
        
        Sequence
        --------
        1. Receive Bob's commitment (ordered: recv_and_ack)
        2. Mark commitment received (start timing barrier)
        3. WAIT Δt (TimingBarrier enforcement)
        4. Reveal bases (ordered: send_with_ack)
        5. Receive Bob's decommitment and outcomes
        6. Verify commitment
        7. Sift keys and estimate QBER
        
        Security
        --------
        The timing barrier between steps 2-4 is the core NSM mechanism.
        Without it, Bob could post-select based on storage success.
        """
        logger = get_logger("protocol.alice.phase2")
        logger.info("=== PHASE II: Sifting (Alice) ===")
        
        # 1. Receive commitment (with ACK)
        commitment_envelope = yield from self._ordered_socket.recv_and_ack(
            self.csocket
        )
        commitment = bytes.fromhex(
            commitment_envelope.payload["commitment"]
        )
        logger.info("Received commitment from Bob")
        
        # 2. Mark quantum phase complete (if not already done)
        # Note: Typically marked at end of Phase I, but commitment receipt
        # can also serve as the reference point for Δt calculation
        if self._timing_barrier is not None:
            self._timing_barrier.mark_quantum_complete()
        
        # 3. WAIT Δt — timing barrier enforcement
        if self._timing_barrier is not None:
            yield from self._timing_barrier.wait_delta_t()
            self._timing_barrier.assert_timing_compliant()
            logger.info("Timing barrier (Δt) satisfied")
        
        # 4. Reveal bases (with ACK)
        yield from self._ordered_socket.send_with_ack(
            socket=self.csocket,
            msg_type=MessageType.BASIS_REVEAL,
            payload={"bases": quantum_result.bases.tobytes().hex()}
        )
        logger.info("Revealed bases to Bob")
        
        # 5. Receive Bob's decommitment
        decommit_envelope = yield from self._ordered_socket.recv_and_ack(
            self.csocket
        )
        bob_outcomes = np.frombuffer(
            bytes.fromhex(decommit_envelope.payload["outcomes"]),
            dtype=np.uint8
        )
        bob_bases = np.frombuffer(
            bytes.fromhex(decommit_envelope.payload["bases"]),
            dtype=np.uint8
        )
        salt = bytes.fromhex(decommit_envelope.payload["salt"])
        
        # 6. Verify commitment
        from caligo.sifting import CommitmentScheme
        if not CommitmentScheme.verify(
            commitment,
            np.concatenate([bob_outcomes, bob_bases]),
            salt
        ):
            raise SecurityError("Commitment verification failed")
        logger.info("Commitment verified successfully")
        
        # 7. Sift keys and estimate QBER
        from caligo.sifting import SiftingManager, QBEREstimator
        sifter = SiftingManager()
        I_0, I_1 = sifter.identify_matching_bases(
            quantum_result.bases, bob_bases
        )
        test_set, key_set = sifter.select_test_sample(
            I_0,
            fraction=self.config.security.test_fraction,
            seed=self.config.seed
        )
        
        qber = QBEREstimator.estimate(
            quantum_result.outcomes,
            bob_outcomes,
            test_set
        )
        logger.info("QBER estimated: %.4f", qber)
        
        # Security check
        from caligo.security import validate_qber
        validate_qber(qber, self.config.security)
        
        return SiftingPhaseResult(
            alice_outcomes=quantum_result.outcomes,
            bob_outcomes=bob_outcomes,
            alice_bases=quantum_result.bases,
            bob_bases=bob_bases,
            I_0=I_0,
            I_1=I_1,
            test_set=test_set,
            key_set=key_set,
            qber=qber,
        )
    
    def _phase4_amplification(
        self, recon_result: ReconciliationPhaseResult
    ) -> Generator[EventExpression, None, AmplificationPhaseResult]:
        """
        Alice's Phase IV: Privacy amplification producing (S_0, S_1).
        
        Sequence
        --------
        1. Compute min-entropy using NSM Max Bound
        2. Calculate secure key length
        3. Generate Toeplitz hash seed
        4. Send seed to Bob
        5. Hash key portions by Alice's basis choices
        6. Produce AliceObliviousKey(S_0, S_1)
        """
        logger = get_logger("protocol.alice.phase4")
        logger.info("=== PHASE IV: Amplification (Alice) ===")
        
        from caligo.amplification import (
            NSMEntropyCalculator,
            SecureKeyLengthCalculator,
            ToeplitzHasher,
            OTOutputFormatter,
        )
        
        # 1-2. Compute secure key length
        entropy_calc = NSMEntropyCalculator(
            storage_noise_r=self.config.nsm.storage_noise_r
        )
        key_calc = SecureKeyLengthCalculator(
            storage_noise_r=self.config.nsm.storage_noise_r,
            epsilon_sec=self.config.security.epsilon_sec,
        )
        
        key_length = key_calc.compute_final_length(
            reconciled_length=len(recon_result.corrected_key),
            syndrome_leakage=recon_result.leakage_bits,
        )
        
        if key_length == 0:
            logger.warning("Death Valley: no secure key extractable")
            # Return empty keys
            return AmplificationPhaseResult(
                oblivious_key=AliceObliviousKey.empty(),
                qber=recon_result.qber,
                key_length=0,
                metrics={},
            )
        
        # 3. Generate Toeplitz seed
        hasher = ToeplitzHasher(
            input_length=len(recon_result.corrected_key),
            output_length=key_length,
        )
        seed = hasher.generate_seed()
        
        # 4. Send seed to Bob
        self.csocket.send(seed.tobytes().hex())
        yield from self.connection.flush()
        logger.info("Sent Toeplitz seed to Bob")
        
        # 5-6. Compute Alice's keys (S_0, S_1)
        alice_keys = OTOutputFormatter.compute_alice_keys(
            reconciled_key=recon_result.corrected_key,
            alice_bases=recon_result.alice_bases,
            key_indices=recon_result.key_set,
            hasher=hasher,
        )
        
        logger.info("Produced keys: |S_0|=%d, |S_1|=%d",
                    len(alice_keys.key_0), len(alice_keys.key_1))
        
        return AmplificationPhaseResult(
            oblivious_key=alice_keys,
            qber=recon_result.qber,
            key_length=key_length,
            metrics={"entropy_rate": entropy_calc.max_bound_entropy_rate()},
        )
```

### 5.4 Module: `bob.py` (~150 LOC)

**Purpose:** Bob's E-HOK protocol implementation.

#### 5.4.1 Bob's Role in E-HOK

Bob has specific responsibilities in each phase:

| Phase | Bob's Actions |
|-------|---------------|
| Phase I | Receive EPR pairs (`recv_keep`) |
| Phase II | Commit to outcomes/bases → Receive Alice's bases → Decommit |
| Phase III | Compute syndrome → Send to Alice → Receive verification |
| Phase IV | Receive seed → Hash to S_C → Derive choice bit |

#### 5.4.2 `BobProgram` Class

```python
class BobProgram(CaligoProgram):
    """
    Bob's E-HOK protocol implementation.
    
    Bob is the "receiver" in the 1-out-of-2 OT terminology. He:
    - Receives EPR pairs (responder role)
    - Commits to his measurements BEFORE learning Alice's bases
    - Obtains ONE key: S_C (where C is his implicit choice bit)
    - CANNOT learn S_{1-C} due to NSM entropy bound
    
    Security Properties
    -------------------
    - Receiver Privacy: Alice cannot determine C (random bases)
    - Sender Privacy: Bob cannot learn S_{1-C} (NSM bound)
    
    Choice Bit Derivation
    ---------------------
    Bob's choice bit C is derived from the I_1/I_0 partition.
    It reflects the implicit "choice" encoded in his random basis
    selections that happened to match Alice's.
    
    Attributes
    ----------
    PEER : str = "alice"
    ROLE : str = "bob"
    """
    
    PEER = "alice"
    ROLE = "bob"
    
    def _phase2_sifting(
        self, quantum_result: QuantumPhaseResult
    ) -> Generator[EventExpression, None, SiftingPhaseResult]:
        """
        Bob's Phase II: Commit, receive bases, decommit, sift.
        
        Sequence
        --------
        1. Commit to outcomes and bases (SHA256)
        2. Send commitment (ordered: send_with_ack)
        3. Receive Alice's bases (ordered: recv_and_ack)
        4. Send decommitment (outcomes, bases, salt)
        5. Sift keys (I_0, I_1 partition)
        
        Security
        --------
        Bob commits BEFORE receiving bases. The commitment is
        binding — he cannot change his claimed detection pattern
        after learning which bases Alice used.
        """
        logger = get_logger("protocol.bob.phase2")
        logger.info("=== PHASE II: Sifting (Bob) ===")
        
        # 1. Create commitment
        from caligo.sifting import CommitmentScheme
        data = np.concatenate([
            quantum_result.outcomes,
            quantum_result.bases
        ])
        commitment, salt = CommitmentScheme.commit(data)
        
        # 2. Send commitment (with ACK)
        yield from self._ordered_socket.send_with_ack(
            socket=self.csocket,
            msg_type=MessageType.DETECTION_COMMITMENT,
            payload={"commitment": commitment.hex()}
        )
        logger.info("Sent commitment to Alice")
        
        # 3. Receive Alice's bases (with ACK)
        bases_envelope = yield from self._ordered_socket.recv_and_ack(
            self.csocket
        )
        alice_bases = np.frombuffer(
            bytes.fromhex(bases_envelope.payload["bases"]),
            dtype=np.uint8
        )
        logger.info("Received Alice's bases")
        
        # 4. Send decommitment
        yield from self._ordered_socket.send_with_ack(
            socket=self.csocket,
            msg_type=MessageType.COMMITMENT_OPENING,
            payload={
                "outcomes": quantum_result.outcomes.tobytes().hex(),
                "bases": quantum_result.bases.tobytes().hex(),
                "salt": salt.hex(),
            }
        )
        logger.info("Sent decommitment to Alice")
        
        # 5. Sift keys
        from caligo.sifting import SiftingManager
        sifter = SiftingManager()
        I_0, I_1 = sifter.identify_matching_bases(
            alice_bases, quantum_result.bases
        )
        test_set, key_set = sifter.select_test_sample(
            I_0,
            fraction=self.config.security.test_fraction,
            seed=self.config.seed
        )
        
        return SiftingPhaseResult(
            alice_outcomes=None,  # Bob doesn't know Alice's outcomes
            bob_outcomes=quantum_result.outcomes,
            alice_bases=alice_bases,
            bob_bases=quantum_result.bases,
            I_0=I_0,
            I_1=I_1,
            test_set=test_set,
            key_set=key_set,
            qber=None,  # QBER computed by Alice
        )
    
    def _phase4_amplification(
        self, recon_result: ReconciliationPhaseResult
    ) -> Generator[EventExpression, None, AmplificationPhaseResult]:
        """
        Bob's Phase IV: Receive seed, hash to S_C.
        
        Sequence
        --------
        1. Receive Toeplitz seed from Alice
        2. Derive choice bit from I_1 fraction
        3. Hash key to produce S_C
        4. Produce BobObliviousKey(S_C, choice_bit)
        
        Choice Bit
        ----------
        Bob's implicit choice C is determined by which of Alice's
        basis values (0 or 1) predominates in his matching indices.
        """
        logger = get_logger("protocol.bob.phase4")
        logger.info("=== PHASE IV: Amplification (Bob) ===")
        
        # 1. Receive Toeplitz seed
        seed_msg = yield from self.csocket.recv()
        seed_bytes = bytes.fromhex(seed_msg)
        
        if len(seed_bytes) == 0:
            # Death Valley — no secure key
            logger.warning("Received empty seed: Death Valley")
            return AmplificationPhaseResult(
                oblivious_key=BobObliviousKey.empty(),
                qber=recon_result.qber,
                key_length=0,
                metrics={},
            )
        
        seed = np.frombuffer(seed_bytes, dtype=np.uint8)
        logger.info("Received Toeplitz seed from Alice")
        
        # 2. Derive choice bit
        from caligo.amplification import OTOutputFormatter
        choice_bit = OTOutputFormatter.derive_choice_bit(
            recon_result.alice_bases,
            recon_result.key_set,
        )
        
        # 3. Hash key to S_C
        from caligo.amplification import ToeplitzHasher
        hasher = ToeplitzHasher(
            input_length=len(recon_result.corrected_key),
            output_length=len(seed) - len(recon_result.corrected_key) + 1,
            seed=seed,
        )
        
        bob_key = OTOutputFormatter.compute_bob_key(
            reconciled_key=recon_result.corrected_key,
            alice_bases=recon_result.alice_bases,
            key_indices=recon_result.key_set,
            hasher=hasher,
            choice_bit=choice_bit,
        )
        
        logger.info("Produced key: |S_C|=%d, choice=%d",
                    len(bob_key.key_c), choice_bit)
        
        return AmplificationPhaseResult(
            oblivious_key=bob_key,
            qber=recon_result.qber,
            key_length=len(bob_key.key_c),
            metrics={"choice_bit": choice_bit},
        )
```

### 5.5 Module: `orchestrator.py` (~80 LOC)

**Purpose:** Phase sequencing validation and simulation runner setup.

```python
@dataclass
class OrchestratorConfig:
    """
    Configuration for protocol orchestration.
    
    Attributes
    ----------
    alice_config : ProtocolConfig
        Alice's protocol configuration.
    bob_config : ProtocolConfig
        Bob's protocol configuration.
    network_config : NetworkConfig
        Network topology and noise parameters.
    enforce_timing : bool
        Whether to inject TimingBarrier (True for production).
    timing_delta_ns : int
        Timing barrier Δt in nanoseconds.
    """
    
    alice_config: "ProtocolConfig"
    bob_config: "ProtocolConfig"
    network_config: "NetworkConfig"
    enforce_timing: bool = True
    timing_delta_ns: int = 1_000_000_000  # 1 second default


class ProtocolOrchestrator:
    """
    Orchestrates E-HOK protocol execution within SquidASM.
    
    Responsibilities
    ----------------
    1. Pre-flight feasibility checks (Phase C integration)
    2. Network construction (Phase B integration)
    3. Program instantiation with dependency injection
    4. Simulation execution and result collection
    
    Usage
    -----
    orchestrator = ProtocolOrchestrator(config)
    
    # Pre-flight check
    feasibility = orchestrator.check_feasibility()
    if not feasibility.is_feasible:
        raise SecurityError(feasibility.reason)
    
    # Run protocol
    results = orchestrator.run()
    alice_key = results["alice"]["oblivious_key"]
    bob_key = results["bob"]["oblivious_key"]
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize orchestrator with configuration."""
        self.config = config
    
    def check_feasibility(self) -> "FeasibilityResult":
        """
        Run pre-flight security checks.
        
        Validates that physical parameters support secure
        protocol execution before committing resources.
        
        Returns
        -------
        FeasibilityResult
            FEASIBLE or abort reason.
        """
        from caligo.security import FeasibilityChecker
        checker = FeasibilityChecker(
            self.config.network_config.physical_params,
            self.config.alice_config.security,
        )
        return checker.check()
    
    def build_programs(self) -> Tuple["AliceProgram", "BobProgram"]:
        """
        Instantiate Alice and Bob programs with dependencies.
        
        Returns
        -------
        Tuple[AliceProgram, BobProgram]
            Configured program instances.
        """
        # Timing barrier (shared reference for coordination)
        timing_barrier = None
        if self.config.enforce_timing:
            from caligo.simulation.timing import TimingBarrier
            timing_barrier = TimingBarrier(
                delta_t_ns=self.config.timing_delta_ns
            )
        
        # Instantiate programs
        alice = AliceProgram(
            config=self.config.alice_config,
            timing_barrier=timing_barrier,
        )
        bob = BobProgram(
            config=self.config.bob_config,
            timing_barrier=None,  # Only Alice enforces the barrier
        )
        
        return alice, bob
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Execute protocol and collect results.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results keyed by role ("alice", "bob").
        """
        from squidasm.run.stack.run import run as squidasm_run
        from caligo.simulation.network_builder import build_network
        
        # Build network
        network = build_network(self.config.network_config)
        
        # Build programs
        alice, bob = self.build_programs()
        
        # Run simulation
        results = squidasm_run(
            config=network,
            programs={"alice": alice, "bob": bob},
            num_times=1,
        )
        
        return {
            "alice": results[0]["alice"],
            "bob": results[0]["bob"],
        }
```

### 5.6 Protocol Package Contract

```python
# caligo/protocol/__init__.py

"""
Protocol package for SquidASM program implementations.

This package provides the executable E-HOK protocol as
SquidASM-compatible programs.

Main Classes
------------
CaligoProgram : ABC
    Template base class for protocol programs.
AliceProgram : CaligoProgram
    Alice's protocol implementation (produces S_0, S_1).
BobProgram : CaligoProgram
    Bob's protocol implementation (produces S_C, C).
ProtocolOrchestrator : class
    Simulation runner with pre-flight checks.

Usage
-----

from caligo.protocol import (
    AliceProgram,
    BobProgram,
    ProtocolOrchestrator,
    OrchestratorConfig,
)

# Configure
config = OrchestratorConfig(
    alice_config=ProtocolConfig.baseline(),
    bob_config=ProtocolConfig.baseline(),
    network_config=NetworkConfig.default(),
)

# Run
orchestrator = ProtocolOrchestrator(config)
results = orchestrator.run()

"""

from caligo.protocol.base import CaligoProgram
from caligo.protocol.alice import AliceProgram
from caligo.protocol.bob import BobProgram
from caligo.protocol.orchestrator import (
    OrchestratorConfig,
    ProtocolOrchestrator,
)

__all__ = [
    "CaligoProgram",
    "AliceProgram",
    "BobProgram",
    "OrchestratorConfig",
    "ProtocolOrchestrator",
]
```

---

## 6. End-to-End Integration Tests

### 6.1 Testing Philosophy

End-to-end tests validate the complete protocol execution within SquidASM, covering:

1. **Protocol Correctness:** OT outputs satisfy correctness property
2. **Security Enforcement:** Ordering and timing constraints are respected
3. **Error Handling:** Proper abort on security violations
4. **Performance:** Execution within acceptable bounds

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    E2E TEST PYRAMID                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────┐                                  │
│                        │  E2E Protocol   │  ← Full simulation tests         │
│                        │     Tests       │    (Section 6.2-6.3)             │
│                        └────────┬────────┘                                  │
│                                 │                                           │
│                    ┌────────────┴────────────┐                              │
│                    │   Integration Tests     │  ← Phase combination tests   │
│                    │                         │    (Section 6.4)             │
│                    └────────────┬────────────┘                              │
│                                 │                                           │
│         ┌───────────────────────┴───────────────────────┐                   │
│         │              Contract Tests                   │  ← Phase boundary │
│         │          (Phase D specification)              │    validation     │
│         └───────────────────────────────────────────────┘                   │
│                                                                             │
│  Test Count Target: 15-20 E2E tests, 30-40 integration tests                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 OT Correctness Tests

These tests validate the fundamental OT property: Bob obtains the key corresponding to his choice bit.

```python
# tests/e2e/test_ot_correctness.py

class TestOTCorrectness:
    """Validate 1-out-of-2 OT correctness property."""
    
    @pytest.fixture
    def perfect_network(self):
        """Create perfect (noiseless) network for correctness testing."""
        from squidasm.run.stack.config import (
            StackNetworkConfig,
            StackConfig,
            LinkConfig,
        )
        alice_cfg = StackConfig.perfect_generic_config("alice")
        bob_cfg = StackConfig.perfect_generic_config("bob")
        link = LinkConfig.perfect_config("alice", "bob")
        return StackNetworkConfig(stacks=[alice_cfg, bob_cfg], links=[link])
    
    def test_bob_key_matches_alice_key_choice_0(
        self, perfect_network, baseline_config
    ):
        """
        Test OT correctness: Bob's key matches Alice's S_0 when C=0.
        
        The correctness property states:
            S_C = Alice's S_C
        
        When Bob's choice bit C=0, his key should equal Alice's S_0.
        """
        # Force choice bit to 0 by controlling random seed
        config = baseline_config.with_seed(42)  # Seed that yields C=0
        
        results = run_protocol(perfect_network, config)
        
        alice_keys = results["alice"]["oblivious_key"]
        bob_key = results["bob"]["oblivious_key"]
        
        assert bob_key.choice_bit == 0
        assert np.array_equal(bob_key.key_c, alice_keys.key_0), \
            "OT correctness violated: Bob's key doesn't match Alice's S_0"
    
    def test_bob_key_matches_alice_key_choice_1(
        self, perfect_network, baseline_config
    ):
        """Test OT correctness when choice bit C=1."""
        config = baseline_config.with_seed(137)  # Seed that yields C=1
        
        results = run_protocol(perfect_network, config)
        
        alice_keys = results["alice"]["oblivious_key"]
        bob_key = results["bob"]["oblivious_key"]
        
        assert bob_key.choice_bit == 1
        assert np.array_equal(bob_key.key_c, alice_keys.key_1), \
            "OT correctness violated: Bob's key doesn't match Alice's S_1"
    
    def test_key_lengths_match(self, perfect_network, baseline_config):
        """Test that all output keys have identical length."""
        results = run_protocol(perfect_network, baseline_config)
        
        alice_keys = results["alice"]["oblivious_key"]
        bob_key = results["bob"]["oblivious_key"]
        
        assert len(alice_keys.key_0) == len(alice_keys.key_1)
        assert len(bob_key.key_c) == len(alice_keys.key_0)
    
    @pytest.mark.parametrize("num_pairs", [500, 1000, 2000])
    def test_correctness_scales_with_key_size(
        self, perfect_network, baseline_config, num_pairs
    ):
        """Test that correctness holds for various key sizes."""
        config = baseline_config.with_quantum(total_pairs=num_pairs)
        
        results = run_protocol(perfect_network, config)
        
        alice_keys = results["alice"]["oblivious_key"]
        bob_key = results["bob"]["oblivious_key"]
        
        # Correctness check
        expected = (
            alice_keys.key_0 if bob_key.choice_bit == 0 
            else alice_keys.key_1
        )
        assert np.array_equal(bob_key.key_c, expected)


class TestEmptyKeyHandling:
    """Test Death Valley scenarios where no key can be extracted."""
    
    def test_high_qber_produces_empty_keys(
        self, noisy_network_high_qber, baseline_config
    ):
        """
        Test that high QBER results in empty keys, not crashes.
        
        When QBER exceeds the security threshold, privacy amplification
        yields zero-length keys. This is correct behavior.
        """
        results = run_protocol(noisy_network_high_qber, baseline_config)
        
        alice_keys = results["alice"]["oblivious_key"]
        bob_key = results["bob"]["oblivious_key"]
        
        assert alice_keys.key_length == 0
        assert bob_key.key_length == 0
        assert results["alice"]["success"] == True  # Graceful completion
    
    def test_very_small_key_produces_empty(
        self, perfect_network, baseline_config
    ):
        """Test that very small inputs produce empty keys (Death Valley)."""
        # Too few pairs for secure key after all deductions
        config = baseline_config.with_quantum(total_pairs=50)
        
        results = run_protocol(perfect_network, config)
        
        assert results["alice"]["oblivious_key"].key_length == 0
```

### 6.3 Security Enforcement Tests

These tests validate that security constraints are properly enforced.

```python
# tests/e2e/test_security_enforcement.py

class TestCommitmentOrdering:
    """Test that commit-then-reveal ordering is enforced."""
    
    def test_bob_commitment_before_alice_bases(
        self, perfect_network, baseline_config
    ):
        """
        Test that Bob's commitment is acknowledged before Alice reveals bases.
        
        This is the fundamental NSM security property. Without it,
        Bob could post-select based on storage success.
        """
        # Inject timing probes into the protocol
        results = run_protocol_with_timing(perfect_network, baseline_config)
        
        alice_timing = results["alice"]["timing"]
        bob_timing = results["bob"]["timing"]
        
        # Bob's commitment send must complete before Alice's basis reveal
        assert bob_timing["commitment_ack_received"] < \
               alice_timing["basis_reveal_started"], \
            "Ordering violation: Alice revealed bases before Bob's commitment ACK"
    
    def test_ordering_violation_aborts_protocol(
        self, perfect_network, malicious_alice_config
    ):
        """
        Test that ordering violations cause protocol abort.
        
        A malicious Alice that tries to reveal bases before receiving
        Bob's commitment should be detected and aborted.
        """
        with pytest.raises(OrderingViolationError):
            run_protocol(perfect_network, malicious_alice_config)


class TestTimingBarrier:
    """Test NSM timing barrier enforcement."""
    
    def test_alice_waits_delta_t_before_revealing(
        self, perfect_network, baseline_config
    ):
        """
        Test that Alice waits Δt after commitment before revealing.
        
        The timing barrier ensures adversary's quantum storage has
        decohered before basis information is available.
        """
        delta_t_ns = 1_000_000_000  # 1 second
        config = baseline_config.with_timing(delta_t_ns=delta_t_ns)
        
        results = run_protocol_with_timing(perfect_network, config)
        
        alice_timing = results["alice"]["timing"]
        
        wait_duration = (
            alice_timing["basis_reveal_started"] - 
            alice_timing["commitment_received"]
        )
        
        assert wait_duration >= delta_t_ns, \
            f"Timing barrier violated: waited {wait_duration}ns, required {delta_t_ns}ns"
    
    def test_timing_violation_detected(
        self, perfect_network, baseline_config
    ):
        """Test that timing barrier violation raises error."""
        # Configure very short Δt but inject a timing violation
        config = baseline_config.with_timing(
            delta_t_ns=1_000_000_000,  # 1 second
            enforce=True,
        )
        
        # This test uses a mock that attempts early reveal
        with pytest.raises(TimingViolationError):
            run_protocol_with_early_reveal(perfect_network, config)


class TestQBERAbort:
    """Test QBER threshold enforcement."""
    
    def test_abort_on_qber_exceeds_threshold(
        self, noisy_network_moderate, baseline_config
    ):
        """
        Test that protocol aborts when QBER exceeds security threshold.
        
        The 22% threshold (Lupo limit) is the hard security boundary.
        """
        # Configure network to produce ~25% QBER
        results = run_protocol(noisy_network_moderate, baseline_config)
        
        assert results["alice"]["success"] == False
        assert "QBER" in results["alice"]["abort_reason"]
    
    def test_warning_at_conservative_threshold(
        self, slightly_noisy_network, baseline_config
    ):
        """
        Test that warning is logged when QBER exceeds conservative threshold.
        
        The 11% threshold (Schaffner) triggers a warning but not abort.
        """
        with pytest.warns(UserWarning, match="QBER.*11%"):
            results = run_protocol(slightly_noisy_network, baseline_config)
        
        # Should still succeed if under hard limit
        assert results["alice"]["success"] == True
```

### 6.4 Phase Integration Tests

These tests validate correct interaction between protocol phases.

```python
# tests/e2e/test_phase_integration.py

class TestQuantumToSifting:
    """Test Phase I → Phase II integration."""
    
    def test_measurement_buffer_preserved(
        self, perfect_network, baseline_config
    ):
        """Test that measurement outcomes are correctly passed to sifting."""
        results = run_protocol_with_metrics(perfect_network, baseline_config)
        
        metrics = results["metrics"]
        
        # Raw count from quantum phase should match sifting input
        assert metrics["quantum_raw_count"] == metrics["sifting_input_count"]
    
    def test_basis_arrays_match_measurement_count(
        self, perfect_network, baseline_config
    ):
        """Test that basis array length matches outcome count."""
        results = run_protocol_with_debug(perfect_network, baseline_config)
        
        alice_debug = results["alice"]["debug"]
        
        assert len(alice_debug["bases"]) == len(alice_debug["outcomes"])


class TestSiftingToReconciliation:
    """Test Phase II → Phase III integration."""
    
    def test_sifted_key_length_matches_reconciliation_input(
        self, perfect_network, baseline_config
    ):
        """Test that sifted key is correctly passed to reconciliation."""
        results = run_protocol_with_metrics(perfect_network, baseline_config)
        
        metrics = results["metrics"]
        
        # Key set from sifting should match reconciliation input
        assert metrics["sifting_key_length"] == metrics["recon_input_length"]
    
    def test_test_set_excluded_from_reconciliation(
        self, perfect_network, baseline_config
    ):
        """Test that test set bits are NOT included in reconciliation."""
        results = run_protocol_with_metrics(perfect_network, baseline_config)
        
        metrics = results["metrics"]
        
        # Key set + test set should equal sifted set
        assert (metrics["sifting_key_length"] + metrics["sifting_test_length"] == 
                metrics["sifting_total_length"])


class TestReconciliationToAmplification:
    """Test Phase III → Phase IV integration."""
    
    def test_corrected_key_passed_to_amplification(
        self, perfect_network, baseline_config
    ):
        """Test that error-corrected key feeds amplification."""
        results = run_protocol_with_metrics(perfect_network, baseline_config)
        
        metrics = results["metrics"]
        
        # Reconciliation output should match amplification input
        assert metrics["recon_output_length"] == metrics["amp_input_length"]
    
    def test_leakage_accounted_in_key_length(
        self, perfect_network, baseline_config
    ):
        """Test that reconciliation leakage reduces final key length."""
        results = run_protocol_with_metrics(perfect_network, baseline_config)
        
        metrics = results["metrics"]
        
        # Final key should be shorter than reconciled key
        # (due to leakage deduction and security parameter)
        assert metrics["final_key_length"] < metrics["recon_output_length"]
```

### 6.5 Simulation Fidelity Tests

These tests validate that the simulation behaves as expected.

```python
# tests/e2e/test_simulation_fidelity.py

class TestNoiseModels:
    """Test that noise models produce expected behavior."""
    
    @pytest.mark.parametrize("depolar_rate,expected_qber_range", [
        (0.0, (0.0, 0.01)),     # Perfect: near-zero QBER
        (0.01, (0.005, 0.02)),  # Low noise: ~1% QBER
        (0.05, (0.03, 0.08)),   # Moderate: ~5% QBER
        (0.10, (0.07, 0.13)),   # High: ~10% QBER
    ])
    def test_qber_correlates_with_depolarization(
        self, depolar_rate, expected_qber_range, baseline_config
    ):
        """Test that QBER increases with depolarization rate."""
        network = create_network_with_depolar(depolar_rate)
        
        results = run_protocol(network, baseline_config)
        
        qber = results["alice"]["qber"]
        low, high = expected_qber_range
        assert low <= qber <= high, \
            f"QBER {qber} outside expected range [{low}, {high}]"
    
    def test_storage_noise_affects_entropy_bound(
        self, perfect_network, baseline_config
    ):
        """Test that storage noise parameter affects entropy calculation."""
        results_low_r = run_protocol(
            perfect_network,
            baseline_config.with_nsm(storage_noise_r=0.5)
        )
        results_high_r = run_protocol(
            perfect_network,
            baseline_config.with_nsm(storage_noise_r=0.9)
        )
        
        # Lower r (more decoherence) should yield MORE extractable entropy
        assert results_low_r["alice"]["metrics"]["entropy_rate"] > \
               results_high_r["alice"]["metrics"]["entropy_rate"]


class TestDeterministicReplay:
    """Test that protocol is deterministic given same seeds."""
    
    def test_same_seed_produces_same_keys(
        self, perfect_network, baseline_config
    ):
        """Test that identical seeds produce identical outputs."""
        config = baseline_config.with_seed(12345)
        
        results1 = run_protocol(perfect_network, config)
        results2 = run_protocol(perfect_network, config)
        
        assert np.array_equal(
            results1["alice"]["oblivious_key"].key_0,
            results2["alice"]["oblivious_key"].key_0
        )
        assert np.array_equal(
            results1["bob"]["oblivious_key"].key_c,
            results2["bob"]["oblivious_key"].key_c
        )
    
    def test_different_seeds_produce_different_keys(
        self, perfect_network, baseline_config
    ):
        """Test that different seeds produce different outputs."""
        results1 = run_protocol(
            perfect_network, baseline_config.with_seed(111)
        )
        results2 = run_protocol(
            perfect_network, baseline_config.with_seed(222)
        )
        
        # Keys should differ (with high probability)
        assert not np.array_equal(
            results1["alice"]["oblivious_key"].key_0,
            results2["alice"]["oblivious_key"].key_0
        )
```

### 6.6 Test Fixtures and Utilities

```python
# tests/e2e/conftest.py

import pytest
from squidasm.run.stack.config import (
    StackNetworkConfig,
    StackConfig,
    LinkConfig,
)
from caligo.protocol import (
    AliceProgram,
    BobProgram,
    ProtocolOrchestrator,
)


@pytest.fixture
def perfect_network():
    """Create noiseless network configuration."""
    alice = StackConfig.perfect_generic_config("alice")
    bob = StackConfig.perfect_generic_config("bob")
    link = LinkConfig.perfect_config("alice", "bob")
    return StackNetworkConfig(stacks=[alice, bob], links=[link])


@pytest.fixture
def baseline_config():
    """Create baseline protocol configuration."""
    from caligo.types import ProtocolConfig
    return ProtocolConfig.baseline()


@pytest.fixture
def noisy_network_high_qber():
    """Create network configuration with high QBER (~25%)."""
    return create_network_with_depolar(depolar_rate=0.20)


@pytest.fixture
def noisy_network_moderate():
    """Create network with moderate noise (~15% QBER)."""
    return create_network_with_depolar(depolar_rate=0.12)


@pytest.fixture
def slightly_noisy_network():
    """Create network with slight noise (~8% QBER)."""
    return create_network_with_depolar(depolar_rate=0.06)


def create_network_with_depolar(depolar_rate: float) -> StackNetworkConfig:
    """Create network with specified depolarization rate."""
    from netsquid.components.models.qerrormodels import DepolarNoiseModel
    
    # Configuration with custom noise model
    alice = StackConfig(
        name="alice",
        qdevice_typ="generic",
        qdevice_cfg={"num_qubits": 10},
    )
    bob = StackConfig(
        name="bob",
        qdevice_typ="generic",
        qdevice_cfg={"num_qubits": 10},
    )
    link = LinkConfig(
        stack1="alice",
        stack2="bob",
        typ="depolarise",
        cfg={"fidelity": 1.0 - depolar_rate},
    )
    return StackNetworkConfig(stacks=[alice, bob], links=[link])


def run_protocol(network, config):
    """Run protocol and return results."""
    from squidasm.run.stack.run import run as squidasm_run
    
    alice = AliceProgram(config=config)
    bob = BobProgram(config=config)
    
    results = squidasm_run(
        config=network,
        programs={"alice": alice, "bob": bob},
        num_times=1,
    )
    
    return {
        "alice": results[0]["alice"],
        "bob": results[0]["bob"],
    }
```

---

## 7. Acceptance Criteria

### 7.1 Module Completeness

| Module | Required Classes | Required Methods | LOC Target |
|--------|------------------|------------------|------------|
| `connection/envelope.py` | `MessageType`, `MessageEnvelope`, `AckPayload` | `to_json`, `from_json`, `to_dict`, `from_dict` | ≤80 |
| `connection/ordered_socket.py` | `SocketState`, `OrderedSocket` | `send_with_ack`, `recv_and_ack`, `create_envelope` | ≤150 |
| `connection/exceptions.py` | 4 exception classes | — | ≤40 |
| `protocol/base.py` | `CaligoProgram` | `meta`, `run`, `_phase1_quantum` | ≤180 |
| `protocol/alice.py` | `AliceProgram` | `_phase2_sifting`, `_phase4_amplification` | ≤150 |
| `protocol/bob.py` | `BobProgram` | `_phase2_sifting`, `_phase4_amplification` | ≤150 |
| `protocol/orchestrator.py` | `OrchestratorConfig`, `ProtocolOrchestrator` | `check_feasibility`, `build_programs`, `run` | ≤80 |

### 7.2 Test Coverage Requirements

| Category | Minimum Coverage | Test Count |
|----------|------------------|------------|
| Unit tests | 90% line coverage | ~30 tests |
| E2E correctness tests | All OT properties | ~10 tests |
| Security enforcement tests | All abort conditions | ~8 tests |
| Phase integration tests | All phase boundaries | ~6 tests |
| Simulation fidelity tests | Noise model validation | ~5 tests |

### 7.3 Performance Targets

| Operation | Target | Environment |
|-----------|--------|-------------|
| Protocol execution (1000 pairs) | <30 seconds | SquidASM simulation |
| Protocol execution (10000 pairs) | <5 minutes | SquidASM simulation |
| Message serialization | <1 ms | Unit test |
| Ordered socket ACK round-trip | <10 ms sim time | Integration test |

### 7.4 SquidASM Compatibility

| Requirement | Validation |
|-------------|------------|
| `Program` interface compliance | `meta` property returns `ProgramMeta` |
| Generator protocol | All I/O uses `yield from` |
| Context usage | Accesses `csockets`, `epr_sockets`, `connection` correctly |
| No blocking calls | No direct `recv()` without `yield from` |
| Flush pattern | `connection.flush()` called after quantum operations |

---

## 8. References

### 8.1 Primary Literature

1. **König, R., et al.** (2012). "Unconditional security from noisy quantum storage."  
   *Key contributions:* Commit-then-reveal semantics, timing barrier requirements.

2. **Lupo, C., et al.** (2023). "Quantum and device-independent unconditionally secure digital signatures."  
   *Key contributions:* E-HOK protocol specification, QBER limits.

3. **Schaffner, C., et al.** (2009). "Simple protocols for oblivious transfer."  
   *Key contributions:* NSM security model, protocol structure.

### 8.2 SquidASM Documentation

4. **SquidASM Documentation** — Program interface, ProgramContext, generator execution model.

5. **NetQASM SDK** — EPRSocket, ClassicalSocket, BaseNetQASMConnection APIs.

### 8.3 Internal Documentation

6. **Phase A Specification** (`phase_a_spec.md`): Type contracts used in protocol
7. **Phase B Specification** (`phase_b_spec.md`): TimingBarrier, network configuration
8. **Phase C Specification** (`phase_c_spec.md`): FeasibilityChecker, security bounds
9. **Phase D Specification** (`phase_d_spec.md`): Quantum, sifting, amplification phases

### 8.4 Source Code References

| ehok Module | Caligo Equivalent | Key Insights |
|-------------|-------------------|--------------|
| `ehok/protocols/ordered_messaging.py` | `connection/` | State machine, ACK protocol |
| `ehok/protocols/base.py` | `protocol/base.py` | Template method pattern |
| `ehok/protocols/alice.py` | `protocol/alice.py` | Timing barrier integration |
| `ehok/protocols/bob.py` | `protocol/bob.py` | Commitment-first flow |
| `ehok/tests/test_integration.py` | `tests/e2e/` | Phase sequencing tests |
| `ehok/tests/test_sprint3_e2e_pipeline.py` | `tests/e2e/` | OT correctness tests |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-01-XX | Caligo Team | Initial draft |
| 1.0 | 2025-01-XX | Caligo Team | Complete specification |

# Caligo: Protocol Technical Reference

**Caligo** (Latin: "fog/mist" — evoking the obscured nature of oblivious transfer) is a production-grade implementation of the 1-out-of-2 Oblivious Transfer protocol secured by the Noisy Storage Model (NSM).

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [System Architecture](#system-architecture)
3. [Protocol Workflow](#protocol-workflow)
4. [Package Organization](#package-organization)
5. [Core Type System](#core-type-system)
6. [Phase Execution Pipeline](#phase-execution-pipeline)
7. [SquidASM Integration (Phase E)](#squidasm-integration-phase-e)
8. [Configuration System](#configuration-system)
9. [Installation & Dependencies](#installation--dependencies)
10. [Entry Points & API Surface](#entry-points--api-surface)
11. [Testing Infrastructure](#testing-infrastructure)
12. [Design Patterns & Conventions](#design-patterns--conventions)
13. [Security Guarantees & Bounds](#security-guarantees--bounds)
14. [References](#references)

---

## Overview & Architecture

Caligo implements a quantum cryptographic construction enabling **1-out-of-2 Oblivious Transfer (OT)**:

- **Alice (Sender)**: Generates two cryptographic keys $(S_0, S_1)$ but remains oblivious to which key Bob receives.
- **Bob (Receiver)**: Selects one key $S_C$ via choice bit $C \in \{0, 1\}$ and cannot learn anything about $S_{1-C}$.

**Security Foundation**: The protocol's security derives from the **Noisy Storage Model (NSM)**, which assumes an adversary has limited quantum memory capacity and must store qubits for a bounded time $\Delta t$ under decoherence.

### Design Philosophy

Caligo is architected around three core principles:

1. **Type-Safe Phase Boundaries**: Each protocol phase (Quantum → Sifting → Reconciliation → Amplification) exchanges data via formally specified contracts (`QuantumPhaseResult`, `SiftingPhaseResult`, etc.) with runtime validation.

2. **Separation of Concerns**: Quantum operations (Phase I–IV) are decoupled from network simulation (Phase E). The core cryptographic logic can be tested independently of SquidASM.

3. **Strategy Pattern for Extensibility**: Critical subsystems (EPR generation, reconciliation, amplification) use the Strategy pattern, enabling algorithm swapping without modifying orchestration logic.

---

## System Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                           │
│  (Protocol Orchestration, CLI, Configuration Management)        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                     PROTOCOL LAYER (Phase E)                    │
│  SquidASM Programs: alice.py, bob.py, orchestrator.py          │
│  • Quantum Phase: EPR generation via NetQASM                    │
│  • Classical Phases: Sifting, Reconciliation, Amplification    │
│  • Ordered Socket: Commit-reveal messaging protocol            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                   CRYPTOGRAPHIC CORE (Phase D)                  │
│  Domain Logic Packages (SquidASM-independent):                  │
│  • quantum/    : EPR, basis selection, measurement buffering    │
│  • sifting/    : Commitment, basis disclosure, QBER estimation  │
│  • reconciliation/ : LDPC codes, syndrome encoding/decoding     │
│  • amplification/  : Entropy calculation, Toeplitz hashing      │
│  • security/   : Feasibility checks, finite-key bounds          │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                          │
│  • types/       : Phase contracts, domain primitives, exceptions│
│  • utils/       : Logging, bitarray ops, entropy functions      │
│  • simulation/  : NSM parameter enforcement, network builders   │
│  • connection/  : Ordered socket, message envelopes            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                   EXTERNAL DEPENDENCIES                         │
│  • SquidASM     : Quantum network simulator (optional)          │
│  • NetQASM      : Quantum assembly language & SDK               │
│  • NetSquid     : Discrete-event quantum simulator              │
│  • NumPy/SciPy  : Numerical computation                         │
│  • Numba        : JIT compilation for LDPC decoding             │
└─────────────────────────────────────────────────────────────────┘
```

### Phase Taxonomy

Caligo distinguishes two execution contexts:

- **Phase D (Domain Logic)**: SquidASM-independent cryptographic primitives that can be unit-tested and benchmarked independently.
- **Phase E (Execution)**: SquidASM `Program` implementations integrating Phase D logic with quantum network simulation.

---

## Protocol Workflow

### High-Level State Machine

```
                  ┌──────────────┐
                  │  Initialize  │
                  │  (NSM Params)│
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Phase I:    │
                  │  Quantum     │──► QuantumPhaseResult
                  │              │    (measurement outcomes,
                  │  • EPR Gen   │     bases, timestamps)
                  │  • Measure   │
                  │  • Timing    │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Phase II:   │
                  │  Sifting     │──► SiftingPhaseResult
                  │              │    (sifted keys, QBER,
                  │  • Commit    │     detection events)
                  │  • Disclose  │
                  │  • QBER Est  │
                  └──────┬───────┘
                         │
           ┌─────────────┴─────────────┐
           │   Security Check:         │
           │   QBER ≤ Threshold?       │
           └───┬─────────────────┬─────┘
              YES               NO
               │                 │
        ┌──────▼───────┐   ┌────▼────┐
        │  Phase III:  │   │  ABORT  │
        │  Reconcile   │   │         │
        │              │   └─────────┘
        │  • LDPC Code │
        │  • Syndrome  │
        │  • Verify    │
        └──────┬───────┘
               │
               │  ReconciliationPhaseResult
               │  (reconciled keys, leakage)
               │
        ┌──────▼───────┐
        │  Phase IV:   │
        │  Amplify     │──► ObliviousTransferOutput
        │              │    (S₀, S₁ for Alice;
        │  • Entropy   │     Sᴄ for Bob)
        │  • Toeplitz  │
        │  • Extract   │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   Success    │
        │   (OT Keys)  │
        └──────────────┘
```

### Execution Modes

1. **Standalone Mode (Phase D)**: Execute cryptographic phases with pre-generated data (no SquidASM). Ideal for:
   - Unit testing individual phase boundaries
   - Performance benchmarking
   - Parameter sweeps without simulation overhead

2. **Simulation Mode (Phase E)**: Full end-to-end execution via SquidASM `run_protocol()`:
   - Quantum EPR generation via NetQASM
   - Classical channel communication with simulated latency
   - NSM timing enforcement via `TimingBarrier`

---

## Package Organization

### Directory Structure

```
caligo/
├── __init__.py                  # Public API exports
├── cli.py                       # Command-line interface (EPR generation)
│
├── types/                       # Type System & Contracts
│   ├── keys.py                  # ObliviousKey, AliceObliviousKey, BobObliviousKey
│   ├── measurements.py          # MeasurementRecord, DetectionEvent, RoundResult
│   ├── phase_contracts.py       # QuantumPhaseResult → AmplificationPhaseResult
│   └── exceptions.py            # Exception hierarchy, ProtocolPhase, AbortReason
│
├── protocol/                    # Phase E: SquidASM Programs
│   ├── base.py                  # CaligoProgram base class, ProtocolParameters
│   ├── alice.py                 # AliceProgram (sender)
│   ├── bob.py                   # BobProgram (receiver)
│   └── orchestrator.py          # run_protocol() entry point
│
├── quantum/                     # Phase D: Quantum Operations
│   ├── epr.py                   # EPRGenerator (NetQASM socket wrapper)
│   ├── basis.py                 # BasisSelector (random Z/X choice)
│   ├── measurement.py           # MeasurementExecutor, MeasurementBuffer
│   ├── batching.py              # BatchingManager for large EPR runs
│   ├── factory.py               # EPRGenerationFactory (Strategy pattern)
│   ├── parallel.py              # ParallelEPROrchestrator (multiprocessing)
│   └── workers.py               # Worker functions for parallel generation
│
├── sifting/                     # Phase D: Basis Sifting & QBER
│   ├── commitment.py            # SHA256Commitment (commit-reveal protocol)
│   ├── sifter.py                # Sifter (basis matching logic)
│   ├── qber.py                  # QBEREstimator (error rate calculation)
│   └── detection_validator.py  # DetectionValidator (Hoeffding bounds)
│
├── reconciliation/              # Phase D: Information Reconciliation
│   ├── orchestrator.py          # ReconciliationOrchestrator (Phase III entry)
│   ├── block_reconciler.py      # BlockReconciler (per-block reconciliation)
│   ├── ldpc_encoder.py          # LDPCEncoder (syndrome computation)
│   ├── ldpc_decoder.py          # BeliefPropagationDecoder (Numba-JIT)
│   ├── matrix_manager.py        # MatrixManager (LDPC code pool)
│   ├── compiled_matrix.py       # CompiledParityCheckMatrix (CSR format)
│   ├── hash_verifier.py         # PolynomialHashVerifier (GF(2) hash)
│   ├── leakage_tracker.py       # LeakageTracker (syndrome + hash leakage)
│   ├── rate_selector.py         # AdaptiveRateSelector (code rate selection)
│   ├── blind_manager.py         # BlindReconciliationManager (iterative)
│   ├── factory.py               # create_strategy() (Baseline vs Blind)
│   └── strategies/              # ReconciliationStrategy implementations
│       ├── baseline.py          # BaselineStrategy (Alice knows QBER)
│       ├── blind.py             # BlindStrategy (Bob-driven, no QBER)
│       └── codec.py             # ClassicalChannelCodec (message serialization)
│
├── amplification/               # Phase D: Privacy Amplification
│   ├── entropy.py               # NSMEntropyCalculator (min-entropy rate)
│   ├── key_length.py            # SecureKeyLengthCalculator (Lupo formula)
│   ├── toeplitz.py              # ToeplitzHasher (universal hash family)
│   └── formatter.py             # OTOutputFormatter (split into S₀, S₁, Sᴄ)
│
├── security/                    # Phase D: Security Analysis
│   ├── bounds.py                # Entropy bounds (Dupuis-König, Lupo, etc.)
│   ├── feasibility.py           # FeasibilityChecker (preflight validation)
│   └── finite_key.py            # Finite-key corrections (Hoeffding, etc.)
│
├── simulation/                  # Infrastructure: SquidASM Integration
│   ├── physical_model.py        # NSMParameters, ChannelParameters
│   ├── network_builder.py       # CaligoNetworkBuilder (StackNetworkConfig)
│   ├── noise_models.py          # NSMStorageNoiseModel, ChannelNoiseProfile
│   ├── timing.py                # TimingBarrier (Δt enforcement)
│   └── constants.py             # QBER thresholds, default parameters
│
├── connection/                  # Infrastructure: Ordered Messaging
│   ├── ordered_socket.py        # OrderedSocket (ACK protocol wrapper)
│   ├── envelope.py              # MessageEnvelope, MessageType, AckPayload
│   └── exceptions.py            # OutOfOrderError, AckTimeoutError
│
├── utils/                       # Infrastructure: Utilities
│   ├── logging.py               # get_logger(), SquidASM-compatible logging
│   ├── math.py                  # binary_entropy, channel_capacity, etc.
│   └── bitarray_utils.py        # Bitarray manipulation (XOR, Hamming, etc.)
│
└── scripts/                     # Utilities: Code Generation
    ├── generate_ldpc_matrices.py    # LDPC code matrix generation
    ├── peg_generator.py             # Progressive Edge Growth algorithm
    ├── generate_ace_mother_code.py  # ACE (mother code) generation
    └── numba_kernels.py             # JIT-compiled LDPC kernels
```

---

## Core Type System

### Phase Boundary Contracts

Caligo enforces **contract-driven development** via immutable dataclasses at each phase boundary. These contracts define:

1. **Pre-conditions**: What must be true before entering a phase.
2. **Post-conditions**: What is guaranteed after a phase completes successfully.
3. **Invariants**: Properties that must hold throughout execution.

#### Contract Hierarchy

```python
# Phase I → Phase II
@dataclass
class QuantumPhaseResult:
    measurement_outcomes: np.ndarray  # Shape (n,), dtype uint8, values ∈ {0,1}
    basis_choices: np.ndarray         # Shape (n,), dtype uint8, values ∈ {0,1}
    round_ids: np.ndarray             # Shape (n,), dtype int64
    generation_timestamp: float       # Simulation time (ns)
    num_pairs_requested: int
    num_pairs_generated: int
    detection_events: List[DetectionEvent]
    timing_barrier_marked: bool       # NSM invariant

# Phase II → Phase III
@dataclass
class SiftingPhaseResult:
    alice_sifted_key: bitarray
    bob_sifted_key: bitarray
    qber_estimate: float
    num_sifted_bits: int
    num_errors_estimated: int
    hoeffding_bound: HoeffdingBound
    detection_validation: ValidationResult

# Phase III → Phase IV
@dataclass
class ReconciliationPhaseResult:
    alice_reconciled_key: bitarray
    bob_reconciled_key: bitarray
    syndrome_leakage: int             # Bits leaked via error correction
    verification_leakage: int         # Bits leaked via hash verification
    total_leakage: int                # syndrome + verification
    num_blocks: int
    block_size: int
    code_rate: float

# Phase IV → Application
@dataclass
class ObliviousTransferOutput:
    alice_output: AliceOTOutput       # Contains (S₀, S₁)
    bob_output: BobOTOutput           # Contains (Sᴄ, choice_bit)
    protocol_metadata: Dict[str, Any]
```

### Exception Hierarchy

```python
CaligoError                       # Base exception
├── SimulationError
│   ├── TimingViolationError      # NSM Δt violated
│   ├── NetworkConfigError        # SquidASM setup failure
│   ├── UnsupportedHardwareError  # Non-Generic QDevice
│   └── EPRGenerationError        # EPR socket failure
├── SecurityError
│   ├── QBERThresholdExceeded     # QBER > threshold
│   ├── InsufficientEntropyError  # Cannot extract secure key
│   ├── LeakageCapExceededError   # Total leakage > budget
│   └── ContractViolation         # Phase boundary invariant broken
├── ProtocolError
│   ├── ReconciliationFailedError # LDPC decode failure
│   ├── VerificationFailedError   # Hash mismatch
│   └── CommitmentViolationError  # Basis commitment failed
└── ConnectionError
    ├── OutOfOrderError           # Message sequence violation
    ├── AckTimeoutError           # No ACK received
    └── SessionMismatchError      # Wrong session_id
```

---

## Phase Execution Pipeline

### Phase I: Quantum Operations

**Objective**: Generate EPR pairs, measure in random bases, enforce NSM timing.

**Components**:
- `EPRGenerator`: Wraps SquidASM EPR socket with error handling
- `BasisSelector`: Generates random basis choices (Z=0, X=1)
- `MeasurementExecutor`: Performs single-qubit measurements
- `MeasurementBuffer`: Accumulates results into `QuantumPhaseResult`
- `TimingBarrier`: Enforces $\Delta t$ wait time before basis revelation

**Execution Flow**:
```python
# Alice side (similar for Bob)
generator = EPRGenerator(epr_socket)
selector = BasisSelector()
executor = MeasurementExecutor(qubit_register)
buffer = MeasurementBuffer()

for round_id in range(num_pairs):
    qubit = yield from generator.create()
    basis = selector.choose()
    outcome = yield from executor.measure(qubit, basis)
    buffer.record(outcome, basis, round_id)

# Enforce NSM timing constraint
yield from timing_barrier.wait()

result = buffer.finalize()  # Returns QuantumPhaseResult
```

**Strategy Pattern**:
- `EPRGenerationFactory` creates either `SequentialEPRStrategy` or `ParallelEPRStrategy`
- Parallel mode uses multiprocessing for Monte Carlo-style batch generation
- Interface: `generate(total_pairs) -> (alice_outcomes, alice_bases, bob_outcomes, bob_bases)`

### Phase II: Sifting & QBER Estimation

**Objective**: Discard basis mismatches, estimate channel error rate.

**Components**:
- `SHA256Commitment`: Commit-reveal protocol for basis choices
- `Sifter`: Filters matching-basis measurement outcomes
- `QBEREstimator`: Computes $\text{QBER} = \frac{\text{errors}}{\text{sifted bits}}$
- `DetectionValidator`: Verifies detection efficiency $\eta$ via Hoeffding bound

**Protocol Steps**:
1. Alice commits to her basis choices: $\text{commit} = H(\text{bases} \| \text{salt})$
2. Bob sends his basis choices in the clear
3. Alice reveals her basis choices + salt; Bob verifies commitment
4. Both parties sift: keep only outcomes where bases match
5. Sample $m$ bits to estimate QBER
6. Abort if $\text{QBER} > \text{threshold}$ (default 11%)

**Security Check**:
```python
if qber_estimate.qber > QBER_CONSERVATIVE_THRESHOLD:
    raise QBERThresholdExceeded(
        f"QBER {qber_estimate.qber:.4f} exceeds threshold "
        f"{QBER_CONSERVATIVE_THRESHOLD}"
    )
```

### Phase III: Information Reconciliation

**Objective**: Correct errors in sifted keys via LDPC error correction.

**Components**:
- `ReconciliationOrchestrator`: Coordinates entire Phase III
- `MatrixManager`: Manages pool of LDPC parity-check matrices
- `LDPCEncoder` (Alice): Computes syndrome $\mathbf{s} = \mathbf{H} \mathbf{k}_A$
- `BeliefPropagationDecoder` (Bob): Decodes to correct $\mathbf{k}_B$
- `PolynomialHashVerifier`: Verifies key agreement via $h(\mathbf{k})$
- `LeakageTracker`: Accounts for syndrome + hash leakage

**Execution Flow**:
```python
# Alice side
encoder = LDPCEncoder(parity_check_matrix)
syndrome = encoder.encode(sifted_key_block)
yield from socket.send(MessageType.SYNDROME, {"syndrome": syndrome})

# Bob side
syndrome_msg = yield from socket.recv()
decoder = BeliefPropagationDecoder(max_iterations=50)
corrected_key = decoder.decode(bob_sifted_key, syndrome_msg["syndrome"])

# Both sides verify via hash
alice_hash = hash_verifier.compute(alice_corrected_key)
bob_hash = hash_verifier.compute(bob_corrected_key)
# Exchange and compare...
```

**Strategy Pattern**:
- `BaselineStrategy`: Alice selects code rate based on known QBER
- `BlindStrategy`: Bob-driven iterative reconciliation without QBER knowledge
- Factory function: `create_strategy(config, mother_code, codec, leakage_tracker)`

### Phase IV: Privacy Amplification

**Objective**: Extract secure keys $S_0, S_1$ (Alice) and $S_C$ (Bob) via universal hashing.

**Components**:
- `NSMEntropyCalculator`: Computes min-entropy rate $h_{\min}$ from NSM parameters
- `SecureKeyLengthCalculator`: Applies Lupo formula to determine extractable length
- `ToeplitzHasher`: Implements universal hash via Toeplitz matrix multiplication
- `OTOutputFormatter`: Splits hashed output into oblivious keys

**Lupo Formula** (Eq. 43):
$$
\ell = \left\lfloor n \cdot h_{\min} - |\Sigma| - 2 \log_2\left(\frac{1}{\epsilon_{\text{sec}}}\right) + 2 \right\rfloor
$$

Where:
- $n$ = reconciled key length
- $h_{\min}$ = min-entropy rate (from NSM bounds)
- $|\Sigma|$ = syndrome + hash leakage
- $\epsilon_{\text{sec}}$ = security parameter (default $10^{-10}$)

**Execution**:
```python
# Calculate secure key length
key_length_result = calculator.calculate(
    reconciled_length=len(alice_key),
    syndrome_leakage=recon_result.total_leakage,
)

# Extract via Toeplitz hashing
hasher = ToeplitzHasher(
    input_length=len(alice_key),
    output_length=key_length_result.final_length,
)
extracted = hasher.hash(alice_key)

# Format into oblivious keys
formatter = OTOutputFormatter()
ot_output = formatter.format(
    alice_extracted=extracted,
    bob_extracted=extracted,
    bob_choice_bit=choice_bit,
)
```

---

## SquidASM Integration (Phase E)

### SquidASM Program Architecture

Caligo implements Phase E as SquidASM `Program` subclasses with generator-based `run()` methods:

```python
class AliceProgram(CaligoProgram):
    PEER = "Bob"
    ROLE = "alice"

    def _run_protocol(self, context) -> Generator[Any, None, Dict[str, Any]]:
        # Phase I: Quantum
        quantum_result = yield from self._phase_quantum(context)
        
        # Phase II: Sifting
        sifting_result = yield from self._phase_sifting(context, quantum_result)
        
        # Phase III: Reconciliation
        recon_result = yield from self._phase_reconciliation(context, sifting_result)
        
        # Phase IV: Amplification
        ot_output = yield from self._phase_amplification(context, recon_result)
        
        return {"role": "alice", "output": ot_output}
```

### Ordered Socket Protocol

SquidASM classical sockets are unordered. Caligo enforces ordering via `OrderedSocket`:

```python
class OrderedSocket:
    def send(self, msg_type: MessageType, payload: dict) -> Generator:
        """Send message and wait for ACK."""
        seq = self._send_seq
        envelope = MessageEnvelope(
            session_id=self._session_id,
            seq=seq,
            msg_type=msg_type,
            payload=payload,
        )
        self._sock.send(envelope.serialize())
        self._state = SocketState.SENT_WAIT_ACK
        
        # Wait for ACK
        ack = yield from self._recv_ack(timeout_rounds=10)
        self._send_seq += 1
        
    def recv(self, expected_type: MessageType) -> Generator:
        """Receive message and send ACK."""
        envelope = yield from self._recv_envelope()
        if envelope.msg_type != expected_type:
            raise OutOfOrderError(...)
        
        # Send ACK
        ack = AckPayload(seq=envelope.seq)
        self._sock.send(ack.serialize())
        return envelope.payload
```

### Network Builder

`CaligoNetworkBuilder` generates `StackNetworkConfig` with NSM parameter enforcement:

```python
builder = CaligoNetworkBuilder()
network_config = builder.build(
    nodes=["Alice", "Bob"],
    nsm_params=NSMParameters(
        delta_t_ns=1_000_000,
        storage_noise_r=0.75,
        detection_eff_eta=0.85,
    ),
    channel_params=ChannelParameters(
        length_km=10.0,
        attenuation_db_per_km=0.2,
    ),
)
```

**Hardware Constraint**: Caligo only supports `"generic"` QDevice type due to NetQASM 2.x / SquidASM 0.13.x incompatibility. NV hardware triggers unimplemented `MOV` instructions.

---

## Configuration System

### YAML Configuration

Example: [`configs/nsm_erven_experimental.yaml`](configs/nsm_erven_experimental.yaml)

```yaml
nsm_parameters:
  storage_noise_r: 0.75          # Storage decoherence parameter
  storage_rate_nu: 0.002         # Fraction of storable qubits
  delta_t_ns: 1_000_000          # Wait time (1 ms)
  channel_fidelity: 0.9999685
  detection_eff_eta: 0.0150      # Total detection efficiency
  detector_error: 0.0093
  dark_count_prob: 1.50e-8

model_selection:
  link_model: "auto"             # "perfect" | "depolarise" | "heralded"
  eta_semantics: "detector_only"

channel_parameters:
  length_km: 0.0
  attenuation_db_per_km: 0.2
  t1_ns: 100_000_000
  t2_ns: 10_000_000
  cycle_time_ns: 1_000_000

network:
  alice_name: "Alice"
  bob_name: "Bob"
  num_qubits: 10
  with_device_noise: true
```

### Programmatic Configuration

```python
from caligo.simulation import NSMParameters, ChannelParameters
from caligo.protocol import ProtocolParameters, run_protocol

params = ProtocolParameters(
    session_id="test-session-001",
    nsm_params=NSMParameters(
        delta_t_ns=1_000_000,
        storage_noise_r=0.75,
        detection_eff_eta=0.85,
    ),
    num_pairs=10_000,
    num_qubits=10,
)

ot_output, raw_results = run_protocol(params, bob_choice_bit=1)
```

---

## Installation & Dependencies

### Installation

```bash
# Clone repository
git clone https://github.com/A-DaRo/qia-challenge-2025.git
cd qia-challenge-2025/caligo

# Basic installation (no simulation)
pip install -e .

# With SquidASM simulation support
pip install -e ".[simulation]"

# With development tools (pytest, mypy, black, ruff)
pip install -e ".[dev]"
```

### Dependency Graph

```
Core Dependencies (always required):
├── numpy >= 1.21              # Numerical arrays
├── numba >= 0.63.1            # JIT compilation (LDPC decoder)
├── bitarray >= 2.6            # Efficient bit manipulation
└── scipy >= 1.7               # Statistical functions

Simulation Dependencies (optional):
├── squidasm >= 0.12           # Quantum network simulator
├── netqasm >= 0.15, < 2       # Quantum assembly SDK
└── netsquid >= 1.1            # Discrete-event simulator

Development Dependencies (optional):
├── pytest >= 7.0
├── pytest-cov >= 4.0
├── mypy >= 1.0
├── black >= 23.0
└── ruff >= 0.1
```

---

## Entry Points & API Surface

### Public API (`from caligo import ...`)

The public API is defined in [`caligo/__init__.py`](caligo/__init__.py) and exports:

#### Type System
```python
# Keys
ObliviousKey, AliceObliviousKey, BobObliviousKey

# Measurements
MeasurementRecord, RoundResult, DetectionEvent

# Phase Contracts
QuantumPhaseResult, SiftingPhaseResult, ReconciliationPhaseResult,
AmplificationPhaseResult, ObliviousTransferOutput

# Enums
ProtocolPhase, AbortReason

# Exceptions
CaligoError, SimulationError, SecurityError, ProtocolError,
ConnectionError, ConfigurationError
```

#### Utilities
```python
# Logging
get_logger, setup_script_logging

# Math
binary_entropy, channel_capacity, finite_size_penalty, gamma_function,
smooth_min_entropy_rate, key_length_bound

# Bitarray Operations
xor_bitarrays, hamming_distance, random_bitarray, bitarray_to_bytes,
bytes_to_bitarray, slice_bitarray, bitarray_from_numpy, bitarray_to_numpy
```

#### Simulation
```python
# Physical Model
NSMParameters, ChannelParameters, create_depolar_noise_model,
create_t1t2_noise_model

# Timing
TimingBarrier, TimingBarrierState

# Noise Models
NSMStorageNoiseModel, ChannelNoiseProfile

# Network Builder
CaligoNetworkBuilder, perfect_network_config, realistic_network_config,
erven_experimental_config
```

#### Security
```python
# Constants
QBER_CONSERVATIVE_THRESHOLD, QBER_ABSOLUTE_THRESHOLD, R_TILDE, R_CROSSOVER,
DEFAULT_EPSILON_SEC, DEFAULT_EPSILON_COR

# Bounds
collision_entropy_rate, dupuis_konig_bound, lupo_virtual_erasure_bound,
max_bound_entropy, rational_adversary_bound, bounded_storage_entropy,
strong_converse_exponent

# Feasibility
FeasibilityChecker, FeasibilityResult, PreflightReport, compute_expected_qber

# Finite-Key
compute_statistical_fluctuation, hoeffding_detection_interval,
compute_finite_key_length
```

#### Quantum Operations
```python
EPRGenerator, EPRGenerationConfig, BasisSelector, MeasurementExecutor,
MeasurementBuffer, BatchingManager, BatchConfig, BatchResult
```

#### Sifting
```python
SHA256Commitment, CommitmentResult, Sifter, SiftingResult, QBEREstimator,
QBEREstimate, DetectionValidator, ValidationResult, HoeffdingBound
```

#### Amplification
```python
NSMEntropyCalculator, SecureKeyLengthCalculator, KeyLengthResult,
ToeplitzHasher, OTOutputFormatter, AliceOTOutput, BobOTOutput
```

### CLI Entry Point

```bash
# Run parallel EPR generation (standalone)
python -m caligo.cli --num-pairs 100000 --parallel --workers 8

# From YAML config
python -m caligo.cli --config configs/parallel.yaml
```

---

## Testing Infrastructure

### Test Organization

```
tests/
├── conftest.py                   # Shared fixtures
├── test_types/                   # Type system validation
├── test_utils/                   # Utility function tests
├── test_quantum/                 # Quantum operation tests
├── test_sifting/                 # Sifting phase tests
├── test_reconciliation/          # LDPC encoding/decoding tests
├── test_amplification/           # Privacy amplification tests
├── test_security/                # Security bound validation
├── test_simulation/              # Network builder tests
├── integration/                  # Multi-package integration (no E2E)
├── e2e/                          # Full SquidASM simulations
└── performance/                  # Benchmarking tests
```

### Pytest Markers

```python
@pytest.mark.integration  # Multi-package integration (no SquidASM)
@pytest.mark.e2e          # End-to-end simulation (requires SquidASM)
@pytest.mark.slow         # Tests > 1s execution time
@pytest.mark.performance  # Performance/overhead checks
@pytest.mark.security     # Statistical security assumption validation
```

### Running Tests

```bash
# All tests
pytest

# Fast tests only (skip slow + e2e)
pytest -m "not slow and not e2e"

# Integration tests
pytest -m integration

# End-to-end tests (requires SquidASM)
pytest -m e2e

# With coverage
pytest --cov=caligo --cov-report=html

# Specific module
pytest tests/test_reconciliation/ -v
```

---

## Design Patterns & Conventions

### Strategy Pattern

**Usage**: Algorithm selection without modifying orchestration logic.

**Implementations**:
1. **EPR Generation** (`quantum/factory.py`):
   - `EPRGenerationStrategy` protocol
   - `SequentialEPRStrategy` (single-process)
   - `ParallelEPRStrategy` (multiprocessing)

2. **Reconciliation** (`reconciliation/factory.py`):
   - `ReconciliationStrategy` protocol
   - `BaselineStrategy` (Alice-driven, known QBER)
   - `BlindStrategy` (Bob-driven, unknown QBER)

**Example**:
```python
# Factory creates strategy based on config
factory = EPRGenerationFactory(config)
strategy = factory.create_strategy()

# Client code is strategy-agnostic
results = strategy.generate(num_pairs)
```

### Factory Pattern

**Usage**: Encapsulate complex object creation.

**Implementations**:
- `EPRGenerationFactory`: Creates EPR strategy from `CaligoConfig`
- `create_strategy()`: Creates reconciliation strategy from `ReconciliationConfig`
- `CaligoNetworkBuilder`: Creates `StackNetworkConfig` from `NSMParameters`

### Contract-Driven Development

**Principle**: Every phase boundary is a formal contract with runtime validation.

**Implementation**:
```python
@dataclass
class QuantumPhaseResult:
    # ... fields ...
    
    def __post_init__(self) -> None:
        """Validate post-conditions."""
        if len(self.measurement_outcomes) != self.num_pairs_generated:
            raise ContractViolation("POST-Q-001: Length mismatch")
        # ... more checks ...
```

### Numpydoc Docstring Standard

**Convention**: All public functions use Numpydoc format.

```python
def calculate_qber(errors: int, total: int) -> float:
    """
    Calculate quantum bit error rate.

    Parameters
    ----------
    errors : int
        Number of bit errors detected.
    total : int
        Total number of sifted bits.

    Returns
    -------
    float
        QBER in range [0, 1].

    Raises
    ------
    ValueError
        If total <= 0 or errors > total.

    Notes
    -----
    QBER is defined as:
    
        Q = errors / total
    
    Security constraint: Q must be ≤ 11% for NSM security.

    References
    ----------
    - Erven et al. (2014), Table I
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if errors > total:
        raise ValueError("errors cannot exceed total")
    return errors / total
```

### Logging Convention

**Rule**: Use `LogManager.get_stack_logger(__name__)`, **never `print()`**.

```python
from caligo.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Starting Phase II sifting")
logger.warning("QBER %.4f exceeds conservative threshold", qber)
logger.error("Reconciliation failed: %s", exc)
```

---

## Security Guarantees & Bounds

### NSM Security Model

**Assumption**: Eve has bounded quantum memory with storage capacity $\nu$ and decoherence parameter $r$.

**Security Condition**:
$$
C_N \cdot \nu < \frac{1}{2}
$$

Where $C_N = 1 - h(\text{depolar\_prob})$ is the quantum channel capacity.

**Verification**:
```python
feasibility_checker = FeasibilityChecker()
result = feasibility_checker.check(nsm_params, expected_qber)

if not result.is_feasible:
    raise SecurityError(f"NSM security impossible: {result.failure_reason}")
```

### Min-Entropy Bounds

Caligo implements multiple entropy bounds from literature:

1. **Dupuis-König Bound** (2012): General NSM bound
2. **Lupo Virtual Erasure Bound** (2023): Optimized for high $\eta$
3. **Rational Adversary Bound**: Conservative lower bound

**Selection**:
```python
calculator = NSMEntropyCalculator(nsm_params)
h_min = calculator.calculate_rate(
    qber=0.05,
    bound_type="max_bound",  # Uses max(Dupuis-König, Lupo)
)
```

### Finite-Key Corrections

**Hoeffding Bound** for detection efficiency:
$$
P(|\hat{\eta} - \eta| \geq \delta) \leq 2 \exp\left(-2n\delta^2\right)
$$

**Leftover Hash Lemma** security penalty:
$$
\text{penalty} = 2 \log_2\left(\frac{1}{\epsilon_{\text{sec}}}\right)
$$

---

## References

### Foundational Papers

1. **Erven, C., et al. (2014)**. "An experimental implementation of oblivious transfer in the noisy storage model." *Nature Communications*, 5, 3418.
   - Table I: Reference NSM parameters

2. **Schaffner, C., et al. (2009)**. "Robust cryptography in the noisy-quantum-storage model." *Physical Review A*, 79(3), 032308.
   - Theoretical foundation of NSM security
   - Definition of ε-secure 1-2 ROT

3. **König, R., et al. (2012)**. "Unconditional security from noisy quantum storage." *IEEE Transactions on Information Theory*, 58(3), 1962-1984.
   - Min-entropy bounds for NSM
   - Storage capacity constraints

4. **Lupo, C., et al. (2023)**. "Practical quantum oblivious key distribution with imperfect devices." *arXiv:2305.xxxxx*.
   - Finite-key length formula (Eq. 43)
   - Virtual erasure channel optimization

### LDPC Coding

5. **Hu, X.-Y., et al. (2005)**. "Regular and irregular progressive edge-growth tanner graphs." *IEEE Transactions on Information Theory*, 51(1), 386-398.
   - Progressive Edge Growth (PEG) algorithm

6. **MacKay, D. J. C., & Neal, R. M. (1997)**. "Near Shannon limit performance of low density parity check codes." *Electronics Letters*, 33(6), 457-458.
   - Belief Propagation decoding

### Implementation References

- **SquidASM Documentation**: [`docs/squidasm_docs/`](docs/squidasm_docs/)
- **NetQASM SDK**: https://netqasm.readthedocs.io/
- **NetSquid Simulator**: https://netsquid.org/

---

## License

MIT License — See [LICENSE](../LICENSE.md) for details.

---

**Maintainer**: Alessandro Da Ros  
**Repository**: https://github.com/A-DaRo/qia-challenge-2025  
**Documentation**: [`qia-challenge-2025/docs/`](../docs/)

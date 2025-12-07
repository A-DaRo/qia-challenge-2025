[← Return to Main Index](../index.md)

# 2.1 The SquidASM Simulation Framework

## 2.1.1 Overview: The Three-Tier Architecture

The implementation of the E-HOK protocol relies on **SquidASM**, a comprehensive quantum network simulation framework consisting of three tightly integrated layers:

```
┌────────────────────────────────────────────────────────────────┐
│                      SquidASM Layer                            │
│  (Application Programs & High-Level Protocol Orchestration)    │
│  - Program, ProgramContext, ProgramMeta                        │
│  - Classical/EPR Socket Abstractions                           │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      NetQASM Layer                             │
│  (Quantum Assembly Language SDK & Network Stack)               │
│  - Connection, Builder, Subroutine Management                  │
│  - Futures, SharedMemory                                       │
│  - EPRSocket (create/recv operations)                          │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      NetSquid Layer                            │
│  (Discrete Event Simulator Core)                               │
│  - Quantum State Management (Qubits, Entanglement)             │
│  - Noise Models (Depolarization, Dephasing, T1/T2)             │
│  - Event Scheduler (pydynaa)                                   │
└────────────────────────────────────────────────────────────────┘
```

Each layer provides distinct abstractions while maintaining strict separation of concerns. This section derives the architectural principles of each layer in order, culminating in the connection points relevant to E-HOK.

---

## 2.1.2 Layer 1: NetSquid — The Discrete Event Simulation Core

### 2.1.2.1 Quantum State Representation

**NetSquid** is the foundational simulation engine built on the **pydynaa** discrete event framework. At its core, NetSquid maintains quantum states as density matrices, allowing full modeling of mixed states and decoherence.

**Qubit State Management:**
NetSquid represents qubits using the `Qubit` class, which internally tracks:
- **Pure states**: $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ (as state vectors)
- **Mixed states**: $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$ (as density matrices)

For entangled systems (e.g., the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$), NetSquid stores the joint state in a shared quantum system, automatically handling partial traces during measurement.

**Example: EPR Pair Generation**
When an EPR pair is generated between nodes Alice and Bob:
1. NetSquid creates a composite quantum system $\mathcal{H}_A \otimes \mathcal{H}_B$
2. Initializes the state to $|\Phi^+\rangle$ (or a noisy approximation)
3. Assigns one subsystem to Alice's quantum memory, the other to Bob's
4. Tracks the entanglement correlation until measurement collapses the joint state

### 2.1.2.2 Noise Models

NetSquid implements physically realistic noise channels that degrade quantum states over time or during operations. The primary models relevant to E-HOK are:

**1. Depolarizing Channel:**
$$\mathcal{E}_{\text{depol}}(\rho) = (1 - p)\rho + p\frac{\mathbb{I}}{2}$$

Where $p$ is the depolarization probability. For an EPR pair, this introduces bit-flip and phase-flip errors, directly contributing to the QBER (Quantum Bit Error Rate).

**Configuration in NetSquid:**
```python
from netsquid.components.models.qerrormodels import DepolarNoiseModel

# Time-dependent depolarization
noise_model = DepolarNoiseModel(
    depolar_rate=1e6,  # 1 MHz depolarization rate
    time_independent=False
)

# Apply to quantum memory
quantum_memory.add_error_model(noise_model)
```

**2. T1/T2 Relaxation:**
Models realistic qubit decay via:
- **T1 (amplitude damping)**: $|1\rangle \to |0\rangle$ energy relaxation
- **T2 (dephasing)**: $|+\rangle \to$ mixed state

**Configuration:**
```python
from netsquid.components.models.qerrormodels import T1T2NoiseModel

noise_model = T1T2NoiseModel(
    T1=1e6,  # 1 ms T1 time (nanoseconds)
    T2=5e5   # 0.5 ms T2 time
)
```

**Impact on E-HOK:**
The fidelity $F$ of the distributed EPR state determines the raw QBER:
$$\text{QBER}_{\text{raw}} \approx \frac{3}{4}(1 - F) \quad \text{(for depolarizing noise)}$$

For the baseline, we target $F \geq 0.95$ (QBER $\leq 3.75\%$), well below the abort threshold of 11%.

### 2.1.2.3 Discrete Event Scheduler

NetSquid uses **pydynaa**, a generator-based discrete event simulation framework. All components (quantum memories, links, protocols) are implemented as Python generators that `yield` control to the scheduler at specific time points.

**Event Scheduling Principle:**
```python
class QuantumProtocol(Protocol):
    def run(self):
        # Wait 1000 ns for EPR generation
        yield self.await_timer(duration=1000)
        
        # Process quantum operation
        qubit = self.quantum_memory.pop()
        
        # Wait for classical message
        yield self.await_port_input(self.port)
        
        # Measurement (instantaneous in simulation time)
        outcome = qubit.measure()
```

**Critical Property:** The scheduler maintains a global simulation clock. Events are processed in **strict chronological order**, ensuring causal consistency (no "spooky action at a distance" artifacts in the simulation).

---

## 2.1.3 Layer 2: NetQASM — The Quantum Network SDK

### 2.1.3.1 Connection & Builder Pattern

**NetQASM** provides a high-level SDK for quantum network programming, abstracting away NetSquid's low-level APIs. The core abstraction is the `Connection` object, which manages:
- **Builder**: Accumulates quantum operations before sending to the quantum node controller
- **SharedMemory**: Classical-quantum interface for storing measurement outcomes
- **Futures**: Promises for quantum operation results

**Execution Model:**
```python
from netqasm.sdk import EPRSocket, Qubit

# Operations are buffered (not executed yet)
with connection:
    q = Qubit(connection)
    q.H()
    m = q.measure()  # Returns a Future (value not yet available)

# flush() compiles and executes the subroutine
yield from connection.flush()

# NOW the measurement outcome is available
outcome_value = int(m)  # Future resolved to integer
```

**Key Insight:** The `flush()` operation is the **synchronization boundary** between the application layer and the quantum hardware. All operations within a `with connection:` block are batched and executed atomically at the `flush()` call.

### 2.1.3.2 The Future Abstraction

**Problem:** Quantum operations take time, and results are not immediately available.

**Solution:** NetQASM uses `Future` objects—placeholders that resolve after `flush()`.

**Implementation (from `netqasm/sdk/futures.py`):**
```python
class Future(int):
    """A Future represents a classical value that becomes available 
    at some point in the future."""
    
    @property
    def value(self) -> Optional[int]:
        """Get the value. Returns None if not yet resolved."""
        if self._value is not None:
            return self._value
        else:
            # Check SharedMemory
            return self._try_get_value()
```

**Usage in E-HOK:**
```python
# Create array for storing 100 measurement outcomes
outcomes = connection.new_array(length=100)

for i in range(100):
    q = Qubit(connection)
    q.H()  # Apply Hadamard (X-basis)
    outcomes.get_future_index(i) << q.measure()

# Execute all 100 measurements
yield from connection.flush()

# Now extract values
for i in range(100):
    bit = int(outcomes.get_future_index(i))  # Resolved!
```

**Critical Property:** Futures ensure **type safety** and prevent "time-travel bugs" where code accidentally accesses quantum results before they're computed.

### 2.1.3.3 EPRSocket API

The `EPRSocket` abstracts entanglement distribution between two nodes. It provides three operational modes:

| Mode | Method | Quantum Memory | Control | E-HOK Usage |
|------|--------|----------------|---------|-------------|
| **Keep** | `create_keep()` / `recv_keep()` | Occupies memory | Full (apply gates, then measure) | Baseline (allows independent basis choice) |
| **Measure** | `create_measure()` / `recv_measure()` | Zero (immediate measurement) | Basis at creation time | High-throughput alternative |
| **Remote** | `create_rsp()` / `recv_rsp()` | Asymmetric | Creator measures, receiver keeps | Not used in baseline |

**E-HOK Baseline Design Decision:**
We use **EPRType.K (Keep)** to maintain protocol symmetry—both Alice and Bob independently choose their measurement basis *after* receiving the qubit.

**API Signature (Create Side):**
```python
def create_keep(
    self,
    number: int = 1,
    sequential: bool = False,
    post_routine: Optional[Callable] = None,
    time_unit: TimeUnit = TimeUnit.MICRO_SECONDS,
    max_time: int = 0,
) -> Union[List[Qubit], FutureQubit]:
    """
    Generate EPR pairs and keep both qubits in memory.
    
    Parameters
    ----------
    number : int
        Number of EPR pairs to generate.
    sequential : bool
        If True, generate one pair at a time (allows processing 
        before next generation—critical for memory-limited systems).
    post_routine : Callable
        Function to call after each pair generation (used with sequential=True).
    """
```

**API Signature (Receive Side):**
```python
def recv_keep(
    self,
    number: int = 1,
    sequential: bool = False,
    post_routine: Optional[Callable] = None,
    time_unit: TimeUnit = TimeUnit.MICRO_SECONDS,
    max_time: int = 0,
) -> Union[List[Qubit], FutureQubit]:
    """
    Receive EPR pairs from remote node.
    
    CRITICAL: Every create_keep() must be matched by a recv_keep().
    The simulation blocks until both nodes execute their respective calls.
    """
```

**Synchronization Semantics:**
The EPR generation is a **blocking bilateral operation**. The simulation timeline does not advance past the EPR creation until *both* nodes have executed their respective `create_*` / `recv_*` calls and the network has distributed the entangled state.

---

## 2.1.4 Layer 3: SquidASM — Application Protocol Orchestration

### 2.1.4.1 The Program Abstraction

**SquidASM** provides the top-level orchestration layer where quantum network protocols are defined. Each node runs a `Program`, which is a Python generator function that coordinates quantum operations and classical communication.

**Program Structure:**
```python
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

class AliceEHOKProgram(Program):
    """Alice's E-HOK protocol implementation."""
    
    @property
    def meta(self) -> ProgramMeta:
        """Declare required network resources."""
        return ProgramMeta(
            name="alice_ehok",
            csockets=["bob"],      # Classical socket to Bob
            epr_sockets=["bob"],   # Quantum (EPR) socket to Bob
            max_qubits=5           # Request 5 qubits in quantum memory
        )
    
    def run(self, context: ProgramContext):
        """Main protocol execution (generator function)."""
        # Access network resources
        conn = context.connection
        epr_socket = context.epr_sockets["bob"]
        csocket = context.csockets["bob"]
        
        # Protocol logic (yields control to scheduler)
        # ...
        yield from self._generate_raw_bits(conn, epr_socket)
        yield from self._perform_commitment(csocket)
        # ...
```

**ProgramMeta Definition (from `squidasm/sim/stack/program.py`):**
```python
@dataclass
class ProgramMeta:
    """Metadata describing a program's network requirements."""
    
    name: str                      # Unique identifier for the program
    csockets: List[str]            # Names of nodes to connect via classical sockets
    epr_sockets: List[str]         # Names of nodes to connect via EPR sockets
    max_qubits: int = 1            # Number of qubits needed in quantum memory
```

**Critical Property:** The `meta` property is read *before* execution to configure the network topology. It acts as a "resource declaration" that the SquidASM runtime uses to set up the simulation.

### 2.1.4.2 Classical Communication: StructuredMessage

SquidASM wraps NetSquid's classical channels with the `ClassicalSocket` abstraction, providing reliable, authenticated message passing.

**Message Structure:**
```python
from squidasm.sim.stack.csocket import StructuredMessage

class StructuredMessage:
    """A structured message with a header and payload."""
    
    def __init__(self, header: str, payload: Any):
        self.header = header
        self.payload = payload
```

**Usage Pattern:**
```python
# Alice sends Bob's commitment request
commitment = compute_commitment(raw_data)
csocket.send_structured(
    StructuredMessage("COMMITMENT", commitment)
)

# Bob receives and verifies
msg = yield from csocket.recv_structured()
if msg.header != "COMMITMENT":
    raise ProtocolError("Expected commitment message")

bob_commitment = msg.payload
```

**Blocking Semantics:**
- `send_structured()`: Non-blocking (message queued)
- `recv_structured()`: **Blocking** (yields until message arrives)

This asymmetry is critical for enforcing protocol ordering in E-HOK (e.g., ensuring Bob's commitment is received before Alice sends bases).

### 2.1.4.3 Network Configuration & Topology

SquidASM uses YAML configuration files to define the network topology, noise parameters, and link properties.

**Example: Two-Node E-HOK Network**
```yaml
# configs/network_baseline.yaml
stacks:
  - name: alice
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 1000000      # 1 ms T1 time (nanoseconds)
      T2: 500000       # 0.5 ms T2 time

  - name: bob
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 1000000
      T2: 500000

links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.97           # 97% fidelity → ~2.25% QBER
      prob_success: 1.0        # No generation failures
      t_cycle: 1000            # 1 μs per EPR pair
```

**Loading Configuration:**
```python
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

# Load YAML config
config = StackNetworkConfig.from_file("configs/network_baseline.yaml")

# Define programs for each node
alice_program = AliceEHOKProgram()
bob_program = BobEHOKProgram()

# Execute simulation
results = run(
    config=config,
    programs={"alice": alice_program, "bob": bob_program},
    num_times=1
)
```

---

## 2.1.5 E-HOK Integration: Key Connection Points

### 2.1.5.1 Phase I: Quantum Generation

**Requirement:** Generate $N = 10,000$ raw correlated bit pairs using entanglement.

**Challenge:** Quantum memory is limited to 5 qubits (configurable, but realistic for near-term hardware).

**Solution: Sequential EPR Generation with Immediate Measurement**

```python
# In AliceEHOKProgram.run()
raw_outcomes = []
raw_bases = []

TOTAL_PAIRS = 10_000
BATCH_SIZE = 5  # Limited by quantum memory

for batch_idx in range(TOTAL_PAIRS // BATCH_SIZE):
    batch_qubits = []
    batch_bases = []
    
    # Generate batch of EPR pairs
    with conn:
        qubits = epr_socket.create_keep(number=BATCH_SIZE)
        
        # Choose random basis for each qubit
        for q in qubits:
            basis = random.randint(0, 1)  # 0=Z, 1=X
            if basis == 1:
                q.H()  # Apply Hadamard for X-basis measurement
            
            batch_qubits.append(q)
            batch_bases.append(basis)
    
    # Execute measurements (blocking)
    yield from conn.flush()
    
    # Extract results (Futures now resolved)
    for q in batch_qubits:
        outcome = int(q.measure())  # Now an integer
        raw_outcomes.append(outcome)
    
    raw_bases.extend(batch_bases)
    
    # Qubits are automatically deallocated after measurement
```

**Bob's Symmetric Implementation:**
```python
# In BobEHOKProgram.run()
raw_outcomes = []
raw_bases = []

for batch_idx in range(TOTAL_PAIRS // BATCH_SIZE):
    batch_qubits = []
    batch_bases = []
    
    with conn:
        # Receive EPR pairs (blocking until Alice creates them)
        qubits = epr_socket.recv_keep(number=BATCH_SIZE)
        
        # Independently choose random basis
        for q in qubits:
            basis = random.randint(0, 1)
            if basis == 1:
                q.H()
            
            batch_qubits.append(q)
            batch_bases.append(basis)
    
    yield from conn.flush()
    
    for q in batch_qubits:
        outcome = int(q.measure())
        raw_outcomes.append(outcome)
    
    raw_bases.extend(batch_bases)
```

**Simulation Timing Guarantee:**
Due to NetSquid's discrete event scheduler and the blocking semantics of EPR operations:
1. Alice's `create_keep()` and Bob's `recv_keep()` execute simultaneously in simulation time
2. The EPR state is generated by the network layer *before* either party's `flush()` completes
3. Both measurements occur at the same simulation timestamp, ensuring no timing-based attacks

### 2.1.5.2 Phase II: Commitment (Classical Communication)

**Requirement:** Bob commits to his measurement outcomes *before* Alice reveals her bases.

**Implementation Using StructuredMessage:**

```python
# Bob's Program (after quantum generation)
import hashlib

# Concatenate outcomes and bases
commitment_string = "".join([f"{o}{b}" for o, b in zip(raw_outcomes, raw_bases)])
commitment_hash = hashlib.sha256(commitment_string.encode()).hexdigest()

# Send commitment to Alice
csocket.send_structured(
    StructuredMessage("COMMITMENT", commitment_hash)
)

# --- Alice's Program ---
# Block until commitment is received
commitment_msg = yield from csocket.recv_structured()

if commitment_msg.header != "COMMITMENT":
    raise ProtocolError("Expected commitment message")

bob_commitment = commitment_msg.payload

# ONLY NOW send basis information
csocket.send_structured(
    StructuredMessage("BASES", raw_bases)
)
```

**Security Property:**
The `recv_structured()` yield ensures Alice's program execution is suspended until Bob's commitment arrives. The SquidASM event scheduler enforces that Bob's commitment is timestamped *before* Alice's basis reveal, preventing look-ahead attacks.

### 2.1.5.3 Extracting Noise Parameters for QBER Estimation

**Runtime Configuration Access:**
```python
from squidasm.run.stack.config import StackNetworkConfig

config = StackNetworkConfig.from_file("configs/network_baseline.yaml")

# Extract link fidelity
for link in config.links:
    if hasattr(link.cfg, 'fidelity'):
        fidelity = link.cfg.fidelity
        print(f"Link fidelity: {fidelity}")
        
        # Estimate QBER for depolarizing noise
        qber_estimate = 0.75 * (1 - fidelity)
        print(f"Expected QBER: {qber_estimate:.4f}")
```

**Output Example:**
```
Link fidelity: 0.97
Expected QBER: 0.0225
```

This allows the protocol to dynamically adjust reconciliation parameters (e.g., LDPC code rate) based on the configured noise level.

---

## 2.1.6 State Generation & Measurement Mechanics

### 2.1.6.1 Bell State Creation (NetSquid Internal)

When `epr_socket.create_keep()` is called, SquidASM delegates to NetSquid's `MagicDistributor` (a simplified entanglement source that directly injects Bell states):

**Ideal Case (No Noise):**
```python
# Internal NetSquid operation (simplified)
from netsquid.qubits import create_qubits, operate

def create_bell_pair():
    """Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    q1, q2 = create_qubits(2)
    
    # Apply H ⊗ I
    operate(q1, ns.H)
    
    # Apply CNOT
    operate([q1, q2], ns.CNOT)
    
    return q1, q2
```

**With Depolarizing Noise:**
```python
def create_noisy_bell_pair(fidelity):
    """Create Werner state ρ = F|Φ⁺⟩⟨Φ⁺| + (1-F)I/4"""
    q1, q2 = create_bell_pair()
    
    # Apply depolarizing channel with probability (1 - fidelity)
    for qubit in [q1, q2]:
        if random.random() < (1 - fidelity):
            # Apply random Pauli error (X, Y, or Z)
            error_gate = random.choice([ns.X, ns.Y, ns.Z])
            operate(qubit, error_gate)
    
    return q1, q2
```

**State Representation After Noise:**
$$\rho_{\text{AB}} = F|\Phi^+\rangle\langle\Phi^+| + (1 - F)\frac{\mathbb{I}_4}{4}$$

This is a **Werner state**, the standard model for noisy entanglement in quantum information.

### 2.1.6.2 Measurement Collapse & Correlation

When Alice and Bob measure in the same basis (e.g., both choose Z):

**Alice measures first (simulation perspective):**
```python
outcome_alice = q_alice.measure()  # Suppose outcome = 0
```

**NetSquid State Update:**
1. The joint state $\rho_{\text{AB}}$ is projected onto $|0\rangle_A \otimes \mathbb{I}_B$
2. Bob's qubit collapses to (ideally) $|0\rangle_B$ due to perfect correlation
3. Bob's subsequent measurement yields outcome = 0 with probability $F$ (or 1 with probability $1-F$ due to noise)

**Measured QBER:**
$$\text{QBER}_{\text{measured}} = \frac{\text{# of mismatched outcomes in same basis}}{\text{# of same-basis measurements}}$$

For depolarizing noise: $\text{QBER} \approx \frac{3}{4}(1 - F)$.

---

## 2.1.7 Summary: Three-Layer Integration in E-HOK

| Layer | Responsibility | E-HOK Usage |
|-------|----------------|-------------|
| **NetSquid** | Quantum state evolution, noise simulation, event scheduling | Generates noisy EPR pairs, enforces timing constraints |
| **NetQASM** | Quantum operation batching, Future resolution, EPR distribution | Buffers measurements, resolves outcomes after flush() |
| **SquidASM** | Protocol orchestration, classical communication, resource management | Implements commitment, sifting, reconciliation phases |

**Key Architectural Insight:**
The strict layering ensures **separation of concerns**:
- Quantum mechanics is handled by NetSquid (physics layer)
- Quantum network operations by NetQASM (SDK layer)
- Protocol security logic by SquidASM (application layer)

This modularity allows E-HOK to:
1. **Swap noise models** without changing protocol code (just update YAML config)
2. **Benchmark different reconciliation algorithms** (LDPC vs. Cascade) by replacing a single module
3. **Extend to multi-party protocols** by adding nodes to the configuration without refactoring the core EPR generation logic

---

[← Return to Main Index](../index.md) | [Next: Theoretical Underpinnings →](./theory.md)

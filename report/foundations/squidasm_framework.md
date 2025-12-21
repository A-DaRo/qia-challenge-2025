[← Return to Main Index](../index.md)

# 2.3 The SquidASM Simulation Framework

## 2.3.1 Overview: Quantum Network Simulation

**SquidASM** (Simulator for Quantum Information Distribution, Application-Specific Modules) [1] is a software framework for simulating quantum network applications built atop the **NetSquid** discrete-event simulator [2]. It provides an application-layer abstraction for quantum protocols while maintaining faithful representation of physical quantum network behavior.

**Design Philosophy**:
- **Application-First**: Protocol designers write Python code without managing low-level simulation details
- **Physically Realistic**: Noise models, timing, and network topology reflect real quantum hardware
- **Modular Architecture**: Separation of concerns (protocol logic vs. network stack)

**Stack Hierarchy**:

```
┌───────────────────────────────────────────────┐
│         Caligo Protocol (User Code)          │
├───────────────────────────────────────────────┤
│        SquidASM Application Layer             │
│  - NetQASMConnection (program compilation)    │
│  - Shared Memory (classical communication)    │
│  - StackNetworkConfig (topology definition)   │
├───────────────────────────────────────────────┤
│            NetQASM SDK Layer                  │
│  - Quantum Instructions (EPR, H, CNOT, etc.)  │
│  - Measurement & Classical Registers          │
│  - Compiler (NetQASM → NetSquid operations)   │
├───────────────────────────────────────────────┤
│          netsquid_netbuilder                  │
│  - Link/Device Configuration                  │
│  - MagicDistributor (EPR pair sources)        │
├───────────────────────────────────────────────┤
│         NetSquid Discrete-Event Core          │
│  - Event Scheduler (priority queue)           │
│  - QuantumProcessor (qubits, gates, noise)    │
│  - Channel (losses, timing)                   │
└───────────────────────────────────────────────┘
```

## 2.3.2 NetSquid: Discrete-Event Simulation

### Event-Driven Execution

NetSquid uses a **discrete-event kernel** [2] where simulation time advances through events, not continuous steps:

```python
# Pseudocode representation
while event_queue.not_empty():
    event = event_queue.pop_earliest()
    simulation_time = event.time
    event.execute()
    # Event execution may schedule new events
```

**Key Concept**: All operations (qubit gates, measurements, channel transmission) are **events** with explicit timestamps. This enables:

1. **Precise Timing**: Protocol delays ($\Delta t$) enforced at nanosecond resolution
2. **Concurrent Operations**: Multiple parties operate in parallel
3. **Reproducibility**: Deterministic execution for given random seed

### Quantum State Representation

NetSquid uses **density matrix** formalism:

$$
\rho = \sum_{i,j} \rho_{ij} |i\rangle\langle j|
$$

**Advantages**:
- Supports **mixed states** (thermal noise, decoherence)
- Enables **partial trace** for subsystem analysis
- Allows **CPTP maps** for noise modeling

**State Vectors**: For efficiency, pure states are tracked as $|\psi\rangle$ until entanglement or measurement forces density matrix conversion.

### Noise Models

NetSquid provides **noise model composition** [2, Section III-B]:

| Noise Model | Physical Basis | Parameters |
|-------------|----------------|------------|
| `DepolarNoiseModel` | Environmental decoherence | `depolar_rate`, `time_independent` |
| `DephaseNoiseModel` | Phase damping | `dephase_rate`, `T2_star` |
| `T1T2NoiseModel` | Amplitude and phase damping | `T1` (relaxation), `T2` (dephasing) |
| `FibreLossModel` | Photon absorption in fiber | `p_loss_init`, `p_loss_length` (per km) |

**Composition**: Models are applied **sequentially** during qubit operations:

$$
\rho_{\text{final}} = \mathcal{N}_k \circ \cdots \circ \mathcal{N}_2 \circ \mathcal{N}_1(\rho_{\text{initial}})
$$

## 2.3.3 SquidASM Application Interface

### NetQASMConnection: Quantum Programs

Quantum protocols are written as **generator functions** yielding NetQASM instructions:

```python
from netqasm.sdk import EPRSocket, Qubit
from squidasm.run.stack.run import run

def alice_program(conn: NetQASMConnection):
    # Create EPR pair with Bob
    epr_socket = EPRSocket("Bob")
    q = epr_socket.recv_keep()[0]  # Receive and keep qubit
    
    # Measure in random basis
    if random.choice([0, 1]) == 0:
        outcome = q.measure()  # Computational basis
    else:
        q.H()  # Hadamard gate
        outcome = q.measure()  # Hadamard basis
    
    yield from conn.flush()  # Execute compiled instructions
    return int(outcome)
```

**Key Features**:
- **Lazy Execution**: Instructions accumulate until `flush()`
- **Classical Registers**: Store measurement outcomes
- **Typed Operations**: Compile-time type checking

### Shared Memory: Classical Communication

Parties communicate classically via `SharedMemoryManager`:

```python
from squidasm.run.stack.context import SharedMemoryManager

# Alice writes
shared_mem = SharedMemoryManager()
shared_mem.set("basis_info", basis_list)

# Bob reads
basis_info = shared_mem.get("basis_info")
```

**Semantics**:
- **Blocking Reads**: `get()` waits until data available
- **Atomic Writes**: No race conditions in single-threaded simulation
- **Persistent**: Data survives across protocol rounds

### Network Configuration

Topology is defined via `StackNetworkConfig`:

```yaml
# network_config.yaml
stacks:
  - name: Alice
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 1000000000  # 1 second (ns)
      T2: 500000000   # 0.5 seconds
      
  - name: Bob
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 1000000000
      T2: 500000000

links:
  - stack1: Alice
    stack2: Bob
    typ: depolarise
    cfg:
      fidelity: 0.95
      t_cycle: 1000  # 1 microsecond
```

**Link Types**:
- `perfect`: Ideal EPR pair generation
- `depolarise`: Depolarizing noise with fixed fidelity
- `heralded-double-click`: Detector-based heralding with dark counts

## 2.3.4 NetQASM: Quantum Assembly Language

### Instruction Set

NetQASM compiles high-level operations to **quantum assembly**:

| High-Level (Python) | NetQASM Instruction | Physical Operation |
|---------------------|---------------------|-------------------|
| `q.H()` | `rot_z q 4 8; rot_x q 4 8` | Hadamard gate |
| `q.measure()` | `meas q M0` | Z-basis measurement |
| `epr_socket.create()` | `create_epr A B M0` | EPR pair generation |
| `q.cnot(q2)` | `cphase q q2 16; rot_z q2 8` | CNOT gate |

**Compilation**: SquidASM compiler translates NetQASM → NetSquid gate sequences with explicit timing.

### Timing Semantics

Each NetQASM instruction has **duration**:

```python
# Example: EPR pair creation timing
t_start = 0 ns
t_entanglement_ready = t_start + t_cycle  # Link cycle time
t_measurement = t_entanglement_ready + gate_duration("M")
```

**NSM Enforcement**: Timing barriers implemented via:

```python
import time
time.sleep(delta_t_seconds)  # Real-time wait in simulation
```

## 2.3.5 netsquid_magic: EPR Pair Distribution

### MagicDistributor

The `MagicDistributor` [3] provides **heralded entanglement generation**:

```python
from netsquid_magic.magic_distributor import (
    MagicDistributor, 
    DoubleClickModelParameters
)

params = DoubleClickModelParameters(
    detector_efficiency=0.9,      # η
    dark_count_probability=1e-6,  # P_dark
    ...
)

distributor = MagicDistributor(
    nodes=["Alice", "Bob"],
    model_params=params,
    state_delay=1000,  # ns
)
```

**Double-Click Model**: Simulates **polarization-encoded photons** with:
1. Photon loss (detector inefficiency)
2. Dark counts (spurious detections)
3. Which-path information leakage

### Noise Injection

Noise is applied at **multiple layers**:

1. **Source Noise**: Fidelity of generated EPR state $|\Phi^+\rangle$
2. **Channel Noise**: Depolarization during transmission
3. **Detector Noise**: Measurement errors, dark counts

**Caligo Mapping**: NSM parameter $r$ maps to source fidelity:

$$
F_{\text{source}} = \frac{1 + 3r}{4}
$$

(Derivation in [Chapter 8.2: NSM-to-Physical Mapping](../nsm/physical_mapping.md))

## 2.3.6 Caligo Integration Patterns

### Pattern 1: Batched EPR Generation

```python
def generate_epr_batch(conn, remote_name, batch_size):
    epr_socket = EPRSocket(remote_name)
    qubits = []
    
    for _ in range(batch_size):
        q = epr_socket.recv_keep()[0]
        qubits.append(q)
    
    yield from conn.flush()  # Single flush for entire batch
    return qubits
```

**Trade-off**: Larger batches amortize compilation overhead but increase memory.

### Pattern 2: Basis-Dependent Measurement

```python
def measure_in_basis(qubit, basis):
    if basis == 0:  # Computational
        outcome = qubit.measure()
    else:  # Hadamard
        qubit.H()
        outcome = qubit.measure()
    return outcome
```

**Timing**: Hadamard gate adds $\sim$ 10-100 ns (hardware-dependent).

### Pattern 3: Timing Barrier

```python
from squidasm.util.routines import timing_barrier

def enforce_nsm_delay(conn, delta_t_ns):
    yield from timing_barrier(delta_t_ns)
```

**Implementation**: Schedules dummy event at `current_time + delta_t_ns`.

## 2.3.7 Simulation Fidelity vs. Reality

### What SquidASM Models Accurately

✅ **Discrete-event timing** (nanosecond precision)  
✅ **Density matrix evolution** (mixed states, decoherence)  
✅ **Link losses** (fiber attenuation, detector efficiency)  
✅ **Gate errors** (depolarization, dephasing)  
✅ **Parallel execution** (asynchronous protocols)

### Limitations

❌ **Continuous dynamics**: No master equation integration  
❌ **Hardware-specific quirks**: Crosstalk, SPAM errors  
❌ **Photon number resolution**: Assumes single-photon regime  
❌ **Non-Markovian noise**: Assumes memoryless channels

**Implication**: Caligo results are **upper bounds** on performance—real hardware will exhibit additional degradation.

---

## References

[1] SquidASM Documentation, QuTech Delft. https://github.com/QuTech-Delft/squidasm

[2] T. Coopmans et al., "NetSquid, a NETwork Simulator for QUantum Information using Discrete events," *Commun. Phys.* **4**, 164 (2021).

[3] NetSquid Magic Documentation. https://netsquid.org/

---

[← Return to Main Index](../index.md) | [← Previous: Cryptographic Primitives](./primitives.md) | [Next: Protocol Literature →](./protocol_literature.md)

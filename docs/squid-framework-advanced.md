# SquidASM Framework Advanced Technical Reference
**Extended Analysis for E-HOK Baseline and Industrial R&D Requirements**

This document extends `e-hok-squid-framework.md` with deeper technical analysis focused on the advanced requirements from `e-hok-baseline.md` and `e-hok-extensions.md`. Topics include network topology configuration, comprehensive noise modeling, batching/streaming operations, distributed simulation architecture, and MDI protocol support.

---

## Table of Contents

1. [Network Topology & Configuration](#1-network-topology--configuration)
2. [Comprehensive Noise Model Architecture](#2-comprehensive-noise-model-architecture)
3. [Batching & Sequential Mode Operations](#3-batching--sequential-mode-operations)
4. [Distributed Simulation Architecture](#4-distributed-simulation-architecture)
5. [Timing Guarantees & Protocol Synchronization](#5-timing-guarantees--protocol-synchronization)
6. [Heralded Links & MDI Architecture Support](#6-heralded-links--mdi-architecture-support)
7. [E-HOK Implementation Patterns](#7-e-hok-implementation-patterns)

---

## 1. Network Topology & Configuration

### 1.1 Configuration Hierarchy

The SquidASM network configuration follows a layered architecture from high-level YAML to low-level NetSquid components:

```
StackNetworkConfig (YAML) → NetworkConfig (netbuilder) → NetSquidNetwork → Nodes + Links
```

**Configuration Entry Point (`squidasm/run/stack/config.py`, lines 103-122):**

```python
class StackNetworkConfig(YamlLoadable):
    """Full network configuration."""

    stacks: List[StackConfig]
    """List of all the stacks in the network."""
    links: List[LinkConfig]
    """List of all the links connecting the stacks in the network."""
    clinks: Optional[List[CLinkConfig]] = None
    """List of all the links connecting the stacks in the network."""
```

### 1.2 Stack (Node) Configuration

Each node is configured via `StackConfig`:

```python
class StackConfig(YamlLoadable):
    """Configuration for a single stack (i.e. end node)."""

    name: str
    """Name of the stack."""
    qdevice_typ: str
    """Type of the quantum device."""
    qdevice_cfg: Any = None
    """Configuration of the quantum device, allowed configuration depends on type."""
```

**Supported QDevice Types:**
- `generic`: Standard quantum processor with configurable noise
- `nv`: NV-center hardware with specialized gate set

**QDevice Construction (`squidasm/sim/network/network.py`, lines 127-146):**

```python
def _build_network(self) -> None:
    for i, node_cfg in enumerate(self._network_config.nodes):
        # Memory fidelity from T1/T2 noise
        mem_fidelities = [T1T2NoiseModel(q.t1, q.t2) for q in node_cfg.qubits]

        if hardware == QuantumHardware.NV:
            qdevice = build_nv_qdevice(
                name=f"{node_cfg.name}_NVQDevice", cfg=self._nv_config
            )
        else:  # use generic hardware
            qdevice = QDevice(
                name=f"{node_cfg.name}_QDevice",
                num_qubits=len(node_cfg.qubits),
                gate_fidelity=node_cfg.gate_fidelity,
                mem_fidelities=mem_fidelities,
            )
        node = Node(name=node_cfg.name, ID=i, qmemory=qdevice)
        self.add_node(node)
```

### 1.3 Link Configuration Types

**Available Link Types (`squidasm/run/stack/config.py`, lines 44-56):**

| Type | Class | Description |
|------|-------|-------------|
| `perfect` | `PerfectQLinkConfig` | No noise, instant success |
| `depolarise` | `DepolariseQLinkConfig` | Depolarizing noise with configurable fidelity |
| `heralded` | `HeraldedDoubleClickQLinkConfig` | Physical heralded link with midpoint BSM |

**Depolarise Link Configuration (`netsquid_netbuilder/modules/qlinks/depolarise.py`, lines 18-33):**

```python
class DepolariseQLinkConfig(IQLinkConfig):
    """Depolarising model config."""

    fidelity: float
    """Fidelity of successfully generated EPR pairs."""
    prob_success: float
    """Probability of successfully generating an EPR pair per cycle."""
    t_cycle: Optional[float] = None
    """Duration of a cycle. [ns]"""
    length: Optional[float] = None
    """length of the link. Will be used to calculate t_cycle. [km]"""
    speed_of_light: float = 200000
    """Speed of light in the optical fiber. [km/s]"""
    random_bell_state: bool = False
    """Determines whether the Bell state is always phi+ or randomly chosen."""
```

### 1.4 Classical Link Configuration

**Classical links (`squidasm/run/stack/config.py`, lines 52-56):**

```python
class InstantCLinkConfig(netbuilder_clinks.InstantCLinkConfig):
    """Instant classical communication - zero delay."""

class DefaultCLinkConfig(netbuilder_clinks.DefaultCLinkConfig):
    """Classical link with configurable delay based on distance."""
```

### E-HOK Topology Requirements

For the baseline E-HOK (2 nodes):

```yaml
stacks:
  - name: alice
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 0  # No memory decay for baseline
      T2: 0

  - name: bob
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5
      T1: 0
      T2: 0

links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.95
      prob_success: 1.0
      t_cycle: 1000  # ns
```

For MDI Extension (3 nodes):

```yaml
stacks:
  - name: alice
  - name: bob
  - name: charlie  # Midpoint station

links:
  - stack1: alice
    stack2: charlie
    typ: heralded
    cfg:
      length: 10  # km
      
  - stack1: bob
    stack2: charlie
    typ: heralded
```

---

## 2. Comprehensive Noise Model Architecture

### 2.1 Noise Model Class Hierarchy

NetSquid provides a rich hierarchy of noise models in `netsquid/components/models/qerrormodels.py`:

```
QuantumErrorModel (base)
├── DepolarNoiseModel
├── DephaseNoiseModel
├── T1T2NoiseModel
└── FibreLossModel
```

### 2.2 DepolarNoiseModel

**Implementation (`netsquid/components/models/qerrormodels.py`, lines 170-240):**

```python
class DepolarNoiseModel(QuantumErrorModel):
    """Model for applying depolarizing noise to qubit(s)."""

    def __init__(self, depolar_rate, time_independent=False, **kwargs):
        super().__init__(**kwargs)
        self.add_property('time_independent', time_independent, value_type=bool)
        self.add_property('depolar_rate', depolar_rate, ...)

    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:
            for qubit in qubits:
                if qubit is not None:
                    qapi.depolarize(qubit, prob=self.depolar_rate)
        else:
            for qubit in qubits:
                if qubit is not None:
                    qapi.delay_depolarize(qubit, depolar_rate=self.depolar_rate, 
                                          delay=delta_time)
```

**Key Properties:**
- `depolar_rate`: Probability (if `time_independent=True`) or rate in Hz
- `time_independent`: Whether noise is a fixed probability or time-dependent

**QBER Relationship:**
For depolarizing noise: `QBER ≈ (3/4) × (1 - fidelity)`

### 2.3 DephaseNoiseModel

**Implementation (`netsquid/components/models/qerrormodels.py`, lines 243-302):**

```python
class DephaseNoiseModel(QuantumErrorModel):
    """Model for applying dephasing noise to qubit(s)."""

    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:
            for qubit in qubits:
                if qubit is not None:
                    qapi.dephase(qubit, prob=self.dephase_rate)
        else:
            for qubit in qubits:
                if qubit is not None:
                    qapi.delay_dephase(qubit, dephase_rate=self.dephase_rate, 
                                       delay=delta_time)
```

**Z-Basis Immunity:** Pure dephasing only affects X/Y basis measurements, not Z-basis.

### 2.4 T1T2NoiseModel (Critical for NSM)

**Implementation (`netsquid/components/models/qerrormodels.py`, lines 305-481):**

```python
class T1T2NoiseModel(QuantumErrorModel):
    """Commonly used phenomenological noise model based on T1 and T2 times.

    Parameters
    ----------
    T1 : float
        T1 time, dictating amplitude damping component.
    T2: float
        T2 time, dictating dephasing component. Note that this is T2 Hahn,
        as opposed to free induction decay T2*
    """

    def apply_noise(self, qubit, t):
        """Applies noise to the qubit, depending on T1/T2 and elapsed time.

        This follows a standard noise model used in experiments:
        1. Apply amplitude damping (depending on T1)
        2. Apply dephasing noise (depending on both T1 and T2)
        """
        if self.T1 == 0 and self.T2 == 0:
            return  # No noise
            
        # Amplitude damping
        if self.T1 > 0:
            probAD = 1 - np.exp(-t / self.T1)
            self._random_amplitude_dampen(qubit, probAD)
            
        # Dephasing (combined T1/T2 effect)
        if self.T2 > 0:
            if self.T1 == 0:
                dp = np.exp(-t * (1 / self.T2))
            else:
                dp = np.exp(-t * (1 / self.T2 - 1 / (2 * self.T1)))
            probZ = (1 - dp) / 2
            self._random_dephasing_noise(qubit, probZ)
```

**Critical for NSM (Noisy Storage Model):**

The T1T2 model is essential for implementing the Noisy Storage Model from `e-hok-extensions.md`. The security parameter depends on:

```
Security Condition: γ × Δt > H_max(K|E)

Where:
- γ = decoherence rate ≈ 1/T2 (for dephasing-dominated)
- Δt = enforced wait time before basis reveal
- H_max(K|E) = max-entropy of key given Eve's information
```

**Memory Application (`squidasm/sim/network/network.py`, lines 141-142):**

```python
mem_fidelities = [T1T2NoiseModel(q.t1, q.t2) for q in node_cfg.qubits]
# Applied to QuantumProcessor via memory_noise_models parameter
```

### 2.5 FibreLossModel

**Implementation (`netsquid/components/models/qerrormodels.py`, lines 484-556):**

```python
class FibreLossModel(QuantumErrorModel):
    """Model for exponential photon loss on fibre optic channels.

    Parameters
    ----------
    p_loss_init : float
        Initial probability of losing a photon when entering channel.
    p_loss_length : float
        Photon survival probability per channel length [dB/km].
    """

    def error_operation(self, qubits, delta_time=0, **kwargs):
        for idx, qubit in enumerate(qubits):
            if qubit is None:
                continue
            prob_loss = 1 - (1 - self.p_loss_init) * \
                        np.power(10, -kwargs['length'] * self.p_loss_length / 10)
            self.lose_qubit(qubits, idx, prob_loss, rng=self.properties['rng'])
```

### 2.6 Link-Level Noise Distribution

**Magic Distributor Types (`squidasm/sim/network/network.py`, lines 203-250):**

```python
def _create_link_distributor(self, link: Link, ...) -> MagicDistributor:
    noise_type = NoiseType(link.noise_type)
    
    if noise_type == NoiseType.NoNoise:
        return PerfectStateMagicDistributor(nodes=[node1, node2], ...)
        
    elif noise_type == NoiseType.Depolarise:
        noise = 1 - link.fidelity
        model_params = LinearDepolariseModelParameters(
            cycle_time=state_delay, prob_success=1, prob_max_mixed=noise
        )
        return LinearDepolariseMagicDistributor(nodes=[node1, node2], ...)
        
    elif noise_type == NoiseType.DiscreteDepolarise:
        # Uses netsquid_magic's DepolariseMagicDistributor
        ...
        
    elif noise_type == NoiseType.Bitflip:
        flip_prob = 1 - link.fidelity
        return BitflipMagicDistributor(...)
```

**State Generation (`squidasm/sim/network/network.py`, lines 575-620):**

```python
class LinearDepolariseStateSamplerFactory(HeraldedStateDeliverySamplerFactory):
    @staticmethod
    def _delivery_func(model_params: LinearDepolariseModelParameters, **kwargs):
        epr_state = np.array([
            [0.5, 0, 0, 0.5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0.5, 0, 0, 0.5]
        ], dtype=complex)  # |Φ+⟩ Bell state
        
        maximally_mixed = np.eye(4) / 4
        
        # Linear interpolation: (1-p)|Φ+⟩⟨Φ+| + p·I/4
        noisy_state = (1 - model_params.prob_max_mixed) * epr_state + \
                      model_params.prob_max_mixed * maximally_mixed
        
        return StateSampler(qreprs=[noisy_state], probabilities=[1]), 1
```

---

## 3. Batching & Sequential Mode Operations

### 3.1 The Memory Constraint Problem

From `e-hok-baseline.md`:
> **Must-Have:** A **Batching Manager** to handle streaming operations. The system must process $N=10,000$ bits using a 5-qubit memory.

**Memory Limitation Evidence (`squidasm/sim/stack/qnos.py`, line 24):**

```python
# TODO: make this a parameter
NUM_QUBITS = 5
```

### 3.2 Sequential Mode API

**EPRSocket Sequential Mode (`netqasm/sdk/epr_socket.py`, lines 146-231):**

```python
def create_keep(
    self,
    number: int = 1,
    post_routine: Optional[Callable] = None,
    sequential: bool = False,
    ...
) -> List[Qubit]:
    """Ask the network stack to generate EPR pairs with the remote node.

    If `sequential` is True, a callback function (`post_routine`) should be
    specified. After generating one EPR pair, this callback will be called,
    before generating the next pair. This method can e.g. be used to generate
    many EPR pairs (more than the number of physical qubits available), by
    measuring (and freeing up) each qubit before the next pair is generated.
    """
```

### 3.3 Streaming Mode Implementation Pattern

**Example (`squidasm/examples/advanced/link_layer/example_link_layer_ck.py`, lines 38-53):**

```python
class ClientProgram(Program):
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=1,  # Only 1 qubit needed at a time!
        )

    def run(self, context: ProgramContext) -> Generator[...]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]

        outcomes = conn.new_array(self._num_pairs)  # Classical storage

        def post_create(conn, q, pair):
            array_entry = outcomes.get_future_index(pair)
            if self._basis == "X":
                q.H()
            elif self._basis == "Y":
                q.K()
            q.measure(array_entry)  # Measure and free qubit

        # Generate N pairs using only 1 qubit memory slot
        epr_socket.create_keep(
            number=self._num_pairs,
            sequential=True,
            post_routine=post_create,
        )

        yield from conn.flush()
        return outcomes
```

### 3.4 E-HOK Batching Pattern

For E-HOK with 10,000 bits and 5-qubit memory:

```python
BATCH_SIZE = 5  # Limited by quantum memory
TOTAL_PAIRS = 10000

class AliceEHOK(Program):
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            max_qubits=BATCH_SIZE,  # Request full memory
            ...
        )

    def run(self, context: ProgramContext):
        conn = context.connection
        epr_socket = context.epr_sockets["bob"]

        # Pre-allocate classical storage
        all_outcomes = conn.new_array(TOTAL_PAIRS)
        all_bases = conn.new_array(TOTAL_PAIRS)

        def post_create(conn, q, pair):
            # Choose random basis
            basis = context.rng.randint(0, 2)  # 0=Z, 1=X
            all_bases.set_value(pair, basis)
            
            if basis == 1:  # X basis
                q.H()
            
            outcome = all_outcomes.get_future_index(pair)
            q.measure(outcome)

        # Stream all pairs through memory
        epr_socket.create_keep(
            number=TOTAL_PAIRS,
            sequential=True,
            post_routine=post_create,
        )

        yield from conn.flush()

        # After flush, all_outcomes and all_bases contain classical data
        return {"outcomes": all_outcomes, "bases": all_bases}
```

### 3.5 EPRType.M for Memory-Efficient Mode

**Advantage of EPRType.M:** The qubit is measured immediately by the network stack, so it **never occupies quantum memory**.

```python
# Memory-efficient alternative using EPRType.M
results = epr_socket.create_measure(
    number=TOTAL_PAIRS,
    random_basis_local=RandomBasis.XZ,   # Random X or Z
    random_basis_remote=RandomBasis.XZ,
)

# Results contain measurement outcomes and bases directly
for r in results:
    outcome = r.measurement_outcome  # 0 or 1
    basis = r.measurement_basis      # X, Y, Z, etc.
```

---

## 4. Distributed Simulation Architecture

### 4.1 Single-Thread vs Multi-Thread Modes

SquidASM provides two execution modes:

**Single-Thread Mode (`squidasm/run/stack/run.py`):**
- All nodes execute as protocols within a single NetSquid simulation
- Uses generator-based coroutines for concurrency
- Deterministic behavior, easier debugging

**Multi-Thread Mode (`squidasm/run/multithread/runtime_mgr.py`):**
- Each node runs in a separate Python thread
- Uses `ThreadPool` for parallel execution
- More realistic but non-deterministic

### 4.2 Single-Thread Execution Model

**Network Setup (`squidasm/run/stack/run.py`, lines 23-81):**

```python
def _setup_network(config: NetworkConfig) -> StackNetwork:
    NetSquidContext.reset()
    ns.sim_reset()
    builder = create_stack_network_builder()
    network = builder.build(config)

    stacks: Dict[str, NodeStack] = {}

    for node_name, node in network.end_nodes.items():
        stack = NodeStack(name=node_name, node=node, ...)
        stacks[node_name] = stack

    # Register peers for inter-node communication
    for s1, s2 in itertools.combinations(stacks.values(), 2):
        s1.qnos_comp.register_peer(s2.node.ID)
        s1.qnos.netstack.register_peer(s2.node.ID)
        ...

    # Setup classical sockets
    for s1 in stacks.values():
        socket_service = ConnectionlessSocketService(node=s1.node)
        for s2 in stacks.values():
            if s2 is s1:
                continue
            socket = socket_service.create_socket()
            socket.bind(port_name="0", remote_node_name=s2.node.name)
            socket.connect(...)

    return StackNetwork(stacks, link_prots, csockets)


def _run(network: StackNetwork) -> List[List[Dict[str, Any]]]:
    # Start node protocols
    for _, stack in network.stacks.items():
        stack.start()

    # Run NetSquid discrete event simulation
    ns.sim_run()

    return [stack.host.get_results() for _, stack in network.stacks.items()]
```

### 4.3 Multi-Thread Execution Model

**Thread Management (`squidasm/run/multithread/runtime_mgr.py`, lines 94-130):**

```python
def start_backend(self) -> None:
    def backend_thread(manager):
        ns.set_qstate_formalism(self.netsquid_formalism)

        for subroutine_handler in self._subroutine_handlers.values():
            subroutine_handler.start()

        self._is_running = True
        put_current_backend(self)
        ns.sim_run()  # NetSquid runs in its own thread
        pop_current_backend()
        self._is_running = False

    t = threading.Thread(target=backend_thread, args=(self,))
    self._backend_thread = t
    t.start()


def run_app(self, app_instance: ApplicationInstance, ...):
    with ThreadPool(len(programs) + 1) as executor:
        program_futures = []
        for program in programs:
            # Each program runs in a separate thread
            future = executor.apply_async(program.entry, kwds=inputs)
            program_futures.append(future)
```

### 4.4 Protocol Synchronization Points

**Host-to-QNodeOS Communication (`squidasm/sim/stack/connection.py`, lines 98-120):**

```python
def commit_subroutine(
    self,
    subroutine: Subroutine,
    block: bool = True,
    callback: Optional[Callable] = None,
) -> Generator[EventExpression, None, None]:
    """Commit a compiled subroutine to the quantum node controller."""
    
    # Send subroutine to QNodeOS
    self._commit_message(
        msg=SubroutineMessage(subroutine=subroutine),
        block=block,
        callback=callback,
    )

    # Block until QNodeOS returns shared memory with results
    result = yield from self._host.receive_qnos_msg()
    self._shared_memory = result
```

**Classical Socket Synchronization (`squidasm/nqasm/singlethread/csocket.py`, lines 34-47):**

```python
def recv(self) -> Generator[EventExpression, None, str]:
    """Receive a message from the remote node."""

    if len(self._protocol.peer_listener.buffer) == 0:
        # Block until message arrives
        yield EventExpression(
            source=self._protocol.peer_listener, 
            event_type=NewClasMsgEvent
        )
    return self._protocol.peer_listener.buffer.pop(0)
```

---

## 5. Timing Guarantees & Protocol Synchronization

### 5.1 Simulation Time Tracking

**Time Access (`netsquid.util`):**

```python
import netsquid as ns

current_time = ns.sim_time()  # Returns simulation time in nanoseconds
```

**Example Usage (`squidasm/sim/stack/netstack.py`, lines 300-345):**

```python
start_time = ns.sim_time()

# ... EPR generation ...

gen_duration_ns_float = ns.sim_time() - start_time
```

### 5.2 Event-Driven Execution

The NetSquid simulation is event-driven. Key events include:

1. **EPR Delivery Event (`squidasm/sim/network/network.py`, line 54):**
   ```python
   EprDeliveredEvent: EventType = EventType(
       "EPR_DELIVERED",
       "Event that an EPR has been delivered by a Distributor"
   )
   ```

2. **Classical Message Event:**
   ```python
   NewClasMsgEvent: EventType = EventType(
       "NewClasMsgEvent",
       "A new classical message from another peer has arrived"
   )
   ```

### 5.3 Ordering Guarantees for E-HOK Security

**Critical for Commitment Before Reveal:**

```python
class BobProgram(Program):
    def run(self, context):
        conn = context.connection
        csocket = context.csockets["alice"]
        epr_socket = context.epr_sockets["alice"]

        # Phase 1: Generate EPR pairs and measure
        outcomes = epr_socket.recv_keep(number=N, ...)
        yield from conn.flush()  # BLOCKS until measurements complete

        # Phase 2: Commit BEFORE receiving Alice's bases
        commitment = compute_commitment(outcomes)
        csocket.send(commitment)  # Non-blocking send

        # Phase 3: Receive Alice's bases
        # This CANNOT happen before Phase 2 due to message ordering
        alice_bases = yield from csocket.recv()  # BLOCKS until message arrives

        # Security: Alice's program BLOCKS on receiving our commitment
        # before sending her bases, so the ordering is enforced
```

**Alice's Corresponding Code:**

```python
class AliceProgram(Program):
    def run(self, context):
        conn = context.connection
        csocket = context.csockets["bob"]
        epr_socket = context.epr_sockets["bob"]

        # Phase 1: Generate and measure
        outcomes = epr_socket.create_keep(number=N, ...)
        yield from conn.flush()

        # Phase 2: WAIT for Bob's commitment (BLOCKS)
        bob_commitment = yield from csocket.recv()  # BLOCKING

        # Phase 3: Only AFTER receiving commitment, send bases
        csocket.send(bases)  # Send basis choices

        # Security Guarantee: 
        # - Bob's commitment was generated from his measurements
        # - Those measurements were fixed at flush() time
        # - Bob cannot change his commitment after seeing our bases
```

### 5.4 Implementing Wait Delays for NSM

For the Noisy Storage Model, we need enforced delays:

```python
import netsquid as ns
from netsquid.protocols import Signals
from pydynaa import EventExpression

class NSMAliceProgram(Program):
    def run(self, context):
        # ... EPR generation and measurement ...
        yield from conn.flush()

        # CRITICAL: Wait for delta_t to allow Bob's memory to decohere
        DELTA_T_NS = 10_000_000  # 10 ms in nanoseconds
        
        # Schedule a wake-up event
        self._schedule_after(DELTA_T_NS, SIGNAL_WAIT_DONE)
        yield self.await_signal(self, SIGNAL_WAIT_DONE)

        # NOW safe to reveal bases (Bob's memory has decohered)
        csocket.send(bases)
```

**Alternative using NetSquid utilities:**

```python
from netsquid.components import Clock

# Create a clock that triggers after delay
wait_clock = Clock(name="wait_clock", frequency=1e9/DELTA_T_NS)
wait_clock.start()

yield self.await_signal(wait_clock, signal_label=Clock.CLOCK_EVT_TYPE)
```

---

## 6. Heralded Links & MDI Architecture Support

### 6.1 Heralded Double-Click Model

The heralded link models a physical setup with a midpoint Bell-state measurement station.

**Configuration (`netsquid_netbuilder/modules/qlinks/heralded_double_click.py`, lines 22-110):**

```python
class HeraldedDoubleClickQLinkConfig(IQLinkConfig):
    """
    Heralded double-click quantum link model.

    The heralded link uses a model with both nodes connected by fiber to a 
    midpoint station with a Bell-state measurement detector.
    """

    length: Optional[float] = None
    """Total length [km] of fiber."""
    
    length_A: Optional[float] = None
    """Length [km] from node A to midpoint."""
    
    length_B: Optional[float] = None
    """Length [km] from node B to midpoint."""
    
    # Photon loss parameters
    p_loss_init: Optional[float] = None
    """Probability photon lost when entering connection."""
    
    p_loss_length: Optional[float] = None
    """Attenuation coefficient [dB/km] of the fiber."""
    
    # Detector parameters
    dark_count_probability: float = 0
    """Dark-count probability per detection."""
    
    detector_efficiency: float = 1
    """Probability photon leads to detection event."""
    
    visibility: float = 1
    """Hong-Ou-Mandel visibility (photon indistinguishability)."""
    
    num_resolving: bool = False
    """Whether photon-number-resolving detectors are used for BSM."""
    
    # Emission parameters (from qdevice)
    emission_fidelity: Optional[float] = None
    """Fidelity of emitted photon-memory entanglement."""
    
    emission_duration: Optional[float] = None
    """Time [ns] for memory to emit entangled photon."""
    
    collection_efficiency: Optional[float] = None
    """Probability of collecting emitted photon."""
```

### 6.2 MDI Architecture Considerations

For MDI-E-HOK from `e-hok-extensions.md`:

**Three-Node Star Topology:**

```yaml
stacks:
  - name: alice
    qdevice_typ: nv
    qdevice_cfg:
      emission_fidelity: 0.99
      emission_duration: 1000  # ns

  - name: bob
    qdevice_typ: nv
    qdevice_cfg:
      emission_fidelity: 0.99
      emission_duration: 1000

  - name: charlie  # Untrusted midpoint
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 2  # Only needs BSM capability

links:
  - stack1: alice
    stack2: charlie
    typ: heralded
    cfg:
      length_A: 5  # Alice to midpoint: 5 km
      length_B: 0  # Charlie is at midpoint
      dark_count_probability: 1e-7
      detector_efficiency: 0.9
      visibility: 0.95

  - stack1: bob
    stack2: charlie
    typ: heralded
    cfg:
      length_A: 0
      length_B: 5  # Bob to midpoint: 5 km
      dark_count_probability: 1e-7
      detector_efficiency: 0.9
      visibility: 0.95
```

### 6.3 MDI Protocol Flow

**Current Limitation:** SquidASM's current EPRSocket API is designed for 2-party entanglement. For MDI with 3 parties, you need to:

1. Alice and Charlie share EPR pair (Alice-Charlie link)
2. Bob and Charlie share EPR pair (Bob-Charlie link)
3. Charlie performs BSM on his two halves
4. Charlie broadcasts BSM outcome to Alice and Bob
5. Alice and Bob now share entanglement (swapped through Charlie)

**Implementation Pattern:**

```python
class CharlieProgram(Program):
    """Untrusted midpoint station."""
    
    def run(self, context):
        conn = context.connection
        alice_socket = context.epr_sockets["alice"]
        bob_socket = context.epr_sockets["bob"]
        csocket_alice = context.csockets["alice"]
        csocket_bob = context.csockets["bob"]

        for round in range(NUM_ROUNDS):
            # Receive halves of EPR pairs from both parties
            q_alice = alice_socket.recv_keep()[0]
            q_bob = bob_socket.recv_keep()[0]
            yield from conn.flush()

            # Perform Bell State Measurement
            q_alice.cnot(q_bob)
            q_alice.H()
            
            m_alice = q_alice.measure()
            m_bob = q_bob.measure()
            yield from conn.flush()

            # Broadcast BSM outcomes (Charlie learns nothing useful)
            bsm_result = (m_alice.value, m_bob.value)
            csocket_alice.send(str(bsm_result))
            csocket_bob.send(str(bsm_result))
```

---

## 7. E-HOK Implementation Patterns

### 7.1 Complete Baseline E-HOK Structure

Based on requirements from `e-hok-baseline.md`:

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# --- Data Structures ---

@dataclass
class ObliviousKey:
    """E-HOK output structure."""
    key_value: np.ndarray      # Final bitstring
    knowledge_mask: np.ndarray # 0=Known, 1=Unknown
    security_param: float      # Estimated epsilon


# --- Abstract Interfaces ---

class ICommitmentScheme:
    """Interface for commitment schemes."""
    def commit(self, data: bytes) -> bytes:
        raise NotImplementedError
    
    def open(self, commitment: bytes, data: bytes, proof: bytes) -> bool:
        raise NotImplementedError


class IReconciliator:
    """Interface for error correction."""
    def encode(self, key: np.ndarray) -> np.ndarray:
        """Generate syndrome."""
        raise NotImplementedError
    
    def decode(self, key: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        """Correct errors using syndrome."""
        raise NotImplementedError


class IPrivacyAmplifier:
    """Interface for privacy amplification."""
    def compress(self, key: np.ndarray, seed: bytes) -> np.ndarray:
        raise NotImplementedError


# --- Baseline Implementations ---

class SHA256Commitment(ICommitmentScheme):
    """Baseline: SHA-256 hash commitment."""
    
    def commit(self, data: bytes) -> bytes:
        import hashlib
        return hashlib.sha256(data).digest()
    
    def open(self, commitment: bytes, data: bytes, proof: bytes) -> bool:
        return self.commit(data) == commitment


class LDPCReconciliator(IReconciliator):
    """Baseline: Standard LDPC using scipy.sparse."""
    
    def __init__(self, parity_check_matrix: np.ndarray):
        from scipy.sparse import csr_matrix
        self.H = csr_matrix(parity_check_matrix)
    
    def encode(self, key: np.ndarray) -> np.ndarray:
        return (self.H @ key) % 2
    
    def decode(self, key: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        # Belief propagation decoder
        # ... implementation ...
        pass


class ToeplitzAmplifier(IPrivacyAmplifier):
    """Baseline: Toeplitz matrix multiplication."""
    
    def compress(self, key: np.ndarray, seed: bytes) -> np.ndarray:
        import numpy as np
        rng = np.random.default_rng(int.from_bytes(seed[:8], 'big'))
        
        output_len = len(key) // 2  # 2:1 compression
        first_col = rng.integers(0, 2, size=output_len)
        first_row = rng.integers(0, 2, size=len(key))
        
        # Construct Toeplitz matrix
        from scipy.linalg import toeplitz
        T = toeplitz(first_col, first_row)
        
        return (T @ key) % 2
```

### 7.2 E-HOK Protocol Manager

```python
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from pydynaa import EventExpression
from typing import Generator, Dict, Any

class AliceEHOKProgram(Program):
    """Alice's E-HOK protocol implementation."""

    def __init__(
        self,
        num_pairs: int = 10000,
        commitment: ICommitmentScheme = None,
        reconciliator: IReconciliator = None,
        amplifier: IPrivacyAmplifier = None,
    ):
        self.num_pairs = num_pairs
        self.commitment = commitment or SHA256Commitment()
        self.reconciliator = reconciliator
        self.amplifier = amplifier

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_ehok",
            csockets=["bob"],
            epr_sockets=["bob"],
            max_qubits=1,  # Streaming mode
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        csocket = context.csockets["bob"]
        epr_socket = context.epr_sockets["bob"]

        # --- Phase I: Quantum Generation ---
        outcomes = conn.new_array(self.num_pairs)
        bases = conn.new_array(self.num_pairs)

        def post_create(conn, q, pair):
            # Random basis choice
            import random
            basis = random.randint(0, 1)
            bases.set_value(pair, basis)
            
            if basis == 1:  # X basis
                q.H()
            
            outcomes.get_future_index(pair)
            q.measure(outcomes.get_future_index(pair))

        epr_socket.create_keep(
            number=self.num_pairs,
            sequential=True,
            post_routine=post_create,
        )
        yield from conn.flush()

        # Convert to numpy after flush
        s = np.array([outcomes.get_value(i) for i in range(self.num_pairs)])
        a = np.array([bases.get_value(i) for i in range(self.num_pairs)])

        # --- Phase II: Receive Bob's Commitment ---
        bob_commitment = yield from csocket.recv()
        
        # --- Phase III: Exchange Bases ---
        csocket.send(a.tobytes())
        bob_bases_bytes = yield from csocket.recv()
        b = np.frombuffer(bob_bases_bytes, dtype=np.int8)

        # Compute matching indices
        I_0 = np.where(a == b)[0]  # Matching bases
        I_1 = np.where(a != b)[0]  # Mismatched bases

        # --- Phase IV: Sampling & Verification ---
        test_size = len(I_0) // 10
        test_indices = np.random.choice(I_0, size=test_size, replace=False)
        
        csocket.send(test_indices.tobytes())
        bob_test_values = yield from csocket.recv()
        bob_test = np.frombuffer(bob_test_values, dtype=np.int8)

        # Calculate QBER
        qber = np.mean(s[test_indices] != bob_test)
        if qber > 0.11:
            return {"status": "ABORT", "qber": qber}

        # --- Phase V: Reconciliation ---
        key_indices = np.setdiff1d(I_0, test_indices)
        syndrome = self.reconciliator.encode(s[key_indices])
        csocket.send(syndrome.tobytes())

        # --- Phase VI: Privacy Amplification ---
        seed = np.random.bytes(32)
        csocket.send(seed)
        
        final_key = self.amplifier.compress(s[key_indices], seed)

        return {
            "status": "SUCCESS",
            "key": ObliviousKey(
                key_value=final_key,
                knowledge_mask=np.zeros(len(final_key)),  # Alice knows all
                security_param=2**(-128),
            ),
            "qber": qber,
        }
```

### 7.3 Testing Patterns

**Honest Execution Test:**

```python
def test_honest_execution():
    """Verify key agreement under ideal conditions."""
    cfg = StackNetworkConfig.from_file("perfect_config.yaml")
    
    alice_program = AliceEHOKProgram(num_pairs=1000)
    bob_program = BobEHOKProgram(num_pairs=1000)
    
    results = run(cfg, {"alice": alice_program, "bob": bob_program})
    
    alice_key = results[0]["key"]
    bob_key = results[1]["key"]
    
    # Keys should match on I_0
    assert np.array_equal(alice_key.key_value, bob_key.key_value)
    
    # Bob's knowledge mask should reflect I_1
    assert np.sum(bob_key.knowledge_mask) > 0
```

**Commitment Ordering Test:**

```python
def test_commitment_ordering():
    """Verify protocol aborts if bases sent before commitment."""
    
    class MaliciousAlice(AliceEHOKProgram):
        def run(self, context):
            # WRONG: Send bases before receiving commitment
            csocket.send(bases.tobytes())  # Security violation!
            bob_commitment = yield from csocket.recv()
            # ...
    
    # Protocol should detect and abort
    results = run(cfg, {"alice": MaliciousAlice(), "bob": bob_program})
    assert results[1]["status"] == "SECURITY_ABORT"
```

---

## Summary & Key Takeaways

### Network Configuration
- Use `StackNetworkConfig` for YAML-based network setup
- Choose `depolarise` links for baseline, `heralded` for physical simulations
- Configure T1/T2 on qdevices for NSM experiments

### Noise Models
- `DepolarNoiseModel`: Uniform noise, `QBER ≈ 0.75 × (1-fidelity)`
- `T1T2NoiseModel`: Critical for NSM, models physical decoherence
- Link-level noise configured via `MagicDistributor` types

### Batching & Streaming
- Use `sequential=True` with `post_routine` to exceed memory limits
- EPRType.M bypasses memory entirely for high-throughput
- Classical buffers accumulate results for batch processing

### Timing & Synchronization
- `yield from conn.flush()` creates synchronization points
- `yield from csocket.recv()` blocks until message arrives
- Use `ns.sim_time()` to track simulation time for NSM delays

### MDI Architecture
- Requires 3-node topology with midpoint station
- Manual entanglement swapping via BSM on Charlie
- Current API designed for 2-party; MDI needs custom protocols
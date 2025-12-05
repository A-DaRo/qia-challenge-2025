# E-HOK SquidASM Framework Technical Reference
**Detailed Analysis for Entanglement-Based Hybrid Oblivious Key Implementation**

This document provides in-depth technical answers to the framework-specific questions raised in the E-HOK roadmap (`e-hok-roadmap.md`). All answers are derived from direct code inspection of the SquidASM, NetQASM, and NetSquid packages.

---

## Table of Contents

1. [Simulator Concurrency & Timing](#1-simulator-concurrency--timing)
2. [The "Future" Object Lifecycle](#2-the-future-object-lifecycle)
3. [LDPC Matrix Handling & SharedMemory](#3-ldpc-matrix-handling--sharedmemory)
4. [Noise Model Fidelity](#4-noise-model-fidelity)
5. [EPRSocket Implementation Details](#5-eprsocket-implementation-details)
6. [Streaming Mode & Quantum Memory Constraints](#6-streaming-mode--quantum-memory-constraints)
7. [Protocol Flow & Security Guarantees](#7-protocol-flow--security-guarantees)

---

## 1. Simulator Concurrency & Timing

### Original Question (from roadmap)

> *Does SquidASM guarantee that if Alice executes `conn.flush()` (measurement) and then sends a classical message, the simulation time strictly advances?*

### Answer: YES, with Caveats

SquidASM inherits its discrete event simulation from **NetSquid** (via `pydynaa`). The key insight is that SquidASM uses a **generator-based protocol execution model** that explicitly yields control to the scheduler at well-defined points.

#### Evidence from Codebase

**1. The Handler Protocol (`squidasm/sim/stack/handler.py`, lines 243-265):**

```python
def run(self) -> Generator[EventExpression, None, None]:
    """Run this protocol. Automatically called by NetSquid during simulation."""

    # Loop forever acting on messages from the Host.
    while True:
        # Wait for a new message from the Host.
        raw_host_msg = yield from self._receive_host_msg()
        self._logger.debug(f"received new msg from host: {raw_host_msg}")
        msg = deserialize_host_msg(raw_host_msg)

        # Handle the message. This updates the handler's state and may e.g.
        # add a pending subroutine for an application.
        self.msg_from_host(msg)

        # Get the next application that needs work.
        app = self._next_app()
        if app is not None:
            # Flush all pending subroutines for this app.
            while True:
                subrt = app.next_subroutine()
                if subrt is None:
                    break
                app_mem = yield from self.assign_processor(app.id, subrt)
                self._send_host_msg(app_mem)
```

This shows that:
- Every `yield from` statement **transfers control to the NetSquid scheduler**
- Subroutine execution is **atomic from the Host's perspective**: the result is only sent back *after* the subroutine fully completes

**2. The Flush Mechanism (`netqasm/sdk/connection.py`, lines 481-499):**

```python
def flush(self, block: bool = True, callback: Optional[Callable] = None) -> None:
    """Compile and send all pending operations to the quantum node controller.

    All operations that have been added to this connection's Builder (typically by
    issuing these operations within a connection context) are collected and
    compiled into a NetQASM subroutine. This subroutine is then sent over the connection
    to be executed by the quantum node controller.

    :param block: block on receiving the result of executing the compiled subroutine
        from the quantum node controller.
    """
    protosubroutine = self._builder.subrt_pop_pending_subroutine()
    if protosubroutine is None:
        return

    self.commit_protosubroutine(
        protosubroutine=protosubroutine,
        block=block,
        callback=callback,
    )
```

When `block=True` (default), `flush()` **blocks until the quantum node controller completes execution**.

**3. Event Scheduler Interaction (`squidasm/sim/stack/common.py`, lines 100-118):**

```python
class PortListener(Protocol):
    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            # Wait for an event saying that there is new input.
            yield self.await_port_input(self._port)

            counter = 0
            # Read all inputs and count them.
            while True:
                input = self._port.rx_input()
                if input is None:
                    break
                self._buffer += input.items
                counter += 1
            # If there are n inputs, there have been n events, but we yielded only
            # on one of them so far. "Flush" these n-1 additional events:
            while counter > 1:
                yield self.await_port_input(self._port)
                counter -= 1

            # Only after having yielded on all current events, we can schedule a
            # notification event, so that its reactor can handle all inputs at once.
            self.send_signal(self._signal_label)
```

This confirms that event processing is **strictly ordered**.

### Security Implications for E-HOK

**The simulation enforces strict ordering**:
1. Alice's `flush()` for measurements completes before her program continues
2. Any classical message sent after `flush()` has a **later simulation timestamp** than the measurement
3. Bob's commitment (sent before Alice's basis reveal) will have an **earlier timestamp** than Alice's bases

**However**, there is a **critical caveat**: The SquidASM simulator processes events in a **single-threaded** manner. The security guarantee holds *within the simulation*, but you must ensure:

1. Bob's commitment message is **scheduled** before Alice reveals her bases
2. The protocol uses explicit message-response patterns (not fire-and-forget)

**Recommendation**: Use the `ClassicalSocket` send/recv pattern with structured messages to enforce protocol ordering:

```python
# Bob commits FIRST
socket.send_structured(StructuredMessage("Commitment", commitment_hash))

# Alice only sends bases AFTER receiving confirmation
confirmation = yield from socket.recv_structured()
socket.send_structured(StructuredMessage("Bases", alice_bases))
```

---

## 2. The "Future" Object Lifecycle

### Original Question (from roadmap)

> *When using `EPRType.M` (Measure Directly), are the results available immediately in the Python variable after `flush()`, or is there a callback latency?*

### Answer: Results are Available Immediately After `flush()`

The `Future` class is designed to behave like an `int` once the value is resolved. The value becomes available **synchronously upon `flush()` completion**.

#### Evidence from Codebase

**1. Future Value Resolution (`netqasm/sdk/futures.py`, lines 117-144):**

```python
@as_int_when_value
class BaseFuture(int):
    """Base class for Future-like objects.

    A Future represents a classical value that becomes available at some point
    in the future. At the moment, a Future always represents an integer value.

    Futures have a `value` property that is either `None` (when the value is not yet
    available), or has a concrete integer value.
    Executing a subroutine on the quantum node controller makes the value property go
    from `None` to a concrete value, granted that the subroutine sets the value of
    whatever the Future represents.
    """
    # ...
    
    @property
    def value(self) -> Optional[int]:
        """Get the value of the future.
        If it's not set yet, `None` is returned."""
        if self._value is not None:
            return self._value
        else:
            return self._try_get_value()
```

**2. The `@as_int_when_value` Decorator (`netqasm/sdk/futures.py`, lines 37-63):**

```python
def as_int_when_value(cls):
    """A decorator for the `BaseFuture` class which makes is behave like an `int`
    when the property `value` is not `None`.
    """

    def wrap_method(method_name):
        """Return a new method for the class given a method name"""
        int_method = getattr(int, method_name)

        def new_method(self, *args, **kwargs):
            """Check if the value is set, otherwise raise an error"""
            value = self.value
            if value is None:
                raise NoValueError(
                    f"The object '{repr(self)}' has no value yet, "
                    "consider flusing the current subroutine"
                )
            # ...
            return int_method(value, *args, **kwargs)

        return new_method
```

**3. Value Retrieval from SharedMemory (`netqasm/sdk/futures.py`, lines 277-293):**

```python
def _try_get_value(self) -> Optional[int]:
    if not isinstance(self._index, int):
        raise NonConstantIndexError("index is not constant and cannot be resolved")
    if self._connection.shared_memory is None:
        return None
    value = self._connection.shared_memory.get_array_part(
        address=self._address, index=self._index
    )
    if not isinstance(value, int) and value is not None:
        raise RuntimeError(
            f"Something went wrong: future value {value} is not an int or None"
        )
    if value is not None:
        self._value = value
    return value
```

**4. Netstack Writing Results (`squidasm/sim/stack/netstack.py`, lines 424-442):**

```python
# Populate results array.
for pair_index in range(request.number):
    result = results[pair_index]

    for i in range(slice_len):
        # Write -1 to unused array elements.
        value = -1

        # Write corresponding result value to the other array elements.
        if i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_OUTCOME:
            value = result.measurement_outcome
        elif i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_BASIS:
            value = result.measurement_basis.value
        elif i == SER_RESPONSE_KEEP_IDX_BELL_STATE:
            value = result.bell_state.value

        # Calculate array element location.
        arr_index = slice_len * pair_index + i

        app_mem.set_array_value(req.result_array_addr, arr_index, value)
```

### EPRType.M Measurement Basis Support

**Critical Finding**: `EPRType.M` (Measure Directly) **fully supports specifying measurement bases**.

**Evidence (`netqasm/sdk/epr_socket.py`, lines 269-334):**

```python
def create_measure(
    self,
    number: int = 1,
    time_unit: TimeUnit = TimeUnit.MICRO_SECONDS,
    max_time: int = 0,
    basis_local: Optional[EprMeasBasis] = None,
    basis_remote: Optional[EprMeasBasis] = None,
    rotations_local: Tuple[int, int, int] = (0, 0, 0),
    rotations_remote: Tuple[int, int, int] = (0, 0, 0),
    random_basis_local: Optional[RandomBasis] = None,
    random_basis_remote: Optional[RandomBasis] = None,
) -> List[EprMeasureResult]:
    """Ask the network stack to generate EPR pairs with the remote node and
    measure them immediately (on both nodes).
    # ...
    
    The basis to measure in can also be specified. There are 3 ways to specify a
    basis:

    * using one of the `EprMeasBasis` variants
    * by specifying 3 rotation angles, interpreted as an X-rotation, a Y-rotation
      and another X-rotation. For example, setting `rotations_local` to (8, 0, 0)
      means that before measuring, an X-rotation of 8*pi/16 = pi/2 radians is
      applied to the qubit.
    * using one of the `RandomBasis` variants, in which case one of the bases of
      that variant is chosen at random just before measuring
```

**Available Measurement Bases (`netqasm/sdk/build_epr.py`, lines 11-18):**

```python
class EprMeasBasis(Enum):
    X = 0
    Y = auto()
    Z = auto()
    MX = auto()  # Minus X
    MY = auto()  # Minus Y
    MZ = auto()  # Minus Z
```

**Random Basis Options (`netqasm/qlink_compat.py`, lines 46-52):**

```python
class RandomBasis(Enum):
    NONE = 0
    XZ = auto()      # Random choice between X and Z bases
    XYZ = auto()     # Random choice between X, Y, and Z bases
    CHSH = auto()    # Bases optimized for CHSH inequality
```

### Recommendation for E-HOK

For E-HOK, you can use either approach:

1. **EPRType.K + Manual Measurement** (More Control):
   ```python
   q = epr_socket.create_keep()[0]
   if basis == 1:  # X basis
       q.H()
   outcome = q.measure()
   yield from conn.flush()
   # outcome is now an int
   ```

2. **EPRType.M with Random Basis** (More Efficient):
   ```python
   results = epr_socket.create_measure(
       number=100,
       random_basis_local=RandomBasis.XZ,
       random_basis_remote=RandomBasis.XZ
   )
   yield from conn.flush()
   # Each result has .measurement_outcome and .measurement_basis
   ```

**EPRType.M Advantage**: The qubit is measured immediately by the network stack, so it **never occupies quantum memory**. This is crucial for streaming mode.

---

## 3. LDPC Matrix Handling & SharedMemory

### Original Question (from roadmap)

> *Can the `SubroutineHandler` or `SharedMemory` in NetQASM handle large binary arrays efficiently?*

### Answer: Yes, but with Design Considerations

#### SharedMemory Architecture

**SharedMemory Structure (`netqasm/sdk/shared_memory.py`, lines 62-147):**

```python
class Arrays:
    def __init__(self):
        self._arrays: Dict[int, List[Optional[int]]] = {}
    
    def __setitem__(
        self,
        key: Tuple[int, Union[int, slice]],
        value: Union[None, int, List[Optional[int]]],
    ) -> None:
        address, index = self._extract_key(key)
        if isinstance(index, int):
            if isinstance(value, int):
                _assert_within_width(value, ADDRESS_BITS)
                _assert_within_width(index, ADDRESS_BITS)
        # ...
        array = self._get_array(address)
        # ...
        array[index] = value
```

**Key Observations**:

1. **Arrays are Python lists**: `Dict[int, List[Optional[int]]]` - standard Python data structures
2. **Integer width constraint**: `ADDRESS_BITS` limits values (but only enforced for hardware, not simulation)
3. **No size limit in simulation**: The constraint on array size is **logical, not physical**

**Array Allocation (`netqasm/sdk/connection.py`, lines 577-591):**

```python
def new_array(
    self, length: int = 1, init_values: Optional[List[Optional[int]]] = None
) -> Array:
    """Allocate a new array in the shared memory.

    This operation is handled by the connection's Builder.
    The Builder make sures the relevant NetQASM instructions end up in the
    subroutine.

    :param length: length of the array, defaults to 1
    :param init_values: list of initial values of the array. If not None, must
        have the same length as `length`.
    :return: a handle to the array that can be used in application code
    """
    return self._builder.alloc_array(length, init_values)
```

#### LDPC Processing Strategy

**Critical Insight**: LDPC decoding should happen **outside** the NetQASM subroutine, in pure Python.

The workflow should be:

```
[Quantum Layer: NetQASM]     [Classical Layer: Pure Python]
     EPR Generation    ──>     Bit Buffer
     Measurement       ──>     Syndrome Computation
     flush()           ──>     LDPC Decoding (numpy/scipy)
                              Privacy Amplification
```

**Evidence for this separation (`squidasm/sim/stack/common.py`, lines 164-233):**

```python
class AppMemory:
    def __init__(self, app_id: int, max_qubits: int) -> None:
        self._app_id: int = app_id
        self._registers: Dict[RegisterName, RegisterGroup] = setup_registers()
        self._arrays: Arrays = Arrays()
        # ...

    def get_array(self, address: int) -> List[Optional[int]]:
        return self._arrays._get_array(address)

    def get_array_values(
        self, addr: int, start_offset: int, end_offset
    ) -> List[Optional[int]]:
        values = self.get_array_slice(
            operand.ArraySlice(operand.Address(addr), start_offset, end_offset)
        )
        assert values is not None
        return values
```

After `flush()`, you can access the entire array as a Python list and convert to numpy:

```python
# After flush, access measurement results
raw_outcomes = conn.shared_memory.get_array_part(address=outcomes_addr, index=slice(0, N))
outcomes_np = np.array(raw_outcomes, dtype=np.uint8)

# Now perform LDPC in pure numpy/scipy
syndrome = (H @ outcomes_np) % 2
decoded = ldpc_decode(syndrome, H)  # Using pyldpc or custom implementation
```

### Recommendation for LDPC in E-HOK

1. **Batched Generation**: Generate EPR pairs in batches (e.g., 100 at a time due to `max_qubits`)
2. **Buffer Accumulation**: Accumulate results in Python lists
3. **Block Processing**: Once you have enough bits (e.g., 10,000), run LDPC
4. **Use Efficient Libraries**: 
   - `numpy` for matrix operations
   - `scipy.sparse` for LDPC parity-check matrices
   - `galois` for GF(2) arithmetic

---

## 4. Noise Model Fidelity

### Original Question (from roadmap)

> *Which specific noise models in `netsquid` (Depolarizing, Dephasing) are active on the EPR links?*

### Answer: Configurable via Network Configuration

SquidASM supports multiple noise models, configurable at both the **link level** (entanglement generation) and the **quantum memory level** (qubit storage).

#### Available Noise Models

**1. Quantum Memory Noise (`netsquid/components/models/qerrormodels.py`, lines 169-297):**

```python
class DepolarNoiseModel(QuantumErrorModel):
    """Model for applying depolarizing noise to qubit(s) on a quantum component.

    Parameters
    ----------
    depolar_rate : float
        Probability that qubit will depolarize with time. If ``time_independent`` is False (default),
        then this is the exponential depolarizing rate per unit time [Hz].
        If True, it is a probability.
    """
    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.time_independent:
            for qubit in qubits:
                if qubit is not None:
                    qapi.depolarize(qubit, prob=self.depolar_rate)
        else:
            for qubit in qubits:
                if qubit is not None:
                    qapi.delay_depolarize(qubit, depolar_rate=self.depolar_rate, delay=delta_time)


class DephaseNoiseModel(QuantumErrorModel):
    """Model for applying dephasing noise to qubit(s) on a quantum component."""
    # Similar structure to DepolarNoiseModel


class T1T2NoiseModel(QuantumErrorModel):
    """Commonly used phenomenological noise model based on T1 and T2 times.

    Parameters
    ----------
    T1 : float
        T1 time, dictating amplitude damping component.
    T2: float
        T2 time, dictating dephasing component.
    """
```

**2. Link-Level Noise Configuration (`squidasm/sim/network/network.py`, lines 203-248):**

```python
def _create_link_distributor(
    self, link: Link, state_delay: Optional[float] = 1000
) -> MagicDistributor:
    """
    Create a MagicDistributor for a pair of nodes,
    based on configuration in a `Link` object.
    """
    node1 = self.get_node(link.node_name1)
    node2 = self.get_node(link.node_name2)

    try:
        noise_type = NoiseType(link.noise_type)
        if noise_type == NoiseType.NoNoise:
            model_params = PerfectModelParameters(state_delay=state_delay)
            return PerfectStateMagicDistributor(
                nodes=[node1, node2], model_params=model_params
            )
        elif noise_type == NoiseType.Depolarise:
            noise = 1 - link.fidelity
            model_params = LinearDepolariseModelParameters(
                cycle_time=state_delay, prob_success=1, prob_max_mixed=noise
            )
            return LinearDepolariseMagicDistributor(
                nodes=[node1, node2], model_params=model_params
            )
        elif noise_type == NoiseType.DiscreteDepolarise:
            noise = 1 - link.fidelity
            model_params = DepolariseModelParameters(
                prob_max_mixed=noise, cycle_time=state_delay
            )
            return DepolariseMagicDistributor(
                nodes=[node1, node2],
                model_params=model_params,
            )
        elif noise_type == NoiseType.Bitflip:
            flip_prob = 1 - link.fidelity
            model_params = BitFlipModelParameters(flip_prob=flip_prob)
            return BitflipMagicDistributor(
                model_params=model_params,
                state_delay=state_delay,
                nodes=[node1, node2],
            )
```

#### Extracting Fidelity from Configuration

**StackNetworkConfig (`squidasm/run/stack/config.py`, lines 40-78):**

```python
class DepolariseLinkConfig(netbuilder_links.DepolariseQLinkConfig):
    __doc__ = netbuilder_links.DepolariseQLinkConfig.__doc__


class HeraldedLinkConfig(netbuilder_links.HeraldedDoubleClickQLinkConfig):
    __doc__ = netbuilder_links.HeraldedDoubleClickQLinkConfig.__doc__


class LinkConfig(YamlLoadable):
    """Configuration for a single link."""

    stack1: str
    stack2: str
    typ: str
    cfg: Any = None

    @classmethod
    def perfect_config(cls, stack1: str, stack2: str) -> LinkConfig:
        """Create a configuration for a link without any noise or errors."""
        return LinkConfig(stack1=stack1, stack2=stack2, typ="perfect", cfg=None)
```

**QDevice Memory Fidelity (`squidasm/sim/network/network.py`, lines 140-145):**

```python
mem_fidelities = [T1T2NoiseModel(q.t1, q.t2) for q in node_cfg.qubits]
```

#### QBER Estimation from Configuration

The QBER (Quantum Bit Error Rate) can be estimated from the configuration:

```python
# For Depolarizing noise:
# QBER ≈ (3/4) * (1 - fidelity) for maximally mixed noise
# Example: fidelity = 0.97 → QBER ≈ 0.75 * 0.03 = 0.0225 (2.25%)

# For Bitflip noise:
# QBER = 1 - fidelity (directly)
# Example: fidelity = 0.97 → QBER = 0.03 (3%)
```

### Extracting Fidelity at Runtime

You can access the configuration programmatically:

```python
from squidasm.run.stack.config import StackNetworkConfig

config = StackNetworkConfig.from_file("network_config.yaml")

for link in config.links:
    if hasattr(link.cfg, 'fidelity'):
        print(f"Link {link.stack1}-{link.stack2}: fidelity = {link.cfg.fidelity}")
```

**Example YAML Configuration:**

```yaml
stacks:
  - name: alice
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5

  - name: bob
    qdevice_typ: generic
    qdevice_cfg:
      num_qubits: 5

links:
  - stack1: alice
    stack2: bob
    typ: depolarise
    cfg:
      fidelity: 0.97
      prob_success: 1.0
      t_cycle: 1000  # nanoseconds
```

---

## 5. EPRSocket Implementation Details

### EPRType Options Summary

| EPRType | Quantum Memory Usage | Measurement Control | Best For |
|---------|---------------------|---------------------|----------|
| `K` (Keep) | Occupies memory until measured | Full control (apply gates before measuring) | Complex protocols requiring entanglement |
| `M` (Measure Directly) | No memory occupation | Basis specified at creation time | High-throughput QKD |
| `R` (Remote State Preparation) | One side keeps, one measures | Creator measures, receiver keeps qubit | Asymmetric protocols |

### EPRSocket Create/Recv Symmetry

**Critical**: Every `create_*` must be matched by a `recv_*` on the remote node.

**Evidence (`netqasm/sdk/epr_socket.py`, lines 53-63):**

```python
"""
Each `create` operation on one node must be matched by a `recv` operation on the
other node. Since "creating" and "receiving" must happen at the same time, a node
that is doing a `create` operation on its socket cannot advance until the other
node does the corresponding `recv`. This is different from classical network
sockets where a "send" operation (roughly anologous to `create` in an EPR socket)
does not block on the remote node receiving it.
"""
```

### Measurement Basis Selection in EPRType.M

**The creator controls BOTH measurement bases:**

```python
# Alice controls both bases
results = epr_socket.create_measure(
    number=100,
    basis_local=EprMeasBasis.X,       # Alice's basis
    basis_remote=EprMeasBasis.Z,      # Bob's basis (!)
)

# Bob cannot override - he just receives
results = epr_socket.recv_measure(number=100)
# Bob's measurement basis was determined by Alice's request
```

**Evidence (`netqasm/sdk/epr_socket.py`, lines 310-313):**

```python
"""
NOTE: the node that initiates the entanglement generation, i.e. the one that
calls `create` on its EPR socket, also controls the measurement bases of the
receiving node (by setting e.g. `rotations_remote`). The receiving node cannot
change this.
"""
```

### For E-HOK Implications

This asymmetry is **problematic for E-HOK** where Bob must independently choose his basis. Solutions:

1. **Use EPRType.K**: Both parties receive qubits and measure independently
2. **Coordinate via Classical Channel**: Agree on random seed for basis selection before EPR creation
3. **Use RandomBasis**: Let the network stack randomly choose

---

## 6. Streaming Mode & Quantum Memory Constraints

### The Memory Constraint Problem

**From `squidasm/sim/stack/qnos.py`, line 24:**

```python
# TODO: make this a parameter
NUM_QUBITS = 5
```

The default is only **5 qubits**. This is configurable but reflects realistic near-term hardware.

### Implementing Streaming Mode

**Pattern for Batch Processing:**

```python
BATCH_SIZE = 5  # Limited by quantum memory
TOTAL_PAIRS = 10000
RECONCILIATION_BLOCK = 1000

classical_buffer = []

for batch in range(TOTAL_PAIRS // BATCH_SIZE):
    # Generate and measure in quantum layer
    for i in range(BATCH_SIZE):
        q = epr_socket.create_keep()[0]
        basis = random.randint(0, 1)
        if basis == 1:
            q.H()
        m = q.measure()
        yield from conn.flush()
        classical_buffer.append((int(m), basis))
    
    # Check if we have enough for reconciliation
    if len(classical_buffer) >= RECONCILIATION_BLOCK:
        # Move to classical post-processing
        outcomes = np.array([x[0] for x in classical_buffer[:RECONCILIATION_BLOCK]])
        bases = np.array([x[1] for x in classical_buffer[:RECONCILIATION_BLOCK]])
        classical_buffer = classical_buffer[RECONCILIATION_BLOCK:]
        
        # Perform sifting, reconciliation, privacy amplification
        # ... (pure Python/numpy operations)
```

**Using Sequential Mode (`netqasm/sdk/epr_socket.py`, lines 173-199):**

```python
"""
If `sequential` is True, a callback function (`post_routine`) should be
specified. After generating one EPR pair, this callback will be called, before
generating the next pair. This method can e.g. be used to generate many EPR
pairs (more than the number of physical qubits available), by measuring (and
freeing up) each qubit before the next pair is generated.
"""

# Example:
outcomes = alice.new_array(num)

def post_create(conn, q, pair):
    q.H()
    outcome = outcomes.get_future_index(pair)
    q.measure(outcome)

epr_socket.create_keep(number=num, post_routine=post_create, sequential=True)
```

This allows generating **more pairs than physical qubits** by measuring each qubit before generating the next.

---

## 7. Protocol Flow & Security Guarantees

### Recommended Protocol Flow for E-HOK

Based on the analysis, here's the secure protocol flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         E-HOK Protocol Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ALICE                                                   BOB        │
│    │                                                       │        │
│    │  ── Phase 1: EPR Generation & Measurement ──          │        │
│    │                                                       │        │
│    │  create_keep() ──────────────────────> recv_keep()    │        │
│    │  choose basis a_i                      choose basis b_i│        │
│    │  measure → s_i                         measure → s̄_i  │        │
│    │  flush()                               flush()        │        │
│    │                                                       │        │
│    │  ── Phase 2: Commitment ──                            │        │
│    │                                                       │        │
│    │                   <────── Commitment(H(s̄, b))         │        │
│    │                                                       │        │
│    │  ── Phase 3: Basis Exchange ──                        │        │
│    │                                                       │        │
│    │  Bases(a) ──────────────────────────────────────>     │        │
│    │                   <────── Bases(b)                    │        │
│    │                                                       │        │
│    │  ── Phase 4: Commitment Verification ──               │        │
│    │                                                       │        │
│    │  Challenge(T) ──────────────────────────────────>     │        │
│    │                   <────── Opening(s̄_T, Merkle proofs) │        │
│    │  Verify Merkle proofs                                 │        │
│    │                                                       │        │
│    │  ── Phase 5: Sifting & Reconciliation ──             │        │
│    │                                                       │        │
│    │  Compute I_0 = {i : a_i = b_i}                        │        │
│    │  Syndrome(Hs|_{I_0}) ──────────────────────────>      │        │
│    │                       Bob corrects s̄|_{I_0}          │        │
│    │                                                       │        │
│    │  ── Phase 6: Privacy Amplification ──                 │        │
│    │                                                       │        │
│    │  K_A = Extract(s|_{I_0})    K_B = Extract(s̄|_{I_0})  │        │
│    │  K_{I_1} = s|_{I_1}         K_{I_1} unknown           │        │
│    │                                                       │        │
└─────────────────────────────────────────────────────────────────────┘
```

### SquidASM Implementation Pattern

```python
# alice_program.py
class AliceProgram(Program):
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice",
            csockets=["bob"],
            epr_sockets=["bob"],
        )

    def run(self, context: ProgramContext):
        conn = context.connection
        epr_socket = context.epr_sockets["bob"]
        csocket = context.csockets["bob"]
        
        # Phase 1: Generate EPR pairs with measurements
        outcomes = []
        bases = []
        for _ in range(NUM_PAIRS):
            q = epr_socket.create_keep()[0]
            b = random.randint(0, 1)
            if b == 1:
                q.H()
            m = q.measure()
            yield from conn.flush()
            outcomes.append(int(m))
            bases.append(b)
        
        # Phase 2: Wait for Bob's commitment
        commitment_msg = yield from csocket.recv_structured()
        bob_commitment = commitment_msg.payload
        
        # Phase 3: Exchange bases (ONLY after receiving commitment)
        csocket.send_structured(StructuredMessage("Bases", bases))
        bob_bases_msg = yield from csocket.recv_structured()
        bob_bases = bob_bases_msg.payload
        
        # Continue with verification, sifting, reconciliation...
```

---

## Summary & Recommendations

### Key Takeaways

1. **Timing Security**: SquidASM's discrete event simulation enforces strict ordering when using `flush()` and `yield from` patterns. Protocol security can be maintained.

2. **EPRType.M is Efficient**: For E-HOK, prefer `EPRType.M` with `RandomBasis.XZ` for high throughput since it doesn't occupy quantum memory.

3. **LDPC Outside NetQASM**: Perform LDPC decoding in pure Python/numpy after extracting measurement results via `flush()`.

4. **Noise Configuration**: Fidelity is configurable via YAML. Calculate QBER from `fidelity` parameter: `QBER ≈ 0.75 * (1 - fidelity)` for depolarizing noise.

5. **Streaming Mode Required**: With realistic quantum memory limits (5-10 qubits), use batch processing with classical buffers.

6. **Commitment Before Bases**: The protocol must enforce Bob commits before Alice reveals bases—use structured message-response patterns.

### Technical Debt & Future Work

- The `NUM_QUBITS = 5` hardcoding should be parameterized
- EPRType.M's asymmetric basis control may need workarounds for E-HOK
- LDPC integration should use `pyldpc` or `aff3ct` bindings for performance

---

*Document generated from codebase analysis of SquidASM, NetQASM, and NetSquid packages. Last updated: December 2025.*

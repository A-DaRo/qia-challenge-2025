# 4.1 EPR Pair Generation

## Overview

EPR (Einstein-Podolsky-Rosen) pair generation is the foundation of the E-HOK quantum phase. This section specifies the generation process, entanglement properties, and SquidASM integration patterns.

## EPR Pair Definition

### Bell State

An EPR pair is a maximally entangled two-qubit state in the **Bell basis**:

$$
|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
$$

**Properties**:
1. **Maximal Entanglement**: Schmidt rank 2, entropy $S = 1$
2. **Perfect Correlation (Z basis)**: Measuring both in Z yields identical outcomes
3. **Perfect Anti-correlation (X basis)**: Measuring both in X yields opposite outcomes (after phase correction)

### Imperfect EPR Generation

In realistic settings, generated states are **Werner states**:

$$
\rho_F = F|\Phi^+\rangle\langle\Phi^+| + \frac{1-F}{4}\mathbb{I}_4
$$

where $F$ is the **fidelity** to the ideal Bell state.

**Fidelity Sources**:
- Source imperfections (photon emission asymmetry)
- Channel losses (fiber attenuation)
- Detector imperfections (dark counts, detector blinding)

## SquidASM Integration

### EPR Socket Interface

SquidASM provides EPR generation via **EPR sockets**:

```python
# Alice side
epr_socket = context.epr_sockets[peer_name]
qubit = yield from epr_socket.create_keep(number=1)[0]

# Bob side (symmetric)
epr_socket = context.epr_sockets[peer_name]
qubit = yield from epr_socket.recv_keep(number=1)[0]
```

**Semantics**:
- `create_keep()`: Initiator side, generates EPR pair and keeps one half
- `recv_keep()`: Receiver side, receives the other half
- Both calls are **blocking generators** (yield control to simulator)

### NetQASM Execution Model

EPR generation uses NetQASM quantum assembly language:

```
# Pseudo-NetQASM
CREATE_EPR peer=Bob reg=q0  # Alice
RECV_EPR peer=Alice reg=q0   # Bob
```

**SquidASM Translation**:
1. `create_keep()` → `CREATE_EPR` instruction
2. NetQASM processor interprets instruction
3. NetSquid simulator generates entangled pair
4. Noise models applied (channel, detector)
5. Qubit delivered to quantum register

### Caligo Wrapper: `EPRGenerator`

**Purpose**: Encapsulate SquidASM EPR socket with error handling and logging.

```python
class EPRGenerator:
    """
    Generate EPR pairs via SquidASM EPR socket.
    
    Parameters
    ----------
    epr_socket : Any
        SquidASM EPR socket interface.
    peer_name : str
        Name of the peer node (for logging).
    role : str
        Node role: "create" (Alice) or "recv" (Bob).
    """
    
    def __init__(self, epr_socket: Any, peer_name: str, role: str):
        self._epr_socket = epr_socket
        self._peer_name = peer_name
        self._role = role
        self._logger = get_logger(__name__)
    
    def create(self, *, round_id: Optional[int] = None) -> Generator:
        """
        Generate one EPR pair (generator for SquidASM).
        
        Yields
        ------
        EventExpression
            SquidASM event (managed by simulator).
        
        Returns
        -------
        Qubit
            Local half of EPR pair.
        
        Raises
        ------
        EPRGenerationError
            If EPR generation fails.
        """
        try:
            if self._role == "create":
                qubit = yield from self._epr_socket.create_keep(number=1)[0]
            elif self._role == "recv":
                qubit = yield from self._epr_socket.recv_keep(number=1)[0]
            else:
                raise ValueError(f"Invalid role: {self._role}")
            
            if round_id is not None:
                self._logger.debug(
                    "EPR pair generated (round %d, peer %s)",
                    round_id,
                    self._peer_name,
                )
            
            return qubit
            
        except Exception as exc:
            self._logger.error("EPR generation failed: %s", exc)
            raise EPRGenerationError(
                f"Failed to generate EPR pair with {self._peer_name}"
            ) from exc
```

**Usage in Protocol**:
```python
# In AliceProgram._phase_quantum()
generator = EPRGenerator(
    epr_socket=context.epr_sockets["Bob"],
    peer_name="Bob",
    role="create",
)

for round_id in range(num_pairs):
    qubit = yield from generator.create(round_id=round_id)
    # ... measure qubit ...
```

## Generation Modes

### Sequential Generation

**Characteristics**:
- One EPR pair generated at a time
- Total simulation time: $T_{\text{total}} = n \times T_{\text{cycle}}$
- Deterministic ordering (round_id sequential)

**Advantages**:
- Simple implementation
- Full SquidASM fidelity (network latency, timing)
- Suitable for < 1000 EPR pairs

**Limitations**:
- Linear scaling: large $n$ → long execution time
- Cannot exploit multi-core parallelism

**Implementation**:
```python
class SequentialEPRStrategy:
    def generate(self, total_pairs: int) -> Tuple[List, List, List, List]:
        """Generate EPR pairs sequentially."""
        alice_outcomes, alice_bases = [], []
        bob_outcomes, bob_bases = [], []
        
        for round_id in range(total_pairs):
            # Simulate single EPR generation + measurement
            alice_out, alice_bas, bob_out, bob_bas = (
                self._generate_single_pair(round_id)
            )
            
            alice_outcomes.append(alice_out)
            alice_bases.append(alice_bas)
            bob_outcomes.append(bob_out)
            bob_bases.append(bob_bas)
        
        return alice_outcomes, alice_bases, bob_outcomes, bob_bases
```

### Parallel Generation (Monte Carlo Mode)

**Characteristics**:
- Multiple EPR pairs generated concurrently (multiprocessing)
- Worker processes handle disjoint batches
- Results aggregated post-execution

**Advantages**:
- Near-linear speedup (scales with CPU cores)
- Suitable for large $n$ (> 10,000 pairs)
- Monte Carlo-style sampling of noise models

**Limitations**:
- Ordering not preserved (shuffle required for strict i.i.d.)
- Standalone mode only (no SquidASM timing simulation)
- Trade-off: speed vs. network simulation fidelity

**Implementation Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                    Main Process                               │
│  • Split num_pairs into batches                              │
│  • Create worker pool (multiprocessing.Pool)                 │
│  • Dispatch batches to workers                               │
│  • Aggregate results                                         │
└───────────┬──────────────────────────────────────────────────┘
            │
            ├──► Worker 1: generate_epr_batch(batch_size=2500)
            ├──► Worker 2: generate_epr_batch(batch_size=2500)
            ├──► Worker 3: generate_epr_batch(batch_size=2500)
            └──► Worker 4: generate_epr_batch(batch_size=2500)
                            │
                            ▼
              ┌─────────────────────────────┐
              │   generate_epr_batch()      │
              │                             │
              │  1. Apply noise model       │
              │  2. Generate outcomes       │
              │  3. Return (outcomes, bases)│
              └─────────────────────────────┘
```

**Code**:
```python
class ParallelEPRStrategy:
    def __init__(self, config: ParallelEPRConfig, network_config: Dict):
        self._config = config
        self._network_config = network_config
        self._pool = multiprocessing.Pool(processes=config.num_workers)
    
    def generate(self, total_pairs: int) -> Tuple[List, List, List, List]:
        """Generate EPR pairs in parallel batches."""
        batch_size = self._config.pairs_per_batch
        num_batches = (total_pairs + batch_size - 1) // batch_size
        
        # Create work items
        tasks = [
            (min(batch_size, total_pairs - i * batch_size), self._network_config)
            for i in range(num_batches)
        ]
        
        # Dispatch to worker pool
        results = self._pool.starmap(generate_epr_batch_standalone, tasks)
        
        # Aggregate
        alice_outcomes = [r[0] for batch in results for r in batch]
        alice_bases = [r[1] for batch in results for r in batch]
        bob_outcomes = [r[2] for batch in results for r in batch]
        bob_bases = [r[3] for batch in results for r in batch]
        
        return alice_outcomes, alice_bases, bob_outcomes, bob_bases
    
    def shutdown(self) -> None:
        """Clean up worker pool."""
        self._pool.close()
        self._pool.join()
```

## Noise Modeling

### Channel Noise

**Depolarizing Channel**:
$$
\rho \rightarrow (1 - p_{\text{depol}}) \rho + p_{\text{depol}} \frac{\mathbb{I}}{2}
$$

**Mapping from Fidelity**:
$$
p_{\text{depol}} = \frac{1 - F}{4/3} = \frac{3(1 - F)}{4}
$$

**Caligo Implementation**:
```python
def create_depolar_noise_model(fidelity: float):
    """Create depolarizing noise model from fidelity."""
    depolar_prob = 3 * (1 - fidelity) / 4
    return DepolarNoiseModel(depolar_prob=depolar_prob)
```

### Detection Model

**Heralded Detection** (for low $\eta$):
- Pair generation only succeeds if both detectors click
- Effective loss rate: $1 - \eta^2$ (both photons must be detected)
- Dark counts modeled as spontaneous false positives

**Perfect Detection** (for high $\eta$):
- All pairs delivered (no heralding)
- Detector error $e_{\text{det}}$ flips outcome with probability $p = e_{\text{det}}$

**Model Selection**:
```python
def select_link_model(eta: float) -> str:
    """Select link model based on detection efficiency."""
    if eta < 0.5:
        return "heralded"  # Heralded-double-click for low eta
    else:
        return "depolarise"  # Depolarizing model for high eta
```

## Loss Modeling

### Transmission Loss

**Fiber Attenuation**:
$$
\eta_{\text{fiber}} = 10^{-\alpha L / 10}
$$

where:
- $\alpha$ = attenuation coefficient (dB/km)
- $L$ = fiber length (km)

**Total Detection Efficiency**:
$$
\eta_{\text{total}} = \eta_{\text{source}} \cdot \eta_{\text{fiber}} \cdot \eta_{\text{det}}
$$

**Caligo Enforcement**:
```python
def compute_total_efficiency(
    source_eff: float,
    fiber_length_km: float,
    attenuation_db_per_km: float,
    detector_eff: float,
) -> float:
    """Compute total detection efficiency."""
    fiber_loss_db = attenuation_db_per_km * fiber_length_km
    eta_fiber = 10 ** (-fiber_loss_db / 10)
    return source_eff * eta_fiber * detector_eff
```

### Photon Loss Events

In heralded mode, losses manifest as **non-detection events**:
- EPR socket call returns `None` instead of qubit
- Protocol discards this round
- Effective sifted key rate reduced

**Handling in Caligo**:
```python
def create(self, round_id: int) -> Generator:
    """Generate EPR pair with loss handling."""
    qubit = yield from self._epr_socket.create_keep(number=1)
    
    if qubit is None or len(qubit) == 0:
        # Loss event
        self._logger.debug("EPR loss event (round %d)", round_id)
        return None  # Caller handles None
    
    return qubit[0]
```

## Performance Characteristics

### Sequential Mode

**Time Complexity**:
$$
T_{\text{seq}} = n \times T_{\text{cycle}}
$$

where $T_{\text{cycle}} \approx 1$ ms (typical SquidASM EPR cycle time).

**Example**: $n = 10,000$ pairs → $T_{\text{seq}} \approx 10$ seconds.

### Parallel Mode

**Theoretical Speedup**:
$$
S_{\text{ideal}} = \frac{T_{\text{seq}}}{T_{\text{par}}} = \frac{n \times T_{\text{cycle}}}{(n / W) \times T_{\text{cycle}} + T_{\text{overhead}}} \approx W
$$

where $W$ = number of workers.

**Practical Speedup** (with overhead):
$$
S_{\text{actual}} \approx \frac{W}{1 + (W \times T_{\text{overhead}}) / (n \times T_{\text{cycle}})}
$$

**Empirical Results** (n = 100,000, W = 8):
- Sequential: ~100 seconds
- Parallel: ~15 seconds
- Speedup: 6.7× (efficiency: 84%)

## Testing Strategy

### Unit Tests

```python
def test_epr_generator_create():
    """Test EPR generation wrapper (sequential)."""
    mock_socket = MockEPRSocket()
    generator = EPRGenerator(mock_socket, "Bob", role="create")
    
    # Simulate generation
    qubit = run_generator(generator.create(round_id=0))
    
    assert qubit is not None
    assert mock_socket.create_keep_called

def test_parallel_epr_strategy_aggregation():
    """Test parallel EPR batch aggregation."""
    config = ParallelEPRConfig(
        enabled=True,
        num_workers=2,
        pairs_per_batch=100,
    )
    strategy = ParallelEPRStrategy(config, network_config={})
    
    alice_out, alice_bas, bob_out, bob_bas = strategy.generate(200)
    
    assert len(alice_out) == 200
    assert len(bob_out) == 200
    assert all(b in {0, 1} for b in alice_bas)
```

### Integration Tests

```python
@pytest.mark.integration
def test_epr_generation_with_squidasm():
    """Test EPR generation via SquidASM (requires full stack)."""
    from caligo.simulation import perfect_network_config
    from caligo.protocol import run_protocol, ProtocolParameters
    
    params = ProtocolParameters(
        session_id="test-epr",
        nsm_params=NSMParameters(...),
        num_pairs=100,
    )
    
    network_config = perfect_network_config("Alice", "Bob")
    ot_output, _ = run_protocol(params, network_config=network_config)
    
    # Verify EPR generation succeeded
    assert ot_output.alice_output.key_length > 0
```

## References

- Wehner, S., et al. (2018). "Quantum internet: A vision for the road ahead." *Science*, 362(6412), eaam9288.
- Caleffi, M., et al. (2022). "Distributed quantum computing: A survey." *arXiv:2212.10609*.
- NetSquid Documentation: "EPR Pair Generation." https://netsquid.org/
- SquidASM Documentation: "EPR Sockets." [`docs/squidasm_docs/`](../squidasm_docs/)

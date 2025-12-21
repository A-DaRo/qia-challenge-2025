# 3.3 Domain-Driven Design

## Introduction

Caligo uses **Domain-Driven Design (DDD)** principles to achieve:
1. **Ubiquitous Language**: Shared terminology between cryptographers and engineers
2. **Bounded Contexts**: Clear separation between quantum, classical, and simulation layers
3. **Type Safety**: Compile-time guarantees via Python type hints and runtime validation

This section formalizes Caligo's domain model and architectural patterns.

## Ubiquitous Language

### Core Domain Terms

The following terms constitute Caligo's **ubiquitous language**, used consistently across code, documentation, and specifications:

#### Cryptographic Primitives

| Term | Definition | Caligo Type |
|------|-----------|-------------|
| **Oblivious Key** | Cryptographic key from 1-2 OT | `ObliviousKey` |
| **Alice** | Sender (possesses $S_0, S_1$) | `AliceProgram`, `AliceOTOutput` |
| **Bob** | Receiver (possesses $S_C$, chooses $C$) | `BobProgram`, `BobOTOutput` |
| **Choice Bit** | Bob's selection $C \in \{0, 1\}$ | `int` (validated) |
| **Sender Privacy** | Alice oblivious to $C$ | — (security property) |
| **Receiver Privacy** | Bob oblivious to $S_{1-C}$ | — (security property) |

#### Quantum Layer

| Term | Definition | Caligo Type |
|------|-----------|-------------|
| **EPR Pair** | Entangled qubit pair (Bell state) | — (SquidASM primitive) |
| **Basis** | Measurement basis: $Z$ (0) or $X$ (1) | `int` ∈ {0, 1} |
| **Outcome** | Measurement result: 0 or 1 | `int` ∈ {0, 1} |
| **Round** | Single EPR generation + measurement | `int` (round_id) |
| **Detection Event** | Successful photon detection | `DetectionEvent` |
| **Timing Barrier** | NSM $\Delta t$ enforcement | `TimingBarrier` |

#### Sifting & QBER

| Term | Definition | Caligo Type |
|------|-----------|-------------|
| **Sifting** | Discard basis-mismatched outcomes | `Sifter.sift()` |
| **Sifted Key** | Outcomes where $b_A = b_B$ | `bitarray` |
| **Raw Key** | Pre-sifting measurement outcomes | `np.ndarray` |
| **QBER** | Quantum Bit Error Rate | `float` ∈ [0, 1] |
| **Commitment** | Cryptographic hash of basis choices | `SHA256Commitment` |
| **Hoeffding Bound** | Statistical confidence interval | `HoeffdingBound` |

#### Reconciliation

| Term | Definition | Caligo Type |
|------|-----------|-------------|
| **LDPC Code** | Low-Density Parity-Check code | `CompiledParityCheckMatrix` |
| **Syndrome** | Error syndrome $\mathbf{s} = \mathbf{H} \cdot \mathbf{k}$ | `bitarray` |
| **Belief Propagation** | Iterative LDPC decoding | `BeliefPropagationDecoder` |
| **Code Rate** | $R = k / n$ (info bits / total bits) | `float` ∈ (0, 1) |
| **Leakage** | Information revealed to adversary | `int` (bits) |
| **Verification Hash** | GF(2) polynomial hash | `PolynomialHashVerifier` |

#### Privacy Amplification

| Term | Definition | Caligo Type |
|------|-----------|-------------|
| **Min-Entropy** | Worst-case unpredictability | `float` (bits) |
| **Toeplitz Matrix** | Universal hash family matrix | `np.ndarray` |
| **Extractable Length** | Secure key length post-amplification | `int` (bits) |
| **Security Parameter** | Adversary's advantage bound $\epsilon$ | `float` |

### Naming Conventions

**Class Names** (PascalCase):
- Nouns for entities: `ObliviousKey`, `DetectionEvent`
- Descriptive for services: `QBEREstimator`, `ToeplitzHasher`

**Function Names** (snake_case):
- Verbs for actions: `compute_qber()`, `extract_keys()`
- Domain terms: `sift_keys()`, `reconcile_block()`

**Constants** (UPPER_SNAKE_CASE):
- Security thresholds: `QBER_CONSERVATIVE_THRESHOLD`
- Physical constants: `DEFAULT_EPSILON_SEC`

## Bounded Contexts

Caligo is organized into **four bounded contexts**, each with distinct responsibilities and minimal inter-context dependencies.

### Context Map

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROTOCOL CONTEXT (Phase E)                    │
│                                                                 │
│  Responsibility: Orchestrate four-phase protocol execution      │
│  Dependencies: All other contexts                               │
│  Exports: run_protocol(), AliceProgram, BobProgram             │
│                                                                 │
│  Anti-Corruption Layer: OrderedSocket wraps SquidASM sockets   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│              CRYPTOGRAPHIC CONTEXT (Phase D)                    │
│                                                                 │
│  Responsibility: Implement protocol phases (Quantum, Sifting,   │
│                  Reconciliation, Amplification)                 │
│  Dependencies: Infrastructure Context only                      │
│  Exports: Sifter, ReconciliationOrchestrator, ToeplitzHasher   │
│                                                                 │
│  Bounded by: Phase boundary contracts (DTOs)                    │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│             INFRASTRUCTURE CONTEXT                              │
│                                                                 │
│  Responsibility: Cross-cutting concerns (types, utils, logging) │
│  Dependencies: None (foundation layer)                          │
│  Exports: Phase contracts, exception hierarchy, bitarray utils  │
│                                                                 │
│  Enforces: Contract validation, type safety, logging standard   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│              SIMULATION CONTEXT                                 │
│                                                                 │
│  Responsibility: SquidASM integration, NSM parameter mapping    │
│  Dependencies: Infrastructure Context, SquidASM (external)      │
│  Exports: CaligoNetworkBuilder, NSMParameters, TimingBarrier    │
│                                                                 │
│  Abstraction: Hides SquidASM complexity from Cryptographic      │
└─────────────────────────────────────────────────────────────────┘
```

### Context Interactions

**Upstream/Downstream Relationships**:
- Protocol Context **depends on** Cryptographic Context (Customer-Supplier)
- Cryptographic Context **depends on** Infrastructure Context (Conformist)
- Simulation Context **translates** between NSM and SquidASM (Anti-Corruption Layer)

**Integration Patterns**:
- **Shared Kernel**: `types/phase_contracts.py` (all contexts use these DTOs)
- **Published Language**: Numpydoc docstrings define interface contracts
- **Open-Host Service**: `run_protocol()` is public API entry point

## Layered Architecture

### Layer Definitions

```
┌───────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                        │
│  • CLI (cli.py)                                               │
│  • Configuration loading (YAML → dataclasses)                 │
│  • Entry points: run_protocol(), parallel EPR CLI            │
└───────────────────────────┬───────────────────────────────────┘
                            │ Calls
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                           │
│  • ReconciliationOrchestrator (coordinates Phase III)         │
│  • FeasibilityChecker (pre-flight validation)                │
│  • OTOutputFormatter (assembles final output)                │
└───────────────────────────┬───────────────────────────────────┘
                            │ Uses
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER                            │
│  • Entities: ObliviousKey, MeasurementRecord                 │
│  • Value Objects: HoeffdingBound, QBEREstimate               │
│  • Domain Services: QBEREstimator, NSMEntropyCalculator      │
└───────────────────────────┬───────────────────────────────────┘
                            │ Operates on
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                        │
│  • Repositories: MatrixManager (LDPC code pool)              │
│  • Technical Services: LogManager, bitarray utilities        │
│  • External Adapters: OrderedSocket (wraps SquidASM)         │
└───────────────────────────────────────────────────────────────┘
```

### Dependency Rule

**Strict Inward Dependency**: Outer layers depend on inner layers, never vice versa.

**Enforcement**:
```python
# ✓ ALLOWED: Service layer imports domain
from caligo.sifting import QBEREstimator

# ✗ FORBIDDEN: Domain layer imports SquidASM
# from squidasm.sim.stack.program import Program  # Would violate DDD
```

## Aggregates and Entities

### Aggregate Roots

An **aggregate** is a cluster of domain objects treated as a single unit for data consistency.

#### Aggregate 1: `QuantumPhaseResult`

**Aggregate Root**: `QuantumPhaseResult`

**Entities**:
- `DetectionEvent` (timestamped measurement record)

**Value Objects**:
- `measurement_outcomes` (np.ndarray)
- `basis_choices` (np.ndarray)

**Invariants**:
- `POST-Q-001`: Array lengths match `num_pairs_generated`
- `POST-Q-003`: All outcomes ∈ {0, 1}
- `POST-Q-005`: Timing barrier marked

**Consistency Boundary**: All data originates from single quantum phase execution; cannot be partially updated.

#### Aggregate 2: `ReconciliationPhaseResult`

**Aggregate Root**: `ReconciliationPhaseResult`

**Entities**:
- `BlockResult` (per-block reconciliation outcome)

**Value Objects**:
- `alice_reconciled_key` (bitarray)
- `bob_reconciled_key` (bitarray)
- `syndrome_leakage`, `verification_leakage` (int)

**Invariants**:
- `POST-R-002`: Keys have zero Hamming distance
- `POST-R-003`: Total leakage = sum of block leakages

**Consistency Boundary**: All blocks reconciled atomically; partial reconciliation is invalid state.

#### Aggregate 3: `ObliviousTransferOutput`

**Aggregate Root**: `ObliviousTransferOutput`

**Entities**:
- `AliceOTOutput` (contains $S_0, S_1$)
- `BobOTOutput` (contains $S_C, C$)

**Value Objects**:
- `protocol_metadata` (Dict[str, Any])

**Invariants**:
- `POST-A-001`: $|S_0| = |S_1| = |S_C|$
- `POST-A-005`: Alice oblivious to $C$, Bob oblivious to $S_{1-C}$

**Consistency Boundary**: OT output is final; no modification allowed after creation (frozen dataclass).

## Value Objects

**Definition**: Immutable objects defined by their attributes, not identity.

### Examples

#### `HoeffdingBound`

```python
@dataclass(frozen=True)
class HoeffdingBound:
    lower_bound: float
    upper_bound: float
    confidence: float
    sample_size: int
    
    def contains(self, value: float) -> bool:
        return self.lower_bound <= value <= self.upper_bound
```

**Value Object Properties**:
- Immutable (`frozen=True`)
- Equality by value (not identity)
- No side effects

#### `QBEREstimate`

```python
@dataclass(frozen=True)
class QBEREstimate:
    qber: float
    num_errors: int
    num_compared: int
    hoeffding_bound: HoeffdingBound
    
    def __post_init__(self) -> None:
        if not (0 <= self.qber <= 1):
            raise ContractViolation("QBER must be in [0, 1]")
```

## Domain Services

**Definition**: Stateless operations that don't naturally belong to an entity or value object.

### Service Pattern

```python
class QBEREstimator:
    """Domain service for QBER estimation.
    
    Stateless: No instance variables (other than config).
    Pure functions: Same input → same output.
    """
    
    def estimate(
        self,
        alice_key: bitarray,
        bob_key: bitarray,
        sample_size: int,
    ) -> QBEREstimate:
        """Estimate QBER from sifted keys."""
        # Sample random positions
        indices = random.sample(range(len(alice_key)), sample_size)
        
        # Count errors
        errors = sum(alice_key[i] != bob_key[i] for i in indices)
        qber = errors / sample_size
        
        # Compute Hoeffding bound
        bound = self._compute_hoeffding(sample_size, qber)
        
        return QBEREstimate(
            qber=qber,
            num_errors=errors,
            num_compared=sample_size,
            hoeffding_bound=bound,
        )
```

### Service Registry

| Service | Responsibility | Statefulness |
|---------|---------------|--------------|
| `QBEREstimator` | Estimate channel error rate | Stateless |
| `NSMEntropyCalculator` | Compute min-entropy bounds | Stateless |
| `ToeplitzHasher` | Universal hashing | Stateful (RNG seed) |
| `ReconciliationOrchestrator` | Coordinate reconciliation | Stateful (leakage tracker) |

## Factories and Strategies

### Factory Pattern

**Purpose**: Encapsulate complex object creation logic.

#### `EPRGenerationFactory`

```python
class EPRGenerationFactory:
    """Factory for EPR generation strategies."""
    
    def __init__(self, config: CaligoConfig):
        self._config = config
    
    def create_strategy(self) -> EPRGenerationStrategy:
        """Create strategy based on configuration."""
        if self._config.parallel_config.enabled:
            return ParallelEPRStrategy(
                config=self._config,
                num_workers=self._config.parallel_config.num_workers,
            )
        else:
            return SequentialEPRStrategy(
                network_config=self._config.network_config,
            )
```

**Benefits**:
- Client code decoupled from strategy implementation
- Centralized configuration logic
- Testable in isolation

### Strategy Pattern

**Purpose**: Enable algorithm swapping without modifying client code.

#### `ReconciliationStrategy`

```python
class ReconciliationStrategy(Protocol):
    """Protocol (interface) for reconciliation strategies."""
    
    def reconcile_block(
        self,
        alice_block: bitarray,
        bob_block: bitarray,
        context: Any,
    ) -> Generator[Any, None, BlockResult]:
        """Reconcile single block (generator for SquidASM)."""
        ...
```

**Implementations**:
1. `BaselineStrategy`: Alice-driven, uses known QBER to select code rate
2. `BlindStrategy`: Bob-driven, iterative refinement without QBER knowledge

**Usage**:
```python
# Factory selects strategy
strategy = create_strategy(config, mother_code, codec, leakage_tracker)

# Client code is strategy-agnostic
for block_id, block_data in enumerate(blocks):
    result = yield from strategy.reconcile_block(
        alice_block, bob_block, context
    )
```

## Repositories

**Definition**: Abstraction over data access (CRUD operations).

### `MatrixManager` (LDPC Repository)

```python
class MatrixManager:
    """Repository for LDPC parity-check matrices.
    
    Manages a pool of compiled LDPC codes with different rates.
    """
    
    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._cache: Dict[str, CompiledParityCheckMatrix] = {}
    
    def get_matrix(
        self,
        code_rate: float,
        frame_size: int,
    ) -> CompiledParityCheckMatrix:
        """Retrieve matrix by rate and size (cached)."""
        key = f"rate_{code_rate:.3f}_n_{frame_size}"
        
        if key not in self._cache:
            matrix = self._load_from_disk(code_rate, frame_size)
            self._cache[key] = matrix
        
        return self._cache[key]
```

**Repository Pattern Benefits**:
- Domain layer doesn't know matrices are stored on disk
- Caching strategy encapsulated
- Easy to swap backend (e.g., in-memory for tests, Redis for distributed)

## Anti-Corruption Layer (ACL)

**Purpose**: Protect domain model from external system complexity (SquidASM).

### `OrderedSocket` (ACL for SquidASM Classical Sockets)

**Problem**: SquidASM sockets are unordered; messages can arrive out-of-sequence.

**Solution**: Wrap SquidASM socket with ordering enforcement:

```python
class OrderedSocket:
    """ACL: Enforce message ordering on SquidASM socket."""
    
    def __init__(self, socket, session_id: str):
        self._sock = socket  # SquidASM socket
        self._session_id = session_id
        self._send_seq = 0
        self._recv_seq = 0
    
    def send(self, msg_type: MessageType, payload: dict) -> Generator:
        """Send message and wait for ACK."""
        envelope = MessageEnvelope(
            session_id=self._session_id,
            seq=self._send_seq,
            msg_type=msg_type,
            payload=payload,
        )
        self._sock.send(envelope.serialize())
        
        # Wait for ACK before returning
        ack = yield from self._recv_ack()
        self._send_seq += 1
    
    def recv(self, expected_type: MessageType) -> Generator:
        """Receive message and send ACK."""
        envelope = yield from self._recv_envelope()
        
        if envelope.seq != self._recv_seq:
            raise OutOfOrderError(...)
        
        # Send ACK
        self._send_ack(envelope.seq)
        self._recv_seq += 1
        
        return envelope.payload
```

**Benefits**:
- Protocol layer sees ordered, reliable channel
- SquidASM complexity hidden
- Can swap underlying transport without changing protocol logic

### `CaligoNetworkBuilder` (ACL for NSM → SquidASM Mapping)

**Problem**: NSM parameters (abstract) must map to SquidASM noise models (concrete).

**Solution**: Builder translates between domains:

```python
class CaligoNetworkBuilder:
    """ACL: Translate NSM parameters to SquidASM config."""
    
    def build(
        self,
        nodes: List[str],
        nsm_params: NSMParameters,
        channel_params: ChannelParameters,
    ) -> StackNetworkConfig:
        """Build SquidASM network from NSM parameters."""
        
        # Translate NSM → Physical noise models
        link_noise = self._create_link_noise(
            eta=nsm_params.detection_eff_eta,
            fidelity=nsm_params.channel_fidelity,
            dark_count=nsm_params.dark_count_prob,
        )
        
        storage_noise = NSMStorageNoiseModel(
            r=nsm_params.storage_noise_r,
            delta_t=nsm_params.delta_t_ns,
        )
        
        # Construct SquidASM config
        return self._build_stack_network(
            nodes, link_noise, storage_noise, channel_params
        )
```

**Benefits**:
- Domain layer works with `NSMParameters` (domain language)
- SquidASM details isolated to Simulation Context
- Can replace SquidASM with different simulator by modifying ACL only

## Specification Pattern

**Purpose**: Encapsulate business rules as reusable, combinable predicates.

### Security Specifications

```python
class NSMSecuritySpecification:
    """Specification: NSM security constraints."""
    
    def __init__(self, nsm_params: NSMParameters):
        self._nsm = nsm_params
    
    def is_satisfied_by(self, qber: float) -> bool:
        """Check if QBER satisfies NSM security."""
        q_storage = (1 - self._nsm.storage_noise_r) / 2
        return qber < q_storage
    
    def and_(self, other: Specification) -> Specification:
        """Combine specifications with AND."""
        return AndSpecification(self, other)
```

**Usage**:
```python
nsm_spec = NSMSecuritySpecification(nsm_params)
qber_spec = QBERThresholdSpecification(threshold=0.11)

combined_spec = nsm_spec.and_(qber_spec)

if not combined_spec.is_satisfied_by(observed_qber):
    raise SecurityError("Security specifications not satisfied")
```

## Contract-Driven Development

### Design by Contract

**Principle**: Every module has a **contract** specifying:
1. **Pre-conditions**: What caller must guarantee
2. **Post-conditions**: What module guarantees
3. **Invariants**: What is always true

### Contract Enforcement

**Level 1: Type Hints (Compile-Time)**
```python
def reconcile_block(
    alice_key: bitarray,
    bob_key: bitarray,
    parity_matrix: CompiledParityCheckMatrix,
) -> BlockResult:
    ...
```

**Level 2: Dataclass Validation (Runtime)**
```python
@dataclass
class SiftingPhaseResult:
    qber_estimate: float
    
    def __post_init__(self) -> None:
        if not (0 <= self.qber_estimate <= 1):
            raise ContractViolation("QBER must be in [0, 1]")
```

**Level 3: Explicit Assertions**
```python
def extract_key(reconciled_key: bitarray, length: int) -> bitarray:
    assert length > 0, "PRE: length must be positive"
    
    extracted = toeplitz_hash(reconciled_key, length)
    
    assert len(extracted) == length, "POST: extracted length matches"
    return extracted
```

## Testing Strategy

### Unit Tests (Per-Aggregate)

```python
class TestQuantumPhaseResult:
    def test_post_condition_q001_length_match(self):
        """POST-Q-001: outcomes length matches num_generated."""
        result = QuantumPhaseResult(
            measurement_outcomes=np.array([0, 1, 0]),
            basis_choices=np.array([0, 1, 0]),
            num_pairs_generated=3,
            # ... other fields
        )
        assert len(result.measurement_outcomes) == 3
    
    def test_post_condition_q003_outcome_range(self):
        """POST-Q-003: outcomes in {0, 1}."""
        with pytest.raises(ContractViolation, match="POST-Q-003"):
            QuantumPhaseResult(
                measurement_outcomes=np.array([0, 2]),  # Invalid!
                num_pairs_generated=2,
                # ... other fields
            )
```

### Integration Tests (Cross-Aggregate)

```python
@pytest.mark.integration
def test_sifting_to_reconciliation_phase_boundary():
    """Verify SiftingPhaseResult → ReconciliationOrchestrator."""
    # Create valid sifting result
    sifting_result = SiftingPhaseResult(
        alice_sifted_key=alice_key,
        bob_sifted_key=bob_key,
        qber_estimate=0.05,
        # ... other fields
    )
    
    # Should be accepted by reconciliation
    orchestrator = ReconciliationOrchestrator(matrix_manager)
    recon_result = orchestrator.reconcile_all_blocks(
        alice_sifted=sifting_result.alice_sifted_key,
        bob_sifted=sifting_result.bob_sifted_key,
        qber=sifting_result.qber_estimate,
    )
    
    # Verify post-conditions
    assert recon_result.total_leakage > 0
    assert hamming_distance(
        recon_result.alice_reconciled_key,
        recon_result.bob_reconciled_key,
    ) == 0
```

### End-to-End Tests (Full Protocol)

```python
@pytest.mark.e2e
def test_full_protocol_erven_parameters():
    """E2E: Full protocol with Erven et al. (2014) parameters."""
    params = ProtocolParameters(
        session_id="erven-e2e",
        nsm_params=load_erven_params(),
        num_pairs=10_000,
    )
    
    ot_output, _ = run_protocol(params, bob_choice_bit=1)
    
    # Verify OT properties
    assert ot_output.alice_output.key_length > 0
    assert ot_output.bob_output.key_length > 0
    assert ot_output.bob_output.choice_bit == 1
```

## Documentation as Specification

### Numpydoc Format

All public APIs use Numpydoc format specifying:
- **Parameters**: Input pre-conditions
- **Returns**: Output post-conditions
- **Raises**: Exception conditions
- **Notes**: Invariants and domain constraints

**Example**:
```python
def compute_qber(errors: int, total: int) -> float:
    """
    Compute Quantum Bit Error Rate.

    Parameters
    ----------
    errors : int
        Number of bit errors, must be ≥ 0.
    total : int
        Total bits compared, must be > 0.

    Returns
    -------
    float
        QBER in range [0, 1].

    Raises
    ------
    ValueError
        If total ≤ 0 or errors > total.

    Notes
    -----
    **Contract**:
    - PRE: errors ≥ 0, total > 0, errors ≤ total
    - POST: 0 ≤ result ≤ 1
    
    QBER is defined as:
    
        Q = errors / total

    References
    ----------
    - Erven et al. (2014), Eq. (1)
    """
    if total <= 0:
        raise ValueError("PRE violated: total must be positive")
    if errors > total:
        raise ValueError("PRE violated: errors cannot exceed total")
    
    qber = errors / total
    
    assert 0 <= qber <= 1, "POST violated"
    return qber
```

## References

- Evans, E. (2003). *Domain-Driven Design: Tackling Complexity in the Heart of Software*. Addison-Wesley.
- Martin, R. C. (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
- Vernon, V. (2013). *Implementing Domain-Driven Design*. Addison-Wesley.
- Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.

# SquidASM Framework Assessment for E-HOK Implementation

This document provides a comprehensive analysis of the SquidASM framework's native capabilities against the requirements of the E-HOK protocol implementation phases. It identifies gaps requiring extensions to lower-level packages (NetQASM/NetSquid) and proposes integration strategies.

---

## Phase I: Quantum Generation & Physical Setup

### 1. Requirements Summary

Phase I establishes the physical foundation of E-HOK with the following critical requirements:

| Requirement | Description | Criticality |
|-------------|-------------|-------------|
| **Pre-Flight Feasibility Check** | Validate that trusted noise < untrusted noise (NSM assumption) | HIGH |
| **QBER Estimation** | Calculate and enforce QBER limits (22% hard, 11% conservative) | HIGH |
| **Wait Time Enforcement (Δt)** | Strict timing between qubit transmission and basis revelation | CRITICAL |
| **Noise Model Parameters** | Access to Source Quality (μ), Detection Efficiency (η), Intrinsic Error (e_det) | HIGH |
| **Modular Commitment** | Support for both Physical (wait time) and TLP-based commitment | MEDIUM |

### 2. SquidASM Native Support Assessment

#### 2.1 EPR Generation and Distribution
**Status: ✅ NATIVELY SUPPORTED**

SquidASM provides EPR pair generation through the `EPRSocket` interface:

```python
# From squidasm/squidasm/sim/stack/program.py
epr_sockets: Dict[str, EPRSocket]
```

The underlying `MagicDistributor` in `netsquid_magic` handles state delivery with configurable noise models:

| Noise Model | Location | Supported Parameters |
|-------------|----------|---------------------|
| `PerfectStateMagicDistributor` | netsquid_magic | state_delay |
| `DepolariseMagicDistributor` | netsquid_magic | prob_max_mixed, cycle_time |
| `BitflipMagicDistributor` | netsquid_magic | flip_prob |
| `LinearDepolariseMagicDistributor` | squidasm/sim/network/network.py | prob_max_mixed, prob_success, cycle_time |

**Source Code Evidence:**
```python
# From squidasm/squidasm/sim/network/network.py:203-230
def _create_link_distributor(self, link: Link, state_delay: Optional[float] = 1000):
    if noise_type == NoiseType.Depolarise:
        noise = 1 - link.fidelity
        model_params = LinearDepolariseModelParameters(
            cycle_time=state_delay, prob_success=1, prob_max_mixed=noise
        )
```

#### 2.2 Noise Model Configuration
**Status: ⚠️ PARTIALLY SUPPORTED - EXTENSION REQUIRED**

**What's Available:**
- `fidelity` parameter on links (translates to depolarizing noise)
- `T1/T2` times on individual qubits via `T1T2NoiseModel`
- `DepolarNoiseModel`, `DephaseNoiseModel` in NetSquid

**What's Missing:**
- **Source Quality (μ)**: No direct parameter in SquidASM configuration
- **Detection Efficiency (η)**: Not configurable at the SquidASM level
- **Dark Count Rate**: Not exposed in the EPR generation interface
- **QBER Calculation**: No built-in QBER estimation after sifting

**NetSquid Resources Available for Extension:**
```python
# From netsquid/components/models/qerrormodels.py
class DepolarNoiseModel(QuantumErrorModel):
    # depolar_rate: exponential depolarizing rate [Hz] or probability
    
class T1T2NoiseModel(QuantumErrorModel):
    # T1: amplitude damping time
    # T2: dephasing time (T2 Hahn)
    
class FibreLossModel(QuantumErrorModel):
    # Fiber-based loss modeling
```

**Extension Point Identified:**
Create a `PhysicalChannelModel` wrapper that:
1. Accepts NSM-compatible parameters (μ, η, e_det)
2. Converts to NetSquid noise models
3. Provides QBER calculation hooks

#### 2.3 Wait Time Enforcement (Δt)
**Status: ❌ NOT NATIVELY SUPPORTED - EXTENSION REQUIRED**

This is the most critical gap. The NSM security model requires strict ordering:
1. Alice sends qubits
2. Bob acknowledges receipt (implicit measurement/storage)
3. **Timer Δt elapses** (adversary's storage decoheres)
4. Alice reveals bases

**What's Available in NetSquid:**
```python
# From netsquid core - sim_time() provides simulation time
import netsquid as ns
current_time = ns.sim_time()

# Event scheduling available via pydynaa
self._schedule_after(delay_time, event)
```

**SquidASM Timing Capabilities:**
```python
# From squidasm tests - time tracking is possible
start_time = ns.sim_time()
# ... operations ...
completion_time = ns.sim_time() - start_time
```

**Gap Analysis:**
- SquidASM's `run()` method is a generator that yields control for async operations
- No built-in "wait for time Δt" primitive
- Classical sockets don't have delay enforcement

**Proposed Extension:**
```python
# Extension: ehok/quantum/timing.py
class TimedProtocolEnforcer:
    """Enforces NSM timing constraints in E-HOK protocols."""
    
    def __init__(self, delta_t_ns: float):
        self.delta_t = delta_t_ns
        
    def enforce_wait(self, context: ProgramContext) -> Generator:
        """Yields control until Δt has elapsed since qubit transmission."""
        start = ns.sim_time()
        while ns.sim_time() - start < self.delta_t:
            yield  # Release control to simulator
```

#### 2.4 Pre-Flight Feasibility Check
**Status: ❌ NOT NATIVELY SUPPORTED - MUST IMPLEMENT**

Phase I requires aborting if:
- $Q_{total} > 22\%$ (Hard limit - security impossible)
- $Q_{total} > 11\%$ (Conservative limit - warning)

**No native function exists to:**
1. Aggregate noise parameters into total trusted noise
2. Compare against NSM security bounds
3. Calculate minimum required storage noise

**Proposed Extension:**
```python
# Extension: ehok/core/feasibility.py
def pre_flight_check(
    source_quality: float,      # μ
    detection_efficiency: float, # η  
    intrinsic_error: float,     # e_det
    storage_noise: float        # r_storage (adversary)
) -> tuple[bool, str]:
    """
    Validates E-HOK feasibility before protocol execution.
    
    Returns
    -------
    (feasible, message) : tuple[bool, str]
        Whether protocol is feasible and diagnostic message.
    """
    q_total = calculate_total_trusted_noise(source_quality, detection_efficiency, intrinsic_error)
    
    if q_total > 0.22:
        return False, f"ABORT: Total trusted noise {q_total:.2%} exceeds hard limit (22%)"
    elif q_total > 0.11:
        return True, f"WARNING: Total trusted noise {q_total:.2%} exceeds conservative limit (11%)"
    else:
        return True, f"OK: Total trusted noise {q_total:.2%} within bounds"
```

### 3. Integration Strategy for Phase I

```
┌─────────────────────────────────────────────────────────────────┐
│                     E-HOK Framework (ehok/)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────┐ │
│  │ FeasibilityChecker│  │ TimingEnforcer    │  │ NoiseAdapter │ │
│  │ - pre_flight()    │  │ - enforce_wait()  │  │ - μ, η, e_det│ │
│  │ - qber_bounds()   │  │ - Δt calibration  │  │ → NetSquid   │ │
│  └────────┬──────────┘  └────────┬──────────┘  └──────┬───────┘ │
│           │                      │                     │         │
├───────────┴──────────────────────┴─────────────────────┴─────────┤
│                         SquidASM Layer                           │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │  EPRSocket     │  │ ClassicalSocket │  │ ProgramContext     │ │
│  │  - create_keep │  │ - send/recv     │  │ - connection       │ │
│  │  - recv_keep   │  │ - timing (none) │  │ - sockets          │ │
│  └────────┬───────┘  └────────┬────────┘  └──────────┬─────────┘ │
├───────────┴───────────────────┴──────────────────────┴───────────┤
│                         NetQASM Layer                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ NetworkConfig   │  │ Link (fidelity)  │  │ Node (T1, T2)   │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬─────────┘  │
├───────────┴────────────────────┴────────────────────┴────────────┤
│                         NetSquid Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ MagicDistributor │  │ T1T2NoiseModel   │  │ DepolarNoise   │  │
│  │ - state_delay    │  │ - per qubit      │  │ Model          │  │
│  │ - label_delay    │  │                  │  │                │  │
│  └──────────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 4. Files to Create/Modify for Phase I

| File | Purpose | Status |
|------|---------|--------|
| `ehok/core/feasibility.py` | Pre-flight checks, QBER bounds | TO CREATE |
| `ehok/quantum/noise_adapter.py` | NSM params → NetSquid models | TO CREATE |
| `ehok/quantum/timing.py` | Δt enforcement primitives | TO CREATE |
| `ehok/configs/network_config.py` | Extended network config with NSM params | TO CREATE |

---

## Phase II: Sifting & Estimation

### 1. Requirements Summary

Phase II implements the "gatekeeper" functionality with strict temporal ordering requirements:

| Requirement | Description | Criticality |
|-------------|-------------|-------------|
| **"Sandwich" Protocol Flow** | Strict ordering: Quantum Tx → Missing Rounds → Wait Δt → Basis Reveal → Sifting | CRITICAL |
| **Missing Rounds Validation** | Verify reported losses against expected channel transmittance | HIGH |
| **Dynamic Statistical Bounds** | Calculate penalty term μ based on finite-size effects | HIGH |
| **QBER Estimation** | Compute observed QBER from test subset with statistical confidence | HIGH |
| **Decoy State Validation** | (Optional) Separate statistics for signal/decoy pulses | MEDIUM |

### 2. SquidASM Native Support Assessment

#### 2.1 Classical Communication for Protocol Flow
**Status: ✅ NATIVELY SUPPORTED (with caveats)**

SquidASM provides `ClassicalSocket` for message exchange:

```python
# From squidasm/squidasm/sim/stack/csocket.py
class ClassicalSocket(Socket):
    def send(self, msg: str) -> None:
        self.netsquid_socket.send(msg)
    
    def recv(self, **kwargs) -> Generator[EventExpression, None, str]:
        return (yield from self.netsquid_socket.recv())
```

**What Works:**
- Send/receive strings, integers, floats
- Structured messages with headers
- Generator-based async model aligns with simulation

**Gap - No Ordered Acknowledgments:**
The classical socket doesn't enforce acknowledgment ordering. For the "Sandwich" protocol:
- Bob's "Missing Rounds" report must be **acknowledged** by Alice before bases are sent
- No built-in mechanism to prevent race conditions

**NetSquid-Level Support:**
```python
# netsquid_driver provides classical socket service
# But timing guarantees depend on simulation scheduling
```

**Proposed Extension:**
```python
# Extension: ehok/protocols/ordered_messaging.py
class OrderedProtocolSocket:
    """Ensures strict message ordering with acknowledgments."""
    
    def send_with_ack(self, msg: str, timeout_ns: float) -> Generator:
        """Send message and wait for acknowledgment before continuing."""
        self.socket.send(msg)
        ack = yield from self._wait_for_ack(timeout_ns)
        if not ack:
            raise ProtocolViolation("Acknowledgment timeout")
        return ack
    
    def recv_and_ack(self) -> Generator:
        """Receive message and send acknowledgment."""
        msg = yield from self.socket.recv()
        self.socket.send("ACK")
        return msg
```

#### 2.2 Basis Sifting
**Status: ✅ NATIVELY SUPPORTED**

NetQASM/SquidASM provides excellent support for BB84-style basis handling:

```python
# From netqasm/sdk/build_epr.py
class EprMeasBasis(Enum):
    X = 0
    Y = auto()
    Z = auto()
    MX = auto()
    MY = auto()
    MZ = auto()

# From netqasm/sdk/qubit.py
class QubitMeasureBasis(Enum):
    X = 0
    Y = auto()
    Z = auto()
```

**Available Functionality:**
- `Qubit.measure(basis=QubitMeasureBasis.X)` - measure in specific basis
- `basis_rotations` parameter for arbitrary measurement angles
- `RandomBasis.XZ` for BB84-style random basis selection

**Source Evidence (netsquid_magic/link_layer.py:491-520):**
```python
@staticmethod
def _sample_basis_choice(random_basis_set, probability_dist_spec):
    if random_basis_set == RandomBasis.XZ:
        weights = [probability_dist_spec[0], 1 - probability_dist_spec[0]]
        basis = random.choices([MeasurementBasis.X, MeasurementBasis.Z], weights)[0]
```

#### 2.3 Missing Rounds Detection
**Status: ⚠️ PARTIALLY SUPPORTED - EXTENSION REQUIRED**

**What's Available:**
- EPR generation provides success/failure per pair
- `EprMeasureResult` contains measurement outcomes
- Qubit IDs can be tracked per round

**What's Missing:**
- No built-in "detection event" reporting
- No channel transmittance estimation
- No Chernoff bound validation

**NetSquid Resources:**
```python
# From netsquid/components/models/qerrormodels.py
class FibreLossModel(QuantumErrorModel):
    # Can model probabilistic qubit loss
```

**Proposed Extension:**
```python
# Extension: ehok/quantum/detection.py
@dataclass
class DetectionReport:
    """Report of detection events from Bob's side."""
    total_rounds: int
    detected_indices: List[int]
    missing_indices: List[int]
    
    @property
    def detection_rate(self) -> float:
        return len(self.detected_indices) / self.total_rounds

class DetectionValidator:
    """Validates detection reports against expected channel parameters."""
    
    def __init__(self, expected_transmittance: float, epsilon_sec: float = 1e-10):
        self.p_trans = expected_transmittance
        self.epsilon = epsilon_sec
    
    def validate(self, report: DetectionReport) -> tuple[bool, str]:
        """Validate using Chernoff bound."""
        n = report.total_rounds
        s = len(report.detected_indices)
        expected = self.p_trans * n
        
        # Chernoff tolerance
        zeta = math.sqrt(math.log(2/self.epsilon) / (2*n))
        
        if abs(s - expected) > zeta * n:
            return False, f"Detection rate {s/n:.2%} outside bounds [{expected/n - zeta:.2%}, {expected/n + zeta:.2%}]"
        return True, "Detection rate within bounds"
```

#### 2.4 Statistical Penalty Calculation (μ)
**Status: ❌ NOT NATIVELY SUPPORTED - MUST IMPLEMENT**

Phase II requires dynamic calculation of:
$$\mu := \sqrt{\frac{n + k}{n k} \frac{k + 1}{k}} \ln \frac{4}{\varepsilon_{sec}}$$

**No native function exists for:**
- Finite-size penalty calculation
- Smooth min-entropy bounds
- Composable security parameters

**Proposed Extension:**
```python
# Extension: ehok/analysis/statistics.py
def calculate_finite_size_penalty(
    n: int,           # Raw key size
    k: int,           # Test sample size
    epsilon_sec: float = 1e-10
) -> float:
    """
    Calculate statistical penalty μ per Erven et al.
    
    Parameters
    ----------
    n : int
        Size of remaining key bits after sampling
    k : int
        Size of test/sample bits
    epsilon_sec : float
        Security parameter
        
    Returns
    -------
    float
        Penalty term μ to add to observed QBER
    """
    import math
    return math.sqrt((n + k) / (n * k) * (k + 1) / k) * math.log(4 / epsilon_sec)

def compute_max_tolerable_qber(
    observed_qber: float,
    penalty_mu: float,
    hard_limit: float = 0.22,
    conservative_limit: float = 0.11
) -> tuple[bool, str]:
    """
    Check if adjusted QBER is within security bounds.
    """
    adjusted = observed_qber + penalty_mu
    
    if adjusted > hard_limit:
        return False, f"ABORT: Adjusted QBER {adjusted:.2%} exceeds hard limit {hard_limit:.2%}"
    elif adjusted > conservative_limit:
        return True, f"WARNING: Adjusted QBER {adjusted:.2%} exceeds conservative limit"
    else:
        return True, f"OK: Adjusted QBER {adjusted:.2%} within bounds"
```

#### 2.5 Decoy State Support
**Status: ❌ NOT NATIVELY SUPPORTED - COMPLEX EXTENSION REQUIRED**

Decoy state implementation requires:
1. Variable pulse intensity during EPR generation
2. Separate statistics tracking per intensity level
3. PNS attack detection algorithms

**NetSquid Capabilities:**
- `StateSampler` can produce different states probabilistically
- `MagicDistributor` could be extended for intensity variation

**Gap Analysis:**
The current `MagicDistributor` assumes uniform EPR generation. Decoy states require:
- Per-round intensity tagging
- Modified `LinearDepolariseStateSamplerFactory` for intensity-dependent noise
- Post-processing to separate μ and ν statistics

**Proposed Architecture (Complex):**
```python
# Extension: ehok/quantum/decoy_states.py
class DecoyStateDistributor(MagicDistributor):
    """Extended distributor with decoy state support."""
    
    def __init__(self, signal_intensity: float, decoy_intensity: float, 
                 decoy_ratio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.mu = signal_intensity  # Signal intensity
        self.nu = decoy_intensity   # Decoy intensity
        self.decoy_ratio = decoy_ratio
        self._round_intensities: List[float] = []
    
    def add_delivery(self, memory_positions, **kwargs) -> Delivery:
        # Randomly select intensity for this round
        intensity = self.nu if random.random() < self.decoy_ratio else self.mu
        self._round_intensities.append(intensity)
        
        # Adjust noise based on intensity
        # (Multi-photon events are more likely with higher intensity)
        ...
```

### 3. Integration Strategy for Phase II

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           E-HOK Phase II Layer                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌─────────────────────┐  ┌───────────────────────┐  │
│  │ OrderedMessaging   │  │ DetectionValidator  │  │ StatisticalBounds     │  │
│  │ - send_with_ack()  │  │ - Chernoff bounds   │  │ - μ penalty           │  │
│  │ - recv_and_ack()   │  │ - loss validation   │  │ - QBER adjustment     │  │
│  └─────────┬──────────┘  └──────────┬──────────┘  └───────────┬───────────┘  │
│            │                        │                         │              │
│  ┌─────────┴──────────┐  ┌──────────┴──────────┐  ┌───────────┴───────────┐  │
│  │ SiftingProtocol    │  │ DecoyStateAnalyzer  │  │ FiniteSizeAnalyzer    │  │
│  │ - basis matching   │  │ - (optional)        │  │ - sample selection    │  │
│  │ - index filtering  │  │ - yield comparison  │  │ - confidence bounds   │  │
│  └─────────┬──────────┘  └──────────┬──────────┘  └───────────┬───────────┘  │
├────────────┴─────────────────────────┴────────────────────────┴──────────────┤
│                              SquidASM Layer                                   │
│  ┌─────────────────┐  ┌───────────────────┐  ┌─────────────────────────────┐ │
│  │ ClassicalSocket │  │ EPRSocket         │  │ Qubit                       │ │
│  │ - send/recv     │  │ - create_measure  │  │ - measure(basis=X/Z)       │ │
│  │                 │  │ - RandomBasis.XZ  │  │ - basis_rotations          │ │
│  └─────────────────┘  └───────────────────┘  └─────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4. Files to Create/Modify for Phase II

| File | Purpose | Status |
|------|---------|--------|
| `ehok/protocols/ordered_messaging.py` | Strict message ordering with ACKs | TO CREATE |
| `ehok/quantum/detection.py` | Detection event handling and validation | TO CREATE |
| `ehok/analysis/statistics.py` | Finite-size bounds, μ penalty | TO CREATE |
| `ehok/quantum/sifting.py` | Basis sifting protocol | TO CREATE |
| `ehok/quantum/decoy_states.py` | (Optional) Decoy state support | TO CREATE |

---

## Phase III: Information Reconciliation

### 1. Requirements Summary

Phase III performs error correction while minimizing information leakage:

| Requirement | Description | Criticality |
|-------------|-------------|-------------|
| **One-Way LDPC** | Unidirectional syndrome-based FEC (no Cascade) | CRITICAL |
| **Wiretap Cost Tracking** | Count syndrome bits as leaked information | CRITICAL |
| **Safety Cap Enforcement** | Hard limit on total leakage $L_{max}$ | HIGH |
| **Adaptive Rate Selection** | Select code rate based on estimated QBER | HIGH |
| **Interactive Hashing (Optional)** | Advanced mode for reduced leakage | MEDIUM |

### 2. SquidASM Native Support Assessment

#### 2.1 Classical Data Exchange for Syndromes
**Status: ✅ NATIVELY SUPPORTED**

SquidASM's classical sockets can transmit arbitrary data:

```python
# Binary data can be encoded and sent
syndrome_hex = syndrome.tobytes().hex()
context.csockets["peer"].send(syndrome_hex)
```

**No special extensions needed** for syndrome transmission.

#### 2.2 LDPC Reconciliation
**Status: ✅ ALREADY IMPLEMENTED IN E-HOK**

The ehok project provides a comprehensive LDPC implementation:

```python
# From ehok/implementations/reconciliation/ldpc_reconciliator.py
class LDPCReconciliator(IReconciliator):
    """Block-based LDPC reconciliator with rate adaptation and integrated QBER estimation."""
    
    def select_rate(self, qber_est: float) -> float:
        """Select appropriate LDPC code rate based on QBER estimate."""
        
    def reconcile_block(self, key_block, syndrome, rate, n_shortened, prng_seed, max_retries=2):
        """Reconcile a single block using LDPC belief-propagation decoding."""
```

**Available Components:**
| Component | File | Purpose |
|-----------|------|---------|
| `LDPCReconciliator` | `ldpc_reconciliator.py` | Main orchestrator |
| `LDPCBeliefPropagation` | `ldpc_bp_decoder.py` | BP decoding algorithm |
| `LDPCMatrixManager` | `ldpc_matrix_manager.py` | Parity-check matrices |
| `PolynomialHashVerifier` | `polynomial_hash.py` | Block verification |
| `IntegratedQBEREstimator` | `qber_estimator.py` | QBER from corrections |

**Interface Design (ehok/interfaces/reconciliation.py):**
```python
class IReconciliator(ABC):
    @abstractmethod
    def select_rate(self, qber_est: float) -> float: ...
    
    @abstractmethod
    def compute_shortening(self, rate, qber_est, target_payload) -> int: ...
    
    @abstractmethod
    def reconcile_block(self, key_block, syndrome, rate, n_shortened, prng_seed) -> Tuple: ...
    
    @abstractmethod
    def estimate_leakage_block(self, syndrome_length, hash_bits=50) -> int: ...
```

#### 2.3 Leakage Tracking (Wiretap Cost)
**Status: ✅ ALREADY IMPLEMENTED IN E-HOK**

The ehok interfaces include leakage estimation:

```python
# From ehok/interfaces/reconciliation.py:142
def estimate_leakage_block(self, syndrome_length: int, hash_bits: int = 50) -> int:
    """Estimate information leakage for a single reconciled block."""
```

**And privacy amplification accounts for it:**
```python
# From ehok/interfaces/privacy_amplification.py:117
def compute_output_length(self, n: int, qber: float, leakage: float, epsilon: float) -> int:
    """
    Compute secure output length:
    m ≤ n · [1 - h(qber)] - leakage - 2log₂(1/ε)
    """
```

#### 2.4 Safety Cap Enforcement
**Status: ⚠️ PARTIALLY IMPLEMENTED - NEEDS EXTENSION**

**What's Available:**
- Leakage accumulation per block
- QBER-based abort conditions

**What's Missing:**
- Hard limit $L_{max}$ calculation from min-entropy bounds
- Per-block leakage cap enforcement
- Abort on exceeding cumulative leakage threshold

**Proposed Extension:**
```python
# Extension: ehok/core/security_bounds.py
class LeakageSafetyManager:
    """Enforces hard limits on information leakage during reconciliation."""
    
    def __init__(self, min_entropy: float, target_key_length: int, epsilon_sec: float):
        self.h_min = min_entropy
        self.l_target = target_key_length
        self.epsilon = epsilon_sec
        self.accumulated_leakage = 0.0
        
    @property
    def max_leakage(self) -> float:
        """Calculate L_max = H_min(X|E) - l_target - security_margin."""
        security_margin = 2 * math.log2(1 / self.epsilon)
        return self.h_min - self.l_target - security_margin
    
    def add_leakage(self, syndrome_bits: int, hash_bits: int = 50) -> bool:
        """Track leakage. Returns False if limit exceeded."""
        self.accumulated_leakage += syndrome_bits + hash_bits
        return self.accumulated_leakage <= self.max_leakage
    
    def should_abort(self) -> bool:
        return self.accumulated_leakage > self.max_leakage
```

#### 2.5 One-Way Constraint Enforcement
**Status: ✅ ENFORCED BY DESIGN**

The existing LDPC implementation is inherently one-way:

```python
# Bob's reconciliation flow (from ehok/protocols/bob.py)
def _phase4_reconciliation(self, bob_key):
    # 1. Receive syndrome from Alice (one-way)
    syndrome_data = yield from self.context.csockets["alice"].recv()
    
    # 2. Decode locally using syndrome (no back-channel)
    block_result = self.reconciliator.reconcile_block(
        bob_key, syndrome, rate, n_shortened, prng_seed
    )
    
    # 3. No per-block feedback to Alice
    return block_result
```

**Cascade is NOT implemented** - this is correct for E-HOK security.

#### 2.6 Interactive Hashing (Advanced)
**Status: ❌ NOT IMPLEMENTED - FUTURE EXTENSION**

Interactive Hashing (as described in König et al.) requires:
- Multiple rounds of hash commitment/reveal
- Bob commits to set $I_0$ without fully revealing it
- Alice learns two possible sets $\{W_0, W_1\}$

**This is complex and recommended for Phase 2 of implementation.**

### 3. SquidASM/NetSquid Integration Points

For Phase III, the integration is primarily at the **classical channel level**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          E-HOK Phase III Layer                              │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌───────────────────────┐  ┌──────────────────┐ │
│  │ LDPCReconciliator    │  │ LeakageSafetyManager  │  │ (Future:         │ │
│  │ - select_rate()      │  │ - max_leakage         │  │  InteractiveHash)│ │
│  │ - reconcile_block()  │  │ - add_leakage()       │  │                  │ │
│  │ - estimate_leakage() │  │ - should_abort()      │  │                  │ │
│  └──────────┬───────────┘  └───────────┬───────────┘  └──────────────────┘ │
│             │                          │                                    │
│  ┌──────────┴──────────────────────────┴────────────────────────────────┐  │
│  │                         SyndromeTransport                             │  │
│  │  - serialize syndrome → hex string                                    │  │
│  │  - deserialize on receiver                                            │  │
│  │  - track bytes transferred for leakage accounting                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                              SquidASM Layer                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                        ClassicalSocket                                  ││
│  │  - send(syndrome_hex)                                                   ││
│  │  - recv() → syndrome_hex                                                ││
│  │  - No timing guarantees (adequate for reconciliation)                   ││
│  └────────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────────┘
```

### 4. Files to Create/Modify for Phase III

| File | Purpose | Status |
|------|---------|--------|
| `ehok/core/security_bounds.py` | Leakage safety cap, $L_{max}$ calculation | TO CREATE |
| `ehok/implementations/reconciliation/ldpc_reconciliator.py` | Add safety cap integration | TO MODIFY |
| `ehok/protocols/alice.py` | Enforce abort on leakage overflow | TO MODIFY |
| `ehok/quantum/interactive_hashing.py` | (Future) Advanced reconciliation mode | FUTURE |

### 5. Key Insight: SquidASM is Sufficient for Phase III

Unlike Phases I-II, Phase III operates **purely classically** after quantum measurements. The SquidASM classical socket infrastructure is adequate. The main work is in the **ehok layer** for:
- Security bound calculations
- Leakage management
- Protocol flow enforcement

---

## Phase IV: Privacy Amplification

### 1. Requirements Summary

Phase IV performs final key distillation with E-HOK-specific security bounds:

| Requirement | Description | Criticality |
|-------------|-------------|-------------|
| **"Max Bound" Calculator** | Implement Lupo et al. dual-bound entropy estimation | CRITICAL |
| **Finite-Key Effects** | Statistical penalty Δ with $1/\sqrt{N}$ scaling | CRITICAL |
| **Feasibility Check** | Pre-check batch size to avoid zero-length keys | HIGH |
| **Toeplitz Hashing** | 2-universal hashing with cryptographic seed | HIGH |
| **Oblivious Output Formatting** | Separate $S_0$, $S_1$ for Alice; $S_C$ for Bob | CRITICAL |

### 2. SquidASM Native Support Assessment

#### 2.1 Toeplitz Hashing
**Status: ✅ ALREADY IMPLEMENTED IN E-HOK**

The ehok project provides comprehensive Toeplitz-based PA:

```python
# From ehok/implementations/privacy_amplification/toeplitz_amplifier.py
class ToeplitzAmplifier(IPrivacyAmplifier):
    def generate_hash_seed(self, input_length: int, output_length: int) -> np.ndarray:
        """Generate cryptographically secure random seed for Toeplitz matrix."""
        # Uses secrets.token_bytes() for cryptographic randomness
    
    def compress(self, key: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Apply Toeplitz matrix multiplication."""
```

**Features:**
- Cryptographic seed generation via `secrets.token_bytes()`
- O(mn) direct and O(n log n) FFT-based compression
- Proper 2-universal hashing for leftover hash lemma

#### 2.2 Finite-Key Security Bounds
**Status: ✅ ALREADY IMPLEMENTED IN E-HOK**

The ehok project implements Tomamichel et al. finite-key analysis:

```python
# From ehok/implementations/privacy_amplification/finite_key.py
def compute_statistical_fluctuation(n: int, k: int, epsilon: float) -> float:
    """
    Compute μ = sqrt((n+k)/(nk) * (k+1)/k) * sqrt(ln(4/ε))
    """

def compute_final_length_finite_key(params: FiniteKeyParams) -> int:
    """
    ℓ ≤ n(1 - h(QBER + μ)) - leak_EC - log2(2/(ε_sec·ε_cor))
    """
```

**Note:** The implementation follows Tomamichel (2012) for QKD. For E-HOK (NSM), the min-entropy bound must be replaced with the Lupo et al. "Max Bound".

#### 2.3 The "Max Bound" Calculator
**Status: ❌ NOT IMPLEMENTED - CRITICAL EXTENSION REQUIRED**

Phase IV requires the E-HOK-specific entropy bound:

$$h_{min} \ge \max \left\{ \Gamma [1 - \log (1 + 3r^2)], 1 - r \right\}$$

**Current State:**
The existing implementation uses standard QKD bounds (`1 - h(QBER)`), which are **not appropriate** for NSM-based E-HOK.

**Proposed Extension:**
```python
# Extension: ehok/analysis/nsm_bounds.py
import math

def dupuis_konig_bound(r: float) -> float:
    """
    Collision entropy bound from Dupuis/König.
    
    Parameters
    ----------
    r : float
        Storage noise parameter (depolarizing strength).
        r = 0: perfect storage (worst case)
        r = 1: complete depolarization (best case)
    
    Returns
    -------
    float
        Min-entropy rate bound Γ[1 - log(1 + 3r²)]
    """
    # Γ is the regularized expression - simplified for depolarizing channel
    return 1 - math.log2(1 + 3 * r**2)

def lupo_virtual_erasure_bound(r: float) -> float:
    """
    Virtual erasure bound from Lupo et al.
    
    Parameters
    ----------
    r : float
        Storage noise parameter.
    
    Returns
    -------
    float
        Min-entropy rate bound (1 - r)
    """
    return 1 - r

def max_bound_entropy(r: float) -> float:
    """
    Compute optimal min-entropy using the "Max Bound" from Lupo et al.
    
    This selects the better bound for the given noise level:
    - For high noise (low r): Dupuis/König is better
    - For low noise (high r): Lupo bound is superior
    
    Parameters
    ----------
    r : float
        Adversary's storage noise parameter.
    
    Returns
    -------
    float
        Maximum min-entropy rate h_min
    """
    bound_a = dupuis_konig_bound(r)
    bound_b = lupo_virtual_erasure_bound(r)
    return max(bound_a, bound_b)

def compute_ehok_final_length(
    n: int,
    storage_noise: float,
    qber: float,
    leakage: float,
    epsilon_sec: float = 1e-9,
    k: Optional[int] = None,
) -> int:
    """
    Compute secure key length for E-HOK (not standard QKD).
    
    Uses NSM security model with Lupo et al. bounds instead of QKD bounds.
    
    Parameters
    ----------
    n : int
        Reconciled key length.
    storage_noise : float
        Adversary's storage noise parameter r.
    qber : float
        Measured QBER on test set.
    leakage : float
        Total reconciliation leakage (syndrome + hash).
    epsilon_sec : float
        Target security parameter.
    k : int, optional
        Test sample size. If None, uses 10% of n.
    
    Returns
    -------
    int
        Maximum secure key length for E-HOK OT.
    """
    if k is None:
        k = max(100, n // 10)
    
    # Statistical fluctuation for QBER estimation
    epsilon_pe = math.sqrt(epsilon_sec)
    mu = compute_statistical_fluctuation(n, k, epsilon_pe)
    qber_effective = min(qber + mu, 0.5)
    
    # NSM min-entropy bound (NOT QKD bound!)
    h_min_rate = max_bound_entropy(storage_noise)
    
    # Total extractable entropy
    extractable = n * h_min_rate
    
    # Subtract losses
    security_cost = 2 * math.log2(1 / epsilon_sec)
    final_length = extractable - leakage - security_cost - n * binary_entropy(qber_effective)
    
    return max(0, int(final_length))
```

#### 2.4 Feasibility Check
**Status: ⚠️ PARTIALLY IMPLEMENTED - NEEDS NSM INTEGRATION**

**Current State:**
The ehok project checks for zero-length keys after calculation but doesn't provide a **pre-flight** feasibility check.

**Gap:** 
- No `CheckFeasibility(N, QBER, epsilon_target)` function
- No batch size recommendations
- No "accumulate more photons" warnings

**Proposed Extension:**
```python
# Extension: ehok/core/feasibility.py
def check_batch_feasibility(
    n: int,
    k: int,
    expected_qber: float,
    storage_noise: float,
    epsilon_sec: float = 1e-9,
) -> tuple[bool, str, int]:
    """
    Pre-flight check: will this batch produce a positive key?
    
    Returns
    -------
    (feasible, message, recommended_n)
        - feasible: True if positive key length expected
        - message: Diagnostic information
        - recommended_n: Minimum n for positive key, or 0 if already feasible
    """
    # Estimate leakage based on expected QBER
    estimated_leakage = estimate_reconciliation_leakage(n, expected_qber)
    
    # Calculate expected key length
    expected_length = compute_ehok_final_length(
        n, storage_noise, expected_qber, estimated_leakage, epsilon_sec, k
    )
    
    if expected_length > 0:
        return True, f"Expected secure key: {expected_length} bits", 0
    
    # Find minimum batch size
    for test_n in [n * 2, n * 5, n * 10, 10**6, 10**7]:
        test_leakage = estimate_reconciliation_leakage(test_n, expected_qber)
        test_length = compute_ehok_final_length(
            test_n, storage_noise, expected_qber, test_leakage, epsilon_sec, test_n // 10
        )
        if test_length > 0:
            return False, f"Batch too small. Need ≥{test_n} bits", test_n
    
    return False, "Infeasible: QBER too high or storage noise too low", 0
```

#### 2.5 Oblivious Output Formatting
**Status: ❌ NOT IMPLEMENTED - CRITICAL FOR E-HOK**

The current implementation produces a **single** final key. E-HOK requires **structured oblivious output**:

- **Alice** produces: $(S_0, S_1)$ — two keys
- **Bob** produces: $(S_C, C)$ — one key + choice bit

**Gap Analysis:**
The existing `compress()` method doesn't support:
- Separate hashing of $I_0$ vs $I_1$ subsets
- Choice-bit-indexed key selection
- Oblivious key packaging

**Proposed Extension:**
```python
# Extension: ehok/core/oblivious_key.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class AliceObliviousKey:
    """Alice's output: two candidate keys."""
    s0: np.ndarray  # Key for choice bit = 0
    s1: np.ndarray  # Key for choice bit = 1
    seed: np.ndarray  # Toeplitz seed used (shared with Bob)

@dataclass
class BobObliviousKey:
    """Bob's output: one key determined by his measurements."""
    s_c: np.ndarray  # Key corresponding to Bob's actual measurements
    c: int  # Implicit choice bit (not directly known to Alice)
    # Note: Bob doesn't learn which "official" C this corresponds to

class ObliviousKeyFormatter:
    """Formats reconciled key into E-HOK oblivious transfer output."""
    
    def __init__(self, amplifier: IPrivacyAmplifier):
        self.amplifier = amplifier
    
    def format_alice(
        self,
        reconciled_key: np.ndarray,
        bases_alice: np.ndarray,
        I_0: np.ndarray,  # Matching basis indices
        I_1: np.ndarray,  # Non-matching basis indices
        output_length: int,
    ) -> AliceObliviousKey:
        """
        Generate Alice's two oblivious keys.
        
        Alice can compute BOTH S_0 and S_1 because she knows her bases.
        """
        seed = self.amplifier.generate_hash_seed(len(reconciled_key), output_length)
        
        # Note: In practice, this is more nuanced - the reconciled key
        # corresponds to sifted bits. The oblivious structure comes from
        # how we interpret the raw transmission.
        
        # For simplicity: hash different derived values for S_0, S_1
        s0 = self.amplifier.compress(reconciled_key, seed)
        
        # S_1 uses a different derivation (implementation-specific)
        # This is a placeholder - real implementation requires careful analysis
        s1_seed = np.roll(seed, 1)  # Different seed component
        s1 = self.amplifier.compress(reconciled_key, s1_seed)
        
        return AliceObliviousKey(s0=s0, s1=s1, seed=seed)
    
    def format_bob(
        self,
        reconciled_key: np.ndarray,
        seed: np.ndarray,
        choice_implicit: int,
    ) -> BobObliviousKey:
        """
        Generate Bob's single oblivious key.
        
        Bob can only compute ONE of {S_0, S_1} because he only measured
        in one basis configuration.
        """
        s_c = self.amplifier.compress(reconciled_key, seed)
        return BobObliviousKey(s_c=s_c, c=choice_implicit)
```

### 3. SquidASM/NetSquid Integration Points

Phase IV is primarily **classical post-processing** with one quantum-related input:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           E-HOK Phase IV Layer                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │ NSMBoundsCalculator │  │ FeasibilityChecker  │  │ ObliviousKeyFormatter │ │
│  │ - max_bound()       │  │ - batch_size_check  │  │ - format_alice()      │ │
│  │ - dupuis_konig()    │  │ - recommended_n     │  │ - format_bob()        │ │
│  │ - lupo_bound()      │  │                     │  │                       │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └───────────┬───────────┘ │
│             │                        │                         │             │
│  ┌──────────┴────────────────────────┴─────────────────────────┴───────────┐ │
│  │                        ToeplitzAmplifier (existing)                      │ │
│  │  - generate_hash_seed()                                                  │ │
│  │  - compress() [O(n log n) FFT available]                                 │ │
│  │  - compute_final_length()                                                │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────────┤
│                            SquidASM Interface                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        ClassicalSocket                                  │  │
│  │  - send(seed.hex())  [Share Toeplitz seed]                              │  │
│  │  - recv() → seed                                                        │  │
│  │  - No quantum operations in Phase IV                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The NSM storage noise parameter ($r$) is **not available from SquidASM**.

This is a **simulation parameter** that must be:
1. Set externally (adversary model assumption)
2. Propagated through configuration
3. Used in security bound calculations

**NetSquid Connection:**
```python
# The adversary's storage noise can be modeled as T1/T2 decay
# Extension needed to translate NetSquid memory parameters → NSM r value

def estimate_storage_noise_from_netsquid(
    t1_ns: float,  # T1 time in nanoseconds
    t2_ns: float,  # T2 time in nanoseconds
    delta_t_ns: float,  # Wait time between phases
) -> float:
    """
    Estimate adversary's effective storage noise r from NetSquid memory model.
    
    Uses T1T2 noise model to estimate depolarizing strength after wait time.
    """
    # Simplified model: exponential decay
    decay_amplitude = math.exp(-delta_t_ns / t1_ns) if t1_ns > 0 else 0
    decay_phase = math.exp(-delta_t_ns / t2_ns) if t2_ns > 0 else 0
    
    # Convert to depolarizing parameter (approximation)
    fidelity = 0.5 * (1 + decay_amplitude * decay_phase)
    r = 1 - fidelity  # r=0: perfect storage, r=1: full depolarization
    
    return r
```

### 4. Files to Create/Modify for Phase IV

| File | Purpose | Status |
|------|---------|--------|
| `ehok/analysis/nsm_bounds.py` | Lupo et al. "Max Bound" implementation | TO CREATE |
| `ehok/core/feasibility.py` | Batch size feasibility checker | TO CREATE |
| `ehok/core/oblivious_key.py` | Oblivious output formatting | TO CREATE |
| `ehok/quantum/noise_adapter.py` | NetSquid → NSM parameter conversion | TO CREATE |
| `ehok/implementations/privacy_amplification/toeplitz_amplifier.py` | Integrate NSM bounds | TO MODIFY |

---

## Summary: Cross-Phase Integration Architecture

### Overall Assessment Matrix

| Phase | Requirement | SquidASM Native | NetQASM | NetSquid | Extension Needed |
|-------|-------------|-----------------|---------|----------|------------------|
| **I** | EPR Generation | ✅ | ✅ | ✅ | No |
| **I** | Noise Models (T1/T2, Depolarizing) | ⚠️ | ✅ | ✅ | Parameter adapter |
| **I** | NSM Parameters (μ, η, e_det) | ❌ | ❌ | ⚠️ | Custom model |
| **I** | Wait Time Enforcement (Δt) | ❌ | ❌ | ⚠️ | Timing primitives |
| **I** | Pre-Flight Feasibility | ❌ | ❌ | ❌ | New module |
| **II** | Classical Communication | ✅ | ✅ | ✅ | No |
| **II** | Basis Sifting | ✅ | ✅ | ✅ | No |
| **II** | Ordered Message Acknowledgment | ❌ | ❌ | ❌ | Protocol layer |
| **II** | Missing Rounds Validation | ❌ | ❌ | ❌ | New module |
| **II** | Statistical Penalty (μ) | ❌ | ❌ | ❌ | Already in ehok |
| **III** | LDPC Reconciliation | ✅ ehok | N/A | N/A | No |
| **III** | Leakage Tracking | ✅ ehok | N/A | N/A | Minor enhancement |
| **III** | Safety Cap Enforcement | ⚠️ ehok | N/A | N/A | Add $L_{max}$ check |
| **IV** | Toeplitz Hashing | ✅ ehok | N/A | N/A | No |
| **IV** | Finite-Key Bounds | ✅ ehok | N/A | N/A | Replace with NSM |
| **IV** | NSM "Max Bound" | ❌ | ❌ | ❌ | Critical extension |
| **IV** | Oblivious Output Format | ❌ | ❌ | ❌ | New module |

### Legend:
- ✅ = Fully supported natively
- ⚠️ = Partially supported, minor adaptation needed
- ❌ = Not supported, extension required
- N/A = Not applicable (pure classical processing)

---

## Critical Path: Required Extensions

### Priority 1: Security-Critical Extensions

These extensions are **required** for E-HOK security guarantees:

#### 1.1 NSM Security Bounds (`ehok/analysis/nsm_bounds.py`)
**Urgency: CRITICAL**

The existing finite-key analysis uses QKD bounds, not NSM bounds. Without the Lupo et al. "Max Bound", the security claims are invalid for E-HOK.

```python
# Key functions needed:
def max_bound_entropy(storage_noise: float) -> float: ...
def compute_ehok_final_length(n, storage_noise, qber, leakage, epsilon_sec) -> int: ...
```

#### 1.2 Timing Enforcement (`ehok/quantum/timing.py`)
**Urgency: CRITICAL**

The NSM security model hinges on the wait time Δt causing adversary storage decoherence. Without timing enforcement, a cheating Bob can measure immediately.

```python
# Key functions needed:
class TimedProtocolEnforcer:
    def enforce_wait(self, delta_t_ns: float) -> Generator: ...
    def verify_order(self, events: List[str]) -> bool: ...
```

#### 1.3 Ordered Protocol Messaging (`ehok/protocols/ordered_messaging.py`)
**Urgency: HIGH**

The "Sandwich" protocol flow (Missing Rounds → Δt → Bases) must be strictly enforced with acknowledgments.

```python
# Key functions needed:
class OrderedProtocolSocket:
    def send_with_ack(self, msg: str, timeout_ns: float) -> Generator: ...
    def recv_and_ack(self) -> Generator: ...
```

### Priority 2: Functionality Extensions

These extensions enable full E-HOK functionality:

#### 2.1 NSM Parameter Adapter (`ehok/quantum/noise_adapter.py`)

Converts between SquidASM/NetSquid configuration and NSM-compatible parameters.

```python
# Key functions needed:
def create_nsm_channel_config(
    source_quality: float,
    detection_efficiency: float,
    intrinsic_error: float,
) -> NetworkConfig: ...

def estimate_storage_noise_from_netsquid(t1, t2, delta_t) -> float: ...
```

#### 2.2 Detection Validation (`ehok/quantum/detection.py`)

Validates Bob's missing rounds reports against expected channel transmittance.

```python
# Key functions needed:
class DetectionValidator:
    def validate(self, report: DetectionReport) -> tuple[bool, str]: ...
    def chernoff_bounds(self, n: int, p: float, epsilon: float) -> tuple[float, float]: ...
```

#### 2.3 Oblivious Key Formatter (`ehok/core/oblivious_key.py`)

Structures the output for 1-out-of-2 OT applications.

```python
# Key functions needed:
class ObliviousKeyFormatter:
    def format_alice(self, ...) -> AliceObliviousKey: ...
    def format_bob(self, ...) -> BobObliviousKey: ...
```

#### 2.4 Feasibility Checker (`ehok/core/feasibility.py`)

Pre-flight validation that a secure key is achievable with given parameters.

```python
# Key functions needed:
def pre_flight_check(source_quality, detection_efficiency, intrinsic_error, storage_noise) -> tuple[bool, str]: ...
def check_batch_feasibility(n, k, expected_qber, storage_noise) -> tuple[bool, str, int]: ...
```

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    E-HOK Application Layer                                        │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐ │
│  │   AliceProtocol     │  │    BobProtocol      │  │  FeasibilityChecker │  │ ObliviousFormatter│ │
│  │   (ehok/protocols)  │  │   (ehok/protocols)  │  │  (ehok/core)        │  │  (ehok/core)      │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  └─────────┬────────┘ │
│             │                        │                        │                       │          │
├─────────────┴────────────────────────┴────────────────────────┴───────────────────────┴──────────┤
│                                    E-HOK Core Layer                                               │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐ │
│  │  TimingEnforcer     │  │  OrderedMessaging   │  │  DetectionValidator │  │ LeakageSafetyMgr │ │
│  │  (NEW)              │  │  (NEW)              │  │  (NEW)              │  │  (NEW)           │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  └─────────┬────────┘ │
│             │                        │                        │                       │          │
│  ┌──────────┴──────────────────────────────────────────────────────────────────────────┴────────┐│
│  │                            NSM Security Analysis (NEW)                                        ││
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────────────────────────┐││
│  │  │  NSMBoundsCalculator│  │  NoiseParameterAdapter│  │  FiniteKeyAnalyzer (MODIFIED)        │││
│  │  │  - max_bound()      │  │  - μ, η, e_det → NQ  │  │  - Use NSM bounds                    │││
│  │  │  - dupuis_konig()   │  │  - T1/T2 → r         │  │  - Storage noise integration         │││
│  │  └─────────────────────┘  └─────────────────────┘  └────────────────────────────────────────┘││
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                  E-HOK Implementation Layer                                       │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────────────────────┐│
│  │  Reconciliation (existing)      │  │  Privacy Amplification (existing + MODIFIED)            ││
│  │  - LDPCReconciliator            │  │  - ToeplitzAmplifier                                     ││
│  │  - LDPCBeliefPropagation        │  │  - compute_ehok_final_length() (NEW)                    ││
│  └─────────────────────────────────┘  └─────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                     SquidASM Layer                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────────┐│
│  │  │  Program         │  │  ProgramContext  │  │  EPRSocket        │  │  ClassicalSocket        ││
│  │  │  - run()         │  │  - connection    │  │  - create_keep()  │  │  - send()/recv()        ││
│  │  │  - meta          │  │  - csockets      │  │  - recv_keep()    │  │  - Generator-based      ││
│  │  └──────────────────┘  │  - epr_sockets   │  │  - create_measure │  └─────────────────────────┘│
│  │                        └──────────────────┘  └───────────────────┘                             │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                      NetQASM Layer                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────────┐│
│  │  │  NetworkConfig   │  │  Link            │  │  Node             │  │  Qubit                  ││
│  │  │  - nodes[]       │  │  - fidelity      │  │  - qubits[]       │  │  - measure(basis)       ││
│  │  │  - links[]       │  │  - noise_type    │  │  - T1, T2         │  │  - basis_rotations      ││
│  │  └──────────────────┘  └──────────────────┘  │  - gate_fidelity  │  └─────────────────────────┘│
│  │                                              └───────────────────┘                             │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                      NetSquid Layer                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  │  ┌──────────────────┐  ┌────────────────────────┐  ┌────────────────────────────────────────┐ │
│  │  │  MagicDistributor│  │  Quantum Error Models  │  │  Discrete Event Simulation             │ │
│  │  │  - state_delay   │  │  - DepolarNoiseModel   │  │  - ns.sim_time()                       │ │
│  │  │  - label_delay   │  │  - T1T2NoiseModel      │  │  - Event scheduling                    │ │
│  │  │  - delivery      │  │  - FibreLossModel      │  │  - _schedule_after()                   │ │
│  │  └──────────────────┘  └────────────────────────┘  └────────────────────────────────────────┘ │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Implementation Roadmap

### Sprint 1: Security Foundation (Week 1-2)
1. `ehok/analysis/nsm_bounds.py` - Lupo et al. "Max Bound"
2. `ehok/quantum/timing.py` - Wait time enforcement primitives
3. `ehok/core/feasibility.py` - Pre-flight security checks

### Sprint 2: Protocol Layer (Week 3-4)
4. `ehok/protocols/ordered_messaging.py` - ACK-based message ordering
5. `ehok/quantum/detection.py` - Missing rounds validation
6. `ehok/quantum/noise_adapter.py` - NSM ↔ NetSquid parameter translation

### Sprint 3: Output & Integration (Week 5-6)
7. `ehok/core/oblivious_key.py` - 1-out-of-2 OT output formatting
8. `ehok/core/security_bounds.py` - Leakage safety cap enforcement
9. Modify `ehok/implementations/privacy_amplification/toeplitz_amplifier.py` - NSM integration
10. Integration testing with full E-HOK protocol flow

---

## Conclusion

**SquidASM provides an excellent foundation** for E-HOK implementation with:
- Full EPR generation and measurement support
- Comprehensive noise model infrastructure
- Classical communication primitives
- Discrete event simulation for timing

**The critical gaps are in the E-HOK-specific security layer:**
1. **NSM security bounds** (currently using QKD bounds - incorrect for E-HOK)
2. **Timing enforcement** (Δt wait time is the security cornerstone)
3. **Oblivious output structure** (1-out-of-2 OT requires special formatting)

The existing ehok codebase provides excellent infrastructure for reconciliation and privacy amplification, but these must be **adapted** to use NSM-specific entropy bounds rather than standard QKD bounds.

**Estimated effort:** 
- Core security extensions: 2-3 weeks
- Protocol layer: 1-2 weeks
- Integration and testing: 2 weeks
- **Total: ~6 weeks** for a complete E-HOK-compatible implementation




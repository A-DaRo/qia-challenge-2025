# System Test Specification: E-HOK on SquidASM Integration Validation

> **Document ID**: STS-001 / E-HOK-on-SquidASM  
> **Classification**: Lead QA Engineer & Systems Integration Architect  
> **Scope**: Final validation gate for E-HOK NSM migration  
> **Version**: 1.0.0  
> **Last Updated**: 2025-12-13

---

## 1. Executive Summary

### 1.1 Scope

This document specifies the **System Test Specification** for validating the E-HOK (Entanglement-based Honest Oblivious Key) protocol implementation on the SquidASM v0.x simulation framework. The specification targets the integration seams between:

- **Application Layer**: `qia-challenge-2025/ehok/*` (custom E-HOK implementation)
- **Simulation Framework**: `squidasm/squidasm/*` (SquidASM application layer)
- **Quantum SDK**: `qia/lib/python3.10/site-packages/netqasm/*` (instruction set layer)
- **Physical Simulator**: `qia/lib/python3.10/site-packages/netsquid*/*` (discrete-event simulation engine)

### 1.2 Objective

To certify that:

1. **NSM Compliance**: Security calculations use the NSM "Max Bound" ($h_{min}(r) = \max\{\Gamma[1-\log_2(1+3r^2)], 1-r\}$), not QKD entropy formulas
2. **Platform Integration**: Custom `ehok` extensions correctly interact with NetSquid simulation state
3. **Failure Mode Correctness**: The system fails safely and deterministically under invalid/adversarial conditions
4. **Oblivious Output Structure**: Final outputs conform to 1-out-of-2 OT semantics: Alice $(S_0, S_1)$, Bob $(S_C, C)$

### 1.3 Prerequisites

| Prerequisite | Description | Validation |
|--------------|-------------|------------|
| Sprint 0 Complete | CI scaffold, dataclass contracts, logging infrastructure | Green CI pipeline |
| Sprint 1 Complete | `NSMBoundsCalculator`, `TimingEnforcer`, `FeasibilityChecker` | Unit tests pass |
| Sprint 2 Complete | `OrderedProtocolSocket`, `DetectionValidator`, `LeakageSafetyManager` | Integration tests pass |
| Sprint 3 Complete | `ObliviousKeyFormatter`, E2E pipeline, validation suite | E2E tests produce valid keys |
| Legacy Removal Complete | No QKD-bound codepaths remain; all security uses NSM bounds | Static analysis confirms |

### 1.4 Non-Goals

- **Performance optimization**: This specification validates correctness, not throughput
- **Adversarial cryptanalysis**: We assume the NSM security proofs hold; we validate correct implementation
- **Hardware-in-the-loop**: All tests execute in simulation

---

## 2. Integration Point Validation (White-Box Extension Tests)

This section specifies **white-box** tests that inspect the underlying simulation state to verify that `ehok` extensions correctly manipulate the SquidASM/NetSquid stack.

### 2.1 Noise Adapter Integration (GAP: NOISE-PARAMS-001)

**Requirement Trace**:
- `squid_assesment.md` §2.2: "Source Quality (μ), Detection Efficiency (η) — Not configurable at SquidASM level"
- `sprint_1_specification.md` §3.3: NOISE-PARAMS-001 task
- `phase_I_analysis.md` §4.2.2: NetSquid noise model mapping

#### Test Case: SYS-INT-NOISE-001 — PhysicalModelAdapter Fidelity Mapping

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-NOISE-001` |
| **Title** | Verify PhysicalModelAdapter correctly translates NSM parameters to NetSquid Link fidelity |
| **Priority** | CRITICAL |
| **Traces To** | GAP: NOISE-PARAMS-001, REQ: PHI-R1 (Pre-flight feasibility) |

**Goal**: Confirm that when `ehok/quantum/noise_adapter.py` is configured with NSM parameters ($\mu$, $\eta$, $e_{det}$), the resulting `netsquid.components.qchannel.QuantumChannel` properties match the calculated theoretical fidelity.

**Pre-Conditions**:
1. E-HOK protocol configuration specifies NSM parameters:
   - Source quality $\mu = 3.145 \times 10^{-5}$
   - Detection efficiency $\eta = 0.0150$
   - Intrinsic error rate $e_{det} = 0.0093$
2. Network is instantiated via SquidASM `StackNetworkConfig`

**Test Logic**:
```
1. Configure PhysicalModelAdapter with NSM parameters
2. Create SquidASM network using adapter's output configuration
3. INSPECT: Retrieve underlying netsquid.components.qchannel.QuantumChannel from the Link
4. INSPECT: Query QuantumChannel.models to obtain noise model parameters
5. COMPUTE: Calculate expected fidelity from NSM formula:
   F_expected = 1 - (μ + (1-η)·0.5 + e_det)
6. ASSERT: Configured noise parameters match expected fidelity within tolerance 1e-6
```

**Expected System State**:
- `QuantumChannel.models` contains a `DepolarNoiseModel` or `LinearDepolariseModelParameters`
- Model's `prob_max_mixed` parameter equals $1 - F_{expected}$

**Inspection Points** (NetSquid Object Access):
```python
# Access pattern for white-box validation
from netsquid.components.qchannel import QuantumChannel

channel = network._get_quantum_channel("Alice", "Bob")  # Implementation-specific accessor
noise_model = channel.models.get("noise_model")
actual_depolar_rate = noise_model.properties["prob_max_mixed"]

assert abs(actual_depolar_rate - (1 - F_expected)) < 1e-6
```

**Failure Criteria**:
- FAIL if `prob_max_mixed` differs from expected by > 1e-6
- FAIL if no noise model is attached to the quantum channel
- FAIL if noise model type is not `DepolarNoiseModel` or compatible

---

#### Test Case: SYS-INT-NOISE-002 — Storage Noise r Derivation from T1/T2

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-NOISE-002` |
| **Title** | Verify adversary storage noise r is correctly derived from NetSquid T1/T2 memory parameters |
| **Priority** | HIGH |
| **Traces To** | GAP: STORAGE-LINK-001, `phase_IV_analysis.md` §3.2 |

**Goal**: Validate that `ehok/quantum/noise_adapter.py` correctly converts NetSquid memory decay parameters to the NSM storage noise parameter $r$.

**Pre-Conditions**:
1. NetSquid memory configured with:
   - $T_1 = 1 \times 10^9$ ns (amplitude damping time)
   - $T_2 = 5 \times 10^8$ ns (dephasing time)
2. Wait time $\Delta t = 1 \times 10^9$ ns

**Test Logic**:
```
1. Configure NetSquid node with T1/T2 memory parameters
2. Call noise_adapter.estimate_storage_noise_from_netsquid(T1, T2, delta_t)
3. COMPUTE: Expected storage noise from decay model:
   decay_amp = exp(-Δt / T1)
   decay_phase = exp(-Δt / T2)
   F_storage = 0.5 * (1 + decay_amp * decay_phase)
   r_expected = 1 - F_storage
4. ASSERT: Returned r equals r_expected within tolerance 1e-4
```

**Expected System State**:
- Returned $r \approx 0.565$ for the specified parameters
- $r$ is in valid range $[0, 1]$

---

### 2.2 Timing Enforcer Integration (GAP: TIMING-001)

**Requirement Trace**:
- `squid_assesment.md` §2.3: "Wait Time Enforcement (Δt) — NOT NATIVELY SUPPORTED"
- `sprint_1_specification.md` §3.1: TASK-TIMING-001
- `phase_I_analysis.md` §1.1: König et al. Markovian noise requirement

#### Test Case: SYS-INT-TIMING-001 — Δt Barrier Simulation Time Verification

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-TIMING-001` |
| **Title** | Verify TimingEnforcer interacts correctly with NetSquid simulation time |
| **Priority** | CRITICAL |
| **Traces To** | GAP: TIMING-001, REQ: PHI-R2 (Strict Δt enforcement) |

**Goal**: Assert that `ns.sim_time()` differs exactly by $\Delta t$ between the "Commit ACK Received" event and the "Basis Reveal Sent" event.

**Pre-Conditions**:
1. `TimingEnforcer` configured with $\Delta t = 1 \times 10^9$ ns
2. Protocol execution reaches Phase II sifting

**Test Logic**:
```
1. Register simulation time observers on TimingEnforcer events
2. Execute protocol through Phase I quantum transmission
3. CAPTURE: t_commit = ns.sim_time() when TIMING_COMMIT_ACK_RECEIVED logged
4. Execute timing wait via generator yield
5. CAPTURE: t_reveal = ns.sim_time() when TIMING_BASIS_REVEAL_ALLOWED logged
6. ASSERT: (t_reveal - t_commit) >= Δt
7. ASSERT: (t_reveal - t_commit) < Δt + tolerance (no unnecessary delay)
```

**Expected System State**:
- NetSquid simulation clock advanced by exactly $\Delta t$ during wait
- Log events emitted in order: `TIMING_COMMIT_ACK_RECEIVED` → `TIMING_BASIS_REVEAL_ALLOWED`

**Inspection Points** (NetSquid Direct Access):
```python
import netsquid as ns

# Capture simulation time at key events
t_commit_ack = None
t_basis_reveal = None

# In protocol execution flow:
timing_enforcer.mark_commit_received(sim_time_ns=int(ns.sim_time()))
t_commit_ack = ns.sim_time()

# ... generator yields control for Δt ...

timing_enforcer.mark_basis_reveal_attempt(sim_time_ns=int(ns.sim_time()))
t_basis_reveal = ns.sim_time()

assert t_basis_reveal - t_commit_ack >= delta_t_ns
```

**Failure Criteria**:
- FAIL if basis reveal occurs before $t_{commit} + \Delta t$
- FAIL if `ns.sim_time()` does not advance during wait
- FAIL if timing events are logged out of order

---

#### Test Case: SYS-INT-TIMING-002 — Premature Basis Reveal Block

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-TIMING-002` |
| **Title** | Verify TimingEnforcer blocks basis reveal if Δt has not elapsed |
| **Priority** | CRITICAL |
| **Traces To** | GAP: TIMING-001, Abort: ABORT-II-TIMING-001 |

**Goal**: Confirm that attempting basis reveal before $\Delta t$ results in a blocked state or raised exception.

**Pre-Conditions**:
1. `TimingEnforcer` configured with $\Delta t = 1 \times 10^9$ ns
2. Commit ACK received at $t_0$

**Test Logic**:
```
1. Mark commit received at t_0
2. Attempt basis reveal at t_0 + (Δt / 2)  # Premature
3. ASSERT: is_basis_reveal_allowed() returns False
4. ASSERT: Log event TIMING_BASIS_REVEAL_BLOCKED emitted
5. Attempt basis reveal at t_0 + Δt + 1  # Valid
6. ASSERT: is_basis_reveal_allowed() returns True
```

**Expected System State**:
- State remains blocked until required time
- No state corruption from premature attempt

---

### 2.3 Ordered Messaging Integration (GAP: ORDERED-MSG-001)

**Requirement Trace**:
- `squid_assesment.md` §2.1: "Ordered Acknowledgments — No built-in mechanism"
- `sprint_2_specification.md` §2.1: TASK-ORDERED-MSG-001
- `phase_II_analysis.md` §3.2: Race condition failure mode

#### Test Case: SYS-INT-MSG-001 — Ordered Socket ACK Blocking

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-MSG-001` |
| **Title** | Verify OrderedProtocolSocket prevents race conditions under network delay |
| **Priority** | CRITICAL |
| **Traces To** | GAP: ORDERED-MSG-001, Security Invariant: Commit-then-Reveal |

**Goal**: Confirm that Alice blocks on `send_with_ack()` until Bob's ACK arrives, even with artificial network delay.

**Pre-Conditions**:
1. Network configured with asymmetric classical delay:
   - Alice→Bob: 100ms
   - Bob→Alice: 500ms (simulates congestion)
2. `OrderedProtocolSocket` wrapping `ClassicalSocket`

**Test Logic**:
```
1. Alice sends DetectionReport via send_with_ack()
2. INJECT: Additional 500ms delay on Bob's ACK response
3. CAPTURE: t_send = time when Alice initiates send
4. CAPTURE: t_ack_received = time when send_with_ack() returns
5. ASSERT: (t_ack_received - t_send) >= 600ms (round-trip delay)
6. ASSERT: Alice's state is SENT_WAIT_ACK during wait
7. ASSERT: Alice does NOT send BasisReveal before ACK received
```

**Expected System State**:
- Alice's generator remains yielded during ACK wait
- Protocol state machine correctly tracks ordering

**Inspection Points**:
```python
# State machine inspection
socket = OrderedProtocolSocket(classical_socket)
socket.send_with_ack("DetectionReport", timeout_ns=10**10)

# During yield:
assert socket.state == OrderedSocketState.SENT_WAIT_ACK

# After ACK received:
assert socket.state == OrderedSocketState.IDLE
```

---

#### Test Case: SYS-INT-MSG-002 — ACK Timeout Triggers Abort

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-MSG-002` |
| **Title** | Verify OrderedProtocolSocket triggers ProtocolViolation on ACK timeout |
| **Priority** | HIGH |
| **Traces To** | GAP: ORDERED-MSG-001, Abort: ABORT-II-ACK-TIMEOUT |

**Goal**: Confirm that missing ACK within timeout causes deterministic abort.

**Pre-Conditions**:
1. Timeout configured at 5 seconds
2. Bob configured to NOT send ACK (simulates malicious/failed node)

**Test Logic**:
```
1. Alice sends message via send_with_ack(timeout_ns=5*10^9)
2. Bob receives message but does NOT send ACK
3. WAIT: 5 seconds simulation time
4. ASSERT: ProtocolViolation exception raised
5. ASSERT: Exception message indicates "ACK timeout"
6. ASSERT: Protocol state transitions to VIOLATION/ABORT
```

**Expected System State**:
- Clean abort with traceable error
- No partial state left behind

---

### 2.4 MagicDistributor Configuration Verification

**Requirement Trace**:
- `squid_assesment.md` §2.1: MagicDistributor noise model configuration
- `phase_I_analysis.md` §4.2.1: EPR generation fidelity mapping

#### Test Case: SYS-INT-MAGIC-001 — EPR Fidelity Matches Configuration

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-INT-MAGIC-001` |
| **Title** | Verify MagicDistributor produces EPR pairs with configured fidelity |
| **Priority** | HIGH |
| **Traces To** | `squid_assesment.md` §2.1, EPR generation |

**Goal**: Confirm that the underlying `netsquid_magic.MagicDistributor` is configured with the correct `prob_max_mixed` parameter derived from `Link.fidelity`.

**Pre-Conditions**:
1. Link configured with `fidelity = 0.95`
2. Network uses `NoiseType.Depolarise`

**Test Logic**:
```
1. Create SquidASM network with specified link fidelity
2. INSPECT: Access underlying MagicDistributor from network internals
3. INSPECT: Query model_params.prob_max_mixed
4. ASSERT: prob_max_mixed == 1 - 0.95 == 0.05
5. Generate 1000 EPR pairs
6. MEASURE: Fidelity of generated states to |Φ+⟩
7. ASSERT: Measured fidelity within [0.94, 0.96] (statistical bounds)
```

**Inspection Points**:
```python
# Access MagicDistributor configuration
distributor = network._link_distributors["Alice-Bob"]
model_params = distributor._state_sampler_factory._model_params
assert model_params.prob_max_mixed == 0.05
```

---

## 3. End-to-End Behavioral Scenarios

This section defines complete protocol execution scenarios testing the integration of all Sprint 1-3 components.

### 3.1 Scenario A: The Golden Run (Happy Path)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-GOLDEN-001` |
| **Title** | Complete protocol execution with valid parameters produces matching oblivious keys |
| **Priority** | CRITICAL |
| **Traces To** | Roadmap §6.3: "Positive key rate at Q=5%, n=10,000" |

**Goal**: Verify end-to-end protocol produces valid oblivious keys under optimal conditions.

**Inputs**:
| Parameter | Value |
|-----------|-------|
| Batch size ($M$) | 20,000 rounds |
| Link fidelity | 0.97 (3% depolarization) |
| Storage noise ($r$) | 0.3 |
| Security parameter ($\varepsilon_{sec}$) | $10^{-6}$ |
| Wait time ($\Delta t$) | $10^9$ ns |
| Expected QBER | ~3% |

**Test Logic**:
```
1. Configure E-HOK protocol with above parameters
2. Execute full protocol: Phase I → II → III → IV
3. CAPTURE: Alice's output (S_0, S_1)
4. CAPTURE: Bob's output (S_C, C)
5. ASSERT: len(S_0) == len(S_1) == len(S_C) > 0
6. ASSERT: S_C == S_0 if C == 0 else S_C == S_1
7. ASSERT: ε_achieved <= ε_sec
8. ASSERT: No ABORT codes triggered
9. VERIFY: Protocol metrics show NSM min-entropy (not QKD)
```

**Expected Outcomes**:
| Metric | Expected Range |
|--------|----------------|
| Final key length ($\ell$) | 1,000 – 5,000 bits |
| Observed QBER | 2% – 5% |
| Reconciliation efficiency | ≥ 95% |
| Blocks failed | < 5% |

**Success Criteria**:
- Keys match according to OT semantics
- No security violations logged
- Protocol completes in bounded simulation time

---

### 3.2 Scenario B: Death Valley (Infeasible Batch Size)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-DEATHVALLEY-001` |
| **Title** | Small batch size triggers pre-flight ABORT before quantum resource consumption |
| **Priority** | CRITICAL |
| **Traces To** | Roadmap §6.4: ABORT-I-FEAS-001, `sprint_1_specification.md` §3.2 |

**Goal**: Verify `FeasibilityChecker` aborts before Phase I quantum transmission when batch size is provably insufficient.

**Inputs**:
| Parameter | Value |
|-----------|-------|
| Batch size ($M$) | 100 rounds (deliberately too small) |
| Expected QBER | 8% |
| Storage noise ($r$) | 0.3 |
| Security parameter | $10^{-6}$ |

**Test Logic**:
```
1. Configure E-HOK with small batch size
2. Invoke pre-flight feasibility check
3. ASSERT: FeasibilityChecker returns INFEASIBLE
4. ASSERT: Abort code == ABORT-I-FEAS-001
5. ASSERT: Recommendation includes minimum viable batch size
6. ASSERT: No EPR pairs were generated (ns.sim_time() == 0 or minimal)
7. ASSERT: No quantum resources consumed
```

**Expected System State**:
- Protocol terminates before `EPRSocket.create_keep()` invoked
- Diagnostic message includes: "Batch Size Too Small"
- Recommended $N_{min}$ provided (e.g., "Need ≥ 50,000 rounds")

---

### 3.3 Scenario C: Active Attack — Order Violation

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-ATTACK-ORDER-001` |
| **Title** | Out-of-order message triggers ProtocolViolation abort |
| **Priority** | CRITICAL |
| **Traces To** | `phase_II_analysis.md` §3.2: Race condition attack, Abort: ABORT-II-ORDER-001 |

**Goal**: Verify that basis reveal before detection report acknowledgment triggers immediate abort.

**Inputs**:
- Standard valid parameters
- **INJECT**: Bob sends BasisRequest before DetectionReport is acknowledged

**Test Logic**:
```
1. Start protocol normally
2. INJECT: At Phase II start, Bob sends message with seq > expected
3. ASSERT: OrderedProtocolSocket detects out-of-order message
4. ASSERT: ProtocolViolation raised
5. ASSERT: Abort code == ABORT-II-ORDER-001 or ABORT-II-ACK-TIMEOUT
6. ASSERT: Protocol state == VIOLATION/ABORT
7. VERIFY: Neither party has usable key material
```

**Expected System State**:
- Clean abort with security invariant preserved
- Log contains: "Protocol ordering violation detected"

---

### 3.4 Scenario D: High QBER (Channel Too Noisy)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-HIGHQBER-001` |
| **Title** | QBER exceeding 22% hard limit triggers security abort |
| **Priority** | CRITICAL |
| **Traces To** | Roadmap §6.4: ABORT-II-QBER-001, `phase_I_analysis.md` §1.3: Lupo 22% limit |

**Goal**: Verify protocol aborts when adjusted QBER exceeds the hard security limit.

**Inputs**:
| Parameter | Value |
|-----------|-------|
| Link fidelity | 0.70 (30% depolarization → ~25% QBER) |
| Batch size | 10,000 |
| Storage noise ($r$) | 0.3 |

**Test Logic**:
```
1. Configure network with high noise (low fidelity)
2. Execute Phase I quantum transmission
3. Execute Phase II sifting and QBER estimation
4. CAPTURE: observed QBER and statistical penalty μ
5. COMPUTE: adjusted_qber = observed + μ
6. ASSERT: adjusted_qber > 0.22
7. ASSERT: Protocol ABORTs with code ABORT-II-QBER-001
8. ASSERT: No key material output
9. VERIFY: Abort message contains "QBER exceeds hard limit"
```

**Decision Tree**:
```
IF adjusted_qber > 0.22:
    → ABORT-II-QBER-001 (Hard limit - security impossible)
ELIF adjusted_qber > 0.11:
    → WARNING logged, continue with reduced key rate
ELSE:
    → Continue normally
```

---

### 3.5 Scenario E: Leakage Cap Exceeded

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-LEAKAGE-001` |
| **Title** | Reconciliation leakage exceeding L_max triggers abort |
| **Priority** | HIGH |
| **Traces To** | Roadmap §6.4: ABORT-III-LEAKAGE-001, `sprint_2_specification.md` §4.2 |

**Goal**: Verify `LeakageSafetyManager` aborts when cumulative syndrome leakage exceeds security bounds.

**Inputs**:
| Parameter | Value |
|-----------|-------|
| Batch size | 5,000 |
| QBER | 15% (high, requiring multiple reconciliation attempts) |
| $L_{max}$ | Derived from NSM bounds |

**Test Logic**:
```
1. Configure with marginal parameters (high QBER, small batch)
2. Execute through Phase III reconciliation
3. INJECT: Force multiple block failures requiring re-transmission
4. TRACK: Cumulative leakage via LeakageSafetyManager
5. ASSERT: When total_leakage > L_max, ABORT triggered
6. ASSERT: Abort code == ABORT-III-LEAKAGE-001
7. ASSERT: No partial key released
```

**Expected System State**:
- `LeakageSafetyManager.should_abort()` returns `True`
- Abort prevents further syndrome transmission
- Security margin preserved

---

### 3.6 Scenario F: Detection Anomaly (Chernoff Violation)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-E2E-DETECTION-001` |
| **Title** | Anomalous detection rate triggers Chernoff validation abort |
| **Priority** | HIGH |
| **Traces To** | `sprint_2_specification.md` §3.1: TASK-DETECT-VALID-001, Abort: ABORT-II-DETECT-001 |

**Goal**: Verify `DetectionValidator` aborts when Bob's reported detection rate violates statistical bounds.

**Inputs**:
| Parameter | Value |
|-----------|-------|
| Expected transmittance ($P_{expected}$) | 0.70 |
| Total rounds ($M$) | 10,000 |
| **INJECT**: Bob reports only 3,000 detections (30% vs expected 70%) |

**Test Logic**:
```
1. Configure expected channel transmittance
2. Execute Phase I
3. INJECT: Bob's DetectionReport claims 30% detection rate
4. Invoke DetectionValidator.validate(report)
5. COMPUTE: Chernoff tolerance ζ = sqrt(ln(2/ε) / 2M)
6. COMPUTE: Acceptance interval [(P-ζ)M, (P+ζ)M]
7. ASSERT: 3000 is outside acceptance interval
8. ASSERT: ABORT triggered with code ABORT-II-DETECT-001
9. ASSERT: Diagnostic contains observed/expected rates
```

**Expected System State**:
- Validation fails: "Detection rate 30% outside bounds [68%, 72%]"
- Protocol aborts before basis reveal (post-selection attack prevented)

---

## 4. Output Artifact Validation

This section specifies tests validating the structure and correctness of protocol outputs.

### 4.1 Oblivious Output Structure Verification

#### Test Case: SYS-OUT-OBLIV-001 — Alice Output Contains Exactly (S_0, S_1)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-OUT-OBLIV-001` |
| **Title** | Verify Alice's output dataclass contains exactly two keys |
| **Priority** | CRITICAL |
| **Traces To** | `sprint_3_specification.md` §4: OBLIV-FORMAT-001 |

**Goal**: Confirm `AliceObliviousKey` structure matches 1-out-of-2 OT specification.

**Test Logic**:
```
1. Execute successful protocol run
2. CAPTURE: alice_output = protocol.get_alice_result()
3. ASSERT: isinstance(alice_output, AliceObliviousKey)
4. ASSERT: hasattr(alice_output, 's0') and hasattr(alice_output, 's1')
5. ASSERT: len(alice_output.s0) == len(alice_output.s1) == ℓ
6. ASSERT: alice_output.s0.dtype == np.uint8
7. ASSERT: alice_output.s1.dtype == np.uint8
8. ASSERT: alice_output.seed is not None
9. ASSERT: Alice does NOT have access to Bob's choice bit C
```

---

#### Test Case: SYS-OUT-OBLIV-002 — Bob Output Contains Exactly (S_C, C)

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-OUT-OBLIV-002` |
| **Title** | Verify Bob's output dataclass contains one key and choice bit |
| **Priority** | CRITICAL |
| **Traces To** | `sprint_3_specification.md` §4: OBLIV-FORMAT-001 |

**Goal**: Confirm `BobObliviousKey` structure matches 1-out-of-2 OT specification.

**Test Logic**:
```
1. Execute successful protocol run
2. CAPTURE: bob_output = protocol.get_bob_result()
3. ASSERT: isinstance(bob_output, BobObliviousKey)
4. ASSERT: hasattr(bob_output, 's_c') and hasattr(bob_output, 'c')
5. ASSERT: bob_output.c in {0, 1}
6. ASSERT: len(bob_output.s_c) == ℓ
7. ASSERT: bob_output does NOT contain s_{1-c}
```

---

#### Test Case: SYS-OUT-OBLIV-003 — Oblivious Property: S_{1-C} Uncorrelated

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-OUT-OBLIV-003` |
| **Title** | Verify Bob cannot derive S_{1-C} from his available information |
| **Priority** | CRITICAL |
| **Traces To** | OT security property |

**Goal**: Statistical test confirming Bob's key $S_C$ is uncorrelated with $S_{1-C}$.

**Test Logic**:
```
1. Execute N=100 successful protocol runs with same parameters
2. For each run, record (S_0, S_1, S_C, C)
3. For runs where C=0: COMPUTE correlation(S_C, S_1)
4. For runs where C=1: COMPUTE correlation(S_C, S_0)
5. ASSERT: Correlation coefficients are within [-0.1, 0.1] (statistical noise)
6. ASSERT: No systematic bias in Bob learning S_{1-C}
```

**Expected Statistical Properties**:
- $\text{Corr}(S_C, S_{1-C}) \approx 0$ (within statistical fluctuation)
- Mutual information $I(S_C; S_{1-C}) \approx 0$

---

### 4.2 NSM Metrics Verification

#### Test Case: SYS-OUT-NSM-001 — Final Output Uses NSM Min-Entropy

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-OUT-NSM-001` |
| **Title** | Verify output metadata reports NSM bounds, not QKD bounds |
| **Priority** | CRITICAL |
| **Traces To** | Roadmap §1.4: "Security bounds use NSM Max Bound" |

**Goal**: Confirm protocol metrics dataclass contains NSM-specific entropy calculation.

**Test Logic**:
```
1. Execute successful protocol with storage_noise r = 0.3
2. CAPTURE: metrics = protocol.get_metrics()
3. ASSERT: hasattr(metrics, 'min_entropy_per_bit')
4. COMPUTE: expected_h_min = max(Γ[1-log₂(1+3r²)], 1-r) ≈ 0.805
5. ASSERT: abs(metrics.min_entropy_per_bit - 0.805) < 0.01
6. ASSERT: metrics does NOT contain field 'qkd_entropy_rate'
7. ASSERT: metrics.storage_noise_assumed == r
```

**Forbidden Patterns**:
- Output must NOT contain `1 - h(QBER)` calculation
- Output must NOT reference "QKD" in entropy field names

---

#### Test Case: SYS-OUT-NSM-002 — Key Length Matches NSM Formula

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-OUT-NSM-002` |
| **Title** | Verify final key length matches NSM secure length formula |
| **Priority** | CRITICAL |
| **Traces To** | `sprint_3_specification.md` §3.4: NSM final length formula |

**Goal**: Confirm output key length equals the NSM-computed bound.

**Test Logic**:
```
1. Execute protocol, capture all intermediate values:
   - n = reconciled key length
   - r = storage noise
   - |Σ| = total leakage
   - ε_sec = security parameter
2. COMPUTE: expected_ℓ = ⌊n·h_min(r) - |Σ| - 2log₂(1/ε_sec) - Δ_finite⌋
3. CAPTURE: actual_ℓ = len(alice_output.s0)
4. ASSERT: actual_ℓ == expected_ℓ (exact match)
5. ASSERT: actual_ℓ <= n·h_min(r) - |Σ| (security constraint satisfied)
```

---

## 5. Performance & Stability Benchmarks

### 5.1 Deterministic Replay

#### Test Case: SYS-PERF-DETERM-001 — Seeded Simulation Reproducibility

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-PERF-DETERM-001` |
| **Title** | Identical seeds produce identical logs and keys |
| **Priority** | HIGH |
| **Traces To** | Roadmap §2.1: "Seeded tests produce identical output" |

**Goal**: Verify simulation determinism for debugging and validation.

**Test Logic**:
```
1. Execute protocol with:
   - NetSquid seed = 42
   - NumPy seed = 42
   - Python random seed = 42
2. CAPTURE: run1_keys, run1_logs, run1_metrics
3. Reset all simulators and RNGs with same seeds
4. Execute identical protocol
5. CAPTURE: run2_keys, run2_logs, run2_metrics
6. ASSERT: run1_keys == run2_keys (byte-for-byte)
7. ASSERT: run1_metrics == run2_metrics
8. ASSERT: Log timestamps and events match
```

**Expected System State**:
- Identical quantum measurement outcomes
- Identical sifting results
- Identical reconciliation corrections
- Identical final keys

---

### 5.2 Memory Discipline

#### Test Case: SYS-PERF-MEM-001 — No Qubit Leaks on Abort

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-PERF-MEM-001` |
| **Title** | Protocol abort releases all quantum resources |
| **Priority** | MEDIUM |
| **Traces To** | Resource management best practices |

**Goal**: Verify no undelivered qubits remain in NetSquid memory after protocol abort.

**Test Logic**:
```
1. Execute protocol that triggers ABORT (e.g., high QBER)
2. After abort handler completes:
3. INSPECT: For each node in network:
   - memory = node.qmemory
   - ASSERT: all memory positions are None or explicitly released
4. INSPECT: MagicDistributor delivery queue
   - ASSERT: No pending deliveries
5. INSPECT: NetSquid event queue
   - ASSERT: No orphaned qubit events scheduled
```

**Inspection Points**:
```python
# Memory inspection after abort
for node in network.nodes:
    for pos in range(node.qmemory.num_positions):
        qubit = node.qmemory.peek(pos)
        assert qubit is None, f"Leaked qubit at {node.name}:{pos}"
```

---

#### Test Case: SYS-PERF-MEM-002 — Memory Stability Over Multiple Runs

| Attribute | Value |
|-----------|-------|
| **ID** | `SYS-PERF-MEM-002` |
| **Title** | Memory usage stable across 100 consecutive protocol executions |
| **Priority** | MEDIUM |
| **Traces To** | Production stability requirements |

**Test Logic**:
```
1. Record baseline memory usage
2. Execute 100 complete protocol runs (mix of success/abort)
3. After each run, force garbage collection
4. Record memory usage
5. ASSERT: Memory growth < 10% of baseline
6. ASSERT: No monotonic memory increase trend
```

---

## 6. Test Traceability Matrix

This matrix maps each identified gap from `squid_assesment.md` to extension modules and system test coverage.

| Gap ID | Gap Description | Source | Extension Module | System Test IDs |
|--------|-----------------|--------|------------------|-----------------|
| **TIMING-001** | Wait Time Enforcement (Δt) not natively supported | `squid_assesment.md` §2.3 | `ehok/core/timing.py` | SYS-INT-TIMING-001, SYS-INT-TIMING-002 |
| **NOISE-PARAMS-001** | NSM Parameters (μ, η, e_det) not exposed | `squid_assesment.md` §2.2 | `ehok/quantum/noise_adapter.py` | SYS-INT-NOISE-001, SYS-INT-NOISE-002 |
| **ORDERED-MSG-001** | No ordered acknowledgment mechanism | `squid_assesment.md` §2.1 | `ehok/protocols/ordered_messaging.py` | SYS-INT-MSG-001, SYS-INT-MSG-002 |
| **FEAS-001** | Pre-flight feasibility check missing | `squid_assesment.md` §2.4 | `ehok/core/feasibility.py` | SYS-E2E-DEATHVALLEY-001 |
| **DETECT-VALID-001** | Missing rounds Chernoff validation | `squid_assesment.md` §2.3 | `ehok/quantum/detection.py` | SYS-E2E-DETECTION-001 |
| **LEAKAGE-MGR-001** | Safety cap $L_{max}$ not enforced | `squid_assesment.md` §2.4 | `ehok/core/security_bounds.py` | SYS-E2E-LEAKAGE-001 |
| **NSM-BOUNDS-001** | QKD bounds used instead of NSM Max Bound | `squid_assesment.md` §2.3 | `ehok/analysis/nsm_bounds.py` | SYS-OUT-NSM-001, SYS-OUT-NSM-002 |
| **OBLIV-FORMAT-001** | Single-key output vs OT structure | `squid_assesment.md` §2.5 | `ehok/core/oblivious_key.py` | SYS-OUT-OBLIV-001, SYS-OUT-OBLIV-002, SYS-OUT-OBLIV-003 |
| **STORAGE-LINK-001** | Storage noise r not wired to Phase IV | `squid_assesment.md` §3 | `ehok/quantum/noise_adapter.py` | SYS-INT-NOISE-002, SYS-OUT-NSM-001 |

---

## 7. Abort Code Coverage Matrix

This matrix ensures all defined abort codes are exercised by at least one system test.

| Abort Code | Phase | Trigger Condition | System Test ID | Covered |
|------------|-------|-------------------|----------------|---------|
| `ABORT-I-FEAS-001` | I | Pre-flight: Q_expected > 22% or infeasible batch | SYS-E2E-DEATHVALLEY-001 | ✅ |
| `ABORT-II-DETECT-001` | II | Detection rate Chernoff violation | SYS-E2E-DETECTION-001 | ✅ |
| `ABORT-II-QBER-001` | II | Adjusted QBER > 22% | SYS-E2E-HIGHQBER-001 | ✅ |
| `ABORT-II-ORDER-001` | II | Message ordering violation | SYS-E2E-ATTACK-ORDER-001 | ✅ |
| `ABORT-II-ACK-TIMEOUT` | II | ACK not received within timeout | SYS-INT-MSG-002 | ✅ |
| `ABORT-III-LEAKAGE-001` | III | Syndrome leakage exceeds L_max | SYS-E2E-LEAKAGE-001 | ✅ |
| `ABORT-IV-FEAS-001` | IV | Min-entropy insufficient for key length | SYS-E2E-DEATHVALLEY-001 | ✅ |

---

## 8. Sprint Completion Validation Checkpoints

| Sprint | Validation Test | Pass Criterion | System Tests Required |
|--------|-----------------|----------------|----------------------|
| **Sprint 1 Complete** | NSM bounds unit + integration | $h_{min}(r)$ matches [Lupo 2023] within 0.1% | SYS-OUT-NSM-001, SYS-OUT-NSM-002 |
| **Sprint 1 Complete** | Timing barrier integration | ns.sim_time() advances by Δt | SYS-INT-TIMING-001, SYS-INT-TIMING-002 |
| **Sprint 2 Complete** | Ordered messaging | ACK blocking verified | SYS-INT-MSG-001, SYS-INT-MSG-002 |
| **Sprint 2 Complete** | Detection validation | Chernoff abort triggered | SYS-E2E-DETECTION-001 |
| **Sprint 3 Complete** | Oblivious output structure | $(S_0, S_1)$ / $(S_C, C)$ validated | SYS-OUT-OBLIV-001, SYS-OUT-OBLIV-002 |
| **Sprint 3 Complete** | E2E golden path | Positive key at Q=5%, n=10,000 | SYS-E2E-GOLDEN-001 |
| **Release Ready** | Full abort coverage | All abort codes triggered correctly | All SYS-E2E-* tests |

---

## Appendix A: NetSquid Inspection API Reference

This appendix documents the NetSquid internal APIs required for white-box testing.

### A.1 Simulation Time Access
```python
import netsquid as ns

# Get current simulation time (nanoseconds)
current_time = ns.sim_time()

# Reset simulation
ns.sim_reset()

# Set random seed for determinism
ns.set_random_state(seed=42)
```

### A.2 Quantum Memory Inspection
```python
from netsquid.components.qmemory import QuantumMemory

# Access node's quantum memory
qmem: QuantumMemory = node.qmemory

# Check memory position occupancy
qubit = qmem.peek(position=0)  # Returns Qubit or None

# Get number of memory positions
num_positions = qmem.num_positions
```

### A.3 Quantum Channel Model Inspection
```python
from netsquid.components.qchannel import QuantumChannel

# Access channel noise model
noise_model = channel.models.get("quantum_noise_model")

# Get depolarizing parameter
if hasattr(noise_model, "properties"):
    depolar_rate = noise_model.properties.get("depolar_rate")
```

### A.4 MagicDistributor Access
```python
# Access from SquidASM network (implementation-specific)
distributor = network._link_distributors["Alice-Bob"]

# Get model parameters
model_params = distributor._state_sampler_factory._model_params

# Check cycle time and success probability
cycle_time = model_params.cycle_time
prob_max_mixed = model_params.prob_max_mixed
```

---

## Appendix B: Test Fixture Requirements

### B.1 Network Configuration Fixtures

```yaml
# test_network_config_nominal.yaml
nodes:
  - name: "Alice"
    qubits:
      - id: 0
        t1: 1e9
        t2: 5e8
    gate_fidelity: 0.999
  - name: "Bob"
    qubits:
      - id: 0
        t1: 1e9
        t2: 5e8
    gate_fidelity: 0.999

links:
  - node1: "Alice"
    node2: "Bob"
    fidelity: 0.97
    noise_type: "Depolarise"
```

### B.2 Protocol Configuration Fixtures

```yaml
# test_protocol_config_nominal.yaml
batch_size: 20000
delta_t_ns: 1000000000  # 1 second
epsilon_sec: 1e-6
epsilon_cor: 1e-9
storage_noise_r: 0.3
qber_hard_limit: 0.22
qber_warning_limit: 0.11
```

---

## Appendix C: Log Event Schema

All system tests must verify the presence and ordering of these log events:

| Event ID | Phase | Description | Required Fields |
|----------|-------|-------------|-----------------|
| `TIMING_COMMIT_ACK_RECEIVED` | II | Bob's commitment acknowledged | `t_commit_ack_ns`, `session_id` |
| `TIMING_BASIS_REVEAL_BLOCKED` | II | Premature reveal attempt | `t_attempt_ns`, `t_required_ns` |
| `TIMING_BASIS_REVEAL_ALLOWED` | II | Reveal permitted after Δt | `t_reveal_ns`, `delta_t_actual_ns` |
| `DETECTION_VALIDATED` | II | Chernoff check passed | `observed_rate`, `expected_rate`, `tolerance` |
| `DETECTION_ANOMALY` | II | Chernoff check failed | `observed_rate`, `expected_rate`, `abort_code` |
| `QBER_COMPUTED` | II | QBER estimation complete | `qber_observed`, `penalty_mu`, `qber_adjusted` |
| `LEAKAGE_TRACKED` | III | Syndrome transmitted | `block_id`, `syndrome_bits`, `cumulative_leakage` |
| `LEAKAGE_CAP_EXCEEDED` | III | Safety limit breached | `cumulative_leakage`, `l_max`, `abort_code` |
| `KEY_LENGTH_COMPUTED` | IV | Final length determined | `n`, `h_min_r`, `leakage`, `final_length` |
| `OBLIVIOUS_OUTPUT_FORMATTED` | IV | OT keys generated | `key_length`, `alice_keys`, `bob_choice` |

---

*End of System Test Specification*

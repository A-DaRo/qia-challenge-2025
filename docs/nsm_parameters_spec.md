# NSM Parameters Implementation Specification

**Document Type:** Implementation Specification  
**Version:** 1.0  
**Date:** December 18, 2025  
**Status:** Draft  
**Parent Document:** [nsm_parameters_enforcement.md](nsm_parameters_enforcement.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Phase 1: Foundation Components](#3-phase-1-foundation-components)
4. [Phase 2: Enhanced Noise Modeling](#4-phase-2-enhanced-noise-modeling)
5. [Phase 3: Validation Framework](#5-phase-3-validation-framework)
6. [Phase 4: Advanced Features](#6-phase-4-advanced-features)
7. [SquidASM Pipeline Injection Specification](#7-squidasm-pipeline-injection-specification)
8. [Module Organization](#8-module-organization)
9. [Integration Contracts](#9-integration-contracts)

---

## 1. Overview

### 1.1 Purpose

This document specifies the implementation architecture for enforcing Noisy Storage Model (NSM) parameters within the Caligo simulation environment. It translates the theoretical requirements from `nsm_parameters_enforcement.md` into actionable component designs.

### 1.2 Design Goals

| Goal | Rationale |
|------|-----------|
| **Separation of Concerns** | NSM parameters (security) vs. channel parameters (physics) vs. simulation config |
| **Fail-Fast Validation** | Invalid parameters rejected at construction, not runtime |
| **Pipeline Transparency** | Clear injection points with no hidden side effects |
| **Testability** | Each component independently testable with mock dependencies |
| **Extensibility** | New noise models addable without modifying existing code |

### 1.3 Key Abstractions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ABSTRACTION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Security Layer          │   Physics Layer           │   Simulation Layer  │
│   ──────────────          │   ─────────────           │   ────────────────  │
│                           │                           │                     │
│   NSMParameters           │   ChannelParameters       │   NetworkConfig     │
│   ├─ storage_noise_r      │   ├─ length_km            │   ├─ nodes          │
│   ├─ storage_rate_nu      │   ├─ attenuation          │   ├─ links          │
│   ├─ delta_t_ns           │   ├─ t1_ns, t2_ns         │   └─ noise_profiles │
│   └─ channel_fidelity     │   └─ cycle_time_ns        │                     │
│                           │                           │                     │
│   SecurityAnalyzer        │   ChannelNoiseProfile     │   NoiseInjector     │
│   ├─ verify_condition()   │   ├─ total_qber           │   ├─ inject()       │
│   └─ compute_bounds()     │   └─ detection_prob       │   └─ configure()    │
│                           │                           │                     │
│   TimingBarrier           │   NoiseModelFactory       │   PipelineAdapter   │
│   ├─ wait_delta_t()       │   ├─ create_depolar()     │   ├─ to_squidasm()  │
│   └─ verify_compliance()  │   └─ create_t1t2()        │   └─ to_netbuilder()│
│                           │                           │                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Principles

### 2.1 Layered Configuration Flow

NSM parameters flow through three transformation stages before reaching the simulation:

```
User Config (YAML/Python)
         │
         ▼
┌─────────────────────────┐
│  1. VALIDATION LAYER    │  ← Invariant checks, type coercion
│     NSMParameters       │  ← ChannelNoiseProfile
│     ChannelParameters   │  ← Dataclass __post_init__
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. TRANSLATION LAYER   │  ← NSM → NetSquid mapping
│     NoiseModelFactory   │  ← Creates DepolarNoiseModel, T1T2NoiseModel
│     ProfileTranslator   │  ← Maps QBER components to model params
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. INJECTION LAYER     │  ← SquidASM pipeline integration
│     NoiseInjector       │  ← Attaches models to QDevice, Link
│     PipelineAdapter     │  ← Bridges Caligo config to StackNetworkConfig
└───────────┬─────────────┘
            │
            ▼
      SquidASM Runtime
```

### 2.2 Dependency Inversion

Components depend on **abstractions** (protocols/interfaces), not concrete implementations:

| Component | Depends On | Not On |
|-----------|------------|--------|
| `CaligoNetworkBuilder` | `INoiseModelFactory` | `DepolarNoiseModel` directly |
| `TimingBarrier` | `ISimulationClock` | `netsquid.sim_time()` directly |
| `SecurityAnalyzer` | `IQBERMeasurement` | Concrete protocol implementation |

### 2.3 Configuration Immutability

All parameter dataclasses are **frozen** after construction:
- Prevents accidental mutation during simulation
- Enables safe sharing across threads/processes
- Simplifies reasoning about state

---

## 3. Phase 1: Foundation Components

### 3.1 NSMParameters Dataclass

**Location:** `caligo/simulation/physical_model.py`

**Responsibility:** Encapsulate and validate all NSM security parameters.

**Invariants:**

| ID | Constraint | Rationale |
|----|------------|-----------|
| INV-NSM-001 | `storage_noise_r ∈ [0, 1]` | Probability bound |
| INV-NSM-002 | `storage_rate_nu ∈ [0, 1]` | Fraction bound |
| INV-NSM-003 | `storage_dimension_d == 2` | Qubit assumption |
| INV-NSM-004 | `delta_t_ns > 0` | Positive wait time |
| INV-NSM-005 | `channel_fidelity ∈ (0.5, 1]` | Above random threshold |
| INV-NSM-006 | `detection_eff_eta ∈ (0, 1]` | Physical efficiency |

**Derived Properties:**

| Property | Formula | Purpose |
|----------|---------|---------|
| `depolar_prob` | `1 - r` | NetSquid `depolar_rate` |
| `qber_channel` | Erven formula | Security condition check |
| `storage_capacity` | `1 - h(depolar_prob)` | Capacity bound |
| `qber_storage` | `(1 - r) / 2` | Security threshold |

**Factory Methods:**

| Method | Use Case |
|--------|----------|
| `from_erven_experimental()` | Reproduce Erven 2014 parameters |
| `for_testing(r, F, Δt)` | Simplified test configurations |
| `from_yaml(path)` | Load from configuration file |

### 3.2 ChannelNoiseProfile Dataclass

**Location:** `caligo/simulation/noise_models.py`

**Responsibility:** Aggregate physical channel noise sources for QBER computation.

**Invariants:**

| ID | Constraint | Rationale |
|----|------------|-----------|
| INV-CNP-001 | `source_fidelity ∈ (0.5, 1]` | Above random |
| INV-CNP-002 | `detector_efficiency ∈ (0, 1]` | Physical bound |
| INV-CNP-003 | `detector_error ∈ [0, 0.5]` | Error rate bound |
| INV-CNP-004 | `dark_count_rate ∈ [0, 1]` | Probability bound |
| INV-CNP-005 | `transmission_loss ∈ [0, 1)` | Loss must allow transmission |

**Key Methods:**

| Method | Behavior |
|--------|----------|
| `total_qber` | Compute Q_channel via Erven formula |
| `is_secure` | Check `total_qber < 0.11` |
| `is_feasible` | Check `total_qber < 0.22` |
| `to_nsm_parameters(r, ν, Δt)` | Convert to full NSMParameters |

### 3.3 ChannelParameters Dataclass

**Location:** `caligo/simulation/physical_model.py`

**Responsibility:** Physical channel characteristics for honest party link.

**Key Properties:**

| Property | Physical Meaning |
|----------|------------------|
| `propagation_delay_ns` | Light travel time |
| `transmittance` | $10^{-\text{loss}/10}$ |
| `total_loss_db` | Channel attenuation |

### 3.4 TimingBarrier Class

**Location:** `caligo/simulation/timing.py`

**Responsibility:** Enforce $\Delta t$ wait time as causal barrier in discrete-event simulation.

**State Machine:**

```
         mark_quantum_complete()          wait_delta_t()
    IDLE ─────────────────────────► WAITING ──────────────► READY
     ▲                                                        │
     └────────────────────────── reset() ─────────────────────┘
```

**Interface Contract:**

| Method | Precondition | Postcondition |
|--------|--------------|---------------|
| `mark_quantum_complete()` | State is IDLE | State is WAITING, timestamp recorded |
| `wait_delta_t()` | State is WAITING | Yields until Δt elapsed, state becomes READY |
| `can_reveal_basis()` | None | Returns `state == READY` |
| `reset()` | None | State becomes IDLE |

**Simulation Integration:**

The `wait_delta_t()` method is a **generator** that yields to the SquidASM event loop:
- Calculates remaining wait time from `ns.sim_time()`
- Yields `EventExpression` to advance simulation clock
- Does NOT block Python thread

### 3.5 CaligoNetworkBuilder Class

**Location:** `caligo/simulation/network_builder.py`

**Responsibility:** Factory for SquidASM `StackNetworkConfig` with NSM-aware noise models.

**Builder Pattern:**

```
CaligoNetworkBuilder(nsm_params, channel_params)
    │
    ├── .with_detection_model(efficiency, dark_count)  [Phase 2]
    ├── .with_memory_noise(T1, T2)
    ├── .with_gate_noise(depolar_prob)
    │
    └── .build_two_node_network(alice, bob, num_qubits)
            │
            └── Returns: StackNetworkConfig
```

**Current Limitation:** Only supports `fidelity` parameter for link noise.

---

## 4. Phase 2: Enhanced Noise Modeling

### 4.1 Detection Efficiency Model

**Goal:** Explicitly model photon loss before measurement.

**Component:** `DetectionEfficiencyModel`

**Integration Point:** Extend `CaligoNetworkBuilder` to accept `detection_efficiency` parameter.

**Semantic Behavior:**
1. Accept $\eta \in (0, 1]$ as detection efficiency
2. Map to `DoubleClickModelParameters.detector_efficiency`
3. Affects EPR heralding success probability
4. Impacts measured QBER through conditional statistics

**Mapping Strategy:**

| NSM Parameter | netsquid_magic Parameter | Relationship |
|---------------|-------------------------|--------------|
| $\eta$ | `detector_efficiency` | Direct mapping |
| - | `prob_success` | Computed from $\eta$ and loss model |

### 4.2 Dark Count Injection

**Goal:** Model spurious detector clicks that introduce errors.

**Component:** `DarkCountInjector`

**Semantic Behavior:**
1. Accept $P_{\text{dark}} \in [0, 1]$ as dark count probability
2. For each detection window without signal photon:
   - With probability $P_{\text{dark}}$, register false click
   - False clicks produce random measurement outcomes
3. Contributes to QBER: $Q_{\text{dark}} = (1-\eta) \cdot P_{\text{dark}} \cdot 0.5$

**Integration Approach:**

Two options exist:

| Approach | Mechanism | Tradeoff |
|----------|-----------|----------|
| **A. Model-level** | Extend `DoubleClickModelParameters` | Accurate but requires netsquid_magic modification |
| **B. Post-processing** | Apply dark count errors after measurement | Approximate but non-invasive |

**Recommendation:** Approach B for Phase 2, migrate to A in Phase 4.

### 4.3 DoubleClick Model Integration

**Goal:** Use netsquid_magic's realistic heralded entanglement model.

**Component:** `HeraldedLinkConfigBuilder`

**Semantic Behavior:**
1. Replace simple `depolarise` link type with `heralded_double_click`
2. Configure full parameter set:

| Parameter | Source | Description |
|-----------|--------|-------------|
| `detector_efficiency` | ChannelNoiseProfile | Combined η |
| `dark_count_probability` | ChannelNoiseProfile | P_dark |
| `visibility` | Derived from source_fidelity | Interference quality |
| `emission_prob` | Fixed or configured | Single-photon emission |
| `length_A`, `length_B` | ChannelParameters | Fiber lengths |

**Registration Requirement:**

Custom link types must be registered with netsquid_netbuilder:
- Define `HeraldedDoubleClickQLinkModule`
- Register in `QLinkModuleRegistry`
- Reference by string name in `LinkConfig.typ`

### 4.4 Measurement Error Mapping

**Goal:** Map intrinsic detector error $e_{\text{det}}$ to gate-level noise.

**Component:** `MeasurementErrorMapper`

**Semantic Behavior:**
1. Accept $e_{\text{det}} \in [0, 0.5]$ as detector error rate
2. Map to `GenericQDeviceConfig.single_qubit_gate_depolar_prob`
3. Approximate mapping: `gate_depolar ≈ 2 × e_det` (worst case)

**Rationale:**
- Measurement in SquidASM is implemented as gate + projection
- Gate depolarization introduces bit flips with probability $p/2$
- Setting $p = 2 \cdot e_{\text{det}}$ achieves approximate error rate

**Limitation:** This is an approximation. Exact modeling requires custom measurement instruction.

---

## 5. Phase 3: Validation Framework

### 5.1 NSM Condition Verifier

**Goal:** Runtime verification of $Q_{\text{channel}} < Q_{\text{storage}}$.

**Component:** `NSMSecurityVerifier`

**Interface:**

| Method | Input | Output |
|--------|-------|--------|
| `verify(measured_qber, nsm_params)` | Empirical QBER, parameters | `bool` or raises `SecurityError` |
| `compute_margin(measured_qber, nsm_params)` | As above | Security margin (float) |
| `is_within_threshold(measured_qber, threshold)` | QBER, threshold | `bool` |

**Verification Logic:**

```
compute Q_storage = (1 - r) / 2
IF measured_qber >= Q_storage:
    RAISE SecurityError("NSM condition violated")
IF measured_qber >= 0.11:
    LOG WARNING "Above Schaffner conservative threshold"
IF measured_qber >= 0.22:
    RAISE SecurityError("Above König hard limit")
RETURN True
```

**Integration Point:** Called by `Orchestrator` after sifting phase completes.

### 5.2 QBER Measurement Validator

**Goal:** Validate empirical QBER matches theoretical prediction.

**Component:** `QBERValidator`

**Interface:**

| Method | Purpose |
|--------|---------|
| `validate(measured, expected, tolerance)` | Compare with absolute tolerance |
| `validate_relative(measured, expected, rel_tol)` | Compare with relative tolerance |
| `compute_deviation(measured, expected)` | Return signed deviation |

**Usage Context:**
- Post-sifting: Compare measured QBER against `ChannelNoiseProfile.total_qber`
- Debugging: Identify simulation configuration errors
- Regression testing: Ensure noise models behave consistently

### 5.3 Timing Compliance Reporter

**Goal:** Track and report timing constraint compliance throughout protocol.

**Component:** `TimingComplianceReporter`

**Tracked Metrics:**

| Metric | Description |
|--------|-------------|
| `quantum_complete_time` | Timestamp when quantum phase ended |
| `basis_reveal_time` | Timestamp when basis was revealed |
| `actual_wait_duration` | `reveal_time - complete_time` |
| `required_wait_duration` | Configured $\Delta t$ |
| `compliance_status` | COMPLIANT / VIOLATED / PENDING |

**Report Generation:**
- JSON export for analysis
- Integration with Caligo logging system
- Failure details on violation

### 5.4 Integration Test Suite

**Goal:** End-to-end tests validating NSM parameter enforcement.

**Test Categories:**

| Category | Tests |
|----------|-------|
| **Parameter Sweep** | Vary $(r, \nu, \Delta t)$ systematically |
| **Boundary Conditions** | Test at QBER thresholds (0.11, 0.22) |
| **Timing Enforcement** | Verify TimingBarrier blocks premature reveal |
| **Noise Injection** | Confirm configured fidelity matches measured |
| **Security Condition** | Verify Q_channel < Q_storage detection |

**Test Fixtures:**
- `perfect_channel_fixture`: F=1.0, no errors
- `erven_experimental_fixture`: Erven 2014 parameters
- `high_noise_fixture`: Near threshold QBER
- `violation_fixture`: Parameters that should trigger security error

---

## 6. Phase 4: Advanced Features

### 6.1 Custom StateSampler Factory

**Goal:** Full control over EPR state generation noise model.

**Component:** `CaligoStateSamplerFactory`

**Semantic Behavior:**
1. Implement `IStateDeliverySamplerFactory` interface
2. Accept arbitrary noise profile configuration
3. Generate `StateSampler` with exact state mixture:
   - Base Bell state fidelity from `source_fidelity`
   - Depolarization from configured profile
   - Optional custom error channels

**Extension Points:**
- Asymmetric noise (different arms)
- Non-Markovian noise models
- Correlated errors across pairs

### 6.2 Configurable Heralding Model

**Goal:** Support heralding models beyond depolarise/double-click.

**Component:** `HeraldingModelRegistry`

**Supported Models:**

| Model | Use Case |
|-------|----------|
| `perfect` | Ideal EPR pairs (testing) |
| `depolarise` | Simple symmetric noise |
| `double_click` | Realistic heralded entanglement |
| `single_click` | Single-photon interference |
| `custom` | User-defined via factory |

**Registration API:**

```
HeraldingModelRegistry.register(
    name="custom_model",
    factory=CustomModelFactory,
    config_class=CustomModelConfig,
)
```

### 6.3 Parameter Sweep Automation

**Goal:** Systematic exploration of NSM parameter space.

**Component:** `NSMParameterSweep`

**Semantic Behavior:**
1. Define parameter ranges: `r ∈ [0.5, 1.0]`, `ν ∈ [0, 0.1]`, `Δt ∈ [1μs, 10ms]`
2. Generate grid or random samples
3. Execute protocol for each configuration
4. Collect metrics: QBER, key rate, security margin
5. Export results for analysis

**Output Format:**
- CSV/Parquet for numerical analysis
- JSON for full protocol traces
- Plots (optional): QBER vs r, key rate vs Δt

### 6.4 Security Margin Visualization

**Goal:** Visual representation of security margins across parameter space.

**Component:** `SecurityMarginVisualizer`

**Visualizations:**

| Plot Type | Axes | Purpose |
|-----------|------|---------|
| **Heatmap** | (r, ν) → margin | Identify secure operating region |
| **Line Plot** | Δt → QBER | Show timing sensitivity |
| **Scatter** | Measured vs Expected QBER | Validate simulation accuracy |
| **Phase Diagram** | (r, QBER) → secure/insecure | Binary security classification |

---

## 7. SquidASM Pipeline Injection Specification

### 7.1 Injection Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SQUIDASM INJECTION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Caligo Layer                                                              │
│   ────────────                                                              │
│                                                                             │
│   ┌───────────────────┐                                                     │
│   │  NSMParameters    │──────────────────────────┐                          │
│   └───────────────────┘                          │                          │
│            │                                     │                          │
│            ▼                                     ▼                          │
│   ┌───────────────────┐              ┌───────────────────┐                  │
│   │ NoiseModelFactory │              │  TimingBarrier    │                  │
│   │  .create_*()      │              │  (injected into   │                  │
│   └────────┬──────────┘              │   CaligoProgram)  │                  │
│            │                         └───────────────────┘                  │
│            ▼                                                                │
│   ┌───────────────────┐                                                     │
│   │ PipelineAdapter   │                                                     │
│   │  .configure()     │                                                     │
│   └────────┬──────────┘                                                     │
│            │                                                                │
├────────────┼────────────────────────────────────────────────────────────────┤
│            │                                                                │
│   SquidASM Layer                                                            │
│   ─────────────                                                             │
│            │                                                                │
│            ▼                                                                │
│   ┌───────────────────┐      ┌───────────────────┐                          │
│   │ StackNetworkConfig│─────►│  StackConfig      │                          │
│   │  .links           │      │  .qdevice_cfg     │                          │
│   └────────┬──────────┘      └────────┬──────────┘                          │
│            │                          │                                     │
│            ▼                          ▼                                     │
│   ┌───────────────────┐      ┌───────────────────┐                          │
│   │   LinkConfig      │      │ GenericQDeviceConfig                         │
│   │  .typ = "custom"  │      │  .T1, .T2         │                          │
│   │  .cfg = {...}     │      │  .gate_depolar    │                          │
│   └────────┬──────────┘      └────────┬──────────┘                          │
│            │                          │                                     │
├────────────┼──────────────────────────┼─────────────────────────────────────┤
│            │                          │                                     │
│   netsquid_netbuilder Layer                                                 │
│   ─────────────────────                                                     │
│            │                          │                                     │
│            ▼                          ▼                                     │
│   ┌───────────────────┐      ┌───────────────────┐                          │
│   │ QLinkModule       │      │ QDeviceModule     │                          │
│   │ .create_distributor()    │ .build_processor()│                          │
│   └────────┬──────────┘      └────────┬──────────┘                          │
│            │                          │                                     │
├────────────┼──────────────────────────┼─────────────────────────────────────┤
│            │                          │                                     │
│   netsquid_magic Layer                                                      │
│   ────────────────────                                                      │
│            │                          │                                     │
│            ▼                          │                                     │
│   ┌───────────────────┐               │                                     │
│   │ MagicDistributor  │               │                                     │
│   │ .state_sampler    │◄──────────────┤                                     │
│   └────────┬──────────┘               │                                     │
│            │                          │                                     │
├────────────┼──────────────────────────┼─────────────────────────────────────┤
│            │                          │                                     │
│   NetSquid Layer                      │                                     │
│   ──────────────                      │                                     │
│            │                          │                                     │
│            ▼                          ▼                                     │
│   ┌───────────────────┐      ┌───────────────────┐                          │
│   │  StateSampler     │      │ QuantumProcessor  │                          │
│   │  (EPR states)     │      │ .memory_noise     │                          │
│   └───────────────────┘      │ .phys_instructions│                          │
│                              └───────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Injection Points

#### 7.2.1 Link-Level Injection (EPR Noise)

**Target:** `LinkConfig` in StackNetworkConfig

**Mechanism:**

1. **Standard Types:** Use built-in `typ` values (`"perfect"`, `"depolarise"`)
2. **Custom Types:** Register custom QLinkModule with netbuilder

**Standard Injection (Current):**

```
LinkConfig(
    stack1="Alice",
    stack2="Bob",
    typ="depolarise",
    cfg={"fidelity": nsm_params.channel_fidelity}
)
```

**Enhanced Injection (Phase 2+):**

```
LinkConfig(
    stack1="Alice",
    stack2="Bob",
    typ="heralded_double_click",  # Custom registered type
    cfg={
        "detector_efficiency": channel_profile.detector_efficiency,
        "dark_count_probability": channel_profile.dark_count_rate,
        "visibility": derived_visibility,
        ...
    }
)
```

#### 7.2.2 Device-Level Injection (Memory/Gate Noise)

**Target:** `GenericQDeviceConfig` in StackConfig

**Mechanism:** Configure QDevice parameters before network construction.

**Injection Points:**

| Parameter | NetSquid Effect |
|-----------|-----------------|
| `T1` | Amplitude damping on stored qubits |
| `T2` | Phase damping on stored qubits |
| `single_qubit_gate_depolar_prob` | Noise per single-qubit gate |
| `two_qubit_gate_depolar_prob` | Noise per two-qubit gate |
| `num_qubits` | Memory capacity |

**Configuration Flow:**

```
ChannelParameters
    │
    ├─► T1, T2 ─────────────────► GenericQDeviceConfig.T1, T2
    │
ChannelNoiseProfile
    │
    └─► detector_error ─────────► GenericQDeviceConfig.single_qubit_gate_depolar_prob
```

#### 7.2.3 Protocol-Level Injection (Timing)

**Target:** `CaligoProgram` base class

**Mechanism:** TimingBarrier created from NSMParameters and integrated into program flow.

**Injection Flow:**

```
ProtocolParameters.nsm_params.delta_t_ns
    │
    └─► TimingBarrier(delta_t_ns)
            │
            └─► CaligoProgram._timing_barrier
                    │
                    └─► Alice._run_protocol()
                            │
                            ├─► mark_quantum_complete()
                            ├─► yield from wait_delta_t()
                            └─► reveal_basis()
```

### 7.3 Custom Link Type Registration

To use advanced noise models (Phase 2+), custom link types must be registered:

**Step 1: Define Configuration Dataclass**

```
@dataclass
class CaligoHeraldedLinkConfig:
    """Configuration for Caligo heralded entanglement link."""
    detector_efficiency: float
    dark_count_probability: float
    visibility: float
    emission_probability: float
    length_alice_km: float
    length_bob_km: float
```

**Step 2: Implement QLinkModule**

```
class CaligoHeraldedQLinkModule(IQLinkModule):
    """NetBuilder module for Caligo heralded links."""
    
    CONFIG_CLASS = CaligoHeraldedLinkConfig
    
    def create_distributor(self, config, node_a, node_b):
        model_params = DoubleClickModelParameters(
            detector_efficiency=config.detector_efficiency,
            dark_count_probability=config.dark_count_probability,
            ...
        )
        return MagicDistributor(
            nodes=[node_a, node_b],
            state_sampler_factory=DoubleClickStateSamplerFactory(),
            model_params=model_params,
        )
```

**Step 3: Register with NetBuilder**

```
QLinkModuleRegistry.register("caligo_heralded", CaligoHeraldedQLinkModule)
```

**Step 4: Use in LinkConfig**

```
LinkConfig(typ="caligo_heralded", cfg={...})
```

### 7.4 Runtime Parameter Propagation

Parameters flow through the stack at different lifecycle stages:

| Stage | When | What Propagates |
|-------|------|-----------------|
| **Configuration** | Before `run_simulation()` | All static parameters |
| **Network Build** | During `_setup_network()` | Link/device configs |
| **Program Init** | During `Program.__init__()` | TimingBarrier, session params |
| **Protocol Runtime** | During `run()` generator | Dynamic state, measurements |
| **Post-Processing** | After simulation completes | Collected metrics |

---

## 8. Module Organization

### 8.1 Package Structure

```
caligo/
├── simulation/
│   ├── __init__.py
│   ├── constants.py           # Literature values, thresholds
│   ├── physical_model.py      # NSMParameters, ChannelParameters
│   ├── noise_models.py        # ChannelNoiseProfile, wrappers
│   ├── timing.py              # TimingBarrier
│   ├── network_builder.py     # CaligoNetworkBuilder
│   │
│   ├── injection/             # Phase 2+ (NEW)
│   │   ├── __init__.py
│   │   ├── link_modules.py    # Custom QLinkModule implementations
│   │   ├── noise_factory.py   # NoiseModelFactory
│   │   └── pipeline_adapter.py # PipelineAdapter
│   │
│   └── validation/            # Phase 3 (NEW)
│       ├── __init__.py
│       ├── security_verifier.py
│       ├── qber_validator.py
│       └── compliance_reporter.py
│
├── analysis/                  # Phase 4 (NEW)
│   ├── __init__.py
│   ├── parameter_sweep.py
│   └── visualization.py
│
└── protocol/
    ├── base.py                # CaligoProgram (integrates TimingBarrier)
    ├── alice.py
    └── bob.py
```

### 8.2 Dependency Graph

```
                    ┌─────────────────┐
                    │    constants    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │physical_model│  │noise_models │  │   timing    │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ network_builder │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ link_modules│  │noise_factory│  │pipeline_adap│
    └─────────────┘  └─────────────┘  └──────┬──────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  protocol/base  │
                                    └─────────────────┘
```

---

## 9. Integration Contracts

### 9.1 NoiseModelFactory Protocol

```
protocol INoiseModelFactory:
    """Factory for creating NetSquid noise models from Caligo parameters."""
    
    def create_depolar_model(params: NSMParameters) -> DepolarNoiseModel:
        """Create depolarizing noise model for storage simulation."""
        ...
    
    def create_t1t2_model(params: ChannelParameters) -> T1T2NoiseModel:
        """Create T1T2 model for memory decoherence."""
        ...
    
    def create_fibre_loss_model(params: ChannelParameters) -> FibreLossModel:
        """Create fibre loss model for channel attenuation."""
        ...
```

### 9.2 SimulationClock Protocol

```
protocol ISimulationClock:
    """Abstraction over simulation time source."""
    
    def current_time_ns() -> float:
        """Return current simulation time in nanoseconds."""
        ...
    
    def wait(duration_ns: float) -> Generator[EventExpression, None, None]:
        """Yield control for specified duration."""
        ...
```

### 9.3 SecurityVerifier Protocol

```
protocol ISecurityVerifier:
    """Verifies NSM security conditions at runtime."""
    
    def verify_nsm_condition(
        measured_qber: float,
        nsm_params: NSMParameters
    ) -> bool:
        """Check Q_channel < Q_storage."""
        ...
    
    def verify_threshold(
        measured_qber: float,
        threshold: float
    ) -> bool:
        """Check QBER against arbitrary threshold."""
        ...
```

### 9.4 PipelineAdapter Protocol

```
protocol IPipelineAdapter:
    """Adapts Caligo configuration to SquidASM structures."""
    
    def to_stack_network_config(
        nsm_params: NSMParameters,
        channel_params: ChannelParameters,
        alice_name: str,
        bob_name: str,
    ) -> StackNetworkConfig:
        """Create SquidASM network configuration."""
        ...
    
    def to_protocol_parameters(
        nsm_params: NSMParameters,
        session_id: str,
        num_pairs: int,
    ) -> ProtocolParameters:
        """Create protocol execution parameters."""
        ...
```

---

## Appendix A: Configuration Examples

### A.1 Minimal Configuration (Testing)

```yaml
# caligo_config_minimal.yaml
nsm:
  storage_noise_r: 0.75
  storage_rate_nu: 0.01
  delta_t_ns: 1_000_000  # 1 ms
  channel_fidelity: 0.95

simulation:
  num_pairs: 100
  num_qubits: 10
```

### A.2 Erven Experimental Configuration

```yaml
# caligo_config_erven.yaml
nsm:
  storage_noise_r: 0.75
  storage_rate_nu: 0.002
  delta_t_ns: 1_000_000

channel:
  source_fidelity: 0.99997  # 1 - μ
  detector_efficiency: 0.0150
  detector_error: 0.0093
  dark_count_rate: 1.50e-8

physical:
  length_km: 0.0  # Table-top
  t1_ns: 100_000_000
  t2_ns: 10_000_000
```

### A.3 High-Noise Stress Test Configuration

```yaml
# caligo_config_stress.yaml
nsm:
  storage_noise_r: 0.60  # High adversary capability
  storage_rate_nu: 0.05
  delta_t_ns: 500_000  # 0.5 ms

channel:
  source_fidelity: 0.90  # Near threshold
  detector_efficiency: 0.80
  detector_error: 0.03
  dark_count_rate: 1.0e-5

# Expected: QBER near 0.11 threshold
```

---

## Appendix B: Migration Path

### B.1 From Current Implementation

| Current | Target | Migration Step |
|---------|--------|----------------|
| Direct fidelity in LinkConfig | NoiseModelFactory | Wrap in factory method |
| Hardcoded ns.sim_time() | ISimulationClock | Inject clock interface |
| Inline QBER computation | ChannelNoiseProfile | Use dataclass property |
| Manual timing checks | TimingBarrier | Use state machine |

### B.2 Deprecation Timeline

| Phase | Deprecated | Replacement | Removal |
|-------|------------|-------------|---------|
| Phase 2 | Direct LinkConfig creation | CaligoNetworkBuilder | Phase 3 |
| Phase 3 | Manual security checks | NSMSecurityVerifier | Phase 4 |
| Phase 4 | String-based link types | Type-safe enums | Future |

---

*Document maintained by the Caligo development team. Last updated: December 18, 2025.*

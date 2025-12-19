# NSM Parameters Physical Enforcement in the Squid Simulation Stack

**Document Type:** Technical Specification  
**Version:** 2.0  
**Date:** December 18, 2025  
**Status:** Draft  
**Parent Documents:** [caligo_architecture.md](caligo/caligo_architecture.md), [phase_b_spec.md](caligo/phase_b_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Squid Stack Noise Model Architecture](#3-squid-stack-noise-model-architecture)
4. [NSM Parameter to Physical Model Mapping](#4-nsm-parameter-to-physical-model-mapping)
5. [Caligo Simulation Layer Architecture](#5-caligo-simulation-layer-architecture)
6. [Integration Architecture](#6-integration-architecture)
7. [Validation Strategy](#7-validation-strategy)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Executive Summary

This document specifies how Noisy Storage Model (NSM) parameters must be translated into physically meaningful noise models within the NetSquid/SquidASM simulation environment for the Caligo $\binom{2}{1}$-OT protocol implementation.

**Critical Insight:** For error tolerance proofs and robustness analysis to be valid, NSM parameters must have **physical consequences at the simulation level**. Abstract security parameters without simulation-level enforcement produce flawed empirical results.

### 1.1 Scope

This specification covers:
1. Translation of NSM theoretical parameters to NetSquid noise models
2. Injection points for noise model configuration in the Squid stack
3. Caligo integration architecture for NSM enforcement
4. Validation methodology for parameter correctness

### 1.2 Key Findings from Codebase Analysis

| Layer | Noise Injection Point | Configurable Parameters |
|-------|----------------------|------------------------|
| **NetSquid Core** | `QuantumErrorModel` subclasses | `depolar_rate`, `T1`, `T2`, `p_loss_*` |
| **netsquid_magic** | `MagicDistributor` + `IModelParameters` | Depolarising/Heralded model params incl. detector/dark-count knobs |
| **netsquid_netbuilder** | `IQLinkConfig`, `IQDeviceConfig` | Link configs (`depolarise`, `heralded-double-click`), device configs (T1/T2, gate depolar) |
| **SquidASM (stack runner)** | `StackNetworkConfig` → netbuilder conversion | Typed link/device configs via `squidasm.run.stack.config` |
| **SquidASM (NetQASM-interface, out-of-scope)** | `NetSquidNetwork._create_link_distributor()` | Simplified link fidelity→noise mapping |

---

## 2. Problem Statement

### 2.1 The NSM Security Foundation

The Noisy Storage Model security relies on the following physical assumption:

$$
\text{Security} \Leftrightarrow Q_{\text{channel}} < Q_{\text{storage}} \land C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

Where:
- $Q_{\text{channel}}$: Quantum bit error rate experienced by honest parties
- $Q_{\text{storage}}$: Noise affecting adversary's quantum storage
- $C_{\mathcal{N}}$: Classical capacity of adversary's storage channel
- $\nu$: Storage rate (fraction of qubits storable)

### 2.2 Current Gap Analysis

There are **two** relevant simulation configuration paths in the SquidASM ecosystem:

1. **Stack runner path (Caligo uses this):** Caligo builds `StackNetworkConfig` and relies on `netsquid_netbuilder` link/device configs.
2. **NetQASM-interface path (not used by Caligo):** SquidASM can also run a `netqasm.runtime.interface.config.NetworkConfig` through `NetSquidNetwork`.

The NetQASM-interface path provides only **simplified** link noise modeling:

```python
# From squidasm/sim/network/network.py - Current approach
noise = 1 - link.fidelity  # Single parameter!
model_params = LinearDepolariseModelParameters(
    cycle_time=state_delay, prob_success=1, prob_max_mixed=noise
)
```

**Caligo-specific gap:** although the stack runner path supports realistic modeling (notably heralded double-click with detector efficiency and dark counts), Caligo’s current network builder configuration only uses a depolarising link configured via a single `channel_fidelity` value and does not yet plumb $\eta$, $P_{\text{dark}}$, or $e_{\text{det}}$ into the link/device configs.

**Missing NSM-Critical Parameters:**

| NSM Parameter | Physical Meaning | Current Status |
|--------------|------------------|----------------|
| Storage noise $r$ | Depolarizing parameter during $\Delta t$ | ❌ Not modeled |
| Storage rate $\nu$ | Fraction of storable qubits | ❌ Not modeled |
| Wait time $\Delta t$ | Adversary storage decoherence time | ❌ Not enforced |
| Detection efficiency $\eta$ | Combined detector efficiency | ❌ Implicit only |
| Dark count rate | Spurious detection probability | ❌ Not exposed |
| Source quality $\mu$ | EPR pair fidelity contribution | ⚠️ Partial (fidelity) |

### 2.3 Why This Matters

Without proper NSM parameter enforcement:

1. **Security proofs are invalid**: The "strictly less" condition cannot be verified
2. **QBER calculations are incorrect**: Missing detection efficiency and dark counts
3. **Timing attacks are possible**: No $\Delta t$ enforcement in simulation
4. **Robustness analysis is flawed**: Cannot sweep NSM parameters empirically

---

## 3. Squid Stack Noise Model Architecture

### 3.1 Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SQUID STACK NOISE HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SquidASM (Application)                       │    │
│  │  • StackNetworkConfig (YAML)                                        │    │
│  │  • Stack runner: StackNetworkConfig → netsquid-netbuilder configs   │    │
│  │  • NetQASM-interface path exists but is out-of-scope for Caligo     │    │
│  │  • QDevice (memory_noise_models, phys_instructions)                 │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     netsquid_netbuilder (Config)                    │    │
│  │  • DepolariseQLinkConfig → DepolariseModelParameters                │    │
│  │  • HeraldedDoubleClickQLinkConfig → DoubleClickModelParameters      │    │
│  │  • GenericQDeviceConfig (T1, T2, gate_depolar_prob)                 │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      netsquid_magic (EPR Layer)                     │    │
│  │  • MagicDistributor (state_delay, label_delay)                      │    │
│  │  • IModelParameters subclasses:                                     │    │
│  │    - DepolariseModelParameters (prob_max_mixed, prob_success)       │    │
│  │    - DoubleClickModelParameters (detector_eff, dark_count, ...)     │    │
│  │  • State delivery samplers are internal (configured via netbuilder) │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       NetSquid (Simulation Core)                    │    │
│  │  • QuantumErrorModel (base class)                                   │    │
│  │    - DepolarNoiseModel (depolar_rate, time_independent)             │    │
│  │    - DephaseNoiseModel (dephase_rate)                               │    │
│  │    - T1T2NoiseModel (T1, T2)                                        │    │
│  │    - FibreLossModel (p_loss_init, p_loss_length)                    │    │
│  │  • QuantumProcessor (phys_instructions, memory_noise_models)        │    │
│  │  • StateSampler (qreprs, probabilities)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Noise Injection Points Identified

#### 3.2.1 EPR Pair Generation (Channel Noise)

**Location (Caligo path):** `netsquid_netbuilder` qlink modules construct a `netsquid_magic.MagicDistributor` based on a selected `IQLinkConfig`.

**Mechanism:** Caligo controls EPR generation noise by selecting the qlink type and fields in `StackNetworkConfig.links[*]`:

- `typ="depolarise"` with `DepolariseQLinkConfig` fields (fidelity→depolarising mixture internally)
- `typ="heralded"` (alias for `"heralded-double-click"`) with `HeraldedDoubleClickQLinkConfig` fields including `detector_efficiency` and `dark_count_probability`

Internally, `netsquid_magic` uses state delivery samplers to realize the configured model parameters; Caligo should treat those internals as implementation details and configure behavior via the netbuilder/stack-runner config surface.

**NSM Relevance:** This is where $Q_{\text{channel}}$ is realized in the simulation. For NSM parameter enforcement in Caligo, the design is to use the existing netbuilder heralded model rather than adding custom sampler factories in Caligo.

#### 3.2.2 Quantum Memory (Honest Party Storage)

**Location:** `netsquid_netbuilder/modules/qdevices/generic.py`

**Mechanism:** Per-qubit T1T2NoiseModel applied during storage:

```python
electron_qubit_noise = T1T2NoiseModel(T1=qdevice_cfg.T1, T2=qdevice_cfg.T2)
mem_noise_models = [electron_qubit_noise] * qdevice_cfg.num_qubits
```

**NSM Relevance:** Models **honest party memory decoherence** (NOT adversary storage noise).

#### 3.2.3 Gate Operations (Measurement Noise)

**Location:** `netsquid_netbuilder/modules/qdevices/generic.py`

**Mechanism:** Depolarizing noise per gate via PhysicalInstruction:

```python
single_qubit_gate_noise = DepolarNoiseModel(
    depolar_rate=qdevice_cfg.single_qubit_gate_depolar_prob,
    time_independent=True,
)
```

**NSM Relevance:** Contributes to $e_{\text{det}}$ (intrinsic detector error).

#### 3.2.4 **CRITICAL GAP: Adversary Storage Noise**

**Status:** ❌ NOT MODELED

The NSM adversary's storage channel $\mathcal{F}_{\Delta t}$ is **not simulated**. The simulator models only honest-party behavior.

**Resolution Strategy:** NSM storage noise must be computed **analytically** in security bounds, NOT simulated. The simulation validates the channel noise $Q_{\text{channel}}$.

---

## 4. NSM Parameter to Physical Model Mapping

### 4.1 Literature Foundations

The mapping from NSM theoretical parameters to physical simulation models is grounded in four key references:

| Reference | Focus | Key Parameters |
|-----------|-------|----------------|
| **Schaffner et al. 2009** | Fundamental NSM theory | $r$, QBER thresholds |
| **Wehner et al. 2010** | Practical implementation | $P_{\text{sent}}^n$, $P_{\text{B,click}}^h$, $e_{\text{det}}$ |
| **Erven et al. 2014** | Experimental realization | $\mu$, $\eta$, $P_{\text{dark}}$, finite-size effects |
| **Lemus et al. 2025** | Practical QOT performance | Commitments, reconciliation |

### 4.2 NSM Parameter Definitions

#### 4.2.1 Adversary Storage Model (Schaffner 2009)

The NSM models adversary storage as a depolarizing channel:

$$
\mathcal{N}_r(\rho) = r \cdot \rho + (1-r) \cdot \frac{\mathbf{I}}{d}
$$

Where:
- $r \in [0, 1]$: **Preservation probability** (higher = less noise = adversary advantage)
- $d = 2$: Qubit dimension
- $r = 1$: Perfect storage (NO security)
- $r = 0$: Complete depolarization (maximum security)

**NetSquid Mapping:**
```python
# NSM r → NetSquid depolar_rate
depolar_rate = 1 - r  # NetSquid uses depolarization probability
```

#### 4.2.2 Storage Rate (Schaffner 2009)

$$
\nu \in [0, 1]: \text{Fraction of qubits adversary can store}
$$

**Security Condition:**
$$
C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

Where $C_{\mathcal{N}}$ is the classical capacity of the depolarizing channel.

**Simulation Implication:** This is an **analytical bound**, not directly simulated. The simulation enforces $\Delta t$ timing; $\nu$ is used in security bound calculations.

#### 4.2.3 Wait Time $\Delta t$ (Erven 2014)

From Erven et al.:
> "Both parties now wait a time, Δt, long enough for any stored quantum information of a dishonest party to decohere."

**NetSquid Mapping:**
```python
# Δt enforced via simulation time
import netsquid as ns

class TimingBarrier:
    def __init__(self, delta_t_ns: float):
        self.delta_t = delta_t_ns
        self.start_time = None
    
    def mark_quantum_complete(self):
        self.start_time = ns.sim_time()
    
    def wait_complete(self) -> bool:
        return (ns.sim_time() - self.start_time) >= self.delta_t
```

### 4.3 Channel Noise Parameters (Wehner 2010, Erven 2014)

#### 4.3.1 Source Parameters

| Parameter | Symbol | Erven Value | Physical Meaning |
|-----------|--------|-------------|------------------|
| Mean photon number | $\mu$ | $3.145 \times 10^{-5}$ | Photons per coherence time |
| Transmittance | $\eta$ | $0.0150$ | Combined detection efficiency |
| Detection error | $e_{\text{det}}$ | $0.0093$ | Intrinsic error rate |
| Dark count prob | $P_{\text{dark}}$ | $1.50 \times 10^{-8}$ | Spurious clicks per pulse |

#### 4.3.2 Total Channel QBER

From Wehner et al. 2010:

$$
P_{\text{B,err}}^h = \text{function}(e_{\text{det}}, P_{\text{dark}}, \eta)
$$

The total effective error rate for honest parties includes:
1. **Detection error** $e_{\text{det}}$: Misalignment and apparatus imperfections
2. **Dark counts**: Random clicks without signal photons
3. **Multi-photon emissions**: Weak coherent source statistics

**Simulation Strategy:** These must be modeled in the EPR state generation layer.

### 4.4 NSM to NetSquid Parameter Translation Table

| NSM Parameter | Physical Meaning | NetSquid Component | Implementation |
|--------------|------------------|-------------------|----------------|
| $r$ | Storage noise parameter | NOT SIMULATED | Security bound calculation |
| $\nu$ | Storage rate | NOT SIMULATED | Security bound calculation |
| $\Delta t$ | Wait time | `ns.sim_time()` | TimingBarrier class |
| $\mu$ (source) | Source quality knob | Link-level effective parameters | Represented indirectly via `channel_fidelity` / `emission_fidelity` (no separate WCP model in Caligo) |
| $\eta$ | Detection efficiency | Heralded link model parameters | `HeraldedDoubleClickQLinkConfig.detector_efficiency` and/or loss fields (`p_loss_init`, `p_loss_length`, `length`) |
| $e_{\text{det}}$ | Intrinsic error | Gate noise | DepolarNoiseModel |
| $P_{\text{dark}}$ | Dark count probability | Heralded link model parameters | `HeraldedDoubleClickQLinkConfig.dark_count_probability` |
| QBER | Total error rate | Composite | All above combined |

### 4.5 The "Strictly Less" Condition

**Schaffner 2009 Theorem:**
> "Security can be obtained as long as the quantum bit-error rate of the channel does not exceed **11%** and the noise on the channel is **strictly less** than the quantum storage noise."

$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1 - r}{2}
$$

**Critical Implication:** The simulation must accurately measure $Q_{\text{channel}}$ to validate this condition at runtime.

---

## 5. Caligo Simulation Layer Architecture

### 5.1 Current Implementation Status

The Caligo simulation layer (located in `caligo/simulation/`) provides a well-structured foundation for NSM parameter enforcement:

| Module | Purpose | NSM Relevance |
|--------|---------|---------------|
| `constants.py` | Literature-derived values | Erven parameters, QBER thresholds |
| `physical_model.py` | NSM/Channel parameter dataclasses | Parameter validation and mappings |
| `noise_models.py` | Wrapper noise models | QBER computation, profile factories |
| `timing.py` | TimingBarrier implementation | $\Delta t$ enforcement via `ns.sim_time()` |
| `network_builder.py` | StackNetworkConfig factory | SquidASM injection point |

### 5.2 Key Dataclasses

#### 5.2.1 `NSMParameters` (physical_model.py)

```python
@dataclass(frozen=True)
class NSMParameters:
    """
    Noisy Storage Model parameters for simulation and security analysis.
    
    Invariants:
    - INV-NSM-001: storage_noise_r ∈ [0, 1]
    - INV-NSM-002: storage_rate_nu ∈ [0, 1]
    - INV-NSM-003: storage_dimension_d == 2
    - INV-NSM-004: delta_t_ns > 0
    - INV-NSM-005: channel_fidelity ∈ (0.5, 1]
    - INV-NSM-006: detection_eff_eta ∈ (0, 1]
    """
    storage_noise_r: float       # NSM r parameter
    storage_rate_nu: float       # NSM ν parameter
    delta_t_ns: float            # Wait time Δt
    channel_fidelity: float      # EPR pair fidelity F
    detection_eff_eta: float = 1.0
    detector_error: float = 0.0
    dark_count_prob: float = 0.0
    storage_dimension_d: int = 2
```

**Derived Properties:**
- `depolar_prob`: NetSquid depolarization rate = $1 - r$
- `qber_channel`: Full Erven formula QBER computation
- `storage_capacity`: $C_{\mathcal{N}} = 1 - h(p)$
- `storage_security_satisfied`: Checks $C_{\mathcal{N}} \cdot \nu < 1/2$

#### 5.2.2 `ChannelNoiseProfile` (noise_models.py)

```python
@dataclass(frozen=True)
class ChannelNoiseProfile:
    """
    Aggregate noise profile for the trusted quantum channel.
    
    Invariants:
    - INV-CNP-001: source_fidelity ∈ (0.5, 1]
    - INV-CNP-002: detector_efficiency ∈ (0, 1]
    - INV-CNP-003: detector_error ∈ [0, 0.5]
    - INV-CNP-004: dark_count_rate ∈ [0, 1]
    - INV-CNP-005: transmission_loss ∈ [0, 1)
    """
    source_fidelity: float
    detector_efficiency: float
    detector_error: float
    dark_count_rate: float
    transmission_loss: float = 0.0
```

**Computed QBER:** Uses `caligo.utils.math.compute_qber_erven()` for exact literature formula.

#### 5.2.3 `ChannelParameters` (physical_model.py)

```python
@dataclass(frozen=True)
class ChannelParameters:
    """Physical channel parameters for quantum link."""
    length_km: float = 0.0
    attenuation_db_per_km: float = 0.2
    speed_of_light_km_s: float = 200_000.0
    t1_ns: float = TYPICAL_T1_NS     # 10 ms
    t2_ns: float = TYPICAL_T2_NS     # 1 ms
    cycle_time_ns: float = TYPICAL_CYCLE_TIME_NS
```

### 5.3 TimingBarrier Implementation

The `TimingBarrier` class (`timing.py`) implements NSM $\Delta t$ enforcement as a state machine:

```
IDLE ──[mark_quantum_complete()]──► WAITING ──[wait_delta_t()]──► READY
```

**Key Features:**
- Integrates with NetSquid discrete-event simulation via `ns.sim_time()`
- State machine prevents basis revelation before $\Delta t$ elapses
- Strict mode raises `TimingViolationError` on violations
- Yields control to simulator during wait period

```python
# From caligo/simulation/timing.py (simplified)
class TimingBarrier:
    def __init__(self, delta_t_ns: float, strict_mode: bool = True):
        self._delta_t_ns = delta_t_ns
        self._state = TimingBarrierState.IDLE
        self._quantum_complete_time: Optional[float] = None
    
    def mark_quantum_complete(self) -> None:
        self._quantum_complete_time = _get_sim_time()  # ns.sim_time()
        self._state = TimingBarrierState.WAITING
    
    def wait_delta_t(self) -> Generator[EventExpression, None, None]:
        """Yield to simulator for Δt nanoseconds."""
        remaining = self._delta_t_ns - (ns.sim_time() - self._quantum_complete_time)
        if remaining > 0:
            yield from self.context.connection.wait(remaining)
        self._state = TimingBarrierState.READY
```

### 5.4 Protocol Integration

The `CaligoProgram` base class (`protocol/base.py`) integrates NSM enforcement:

```python
class CaligoProgram(Program, ABC):
    def __init__(self, params: ProtocolParameters) -> None:
        self._params = params
        # TimingBarrier created from NSMParameters.delta_t_ns
        self._timing_barrier = TimingBarrier(delta_t_ns=params.nsm_params.delta_t_ns)
```

**Alice's Protocol Flow (alice.py):**
```python
# After quantum measurements complete:
self._timing_barrier.mark_quantum_complete()

# Wait for Δt (yields control to discrete-event simulator)
yield from self._timing_barrier.wait_delta_t()

# NOW safe to reveal basis choices
yield from self._ordered_socket.send(basis_message)
```

---

## 6. Integration Architecture

### 6.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALIGO NSM INTEGRATION DATA FLOW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐      ┌─────────────────────┐                      │
│   │   User Config       │─────►│   NSMParameters     │                      │
│   │   (YAML/Python)     │      │   + validation      │                      │
│   └─────────────────────┘      └─────────┬───────────┘                      │
│                                          │                                  │
│                 ┌────────────────────────┼────────────────────────┐         │
│                 │                        │                        │         │
│                 ▼                        ▼                        ▼         │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│   │  CaligoNetwork      │  │    TimingBarrier    │  │  SecurityAnalyzer   │ │
│   │  Builder            │  │    (Δt enforcement) │  │  (r, ν bounds)      │ │
│   └─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘ │
│             │                        │                        │             │
│             ▼                        ▼                        ▼             │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│   │  StackNetworkConfig │  │  ProtocolParameters │  │  Security Bounds    │ │
│   │  (SquidASM)         │  │  (program params)   │  │  (key rate calc)    │ │
│   └─────────┬───────────┘  └─────────┬───────────┘  └─────────────────────┘ │
│             │                        │                                      │
│             ▼                        ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SquidASM Simulation Runtime                      │   │
│   │  • NetSquidNetwork with configured noise models                     │   │
│   │  • CaligoProgram instances (Alice, Bob)                             │   │
│   │  • TimingBarrier enforced via ns.sim_time()                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Current Noise Model Injection

The `CaligoNetworkBuilder` class provides the primary injection point:

```python
# From caligo/simulation/network_builder.py
class CaligoNetworkBuilder:
    def build_two_node_network(self, alice_name, bob_name, num_qubits):
        # Noise type selection based on fidelity
        if self._nsm_params.channel_fidelity == 1.0:
            link_noise_type = "perfect"
            link_cfg_payload = None
        else:
            link_noise_type = "depolarise"  # <-- SquidASM built-in
            link_cfg_payload = {"fidelity": fidelity_param}
        
        link_cfg = LinkConfig(
            stack1=alice_name,
            stack2=bob_name,
            typ=link_noise_type,
            cfg=link_cfg_payload,
        )
```

**Limitation:** Uses only `fidelity` parameter; missing $\eta$, $P_{\text{dark}}$, $e_{\text{det}}$.

### 6.3 Enhanced Noise Model Requirements

To fully implement NSM parameter enforcement, the following enhancements are needed:

#### 6.3.1 Detection Efficiency Modeling

**Current:** Detection efficiency $\eta$ is implicit in fidelity mapping.

**Required (Caligo path):** Use the existing `netsquid_netbuilder` heralded double-click link model via the stack runner link config.

```python
# Proposed extension (stack runner path)
from squidasm.run.stack.config import HeraldedLinkConfig

def build_heralded_link_config(
    detection_efficiency: float,
    dark_count_prob: float,
    channel_fidelity: float,
    *,
    length_km: float = 0.0,
    p_loss_length: float = 0.0,
    p_loss_init: float = 0.0,
    speed_of_light_km_s: float = 200_000.0,
) -> HeraldedLinkConfig:
    """Create heralded-double-click link config (no custom registration required)."""

    return HeraldedLinkConfig(
        length=float(length_km),
        p_loss_length=float(p_loss_length),
        p_loss_init=float(p_loss_init),
        speed_of_light=float(speed_of_light_km_s),
        detector_efficiency=float(detection_efficiency),
        dark_count_probability=float(dark_count_prob),
        visibility=1.0,
        emission_fidelity=float(channel_fidelity),
        emission_duration=0.0,
        collection_efficiency=1.0,
        num_multiplexing_modes=1,
    )
```

#### 6.3.2 Dark Count Injection

**Current:** Not modeled.

**Required:** Probabilistic insertion of spurious detection events.

**Approach (Caligo path):** Model dark counts by selecting the heralded-double-click link model and setting `dark_count_probability`. Avoid injecting synthetic “dark count outcomes” in Caligo, because it risks double-counting and bypasses the dependency’s intended semantics.

#### 6.3.3 Measurement Error Modeling

**Current:** Via `GenericQDeviceConfig.single_qubit_gate_depolar_prob`.

**Required:** Map $e_{\text{det}}$ to gate noise:

```python
qdevice_cfg.single_qubit_gate_depolar_prob = detector_error * 2  # Approximate
```

### 6.4 Netbuilder Device Config Factory Functions

For the stack runner path, prefer constructing `netsquid_netbuilder`/SquidASM device config objects rather than building NetSquid noise models directly in Caligo.

The primary knobs are:

- memory decoherence: `GenericQDeviceConfig.T1` / `GenericQDeviceConfig.T2`
- gate noise: `GenericQDeviceConfig.single_qubit_gate_depolar_prob` (and optionally `two_qubit_gate_depolar_prob`)

**Note:** Adversary storage noise parameters $(r, \nu)$ remain analytical and should not be mapped to honest-party device configs.

---

## 7. Validation Strategy

### 7.1 NSM Condition Verification

The simulation must validate the fundamental NSM security condition:

$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1 - r}{2}
$$

**Runtime Check:**
```python
def verify_nsm_security_condition(
    measured_qber: float,
    nsm_params: NSMParameters,
) -> bool:
    """
    Verify that measured channel QBER satisfies NSM security condition.
    
    Parameters
    ----------
    measured_qber : float
        Empirically measured QBER from protocol execution.
    nsm_params : NSMParameters
        Configured NSM parameters.
    
    Returns
    -------
    bool
        True if Q_channel < (1 - r) / 2.
    
    Raises
    ------
    SecurityError
        If QBER exceeds storage noise bound.
    """
    storage_noise_bound = (1.0 - nsm_params.storage_noise_r) / 2.0
    
    if measured_qber >= storage_noise_bound:
        raise SecurityError(
            f"NSM security violated: Q_channel={measured_qber:.4f} >= "
            f"Q_storage={(storage_noise_bound):.4f}"
        )
    
    return True
```

### 7.2 QBER Measurement Validation

Compare empirical QBER against theoretical prediction:

```python
def validate_qber_measurement(
    measured_qber: float,
    expected_qber: float,
    tolerance: float = 0.01,
) -> bool:
    """
    Validate that measured QBER matches theoretical prediction.
    
    Uses relative tolerance for comparison.
    """
    if abs(measured_qber - expected_qber) > tolerance:
        logger.warning(
            f"QBER deviation: measured={measured_qber:.4f}, "
            f"expected={expected_qber:.4f}, diff={abs(measured_qber - expected_qber):.4f}"
        )
        return False
    return True
```

### 7.3 Timing Enforcement Validation

The `TimingBarrier` includes compliance tracking:

```python
# Post-protocol validation
if not barrier.timing_compliant:
    raise SecurityError("Timing constraint violated during protocol execution")

# Verify actual wait duration
actual_wait = barrier.actual_wait_duration_ns
expected_wait = nsm_params.delta_t_ns
if actual_wait < expected_wait * 0.99:  # 1% tolerance for simulation precision
    raise SecurityError(f"Insufficient wait time: {actual_wait} < {expected_wait}")
```

### 7.4 Test Coverage Requirements

| Test Category | Requirement | Priority |
|--------------|-------------|----------|
| Parameter validation | All invariants (INV-NSM-*, INV-CNP-*) | P0 |
| QBER computation | Match Erven formula to 6 decimal places | P0 |
| Timing enforcement | TimingBarrier state machine correctness | P0 |
| Security condition | Q_channel < Q_storage verification | P0 |
| Network builder | StackNetworkConfig generation | P1 |
| Noise injection | EPR fidelity matches configuration | P1 |
| Factory functions | NetSquid model creation | P2 |

---

### 7.5 Integration Testing (SquidASM + NetSquid Required)

This section specifies **integration tests** that validate NSM parameter enforcement by initializing the simulation *exactly as production does* via SquidASM’s **stack runner path** and then inspecting the **generated netbuilder configuration** and the **instantiated link/device noise models**.

This is intentionally **not** done by directly constructing a NetSquid `DepolarNoiseModel`, `T1T2NoiseModel`, etc. in isolation. Those unit-level tests are useful, but they do not prove that Caligo’s configuration successfully flows through:

1. `caligo.simulation.network_builder` → `squidasm.run.stack.config.StackNetworkConfig`
2. `squidasm.run.stack.config._convert_stack_network_config` → `netsquid_netbuilder.network_config.NetworkConfig`
3. `squidasm.run.stack.build.create_stack_network_builder().build(...)` → instantiated `MagicLinkLayerProtocolWithSignaling`, `MagicDistributor`, and `QuantumProcessor` objects

#### 7.5.1 Current Test Suite: What Exists and What’s Missing

Caligo already has good unit tests for simulation-adjacent logic:

- `caligo/tests/test_simulation/test_network_builder.py` verifies that `CaligoNetworkBuilder` can build a `StackNetworkConfig` (or raises a clean error if SquidASM is missing).
- `caligo/tests/test_simulation/test_timing.py` exercises `TimingBarrier` and correctly distinguishes “no NetSquid engine” vs. real simulation time.
- `caligo/tests/e2e/test_parallel_simulation.py` and `caligo/tests/integration/test_parallel_protocol.py` validate end-to-end data shape and QBER trends.

However, these do **not** yet verify that the simulation pipeline is initialized with:

- the correct *link model type* (e.g., `depolarise` vs `heralded-double-click`),
- the correct *model parameters* (e.g., `dark_count_probability`, `detector_efficiency`, `prob_max_mixed`), and
- the correct *device noise hooks* (gate depolarization and memory T1/T2) *as installed into the instantiated NetSquid components*.

That missing coverage is critical for NSM enforcement, because otherwise it is possible for Caligo to compute analytical QBER/security values while the underlying Squid stack silently runs with a different physical model.

#### 7.5.2 Integration Test Invariants (What We Must Assert)

For each supported link/device configuration, integration tests must assert **all** of the following:

1. **Translation invariant (config → netbuilder config):** the `StackNetworkConfig` produced by Caligo converts to a `netsquid_netbuilder.network_config.NetworkConfig` whose:
   - `qlinks[*].typ` matches the expected netbuilder model name, and
   - `qlinks[*].cfg` contains the expected fields (and types), not silently dropped/renamed values.

2. **Instantiation invariant (netbuilder config → built network):** building the network via `create_stack_network_builder().build(...)` produces:
   - a `MagicLinkLayerProtocolWithSignaling` for the link,
   - a `MagicDistributor` with the expected distributor subclass,
   - a `MagicDistributor` model-parameter object of the expected type with expected values.

3. **Device invariant:** the node qdevices built by netbuilder contain:
   - `T1T2NoiseModel(T1=..., T2=...)` installed as memory noise,
   - `DepolarNoiseModel(depolar_rate=...)` installed on the relevant physical instructions.

The objective is to validate “**the right model is installed**” rather than only validating “a value was computed somewhere”.

#### 7.5.3 Recommended Harness Pattern (Production-Equivalent Initialization)

The stack runner uses this production path:

- `squidasm.run.stack.run.run(config=StackNetworkConfig, ...)` which internally calls:
  - `squidasm.run.stack.config._convert_stack_network_config`, then
  - `squidasm.run.stack.build.create_stack_network_builder().build(...)`, then
  - NetSquid reset + protocol wiring.

For *configuration/noise-model introspection*, integration tests should avoid running full protocol programs and instead:

1. create a `StackNetworkConfig` (preferably using Caligo’s builder),
2. convert it with `_convert_stack_network_config`,
3. build the network with `create_stack_network_builder().build(...)`, and
4. inspect:
   - `network.qlinks[("Alice", "Bob")]` to access the link protocol,
   - `network.end_nodes["Alice"].qmemory` to access the NetSquid `QuantumProcessor`.

This gives you production-equivalent initialization without adding NetQASM instruction-set fragility.

**Pytest gating:** these integration tests must require SquidASM and NetSquid:

```python
import pytest

pytest.importorskip("squidasm")
pytest.importorskip("netsquid")
```

If a test also runs programs via `squidasm.run.stack.run.run`, add a defensive guard for NetQASM 2.x incompatibility (mirroring the note in `caligo/tests/e2e/test_phase_e_protocol.py`).

#### 7.5.4 Concrete Test Cases (PR-ready)

Below are concrete tests that should be added under `caligo/tests/integration/` (or similar). They are written as **skeletons** but reference the *actual* SquidASM/netbuilder objects and fields.

##### Test A: Depolarise link config survives stack-runner conversion

Goal: verify Caligo’s `StackNetworkConfig` produces the expected netbuilder `QLinkConfig` with the correct type and cfg fields.

```python
import math
import pytest

pytest.importorskip("squidasm")
pytest.importorskip("netsquid")

from squidasm.run.stack.config import _convert_stack_network_config

from caligo.simulation.network_builder import CaligoNetworkBuilder
from caligo.simulation.physical_model import NSMParameters


def test_stackrunner_conversion_depolarise_preserves_fields() -> None:
    nsm = NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.01,
        delta_t_ns=1_000_000,
        channel_fidelity=0.92,
    )
    stack_cfg = CaligoNetworkBuilder(nsm).build_two_node_network(
        alice_name="Alice",
        bob_name="Bob",
        num_qubits=4,
    )

    netbuilder_cfg = _convert_stack_network_config(stack_cfg)

    assert len(netbuilder_cfg.qlinks) == 1
    qlink = netbuilder_cfg.qlinks[0]
    assert qlink.typ == "depolarise"

    # cfg may be a dict or a DepolariseQLinkConfig instance; accept both.
    cfg = qlink.cfg
    fidelity = cfg["fidelity"] if isinstance(cfg, dict) else cfg.fidelity
    prob_success = cfg["prob_success"] if isinstance(cfg, dict) else cfg.prob_success
    t_cycle = cfg["t_cycle"] if isinstance(cfg, dict) else cfg.t_cycle

    assert math.isclose(float(fidelity), 0.92)
    assert math.isclose(float(prob_success), 1.0)
    assert float(t_cycle) > 0
```

##### Test B: Depolarise link builds a distributor with correct model parameters

Goal: verify that building the network produces the expected `MagicDistributor` subclass and that the *model parameters in the instantiated distributor* match the requested fidelity mapping.

```python
import math
import pytest

pytest.importorskip("squidasm")
pytest.importorskip("netsquid")

from netsquid_netbuilder.util.fidelity import fidelity_to_prob_max_mixed
from netsquid_magic.magic_distributor import DepolariseWithFailureMagicDistributor

from squidasm.run.stack.build import create_stack_network_builder
from squidasm.run.stack.config import _convert_stack_network_config

from caligo.simulation.network_builder import CaligoNetworkBuilder
from caligo.simulation.physical_model import NSMParameters


def test_stackrunner_build_depolarise_installs_expected_magic_model_params() -> None:
    nsm = NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.01,
        delta_t_ns=1_000_000,
        channel_fidelity=0.92,
    )
    stack_cfg = CaligoNetworkBuilder(nsm).build_two_node_network(num_qubits=2)
    netbuilder_cfg = _convert_stack_network_config(stack_cfg)

    builder = create_stack_network_builder()
    network = builder.build(netbuilder_cfg)

    link = network.qlinks[("Alice", "Bob")]
    dist = link.magic_distributor
    assert isinstance(dist, DepolariseWithFailureMagicDistributor)

    # netsquid_magic stores model params internally (list per merged distributor).
    model = dist._model_parameters[0]

    assert math.isclose(float(model.prob_success), 1.0)
    assert math.isclose(float(model.prob_max_mixed), float(fidelity_to_prob_max_mixed(0.92)))
```

##### Test C: Heralded-double-click link installs dark counts and detector efficiency

Goal: verify that selecting the heralded link path installs a `DoubleClickMagicDistributor` whose model params include the requested `dark_count_probability` and `detector_efficiency`.

```python
import math
import pytest

pytest.importorskip("squidasm")
pytest.importorskip("netsquid")

from netsquid_magic.magic_distributor import DoubleClickMagicDistributor
from squidasm.run.stack.build import create_stack_network_builder
from squidasm.run.stack.config import (
    StackNetworkConfig,
    StackConfig,
    LinkConfig,
    GenericQDeviceConfig,
    HeraldedLinkConfig,
    _convert_stack_network_config,
)


def test_stackrunner_build_heralded_double_click_installs_detector_and_dark_counts() -> None:
    alice = StackConfig(
        name="Alice",
        qdevice_typ="generic",
        qdevice_cfg=GenericQDeviceConfig.perfect_config(num_qubits=2),
    )
    bob = StackConfig(
        name="Bob",
        qdevice_typ="generic",
        qdevice_cfg=GenericQDeviceConfig.perfect_config(num_qubits=2),
    )

    heralded_cfg = HeraldedLinkConfig(
        length=10.0,
        p_loss_length=0.2,
        p_loss_init=0.0,
        speed_of_light=200_000.0,
        detector_efficiency=0.015,
        dark_count_probability=1.5e-8,
        visibility=1.0,
        emission_fidelity=0.99,
        emission_duration=0.0,
        collection_efficiency=1.0,
        num_multiplexing_modes=1,
    )

    link = LinkConfig(stack1="Alice", stack2="Bob", typ="heralded", cfg=heralded_cfg)
    stack_cfg = StackNetworkConfig(stacks=[alice, bob], links=[link])
    netbuilder_cfg = _convert_stack_network_config(stack_cfg)

    builder = create_stack_network_builder()
    network = builder.build(netbuilder_cfg)

    link = network.qlinks[("Alice", "Bob")]
    dist = link.magic_distributor
    assert isinstance(dist, DoubleClickMagicDistributor)

    model = dist._model_parameters[0]
    assert math.isclose(float(model.detector_efficiency), 0.015)
    assert math.isclose(float(model.dark_count_probability), 1.5e-8)
```

Notes:

- The heralded model expands “global” length/loss parameters into side-specific fields (`*_A`, `*_B`) during preprocessing; tests should assert on stable fields like `detector_efficiency`/`dark_count_probability` and (optionally) check `length_A == length_B == length/2`.
- Keep `collection_efficiency=1.0` in tests to avoid the model’s internal absorption step altering `p_loss_init`.

##### Test D: Device noise is installed into NetSquid `QuantumProcessor`

Goal: verify that gate depolarization and memory T1/T2 are applied to the built device (not merely present in a config object).

```python
import pytest

pytest.importorskip("squidasm")
pytest.importorskip("netsquid")

from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel

from squidasm.run.stack.build import create_stack_network_builder
from squidasm.run.stack.config import (
    StackNetworkConfig,
    StackConfig,
    LinkConfig,
    DepolariseLinkConfig,
    GenericQDeviceConfig,
    _convert_stack_network_config,
)


def test_stackrunner_build_installs_qdevice_gate_and_memory_noise() -> None:
    qdev = GenericQDeviceConfig.perfect_config(num_qubits=3)
    qdev.T1 = 20_000_000
    qdev.T2 = 2_000_000
    qdev.single_qubit_gate_depolar_prob = 0.02
    qdev.two_qubit_gate_depolar_prob = 0.03

    alice = StackConfig(name="Alice", qdevice_typ="generic", qdevice_cfg=qdev)
    bob = StackConfig(name="Bob", qdevice_typ="generic", qdevice_cfg=qdev)
    link = LinkConfig(
        stack1="Alice",
        stack2="Bob",
        typ="depolarise",
        cfg=DepolariseLinkConfig(fidelity=0.99, prob_success=1.0, t_cycle=1000.0),
    )
    stack_cfg = StackNetworkConfig(stacks=[alice, bob], links=[link])
    netbuilder_cfg = _convert_stack_network_config(stack_cfg)

    builder = create_stack_network_builder()
    network = builder.build(netbuilder_cfg)

    qproc = network.end_nodes["Alice"].qmemory
    assert qproc is not None

    # Memory noise: list of T1T2NoiseModel
    mem_noise_models = getattr(qproc, "mem_noise_models", None)
    assert mem_noise_models is not None
    assert isinstance(mem_noise_models[0], T1T2NoiseModel)

    # Gate noise: DepolarNoiseModel installed on physical instructions
    phys_instr = getattr(qproc, "phys_instructions", None) or getattr(qproc, "physical_instructions", None)
    assert phys_instr is not None
    depolar_models = [pi.quantum_noise_model for pi in phys_instr if hasattr(pi, "quantum_noise_model")]
    depolar_models = [m for m in depolar_models if isinstance(m, DepolarNoiseModel)]
    assert len(depolar_models) > 0
    assert any(abs(float(m.depolar_rate) - 0.02) < 1e-12 for m in depolar_models)
```

#### 7.5.5 Where These Tests Fit (Repository Layout)

Recommended placement:

- new tests that require SquidASM/NetSquid and perform stack-runner initialization should live under `caligo/tests/integration/` and be marked with `@pytest.mark.integration` (and optionally `@pytest.mark.slow` if they build larger networks).

This keeps unit tests fast and deterministic while providing a dedicated suite that validates “noise model wiring correctness”.

---

## 8. Implementation Roadmap

This section replaces the high-level roadmap with a **PR-ready design document** based on an implementation-level audit of:

- Caligo simulation layer (network builder, timing barrier, QBER computation)
- SquidASM stack runner configuration layer (`StackNetworkConfig` → netsquid-netbuilder)
- `netsquid_netbuilder` link and device config schemas
- `netsquid_magic` entanglement generation model parameter classes
- The separate SquidASM NetQASM-interface simulation path (`NetSquidNetwork._create_link_distributor`)

The goal is to make NSM parameters have **physical consequences** in the simulation in a way that is:

1. aligned with the *actual* dependency APIs,
2. testable and reproducible, and
3. explicitly clear about which NSM parameters are simulated vs. analytically enforced.

---

### 8.1 Scope, Assumptions, and Non-Goals

#### 8.1.1 In Scope (this PR series)

1. **Δt timing enforcement** (already implemented) and its validation.
2. **Trusted channel physical modeling** for $(F, \eta, P_{\text{dark}}, e_{\text{det}})$ by configuring:
   - quantum link type and parameters, and
   - honest-party device gate/memory noise.
3. **Runtime safety checks** ensuring NSM conditions are met:
   - $Q_{\text{channel}} < Q_{\text{storage}} = \frac{1-r}{2}$
   - optional measured-vs-expected QBER consistency check.

#### 8.1.2 Explicit Non-Goals (do not implement in this PR)

1. Simulating an explicit adversary with quantum storage. Under NSM, adversary storage is a **security assumption**; Caligo correctly treats it analytically.
2. Replacing SquidASM or rewriting link-layer protocols.
3. Implementing brand-new custom NetSquid `QuantumErrorModel` classes unless strictly required.
4. Parameter sweep orchestration tooling (future work).

---

### 8.2 Architectural Decision: Which Squid Stack Path Caligo Uses

Caligo currently builds **StackNetworkConfig** objects in `caligo/simulation/network_builder.py` and feeds them to SquidASM’s **stack runner**, which converts them to a `netsquid_netbuilder.network_config.NetworkConfig`.

This is distinct from SquidASM’s NetQASM-interface simulation path (the `NetSquidNetwork` in `squidasm/sim/network/network.py`) which consumes `netqasm.runtime.interface.config.NetworkConfig` and uses `Link.noise_type` + `Link.fidelity` to construct a `MagicDistributor` directly.

**PR decision:** implement all NSM physical enforcement via the **stack runner + netsquid-netbuilder** path, because:

- It already supports realistic heralded models (single/double click) with detector and dark-count parameters.
- It validates config schema via netbuilder config objects.
- It is the path Caligo already uses.

We will keep the NetQASM-interface path out-of-scope for Caligo, but we will document the semantic differences (see 8.3.2).

---

### 8.3 Correctness Notes from Dependency Audit (Critical)

#### 8.3.1 Depolarising “fidelity → mixture” mapping differs by path

There are two common conversions from a target Bell-state fidelity $F$ to a “maximally mixed fraction” $p$:

- **netsquid-netbuilder depolarise link** uses:
  $$p = \frac{4}{3}(1-F)$$
  implemented in `netsquid_netbuilder.util.fidelity.fidelity_to_prob_max_mixed`.

- The **SquidASM NetQASM-interface path** contains a simplified mapping in `squidasm/sim/network/network.py`:
  ```python
  noise = 1 - link.fidelity
  model_params = LinearDepolariseModelParameters(prob_max_mixed=noise, ...)
  ```
  This is a different contract and should not be mixed into the stack runner path.

**Design implication:** in Caligo, treat `NSMParameters.channel_fidelity` as “EPR pair fidelity” and rely on netbuilder’s conversion for `typ="depolarise"`.

#### 8.3.2 “Use DoubleClickModelParameters directly” is not the stack injection mechanism

In the stack runner path, the configuration object is `HeraldedDoubleClickQLinkConfig` (wrapped as `HeraldedLinkConfig` in `squidasm.run.stack.config`). Netbuilder converts that config into a `netsquid_magic.model_parameters.DoubleClickModelParameters` internally.

Therefore the correct injection is:

- `LinkConfig.typ = "heralded"` (alias for `"heralded-double-click"`) and
- `LinkConfig.cfg = HeraldedLinkConfig(...)` or a dict with the same fields.

Not: constructing `DoubleClickModelParameters` and “registering” a custom model.

---

### 8.4 PR-ready Design: NSM Parameter Physical Enforcement

#### 8.4.1 Public configuration surface

The existing dataclasses already provide the parameters we need:

- `NSMParameters`: `channel_fidelity`, `detection_eff_eta`, `detector_error`, `dark_count_prob`, plus analytical `storage_noise_r`, `storage_rate_nu`, and timing `delta_t_ns`.
- `ChannelParameters`: `length_km`, `attenuation_db_per_km`, `speed_of_light_km_s`, and honest memory `t1_ns/t2_ns`.

We will add **one explicit selector** that prevents ambiguous interpretation:

```python
@dataclass(frozen=True)
class ChannelModelSelection:
    """Select which physical link model to use in the simulator."""

    link_model: str = "auto"  # one of: auto | depolarise | heralded-double-click
    eta_semantics: str = "detector_only"  # one of: detector_only | end_to_end
```

Rationale:

- `link_model="auto"` keeps backward compatibility while enabling realistic modeling when needed.
- `eta_semantics` forces us to be explicit whether $\eta$ is mapped to `detector_efficiency` alone or to an end-to-end loss model.

#### 8.4.2 Link model selection rules (deterministic and documented)

Given `NSMParameters` and `ChannelParameters`:

1. If `link_model == "depolarise"`:
   - Use `typ="depolarise"`.
   - Configure fidelity via `DepolariseQLinkConfig.fidelity`.
   - **Do not** attempt to simulate dark counts or loss; those are outside this link model’s semantics.

2. If `link_model == "heralded-double-click"`:
   - Use `typ="heralded"` (alias to double-click).
   - Configure `detector_efficiency` and `dark_count_probability`.
   - Configure loss via `length` and `p_loss_length` (and optional `p_loss_init`).

3. If `link_model == "auto"` (recommended default):
   - Choose `"perfect"` if all are ideal (fidelity=1, eta=1, dark=0, detector_error=0, and optionally length=0).
   - Choose `"depolarise"` if $(\eta=1)$ and $(P_{\text{dark}}=0)$ and user only targets a Bell fidelity.
   - Choose `"heralded"` if $\eta<1$ or $P_{\text{dark}}>0$ or if a nontrivial physical link (length/loss) is requested.

This yields consistent and testable behavior.

#### 8.4.3 Concrete dependency-correct configuration (code skeleton)

Below is the skeleton to implement inside `caligo/simulation/network_builder.py`.

**A) Depolarise link config (netbuilder schema):**

```python
from squidasm.run.stack.config import DepolariseLinkConfig

def _make_depolarise_link_cfg(nsm_params: NSMParameters, channel_params: ChannelParameters) -> DepolariseLinkConfig:
    return DepolariseLinkConfig(
        fidelity=float(nsm_params.channel_fidelity),
        prob_success=1.0,
        t_cycle=float(channel_params.cycle_time_ns),
        random_bell_state=False,
    )
```

This matches netbuilder’s `DepolariseQLinkConfig` fields (fidelity/prob_success/t_cycle).

**B) Heralded double-click link config (netbuilder schema):**

```python
from squidasm.run.stack.config import HeraldedLinkConfig

def _make_heralded_double_click_cfg(
    nsm_params: NSMParameters,
    channel_params: ChannelParameters,
    eta_semantics: str,
) -> HeraldedLinkConfig:
    # Interpretation choice:
    # - detector_only: map eta directly to detector_efficiency and use zero physical loss
    # - end_to_end: encode loss via length/p_loss_length/p_loss_init; optionally keep detector_efficiency fixed

    if eta_semantics == "detector_only":
        length_km = 0.0
        p_loss_length = float(channel_params.attenuation_db_per_km)
        detector_efficiency = float(nsm_params.detection_eff_eta)
        p_loss_init = 0.0
    else:
        # simplest “end-to-end” interpretation: put loss into p_loss_init and keep detector ideal.
        # A more physical mapping can distribute loss across length and detector; that mapping should be documented.
        length_km = float(channel_params.length_km)
        p_loss_length = float(channel_params.attenuation_db_per_km)
        detector_efficiency = 1.0
        p_loss_init = 1.0 - float(nsm_params.detection_eff_eta)

    return HeraldedLinkConfig(
        length=length_km,
        p_loss_length=p_loss_length,
        p_loss_init=p_loss_init,
        speed_of_light=float(channel_params.speed_of_light_km_s),

        detector_efficiency=detector_efficiency,
        dark_count_probability=float(nsm_params.dark_count_prob),
        visibility=1.0,

        # The heralded model produces a fidelity that depends on multiple parameters.
        # We use emission_fidelity as the closest “source quality” knob.
        emission_fidelity=float(nsm_params.channel_fidelity),
        emission_duration=0.0,
        collection_efficiency=1.0,
        num_multiplexing_modes=1,
    )
```

This matches `HeraldedDoubleClickQLinkConfig` and ultimately `DoubleClickModelParameters` field names.

**C) Honest device noise mapping ($e_{\text{det}} \rightarrow$ gate depolar):**

`netsquid_netbuilder.modules.qdevices.generic.GenericQDeviceConfig` includes:

- `single_qubit_gate_depolar_prob`
- `two_qubit_gate_depolar_prob`

which are passed to NetSquid `DepolarNoiseModel(depolar_rate=...)` per instruction.

We will implement a deliberately conservative mapping:

```python
def _map_detector_error_to_gate_depolar(detector_error: float) -> float:
    # Heuristic mapping: treat e_det as a “bit flip” likelihood and map to depolar.
    # Must be validated empirically; kept simple to avoid overfitting.
    return min(1.0, max(0.0, 2.0 * float(detector_error)))
```

**Validation requirement:** demonstrate monotonicity (higher `detector_error` produces higher observed QBER) and tune factor if needed.

---

### 8.5 Validation Framework (PR-ready)

This section turns the prior “planned” items into concrete acceptance criteria and test hooks.

#### 8.5.1 Runtime security checks

Add a small verifier that is invoked after QBER is estimated (post-sifting, pre-reconciliation):

```python
def verify_nsm_security_condition(measured_qber: float, nsm_params: NSMParameters) -> None:
    q_storage = (1.0 - float(nsm_params.storage_noise_r)) / 2.0
    if measured_qber >= q_storage:
        raise SecurityError(
            f"NSM security violated: Q_channel={measured_qber:.4f} >= Q_storage={q_storage:.4f}"
        )
```

Acceptance criteria:

- A run with parameters satisfying $Q_{\text{channel}} < \frac{1-r}{2}$ does not abort.
- A run with parameters violating the inequality aborts deterministically.

#### 8.5.2 QBER prediction vs measurement

Caligo already computes an analytical QBER estimate (Erven-style) via `ChannelNoiseProfile.total_qber`.

Add a validator:

```python
def validate_qber_measurement(measured_qber: float, expected_qber: float, tolerance: float = 0.01) -> bool:
    return abs(measured_qber - expected_qber) <= tolerance
```

Acceptance criteria:

- Under `link_model="depolarise"` with ideal detector (eta=1, dark=0, detector_error=0), measured QBER tracks approximately $(1-F)/2$.
- Under `link_model="heralded-double-click"`, increasing `dark_count_prob` increases measured QBER (directional correctness).

#### 8.5.3 TimingBarrier reporting

TimingBarrier already tracks compliance. Extend (or expose) a single metric:

- `actual_wait_duration_ns`

Acceptance criteria:

- When the protocol waits via `yield from wait_delta_t()`, the actual wait is $\ge 0.99 \Delta t$.

---

### 8.6 Implementation Plan (Work Items and PR Breakdown)

#### PR-1: Network builder physical link selection

Changes:

1. Extend `caligo/simulation/network_builder.py`:
   - Accept a `ChannelModelSelection` (or equivalent) input.
   - Implement deterministic link model selection rules.
   - Construct typed link config objects (`DepolariseLinkConfig`, `HeraldedLinkConfig`) instead of dict payloads.

Acceptance tests:

- Unit test constructing configs for all three modes: perfect, depolarise, heralded.
- Smoke test that the resulting `StackNetworkConfig` passes validation and can be consumed by SquidASM stack runner.

#### PR-2: Honest device noise mapping

Changes:

1. Update `CaligoNetworkBuilder.build_stack_config(..., with_memory_noise=True)` to also map:
   - `nsm_params.detector_error` → `single_qubit_gate_depolar_prob` (and two-qubit) via `_map_detector_error_to_gate_depolar`.

Acceptance tests:

- Monotonicity test: QBER increases as detector_error increases (small statistical tolerance).

#### PR-3: Runtime NSM security verifier

Changes:

1. Add verifier function and call site post-QBER estimation (sifting-to-reconciliation boundary).

Acceptance tests:

- Run aborts when $Q_{\text{channel}} \ge \frac{1-r}{2}$.

#### PR-4: QBER validation and documentation

Changes:

1. Add measured-vs-expected QBER validator (warning-level by default).
2. Document the difference between:
   - `depolarise` fidelity semantics, and
   - heralded model calibration limitations.

---

### 8.7 Risks and Mitigations

1. **Ambiguous meaning of $\eta$ (detector-only vs end-to-end).**
   - Mitigation: explicit `eta_semantics` switch and documentation; do not silently reinterpret.

2. **Heralded model “fidelity” is multi-parameter and not directly equal to `channel_fidelity`.**
   - Mitigation: treat `channel_fidelity` as `emission_fidelity` knob; validate empirically and optionally add a calibration step later.

3. **Statistical noise in short simulations.**
   - Mitigation: tests focus on monotonic trends and use relaxed tolerances for small N.

---

### 8.8 Definition of Done

This roadmap is complete when:

1. Caligo can run with `link_model="depolarise"` and `link_model="heralded-double-click"` using dependency-correct config objects.
2. $\Delta t$ is enforced and measurable in simulation time.
3. NSM inequality aborts runs that violate $Q_{\text{channel}} < \frac{1-r}{2}$.
4. QBER validations are in place (warning or strict mode) and backed by tests.

---

## References

1. **König, R., Wehner, S., & Wullschleger, J.** (2012). Unconditional security from noisy quantum storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.

2. **Schaffner, C., Terhal, B. M., & Wehner, S.** (2009). Robust cryptography in the noisy-quantum-storage model. *Quantum Information & Computation*, 9(11-12), 963-996.

3. **Wehner, S., Curty, M., Schaffner, C., & Lo, H. K.** (2010). Implementation of two-party protocols in the noisy-storage model. *Physical Review A*, 81(5), 052336.

4. **Erven, C., et al.** (2014). An experimental implementation of oblivious transfer in the noisy storage model. *Nature Communications*, 5, 3418.

5. **Lemus, M., Ramos, M. F., Yadav, P., Silva, N. A., Muga, N. J., & Pinto, A. N.** (2025). Practical quantum oblivious transfer with a single photon. *arXiv:2505.03803*.

---

*Document maintained by the Caligo development team. Last updated: December 18, 2025.*

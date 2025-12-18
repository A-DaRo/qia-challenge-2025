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
5. [Custom Noise Model Implementation](#5-custom-noise-model-implementation)
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
2. Injection points for custom noise models in the Squid stack
3. Caligo integration architecture for NSM enforcement
4. Validation methodology for parameter correctness

### 1.2 Key Findings from Codebase Analysis

| Layer | Noise Injection Point | Configurable Parameters |
|-------|----------------------|------------------------|
| **NetSquid Core** | `QuantumErrorModel` subclasses | `depolar_rate`, `T1`, `T2`, `p_loss_*` |
| **netsquid_magic** | `MagicDistributor` + `IModelParameters` | `prob_max_mixed`, `fidelity`, detection params |
| **netsquid_netbuilder** | `IQLinkConfig`, `IQDeviceConfig` | Link fidelity, gate noise, memory T1/T2 |
| **SquidASM** | `NetSquidNetwork._create_link_distributor()` | Linear interpolation via `prob_max_mixed` |

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

The current SquidASM default configuration provides only **simplified** noise modeling:

```python
# From squidasm/sim/network/network.py - Current approach
noise = 1 - link.fidelity  # Single parameter!
model_params = LinearDepolariseModelParameters(
    cycle_time=state_delay, prob_success=1, prob_max_mixed=noise
)
```

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
│  │  • NetSquidNetwork._create_link_distributor()                       │    │
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
│  │  • StateDeliverySamplerFactory → StateSampler                       │    │
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

**Location:** `netsquid_magic/magic_distributor.py` + custom sampler factories

**Mechanism:** The `MagicDistributor` uses a `StateDeliverySamplerFactory` to create noisy EPR states:

```python
# From netsquid_magic/state_delivery_sampler.py
class DepolariseStateSamplerFactory(HeraldedStateDeliverySamplerFactory):
    @staticmethod
    def _delivery_func(model_params: DepolariseModelParameters, **kwargs):
        # Creates (1-p)|Φ+⟩⟨Φ+| + p·I/4 mixed state
        ...
```

**NSM Relevance:** This models the **channel noise** $Q_{\text{channel}}$ but needs extension for:
- Detection efficiency impact on measured QBER
- Dark count contributions
- Loss-induced heralding failures

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
| $\mu$ (source) | EPR fidelity contribution | `prob_max_mixed` | StateDeliverySampler |
| $\eta$ | Detection efficiency | Loss model | Custom FibreLossModel |
| $e_{\text{det}}$ | Intrinsic error | Gate noise | DepolarNoiseModel |
| $P_{\text{dark}}$ | Dark count rate | State sampler | Custom dark count injection |
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

**Required:** Explicit loss model for photon loss before measurement.

```python
# Proposed extension
def build_link_config_with_loss(
    self, 
    detection_efficiency: float,
    dark_count_prob: float,
) -> LinkConfig:
    """
    Create link configuration with explicit detection model.
    
    Uses DoubleClickModelParameters for realistic heralded entanglement.
    """
    from netsquid_magic.model_parameters import DoubleClickModelParameters
    
    model_params = DoubleClickModelParameters(
        detector_efficiency=detection_efficiency,
        dark_count_probability=dark_count_prob,
        visibility=1.0,  # Perfect source visibility
        # ... other parameters
    )
    # Register custom model type with netbuilder
```

#### 6.3.2 Dark Count Injection

**Current:** Not modeled.

**Required:** Probabilistic insertion of spurious detection events.

**Approach:** Extend the `ChannelNoiseProfile` to generate "dark count" measurement outcomes that flip bits with probability $P_{\text{dark}} \cdot (1 - \eta)$.

#### 6.3.3 Measurement Error Modeling

**Current:** Via `GenericQDeviceConfig.single_qubit_gate_depolar_prob`.

**Required:** Map $e_{\text{det}}$ to gate noise:

```python
qdevice_cfg.single_qubit_gate_depolar_prob = detector_error * 2  # Approximate
```

### 6.4 NetSquid Model Factory Functions

The `physical_model.py` module provides factory functions for NetSquid models:

```python
def create_depolar_noise_model(params: NSMParameters) -> DepolarNoiseModel:
    """Create NetSquid DepolarNoiseModel from NSM parameters."""
    return DepolarNoiseModel(
        depolar_rate=params.depolar_prob,  # = 1 - r
        time_independent=True,
    )

def create_t1t2_noise_model(params: ChannelParameters) -> T1T2NoiseModel:
    """Create NetSquid T1T2NoiseModel for memory decoherence."""
    return T1T2NoiseModel(T1=params.t1_ns, T2=params.t2_ns)
```

**Note:** These are for honest party memory, NOT adversary storage simulation.

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

## 8. Implementation Roadmap

### Phase 1: Foundation (Current State ✓)

| Component | Status | Location |
|-----------|--------|----------|
| NSMParameters dataclass | ✓ Complete | `physical_model.py` |
| ChannelNoiseProfile dataclass | ✓ Complete | `noise_models.py` |
| ChannelParameters dataclass | ✓ Complete | `physical_model.py` |
| TimingBarrier class | ✓ Complete | `timing.py` |
| QBER computation (Erven) | ✓ Complete | `utils/math.py` |
| CaligoNetworkBuilder | ✓ Complete | `network_builder.py` |
| Protocol base class integration | ✓ Complete | `protocol/base.py` |

### Phase 2: Enhanced Noise Modeling (Planned)

| Component | Status | Description |
|-----------|--------|-------------|
| Detection efficiency model | 🔲 Planned | Extend `build_link_config` with $\eta$ |
| Dark count injection | 🔲 Planned | Add to state delivery sampler |
| DoubleClick model integration | 🔲 Planned | Use netsquid_magic `DoubleClickModelParameters` |
| Measurement error mapping | 🔲 Planned | $e_{\text{det}} \rightarrow$ gate noise |

### Phase 3: Validation Framework (Planned)

| Component | Status | Description |
|-----------|--------|-------------|
| NSM condition verifier | 🔲 Planned | Runtime Q_channel < Q_storage check |
| QBER measurement validator | 🔲 Planned | Empirical vs theoretical comparison |
| Timing compliance reporter | ⚠️ Partial | TimingBarrier tracks compliance |
| Integration test suite | 🔲 Planned | End-to-end NSM parameter sweeps |

### Phase 4: Advanced Features (Future)

| Component | Status | Description |
|-----------|--------|-------------|
| Custom StateSampler factory | 🔲 Future | Full control over EPR state generation |
| Configurable heralding model | 🔲 Future | Beyond depolarise/double-click |
| Parameter sweep automation | 🔲 Future | Systematic $(r, \nu, \Delta t)$ exploration |
| Security margin visualization | 🔲 Future | Plot QBER vs storage noise bounds |

---

## References

1. **König, R., Wehner, S., & Wullschleger, J.** (2012). Unconditional security from noisy quantum storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.

2. **Schaffner, C., Terhal, B. M., & Wehner, S.** (2009). Robust cryptography in the noisy-quantum-storage model. *Quantum Information & Computation*, 9(11-12), 963-996.

3. **Wehner, S., Curty, M., Schaffner, C., & Lo, H. K.** (2010). Implementation of two-party protocols in the noisy-storage model. *Physical Review A*, 81(5), 052336.

4. **Erven, C., et al.** (2014). An experimental implementation of oblivious transfer in the noisy storage model. *Nature Communications*, 5, 3418.

5. **Lemus, M., Ramos, M. F., Yadav, P., Silva, N. A., Muga, N. J., & Pinto, A. N.** (2025). Practical quantum oblivious transfer with a single photon. *arXiv:2505.03803*.

---

*Document maintained by the Caligo development team. Last updated: December 18, 2025.*

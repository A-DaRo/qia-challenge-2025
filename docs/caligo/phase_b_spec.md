# Caligo Phase B: Simulation Layer Specification

**Document Type:** Formal Specification  
**Version:** 1.0  
**Date:** December 16, 2025  
**Status:** Draft  
**Parent Document:** [caligo_architecture.md](caligo_architecture.md)  
**Prerequisite:** [phase_a_spec.md](phase_a_spec.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope & Deliverables](#2-scope--deliverables)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Package: `simulation/`](#4-package-simulation)
5. [NetSquid/SquidASM Integration Reference](#5-netsquidsquidasm-integration-reference)
6. [Configuration System](#6-configuration-system)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

## 1. Executive Summary

**Phase B** establishes the simulation infrastructure that bridges NSM theoretical parameters to NetSquid's discrete-event simulation engine. This phase is **critical** because it implements the temporal semantics required for NSM security.

| Component | Purpose | Est. LOC |
|-----------|---------|----------|
| `simulation/physical_model.py` | NSM parameters ↔ NetSquid noise models | ~180 |
| `simulation/timing.py` | TimingBarrier with `ns.sim_time()` enforcement | ~150 |
| `simulation/network_builder.py` | SquidASM network configuration factory | ~200 |
| `simulation/noise_models.py` | Custom noise model wrappers | ~120 |

**Critical Insight from Architecture Analysis:**

> "The commitment scheme in the legacy implementation is hash-based but lacks NSM 'physical wait' enforcement semantics."
> — [phase_I_analysis.md](../implementation%20plan/phase_I_analysis.md)

Phase B directly addresses this gap by implementing `TimingBarrier` — a simulation-aware mechanism that enforces the wait time Δt as a **causal barrier** in the discrete-event timeline.

---

## 2. Scope & Deliverables

### 2.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `caligo/simulation/physical_model.py` | NSM parameter dataclass → NetSquid model mapping |
| `caligo/simulation/timing.py` | `TimingBarrier` for Δt enforcement |
| `caligo/simulation/network_builder.py` | Network configuration factory |
| `caligo/simulation/noise_models.py` | Custom noise model adapters |
| `caligo/simulation/__init__.py` | Package exports |
| Integration tests | SquidASM simulation validation |

### 2.2 Out of Scope

- Protocol phase implementations (Phase D)
- Security bounds calculation (Phase C)
- Type definitions (Phase A, prerequisite)

### 2.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Phase A types | — | `CaligoError`, phase contracts |
| SquidASM | ≥0.12 | Simulation framework |
| NetSquid | ≥1.1 | Discrete-event quantum simulation |
| netsquid-magic | ≥11.0 | EPR distribution models |
| netsquid-netbuilder | ≥0.4 | Network construction |

---

## 3. Theoretical Foundations

### 3.1 The Noisy Storage Model (NSM)

The NSM provides the security foundation for $\binom{2}{1}$-OT. Understanding its mathematical structure is essential for correct simulation mapping.

#### 3.1.1 NSM Definition (König et al., 2012)

The adversary's quantum storage is modeled as a noisy channel $\mathcal{F}$:

$$
\mathcal{F}: \mathcal{B}(\mathcal{H}_{\text{in}}) \rightarrow \mathcal{B}(\mathcal{H}_{\text{out}})
$$

The noise is assumed **Markovian** with the semigroup property:

$$
\mathcal{F}_0 = \mathbb{1} \quad \text{and} \quad \mathcal{F}_{t_1+t_2} = \mathcal{F}_{t_1} \circ \mathcal{F}_{t_2}
$$

**Implementation Implication:** The adversary cannot gain information by delaying readout. This justifies using a fixed $\mathcal{F} = \mathcal{F}_{\Delta t}$ for security analysis.

#### 3.1.2 Storage Capacity Constraint

Security holds when the classical capacity of the adversary's storage channel satisfies:

$$
C_{\mathcal{N}} \cdot \nu < \frac{1}{2}
$$

Where:
- $C_{\mathcal{N}}$ = Classical capacity of the storage channel
- $\nu$ = Storage rate (fraction of qubits storable)

**Reference:** König et al. (2012), Section I-C

#### 3.1.3 Depolarizing Channel Model

For a depolarizing channel with parameter $r \in [0, 1]$:

$$
\mathcal{D}_r(\rho) = r \cdot \rho + (1-r) \cdot \frac{\mathbb{I}}{d}
$$

Where:
- $r = 1$: Perfect storage (no noise)
- $r = 0$: Complete depolarization (maximum noise)
- $d = 2$: Qubit dimension

**NetSquid Mapping:** `DepolarNoiseModel(depolar_rate=1-r, time_independent=True)`

### 3.2 Channel Error Model

The trusted channel between honest parties experiences noise characterized by:

$$
Q_{\text{total}} = \mu + \eta \cdot e_{\text{det}} + (1 - \eta) \cdot \frac{1}{2}
$$

Where:
- $\mu$ = Source quality (EPR pair fidelity)
- $\eta$ = Channel transmittance (detection efficiency)
- $e_{\text{det}}$ = Intrinsic detector error rate

**Reference:** Erven et al. (2014), Table I

#### 3.2.1 Experimental Parameters (Erven et al.)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Source quality | $\mu$ | $3.145 \times 10^{-5}$ | EPR pair error contribution |
| Transmittance | $\eta$ | 0.0150 | Combined detection efficiency |
| Detector error | $e_{\text{det}}$ | 0.0093 | Intrinsic error rate |
| Dark count rate | $P_{\text{dark}}$ | $1.50 \times 10^{-8}$ | Dark counts per detector |
| Storage noise | $r$ | 0.75 | Depolarizing parameter |
| Storage rate | $\nu$ | 0.002 | Adversary storage fraction |

### 3.3 The "Strictly Less" Condition

**Security Requirement (Schaffner et al., 2009):**

> "Security can be obtained in the noisy-storage model with depolarizing storage noise, as long as the quantum bit-error rate of the channel does not exceed **11%** and the noise on the channel is **strictly less** than the noise during quantum storage."

Formally: $Q_{\text{channel}} < Q_{\text{storage}}$

This is the **hard feasibility limit** that Phase B's `FeasibilityChecker` must validate.

### 3.4 Timing Semantics

#### 3.4.1 The Wait Time Δt

The protocol enforces a wait time Δt between:
1. Bob's measurement completion
2. Alice's basis revelation

During this interval, any quantum information stored by a dishonest Bob degrades according to $\mathcal{F}_{\Delta t}$.

**From Erven et al. (2014):**
> "Both parties now wait a time, Δt, long enough for any stored quantum information of a dishonest party to decohere."

#### 3.4.2 NetSquid Time Model

NetSquid uses **nanoseconds** as the base time unit:

```python
# From netsquid/util/simtools.py
SECOND = 1e9      # ns
MILLISECOND = 1e6 # ns
MICROSECOND = 1e3 # ns
NANOSECOND = 1    # ns

def sim_time(magnitude=NANOSECOND) -> float:
    """Returns the current simulation time."""
    return _simengine.current_time / magnitude
```

**Implementation Requirement:** All timing in Caligo must be expressed in nanoseconds for NetSquid compatibility.

---

## 4. Package: `simulation/`

### 4.1 Module: `physical_model.py` (< 180 LOC)

**Purpose:** Define NSM physical parameters and map them to NetSquid noise models.

#### 4.1.1 `NSMParameters` Dataclass

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NSMParameters Specification                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Encapsulates all Noisy Storage Model parameters required for simulation    │
│  and security analysis. Provides validated mappings to NetSquid models.     │
│                                                                             │
│  Attributes:                                                                │
│  ┌─────────────────────┬────────────┬────────────────────────────────────┐  │
│  │ Name                │ Type       │ Description                        │  │
│  ├─────────────────────┼────────────┼────────────────────────────────────┤  │
│  │ storage_noise_r     │ float      │ Depolarizing parameter r ∈ [0,1]   │  │
│  │ storage_rate_nu     │ float      │ Adversary storage rate ν ∈ [0,1]   │  │
│  │ storage_dimension_d │ int        │ Qubit dimension (always 2)         │  │
│  │ delta_t_ns          │ float      │ Wait time in nanoseconds           │  │
│  │ channel_fidelity    │ float      │ EPR pair fidelity F ∈ (0.5, 1]     │  │
│  │ detection_eff_eta   │ float      │ Combined detection efficiency      │  │
│  │ detector_error      │ float      │ Intrinsic detector error rate      │  │
│  │ dark_count_prob     │ float      │ Dark count probability per event   │  │
│  └─────────────────────┴────────────┴────────────────────────────────────┘  │
│                                                                             │
│  Derived Properties:                                                        │
│  ┌─────────────────────┬────────────────────────────────────────────────┐   │
│  │ Property            │ Formula                                        │   │
│  ├─────────────────────┼────────────────────────────────────────────────┤   │
│  │ qber_channel        │ (1 - channel_fidelity) / 2 + detector_error    │   │
│  │ depolar_prob        │ 1 - storage_noise_r (for NetSquid)             │   │
│  │ storage_capacity    │ 1 - h(depolar_prob) where h = binary entropy   │   │
│  │ security_possible   │ qber_channel < 0.11 (Schaffner threshold)      │   │
│  └─────────────────────┴────────────────────────────────────────────────┘   │
│                                                                             │
│  Invariants:                                                                │
│  • INV-NSM-001: storage_noise_r ∈ [0, 1]                                    │
│  • INV-NSM-002: storage_rate_nu ∈ [0, 1]                                    │
│  • INV-NSM-003: storage_dimension_d == 2                                    │
│  • INV-NSM-004: delta_t_ns > 0                                              │
│  • INV-NSM-005: channel_fidelity ∈ (0.5, 1]                                 │
│  • INV-NSM-006: detection_eff_eta ∈ (0, 1]                                  │
│                                                                             │
│  References:                                                                │
│  • König et al. (2012) Section I-C: NSM definition                          │
│  • Erven et al. (2014) Table I: Experimental parameters                     │
│  • Schaffner et al. (2009) Corollary 7: 11% QBER threshold                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 `ChannelParameters` Dataclass

```python
@dataclass(frozen=True)
class ChannelParameters:
    """
    Physical channel parameters for the quantum link.
    
    These parameters characterize the trusted channel between honest
    Alice and honest Bob, separate from the adversary's storage model.
    
    Attributes
    ----------
    length_km : float
        Fiber length in kilometers.
    attenuation_db_per_km : float
        Fiber loss in dB/km. Default: 0.2 (standard telecom fiber).
    speed_of_light_km_s : float
        Speed of light in fiber (km/s). Default: 200,000.
    t1_ns : float
        T1 relaxation time for memory qubits (ns). Default: 10_000_000.
    t2_ns : float
        T2 dephasing time for memory qubits (ns). Default: 1_000_000.
    cycle_time_ns : float
        EPR generation cycle time (ns). Default: 1_000_000 (1 ms).
    
    Derived Properties
    ------------------
    propagation_delay_ns : float
        Light travel time through fiber: length_km / speed_of_light_km_s * 1e9
    total_loss_db : float
        Total fiber loss: length_km * attenuation_db_per_km
    transmittance : float
        Transmission probability: 10^(-total_loss_db / 10)
    
    References
    ----------
    - netsquid-magic HeraldedModelParameters: length_A, length_B, p_loss_length
    - NetSquid T1T2NoiseModel: T1, T2 parameters
    """
```

#### 4.1.3 Factory Functions

```python
def create_depolar_noise_model(params: NSMParameters) -> DepolarNoiseModel:
    """
    Create NetSquid DepolarNoiseModel from NSM parameters.
    
    Parameters
    ----------
    params : NSMParameters
        NSM configuration with storage_noise_r.
    
    Returns
    -------
    DepolarNoiseModel
        Configured with depolar_rate = 1 - params.storage_noise_r.
    
    Notes
    -----
    NetSquid's DepolarNoiseModel uses `depolar_rate` as the probability
    of depolarization, whereas NSM uses `r` as the probability of
    preservation. Hence: depolar_rate = 1 - r.
    
    For time-independent storage noise (adversary stores for exactly Δt):
        time_independent=True
    
    For T1T2-style continuous decoherence (honest party memory):
        Use create_t1t2_noise_model instead.
    """

def create_t1t2_noise_model(params: ChannelParameters) -> T1T2NoiseModel:
    """
    Create NetSquid T1T2NoiseModel for quantum memory decoherence.
    
    Parameters
    ----------
    params : ChannelParameters
        Channel configuration with t1_ns and t2_ns.
    
    Returns
    -------
    T1T2NoiseModel
        Configured with T1=params.t1_ns, T2=params.t2_ns.
    
    Notes
    -----
    The T1T2 model is phenomenological:
    - T1: Amplitude damping (energy relaxation)
    - T2: Phase damping (dephasing), T2 ≤ T1
    
    This models the honest party's imperfect quantum memory,
    NOT the adversary's noisy storage (use depolar model for that).
    
    References
    ----------
    - netsquid/components/models/qerrormodels.py: T1T2NoiseModel
    """

def create_link_model_params(
    nsm_params: NSMParameters,
    channel_params: ChannelParameters,
    model_type: str = "depolarise"
) -> IModelParameters:
    """
    Create netsquid-magic model parameters for EPR distribution.
    
    Parameters
    ----------
    nsm_params : NSMParameters
        NSM security parameters.
    channel_params : ChannelParameters
        Physical channel parameters.
    model_type : str
        One of: "perfect", "depolarise", "single_click", "double_click"
    
    Returns
    -------
    IModelParameters
        Appropriate model parameters subclass for netsquid-magic.
    
    Model Type Selection
    --------------------
    - "perfect": PerfectModelParameters (testing only)
    - "depolarise": DepolariseModelParameters (basic noise model)
    - "single_click": SingleClickModelParameters (heralded, realistic)
    - "double_click": DoubleClickModelParameters (heralded, high-fidelity)
    
    References
    ----------
    - netsquid_magic/model_parameters.py: Parameter classes
    - Erven et al. (2014): Experimental setup used double-click BSM
    """
```

#### 4.1.4 Note on `FeasibilityChecker`

> **Important:** The `FeasibilityChecker` class is defined in **Phase C** (`security/feasibility.py`), 
> not in Phase B. Phase B's `NSMParameters` dataclass validates its own invariants in `__post_init__`,
> but comprehensive security feasibility checking (QBER thresholds, storage capacity constraint, 
> "strictly less" condition) belongs to the security layer.
>
> See [phase_c_spec.md](phase_c_spec.md) Section 4.2 for the full `FeasibilityChecker` specification.

---

### 4.2 Module: `timing.py` (< 150 LOC)
        """
        Verify the "strictly less" condition: Q_channel < Q_storage.
        
        The fundamental NSM security condition requires that the noise
        experienced by honest parties (channel errors) is strictly less
        than the noise affecting the adversary's quantum storage.
        
        Returns
        -------
        bool
            True if Q_channel < Q_storage (security possible).
        
        Notes
        -----
        For depolarizing storage with parameter r:
            Q_storage ≈ (1 - r) / 2 (mixing toward |01⟩+|10⟩)
        
        If this condition fails, a "rational" adversary would simply
        measure immediately rather than store, gaining more information
        than honest parties.
        """
```

---

### 4.2 Module: `timing.py` (< 150 LOC)

**Purpose:** Implement the `TimingBarrier` that enforces NSM wait time Δt.

#### 4.2.1 `TimingBarrier` Class

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TimingBarrier Specification                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Enforces the NSM timing constraint Δt as a causal barrier in the           │
│  discrete-event simulation. Prevents basis revelation before Δt expires.    │
│                                                                             │
│  The TimingBarrier is the SOFTWARE IMPLEMENTATION of the physical           │
│  requirement that adversary storage noise degrades over time Δt.            │
│                                                                             │
│  State Machine:                                                             │
│  ┌────────────┐     mark_quantum_complete()     ┌────────────────────┐      │
│  │   IDLE     │ ──────────────────────────────► │  WAITING           │      │
│  │            │                                 │  (start_time set)  │      │
│  └────────────┘                                 └─────────┬──────────┘      │
│        ▲                                                  │                 │
│        │                                                  │                 │
│        │ reset()                         wait_complete() │                  │
│        │                                 (after Δt)      ▼                  │
│  ┌─────┴──────┐                                 ┌────────────────────┐      │
│  │  EXPIRED   │ ◄───────────────────────────────│  READY             │      │
│  │            │      (timing_compliant=True)    │  (Δt elapsed)      │      │
│  └────────────┘                                 └────────────────────┘      │
│                                                                             │
│  Methods:                                                                   │
│  ┌─────────────────────────┬───────────────────────────────────────────┐    │
│  │ Method                  │ Description                               │    │
│  ├─────────────────────────┼───────────────────────────────────────────┤    │
│  │ mark_quantum_complete() │ Record end of quantum phase               │    │
│  │ wait_delta_t()          │ Yield until Δt has elapsed                │    │
│  │ can_reveal_basis()      │ Check if basis revelation is permitted    │    │
│  │ get_elapsed_ns()        │ Return time since quantum phase ended     │    │
│  │ reset()                 │ Reset barrier for next protocol run       │    │
│  └─────────────────────────┴───────────────────────────────────────────┘    │
│                                                                             │
│  Integration with SquidASM:                                                 │
│  • Uses ns.sim_time() for current simulation time                           │
│  • Generator-based wait compatible with SquidASM protocols                  │
│  • Can be used in both Program.run() and standalone contexts                │
│                                                                             │
│  Security Properties:                                                       │
│  • PROP-TIME-001: Basis revelation MUST NOT occur before Δt                 │
│  • PROP-TIME-002: All parties observe same Δt (simulation synchronized)     │
│  • PROP-TIME-003: Barrier state persists across yields                      │
│                                                                             │
│  References:                                                                │
│  • Erven et al. (2014): "Both parties now wait a time, Δt..."               │
│  • König et al. (2012): Markovian noise semigroup property                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 Implementation Specification

```python
from typing import Generator, Optional
import netsquid as ns
from enum import Enum, auto

class TimingBarrierState(Enum):
    """State machine states for TimingBarrier."""
    IDLE = auto()          # Initial state, waiting for quantum phase
    WAITING = auto()       # Quantum complete, waiting for Δt
    READY = auto()         # Δt elapsed, can proceed
    EXPIRED = auto()       # Protocol completed, barrier used


class TimingBarrier:
    """
    Enforces NSM timing constraint Δt in discrete-event simulation.
    
    The TimingBarrier ensures that Alice cannot reveal her basis choices
    until time Δt has elapsed since the quantum phase completed. This
    allows any quantum information stored by a dishonest Bob to decohere
    according to the NSM assumption.
    
    Parameters
    ----------
    delta_t_ns : float
        Required wait time in nanoseconds.
    strict_mode : bool
        If True, raises TimingViolationError on premature access.
        If False, logs warning but allows continuation. Default: True.
    
    Attributes
    ----------
    state : TimingBarrierState
        Current state machine state.
    quantum_complete_time : Optional[float]
        Simulation time when quantum phase ended (ns).
    delta_t_ns : float
        Configured wait time.
    timing_compliant : bool
        True if protocol respected timing constraints.
    
    Examples
    --------
    In a SquidASM Program:
    
    >>> barrier = TimingBarrier(delta_t_ns=1_000_000)  # 1 ms
    >>> 
    >>> # After quantum measurements complete:
    >>> barrier.mark_quantum_complete()
    >>> 
    >>> # Wait for Δt (yields control to simulator)
    >>> yield from barrier.wait_delta_t()
    >>> 
    >>> # Now safe to reveal basis
    >>> if barrier.can_reveal_basis():
    >>>     socket.send(bases)
    
    References
    ----------
    - phase_I_analysis.md Section 7.2: Timing barrier requirements
    - König et al. (2012): Markovian storage noise assumption
    """
    
    def mark_quantum_complete(self) -> None:
        """
        Mark the end of the quantum phase.
        
        Records the current simulation time as the start of the Δt
        wait period. Must be called after all quantum measurements
        are complete but before any basis information is exchanged.
        
        Raises
        ------
        TimingViolationError
            If called when not in IDLE state.
        
        Side Effects
        ------------
        - Sets quantum_complete_time to ns.sim_time()
        - Transitions state from IDLE to WAITING
        """
    
    def wait_delta_t(self) -> Generator[None, None, None]:
        """
        Generator that waits until Δt has elapsed.
        
        Yields control back to the SquidASM simulation engine until
        the required wait time has passed. Compatible with the
        generator-based programming model.
        
        Yields
        ------
        None
            Control is yielded until Δt elapses.
        
        Raises
        ------
        TimingViolationError
            If called when not in WAITING state.
        
        Notes
        -----
        In SquidASM, use: `yield from barrier.wait_delta_t()`
        
        This does NOT block the simulation — other events can occur.
        The barrier simply tracks when the protocol is allowed to
        proceed with basis revelation.
        """
    
    def can_reveal_basis(self) -> bool:
        """
        Check if basis revelation is permitted.
        
        Returns
        -------
        bool
            True if Δt has elapsed and state is READY.
        
        Side Effects
        ------------
        In strict_mode, raises TimingViolationError if called in
        wrong state. Otherwise, returns False with logged warning.
        """
    
    def get_elapsed_ns(self) -> float:
        """
        Get time elapsed since quantum phase completed.
        
        Returns
        -------
        float
            Elapsed time in nanoseconds, or 0.0 if not started.
        """
    
    def assert_timing_compliant(self) -> None:
        """
        Assert that timing constraints are satisfied.
        
        Raises
        ------
        TimingViolationError
            If Δt has not elapsed since quantum phase completion.
            Error includes diagnostic information about elapsed time
            and required wait time.
        
        Notes
        -----
        Call this method before revealing basis information.
        Unlike can_reveal_basis(), this method ALWAYS raises if
        timing is violated (no strict_mode flag needed).
        
        Example
        -------
        >>> barrier.mark_quantum_complete()
        >>> yield from barrier.wait_delta_t()
        >>> barrier.assert_timing_compliant()  # Raises if Δt not satisfied
        >>> socket.send(bases)  # Safe to reveal
        """
```

#### 4.2.3 Integration Pattern

```python
# Pattern for using TimingBarrier in a SquidASM Program

class EHOKAliceProgram(Program):
    def __init__(self, params: NSMParameters):
        self.barrier = TimingBarrier(delta_t_ns=params.delta_t_ns)
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="ehok_alice",
            csockets=["bob"],
            epr_sockets=["bob"],
        )
    
    def run(self, context: ProgramContext) -> Generator[...]:
        epr_socket = context.epr_sockets["bob"]
        csocket = context.csockets["bob"]
        
        # === PHASE I: Quantum Generation ===
        outcomes, bases = [], []
        for _ in range(self.num_pairs):
            epr = yield from epr_socket.create_keep()[0]
            basis = random.randint(0, 1)
            if basis == 1:
                yield from epr.H()
            m = yield from epr.measure()
            outcomes.append(m)
            bases.append(basis)
        
        # Mark quantum phase complete — starts Δt timer
        self.barrier.mark_quantum_complete()
        
        # === WAIT FOR Δt ===
        # This yields control; simulation time advances
        yield from self.barrier.wait_delta_t()
        
        # === PHASE II: Can now reveal basis ===
        if self.barrier.can_reveal_basis():
            csocket.send(bases)
        else:
            raise TimingViolationError("Basis revelation before Δt")
```

---

### 4.3 Module: `network_builder.py` (< 200 LOC)

**Purpose:** Factory for creating SquidASM network configurations.

#### 4.3.1 `CaligoNetworkBuilder` Class

```python
class CaligoNetworkBuilder:
    """
    Factory for creating SquidASM network configurations for $\binom{2}{1}$-OT.
    
    Provides a high-level API for constructing network configurations
    that incorporate NSM-specific parameters. Wraps the complexity of
    SquidASM's StackNetworkConfig and netsquid-netbuilder.
    
    Attributes
    ----------
    nsm_params : NSMParameters
        NSM security parameters.
    channel_params : ChannelParameters
        Physical channel parameters.
    
    Methods
    -------
    build_two_node_network(alice_name, bob_name) -> StackNetworkConfig
        Create minimal Alice-Bob network for $\binom{2}{1}$-OT.
    build_stack_config(name, num_qubits) -> StackConfig
        Create single node configuration.
    build_link_config(node1, node2, noise_type) -> LinkConfig
        Create quantum link configuration.
    
    Examples
    --------
    >>> builder = CaligoNetworkBuilder(nsm_params, channel_params)
    >>> config = builder.build_two_node_network("Alice", "Bob")
    >>> network = run_network(config, programs)
    
    References
    ----------
    - squidasm/run/stack/config.py: StackNetworkConfig
    - squidasm/run/stack/build.py: create_stack_network_builder
    """
    
    def build_two_node_network(
        self,
        alice_name: str = "Alice",
        bob_name: str = "Bob",
        num_qubits: int = 10
    ) -> StackNetworkConfig:
        """
        Create a two-node network configuration for $\binom{2}{1}$-OT.
        
        Parameters
        ----------
        alice_name : str
            Name for Alice's node. Default: "Alice".
        bob_name : str
            Name for Bob's node. Default: "Bob".
        num_qubits : int
            Number of qubit positions per node. Default: 10.
        
        Returns
        -------
        StackNetworkConfig
            Complete network configuration ready for simulation.
        
        Notes
        -----
        The returned configuration includes:
        - Two StackConfig nodes with Generic QDevice
        - One LinkConfig with appropriate noise model
        - Instant classical links (no propagation delay)
        
        The quantum link noise model is selected based on NSM parameters:
        - If channel_fidelity == 1.0: NoNoise (testing only)
        - Otherwise: Depolarise with fidelity = channel_fidelity
        """
```

#### 4.3.2 Configuration Presets

```python
def perfect_network_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 10
) -> StackNetworkConfig:
    """
    Create a perfect (noiseless) network for unit testing.
    
    All operations are ideal:
    - Perfect EPR pairs (F = 1.0)
    - No memory decoherence
    - No channel losses
    - No gate errors
    
    Use this for testing protocol logic without noise effects.
    """

def realistic_network_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob",
    num_qubits: int = 10,
    fidelity: float = 0.95,
    t1_ns: float = 10_000_000,
    t2_ns: float = 1_000_000
) -> StackNetworkConfig:
    """
    Create a network with realistic noise parameters.
    
    Based on Erven et al. (2014) experimental parameters,
    scaled for simulation efficiency.
    """

def erven_experimental_config(
    alice_name: str = "Alice",
    bob_name: str = "Bob"
) -> StackNetworkConfig:
    """
    Create network matching Erven et al. (2014) experiment.
    
    Uses exact parameters from Table I:
    - μ = 3.145 × 10^{-5} (source quality)
    - η = 0.0150 (transmittance)
    - e_det = 0.0093 (detector error)
    - r = 0.75 (storage noise)
    - ν = 0.002 (storage rate)
    
    Warning: This configuration has very low transmittance,
    requiring many rounds for meaningful key generation.
    """
```

---

### 4.4 Module: `noise_models.py` (< 120 LOC)

**Purpose:** Custom noise model wrappers for NSM-specific behavior.

#### 4.4.1 `NSMStorageNoiseModel`

```python
class NSMStorageNoiseModel:
    """
    Wrapper combining depolarization and timing for NSM storage noise.
    
    Models the adversary's quantum storage as experiencing depolarizing
    noise over the wait time Δt. This is used for theoretical analysis
    and test oracle generation, NOT for actual simulation (where the
    adversary's storage is not simulated).
    
    Attributes
    ----------
    r : float
        Depolarizing parameter (preservation probability).
    delta_t_ns : float
        Wait time in nanoseconds.
    
    Methods
    -------
    apply_noise(state: np.ndarray) -> np.ndarray
        Apply depolarizing channel to quantum state.
    get_effective_fidelity() -> float
        Calculate fidelity after noise application.
    get_min_entropy_bound() -> float
        Calculate adversary's min-entropy bound.
    
    References
    ----------
    - König et al. (2012): Markovian storage noise model
    - Schaffner et al. (2009): Depolarizing channel analysis
    """
```

#### 4.4.2 `ChannelNoiseProfile`

```python
@dataclass(frozen=True)
class ChannelNoiseProfile:
    """
    Aggregate noise profile for the trusted quantum channel.
    
    Combines multiple noise sources into a single QBER estimate.
    
    Attributes
    ----------
    source_fidelity : float
        EPR source fidelity (intrinsic pair quality).
    transmission_loss : float
        Probability of photon loss in fiber.
    detector_efficiency : float
        Probability of detection given arrival.
    detector_error : float
        Probability of measurement error.
    dark_count_rate : float
        Dark count probability per detection window.
    
    Derived Properties
    ------------------
    total_qber : float
        Combined QBER from all sources.
    is_secure : bool
        True if total_qber < QBER_CONSERVATIVE_LIMIT.
    
    Methods
    -------
    to_nsm_parameters() -> NSMParameters
        Convert to NSMParameters for simulation.
    
    References
    ----------
    - Erven et al. (2014) Table I: Experimental parameters
    """
```

---

## 5. NetSquid/SquidASM Integration Reference

### 5.1 Package Structure

```
SquidASM (0.12+)
├── squidasm/
│   ├── run/
│   │   └── stack/
│   │       ├── config.py      # StackNetworkConfig, LinkConfig
│   │       ├── build.py       # create_stack_network_builder
│   │       └── run.py         # run() entry point
│   ├── sim/
│   │   ├── stack/
│   │   │   └── stack.py       # StackNode, NodeStack
│   │   └── network/
│   │       └── network.py     # NetSquidNetwork
│   └── util/
│       └── sim.py             # get_qubit_state

NetSquid (1.1+)
├── components/models/
│   └── qerrormodels.py        # DepolarNoiseModel, T1T2NoiseModel
├── util/
│   └── simtools.py            # sim_time(), NANOSECOND constants

netsquid-magic (11.0+)
├── model_parameters.py        # IModelParameters hierarchy
├── magic_distributor.py       # MagicDistributor classes
└── state_delivery_sampler.py  # State sampling for EPR
```

### 5.2 Key API Mappings

| Caligo Concept | SquidASM/NetSquid API |
|----------------|----------------------|
| NSM storage noise r | `DepolarNoiseModel(depolar_rate=1-r)` |
| Memory T1/T2 | `T1T2NoiseModel(T1=t1_ns, T2=t2_ns)` |
| EPR fidelity | `DepolariseModelParameters(prob_max_mixed=1-F)` |
| Simulation time | `netsquid.sim_time()` returns ns |
| Time delay | `yield from context.connection.flush()` |
| Network config | `StackNetworkConfig` from `squidasm.run.stack.config` |

### 5.3 Time Unit Conventions

```python
# NetSquid uses nanoseconds as base unit
from netsquid.util.simtools import SECOND, MILLISECOND, MICROSECOND, NANOSECOND

# Constants for Caligo
CALIGO_TIME_UNIT = NANOSECOND  # Base unit: ns

# Typical timing values
TYPICAL_DELTA_T_NS = 1_000_000       # 1 ms (Δt for NSM)
TYPICAL_CYCLE_TIME_NS = 10_000       # 10 μs (EPR generation)
TYPICAL_T1_NS = 10_000_000           # 10 ms (T1 relaxation)
TYPICAL_T2_NS = 1_000_000            # 1 ms (T2 dephasing)
```

### 5.4 Noise Model Mappings

| Physical Effect | NetSquid Model | Parameters |
|-----------------|----------------|------------|
| Depolarizing storage | `DepolarNoiseModel` | `depolar_rate`, `time_independent=True` |
| Memory decoherence | `T1T2NoiseModel` | `T1`, `T2` (in ns) |
| Fiber loss | `FibreLossModel` | `p_loss_length` (dB/km) |
| EPR imperfection | `DepolariseModelParameters` | `prob_max_mixed`, `prob_success` |
| Detection inefficiency | `SingleClickModelParameters` | `detector_efficiency`, `dark_count_probability` |

---

## 6. Configuration System

### 6.1 YAML Configuration Schema

```yaml
# caligo_config.yaml
nsm:
  storage_noise_r: 0.75
  storage_rate_nu: 0.002
  delta_t_ns: 1_000_000
  
channel:
  fidelity: 0.95
  length_km: 1.0
  attenuation_db_per_km: 0.2
  t1_ns: 10_000_000
  t2_ns: 1_000_000
  
protocol:
  num_pairs: 100_000
  qber_abort_threshold: 0.11
  test_fraction: 0.1
  
simulation:
  seed: 42
  log_level: INFO
```

### 6.2 `CaligoConfig` Dataclass

```python
@dataclass
class CaligoConfig:
    """
    Complete configuration for a Caligo simulation run.
    
    Aggregates NSM parameters, channel parameters, protocol settings,
    and simulation options into a single validated configuration object.
    
    Attributes
    ----------
    nsm : NSMParameters
        Noisy Storage Model parameters.
    channel : ChannelParameters
        Physical channel parameters.
    protocol : ProtocolParameters
        Protocol execution parameters.
    simulation : SimulationParameters
        Simulation engine parameters.
    
    Class Methods
    -------------
    from_yaml(path: Path) -> CaligoConfig
        Load configuration from YAML file.
    from_dict(data: dict) -> CaligoConfig
        Create from dictionary (e.g., from JSON).
    default() -> CaligoConfig
        Create configuration with sensible defaults.
    erven_experimental() -> CaligoConfig
        Create configuration matching Erven et al. (2014).
    
    Instance Methods
    ----------------
    validate() -> None
        Run all validation checks, raise ConfigurationError on failure.
    to_yaml(path: Path) -> None
        Save configuration to YAML file.
    to_dict() -> dict
        Convert to dictionary.
    """
```

---

## 7. Testing Strategy

### 7.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Validate individual classes | `tests/test_simulation/` |
| Integration Tests | SquidASM simulation runs | `tests/integration/` |
| Timing Tests | Verify Δt enforcement | `tests/test_simulation/test_timing.py` |
| Noise Model Tests | Verify noise mappings | `tests/test_simulation/test_noise.py` |

### 7.2 Integration Test Fixtures

```python
# tests/integration/conftest.py

@pytest.fixture
def perfect_network():
    """Create a perfect (noiseless) network for testing."""
    return perfect_network_config()

@pytest.fixture
def realistic_network():
    """Create a realistic noisy network for testing."""
    return realistic_network_config(fidelity=0.95)

@pytest.fixture
def nsm_params():
    """Default NSM parameters for testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.95,
    )

@pytest.fixture
def timing_barrier(nsm_params):
    """Pre-configured TimingBarrier for testing."""
    return TimingBarrier(delta_t_ns=nsm_params.delta_t_ns)
```

### 7.3 Test Examples

```python
# tests/test_simulation/test_timing.py

class TestTimingBarrier:
    """Tests for TimingBarrier NSM enforcement."""
    
    def test_cannot_reveal_before_delta_t(self, timing_barrier):
        """Verify basis revelation blocked before Δt."""
        timing_barrier.mark_quantum_complete()
        
        # Immediately after marking, should not be able to reveal
        assert not timing_barrier.can_reveal_basis()
        assert timing_barrier.state == TimingBarrierState.WAITING
    
    def test_can_reveal_after_delta_t(self, timing_barrier):
        """Verify basis revelation allowed after Δt."""
        timing_barrier.mark_quantum_complete()
        
        # Simulate time passing
        ns.sim_run(duration=timing_barrier.delta_t_ns + 1)
        
        assert timing_barrier.can_reveal_basis()
        assert timing_barrier.state == TimingBarrierState.READY
    
    def test_timing_violation_raises_in_strict_mode(self):
        """Verify TimingViolationError in strict mode."""
        barrier = TimingBarrier(delta_t_ns=1_000_000, strict_mode=True)
        barrier.mark_quantum_complete()
        
        with pytest.raises(TimingViolationError):
            # Try to reveal without waiting
            barrier.can_reveal_basis()  # Should raise


# tests/test_simulation/test_noise.py

class TestNoiseModelMapping:
    """Tests for NSM → NetSquid noise model conversion."""
    
    def test_depolar_rate_inversion(self):
        """Verify r → depolar_rate mapping."""
        params = NSMParameters(storage_noise_r=0.75)
        model = create_depolar_noise_model(params)
        
        # NetSquid: depolar_rate = probability of depolarization
        # NSM: r = probability of preservation
        # Therefore: depolar_rate = 1 - r
        assert model.depolar_rate == 0.25
    
    def test_perfect_storage_has_zero_depolar(self):
        """r=1 means perfect storage, no depolarization."""
        params = NSMParameters(storage_noise_r=1.0)
        model = create_depolar_noise_model(params)
        
        assert model.depolar_rate == 0.0


# tests/integration/test_squidasm_integration.py

class TestSquidASMIntegration:
    """Integration tests with SquidASM simulation engine."""
    
    def test_epr_generation_with_depolarizing_noise(self, realistic_network):
        """Verify EPR pairs have expected fidelity."""
        # Run a simple EPR generation program
        results = run_simulation(realistic_network, [AliceEPRProgram(), BobEPRProgram()])
        
        # Extract fidelities from results
        fidelities = [r['fidelity'] for r in results]
        avg_fidelity = np.mean(fidelities)
        
        # Should match configured channel fidelity (within statistical error)
        assert 0.90 < avg_fidelity < 1.0
    
    def test_timing_barrier_in_simulation(self, perfect_network, nsm_params):
        """Verify TimingBarrier integrates with SquidASM."""
        barrier = TimingBarrier(delta_t_ns=nsm_params.delta_t_ns)
        
        class TimingTestProgram(Program):
            def run(self, context):
                # Do some work
                yield from context.connection.flush()
                
                # Mark quantum complete
                barrier.mark_quantum_complete()
                start_time = ns.sim_time()
                
                # Wait for Δt
                yield from barrier.wait_delta_t()
                
                end_time = ns.sim_time()
                elapsed = end_time - start_time
                
                return {"elapsed_ns": elapsed}
        
        results = run_simulation(perfect_network, [TimingTestProgram()])
        
        # Verify elapsed time matches Δt
        assert results[0]["elapsed_ns"] >= nsm_params.delta_t_ns
```

---

## 8. Acceptance Criteria

### 8.1 Functional Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-B-001 | `TimingBarrier` prevents basis revelation before Δt | Unit test + integration test |
| AC-B-002 | `NSMParameters` validates all invariants in `__post_init__` | Unit tests |
| AC-B-003 | `FeasibilityChecker` correctly rejects insecure configurations | Unit tests with edge cases |
| AC-B-004 | Noise models produce expected fidelity in simulation | Integration tests |
| AC-B-005 | Network builder creates valid SquidASM configurations | Integration tests |

### 8.2 Quality Criteria

| ID | Criterion | Target |
|----|-----------|--------|
| AC-B-006 | Test coverage for `simulation/` package | ≥90% |
| AC-B-007 | All modules ≤ 200 LOC | Static analysis |
| AC-B-008 | Numpydoc format for all public APIs | Linter check |
| AC-B-009 | Integration tests pass with SquidASM 0.12+ | CI pipeline |
| AC-B-010 | No `print()` statements | Grep check |

### 8.3 Performance Criteria

| ID | Criterion | Target |
|----|-----------|--------|
| AC-B-011 | `TimingBarrier` overhead | < 1 μs per call |
| AC-B-012 | Network configuration creation | < 100 ms |
| AC-B-013 | Noise model application | O(1) per qubit |

---

## 9. References

### 9.1 Primary Literature

| Citation | Title | Usage in Phase B |
|----------|-------|------------------|
| König et al. (2012) | "Unconditional Security From Noisy Quantum Storage" | NSM definition, Markovian noise, storage capacity constraint |
| Schaffner et al. (2009) | "Robust Cryptography in the Noisy-Quantum-Storage Model" | 11% QBER threshold, "strictly less" condition, depolarizing analysis |
| Erven et al. (2014) | "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model" | Experimental parameters, Δt semantics, protocol flow |

### 9.2 SquidASM/NetSquid Documentation

| Document | Location | Relevance |
|----------|----------|-----------|
| SquidASM Architecture | `squidasm/sim/stack/arch/ARCH.md` | Stack architecture |
| NetSquid simtools | `netsquid/util/simtools.py` | `sim_time()`, time constants |
| netsquid-magic models | `netsquid_magic/model_parameters.py` | EPR distribution parameters |
| StackNetworkConfig | `squidasm/run/stack/config.py` | Network configuration API |

### 9.3 Internal Documents

| Document | Relevance |
|----------|-----------|
| [caligo_architecture.md](caligo_architecture.md) | Parent architecture document |
| [phase_a_spec.md](phase_a_spec.md) | Prerequisite types and utils |
| [phase_I_analysis.md](../implementation%20plan/phase_I_analysis.md) | Timing barrier requirements |
| [phase_II_analysis.md](../implementation%20plan/phase_II_analysis.md) | Sifting timing constraints |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Lead Architect | Initial specification |

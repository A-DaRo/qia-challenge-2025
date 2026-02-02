# NSM Parameters Implementation Specification

**Document Type:** Implementation Specification  
**Version:** 2.0  
**Date:** December 18, 2025  
**Status:** Final Draft  
**Parent Document:** [nsm_parameters_enforcement.md](nsm_parameters_enforcement.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Core Components](#3-core-components)
4. [NSM-to-Physical Parameter Mapping](#4-nsm-to-physical-parameter-mapping)
5. [SquidASM Pipeline Injection](#5-squidasm-pipeline-injection)
6. [Squid Package Dependencies](#6-squid-package-dependencies)
7. [Confrontation With Enforcement Spec](#7-confrontation-with-enforcement-spec)
8. [Implementation Contracts](#8-implementation-contracts)
9. [Package Structure (Final)](#9-package-structure-final)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Overview

### 1.1 Purpose

This document specifies the **single-phase** implementation architecture for enforcing Noisy Storage Model (NSM) parameters within the Caligo simulation environment. It bridges NSM theoretical parameters to their physical representations via runtime injection into SquidASM.

### 1.2 Design Goals

| Goal | Rationale |
|------|-----------|
| **Direct NSM-to-Config Mapping** | NSM/channel parameters translate directly to SquidASM/netbuilder configs (which then construct NetSquid noise processes) |
| **Fail-Fast Validation** | Invalid parameters rejected at construction via `__post_init__` |
| **Pipeline Transparency** | Clear injection points at Link, Device, and Protocol levels |
| **Literature-Grounded Formulas** | All QBER/security computations derive from published results |

### 1.3 Scope Boundaries

**In Scope:**
- NSM parameter dataclasses with validation
- SquidASM/netbuilder configuration injection (links + qdevices)
- SquidASM pipeline injection mechanisms
- TimingBarrier for Δt enforcement
- Analytic feasibility/security condition verification

**Out of Scope (external tooling):**
- Parameter sweep automation
- Visualization components
- Statistical analysis pipelines

---

## 2. Architecture Principles

### 2.1 Unified Configuration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NSM PARAMETER FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Configuration                                                        │
│   ──────────────────                                                        │
│   NSMParameters(r, ν, Δt, F, η, e_det, P_dark)                              │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    VALIDATION + TRANSLATION                         │   │
│   │  • Invariant enforcement via __post_init__                          │   │
│   │  • Derived properties: depolar_prob, qber_channel, qber_storage     │   │
│   │  • Security condition: Q_channel < Q_storage                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ├──────────────────┬──────────────────┬─────────────────────┐    │
│            ▼                  ▼                  ▼                     ▼    │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │ LinkConfig  │    │ QDeviceConfig│   │TimingBarrier│    │  Security   │  │
│   │ (EPR noise) │    │ (T1/T2/gate)│    │ (Δt wait)   │    │  Verifier   │  │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘  │
│          │                  │                  │                            │
│          └──────────────────┴──────────────────┘                            │
│                             │                                               │
│                             ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SQUIDASM RUNTIME INJECTION                       │   │
│   │  • MagicDistributor with DoubleClickModelParameters                 │   │
│   │  • QuantumProcessor with T1T2NoiseModel + DepolarNoiseModel         │   │
│   │  • CaligoProgram with integrated TimingBarrier                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Immutable Configuration

All parameter dataclasses use `@dataclass(frozen=True)`:
- Thread-safe sharing across simulation components
- Prevents accidental mutation during protocol execution
- Enables deterministic replay of simulations

---

## 3. Core Components

### 3.1 NSMParameters

**Location:** `caligo/simulation/physical_model.py`

```python
@dataclass(frozen=True)
class NSMParameters:
    """
    Noisy Storage Model parameters for security analysis and simulation.

    Encapsulates all NSM parameters required for E-HOK protocol execution
    with validated mappings to NetSquid noise models.

    Parameters
    ----------
    storage_noise_r : float
        Depolarizing parameter r ∈ [0, 1]. Represents preservation probability
        in adversary's noisy quantum storage. r=1 means perfect storage (no
        security), r=0 means complete depolarization (maximum security).
    storage_rate_nu : float
        Adversary storage rate ν ∈ [0, 1]. Fraction of qubits adversary can
        store. Security requires C_N · ν < 1/2.
    delta_t_ns : float
        Wait time Δt in nanoseconds. Time Alice waits before revealing basis
        to allow adversary's stored qubits to decohere.
    channel_fidelity : float
        EPR pair fidelity F ∈ (0.5, 1]. Source quality parameter.
    detection_eff_eta : float
        Combined detection efficiency η ∈ (0, 1]. Includes transmittance
        and detector efficiency.
    detector_error : float
        Intrinsic detector error e_det ∈ [0, 0.5]. Measurement apparatus
        imperfections.
    dark_count_prob : float
        Dark count probability P_dark ∈ [0, 1]. Spurious detector clicks.

    Attributes
    ----------
    depolar_prob : float
        NetSquid depolarization probability = 1 - r.
    qber_storage : float
        Storage-induced QBER threshold = (1 - r) / 2.
    qber_channel : float
        Total channel QBER from Erven et al. formula.
    storage_capacity : float
        Classical capacity C_N of depolarizing storage channel.

    Raises
    ------
    InvalidParameterError
        If any invariant is violated during construction.

    Notes
    -----
    **Invariants:**
    
    - INV-NSM-001: storage_noise_r ∈ [0, 1]
    - INV-NSM-002: storage_rate_nu ∈ [0, 1]  
    - INV-NSM-003: delta_t_ns > 0
    - INV-NSM-004: channel_fidelity ∈ (0.5, 1]
    - INV-NSM-005: detection_eff_eta ∈ (0, 1]
    - INV-NSM-006: detector_error ∈ [0, 0.5]
    - INV-NSM-007: dark_count_prob ∈ [0, 1]

    **Literature References:**
    
    - König et al. (2012): NSM definition, storage capacity constraint
    - Schaffner et al. (2009): 11% QBER threshold, depolarizing analysis
    - Erven et al. (2014): Experimental parameters (Table I)

    Examples
    --------
    >>> params = NSMParameters(
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ...     delta_t_ns=1_000_000,
    ...     channel_fidelity=0.95,
    ...     detection_eff_eta=0.015,
    ...     detector_error=0.0093,
    ...     dark_count_prob=1.5e-8,
    ... )
    >>> params.qber_storage
    0.125
    >>> params.verify_security_condition()
    True
    """
    
    storage_noise_r: float
    storage_rate_nu: float
    delta_t_ns: float
    channel_fidelity: float
    detection_eff_eta: float = 1.0
    detector_error: float = 0.0
    dark_count_prob: float = 0.0

    def __post_init__(self) -> None:
        """Validate all invariants after initialization."""
        ...

    @property
    def depolar_prob(self) -> float:
        """NetSquid depolarization probability = 1 - r."""
        ...

    @property
    def qber_storage(self) -> float:
        """Storage-induced QBER threshold = (1 - r) / 2."""
        ...

    @property
    def qber_channel(self) -> float:
        """Total channel QBER using Erven et al. formula."""
        ...

    @property
    def storage_capacity(self) -> float:
        """Classical capacity C_N = 1 - h((1-r)/2) of depolarizing channel."""
        ...

    def verify_security_condition(self) -> bool:
        """
        Verify fundamental NSM security condition.

        Returns
        -------
        bool
            True if Q_channel < Q_storage AND C_N · ν < 1/2.

        Raises
        ------
        SecurityError
            If security condition is violated.
        """
        ...

    @classmethod
    def from_erven_experimental(cls) -> "NSMParameters":
        """Create parameters matching Erven et al. (2014) Table I."""
        ...
```

### 3.2 ChannelNoiseProfile

**Location:** `caligo/simulation/noise_models.py`

```python
@dataclass(frozen=True)
class ChannelNoiseProfile:
    """
    Aggregate noise profile for the trusted quantum channel.

    Combines multiple noise sources into a unified QBER estimate using
    the Erven et al. (2014) formulation.

    Parameters
    ----------
    source_fidelity : float
        EPR source fidelity F ∈ (0.5, 1]. Intrinsic pair quality.
    detector_efficiency : float
        Combined detection efficiency η ∈ (0, 1].
    detector_error : float
        Intrinsic detector error e_det ∈ [0, 0.5].
    dark_count_rate : float
        Dark count probability P_dark ∈ [0, 1].

    Attributes
    ----------
    total_qber : float
        Combined QBER = Q_source + Q_det + Q_dark.
    is_secure : bool
        True if total_qber < 0.11 (Schaffner threshold).
    is_feasible : bool
        True if total_qber < 0.22 (König hard limit).

    Notes
    -----
    **QBER Components (Erven et al. 2014):**
    
    - Q_source = (1 - F) / 2
    - Q_det = e_det  
    - Q_dark = (1 - η) · P_dark · 0.5
    - Q_total = Q_source + Q_det + Q_dark
    """

    source_fidelity: float
    detector_efficiency: float
    detector_error: float
    dark_count_rate: float

    @property
    def total_qber(self) -> float:
        """Compute total QBER using Erven formula."""
        ...

    def to_double_click_params(self) -> "DoubleClickModelParameters":
        """
        Convert to netsquid_magic DoubleClickModelParameters.

        Returns
        -------
        DoubleClickModelParameters
            Parameters for heralded entanglement generation.
        """
        ...
```

### 3.3 TimingBarrier

**Location:** `caligo/simulation/timing.py`

```python
class TimingBarrier:
    """
    Enforces NSM timing constraint Δt in discrete-event simulation.

    Implements a state machine ensuring Alice cannot reveal basis choices
    until time Δt has elapsed since quantum phase completed.

    Parameters
    ----------
    delta_t_ns : float
        Required wait time in nanoseconds.
    strict_mode : bool
        If True, raises TimingViolationError on violations. Default: True.

    Attributes
    ----------
    state : TimingBarrierState
        Current state: IDLE → WAITING → READY.
    quantum_complete_time : Optional[float]
        Simulation time when quantum phase ended.
    timing_compliant : bool
        True if protocol respected timing constraints.

    Notes
    -----
    **State Machine:**
    
    ```
    IDLE ──[mark_quantum_complete()]──► WAITING ──[wait_delta_t()]──► READY
      ▲                                                                 │
      └─────────────────────────── reset() ─────────────────────────────┘
    ```

    **Integration with SquidASM:**
    
    The `wait_delta_t()` method is a generator that yields to NetSquid's
    discrete-event simulator via `ns.sim_run(duration=remaining)`.
    """

    def __init__(self, delta_t_ns: float, strict_mode: bool = True) -> None:
        ...

    def mark_quantum_complete(self) -> None:
        """
        Record quantum phase completion timestamp.

        Transitions state from IDLE to WAITING.
        Records current simulation time via ns.sim_time().

        Raises
        ------
        TimingViolationError
            If called when state is not IDLE.
        """
        ...

    def wait_delta_t(self) -> Generator[EventExpression, None, None]:
        """
        Yield control to simulator for Δt nanoseconds.

        Transitions state from WAITING to READY after wait completes.

        Yields
        ------
        EventExpression
            NetSquid event expression for simulation advancement.

        Raises
        ------
        TimingViolationError
            If called when state is not WAITING.
        """
        ...

    def can_reveal_basis(self) -> bool:
        """Check if basis revelation is permitted (state == READY)."""
        ...
```

### 3.4 CaligoNetworkBuilder (Runtime Injection Adapter)

**Location:** `caligo/simulation/network_builder.py`

This is the *primary* integration surface between Caligo’s NSM parameters and the SquidASM execution pipeline. It produces a SquidASM `StackNetworkConfig` that can be passed directly to SquidASM’s stack runner.

Key responsibilities:

- Select the correct netsquid-netbuilder link model (`"perfect"`, `"depolarise"`, `"heralded-double-click"`)
- Provide the correct **config payload** for that model (dict or config object)
- Configure device-level noise via `GenericQDeviceConfig` (T1/T2, gate depolarization)
- Remain compatible with SquidASM’s `StackNetworkConfig` conversion (`squidasm/run/stack/config.py::_convert_stack_network_config`)

### 3.5 Feasibility / “Strictly Less” Validation

**Location:** `caligo/security/feasibility.py`

Caligo already contains a pre-flight validation layer that operationalizes literature requirements from Schaffner/Wehner/Erven:

- QBER thresholds (11% conservative, 22% absolute)
- Storage capacity constraint $C_{\mathcal{N}}\,\nu < 1/2$
- “Strictly less” channel-vs-storage noise condition (as a feasibility check)

This spec treats feasibility checks as the canonical place to enforce *analytic* NSM constraints (those that cannot be simulated because the adversary is not modeled as an in-simulator agent).

---

## 4. NSM-to-Physical Parameter Mapping

### 4.1 Literature-Derived Formulas


All formulas in this section are directly extracted from published literature with precise citations.

#### 4.1.1 Depolarizing Storage Channel (Wehner, Schaffner, Terhal 2008)

From *Cryptography from Noisy Storage* (PRL 100, 220502):

> "Let $\mathcal{N}(\rho) := r\rho + (1-r)(\mathbf{1}/2)$ be the fixed depolarizing 'quantum-storage' channel that Bob cannot influence."

The NSM models adversary storage as a depolarizing channel:

$$
\mathcal{N}_r(\rho) = r \cdot \rho + (1-r) \cdot \frac{\mathbf{I}}{d}
$$

Where:
- $r \in [0, 1]$: Preservation probability (state retention)
- $d = 2$: Qubit dimension
- $(1-r)$: Depolarization probability → **NetSquid `depolar_rate`**

**All-or-Nothing Result (Wehner et al. 2008):**

> "For $r \geq 1/\sqrt{2}$ we have $\max_{S_i} \Delta(S_i) = 1$ and for $r < 1/\sqrt{2}$, $\max_{S_i} \Delta(S_i) = \frac{1}{2} + \frac{r}{2\sqrt{2}}$."

This defines the threshold where adversary strategy switches from measurement to storage.

#### 4.1.2 Classical Capacity (Schaffner, Terhal, Wehner 2009)

From *Robust Cryptography in the Noisy-Quantum-Storage Model* (QIC 9:11&12, Eq. 6):

> "A memory has one other pertinent parameter we need; namely, a storage rate, $\nu$, which represents the fraction of qubits which an adversarial Bob can store in memory."

**Depolarizing Channel Definition (Eq. 6):**
$$
\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{\mathbf{I}}{d} \quad \text{for } 0 \le r \le 1
$$

**Security requires (from Erven et al. 2014, Eq. 7):**
$$
C_{\mathcal{N}_r, \nu n} < \frac{C_{\text{BB84}} \cdot m_1}{2} - 1 - \log_2\left(\frac{1}{2\varepsilon}\right)
$$

**Classical Capacity of Depolarizing Channel:**
$$
C_{\mathcal{N}} = 1 - h\left(\frac{1-r}{2}\right)
$$

Where $h(x) = -x \log_2(x) - (1-x) \log_2(1-x)$ is binary entropy.

#### 4.1.3 QBER Security Threshold (Schaffner et al. 2009, Table 1)

From *Robust Cryptography in the Noisy-Quantum-Storage Model*:

> "We can obtain secure oblivious transfer as long as the quantum bit-error rate of the channel does not exceed **11%** and the noise on the channel is **strictly less** than the quantum storage noise. This is optimal for the protocol considered."

**Security Condition:**
$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1-r}{2}
$$

**QBER Thresholds:**
- **11% (Schaffner):** Conservative threshold for depolarizing storage
- **22% (König):** Absolute maximum (hard limit)

#### 4.1.4 Channel QBER Components (Erven et al. 2014)

From *An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model*, the total channel QBER comprises three components:

**Source Error (from fidelity):**
$$
Q_{\text{source}} = \frac{1 - F}{2}
$$

**Detector Error:**
$$
Q_{\text{det}} = e_{\text{det}}
$$

**Dark Count Error:**
$$
Q_{\text{dark}} = (1 - \eta) \cdot P_{\text{dark}} \cdot 0.5
$$

**Total QBER:**
$$
Q_{\text{channel}} = Q_{\text{source}} + Q_{\text{det}} + Q_{\text{dark}}
$$

#### 4.1.5 Honest Bob Detection Probability (Wehner et al. 2010, Eq. 3)

From *Implementation of two-party protocols in the noisy-storage model* (Phys. Rev. A 81, 052336):

$$
P_{B,\text{no click}}^h = \sum_{n=0}^{\infty} P_{\text{src}}^n P_{B,\text{no click}}^{h|n}
$$

**Click probability:**
$$
P_{B,\text{click}}^h = \eta + (1 - \eta) \cdot P_{\text{dark}}
$$

**Conditional Error Rate:**
$$
P_{B,\text{err}}^h = \frac{\eta \cdot (Q_{\text{source}} + Q_{\text{det}}) + (1-\eta) \cdot P_{\text{dark}} \cdot 0.5}{P_{B,\text{click}}^h}
$$

#### 4.1.6 ROT Rate Formula (Erven et al. 2014, Eq. 8)

The secure ROT string length formula:

$$
l \le \frac{1}{2}\nu \gamma^{\mathcal{N}_r}\left(\frac{R}{\nu}\right)\frac{m}{n} - n \cdot f \cdot h(p_{\text{err}}) - \log_2\left(\frac{1}{2\varepsilon}\right)
$$

Where:
- $\gamma^{\mathcal{N}_r}$: Strong converse parameter of adversary's memory
- $R$: Rate at which dishonest Bob would need to store quantum information
- $m$: Number of rounds where both parties measured
- $f$: Error correction efficiency relative to Shannon limit
- $p_{\text{err}}$: Total error probability

### 4.2 Parameter Translation Table

| NSM Parameter | Symbol | Where it lives | Translation / Injection |
|--------------|--------|----------------|--------------------------|
| Storage noise | $r$ | Analytic security bound (not simulated adversary) | $Q_{\text{storage}}=(1-r)/2$ and $C_{\mathcal{N}}=1-h(\frac{1-r}{2})$ |
| Storage rate | $\nu$ | Analytic security bound | Check $C_{\mathcal{N}}\,\nu < 1/2$ |
| Wait time | $\Delta t$ | SquidASM program runtime | `TimingBarrier(delta_t_ns)` enforced in `Program.run()` |
| Channel fidelity | $F$ | netsquid-netbuilder qlink config | **Depolarise link:** `DepolariseQLinkConfig.fidelity = F` and internal $p_{\max\,mix}=\frac{4}{3}(1-F)$ |
| Detection efficiency | $\eta$ | netsquid-netbuilder qlink config | **Heralded double-click:** `HeraldedDoubleClickQLinkConfig.detector_efficiency` (or split into fiber loss × detector eff) |
| Detector error | $e_{\text{det}}$ | netsquid-netbuilder qdevice config | `GenericQDeviceConfig.*_gate_depolar_prob ≈ 2 e_{\text{det}}` (documented approximation) |
| Dark count | $P_{\text{dark}}$ | netsquid-netbuilder qlink config | `HeraldedDoubleClickQLinkConfig.dark_count_probability = P_dark` |

### 4.3 Erven et al. (2014) Experimental Values

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Mean photon number | $\mu$ | $3.145 \times 10^{-5}$ | Per coherence time |
| Transmittance | $\eta$ | $0.0150$ | Combined efficiency |
| Detection error | $e_{\text{det}}$ | $0.0093$ | Intrinsic error rate |
| Dark count prob | $P_{\text{dark}}$ | $1.50 \times 10^{-8}$ | Per pulse |
| Storage noise | $r$ | $0.75$ | Depolarizing parameter |
| Storage rate | $\nu$ | $0.002$ | Fraction storable |

---

## 5. SquidASM Pipeline Injection

This section is the implementation core of the spec. It answers a single question:

> How do Caligo’s NSM parameters become **real NetSquid noise processes** inside SquidASM when we run Phase E?

The key insight (mirroring the enforcement spec) is that Caligo does **not** inject `netsquid.components.models.qerrormodels.*` objects directly.

Instead, Caligo injects **configuration objects** into SquidASM:

- SquidASM converts `StackNetworkConfig` → netsquid-netbuilder `NetworkConfig`.
- netsquid-netbuilder selects registered qlink/qdevice builders by string `typ`.
- those builders construct the `netsquid_magic` distributor protocols and the NetSquid noise models.

This distinction matters because it determines the *correct API boundary* for runtime injection.

### 5.1 Injection Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SQUIDASM INJECTION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CALIGO LAYER                                                              │
│   ════════════                                                              │
│                                                                             │
│   NSMParameters + ChannelParameters                                         │
│        │                                                                    │
│        ├──► CaligoNetworkBuilder ──► StackNetworkConfig                      │
│        │                              ├── stacks[*].qdevice_cfg (Generic)   │
│        │                              └── links[*] (typ + cfg)              │
│        │                                                                    │
│        └──► TimingBarrier ─────────► AliceProgram (Δt enforced in run())    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SQUIDASM STACK RUNNER                                                     │
│   ══════════════════                                                        │
│                                                                             │
│   run(StackNetworkConfig, programs)                                         │
│        │                                                                    │
│        └──► _convert_stack_network_config()                                 │
│                ("heralded" → "heralded-double-click")                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   NETSQUID_NETBUILDER                                                       │
│   ═════════════════                                                         │
│                                                                             │
│   Registered qlink models:                                                  │
│     "perfect", "depolarise", "heralded-single-click", "heralded-double-click"│
│   Builds netsquid_magic distributors + NetSquid processors/noise             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Link-Level Injection (EPR Noise)

**Target:** EPR pair generation noise via `LinkConfig`

**Injection Point:** `squidasm/run/stack/config.py::LinkConfig` (and downstream conversion via `_convert_stack_network_config`)

```python
@dataclass
class LinkConfig:
    stack1: str
    stack2: str
    # Important: these strings must match netsquid-netbuilder model names.
    # Link models registered by netsquid-netbuilder:
    # "perfect", "depolarise", "heralded-single-click", "heralded-double-click"
    typ: str
    cfg: Optional[Dict[str, Any]] = None
```

#### 5.2.1 What “cfg” Really Is

In SquidASM `LinkConfig.cfg` is passed through to netsquid-netbuilder. It can be either:

- A dict that can be unpacked into the corresponding `...QLinkConfig` dataclass (netbuilder will construct it), **or**
- An explicit config object, e.g. `DepolariseLinkConfig(...)` / `HeraldedLinkConfig(...)` from `squidasm/run/stack/config.py`.

If a dict is used, it must contain **all required fields** of the target config dataclass. In particular, depolarising links require `prob_success` and either `t_cycle` or (`length`, `speed_of_light`).

This is the most common runtime-injection failure mode: the pipeline appears correct, but the builder fails during preprocessing because the dict is incomplete.

#### 5.2.2 Canonical Injection Flow (Runtime)

Caligo ultimately calls SquidASM’s stack runner:

```python
from squidasm.run.stack.run import run as squidasm_run

results = squidasm_run(stack_network_config, {"Alice": alice_program, "Bob": bob_program})
```

The runner accepts either a `StackNetworkConfig` or a netsquid-netbuilder `NetworkConfig`. If given `StackNetworkConfig`, SquidASM will convert it internally using `_convert_stack_network_config`.

This makes **`CaligoNetworkBuilder`** the correct injection point for all runtime noise configuration.

#### 5.2.3 Implementation: Depolarising Link (Fidelity-Only)

This is the minimal, already-supported injection pattern and corresponds to the “current SquidASM default configuration” called out in the enforcement spec.

**Correct payload (config object form, preferred):**

```python
from squidasm.run.stack.config import DepolariseLinkConfig, LinkConfig

link_cfg = DepolariseLinkConfig(
    fidelity=float(nsm_params.channel_fidelity),
    prob_success=1.0,
    t_cycle=float(channel_params.cycle_time_ns),
    random_bell_state=False,
)

link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=link_cfg)
```

**Correct payload (dict form):**

```python
link = LinkConfig(
    stack1="Alice",
    stack2="Bob",
    typ="depolarise",
    cfg={
        "fidelity": float(nsm_params.channel_fidelity),
        "prob_success": 1.0,
        "t_cycle": float(channel_params.cycle_time_ns),
        "random_bell_state": False,
    },
)
```

**How fidelity becomes depolarising noise** (what actually happens in the pipeline):

- netsquid-netbuilder converts $F$ to a maximally-mixed fraction using `fidelity_to_prob_max_mixed(F)`.
- For a depolarised Bell state $\rho=(1-p)|\Phi^+\rangle\langle\Phi^+| + p\,I/4$, fidelity satisfies:

$$
F = 1 - \frac{3p}{4}\quad\Rightarrow\quad p = \frac{4}{3}(1-F)
$$

This corrects a common (but slightly wrong) shortcut `prob_max_mixed = 1 - F`.

#### 5.2.4 Implementation: Heralded Double-Click Link (η, P_dark, visibility)

This is the enforcement spec’s main missing feature: explicit detection efficiency and dark counts in the entanglement generation layer.

**Correct model name:** `typ="heralded-double-click"`.

**Correct payload (config object form, preferred):**

```python
from squidasm.run.stack.config import HeraldedLinkConfig, LinkConfig

visibility = max(0.0, min(1.0, 2.0 * float(nsm_params.channel_fidelity) - 1.0))

qlink_cfg = HeraldedLinkConfig(
    # Fiber / propagation
    length=float(channel_params.length_km),
    p_loss_length=float(channel_params.attenuation_db_per_km),
    speed_of_light=float(channel_params.speed_of_light_km_s),
    # Detection model
    detector_efficiency=float(nsm_params.detection_eff_eta),
    dark_count_probability=float(nsm_params.dark_count_prob),
    visibility=visibility,
    # Multiplexing / coincidence defaults
    num_multiplexing_modes=1,
    num_resolving=False,
    coin_prob_ph_ph=1.0,
    coin_prob_ph_dc=1.0,
    coin_prob_dc_dc=1.0,
)

link = LinkConfig(
    stack1="Alice",
    stack2="Bob",
    typ="heralded-double-click",
    cfg=qlink_cfg,
)
```

**Optional: split η into fiber loss × detector efficiency**

In NSM notation, $\eta$ is often used as a *combined* transmittance/detection efficiency (Erven Table I). The heralded-double-click model separates fiber loss and detector efficiency.

If Caligo is configured with a physical channel length $L$ and attenuation $\alpha$ (dB/km), one can compute fiber transmittance:

$$
\eta_{\text{fiber}} = 10^{-\alpha L / 10}
$$

and then set:

$$
\eta_{\text{det}} = \min\left(1, \frac{\eta_{\text{total}}}{\eta_{\text{fiber}}}\right)
$$

This keeps the operational meaning of $\eta$ consistent with Erven/Wehner while using the more detailed netbuilder model.

**Implementation note:** Caligo already implements `CaligoNetworkBuilder` in `caligo/simulation/network_builder.py` with the authoritative signature `(nsm_params, channel_params)`. The current implementation supports `"perfect"` and `"depolarise"`; this spec requires extending it to support `"heralded-double-click"` when $\eta$/$P_{\text{dark}}$ must be injected at runtime.

### 5.3 Device-Level Injection (Memory/Gate Noise)

**Target:** Quantum processor noise via `GenericQDeviceConfig`

**Injection Point:** `netsquid_netbuilder/modules/qdevices/generic.py`

**Key Parameters:**

| Parameter | Effect | NSM Source |
|-----------|--------|------------|
| `T1` | Amplitude damping time | ChannelParameters.t1_ns |
| `T2` | Dephasing time | ChannelParameters.t2_ns |
| `single_qubit_gate_depolar_prob` | Gate error rate | 2 × detector_error |
| `num_qubits` | Memory capacity | Protocol requirement |

#### 5.3.1 What This Actually Models (and What It Does Not)

- These parameters model **honest-party device noise** (memory decoherence and gate imperfections) in the SquidASM node stacks.
- They do **not** model adversary storage noise $r$ (the dishonest Bob is not an in-simulator actor; see enforcement spec Section 3.2.4).

This matters for interpretation: $r$ and $\nu$ must be enforced in *analytic* security checks (capacity bounds, min-entropy bounds), while the simulation must faithfully measure $Q_{\text{channel}}$ under the configured channel model.

### 5.4 Protocol-Level Injection (Timing)

**Target:** CaligoProgram base class

**Injection Flow:**

```python
class CaligoProgram(Program, ABC):
    """Base class for Caligo protocol programs with timing enforcement."""

    def __init__(self, params: ProtocolParameters) -> None:
        self._params = params
        self._timing_barrier = TimingBarrier(
            delta_t_ns=params.nsm_params.delta_t_ns
        )

    def run(self, context: Any) -> Generator[Any, None, Dict[str, Any]]:
        """Protocol execution with timing barrier integration."""
        # ... quantum phase ...

        # Mark quantum phase complete
        self._timing_barrier.mark_quantum_complete()

        # Wait for Δt (yields to NetSquid simulator)
        yield from self._timing_barrier.wait_delta_t()

        # Now safe to reveal basis
        yield from self._reveal_basis(context)
```

#### 5.4.1 Why Timing Is the Only “Adversary Storage” Enforcement in Simulation

As the enforcement spec emphasizes, the adversary is not explicitly simulated. The only “physical enforcement” of the NSM storage assumption inside the simulator is the causal structure created by the enforced wait $\Delta t$:

- Alice’s basis information is unavailable during $[t_{\text{quantum\_done}},\, t_{\text{quantum\_done}} + \Delta t)$.
- This matches the NSM assumption that any stored quantum information must undergo $\mathcal{F}_{\Delta t}$ before it can be exploited.

Everything else about adversary storage quality ($r$) and quantity ($\nu$) is necessarily analytic.

### 5.5 Runtime Injection Failure Modes (Checklist)

These issues are easy to miss because they occur *below* Caligo, inside SquidASM/netbuilder:

1. Wrong qlink model name
    - Use `"heralded-double-click"`, not `"heralded_double_click"`.
    - `"heralded"` is accepted by SquidASM and converted to `"heralded-double-click"`.
2. Incomplete config dicts
    - Depolarise links require `prob_success` and `t_cycle` (or `length` + `speed_of_light`).
3. Ambiguous meaning of $\eta$
    - If $\eta$ is “total detection efficiency”, decide whether it maps entirely to `detector_efficiency` (short links), or is split between fiber loss and detector efficiency (realistic links).
4. Hidden defaults
    - Heralded-double-click has many defaults; specify explicitly in tests to ensure determinism.
5. Environment incompatibilities
    - Some environments combine NetQASM 2.x with SquidASM versions that cannot execute the emitted instruction set (MOV, etc.). E2E tests must skip or pin versions accordingly.

---

## 6. Squid Package Dependencies

This section documents exact class signatures and file locations from the squid packages for NSM parameter injection.

### 6.1 NetSquid Core (`netsquid`)

**Package Path:** `netsquid/components/models/qerrormodels.py`

#### 6.1.1 QuantumErrorModel (Base Class)

```python
class QuantumErrorModel(Model):
    """
    Base class for quantum error models.

    File: netsquid/components/models/qerrormodels.py
    """

    def error_operation(
        self,
        qubits: List[Qubit],
        delta_time: float = 0,
        **kwargs,
    ) -> None:
        """
        Apply error to qubits.

        Parameters
        ----------
        qubits : List[Qubit]
            Qubits to apply noise to.
        delta_time : float
            Time elapsed since last operation (ns).
        """
        ...
```

#### 6.1.2 DepolarNoiseModel

```python
class DepolarNoiseModel(QuantumErrorModel):
    """
    Depolarizing noise model for qubits.

    File: netsquid/components/models/qerrormodels.py

    Parameters
    ----------
    depolar_rate : float
        Depolarization probability per time unit (or total if time_independent).
    time_independent : bool
        If True, depolar_rate is total probability regardless of time.
    """

    def __init__(
        self,
        depolar_rate: float = 0,
        time_independent: bool = False,
    ) -> None:
        ...

    @property
    def depolar_rate(self) -> float:
        """Depolarization rate."""
        ...
```

**NSM Mapping:** `depolar_rate = 1 - r` (from Wehner et al. depolarizing channel definition)

**Important clarification:** In the current Caligo/SquidASM architecture, this `DepolarNoiseModel` mapping is primarily used for *theoretical modeling* (e.g., storage bounds) and honest-device noise modeling. The adversary storage channel $\mathcal{N}_r$ is not instantiated as an in-simulator component.

#### 6.1.3 T1T2NoiseModel

```python
class T1T2NoiseModel(QuantumErrorModel):
    """
    Combined T1 (amplitude damping) and T2 (dephasing) noise.

    File: netsquid/components/models/qerrormodels.py

    Parameters
    ----------
    T1 : float
        Amplitude damping time constant (ns).
    T2 : float
        Dephasing time constant (ns). Must satisfy T2 ≤ T1.
    """

    def __init__(self, T1: float = 0, T2: float = 0) -> None:
        ...
```

### 6.2 netsquid_magic

**Package Path:** `netsquid_magic/`

#### 6.2.1 IModelParameters Base (Exact Signature)

```python
# File: netsquid_magic/model_parameters.py:8

@dataclasses.dataclass
class IModelParameters(metaclass=ABCMeta):
    """Abstract base data class for the parameters of entanglement generation models."""
    cycle_time: float = 0
    """Duration [ns] of each round of entanglement distribution."""

    @abstractmethod
    def verify(self):
        self.verify_not_negative_value("cycle_time")
```

#### 6.2.2 DepolariseModelParameters (Exact Signature)

```python
# File: netsquid_magic/model_parameters.py:63

@dataclasses.dataclass
class DepolariseModelParameters(IModelParameters):
    """Data class for the parameters of the depolarising entanglement generation model."""
    prob_max_mixed: float = 0
    """Fraction of maximally mixed state in the EPR state generated."""
    prob_success: float = 1
    """Probability of successfully generating an EPR state per cycle."""
    random_bell_state: bool = False
    """Determines whether the Bell state is always phi+ or randomly chosen."""
```

**NSM Mapping:**
- `prob_max_mixed = fidelity_to_prob_max_mixed(F) = \frac{4}{3}(1-F)` for depolarised Bell pairs
- Maps to QBER via: $Q_{\text{source}} = \frac{\text{prob\_max\_mixed}}{2}$

#### 6.2.3 DoubleClickModelParameters (Exact Signature)

```python
# File: netsquid_magic/model_parameters.py:202

@dataclasses.dataclass
class DoubleClickModelParameters(HeraldedModelParameters):
    """Data class for the parameters of the double click entanglement generation model."""
    detector_efficiency: float = 1
    """Probability that the presence of a photon leads to a detection event."""
    dark_count_probability: float = 0
    """Dark-count probability per detection."""
    visibility: float = 1
    """Hong-Ou-Mandel visibility of photons being interfered."""
    num_resolving: bool = False
    """Whether photon-number-resolving detectors are used."""
    num_multiplexing_modes: int = 1
    """Number of modes used for multiplexing."""
    emission_fidelity_A: float = 1
    """Fidelity of state on A side to PHI_PLUS after emission."""
    emission_fidelity_B: float = 1
    """Fidelity of state on B side to PHI_PLUS after emission."""
    coin_prob_ph_ph: float = 1
    """Coincidence probability for two photons."""
    coin_prob_ph_dc: float = 1
    """Coincidence probability for photon and dark count."""
    coin_prob_dc_dc: float = 1
    """Coincidence probability for two dark counts."""
```

**NSM Parameter Mapping:**
| NSM Parameter | DoubleClickModelParameters Field |
|--------------|----------------------------------|
| $\eta$ (detection efficiency) | `detector_efficiency` |
| $P_{\text{dark}}$ (dark count) | `dark_count_probability` |
| $F$ (fidelity) | `visibility` (via $V = 2F - 1$) |

#### 6.2.4 StateDeliverySampler (Exact Signature)

```python
# File: netsquid_magic/state_delivery_sampler.py:17

DeliverySample = namedtuple("DeliverySample", ["state", "delivery_duration", "label"])

class StateDeliverySampler(StateSampler):
    """Class for sampling both a quantum state and the time its generation took."""

    def __init__(
        self,
        state_sampler: StateSampler,
        cycle_time: float,
        long_distance_interface: ILongDistanceInterface = None,
    ) -> None:
        ...

    def sample(self, skip_rounds: bool = True) -> DeliverySample:
        """
        Samples quantum state and generation time.

        Returns
        -------
        DeliverySample
            Named tuple: (state, delivery_duration, label)
        """
        ...
```

### 6.3 netsquid_netbuilder

**Package Path:** `netsquid_netbuilder/modules/`

#### 6.3.1 GenericQDeviceConfig (via SquidASM)

```python
# File: squidasm/run/stack/config.py:13 (imports from netsquid_netbuilder)

class GenericQDeviceConfig(netbuilder_qdevices.GenericQDeviceConfig):
    """Configuration for generic quantum devices.
    
    Key attributes for NSM mapping:
    - num_qubits: int - Number of qubit positions
    - T1: float - Amplitude damping time (ns)
    - T2: float - Dephasing time (ns), T2 ≤ T1
    - single_qubit_gate_depolar_prob: float - Gate error rate
    """
    
    @classmethod
    def perfect_config(cls) -> "GenericQDeviceConfig":
        """Create noise-free configuration."""
        ...
```

### 6.4 SquidASM Configuration Classes

**Package Path:** `squidasm/run/stack/config.py`

#### 6.4.1 StackConfig (Exact Signature)

```python
# File: squidasm/run/stack/config.py:22

class StackConfig(YamlLoadable):
    """Configuration for a single stack (i.e. end node)."""

    name: str
    """Name of the stack."""
    qdevice_typ: str
    """Type of the quantum device."""
    qdevice_cfg: Any = None
    """Configuration of the quantum device, allowed configuration depends on type."""

    @classmethod
    def perfect_generic_config(cls, name: str) -> StackConfig:
        """Create a configuration for a stack with a generic quantum device 
        without any noise or errors."""
        return StackConfig(
            name=name,
            qdevice_typ="generic",
            qdevice_cfg=netbuilder_qdevices.GenericQDeviceConfig.perfect_config(),
        )
```

#### 6.4.2 LinkConfig (Exact Signature)

```python
# File: squidasm/run/stack/config.py:57

class LinkConfig(YamlLoadable):
    """Configuration for a single link."""

    stack1: str
    """Name of the first stack being connected via link."""
    stack2: str
    """Name of the second stack being connected via link."""
    typ: str
    """Type of the link."""
    cfg: Any = None
    """Configuration of the link, allowed configuration depends on type."""

    @classmethod
    def perfect_config(cls, stack1: str, stack2: str) -> LinkConfig:
        """Create a configuration for a link without any noise or errors."""
        return LinkConfig(stack1=stack1, stack2=stack2, typ="perfect", cfg=None)
```

**Link Types and NSM Mapping:**
| typ | Model | NSM Parameters Used |
|-----|-------|---------------------|
| `"perfect"` | No noise | None (baseline testing) |
| `"depolarise"` | `DepolariseModelParameters` | `channel_fidelity` → `fidelity` |
| `"heralded"` | Alias (SquidASM conversion) | Converted to `"heralded-double-click"` |
| `"heralded-double-click"` | Double-click distributor model | Full: $\eta$, $P_{\text{dark}}$, visibility ($\approx 2F-1$) |

#### 6.4.3 StackNetworkConfig (Exact Signature)

```python
# File: squidasm/run/stack/config.py:95

class StackNetworkConfig(YamlLoadable):
    """Full network configuration."""

    stacks: List[StackConfig]
    """List of all the stacks in the network."""
    links: List[LinkConfig]
    """List of all the links connecting the stacks in the network."""
    clinks: Optional[List[CLinkConfig]] = None
    """List of all the classical links connecting the stacks in the network."""

    @classmethod
    def from_file(cls, path: str) -> StackNetworkConfig:
        return super().from_file(path)
```

#### 6.4.4 Configuration Conversion (Internal)

```python
# File: squidasm/run/stack/config.py:108

def _convert_stack_network_config(
    stack_network_config: StackNetworkConfig,
) -> netbuilder_configs.NetworkConfig:
    """Method to convert a StackNetworkConfig into a netsquid-netbuilder 
    NetworkConfig object.
    
    Key transformations:
    - link_typ "heralded" → "heralded-double-click"
    - link_cfg None + typ "perfect" → PerfectQLinkConfig()
    - Creates ProcessingNodeConfig from each StackConfig
    - Creates QLinkConfig from each LinkConfig
    """
    ...
```

### 6.5 Injection Point Summary

| Injection Level | Target Class | NSM Parameters | NetSquid Effect |
|----------------|--------------|----------------|-----------------|
| **Link** | `LinkConfig` | $F$, $\eta$, $P_{\text{dark}}$ | EPR state noise |
| **Device** | `GenericQDeviceConfig` | T1, T2, $e_{\text{det}}$ | Memory/gate noise |
| **Protocol** | `CaligoProgram` | $\Delta t$ | Timing enforcement |

---

## 7. Confrontation With Enforcement Spec

This section explicitly aligns (and, where needed, corrects) this implementation spec against [nsm_parameters_enforcement.md](nsm_parameters_enforcement.md). The enforcement document is treated as the “problem statement and gaps” source; this document is the “how we implement it” source.

### 7.1 Shared Security Foundation (Both Documents)

Both documents agree on the core NSM requirement:

$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1-r}{2}\;\;\wedge\;\; C_{\mathcal{N}}\,\nu < \frac{1}{2}
$$

and on the key operational QBER thresholds:

- 11% (Schaffner et al. 2009) as a conservative threshold for the depolarising-storage analysis
- 22% as an absolute hard limit (various references; encoded in Caligo constants)

### 7.2 Gap Table → Concrete Implementation Decisions

The enforcement spec lists several “missing NSM-critical parameters” at the SquidASM level. The table below translates each gap into a concrete Caligo implementation decision, including *where it is enforced* (simulation vs analytic).

| Enforcement Spec Gap | What it means | Where enforced | Concrete mechanism |
|----------------------|---------------|----------------|-------------------|
| $r$ not modeled | Adversary storage decoherence | Analytic | `NSMParameters.qber_storage`, `storage_capacity`, feasibility checks (`caligo/security/feasibility.py`) |
| $\nu$ not modeled | Adversary can store fraction of qubits | Analytic | `NSMParameters.storage_security_satisfied` and preflight report; affects key-rate bounds |
| $\Delta t$ not enforced | Alice might leak basis “too early” | Simulation | `TimingBarrier` integrated into `AliceProgram` and program base |
| $\eta$ missing | Detection efficiency affects QBER and click rates | Simulation (preferred), analytic fallback | Inject `heralded-double-click` qlink config; if not available, incorporate via analytic QBER estimator |
| $P_{\text{dark}}$ missing | Dark counts contribute to QBER | Simulation (preferred), analytic fallback | Inject `dark_count_probability` into heralded-double-click link; otherwise use Erven QBER formula |
| Source quality only via fidelity | Multi-parameter source model is simplified | Both | Keep fidelity-based depolarising link as baseline; optionally add PDC-model based feasibility computations (Erven) |

### 7.3 Explicit Corrections vs Enforcement Spec

1. Depolarising-link payload completeness
    - Enforcement spec examples show a minimal payload (`fidelity` only). In the actual netbuilder API, depolarising links require `prob_success` and a cycle time (`t_cycle`) or (`length`, `speed_of_light`).

2. Model-name precision
    - The canonical netbuilder model name is `"heralded-double-click"`.
    - SquidASM additionally accepts `"heralded"` and converts it to `"heralded-double-click"` during config conversion.

3. “Strictly less” interpretation
    - The enforcement spec frames this as “noise on the channel strictly less than storage noise.” In implementation, Caligo evaluates it via feasibility checks using entropy/capacity bounds (see `caligo/security/feasibility.py`).

### 7.4 What Must Be Measured in Simulation

To validate the NSM assumption empirically (as demanded by the enforcement spec), the simulator must measure:

- Channel QBER ($Q_{\text{channel}}$) from the sifting test sample
- Detection/click statistics (optional but strongly recommended if using heralded link models)
- Timing compliance (barrier state) to ensure $\Delta t$ was respected

These quantities are then compared against the analytic storage bounds derived from $r$ and $\nu$.

## 8. Implementation Contracts

This section defines **implementation-facing contracts** that keep the system aligned with the actual SquidASM/netbuilder injection boundary.

The critical rule is:

> Caligo injects *configs* (StackNetworkConfig, LinkConfig, GenericQDeviceConfig), not raw NetSquid noise model instances.

### 8.1 Link/QDevice Injection Contracts (Config-First)

```python
from typing import Any, Protocol


class ILinkConfigProvider(Protocol):
    """Create a SquidASM LinkConfig payload from Caligo parameters."""

    def build_link_config(
        self,
        alice_name: str,
        bob_name: str,
        nsm_params: "NSMParameters",
        channel_params: "ChannelParameters",
    ) -> Any:
        """Return a SquidASM LinkConfig (typ + cfg) with complete payload."""


class IQDeviceConfigProvider(Protocol):
    """Create a GenericQDeviceConfig payload from Caligo parameters."""

    def build_qdevice_config(
        self,
        num_qubits: int,
        nsm_params: "NSMParameters",
        channel_params: "ChannelParameters",
    ) -> Any:
        """Return a GenericQDeviceConfig (T1/T2 + gate depolarization)."""


class INetworkConfigProvider(Protocol):
    """Build a StackNetworkConfig suitable for squidasm.run()."""

    def build_two_node_network(
        self,
        alice_name: str,
        bob_name: str,
        num_qubits: int,
    ) -> Any:
        """Return a StackNetworkConfig with stacks + links configured."""
```

**Reference implementation:** `caligo/simulation/network_builder.py::CaligoNetworkBuilder`.

### 8.2 Timing Contract (Δt Enforcement)

**Requirement:** Alice’s program must enforce a wait of exactly $\Delta t$ between “quantum done” and “basis reveal”, implemented as a generator `yield` to NetSquid.

Contractually, this means:

- `TimingBarrier.mark_quantum_complete()` is called immediately after the quantum phase ends.
- `yield from TimingBarrier.wait_delta_t()` occurs before any classical information that would break the NSM assumption is sent.

### 8.3 Security/Feasibility Evaluation Contract (Analytic)

Because the adversary is not simulated, security is validated via analytic checks that consume *measured simulation outputs*.

```python
from typing import Protocol


class ISecurityEvaluator(Protocol):
    """Evaluate measured simulation outputs against NSM bounds."""

    def verify_capacity_bound(self, nsm_params: "NSMParameters") -> bool:
        """Check C_N · ν < 1/2."""

    def verify_channel_vs_storage(self, measured_qber: float, nsm_params: "NSMParameters") -> bool:
        """Check Q_channel < (1-r)/2 using measured_qber from the run."""

    def security_margin(self, measured_qber: float, nsm_params: "NSMParameters") -> float:
        """Return (1-r)/2 - measured_qber; positive indicates margin."""
```

**Reference location:** feasibility checks belong in `caligo/security/feasibility.py` (and related bounds modules).

---

## 9. Package Structure (Final)

This section finalizes the package/module structure **as it exists in the Caligo codebase today**, and specifies the *new or updated responsibilities* needed to fully realize the enforcement spec.

### 9.1 Current Caligo Reality (Authoritative)

The implementation lives under `qia-challenge-2025/caligo/caligo/` and already contains the correct high-level decomposition:

- `simulation/`
    - `constants.py` — literature constants (Erven Table I, 11%/22% thresholds)
    - `physical_model.py` — NSM/channel parameter dataclasses + derived analytic quantities
    - `noise_models.py` — QBER/entropy helpers and wrappers used as “theory → numbers” glue
    - `timing.py` — `TimingBarrier` and NetSquid-compatible waiting semantics
    - `network_builder.py` — runtime injection adapter producing SquidASM `StackNetworkConfig`

- `protocol/`
    - `base.py` — SquidASM `Program` integration + `TimingBarrier` ownership
    - `alice.py` / `bob.py` — Phase E programs; Alice enforces the timing barrier before basis reveal
    - `orchestrator.py` — runtime entrypoint calling `squidasm.run.stack.run.run`

- `security/`
    - `feasibility.py` — preflight security checks implementing “strictly less”, capacity bounds, QBER thresholds
    - `bounds.py`, `finite_key.py`, … — rate/bounds and finite-size handling

### 9.2 New / Updated Component Responsibilities (Spec-Driven)

No new files are created by this document, but it specifies *implementation changes* (to be performed later) in terms of responsibilities and testable behavior:

1. Updated: `caligo/simulation/network_builder.py`
     - Add a “link model selection” mode beyond {perfect,depolarise}:
         - Support `"heralded-double-click"` when NSM parameters include nontrivial $\eta$ and $P_{\text{dark}}$.
     - Ensure the link config payload is complete (`prob_success`, `t_cycle` for depolarise; required fields for heralded-double-click).

2. Updated: `caligo/simulation/physical_model.py`
     - Make the fidelity→depolarisation relation explicit in documentation and (if needed) helper functions:
         - $p_{\max\,mix}=\frac{4}{3}(1-F)$ for depolarised Bell states.
     - Keep the distinction between analytic adversary parameters ($r$, $\nu$) and simulated channel parameters explicit.

3. Updated: `caligo/security/feasibility.py`
     - Treat feasibility as the authoritative place for:
         - $C_{\mathcal{N}}\,\nu < 1/2$ capacity checks
         - “Strictly less” comparisons between channel error and storage noise
         - Minimum batch-size recommendations (finite-size effects)

4. Updated: `caligo/protocol/orchestrator.py`
     - Surface diagnostic outputs in raw results (optional):
         - Chosen qlink model name and relevant injected parameters
         - Timing compliance boolean
         - Measured QBER and sample sizes

## 10. Testing Strategy

Testing must mirror the architecture split:

- Analytic NSM constraints ($r$, $\nu$) are validated in pure-Python unit tests.
- Runtime injection correctness (SquidASM/netbuilder integration) is validated in conditional tests that import SquidASM/NetSquid.

### 10.1 Unit Tests (No SquidASM Required)

These tests run in environments where SquidASM/NetSquid may not be installed.

**Targets:**

- `caligo/simulation/physical_model.py`
    - Invariant enforcement in `NSMParameters.__post_init__`
    - Derived properties: `qber_channel`, `qber_storage`, `storage_capacity`, `storage_security_satisfied`
    - Regression tests for literature values (Erven Table I, thresholds)

- `caligo/security/feasibility.py`
    - `compute_expected_qber` matches the Erven decomposition:
        - $Q_{\text{source}}=(1-F)/2$, $Q_{\text{det}}=e_{\text{det}}$, $Q_{\text{dark}}=(1-\eta)P_{\text{dark}}/2$
    - “Strictly less” check behavior around boundary conditions

- `caligo/simulation/timing.py`
    - State machine transitions IDLE→WAITING→READY
    - Non-NetSquid fallback path yields at least once

**Placement:** extend existing `caligo/tests/test_simulation/` and `caligo/tests/test_security/` suites.

### 10.2 Integration Unit Tests (SquidASM Optional)

These tests verify that Caligo constructs *valid* SquidASM configs, but skip if SquidASM is absent.

**Targets:**

- `caligo/simulation/network_builder.py`
    - If SquidASM is present: `build_two_node_network()` returns a `StackNetworkConfig` whose `links[0].typ` is a registered model name.
    - Validate “payload completeness” by attempting to pass the config into SquidASM’s converter:
        - `squidasm.run.stack.config._convert_stack_network_config(network_cfg)`

This catches the most common runtime-injection failure (missing required qlink config fields) without needing a full end-to-end protocol run.

### 10.3 End-to-End Tests (Phase E)

These tests run the full protocol through SquidASM’s stack runner and validate outcome contracts.

**Existing anchor:** `caligo/tests/e2e/test_phase_e_protocol.py`.

**Required additions (spec-driven):**

- Parameterized E2E runs over:
    - depolarise link (fidelity-only)
    - heralded-double-click link (η/dark counts enabled)
- Assertions:
    - Protocol succeeds and key length is positive for feasible regimes
    - Measured QBER stays below thresholds when parameters are configured as such
    - Timing barrier compliance is true

**Environment guards:**

E2E must skip if the NetQASM/SquidASM instruction set combination is incompatible (as already hinted in the existing E2E test file). This is not a Caligo logic issue but an environment dependency constraint.

---

## Appendix A: Configuration Examples

### A.1 Erven et al. (2014) Experimental Configuration

```python
# Reproduce Erven et al. Table I parameters
nsm_params = NSMParameters(
    storage_noise_r=0.75,
    storage_rate_nu=0.002,
    delta_t_ns=1_000_000,  # 1 ms
    channel_fidelity=1.0 - 3.145e-5,  # From μ
    detection_eff_eta=0.0150,
    detector_error=0.0093,
    dark_count_prob=1.50e-8,
)

channel_params = ChannelParameters.from_erven_experimental()

# Build network
builder = CaligoNetworkBuilder(nsm_params, channel_params)
network_config = builder.build_two_node_network()
```

### A.2 Simplified Testing Configuration

```python
# For unit tests with controllable noise
nsm_params = NSMParameters.for_testing(
    storage_noise_r=0.75,
    channel_fidelity=0.95,
    delta_t_ns=1_000_000,
)

channel_params = ChannelParameters.for_testing()
builder = CaligoNetworkBuilder(nsm_params, channel_params)
network_config = builder.build_two_node_network()
```

---

## Appendix B: Literature Reference Summary

| Reference | Key Contribution | Parameters Used |
|-----------|-----------------|-----------------|
| König et al. (2012) | NSM definition, capacity bounds | $C_{\mathcal{N}}$, $\nu$ |
| Schaffner et al. (2009) | 11% QBER threshold, depolarizing analysis | $r$, QBER thresholds |
| Wehner et al. (2010) | Practical implementation, detection model | $P_{B,\text{click}}^h$, $P_{B,\text{err}}^h$ |
| Erven et al. (2014) | Experimental realization, Table I values | $\mu$, $\eta$, $e_{\text{det}}$, $P_{\text{dark}}$ |

---

*Document maintained by the Caligo development team. Last updated: December 18, 2025.*

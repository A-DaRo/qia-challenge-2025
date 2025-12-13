# Phase I Technical Analysis and Migration Guide: Quantum Generation & Physical Setup

**E-HOK Protocol on SquidASM/NetQASM/NetSquid Stack**

---

## Abstract

This document constitutes the authoritative technical analysis and migration guide for **Phase I** of the E-HOK (Ephemeral Honest-but-curious Oblivious Key) protocol, formalizing the foundational quantum generation layer onto the SquidASM/NetQASM/NetSquid simulation stack. Phase I establishes the physical bedrock of the protocol: the generation of correlated raw material (bit strings $s, \bar{s}$ and basis strings $a, \bar{a}$) via quantum entanglement, and the establishment of the **root of trust** derived from the **Noisy Storage Model (NSM)**.

The analysis is grounded in source-code-level inspection of the legacy `ehok/` implementation, the target `Squid` packages, and critically validated against the theoretical foundations in the referenced literature. The document culminates in a MoSCoW-prioritized development roadmap with formal invariants, pre-/post-conditions, and a deprecation strategy ensuring convergence to NSM-correct behavior.

**Central Conclusions:**

1. The SquidASM framework provides robust EPR generation and measurement primitives via `EPRSocket` and configurable noise models, satisfying the basic quantum channel requirements.
2. **Critical Gap**: No mechanism exists to enforce the NSM waiting window $\Delta t$ as a causal barrier between Bob's measurement commitment and Alice's basis revelation.
3. **Critical Gap**: No pre-flight feasibility check validates that trusted noise remains strictly below untrusted storage noise before resource consumption.
4. The commitment scheme in the legacy implementation is hash-based but lacks NSM "physical wait" enforcement semantics.
5. Noise model configuration in SquidASM expresses fidelity abstractly but does not expose NSM-specific parameters ($\mu$, $\eta$, $e_{det}$) required for rigorous security analysis.

---

## Table of Contents

1. [Literature Review & Theoretical Basis](#1-literature-review--theoretical-basis)
2. [Ontology & Terminology](#2-ontology--terminology)
3. [Traceability Matrix](#3-traceability-matrix)
4. [Phase I Deep-Dive & Gap Analysis](#4-phase-i-deep-dive--gap-analysis)
5. [Formalization of Physical and Protocol Constraints](#5-formalization-of-physical-and-protocol-constraints)
6. [MoSCoW-Prioritized Strategic Roadmap](#6-moscow-prioritized-strategic-roadmap)
7. [Formal System Invariants and Contracts](#7-formal-system-invariants-and-contracts)
8. [Deprecation Strategy and Migration Checkpoints](#8-deprecation-strategy-and-migration-checkpoints)

---

## 1. Literature Review & Theoretical Basis

Phase I's security derives from the **Noisy Storage Model (NSM)**, which assumes that any adversary's quantum storage is subject to noise that degrades stored quantum information over time. This section maps the theoretical contributions of the core literature to specific Phase I implementation requirements.

### 1.1 König, Wehner & Wullschleger (2012) — Unconditional Security from Noisy Quantum Storage

**Citation**: *IEEE Transactions on Information Theory, Vol. 58, No. 3*

**Core Contributions to Phase I**:

1. **Weak String Erasure (WSE) Primitive**: The E-HOK protocol implements WSE, wherein Alice generates random strings while Bob receives a noisy version. Security holds when the classical capacity of the adversary's storage channel, multiplied by the storage rate, satisfies:
   $$C_{\mathcal{N}} \cdot \nu < \frac{1}{2}$$
   This inequality directly constrains the minimum wait time $\Delta t$: it must be sufficient to degrade the adversary's effective channel capacity below this threshold.

2. **Timing Semantics**: The authors formalize that "whenever the protocol requires the adversary to wait for a time $\Delta t$, he has to measure/discard all his quantum information except what he can encode [...] into $H_{in}$. This information then undergoes noise described by $F$."

3. **Markovian Noise Assumption**: The noise model must be Markovian (Eq. 1 in the paper): $F_0 = \mathbb{1}$ and $F_{t_1+t_2} = F_{t_1} \circ F_{t_2}$, ensuring that delay cannot benefit the adversary.

**Phase I Mapping**:
- **Requirement PHI-R2**: The wait time $\Delta t$ must be a discrete-event simulation primitive, not merely message ordering.
- **Requirement PHI-R1**: Pre-flight checks must validate that $C_{\mathcal{N}} \cdot \nu < 1/2$ given calibrated storage parameters.

### 1.2 Schaffner, Terhal & Wehner (2009) — Robust Cryptography in the Noisy-Quantum-Storage Model

**Citation**: *Quantum Information and Computation, Vol. 9, No. 11&12*

**Core Contributions to Phase I**:

1. **QBER Tolerance Bound**: For individual-storage attacks with depolarizing noise, "1-2 oblivious transfer [...] can be achieved [...] as long as the quantum bit-error rate of the channel does not exceed **11%** and the noise on the channel is strictly less than the noise during quantum storage."

2. **The "Strictly Less" Condition**: The paper establishes that trusted noise (channel + device errors) must be **strictly less** than untrusted storage noise. This is the foundational security invariant for Phase I:
   $$Q_{trusted} < r_{storage}$$
   where $r$ parameterizes the depolarizing channel acting on the adversary's memory.

3. **Individual-Storage Attack Model**: The adversary may (i) partially measure qubits upon reception, (ii) store remaining qubits separately subject to noise $N_j$ per qubit, and (iii) perform coherent measurement after receiving Alice's basis information.

**Phase I Mapping**:
- **Requirement PHI-R1**: The 11% QBER threshold (conservative bound) and the strictly-less condition must be enforced as abort criteria.
- **Channel Model Binding**: NetSquid's `DepolarNoiseModel` must accurately reflect the trusted noise assumptions.

### 1.3 Lupo, Peat, Andersson & Kok (2023) — Error-tolerant Oblivious Transfer in the Noisy-Storage Model

**Citation**: *arXiv:2309.xxxxx*

**Core Contributions to Phase I**:

1. **Tighter Entropic Bounds**: Leveraging entropic uncertainty relations, the authors establish that for a depolarizing storage channel with parameter $r$, the protocol remains secure up to:
   $$h\left(\frac{1+r_{trusted}}{2}\right) \le \frac{1}{2} \implies r_{trusted} \gtrsim 0.78$$
   This corresponds to a **hard limit** of approximately $P_{error} \lesssim 22\%$ (Eq. 43 and Section VI).

2. **Min-Entropy Rate Analysis**: The smooth min-entropy rate $h_{min}$ is bounded by the strong-converse exponent $\gamma_r(R)$ of the depolarizing channel (Eq. 19):
   $$h_{min} = \lim_{n \to \infty} \frac{1}{n} H_{min}^{\epsilon_s}(X | \mathcal{F}(Q)\Theta) \ge \gamma_r(1/2)$$

3. **Storage Capacity Formula**: For a depolarizing channel, the strong-converse capacity is (Eq. 14):
   $$C_{\mathcal{N}} = 1 - h\left(\frac{1+r}{2}\right)$$
   where $h(x) = -x \log x - (1-x) \log(1-x)$ is the binary Shannon entropy.

**Phase I Mapping**:
- **Requirement PHI-R1**: Two-tiered abort: hard abort at $Q_{total} > 22\%$, warning at $Q_{total} > 11\%$.
- **NSM Bound Calculator**: The derivation of $h_{min}$ from storage noise $r$ must be implemented for Phase IV, but the feasibility check in Phase I must validate that positive-rate keys are theoretically achievable.

### 1.4 Erven et al. (2014) — An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model

**Citation**: *arXiv:1308.5098*

**Core Contributions to Phase I**:

1. **Experimental Parameter Characterization**: Table I provides calibrated physical parameters:
   - Mean photon pair number per pulse: $\mu = 3.145 \times 10^{-5}$
   - Total transmittance: $\eta = 0.0150$
   - Intrinsic detection error rate: $e_{det} = 0.0093$
   - Dark count probability: $P_{dark} = 1.50 \times 10^{-8}$

2. **Secure ROT String Rate Formula** (Eq. 8):
   $$\ell \le \frac{1}{2}\nu \gamma^{\mathcal{N}_r}\left(\frac{r}{\nu}\right)\frac{m}{n} - n \cdot f \cdot h(p_{err}) - \log_2\left(\frac{1}{2\varepsilon}\right)$$
   where the first term represents Bob's uncertainty, the second term is leakage from error correction, and the third is a finite-size safety margin.

3. **Loss Sensitivity Analysis**: Figure 2 demonstrates that secure rate drops drastically with decreasing transmittance. Below $\eta < 1.5\%$, the "strictly less" condition may become impossible to satisfy.

4. **Wait Time Semantics**: "Both parties now wait a time, $\Delta t$, long enough for any stored quantum information of a dishonest party to decohere."

**Phase I Mapping**:
- **Noise Adapter**: Configuration must accept NSM-compatible parameters ($\mu$, $\eta$, $e_{det}$, $P_{dark}$) and map them to NetSquid models.
- **$\Delta t$ Calibration**: The wait time must be calibrated to the assumed decoherence times (milliseconds to seconds for atomic ensemble memories).

### 1.5 Lemus et al. (2020) — Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation

**Citation**: *arXiv:1909.11701*

**Core Contributions to Phase I**:

1. **Hybrid Commitment Architecture**: The protocol uses classical cryptographic hash functions for commitments ($\pi_{COMH}$). The commitment phase ensures Bob measures/commits before learning Alice's bases.

2. **BB84-Style Encoding**: State preparation uses:
   $$|(s_i, a_i)\rangle: \quad |(0,0)\rangle = |0\rangle, \quad |(0,1)\rangle = |+\rangle, \quad |(1,0)\rangle = |1\rangle, \quad |(1,1)\rangle = |-\rangle$$

3. **Correlation Test**: A random subset is revealed to verify that Bob measured as stated, detecting coherent attacks that would fail specific correlation checks.

**Phase I Mapping**:
- **Basis Selection Logic**: The current implementation correctly uses Hadamard rotation for X-basis measurement.
- **Commitment Module**: The hash-based commitment is implemented but lacks NSM physical-wait enforcement.

---

## 2. Ontology & Terminology

This section provides a definitive mapping from abstract protocol concepts to their concrete software representations across the technology stack.

### 2.1 Core Security Model Vocabulary

| Term | Definition | Software Manifestation |
|------|------------|------------------------|
| **NSM Adversary** | Bob modeled as honest-but-curious, with advantage limited by noisy quantum storage over delay window $\Delta t$ | Protocol assumptions; not directly encoded but enforced via timing |
| **Trusted Noise** | Imperfections attributable to honest physical layer: source preparation, channel, detection | Configurable via `Link.fidelity`, `T1T2NoiseModel`; mapped to empirical QBER |
| **Untrusted Noise** | Storage decoherence in adversary's memory, parameterized by depolarizing $r$ | Assumption parameter in security analysis; not simulated for adversary |
| **Wait Time ($\Delta t$)** | NSM causal barrier between qubit reception and basis revelation | **GAP**: Must be implemented via NetSquid timeline |

### 2.2 Quantum Operations Mapping

| Protocol Concept | Mathematical Representation | SquidASM/NetQASM Realization |
|------------------|----------------------------|------------------------------|
| **Qubit Source** | EPR pair generation $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ | `EPRSocket.create_keep()` / `EPRSocket.recv_keep()` |
| **Basis Selection** | $\theta \in \{0, 1\}$ (Z or X) | `Qubit.H()` for X-basis rotation, then `Qubit.measure()` |
| **Measurement** | Projective measurement in chosen basis | `Qubit.measure(basis=QubitMeasureBasis.Z)` or post-Hadamard Z-measurement |
| **Depolarizing Channel** | $\rho \to r\rho + (1-r)\frac{I}{2}$ | `DepolarNoiseModel(depolar_rate=1-fidelity)` in NetSquid |
| **T1/T2 Decoherence** | Amplitude damping ($T_1$), dephasing ($T_2$) | `T1T2NoiseModel(T1, T2)` on qdevice memory positions |

### 2.3 Classical Post-Processing Mapping

| Protocol Concept | Legacy Implementation | Target Stack Mapping |
|------------------|----------------------|----------------------|
| **Commitment** | `SHA256Commitment.commit(data)` → `(commitment, salt)` | Preserved; add timing enforcement wrapper |
| **Basis Sifting** | `SiftingManager.identify_matching_bases()` → `(I_0, I_1)` | Preserved; pure classical logic |
| **Test Set Selection** | `SiftingManager.select_test_set()` → `(test_set, key_set)` | Preserved; deterministic seeding |
| **QBER Estimation** | `SiftingManager.estimate_qber()` | Preserved; add statistical penalty $\mu$ |

### 2.4 NetSquid/SquidASM Component Hierarchy

```
StackNetworkConfig (YAML)
    └── NetworkConfig
        ├── nodes: List[NodeConfig]
        │   ├── name: str
        │   ├── qubits: List[QubitConfig]  # T1, T2 per position
        │   └── gate_fidelity: float
        └── links: List[Link]
            ├── node_name1, node_name2
            ├── fidelity: float
            └── noise_type: NoiseType
                ├── NoNoise → PerfectStateMagicDistributor
                ├── Depolarise → LinearDepolariseMagicDistributor
                └── Bitflip → BitflipMagicDistributor
```

---

## 3. Traceability Matrix

### 3.1 Requirement Identifier Scheme

- **Spec Requirements**: `PHI-R{ordinal}` (Phase I Requirement)
- **Literature Citations**: `[Author, Year]`
- **Roadmap Tasks**: `TASK-{subsystem}-{index}`
- **Abort Conditions**: `ABORT-{phase}-{category}-{index}`

### 3.2 Phase I Requirements Traceability

| Req ID | Title | Literature Basis | Legacy Status | Target Stack Capability | Migration Task |
|--------|-------|------------------|---------------|------------------------|----------------|
| **PHI-R1** | Pre-Flight Feasibility Check | [König 2012] Corollary I.2; [Lupo 2023] Eq. 43; [Schaffner 2009] Corollary 7 | **Missing** | Parameters available but not aggregated | TASK-FEAS-001 |
| **PHI-R2** | Strict Enforcement of $\Delta t$ | [König 2012] Section I-C; [Erven 2014] Remark 2 | **Missing** | `ns.sim_time()` available; no wait primitive | TASK-TIMING-001 |
| **PHI-R3** | Modular Commitment (TLP vs Physical) | [Lemus 2020]; [Schaffner 2009] | **Partial** (hash only) | Hash-based implemented; physical wait missing | TASK-COMMIT-001 |

### 3.3 Literature Citation Index

| Citation Key | Full Reference | Phase I Relevance |
|--------------|----------------|-------------------|
| [König 2012] | König, Wehner, Wullschleger. "Unconditional Security from Noisy Quantum Storage." IEEE TIT 2012. | NSM definition, WSE, capacity bounds |
| [Schaffner 2009] | Schaffner, Terhal, Wehner. "Robust Cryptography in the Noisy-Quantum-Storage Model." QIC 2009. | 11% QBER bound, individual-storage attacks |
| [Lupo 2023] | Lupo, Peat, Andersson, Kok. "Error-tolerant OT in the Noisy-Storage Model." arXiv 2023. | 22% hard limit, entropic bounds |
| [Erven 2014] | Erven et al. "Experimental Implementation of OT in the Noisy Storage Model." arXiv 2014. | Physical parameters, rate formula |
| [Lemus 2020] | Lemus et al. "Generation and Distribution of Quantum Oblivious Keys." arXiv 2020. | Hybrid commitment, BB84 encoding |

---

## 4. Phase I Deep-Dive & Gap Analysis

### 4.1 Semantic Goal of Phase I

Phase I is not merely "generate qubits and measure." It is the phase where the **NSM security model becomes operational** by ensuring:

1. **Trusted noise** (honest channel/device imperfections) remains **strictly below** untrusted storage noise.
2. A **real waiting semantics** $\Delta t$ exists in the discrete-event timeline, creating an unforgeable causal barrier.
3. **Pre-flight validation** aborts before resource consumption if security is mathematically impossible.

### 4.2 Existing Capabilities in the SquidASM Stack

#### 4.2.1 EPR Generation and Distribution

**Status**: ✅ **NATIVELY SUPPORTED**

The SquidASM framework provides EPR pair generation through the `EPRSocket` interface, with the underlying `MagicDistributor` handling state delivery with configurable noise models.

**Source Evidence** ([network.py#L203-L230](squidasm/squidasm/sim/network/network.py#L203-L230)):

The `_create_link_distributor` method maps `Link.fidelity` to noise model parameters:

- `NoiseType.NoNoise` → `PerfectStateMagicDistributor`
- `NoiseType.Depolarise` → `LinearDepolariseMagicDistributor` with `prob_max_mixed = 1 - fidelity`
- `NoiseType.Bitflip` → `BitflipMagicDistributor` with `flip_prob = 1 - fidelity`

**Adequacy for Phase I**: The fidelity-based noise model abstracts the trusted channel noise. However, it does not directly expose NSM-specific parameters like source quality $\mu$ or detection efficiency $\eta$.

#### 4.2.2 Basis Selection and Measurement

**Status**: ✅ **NATIVELY SUPPORTED**

NetQASM provides measurement basis selection through:

- `Qubit.measure(basis=QubitMeasureBasis.Z)` for computational basis
- `Qubit.H()` followed by Z-measurement for Hadamard (X) basis
- `RandomBasis.XZ` for BB84-style random selection

**Legacy Implementation** ([measurement.py](ehok/quantum/measurement.py)): The current E-HOK code uses Hadamard rotation before computational-basis measurement, consistent with standard BB84 encoding.

#### 4.2.3 Noise Model Configuration

**Status**: ⚠️ **PARTIALLY SUPPORTED — EXTENSION REQUIRED**

**Available Parameters**:
- `Link.fidelity` → translates to depolarizing noise $1 - F$
- `T1`, `T2` times on individual qubits via `T1T2NoiseModel`
- `DepolarNoiseModel`, `DephaseNoiseModel` in NetSquid

**Missing Parameters**:
- **Source Quality ($\mu$)**: No direct parameter for mean photon pair number
- **Detection Efficiency ($\eta$)**: Not configurable at SquidASM level
- **Dark Count Rate ($P_{dark}$)**: Not exposed in EPR generation interface
- **Intrinsic Error ($e_{det}$)**: Must be inferred from aggregate fidelity

**NetSquid Resources for Extension** ([qerrormodels.py](netsquid/components/models/qerrormodels.py)):

```
QuantumErrorModel (base)
├── DepolarNoiseModel     # depolar_rate: exponential [Hz] or probability
├── DephaseNoiseModel     # dephase_rate: pure dephasing
├── T1T2NoiseModel        # T1: amplitude damping, T2: Hahn dephasing
└── FibreLossModel        # Fiber-based probabilistic loss
```

**Extension Point**: Create a `PhysicalChannelModel` adapter that accepts NSM-compatible parameters and converts them to NetSquid noise models.

#### 4.2.4 Timing Primitives

**Status**: ❌ **NOT NATIVELY SUPPORTED — EXTENSION REQUIRED**

This is the **most critical gap**. The NSM security model requires strict ordering:

1. Alice sends qubits
2. Bob acknowledges receipt (implicit measurement/storage)
3. **Timer $\Delta t$ elapses** (adversary's storage decoheres)
4. Alice reveals bases

**Available NetSquid Primitives**:
- `ns.sim_time()` returns current simulation time in nanoseconds
- Event scheduling via `pydynaa._schedule_after(delay, event)`
- `netsquid_magic.Sleeper` for yielding without blocking

**Gap Analysis**: SquidASM's `run()` method is a generator that yields control for async operations, but there is no built-in "wait for time $\Delta t$" primitive. Classical sockets have no delay enforcement.

#### 4.2.5 Pre-Flight Feasibility Validation

**Status**: ❌ **NOT NATIVELY SUPPORTED — MUST IMPLEMENT**

Phase I requires aborting if:
- $Q_{total} > 22\%$ (Hard limit — security mathematically impossible)
- $Q_{total} > 11\%$ (Conservative limit — warning)

**No native function exists to**:
1. Aggregate noise parameters into total trusted noise $Q_{total}$
2. Compare against NSM security bounds from [Lupo 2023]
3. Calculate minimum required storage noise given channel parameters

### 4.3 Observed Legacy Implementation Behavior

#### 4.3.1 Quantum Phase Runner

**Source**: [runner.py](ehok/quantum/runner.py)

The `QuantumPhaseRunner` class encapsulates NetSquid-specific batching logic:

- Iterates over batches (default 5 qubits per batch, limited by quantum memory)
- Records `ns.sim_time()` timestamps per batch
- Performs measurement immediately after EPR delivery via `EPRGenerator`

**Gap**: No explicit waiting interval exists between "receipt" and "basis reveal" beyond the ordering implied by subsequent classical messages.

#### 4.3.2 Commitment Scheme

**Source**: [commitment/sha256_commitment.py](ehok/implementations/commitment/sha256_commitment.py)

The SHA256-based commitment implements:
- `commit(data) → (commitment_hash, salt)`
- `verify(commitment, data, salt) → bool`

**Gap**: The commitment is purely cryptographic. It does not enforce or record the timing of Bob's measurement relative to the commitment, which is the NSM security requirement.

#### 4.3.3 Sifting and QBER Estimation

**Source**: [sifting.py](ehok/core/sifting.py)

The `SiftingManager` provides:
- `identify_matching_bases(bases_alice, bases_bob) → (I_0, I_1)`
- `select_test_set(I_0, fraction, seed) → (test_set, key_set)`
- `estimate_qber(outcomes_alice, outcomes_bob, test_set) → float`
- `check_qber_abort(qber, threshold)` — raises `QBERTooHighError` if violated

**Adequacy**: The 11% threshold is hardcoded in `QBER_THRESHOLD`. The implementation correctly aborts on high QBER but lacks the two-tiered warning system.

### 4.4 Requirement-by-Requirement Gap Analysis

| Req ID | Title | Current Alignment | Primary Gap | Target Stack Surfaces |
|--------|-------|-------------------|-------------|----------------------|
| **PHI-R1** | Pre-Flight Feasibility Check | **Missing** | No pre-flight calculation ties trusted noise and storage assumptions to an abort decision before qubit generation | Protocol config + analysis layer; NetSquid link/memory noise params; abort taxonomy |
| **PHI-R2** | Strict Enforcement of $\Delta t$ | **Missing** | No enforced $\Delta t$ barrier between Bob's measurement commitment and Alice's basis disclosure | NetSquid timeline via `ns.sim_time()`, `netsquid_magic.Sleeper`, ordered messaging |
| **PHI-R3** | Modular Commitment (TLP vs Physical) | **Partial** | Hash commitment exists but lacks NSM "physical wait" enforcement; no TLP mode | Commitment strategy layer + timing enforcement wrapper |

### 4.5 Architectural Responsibility Mapping

This subsection assigns responsibilities to conceptual modules. These are semantic contracts, not implementation prescriptions.

#### 4.5.1 PhysicalModelAdapter

**Responsibility**: Owns the mapping from calibration parameters ($\mu$, $\eta$, $e_{det}$, $P_{dark}$) to NetSquid channel and memory models.

**Semantic Contract**:
- **Input**: Experimental parameter bundle conforming to [Erven 2014] Table I format
- **Output**: Configured `Link` object and `MagicDistributor` with noise models
- **Derived Output**: Expected QBER $Q_{expected}$ from aggregate noise model

**Target Location**: `ehok/quantum/noise_adapter.py`

#### 4.5.2 TimingEnforcer

**Responsibility**: Owns enforcement of the NSM waiting window $\Delta t$ in the discrete-event timeline.

**Semantic Contract**:
- **Precondition**: Bob has acknowledged qubit receipt (commitment sent)
- **Operation**: Yield control until `ns.sim_time() - t_receipt >= delta_t`
- **Postcondition**: Produce transcript marker indicating causal boundary
- **Violation**: Abort with `ABORT-I-TIMING-001` if bases revealed prematurely

**Target Location**: `ehok/quantum/timing.py`

#### 4.5.3 FeasibilityAnalyzer

**Responsibility**: Owns pre-flight impossibility check (hard abort threshold and conservative warning).

**Semantic Contract**:
- **Input**: Noise model parameters, assumed adversary storage noise $r$, security parameter $\varepsilon_{sec}$
- **Output**: `(feasible: bool, message: str, proof_obligation: SecurityProofBundle)`
- **Abort Codes**:
  - `ABORT-I-FEAS-001`: $Q_{total} > 22\%$ (hard limit)
  - `WARN-I-FEAS-001`: $Q_{total} > 11\%$ (conservative limit)

**Target Location**: `ehok/core/feasibility.py`

---

## 5. Formalization of Physical and Protocol Constraints

This section formalizes the mathematical constraints governing Phase I, using LaTeX notation and citing the literature where applicable.

### 5.1 QBER Bounds (Raw Key)

#### 5.1.1 Hard Limit (Lupo et al.)

For a depolarizing storage channel with parameter $r$, security requires that the trusted noise parameter $r_{trusted}$ satisfies [Lupo 2023, Eq. 43]:

$$h\left(\frac{1 + r_{trusted}}{2}\right) \le \frac{1}{2}$$

where $h(x) = -x \log_2 x - (1-x) \log_2(1-x)$ is the binary Shannon entropy.

Solving this inequality:

$$\frac{1 + r_{trusted}}{2} \le h^{-1}(0.5) \approx 0.89$$

$$r_{trusted} \le 0.78$$

The corresponding QBER limit is:

$$P_{error} = \frac{1 - r_{trusted}}{2} \lesssim 0.22$$

**Implementation**: Abort if estimated $Q_{total} > 0.22$.

#### 5.1.2 Conservative Limit (Schaffner et al.)

For robust operation with margin against parameter estimation errors [Schaffner 2009, Corollary 7]:

$$P_{error} \lesssim 0.11$$

**Implementation**: Warn if estimated $Q_{total} > 0.11$.

### 5.2 Source Characterization

Following [Erven 2014], the photon source is characterized by:

| Parameter | Symbol | Physical Meaning |
|-----------|--------|------------------|
| Mean photon pair number | $\mu$ | Pairs per coherence time (PDC source) |
| Total transmittance | $\eta$ | Probability photon reaches detector |
| Intrinsic detection error | $e_{det}$ | Wrong-detector click probability |
| Dark count probability | $P_{dark}$ | Spurious detection per coherence time |

The expected QBER from honest devices is approximately:

$$Q_{expected} \approx e_{det} + \frac{P_{dark}}{\eta \cdot \mu}$$

where the second term represents the contribution of dark counts relative to signal.

**Implementation**: `PhysicalModelAdapter` must accept these parameters and compute $Q_{expected}$.

### 5.3 Timing Semantics

The wait time $\Delta t$ must satisfy [König 2012, Section I-D]:

$$C_{\mathcal{N}(\Delta t)} \cdot \nu < \frac{1}{2}$$

where:
- $C_{\mathcal{N}(\Delta t)}$ is the classical capacity of the adversary's storage channel after time $\Delta t$
- $\nu$ is the storage rate (fraction of qubits adversary can store)

For a depolarizing memory with rate $\gamma$ (decoherence per unit time):

$$r(\Delta t) = e^{-\gamma \Delta t}$$

$$C_{\mathcal{N}} = 1 - h\left(\frac{1 + r(\Delta t)}{2}\right)$$

**Calibration**: $\Delta t$ should be chosen such that even with $\nu = 1$ (full storage), $C_{\mathcal{N}} < 0.5$. For typical atomic ensemble memories ([Erven 2014]), $\Delta t \sim 1-10$ ms.

**Implementation**: `TimingEnforcer.delta_t` must be configurable in nanoseconds, with values typically $10^6 - 10^{10}$ ns.

### 5.4 Channel Model Mapping

The NetSquid depolarizing channel applies:

$$\rho \to (1 - p_{max\_mixed}) \rho + p_{max\_mixed} \cdot \frac{I}{2}$$

where $p_{max\_mixed} = 1 - \text{fidelity}$.

This corresponds to the depolarizing parameter:

$$r = 1 - \frac{3}{2} p_{max\_mixed} = 1 - \frac{3}{2}(1 - F) = \frac{3F - 1}{2}$$

**Mapping Table**:

| Fidelity $F$ | $p_{max\_mixed}$ | Depolarizing $r$ | Expected QBER |
|--------------|------------------|------------------|---------------|
| 1.00 | 0.00 | 1.00 | 0.00 |
| 0.99 | 0.01 | 0.985 | 0.0075 |
| 0.95 | 0.05 | 0.925 | 0.0375 |
| 0.90 | 0.10 | 0.85 | 0.075 |
| 0.85 | 0.15 | 0.775 | 0.1125 |

**Implementation**: The `PhysicalModelAdapter` must provide this bidirectional mapping.

### 5.5 Basis Selection Logic

For BB84-style protocols, the basis choice is typically uniform:

$$P(\theta = 0) = P(\theta = 1) = 0.5$$

The expected sifted fraction is:

$$P_{sift} = P(\theta_A = \theta_B) = 0.5$$

**Biased Basis Option**: Some protocols use biased basis selection for improved rate. NetSquid supports this via `probability_dist_spec` in `RandomBasis.XZ` selection.

**Implementation**: The basis selection probability should be configurable via `ProtocolConfig.quantum.basis_bias`.

---

## 6. MoSCoW-Prioritized Strategic Roadmap

### 6.1 Must Have

These items constitute the critical path to achieving baseline Phase I functionality with NSM compliance.

| Task ID | Objective | Depends On | Verification Criteria |
|---------|-----------|------------|----------------------|
| **TASK-TIMING-001** | Enforce $\Delta t$ as a hard causal barrier between Bob's receipt acknowledgment and Alice's basis reveal | PHI-R2 | Simulation trace shows minimum $\Delta t$ elapsed between commitment and basis messages |
| **TASK-FEAS-001** | Implement pre-flight feasibility analysis with hard abort at $Q > 22\%$ and warning at $Q > 11\%$ | PHI-R1 | Protocol aborts before EPR generation if channel parameters violate bounds |
| **TASK-NOISE-PARAMS-001** | Expose NSM-compatible noise parameters ($\mu$, $\eta$, $e_{det}$) in configuration | PHI-R1, TASK-FEAS-001 | Configuration schema accepts [Erven 2014] parameter format |
| **TASK-QBER-TWOITER-001** | Implement two-tiered QBER validation: hard abort at 22%, warning at 11% | PHI-R1 | Abort codes distinguish `ABORT-I-FEAS-HARD` from `WARN-I-FEAS-CONSERVATIVE` |

#### 6.1.1 TASK-TIMING-001: Wait Time Enforcement

**Semantic Objective**: Introduce an explicit simulation-time barrier that prevents Alice from revealing basis information until $\Delta t$ nanoseconds have elapsed since Bob acknowledged qubit receipt.

**Behavioral Description**:
1. After Bob sends his commitment message, the `TimingEnforcer` records `t_receipt = ns.sim_time()`.
2. When Alice prepares to send basis information, she queries `TimingEnforcer.can_reveal_bases()`.
3. If `ns.sim_time() - t_receipt < delta_t`, the enforcer yields control via a generator pattern until sufficient time has elapsed.
4. The transcript must record explicit markers: `TIMING_BARRIER_START` and `TIMING_BARRIER_END`.

**Dependencies**: Requires access to `ns.sim_time()` from within SquidASM program context.

#### 6.1.2 TASK-FEAS-001: Pre-Flight Feasibility Check

**Semantic Objective**: Before any quantum resources are consumed, validate that the configured channel parameters permit a positive-rate secure key under NSM assumptions.

**Behavioral Description**:
1. Accept inputs: source quality, detection efficiency, intrinsic error, dark count rate, assumed storage noise $r_{storage}$, security parameter $\varepsilon_{sec}$.
2. Compute expected total trusted noise $Q_{total}$ using the formulas in Section 5.2.
3. Compute the minimum achievable key rate using [Lupo 2023] min-entropy bounds.
4. **Abort** with `ABORT-I-FEAS-001` if $Q_{total} > 0.22$.
5. **Warn** with `WARN-I-FEAS-001` if $Q_{total} > 0.11$.
6. Return a `FeasibilityResult` dataclass containing proof obligations for downstream phases.

### 6.2 Should Have

These items enhance simulation accuracy and align with literature-specific noise models.

| Task ID | Objective | Depends On | Verification Criteria |
|---------|-----------|------------|----------------------|
| **TASK-NOISE-ADAPTER-001** | Create `PhysicalModelAdapter` mapping NSM params to NetSquid models | TASK-NOISE-PARAMS-001 | Unit tests verify parameter conversion matches [Erven 2014] |
| **TASK-COMMIT-TIMING-001** | Wrap commitment module with timing enforcement | TASK-TIMING-001 | Commitment object carries timestamp metadata |
| **TASK-TRANSCRIPT-001** | Introduce protocol transcript object recording message types and timing | TASK-TIMING-001 | Transcript can be audited for causal ordering |

#### 6.2.1 TASK-NOISE-ADAPTER-001: Physical Model Adapter

**Semantic Objective**: Provide a translation layer between experimentally-calibrated physical parameters and the abstract noise models in NetSquid/SquidASM.

**Behavioral Description**:
1. Accept an `ExperimentalParams` dataclass with fields: `mu`, `eta`, `e_det`, `p_dark`.
2. Compute expected QBER: $Q_{expected} = e_{det} + P_{dark} / (\eta \cdot \mu)$.
3. Compute equivalent fidelity: $F = 1 - 2 \cdot Q_{expected}$ (for depolarizing model).
4. Generate a `LinkConfig` with `fidelity` and `noise_type` set appropriately.
5. Optionally compute memory decoherence parameters for T1/T2 models.

### 6.3 Could Have

Extended telemetry, debugging capabilities, and advanced commitment modes.

| Task ID | Objective | Depends On | Verification Criteria |
|---------|-----------|------------|----------------------|
| **TASK-TLP-COMMIT-001** | Implement Time-Lock Puzzle (TLP) commitment mode per [Faleiro] | TASK-COMMIT-TIMING-001 | TLP mode selectable; puzzle difficulty calibrated to $\Delta t$ |
| **TASK-CHANNEL-TELEMETRY-001** | Expose per-batch QBER telemetry during quantum generation | — | Telemetry data available for real-time monitoring |
| **TASK-DECOY-PREP-001** | Prepare infrastructure for decoy-state intensity tagging | — | Metadata channel supports per-round intensity labels |

---

## 7. Formal System Invariants and Contracts

This section formalizes behavioral contracts as runtime-enforceable invariants.

### 7.1 Global Invariants

#### INV-PHI-001: Causality Barrier

> No message containing Alice's basis information may appear in the protocol transcript prior to a timing marker indicating that $\Delta t$ has elapsed since Bob's commitment message.

**Formal Statement**:
$$\forall m \in T: \text{type}(m) = \text{BASIS\_REVEAL} \implies \exists m' \in T: m' < m \land \text{type}(m') = \text{TIMING\_BARRIER\_END}$$

#### INV-PHI-002: Abort Safety

> If any hard abort condition holds, the protocol must terminate without producing non-trivial output.

**Formal Statement**:
$$Q_{total} > 0.22 \implies \text{output} = \text{ABORT}$$

#### INV-PHI-003: Noise Monotonicity

> The trusted noise model must be Markovian: additional delay cannot decrease noise.

**Formal Statement**:
$$F_{t_1 + t_2} = F_{t_1} \circ F_{t_2}$$

This is guaranteed by NetSquid's `T1T2NoiseModel` and `DepolarNoiseModel` implementations.

### 7.2 Phase-Specific Contracts

#### PRE-PHI-001: Feasibility Precondition

> Before EPR generation begins, the feasibility predicate must hold.

**Formal Statement**:
```
PRE: FeasibilityAnalyzer.check(params) returns (True, _)
```

If violated, abort with `ABORT-I-FEAS-001` before any quantum resources are consumed.

#### POST-PHI-001: Raw Material Generation

> After Phase I completes, Alice and Bob each hold arrays of measurement outcomes and basis choices of equal length.

**Formal Statement**:
```
POST: len(outcomes_alice) == len(outcomes_bob) == len(bases_alice) == len(bases_bob) == n_pairs
```

#### POST-PHI-002: Correlation Guarantee

> For indices where bases match, outcomes should be equal (modulo channel noise).

**Formal Statement**:
$$\forall i \in I_0: P(s_i = \bar{s}_i) = 1 - Q_{channel}$$

This is verified statistically during Phase II sifting.

### 7.3 Abort Taxonomy for Phase I

| Abort Code | Condition | Response |
|------------|-----------|----------|
| `ABORT-I-FEAS-001` | $Q_{total} > 0.22$ | Hard abort before EPR generation |
| `ABORT-I-TIMING-001` | Basis revealed before $\Delta t$ | Hard abort, protocol violation |
| `WARN-I-FEAS-001` | $0.11 < Q_{total} \le 0.22$ | Warning, continue with caution |
| `INFO-I-QBER-001` | Observed QBER > expected | Informational, flag for investigation |

---

## 8. Aggressive Legacy Removal Strategy

### 8.1 Migration Approach: No Rollback

The migration must enforce NSM semantics **without maintaining backward compatibility** with legacy non-NSM code.

#### 8.1.1 Phased Removal (No Rollback)

1. **Phase 1a**: Introduce `TimingEnforcer` and `FeasibilityAnalyzer` as new modules alongside legacy code.
2. **Phase 1b**: Write comprehensive parity tests comparing legacy vs. new implementations.
3. **Phase 1c**: Upon validation, **PERMANENTLY DELETE** legacy non-enforcing components.
4. **Phase 1d**: All downstream code is rewritten to use NSM-compliant modules.

#### 8.1.2 Configuration Transition (Breaking Change)

Old configurations are no longer supported. Migration requires explicit conversion:

```yaml
# DELETED: Legacy format is no longer accepted
# links:
#   - stack1: alice
#     stack2: bob
#     typ: depolarise
#     cfg:
#       fidelity: 0.95

# REQUIRED: New NSM-aware format (mandatory)
links:
  - stack1: alice
    stack2: bob
    typ: depolarise_nsm  # Only valid type after migration
    cfg:
      fidelity: 0.95
      nsm_params:
        mu: 3.145e-5
        eta: 0.015
        e_det: 0.0093
        p_dark: 1.5e-8
        delta_t_ns: 1e7  # 10 ms
        assumed_r_storage: 0.75  # Must be explicitly set
```

**Breaking Change Warning**: Users upgrading to post-migration versions must explicitly update configurations. No automatic fallback or compatibility layer.

### 8.2 Validation Checkpoints (Gates to Deletion)

| Checkpoint | Criteria | Test Method | Status for Deletion |
|------------|----------|-------------|---------------------|
| **CP-PHI-001** | Timing enforcement exists and is testable | Parity test: legacy vs. enforcer | ✓ Both implementations agree |
| **CP-PHI-002** | Pre-flight check behavior matches spec | Parity test: abort/warn boundaries | ✓ Both implementations abort at 22%, warn at 11% |
| **CP-PHI-003** | NSM noise parameters configurable | Parity test: [Erven 2014] params | ✓ Both yield identical results |
| **CP-PHI-004** | Transcript records causal ordering | Parity test: timing barrier markers | ✓ Both record identical markers |
| **CP-PHI-005** | All references to legacy code removed | Code review + grep | ✓ Zero legacy references in codebase |

Once **ALL** validation checkpoints pass, legacy code is deleted immediately. No "deprecation period."

### 8.3 Deletion Timeline

| Version | Action | Impact |
|---------|--------|--------|
| v0.2.0 | Introduce `TimingEnforcer`, `FeasibilityAnalyzer` (alongside legacy) | Non-breaking; legacy still works |
| v0.3.0 | Validation complete; delete legacy code | **BREAKING** — configurations must be updated |
| v0.4.0+ | NSM semantics are mandatory; no legacy modes exist | All code assumes NSM correctness |

---

## Appendix A: Dependency Graph for Phase I Tasks

```
TASK-NOISE-PARAMS-001
        │
        ▼
TASK-NOISE-ADAPTER-001
        │
        ├───────────────────┐
        ▼                   ▼
TASK-FEAS-001         TASK-TIMING-001
        │                   │
        │                   ▼
        │           TASK-COMMIT-TIMING-001
        │                   │
        └───────────────────┴───────────────┐
                                            ▼
                                    TASK-TRANSCRIPT-001
                                            │
                                            ▼
                                    TASK-QBER-TWOITER-001
```

---

## Appendix B: Open Specification Decisions

The following decisions must be fixed to claim NSM-correctness:

| Decision ID | Question | Options | Recommendation |
|-------------|----------|---------|----------------|
| **A-DEC-PHI-001** | Storage model parameterization | Per-qubit probability vs per-unit-time rate | Use effective parameter after $\Delta t$ as per [König 2012] |
| **A-DEC-PHI-002** | $\Delta t$ value for simulation | Fixed vs calibrated to storage assumptions | Configurable, default $10^7$ ns (10 ms) |
| **A-DEC-PHI-003** | Basis selection bias | Uniform (0.5/0.5) vs biased | Uniform for baseline, configurable for extensions |
| **A-DEC-PHI-004** | Commitment mode | Hash-only vs TLP-capable | Hash-only for baseline; TLP as Could-Have extension |

---

## References

1. König, R., Wehner, S., & Wullschleger, J. (2012). Unconditional Security from Noisy Quantum Storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.

2. Schaffner, C., Terhal, B., & Wehner, S. (2009). Robust Cryptography in the Noisy-Quantum-Storage Model. *Quantum Information and Computation*, 9(11&12), 963-996.

3. Lupo, C., Peat, J. T., Andersson, E., & Kok, P. (2023). Error-tolerant oblivious transfer in the noisy-storage model. *arXiv:2309.xxxxx*.

4. Erven, C., Ng, N. H. Y., Gigov, N., Laflamme, R., Wehner, S., & Weihs, G. (2014). An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model. *arXiv:1308.5098*.

5. Lemus, M., et al. (2020). Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation. *arXiv:1909.11701*.

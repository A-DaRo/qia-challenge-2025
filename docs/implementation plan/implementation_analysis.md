# E-HOK Implementation Analysis & Migration Guide: NSM Compliance and SquidASM/NetQASM/NetSquid Integration

## Abstract

This document is the authoritative migration guide for refactoring the legacy E-HOK (Ephemeral Honest-but-curious Oblivious Key) prototype in `qia-challenge-2025/ehok/` onto the SquidASM/NetQASM/NetSquid simulation stack while restoring full compliance with the Noisy Storage Model (NSM) security semantics described in the phase specifications.

The analysis is grounded in source code as the primary truth: (i) the current E-HOK role programs, classical post-processing, and data structures in the repository; (ii) the application-facing primitives in SquidASM; and (iii) the lower-layer services and timing/noise semantics exposed through NetQASM, NetSquid, and netsquid_magic.

Central conclusions derived from the codebase inspection:
- The repository already contains a coherent and test-backed classical pipeline (sampling, LDPC reconciliation machinery, Toeplitz hashing, finite-key length computation).
- The implementation is not yet NSM-compliant because several security-critical semantic constraints are not enforced at runtime: $\Delta t$ ordering, missing-rounds / loss validation, pre-flight feasibility checks, and OT-aligned output semantics.
- The current reconciliation direction is Bob $	o$ Alice (Bob transmits syndrome), whereas the Phase III requirement set is written for Alice $	o$ Bob one-way FEC to avoid leaking Bob’s choice semantics.
- The produced `ObliviousKey` object is a QKD-style “key + approximate knowledge mask” artifact, not the OT-style output contract $(S_0,S_1)$ for Alice and $(S_C,C)$ for Bob.

The remainder of this document provides a phase-structured gap analysis and a MoSCoW-prioritized development roadmap with formal invariants, pre-/post-conditions, and a deprecation strategy that preserves backward compatibility while converging to NSM-correct behavior.

## Table of Contents

- 1. Terminology, Ontology, and Software Mapping
- 2. Traceability Matrix Schema
- 3. Phase I Deep-Dive: Quantum Generation & Physical Setup
- 4. Phase II Deep-Dive: Sifting & Estimation
- 5. Phase III Deep-Dive: Information Reconciliation
- 6. Phase IV Deep-Dive: Privacy Amplification and OT Output Semantics
- 7. MoSCoW-Prioritized Migration Roadmap
- 8. Formal System Invariants and Contracts
- 9. Deprecation Strategy and Migration Checkpoints
- Appendix A. Open Specification Decisions
- Appendix B. Mathematical Formalizations (NSM bounds, finite-key, leakage)
- Appendix C. Source Inventory (ehok, squidasm, netqasm, netsquid_magic)
- Appendix D. Symbol Index (by module)

## 1. Terminology, Ontology, and Software Mapping

### 1.1 Security Model Vocabulary

- NSM adversary: Bob is modeled as honest-but-curious at the protocol interface but may deviate strategically; his advantage is limited by a noisy quantum storage channel that acts over a delay window $\Delta t$.
- Trusted noise: all imperfections attributable to the honest physical layer and measurement (source preparation, channel, detection) that appear as an empirical QBER and loss process.
- Untrusted noise: the storage decoherence experienced by an adversary attempting to retain quantum information beyond $\Delta t$, typically parameterized by a depolarizing parameter $r$ in the phase specs.

### 1.2 Core Symbols and Their Concrete Counterparts

This table maps protocol-level symbols to their concrete software representation in the current codebase and in the target stack. The mappings below are descriptive contracts, not implementation prescriptions.

| Symbol | Meaning | Concrete representation (current E-HOK) | Concrete representation (Squid stack) |
|---|---|---|---|
| $\Delta t$ | NSM waiting window between reception and basis reveal | Not enforced; implicit in message order only | NetSquid simulation time and event scheduling (e.g., `netsquid.sim_time()` and netsquid_magic `Sleeper`) |
| $Q_{\mathrm{QBER}}$ | Observed bit error rate on test subset | `ehok.core.sifting.SiftingManager.estimate_qber` and abort via `check_qber_abort` | Measurements produced by NetQASM qubits; error-rate estimation can reuse SquidASM `util.qkd_routine` pattern |
| $\mu$ | Finite-size fluctuation penalty | `ehok.implementations.privacy_amplification.finite_key.compute_statistical_fluctuation` | Same; fed by observed counts and $arepsilon_{sec}$ |
| $I_0, I_1$ | Match/mismatch basis index sets | `SiftingManager.identify_matching_bases` | Same; bases produced by measurements; indices are classical |
| $L_{\mathrm{EC}}$ | Leakage due to reconciliation messaging | Partially tracked as syndrome length + hash bits | Classical channel capacity is explicit in SquidASM sockets; leakage accounting is a protocol-level invariant |
| $(S_0,S_1)$, $(S_C,C)$ | OT-style outputs | Not implemented; `ObliviousKey` is QKD-like | Must be explicit output dataclasses and application-facing contract |

## 2. Traceability Matrix Schema

This document uses a requirement identifier scheme to link specification requirements to (a) observed source-code behavior, (b) target-stack capabilities, and (c) roadmap items.

### 2.1 Requirement Identifier Scheme

- Spec requirement IDs are normalized as: `PH{phase}-R{ordinal}` (e.g., `PHI-R2`, `PHIII-R1`).
- Each roadmap item is normalized as: `TASK-{subsystem}-{index}` (e.g., `TASK-TIMING-001`).
- Each abort condition is normalized as: `ABORT-{phase}-{category}-{index}` (e.g., `ABORT-II-LOSS-001`).

### 2.2 Cross-Reference Columns

Every requirement row in matrices includes:
- Spec statement anchor (source markdown section)
- Observed implementation location(s) in `ehok/` (files and semantic behavior)
- Candidate target-stack primitive(s) (SquidASM / NetQASM / NetSquid)
- Compliance status: Implemented / Partial / Missing / Implemented-but-wrong-direction
- Migration delta: what must change semantically
- Roadmap dependency references

### 2.3 Aggregated Requirement List (from phase specifications)

| Phase | Spec ordinal | Normalized ID | Title |
|---|---:|---|---|
| I | 1 | PHI-R1 | The "Pre-Flight" Feasibility Check |
| I | 2 | PHI-R2 | Strict Enforcement of Wait Time ($\Delta t$) |
| I | 3 | PHI-R3 | Modular Commitment Module (TLP vs. Physical) |
| II | 1 | PHII-R1 | The "Sandwich" Protocol Flow (Strict Ordering) |
| II | 2 | PHII-R2 | Dynamic "Pre-Processing" Calculation |
| II | 3 | PHII-R3 | Decoy State Validator (Optional but Recommended) |
| III | 1 | PHIII-R1 | One-Way LDPC Implementation |
| III | 2 | PHIII-R2 | The "Safety Cap" (Leakage Accumulator) |
| III | 3 | PHIII-R3 | "Send-All" vs. "Interactive Hashing" Selector |
| IV | 1 | PHIV-R1 | The "Max Bound" Calculator |
| IV | 2 | PHIV-R2 | Feasibility & Batch Sizing |
| IV | 3 | PHIV-R3 | Oblivious Output Formatting |

## 3. Phase I: Quantum Generation & Physical Setup

###  Overview: semantic goal of this phase

Phase I establishes the quantum raw material and the physical root-of-trust. In NSM, Phase I is not just “generate qubits”; it is the step where the simulation must make the security model operational by ensuring that trusted noise remains strictly below the untrusted storage noise and that a real waiting semantics $\Delta t$ exists in the discrete-event timeline.

###  Existing capabilities in the Squid stack

- Entanglement distribution is exposed to applications via `ProgramContext.epr_sockets` and NetQASM `EPRSocket` operations, while the network-level semantics are realized by NetSquid network objects and netsquid_magic distributors (configured through NetQASM `NetworkConfig` and SquidASM network construction).
- Basis selection and measurement semantics are available through NetQASM `Qubit` operations and application-side rotations (the current E-HOK code uses a Hadamard rotation to implement $X$ basis before a computational-basis measurement).
- Classical communication is available via SquidASM `ClassicalSocket` objects and generator-based receive operations that yield `EventExpression`s.
- Timing primitives can be implemented at the NetSquid layer using the simulator timeline and event scheduling; netsquid_magic provides `Sleeper` which yields events without blocking the simulation.

###  Observed baseline implementation behavior (source-grounded)

- The current quantum phase is implemented as a role-independent runner (`ehok.quantum.runner.QuantumPhaseRunner`) that iterates over batches; it records `netsquid.sim_time()` timestamps per batch and performs measurement immediately after EPR delivery.
- EPR acquisition uses `EPRSocket.create_keep` on Alice and `EPRSocket.recv_keep` on Bob, consistent with the SquidASM application model. No explicit waiting interval exists between “receipt” and “basis reveal” beyond the ordering implied by subsequent classical messages.

###  Requirement-by-requirement gap analysis

| Requirement ID | Title | Current alignment | Primary gap | Primary target surfaces |
|---|---|---|---|---|
| PHI-R1 | The "Pre-Flight" Feasibility Check | Missing | No pre-flight feasibility calculation ties trusted noise and storage assumptions to an abort decision. | Protocol config + analysis layer; NetSquid link/memory noise parameters; run-time abort taxonomy |
| PHI-R2 | Strict Enforcement of Wait Time ($\Delta t$) | Missing | No enforced $\Delta t$ barrier between Bob’s measurement/storage commitment and Alice’s basis disclosure. | NetSquid timeline, netsquid_magic Sleeper, protocol-level ordered messaging |
| PHI-R3 | Modular Commitment Module (TLP vs. Physical) | Partial | Commitment exists (hash-based), but does not provide NSM “physical wait” enforcement nor a TLP mode. | Commitment strategy layer + timing enforcement layer |

###  Architectural responsibility mapping (semantic, not code)

This subsection assigns responsibilities to conceptual modules. Names are descriptive and may be realized as new modules or as refactors of existing ones.

- PhysicalModelAdapter: owns the mapping from calibration parameters (e.g., $\mu$, $\eta$, $e_{det}$) to NetSquid channel and memory models, including the mapping of “trusted noise” into QBER expectations.
- TimingEnforcer: owns the enforcement of the NSM waiting window $\Delta t$ in the discrete-event timeline, and produces explicit transcript markers indicating the causal boundary between reception and basis revelation.
- FeasibilityAnalyzer: owns the pre-flight impossibility check (hard abort threshold and conservative warning), and returns a proof obligation summary object consumed by the orchestrator.

## 4. Phase II: Sifting & Estimation

###  Overview: semantic goal of this phase

Phase II converts raw measurement streams into (i) a sifted set and (ii) statistically justified estimates for the error and loss processes. Its defining property is order: Bob must commit to detection/measurement evidence before learning bases, and Alice must verify that missing rounds are consistent with the declared physical channel rather than adversarial filtering.

###  Existing capabilities in the Squid stack

- Entanglement distribution is exposed to applications via `ProgramContext.epr_sockets` and NetQASM `EPRSocket` operations, while the network-level semantics are realized by NetSquid network objects and netsquid_magic distributors (configured through NetQASM `NetworkConfig` and SquidASM network construction).
- Basis selection and measurement semantics are available through NetQASM `Qubit` operations and application-side rotations (the current E-HOK code uses a Hadamard rotation to implement $X$ basis before a computational-basis measurement).
- Classical communication is available via SquidASM `ClassicalSocket` objects and generator-based receive operations that yield `EventExpression`s.

###  Observed baseline implementation behavior (source-grounded)

- The commitment transcript is implemented as a SHA-256 commitment over Bob’s concatenated outcomes and bases plus salt, sent to Alice before Alice reveals her bases.
- Sifting is implemented by direct comparison of two basis arrays and returns index sets $I_0$ and $I_1$. Test-set selection is deterministic when a seed is not explicitly set (seed derived from the indices), ensuring both roles select the same test subset without interactive negotiation.
- There is no explicit modeling or validation of missing rounds / losses. The baseline assumes that every requested EPR pair produces a measurement outcome.

###  Requirement-by-requirement gap analysis

| Requirement ID | Title | Current alignment | Primary gap | Primary target surfaces |
|---|---|---|---|---|
| PHII-R1 | The "Sandwich" Protocol Flow (Strict Ordering) | Partial | Commit-then-basis order exists, but lacks explicit receipt acknowledgment, $\Delta t$ gate, and loss-report ordering. | Classical sockets + ordered protocol transcript management |
| PHII-R2 | Dynamic "Pre-Processing" Calculation | Partial | Finite-size penalties exist in PA layer, but Phase II does not feed a unified parameter-estimation object into later phases. | Security analysis module + protocol state machine |
| PHII-R3 | Decoy State Validator (Optional but Recommended) | Missing | No decoy-state machinery; EPR generation is uniform and untagged. | Network/distributor configuration + per-round metadata channel |

###  Architectural responsibility mapping (semantic, not code)

This subsection assigns responsibilities to conceptual modules. Names are descriptive and may be realized as new modules or as refactors of existing ones.

- TranscriptController: owns the strict ordering of classical messages and acknowledgments, ensuring that bases cannot be revealed until commitment and (if modeled) loss reports are locked in.
- LossAndDetectionValidator: owns Chernoff-style acceptance checks for reported missing rounds based on declared $P_{trans}$ (or more general loss models) and produces abort codes when violated.
- ParameterEstimationEngine: owns finite-size penalties and produces a parameter bundle used by Phase III and IV (rather than scattering $\mu$ inside PA alone).

## 5. Phase III: Information Reconciliation

###  Overview: semantic goal of this phase

Phase III performs information reconciliation subject to an OT-specific non-leakage constraint. The syndrome transcript is treated as adversary-accessible leakage and must be capped. Directionality (who sends what) is a semantic security choice: the Phase III spec text is aligned with one-way reconciliation from Alice to Bob.

###  Existing capabilities in the Squid stack

- Classical transcripts are naturally observable to the adversary model unless explicitly assumed authenticated/private; therefore, the stack is compatible with “leakage as transcript length” accounting.

###  Observed baseline implementation behavior (source-grounded)

- Reconciliation is currently orchestrated by Bob selecting an LDPC rate and sending a syndrome to Alice, along with a verification hash. Alice attempts to decode and replies with a verification acknowledgement.
- Leakage accounting exists as a function of syndrome length and verification hash bits in the reconciliator; however, the protocol does not enforce a system-wide leakage cap $L_{max}$ as an abort condition.

###  Requirement-by-requirement gap analysis

| Requirement ID | Title | Current alignment | Primary gap | Primary target surfaces |
|---|---|---|---|---|
| PHIII-R1 | One-Way LDPC Implementation | Implemented-but-wrong-direction | Syndrome currently flows Bob $	o$ Alice, contradicting the spec’s one-way Alice $	o$ Bob assumption for obliviousness. | Protocol role orchestration + reconciliator API directionality |
| PHIII-R2 | The "Safety Cap" (Leakage Accumulator) | Partial | Leakage can be estimated but is not enforced as a hard cap with abort semantics across retries/blocks. | Global leakage accumulator + abort taxonomy |
| PHIII-R3 | "Send-All" vs. "Interactive Hashing" Selector | Not assessed | Advanced interactive hashing mode not present in baseline; requires explicit transcript design. | New protocol subroutine design |

###  Architectural responsibility mapping (semantic, not code)

This subsection assigns responsibilities to conceptual modules. Names are descriptive and may be realized as new modules or as refactors of existing ones.

- ReconciliationOrchestrator: owns which party sends which artifacts (syndrome, rate, shortening, verification), and enforces one-way leakage constraints at the transcript level.
- LeakageLedger: owns global accounting of all transcript bits that are security-relevant leakage, including verification hashes and any blind-reconciliation increments.
- SafetyCapPolicy: owns the derivation of $L_{max}$ and enforces abort when exceeded (block-level or run-level).

## 6. Phase IV: Privacy Amplification

###  Overview: semantic goal of this phase

Phase IV distills the final OT keys using privacy amplification and explicitly binds output semantics to 1-out-of-2 OT: Alice obtains two keys while Bob obtains exactly one key and a choice bit. The min-entropy bound must be NSM-appropriate (e.g., the Lupo et al. “max bound”) rather than a pure-QKD bound.

###  Existing capabilities in the Squid stack

- Classical transcripts are naturally observable to the adversary model unless explicitly assumed authenticated/private; therefore, the stack is compatible with “leakage as transcript length” accounting.

###  Observed baseline implementation behavior (source-grounded)

- Privacy amplification is implemented via Toeplitz hashing. Final length is computed using a finite-key QKD-style bound based on $1 - h(Q + \mu)$ and a leftover-hash security cost.
- Output is a single `ObliviousKey` with a “knowledge mask” (Alice: all known; Bob: approximate unknown fraction). This is not equivalent to OT output semantics and does not preserve a provable mapping from $I_1$ to post-hash ignorance.

###  Requirement-by-requirement gap analysis

| Requirement ID | Title | Current alignment | Primary gap | Primary target surfaces |
|---|---|---|---|---|
| PHIV-R1 | The "Max Bound" Calculator | Partial | Finite-key length computation exists but uses a QKD-style min-entropy model rather than NSM max bound dependent on storage noise $r$. | NSM bound calculator + parameter object that includes $r$ and $
u$ assumptions |
| PHIV-R2 | Feasibility & Batch Sizing | Partial | Death-valley detection exists late (final_length==0), but feasibility is not validated before consuming resources. | Pre-flight and mid-flight feasibility checks |
| PHIV-R3 | Oblivious Output Formatting | Missing | OT output formatting $(S_0,S_1)$ and $(S_C,C)$ is not implemented; current output is a single key with an approximate mask. | New output dataclasses and post-processing flow |

###  Architectural responsibility mapping (semantic, not code)

This subsection assigns responsibilities to conceptual modules. Names are descriptive and may be realized as new modules or as refactors of existing ones.

- NSMEntropyBounder: owns the mapping from storage-noise assumptions to a smooth min-entropy lower bound per Lupo et al. style “max bound”, producing a rate $h_{min}$ used in key-length formulas.
- PrivacyAmplificationEngine: owns universal hashing instantiation, seed generation policies, and leftover-hash security composition accounting.
- OTKeyFormatter: owns the final semantic mapping from reconciled raw material and basis partitions into $(S_0,S_1)$ for Alice and $(S_C,C)$ for Bob; it forbids the “knowledge mask” approximation as a primary interface.

## 7. MoSCoW-Prioritized Migration Roadmap

Roadmap items are grouped by subsystem; each item states a semantic objective, dependencies, and verification criteria. The list is intentionally explicit to support incremental migration without regressions.

### 7.1 Must Have

| ID | Objective | Depends on |
|---|---|---|
| TASK-TIMING-001 | Enforce $\Delta t$ as a hard causal barrier between Bob’s receipt/measurement commitment and Alice’s basis reveal; violations abort. | PHI-R2, PHII-R1 |
| TASK-FEAS-001 | Implement pre-flight feasibility analysis for trusted-noise vs storage-noise assumptions with hard abort and conservative warnings. | PHI-R1, PHIV-R2 |
| TASK-LOSS-001 | Introduce explicit missing-rounds reporting and Chernoff-style validation to prevent adversarial filtering via selective loss claims. | PHII-R1 |
| TASK-REC-DIR-001 | Refactor reconciliation transcript so the one-way FEC direction matches the Phase III OT security semantics (Alice sends syndrome). | PHIII-R1 |
| TASK-LEAK-001 | Add a global leakage ledger and enforce $L_{max}$ as an abort condition across all reconciliation attempts. | PHIII-R2 |
| TASK-OT-OUT-001 | Replace QKD-style `ObliviousKey` output with explicit OT outputs $(S_0,S_1)$ and $(S_C,C)$ and document how they bind to protocol state. | PHIV-R3 |
| TASK-NSM-ENT-001 | Implement NSM min-entropy bound computation that depends on adversarial storage noise $r$ and produces $h_{min}$ per Lupo-style max bound. | PHIV-R1 |

### 7.2 Should Have

| ID | Objective | Depends on |
|---|---|---|
| TASK-TRANS-001 | Introduce a protocol transcript object that records message lengths/types and binds leakage accounting to the actual wire representation. | TASK-LEAK-001 |
| TASK-PARAM-001 | Unify Phase II statistical penalties and Phase IV finite-key costs into a single parameter-estimation bundle with explicit $\varepsilon_{sec}$ splitting. | PHII-R2, PHIV-R2 |
| TASK-NOISE-001 | Expose calibrated noise parameters ($\mu,\eta,e_{det}$) through configuration and map them into NetSquid link/memory models. | PHI-R1 |
| TASK-ACK-001 | Implement explicit ACK handshakes in classical messaging to eliminate race-condition interpretations of “send order”. | PHII-R1 |

### 7.3 Could Have

| ID | Objective | Depends on |
|---|---|---|
| TASK-DECOY-001 | Add decoy-state support with per-round intensity tagging and yield statistics checks. | PHII-R3 |
| TASK-IHASH-001 | Implement interactive hashing mode for improved reconciliation efficiency while preserving obliviousness. | PHIII-R3 |
| TASK-MULTI-001 | Generalize beyond depolarizing storage models to alternative memory channels by abstracting $\Gamma[\cdot]$. | PHIV-R1 |

## 8. Formal System Invariants and Contracts

This section formalizes behavioral contracts in a software-facing way. Each item is expressed as an invariant or a pre-/post-condition, written in mathematical language but intended to be enforceable as runtime checks and test assertions.

### 8.1 Global invariants

- INV-GLOBAL-001 (Transcript monotonicity): Let $T$ be the ordered transcript of all classical messages. The leakage ledger $I_{leak}(T)$ is monotonically non-decreasing with transcript extension: if $T$ is a prefix of $T’$ then $I_{leak}(T) \le I_{leak}(T’)$.
- INV-GLOBAL-002 (Causality barrier): No message containing Alice’s basis information may occur in $T$ prior to a transcript marker indicating that $\Delta t$ has elapsed since Bob’s receipt acknowledgement.
- INV-GLOBAL-003 (Abort safety): If any hard abort condition holds, the protocol must terminate without outputting non-empty OT keys.

### 8.2 Phase-specific contracts (selected critical ones)

- PRE-PHI-001 (Feasibility): Inputs must satisfy the feasibility predicate derived from Phase I bounds; if violated, abort with `ABORT-I-FEAS-001` before generating qubits.
- POST-PHII-001 (Loss validation): If missing-round validation passes, the accepted detection set must be statistically consistent with declared $P_{trans}$ at level $arepsilon$; otherwise abort with `ABORT-II-LOSS-001`.
- PRE-PHIII-001 (Leakage budget): A leakage budget $L_{max}$ must be computed before reconciliation and must upper-bound the total reconciliation transcript length to preserve min-entropy.
- POST-PHIV-001 (OT output): Alice outputs $(S_0,S_1)$ and Bob outputs $(S_C,C)$ such that $S_C$ equals one of Alice’s keys and Bob’s view has negligible information about $S_{1-C}$ under the NSM assumptions.

## 9. Deprecation Strategy and Migration Checkpoints

The existing `ehok/` codebase is a working simulation pipeline. Migration must preserve runnability while incrementally tightening semantics. Deprecation is therefore staged with explicit checkpoints.

### 9.1 Compatibility shims

- Keep the existing role programs as wrappers, but introduce a new OT-native output type and provide a temporary adapter that can still produce the legacy `ObliviousKey` view for downstream experiments.
- Keep the existing Toeplitz hashing and finite-key infrastructure, but re-root the entropy source to an NSM bounder; the old QKD-style entropy model becomes a deprecated strategy.

### 9.2 Checkpoints

- CP-001: Timing enforcement $\Delta t$ exists and is testable in simulation traces.
- CP-002: Missing-round reporting and validation exist; adversarial filtering is detectable as an abort.
- CP-003: Reconciliation direction matches Phase III semantics; leakage ledger enforces $L_{max}$.
- CP-004: OT output contract exists and is the primary public API; legacy outputs are adapters.
- CP-005: NSM entropy bounder is integrated; key lengths are derived from NSM assumptions rather than QKD asymptotics.

## Appendix A. Open Specification Decisions

The phase specifications leave several decisions implicit or parameter-dependent. These must be fixed to claim NSM-correctness, because different choices affect both the transcript and the min-entropy budget.

- A-DEC-001: Storage model parameterization. Is the adversary storage channel parameter $r$ intended as a per-qubit depolarizing probability, a per-unit-time rate, or a coarse effective parameter after $\Delta t$? The simulation must choose a consistent operational meaning.
- A-DEC-002: Authenticated classical channel model. Does the protocol assume an authenticated classical channel (typical in QKD/OT proofs) and if so, how is this represented in simulation?
- A-DEC-003: Loss model binding. Does $P_{trans}$ include detector effects, source emission probability, and link loss, or only link loss? This affects the missing-round validator acceptance region.
- A-DEC-004: Choice-bit semantics. Is Bob’s choice bit $C$ derived from basis partition, from index set selection, or from an explicit choice selection step after sifting? The output formatter must align with the proof model.

## Appendix B. Mathematical Formalizations (NSM bounds, finite-key, leakage)

This appendix collects the principal equations used in the migration. The objective is to make every security-critical subtraction term explicit and to specify which quantities must be computed from observed data versus which are configured assumptions.

### B.1 Finite-size fluctuation term

The codebase already implements a Tomamichel-style statistical fluctuation term $\mu$ that corrects a measured QBER into a conservative effective QBER.

$$\mu = \sqrt{rac{n+k}{nk}\,rac{k+1}{k}}\,\sqrt{\lnrac{4}{arepsilon}}$$

### B.2 Leftover hash lemma security cost

For a target security parameter pair $(arepsilon_{sec},arepsilon_{cor})$, privacy amplification incurs a security cost term of the form:

$$\mathrm{cost}_{sec} = \log_2rac{2}{arepsilon_{sec}\,arepsilon_{cor}}$$

### B.3 Leakage accounting

Let the reconciliation transcript expose $\Sigma$ syndrome bits and $H$ verification bits. In the simplest model, leakage is:

$$L_{\mathrm{EC}} = |\Sigma| + |H|$$

In a blind-reconciliation variant with multiple increments, leakage must sum all increments: $L_{\mathrm{EC}} = \sum_i |\Sigma_i| + |H|$.

### B.4 NSM entropy bound placeholder

The Phase IV spec references an NSM “max bound” of the form:

$$h_{min}(r) \ge \max\{\Gamma[1-\log(1+3r^2)],\, 1-r\}$$

In the migration, this bound must be instantiated with a precise operational meaning of $r$ and a concrete definition of $\Gamma[\cdot]$ for the modeled storage channel.

## Appendix C. Source Inventory (ehok, squidasm, netqasm, netsquid_magic)

This appendix is auto-generated from the repository and installed packages and is intended to provide a source-of-truth inventory for refactoring. It lists modules and their public symbol surfaces (classes, functions, constants). No code is reproduced; only identifiers are indexed.

### C.1 Inventory: ehok

- Total python files indexed: 63

#### qia-challenge-2025/ehok/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/analysis/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/analysis/metrics.py
- Constants: (none)
- Functions: compute_execution_metrics
- Classes: (none)

#### qia-challenge-2025/ehok/configs/generate_ldpc.py
- Constants: (none)
- Functions: _get_distributions, generate_all, main
- Classes: (none)

#### qia-challenge-2025/ehok/core/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/core/config.py
- Constants: (none)
- Functions: (none)
- Classes:
  - PrivacyAmplificationConfig
    - Methods (1):
      - __post_init__
  - ProtocolConfig
    - Methods (4):
      - __post_init__
      - baseline
      - copy_with
      - to_dict
  - QuantumConfig
    - Methods (1):
      - __post_init__
  - ReconciliationConfig
    - Methods (1):
      - __post_init__
  - SecurityConfig
    - Methods (1):
      - __post_init__

#### qia-challenge-2025/ehok/core/constants.py
- Constants: BATCH_SIZE, CLASSICAL_TIMEOUT_SEC, LDPC_AVAILABLE_RATES, LDPC_BP_THRESHOLD, LDPC_CODE_RATE, LDPC_CODE_RATES, LDPC_CRITICAL_EFFICIENCY, LDPC_DEFAULT_RATE, LDPC_DEGREE_DISTRIBUTIONS, LDPC_DEGREE_DISTRIBUTIONS_PATH, LDPC_FRAME_SIZE, LDPC_F_CRIT, LDPC_HASH_BITS, LDPC_MATRIX_FILE_PATTERN, LDPC_MAX_ITERATIONS, LDPC_QBER_WINDOW_SIZE, LDPC_TEST_FRAME_SIZES, LDPC_TEST_MATRIX_SUBDIR, LINK_FIDELITY_MIN, LOG_LEVEL, LOG_TO_FILE, MIN_TEST_SET_SIZE, PA_SECURITY_MARGIN, PEG_DEFAULT_SEED, PEG_MAX_TREE_DEPTH, QBER_THRESHOLD, TARGET_EPSILON_SEC, TEST_SET_FRACTION, TOTAL_EPR_PAIRS
- Functions: _load_ldpc_degree_distributions
- Classes: (none)

#### qia-challenge-2025/ehok/core/data_structures.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ExecutionMetrics
    - Methods (1):
      - __post_init__
  - LDPCBlockResult
    - Methods (1):
      - __post_init__
  - LDPCMatrixPool
    - Methods (1):
      - __post_init__
  - LDPCReconciliationResult
    - Methods (1):
      - __post_init__
  - MeasurementRecord
    - Methods (1):
      - __post_init__
  - ObliviousKey
    - Methods (1):
      - __post_init__
  - ProtocolResult
    - Methods (1):
      - __post_init__

#### qia-challenge-2025/ehok/core/exceptions.py
- Constants: (none)
- Functions: (none)
- Classes:
  - CommitmentVerificationError
    - Methods: (none detected)
  - EHOKException
    - Methods: (none detected)
  - MatrixSynchronizationError
    - Methods (1):
      - __init__
  - ProtocolError
    - Methods: (none detected)
  - QBERTooHighError
    - Methods (1):
      - __init__
  - ReconciliationFailedError
    - Methods: (none detected)
  - SecurityException
    - Methods: (none detected)

#### qia-challenge-2025/ehok/core/sifting.py
- Constants: (none)
- Functions: (none)
- Classes:
  - SiftingManager
    - Methods (4):
      - check_qber_abort
      - estimate_qber
      - identify_matching_bases
      - select_test_set

#### qia-challenge-2025/ehok/examples/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/examples/debug_qber.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/examples/run_baseline.py
- Constants: (none)
- Functions: print_results, run_ehok_baseline
- Classes: (none)

#### qia-challenge-2025/ehok/examples/test_noise.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AliceTestProgram
    - Methods (3):
      - __init__
      - meta
      - run
  - BobTestProgram
    - Methods (3):
      - __init__
      - meta
      - run

#### qia-challenge-2025/ehok/implementations/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/commitment/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/commitment/merkle_commitment.py
- Constants: (none)
- Functions: (none)
- Classes:
  - MerkleCommitment
    - Methods (4):
      - __init__
      - commit
      - open_subset
      - verify
  - MerkleTree
    - Methods (5):
      - __init__
      - _build_tree
      - get_proof
      - root
      - verify_proof

#### qia-challenge-2025/ehok/implementations/commitment/sha256_commitment.py
- Constants: (none)
- Functions: (none)
- Classes:
  - SHA256Commitment
    - Methods (4):
      - __init__
      - commit
      - open_subset
      - verify

#### qia-challenge-2025/ehok/implementations/factories.py
- Constants: (none)
- Functions: build_commitment_scheme, build_noise_estimator, build_privacy_amplifier, build_reconciliator, build_sampling_strategy
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/noise/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/noise/simple_noise_estimator.py
- Constants: (none)
- Functions: (none)
- Classes:
  - SimpleNoiseEstimator
    - Methods (1):
      - estimate_leakage

#### qia-challenge-2025/ehok/implementations/privacy_amplification/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/privacy_amplification/finite_key.py
- Constants: DEFAULT_EPSILON_COR, DEFAULT_EPSILON_SEC
- Functions: binary_entropy, compute_blind_reconciliation_leakage, compute_final_length_blind_mode, compute_final_length_finite_key, compute_statistical_fluctuation, estimate_qber_from_reconciliation
- Classes:
  - FiniteKeyParams
    - Methods (1):
      - __post_init__

#### qia-challenge-2025/ehok/implementations/privacy_amplification/toeplitz_amplifier.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ToeplitzAmplifier
    - Methods (8):
      - __init__
      - _compress_direct
      - _compress_fft
      - compress
      - compute_final_length
      - compute_final_length_asymptotic
      - compute_final_length_blind
      - generate_hash_seed

#### qia-challenge-2025/ehok/implementations/reconciliation/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_bp_decoder.py
- Constants: (none)
- Functions: (none)
- Classes:
  - LDPCBeliefPropagation
    - Methods (2):
      - __init__
      - decode

#### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_matrix_manager.py
- Constants: (none)
- Functions: (none)
- Classes:
  - LDPCMatrixManager
    - Methods (10):
      - __init__
      - _apply_puncturing
      - _autogenerate_matrix
      - _compute_checksum
      - checksum
      - frame_size
      - from_directory
      - get_matrix
      - rates
      - verify_checksum

#### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_reconciliator.py
- Constants: (none)
- Functions: _binary_entropy
- Classes:
  - LDPCReconciliator
    - Methods (12):
      - __init__
      - _build_error_llrs
      - _generate_padding
      - aggregate_results
      - compute_adaptive_iterations
      - compute_shortening
      - compute_syndrome_block
      - estimate_leakage_block
      - reconcile_block
      - select_rate
      - syndrome_guided_llr_init
      - verify_block

#### qia-challenge-2025/ehok/implementations/reconciliation/peg_generator.py
- Constants: (none)
- Functions: (none)
- Classes:
  - DegreeDistribution
    - Methods (1):
      - __post_init__
  - PEGMatrixGenerator
    - Methods (6):
      - __init__
      - _assign_node_degrees
      - _bfs_reachable
      - _sample_degrees
      - _select_check_node
      - generate

#### qia-challenge-2025/ehok/implementations/reconciliation/polynomial_hash.py
- Constants: (none)
- Functions: (none)
- Classes:
  - PolynomialHashVerifier
    - Methods (5):
      - __init__
      - compute_hash
      - hash_and_seed
      - hash_length_bits
      - verify

#### qia-challenge-2025/ehok/implementations/reconciliation/qber_estimator.py
- Constants: (none)
- Functions: (none)
- Classes:
  - IntegratedQBEREstimator
    - Methods (4):
      - __init__
      - estimate
      - update_rolling
      - window

#### qia-challenge-2025/ehok/implementations/sampling/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/implementations/sampling/random_sampling.py
- Constants: (none)
- Functions: (none)
- Classes:
  - RandomSamplingStrategy
    - Methods (1):
      - select_sets

#### qia-challenge-2025/ehok/interfaces/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/interfaces/commitment.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ICommitmentScheme
    - Methods (3):
      - commit
      - open_subset
      - verify

#### qia-challenge-2025/ehok/interfaces/noise_estimator.py
- Constants: (none)
- Functions: (none)
- Classes:
  - INoiseEstimator
    - Methods (1):
      - estimate_leakage

#### qia-challenge-2025/ehok/interfaces/privacy_amplification.py
- Constants: (none)
- Functions: (none)
- Classes:
  - IPrivacyAmplifier
    - Methods (3):
      - compress
      - compute_final_length
      - generate_hash_seed

#### qia-challenge-2025/ehok/interfaces/reconciliation.py
- Constants: (none)
- Functions: (none)
- Classes:
  - IReconciliator
    - Methods (6):
      - compute_shortening
      - compute_syndrome_block
      - estimate_leakage_block
      - reconcile_block
      - select_rate
      - verify_block

#### qia-challenge-2025/ehok/interfaces/sampling_strategy.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ISamplingStrategy
    - Methods (1):
      - select_sets

#### qia-challenge-2025/ehok/protocols/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/protocols/alice.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AliceBaselineEHOK
    - Methods (7):
      - __init__
      - _build_reconciliator
      - _execute_remaining_phases
      - _phase2_receive_commitment
      - _phase3_sifting_sampling
      - _phase4_reconciliation
      - _phase5_privacy_amplification

#### qia-challenge-2025/ehok/protocols/base.py
- Constants: (none)
- Functions: (none)
- Classes:
  - EHOKRole
    - Methods (11):
      - __init__
      - _build_privacy_amplifier
      - _build_quantum_runner
      - _build_reconciliator
      - _build_strategies
      - _execute_remaining_phases
      - _phase1_quantum
      - _result_abort
      - _result_success
      - meta
      - run

#### qia-challenge-2025/ehok/protocols/bob.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BobBaselineEHOK
    - Methods (7):
      - __init__
      - _build_reconciliator
      - _execute_remaining_phases
      - _phase2_send_commitment
      - _phase3_sifting_sampling
      - _phase4_reconciliation
      - _phase5_privacy_amplification

#### qia-challenge-2025/ehok/quantum/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/quantum/basis_selection.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BasisSelector
    - Methods (3):
      - __init__
      - basis_to_string
      - generate_bases

#### qia-challenge-2025/ehok/quantum/batching_manager.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BatchResult
    - Methods: (none detected)
  - BatchingManager
    - Methods (2):
      - __init__
      - compute_batch_sizes
  - EPRGenerator
    - Methods (5):
      - __init__
      - extract_batch_results
      - generate_batch_alice
      - generate_batch_bob
      - measure_batch

#### qia-challenge-2025/ehok/quantum/measurement.py
- Constants: (none)
- Functions: (none)
- Classes:
  - MeasurementBuffer
    - Methods (6):
      - __init__
      - __len__
      - add_batch
      - clear
      - get_bases
      - get_outcomes

#### qia-challenge-2025/ehok/quantum/runner.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QuantumPhaseResult
    - Methods: (none detected)
  - QuantumPhaseRunner
    - Methods (4):
      - __init__
      - connection
      - csocket
      - run

#### qia-challenge-2025/ehok/tests/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/tests/conftest.py
- Constants: TEST_FRAME_SIZE, TEST_LDPC_RATES
- Functions: _generate_test_matrix, _get_test_degree_distributions, _require_ldpc_matrix, baseline_config, fast_test_config, pytest_addoption, rng, sample_sifted_key, sample_sifted_key_pair, test_ldpc_matrix_dir, test_ldpc_matrix_pool, test_matrix_manager, test_reconciliator
- Classes: (none)

#### qia-challenge-2025/ehok/tests/test_commitment.py
- Constants: (none)
- Functions: test_benchmark_commitments
- Classes:
  - TestMerkleCommitment
    - Methods (5):
      - setup_method
      - test_binding_property
      - test_commit_verify_correctness
      - test_subset_opening
      - test_tampered_subset_rejected
  - TestSHA256Commitment
    - Methods (5):
      - setup_method
      - test_binding_property
      - test_commit_verify_correctness
      - test_subset_opening
      - test_tampered_subset_rejected

#### qia-challenge-2025/ehok/tests/test_foundation.py
- Constants: (none)
- Functions: (none)
- Classes:
  - TestAbstractInterfaces
    - Methods (6):
      - test_icommitment_scheme_abstract_methods
      - test_icommitment_scheme_not_instantiable
      - test_iprivacy_amplifier_abstract_methods
      - test_iprivacy_amplifier_not_instantiable
      - test_ireconciliator_abstract_methods
      - test_ireconciliator_not_instantiable
  - TestConstants
    - Methods (6):
      - test_ldpc_parameters
      - test_logging_configuration
      - test_network_configuration
      - test_privacy_amplification_parameters
      - test_protocol_parameters
      - test_quantum_parameters
  - TestDataStructures
    - Methods (11):
      - test_measurement_record_construction_valid
      - test_measurement_record_invalid_basis
      - test_measurement_record_invalid_outcome
      - test_oblivious_key_construction_valid
      - test_oblivious_key_invalid_key_value_range
      - test_oblivious_key_invalid_key_value_type
      - test_oblivious_key_length_mismatch
      - test_protocol_result_construction_valid
      - test_protocol_result_invariants_fail_on_counts
      - test_protocol_result_invariants_fail_on_qber_and_key_length
      - test_protocol_result_with_abort
  - TestDocstrings
    - Methods (1):
      - test_public_classes_have_docstrings
  - TestExceptionHierarchy
    - Methods (9):
      - test_all_exceptions_are_catchable_as_ehok_exception
      - test_commitment_verification_error_inherits_from_security_exception
      - test_ehok_exception_inherits_from_exception
      - test_matrix_synchronization_error_inherits_from_protocol_error
      - test_protocol_error_inherits_from_ehok_exception
      - test_qber_too_high_error_custom_attributes
      - test_qber_too_high_error_inherits_from_security_exception
      - test_reconciliation_failed_error_inherits_from_protocol_error
      - test_security_exception_inherits_from_ehok_exception
  - TestLogging
    - Methods (5):
      - test_get_logger_returns_logger
      - test_hierarchical_logger_names
      - test_no_print_statements_in_production_code
      - test_setup_ehok_logging_console_only
      - test_setup_ehok_logging_with_file
  - TestPhase0Integration
    - Methods (2):
      - test_data_structures_with_exceptions
      - test_logging_with_exceptions
  - TestProtocolConfigBinding
    - Methods (2):
      - test_protocols_accept_protocol_config
      - test_protocols_do_not_instantiate_concretes

#### qia-challenge-2025/ehok/tests/test_integration.py
- Constants: (none)
- Functions: _make_perfect_network_config, test_phase_sequencing_commitment_before_bases, test_synchronization_flush_required
- Classes:
  - AliceNoFlush
    - Methods (1):
      - _execute_remaining_phases
  - AlicePhaseSeq
    - Methods (1):
      - _execute_remaining_phases
  - BobNoFlush
    - Methods (1):
      - _execute_remaining_phases
  - BobPhaseSeq
    - Methods (1):
      - _execute_remaining_phases

#### qia-challenge-2025/ehok/tests/test_ldpc_integration.py
- Constants: FRAME_SIZE, RATE
- Functions: _build_manager, _simulate_bob, test_alice_bob_handshake_success, test_matrix_sync_failure, test_reconciliation_failure_on_hash_mismatch
- Classes: (none)

#### qia-challenge-2025/ehok/tests/test_ldpc_reconciliation.py
- Constants: FRAME_SIZE, RATE
- Functions: _write_test_matrix, test_constants_load_normalized_distributions, test_degree_distribution_normalization, test_ldpc_reconciliator_block_roundtrip, test_ldpc_reconciliator_hash_and_leakage, test_matrix_manager_checksum_and_access, test_peg_generator_respects_regular_degrees
- Classes: (none)

#### qia-challenge-2025/ehok/tests/test_privacy_amplification.py
- Constants: (none)
- Functions: (none)
- Classes:
  - TestFiniteKeyFormula
    - Methods (12):
      - test_binary_entropy_properties
      - test_pt1_finite_key_formula_correctness
      - test_pt2_finite_key_vs_asymptotic
      - test_pt3_no_fixed_output_length_required
      - test_pt3_small_key_correctly_conservative
      - test_pt4_pa_robustness_qber_range
      - test_pt4_qber_monotonicity
      - test_pt5_blind_leakage_calculation
      - test_pt5_blind_mode_leakage_accounting
      - test_pt6_independence_different_seeds
      - test_pt6_output_key_uniformity
      - test_statistical_fluctuation_scaling
  - TestPrivacyAmplification
    - Methods (12):
      - setup_method
      - test_compress_returns_zero_for_m_zero
      - test_compression_correctness_small_example
      - test_compression_execution
      - test_compression_length_calculation
      - test_compute_final_length_security_bound
      - test_hankel_matrix_equivalence
      - test_invalid_seed_length
      - test_output_uniformity
      - test_output_uniformity_chi_square_10bits
      - test_toeplitz_construction
      - test_zero_length_output

#### qia-challenge-2025/ehok/tests/test_quantum.py
- Constants: (none)
- Functions: (none)
- Classes:
  - TestBasisRandomness
    - Methods (2):
      - test_independence
      - test_uniform_distribution
  - TestBatchingManager
    - Methods (1):
      - test_batch_size_computation
  - TestEPRGeneration
    - Methods (3):
      - _create_network_config
      - test_epr_generation_noisy
      - test_epr_generation_perfect

#### qia-challenge-2025/ehok/tests/test_reconciliation_integration.py
- Constants: (none)
- Functions: test_build_reconciliator_autogeneration_disabled, test_build_reconciliator_requires_existing_matrices, test_ldpc_manager_requires_on_disk_files, test_matrix_checksum_mismatch_raises_during_run, test_protocol_aborts_when_privacy_amplification_yields_zero_length, test_protocol_runs_with_local_ldpc_files_no_deadlock
- Classes: (none)

#### qia-challenge-2025/ehok/tests/test_sifting.py
- Constants: (none)
- Functions: (none)
- Classes:
  - TestSifting
    - Methods (5):
      - setup_method
      - test_basis_matching
      - test_qber_abort_threshold
      - test_qber_estimation_exact
      - test_test_set_selection

#### qia-challenge-2025/ehok/tests/test_system.py
- Constants: (none)
- Functions: _depolarise_network_config, _perfect_network_config, test_commitment_ordering_security, test_honest_execution_perfect, test_noise_tolerance_5pct, test_qber_abort_threshold
- Classes:
  - MaliciousAliceProgram
    - Methods (1):
      - _execute_remaining_phases

#### qia-challenge-2025/ehok/utils/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia-challenge-2025/ehok/utils/classical_sockets.py
- Constants: T
- Functions: (none)
- Classes:
  - StructuredSocket
    - Methods (5):
      - __init__
      - recv_str
      - recv_structured
      - send_str
      - send_structured

#### qia-challenge-2025/ehok/utils/logging.py
- Constants: (none)
- Functions: get_logger, setup_ehok_logging, setup_script_logging
- Classes: (none)

### C.2 Inventory: squidasm

- Total python files indexed: 54

#### squidasm/squidasm/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/nqasm/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/nqasm/executor/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/nqasm/executor/base.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidExecutor
    - Methods (20):
      - __init__
      - _clear_phys_qubit_in_memory
      - _do_controlled_qubit_rotation
      - _do_meas
      - _do_single_qubit_instr
      - _do_single_qubit_rotation
      - _do_two_qubit_instr
      - _do_wait
      - _execute_command
      - _execute_qdevice_instruction
      - _get_netsquid_instruction
      - _get_qubit
      - _get_simulated_time
      - _get_unused_physical_qubit
      - _meas_physical_qubit
      - _reserve_physical_qubit
      - _wait_to_handle_epr_responses
      - execute_subroutine
      - node_id
      - qdevice

#### squidasm/squidasm/nqasm/executor/nv.py
- Constants: NV_NS_INSTR_MAPPING
- Functions: (none)
- Classes:
  - NVNetSquidExecutor
    - Methods (2):
      - __init__
      - _do_meas

#### squidasm/squidasm/nqasm/executor/vanilla.py
- Constants: VANILLA_NS_INSTR_MAPPING
- Functions: (none)
- Classes:
  - VanillaNetSquidExecutor
    - Methods (1):
      - __init__

#### squidasm/squidasm/nqasm/multithread.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidConnection
    - Methods (5):
      - __init__
      - _commit_serialized_message
      - _execute_callback
      - _get_network_info
      - block
  - NetSquidNetworkInfo
    - Methods (4):
      - _get_node_id
      - _get_node_name
      - get_node_id_for_app
      - get_node_name_for_app

#### squidasm/squidasm/nqasm/netstack.py
- Constants: _SIGNALING_PROTOCOL
- Functions: reset_network
- Classes:
  - NetworkStack
    - Methods (7):
      - __init__
      - _get_recv_request
      - _setup_recv_rule
      - _wait_for_remote_node
      - get_purpose_id
      - put
      - setup_epr_socket
  - SignalingProtocol
    - Methods (7):
      - __init__
      - _assign_purpose_id
      - _get_purpose_id
      - get_circuit_id
      - has_circuit
      - reset
      - setup_circuit

#### squidasm/squidasm/nqasm/output.py
- Constants: (none)
- Functions: (none)
- Classes:
  - InstrLogger
    - Methods (5):
      - _get_node_name
      - _get_qubit_groups
      - _get_qubit_in_mem
      - _get_qubit_states
      - _update_qubits

#### squidasm/squidasm/nqasm/qnodeos.py
- Constants: _WAIT_EVENT_NAMES
- Functions: is_waiting_event
- Classes:
  - SubroutineHandler
    - Methods (16):
      - __init__
      - _get_executor_class
      - _get_next_other_task
      - _get_next_subroutine_task
      - _get_next_task_event
      - _handle_message
      - _handle_signal
      - _mark_message_finished
      - _next_message
      - _task_done
      - get_epr_reaction_handler
      - has_active_apps
      - network_stack
      - network_stack
      - run
      - stop
  - Task
    - Methods (6):
      - __init__
      - is_finished
      - is_waiting
      - msg
      - pop_next_event
      - update_next_event

#### squidasm/squidasm/nqasm/singlethread/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/nqasm/singlethread/connection.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidConnection
    - Methods (12):
      - __enter__
      - __exit__
      - __init__
      - _commit_message
      - _commit_open_epr_socket
      - _commit_serialized_message
      - _get_network_info
      - _wait_for_results
      - close
      - commit_protosubroutine
      - flush
      - shared_memory

#### squidasm/squidasm/nqasm/singlethread/csocket.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidSocket
    - Methods (3):
      - __init__
      - recv
      - send

#### squidasm/squidasm/run/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/run/multithread/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/run/multithread/runtime_mgr.py
- Constants: (none)
- Functions: (none)
- Classes:
  - SquidAsmRuntimeManager
    - Methods (19):
      - __init__
      - _create_subroutine_handlers
      - app_node_map
      - backend_log_dir
      - backend_log_dir
      - executors
      - is_running
      - netsquid_formalism
      - netsquid_formalism
      - network
      - nodes
      - party_map
      - qmemories
      - reset_backend
      - run_app
      - set_network
      - start_backend
      - stop_backend
      - subroutine_handlers

#### squidasm/squidasm/run/multithread/simulate.py
- Constants: _NS_FORMALISMS
- Functions: create_nv_cfg, simulate_application
- Classes: (none)

#### squidasm/squidasm/run/singlethread/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/run/singlethread/context.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidContext
    - Methods (6):
      - add_node
      - add_protocol
      - get_nodes
      - get_protocols
      - set_nodes
      - set_protocols
  - NetSquidNetworkInfo
    - Methods (4):
      - _get_node_id
      - _get_node_name
      - get_node_id_for_app
      - get_node_name_for_app

#### squidasm/squidasm/run/singlethread/protocols.py
- Constants: SUBRT_FINISHED
- Functions: (none)
- Classes:
  - HostPeerListener
    - Methods (3):
      - __init__
      - buffer
      - run
  - HostProtocol
    - Methods (14):
      - __init__
      - _receive_results
      - _recv_classical
      - _send_classical
      - _send_init_app_msg
      - _send_text_subroutine
      - get_result
      - peer_listener
      - peer_port
      - qnos_port
      - results_listener
      - run
      - start
      - stop
  - QNodeOsListener
    - Methods (3):
      - __init__
      - buffer
      - run
  - QNodeOsProtocol
    - Methods (10):
      - __init__
      - _receive_init_msg
      - _receive_msg
      - _receive_subroutine
      - executor
      - host_port
      - run
      - set_network_stack
      - start
      - stop
  - ResultsListener
    - Methods (3):
      - __init__
      - buffer
      - run

#### squidasm/squidasm/run/singlethread/run.py
- Constants: (none)
- Functions: _setup_connections, _setup_network_stacks, run_files, run_programs, run_protocols
- Classes: (none)

#### squidasm/squidasm/run/singlethread/util.py
- Constants: (none)
- Functions: load_program, modify_and_import
- Classes: (none)

#### squidasm/squidasm/run/stack/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/run/stack/build.py
- Constants: (none)
- Functions: create_stack_network_builder
- Classes:
  - StackNodeBuilder
    - Methods (1):
      - build

#### squidasm/squidasm/run/stack/config.py
- Constants: (none)
- Functions: _convert_stack_network_config
- Classes:
  - CLinkConfig
    - Methods (2):
      - from_file
      - perfect_config
  - DefaultCLinkConfig
    - Methods: (none detected)
  - DepolariseLinkConfig
    - Methods: (none detected)
  - GenericQDeviceConfig
    - Methods: (none detected)
  - HeraldedLinkConfig
    - Methods: (none detected)
  - InstantCLinkConfig
    - Methods: (none detected)
  - LinkConfig
    - Methods (2):
      - from_file
      - perfect_config
  - NVQDeviceConfig
    - Methods: (none detected)
  - StackConfig
    - Methods (2):
      - from_file
      - perfect_generic_config
  - StackNetworkConfig
    - Methods (1):
      - from_file

#### squidasm/squidasm/run/stack/run.py
- Constants: (none)
- Functions: _run, _setup_network, run
- Classes: (none)

#### squidasm/squidasm/sim/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/sim/glob.py
- Constants: _CURRENT_BACKEND
- Functions: get_current_app_node_mapping, get_current_node_ids, get_current_node_names, get_current_nodes, get_node_id, get_node_id_for_app, get_node_name, get_node_name_for_app, get_running_backend, pop_current_backend, put_current_backend
- Classes:
  - QubitInfo
    - Methods (2):
      - get_qubit_groups
      - update_qubits_used

#### squidasm/squidasm/sim/network/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/sim/network/network.py
- Constants: (none)
- Functions: (none)
- Classes:
  - LinearDepolariseMagicDistributor
    - Methods (1):
      - __init__
  - LinearDepolariseModelParameters
    - Methods (1):
      - verify
  - LinearDepolariseStateSamplerFactory
    - Methods (3):
      - __init__
      - _delivery_func
      - _get_success_probability
  - MagicNetworkLayerProtocol
    - Methods (5):
      - __init__
      - _get_log_data
      - _get_qubit_state
      - _get_unused_memory_positions
      - _handle_label_delivery
  - NVQDevice
    - Methods (1):
      - __init__
  - NetSquidNetwork
    - Methods (10):
      - __init__
      - _build_network
      - _create_link_distributor
      - _create_link_layer_services
      - global_log
      - host_latency
      - instr_proc_time
      - link_layer_services
      - node_hardware_types
      - set_logger
  - QDevice
    - Methods (1):
      - __init__

#### squidasm/squidasm/sim/network/nv_config.py
- Constants: (none)
- Functions: build_nv_qdevice, nv_cfg_from_file, parse_nv_config
- Classes:
  - NVConfig
    - Methods: (none detected)

#### squidasm/squidasm/sim/queues.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QueueManager
    - Methods (4):
      - create_queue
      - destroy_queues
      - get_queue
      - reset_queues
  - TaskQueue
    - Methods (10):
      - __init__
      - empty
      - full
      - get
      - join
      - join_task
      - put
      - qsize
      - reset
      - task_done

#### squidasm/squidasm/sim/stack/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/sim/stack/common.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AllocError
    - Methods: (none detected)
  - AppMemory
    - Methods (23):
      - __init__
      - expand_array_part
      - get_array
      - get_array_entry
      - get_array_part
      - get_array_slice
      - get_array_value
      - get_array_values
      - get_reg_value
      - get_register
      - increment_prog_counter
      - init_new_array
      - map_virt_id
      - phys_id_for
      - prog_counter
      - qubit_mapping
      - set_array_entry
      - set_array_value
      - set_prog_counter
      - set_reg_value
      - unmap_all
      - unmap_virt_id
      - virt_id_for
  - ComponentProtocol
    - Methods (5):
      - __init__
      - _receive_msg
      - add_listener
      - start
      - stop
  - LogManager
    - Methods (5):
      - _setup_stack_logger
      - get_log_level
      - get_stack_logger
      - log_to_file
      - set_log_level
  - NVPhysicalQuantumMemory
    - Methods (1):
      - __init__
  - NetstackBreakpointCreateRequest
    - Methods: (none detected)
  - NetstackBreakpointReceiveRequest
    - Methods: (none detected)
  - NetstackCreateRequest
    - Methods: (none detected)
  - NetstackReceiveRequest
    - Methods: (none detected)
  - PhysicalQuantumMemory
    - Methods (9):
      - __init__
      - allocate
      - allocate_comm
      - allocate_mem
      - clear
      - comm_qubit_count
      - free
      - is_allocated
      - qubit_count
  - PortListener
    - Methods (3):
      - __init__
      - buffer
      - run
  - RegisterMeta
    - Methods (2):
      - parse
      - prefixes
  - SimTimeFilter
    - Methods (1):
      - filter

#### squidasm/squidasm/sim/stack/connection.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QnosConnection
    - Methods (10):
      - __enter__
      - __exit__
      - __init__
      - _commit_message
      - _commit_serialized_message
      - _get_network_info
      - commit_protosubroutine
      - commit_subroutine
      - flush
      - shared_memory

#### squidasm/squidasm/sim/stack/context.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetSquidContext
    - Methods (7):
      - add_node
      - add_protocol
      - get_nodes
      - get_protocols
      - reset
      - set_nodes
      - set_protocols
  - NetSquidNetworkInfo
    - Methods (4):
      - _get_node_id
      - _get_node_name
      - get_node_id_for_app
      - get_node_name_for_app

#### squidasm/squidasm/sim/stack/csocket.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ClassicalSocket
    - Methods (11):
      - __init__
      - recv
      - recv_float
      - recv_int
      - recv_silent
      - recv_structured
      - send
      - send_float
      - send_int
      - send_silent
      - send_structured

#### squidasm/squidasm/sim/stack/globals.py
- Constants: (none)
- Functions: (none)
- Classes:
  - GlobalSimData
    - Methods (4):
      - get_last_breakpoint_state
      - get_network
      - get_quantum_state
      - set_network

#### squidasm/squidasm/sim/stack/handler.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Handler
    - Methods (23):
      - __init__
      - _deserialize_subroutine
      - _next_app
      - _receive_host_msg
      - _receive_processor_msg
      - _send_host_msg
      - _send_processor_msg
      - add_subroutine
      - app_memories
      - assign_processor
      - clear_application
      - flavour
      - flavour
      - init_new_app
      - msg_from_host
      - netstack
      - open_epr_socket
      - physical_memory
      - qnos
      - run
      - should_clear_memory
      - should_clear_memory
      - stop_application
  - HandlerComponent
    - Methods (9):
      - __init__
      - host_in_port
      - host_out_port
      - netstack_comp
      - node
      - processor_comp
      - processor_in_port
      - processor_out_port
      - qnos_comp
  - RunningApp
    - Methods (4):
      - __init__
      - add_subroutine
      - id
      - next_subroutine

#### squidasm/squidasm/sim/stack/host.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Host
    - Methods (9):
      - __init__
      - compiler
      - compiler
      - enqueue_program
      - get_results
      - receive_qnos_msg
      - register_netsquid_socket
      - run
      - send_qnos_msg
  - HostComponent
    - Methods (3):
      - __init__
      - qnos_in_port
      - qnos_out_port

#### squidasm/squidasm/sim/stack/netstack.py
- Constants: PI, PI_OVER_2
- Functions: (none)
- Classes:
  - EprSocket
    - Methods: (none detected)
  - Netstack
    - Methods (25):
      - __init__
      - _construct_request
      - _read_request_args_array
      - _receive_peer_msg
      - _receive_processor_msg
      - _send_peer_msg
      - _send_processor_msg
      - app_memories
      - assign_egp
      - find_epr_socket
      - handle_breakpoint_create_request
      - handle_breakpoint_receive_request
      - handle_create_ck_request
      - handle_create_md_request
      - handle_create_request
      - handle_receive_ck_request
      - handle_receive_md_request
      - handle_receive_request
      - open_epr_socket
      - physical_memory
      - qdevice
      - register_peer
      - run
      - start
      - stop
  - NetstackComponent
    - Methods (7):
      - __init__
      - node
      - peer_in_port
      - peer_out_port
      - processor_in_port
      - processor_out_port
      - register_peer

#### squidasm/squidasm/sim/stack/processor.py
- Constants: PI, PI_OVER_2
- Functions: (none)
- Classes:
  - GenericProcessor
    - Methods (6):
      - _interpret_controlled_rotation_instr
      - _interpret_init
      - _interpret_meas
      - _interpret_single_qubit_instr
      - _interpret_single_rotation_instr
      - _interpret_two_qubit_instr
  - NVProcessor
    - Methods (8):
      - _interpret_controlled_rotation_instr
      - _interpret_init
      - _interpret_meas
      - _interpret_qalloc
      - _interpret_single_rotation_instr
      - _measure_electron
      - _move_carbon_to_electron_for_measure
      - _move_electron_to_carbon
  - Processor
    - Methods (38):
      - __init__
      - _compute_binary_classical_instr
      - _do_controlled_rotation
      - _do_single_rotation
      - _flush_netstack_msgs
      - _get_rotation_angle_from_operands
      - _interpret_array
      - _interpret_binary_classical_instr
      - _interpret_branch_instr
      - _interpret_breakpoint
      - _interpret_controlled_rotation_instr
      - _interpret_create_epr
      - _interpret_init
      - _interpret_instruction
      - _interpret_lea
      - _interpret_load
      - _interpret_meas
      - _interpret_qalloc
      - _interpret_qfree
      - _interpret_recv_epr
      - _interpret_ret_arr
      - _interpret_ret_reg
      - _interpret_set
      - _interpret_single_qubit_instr
      - _interpret_single_rotation_instr
      - _interpret_store
      - _interpret_two_qubit_instr
      - _interpret_undef
      - _interpret_wait_all
      - _receive_handler_msg
      - _receive_netstack_msg
      - _send_handler_msg
      - _send_netstack_msg
      - app_memories
      - execute_subroutine
      - physical_memory
      - qdevice
      - run
  - ProcessorComponent
    - Methods (7):
      - __init__
      - handler_in_port
      - handler_out_port
      - netstack_in_port
      - netstack_out_port
      - node
      - qdevice

#### squidasm/squidasm/sim/stack/program.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Program
    - Methods (2):
      - meta
      - run
  - ProgramContext
    - Methods (5):
      - __init__
      - app_id
      - connection
      - csockets
      - epr_sockets
  - ProgramMeta
    - Methods: (none detected)

#### squidasm/squidasm/sim/stack/qnos.py
- Constants: NUM_QUBITS
- Functions: (none)
- Classes:
  - Qnos
    - Methods (13):
      - __init__
      - app_memories
      - assign_egp
      - get_virt_qubit_for_phys_id
      - handler
      - handler
      - netstack
      - netstack
      - physical_memory
      - processor
      - processor
      - start
      - stop
  - QnosComponent
    - Methods (11):
      - __init__
      - handler_comp
      - host_in_port
      - host_out_port
      - netstack_comp
      - node
      - peer_in_port
      - peer_out_port
      - processor_comp
      - qdevice
      - register_peer

#### squidasm/squidasm/sim/stack/qnos_network_service.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QNOSNetworkService
    - Methods (8):
      - __init__
      - _build_combined_port_output_event_expr
      - receive_qnos_message
      - register_remote_node
      - run
      - send_qnos_message
      - start
      - stop
  - ReqQNOSMessage
    - Methods: (none detected)

#### squidasm/squidasm/sim/stack/signals.py
- Constants: SIGNAL_HAND_HOST_MSG, SIGNAL_HAND_PROC_MSG, SIGNAL_HOST_HAND_MSG, SIGNAL_HOST_HOST_MSG, SIGNAL_MEMORY_FREED, SIGNAL_NSTK_PROC_MSG, SIGNAL_PEER_NSTK_MSG, SIGNAL_PEER_RECV_MSG, SIGNAL_PROC_HAND_MSG, SIGNAL_PROC_NSTK_MSG
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/sim/stack/stack.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NodeStack
    - Methods (12):
      - __init__
      - assign_egp
      - host
      - host
      - host_comp
      - node
      - qdevice
      - qnos
      - qnos
      - qnos_comp
      - start
      - stop
  - StackNetwork
    - Methods (5):
      - __init__
      - csockets
      - links
      - qdevices
      - stacks
  - StackNode
    - Methods (5):
      - __init__
      - host_comp
      - qnos_comp
      - qnos_peer_port
      - register_peer

#### squidasm/squidasm/util/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### squidasm/squidasm/util/ns.py
- Constants: (none)
- Functions: is_dm_pure, is_ppt, is_pure_state_entangled, is_state_entangled, partial_transpose
- Classes: (none)

#### squidasm/squidasm/util/qkd_routine.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QKDRoutine
    - Methods (4):
      - _distribute_states
      - _estimate_error_rate
      - _filter_bases
      - run
  - _PairInfo
    - Methods: (none detected)

#### squidasm/squidasm/util/routines.py
- Constants: (none)
- Functions: create_ghz, distributed_CNOT_control, distributed_CNOT_target, distributed_CPhase_control, distributed_CPhase_target, measXY, recv_float, recv_int, recv_remote_state_preparation, remote_state_preparation, send_float, send_int, teleport_recv, teleport_send
- Classes:
  - _Role
    - Methods: (none detected)

#### squidasm/squidasm/util/sim.py
- Constants: (none)
- Functions: get_qubit_state
- Classes: (none)

#### squidasm/squidasm/util/thread.py
- Constants: (none)
- Functions: as_completed
- Classes: (none)

#### squidasm/squidasm/util/util.py
- Constants: (none)
- Functions: create_complete_graph_network, create_two_node_network, get_qubit_state, get_reference_state
- Classes: (none)

### C.3 Inventory: netqasm

- Total python files indexed: 163

#### qia/lib/python3.10/site-packages/netqasm/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/backend/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/backend/executor.py
- Constants: (none)
- Functions: inc_program_counter
- Classes:
  - EprCmdData
    - Methods: (none detected)
  - Executor
    - Methods (98):
      - __init__
      - _allocate_physical_qubit
      - _clear_arrays
      - _clear_phys_qubit_in_memory
      - _clear_qubits
      - _clear_registers
      - _clear_shared_memory
      - _clear_subroutine
      - _compute_binary_classical_instr
      - _do_controlled_qubit_rotation
      - _do_create_epr
      - _do_meas
      - _do_recv_epr
      - _do_single_qubit_instr
      - _do_single_qubit_rotation
      - _do_two_qubit_instr
      - _do_wait
      - _execute_command
      - _execute_commands
      - _expand_array_part
      - _extract_epr_info
      - _free_physical_qubit
      - _get_app_id
      - _get_array
      - _get_array_entry
      - _get_array_slice
      - _get_create_request
      - _get_epr_response_handlers
      - _get_instruction_handlers
      - _get_new_qubit_unit_module
      - _get_new_subroutine_id
      - _get_num_pairs_from_array
      - _get_position
      - _get_position_in_unit_module
      - _get_positions
      - _get_purpose_id
      - _get_qubit
      - _get_qubit_state
      - _get_register
      - _get_rotation_angle_from_operands
      - _get_simulated_time
      - _get_unit_module
      - _get_unused_physical_qubit
      - _get_virtual_address_from_epr_data
      - _handle_binary_classical_instr
      - _handle_branch_instr
      - _handle_command_exception
      - _handle_controlled_qubit_rotation
      - _handle_epr_err_response
      - _handle_epr_ok_k_response
      - _handle_epr_ok_m_response
      - _handle_epr_ok_r_response
      - _handle_epr_response
      - _handle_last_epr_pair
      - _handle_pending_epr_responses
      - _handle_single_qubit_instr
      - _handle_single_qubit_rotation
      - _handle_two_qubit_instr
      - _has_virtual_address
      - _initialize_array
      - _instr_array
      - _instr_create_epr
      - _instr_lea
      - _instr_load
      - _instr_meas
      - _instr_qalloc
      - _instr_qfree
      - _instr_recv_epr
      - _instr_ret_arr
      - _instr_ret_reg
      - _instr_set
      - _instr_store
      - _instr_undef
      - _instr_wait_all
      - _instr_wait_any
      - _instr_wait_single
      - _new_shared_memory
      - _reserve_physical_qubit
      - _reset_program_counter
      - _set_array_entry
      - _set_register
      - _setup_arrays
      - _setup_registers
      - _store_ent_info
      - _update_shared_memory
      - _wait_to_handle_epr_responses
      - allocate_new_qubit_unit_module
      - consume_execute_subroutine
      - execute_subroutine
      - get_instr_logger
      - init_new_application
      - name
      - network_stack
      - network_stack
      - node_id
      - set_instr_logger
      - setup_epr_socket
      - stop_application

#### qia/lib/python3.10/site-packages/netqasm/backend/messages.py
- Constants: APP_ID, EPR_FIDELITY, EPR_SOCKET_ID, MESSAGE_CLASSES, MESSAGE_ID, MESSAGE_TYPE, MESSAGE_TYPE_BYTES, NODE_ID, NUM_QUBITS, RETURN_MESSAGE_CLASSES, SIGNAL
- Functions: deserialize_host_msg, deserialize_return_msg
- Classes:
  - ErrorCode
    - Methods: (none detected)
  - ErrorMessage
    - Methods (1):
      - __init__
  - InitNewAppMessage
    - Methods (1):
      - __init__
  - Message
    - Methods (3):
      - __len__
      - __str__
      - deserialize_from
  - MessageHeader
    - Methods (2):
      - __str__
      - len
  - MessageType
    - Methods: (none detected)
  - MsgDoneMessage
    - Methods (1):
      - __init__
  - OpenEPRSocketMessage
    - Methods (1):
      - __init__
  - ReturnArrayMessage
    - Methods (5):
      - __bytes__
      - __init__
      - __len__
      - __str__
      - deserialize_from
  - ReturnArrayMessageHeader
    - Methods (1):
      - len
  - ReturnMessage
    - Methods: (none detected)
  - ReturnMessageType
    - Methods: (none detected)
  - ReturnRegMessage
    - Methods (1):
      - __init__
  - Signal
    - Methods: (none detected)
  - SignalMessage
    - Methods (1):
      - __init__
  - StopAppMessage
    - Methods (1):
      - __init__
  - SubroutineMessage
    - Methods (4):
      - __bytes__
      - __init__
      - __len__
      - deserialize_from

#### qia/lib/python3.10/site-packages/netqasm/backend/network_stack.py
- Constants: CREATE_FIELDS, OK_FIELDS_K, OK_FIELDS_M
- Functions: (none)
- Classes:
  - Address
    - Methods: (none detected)
  - BaseNetworkStack
    - Methods (3):
      - get_purpose_id
      - put
      - setup_epr_socket

#### qia/lib/python3.10/site-packages/netqasm/backend/qnodeos.py
- Constants: (none)
- Functions: (none)
- Classes:
  - QNodeController
    - Methods (20):
      - __init__
      - _add_app
      - _execute_subroutine
      - _get_executor_class
      - _get_message_handlers
      - _handle_init_new_app
      - _handle_message
      - _handle_open_epr_socket
      - _handle_signal
      - _handle_stop_app
      - _handle_subroutine
      - _mark_message_finished
      - _remove_app
      - add_network_stack
      - finished
      - handle_netqasm_message
      - has_active_apps
      - network_stack
      - network_stack
      - stop

#### qia/lib/python3.10/site-packages/netqasm/examples/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_alice.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_bob.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_charlie.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_david.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/conf.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/protocol.py
- Constants: (none)
- Functions: anonymous_transmission
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/sub_protocols.py
- Constants: (none)
- Functions: _setup_broadcast_channel, _setup_down_sockets, _setup_sockets, _setup_up_sockets, anonymous_epr, classical_anonymous_transmission, quantum_anonymous_tranmission, setup_sockets
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/app_alice.py
- Constants: ALL_MEASURED, EOF
- Functions: distribute_bb84_states, estimate_error_rate, extract_key, filter_bases, h, main, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Classes:
  - PairInfo
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/app_bob.py
- Constants: ALL_MEASURED, EOF
- Functions: estimate_error_rate, extract_key, filter_bases, h, main, receive_bb84_states, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Classes:
  - PairInfo
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/app_client.py
- Constants: (none)
- Functions: get_phi_for_oracle, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/app_server.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/app_client.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/app_server.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_alice.py
- Constants: (none)
- Functions: main, measure_basis_0, measure_basis_1
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_bob.py
- Constants: (none)
- Functions: main, measure_basis_0, measure_basis_1
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_repeater.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/app_controller.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/app_target.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/app_client.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/app_server.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/app_client.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/app_server.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/app_player1.py
- Constants: (none)
- Functions: _get_default_strategy, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/app_player2.py
- Constants: (none)
- Functions: _get_default_strategy, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/app_alice.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/app_bob.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/shared/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/shared/myfuncs.py
- Constants: (none)
- Functions: custom_measure, custom_recv, custom_send
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/single_node/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/single_node/app_alice.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/app_receiver.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/app_sender.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_alice.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_bob.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_charlie.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/lib/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/lib/bqc.py
- Constants: (none)
- Functions: measXY, recv_meas_cmd, recv_meas_outcome, recv_teleported_state, send_meas_cmd, send_meas_outcome, teleport_state
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/app_client.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/app_server.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_alice.py
- Constants: (none)
- Functions: format_measurement_basis, main, measure_basis_0, measure_basis_1
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_bob.py
- Constants: (none)
- Functions: format_corrections, format_measurement_basis, game_won, main, measure_basis_0, measure_basis_1
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_repeater.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/app_controller.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/app_target.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/app_player1.py
- Constants: (none)
- Functions: _get_default_strategy, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/app_player2.py
- Constants: (none)
- Functions: _get_default_strategy, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/app_alice.py
- Constants: ALL_MEASURED, EOF
- Functions: distribute_bb84_states, estimate_error_rate, extract_key, filter_bases, h, main, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Classes:
  - PairInfo
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/app_bob.py
- Constants: ALL_MEASURED, EOF
- Functions: estimate_error_rate, extract_key, filter_bases, h, main, receive_bb84_states, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Classes:
  - PairInfo
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/app_receiver.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/app_sender.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/run_examples.py
- Constants: (none)
- Functions: _has_first_argument, main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_bb84.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_enumerate.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_loop.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_post_epr.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_rsp.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_simple_loop.py
- Constants: (none)
- Functions: main
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_scripts/epr.py
- Constants: (none)
- Functions: create_epr, post_function, run_alice, run_bob
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/examples/sdk_scripts/rsp.py
- Constants: PRECOMPILE
- Functions: run_client, run_server
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/lang/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/lang/encoding.py
- Constants: ADDRESS, ADDRESS_BITS, APP_ID, COMMANDS, COMMAND_BYTES, IMMEDIATE, IMMEDIATE_BITS, INSTR_ID, INTEGER, INTEGER_BITS, METADATA_BYTES, NETQASM_VERSION, PADDING_FIELD, REG_BITS, REG_INDEX_BITS, REG_NAME_BITS, REG_TYPE
- Functions: add_padding
- Classes:
  - AddrCommand
    - Methods: (none detected)
  - Address
    - Methods: (none detected)
  - ArrayCommand
    - Methods: (none detected)
  - ArrayEntry
    - Methods: (none detected)
  - ArrayEntryCommand
    - Methods: (none detected)
  - ArraySlice
    - Methods: (none detected)
  - ArraySliceCommand
    - Methods: (none detected)
  - Command
    - Methods (1):
      - __init__
  - ImmCommand
    - Methods: (none detected)
  - ImmImmCommand
    - Methods: (none detected)
  - MeasCommand
    - Methods: (none detected)
  - Metadata
    - Methods: (none detected)
  - NoOperandCommand
    - Methods: (none detected)
  - OptionalInt
    - Methods (2):
      - __init__
      - value
  - RecvEPRCommand
    - Methods: (none detected)
  - Reg5Command
    - Methods: (none detected)
  - RegAddrCommand
    - Methods: (none detected)
  - RegCommand
    - Methods: (none detected)
  - RegEntryCommand
    - Methods: (none detected)
  - RegImmCommand
    - Methods: (none detected)
  - RegImmImmCommand
    - Methods: (none detected)
  - RegRegCommand
    - Methods: (none detected)
  - RegRegImm4Command
    - Methods: (none detected)
  - RegRegImmCommand
    - Methods: (none detected)
  - RegRegImmImmCommand
    - Methods: (none detected)
  - RegRegRegCommand
    - Methods: (none detected)
  - RegRegRegRegCommand
    - Methods: (none detected)
  - Register
    - Methods: (none detected)
  - RegisterName
    - Methods: (none detected)
  - SingleRegisterCommand
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/base.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AddrInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - ArrayEntryInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - ArraySliceInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - DebugInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - ImmImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - ImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - NetQASMInstruction
    - Methods (10):
      - __str__
      - _build_str
      - _get_lineno_str
      - _pretty_print
      - debug_str
      - deserialize_from
      - from_operands
      - operands
      - serialize
      - writes_to
  - NoOperandInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - Reg5Instruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegAddrInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegEntryInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegImmImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegImm4Instruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegImmImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegImmInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegRegInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize
  - RegRegRegRegInstruction
    - Methods (5):
      - _pretty_print
      - deserialize_from
      - from_operands
      - operands
      - serialize

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/core.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AddInstruction
    - Methods: (none detected)
  - AddmInstruction
    - Methods: (none detected)
  - ArrayInstruction
    - Methods (2):
      - size
      - size
  - BeqInstruction
    - Methods (1):
      - check_condition
  - BezInstruction
    - Methods (1):
      - check_condition
  - BgeInstruction
    - Methods (1):
      - check_condition
  - BltInstruction
    - Methods (1):
      - check_condition
  - BneInstruction
    - Methods (1):
      - check_condition
  - BnzInstruction
    - Methods (1):
      - check_condition
  - BranchBinaryInstruction
    - Methods (3):
      - check_condition
      - line
      - line
  - BranchUnaryInstruction
    - Methods (3):
      - check_condition
      - line
      - line
  - BreakpointInstruction
    - Methods (4):
      - action
      - action
      - role
      - role
  - ClassicalOpInstruction
    - Methods (7):
      - regin0
      - regin0
      - regin1
      - regin1
      - regout
      - regout
      - writes_to
  - ClassicalOpModInstruction
    - Methods (9):
      - regin0
      - regin0
      - regin1
      - regin1
      - regmod
      - regmod
      - regout
      - regout
      - writes_to
  - ControlledRotationInstruction
    - Methods (9):
      - angle_denom
      - angle_denom
      - angle_num
      - angle_num
      - qreg0
      - qreg0
      - qreg1
      - qreg1
      - to_matrix
  - CreateEPRInstruction
    - Methods (10):
      - arg_array
      - arg_array
      - ent_results_array
      - ent_results_array
      - epr_socket_id
      - epr_socket_id
      - qubit_addr_array
      - qubit_addr_array
      - remote_node_id
      - remote_node_id
  - DivInstruction
    - Methods: (none detected)
  - InitInstruction
    - Methods (2):
      - qreg
      - qreg
  - JmpInstruction
    - Methods (2):
      - line
      - line
  - LeaInstruction
    - Methods (1):
      - writes_to
  - LoadInstruction
    - Methods (1):
      - writes_to
  - MeasBasisInstruction
    - Methods (13):
      - angle_denom
      - angle_denom
      - angle_num_x1
      - angle_num_x1
      - angle_num_x2
      - angle_num_x2
      - angle_num_y
      - angle_num_y
      - creg
      - creg
      - qreg
      - qreg
      - writes_to
  - MeasInstruction
    - Methods (5):
      - creg
      - creg
      - qreg
      - qreg
      - writes_to
  - MulInstruction
    - Methods: (none detected)
  - QAllocInstruction
    - Methods (2):
      - qreg
      - qreg
  - QFreeInstruction
    - Methods (2):
      - qreg
      - qreg
  - RecvEPRInstruction
    - Methods (8):
      - ent_results_array
      - ent_results_array
      - epr_socket_id
      - epr_socket_id
      - qubit_addr_array
      - qubit_addr_array
      - remote_node_id
      - remote_node_id
  - RemInstruction
    - Methods: (none detected)
  - RetArrInstruction
    - Methods: (none detected)
  - RetRegInstruction
    - Methods: (none detected)
  - RotationInstruction
    - Methods (8):
      - angle_denom
      - angle_denom
      - angle_num
      - angle_num
      - from_operands
      - qreg
      - qreg
      - to_matrix
  - SetInstruction
    - Methods (2):
      - from_operands
      - writes_to
  - SingleQubitInstruction
    - Methods (3):
      - qreg
      - qreg
      - to_matrix
  - StoreInstruction
    - Methods: (none detected)
  - SubInstruction
    - Methods: (none detected)
  - SubmInstruction
    - Methods: (none detected)
  - TwoQubitInstruction
    - Methods (6):
      - qreg0
      - qreg0
      - qreg1
      - qreg1
      - to_matrix
      - to_matrix_target_only
  - UndefInstruction
    - Methods: (none detected)
  - WaitAllInstruction
    - Methods: (none detected)
  - WaitAnyInstruction
    - Methods: (none detected)
  - WaitSingleInstruction
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/flavour.py
- Constants: CORE_INSTRUCTIONS
- Functions: (none)
- Classes:
  - Flavour
    - Methods (4):
      - __init__
      - get_instr_by_id
      - get_instr_by_name
      - instrs
  - InstrMap
    - Methods: (none detected)
  - NVFlavour
    - Methods (2):
      - __init__
      - instrs
  - REIDSFlavour
    - Methods (2):
      - __init__
      - instrs
  - TrappedIonFlavour
    - Methods (2):
      - __init__
      - instrs
  - VanillaFlavour
    - Methods (2):
      - __init__
      - instrs

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/nv.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ControlledRotXInstruction
    - Methods (2):
      - to_matrix
      - to_matrix_target_only
  - ControlledRotYInstruction
    - Methods (2):
      - to_matrix
      - to_matrix_target_only
  - RotXInstruction
    - Methods (1):
      - to_matrix
  - RotYInstruction
    - Methods (1):
      - to_matrix
  - RotZInstruction
    - Methods (1):
      - to_matrix

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/trapped_ion.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AllQubitsInitInstruction
    - Methods: (none detected)
  - AllQubitsMeasInstruction
    - Methods: (none detected)
  - AllQubitsRotXInstruction
    - Methods: (none detected)
  - AllQubitsRotYInstruction
    - Methods: (none detected)
  - AllQubitsRotZInstruction
    - Methods: (none detected)
  - AllQubitsRotationInstruction
    - Methods (5):
      - angle_denom
      - angle_denom
      - angle_num
      - angle_num
      - from_operands
  - BichromaticInstruction
    - Methods (5):
      - angle_denom
      - angle_denom
      - angle_num
      - angle_num
      - from_operands
  - RotZInstruction
    - Methods (1):
      - to_matrix

#### qia/lib/python3.10/site-packages/netqasm/lang/instr/vanilla.py
- Constants: (none)
- Functions: (none)
- Classes:
  - CnotInstruction
    - Methods (2):
      - to_matrix
      - to_matrix_target_only
  - CphaseInstruction
    - Methods (2):
      - to_matrix
      - to_matrix_target_only
  - GateHInstruction
    - Methods (1):
      - to_matrix
  - GateKInstruction
    - Methods (1):
      - to_matrix
  - GateSInstruction
    - Methods (1):
      - to_matrix
  - GateTInstruction
    - Methods (1):
      - to_matrix
  - GateXInstruction
    - Methods (1):
      - to_matrix
  - GateYInstruction
    - Methods (1):
      - to_matrix
  - GateZInstruction
    - Methods (1):
      - to_matrix
  - MovInstruction
    - Methods (2):
      - to_matrix
      - to_matrix_target_only
  - RotXInstruction
    - Methods (1):
      - to_matrix
  - RotYInstruction
    - Methods (1):
      - to_matrix
  - RotZInstruction
    - Methods (1):
      - to_matrix

#### qia/lib/python3.10/site-packages/netqasm/lang/ir.py
- Constants: _STRING_TO_INSTRUCTION
- Functions: _get_lineo_str, flip_branch_instr, instruction_to_string, string_to_instruction
- Classes:
  - BranchLabel
    - Methods (4):
      - __str__
      - _assert_types
      - _build_str
      - debug_str
  - BreakpointAction
    - Methods: (none detected)
  - BreakpointRole
    - Methods: (none detected)
  - GenericInstr
    - Methods: (none detected)
  - ICmd
    - Methods (4):
      - __post_init__
      - __str__
      - _build_str
      - debug_str
  - ProtoSubroutine
    - Methods (8):
      - __init__
      - __str__
      - app_id
      - arguments
      - commands
      - commands
      - instantiate
      - netqasm_version

#### qia/lib/python3.10/site-packages/netqasm/lang/operand.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Address
    - Methods (5):
      - __bytes__
      - __str__
      - _assert_types
      - cstruct
      - from_raw
  - ArrayEntry
    - Methods (6):
      - __bytes__
      - __post_init__
      - __str__
      - _assert_types
      - cstruct
      - from_raw
  - ArraySlice
    - Methods (6):
      - __bytes__
      - __post_init__
      - __str__
      - _assert_types
      - cstruct
      - from_raw
  - Immediate
    - Methods (1):
      - __str__
  - Label
    - Methods (2):
      - __str__
      - _assert_types
  - Operand
    - Methods: (none detected)
  - Register
    - Methods (6):
      - __bytes__
      - __str__
      - _assert_types
      - cstruct
      - from_raw
      - from_str
  - RegisterMeta
    - Methods (2):
      - parse
      - prefixes
  - Template
    - Methods (2):
      - __str__
      - _assert_types

#### qia/lib/python3.10/site-packages/netqasm/lang/parsing/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/lang/parsing/binary.py
- Constants: INSTR_ID
- Functions: deserialize
- Classes:
  - Deserializer
    - Methods (4):
      - __init__
      - _parse_metadata
      - deserialize_command
      - deserialize_subroutine

#### qia/lib/python3.10/site-packages/netqasm/lang/parsing/text.py
- Constants: _ALLOW_LABEL_INSTRUCTIONS, _REGISTER_NAMES, _REPLACE_CONSTANTS_EXCEPTION
- Functions: _apply_macros, _assert_single_preamble_arg, _assert_single_preamble_instr, _assert_valid_preamble_instr_appid, _assert_valid_preamble_instr_define, _assert_valid_preamble_instr_netqasm, _assert_valid_preamble_instructions, _assign_branch_labels, _build_subroutine, _create_subroutine, _get_unused_address, _is_byte, _make_args_operands, _parse_args, _parse_base_address, _parse_constant, _parse_index, _parse_label, _parse_netqasm_version, _parse_operand, _parse_operands, _parse_preamble, _parse_template, _parse_value, _remove_comments_from_line, _replace_constants, _split_instr_and_args, _split_of_bracket, _split_preamble_body, _update_labels, _update_labels_in_command, _update_labels_in_operand, assemble_subroutine, get_current_registers, parse_address, parse_register, parse_text_protosubroutine, parse_text_subroutine
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/lang/subroutine.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Subroutine
    - Methods (15):
      - __bytes__
      - __eq__
      - __init__
      - __len__
      - __str__
      - app_id
      - app_id
      - arguments
      - cstructs
      - instantiate
      - instructions
      - instructions
      - netqasm_version
      - pretty_print
      - print_instructions

#### qia/lib/python3.10/site-packages/netqasm/lang/symbols.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Symbols
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/lang/version.py
- Constants: NETQASM_VERSION
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/logging/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/logging/glob.py
- Constants: NETQASM_LOGGER
- Functions: _setup_netqasm_logger, get_log_level, get_netqasm_logger, set_log_level
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/logging/output.py
- Constants: _STRUCT_LOGGERS
- Functions: get_new_app_logger, reset_struct_loggers, save_all_struct_loggers, should_ignore_instr
- Classes:
  - AppLogger
    - Methods (2):
      - __init__
      - _construct_entry
  - ClassCommLogger
    - Methods (1):
      - _construct_entry
  - InstrLogger
    - Methods (9):
      - __init__
      - _construct_entry
      - _get_app_id
      - _get_node_name
      - _get_physical_qubit_ids
      - _get_qubit_groups
      - _get_qubit_ids
      - _get_qubit_states
      - _update_qubits
  - NetworkLogger
    - Methods (2):
      - __init__
      - _construct_entry
  - SocketOperation
    - Methods: (none detected)
  - StructuredLogger
    - Methods (6):
      - __init__
      - _construct_entry
      - _get_op_value
      - _get_op_values
      - log
      - save

#### qia/lib/python3.10/site-packages/netqasm/qlink_compat.py
- Constants: (none)
- Functions: get_creator_node_id, request_to_qlink_1_0, response_from_qlink_1_0
- Classes:
  - Basis
    - Methods: (none detected)
  - BellState
    - Methods: (none detected)
  - EPRRole
    - Methods: (none detected)
  - EPRType
    - Methods: (none detected)
  - ErrorCode
    - Methods: (none detected)
  - RandomBasis
    - Methods: (none detected)
  - RequestType
    - Methods: (none detected)
  - ReturnType
    - Methods: (none detected)
  - TimeUnit
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/runtime/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/app_config.py
- Constants: (none)
- Functions: default_app_config
- Classes:
  - AppConfig
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/runtime/application.py
- Constants: (none)
- Functions: app_instance_from_path, default_app_instance, load_yaml_file, network_cfg_from_path, post_function_from_path
- Classes:
  - AppMetadata
    - Methods: (none detected)
  - Application
    - Methods: (none detected)
  - ApplicationInstance
    - Methods: (none detected)
  - ApplicationOutput
    - Methods: (none detected)
  - Program
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/runtime/cli.py
- Constants: CONTEXT_SETTINGS, EXAMPLE_APPS, QNE_FOLDER_PATH
- Functions: cli, init, new, qne, run, simulate, version
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/debug.py
- Constants: (none)
- Functions: get_qubit_state, run_application
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/env.py
- Constants: EXAMPLE_APPS_DIR, IGNORED_FILES
- Functions: _create_new_input_file, _create_new_network_file, _create_new_readme_file, _create_new_results_config_file, _create_new_roles_file, _find_argument_for_app_file, file_creation_notify, get_example_apps, get_log_dir, get_post_function_path, get_results_path, get_roles_config_path, get_timed_log_dir, init_folder, load_app_config_file, load_app_files, load_post_function, load_roles_config, new_folder
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/hardware.py
- Constants: (none)
- Functions: run_application, save_results
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/interface/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/interface/config.py
- Constants: _DEFAULT_NUM_QUBITS
- Functions: default_network_config, network_cfg_from_file, parse_network_config
- Classes:
  - Link
    - Methods: (none detected)
  - NetworkConfig
    - Methods: (none detected)
  - Node
    - Methods: (none detected)
  - NoiseType
    - Methods: (none detected)
  - QuantumHardware
    - Methods: (none detected)
  - Qubit
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/runtime/interface/logging.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AppLogEntry
    - Methods: (none detected)
  - ClassCommLogEntry
    - Methods: (none detected)
  - EntanglementStage
    - Methods: (none detected)
  - EntanglementType
    - Methods: (none detected)
  - InstrLogEntry
    - Methods: (none detected)
  - NetworkLogEntry
    - Methods: (none detected)
  - QubitGroup
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/runtime/interface/results.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/process_logs.py
- Constants: _LAST_LOG
- Functions: _add_hln_to_log, _add_hln_to_log_entry, _add_hln_to_logs, create_app_instr_logs, make_last_log, process_log
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/runtime/runtime_mgr.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ApplicationInstance
    - Methods: (none detected)
  - NetworkConfig
    - Methods: (none detected)
  - NetworkInstance
    - Methods: (none detected)
  - RuntimeManager
    - Methods (4):
      - get_network
      - run_app
      - set_network
      - start_backend

#### qia/lib/python3.10/site-packages/netqasm/runtime/settings.py
- Constants: SIMULATOR_ENV
- Functions: _default_simulator, get_is_using_hardware, get_simulator, set_is_using_hardware, set_simulator
- Classes:
  - Flavour
    - Methods: (none detected)
  - Formalism
    - Methods: (none detected)
  - Simulator
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/build_epr.py
- Constants: SER_CREATE_IDX_ATOMIC, SER_CREATE_IDX_CONSECUTIVE, SER_CREATE_IDX_MAX_TIME, SER_CREATE_IDX_MINIMUM_FIDELITY, SER_CREATE_IDX_NUMBER, SER_CREATE_IDX_PRIORITY, SER_CREATE_IDX_PROBABILITY_DIST_LOCAL1, SER_CREATE_IDX_PROBABILITY_DIST_REMOTE1, SER_CREATE_IDX_PROBABLIITY_DIST_LOCAL2, SER_CREATE_IDX_PROBABLIITY_DIST_REMOTE2, SER_CREATE_IDX_RANDOM_BASIS_LOCAL, SER_CREATE_IDX_RANDOM_BASIS_REMOTE, SER_CREATE_IDX_ROTATION_X_LOCAL1, SER_CREATE_IDX_ROTATION_X_LOCAL2, SER_CREATE_IDX_ROTATION_X_REMOTE1, SER_CREATE_IDX_ROTATION_X_REMOTE2, SER_CREATE_IDX_ROTATION_Y_LOCAL, SER_CREATE_IDX_ROTATION_Y_REMOTE, SER_CREATE_IDX_TIME_UNIT, SER_CREATE_IDX_TYPE, SER_CREATE_LEN, SER_RESPONSE_KEEP_IDX_BELL_STATE, SER_RESPONSE_KEEP_IDX_CREATE_ID, SER_RESPONSE_KEEP_IDX_DIRECTONIALITY_FLAG, SER_RESPONSE_KEEP_IDX_GOODNESS, SER_RESPONSE_KEEP_IDX_GOODNESS_TIME, SER_RESPONSE_KEEP_IDX_LOGICAL_QUBIT_ID, SER_RESPONSE_KEEP_IDX_PURPOSE_ID, SER_RESPONSE_KEEP_IDX_REMOTE_NODE_ID, SER_RESPONSE_KEEP_IDX_SEQUENCE_NUMBER, SER_RESPONSE_KEEP_IDX_TYPE, SER_RESPONSE_KEEP_LEN, SER_RESPONSE_MEASURE_IDX_BELL_STATE, SER_RESPONSE_MEASURE_IDX_CREATE_ID, SER_RESPONSE_MEASURE_IDX_DIRECTONIALITY_FLAG, SER_RESPONSE_MEASURE_IDX_GOODNESS, SER_RESPONSE_MEASURE_IDX_MEASUREMENT_BASIS, SER_RESPONSE_MEASURE_IDX_MEASUREMENT_OUTCOME, SER_RESPONSE_MEASURE_IDX_PURPOSE_ID, SER_RESPONSE_MEASURE_IDX_REMOTE_NODE_ID, SER_RESPONSE_MEASURE_IDX_SEQUENCE_NUMBER, SER_RESPONSE_MEASURE_IDX_TYPE, SER_RESPONSE_MEASURE_LEN
- Functions: basis_to_rotation, deserialize_epr_keep_results, deserialize_epr_measure_results, rotation_to_basis, serialize_request
- Classes:
  - EntRequestParams
    - Methods: (none detected)
  - EprKeepResult
    - Methods (1):
      - bell_state
  - EprMeasBasis
    - Methods: (none detected)
  - EprMeasureResult
    - Methods (2):
      - bell_state
      - measurement_outcome

#### qia/lib/python3.10/site-packages/netqasm/sdk/build_nv.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NVEprCompiler
    - Methods (1):
      - get_max_time_for_fidelity

#### qia/lib/python3.10/site-packages/netqasm/sdk/build_types.py
- Constants: (none)
- Functions: (none)
- Classes:
  - GenericHardwareConfig
    - Methods (1):
      - __init__
  - HardwareConfig
    - Methods (4):
      - __init__
      - comm_qubit_count
      - mem_qubit_count
      - qubit_count
  - NVHardwareConfig
    - Methods (1):
      - __init__

#### qia/lib/python3.10/site-packages/netqasm/sdk/builder.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Builder
    - Methods (97):
      - __init__
      - _activate_register
      - _add_wait_for_ent_info_cmd
      - _alloc_ent_results_array
      - _alloc_epr_create_args
      - _assert_epr_args
      - _build_cmds_allocated_arrays
      - _build_cmds_breakpoint
      - _build_cmds_condition
      - _build_cmds_epr_create_keep
      - _build_cmds_epr_create_measure
      - _build_cmds_epr_create_rsp
      - _build_cmds_epr_keep_corrections
      - _build_cmds_epr_keep_corrections_single_pair
      - _build_cmds_epr_recv_keep
      - _build_cmds_epr_recv_measure
      - _build_cmds_epr_recv_rsp
      - _build_cmds_free_up_qubit_location
      - _build_cmds_if_stmt
      - _build_cmds_init_array
      - _build_cmds_init_qubit
      - _build_cmds_loop
      - _build_cmds_loop_body
      - _build_cmds_loop_until
      - _build_cmds_measure
      - _build_cmds_move_qubit
      - _build_cmds_new_qubit
      - _build_cmds_post_epr
      - _build_cmds_qfree
      - _build_cmds_return_array
      - _build_cmds_return_registers
      - _build_cmds_set_register_value
      - _build_cmds_single_qubit
      - _build_cmds_single_qubit_rotation
      - _build_cmds_two_qubit
      - _build_cmds_undefine_array
      - _build_cmds_wait_move_epr_to_mem
      - _check_epr_args
      - _create_ent_info_k_slices
      - _create_ent_qubits
      - _foreach_context_enter
      - _foreach_context_exit
      - _get_branch_commands
      - _get_branch_commands_single_operand
      - _get_condition_operand
      - _get_qubit_futures
      - _get_qubit_register
      - _get_raw_bell_state
      - _log_subroutine
      - _loop_get_entry_commands
      - _loop_get_exit_commands
      - _loop_get_register
      - _loop_until_context_enter
      - _loop_until_context_exit
      - _loop_until_get_break_commands
      - _loop_until_get_entry_commands
      - _loop_until_get_exit_commands
      - _post_epr_context
      - _pre_epr_context
      - _reset
      - alloc_array
      - app_id
      - app_id
      - committed_subroutines
      - if_context_enter
      - if_context_exit
      - inactivate_qubits
      - new_qubit_id
      - new_register
      - sdk_create_epr_context
      - sdk_create_epr_keep
      - sdk_create_epr_measure
      - sdk_create_epr_rsp
      - sdk_epr_keep
      - sdk_epr_measure
      - sdk_epr_rsp_create
      - sdk_epr_rsp_recv
      - sdk_if_eq
      - sdk_if_ez
      - sdk_if_ge
      - sdk_if_lt
      - sdk_if_ne
      - sdk_if_nz
      - sdk_loop_body
      - sdk_loop_context
      - sdk_new_foreach_context
      - sdk_new_if_context
      - sdk_new_loop_until_context
      - sdk_recv_epr_keep
      - sdk_recv_epr_measure
      - sdk_recv_epr_rsp
      - sdk_try_context
      - subrt_add_pending_command
      - subrt_add_pending_commands
      - subrt_compile_subroutine
      - subrt_pop_all_pending_commands
      - subrt_pop_pending_subroutine
  - LabelManager
    - Methods (2):
      - __init__
      - new_label
  - SdkForEachContext
    - Methods (3):
      - __enter__
      - __exit__
      - __init__
  - SdkIfContext
    - Methods (3):
      - __enter__
      - __exit__
      - __init__
  - SdkLoopUntilContext
    - Methods (8):
      - __init__
      - cleanup_code
      - exit_condition
      - loop_register
      - max_iterations
      - set_cleanup_code
      - set_exit_condition
      - set_loop_register

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/broadcast_channel.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BroadcastChannel
    - Methods (5):
      - __init__
      - conn_lost_callback
      - recv
      - recv_callback
      - send
  - BroadcastChannelBySockets
    - Methods (4):
      - __init__
      - _socket_class
      - recv
      - send

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/message.py
- Constants: (none)
- Functions: (none)
- Classes:
  - StructuredMessage
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/socket.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Socket
    - Methods (9):
      - __init__
      - conn_lost_callback
      - recv
      - recv_callback
      - recv_silent
      - recv_structured
      - send
      - send_silent
      - send_structured

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/broadcast_channel.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ThreadBroadcastChannel
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/socket.py
- Constants: (none)
- Functions: log_recv, log_recv_structured, log_send, log_send_structured, trim_msg
- Classes:
  - StorageThreadSocket
    - Methods (2):
      - __init__
      - recv_callback
  - ThreadSocket
    - Methods (18):
      - __del__
      - __init__
      - app_name
      - connected
      - get_comm_logger
      - id
      - key
      - recv
      - recv_silent
      - recv_structured
      - remote_app_name
      - remote_key
      - send
      - send_silent
      - send_structured
      - use_callbacks
      - use_callbacks
      - wait

#### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/socket_hub.py
- Constants: (none)
- Functions: reset_socket_hub
- Classes:
  - _SocketHub
    - Methods (8):
      - __init__
      - _add_callbacks
      - _wait_for_remote
      - connect
      - disconnect
      - is_connected
      - recv
      - send

#### qia/lib/python3.10/site-packages/netqasm/sdk/config.py
- Constants: (none)
- Functions: (none)
- Classes:
  - LogConfig
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/connection.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BaseNetQASMConnection
    - Methods (45):
      - __enter__
      - __exit__
      - __init__
      - __str__
      - _commit_message
      - _commit_serialized_message
      - _get_network_info
      - _get_new_app_id
      - _init_new_app
      - _pop_app_id
      - _save_log_subroutines
      - _setup_epr_socket
      - _setup_epr_sockets
      - _signal_stop
      - active_qubits
      - app_id
      - app_id
      - app_name
      - block
      - builder
      - clear
      - close
      - commit_protosubroutine
      - commit_subroutine
      - compile
      - flush
      - get_app_ids
      - get_app_names
      - if_eq
      - if_ez
      - if_ge
      - if_lt
      - if_ne
      - if_nz
      - insert_breakpoint
      - loop
      - loop_body
      - loop_until
      - network_info
      - new_array
      - node_name
      - shared_memory
      - test_preparation
      - tomography
      - try_until_success
  - DebugConnection
    - Methods (4):
      - __init__
      - _commit_serialized_message
      - _get_network_info
      - shared_memory
  - DebugNetworkInfo
    - Methods (4):
      - _get_node_id
      - _get_node_name
      - get_node_id_for_app
      - get_node_name_for_app

#### qia/lib/python3.10/site-packages/netqasm/sdk/constraint.py
- Constants: (none)
- Functions: (none)
- Classes:
  - SdkConstraint
    - Methods: (none detected)
  - ValueAtMostConstraint
    - Methods (3):
      - __init__
      - future
      - value

#### qia/lib/python3.10/site-packages/netqasm/sdk/epr_socket.py
- Constants: (none)
- Functions: (none)
- Classes:
  - EPRSocket
    - Methods (22):
      - __init__
      - _get_node_id
      - conn
      - conn
      - create
      - create_context
      - create_keep
      - create_keep_with_info
      - create_measure
      - create_rsp
      - epr_socket_id
      - min_fidelity
      - recv
      - recv_context
      - recv_keep
      - recv_keep_with_info
      - recv_measure
      - recv_rsp
      - recv_rsp_with_info
      - remote_app_name
      - remote_epr_socket_id
      - remote_node_id

#### qia/lib/python3.10/site-packages/netqasm/sdk/external.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/futures.py
- Constants: (none)
- Functions: as_int_when_value
- Classes:
  - Array
    - Methods (11):
      - __getitem__
      - __init__
      - __len__
      - address
      - builder
      - enumerate
      - foreach
      - get_future_index
      - get_future_slice
      - lineno
      - undefine
  - BaseFuture
    - Methods (15):
      - __init__
      - __new__
      - __repr__
      - _try_get_value
      - add
      - builder
      - if_eq
      - if_ez
      - if_ge
      - if_lt
      - if_ne
      - if_nz
      - subrt_result
      - subrt_result
      - value
  - Future
    - Methods (9):
      - __init__
      - __new__
      - __str__
      - _get_access_commands
      - _get_store_commands
      - _try_get_value
      - add
      - get_address_entry
      - get_load_commands
  - NoValueError
    - Methods: (none detected)
  - NonConstantIndexError
    - Methods: (none detected)
  - RegFuture
    - Methods (6):
      - __init__
      - __str__
      - _try_get_value
      - add
      - reg
      - reg

#### qia/lib/python3.10/site-packages/netqasm/sdk/memmgr.py
- Constants: (none)
- Functions: (none)
- Classes:
  - MemoryManager
    - Methods (24):
      - __init__
      - activate_qubit
      - add_active_register
      - add_array_to_return
      - add_register_to_return
      - deactivate_qubit
      - get_active_qubits
      - get_arrays_to_return
      - get_inactive_register
      - get_new_array_address
      - get_new_meas_outcome_register
      - get_new_qubit_address
      - get_registers_to_return
      - inactivate_qubits
      - is_qubit_active
      - is_qubit_id_used
      - is_register_active
      - meas_register_set_unused
      - meas_register_set_used
      - remove_active_register
      - reset
      - reset_arrays_to_return
      - reset_registers_to_return
      - reset_used_meas_registers

#### qia/lib/python3.10/site-packages/netqasm/sdk/network.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetworkInfo
    - Methods (4):
      - _get_node_id
      - _get_node_name
      - get_node_id_for_app
      - get_node_name_for_app

#### qia/lib/python3.10/site-packages/netqasm/sdk/progress_bar.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ProgressBar
    - Methods (4):
      - __init__
      - close
      - increase
      - update

#### qia/lib/python3.10/site-packages/netqasm/sdk/qubit.py
- Constants: (none)
- Functions: (none)
- Classes:
  - FutureQubit
    - Methods (3):
      - __init__
      - entanglement_info
      - remote_entangled_node
  - Qubit
    - Methods (28):
      - H
      - K
      - S
      - T
      - X
      - Y
      - Z
      - __init__
      - __str__
      - _activate
      - _deactivate
      - active
      - active
      - assert_active
      - builder
      - cnot
      - connection
      - cphase
      - entanglement_info
      - free
      - measure
      - qubit_id
      - qubit_id
      - remote_entangled_node
      - reset
      - rot_X
      - rot_Y
      - rot_Z
  - QubitMeasureBasis
    - Methods: (none detected)
  - QubitNotActiveError
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/shared_memory.py
- Constants: (none)
- Functions: _assert_within_width, setup_registers
- Classes:
  - Arrays
    - Methods (11):
      - __getitem__
      - __init__
      - __setitem__
      - __str__
      - _assert_list
      - _extract_key
      - _get_active_values
      - _get_array
      - _set_array
      - has_array
      - init_new_array
  - RegisterGroup
    - Methods (7):
      - __getitem__
      - __init__
      - __len__
      - __setitem__
      - __str__
      - _assert_within_length
      - _get_active_values
  - SharedMemory
    - Methods (9):
      - __getitem__
      - __init__
      - _get_active_values
      - _get_array
      - get_array_part
      - get_register
      - init_new_array
      - set_array_part
      - set_register
  - SharedMemoryManager
    - Methods (3):
      - create_shared_memory
      - get_shared_memory
      - reset_memories

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/gates.py
- Constants: (none)
- Functions: t_inverse, toffoli_gate
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/measurements.py
- Constants: (none)
- Functions: parity_meas
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/multi_node.py
- Constants: (none)
- Functions: create_ghz
- Classes:
  - _Role
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/sim_states.py
- Constants: (none)
- Functions: get_fidelity, qubit_from, to_dm
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/state_prep.py
- Constants: (none)
- Functions: get_angle_spec_from_float, set_qubit_state
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/sdk/transpile.py
- Constants: (none)
- Functions: get_hardware_num_denom
- Classes:
  - NVSubroutineTranspiler
    - Methods (15):
      - __init__
      - _handle_single_qubit_gate
      - _handle_two_qubit_gate
      - _map_cnot_carbon_carbon
      - _map_cnot_carbon_electron
      - _map_cnot_electron_carbon
      - _map_cphase_carbon_carbon
      - _map_cphase_electron_carbon
      - _map_single_gate
      - _move_carbon_electron
      - _move_electron_carbon
      - get_reg_value
      - get_unused_register
      - swap
      - transpile
  - REIDSSubroutineTranspiler
    - Methods (2):
      - __init__
      - transpile
  - SubroutineTranspiler
    - Methods (2):
      - __init__
      - transpile

#### qia/lib/python3.10/site-packages/netqasm/typedefs.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/typing.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/error.py
- Constants: (none)
- Functions: (none)
- Classes:
  - NetQASMInstrError
    - Methods: (none detected)
  - NetQASMSyntaxError
    - Methods: (none detected)
  - NoCircuitRuleError
    - Methods: (none detected)
  - NotAllocatedError
    - Methods: (none detected)
  - SubroutineAbortedError
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netqasm/util/log.py
- Constants: (none)
- Functions: (none)
- Classes:
  - HostLine
    - Methods (2):
      - __init__
      - __str__
  - LineTracker
    - Methods (3):
      - __init__
      - _get_file_from_frame
      - get_line

#### qia/lib/python3.10/site-packages/netqasm/util/quantum_gates.py
- Constants: CNOT, CPHASE, H, K, PAULIS, S, STATIC_QUBIT_GATE_TO_MATRIX, T, X, Y, Z
- Functions: are_matrices_equal, gate_to_matrix, get_controlled_rotation_matrix, get_rotation_matrix
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/states.py
- Constants: (none)
- Functions: bloch_sphere_rep
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/string.py
- Constants: ALPHA_ALL, ALPHA_CAPITAL, ALPHA_LOWER, ALPHA_NUM, NUM
- Functions: _assert_valid_brackets, _assert_valid_seperator, group_by_word, is_float, is_number, is_variable_name, rspaces
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/thread.py
- Constants: (none)
- Functions: as_completed
- Classes: (none)

#### qia/lib/python3.10/site-packages/netqasm/util/yaml.py
- Constants: (none)
- Functions: dump_yaml, load_yaml
- Classes: (none)

### C.4 Inventory: netsquid_magic

- Total python files indexed: 13

#### qia/lib/python3.10/site-packages/netsquid_magic/__init__.py
- Constants: (none)
- Functions: (none)
- Classes: (none)

#### qia/lib/python3.10/site-packages/netsquid_magic/abstract_heralded_connection.py
- Constants: (none)
- Functions: (none)
- Classes:
  - AbstractHeraldedConnection
    - Methods (10):
      - __init__
      - attenuation_coefficient
      - efficiency
      - fidelity
      - length
      - num_modes
      - prob_max_mixed
      - speed_of_light
      - speed_of_light_delay
      - success_probability
  - AbstractHeraldedMagic
    - Methods (1):
      - __init__
  - AbstractHeraldedModelParameters
    - Methods (1):
      - verify

#### qia/lib/python3.10/site-packages/netsquid_magic/egp.py
- Constants: (none)
- Functions: (none)
- Classes:
  - EgpProtocol
    - Methods (8):
      - __init__
      - _handle_error
      - create_and_keep
      - measure_directly
      - receive
      - remote_state_preparation
      - run
      - stop_receive

#### qia/lib/python3.10/site-packages/netsquid_magic/entanglement_magic.py
- Constants: (none)
- Functions: (none)
- Classes:
  - EntanglementMagic
    - Methods (18):
      - __init__
      - _are_all_parallel_attempts_taken
      - _can_execute_request
      - _execute_as_many_requests_from_queue_as_possible
      - _execute_request
      - _get_subprot_by_remote_node_name
      - _get_subprot_name_by_remote_node_name
      - _num_running_entanglement_attempts
      - _try_to_execute_request_and_return_whether_executing
      - abort
      - check_queue
      - generate_entanglement
      - get_in_use
      - is_connected
      - run
      - set_in_use
      - start
      - stop

#### qia/lib/python3.10/site-packages/netsquid_magic/link_layer.py
- Constants: (none)
- Functions: (none)
- Classes:
  - LinkLayerService
    - Methods: (none detected)
  - MagicLinkLayerProtocol
    - Methods (36):
      - __init__
      - _add_to_queue_item_value
      - _add_to_request_queue
      - _add_to_requests_in_process
      - _assert_has_q_processor
      - _construct_rotation_operator
      - _decrement_pairs_in_process
      - _decrement_pairs_left
      - _defer_handle_next
      - _find_request_from_delivery
      - _get_bell_state
      - _get_create_id
      - _get_next_sequence_number
      - _get_unused_memory_positions
      - _handle_create_request
      - _handle_label_delivery
      - _handle_next
      - _handle_recv_request
      - _handle_state_delivery
      - _handle_stop_recv_request
      - _increment_pairs_in_process
      - _is_valid_request
      - _measure_qubit
      - _peek_from_request_queue
      - _pop_from_request_queue
      - _pop_from_requests_in_process
      - _reserve_memory_position
      - _sample_basis_choice
      - _set_pairs_in_process_to_zero
      - abort_request
      - capacity
      - close
      - num_requests_in_queue
      - open
      - put_from
      - stop
  - MagicLinkLayerProtocolWithSignaling
    - Methods (3):
      - __init__
      - magic_distributor
      - react_to
  - QueueItem
    - Methods: (none detected)
  - SingleClickTranslationUnit
    - Methods (1):
      - request_to_parameters
  - TranslationUnit
    - Methods (2):
      - __call__
      - request_to_parameters

#### qia/lib/python3.10/site-packages/netsquid_magic/long_distance_interface.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ILongDistanceInterface
    - Methods (2):
      - operate
      - probability_success

#### qia/lib/python3.10/site-packages/netsquid_magic/magic.py
- Constants: (none)
- Functions: (none)
- Classes:
  - MagicProtocol
    - Methods (10):
      - __init__
      - _custom_start
      - _custom_stop
      - is_connected
      - nodes
      - put_from
      - react_to
      - service_interfaces
      - start
      - stop

#### qia/lib/python3.10/site-packages/netsquid_magic/magic_distributor.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BitflipMagicDistributor
    - Methods (1):
      - __init__
  - Delivery
    - Methods (1):
      - in_process
  - DepolariseMagicDistributor
    - Methods (1):
      - __init__
  - DepolariseWithFailureMagicDistributor
    - Methods (2):
      - __init__
      - get_bell_state
  - DoubleClickMagicDistributor
    - Methods (2):
      - __init__
      - get_bell_state
  - HeraldedConnectionMagicDistributor
    - Methods (2):
      - __init__
      - _calculate_cycle_times_and_delays
  - MagicDistributor
    - Methods (37):
      - __init__
      - _add_pair_delivery
      - _apply_noise
      - _archive_delivery
      - _create_entangled_qubits
      - _free_delivery_memory_positions
      - _get_delivery_from_event
      - _get_delivery_from_node_delivery
      - _get_node_from_id
      - _get_total_state_delay
      - _handle_label_delivery
      - _handle_node_delivery
      - _handle_state_delivery
      - _matches_existing_request
      - _pop_delivery
      - _pop_node_delivery
      - _schedule_label_delivery_event
      - _schedule_state_delivery_events
      - abort_all_delivery
      - abort_delivery
      - add_callback
      - add_delivery
      - add_pair_request
      - clear_all_callbacks
      - get_bell_state
      - get_label
      - get_qmemories_from_nodes
      - long_distance_interface
      - long_distance_interface
      - merge_magic_distributor
      - nodes
      - peek_delivery
      - peek_node_delivery
      - reset
      - set_skip_rounds
      - start
      - stop
  - NodeDelivery
    - Methods: (none detected)
  - PerfectStateMagicDistributor
    - Methods (1):
      - __init__
  - SingleClickMagicDistributor
    - Methods (3):
      - __init__
      - add_delivery
      - get_bell_state
  - _EntGenRequest
    - Methods (4):
      - __eq__
      - __init__
      - __neq__
      - matches

#### qia/lib/python3.10/site-packages/netsquid_magic/model_parameters.py
- Constants: (none)
- Functions: (none)
- Classes:
  - BitFlipModelParameters
    - Methods (1):
      - verify
  - DepolariseModelParameters
    - Methods (1):
      - verify
  - DoubleClickModelParameters
    - Methods (1):
      - verify
  - HeraldedModelParameters
    - Methods (3):
      - absorb_collection_efficiency_in_p_init
      - verify
      - verify_only_heralded_params
  - IModelParameters
    - Methods (9):
      - _error_msg_base
      - _raise_error_if_none_not_allowed
      - _verify_is_real_number
      - verify
      - verify_between_0_and_1
      - verify_equal
      - verify_is_real_number
      - verify_is_type
      - verify_not_negative_value
  - PerfectModelParameters
    - Methods (1):
      - verify
  - SingleClickModelParameters
    - Methods (1):
      - verify

#### qia/lib/python3.10/site-packages/netsquid_magic/qlink.py
- Constants: (none)
- Functions: (none)
- Classes:
  - IQLink
    - Methods (3):
      - close
      - num_requests_in_queue
      - open
  - MagicQLink
    - Methods (14):
      - __init__
      - _add_pair_delivery
      - _delivery_mem_position_identical
      - _handle_label_delivery
      - _handle_state_delivery
      - _matches_existing_request
      - abort
      - add_callback
      - add_pair_request
      - close
      - get_bell_state
      - num_requests_in_queue
      - open
      - peek_node_delivery
  - _EntDeliveryRequest
    - Methods: (none detected)

#### qia/lib/python3.10/site-packages/netsquid_magic/services.py
- Constants: (none)
- Functions: (none)
- Classes:
  - ServiceInterface
    - Methods (23):
      - __init__
      - _custom_check_reaction
      - _custom_check_request
      - _custom_start
      - _custom_stop
      - _magic_put
      - _magic_start
      - _magic_stop
      - _put
      - _valid_reaction
      - _valid_request
      - add_magic_protocol
      - add_protocol
      - add_reaction_handler
      - is_magic
      - is_running
      - node
      - put
      - react
      - reaction_types
      - request_types
      - start
      - stop
  - ServiceProtocol
    - Methods (3):
      - put
      - react
      - service_interface

#### qia/lib/python3.10/site-packages/netsquid_magic/sleeper.py
- Constants: (none)
- Functions: (none)
- Classes:
  - Sleeper
    - Methods (2):
      - __init__
      - sleep

#### qia/lib/python3.10/site-packages/netsquid_magic/state_delivery_sampler.py
- Constants: (none)
- Functions: success_prob_and_fidelity_from_heralded_state_delivery_sampler_factory
- Classes:
  - BitflipStateSamplerFactory
    - Methods (2):
      - __init__
      - _delivery_bit_flip
  - DepolariseStateSamplerFactory
    - Methods (2):
      - __init__
      - _delivery_func
  - DepolariseWithFailureStateSamplerFactory
    - Methods (2):
      - __init__
      - _delivery_func
  - DoubleClickDeliverySamplerFactory
    - Methods (5):
      - __init__
      - _calculate_density_matrix
      - _calculate_probabilities
      - _calculate_success_probability
      - _func_delivery
  - HeraldedStateDeliverySamplerFactory
    - Methods (2):
      - __init__
      - create_state_delivery_sampler
  - IStateDeliverySamplerFactory
    - Methods (2):
      - __init__
      - create_state_delivery_sampler
  - PerfectStateSamplerFactory
    - Methods (2):
      - __init__
      - _get_perfect_state_sampler
  - SingleClickDeliverySamplerFactory
    - Methods (4):
      - __init__
      - _compute_detection_probabilities
      - _compute_total_detection_probability
      - _get_single_click_state_sampler
  - StateDeliverySampler
    - Methods (3):
      - __init__
      - _assert_positive_time
      - sample

## Appendix D. Symbol Index (by module)

This appendix re-lists public symbols per module with a normalized “surface signature” to help refactoring teams locate responsibilities quickly.

### qia-challenge-2025/ehok/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/analysis/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/analysis/metrics.py
- Root: ehok
- Public constants: (none)
- Public functions: compute_execution_metrics
- Public classes: (none)

### qia-challenge-2025/ehok/configs/generate_ldpc.py
- Root: ehok
- Public constants: (none)
- Public functions: generate_all, main
- Public classes: (none)

### qia-challenge-2025/ehok/core/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/core/config.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: PrivacyAmplificationConfig, ProtocolConfig, QuantumConfig, ReconciliationConfig, SecurityConfig
- Class PrivacyAmplificationConfig:
  - (no public methods detected)
- Class ProtocolConfig:
  - baseline
  - copy_with
  - to_dict
- Class QuantumConfig:
  - (no public methods detected)
- Class ReconciliationConfig:
  - (no public methods detected)
- Class SecurityConfig:
  - (no public methods detected)

### qia-challenge-2025/ehok/core/constants.py
- Root: ehok
- Public constants: BATCH_SIZE, CLASSICAL_TIMEOUT_SEC, LDPC_AVAILABLE_RATES, LDPC_BP_THRESHOLD, LDPC_CODE_RATE, LDPC_CODE_RATES, LDPC_CRITICAL_EFFICIENCY, LDPC_DEFAULT_RATE, LDPC_DEGREE_DISTRIBUTIONS, LDPC_DEGREE_DISTRIBUTIONS_PATH, LDPC_FRAME_SIZE, LDPC_F_CRIT, LDPC_HASH_BITS, LDPC_MATRIX_FILE_PATTERN, LDPC_MAX_ITERATIONS, LDPC_QBER_WINDOW_SIZE, LDPC_TEST_FRAME_SIZES, LDPC_TEST_MATRIX_SUBDIR, LINK_FIDELITY_MIN, LOG_LEVEL, LOG_TO_FILE, MIN_TEST_SET_SIZE, PA_SECURITY_MARGIN, PEG_DEFAULT_SEED, PEG_MAX_TREE_DEPTH, QBER_THRESHOLD, TARGET_EPSILON_SEC, TEST_SET_FRACTION, TOTAL_EPR_PAIRS
- Public functions: 
- Public classes: (none)

### qia-challenge-2025/ehok/core/data_structures.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: ExecutionMetrics, LDPCBlockResult, LDPCMatrixPool, LDPCReconciliationResult, MeasurementRecord, ObliviousKey, ProtocolResult
- Class ExecutionMetrics:
  - (no public methods detected)
- Class LDPCBlockResult:
  - (no public methods detected)
- Class LDPCMatrixPool:
  - (no public methods detected)
- Class LDPCReconciliationResult:
  - (no public methods detected)
- Class MeasurementRecord:
  - (no public methods detected)
- Class ObliviousKey:
  - (no public methods detected)
- Class ProtocolResult:
  - (no public methods detected)

### qia-challenge-2025/ehok/core/exceptions.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: CommitmentVerificationError, EHOKException, MatrixSynchronizationError, ProtocolError, QBERTooHighError, ReconciliationFailedError, SecurityException
- Class CommitmentVerificationError:
  - (no public methods detected)
- Class EHOKException:
  - (no public methods detected)
- Class MatrixSynchronizationError:
  - (no public methods detected)
- Class ProtocolError:
  - (no public methods detected)
- Class QBERTooHighError:
  - (no public methods detected)
- Class ReconciliationFailedError:
  - (no public methods detected)
- Class SecurityException:
  - (no public methods detected)

### qia-challenge-2025/ehok/core/sifting.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: SiftingManager
- Class SiftingManager:
  - check_qber_abort
  - estimate_qber
  - identify_matching_bases
  - select_test_set

### qia-challenge-2025/ehok/examples/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/examples/debug_qber.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/examples/run_baseline.py
- Root: ehok
- Public constants: (none)
- Public functions: print_results, run_ehok_baseline
- Public classes: (none)

### qia-challenge-2025/ehok/examples/test_noise.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: AliceTestProgram, BobTestProgram
- Class AliceTestProgram:
  - meta
  - run
- Class BobTestProgram:
  - meta
  - run

### qia-challenge-2025/ehok/implementations/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/commitment/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/commitment/merkle_commitment.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: MerkleCommitment, MerkleTree
- Class MerkleCommitment:
  - commit
  - open_subset
  - verify
- Class MerkleTree:
  - get_proof
  - root
  - verify_proof

### qia-challenge-2025/ehok/implementations/commitment/sha256_commitment.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: SHA256Commitment
- Class SHA256Commitment:
  - commit
  - open_subset
  - verify

### qia-challenge-2025/ehok/implementations/factories.py
- Root: ehok
- Public constants: (none)
- Public functions: build_commitment_scheme, build_noise_estimator, build_privacy_amplifier, build_reconciliator, build_sampling_strategy
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/noise/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/noise/simple_noise_estimator.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: SimpleNoiseEstimator
- Class SimpleNoiseEstimator:
  - estimate_leakage

### qia-challenge-2025/ehok/implementations/privacy_amplification/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/privacy_amplification/finite_key.py
- Root: ehok
- Public constants: DEFAULT_EPSILON_COR, DEFAULT_EPSILON_SEC
- Public functions: binary_entropy, compute_blind_reconciliation_leakage, compute_final_length_blind_mode, compute_final_length_finite_key, compute_statistical_fluctuation, estimate_qber_from_reconciliation
- Public classes: FiniteKeyParams
- Class FiniteKeyParams:
  - (no public methods detected)

### qia-challenge-2025/ehok/implementations/privacy_amplification/toeplitz_amplifier.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: ToeplitzAmplifier
- Class ToeplitzAmplifier:
  - compress
  - compute_final_length
  - compute_final_length_asymptotic
  - compute_final_length_blind
  - generate_hash_seed

### qia-challenge-2025/ehok/implementations/reconciliation/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_bp_decoder.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: LDPCBeliefPropagation
- Class LDPCBeliefPropagation:
  - decode

### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_matrix_manager.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: LDPCMatrixManager
- Class LDPCMatrixManager:
  - checksum
  - frame_size
  - from_directory
  - get_matrix
  - rates
  - verify_checksum

### qia-challenge-2025/ehok/implementations/reconciliation/ldpc_reconciliator.py
- Root: ehok
- Public constants: (none)
- Public functions: 
- Public classes: LDPCReconciliator
- Class LDPCReconciliator:
  - aggregate_results
  - compute_adaptive_iterations
  - compute_shortening
  - compute_syndrome_block
  - estimate_leakage_block
  - reconcile_block
  - select_rate
  - syndrome_guided_llr_init
  - verify_block

### qia-challenge-2025/ehok/implementations/reconciliation/peg_generator.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: DegreeDistribution, PEGMatrixGenerator
- Class DegreeDistribution:
  - (no public methods detected)
- Class PEGMatrixGenerator:
  - generate

### qia-challenge-2025/ehok/implementations/reconciliation/polynomial_hash.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: PolynomialHashVerifier
- Class PolynomialHashVerifier:
  - compute_hash
  - hash_and_seed
  - hash_length_bits
  - verify

### qia-challenge-2025/ehok/implementations/reconciliation/qber_estimator.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: IntegratedQBEREstimator
- Class IntegratedQBEREstimator:
  - estimate
  - update_rolling
  - window

### qia-challenge-2025/ehok/implementations/sampling/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/implementations/sampling/random_sampling.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: RandomSamplingStrategy
- Class RandomSamplingStrategy:
  - select_sets

### qia-challenge-2025/ehok/interfaces/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/interfaces/commitment.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: ICommitmentScheme
- Class ICommitmentScheme:
  - commit
  - open_subset
  - verify

### qia-challenge-2025/ehok/interfaces/noise_estimator.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: INoiseEstimator
- Class INoiseEstimator:
  - estimate_leakage

### qia-challenge-2025/ehok/interfaces/privacy_amplification.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: IPrivacyAmplifier
- Class IPrivacyAmplifier:
  - compress
  - compute_final_length
  - generate_hash_seed

### qia-challenge-2025/ehok/interfaces/reconciliation.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: IReconciliator
- Class IReconciliator:
  - compute_shortening
  - compute_syndrome_block
  - estimate_leakage_block
  - reconcile_block
  - select_rate
  - verify_block

### qia-challenge-2025/ehok/interfaces/sampling_strategy.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: ISamplingStrategy
- Class ISamplingStrategy:
  - select_sets

### qia-challenge-2025/ehok/protocols/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/protocols/alice.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: AliceBaselineEHOK
- Class AliceBaselineEHOK:
  - (no public methods detected)

### qia-challenge-2025/ehok/protocols/base.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: EHOKRole
- Class EHOKRole:
  - meta
  - run

### qia-challenge-2025/ehok/protocols/bob.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: BobBaselineEHOK
- Class BobBaselineEHOK:
  - (no public methods detected)

### qia-challenge-2025/ehok/quantum/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/quantum/basis_selection.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: BasisSelector
- Class BasisSelector:
  - basis_to_string
  - generate_bases

### qia-challenge-2025/ehok/quantum/batching_manager.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: BatchResult, BatchingManager, EPRGenerator
- Class BatchResult:
  - (no public methods detected)
- Class BatchingManager:
  - compute_batch_sizes
- Class EPRGenerator:
  - extract_batch_results
  - generate_batch_alice
  - generate_batch_bob
  - measure_batch

### qia-challenge-2025/ehok/quantum/measurement.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: MeasurementBuffer
- Class MeasurementBuffer:
  - add_batch
  - clear
  - get_bases
  - get_outcomes

### qia-challenge-2025/ehok/quantum/runner.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: QuantumPhaseResult, QuantumPhaseRunner
- Class QuantumPhaseResult:
  - (no public methods detected)
- Class QuantumPhaseRunner:
  - connection
  - csocket
  - run

### qia-challenge-2025/ehok/tests/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/tests/conftest.py
- Root: ehok
- Public constants: TEST_FRAME_SIZE, TEST_LDPC_RATES
- Public functions: baseline_config, fast_test_config, pytest_addoption, rng, sample_sifted_key, sample_sifted_key_pair, test_ldpc_matrix_dir, test_ldpc_matrix_pool, test_matrix_manager, test_reconciliator
- Public classes: (none)

### qia-challenge-2025/ehok/tests/test_commitment.py
- Root: ehok
- Public constants: (none)
- Public functions: test_benchmark_commitments
- Public classes: TestMerkleCommitment, TestSHA256Commitment
- Class TestMerkleCommitment:
  - setup_method
  - test_binding_property
  - test_commit_verify_correctness
  - test_subset_opening
  - test_tampered_subset_rejected
- Class TestSHA256Commitment:
  - setup_method
  - test_binding_property
  - test_commit_verify_correctness
  - test_subset_opening
  - test_tampered_subset_rejected

### qia-challenge-2025/ehok/tests/test_foundation.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: TestAbstractInterfaces, TestConstants, TestDataStructures, TestDocstrings, TestExceptionHierarchy, TestLogging, TestPhase0Integration, TestProtocolConfigBinding
- Class TestAbstractInterfaces:
  - test_icommitment_scheme_abstract_methods
  - test_icommitment_scheme_not_instantiable
  - test_iprivacy_amplifier_abstract_methods
  - test_iprivacy_amplifier_not_instantiable
  - test_ireconciliator_abstract_methods
  - test_ireconciliator_not_instantiable
- Class TestConstants:
  - test_ldpc_parameters
  - test_logging_configuration
  - test_network_configuration
  - test_privacy_amplification_parameters
  - test_protocol_parameters
  - test_quantum_parameters
- Class TestDataStructures:
  - test_measurement_record_construction_valid
  - test_measurement_record_invalid_basis
  - test_measurement_record_invalid_outcome
  - test_oblivious_key_construction_valid
  - test_oblivious_key_invalid_key_value_range
  - test_oblivious_key_invalid_key_value_type
  - test_oblivious_key_length_mismatch
  - test_protocol_result_construction_valid
  - test_protocol_result_invariants_fail_on_counts
  - test_protocol_result_invariants_fail_on_qber_and_key_length
  - test_protocol_result_with_abort
- Class TestDocstrings:
  - test_public_classes_have_docstrings
- Class TestExceptionHierarchy:
  - test_all_exceptions_are_catchable_as_ehok_exception
  - test_commitment_verification_error_inherits_from_security_exception
  - test_ehok_exception_inherits_from_exception
  - test_matrix_synchronization_error_inherits_from_protocol_error
  - test_protocol_error_inherits_from_ehok_exception
  - test_qber_too_high_error_custom_attributes
  - test_qber_too_high_error_inherits_from_security_exception
  - test_reconciliation_failed_error_inherits_from_protocol_error
  - test_security_exception_inherits_from_ehok_exception
- Class TestLogging:
  - test_get_logger_returns_logger
  - test_hierarchical_logger_names
  - test_no_print_statements_in_production_code
  - test_setup_ehok_logging_console_only
  - test_setup_ehok_logging_with_file
- Class TestPhase0Integration:
  - test_data_structures_with_exceptions
  - test_logging_with_exceptions
- Class TestProtocolConfigBinding:
  - test_protocols_accept_protocol_config
  - test_protocols_do_not_instantiate_concretes

### qia-challenge-2025/ehok/tests/test_integration.py
- Root: ehok
- Public constants: (none)
- Public functions: test_phase_sequencing_commitment_before_bases, test_synchronization_flush_required
- Public classes: AliceNoFlush, AlicePhaseSeq, BobNoFlush, BobPhaseSeq
- Class AliceNoFlush:
  - (no public methods detected)
- Class AlicePhaseSeq:
  - (no public methods detected)
- Class BobNoFlush:
  - (no public methods detected)
- Class BobPhaseSeq:
  - (no public methods detected)

### qia-challenge-2025/ehok/tests/test_ldpc_integration.py
- Root: ehok
- Public constants: FRAME_SIZE, RATE
- Public functions: test_alice_bob_handshake_success, test_matrix_sync_failure, test_reconciliation_failure_on_hash_mismatch
- Public classes: (none)

### qia-challenge-2025/ehok/tests/test_ldpc_reconciliation.py
- Root: ehok
- Public constants: FRAME_SIZE, RATE
- Public functions: test_constants_load_normalized_distributions, test_degree_distribution_normalization, test_ldpc_reconciliator_block_roundtrip, test_ldpc_reconciliator_hash_and_leakage, test_matrix_manager_checksum_and_access, test_peg_generator_respects_regular_degrees
- Public classes: (none)

### qia-challenge-2025/ehok/tests/test_privacy_amplification.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: TestFiniteKeyFormula, TestPrivacyAmplification
- Class TestFiniteKeyFormula:
  - test_binary_entropy_properties
  - test_pt1_finite_key_formula_correctness
  - test_pt2_finite_key_vs_asymptotic
  - test_pt3_no_fixed_output_length_required
  - test_pt3_small_key_correctly_conservative
  - test_pt4_pa_robustness_qber_range
  - test_pt4_qber_monotonicity
  - test_pt5_blind_leakage_calculation
  - test_pt5_blind_mode_leakage_accounting
  - test_pt6_independence_different_seeds
  - test_pt6_output_key_uniformity
  - test_statistical_fluctuation_scaling
- Class TestPrivacyAmplification:
  - setup_method
  - test_compress_returns_zero_for_m_zero
  - test_compression_correctness_small_example
  - test_compression_execution
  - test_compression_length_calculation
  - test_compute_final_length_security_bound
  - test_hankel_matrix_equivalence
  - test_invalid_seed_length
  - test_output_uniformity
  - test_output_uniformity_chi_square_10bits
  - test_toeplitz_construction
  - test_zero_length_output

### qia-challenge-2025/ehok/tests/test_quantum.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: TestBasisRandomness, TestBatchingManager, TestEPRGeneration
- Class TestBasisRandomness:
  - test_independence
  - test_uniform_distribution
- Class TestBatchingManager:
  - test_batch_size_computation
- Class TestEPRGeneration:
  - test_epr_generation_noisy
  - test_epr_generation_perfect

### qia-challenge-2025/ehok/tests/test_reconciliation_integration.py
- Root: ehok
- Public constants: (none)
- Public functions: test_build_reconciliator_autogeneration_disabled, test_build_reconciliator_requires_existing_matrices, test_ldpc_manager_requires_on_disk_files, test_matrix_checksum_mismatch_raises_during_run, test_protocol_aborts_when_privacy_amplification_yields_zero_length, test_protocol_runs_with_local_ldpc_files_no_deadlock
- Public classes: (none)

### qia-challenge-2025/ehok/tests/test_sifting.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: TestSifting
- Class TestSifting:
  - setup_method
  - test_basis_matching
  - test_qber_abort_threshold
  - test_qber_estimation_exact
  - test_test_set_selection

### qia-challenge-2025/ehok/tests/test_system.py
- Root: ehok
- Public constants: (none)
- Public functions: test_commitment_ordering_security, test_honest_execution_perfect, test_noise_tolerance_5pct, test_qber_abort_threshold
- Public classes: MaliciousAliceProgram
- Class MaliciousAliceProgram:
  - (no public methods detected)

### qia-challenge-2025/ehok/utils/__init__.py
- Root: ehok
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia-challenge-2025/ehok/utils/classical_sockets.py
- Root: ehok
- Public constants: T
- Public functions: (none)
- Public classes: StructuredSocket
- Class StructuredSocket:
  - recv_str
  - recv_structured
  - send_str
  - send_structured

### qia-challenge-2025/ehok/utils/logging.py
- Root: ehok
- Public constants: (none)
- Public functions: get_logger, setup_ehok_logging, setup_script_logging
- Public classes: (none)

### squidasm/squidasm/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/nqasm/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/nqasm/executor/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/nqasm/executor/base.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidExecutor
- Class NetSquidExecutor:
  - execute_subroutine
  - node_id
  - qdevice

### squidasm/squidasm/nqasm/executor/nv.py
- Root: squidasm
- Public constants: NV_NS_INSTR_MAPPING
- Public functions: (none)
- Public classes: NVNetSquidExecutor
- Class NVNetSquidExecutor:
  - (no public methods detected)

### squidasm/squidasm/nqasm/executor/vanilla.py
- Root: squidasm
- Public constants: VANILLA_NS_INSTR_MAPPING
- Public functions: (none)
- Public classes: VanillaNetSquidExecutor
- Class VanillaNetSquidExecutor:
  - (no public methods detected)

### squidasm/squidasm/nqasm/multithread.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidConnection, NetSquidNetworkInfo
- Class NetSquidConnection:
  - block
- Class NetSquidNetworkInfo:
  - get_node_id_for_app
  - get_node_name_for_app

### squidasm/squidasm/nqasm/netstack.py
- Root: squidasm
- Public constants: 
- Public functions: reset_network
- Public classes: NetworkStack, SignalingProtocol
- Class NetworkStack:
  - get_purpose_id
  - put
  - setup_epr_socket
- Class SignalingProtocol:
  - get_circuit_id
  - has_circuit
  - reset
  - setup_circuit

### squidasm/squidasm/nqasm/output.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: InstrLogger
- Class InstrLogger:
  - (no public methods detected)

### squidasm/squidasm/nqasm/qnodeos.py
- Root: squidasm
- Public constants: 
- Public functions: is_waiting_event
- Public classes: SubroutineHandler, Task
- Class SubroutineHandler:
  - get_epr_reaction_handler
  - has_active_apps
  - network_stack
  - network_stack
  - run
  - stop
- Class Task:
  - is_finished
  - is_waiting
  - msg
  - pop_next_event
  - update_next_event

### squidasm/squidasm/nqasm/singlethread/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/nqasm/singlethread/connection.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidConnection
- Class NetSquidConnection:
  - close
  - commit_protosubroutine
  - flush
  - shared_memory

### squidasm/squidasm/nqasm/singlethread/csocket.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidSocket
- Class NetSquidSocket:
  - recv
  - send

### squidasm/squidasm/run/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/run/multithread/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/run/multithread/runtime_mgr.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: SquidAsmRuntimeManager
- Class SquidAsmRuntimeManager:
  - app_node_map
  - backend_log_dir
  - backend_log_dir
  - executors
  - is_running
  - netsquid_formalism
  - netsquid_formalism
  - network
  - nodes
  - party_map
  - qmemories
  - reset_backend
  - run_app
  - set_network
  - start_backend
  - stop_backend
  - subroutine_handlers

### squidasm/squidasm/run/multithread/simulate.py
- Root: squidasm
- Public constants: 
- Public functions: create_nv_cfg, simulate_application
- Public classes: (none)

### squidasm/squidasm/run/singlethread/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/run/singlethread/context.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidContext, NetSquidNetworkInfo
- Class NetSquidContext:
  - add_node
  - add_protocol
  - get_nodes
  - get_protocols
  - set_nodes
  - set_protocols
- Class NetSquidNetworkInfo:
  - get_node_id_for_app
  - get_node_name_for_app

### squidasm/squidasm/run/singlethread/protocols.py
- Root: squidasm
- Public constants: SUBRT_FINISHED
- Public functions: (none)
- Public classes: HostPeerListener, HostProtocol, QNodeOsListener, QNodeOsProtocol, ResultsListener
- Class HostPeerListener:
  - buffer
  - run
- Class HostProtocol:
  - get_result
  - peer_listener
  - peer_port
  - qnos_port
  - results_listener
  - run
  - start
  - stop
- Class QNodeOsListener:
  - buffer
  - run
- Class QNodeOsProtocol:
  - executor
  - host_port
  - run
  - set_network_stack
  - start
  - stop
- Class ResultsListener:
  - buffer
  - run

### squidasm/squidasm/run/singlethread/run.py
- Root: squidasm
- Public constants: (none)
- Public functions: run_files, run_programs, run_protocols
- Public classes: (none)

### squidasm/squidasm/run/singlethread/util.py
- Root: squidasm
- Public constants: (none)
- Public functions: load_program, modify_and_import
- Public classes: (none)

### squidasm/squidasm/run/stack/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/run/stack/build.py
- Root: squidasm
- Public constants: (none)
- Public functions: create_stack_network_builder
- Public classes: StackNodeBuilder
- Class StackNodeBuilder:
  - build

### squidasm/squidasm/run/stack/config.py
- Root: squidasm
- Public constants: (none)
- Public functions: 
- Public classes: CLinkConfig, DefaultCLinkConfig, DepolariseLinkConfig, GenericQDeviceConfig, HeraldedLinkConfig, InstantCLinkConfig, LinkConfig, NVQDeviceConfig, StackConfig, StackNetworkConfig
- Class CLinkConfig:
  - from_file
  - perfect_config
- Class DefaultCLinkConfig:
  - (no public methods detected)
- Class DepolariseLinkConfig:
  - (no public methods detected)
- Class GenericQDeviceConfig:
  - (no public methods detected)
- Class HeraldedLinkConfig:
  - (no public methods detected)
- Class InstantCLinkConfig:
  - (no public methods detected)
- Class LinkConfig:
  - from_file
  - perfect_config
- Class NVQDeviceConfig:
  - (no public methods detected)
- Class StackConfig:
  - from_file
  - perfect_generic_config
- Class StackNetworkConfig:
  - from_file

### squidasm/squidasm/run/stack/run.py
- Root: squidasm
- Public constants: (none)
- Public functions: run
- Public classes: (none)

### squidasm/squidasm/sim/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/sim/glob.py
- Root: squidasm
- Public constants: 
- Public functions: get_current_app_node_mapping, get_current_node_ids, get_current_node_names, get_current_nodes, get_node_id, get_node_id_for_app, get_node_name, get_node_name_for_app, get_running_backend, pop_current_backend, put_current_backend
- Public classes: QubitInfo
- Class QubitInfo:
  - get_qubit_groups
  - update_qubits_used

### squidasm/squidasm/sim/network/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/sim/network/network.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: LinearDepolariseMagicDistributor, LinearDepolariseModelParameters, LinearDepolariseStateSamplerFactory, MagicNetworkLayerProtocol, NVQDevice, NetSquidNetwork, QDevice
- Class LinearDepolariseMagicDistributor:
  - (no public methods detected)
- Class LinearDepolariseModelParameters:
  - verify
- Class LinearDepolariseStateSamplerFactory:
  - (no public methods detected)
- Class MagicNetworkLayerProtocol:
  - (no public methods detected)
- Class NVQDevice:
  - (no public methods detected)
- Class NetSquidNetwork:
  - global_log
  - host_latency
  - instr_proc_time
  - link_layer_services
  - node_hardware_types
  - set_logger
- Class QDevice:
  - (no public methods detected)

### squidasm/squidasm/sim/network/nv_config.py
- Root: squidasm
- Public constants: (none)
- Public functions: build_nv_qdevice, nv_cfg_from_file, parse_nv_config
- Public classes: NVConfig
- Class NVConfig:
  - (no public methods detected)

### squidasm/squidasm/sim/queues.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: QueueManager, TaskQueue
- Class QueueManager:
  - create_queue
  - destroy_queues
  - get_queue
  - reset_queues
- Class TaskQueue:
  - empty
  - full
  - get
  - join
  - join_task
  - put
  - qsize
  - reset
  - task_done

### squidasm/squidasm/sim/stack/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/sim/stack/common.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: AllocError, AppMemory, ComponentProtocol, LogManager, NVPhysicalQuantumMemory, NetstackBreakpointCreateRequest, NetstackBreakpointReceiveRequest, NetstackCreateRequest, NetstackReceiveRequest, PhysicalQuantumMemory, PortListener, RegisterMeta, SimTimeFilter
- Class AllocError:
  - (no public methods detected)
- Class AppMemory:
  - expand_array_part
  - get_array
  - get_array_entry
  - get_array_part
  - get_array_slice
  - get_array_value
  - get_array_values
  - get_reg_value
  - get_register
  - increment_prog_counter
  - init_new_array
  - map_virt_id
  - phys_id_for
  - prog_counter
  - qubit_mapping
  - set_array_entry
  - set_array_value
  - set_prog_counter
  - set_reg_value
  - unmap_all
  - unmap_virt_id
  - virt_id_for
- Class ComponentProtocol:
  - add_listener
  - start
  - stop
- Class LogManager:
  - get_log_level
  - get_stack_logger
  - log_to_file
  - set_log_level
- Class NVPhysicalQuantumMemory:
  - (no public methods detected)
- Class NetstackBreakpointCreateRequest:
  - (no public methods detected)
- Class NetstackBreakpointReceiveRequest:
  - (no public methods detected)
- Class NetstackCreateRequest:
  - (no public methods detected)
- Class NetstackReceiveRequest:
  - (no public methods detected)
- Class PhysicalQuantumMemory:
  - allocate
  - allocate_comm
  - allocate_mem
  - clear
  - comm_qubit_count
  - free
  - is_allocated
  - qubit_count
- Class PortListener:
  - buffer
  - run
- Class RegisterMeta:
  - parse
  - prefixes
- Class SimTimeFilter:
  - filter

### squidasm/squidasm/sim/stack/connection.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: QnosConnection
- Class QnosConnection:
  - commit_protosubroutine
  - commit_subroutine
  - flush
  - shared_memory

### squidasm/squidasm/sim/stack/context.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetSquidContext, NetSquidNetworkInfo
- Class NetSquidContext:
  - add_node
  - add_protocol
  - get_nodes
  - get_protocols
  - reset
  - set_nodes
  - set_protocols
- Class NetSquidNetworkInfo:
  - get_node_id_for_app
  - get_node_name_for_app

### squidasm/squidasm/sim/stack/csocket.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: ClassicalSocket
- Class ClassicalSocket:
  - recv
  - recv_float
  - recv_int
  - recv_silent
  - recv_structured
  - send
  - send_float
  - send_int
  - send_silent
  - send_structured

### squidasm/squidasm/sim/stack/globals.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: GlobalSimData
- Class GlobalSimData:
  - get_last_breakpoint_state
  - get_network
  - get_quantum_state
  - set_network

### squidasm/squidasm/sim/stack/handler.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Handler, HandlerComponent, RunningApp
- Class Handler:
  - add_subroutine
  - app_memories
  - assign_processor
  - clear_application
  - flavour
  - flavour
  - init_new_app
  - msg_from_host
  - netstack
  - open_epr_socket
  - physical_memory
  - qnos
  - run
  - should_clear_memory
  - should_clear_memory
  - stop_application
- Class HandlerComponent:
  - host_in_port
  - host_out_port
  - netstack_comp
  - node
  - processor_comp
  - processor_in_port
  - processor_out_port
  - qnos_comp
- Class RunningApp:
  - add_subroutine
  - id
  - next_subroutine

### squidasm/squidasm/sim/stack/host.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Host, HostComponent
- Class Host:
  - compiler
  - compiler
  - enqueue_program
  - get_results
  - receive_qnos_msg
  - register_netsquid_socket
  - run
  - send_qnos_msg
- Class HostComponent:
  - qnos_in_port
  - qnos_out_port

### squidasm/squidasm/sim/stack/netstack.py
- Root: squidasm
- Public constants: PI, PI_OVER_2
- Public functions: (none)
- Public classes: EprSocket, Netstack, NetstackComponent
- Class EprSocket:
  - (no public methods detected)
- Class Netstack:
  - app_memories
  - assign_egp
  - find_epr_socket
  - handle_breakpoint_create_request
  - handle_breakpoint_receive_request
  - handle_create_ck_request
  - handle_create_md_request
  - handle_create_request
  - handle_receive_ck_request
  - handle_receive_md_request
  - handle_receive_request
  - open_epr_socket
  - physical_memory
  - qdevice
  - register_peer
  - run
  - start
  - stop
- Class NetstackComponent:
  - node
  - peer_in_port
  - peer_out_port
  - processor_in_port
  - processor_out_port
  - register_peer

### squidasm/squidasm/sim/stack/processor.py
- Root: squidasm
- Public constants: PI, PI_OVER_2
- Public functions: (none)
- Public classes: GenericProcessor, NVProcessor, Processor, ProcessorComponent
- Class GenericProcessor:
  - (no public methods detected)
- Class NVProcessor:
  - (no public methods detected)
- Class Processor:
  - app_memories
  - execute_subroutine
  - physical_memory
  - qdevice
  - run
- Class ProcessorComponent:
  - handler_in_port
  - handler_out_port
  - netstack_in_port
  - netstack_out_port
  - node
  - qdevice

### squidasm/squidasm/sim/stack/program.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Program, ProgramContext, ProgramMeta
- Class Program:
  - meta
  - run
- Class ProgramContext:
  - app_id
  - connection
  - csockets
  - epr_sockets
- Class ProgramMeta:
  - (no public methods detected)

### squidasm/squidasm/sim/stack/qnos.py
- Root: squidasm
- Public constants: NUM_QUBITS
- Public functions: (none)
- Public classes: Qnos, QnosComponent
- Class Qnos:
  - app_memories
  - assign_egp
  - get_virt_qubit_for_phys_id
  - handler
  - handler
  - netstack
  - netstack
  - physical_memory
  - processor
  - processor
  - start
  - stop
- Class QnosComponent:
  - handler_comp
  - host_in_port
  - host_out_port
  - netstack_comp
  - node
  - peer_in_port
  - peer_out_port
  - processor_comp
  - qdevice
  - register_peer

### squidasm/squidasm/sim/stack/qnos_network_service.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: QNOSNetworkService, ReqQNOSMessage
- Class QNOSNetworkService:
  - receive_qnos_message
  - register_remote_node
  - run
  - send_qnos_message
  - start
  - stop
- Class ReqQNOSMessage:
  - (no public methods detected)

### squidasm/squidasm/sim/stack/signals.py
- Root: squidasm
- Public constants: SIGNAL_HAND_HOST_MSG, SIGNAL_HAND_PROC_MSG, SIGNAL_HOST_HAND_MSG, SIGNAL_HOST_HOST_MSG, SIGNAL_MEMORY_FREED, SIGNAL_NSTK_PROC_MSG, SIGNAL_PEER_NSTK_MSG, SIGNAL_PEER_RECV_MSG, SIGNAL_PROC_HAND_MSG, SIGNAL_PROC_NSTK_MSG
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/sim/stack/stack.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NodeStack, StackNetwork, StackNode
- Class NodeStack:
  - assign_egp
  - host
  - host
  - host_comp
  - node
  - qdevice
  - qnos
  - qnos
  - qnos_comp
  - start
  - stop
- Class StackNetwork:
  - csockets
  - links
  - qdevices
  - stacks
- Class StackNode:
  - host_comp
  - qnos_comp
  - qnos_peer_port
  - register_peer

### squidasm/squidasm/util/__init__.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### squidasm/squidasm/util/ns.py
- Root: squidasm
- Public constants: (none)
- Public functions: is_dm_pure, is_ppt, is_pure_state_entangled, is_state_entangled, partial_transpose
- Public classes: (none)

### squidasm/squidasm/util/qkd_routine.py
- Root: squidasm
- Public constants: (none)
- Public functions: (none)
- Public classes: QKDRoutine
- Class QKDRoutine:
  - run

### squidasm/squidasm/util/routines.py
- Root: squidasm
- Public constants: (none)
- Public functions: create_ghz, distributed_CNOT_control, distributed_CNOT_target, distributed_CPhase_control, distributed_CPhase_target, measXY, recv_float, recv_int, recv_remote_state_preparation, remote_state_preparation, send_float, send_int, teleport_recv, teleport_send
- Public classes: (none)

### squidasm/squidasm/util/sim.py
- Root: squidasm
- Public constants: (none)
- Public functions: get_qubit_state
- Public classes: (none)

### squidasm/squidasm/util/thread.py
- Root: squidasm
- Public constants: (none)
- Public functions: as_completed
- Public classes: (none)

### squidasm/squidasm/util/util.py
- Root: squidasm
- Public constants: (none)
- Public functions: create_complete_graph_network, create_two_node_network, get_qubit_state, get_reference_state
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/backend/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/backend/executor.py
- Root: netqasm
- Public constants: (none)
- Public functions: inc_program_counter
- Public classes: EprCmdData, Executor
- Class EprCmdData:
  - (no public methods detected)
- Class Executor:
  - allocate_new_qubit_unit_module
  - consume_execute_subroutine
  - execute_subroutine
  - get_instr_logger
  - init_new_application
  - name
  - network_stack
  - network_stack
  - node_id
  - set_instr_logger
  - setup_epr_socket
  - stop_application

### qia/lib/python3.10/site-packages/netqasm/backend/messages.py
- Root: netqasm
- Public constants: APP_ID, EPR_FIDELITY, EPR_SOCKET_ID, MESSAGE_CLASSES, MESSAGE_ID, MESSAGE_TYPE, MESSAGE_TYPE_BYTES, NODE_ID, NUM_QUBITS, RETURN_MESSAGE_CLASSES, SIGNAL
- Public functions: deserialize_host_msg, deserialize_return_msg
- Public classes: ErrorCode, ErrorMessage, InitNewAppMessage, Message, MessageHeader, MessageType, MsgDoneMessage, OpenEPRSocketMessage, ReturnArrayMessage, ReturnArrayMessageHeader, ReturnMessage, ReturnMessageType, ReturnRegMessage, Signal, SignalMessage, StopAppMessage, SubroutineMessage
- Class ErrorCode:
  - (no public methods detected)
- Class ErrorMessage:
  - (no public methods detected)
- Class InitNewAppMessage:
  - (no public methods detected)
- Class Message:
  - deserialize_from
- Class MessageHeader:
  - len
- Class MessageType:
  - (no public methods detected)
- Class MsgDoneMessage:
  - (no public methods detected)
- Class OpenEPRSocketMessage:
  - (no public methods detected)
- Class ReturnArrayMessage:
  - deserialize_from
- Class ReturnArrayMessageHeader:
  - len
- Class ReturnMessage:
  - (no public methods detected)
- Class ReturnMessageType:
  - (no public methods detected)
- Class ReturnRegMessage:
  - (no public methods detected)
- Class Signal:
  - (no public methods detected)
- Class SignalMessage:
  - (no public methods detected)
- Class StopAppMessage:
  - (no public methods detected)
- Class SubroutineMessage:
  - deserialize_from

### qia/lib/python3.10/site-packages/netqasm/backend/network_stack.py
- Root: netqasm
- Public constants: CREATE_FIELDS, OK_FIELDS_K, OK_FIELDS_M
- Public functions: (none)
- Public classes: Address, BaseNetworkStack
- Class Address:
  - (no public methods detected)
- Class BaseNetworkStack:
  - get_purpose_id
  - put
  - setup_epr_socket

### qia/lib/python3.10/site-packages/netqasm/backend/qnodeos.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: QNodeController
- Class QNodeController:
  - add_network_stack
  - finished
  - handle_netqasm_message
  - has_active_apps
  - network_stack
  - network_stack
  - stop

### qia/lib/python3.10/site-packages/netqasm/examples/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_bob.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_charlie.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/app_david.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/conf.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/protocol.py
- Root: netqasm
- Public constants: (none)
- Public functions: anonymous_transmission
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/anonymous_transmission/src/sub_protocols.py
- Root: netqasm
- Public constants: (none)
- Public functions: anonymous_epr, classical_anonymous_transmission, quantum_anonymous_tranmission, setup_sockets
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/app_alice.py
- Root: netqasm
- Public constants: ALL_MEASURED, EOF
- Public functions: distribute_bb84_states, estimate_error_rate, extract_key, filter_bases, h, main, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Public classes: PairInfo
- Class PairInfo:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/bb84/app_bob.py
- Root: netqasm
- Public constants: ALL_MEASURED, EOF
- Public functions: estimate_error_rate, extract_key, filter_bases, h, main, receive_bb84_states, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Public classes: PairInfo
- Class PairInfo:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/app_client.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_phi_for_oracle, main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_grover/app_server.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/app_client.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/blind_rotation/app_server.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: main, measure_basis_0, measure_basis_1
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_bob.py
- Root: netqasm
- Public constants: (none)
- Public functions: main, measure_basis_0, measure_basis_1
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/chsh/app_repeater.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/app_controller.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/dist_cnot/app_target.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/app_client.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_ck/app_server.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/app_client.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/link_layer_md/app_server.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/app_player1.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/magic_square/app_player2.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/app_bob.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/shared/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/multiple_files/shared/myfuncs.py
- Root: netqasm
- Public constants: (none)
- Public functions: custom_measure, custom_recv, custom_send
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/single_node/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/single_node/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/app_receiver.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/teleport/app_sender.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_bob.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/apps/three_nodes/app_charlie.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/lib/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/lib/bqc.py
- Root: netqasm
- Public constants: (none)
- Public functions: measXY, recv_meas_cmd, recv_meas_outcome, recv_teleported_state, send_meas_cmd, send_meas_outcome, teleport_state
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/app_client.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/blind_computation/app_server.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_alice.py
- Root: netqasm
- Public constants: (none)
- Public functions: format_measurement_basis, main, measure_basis_0, measure_basis_1
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_bob.py
- Root: netqasm
- Public constants: (none)
- Public functions: format_corrections, format_measurement_basis, game_won, main, measure_basis_0, measure_basis_1
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/chsh/app_repeater.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/app_controller.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/dist_cnot/app_target.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/app_player1.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/magic_square/app_player2.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/app_alice.py
- Root: netqasm
- Public constants: ALL_MEASURED, EOF
- Public functions: distribute_bb84_states, estimate_error_rate, extract_key, filter_bases, h, main, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Public classes: PairInfo
- Class PairInfo:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/qkd/app_bob.py
- Root: netqasm
- Public constants: ALL_MEASURED, EOF
- Public functions: estimate_error_rate, extract_key, filter_bases, h, main, receive_bb84_states, recvClassicalAssured, recv_single_msg, sendClassicalAssured, send_single_msg
- Public classes: PairInfo
- Class PairInfo:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/app_receiver.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/qne_apps/teleport/app_sender.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/run_examples.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_bb84.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_enumerate.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_loop.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_post_epr.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_rsp.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_compilation/example_simple_loop.py
- Root: netqasm
- Public constants: (none)
- Public functions: main
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_scripts/epr.py
- Root: netqasm
- Public constants: (none)
- Public functions: create_epr, post_function, run_alice, run_bob
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/examples/sdk_scripts/rsp.py
- Root: netqasm
- Public constants: PRECOMPILE
- Public functions: run_client, run_server
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/lang/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/lang/encoding.py
- Root: netqasm
- Public constants: ADDRESS, ADDRESS_BITS, APP_ID, COMMANDS, COMMAND_BYTES, IMMEDIATE, IMMEDIATE_BITS, INSTR_ID, INTEGER, INTEGER_BITS, METADATA_BYTES, NETQASM_VERSION, PADDING_FIELD, REG_BITS, REG_INDEX_BITS, REG_NAME_BITS, REG_TYPE
- Public functions: add_padding
- Public classes: AddrCommand, Address, ArrayCommand, ArrayEntry, ArrayEntryCommand, ArraySlice, ArraySliceCommand, Command, ImmCommand, ImmImmCommand, MeasCommand, Metadata, NoOperandCommand, OptionalInt, RecvEPRCommand, Reg5Command, RegAddrCommand, RegCommand, RegEntryCommand, RegImmCommand, RegImmImmCommand, RegRegCommand, RegRegImm4Command, RegRegImmCommand, RegRegImmImmCommand, RegRegRegCommand, RegRegRegRegCommand, Register, RegisterName, SingleRegisterCommand
- Class AddrCommand:
  - (no public methods detected)
- Class Address:
  - (no public methods detected)
- Class ArrayCommand:
  - (no public methods detected)
- Class ArrayEntry:
  - (no public methods detected)
- Class ArrayEntryCommand:
  - (no public methods detected)
- Class ArraySlice:
  - (no public methods detected)
- Class ArraySliceCommand:
  - (no public methods detected)
- Class Command:
  - (no public methods detected)
- Class ImmCommand:
  - (no public methods detected)
- Class ImmImmCommand:
  - (no public methods detected)
- Class MeasCommand:
  - (no public methods detected)
- Class Metadata:
  - (no public methods detected)
- Class NoOperandCommand:
  - (no public methods detected)
- Class OptionalInt:
  - value
- Class RecvEPRCommand:
  - (no public methods detected)
- Class Reg5Command:
  - (no public methods detected)
- Class RegAddrCommand:
  - (no public methods detected)
- Class RegCommand:
  - (no public methods detected)
- Class RegEntryCommand:
  - (no public methods detected)
- Class RegImmCommand:
  - (no public methods detected)
- Class RegImmImmCommand:
  - (no public methods detected)
- Class RegRegCommand:
  - (no public methods detected)
- Class RegRegImm4Command:
  - (no public methods detected)
- Class RegRegImmCommand:
  - (no public methods detected)
- Class RegRegImmImmCommand:
  - (no public methods detected)
- Class RegRegRegCommand:
  - (no public methods detected)
- Class RegRegRegRegCommand:
  - (no public methods detected)
- Class Register:
  - (no public methods detected)
- Class RegisterName:
  - (no public methods detected)
- Class SingleRegisterCommand:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/lang/instr/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/lang/instr/base.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: AddrInstruction, ArrayEntryInstruction, ArraySliceInstruction, DebugInstruction, ImmImmInstruction, ImmInstruction, NetQASMInstruction, NoOperandInstruction, Reg5Instruction, RegAddrInstruction, RegEntryInstruction, RegImmImmInstruction, RegImmInstruction, RegInstruction, RegRegImm4Instruction, RegRegImmImmInstruction, RegRegImmInstruction, RegRegInstruction, RegRegRegInstruction, RegRegRegRegInstruction
- Class AddrInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class ArrayEntryInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class ArraySliceInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class DebugInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class ImmImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class ImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class NetQASMInstruction:
  - debug_str
  - deserialize_from
  - from_operands
  - operands
  - serialize
  - writes_to
- Class NoOperandInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class Reg5Instruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegAddrInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegEntryInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegImmImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegImm4Instruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegImmImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegImmInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegRegInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize
- Class RegRegRegRegInstruction:
  - deserialize_from
  - from_operands
  - operands
  - serialize

### qia/lib/python3.10/site-packages/netqasm/lang/instr/core.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: AddInstruction, AddmInstruction, ArrayInstruction, BeqInstruction, BezInstruction, BgeInstruction, BltInstruction, BneInstruction, BnzInstruction, BranchBinaryInstruction, BranchUnaryInstruction, BreakpointInstruction, ClassicalOpInstruction, ClassicalOpModInstruction, ControlledRotationInstruction, CreateEPRInstruction, DivInstruction, InitInstruction, JmpInstruction, LeaInstruction, LoadInstruction, MeasBasisInstruction, MeasInstruction, MulInstruction, QAllocInstruction, QFreeInstruction, RecvEPRInstruction, RemInstruction, RetArrInstruction, RetRegInstruction, RotationInstruction, SetInstruction, SingleQubitInstruction, StoreInstruction, SubInstruction, SubmInstruction, TwoQubitInstruction, UndefInstruction, WaitAllInstruction, WaitAnyInstruction, WaitSingleInstruction
- Class AddInstruction:
  - (no public methods detected)
- Class AddmInstruction:
  - (no public methods detected)
- Class ArrayInstruction:
  - size
  - size
- Class BeqInstruction:
  - check_condition
- Class BezInstruction:
  - check_condition
- Class BgeInstruction:
  - check_condition
- Class BltInstruction:
  - check_condition
- Class BneInstruction:
  - check_condition
- Class BnzInstruction:
  - check_condition
- Class BranchBinaryInstruction:
  - check_condition
  - line
  - line
- Class BranchUnaryInstruction:
  - check_condition
  - line
  - line
- Class BreakpointInstruction:
  - action
  - action
  - role
  - role
- Class ClassicalOpInstruction:
  - regin0
  - regin0
  - regin1
  - regin1
  - regout
  - regout
  - writes_to
- Class ClassicalOpModInstruction:
  - regin0
  - regin0
  - regin1
  - regin1
  - regmod
  - regmod
  - regout
  - regout
  - writes_to
- Class ControlledRotationInstruction:
  - angle_denom
  - angle_denom
  - angle_num
  - angle_num
  - qreg0
  - qreg0
  - qreg1
  - qreg1
  - to_matrix
- Class CreateEPRInstruction:
  - arg_array
  - arg_array
  - ent_results_array
  - ent_results_array
  - epr_socket_id
  - epr_socket_id
  - qubit_addr_array
  - qubit_addr_array
  - remote_node_id
  - remote_node_id
- Class DivInstruction:
  - (no public methods detected)
- Class InitInstruction:
  - qreg
  - qreg
- Class JmpInstruction:
  - line
  - line
- Class LeaInstruction:
  - writes_to
- Class LoadInstruction:
  - writes_to
- Class MeasBasisInstruction:
  - angle_denom
  - angle_denom
  - angle_num_x1
  - angle_num_x1
  - angle_num_x2
  - angle_num_x2
  - angle_num_y
  - angle_num_y
  - creg
  - creg
  - qreg
  - qreg
  - writes_to
- Class MeasInstruction:
  - creg
  - creg
  - qreg
  - qreg
  - writes_to
- Class MulInstruction:
  - (no public methods detected)
- Class QAllocInstruction:
  - qreg
  - qreg
- Class QFreeInstruction:
  - qreg
  - qreg
- Class RecvEPRInstruction:
  - ent_results_array
  - ent_results_array
  - epr_socket_id
  - epr_socket_id
  - qubit_addr_array
  - qubit_addr_array
  - remote_node_id
  - remote_node_id
- Class RemInstruction:
  - (no public methods detected)
- Class RetArrInstruction:
  - (no public methods detected)
- Class RetRegInstruction:
  - (no public methods detected)
- Class RotationInstruction:
  - angle_denom
  - angle_denom
  - angle_num
  - angle_num
  - from_operands
  - qreg
  - qreg
  - to_matrix
- Class SetInstruction:
  - from_operands
  - writes_to
- Class SingleQubitInstruction:
  - qreg
  - qreg
  - to_matrix
- Class StoreInstruction:
  - (no public methods detected)
- Class SubInstruction:
  - (no public methods detected)
- Class SubmInstruction:
  - (no public methods detected)
- Class TwoQubitInstruction:
  - qreg0
  - qreg0
  - qreg1
  - qreg1
  - to_matrix
  - to_matrix_target_only
- Class UndefInstruction:
  - (no public methods detected)
- Class WaitAllInstruction:
  - (no public methods detected)
- Class WaitAnyInstruction:
  - (no public methods detected)
- Class WaitSingleInstruction:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/lang/instr/flavour.py
- Root: netqasm
- Public constants: CORE_INSTRUCTIONS
- Public functions: (none)
- Public classes: Flavour, InstrMap, NVFlavour, REIDSFlavour, TrappedIonFlavour, VanillaFlavour
- Class Flavour:
  - get_instr_by_id
  - get_instr_by_name
  - instrs
- Class InstrMap:
  - (no public methods detected)
- Class NVFlavour:
  - instrs
- Class REIDSFlavour:
  - instrs
- Class TrappedIonFlavour:
  - instrs
- Class VanillaFlavour:
  - instrs

### qia/lib/python3.10/site-packages/netqasm/lang/instr/nv.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: ControlledRotXInstruction, ControlledRotYInstruction, RotXInstruction, RotYInstruction, RotZInstruction
- Class ControlledRotXInstruction:
  - to_matrix
  - to_matrix_target_only
- Class ControlledRotYInstruction:
  - to_matrix
  - to_matrix_target_only
- Class RotXInstruction:
  - to_matrix
- Class RotYInstruction:
  - to_matrix
- Class RotZInstruction:
  - to_matrix

### qia/lib/python3.10/site-packages/netqasm/lang/instr/trapped_ion.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: AllQubitsInitInstruction, AllQubitsMeasInstruction, AllQubitsRotXInstruction, AllQubitsRotYInstruction, AllQubitsRotZInstruction, AllQubitsRotationInstruction, BichromaticInstruction, RotZInstruction
- Class AllQubitsInitInstruction:
  - (no public methods detected)
- Class AllQubitsMeasInstruction:
  - (no public methods detected)
- Class AllQubitsRotXInstruction:
  - (no public methods detected)
- Class AllQubitsRotYInstruction:
  - (no public methods detected)
- Class AllQubitsRotZInstruction:
  - (no public methods detected)
- Class AllQubitsRotationInstruction:
  - angle_denom
  - angle_denom
  - angle_num
  - angle_num
  - from_operands
- Class BichromaticInstruction:
  - angle_denom
  - angle_denom
  - angle_num
  - angle_num
  - from_operands
- Class RotZInstruction:
  - to_matrix

### qia/lib/python3.10/site-packages/netqasm/lang/instr/vanilla.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: CnotInstruction, CphaseInstruction, GateHInstruction, GateKInstruction, GateSInstruction, GateTInstruction, GateXInstruction, GateYInstruction, GateZInstruction, MovInstruction, RotXInstruction, RotYInstruction, RotZInstruction
- Class CnotInstruction:
  - to_matrix
  - to_matrix_target_only
- Class CphaseInstruction:
  - to_matrix
  - to_matrix_target_only
- Class GateHInstruction:
  - to_matrix
- Class GateKInstruction:
  - to_matrix
- Class GateSInstruction:
  - to_matrix
- Class GateTInstruction:
  - to_matrix
- Class GateXInstruction:
  - to_matrix
- Class GateYInstruction:
  - to_matrix
- Class GateZInstruction:
  - to_matrix
- Class MovInstruction:
  - to_matrix
  - to_matrix_target_only
- Class RotXInstruction:
  - to_matrix
- Class RotYInstruction:
  - to_matrix
- Class RotZInstruction:
  - to_matrix

### qia/lib/python3.10/site-packages/netqasm/lang/ir.py
- Root: netqasm
- Public constants: 
- Public functions: flip_branch_instr, instruction_to_string, string_to_instruction
- Public classes: BranchLabel, BreakpointAction, BreakpointRole, GenericInstr, ICmd, ProtoSubroutine
- Class BranchLabel:
  - debug_str
- Class BreakpointAction:
  - (no public methods detected)
- Class BreakpointRole:
  - (no public methods detected)
- Class GenericInstr:
  - (no public methods detected)
- Class ICmd:
  - debug_str
- Class ProtoSubroutine:
  - app_id
  - arguments
  - commands
  - commands
  - instantiate
  - netqasm_version

### qia/lib/python3.10/site-packages/netqasm/lang/operand.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Address, ArrayEntry, ArraySlice, Immediate, Label, Operand, Register, RegisterMeta, Template
- Class Address:
  - cstruct
  - from_raw
- Class ArrayEntry:
  - cstruct
  - from_raw
- Class ArraySlice:
  - cstruct
  - from_raw
- Class Immediate:
  - (no public methods detected)
- Class Label:
  - (no public methods detected)
- Class Operand:
  - (no public methods detected)
- Class Register:
  - cstruct
  - from_raw
  - from_str
- Class RegisterMeta:
  - parse
  - prefixes
- Class Template:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/lang/parsing/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/lang/parsing/binary.py
- Root: netqasm
- Public constants: INSTR_ID
- Public functions: deserialize
- Public classes: Deserializer
- Class Deserializer:
  - deserialize_command
  - deserialize_subroutine

### qia/lib/python3.10/site-packages/netqasm/lang/parsing/text.py
- Root: netqasm
- Public constants: 
- Public functions: assemble_subroutine, get_current_registers, parse_address, parse_register, parse_text_protosubroutine, parse_text_subroutine
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/lang/subroutine.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Subroutine
- Class Subroutine:
  - app_id
  - app_id
  - arguments
  - cstructs
  - instantiate
  - instructions
  - instructions
  - netqasm_version
  - pretty_print
  - print_instructions

### qia/lib/python3.10/site-packages/netqasm/lang/symbols.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Symbols
- Class Symbols:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/lang/version.py
- Root: netqasm
- Public constants: NETQASM_VERSION
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/logging/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/logging/glob.py
- Root: netqasm
- Public constants: NETQASM_LOGGER
- Public functions: get_log_level, get_netqasm_logger, set_log_level
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/logging/output.py
- Root: netqasm
- Public constants: 
- Public functions: get_new_app_logger, reset_struct_loggers, save_all_struct_loggers, should_ignore_instr
- Public classes: AppLogger, ClassCommLogger, InstrLogger, NetworkLogger, SocketOperation, StructuredLogger
- Class AppLogger:
  - (no public methods detected)
- Class ClassCommLogger:
  - (no public methods detected)
- Class InstrLogger:
  - (no public methods detected)
- Class NetworkLogger:
  - (no public methods detected)
- Class SocketOperation:
  - (no public methods detected)
- Class StructuredLogger:
  - log
  - save

### qia/lib/python3.10/site-packages/netqasm/qlink_compat.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_creator_node_id, request_to_qlink_1_0, response_from_qlink_1_0
- Public classes: Basis, BellState, EPRRole, EPRType, ErrorCode, RandomBasis, RequestType, ReturnType, TimeUnit
- Class Basis:
  - (no public methods detected)
- Class BellState:
  - (no public methods detected)
- Class EPRRole:
  - (no public methods detected)
- Class EPRType:
  - (no public methods detected)
- Class ErrorCode:
  - (no public methods detected)
- Class RandomBasis:
  - (no public methods detected)
- Class RequestType:
  - (no public methods detected)
- Class ReturnType:
  - (no public methods detected)
- Class TimeUnit:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/runtime/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/app_config.py
- Root: netqasm
- Public constants: (none)
- Public functions: default_app_config
- Public classes: AppConfig
- Class AppConfig:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/runtime/application.py
- Root: netqasm
- Public constants: (none)
- Public functions: app_instance_from_path, default_app_instance, load_yaml_file, network_cfg_from_path, post_function_from_path
- Public classes: AppMetadata, Application, ApplicationInstance, ApplicationOutput, Program
- Class AppMetadata:
  - (no public methods detected)
- Class Application:
  - (no public methods detected)
- Class ApplicationInstance:
  - (no public methods detected)
- Class ApplicationOutput:
  - (no public methods detected)
- Class Program:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/runtime/cli.py
- Root: netqasm
- Public constants: CONTEXT_SETTINGS, EXAMPLE_APPS, QNE_FOLDER_PATH
- Public functions: cli, init, new, qne, run, simulate, version
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/debug.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_qubit_state, run_application
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/env.py
- Root: netqasm
- Public constants: EXAMPLE_APPS_DIR, IGNORED_FILES
- Public functions: file_creation_notify, get_example_apps, get_log_dir, get_post_function_path, get_results_path, get_roles_config_path, get_timed_log_dir, init_folder, load_app_config_file, load_app_files, load_post_function, load_roles_config, new_folder
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/hardware.py
- Root: netqasm
- Public constants: (none)
- Public functions: run_application, save_results
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/interface/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/interface/config.py
- Root: netqasm
- Public constants: 
- Public functions: default_network_config, network_cfg_from_file, parse_network_config
- Public classes: Link, NetworkConfig, Node, NoiseType, QuantumHardware, Qubit
- Class Link:
  - (no public methods detected)
- Class NetworkConfig:
  - (no public methods detected)
- Class Node:
  - (no public methods detected)
- Class NoiseType:
  - (no public methods detected)
- Class QuantumHardware:
  - (no public methods detected)
- Class Qubit:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/runtime/interface/logging.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: AppLogEntry, ClassCommLogEntry, EntanglementStage, EntanglementType, InstrLogEntry, NetworkLogEntry, QubitGroup
- Class AppLogEntry:
  - (no public methods detected)
- Class ClassCommLogEntry:
  - (no public methods detected)
- Class EntanglementStage:
  - (no public methods detected)
- Class EntanglementType:
  - (no public methods detected)
- Class InstrLogEntry:
  - (no public methods detected)
- Class NetworkLogEntry:
  - (no public methods detected)
- Class QubitGroup:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/runtime/interface/results.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/process_logs.py
- Root: netqasm
- Public constants: 
- Public functions: create_app_instr_logs, make_last_log, process_log
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/runtime/runtime_mgr.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: ApplicationInstance, NetworkConfig, NetworkInstance, RuntimeManager
- Class ApplicationInstance:
  - (no public methods detected)
- Class NetworkConfig:
  - (no public methods detected)
- Class NetworkInstance:
  - (no public methods detected)
- Class RuntimeManager:
  - get_network
  - run_app
  - set_network
  - start_backend

### qia/lib/python3.10/site-packages/netqasm/runtime/settings.py
- Root: netqasm
- Public constants: SIMULATOR_ENV
- Public functions: get_is_using_hardware, get_simulator, set_is_using_hardware, set_simulator
- Public classes: Flavour, Formalism, Simulator
- Class Flavour:
  - (no public methods detected)
- Class Formalism:
  - (no public methods detected)
- Class Simulator:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/build_epr.py
- Root: netqasm
- Public constants: SER_CREATE_IDX_ATOMIC, SER_CREATE_IDX_CONSECUTIVE, SER_CREATE_IDX_MAX_TIME, SER_CREATE_IDX_MINIMUM_FIDELITY, SER_CREATE_IDX_NUMBER, SER_CREATE_IDX_PRIORITY, SER_CREATE_IDX_PROBABILITY_DIST_LOCAL1, SER_CREATE_IDX_PROBABILITY_DIST_REMOTE1, SER_CREATE_IDX_PROBABLIITY_DIST_LOCAL2, SER_CREATE_IDX_PROBABLIITY_DIST_REMOTE2, SER_CREATE_IDX_RANDOM_BASIS_LOCAL, SER_CREATE_IDX_RANDOM_BASIS_REMOTE, SER_CREATE_IDX_ROTATION_X_LOCAL1, SER_CREATE_IDX_ROTATION_X_LOCAL2, SER_CREATE_IDX_ROTATION_X_REMOTE1, SER_CREATE_IDX_ROTATION_X_REMOTE2, SER_CREATE_IDX_ROTATION_Y_LOCAL, SER_CREATE_IDX_ROTATION_Y_REMOTE, SER_CREATE_IDX_TIME_UNIT, SER_CREATE_IDX_TYPE, SER_CREATE_LEN, SER_RESPONSE_KEEP_IDX_BELL_STATE, SER_RESPONSE_KEEP_IDX_CREATE_ID, SER_RESPONSE_KEEP_IDX_DIRECTONIALITY_FLAG, SER_RESPONSE_KEEP_IDX_GOODNESS, SER_RESPONSE_KEEP_IDX_GOODNESS_TIME, SER_RESPONSE_KEEP_IDX_LOGICAL_QUBIT_ID, SER_RESPONSE_KEEP_IDX_PURPOSE_ID, SER_RESPONSE_KEEP_IDX_REMOTE_NODE_ID, SER_RESPONSE_KEEP_IDX_SEQUENCE_NUMBER, SER_RESPONSE_KEEP_IDX_TYPE, SER_RESPONSE_KEEP_LEN, SER_RESPONSE_MEASURE_IDX_BELL_STATE, SER_RESPONSE_MEASURE_IDX_CREATE_ID, SER_RESPONSE_MEASURE_IDX_DIRECTONIALITY_FLAG, SER_RESPONSE_MEASURE_IDX_GOODNESS, SER_RESPONSE_MEASURE_IDX_MEASUREMENT_BASIS, SER_RESPONSE_MEASURE_IDX_MEASUREMENT_OUTCOME, SER_RESPONSE_MEASURE_IDX_PURPOSE_ID, SER_RESPONSE_MEASURE_IDX_REMOTE_NODE_ID, SER_RESPONSE_MEASURE_IDX_SEQUENCE_NUMBER, SER_RESPONSE_MEASURE_IDX_TYPE, SER_RESPONSE_MEASURE_LEN
- Public functions: basis_to_rotation, deserialize_epr_keep_results, deserialize_epr_measure_results, rotation_to_basis, serialize_request
- Public classes: EntRequestParams, EprKeepResult, EprMeasBasis, EprMeasureResult
- Class EntRequestParams:
  - (no public methods detected)
- Class EprKeepResult:
  - bell_state
- Class EprMeasBasis:
  - (no public methods detected)
- Class EprMeasureResult:
  - bell_state
  - measurement_outcome

### qia/lib/python3.10/site-packages/netqasm/sdk/build_nv.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NVEprCompiler
- Class NVEprCompiler:
  - get_max_time_for_fidelity

### qia/lib/python3.10/site-packages/netqasm/sdk/build_types.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: GenericHardwareConfig, HardwareConfig, NVHardwareConfig
- Class GenericHardwareConfig:
  - (no public methods detected)
- Class HardwareConfig:
  - comm_qubit_count
  - mem_qubit_count
  - qubit_count
- Class NVHardwareConfig:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/builder.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Builder, LabelManager, SdkForEachContext, SdkIfContext, SdkLoopUntilContext
- Class Builder:
  - alloc_array
  - app_id
  - app_id
  - committed_subroutines
  - if_context_enter
  - if_context_exit
  - inactivate_qubits
  - new_qubit_id
  - new_register
  - sdk_create_epr_context
  - sdk_create_epr_keep
  - sdk_create_epr_measure
  - sdk_create_epr_rsp
  - sdk_epr_keep
  - sdk_epr_measure
  - sdk_epr_rsp_create
  - sdk_epr_rsp_recv
  - sdk_if_eq
  - sdk_if_ez
  - sdk_if_ge
  - sdk_if_lt
  - sdk_if_ne
  - sdk_if_nz
  - sdk_loop_body
  - sdk_loop_context
  - sdk_new_foreach_context
  - sdk_new_if_context
  - sdk_new_loop_until_context
  - sdk_recv_epr_keep
  - sdk_recv_epr_measure
  - sdk_recv_epr_rsp
  - sdk_try_context
  - subrt_add_pending_command
  - subrt_add_pending_commands
  - subrt_compile_subroutine
  - subrt_pop_all_pending_commands
  - subrt_pop_pending_subroutine
- Class LabelManager:
  - new_label
- Class SdkForEachContext:
  - (no public methods detected)
- Class SdkIfContext:
  - (no public methods detected)
- Class SdkLoopUntilContext:
  - cleanup_code
  - exit_condition
  - loop_register
  - max_iterations
  - set_cleanup_code
  - set_exit_condition
  - set_loop_register

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/broadcast_channel.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: BroadcastChannel, BroadcastChannelBySockets
- Class BroadcastChannel:
  - conn_lost_callback
  - recv
  - recv_callback
  - send
- Class BroadcastChannelBySockets:
  - recv
  - send

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/message.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: StructuredMessage
- Class StructuredMessage:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/socket.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: Socket
- Class Socket:
  - conn_lost_callback
  - recv
  - recv_callback
  - recv_silent
  - recv_structured
  - send
  - send_silent
  - send_structured

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/broadcast_channel.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: ThreadBroadcastChannel
- Class ThreadBroadcastChannel:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/socket.py
- Root: netqasm
- Public constants: (none)
- Public functions: log_recv, log_recv_structured, log_send, log_send_structured, trim_msg
- Public classes: StorageThreadSocket, ThreadSocket
- Class StorageThreadSocket:
  - recv_callback
- Class ThreadSocket:
  - app_name
  - connected
  - get_comm_logger
  - id
  - key
  - recv
  - recv_silent
  - recv_structured
  - remote_app_name
  - remote_key
  - send
  - send_silent
  - send_structured
  - use_callbacks
  - use_callbacks
  - wait

### qia/lib/python3.10/site-packages/netqasm/sdk/classical_communication/thread_socket/socket_hub.py
- Root: netqasm
- Public constants: (none)
- Public functions: reset_socket_hub
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/config.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: LogConfig
- Class LogConfig:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/connection.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: BaseNetQASMConnection, DebugConnection, DebugNetworkInfo
- Class BaseNetQASMConnection:
  - active_qubits
  - app_id
  - app_id
  - app_name
  - block
  - builder
  - clear
  - close
  - commit_protosubroutine
  - commit_subroutine
  - compile
  - flush
  - get_app_ids
  - get_app_names
  - if_eq
  - if_ez
  - if_ge
  - if_lt
  - if_ne
  - if_nz
  - insert_breakpoint
  - loop
  - loop_body
  - loop_until
  - network_info
  - new_array
  - node_name
  - shared_memory
  - test_preparation
  - tomography
  - try_until_success
- Class DebugConnection:
  - shared_memory
- Class DebugNetworkInfo:
  - get_node_id_for_app
  - get_node_name_for_app

### qia/lib/python3.10/site-packages/netqasm/sdk/constraint.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: SdkConstraint, ValueAtMostConstraint
- Class SdkConstraint:
  - (no public methods detected)
- Class ValueAtMostConstraint:
  - future
  - value

### qia/lib/python3.10/site-packages/netqasm/sdk/epr_socket.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: EPRSocket
- Class EPRSocket:
  - conn
  - conn
  - create
  - create_context
  - create_keep
  - create_keep_with_info
  - create_measure
  - create_rsp
  - epr_socket_id
  - min_fidelity
  - recv
  - recv_context
  - recv_keep
  - recv_keep_with_info
  - recv_measure
  - recv_rsp
  - recv_rsp_with_info
  - remote_app_name
  - remote_epr_socket_id
  - remote_node_id

### qia/lib/python3.10/site-packages/netqasm/sdk/external.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/futures.py
- Root: netqasm
- Public constants: (none)
- Public functions: as_int_when_value
- Public classes: Array, BaseFuture, Future, NoValueError, NonConstantIndexError, RegFuture
- Class Array:
  - address
  - builder
  - enumerate
  - foreach
  - get_future_index
  - get_future_slice
  - lineno
  - undefine
- Class BaseFuture:
  - add
  - builder
  - if_eq
  - if_ez
  - if_ge
  - if_lt
  - if_ne
  - if_nz
  - subrt_result
  - subrt_result
  - value
- Class Future:
  - add
  - get_address_entry
  - get_load_commands
- Class NoValueError:
  - (no public methods detected)
- Class NonConstantIndexError:
  - (no public methods detected)
- Class RegFuture:
  - add
  - reg
  - reg

### qia/lib/python3.10/site-packages/netqasm/sdk/memmgr.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: MemoryManager
- Class MemoryManager:
  - activate_qubit
  - add_active_register
  - add_array_to_return
  - add_register_to_return
  - deactivate_qubit
  - get_active_qubits
  - get_arrays_to_return
  - get_inactive_register
  - get_new_array_address
  - get_new_meas_outcome_register
  - get_new_qubit_address
  - get_registers_to_return
  - inactivate_qubits
  - is_qubit_active
  - is_qubit_id_used
  - is_register_active
  - meas_register_set_unused
  - meas_register_set_used
  - remove_active_register
  - reset
  - reset_arrays_to_return
  - reset_registers_to_return
  - reset_used_meas_registers

### qia/lib/python3.10/site-packages/netqasm/sdk/network.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetworkInfo
- Class NetworkInfo:
  - get_node_id_for_app
  - get_node_name_for_app

### qia/lib/python3.10/site-packages/netqasm/sdk/progress_bar.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: ProgressBar
- Class ProgressBar:
  - close
  - increase
  - update

### qia/lib/python3.10/site-packages/netqasm/sdk/qubit.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: FutureQubit, Qubit, QubitMeasureBasis, QubitNotActiveError
- Class FutureQubit:
  - entanglement_info
  - remote_entangled_node
- Class Qubit:
  - H
  - K
  - S
  - T
  - X
  - Y
  - Z
  - active
  - active
  - assert_active
  - builder
  - cnot
  - connection
  - cphase
  - entanglement_info
  - free
  - measure
  - qubit_id
  - qubit_id
  - remote_entangled_node
  - reset
  - rot_X
  - rot_Y
  - rot_Z
- Class QubitMeasureBasis:
  - (no public methods detected)
- Class QubitNotActiveError:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/sdk/shared_memory.py
- Root: netqasm
- Public constants: (none)
- Public functions: setup_registers
- Public classes: Arrays, RegisterGroup, SharedMemory, SharedMemoryManager
- Class Arrays:
  - has_array
  - init_new_array
- Class RegisterGroup:
  - (no public methods detected)
- Class SharedMemory:
  - get_array_part
  - get_register
  - init_new_array
  - set_array_part
  - set_register
- Class SharedMemoryManager:
  - create_shared_memory
  - get_shared_memory
  - reset_memories

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/gates.py
- Root: netqasm
- Public constants: (none)
- Public functions: t_inverse, toffoli_gate
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/measurements.py
- Root: netqasm
- Public constants: (none)
- Public functions: parity_meas
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/multi_node.py
- Root: netqasm
- Public constants: (none)
- Public functions: create_ghz
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/sim_states.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_fidelity, qubit_from, to_dm
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/toolbox/state_prep.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_angle_spec_from_float, set_qubit_state
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/sdk/transpile.py
- Root: netqasm
- Public constants: (none)
- Public functions: get_hardware_num_denom
- Public classes: NVSubroutineTranspiler, REIDSSubroutineTranspiler, SubroutineTranspiler
- Class NVSubroutineTranspiler:
  - get_reg_value
  - get_unused_register
  - swap
  - transpile
- Class REIDSSubroutineTranspiler:
  - transpile
- Class SubroutineTranspiler:
  - transpile

### qia/lib/python3.10/site-packages/netqasm/typedefs.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/typing.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/__init__.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/error.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: NetQASMInstrError, NetQASMSyntaxError, NoCircuitRuleError, NotAllocatedError, SubroutineAbortedError
- Class NetQASMInstrError:
  - (no public methods detected)
- Class NetQASMSyntaxError:
  - (no public methods detected)
- Class NoCircuitRuleError:
  - (no public methods detected)
- Class NotAllocatedError:
  - (no public methods detected)
- Class SubroutineAbortedError:
  - (no public methods detected)

### qia/lib/python3.10/site-packages/netqasm/util/log.py
- Root: netqasm
- Public constants: (none)
- Public functions: (none)
- Public classes: HostLine, LineTracker
- Class HostLine:
  - (no public methods detected)
- Class LineTracker:
  - get_line

### qia/lib/python3.10/site-packages/netqasm/util/quantum_gates.py
- Root: netqasm
- Public constants: CNOT, CPHASE, H, K, PAULIS, S, STATIC_QUBIT_GATE_TO_MATRIX, T, X, Y, Z
- Public functions: are_matrices_equal, gate_to_matrix, get_controlled_rotation_matrix, get_rotation_matrix
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/states.py
- Root: netqasm
- Public constants: (none)
- Public functions: bloch_sphere_rep
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/string.py
- Root: netqasm
- Public constants: ALPHA_ALL, ALPHA_CAPITAL, ALPHA_LOWER, ALPHA_NUM, NUM
- Public functions: group_by_word, is_float, is_number, is_variable_name, rspaces
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/thread.py
- Root: netqasm
- Public constants: (none)
- Public functions: as_completed
- Public classes: (none)

### qia/lib/python3.10/site-packages/netqasm/util/yaml.py
- Root: netqasm
- Public constants: (none)
- Public functions: dump_yaml, load_yaml
- Public classes: (none)

### qia/lib/python3.10/site-packages/netsquid_magic/__init__.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: (none)

### qia/lib/python3.10/site-packages/netsquid_magic/abstract_heralded_connection.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: AbstractHeraldedConnection, AbstractHeraldedMagic, AbstractHeraldedModelParameters
- Class AbstractHeraldedConnection:
  - attenuation_coefficient
  - efficiency
  - fidelity
  - length
  - num_modes
  - prob_max_mixed
  - speed_of_light
  - speed_of_light_delay
  - success_probability
- Class AbstractHeraldedMagic:
  - (no public methods detected)
- Class AbstractHeraldedModelParameters:
  - verify

### qia/lib/python3.10/site-packages/netsquid_magic/egp.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: EgpProtocol
- Class EgpProtocol:
  - create_and_keep
  - measure_directly
  - receive
  - remote_state_preparation
  - run
  - stop_receive

### qia/lib/python3.10/site-packages/netsquid_magic/entanglement_magic.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: EntanglementMagic
- Class EntanglementMagic:
  - abort
  - check_queue
  - generate_entanglement
  - get_in_use
  - is_connected
  - run
  - set_in_use
  - start
  - stop

### qia/lib/python3.10/site-packages/netsquid_magic/link_layer.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: LinkLayerService, MagicLinkLayerProtocol, MagicLinkLayerProtocolWithSignaling, QueueItem, SingleClickTranslationUnit, TranslationUnit
- Class LinkLayerService:
  - (no public methods detected)
- Class MagicLinkLayerProtocol:
  - abort_request
  - capacity
  - close
  - num_requests_in_queue
  - open
  - put_from
  - stop
- Class MagicLinkLayerProtocolWithSignaling:
  - magic_distributor
  - react_to
- Class QueueItem:
  - (no public methods detected)
- Class SingleClickTranslationUnit:
  - request_to_parameters
- Class TranslationUnit:
  - request_to_parameters

### qia/lib/python3.10/site-packages/netsquid_magic/long_distance_interface.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: ILongDistanceInterface
- Class ILongDistanceInterface:
  - operate
  - probability_success

### qia/lib/python3.10/site-packages/netsquid_magic/magic.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: MagicProtocol
- Class MagicProtocol:
  - is_connected
  - nodes
  - put_from
  - react_to
  - service_interfaces
  - start
  - stop

### qia/lib/python3.10/site-packages/netsquid_magic/magic_distributor.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: BitflipMagicDistributor, Delivery, DepolariseMagicDistributor, DepolariseWithFailureMagicDistributor, DoubleClickMagicDistributor, HeraldedConnectionMagicDistributor, MagicDistributor, NodeDelivery, PerfectStateMagicDistributor, SingleClickMagicDistributor
- Class BitflipMagicDistributor:
  - (no public methods detected)
- Class Delivery:
  - in_process
- Class DepolariseMagicDistributor:
  - (no public methods detected)
- Class DepolariseWithFailureMagicDistributor:
  - get_bell_state
- Class DoubleClickMagicDistributor:
  - get_bell_state
- Class HeraldedConnectionMagicDistributor:
  - (no public methods detected)
- Class MagicDistributor:
  - abort_all_delivery
  - abort_delivery
  - add_callback
  - add_delivery
  - add_pair_request
  - clear_all_callbacks
  - get_bell_state
  - get_label
  - get_qmemories_from_nodes
  - long_distance_interface
  - long_distance_interface
  - merge_magic_distributor
  - nodes
  - peek_delivery
  - peek_node_delivery
  - reset
  - set_skip_rounds
  - start
  - stop
- Class NodeDelivery:
  - (no public methods detected)
- Class PerfectStateMagicDistributor:
  - (no public methods detected)
- Class SingleClickMagicDistributor:
  - add_delivery
  - get_bell_state

### qia/lib/python3.10/site-packages/netsquid_magic/model_parameters.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: BitFlipModelParameters, DepolariseModelParameters, DoubleClickModelParameters, HeraldedModelParameters, IModelParameters, PerfectModelParameters, SingleClickModelParameters
- Class BitFlipModelParameters:
  - verify
- Class DepolariseModelParameters:
  - verify
- Class DoubleClickModelParameters:
  - verify
- Class HeraldedModelParameters:
  - absorb_collection_efficiency_in_p_init
  - verify
  - verify_only_heralded_params
- Class IModelParameters:
  - verify
  - verify_between_0_and_1
  - verify_equal
  - verify_is_real_number
  - verify_is_type
  - verify_not_negative_value
- Class PerfectModelParameters:
  - verify
- Class SingleClickModelParameters:
  - verify

### qia/lib/python3.10/site-packages/netsquid_magic/qlink.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: IQLink, MagicQLink
- Class IQLink:
  - close
  - num_requests_in_queue
  - open
- Class MagicQLink:
  - abort
  - add_callback
  - add_pair_request
  - close
  - get_bell_state
  - num_requests_in_queue
  - open
  - peek_node_delivery

### qia/lib/python3.10/site-packages/netsquid_magic/services.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: ServiceInterface, ServiceProtocol
- Class ServiceInterface:
  - add_magic_protocol
  - add_protocol
  - add_reaction_handler
  - is_magic
  - is_running
  - node
  - put
  - react
  - reaction_types
  - request_types
  - start
  - stop
- Class ServiceProtocol:
  - put
  - react
  - service_interface

### qia/lib/python3.10/site-packages/netsquid_magic/sleeper.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: (none)
- Public classes: Sleeper
- Class Sleeper:
  - sleep

### qia/lib/python3.10/site-packages/netsquid_magic/state_delivery_sampler.py
- Root: netsquid_magic
- Public constants: (none)
- Public functions: success_prob_and_fidelity_from_heralded_state_delivery_sampler_factory
- Public classes: BitflipStateSamplerFactory, DepolariseStateSamplerFactory, DepolariseWithFailureStateSamplerFactory, DoubleClickDeliverySamplerFactory, HeraldedStateDeliverySamplerFactory, IStateDeliverySamplerFactory, PerfectStateSamplerFactory, SingleClickDeliverySamplerFactory, StateDeliverySampler
- Class BitflipStateSamplerFactory:
  - (no public methods detected)
- Class DepolariseStateSamplerFactory:
  - (no public methods detected)
- Class DepolariseWithFailureStateSamplerFactory:
  - (no public methods detected)
- Class DoubleClickDeliverySamplerFactory:
  - (no public methods detected)
- Class HeraldedStateDeliverySamplerFactory:
  - create_state_delivery_sampler
- Class IStateDeliverySamplerFactory:
  - create_state_delivery_sampler
- Class PerfectStateSamplerFactory:
  - (no public methods detected)
- Class SingleClickDeliverySamplerFactory:
  - (no public methods detected)
- Class StateDeliverySampler:
  - sample


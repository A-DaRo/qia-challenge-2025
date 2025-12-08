# E-HOK Baseline Codebase Review (8 Dec 2025)

## 1. Scope and Methodology

This document reviews the **baseline** E-HOK implementation under `ehok/`, focusing on:

- High-level software architecture and adherence to object-oriented design principles.
- Formal coding aspects (type safety, Numpydoc-style documentation, and interface usage).
- Identification of architectural *code smells* and potential mathematical inconsistencies.
- A proposed **restructured architecture** that better supports iterative evolution toward advanced extensions.
- Concrete coding guidelines that can be enforced via tests (e.g. `test_foundation.py`).

The review is informed by the baseline specifications in `docs/e-hok-baseline.md` and the literature review in `docs/e-hok-baseline-literature-review.md`.

---

## 2. Current Baseline Architecture (High-Level)

### 2.1 Module Overview

The baseline codebase is organized as follows:

- `ehok/core/`
  - `constants.py`: protocol parameters (QBER thresholds, LDPC parameters, logging config).
  - `data_structures.py`: `ObliviousKey`, `MeasurementRecord`, `ProtocolResult`.
  - `exceptions.py`: hierarchy of `EHOKException`, `SecurityException`, etc.
  - `sifting.py`: `SiftingManager` for basis matching and QBER estimation.
- `ehok/interfaces/`
  - `commitment.py`: `ICommitmentScheme`.
  - `reconciliation.py`: `IReconciliator`.
  - `privacy_amplification.py`: `IPrivacyAmplifier`.
- `ehok/implementations/`
  - `commitment/sha256_commitment.py`, `merkle_commitment.py`.
  - `reconciliation/ldpc_reconciliator.py`.
  - `privacy_amplification/toeplitz_amplifier.py`.
- `ehok/quantum/`
  - `basis_selection.py`, `batching_manager.py`, `measurement.py`.
- `ehok/protocols/`
  - `alice.py`: `AliceEHOKProgram` (full 5-phase baseline flow).
  - `bob.py`: `BobEHOKProgram` (full 5-phase baseline flow).
- `ehok/utils/`
  - `logging.py`, `classical_sockets.py`.
- `ehok/tests/`
  - Foundation and component tests, including `test_foundation.py`.

### 2.2 Positive Design Aspects

- **Clear phase decomposition** in `AliceEHOKProgram` and `BobEHOKProgram` (`_phase1_quantum_generation` through `_phase5_privacy_amplification`).
- **Separated interfaces** under `ehok/interfaces` with concrete baseline implementations under `ehok/implementations`.
- **Centralized configuration** in `core/constants.py` with descriptive docstrings.
- **Well-typed data structures** in `core/data_structures.py` with validation via `__post_init__` and covered by tests in `test_foundation.py`.
- **Consistent logging** via `LogManager.get_stack_logger`, avoiding `print()`.

---

## 3. Identified Code Smells and Architectural Issues

This section concentrates on structural and security-relevant concerns, not minor stylistic nits.

### 3.1 Mixed Responsibilities in Protocol Programs

**Observation:** `AliceEHOKProgram` and `BobEHOKProgram` currently embed:

- High-level protocol sequencing (phases I–V).
- Detailed **quantum operations** (batch generation, measurement, NetSquid flush calls).
- Detailed **classical logic** (sifting, commitment verification, QBER checks).
- Direct construction and use of concrete implementations (`SHA256Commitment`, `LDPCReconciliator`, `ToeplitzAmplifier`).

**Issue (SRP & Open/Closed Principle):**

- Each Program class has multiple responsibilities: orchestrating the protocol, performing quantum I/O, and selecting algorithmic strategies.
- Swapping to NSM-based commitments, MET-LDPC, or alternative privacy amplifiers will require **modifying** these core protocol classes instead of injecting new strategies, which violates the **Open/Closed Principle**.

**Consequence:**

- Difficult to compare baselines vs. extensions in a controlled manner.
- Harder to reason formally about security properties for each phase because protocol logic and algorithm choices are tightly coupled.

### 3.2 Incomplete Encapsulation of Security-Critical Configuration

**Observation:** `core/constants.py` defines thresholds and parameters (e.g. `QBER_THRESHOLD`, `TEST_SET_FRACTION`, `LDPC_CODE_RATE`) as mutable module-level constants.

**Issues:**

- These values can be imported and modified at runtime without centralized validation.
- There is no `ProtocolConfig` dataclass to capture a **coherent configuration snapshot** for a given run.

**Consequence:**

- Makes it harder to reproduce simulation runs and to reason about which parameter sets correspond to which security guarantees.
- Hinders later extension to finite-key analysis, where key length formulas must be explicitly tied to configured parameters.

### 3.3 Asymmetry and Approximation in Bob’s `knowledge_mask`

**Observation (from `bob.py`, Phase V):**

```python
# Bob doesn't know bits at mismatched basis positions (I_1)
# But knowledge_mask is for the FINAL key, not the raw key
# For simplicity in baseline: mark proportion of final key as unknown
# This is a simplified model; in reality, the oblivious property
# is more nuanced (related to which raw bits contributed to final bits)
```

**Issue (Mathematical Correctness):**

- The current baseline **does not track exactly** how the Toeplitz privacy amplification maps raw indices (particularly $I_1$) to positions in the final key.
- Instead, an approximate proportional mapping is used to fill Bob’s `knowledge_mask`.

**Consequence:**

- The `ObliviousKey` object **overstates structure**: it suggests precise bit-wise knowledge/ignorance as in Lemus et al., but the current mask is only an approximation.
- This can be misleading for higher-level protocols (e.g. OT extension) that assume the `knowledge_mask` is mathematically consistent with the linear transformation used in privacy amplification.

**Recommendation:**

- Explicitly document this as a **baseline approximation** and prevent higher-level protocols from assuming bit-exact obliviousness until the mask construction is upgraded to follow the actual Toeplitz linear map.

### 3.4 Tight Coupling to Specific LDPC Matrices

**Observation:**

- Both Alice and Bob call a local `_load_ldpc_matrix` function (not shown in the excerpt) and then perform ad-hoc handling:

```python
if sifted_length > H.shape[1]:
    logger.warning(
        f"Truncating key from {sifted_length} to {H.shape[1]} to match LDPC matrix"
    )
    alice_key = alice_key[:H.shape[1]]
```

**Issues:**

- The LDPC code length is not treated as a first-class parameter in a shared `ReconciliationConfig`.
- Truncation is performed silently (with only a warning), which **changes the semantics** of the key bits used for reconciliation and subsequently for privacy amplification.

**Consequence:**

- Potential **mathematical inconsistency** between the conceptual key (as defined in the literature) and the implemented key (actually used in the LDPC code and then in privacy amplification).
- If the truncated bits are not accounted for when computing leakage and final key length, security bounds may be misestimated.

### 3.5 Direct Use of Assertions for Runtime Validation

**Observation:**

- `ObliviousKey.__post_init__` and `MeasurementRecord.__post_init__` use plain `assert` statements.

**Issues:**

- Python’s `assert` is a debugging aid, not a robust runtime validation mechanism; it can be disabled with `-O` (optimize) flags.

**Consequence:**

- In optimized runs (or when integrated into other stacks), invalid data might slip through without raising errors.

**Recommendation:**

- Replace `assert` with explicit exception types (e.g. `ValueError`, or specialized `EHOKException` subclasses) for invariants that must always hold.

### 3.6 Implicit Randomness Contracts in `SiftingManager`

**Observation:**

- In `bob.py`, Phase III:

```python
# NOTE: Both parties use SiftingManager with default seed (None)
# This means they will get the same random selection if I_0 is the same
```

**Issue:**

- This relies on an implicit assumption about the RNG seeding and call order across Alice and Bob.
- Any deviation (e.g. logging, debugging, or added randomness elsewhere) can break synchronization of `test_set` and `key_set`.

**Consequence:**

- Potentially subtle **synchronization bugs** and inconsistent views of which positions belong to test vs key sets.
- Harder to later extend to deterministic or cryptographically seeded selection (as required for cut-and-choose security proofs).

**Recommendation:**

- Promote an explicit `SamplingStrategy` object/config that is instantiated from a shared seed and passed into `SiftingManager`.

### 3.7 Limited Formalization of Protocol Result Semantics

**Observation:**

- `ProtocolResult` aggregates summary statistics, but there is no explicit mapping from these counts to the theoretical quantities used in security proofs (e.g. $N$, $|I_0|$, $|T|$, `leak_EC`).

**Issue:**

- Without clear semantic linkage, a later finite-key analysis module has to reconstruct semantics from counts in an ad-hoc manner.

**Recommendation:**

- Document and enforce invariants such as:
  - `raw_count ≥ sifted_count ≥ test_count + final_count`.
  - `final_count == oblivious_key.final_length` when `success=True`.
- Add assertions/tests for these invariants.

---

## 4. Proposed Restructured Architecture

The goal is to move from a monolithic "baseline" implementation to an **extensible protocol framework** where advanced features (NSM commitments, MET-LDPC, alternative PA) can be added without rewriting core logic.

### 4.1 Layered Architecture

1. **Data & Config Layer** (`ehok/core`)
   - `ProtocolConfig` dataclass encapsulating:
     - Quantum parameters (batch size, total pairs, memory limits).
     - Security parameters (`QBER_THRESHOLD`, target $\varepsilon$, `TEST_SET_FRACTION`).
     - Reconciliation parameters (code rate, matrix choice, max iterations).
     - Privacy amplification parameters (security margin, Toeplitz dimensions).
   - `ExecutionMetrics` dataclass extending `ProtocolResult` with derived quantities.

2. **Interfaces / Strategies Layer** (`ehok/interfaces`, `ehok/implementations`)
   - Existing interfaces (`ICommitmentScheme`, `IReconciliator`, `IPrivacyAmplifier`) remain but are **wired** via factories:
     - `CommitmentFactory(config) -> ICommitmentScheme`.
     - `ReconciliatorFactory(config) -> IReconciliator`.
     - `PrivacyAmplifierFactory(config) -> IPrivacyAmplifier`.
   - New strategy types:
     - `ISamplingStrategy` for selecting test sets (`T`) and key sets.
     - `INoiseEstimator` for mapping observed QBER into leakage estimates.

3. **Quantum I/O Layer** (`ehok/quantum`)
   - `QuantumPhaseRunner` (or `EPRPhaseRunner`) encapsulating all NetSquid-specific batch management and measurement logic.
   - Programs (`AliceEHOKProgram`, `BobEHOKProgram`) delegate *all* qubit handling to this layer.

4. **Protocol Orchestration Layer** (`ehok/protocols`)
   - `EHOKRole` base class with abstract methods:
     - `_phase1_quantum(self, quantum_runner)`
     - `_phase2_commitment(self, commitment_scheme)`
     - `_phase3_sifting(self, sifting_manager, sampling_strategy)`
     - `_phase4_reconciliation(self, reconciliator)`
     - `_phase5_privacy_amplification(self, amplifier)`
   - `AliceBaselineEHOK(EHOKRole)` and `BobBaselineEHOK(EHOKRole)` implement baseline behavior by composing strategies.
   - Future `AliceNSMEHOK`, `BobNSMEHOK`, etc. can override only specific phases or inject new strategies.

5. **Analysis & Verification Layer** (`ehok/analysis` – new)
   - Functions to compute theoretical quantities from `ProtocolResult` and `ProtocolConfig`.
   - Optional future integration with finite-key formulas and leakage accounting.

### 4.2 Benefits w.r.t Open/Closed Principle

- Existing baseline strategies **remain unchanged** when extensions are added.
- New implementations (e.g. MET-LDPC) are introduced as new concrete classes of `IReconciliator` and registered in factories, without modifying `AliceEHOKProgram`/`BobEHOKProgram`.
- Security proofs and simulations can test multiple configurations by swapping strategies through a `ProtocolConfig` object.

---

## 5. Coding Guidelines for Strict Enforcement

This section enumerates **precise rules** that can be enforced by tests (e.g. extended `test_foundation.py`) and by static analysis.

### 5.1 General Design Rules

1. **Strategy Injection Only:**
   - Protocol classes (`ehok/protocols/*`) must **not** instantiate concrete commitment/reconciliation/PA implementations directly.
   - Instead, they receive `ICommitmentScheme`, `IReconciliator`, and `IPrivacyAmplifier` instances via constructor or factory functions.

2. **No Hard-Coded Path Logic:**
   - Paths to LDPC matrices, network configs, etc. must be derived from `ProtocolConfig` or environment variables, not hard-coded inside protocol methods.

3. **Explicit Config Snapshot:**
   - Every protocol run must be parameterized by a `ProtocolConfig` instance; tests should assert that `ProtocolResult` JSON-serializes together with the config.

4. **Side-Effect Transparency:**
   - `__post_init__` methods in dataclasses must **only validate** and must not trigger I/O or randomness.

### 5.2 Type and Invariant Enforcement

1. **No `assert` for Public Invariants:**
   - Replace `assert` in public dataclasses (e.g. `ObliviousKey`, `MeasurementRecord`) with explicit exceptions.
   - Add tests that confirm invalid inputs raise the correct exception types.

2. **Shape and Semantics for `ObliviousKey`:**
   - `len(key_value) == final_length == knowledge_mask.size` must hold.
   - When `success=True` in `ProtocolResult`, enforce via tests that `final_count == oblivious_key.final_length`.

3. **Protocol Count Invariants:**
   - Add tests that confirm relations such as:
     - `raw_count >= sifted_count >= test_count + final_count`.
     - `qber` in `[0, 1]`.

### 5.3 Documentation and Style (Numpydoc)

1. **Numpydoc Docstrings:**
   - All public classes and functions must use Numpydoc sections: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` as appropriate.
   - `test_foundation.py` can add reflection-based tests that scan modules and assert presence of docstrings on public APIs.

2. **Explicit Types in Signatures:**
   - Use full type hints, e.g. `np.ndarray` for arrays, `Sequence[int]` for index sets.

3. **No `print()` in Library Code:**
   - Enforce via tests that `print(` does not appear under `ehok/` (except in examples).

### 5.4 Randomness and Determinism

1. **Explicit RNG Sources:**
   - All randomness used for:
     - Basis selection,
     - Test set selection,
     - Toeplitz seed generation

   must derive from explicit RNG objects passed via config or constructed from shared seeds.

2. **Deterministic Sifting:**
   - Tests should assert that, given a fixed `ProtocolConfig` and seed, repeated runs of the same role yield **identical** `test_set` and `key_set` indices.

### 5.5 Knowledge Mask Correctness (Future Enforcement)

1. **Baseline Acknowledged Approximation:**
   - For now, add a flag in `ProtocolConfig` (e.g. `approximate_knowledge_mask=True`) and document this clearly in `ObliviousKey` docstrings.

2. **Exact Mapping for Extensions:**
   - Future tests must ensure that, when `approximate_knowledge_mask=False`, the knowledge mask is derived from the **actual linear map** used in privacy amplification (by tracking which raw indices influence each final bit).

---

## 6. Suggested Additions to `test_foundation.py`

To make the above guidelines enforceable, the following concrete tests can be added:

1. **Dataclass Invariant Tests:**
   - Replace assertion-based failures with explicit exceptions and test their raising.

2. **Docstring Presence Test:**
   - Introspect `ehok.core`, `ehok.interfaces`, and `ehok.implementations` to ensure all public classes/methods have non-empty docstrings.

3. **No Direct Implementation Instantiation in Protocols:**
   - A structural test that inspects AST of `ehok/protocols/*.py` and fails if `SHA256Commitment(`, `LDPCReconciliator(`, or `ToeplitzAmplifier(` are constructed directly.

4. **Print Ban Test:**
   - Search for the string `"print("` in `ehok/` (excluding `examples/`) and fail if found.

5. **Config Binding Test:**
   - Ensure there exists a `ProtocolConfig` dataclass and that `AliceEHOKProgram` and `BobEHOKProgram` accept it (or equivalent) in their constructors.

6. **Count Invariant Test:**
   - After running a minimal simulation (possibly mocked quantum layer), assert invariants on `ProtocolResult` counts.

---

## 7. Summary of Key Architectural Failures

- **Overloaded protocol classes** that mix orchestration, quantum I/O, and specific algorithm implementations, violating SRP and Open/Closed Principle.
- **Approximate treatment** of Bob’s `knowledge_mask`, which is not yet formally tied to the Toeplitz linear transformation and thus only partially reflects the mathematical oblivious-key definition.
- **Ad-hoc LDPC handling** (truncation, implicit configuration) that can undermine the formal correspondence between the implemented key and theoretical security proofs.
- **Implicit randomness contracts** in sifting and sampling strategies that are fragile and make formal reasoning about cut-and-choose style sampling harder.

The proposed restructured architecture and coding guidelines aim to **stabilize** the baseline as a clean reference implementation while making it straightforward to add and compare advanced E-HOK extensions without introducing new architectural debt.
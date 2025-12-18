# Caligo Extended Test Strategy Specification (Unit + Multi-Package Integration)

**Document Type:** Testing Strategy Specification (verifiable, non‑E2E)

**Target Audience:** Caligo developers writing/maintaining tests in `qia-challenge-2025/caligo/tests/`

**Scope:** Increase line coverage and test quality by adding (1) unit tests for low-coverage modules and (2) multi-package integration tests that validate **phase-contract DTO usage** and **protocol conventions** (Phase I→II, II→III, III→IV). **End-to-end SquidASM simulations are explicitly out of scope** for this document.

**Primary Inputs**
- Current coverage report (Dec 18, 2025)
- Architecture constraints in [qia-challenge-2025/docs/caligo/caligo_architecture.md](caligo_architecture.md)
- Protocol flow & invariants in [qia-challenge-2025/docs/$\binom{2}{1}$-OT-overview.md](../$\binom{2}{1}$-OT-overview.md)
- Phase contract dataclasses in [qia-challenge-2025/caligo/caligo/types/phase_contracts.py](../../caligo/caligo/types/phase_contracts.py)
- Literature grounding (NSM/WSE/OT context): [qia-challenge-2025/docs/literature/Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation.md](../literature/Generation%20and%20Distribution%20of%20Quantum%20Oblivious%20Keys%20for%20Secure%20Multiparty%20Computation.md)

**Note on baseline docs**
- Some internal SquidASM integration notes reference `e-hok-baseline.md` and `QOK_for_SMC.md`. If those files are not present in this repository snapshot, use the combination of the protocol overview, the literature document above, and the notes in `qia-challenge-2025/docs/squidasm_docs/ehok_specific/` as the normative grounding for test intent.

---

## 0. Definitions & Conventions (Normative)

This section defines the dictionary (ontology) and cross-package data conventions that tests MUST enforce.

### 0.1 Protocol phases and canonical DTOs

Caligo is organized as a phase pipeline with **explicit boundary contracts**:

- **Phase I (Quantum)** output: `QuantumPhaseResult`
- **Phase II (Sifting & Estimation)** output: `SiftingPhaseResult`
- **Phase III (Reconciliation)** output: `ReconciliationPhaseResult`
- **Phase IV (Amplification / OT formatting)** output: `AliceObliviousKey`, `BobObliviousKey`, and `ObliviousTransferOutput`

**REQ-DTO-000 (Single wire format):** Phase boundary tests MUST treat these DTOs as the only supported cross-package “wire format” (within Python). If a phase produces additional ad-hoc dicts/tuples that cross package boundaries, that is a test smell.

### 0.2 Bitstring representation conventions

The implementation uses three representations that MUST remain compatible:

- `np.ndarray` of dtype `np.uint8`, values in {0,1} (vectorized operations)
- `bitarray.bitarray` (DTO-level “key-as-bits” types)
- `bytes` as raw memory (used by some reconciliation entry points)

**REQ-BITS-001 (Round-trip stability):** For any bitstring $x$ of length $n$ with values in {0,1}, converting `np.uint8 bits → bitarray → np.uint8 bits` MUST preserve length and bit values exactly.

**REQ-BITS-002 (Byte-view safety):** If `bytes` is used as a carrier for bits, tests MUST assert that the code path treats the bytes as a `np.uint8` array of 0/1 values (not packed bits). Mixing packed-bit and byte-per-bit encodings is a protocol-breaking bug.

### 0.3 Error schema convention

When input violates a boundary contract:

- DTO postconditions MUST raise `ContractViolation`.
- Parameter invariants MUST raise `InvalidParameterError` (or `ValueError` where established).

**REQ-ERR-001 (Diagnosable failures):** Tests MUST assert both the exception type and that the message identifies the violated postcondition/invariant identifier (e.g., `POST-Q-001`).

## 1. Goals & Non‑Goals

### 1.1 Goals
- **Raise total line coverage above 81%** by prioritizing the highest-miss files and the highest-risk code paths.
- Improve **test quality** by validating:
  - **Safety properties** ("something bad never happens"), e.g. contract invariants, hash checks, matrix checksum sync.
  - **Liveness properties** ("something good eventually happens"), e.g. blind reconciliation produces a result or fails deterministically with metadata.
- Add **multi-package tests** that prove the **DTO contracts are used correctly** across phase boundaries.
- Ensure tests are deterministic (fixed RNG seeds) and do not rely on external services.

### 1.2 Non‑Goals
- No full SquidASM/NetSquid network runs.
- No performance benchmarking.
- No cryptographic proofs; tests validate implementation contracts and conventions.

---

## 2. Coverage Triage (What to Hit First)

### 2.1 Highest impact by missed lines
Prioritize in this order, because it combines low coverage with centrality:

1) **Reconciliation factory + wiring**
- `caligo/reconciliation/factory.py` (37% coverage, many missed lines)

2) **Quantum phase helpers**
- `caligo/quantum/epr.py` (64% coverage)
- `caligo/quantum/measurement.py` (85% coverage)

3) **Simulation parameter model**
- `caligo/simulation/physical_model.py` (72% coverage)

Also high-value because they are shared “oracles” across simulation/security/reconciliation:
- `caligo/simulation/noise_models.py` (80% coverage)
- `caligo/utils/math.py` (74% coverage)

4) **Matrix compilation/caching**
- `caligo/reconciliation/matrix_manager.py` (77% coverage)
- `caligo/reconciliation/compiled_matrix.py` (86% coverage)

5) **Amplification formatter correctness paths**
- `caligo/amplification/formatter.py` (96% coverage, a few uncovered correctness/error branches)

6) **Offline scripts (optional coverage)**
- `caligo/scripts/generate_ldpc_matrices.py` and `caligo/scripts/peg_generator.py` are at 0%. These are offline tools; treat as **secondary**. Add minimal tests for core pure functions (degree distribution validation, selection fallback) without running full PEG generation.

---

## 3. Test Design Principles (Normative)

### 3.1 Determinism & reproducibility
- Every test that uses randomness MUST set a seed via `np.random.default_rng(seed)` or fixtures.
- Never depend on wall-clock time.

### 3.2 Contract-first testing
All phase boundary DTOs are **design-by-contract**. Tests MUST validate:
- Happy path objects satisfy postconditions.
- Invalid objects fail fast with `ContractViolation`.

Additionally, tests MUST validate that producers populate DTO fields with the expected **dtype**, **shape**, and **units** (e.g., timestamps represent simulation nanoseconds, not wall-clock seconds).

### 3.3 Prefer “small integration”
Multi-package tests should integrate **2–4 modules** at a time (e.g., encoder→decoder→verifier), not entire simulations.

**Definition (Small integration):** An integration test that crosses at least one package boundary (e.g., `quantum`→`types`, `simulation`→`reconciliation`, `reconciliation`→`amplification`) while still running fully in pytest without starting a SquidASM/NetSquid network.

### 3.4 Test naming & markers
- Unit tests: no markers.
- Multi-package integration tests: use `@pytest.mark.integration` (already used in reconciliation tests).

---

## 4. Unit Test Additions (by module)

### 4.1 `caligo/reconciliation/factory.py` (highest priority)

**Risk/Reason:** This is configuration + dependency wiring and contains the project’s "runtime selection" behavior. Currently under-tested.

#### 4.1.1 Unit tests for `ReconciliationType`
Add tests in a new file: `qia-challenge-2025/caligo/tests/reconciliation/test_factory.py`

- **REQ-FAC-001**: `ReconciliationType.from_string` accepts case/whitespace variants.
  - Inputs: `"baseline"`, `" BASELINE "`, `"Blind"`, `"interactive"`.
  - Verify enum equality and `requires_qber_estimation` semantics.

- **REQ-FAC-002**: `ReconciliationType.from_string` rejects unknown type.
  - Input: `"foo"`.
  - Expect `ValueError` containing valid types.

#### 4.1.2 Unit tests for `ReconciliationConfig`
- **REQ-FAC-010**: `ReconciliationConfig.__post_init__` rejects invalid bounds.
  - `frame_size < 256`, `max_iterations < 1`, `target_rate` out of bounds, `safety_margin` out of bounds, `max_blind_rounds` out of bounds.

- **REQ-FAC-011**: `ReconciliationConfig.from_dict` correctly maps `type` → enum and preserves extras.
  - Provide config dict with known keys plus e.g. `{"unknown_param": 123}`.
  - Expect `config.to_dict()` includes `unknown_param` and `type` is normalized.

- **REQ-FAC-012**: `requires_qber_estimation`/`skips_qber_estimation` reflect chosen type.

#### 4.1.3 Unit tests for `create_reconciler`
Because `BaselineReconciler` is a placeholder, tests focus on **selection** and **guard behavior**.

- **REQ-FAC-020**: baseline returns `BaselineReconciler`; calling `reconcile` without QBER raises `ValueError`.
- **REQ-FAC-021**: interactive raises `NotImplementedError`.
- **REQ-FAC-022**: blind returns `BlindReconciler` and its metadata states `qber_estimation_required=False`.

- **REQ-FAC-023**: `create_reconciler(..., channel_profile=ChannelNoiseProfile(...))` is accepted and does not change type selection semantics.

#### 4.1.3b Unit tests for YAML convenience helpers

- **REQ-FAC-040**: `ReconciliationConfig.from_yaml_file` loads a `reconciliation:` section and correctly maps `type`.
  - Use `tmp_path` to write a minimal YAML file.
- **REQ-FAC-041**: `create_reconciler_from_yaml` returns the same reconciler type as `create_reconciler(ReconciliationConfig.from_yaml_file(...))`.

#### 4.1.4 Unit tests for `BlindReconciler._get_orchestrator` wiring
This method imports several objects and constructs a matrix manager/leakage tracker.

- **REQ-FAC-030**: Use `monkeypatch` to replace `MatrixManager`, `LeakageTracker`, and `ReconciliationOrchestrator` with fakes; assert:
  - matrix path uses `config.ldpc_matrix_path` if set, else default.
  - orchestrator config fields match `frame_size`, `max_iterations`, `max_retries`.

This increases coverage without needing LDPC assets.

**Implementation note:** As of the current code snapshot, `factory.py` constructs `MatrixManager(base_path=...)` while `caligo/reconciliation/matrix_manager.py` exposes `MatrixManager.from_directory(...)` and `MatrixManager.__init__(pool: MatrixPool)`. This is likely wiring drift; the tests above are intentionally designed to expose it early.

---

### 4.2 `caligo/quantum/epr.py`

**Risk/Reason:** Contains generator logic and error handling that can easily regress.

Add tests in `qia-challenge-2025/caligo/tests/test_quantum/test_epr.py` (or extend existing).

- **REQ-EPR-001**: `generate_batch_sync` increments counters and returns correct shapes.
- **REQ-EPR-002**: `reset_counters` resets `total_generated` and batch counter behavior.
- **REQ-EPR-010**: `generate_batch` retries and raises `EPRGenerationError` after N attempts.
  - Provide fake `epr_socket.create_keep` that throws.
  - Provide `context=None` to avoid SquidASM.
- **REQ-EPR-011**: successful `generate_batch` returns `EPRBatch` with list conversion.
  - Fake `create_keep` returns an iterable.

**Note:** `generate_batch` contains unusual timing extraction via `context.csocket.msg_from_peer()`; tests should cover both `context=None` and a stub context with `.connection.flush()`.

---

### 4.3 `caligo/quantum/measurement.py`

**Risk/Reason:** Measurement buffering and fallback behavior (ImportError path) affect correctness and determinism.

Add tests in `qia-challenge-2025/caligo/tests/test_quantum/test_measurement_executor.py`.

- **REQ-MEAS-001**: `MeasurementBuffer.add_outcome` grows capacity and preserves data.
  - Start with small capacity=2, add 3 outcomes.
- **REQ-MEAS-002**: `MeasurementBuffer.add_batch` grows repeatedly until it fits.
- **REQ-MEAS-003**: `get_batch` returns copies (mutating return does not mutate buffer).

- **REQ-MEAS-010**: `MeasurementExecutor.measure_qubit_sync` respects `simulated_outcome`.
- **REQ-MEAS-011**: `MeasurementExecutor.measure_batch_sync` respects provided `simulated_outcomes` and uses correct dtype.
- **REQ-MEAS-012**: `MeasurementExecutor.get_results` returns aligned arrays.

- **REQ-MEAS-020**: `MeasurementExecutor.measure_qubit` ImportError fallback path is deterministic under monkeypatch.
  - Monkeypatch import to raise `ImportError` or run in environment without NetQASM.
  - Fix numpy RNG seed and assert outcome is 0/1 and buffer count increments.

---

### 4.4 `caligo/simulation/physical_model.py`

**Risk/Reason:** This module converts NSM parameters into derived QBER/capacity and into NetSquid model constructors.

Add tests in `qia-challenge-2025/caligo/tests/test_simulation/test_physical_model_contracts.py`.

#### 4.4.1 Invariants and derived properties
- **REQ-NSM-001**: invalid invariants raise `InvalidParameterError`.
  - storage_noise_r out of range, storage_rate_nu out of range, delta_t_ns <=0, channel_fidelity <=0.5, detection_eff_eta <=0.

- **REQ-NSM-010**: `depolar_prob == 1 - r`.
- **REQ-NSM-011**: `qber_simple` equals `(1-F)/2 + e_det`.
- **REQ-NSM-012**: `qber_channel` equals `compute_qber_erven(...)` output.
- **REQ-NSM-013**: `storage_security_satisfied` matches `storage_capacity * nu < 0.5`.

#### 4.4.2 Factory methods
- **REQ-NSM-020**: `from_erven_experimental` returns values in expected ranges and has `security_possible` consistent with conservative threshold.
- **REQ-NSM-021**: `for_testing` produces a minimal valid config.

#### 4.4.3 NetSquid factory functions (ImportError behavior)
- **REQ-NSM-030**: `create_depolar_noise_model` raises `ImportError` with clear message when NetSquid is absent.
- **REQ-NSM-031**: `create_t1t2_noise_model` same behavior.

This directly covers previously missed branches.

#### 4.4.4 PDC source helper functions (Erven-model utilities)

These are pure functions and should be tested without any simulator.

- **REQ-PDC-001**: `pdc_probability(n, mu)` returns 0 for n<0 and raises `ValueError` for mu<=0.
- **REQ-PDC-010**: `pdc_probability(n, mu)` is non-negative and decreases for large n (sanity check: truncated tail small by n=20 for μ in Erven range).
- **REQ-PDC-020**: `p_sent(mu)` returns a value in [0,1] and handles the p0→1 edge case.
- **REQ-PDC-030**: `p_b_noclick(mu, eta, p_dark)` returns a probability in [0,1] for typical parameter ranges.
- **REQ-PDC-031**: `p_b_noclick_min(mu, eta, p_dark)` returns a value ≥ `pdc_probability(0, mu)` and is in [0,1].

#### 4.4.5 Channel parameters (timing + fiber-loss utilities)

- **REQ-CHAN-001**: `ChannelParameters.transmittance` returns 1.0 at 0 dB and decreases with increasing loss.
- **REQ-CHAN-010**: `ChannelParameters.for_testing()` and `.from_erven_experimental()` produce internally consistent values (positive T1/T2/cycle_time).

---

### 4.5 `caligo/reconciliation/compiled_matrix.py`

**Risk/Reason:** Critical for performance and correctness in syndrome checks; has IO branch handling.

Add tests in `qia-challenge-2025/caligo/tests/reconciliation/test_compiled_matrix.py`.

- **REQ-CMAT-001**: `compile_parity_check_matrix` produces consistent `edge_count`, `max_variable_degree`.
  - Build a tiny CSR matrix manually (e.g., 3×5).

- **REQ-CMAT-010**: `compute_syndrome` matches sparse multiplication mod 2.
- **REQ-CMAT-011**: `count_syndrome_errors` equals Hamming distance between computed and target.
- **REQ-CMAT-012**: length mismatch raises `ValueError`.

- **REQ-CMAT-020**: cache round-trip.
  - Use `tmp_path` and `save_compiled_cache` then `load_compiled_cache` with correct checksum.
- **REQ-CMAT-021**: checksum mismatch returns `None`.

---

### 4.6 `caligo/reconciliation/matrix_manager.py`

**Risk/Reason:** Loading and synchronization are protocol-critical. Missing lines likely include error branches.

Add tests in `qia-challenge-2025/caligo/tests/reconciliation/test_matrix_manager_errors.py`.

- **REQ-MM-001**: `from_directory` raises `FileNotFoundError` on missing directory.
- **REQ-MM-002**: missing file for a required rate raises `FileNotFoundError`.
  - Create temp directory with only a subset.

- **REQ-MM-010**: `get_matrix` raises `KeyError` with available rates.
- **REQ-MM-011**: `verify_checksum` true/false.

- **REQ-MM-020**: `get_compiled` caches compiled matrices.
  - Use small generated matrix; call twice and assert identity.

*Note:* The current test suite already loads real assets via `constants.LDPC_MATRICES_PATH`. Keep these new tests asset‑free by using `tmp_path` and small `.npz` matrices.

---

### 4.7 `caligo/amplification/formatter.py`

**Risk/Reason:** Correctness check enforces OT invariant `Sᴄ == S_choice`. Uncovered lines likely include failure path.

Add tests in `qia-challenge-2025/caligo/tests/test_amplification/test_formatter_contracts.py`.

- **REQ-FMT-001**: `compute_alice_keys` rejects too-short inputs.
- **REQ-FMT-002**: `compute_bob_key` rejects invalid `choice_bit`.
- **REQ-FMT-010**: `format_final_output` raises `ContractViolation` when Bob’s key does not match choice.
  - Create `AliceOTOutput`/`BobOTOutput` with deliberate mismatch.

---

### 4.8 Additional high-impact modules (shared oracle + missed branches)

These modules have relatively high missed-line counts and directly influence cross-package correctness. They should be tested even if they are not the *lowest* coverage percentages.

#### 4.8.1 `caligo/simulation/noise_models.py`

Add tests in `qia-challenge-2025/caligo/tests/test_simulation/test_noise_models_contracts.py`.

- **REQ-NOISE-001**: `ChannelNoiseProfile.__post_init__` enforces invariants.
  - invalid `source_fidelity <= 0.5`, `detector_efficiency <= 0`, `detector_error > 0.5`, `dark_count_rate > 1`, `transmission_loss >= 1`.
- **REQ-NOISE-010**: `qber_conditional` returns 0.5 when `p_detect <= 0`.
  - use `detector_efficiency=0` and `dark_count_rate=0`.
- **REQ-NOISE-011**: `signal_to_noise_ratio` returns `inf` when dark contribution is zero.
  - use `dark_count_rate=0` or `detector_efficiency=1`.
- **REQ-NOISE-020**: `to_nsm_parameters` preserves the QBER model fields.
  - assert `ChannelNoiseProfile.total_qber == NSMParameters.qber_channel` for mapped parameters.
- **REQ-NOISE-021**: `get_diagnostic_info` includes required keys and is internally consistent (`security_margin == 0.11 - total_qber`).

#### 4.8.2 `caligo/utils/math.py`

Add tests in `qia-challenge-2025/caligo/tests/test_utils/test_math_contracts.py`.

- **REQ-MATH-001**: `binary_entropy` rejects p outside [0,1] and returns 0 for p∈{0,1}.
- **REQ-MATH-010**: `channel_capacity` rejects qber outside [0,0.5].
- **REQ-MATH-020**: `suggested_ldpc_rate_from_qber` returns highest rate for very low QBER, lowest for high QBER; include non-zero `safety_margin` behavior.
- **REQ-MATH-030**: `blind_reconciliation_initial_config` chooses the intended adaptation regime at threshold points (0.02, 0.05, 0.08).
- **REQ-MATH-040**: `finite_size_penalty` enforces preconditions (n>0, k>0, 0<ε<1) and is monotone decreasing in k for fixed n.
- **REQ-MATH-050**: `gamma_function` enforces r∈[0,1] and returns finite values at r∈{0,1}.
- **REQ-MATH-060**: `key_length_bound` exercises both gamma=0 (QKD-style) and gamma≠0 (NSM-style) branches and floors at 0.

#### 4.8.3 `caligo/reconciliation/hash_verifier.py`

Add tests in `qia-challenge-2025/caligo/tests/reconciliation/test_hash_verifier_contracts.py`.

- **REQ-HASH-001**: constructor rejects non-positive hash length; `hash_bits` alias matches `output_bits`.
- **REQ-HASH-010**: `compute_hash_bytes` length is `ceil(bits/8)` and is deterministic for fixed seed.
- **REQ-HASH-011**: `verify_bytes` agrees with `verify` on the same input.
- **REQ-HASH-020**: mismatch bits causes verification failure (False), not an exception.

#### 4.8.4 `caligo/quantum/basis.py`

Add tests in `qia-challenge-2025/caligo/tests/test_quantum/test_basis_additional.py`.

- **REQ-BASIS-001**: `select_batch(n)` rejects n<=0.
- **REQ-BASIS-010**: seeded mode is deterministic across runs for the same seed.
- **REQ-BASIS-020**: `select_weighted` enforces p_x∈[0,1] and respects edge cases p_x=0 and p_x=1.
- **REQ-BASIS-030**: `compute_matching_mask` raises on length mismatch.

#### 4.8.5 `caligo/types/phase_contracts.py` (target missed contract branches)

Add tests in `qia-challenge-2025/caligo/tests/test_types/test_phase_contracts_negative.py`.

- **REQ-DTO-001**: `QuantumPhaseResult` rejects invalid values and mismatched lengths (POST-Q-001..004).
- **REQ-DTO-010**: `SiftingPhaseResult` rejects qber_adjusted mismatch with penalty (POST-S-002) and rejects qber_adjusted > hard limit (POST-S-003).
- **REQ-DTO-020**: `ReconciliationPhaseResult` rejects `hash_verified=False` (POST-R-002).
- **REQ-DTO-021**: `ReconciliationPhaseResult` rejects leakage exceeding cap when `leakage_within_cap=False` (POST-R-001).
- **REQ-DTO-030**: `ObliviousTransferOutput` rejects mismatch between Bob’s `sc` and Alice’s selected key (POST-OT-003).

This group is intentionally negative-test heavy: contract branches are primarily error-handling logic.

---

## 5. Multi‑Package Integration Tests (Phase Boundary & Protocol Convention)

These tests validate interactions among modules and enforce DTO boundary contracts.

### 5.1 Phase I → Phase II DTO compatibility (Quantum results → Sifting expectations)

**Objective:** Ensure that arrays produced by quantum tooling are shape/type compatible with `QuantumPhaseResult`, and that invalid shapes fail.

Add tests in `qia-challenge-2025/caligo/tests/test_types/test_phase_boundary_quantum_to_sifting.py`.

- **REQ-P12-001**: Build outcomes via `MeasurementExecutor.measure_batch_sync` and bases via `BasisSelector(seed=...)`; construct `QuantumPhaseResult`.
  - Assert: DTO accepts correct lengths and values in {0,1}.

- **REQ-P12-010**: Construct `QuantumPhaseResult` with mismatched lengths and expect `ContractViolation`.

- **REQ-P12-020**: Construct `QuantumPhaseResult` with invalid basis/outcome values (e.g., 2) and expect `ContractViolation`.

**Protocol convention:** `QuantumPhaseResult.generation_timestamp` is simulation time in nanoseconds; tests MUST treat it as an opaque float and MUST NOT depend on wall-clock.

This is a *multi-package* test: `quantum/*` → `types/phase_contracts`.

### 5.2 Phase II → Phase III DTO convention (Sifted bitarrays → reconciliation inputs)

**Objective:** Verify that the reconciliation layer can consume sifted keys represented as `bitarray` and that conversions to NumPy/bytes are consistent.

Add tests in `qia-challenge-2025/caligo/tests/test_phase_boundary_sifting_to_reconciliation.py`.

- **REQ-P23-001**: From a `SiftingPhaseResult`, convert `sifted_key_*` to NumPy `uint8` arrays and run a **single-block** reconciliation with existing `ReconciliationOrchestrator` (already integration-tested).
  - Postcondition: if `verified=True`, corrected payload equals Alice.

- **REQ-P23-010**: Construct `SiftingPhaseResult` with `qber_adjusted > QBER_HARD_LIMIT` and expect `ContractViolation`.
  - This ensures upstream abort would happen before reconciliation.

This validates DTO correctness and expected abort boundary.

**REQ-P23-020 (Bit representation boundary):** Convert `bitarray` → `np.uint8` → `bytes` and ensure the chosen reconciler consumes the same semantics. This test exists explicitly to prevent accidental adoption of packed-bit `bytes` encoding.

### 5.3 Phase III → Phase IV bitstring interaction (reconciled bits → Toeplitz formatter)

**Objective:** Ensure that reconciliation output can be partitioned into I₀/I₁ arrays and hashed to consistent OT outputs.

Add tests in `qia-challenge-2025/caligo/tests/test_phase_boundary_reconciliation_to_amplification.py`.

- **REQ-P34-001**: Create a `ReconciliationPhaseResult` with a reconciled `bitarray`, then:
  - Convert to `np.ndarray` (`uint8`).
  - Split into two segments representing I₀/I₁.
  - Use `OTOutputFormatter` to compute `AliceOTOutput` and `BobOTOutput`, then `format_final_output`.
  - Assert DTO invariant: Bob’s `sc` equals Alice’s selected `s0/s1`.

- **REQ-P34-010**: For deliberate mismatch, assert `ContractViolation`.

**REQ-P34-020 (DTO-level OT invariant):** After producing `AliceObliviousKey` and `BobObliviousKey`, construct `ObliviousTransferOutput` and assert its postconditions accept the object.

This provides a second-layer guard: formatter correctness AND DTO correctness.

This is a *multi-package* test: `types` + `amplification` (and optionally `reconciliation` if you produce keys via the orchestrator).

### 5.4 Reconciliation factory integration (factory → orchestrator → matrix manager)

**Objective:** Cover the currently low-tested runtime wiring while remaining non‑E2E.

Add tests in `qia-challenge-2025/caligo/tests/reconciliation/test_factory_integration.py`.

- **REQ-FINT-001**: `create_reconciler(ReconciliationType.BLIND)` returns reconciler that can run on one block when LDPC assets exist.
  - Use `constants.LDPC_MATRICES_PATH` (existing integration tests already rely on it).
  - Use deterministic Alice/Bob arrays with small error rate.
  - Validate metadata fields:
    - `qber_estimation_required` is `False`
    - `status` in {`success`,`failed`}

This directly raises coverage in the factory and validates intended behavior.

**Blocker note:** If the factory wiring is inconsistent with `MatrixManager` construction (see Section 4.1.4 note), this integration test is expected to reveal the drift. Resolve the drift before making this test a required CI gate.

### 5.5 Simulation → Reconciliation cross-package mapping (QBER/rate oracle)

**Objective:** Ensure the simulation-layer noise/parameter models are consistent with reconciliation-layer rate selection assumptions.

Add tests in `qia-challenge-2025/caligo/tests/test_phase_boundary_simulation_to_reconciliation.py`.

- **REQ-P0R-001**: Construct `ChannelNoiseProfile` and map to `NSMParameters` via `to_nsm_parameters`; assert that:
  - QBER values agree (`total_qber == qber_channel`).
  - `suggested_ldpc_rate(safety_margin=...)` returns a rate present in `MatrixManager.rates` (when a manager is available).
- **REQ-P0R-010**: A profile with `total_qber >= QBER_HARD_LIMIT` must be treated as infeasible (`is_feasible=False`) and must not be passed into reconciliation selection in higher-level code.

This is intentionally non-E2E: it checks the “oracle consistency” that would otherwise only be caught in a full simulation.

---

## 6. Offline Scripts (Optional / Secondary)

Because scripts are not runtime-critical, keep tests minimal and pure.

### 6.1 `caligo/scripts/peg_generator.py`
Add `qia-challenge-2025/caligo/tests/test_scripts/test_peg_generator_unit.py`.

- **REQ-PEG-001**: `DegreeDistribution` normalizes probabilities and rejects invalid input.
- **REQ-PEG-002**: `PEGMatrixGenerator.__init__` rejects invalid `n` and invalid `rate`.

Avoid running full `generate()` in CI if it is slow.

### 6.2 `caligo/scripts/generate_ldpc_matrices.py`
Add `qia-challenge-2025/caligo/tests/test_scripts/test_generate_ldpc_matrices_unit.py`.

- **REQ-GEN-001**: `_get_distributions` falls back to rate=0.50 when missing.
- **REQ-GEN-002**: `_load_degree_distributions` raises `FileNotFoundError` if YAML missing (monkeypatch path to temp).

---

## 7. Acceptance Criteria (Verifiable)

### 7.1 Coverage
- **AC-COV-001:** `caligo/reconciliation/factory.py` coverage increases substantially (target: ≥75% lines).
- **AC-COV-002:** `caligo/quantum/epr.py` coverage increases to ≥85%.
- **AC-COV-003:** `caligo/simulation/physical_model.py` coverage increases to ≥85%.
- **AC-COV-003b:** `caligo/utils/math.py` coverage increases to ≥90% (small, shared, high-impact oracle).
- **AC-COV-003c:** `caligo/simulation/noise_models.py` coverage increases to ≥90% (shared oracle).
- **AC-COV-004:** Total project coverage rises from 81% to **≥86%**.

### 7.2 Contract & Convention checks
- **AC-CON-001:** Phase-contract negative tests assert `ContractViolation` for malformed DTOs.
- **AC-CON-002:** OT correctness invariant (`Sᴄ == S_choice`) is tested in `formatter` and in Phase III→IV integration.

### 7.3 Determinism
- **AC-DET-001:** All newly added tests are deterministic across runs.

---

## 8. Implementation Notes (How to stage the work)

Recommended PR order to keep diffs reviewable:
1) Add `test_factory.py` + `test_factory_integration.py` (highest ROI)
2) Add EPR + measurement + basis unit tests
3) Add physical model + noise model + utils/math unit tests
4) Add compiled matrix + matrix manager tests
5) Add hash verifier + phase-contract negative tests
6) Add phase boundary integration tests
7) Add optional script tests

---

## 9. Open Questions (Resolve before adding some tests)

- Do we want to treat the offline scripts as “coverage-exempt” (e.g., via coverage omit), or explicitly bring them above 0%?
- Should the factory’s `MatrixManager` usage be unified? (In `factory.py` it constructs `MatrixManager(base_path=...)` but in `matrix_manager.py` the class method is `from_directory(...)`). Tests will reveal whether this is intentional or legacy drift.

- Should `POST-Q-005` (“timing_barrier_marked == True”) become a hard postcondition (raise) instead of a soft check? If yes, add negative tests that assert `ContractViolation` when False.

---

## 10. Traceability Matrix (Requirements → Tests)

- Factory selection & policy: REQ-FAC-001..030 → `tests/reconciliation/test_factory*.py`
- Quantum generation robustness: REQ-EPR-* → `tests/test_quantum/test_epr.py`
- Measurement buffering correctness: REQ-MEAS-* → `tests/test_quantum/test_measurement_executor.py`
- NSM parameter invariants: REQ-NSM-* → `tests/test_simulation/test_physical_model_contracts.py`
- PDC + channel helper functions: REQ-PDC-*/REQ-CHAN-* → `tests/test_simulation/test_physical_model_contracts.py`
- Noise profile invariants + mapping: REQ-NOISE-* → `tests/test_simulation/test_noise_models_contracts.py`
- Shared math oracle: REQ-MATH-* → `tests/test_utils/test_math_contracts.py`
- Hash verifier correctness: REQ-HASH-* → `tests/reconciliation/test_hash_verifier_contracts.py`
- Basis selection: REQ-BASIS-* → `tests/test_quantum/test_basis_additional.py`
- DTO postcondition negative coverage: REQ-DTO-* → `tests/test_types/test_phase_contracts_negative.py`
- Matrix compilation & caching: REQ-CMAT-* → `tests/reconciliation/test_compiled_matrix.py`
- Matrix manager error branches: REQ-MM-* → `tests/reconciliation/test_matrix_manager_errors.py`
- Amplification formatter OT invariant: REQ-FMT-* → `tests/test_amplification/test_formatter_contracts.py`
- Phase boundary interactions: REQ-P12/P23/P34/P0R-* → `tests/test_phase_boundary_*.py`


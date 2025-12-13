# Sprint 0 Specification: Foundation

**Document ID**: Sprint-0 / E-HOK-on-SquidASM

This document is the implementation-focused specification for Sprint 0, derived directly from the Master Roadmap and constrained to Sprint 0 scope (infrastructure + contracts + workflow).

---

## 1. Sprint 0 Overview

### Goal
Establish the project foundation required to execute Test-Driven Migration (TDM): a working CI scaffold, a reliable test harness, and strictly defined Phase I/O dataclass contracts that enable future parity testing and aggressive legacy removal.

### References
- Master roadmap (Sprint 0 tasks + deliverables): [master_roadmap.md](master_roadmap.md#41-sprint-0-foundation-days-1-3)
- CI gate intent (quality/integration/e2e): [master_roadmap.md](master_roadmap.md#23-cicd-gate-structure)
- TDM protocol (interfaces → parity tests → delete legacy): [master_roadmap.md](master_roadmap.md#24-test-driven-migration-protocol)
- Phase I: timing semantics ($\Delta t$ as causal barrier) + invariants/contracts: [phase_I_analysis.md](phase_I_analysis.md#1-literature-review--theoretical-basis), [phase_I_analysis.md](phase_I_analysis.md#7-formal-system-invariants-and-contracts)
- Phase II: ordered acknowledgement (“sandwich flow”) semantics: [phase_II_analysis.md](phase_II_analysis.md#12-temporal-ordering-semantics), [phase_II_analysis.md](phase_II_analysis.md#31-ordered-message-protocol-sandwich-flow)
- Phase III: leakage as wiretap cost ($|\Sigma|$) and one-way constraint: [phase_III_analysis.md](phase_III_analysis.md#11-the-efficiency-vs-security-tradeoff), [phase_III_analysis.md](phase_III_analysis.md#21-theoretical-corpus)
- Phase IV: NSM vs QKD bounds and min-entropy inputs: [phase_IV_analysis.md](phase_IV_analysis.md#12-the-nsm-security-model-vs-qkd), [phase_IV_analysis.md](phase_IV_analysis.md#11-core-concepts-mapping)

### Dependencies (Prerequisites)
- Repository has an installable Python project (already present via `pyproject.toml`).
- Unit test runner available (`pytest`).
- SquidASM/NetQASM importability in CI environment (or tests must isolate non-simulation logic from simulator imports).

---

### INFRA-001 — Configure CI pipeline (GitHub Actions)

**Description**
Create a CI pipeline that enforces the Roadmap’s Stage 1 “Quality Gate” on every **push** to the repository. This ensures the codebase remains clean, reproducible, and tested at every save point, preventing technical debt accumulation.

**Requirements (Measurable & Testable)**
1. **CI Triggers**:
   - MUST trigger on `push` events to the default branch (e.g., `main` or `master`).
2. **Environment**:
   - CI jobs MUST run on Linux.
   - CI jobs MUST use Python 3.10 (project baseline).
3. **Stage 1 Quality Gate** (MUST run the following sequence):
   - `mypy` in strict mode (per Roadmap intent; consistent with Phase-contract rigor).
   - `flake8`.
   - `pydocstyle --convention=numpy` (Roadmap requirement).
   - `pytest -m "not integration"` (or equivalent) as “unit gate”.
4. **Pass/Fail Condition**:
   - The job MUST fail (red status) if *any* of the above steps fail or return a non-zero exit code.
5. **Runtime Budget**:
   - Stage 1 MUST complete in ≤ 5 minutes on GitHub-hosted runners to ensure rapid feedback.

**Deliverables**
- `.github/workflows/ci.yml` implementing the Stage 1 Quality Gate.
- Documentation snippet in repo docs `qia-challenge-2025/docs/contracts` describing how to run the same checks locally.

**Success Criteria (How to Prove Completion)**
- **Test Type**: CI validation (integration of tooling).
- **Test Data**: Not applicable (tooling-level).
- **Expected Results**:
  - Pushing a clean commit results in a **Success (Green)** status in the GitHub "Actions" tab.
  - Pushing a commit with a deliberate type error, linting error, or docstring violation results in a **Failure (Red)** status.
- **Failure Points & Debugging**:
  - *Dependency failures*: missing dev dependencies in CI → verify extras installation (e.g., `[project.optional-dependencies].dev`).
  - *Mypy false positives*: isolate SquidASM imports behind `ignore_missing_imports` and/or use stubs; prefer tightening within `ehok/` first.
  - *Docstring validation noise*: ensure only intended paths are checked; avoid validating external vendored code.

---

### INFRA-002 — Define Phase I/O dataclasses (contracts)

**Description**
Define (or finalize) the canonical dataclass contracts at phase boundaries so later sprints can implement modules against stable interfaces and enable deterministic parity tests between legacy and SquidASM-native components.

**Requirements (Measurable & Testable)**
1. Contracts MUST be represented as `@dataclass` types with:
   - Full type hints for all fields.
   - Numpydoc docstrings (project standard).
   - Internal invariants validated in `__post_init__` (Design-by-Contract).
2. Contracts MUST cover at least these phase boundaries:
   - Phase I → Phase II (quantum measurement outputs + detection/commit metadata).
   - Phase II → Phase III (sifted keys + adjusted QBER inputs).
   - Phase III → Phase IV (reconciled key + leakage accounting).
   - Phase IV output (oblivious key material / OT structure).
3. Contract fields MUST enable enforcement of Phase I/II temporal invariants:
   - A representation for simulation timestamps and/or event ordering markers sufficient to verify “commit/ACK before basis reveal” and “$\Delta t$ barrier exists.”
   - Justification: NSM timing semantics ([phase_I_analysis.md](phase_I_analysis.md#11-konig-wehner--wullschleger-2012--unconditional-security-from-noisy-quantum-storage)) and Phase II ordering ([phase_II_analysis.md](phase_II_analysis.md#12-temporal-ordering-semantics)).
4. Contract fields MUST enable leakage accounting in later sprints:
   - Syndrome length and verification hash bits MUST be representable at Phase III/IV boundary.
   - Justification: wiretap-cost subtraction ([phase_III_analysis.md](phase_III_analysis.md#13-the-wiretap-channel-model), [phase_IV_analysis.md](phase_IV_analysis.md#23-the-error-correction-penalty)).
5. Contracts MUST be serializable to/from JSON-like structures for logging and debugging (implementation may be later), therefore:
   - Field types must be either primitives, `Enum`, or explicitly documented arrays/bytes with deterministic encoding.

**Deliverables**
- Phase boundary dataclasses consolidated in `ehok/core/data_structures.py` (per Roadmap deliverable) or a clearly documented equivalent location.
- A “contract map” table in Sprint 0 documentation (can be in this file’s PR) listing each dataclass and which phase boundary it supports.

**Success Criteria (How to Prove Completion)**
- **Test Type**: Unit tests + structural validation.
- **Test Data**:
  - At least 50 randomized valid instances per dataclass (sizes bounded, e.g., bit arrays of length 0, 1, 16, 1024).
  - At least 25 invalid instances per dataclass (wrong dtype, inconsistent lengths, invalid probability ranges).
- **Expected Results**:
  - Valid instances construct successfully.
  - Invalid instances fail deterministically with a specific exception type (`ValueError` or domain exception), and messages are actionable.
  - `mypy --strict` passes for all contract modules.
- **Failure Points & Debugging**:
  - *Nondeterminism*: tests that depend on simulator runtime ordering → isolate contract tests from simulation.
  - *Ambiguous semantics*: missing fields for ordering/timing → add explicit timestamp fields rather than inferring from logs.

**Legacy Interaction (Migration Strategy)**
- These contracts are the “WRAP” boundary in the Roadmap’s aggressive removal pattern.
- Parity tests in later sprints must compare outputs of legacy vs new implementations by converting both into the same contract dataclasses (byte-for-byte equality where deterministic is expected; see Roadmap TDM parity requirement in [master_roadmap.md](master_roadmap.md#12-migration-methodology-aggressive-legacy-removal)).

---

### INFRA-003 — Establish logging infrastructure (LogManager)

**Description**
Ensure project-wide structured logging is available via SquidASM’s `LogManager`, with predictable logger naming and file output suitable for debugging protocol traces and CI artifacts.

**Requirements (Measurable & Testable)**
1. All E-HOK modules MUST obtain loggers through a single utility surface (e.g., `LogManager.get_stack_logger(__name__)` convention).
2. Script logging MUST support:
   - Always-on file logging to a configured directory.
   - Optional terminal logging (disabled by default for CI cleanliness).
3. Logging MUST NOT duplicate handlers on repeated setup.
4. Logging MUST be compatible with non-simulation unit tests (i.e., logger construction does not require a running NetSquid simulation).

**Deliverables**
- Logging utilities in `ehok/utils/logging.py` confirmed/updated to meet the above requirements (Roadmap deliverable).
- A minimal logging usage guideline section in docs or README (optional but recommended).

**Success Criteria (How to Prove Completion)**
- **Test Type**: Unit tests.
- **Test Data**: Use a temporary directory for log output.
- **Expected Results**:
  - Calling the setup function twice does not create duplicate handlers.
  - Log file is created and contains an initialization header + at least one logged message.
  - Logger names are stable and hierarchical under `ehok.*`.
- **Failure Points & Debugging**:
  - *CI filesystem permissions*: log dir creation fails → redirect logs to workspace temp directory in CI.
  - *SquidASM import issues*: `LogManager` unavailable in minimal environments → define a fallback policy (documented) or ensure CI installs SquidASM.

**Legacy Interaction (Migration Strategy)**
- Logging is required to support parity debugging when legacy and new implementations diverge (Roadmap “Validate” phase).

---

## 3. Implementation Tasks (Should Have & Could Have)

### INFRA-004 — Create test fixtures for deterministic runs (SHOULD)

**Description**
Provide pytest fixtures and conventions that guarantee reproducible results for deterministic components and controlled reproducibility for simulation-adjacent tests.

**Requirements (Measurable & Testable)**
1. Provide a fixture that seeds:
   - Python `random`.
   - NumPy RNG.
2. Provide a documented policy for simulation randomness:
   - If NetSquid/SquidASM randomness can be seeded, document the method and require it for any deterministic parity test.
   - If not seedable, explicitly mark such tests as non-deterministic and exclude them from parity gating.
3. Deterministic tests MUST be repeatable:
   - Same seed ⇒ identical outputs (byte-for-byte for arrays, identical structured dataclass equality).

**Deliverables**
- Pytest fixtures (recommended location: `qia-challenge-2025/conftest.py` or `ehok/tests/conftest.py`) that implement the deterministic seed policy.
- Test marker policy:
  - Keep existing `--run-long` mechanism.
  - Add (or standardize) markers for `unit`, `integration`, and `deterministic` as needed.

**Success Criteria (How to Prove Completion)**
- **Test Type**: Unit tests.
- **Test Data**:
  - Run the same deterministic test case 20 times in a loop with a fixed seed.
- **Expected Results**:
  - All 20 runs produce the same digest (e.g., SHA256 of serialized output) and/or identical arrays.
- **Failure Points & Debugging**:
  - *Hidden RNG sources*: randomness from time or OS entropy → locate and route through fixture.
  - *Simulator nondeterminism*: event ordering differences → mark test as integration/non-deterministic and exclude from parity gates.

**Literature Justification (Why determinism matters)**
Deterministic parity tests are required by the Roadmap’s TDM strategy (“byte-for-byte parity on deterministic runs”). This is critical for validating timing/order constraints (commit-then-reveal and $\Delta t$ semantics) discussed in the NSM literature and analyses; without reproducible traces, diagnosing ordering violations is unreliable.

---

## 4. Test & Validation Framework

### Testing Infrastructure Setup
Required tooling for Sprint 0 completion (must be runnable in CI):
- `pytest` and `pytest-cov` (coverage optional for Sprint 0 but recommended).
- `mypy` strict typing checks.
- `flake8` lint.
- `pydocstyle --convention=numpy` docstring validation.

Recommended conventions:
- Separate **unit** tests (pure Python, no simulation) from **integration** tests (requires SquidASM/NetSquid runtime).
- Use the Roadmap’s tiering: unit on every push; integration on PR; e2e pre-merge.

### Performance Metrics (Sprint 0)
Sprint 0 is infrastructure-centric; therefore performance is measured in build/test reliability, not protocol throughput:
- **CI latency**: Stage 1 Quality Gate completes in ≤ 5 minutes.
- **Determinism**: deterministic tests show 0 flaky failures over 20 repeated runs.
- **Static quality**: 0 mypy errors in contract modules; 0 flake8 errors in `ehok/`.

### Validation Protocol
- Run CI on a clean runner to confirm no reliance on local state.
- Intentionally introduce one failure per tool category (type, lint, docstring, unit test) to confirm CI gating.
- For determinism: run deterministic unit tests multiple times within the same CI job step and ensure identical results.

---

## 5. Technical Risks & Mitigation

1. **Risk**: Simulator imports break unit-only CI environments.
   - **Mitigation**: enforce test layering; unit tests must not require a running simulation. Provide guarded imports or dependency installation in CI.
2. **Risk**: Dataclass contracts drift as phases are implemented.
   - **Mitigation**: treat contracts as versioned interfaces; changes require updating contract unit tests and explicit review. Contracts must remain backward-compatible within a sprint.
3. **Risk**: Non-deterministic behavior prevents parity testing.
   - **Mitigation**: formalize a “deterministic parity” subset (seeded, simulator-stable). Mark non-deterministic integration tests separately.
4. **Risk**: Logging duplication / handler leakage causes noisy CI and huge artifacts.
   - **Mitigation**: handler de-duplication tests; cap log size in CI artifacts if enabled.

---

## 6. Sprint Completion Criteria

Sprint 0 is complete when all of the following are true:
1. A CI workflow exists and enforces the Stage 1 Quality Gate (type, lint, docstrings, unit tests) per [master_roadmap.md](master_roadmap.md#23-cicd-gate-structure).
2. Phase boundary dataclass contracts exist with strict typing, Numpydoc docstrings, and invariant validation, and are covered by unit tests.
3. Logging utilities produce stable, hierarchical loggers under `ehok.*`, with tests proving no duplicate handlers and successful file output.
4. Deterministic testing fixtures exist (SHOULD item), and at least one deterministic test demonstrates repeatability over 20 runs.

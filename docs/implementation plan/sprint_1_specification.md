# Sprint 1 Specification: Security Core

**Document ID**: Sprint-1 / E-HOK-on-SquidASM

This document specifies Sprint 1 (“Security Core”) as an implementation-grade, test-first contract. It is strictly derived from Sprint 1 in the master roadmap (Section 4.2) and the referenced Phase I/IV analyses + literature.

---

## 1. Sprint 1 Executive Summary

### Goal
Implement the Noisy Storage Model (NSM) security foundation that blocks all subsequent protocol phases: (1) an NSM min-entropy / key-length calculator using the NSM “Max Bound”, (2) a simulation-time timing barrier enforcing the mandatory wait $\Delta t$, (3) a pre-flight feasibility gate that aborts early under mathematically impossible security conditions, and (4) protocol configuration schema exposing NSM/physical parameters.

**Roadmap source**: Sprint 1 MUST items in [master_roadmap.md](master_roadmap.md#42-sprint-1-security-core-days-4-10).

### Critical Path
`TASK-NSM-001` (`NSMBoundsCalculator`) blocks all later key-length decisions (reconciliation leakage accounting in Phase III and privacy amplification length selection in Phase IV).

**Roadmap source**: “All security calculations depend on TASK-NSM-001” in [master_roadmap.md](master_roadmap.md#32-blocking-dependencies-summary).

### Reference Map (Roadmap → Analyses → Literature)
- **NSM-001 (NSMBoundsCalculator / Max Bound)**
  - Roadmap: [master_roadmap.md](master_roadmap.md#421-must-items)
  - Phase IV (Max Bound + $\Gamma$ definition): [phase_IV_analysis.md](phase_IV_analysis.md#14-the-Γ-function)
  - Literature (exact equations for $\Gamma$, collision-entropy bound, and Max Bound):
    - “Error-tolerant oblivious transfer in the noisy-storage model” (Lupo et al., 2023): [docs/literature/Error-tolerant oblivious transfer in the noisy-storage model.md](../literature/Error-tolerant%20oblivious%20transfer%20in%20the%20noisy-storage%20model.md)
- **TIMING-001 (TimingEnforcer / $\Delta t$ barrier)**
  - Roadmap: [master_roadmap.md](master_roadmap.md#421-must-items)
  - Phase I (NSM timing semantics + Markovian storage noise assumption): [phase_I_analysis.md](phase_I_analysis.md#11-konig-wehner--wullschleger-2012--unconditional-security-from-noisy-quantum-storage)
  - Literature (Markovian family and NSM semantics):
    - “Unconditional security from noisy quantum storage” (König et al., 2012): [docs/literature/Unconditional-security-from-noisy-quantum-storage.md](../literature/Unconditional-security-from-noisy-quantum-storage.md)
    - “An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model” (Erven et al., 2014) for $\Delta t$ example: [docs/literature/An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model.md](../literature/An%20Experimental%20Implementation%20of%20Oblivious%20Transfer%20in%20the%20Noisy%20Storage%20Model.md)
- **FEAS-001 (Pre-flight feasibility gate)**
  - Roadmap: [master_roadmap.md](master_roadmap.md#421-must-items), abort taxonomy: [master_roadmap.md](master_roadmap.md#64-abort-code-taxonomy)
  - Phase I (two-tiered abort: 11% warning / 22% hard abort; “strictly less” condition): [phase_I_analysis.md](phase_I_analysis.md#12-schaffner-terhal--wehner-2009--robust-cryptography-in-the-noisy-quantum-storage-model)
  - Literature:
    - “Robust cryptography in the noisy-quantum-storage model” (Schaffner et al., 2009): [docs/literature/ROBUST CRYPTOGRAPHY IN THE NOISY-QUANTUM-STORAGE MODEL.md](../literature/ROBUST%20CRYPTOGRAPHY%20IN%20THE%20NOISY-QUANTUM-STORAGE%20MODEL.md)
    - “Error-tolerant oblivious transfer…” (Lupo et al., 2023) for the hard-limit intuition and bounds
- **NOISE-PARAMS-001 (Config schema: $\mu,\eta,e_{det}$)**
  - Roadmap: [master_roadmap.md](master_roadmap.md#421-must-items)
  - Phase I (physical parameters and rationale): [phase_I_analysis.md](phase_I_analysis.md#14-erven-et-al-2014--an-experimental-implementation-of-oblivious-transfer-in-the-noisy-storage-model)
  - Literature (Table I numeric parameters): [docs/literature/An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model.md](../literature/An%20Experimental%20Implementation%20of%20Oblivious%20Transfer%20in%20the%20Noisy%20Storage%20Model.md)

---

## 2. Component Specification: The NSM Calculator (TASK-NSM-001)

### Context
This is `NSM-001` in the Sprint 1 MUST items list: “Implement `NSMBoundsCalculator` with Max Bound” with validation against literature.

**Roadmap source**: [master_roadmap.md](master_roadmap.md#421-must-items).

### Separation of Concerns (Hard Requirement)
- **Mathematical logic** MUST be pure-Python, deterministic, and must not import NetSquid/SquidASM.
  - Target module: `ehok/analysis/nsm_bounds.py`.
- **System logic** (simulation time, sockets, orchestration) MUST NOT be implemented in this module.

### Module Definition
**File**: `ehok/analysis/nsm_bounds.py`

**Public API (required)**

1) Pure min-entropy rate calculator:

- `def max_bound_entropy_rate(storage_noise_r: float) -> float:`
  - Returns $h_{min}(r)$ (min-entropy rate per sifted bit) using the NSM “Max Bound”.

2) Supporting primitives:

- `def gamma_function(x: float) -> float:`
- `def collision_entropy_rate(storage_noise_r: float) -> float:`

3) Bounds calculator object (stateful contract):

- `@dataclass(frozen=True)
  class NSMBoundsInputs:`
  - `storage_noise_r: float`
  - `adjusted_qber: float`
  - `total_leakage_bits: int`
  - `epsilon_sec: float`
  - `n_sifted_bits: int`

- `class NSMBoundsCalculator:`
  - `def compute(self, inputs: NSMBoundsInputs) -> "NSMBoundsResult":`

- `@dataclass(frozen=True)
  class NSMBoundsResult:`
  - `max_secure_key_length_bits: int`
  - `min_entropy_per_bit: float`
  - `feasibility_status: "FeasibilityResult"`

- `class FeasibilityResult(Enum):`
  - `FEASIBLE`
  - `INFEASIBLE_QBER_TOO_HIGH`
  - `INFEASIBLE_INSUFFICIENT_ENTROPY`
  - `INFEASIBLE_INVALID_PARAMETERS`

**Note (roadmap alignment)**: The roadmap lists the inputs without `n_sifted_bits`. For a computable key length, Sprint 1 MUST introduce `n_sifted_bits` either as (a) a field on the input dataclass or (b) an initialization parameter on `NSMBoundsCalculator`. The specification requires it to be explicitly typed and validated.

### Inputs (Precise Definitions)

- $r$ (`storage_noise_r: float`): depolarising channel “retention probability” parameter in $[0,1]$.
  - Interpretation from literature: Eq. (26) in Lupo et al. 2023 shows $\tau = r\Psi + (1-r)I/2\otimes I/2$.
- $Q$ (`adjusted_qber: float`): adjusted QBER used for security (Phase II defines $Q_{eff}=Q_{measured}+\mu$; Sprint 1 consumes this value but does not compute $\mu$).
  - Must be validated in $[0,0.5]$ as a probability.
  - Sprint 1 feasibility must hard-abort (or return infeasible) for $Q > 0.22$ as per Phase I analysis.
- `total_leakage_bits: int`: non-negative integer representing all publicly revealed bits that must be subtracted from extractable entropy (wiretap cost). In Sprint 1 this is an input placeholder (Phase III will later compute it).
- `epsilon_sec: float`: security parameter; must satisfy $0 < \epsilon_{sec} < 1$.
- `n_sifted_bits: int`: number of candidate bits available for privacy amplification (typically $|I_0|-k$ from Phase II); must satisfy `n_sifted_bits >= 0`.

### Outputs

- `min_entropy_per_bit: float` equals $h_{min}(r)$.
- `max_secure_key_length_bits: int` equals the maximum extractable key length given the Sprint-1 bound model and inputs.
- `feasibility_status` captures the exact reason for infeasibility.

### Mathematical Verification (What Must Be Implemented)

#### Max Bound (NSM min-entropy rate)
Sprint 1 MUST implement the depolarising-channel Max Bound exactly as Eq. (36) in Lupo et al. (2023):

$$
 h_{min}(r) \ge \max\left\{ \Gamma\left[1 - \log_2(1 + 3r^2)\right],\; 1-r \right\}.
$$

Supporting definitions (Lupo et al. 2023, Eqs. (24)–(27)):

$$
\Gamma(x) = \begin{cases}
 x & \text{if } x \ge 1/2,\\
 g^{-1}(x) & \text{if } x < 1/2,
\end{cases}
\qquad
g(y) = -y\log_2 y - (1-y)\log_2(1-y) + y - 1.
$$

and

$$
 h_2(\sigma)= 1 - \log_2(1 + 3r^2).
$$

#### Sprint‑1 Key-Length Upper Bound (System‑Independent)
Sprint 1 MUST provide a conservative, simulator-independent length bound for feasibility and configuration validation:

$$
\ell_{max} = \left\lfloor n\, h_{min}(r) - L - 2\log_2\left(\frac{1}{\epsilon_{sec}}\right)\right\rfloor,
$$

where:
- $n$ is `n_sifted_bits`,
- $L$ is `total_leakage_bits` (placeholder, Phase III will later provide).

This matches the Roadmap’s general penalty term pattern (privacy amplification penalty) and Phase IV workflow outline.

### Error Conditions (Must Be Deterministic)

The calculator MUST reject invalid inputs with explicit error codes/messages:
- If `storage_noise_r` is outside $[0,1]$ → `INFEASIBLE_INVALID_PARAMETERS`.
- If `adjusted_qber` outside $[0,0.5]$ → `INFEASIBLE_INVALID_PARAMETERS`.
- If `epsilon_sec` is not in $(0,1)$ → `INFEASIBLE_INVALID_PARAMETERS`.
- If `total_leakage_bits < 0` or `n_sifted_bits < 0` → `INFEASIBLE_INVALID_PARAMETERS`.
- If `adjusted_qber > 0.22` → `INFEASIBLE_QBER_TOO_HIGH`.
- If computed $\ell_{max} \le 0$ → `INFEASIBLE_INSUFFICIENT_ENTROPY`.

### Validation Requirement (Literature as Unit Test Data)

**Roadmap mandate**: “Unit tests against [Lupo 2023] Table 1” with tolerance 0.1%.

Because the repository’s literature copy provides the full defining equations (and figure descriptions) but not a numeric “Table 1” dump, Sprint 1 MUST include a reproducible fixture set derived directly from the literature equations and/or digitized figure points:

1) **Equation-derived regression points (required)**
- Construct a test fixture of $(r, h_{min})$ points computed from Eq. (36) with a high-precision reference implementation.
- Minimum fixture set:
  - $r \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$
  - Expected $h_{min}$ values MUST match the Phase IV analysis sanity table:
    - $h_{min}(0.1) \approx 0.957$
    - $h_{min}(0.3) \approx 0.805$
    - $h_{min}(0.5) \approx 0.585$
    - $h_{min}(0.7) \approx 0.322$
    - $h_{min}(0.9) \approx 0.100$
  - Acceptance tolerance: relative error ≤ 0.1%.

2) **Structural properties (required)**
- Monotonicity: for a sorted grid of $r$ values in $[0,1]$, assert $h_{min}(r)$ is non-increasing.
- Bounds: assert $0 \le h_{min}(r) \le 1$ for all tested points.
- Crossover check: assert the chosen branch switches near $r \approx 0.82$ (Phase IV analysis statement), within ±0.03.

3) **Figure/experimental cross-check (recommended but optional within Sprint 1)**
- Digitize 5–10 points from the figure description in the Lupo et al. literature markdown (Fig. 2) and verify the computed curve is consistent within an absolute error budget (e.g., ≤ 0.02). This is a weaker check but ties directly to literature artifacts stored in-repo.

---

## 3. Component Specification: Timing & Feasibility (TASK-TIMING-001, TASK-FEAS-001, NOISE-PARAMS-001)

### 3.1 Timing Enforcer (TASK-TIMING-001)

**Objective**
Enforce the NSM mandatory wait window $\Delta t$ as a discrete-event simulation barrier; not merely message ordering.

**Roadmap source**: [master_roadmap.md](master_roadmap.md#421-must-items).

**Literature basis**
- NSM timing semantics and Markovian noise family: König et al. 2012 (see Phase I analysis mapping).
- Example $\Delta t$ selection: Erven et al. 2014 uses $\Delta t = 1\,\mathrm{s}$ before basis reveal.

**File**: `ehok/core/timing.py`

**Public API (required)**

- `@dataclass(frozen=True)
  class TimingConfig:`
  - `delta_t_ns: int`  (simulation-time nanoseconds)

- `class TimingEnforcer:`
  - `def __init__(self, config: TimingConfig) -> None:`
  - `def mark_commit_received(self, *, sim_time_ns: int) -> None:`
  - `def mark_basis_reveal_attempt(self, *, sim_time_ns: int) -> None:`
  - `def is_basis_reveal_allowed(self, *, sim_time_ns: int) -> bool:`
  - `def required_release_time_ns(self) -> int:`

**Pre-conditions**
- `delta_t_ns > 0`.
- `mark_basis_reveal_attempt` MUST NOT be called before `mark_commit_received` (otherwise raise `ValueError` or a domain exception).

**Post-conditions (invariants to enforce)**
- Safety property: basis reveal MUST NOT occur unless
  $$
  t_{basis} - t_{commit\_ack} \ge \Delta t.
  $$
- `required_release_time_ns()` MUST equal $t_{commit\_ack}+\Delta t$ once commit is marked.

**How “time” is measured**
- TimingEnforcer MUST operate solely on simulation time in integer nanoseconds.
- The integration layer (protocol coroutines) is responsible for providing `sim_time_ns` using the simulator time source (e.g., NetSquid `ns.sim_time()`), but TimingEnforcer itself must remain deterministic given time inputs.

**Validation (required)**
- “Simulation trace shows barrier markers” means:
  - Integration tests must record both event timestamps and assert the inequality above.
  - Additionally, logs (via LogManager) must emit structured events:
    - `TIMING_COMMIT_ACK_RECEIVED` with `t_commit_ack_ns`
    - `TIMING_BASIS_REVEAL_BLOCKED` when attempted too early
    - `TIMING_BASIS_REVEAL_ALLOWED` when allowed


### 3.2 Pre-Flight Feasibility Gate (TASK-FEAS-001)

**Objective**
Abort before consuming quantum resources when the session is provably insecure or produces no key (“Death Valley” conditions).

**Roadmap source**: [master_roadmap.md](master_roadmap.md#421-must-items) and abort taxonomy [master_roadmap.md](master_roadmap.md#64-abort-code-taxonomy).

**File**: `ehok/core/feasibility.py`

**Public API (required)**

- `@dataclass(frozen=True)
  class FeasibilityInputs:`
  - `expected_qber: float`
  - `storage_noise_r: float`
  - `storage_rate_nu: float`
  - `epsilon_sec: float`
  - `n_target_sifted_bits: int`
  - `expected_leakage_bits: int`

- `@dataclass(frozen=True)
  class FeasibilityDecision:`
  - `is_feasible: bool`
  - `abort_code: str | None`
  - `reason: str`
  - `recommended_min_n: int | None`

- `class FeasibilityChecker:`
  - `def check(self, inputs: FeasibilityInputs) -> FeasibilityDecision:`

**Hard abort rules (must)**
- If `expected_qber > 0.22` → `ABORT-I-FEAS-001` (roadmap).
- If strict-less condition violated ($Q_{trusted} \ge r_{storage}$) → abort with a dedicated feasibility code (Sprint 1 introduces code; roadmap references the invariant).
  - Literature basis: Schaffner et al. 2009 and Phase I analysis.
- If storage-capacity condition violated ($C_{\mathcal{N}}\cdot\nu \ge 1/2$) → abort.
  - Compute $C_{\mathcal{N}}$ using Phase I analysis (Lupo 2023 Eq. (14) shown there):
    $$
    C_{\mathcal{N}} = 1 - h\left(\frac{1+r}{2}\right)
    $$
  - Literature basis: König et al. 2012 (capacity × rate constraint), as mapped in Phase I analysis.

**Death Valley check (must)**
- Use the Sprint‑1 key-length bound (Section 2) to estimate whether $\ell_{max} > 0$.
- If $\ell_{max} \le 0$ → abort with feasibility code and a recommended minimum $n$ (computed by solving for $n$ s.t. $\ell_{max} > 0$).

**Validation (required)**
- Unit tests for the checker must include:
  - Hard abort at `expected_qber = 0.2201`.
  - Pass at `expected_qber = 0.05` with otherwise safe parameters.
  - Strict-less abort when `expected_qber >= storage_noise_r` for at least 3 pairs of values.


### 3.3 Config Schema (NOISE-PARAMS-001)

**Objective**
Expose NSM parameters and physical-model parameters through a typed config schema.

**Roadmap source**: [master_roadmap.md](master_roadmap.md#421-must-items).

**File to create**: `ehok/configs/protocol_config.py`

**Schema requirements (must)**
- Configuration must be representable as `@dataclass(frozen=True)` with explicit units:
  - `mu_pair_per_coherence: float` (dimensionless mean photon pairs per coherence time)
  - `eta_total_transmittance: float` in $[0,1]$
  - `e_det: float` in $[0,1]$
  - `p_dark: float` in $[0,1]$
  - `delta_t_s: float` (seconds) and/or `delta_t_ns: int` (nanoseconds; must be derived exactly)
  - `storage_noise_r: float` in $[0,1]$
  - `storage_rate_nu: float` in $[0,1]$
  - `epsilon_sec: float` in $(0,1)$

**Default values (must)**
Default physical parameters MUST be taken from Erven et al. 2014 Table I:
- $\mu = 3.145\times 10^{-5}$
- $\eta = 0.0150$
- $e_{det} = 0.0093$
- $P_{dark} = 1.50\times 10^{-8}$

**Validation (must)**
- Unit tests must confirm:
  - schema rejects out-of-range probabilities.
  - defaults match literature values to within the stated precision.

---

## 4. Component Specification: Physical Adapter (SHOULD) (TASK-NOISE-ADAPTER-001)

### Context
This is a Sprint 1 SHOULD item: provide a physical-model adapter mapping $(\mu,\eta,e_{det})$ parameters to simulator-level noise knobs.

**Roadmap source**: [master_roadmap.md](master_roadmap.md#422-should-items).

### Scope Boundaries (important)
- The adapter is **not** a full security proof.
- It is a translation layer for simulation configuration and for expected-rate checks.
- It must be designed so later phases can replace/extend it without changing the NSM math module.

### File
`ehok/quantum/noise_adapter.py`

### Public API (required)

- `@dataclass(frozen=True)
  class PhysicalParams:`
  - `mu_pair_per_coherence: float`
  - `eta_total_transmittance: float`
  - `e_det: float`
  - `p_dark: float`

- `@dataclass(frozen=True)
  class SimulatorNoiseParams:`
  - `link_fidelity: float`
  - `measurement_bitflip_prob: float`
  - `expected_detection_prob: float`

- `def physical_to_simulator(params: PhysicalParams) -> SimulatorNoiseParams:`

### Translation Logic (normative requirements)
- The adapter MUST be explicit about what is modeled and what is not.
- At minimum:
  - `measurement_bitflip_prob` MUST equal `e_det` (first-order approximation for intrinsic detection errors).
  - `expected_detection_prob` MUST be monotone increasing in $\eta$ and $\mu$.
  - `link_fidelity` MUST be in $[0,1]$ and monotone increasing in $\eta$ (higher loss should not increase fidelity).

### Validation Requirement (Erven et al. 2014 as unit-test data)
- Construct unit-test fixtures using Table I values from Erven et al.:
  - $(\mu,\eta,e_{det},P_{dark})=(3.145\times 10^{-5},0.0150,0.0093,1.50\times 10^{-8})$.
- Expected results:
  - All returned probabilities are within $[0,1]$.
  - `measurement_bitflip_prob == 0.0093` exactly (or within 1e-12).
  - `link_fidelity` and `expected_detection_prob` must be stable (no NaNs/Infs) and reproducible.

---

## 5. Testing & Quality Assurance Plan

### 5.1 Unit Testing Strategy (Mathematical Functions)

**Targets**
- `ehok/analysis/nsm_bounds.py`:
  - $\Gamma$ definition correctness (branching + inversion behavior).
  - Eq. (27) collision entropy rate.
  - Eq. (36) Max Bound.
  - Key-length penalty formula.

**Determinism requirement**
- All math tests MUST be deterministic and run without simulator imports.

**Numerical tolerance policy (must)**
- For literature regression points: relative error ≤ 0.1% (roadmap).
- For invariant/property tests: exact inequalities (no tolerance) where applicable.


### 5.2 Integration Testing Strategy (Timing Enforcement)

**Targets**
- `ehok/core/timing.py` integrated into a minimal two-party simulation harness (Alice/Bob coroutines) that produces timestamps.

**What must be proven**
- The timing barrier blocks basis reveal attempts until $\Delta t$ has elapsed since commit/ack.

**Pass/fail criteria**
- Pass if:
  - For at least 10 runs (seeded where possible), the observed difference satisfies $t_{basis} - t_{commit\_ack} \ge \Delta t$.
  - At least one negative test verifies that attempting basis reveal earlier triggers a block event and does not progress the protocol state.


### 5.3 Data Fixtures (Standard Inputs)

A shared fixture set MUST be defined for Sprint 1 tests:

- **NSM curve fixture**: points in $r$ space (at least 5 points) with expected $h_{min}$.
- **Feasibility fixture**:
  - safe case: `expected_qber = 0.05`, `storage_noise_r = 0.75`, `storage_rate_nu = 0.002`, `epsilon_sec = 1e-6`.
  - abort cases:
    - `expected_qber = 0.2201` (hard abort)
    - `expected_qber = storage_noise_r` (strict-less violation)
- **Physical params fixture** (Erven Table I) for config + adapter.

---

## 6. Risks & Definition of Done

### 6.1 Specific Risks

1. **NetSquid timing API incompatibility**
- Risk noted in the roadmap: inability to access simulator time or enforce barriers in SquidASM program context.
- Mitigation requirement:
  - TimingEnforcer must be designed to accept time as input (pure), while integration provides the time source.
  - Provide a fallback integration strategy if `ns.sim_time()` is inaccessible: implement barrier at NetSquid Protocol level (documented divergence).

2. **Numerical instability / incorrect log base**
- Risk: mixing $\log$ bases yields incorrect entropy rates.
- Mitigation requirement:
  - All entropy-related logs MUST be base-2; unit tests must catch base errors using regression points.

3. **Ambiguous parameter semantics (r meaning)**
- Risk: confusion between “depolarizing rate” vs “retention probability”.
- Mitigation requirement:
  - Documentation must define $r$ exactly as in Lupo Eq. (26).
  - The config field name must reflect the meaning (e.g., `storage_noise_r_retention`).


### 6.2 Definition of Done (Sprint 1)

Sprint 1 is complete when all of the following are true:

1. **NSMBoundsCalculator**
- Implements Eq. (36) Max Bound and produces deterministic outputs.
- Unit tests pass and include literature-derived regression points.
- Regression tolerance meets roadmap: ≤ 0.1% relative error on the fixed fixture set.

2. **TimingEnforcer**
- Enforces $\Delta t$ barrier and exposes timestamps/markers.
- Integration tests prove the barrier blocks basis reveal.

3. **Feasibility checker**
- Aborts before quantum generation at `Q > 22%` (roadmap) and reports explicit abort code.
- Includes strict-less and capacity×rate feasibility checks as specified.

4. **Config schema**
- Provides typed access to $(\mu,\eta,e_{det},P_{dark},\Delta t,r,\nu,\epsilon_{sec})$.
- Defaults match Erven et al. Table I values.

5. **Quality metrics**
- ≥ 90% line coverage for `ehok/analysis/nsm_bounds.py`.
- All Sprint 1 tests run in CI within the Stage‑1 budget established in Sprint 0.

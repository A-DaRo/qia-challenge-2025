# Phase II Technical Analysis: Sifting & Estimation

> **Definitive Migration Guide for Classical Post-Processing in E-HOK**
> 
> Version: 1.0  
> Last Updated: 2025  
> Authors: AI Technical Analysis

---

## Abstract

Phase II of the E-HOK protocol implements the critical "gatekeeper" functionality that bridges raw quantum measurement data (Phase I) with information-theoretic post-processing (Phase III). This phase operationalizes the **Commit-then-Reveal** logic fundamental to the Weak String Erasure (WSE) primitive, enforcing strict temporal ordering to prevent adversarial post-selection attacks. The central security guarantee—that Bob cannot retrospectively filter detection reports after learning basis information—depends entirely on the integrity of this phase.

This analysis examines three core algorithmic domains: (1) the **"Sandwich" Protocol Flow** implementing ordered message acknowledgment, (2) **Missing Rounds Validation** using Chernoff-bound statistical tests, and (3) **Finite-Size Statistical Penalty** ($\mu$) calculation for composable security. We map theoretical requirements from Schaffner et al., Erven et al., and Lupo et al. against the SquidASM/NetQASM/NetSquid stack, identifying semantic gaps and proposing extension architectures for migration.

The analysis concludes that while SquidASM provides native support for basis sifting and classical communication, the framework lacks built-in mechanisms for ordered acknowledgment enforcement, Chernoff-bound validation, and finite-size penalty calculation—all of which require custom implementation within the `ehok/` workspace.

---

## 1. Ontology: Core Concepts of Phase II

### 1.1 The "Gatekeeper" Responsibility

Phase II acts as the protocol's security checkpoint, ensuring that:

| Concept | Definition | Security Role |
|---------|------------|---------------|
| **Weak String Erasure (WSE)** | Primitive where Bob receives a string with some positions erased; Alice knows which positions were erased | Foundation for 1-2 Random OT construction |
| **Commit-then-Reveal** | Temporal ordering where Bob's detection report is committed before basis revelation | Prevents adversarial post-selection of favorable rounds |
| **Missing Rounds** | Indices where Bob claims no photon detection occurred | Must be validated against expected channel transmittance |
| **Finite-Size Penalty ($\mu$)** | Statistical adjustment accounting for sample variance | Bridges observed QBER to worst-case security bounds |

### 1.2 Temporal Ordering Semantics

The security model mandates a strict causal ordering of protocol messages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase II: Temporal Ordering Diagram                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TIME ────────────────────────────────────────────────────────────────▶ │
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐ │
│  │ Quantum │    │ Missing │    │  Wait   │    │  Basis  │    │Sifting │ │
│  │   Tx    │───▶│ Rounds  │───▶│   Δt    │───▶│ Reveal  │───▶│& QBER  │ │
│  │         │    │ Report  │    │         │    │         │    │        │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘ │
│       │              │              │              │              │      │
│       │              │              │              │              │      │
│   Alice sends    Bob commits    Storage       Alice sends    Classical  │
│   qubits to      detection      decoheres     basis string   sifting &  │
│   Bob            events         (NSM)         α^m            estimation │
│                                                                          │
│  ◄──────────────── CRITICAL ORDERING CONSTRAINT ─────────────────────▶  │
│                                                                          │
│  Bob's Missing Rounds report MUST be acknowledged BEFORE bases sent     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Security Invariant**: If Bob receives basis information $\alpha^m$ before committing his detection report $\mathcal{M}$, he can selectively claim "loss" only on rounds where his noisy storage failed, effectively post-selecting a lower-noise sub-key. This breaks the WSE security guarantee.

### 1.3 Conceptual Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Phase II Conceptual Flow                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────┐                                                   │
│  │  Raw Quantum Data   │ From Phase I: m rounds with outcomes X^m, Y^m    │
│  │  (X^m, Y^m, α^m)    │ and bases α^m, β^m                               │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ Missing Rounds      │ Bob reports indices 𝓜 where no detection         │
│  │ Validation          │ Alice validates |𝓜| against Chernoff bound       │
│  └──────────┬──────────┘                                                   │
│             │ Pass/Abort                                                   │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ Basis Sifting       │ Compute I₀ (match) and I₁ (mismatch)             │
│  │ Compute I₀, I₁      │ I_C → S_C (Bob's chosen string)                  │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ Test Set Sampling   │ Random subset k from I₀ for QBER estimation      │
│  │ T ⊂ I₀, |T| = k     │ Remaining n = |I₀| - k bits form raw key        │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ QBER Estimation     │ e_obs = (errors in T) / k                        │
│  │ with Penalty μ      │ e_adj = e_obs + μ where μ = f(n, k, ε_sec)       │
│  └──────────┬──────────┘                                                   │
│             │ e_adj ≤ 22%?                                                 │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ To Phase III        │ Sifted key indices, adjusted QBER                │
│  │ (Reconciliation)    │ for privacy amplification calculation            │
│  └─────────────────────┘                                                   │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Literature Alignment & Mathematical Foundations

### 2.1 Theoretical Corpus

Phase II draws from three primary literature sources, each contributing distinct mathematical machinery:

| Source | Primary Contribution | Key Equations |
|--------|---------------------|---------------|
| **Schaffner et al.** (2009) | WSE primitive definition; individual-storage attack model; 11% QBER bound | Smooth min-entropy bounds; Protocol 1 (1-2 ROT) |
| **Erven et al.** (2014) | Finite-size security analysis; experimental implementation | Penalty term $\mu$; Eq. (8) for ROT rate |
| **Lupo et al.** (2023) | Tight entropic uncertainty relations; 22% hard limit derivation | Eq. (36) min-entropy bound; Eq. (43) bit rate |

### 2.2 Missing Rounds Constraint (Chernoff Validation)

**Source**: Schaffner et al. Section 4, Remark 4; Erven et al. Section "Correctness"

**Problem Statement**: A cheating Bob with imperfect quantum storage could exploit detection loss by claiming "missing" precisely those rounds where his memory failed. This post-selection attack would effectively give him a lower-error sub-key.

**Mathematical Formulation**: Let $M$ be the total number of transmitted rounds and $S$ the number Bob reports as detected. The expected detection rate is $P_{expected}$ (calibrated from channel transmittance in Phase I). Alice validates:

$$\text{Prob}\left[|S - P_{expected} \cdot M| \geq \zeta \cdot M\right] < \varepsilon$$

where the Chernoff tolerance $\zeta$ is derived from Hoeffding's inequality:

$$\zeta = \sqrt{\frac{\ln(2/\varepsilon)}{2M}}$$

**Security Semantics**: If Bob's reported detection count falls outside the interval $[(P_{expected} - \zeta)M, (P_{expected} + \zeta)M]$, the protocol aborts. This bounds Bob's ability to post-select favorable rounds.

**Literature Quote** (Erven et al.):
> "According to Hoeffding's inequality, the number of detected rounds fall out of this interval with probability less than $2\varepsilon$... This test prevents a dishonest Bob from using the fact that he can report rounds as lost to discard some or all of the single photon rounds."

### 2.3 Finite-Size Statistical Penalty ($\mu$)

**Source**: Erven et al. Theorem 2, Eq. (2); Schaffner et al. Corollary 2

**Problem Statement**: The observed QBER on a test subset of size $k$ is a point estimate. The true error rate on the remaining key of size $n$ may be higher due to statistical fluctuation. The protocol must account for this uncertainty to maintain composable security.

**Mathematical Formulation**: The penalty term $\mu$ bridges observed QBER to worst-case bounds:

$$\mu := \sqrt{\frac{n + k}{nk} \cdot \frac{k + 1}{k}} \cdot \ln\frac{4}{\varepsilon_{sec}}$$

The adjusted QBER used for security calculations becomes:

$$e_{adj} = e_{obs} + \mu$$

**Scaling Behavior**:
- $\mu \propto 1/\sqrt{k}$ — larger test sets reduce uncertainty
- $\mu \propto \sqrt{\ln(1/\varepsilon_{sec})}$ — tighter security requires larger penalty
- For $k = 10^5$, $\varepsilon_{sec} = 10^{-10}$: $\mu \approx 0.003$ (0.3%)

**Security Semantics**: Privacy amplification calculations must use $e_{adj}$, not $e_{obs}$, to ensure the final key satisfies the target security parameter.

### 2.4 QBER Thresholds

**Source**: Lupo et al. Section VI; Schaffner et al. Section 5

Two distinct thresholds govern protocol viability:

| Threshold | Value | Derivation | Action |
|-----------|-------|------------|--------|
| **Hard Limit** | 22% | Lupo et al. Eq. (43): $h((1+r_j)/2) \leq 1/2 \Rightarrow r_j \geq 0.78$ | ABORT — security impossible |
| **Conservative Limit** | 11% | Schaffner et al. Section 5.2: $t \geq 0.22$ for depolarizing noise | WARNING — reduced key rate |

**Physical Interpretation**: The 22% limit arises from the requirement that error correction leakage not exceed Bob's min-entropy about Alice's string. Beyond this threshold, the information leaked through syndromes exceeds the uncertainty provided by the noisy storage assumption.

**Literature Quote** (Lupo et al.):
> "Secure OT is possible only if the trusted noise parameter $r_j$ is such that $h((1+r_j)/2) \leq 1/2$, i.e. $r_j \geq 0.78$. This corresponds to a maximum tolerable trusted noise of about 22%."

### 2.5 Basis Sifting Semantics

**Source**: Damgård et al. (1999); Schaffner et al. Protocol 1

After basis revelation, the measurement outcomes are partitioned:

- $I_C = \{i \mid \alpha_i = \beta_i = C\}$ — Bob's chosen basis matches
- $I_{\bar{C}} = \{i \mid \alpha_i = \beta_i \neq C\}$ — Bob's chosen basis mismatches
- $I_0 \cup I_1$ — union forms the sifted key candidates

**Security Property**: Bob's labeling of which subset is $I_C$ (his chosen string) vs $I_{\bar{C}}$ is hidden from Alice because both subsets are statistically indistinguishable.

---

## 3. Protocol Logic & Flow Analysis

### 3.1 Ordered Message Protocol ("Sandwich" Flow)

The Commit-then-Reveal security depends on a strict message ordering protocol:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Ordered Message Protocol Flow                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│     ALICE                                           BOB                   │
│       │                                               │                   │
│       │───────────[Quantum Transmission]──────────────▶│                  │
│       │              (EPR pairs, time t=0)            │                   │
│       │                                               │                   │
│       │◀──────────[Detection Report 𝓜]────────────────│                  │
│       │         Bob commits missing indices           │                   │
│       │                                               │                   │
│       │───────────[ACK: Report Received]──────────────▶│                  │
│       │        ▲                                      │                   │
│       │        │                                      │                   │
│       │   ┌────┴─────────────────────────────────┐    │                   │
│       │   │ CRITICAL: ACK must be received by    │    │                   │
│       │   │ Bob BEFORE Alice sends basis string  │    │                   │
│       │   └──────────────────────────────────────┘    │                   │
│       │                                               │                   │
│       │        [WAIT Δt - Storage Decoherence]        │                   │
│       │                                               │                   │
│       │───────────[Basis String α^m]──────────────────▶│                  │
│       │                                               │                   │
│       │◀──────────[Index Lists I₀, I₁]────────────────│                  │
│       │                                               │                   │
│       │───────────[Test Subset Challenge]─────────────▶│                  │
│       │                                               │                   │
│       │◀──────────[Test Outcomes Y|_T]────────────────│                  │
│       │                                               │                   │
│   [Compute QBER, Apply μ Penalty]                     │                   │
│       │                                               │                   │
│       ▼                                               │                   │
│   Continue to Phase III or ABORT                      │                   │
│                                                       │                   │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Failure Mode: Race Condition

**Scenario**: Classical communication latency causes Alice to send basis string $\alpha^m$ before Bob's acknowledgment of his detection report is registered.

**Attack Vector**: If Bob receives $\alpha^m$ before his report is committed (from Alice's perspective), he can:
1. Wait for basis information
2. Selectively measure stored qubits in the correct bases
3. Report "missing" only those rounds where he failed
4. Achieve effectively zero storage noise

**Mitigation Requirement**: The protocol must enforce synchronous acknowledgment—Alice's basis transmission is blocked until Bob's acknowledgment is processed.

### 3.3 State Machine Representation

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Phase II State Machine                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                           │
│   │   START     │                                                           │
│   │ (Phase I    │                                                           │
│   │  Complete)  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ AWAIT_      │◀─────────────────────────┐                               │
│   │ DETECTION_  │                          │ Timeout: Retry                │
│   │ REPORT      │──────────────────────────┘                               │
│   └──────┬──────┘                                                           │
│          │ Report Received                                                  │
│          ▼                                                                  │
│   ┌─────────────┐     Chernoff                                             │
│   │ VALIDATE_   │     Failed      ┌─────────────┐                          │
│   │ MISSING_    │────────────────▶│   ABORT     │                          │
│   │ ROUNDS      │                 │ (Detection  │                          │
│   └──────┬──────┘                 │  Anomaly)   │                          │
│          │ Validation Passed      └─────────────┘                          │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ WAIT_DELTA  │ Timer: Δt nanoseconds                                    │
│   │ _T          │ (Storage decoherence)                                    │
│   └──────┬──────┘                                                           │
│          │ Timer Expired                                                    │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ REVEAL_     │ Send α^m to Bob                                          │
│   │ BASES       │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ RECEIVE_    │ Bob sends I₀, I₁                                         │
│   │ INDICES     │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ SAMPLE_     │ Select random T ⊂ I₀                                     │
│   │ TEST_SET    │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ COMPUTE_    │ e_obs, compute μ, e_adj = e_obs + μ                      │
│   │ QBER        │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐     e_adj > 22%     ┌─────────────┐                      │
│   │ CHECK_      │────────────────────▶│   ABORT     │                      │
│   │ THRESHOLDS  │                     │ (QBER High) │                      │
│   └──────┬──────┘                     └─────────────┘                      │
│          │ e_adj ≤ 22%                                                      │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │  SUCCESS    │ Pass to Phase III with sifted data                       │
│   │ (To Phase   │                                                           │
│   │   III)      │                                                           │
│   └─────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Gap Analysis

### 4.1 Gap Summary Matrix

| Capability | SquidASM Native | Legacy ehok | Gap Status | Extension Required |
|------------|-----------------|-------------|------------|-------------------|
| Classical Socket Communication | ✅ `ClassicalSocket` | N/A | SUPPORTED | None |
| Basis Sifting (I₀/I₁ computation) | ✅ Basis enums | ✅ `SiftingManager.identify_matching_bases()` | SUPPORTED | None |
| Random Test Set Selection | Partial | ✅ `SiftingManager.select_test_set()` | SUPPORTED | Migrate logic |
| QBER Estimation | Not built-in | ✅ `SiftingManager.estimate_qber()` | SUPPORTED | Migrate logic |
| **Ordered Acknowledgment** | ❌ Not built-in | ❌ Not implemented | **GAP** | Create `OrderedProtocolSocket` |
| **Chernoff Bound Validation** | ❌ Not built-in | ❌ Not implemented | **GAP** | Create `DetectionValidator` |
| **Finite-Size Penalty ($\mu$)** | ❌ Not built-in | ❌ Not implemented | **GAP** | Create `calculate_finite_size_penalty()` |
| **QBER Adjustment with $\mu$** | ❌ Not built-in | ❌ Not implemented | **GAP** | Create `compute_adjusted_qber()` |
| Decoy State Statistics | ❌ Not built-in | ❌ Not implemented | **GAP** | Complex extension (optional) |

### 4.2 Gap Analysis: Ordered Acknowledgment

**Current State (SquidASM)**:
The `ClassicalSocket` class provides basic send/receive operations without ordering guarantees:

- `send(msg: str)` — non-blocking message dispatch
- `recv()` — generator-based blocking receive

**Gap**: No mechanism to ensure message ordering across parties. Alice's `send(bases)` can race ahead of Bob's acknowledgment receipt.

**Security Impact**: Without ordered acknowledgment, the fundamental WSE security guarantee is violated.

**Proposed Extension Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OrderedProtocolSocket Extension                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    OrderedProtocolSocket                          │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Wraps: ClassicalSocket                                      │  │   │
│  │  │                                                             │  │   │
│  │  │ Methods:                                                    │  │   │
│  │  │   send_with_ack(msg, timeout) → Generator                   │  │   │
│  │  │     - Sends message                                         │  │   │
│  │  │     - Blocks until ACK received or timeout                  │  │   │
│  │  │     - Raises ProtocolViolation on timeout                   │  │   │
│  │  │                                                             │  │   │
│  │  │   recv_and_ack() → Generator                                │  │   │
│  │  │     - Receives message                                      │  │   │
│  │  │     - Automatically sends ACK                               │  │   │
│  │  │     - Returns message content                               │  │   │
│  │  │                                                             │  │   │
│  │  │ State:                                                      │  │   │
│  │  │   _sequence_number: int                                     │  │   │
│  │  │   _pending_acks: Dict[int, Event]                           │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Location: ehok/protocols/ordered_messaging.py                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Gap Analysis: Chernoff Bound Validation

**Current State (Legacy ehok)**: 
The `SiftingManager` class has no detection report validation logic.

**Gap**: Missing rounds are accepted without statistical validation against expected channel parameters.

**Security Impact**: Bob can selectively report losses without detection.

**Proposed Extension Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DetectionValidator Extension                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DetectionReport (Dataclass)                    │   │
│  │                                                                   │   │
│  │  Fields:                                                          │   │
│  │    total_rounds: int                                              │   │
│  │    detected_indices: List[int]                                    │   │
│  │    missing_indices: List[int]                                     │   │
│  │                                                                   │   │
│  │  Properties:                                                      │   │
│  │    detection_rate → len(detected_indices) / total_rounds         │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DetectionValidator                             │   │
│  │                                                                   │   │
│  │  Constructor:                                                     │   │
│  │    expected_transmittance: float (P_expected from Phase I)       │   │
│  │    epsilon_sec: float (security parameter, default 10^-10)       │   │
│  │                                                                   │   │
│  │  Methods:                                                         │   │
│  │    validate(report: DetectionReport) → (bool, str)               │   │
│  │      - Computes ζ = sqrt(ln(2/ε) / (2M))                         │   │
│  │      - Checks |S - P·M| ≤ ζ·M                                    │   │
│  │      - Returns (passed, diagnostic_message)                      │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Location: ehok/quantum/detection.py                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Gap Analysis: Finite-Size Penalty Calculation

**Current State (Legacy ehok)**:
`SiftingManager.check_qber_abort()` compares raw QBER against threshold without adjustment.

**Gap**: No finite-size penalty calculation. Observed QBER used directly.

**Security Impact**: For small test sets, actual QBER may exceed bounds with non-negligible probability.

**Proposed Extension Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Statistics Extension                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FiniteSizeAnalyzer                             │   │
│  │                                                                   │   │
│  │  Functions (Static/Module-level):                                 │   │
│  │                                                                   │   │
│  │    calculate_finite_size_penalty(n, k, epsilon_sec) → float      │   │
│  │      - μ = sqrt((n+k)/(nk) · (k+1)/k) · ln(4/ε_sec)             │   │
│  │      - Returns penalty to add to observed QBER                   │   │
│  │                                                                   │   │
│  │    compute_adjusted_qber(e_obs, n, k, epsilon_sec) → float       │   │
│  │      - Computes μ internally                                     │   │
│  │      - Returns e_obs + μ                                         │   │
│  │                                                                   │   │
│  │    check_security_bounds(e_adj, hard=0.22, cons=0.11) → Result   │   │
│  │      - Returns (status, message) tuple                           │   │
│  │      - status ∈ {ABORT, WARNING, OK}                             │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Location: ehok/analysis/statistics.py                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Legacy Code Assessment & Removal Plan

**File**: `ehok/core/sifting.py`

| Method | Purpose | Migration Status |
|--------|---------|------------------|
| `identify_matching_bases(bases_alice, bases_bob)` | Computes $I_0$, $I_1$ from basis arrays | ✅ Extract logic, reimplement in SquidASM-native module |
| `select_test_set(I_0, fraction, seed)` | Random test subset selection | ✅ Migrate to SquidASM context; delete legacy |
| `estimate_qber(outcomes_alice, outcomes_bob, test_indices)` | Computes $e_{obs}$ | ✅ Reimplement with $\mu$ integration; delete legacy |
| `check_qber_abort(qber, threshold)` | Threshold check | ⚠️ Rewrite to use $e_{adj}$; delete legacy version |

**Assessment**: The legacy implementation provides correct baseline sifting logic but lacks:
1. Ordered acknowledgment integration (security-critical)
2. Chernoff bound validation (security-critical)
3. Finite-size penalty calculation (security-critical)

**Deletion Plan**: Once parity tests confirm all three gaps are closed in the new SquidASM-native sifting module:
- Delete `ehok/core/sifting.py` entirely
- Remove all imports of legacy sifting functions
- Confirm test suite uses only SquidASM-native implementations
- Update documentation to reference new module paths

No deprecation period—deletion is immediate upon validation.

---

## 5. Formalized Metrics & Constraints

### 5.1 Security Metrics with LaTeX Formalization

#### 5.1.1 Chernoff Tolerance Bound

$$\zeta(\varepsilon, M) := \sqrt{\frac{\ln(2/\varepsilon)}{2M}}$$

where:
- $\varepsilon$ — Security parameter (typical: $10^{-10}$)
- $M$ — Total transmitted rounds

**Numerical Example**: For $M = 10^6$ and $\varepsilon = 10^{-10}$:

$$\zeta = \sqrt{\frac{\ln(2 \times 10^{10})}{2 \times 10^6}} \approx \sqrt{\frac{24.02}{2 \times 10^6}} \approx 0.0035$$

This means detection rate must be within ±0.35% of expected.

#### 5.1.2 Finite-Size Statistical Penalty

$$\mu(n, k, \varepsilon_{sec}) := \sqrt{\frac{n + k}{nk} \cdot \frac{k + 1}{k}} \cdot \ln\frac{4}{\varepsilon_{sec}}$$

**Scaling Analysis**:

| Test Set Size $k$ | Key Size $n$ | $\varepsilon_{sec}$ | $\mu$ |
|------------------|--------------|---------------------|-------|
| $10^3$ | $10^4$ | $10^{-10}$ | 0.074 (7.4%) |
| $10^4$ | $10^5$ | $10^{-10}$ | 0.023 (2.3%) |
| $10^5$ | $10^6$ | $10^{-10}$ | 0.007 (0.7%) |
| $10^6$ | $10^7$ | $10^{-10}$ | 0.002 (0.2%) |

**Insight**: For practical security, test sets should be at least $10^4$ bits to keep $\mu < 3\%$.

#### 5.1.3 Adjusted QBER

$$e_{adj} := e_{obs} + \mu(n, k, \varepsilon_{sec})$$

**Security Constraint**:

$$e_{adj} \leq Q_{hard} = 0.22$$

**Conservative Constraint**:

$$e_{adj} \leq Q_{cons} = 0.11$$

#### 5.1.4 Sifting Efficiency

$$\eta_{sift} := \frac{|I_0|}{M_{detected}} \approx 0.5$$

For random basis selection (BB84-style), approximately half of detected rounds have matching bases.

### 5.2 Success Criteria

| Criterion | Condition | Action on Failure |
|-----------|-----------|-------------------|
| Detection Bound | $\|S - P_{exp} \cdot M\| \leq \zeta \cdot M$ | ABORT with `DetectionAnomalyError` |
| Hard QBER Limit | $e_{adj} \leq 0.22$ | ABORT with `QBERTooHighError` |
| Conservative QBER | $e_{adj} \leq 0.11$ | WARNING logged; reduced key rate expected |
| Minimum Test Size | $k \geq 100$ | ABORT with `InsufficientTestSampleError` |
| Temporal Ordering | ACK received before basis reveal | ABORT with `ProtocolViolationError` |

---

## 6. Integration Architecture

### 6.1 Component Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase II Integration Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         E-HOK Application Layer                      │    │
│  │                                                                      │    │
│  │   ┌──────────────────┐   ┌──────────────────┐   ┌───────────────┐   │    │
│  │   │  SiftingManager  │   │ DetectionValidator│   │ FiniteSizeAna │   │    │
│  │   │  (legacy core)   │   │   (new)          │   │ lyzer (new)   │   │    │
│  │   └────────┬─────────┘   └────────┬─────────┘   └───────┬───────┘   │    │
│  │            │                      │                     │           │    │
│  │            └──────────────┬───────┴─────────────────────┘           │    │
│  │                           │                                          │    │
│  │                           ▼                                          │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │                  PhaseIIOrchestrator                           │ │    │
│  │   │  - Coordinates ordered messaging                               │ │    │
│  │   │  - Invokes validation and sifting components                   │ │    │
│  │   │  - Manages state transitions                                   │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  │                           │                                          │    │
│  └───────────────────────────┼──────────────────────────────────────────┘    │
│                              │                                               │
│  ┌───────────────────────────┼──────────────────────────────────────────┐    │
│  │                    SquidASM Layer                                     │    │
│  │                           │                                           │    │
│  │   ┌───────────────────────▼────────────────────────────────────────┐ │    │
│  │   │                  OrderedProtocolSocket                          │ │    │
│  │   │  - Wraps ClassicalSocket                                        │ │    │
│  │   │  - Provides send_with_ack / recv_and_ack                        │ │    │
│  │   └───────────────────────────────────────────────────────────────┘  │    │
│  │                           │                                           │    │
│  │   ┌───────────────────────▼────────────────────────────────────────┐ │    │
│  │   │                  ClassicalSocket (Native)                       │ │    │
│  │   └───────────────────────────────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Flow Sequence

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     Phase II Data Flow Sequence                                │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Phase I Output                                                                │
│  ┌─────────────────────────────────────────┐                                  │
│  │ RawQuantumData                          │                                  │
│  │   - outcomes_alice: np.ndarray          │                                  │
│  │   - outcomes_bob: np.ndarray            │                                  │
│  │   - bases_alice: np.ndarray             │                                  │
│  │   - bases_bob: np.ndarray               │                                  │
│  │   - total_rounds: int                   │                                  │
│  │   - expected_transmittance: float       │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ DetectionReport (from Bob)              │                                  │
│  │   - detected_indices                    │                                  │
│  │   - missing_indices                     │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ DetectionValidator.validate()           │──▶ ABORT if Chernoff fails       │
│  └─────────────────┬───────────────────────┘                                  │
│                    │ (validation passed)                                       │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ SiftingManager.identify_matching_bases()│                                  │
│  │   Output: I_0, I_1                      │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ SiftingManager.select_test_set()        │                                  │
│  │   Output: test_set (T), key_set         │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ SiftingManager.estimate_qber()          │                                  │
│  │   Output: e_obs                         │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ FiniteSizeAnalyzer                      │                                  │
│  │   - calculate_finite_size_penalty()     │                                  │
│  │   - compute_adjusted_qber()             │                                  │
│  │   Output: e_adj = e_obs + μ             │                                  │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ check_security_bounds(e_adj)            │──▶ ABORT if e_adj > 22%          │
│  └─────────────────┬───────────────────────┘                                  │
│                    │                                                           │
│                    ▼                                                           │
│  ┌─────────────────────────────────────────┐                                  │
│  │ Phase III Input                         │                                  │
│  │ SiftedData                              │                                  │
│  │   - key_indices: np.ndarray             │                                  │
│  │   - adjusted_qber: float                │                                  │
│  │   - statistical_penalty: float          │                                  │
│  │   - security_parameter: float           │                                  │
│  └─────────────────────────────────────────┘                                  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. MoSCoW Prioritized Roadmap

### 7.1 Priority Matrix

| Priority | Capability | Rationale | Effort Est. |
|----------|------------|-----------|-------------|
| **MUST** | Ordered Acknowledgment (`OrderedProtocolSocket`) | Security-critical; WSE depends on ordering | Medium |
| **MUST** | Finite-Size Penalty Calculation | Composable security requires adjusted QBER | Low |
| **MUST** | QBER Threshold Check with Adjustment | Current impl uses raw QBER | Low |
| **SHOULD** | Chernoff Bound Validation | Prevents post-selection attacks | Medium |
| **SHOULD** | Migrate `SiftingManager` to Protocol | Integrate with SquidASM generator model | Medium |
| **COULD** | Decoy State Statistics Separation | PNS attack mitigation (robust variant) | High |
| **WONT** | Real-time Channel Calibration | Out of scope for simulation environment | — |

### 7.2 Implementation Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase II Implementation Dependencies                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────────┐                            │
│                          │ Phase I Complete    │                            │
│                          │ (Prerequisite)      │                            │
│                          └──────────┬──────────┘                            │
│                                     │                                        │
│                                     ▼                                        │
│     ┌───────────────────────────────┴───────────────────────────────┐       │
│     │                                                                │       │
│     ▼                                ▼                               ▼       │
│ ┌─────────────┐            ┌─────────────────┐            ┌──────────────┐  │
│ │ Ordered     │            │ Finite-Size     │            │ Detection    │  │
│ │ Protocol    │            │ Penalty (μ)     │            │ Validator    │  │
│ │ Socket      │            │                 │            │ (Chernoff)   │  │
│ │ [MUST]      │            │ [MUST]          │            │ [SHOULD]     │  │
│ └──────┬──────┘            └────────┬────────┘            └──────┬───────┘  │
│        │                            │                            │          │
│        │                            │                            │          │
│        └────────────────────────────┼────────────────────────────┘          │
│                                     │                                        │
│                                     ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │ PhaseIIOrchestrator │                             │
│                         │ (Integration Point) │                             │
│                         │ [MUST]              │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                         │
│                                    ▼                                         │
│                         ┌─────────────────────┐                             │
│                         │ Phase III Ready     │                             │
│                         └─────────────────────┘                             │
│                                                                              │
│  Legend:                                                                     │
│  ───▶ Dependency (A must complete before B)                                │
│  [MUST] = Critical path                                                      │
│  [SHOULD] = Important but not blocking                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Files to Create/Modify

| File | Purpose | Priority | Status |
|------|---------|----------|--------|
| `ehok/protocols/ordered_messaging.py` | `OrderedProtocolSocket` wrapper | MUST | TO CREATE |
| `ehok/analysis/statistics.py` | Finite-size penalty and adjusted QBER | MUST | TO CREATE |
| `ehok/quantum/detection.py` | `DetectionReport`, `DetectionValidator` | SHOULD | TO CREATE |
| `ehok/core/sifting.py` | Update `check_qber_abort()` to use adjusted QBER | MUST | TO MODIFY |
| `ehok/protocols/phase_ii.py` | `PhaseIIOrchestrator` coordinator | MUST | TO CREATE |

---

## 8. Risks & Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Race condition in classical messaging** | Medium | Critical (security breach) | Implement strict acknowledgment protocol with timeouts |
| **Finite-size penalty too large for small experiments** | High | Medium (protocol aborts) | Document minimum viable test set sizes; adjust experimental parameters |
| **Chernoff validation too strict with unstable channels** | Medium | Medium (false positive aborts) | Allow configurable tolerance; implement channel calibration phase |
| **Generator model integration complexity** | Medium | Low (development delay) | Leverage existing SquidASM patterns from examples |

### 8.2 Theoretical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Parameter estimation before protocol** | Low | High | Per Erven et al., parameters must be stable; enforce pre-calibration |
| **Storage model deviation** | Low | Medium | NSM security holds for any noise above threshold; conservative bounds |

---

## 9. Conclusion

Phase II represents the security-critical transition from quantum physical layer to classical post-processing. The analysis identifies three primary implementation gaps requiring custom development:

1. **Ordered Acknowledgment Protocol**: Essential for WSE security; SquidASM's `ClassicalSocket` must be wrapped with acknowledgment logic.

2. **Finite-Size Penalty Calculation**: The $\mu$ parameter bridges sample statistics to composable security bounds; must be integrated into QBER threshold checks.

3. **Chernoff Bound Detection Validation**: Prevents post-selection attacks by validating Bob's detection report against expected channel parameters.

The legacy `ehok/core/sifting.py` provides a solid foundation for basis sifting and QBER estimation, but requires extension to incorporate the security-critical statistical adjustments mandated by the theoretical literature.

Upon completion of Phase II implementation, the protocol will have validated its security gatekeeper role, producing sifted key indices and an adjusted QBER suitable for Phase III privacy amplification calculations.

---

## References

1. Schaffner, C., Terhal, B., & Wehner, S. (2009). *Robust Cryptography in the Noisy-Quantum-Storage Model*. Theory of Cryptography Conference.

2. Erven, C., et al. (2014). *An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model*. arXiv:1308.5098v4.

3. Lupo, C., Peat, J.T., Andersson, E., & Kok, P. (2023). *Error-tolerant oblivious transfer in the noisy-storage model*.

4. Damgård, I., Fehr, S., Salvail, L., & Schaffner, C. (2005). *Cryptography in the Bounded Quantum-Storage Model*. FOCS.

5. Lemus, M., et al. (2020). *Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation*. arXiv:1909.11701.

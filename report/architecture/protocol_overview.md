# 3.1 Protocol Overview: Four-Phase Pipeline

## Introduction

The Caligo project implements 1-out-of-2 Oblivious Transfer (OT) via a four-phase execution pipeline. This section provides a formal specification of the protocol's state machine, phase boundaries, and data flow contracts.

## Protocol Definition

### 1-out-of-2 Oblivious Transfer

**Definition (ε-secure 1-2 ROT)** [Schaffner et al., 2009]

A protocol between Alice (sender) and Bob (receiver) implementing $(n, \epsilon)$-secure Random Oblivious Transfer if:

1. **Correctness**: Bob obtains $S_C$ where $C \in \{0, 1\}$ is his choice bit
2. **Sender Privacy**: Bob learns negligible information about $S_{1-C}$
3. **Receiver Privacy**: Alice learns negligible information about $C$
4. **Security Parameter**: Both privacy guarantees hold except with probability $\epsilon$

**Output**:
- Alice: $(S_0, S_1)$ where $S_0, S_1 \in \{0,1\}^{\ell}$
- Bob: $(S_C, C)$ where $C \in \{0,1\}$

## Four-Phase State Machine

### State Transition Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                              │
│  • Parse NSMParameters (Δt, r, ν, η, F)                             │
│  • Validate security constraints                                    │
│  • Initialize quantum/classical channels                            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────────┐
        │               PHASE I: QUANTUM                    │
        │                                                   │
        │  Objective: Generate EPR pairs, measure in        │
        │             random bases, enforce Δt              │
        │                                                   │
        │  Alice Operations:                                │
        │    1. Create EPR pair with Bob                    │
        │    2. Choose random basis b_A ∈ {Z, X}            │
        │    3. Measure in basis b_A → outcome m_A          │
        │    4. Buffer (m_A, b_A, round_id)                 │
        │                                                   │
        │  Bob Operations (symmetric):                      │
        │    1. Receive EPR half                            │
        │    2. Choose random basis b_B ∈ {Z, X}            │
        │    3. Measure in basis b_B → outcome m_B          │
        │    4. Buffer (m_B, b_B, round_id)                 │
        │                                                   │
        │  NSM Enforcement:                                 │
        │    • TimingBarrier.mark_quantum_complete()        │
        │    • Wait Δt before basis revelation              │
        │                                                   │
        │  Output Contract: QuantumPhaseResult              │
        └───────────────────┬───────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────────┐
        │            PHASE II: SIFTING & QBER               │
        │                                                   │
        │  Objective: Discard basis mismatches, estimate    │
        │             channel error rate                    │
        │                                                   │
        │  Protocol Steps:                                  │
        │    1. Alice → Bob: Commitment to bases            │
        │       commit_A = SHA256(bases_A || salt)          │
        │                                                   │
        │    2. Bob → Alice: Reveal bases_B                 │
        │                                                   │
        │    3. Alice → Bob: Reveal (bases_A, salt)         │
        │       Bob verifies: SHA256(bases_A||salt)==commit │
        │                                                   │
        │    4. Sifting: Keep (m_A, m_B) where b_A == b_B   │
        │                                                   │
        │    5. QBER Estimation:                            │
        │       - Sample m bits for comparison              │
        │       - QBER = (# errors) / m                     │
        │       - Compute Hoeffding bound                   │
        │                                                   │
        │  Security Check:                                  │
        │    IF QBER >  0.11 → WARNING                      │
        │    IF QBER >  0.22 → ABORT                        │
        │                                                   │
        │  Output Contract: SiftingPhaseResult              │
        └───────────────────┬───────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────────┐
        │        PHASE III: INFORMATION RECONCILIATION      │
        │                                                   │
        │  Objective: Correct errors via LDPC codes         │
        │                                                   │
        │  Strategy Selection:                              │
        │    • Baseline: Alice knows QBER, selects rate     │
        │    • Blind: Bob-driven iterative reconciliation   │
        │                                                   │
        │  Block Processing:                                │
        │    1. Partition sifted key into blocks            │
        │    2. For each block:                             │
        │       a) Alice: Compute syndrome s = H·k_A        │
        │       b) Alice → Bob: syndrome s                  │
        │       c) Bob: BP decode k_B' given (k_B, s)       │
        │       d) Hash verification:                       │
        │          Alice: h_A = Hash(k_A)                   │
        │          Bob:   h_B = Hash(k_B')                  │
        │          Exchange and compare                     │
        │                                                   │
        │  Leakage Tracking:                                │
        │    total_leakage = syndrome_bits + hash_bits      │
        │                                                   │
        │  Output Contract: ReconciliationPhaseResult       │
        └───────────────────┬───────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────────┐
        │        PHASE IV: PRIVACY AMPLIFICATION            │
        │                                                   │
        │  Objective: Extract secure oblivious keys via     │
        │             universal hashing                     │
        │                                                   │
        │  Key Length Calculation (Lupo Formula):           │
        │    ℓ = ⌊n·h_min - |Σ| - 2·log₂(1/ε_sec) + 2⌋       │
        │                                                   │
        │    Where:                                         │
        │      n = reconciled key length                    │
        │      h_min = min-entropy rate (from NSM bounds)   │
        │      |Σ| = total leakage (syndrome + hash)        │
        │      ε_sec = security parameter (10^{-10})        │
        │                                                   │
        │  Extraction (Toeplitz Hashing):                   │
        │    1. Generate random Toeplitz matrix T           │
        │    2. extracted = T · reconciled_key              │
        │    3. |extracted| = ℓ bits                        │
        │                                                   │
        │  Key Derivation:                                  │
        │    • Alice: Split extracted into (S₀, S₁)         │
        │      S₀ = extracted[0 : ℓ/2]                      │
        │      S₁ = extracted[ℓ/2 : ℓ]                      │
        │                                                   │
        │    • Bob: Derive S_C based on choice bit C        │
        │      S_C = extracted ⊕ correction_function(C)    │
        │                                                   │
        │  Output Contract: ObliviousTransferOutput         │
        └───────────────────┬───────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    SUCCESS    │
                    │               │
                    │ Alice: (S₀,S₁)│
                    │ Bob: (S_C, C) │
                    └───────────────┘
```

## Phase Boundary Contracts

### Contract Algebra

Each phase boundary defines a **data transfer object (DTO)** with:
- **Pre-conditions**: $\text{PRE}_\phi$ — Required properties of input
- **Post-conditions**: $\text{POST}_\phi$ — Guaranteed properties of output
- **Invariants**: $\text{INV}_\phi$ — Properties preserved throughout execution

**Formal Contract**:
$$
\{\text{PRE}_\phi\} \quad \phi \quad \{\text{POST}_\phi\}
$$

### Phase I → Phase II Contract

**DTO**: `QuantumPhaseResult`

**Post-conditions**:
- `POST-Q-001`: $|\text{measurement\_outcomes}| = n_{\text{generated}}$
- `POST-Q-002`: $|\text{basis\_choices}| = n_{\text{generated}}$
- `POST-Q-003`: $\forall m \in \text{outcomes}, \, m \in \{0, 1\}$
- `POST-Q-004`: $\forall b \in \text{bases}, \, b \in \{0, 1\}$ (Z=0, X=1)
- `POST-Q-005`: $\text{timing\_barrier\_marked} = \texttt{True}$

**Invariants**:
- `INV-Q-001`: $\text{generation\_timestamp} \geq 0$
- `INV-Q-002`: $n_{\text{generated}} \leq n_{\text{requested}}$ (loss model)

### Phase II → Phase III Contract

**DTO**: `SiftingPhaseResult`

**Post-conditions**:
- `POST-S-001`: $|\text{alice\_sifted\_key}| = |\text{bob\_sifted\_key}| = n_{\text{sifted}}$
- `POST-S-002`: $0 \leq \text{qber\_estimate} \leq 1$
- `POST-S-003`: $\text{qber\_estimate} \leq \text{QBER\_THRESHOLD}$ (security check passed)
- `POST-S-004`: $\text{hoeffding\_bound.is\_valid} = \texttt{True}$

**Invariants**:
- `INV-S-001`: $n_{\text{sifted}} \leq n_{\text{raw}}$ (basis matching reduces size)
- `INV-S-002`: $\text{hamming\_distance}(\text{alice\_key}, \text{bob\_key}) \approx \text{qber} \times n_{\text{sifted}}$

### Phase III → Phase IV Contract

**DTO**: `ReconciliationPhaseResult`

**Post-conditions**:
- `POST-R-001`: $|\text{alice\_reconciled\_key}| = |\text{bob\_reconciled\_key}| = n_{\text{reconciled}}$
- `POST-R-002`: $\text{hamming\_distance}(\text{alice\_key}, \text{bob\_key}) = 0$ (perfect agreement)
- `POST-R-003`: $\text{total\_leakage} = \text{syndrome\_leakage} + \text{verification\_leakage}$
- `POST-R-004`: $\text{total\_leakage} \leq \text{leakage\_cap}$ (if enforced)

**Invariants**:
- `INV-R-001`: $n_{\text{reconciled}} \leq n_{\text{sifted}}$ (sampling for QBER reduces size)
- `INV-R-002`: $\text{syndrome\_leakage} = \sum_{i=1}^{N_{\text{blocks}}} |\text{syndrome}_i|$

### Phase IV → Application Contract

**DTO**: `ObliviousTransferOutput`

**Post-conditions**:
- `POST-A-001`: $|S_0| = |S_1| = \ell_{\text{final}}$
- `POST-A-002`: $|S_C| = \ell_{\text{final}}$
- `POST-A-003`: $\ell_{\text{final}} = \lfloor n \cdot h_{\min} - |\Sigma| - 2\log_2(1/\epsilon_{\text{sec}}) + 2 \rfloor$
- `POST-A-004`: $\ell_{\text{final}} > 0$ (viability check)
- `POST-A-005`: Alice oblivious to $C$, Bob oblivious to $S_{1-C}$

**Security Guarantee**:
$$
\Pr[\text{Eve distinguishes } S_C \text{ from random}] \leq \epsilon_{\text{sec}}
$$

## Execution Modes

### Mode D (Domain Logic)

**Characteristics**:
- SquidASM-independent execution
- Phases executed with pre-generated data
- Ideal for unit testing and parameter sweeps

**Entry Point**:
```python
from caligo.quantum import EPRGenerator
from caligo.sifting import Sifter, QBEREstimator
from caligo.reconciliation import ReconciliationOrchestrator
from caligo.amplification import OTOutputFormatter

# Phase I: Pre-generated data
quantum_result = QuantumPhaseResult(
    measurement_outcomes=alice_outcomes,
    basis_choices=alice_bases,
    # ... other fields
)

# Phase II: Sifting
sifter = Sifter()
sifting_result = sifter.sift(quantum_result_alice, quantum_result_bob)

# Phase III: Reconciliation
orchestrator = ReconciliationOrchestrator(matrix_manager)
recon_result = orchestrator.reconcile_all_blocks(
    alice_sifted=sifting_result.alice_sifted_key,
    bob_sifted=sifting_result.bob_sifted_key,
    qber=sifting_result.qber_estimate,
)

# Phase IV: Amplification
formatter = OTOutputFormatter()
ot_output = formatter.format(
    alice_extracted=extracted_alice,
    bob_extracted=extracted_bob,
    bob_choice_bit=1,
)
```

### Mode E (Execution via SquidASM)

**Characteristics**:
- Full quantum network simulation
- NetQASM EPR socket operations
- Classical channel with simulated latency
- NSM timing enforcement via `TimingBarrier`

**Entry Point**:
```python
from caligo.protocol import run_protocol, ProtocolParameters
from caligo.simulation import NSMParameters

params = ProtocolParameters(
    session_id="session-001",
    nsm_params=NSMParameters(
        delta_t_ns=1_000_000,
        storage_noise_r=0.75,
        detection_eff_eta=0.85,
    ),
    num_pairs=10_000,
)

ot_output, raw_results = run_protocol(params, bob_choice_bit=1)
```

## Abort Conditions

The protocol may abort at any phase if security constraints are violated.

### Abort Reasons (Enumeration)

```python
class AbortReason(str, Enum):
    # Phase I
    QUANTUM_GENERATION_FAILED = "PHASE_I_EPR_GENERATION_FAILED"
    QUANTUM_TIMING_VIOLATED = "PHASE_I_TIMING_CONSTRAINT_VIOLATED"
    QUANTUM_INSUFFICIENT_PAIRS = "PHASE_I_INSUFFICIENT_EPR_PAIRS"
    
    # Phase II
    SIFTING_QBER_EXCEEDED = "PHASE_II_QBER_THRESHOLD_EXCEEDED"
    SIFTING_COMMITMENT_FAILED = "PHASE_II_BASIS_COMMITMENT_VIOLATED"
    SIFTING_INSUFFICIENT_SIFTED = "PHASE_II_INSUFFICIENT_SIFTED_BITS"
    
    # Phase III
    RECONCILIATION_FAILED = "PHASE_III_LDPC_DECODE_FAILED"
    RECONCILIATION_VERIFICATION_FAILED = "PHASE_III_HASH_MISMATCH"
    RECONCILIATION_LEAKAGE_CAP_EXCEEDED = "PHASE_III_LEAKAGE_CAP_EXCEEDED"
    
    # Phase IV
    AMPLIFICATION_ENTROPY_DEPLETED = "PHASE_IV_INSUFFICIENT_ENTROPY"
    AMPLIFICATION_ZERO_LENGTH = "PHASE_IV_ZERO_FINAL_LENGTH"
```

### Abort State Transition

```
Any Phase → ABORT → {
    Log diagnostic event with AbortReason
    Cleanup resources (close sockets, deallocate qubits)
    Return aborted=True to orchestrator
    TERMINATE
}
```

## Performance Characteristics

### Computational Complexity

| Phase | Operation | Complexity | Bottleneck |
|-------|-----------|-----------|------------|
| I | EPR Generation | $O(n)$ | Network latency, quantum ops |
| II | Basis Commitment | $O(n)$ | SHA-256 hashing |
| II | Sifting | $O(n)$ | Basis comparison |
| II | QBER Estimation | $O(m)$ where $m \ll n$ | Sampling |
| III | LDPC Encoding | $O(n \cdot d_c)$ | Sparse matrix-vector multiply |
| III | BP Decoding | $O(i \cdot n \cdot d_v)$ | Belief propagation iterations |
| IV | Toeplitz Hashing | $O(n \cdot \ell)$ | Matrix multiplication |

Where:
- $n$ = key length
- $d_c$ = check node degree (LDPC)
- $d_v$ = variable node degree (LDPC)
- $i$ = BP iterations (typically 50–100)
- $\ell$ = final extracted key length

### Communication Complexity

| Phase | Direction | Size | Type |
|-------|-----------|------|------|
| I | Quantum | $O(n)$ qubits | EPR pairs |
| II | Alice → Bob | $O(1)$ | Commitment (32 bytes) |
| II | Bob → Alice | $O(n)$ bits | Basis choices |
| II | Alice → Bob | $O(n)$ bits + salt | Basis revelation |
| III | Alice → Bob | $O(n \cdot R)$ bits | Syndromes (R = code rate) |
| III | Both directions | $O(\log n)$ bits | Hash verification |
| IV | Alice → Bob | $O(\ell)$ bits | Toeplitz seed (optional) |

## Security Analysis

### Threat Model

**Adversary (Eve)**:
- Intercepts quantum channel with detection efficiency $\eta_E$
- Has bounded quantum storage with parameters $(r, \nu)$
- Can access classical channel (but not modify due to authentication)
- Limited to individual attacks (no coherent attacks across rounds)

**Security Assumption (NSM)**:
$$
C_N \cdot \nu < \frac{1}{2}
$$

Where $C_N = 1 - h(\rho)$ and $\rho$ is the depolarization probability from storage noise parameter $r$.

### Security Proof Sketch

1. **Phase I Security**: EPR pairs measured in random bases → computational basis outcomes reveal no information about X-basis outcomes (and vice versa) due to complementarity.

2. **Phase II Security**: Commitment scheme prevents basis-dependent attacks. Honest parties abort if $\text{QBER} > 11\%$, ensuring Eve's mutual information is bounded.

3. **Phase III Security**: Syndrome leakage $|\Sigma|$ is accounted for in Phase IV. Error correction does not reduce security if leakage is properly tracked.

4. **Phase IV Security**: Leftover hash lemma (quantum version) guarantees:
   $$
   \delta(S_C, U_\ell) \leq \epsilon_{\text{sec}}
   $$
   where $\delta$ is trace distance and $U_\ell$ is uniform distribution.

**Result**: The protocol achieves $(\ell, \epsilon_{\text{sec}})$-secure 1-2 OT under the NSM.

## References

- Schaffner, C., et al. (2009). "Robust cryptography in the noisy-quantum-storage model." *Phys. Rev. A*, 79(3), 032308.
- König, R., et al. (2012). "Unconditional security from noisy quantum storage." *IEEE Trans. Inf. Theory*, 58(3), 1962-1984.
- Erven, C., et al. (2014). "An experimental implementation of oblivious transfer in the noisy storage model." *Nat. Commun.*, 5, 3418.
- Lupo, C., et al. (2023). "Practical quantum oblivious key distribution." arXiv:2305.xxxxx.

# E-HOK Phase Boundary Contracts

This document describes the dataclass contracts at phase boundaries for the E-HOK
protocol, as defined in [sprint_0_specification.md](../implementation%20plan/sprint_0_specification.md)
(INFRA-002).

## Overview

The E-HOK protocol is divided into four phases, each with well-defined input/output
contracts. These contracts enable:

- **Deterministic parity tests** between legacy and new implementations
- **Type-safe data exchange** between protocol phases
- **Runtime validation** of protocol invariants via Design-by-Contract

## Contract Map

| Phase Boundary     | Input Contract        | Output Contract            | Module Location |
|--------------------|----------------------|----------------------------|-----------------|
| Phase I → Phase II | —                    | `QuantumPhaseOutput`       | `ehok/core/data_structures.py` |
| Phase II → Phase III | `QuantumPhaseOutput` | `SiftedKeyData`            | `ehok/core/data_structures.py` |
| Phase III → Phase IV | `SiftedKeyData`      | `ReconciledKeyData`        | `ehok/core/data_structures.py` |
| Phase IV output    | `ReconciledKeyData`  | `ObliviousTransferOutput`  | `ehok/core/data_structures.py` |

## Contract Descriptions

### Phase I → Phase II: `QuantumPhaseOutput`

**Purpose**: Captures the complete output of Phase I quantum generation, including
timing markers for NSM causal barrier verification.

**Key Fields**:
- `outcomes_alice`, `outcomes_bob`: Measurement outcomes (uint8, values 0/1)
- `bases_alice`, `bases_bob`: Basis choices (uint8, 0=Z, 1=X)
- `timing_markers`: List of `TimingMarker` for causal ordering audit
- `commitment`: Optional `CommitmentRecord` for commit-then-reveal
- `missing_rounds`: Indices where Bob reported no detection

**Invariants** (from phase_I_analysis.md):
- POST-PHI-001: All arrays have equal length `n_pairs`
- POST-PHI-002: All arrays have dtype uint8
- INV-PHI-001: Timing markers must show commitment before basis reveal

### Phase II → Phase III: `SiftedKeyData`

**Purpose**: Represents sifted key material ready for reconciliation, with QBER
estimates including finite-size statistical penalties.

**Key Fields**:
- `key_alice`, `key_bob`: Sifted key bits (uint8)
- `i_0_indices`, `i_1_indices`: Index partitions for OT
- `observed_qber`, `adjusted_qber`, `statistical_penalty`: Error estimates

**Invariants** (from phase_II_analysis.md):
- `adjusted_qber = observed_qber + statistical_penalty`
- `|i_0_indices| + |i_1_indices| == sifted_length`

### Phase III → Phase IV: `ReconciledKeyData`

**Purpose**: Error-corrected key material with complete leakage accounting.

**Key Fields**:
- `reconciled_key`: Error-corrected key bits
- `total_syndrome_bits`, `total_hash_bits`, `total_leakage`: Leakage tracking
- `safety_cap_bits`, `safety_cap_utilization`: Safety cap enforcement

**Invariants** (from phase_III_analysis.md):
- `total_leakage = total_syndrome_bits + total_hash_bits`
- `safety_cap_utilization = total_leakage / safety_cap_bits`

### Phase IV Output: `ObliviousTransferOutput`

**Purpose**: Final 1-out-of-2 Oblivious Transfer output.

**Key Fields**:
- `alice_key_0`, `alice_key_1`: Alice's two keys
- `bob_key`: Bob's chosen key
- `bob_choice_bit`: Bob's choice (0 or 1)
- `entropy_bound_used`: Which entropy bound was applied ("dupuis_konig", "lupo", "max_bound")

**Invariants** (from phase_IV_analysis.md):
- OT Correctness: `bob_key == alice_key_{bob_choice_bit}`
- All keys have length `final_length`

## Supporting Dataclasses

### `TimingMarker`

Records protocol timing events for NSM causal barrier enforcement.

```python
@dataclass
class TimingMarker:
    event_type: str       # e.g., "COMMITMENT_SENT", "TIMING_BARRIER_END"
    timestamp_ns: int     # Simulation timestamp
    description: str      # Human-readable description
```

### `CommitmentRecord`

Records cryptographic commitments for audit.

```python
@dataclass
class CommitmentRecord:
    commitment_hash: bytes
    salt: bytes
    timestamp_ns: int
    verified: Optional[bool]
    data_length: int
```

### `ProtocolTranscript`

Complete protocol execution transcript.

```python
@dataclass
class ProtocolTranscript:
    session_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    final_phase: ProtocolPhase
    abort_reason: Optional[AbortReason]
    warnings: List[WarningCode]
    timing_markers: List[TimingMarker]
```

## Abort Taxonomy

The protocol defines a structured abort code taxonomy:

| Code Pattern       | Phase | Description |
|--------------------|-------|-------------|
| `ABORT-I-*`        | I     | Feasibility, timing violations |
| `ABORT-II-*`       | II    | Detection anomaly, QBER limits |
| `ABORT-III-*`      | III   | Leakage cap, reconciliation failure |
| `ABORT-IV-*`       | IV    | Entropy depletion |

See `AbortReason` enum in `ehok/core/data_structures.py` for complete list.

## Usage Example

```python
from ehok.core.data_structures import (
    QuantumPhaseOutput,
    SiftedKeyData,
    TimingMarker,
)
import numpy as np

# Create Phase I output
n_pairs = 1000
output = QuantumPhaseOutput(
    outcomes_alice=np.random.randint(0, 2, n_pairs, dtype=np.uint8),
    outcomes_bob=np.random.randint(0, 2, n_pairs, dtype=np.uint8),
    bases_alice=np.random.randint(0, 2, n_pairs, dtype=np.uint8),
    bases_bob=np.random.randint(0, 2, n_pairs, dtype=np.uint8),
    n_pairs=n_pairs,
    start_timestamp_ns=0,
    end_timestamp_ns=1_000_000,
    timing_markers=[
        TimingMarker("QUANTUM_START", 0, "EPR generation started"),
        TimingMarker("QUANTUM_END", 1_000_000, "EPR generation completed"),
    ],
)

# Invariants are validated in __post_init__
# Invalid data will raise ValueError
```

## Testing

Run contract validation tests:

```bash
pytest ehok/tests/test_contracts.py -v -m unit
```

The tests validate:
- 50+ randomized valid instances per dataclass
- 25+ invalid instances per dataclass (wrong dtype, inconsistent lengths, etc.)

## References

- [sprint_0_specification.md](../implementation%20plan/sprint_0_specification.md) — INFRA-002 requirements
- [phase_I_analysis.md](../implementation%20plan/phase_I_analysis.md) — Timing semantics, NSM invariants
- [phase_II_analysis.md](../implementation%20plan/phase_II_analysis.md) — Sifting, QBER thresholds
- [phase_III_analysis.md](../implementation%20plan/phase_III_analysis.md) — Wiretap cost, leakage
- [phase_IV_analysis.md](../implementation%20plan/phase_IV_analysis.md) — NSM max bound

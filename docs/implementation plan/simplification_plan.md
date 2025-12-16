# Major Simplification & Decoupling Plan

**Author:** Lead Software Architect  
**Date:** December 16, 2025  
**Objective:** Transform the `ehok` package into a lean, domain-driven architecture adhering to a strict 200 LOC limit per module.

---

## 1. Executive Summary

The current `ehok` codebase suffers from **architecture astronaut syndrome**—excessive abstraction layers (interfaces, factories), mixed responsibilities in bloated modules, and a directory structure that obscures rather than reveals the protocol's domain logic.

### Key Problems Identified

| Issue | Evidence | Impact |
|-------|----------|--------|
| **Bloated modules** | `data_structures.py` (1076 LOC), `ordered_messaging.py` (1021 LOC), `nsm_bounds.py` (747 LOC) | Violates SRP, hard to navigate |
| **Redundant abstraction** | `interfaces/` + `implementations/factories.py` | Unnecessary indirection for single-implementation strategies |
| **Configuration sprawl** | `configs/protocol_config.py` (deprecated) + `core/config.py` | Confusing, duplicate logic |
| **Misplaced assets** | LDPC matrices in `configs/` | Configuration vs. generated artifacts confusion |
| **Non-domain packaging** | `core/`, `implementations/` | Package names don't reflect protocol phases |
| **Dead code** | Deprecated modules, unused noise estimator interface | Technical debt |

### Target State

A **domain-driven** package where:
- Directory structure mirrors the **4-phase E-HOK protocol**
- Every module ≤ 200 LOC
- No abstract interfaces for single-implementation strategies
- Configuration external to package
- LDPC generation as utility, matrices as external assets

---

## 2. Domain Model Analysis

The E-HOK protocol has **four distinct phases** with clear boundaries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           E-HOK PROTOCOL PHASES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE I: Quantum Generation                                                │
│  └── EPR generation, basis selection, measurement, NSM timing barrier      │
│      Domain concepts: EPR pairs, bases, outcomes, Δt timing                │
│                                                                             │
│  PHASE II: Sifting & Estimation                                             │
│  └── Commitment, detection report, basis reveal, QBER estimation           │
│      Domain concepts: Commitment, sifting, test set, QBER, finite-size μ   │
│                                                                             │
│  PHASE III: Information Reconciliation                                      │
│  └── LDPC syndrome, BP decoding, leakage tracking, hash verification       │
│      Domain concepts: Syndrome, LDPC codes, wiretap cost |Σ|               │
│                                                                             │
│  PHASE IV: Privacy Amplification                                            │
│  └── NSM entropy bounds, Toeplitz hashing, OT output formatting            │
│      Domain concepts: Max Bound, h_min(r), Toeplitz matrix, S_0/S_1/S_C    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Cutting Concerns

| Concern | Current Location | Proper Location |
|---------|-----------------|-----------------|
| NSM security bounds | `analysis/nsm_bounds.py` | `nsm/` (dedicated package) |
| Protocol timing | `core/timing.py` | `nsm/timing.py` |
| Feasibility checking | `core/feasibility.py` | `nsm/feasibility.py` |
| Socket communication | `protocols/ordered_messaging.py` | `connection/` |
| Quantum operations | `quantum/` | `quantum/` (keep, decompose) |
| LDPC utilities | `implementations/reconciliation/` | `ldpc/` |

---

## 3. Proposed Architecture

### 3.1 New Package Structure

```
ehok/
├── __init__.py                      # Public API (< 50 LOC)
│
├── types/                           # Domain primitives (formerly core/)
│   ├── __init__.py                  # Type exports
│   ├── keys.py                      # ObliviousKey, AliceKey, BobKey (< 100 LOC)
│   ├── records.py                   # MeasurementRecord, BlockResult (< 100 LOC)
│   ├── contracts.py                 # Phase boundary dataclasses (< 150 LOC)
│   └── exceptions.py                # Exception hierarchy (< 100 LOC)
│
├── nsm/                             # Noisy Storage Model security layer
│   ├── __init__.py                  
│   ├── bounds.py                    # Max Bound, Γ function, entropy (< 200 LOC)
│   ├── feasibility.py               # Pre-flight checker (< 150 LOC)
│   ├── timing.py                    # TimingEnforcer, Δt barrier (< 150 LOC)
│   └── parameters.py                # NSM config: r, ν, ε_sec (< 100 LOC)
│
├── quantum/                         # Phase I: Quantum Generation
│   ├── __init__.py
│   ├── adapter.py                   # PhysicalModelAdapter → SquidASM config (< 150 LOC) [CRITICAL]
│   ├── epr.py                       # EPR generation via SquidASM (< 150 LOC)
│   ├── basis.py                     # Basis selection (Z/X) (< 80 LOC)
│   ├── measurement.py               # Measurement buffering (< 100 LOC)
│   └── batching.py                  # Memory-constrained batching (< 120 LOC)
│
├── sifting/                         # Phase II: Sifting & Estimation
│   ├── __init__.py
│   ├── commitment.py                # SHA256 commitment (< 120 LOC)
│   ├── sifter.py                    # I_0/I_1 partitioning, test set (< 150 LOC)
│   ├── qber.py                      # QBER estimation + μ penalty (< 120 LOC)
│   └── validation.py                # Detection validation, Chernoff (< 150 LOC)
│
├── reconciliation/                  # Phase III: Information Reconciliation
│   ├── __init__.py
│   ├── syndrome.py                  # Syndrome computation (< 100 LOC)
│   ├── decoder.py                   # BP decoder (< 180 LOC)
│   ├── verifier.py                  # Hash verification (< 100 LOC)
│   └── leakage.py                   # Wiretap cost tracking (< 120 LOC)
│
├── amplification/                   # Phase IV: Privacy Amplification
│   ├── __init__.py
│   ├── toeplitz.py                  # Toeplitz matrix hashing (< 150 LOC)
│   ├── key_length.py                # Secure length calculation (< 120 LOC)
│   └── formatter.py                 # OT output: S_0, S_1, S_C (< 150 LOC)
│
├── ldpc/                            # LDPC code utilities (external to protocol)
│   ├── __init__.py
│   ├── generator.py                 # PEG matrix generation (< 200 LOC)
│   ├── manager.py                   # Matrix loading, checksum (< 150 LOC)
│   └── distributions.py             # Degree distribution handling (< 100 LOC)
│
├── connection/                      # Network communication layer
│   ├── __init__.py
│   ├── socket.py                    # Classical socket wrapper (< 100 LOC)
│   ├── messaging.py                 # Message envelope, ACK (< 150 LOC)
│   └── ordering.py                  # Commit-then-reveal state machine (< 150 LOC)
│
├── protocol/                        # Protocol orchestration
│   ├── __init__.py
│   ├── config.py                    # Unified ProtocolConfig (< 150 LOC)
│   ├── orchestrator.py              # Phase sequencing (< 200 LOC)
│   ├── alice.py                     # Alice SquidASM Program (< 180 LOC)
│   └── bob.py                       # Bob SquidASM Program (< 180 LOC)
│
├── utils/                           # Cross-cutting utilities
│   ├── __init__.py
│   ├── logging.py                   # Logging infrastructure (< 150 LOC)
│   ├── entropy.py                   # Binary entropy, shared math (< 50 LOC)
│   └── generated_ldpc.py            # generated ldpc matrices (< 150 LOC)
│
└── tests/                           # Test suite (external to LOC limit)
    ├── conftest.py
    ├── unit/
    │   ├── test_nsm_bounds.py
    │   ├── test_sifting.py
    │   ├── test_reconciliation.py
    │   └── test_amplification.py
    ├── integration/
    │   ├── test_phase_contracts.py
    │   └── test_protocol_flow.py
    └── system/
        └── test_e2e.py
```

### 3.2 Module Decomposition Plan

#### 3.2.1 `data_structures.py` (1076 LOC → 4 modules ~400 LOC total)

**Current:** Monolithic file with enums, phase contracts, LDPC results, OT outputs.

**Decomposition:**

| New Module | Contents | Est. LOC |
|------------|----------|----------|
| `types/keys.py` | `ObliviousKey`, `AliceObliviousKey`, `BobObliviousKey` | ~100 |
| `types/records.py` | `MeasurementRecord`, `LDPCBlockResult`, `LDPCReconciliationResult` | ~100 |
| `types/contracts.py` | `QuantumPhaseOutput`, `SiftedKeyData`, `ReconciledKeyData`, `ObliviousTransferOutput` | ~150 |
| `types/exceptions.py` | All enums (`ProtocolPhase`, `AbortReason`, `WarningCode`) + exceptions | ~100 |

#### 3.2.2 `ordered_messaging.py` (1021 LOC → 3 modules ~400 LOC total)

**Current:** Message types, socket state machine, envelope serialization, data structures all in one file.

**Decomposition:**

| New Module | Contents | Est. LOC |
|------------|----------|----------|
| `connection/messaging.py` | `MessageType`, `MessageEnvelope`, serialization | ~150 |
| `connection/ordering.py` | `SocketState`, `OrderingViolationError`, state machine logic | ~150 |
| `connection/socket.py` | `OrderedProtocolSocket` wrapper, ACK handling | ~100 |

#### 3.2.3 `nsm_bounds.py` (747 LOC → 2 modules ~350 LOC total)

**Current:** Mathematical functions + calculator class + feasibility enum all mixed.

**Decomposition:**

| New Module | Contents | Est. LOC |
|------------|----------|----------|
| `nsm/bounds.py` | `gamma_function`, `collision_entropy_rate`, `max_bound_entropy_rate`, `channel_capacity` | ~200 |
| `nsm/feasibility.py` | `FeasibilityResult`, `FeasibilityChecker`, validation logic | ~150 |

#### 3.2.4 `statistical_validation.py` (707 LOC → 2 modules ~300 LOC total)

**Current:** Detection validation + QBER adjustment + finite-size penalty in one file.

**Decomposition:**

| New Module | Contents | Est. LOC |
|------------|----------|----------|
| `sifting/validation.py` | `DetectionValidator`, Chernoff bounds | ~150 |
| `sifting/qber.py` | `QBERAdjuster`, finite-size penalty μ, QBER threshold checks | ~150 |

#### 3.2.5 `ldpc_reconciliator.py` (677 LOC → 3 modules ~450 LOC total)

**Current:** Rate selection + syndrome computation + BP orchestration + verification.

**Decomposition:**

| New Module | Contents | Est. LOC |
|------------|----------|----------|
| `reconciliation/syndrome.py` | `compute_syndrome`, rate selection, shortening | ~100 |
| `reconciliation/decoder.py` | BP decoding logic, LLR computation | ~200 |
| `reconciliation/verifier.py` | Hash verification, block result construction | ~150 |

---

## 4. Elimination Plan

### 4.1 Interfaces Package (DELETE)

**Rationale:** The Strategy pattern is over-engineered when there's only one implementation. Direct imports are cleaner.

| Interface | Current Implementation | Action |
|-----------|----------------------|--------|
| `ICommitmentScheme` | `SHA256Commitment` | Keep SHA256 as single class, delete interface |
| `IReconciliator` | `LDPCReconciliator` | LDPC is the only implementation, delete interface |
| `IPrivacyAmplifier` | `ToeplitzAmplifier` | Toeplitz is canonical, delete interface |
| `ISamplingStrategy` | `RandomSamplingStrategy` | Inline into `sifting/sifter.py` |
| `INoiseEstimator` | `SimpleNoiseEstimator` | Delete entirely (unused) |

### 4.2 Factories Module (DELETE)

**Rationale:** With interfaces removed, factory pattern adds no value.

**Current:** `implementations/factories.py` (90 LOC)

**Action:** Delete. Move any necessary construction logic into `protocol/config.py`.

### 4.3 Configs Package (DELETE)

**Rationale:** Configuration belongs outside the package. LDPC matrices are generated assets.

| Current Location | Action |
|-----------------|--------|
| `configs/protocol_config.py` | DELETE (deprecated, merged into `core/config.py`) |
| `configs/network_*.yaml` | MOVE to `qia-challenge-2025/config/` (outside ehok) |
| `configs/ldpc_matrices/` | MOVE to `qia-challenge-2025/assets/ldpc/` |
| `configs/test_ldpc_matrices/` | MOVE to `qia-challenge-2025/assets/ldpc_test/` |
| `configs/ldpc_degree_distributions.yaml` | MOVE to `ldpc/distributions.py` as constants |
| `configs/generate_ldpc.py` | MOVE to `ldpc/generator.py` |

### 4.4 Examples Package (DELETE)

**Rationale:** Examples belong outside the distributable package.

| Current Location | Action |
|-----------------|--------|
| `examples/run_baseline.py` | MOVE to `qia-challenge-2025/scripts/` |
| `examples/debug_qber.py` | MOVE to `qia-challenge-2025/scripts/` |
| `examples/test_noise.py` | DELETE (test file in wrong location) |

### 4.5 Dead Code Removal

| File | Reason | Action |
|------|--------|--------|
| `implementations/noise/simple_noise_estimator.py` | Unused, trivial | DELETE |
| `quantum/noise_adapter.py` (709 LOC) | Complex adapter with unclear usage | REVIEW & potentially delete |
| `protocols/__init__.py` (102 LOC) | Over-exports internal components | SIMPLIFY to ~20 LOC |

---

## 5. Module Mapping: Old → New

### 5.1 Core Package

| Old Path | New Path | Notes |
|----------|----------|-------|
| `core/__init__.py` | `types/__init__.py` | Rename |
| `core/data_structures.py` | `types/{keys,records,contracts}.py` | Split |
| `core/exceptions.py` | `types/exceptions.py` | Move |
| `core/constants.py` | `protocol/config.py` + `nsm/parameters.py` | Split by domain |
| `core/config.py` | `protocol/config.py` | Move |
| `core/timing.py` | `nsm/timing.py` | Move |
| `core/feasibility.py` | `nsm/feasibility.py` | Move |
| `core/sifting.py` | `sifting/sifter.py` | Move |
| `core/oblivious_formatter.py` | `amplification/formatter.py` | Move |

### 5.2 Analysis Package

| Old Path | New Path | Notes |
|----------|----------|-------|
| `analysis/nsm_bounds.py` | `nsm/bounds.py` | Move & split |
| `analysis/metrics.py` | DELETE | Minimal, unused |

### 5.3 Implementations Package (ABOLISH)

| Old Path | New Path | Notes |
|----------|----------|-------|
| `implementations/commitment/sha256_commitment.py` | `sifting/commitment.py` | Move |
| `implementations/commitment/merkle_commitment.py` | DELETE | Not used |
| `implementations/reconciliation/ldpc_*.py` | `reconciliation/*.py` + `ldpc/*.py` | Split |
| `implementations/privacy_amplification/*.py` | `amplification/*.py` | Move |
| `implementations/sampling/random_sampling.py` | `sifting/sifter.py` | Inline |
| `implementations/factories.py` | DELETE | Unnecessary |

### 5.4 Protocols Package

| Old Path | New Path | Notes |
|----------|----------|-------|
| `protocols/base.py` | `protocol/orchestrator.py` | Rename & refactor |
| `protocols/alice.py` | `protocol/alice.py` | Keep, simplify |
| `protocols/bob.py` | `protocol/bob.py` | Keep, simplify |
| `protocols/ordered_messaging.py` | `connection/{messaging,ordering,socket}.py` | Split |
| `protocols/leakage_manager.py` | `reconciliation/leakage.py` | Move |
| `protocols/statistical_validation.py` | `sifting/{validation,qber}.py` | Split |

### 5.5 Quantum Package

| Old Path | New Path | Notes |
|----------|----------|-------|
| `quantum/runner.py` | `quantum/epr.py` | Rename |
| `quantum/basis_selection.py` | `quantum/basis.py` | Rename |
| `quantum/measurement.py` | `quantum/measurement.py` | Keep |
| `quantum/batching_manager.py` | `quantum/batching.py` | Rename |
| `quantum/noise_adapter.py` | REVIEW | May be deleted |

---

## 6. Unified Configuration Strategy

### 6.1 New Configuration Hierarchy

```python
# protocol/config.py (< 150 LOC)

@dataclass(frozen=True)
class NSMParameters:
    """Adversary model assumptions."""
    storage_noise_r: float = 0.75
    storage_rate_nu: float = 0.002
    delta_t_ns: int = 1_000_000_000
    
@dataclass(frozen=True)  
class SecurityParameters:
    """Protocol security targets."""
    epsilon_sec: float = 1e-9
    epsilon_cor: float = 1e-15
    qber_warning: float = 0.11
    qber_abort: float = 0.22
    test_set_fraction: float = 0.1

@dataclass(frozen=True)
class QuantumParameters:
    """Quantum generation settings."""
    total_pairs: int = 10_000
    batch_size: int = 5
    max_qubits: int = 5

@dataclass(frozen=True)
class ReconciliationParameters:
    """LDPC reconciliation settings."""
    ldpc_frame_size: int = 4096
    max_iterations: int = 60
    bp_threshold: float = 1e-6

@dataclass
class ProtocolConfig:
    """Unified protocol configuration."""
    nsm: NSMParameters = field(default_factory=NSMParameters)
    security: SecurityParameters = field(default_factory=SecurityParameters)
    quantum: QuantumParameters = field(default_factory=QuantumParameters)
    reconciliation: ReconciliationParameters = field(default_factory=ReconciliationParameters)
    
    # External paths (not part of package)
    ldpc_matrix_dir: Path | None = None
    network_config_path: Path | None = None
```

### 6.2 YAML-Based External Configuration

The `ehok` package shall accept **ALL parameters via a unified YAML configuration file** placed outside the package. The package contains only code—no default configuration files, no example scripts, no generated assets.

#### 6.2.1 External Configuration Schema

```yaml
# qia-challenge-2025/config/ehok_config.yaml
# Unified E-HOK Protocol Configuration

# =============================================================================
# Physical Parameters (Device Characterization)
# Ref: Erven et al. (2014) Table I
# =============================================================================
physical:
  mu_pair_per_coherence: 3.145e-5    # Mean photon pair number per pulse
  eta_total_transmittance: 0.0150   # Total transmission efficiency (0, 1]
  e_det: 0.0093                     # Intrinsic detection error rate [0, 0.5]
  p_dark: 1.50e-8                   # Dark count probability per coherence time

# =============================================================================
# Noisy Storage Model (Adversary Assumptions)
# Ref: König et al. (2012), Schaffner et al. (2009)
# =============================================================================
nsm:
  storage_noise_r: 0.75             # Depolarizing retention r ∈ [0, 1]
  storage_rate_nu: 0.002            # Storage rate ν ∈ [0, 1]
  delta_t_ns: 1_000_000_000         # Wait time Δt in nanoseconds (1 second)
  
  # Memory T1/T2 for storage noise derivation (optional)
  memory_T1_ns: null                # Amplitude damping time [ns]
  memory_T2_ns: null                # Dephasing time [ns]

# =============================================================================
# Security Targets
# =============================================================================
security:
  epsilon_sec: 1.0e-9               # Security parameter (trace distance)
  epsilon_cor: 1.0e-15              # Correctness parameter (hash collision)
  qber_warning_threshold: 0.11      # Conservative QBER limit (Schaffner)
  qber_abort_threshold: 0.22        # Hard QBER limit (Lupo)
  test_set_fraction: 0.1            # Fraction of sifted bits for QBER estimation
  min_test_set_size: 100            # Minimum test bits required

# =============================================================================
# Quantum Generation (Phase I)
# =============================================================================
quantum:
  total_pairs: 10000                # Total EPR pairs to generate
  batch_size: 5                     # EPR pairs per batch (memory constraint)
  max_qubits: 5                     # Qubits available per node

# =============================================================================
# Information Reconciliation (Phase III)
# =============================================================================
reconciliation:
  ldpc_frame_size: 4096             # LDPC codeword length
  max_iterations: 60                # BP decoder max iterations
  bp_threshold: 1.0e-6              # BP convergence threshold
  leakage_safety_margin: 0.1        # Safety margin for wiretap cost
  max_leakage_fraction: 0.5         # Abort if leakage exceeds this fraction

# =============================================================================
# Privacy Amplification (Phase IV)
# =============================================================================
amplification:
  use_nsm_bounds: true              # Use NSM Max Bound (vs QKD bounds)
  min_output_bits: 64               # Minimum key length to avoid abort

# =============================================================================
# Connection & Timing
# =============================================================================
connection:
  ack_timeout_ns: 1_000_000_000     # ACK timeout (1 second)
  max_retries: 3                    # Message retry count
  
# =============================================================================
# Asset Paths (relative to config file location)
# =============================================================================
paths:
  ldpc_matrices: "../assets/ldpc"
  network_config: "network_baseline.yaml"
  output_dir: "../results"
  log_dir: "../logs"
```

#### 6.2.2 Project Directory Structure

```
qia-challenge-2025/
├── config/                          # External configuration
│   ├── ehok_config.yaml            # Primary protocol config
│   ├── ehok_config_test.yaml       # Test configuration (smaller batches)
│   ├── network_baseline.yaml       # SquidASM network topology
│   └── network_perfect.yaml        # Noiseless network for debugging
│
├── assets/                          # Generated/static assets
│   ├── ldpc/                       # Production LDPC matrices
│   │   └── ldpc_4096_rate*.npz
│   └── ldpc_test/                  # Test LDPC matrices
│       └── ldpc_128_rate*.npz
│
├── scripts/                         # Executable scripts (formerly examples/)
│   ├── run_baseline.py             # Execute baseline protocol
│   ├── generate_ldpc.py            # Generate LDPC matrices
│   └── analyze_results.py          # Post-run analysis
│
├── results/                         # Protocol execution outputs
│   └── run_YYYYMMDD_HHMMSS/
│
├── logs/                            # Log files
│
├── docs/                            # Documentation
│
├── ehok/                            # Package code ONLY
│   ├── __init__.py
│   ├── types/
│   ├── nsm/
│   ├── quantum/
│   ├── sifting/
│   ├── reconciliation/
│   ├── amplification/
│   ├── ldpc/
│   ├── connection/
│   ├── protocol/
│   ├── utils/
│   └── tests/
│
├── conftest.py                      # pytest fixtures
├── pyproject.toml
└── pytest.ini
```

#### 6.2.3 Configuration Loading in Package

```python
# ehok/protocol/config.py (< 150 LOC)

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ehok.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical device characterization (Erven et al. Table I)."""
    mu_pair_per_coherence: float = 3.145e-5
    eta_total_transmittance: float = 0.0150
    e_det: float = 0.0093
    p_dark: float = 1.50e-8


@dataclass(frozen=True)
class NSMParameters:
    """Noisy Storage Model adversary assumptions."""
    storage_noise_r: float = 0.75
    storage_rate_nu: float = 0.002
    delta_t_ns: int = 1_000_000_000
    memory_T1_ns: float | None = None
    memory_T2_ns: float | None = None


@dataclass(frozen=True)
class SecurityParameters:
    """Protocol security targets."""
    epsilon_sec: float = 1e-9
    epsilon_cor: float = 1e-15
    qber_warning_threshold: float = 0.11
    qber_abort_threshold: float = 0.22
    test_set_fraction: float = 0.1
    min_test_set_size: int = 100


@dataclass(frozen=True)
class QuantumParameters:
    """Quantum generation settings."""
    total_pairs: int = 10_000
    batch_size: int = 5
    max_qubits: int = 5


@dataclass(frozen=True)
class ReconciliationParameters:
    """LDPC reconciliation settings."""
    ldpc_frame_size: int = 4096
    max_iterations: int = 60
    bp_threshold: float = 1e-6
    leakage_safety_margin: float = 0.1
    max_leakage_fraction: float = 0.5


@dataclass(frozen=True)
class AmplificationParameters:
    """Privacy amplification settings."""
    use_nsm_bounds: bool = True
    min_output_bits: int = 64


@dataclass(frozen=True)
class ConnectionParameters:
    """Connection and timing settings."""
    ack_timeout_ns: int = 1_000_000_000
    max_retries: int = 3


@dataclass(frozen=True)
class PathParameters:
    """External asset paths."""
    ldpc_matrices: Path | None = None
    network_config: Path | None = None
    output_dir: Path | None = None
    log_dir: Path | None = None


@dataclass
class ProtocolConfig:
    """Unified protocol configuration loaded from YAML."""
    
    physical: PhysicalParameters = field(default_factory=PhysicalParameters)
    nsm: NSMParameters = field(default_factory=NSMParameters)
    security: SecurityParameters = field(default_factory=SecurityParameters)
    quantum: QuantumParameters = field(default_factory=QuantumParameters)
    reconciliation: ReconciliationParameters = field(default_factory=ReconciliationParameters)
    amplification: AmplificationParameters = field(default_factory=AmplificationParameters)
    connection: ConnectionParameters = field(default_factory=ConnectionParameters)
    paths: PathParameters = field(default_factory=PathParameters)
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> ProtocolConfig:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : Path | str
            Path to YAML configuration file.
        
        Returns
        -------
        ProtocolConfig
            Populated configuration object.
        
        Raises
        ------
        FileNotFoundError
            If configuration file does not exist.
        ValueError
            If configuration validation fails.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info("Loading configuration from %s", config_path)
        
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        
        # Resolve relative paths against config file location
        config_dir = config_path.parent
        
        return cls(
            physical=PhysicalParameters(**raw.get("physical", {})),
            nsm=NSMParameters(**raw.get("nsm", {})),
            security=SecurityParameters(**raw.get("security", {})),
            quantum=QuantumParameters(**raw.get("quantum", {})),
            reconciliation=ReconciliationParameters(**raw.get("reconciliation", {})),
            amplification=AmplificationParameters(**raw.get("amplification", {})),
            connection=ConnectionParameters(**raw.get("connection", {})),
            paths=cls._resolve_paths(raw.get("paths", {}), config_dir),
        )
    
    @staticmethod
    def _resolve_paths(paths_dict: dict[str, Any], base_dir: Path) -> PathParameters:
        """Resolve relative paths against configuration directory."""
        resolved = {}
        for key, value in paths_dict.items():
            if value is not None:
                path = Path(value)
                if not path.is_absolute():
                    path = (base_dir / path).resolve()
                resolved[key] = path
        return PathParameters(**resolved)
    
    @classmethod
    def default(cls) -> ProtocolConfig:
        """Create default configuration (for testing/development)."""
        return cls()
```

#### 6.2.4 Network Configuration (Separate YAML)

```yaml
# qia-challenge-2025/config/network_baseline.yaml
# SquidASM Network Configuration

nodes:
  - name: "Alice"
    qubits: 5
    gate_fidelity: 0.99
    T1: 1_000_000_000    # 1 second
    T2: 500_000_000      # 500 ms
    
  - name: "Bob"
    qubits: 5
    gate_fidelity: 0.99
    T1: 1_000_000_000
    T2: 500_000_000

links:
  - node1: "Alice"
    node2: "Bob"
    noise_type: "depolarise"    # "none", "depolarise", "bitflip"
    fidelity: 0.9907            # Derived from e_det = 0.0093
    t_cycle: 5000               # 5 μs cycle time

classical_links:
  - node1: "Alice"
    node2: "Bob"
    delay: 100_000              # 100 μs classical link delay
```

---

## 7. Expanded Decomposition Plan

This section provides detailed decomposition strategies for each bloated module, including class-level assignments and interface boundaries.

### 7.1 `data_structures.py` Decomposition (1076 LOC → 4 modules)

#### 7.1.1 `types/keys.py` (~100 LOC)

**Contents:**
```python
@dataclass
class ObliviousKey:
    """Base key representation for OT protocol."""
    bits: np.ndarray
    length: int
    creation_timestamp: float

@dataclass  
class AliceObliviousKey(ObliviousKey):
    """Alice's dual-key output (S_0, S_1)."""
    s0: np.ndarray
    s1: np.ndarray

@dataclass
class BobObliviousKey(ObliviousKey):
    """Bob's chosen key output (S_C, C)."""
    s_c: np.ndarray
    choice_bit: int
```

#### 7.1.2 `types/records.py` (~100 LOC)

**Contents:**
```python
@dataclass
class MeasurementRecord:
    """Single qubit measurement outcome."""
    index: int
    outcome: int
    basis: int
    timestamp_ns: int

@dataclass
class LDPCBlockResult:
    """Single LDPC block reconciliation result."""
    block_index: int
    converged: bool
    iterations: int
    syndrome_bits: int
    hash_verified: bool

@dataclass
class LDPCReconciliationResult:
    """Aggregate reconciliation statistics."""
    total_blocks: int
    successful_blocks: int
    total_leakage_bits: int
    block_results: list[LDPCBlockResult]
```

#### 7.1.3 `types/contracts.py` (~150 LOC)

**Contents:**
```python
@dataclass(frozen=True)
class QuantumPhaseOutput:
    """Phase I → Phase II boundary contract."""
    outcomes: np.ndarray
    bases: np.ndarray
    pair_count: int
    generation_time_ns: int

@dataclass(frozen=True)
class SiftedKeyData:
    """Phase II → Phase III boundary contract."""
    key_bits: np.ndarray
    matching_indices: np.ndarray
    mismatching_indices: np.ndarray
    test_indices: np.ndarray
    estimated_qber: float
    qber_adjusted: float

@dataclass(frozen=True)
class ReconciledKeyData:
    """Phase III → Phase IV boundary contract."""
    reconciled_key: np.ndarray
    total_leakage_bits: int
    reconciliation_result: LDPCReconciliationResult

@dataclass(frozen=True)
class ObliviousTransferOutput:
    """Phase IV final output."""
    alice_keys: AliceObliviousKey | None
    bob_key: BobObliviousKey | None
    protocol_succeeded: bool
    abort_reason: AbortReason | None
```

#### 7.1.4 `types/exceptions.py` (~100 LOC)

**Contents:**
```python
class ProtocolPhase(Enum):
    QUANTUM_GENERATION = 1
    SIFTING_ESTIMATION = 2
    RECONCILIATION = 3
    PRIVACY_AMPLIFICATION = 4

class AbortReason(Enum):
    QBER_EXCEEDED = "qber_exceeded"
    INSUFFICIENT_BITS = "insufficient_bits"
    RECONCILIATION_FAILED = "reconciliation_failed"
    LEAKAGE_CAP_EXCEEDED = "leakage_cap_exceeded"
    FEASIBILITY_FAILED = "feasibility_failed"
    TIMING_VIOLATION = "timing_violation"
    PROTOCOL_VIOLATION = "protocol_violation"

class WarningCode(Enum):
    HIGH_QBER = "high_qber"
    LOW_SIFTING_RATE = "low_sifting_rate"
    APPROACHING_LEAKAGE_CAP = "approaching_leakage_cap"

class EHOKProtocolError(Exception):
    """Base exception for protocol errors."""

class FeasibilityError(EHOKProtocolError):
    """Pre-flight feasibility check failed."""

class TimingViolationError(EHOKProtocolError):
    """NSM timing barrier violated."""

class LeakageCapExceededError(EHOKProtocolError):
    """Wiretap cost safety cap exceeded."""

class ReconciliationError(EHOKProtocolError):
    """Information reconciliation failed."""
```

### 7.2 `ordered_messaging.py` Decomposition (1021 LOC → 3 modules)

#### 7.2.1 `connection/messaging.py` (~150 LOC)

**Contents:**
```python
class MessageType(Enum):
    """Protocol message types with ordering semantics."""
    COMMITMENT = "commitment"
    COMMITMENT_ACK = "commitment_ack"
    DETECTION_REPORT = "detection_report"
    DETECTION_REPORT_ACK = "detection_report_ack"
    BASIS_REVEAL = "basis_reveal"
    BASIS_REVEAL_ACK = "basis_reveal_ack"
    SYNDROME = "syndrome"
    SYNDROME_ACK = "syndrome_ack"
    HASH_VERIFICATION = "hash_verification"
    KEY_CONFIRMATION = "key_confirmation"

@dataclass
class MessageEnvelope:
    """Typed message container with ACK tracking."""
    message_type: MessageType
    sequence_number: int
    payload: dict[str, Any]
    timestamp_ns: int
    requires_ack: bool = True
    
    def to_json(self) -> str: ...
    
    @classmethod
    def from_json(cls, data: str) -> MessageEnvelope: ...

# Payload dataclasses
@dataclass
class CommitmentPayload:
    commitment_hash: str
    round_count: int

@dataclass
class DetectionReportPayload:
    missing_indices: list[int]
    detection_count: int

@dataclass
class BasisRevealPayload:
    bases: list[int]
    total_rounds: int
```

#### 7.2.2 `connection/ordering.py` (~150 LOC)

**Contents:**
```python
class SocketState(Enum):
    """Commit-then-reveal state machine states."""
    INITIALIZED = "initialized"
    COMMITMENT_SENT = "commitment_sent"
    COMMITMENT_ACKED = "commitment_acked"
    DETECTION_SENT = "detection_sent"
    DETECTION_ACKED = "detection_acked"
    WAITING_TIMING_BARRIER = "waiting_timing_barrier"
    TIMING_BARRIER_SATISFIED = "timing_barrier_satisfied"
    BASIS_REVEALED = "basis_revealed"
    COMPLETED = "completed"
    ERROR = "error"

class OrderingViolationError(Exception):
    """Protocol message ordering violated."""

class OrderingStateMachine:
    """Enforces commit-then-reveal message ordering."""
    
    def __init__(self, role: str):
        self._state = SocketState.INITIALIZED
        self._role = role
    
    def validate_transition(self, message_type: MessageType) -> None:
        """Validate state transition is legal."""
        
    def transition(self, message_type: MessageType) -> None:
        """Execute state transition."""
        
    @property
    def can_reveal_basis(self) -> bool:
        """Check if basis reveal is currently legal."""
```

#### 7.2.3 `connection/socket.py` (~150 LOC)

**Contents:**
```python
class OrderedProtocolSocket:
    """
    Classical socket wrapper with ordering enforcement.
    
    Wraps SquidASM ClassicalSocket with:
    - Message envelope serialization
    - ACK tracking and timeout handling
    - State machine integration
    """
    
    def __init__(
        self,
        raw_socket: Any,  # SquidASM ClassicalSocket
        role: str,
        timing_enforcer: TimingEnforcer | None = None,
    ):
        self._socket = raw_socket
        self._state_machine = OrderingStateMachine(role)
        self._timing = timing_enforcer
        self._pending_acks: dict[int, MessageEnvelope] = {}
    
    def send_with_ack(
        self,
        envelope: MessageEnvelope,
        timeout_ns: int = 1_000_000_000,
    ) -> Generator[EventExpression, None, None]:
        """Send message and yield until ACK received."""
        
    def recv_and_ack(self) -> Generator[EventExpression, None, MessageEnvelope]:
        """Receive message and send ACK."""
        
    def mark_timing_barrier_satisfied(self) -> None:
        """Mark that NSM timing barrier has elapsed."""
```

### 7.3 `nsm_bounds.py` Decomposition (747 LOC → 2 modules)

#### 7.3.1 `nsm/bounds.py` (~200 LOC)

**Contents:**
```python
def binary_entropy(p: float) -> float:
    """Binary Shannon entropy H(p) = -p log₂(p) - (1-p) log₂(1-p)."""

def gamma_function(r: float, rate: float) -> float:
    """
    Strong-converse exponent Γ_r(R) for depolarizing channel.
    Ref: Lupo et al. (2023) Eq. (19).
    """

def collision_entropy_rate(r: float) -> float:
    """
    Collision entropy rate h_A = Γ[1 - log₂(1 + 3r²)].
    Dupuis-König bound.
    """

def virtual_erasure_rate(r: float) -> float:
    """
    Virtual erasure entropy rate h_B = 1 - r.
    Lupo bound.
    """

def max_bound_entropy_rate(r: float) -> float:
    """
    NSM Max Bound: max{h_A, h_B}.
    Selects optimal bound based on storage noise.
    Ref: Lupo et al. (2023) Eq. (36).
    """

def channel_capacity(r: float) -> float:
    """Classical capacity of depolarizing channel C_N = 1 - h((1+r)/2)."""

def extractable_key_length(
    n_sifted: int,
    r: float,
    leakage_bits: int,
    epsilon_sec: float,
) -> int:
    """
    Calculate secure key length using NSM bounds.
    
    ℓ = n × h_min(r) - leakage - log₂(1/ε_sec)
    """
```

#### 7.3.2 `nsm/feasibility.py` (~150 LOC)

**Contents:**
```python
class FeasibilityResult(Enum):
    FEASIBLE = "feasible"
    QBER_TOO_HIGH = "qber_too_high"
    INSUFFICIENT_ENTROPY = "insufficient_entropy"
    STORAGE_RATE_EXCEEDED = "storage_rate_exceeded"
    TIMING_INSUFFICIENT = "timing_insufficient"

@dataclass(frozen=True)
class FeasibilityInputs:
    """Inputs for pre-flight feasibility check."""
    expected_qber: float
    storage_noise_r: float
    storage_rate_nu: float
    epsilon_sec: float
    n_target_sifted_bits: int
    expected_leakage_bits: int
    batch_size: int = 0  # 0 = full-session mode

@dataclass(frozen=True)
class FeasibilityReport:
    """Detailed feasibility analysis output."""
    result: FeasibilityResult
    estimated_key_length: int
    min_entropy_rate: float
    channel_capacity: float
    message: str

class FeasibilityChecker:
    """Pre-flight security feasibility validation."""
    
    def check(self, inputs: FeasibilityInputs) -> FeasibilityReport:
        """
        Validate protocol can succeed with given parameters.
        
        Checks:
        1. QBER below hard limit (22%)
        2. C_N × ν < 1/2 (capacity constraint)
        3. Positive extractable entropy
        4. Sufficient bits for finite-size effects
        """
```

### 7.4 `ldpc_reconciliator.py` Decomposition (677 LOC → 4 modules)

#### 7.4.1 `reconciliation/syndrome.py` (~100 LOC)

**Contents:**
```python
def compute_syndrome(key: np.ndarray, parity_matrix: np.ndarray) -> np.ndarray:
    """Compute LDPC syndrome S = H × X mod 2."""

def select_code_rate(qber: float, available_rates: list[float]) -> float:
    """Select LDPC code rate based on estimated QBER."""

def shorten_key_to_frame(
    key: np.ndarray,
    frame_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shorten key to fit LDPC frame, return (shortened, remainder)."""
```

#### 7.4.2 `reconciliation/decoder.py` (~180 LOC)

**Contents:**
```python
def compute_llr(bit: int, qber: float) -> float:
    """Compute log-likelihood ratio for BP initialization."""

class BeliefPropagationDecoder:
    """Sum-product BP decoder for LDPC codes."""
    
    def __init__(
        self,
        parity_matrix: np.ndarray,
        max_iterations: int = 60,
        convergence_threshold: float = 1e-6,
    ): ...
    
    def decode(
        self,
        received: np.ndarray,
        syndrome: np.ndarray,
        channel_llr: np.ndarray,
    ) -> tuple[np.ndarray, bool, int]:
        """
        Decode using belief propagation.
        
        Returns
        -------
        corrected : np.ndarray
            Corrected codeword.
        converged : bool
            Whether decoder converged.
        iterations : int
            Number of iterations performed.
        """
```

#### 7.4.3 `reconciliation/verifier.py` (~100 LOC)

**Contents:**
```python
def compute_verification_hash(
    key: np.ndarray,
    hash_length_bits: int = 64,
) -> str:
    """Compute SHA256-based verification hash."""

def verify_reconciliation(
    alice_key: np.ndarray,
    bob_key: np.ndarray,
    hash_length_bits: int = 64,
) -> bool:
    """Verify keys match via hash comparison."""

@dataclass
class VerificationResult:
    """Hash verification outcome."""
    verified: bool
    alice_hash: str
    bob_hash: str
    hash_bits_leaked: int
```

#### 7.4.4 `reconciliation/leakage.py` (~120 LOC)

**Contents:**
```python
@dataclass
class BlockLeakageReport:
    """Leakage accounting for single reconciliation block."""
    block_index: int
    syndrome_bits: int
    hash_bits: int
    total_bits: int

class LeakageSafetyManager:
    """
    Wiretap cost tracking with safety cap enforcement.
    
    Tracks cumulative syndrome + hash leakage and enforces
    abort if safety cap exceeded (prevents "feigned failure" attack).
    """
    
    def __init__(
        self,
        max_leakage_bits: int,
        safety_margin: float = 0.1,
    ): ...
    
    def account_block(self, report: BlockLeakageReport) -> None:
        """Record leakage from reconciliation block."""
    
    @property
    def total_leakage_bits(self) -> int:
        """Current cumulative leakage."""
    
    @property
    def is_cap_exceeded(self) -> bool:
        """Check if safety cap has been exceeded."""
    
    @property
    def remaining_budget(self) -> int:
        """Remaining leakage budget."""
```

---

## 8. Expanded Elimination Plan

### 8.1 Interfaces Package Elimination

**Rationale:** The Strategy pattern adds value when:
1. Multiple implementations exist
2. Runtime switching is needed
3. Testing requires mocking

In E-HOK, none of these conditions hold meaningfully. Each "interface" has exactly one production implementation, and mocking can use `unittest.mock` directly.

#### Elimination Strategy

| Interface | Implementation | Action | New Location |
|-----------|---------------|--------|--------------|
| `ICommitmentScheme` | `SHA256Commitment` | Inline | `sifting/commitment.py` |
| `IReconciliator` | `LDPCReconciliator` | Inline | `reconciliation/` package |
| `IPrivacyAmplifier` | `ToeplitzAmplifier` | Inline | `amplification/toeplitz.py` |
| `ISamplingStrategy` | `RandomSamplingStrategy` | Delete | Inline into `sifting/sifter.py` |
| `INoiseEstimator` | `SimpleNoiseEstimator` | Delete | Unused, remove entirely |

**Migration Path:**
1. Move implementation to new location
2. Update all imports project-wide
3. Add `__all__` exports to new module
4. Delete interface file
5. Delete `interfaces/__init__.py`

### 8.2 Factories Module Elimination

**Current State:** `implementations/factories.py` (90 LOC)

```python
# Current factory pattern (to be eliminated)
def build_commitment_scheme(config) -> ICommitmentScheme:
    return SHA256Commitment()

def build_reconciliator(config, matrix=None) -> IReconciliator:
    return LDPCReconciliator(config, matrix)
```

**Replacement:** Direct construction in protocol orchestrator.

```python
# New direct construction in protocol/orchestrator.py
from ehok.sifting.commitment import SHA256Commitment
from ehok.reconciliation import LDPCReconciliator

class ProtocolOrchestrator:
    def __init__(self, config: ProtocolConfig):
        self.commitment = SHA256Commitment()
        self.reconciliator = LDPCReconciliator(
            frame_size=config.reconciliation.ldpc_frame_size,
            max_iterations=config.reconciliation.max_iterations,
        )
```

### 8.3 Configs Package Elimination

**Current Contents:**
```
configs/
├── __init__.py
├── protocol_config.py      # DEPRECATED - delete
├── network_baseline.yaml   # MOVE to qia-challenge-2025/config/
├── network_perfect.yaml    # MOVE to qia-challenge-2025/config/
├── ldpc_degree_distributions.yaml  # CONVERT to Python constants
├── ldpc_matrices/          # MOVE to qia-challenge-2025/assets/ldpc/
├── test_ldpc_matrices/     # MOVE to qia-challenge-2025/assets/ldpc_test/
└── generate_ldpc.py        # MOVE to scripts/
```

**Migration Commands:**
```bash
# Execute from qia-challenge-2025/
mkdir -p config assets/ldpc assets/ldpc_test scripts

# Move YAML configs
mv ehok/configs/network_*.yaml config/

# Move LDPC matrices
mv ehok/configs/ldpc_matrices/* assets/ldpc/
mv ehok/configs/test_ldpc_matrices/* assets/ldpc_test/

# Move generation script
mv ehok/configs/generate_ldpc.py scripts/

# Delete deprecated module
rm ehok/configs/protocol_config.py

# Convert degree distributions to Python
# (manual: create ldpc/distributions.py with YAML content as dict)

# Remove empty configs package
rm -rf ehok/configs/
```

### 8.4 Examples Package Elimination

**Rationale:** Examples are not part of the distributable package. They belong in project-level `scripts/`.

**Migration:**
```bash
mkdir -p scripts
mv ehok/examples/run_baseline.py scripts/
mv ehok/examples/debug_qber.py scripts/
rm ehok/examples/test_noise.py  # Misplaced test
rm -rf ehok/examples/
```

### 8.5 Dead Code Removal Manifest

| File | LOC | Status | Evidence | Action |
|------|-----|--------|----------|--------|
| `implementations/noise/simple_noise_estimator.py` | ~50 | Unused | No imports found | DELETE |
| `implementations/noise/__init__.py` | ~10 | Orphan | Only exports deleted class | DELETE |
| `analysis/metrics.py` | ~80 | Minimal | Only basic counters, unused | DELETE |
| `quantum/noise_adapter.py` | 709 | Partial | Some functions used, class missing | REFACTOR |
| `implementations/privacy_amplification/finite_key.py` | ~200 | Legacy | QKD bounds, not NSM | REVIEW → DELETE |

### 8.6 Duplicate Code Consolidation

#### Binary Entropy (3 implementations → 1)

**Current Locations:**
- `analysis/nsm_bounds.py:_binary_entropy()` (private)
- `implementations/reconciliation/ldpc_reconciliator.py:_binary_entropy()` (private)
- `implementations/privacy_amplification/finite_key.py:binary_entropy()` (public)

**Consolidation:**
```python
# utils/entropy.py
def binary_entropy(p: float) -> float:
    """
    Binary Shannon entropy H(p).
    
    H(p) = -p log₂(p) - (1-p) log₂(1-p)
    
    Parameters
    ----------
    p : float
        Probability in [0, 1].
    
    Returns
    -------
    float
        Entropy in bits.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
```

**Update all imports:**
```python
# In nsm/bounds.py, reconciliation/decoder.py, etc.
from ehok.utils.entropy import binary_entropy
```

#### Constants Consolidation

**Current Locations:**
- `core/constants.py` (scattered)
- `analysis/nsm_bounds.py` (duplicates)
- `protocols/alice.py` (hardcoded)

**Consolidation Strategy:**
- Security thresholds → `nsm/parameters.py`
- Protocol defaults → `protocol/config.py` (dataclass defaults)
- Delete `core/constants.py` entirely

---

## 9. Refactoring Execution Plan

This section provides a step-by-step migration sequence ensuring continuous test passage and minimal disruption.

### 9.1 Phase 0: Preparation (Non-Breaking)

**Duration:** 1 day  
**Objective:** Create new structure without modifying existing code.

| Step | Task | Command/Action |
|------|------|----------------|
| 0.1 | Create new package directories | `mkdir -p ehok/{types,nsm,sifting,reconciliation,amplification,ldpc,connection,protocol}` |
| 0.2 | Create `__init__.py` stubs | Touch empty `__init__.py` in each new package |
| 0.3 | Create external config structure | `mkdir -p config assets/{ldpc,ldpc_test} scripts` |
| 0.4 | Copy (don't move) YAML configs | `cp ehok/configs/*.yaml config/` |
| 0.5 | Run test suite | `pytest -v` — must pass |

### 9.2 Phase 1: Types Package Migration

**Duration:** 2 days  
**Objective:** Extract domain types from monolithic `data_structures.py`.

| Step | Task | Files Affected | Test Command |
|------|------|---------------|--------------|
| 1.1 | Create `types/exceptions.py` | New file | `pytest tests/unit/test_core/` |
| 1.2 | Create `types/keys.py` | New file | — |
| 1.3 | Create `types/records.py` | New file | — |
| 1.4 | Create `types/contracts.py` | New file | — |
| 1.5 | Update `types/__init__.py` | Re-export all types | — |
| 1.6 | Add deprecation to `core/data_structures.py` | Add re-exports + warning | `pytest` |
| 1.7 | Update imports across codebase | Global search/replace | `pytest` |
| 1.8 | Delete `core/data_structures.py` | Remove file | `pytest` |

**Migration Pattern for Step 1.6:**
```python
# core/data_structures.py (deprecated)
import warnings
warnings.warn(
    "ehok.core.data_structures is deprecated. "
    "Import from ehok.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
from ehok.types.keys import ObliviousKey, AliceObliviousKey, BobObliviousKey
from ehok.types.records import MeasurementRecord, LDPCBlockResult
from ehok.types.contracts import QuantumPhaseOutput, SiftedKeyData
from ehok.types.exceptions import AbortReason, ProtocolPhase, EHOKProtocolError

__all__ = [
    "ObliviousKey", "AliceObliviousKey", "BobObliviousKey",
    "MeasurementRecord", "LDPCBlockResult",
    "QuantumPhaseOutput", "SiftedKeyData",
    "AbortReason", "ProtocolPhase", "EHOKProtocolError",
]
```

### 9.3 Phase 2: NSM Package Migration

**Duration:** 2 days  
**Objective:** Consolidate NSM-related code into dedicated package.

| Step | Task | Source | Target |
|------|------|--------|--------|
| 2.1 | Create `nsm/bounds.py` | `analysis/nsm_bounds.py` (partial) | New file |
| 2.2 | Create `nsm/feasibility.py` | `core/feasibility.py` | New file |
| 2.3 | Create `nsm/timing.py` | `core/timing.py` | New file |
| 2.4 | Create `nsm/parameters.py` | `core/config.py` (partial) | New file |
| 2.5 | Update imports | Global | — |
| 2.6 | Add deprecation wrappers | Old locations | — |
| 2.7 | Delete old files | `analysis/nsm_bounds.py`, `core/timing.py`, etc. | — |

### 9.4 Phase 3: Domain Packages Migration

**Duration:** 3 days  
**Objective:** Create phase-aligned domain packages.

#### 9.4.1 Sifting Package
| Source | Target |
|--------|--------|
| `core/sifting.py` | `sifting/sifter.py` |
| `implementations/commitment/sha256_commitment.py` | `sifting/commitment.py` |
| `implementations/sampling/random_sampling.py` | Inline into `sifting/sifter.py` |
| `protocols/statistical_validation.py` (partial) | `sifting/validation.py`, `sifting/qber.py` |

#### 9.4.2 Reconciliation Package
| Source | Target |
|--------|--------|
| `implementations/reconciliation/ldpc_reconciliator.py` | Split into `syndrome.py`, `decoder.py`, `verifier.py` |
| `implementations/reconciliation/ldpc_belief_propagation.py` | `reconciliation/decoder.py` |
| `implementations/reconciliation/ldpc_matrix_manager.py` | `ldpc/manager.py` |
| `protocols/leakage_manager.py` | `reconciliation/leakage.py` |

#### 9.4.3 Amplification Package
| Source | Target |
|--------|--------|
| `implementations/privacy_amplification/toeplitz_amplifier.py` | `amplification/toeplitz.py` |
| `implementations/privacy_amplification/nsm_privacy_amplifier.py` | `amplification/key_length.py` |
| `core/oblivious_formatter.py` | `amplification/formatter.py` |

### 9.5 Phase 4: Connection Package Migration

**Duration:** 2 days  
**Objective:** Split `ordered_messaging.py` into focused modules.

| Step | Task | LOC |
|------|------|-----|
| 4.1 | Extract `MessageType`, `MessageEnvelope` → `connection/messaging.py` | ~150 |
| 4.2 | Extract `SocketState`, `OrderingStateMachine` → `connection/ordering.py` | ~150 |
| 4.3 | Extract `OrderedProtocolSocket` → `connection/socket.py` | ~150 |
| 4.4 | Delete `protocols/ordered_messaging.py` | -1021 |

### 9.6 Phase 5: Protocol Package Refactoring

**Duration:** 2 days  
**Objective:** Simplify orchestration layer.

| Step | Task |
|------|------|
| 5.1 | Create unified `protocol/config.py` with YAML loading |
| 5.2 | Create `protocol/orchestrator.py` from `protocols/base.py` |
| 5.3 | Refactor `protocol/alice.py` to use new imports |
| 5.4 | Refactor `protocol/bob.py` to use new imports |
| 5.5 | Delete old `protocols/` package |

### 9.7 Phase 6: Cleanup and Verification

**Duration:** 2 days  
**Objective:** Final cleanup and validation.

| Step | Task |
|------|------|
| 6.1 | Delete `interfaces/` package entirely |
| 6.2 | Delete `implementations/` package entirely |
| 6.3 | Delete `configs/` package entirely |
| 6.4 | Delete `examples/` package entirely |
| 6.5 | Delete `core/` package (replaced by `types/` + `nsm/`) |
| 6.6 | Delete `analysis/` package (merged into `nsm/`) |
| 6.7 | Run full test suite with coverage |
| 6.8 | Update documentation |
| 6.9 | Update `pyproject.toml` package discovery |

---

## 10. Implementation Roadmap

This is a consolidated timeline merging the detailed refactoring steps from Section 9 into weekly milestones.

### Week 1: Foundation
| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1-2 | Phase 0 (Preparation) | New directory structure, external config |
| 3-5 | Phase 1 (Types Package) | `types/` package complete |

### Week 2: NSM & Domain Setup
| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1-2 | Phase 2 (NSM Package) | `nsm/` package complete |
| 3-5 | Phase 3.1 (Sifting) | `sifting/` package complete |

### Week 3: Reconciliation & Amplification
| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1-3 | Phase 3.2 (Reconciliation) | `reconciliation/`, `ldpc/` packages |
| 4-5 | Phase 3.3 (Amplification) | `amplification/` package complete |

### Week 4: Infrastructure & Protocol
| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1-2 | Phase 4 (Connection) | `connection/` package complete |
| 3-5 | Phase 5 (Protocol) | `protocol/` package, YAML config |

### Week 5: Cleanup & Validation
| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1-3 | Phase 6 (Cleanup) | Delete obsolete packages |
| 4-5 | Validation | Full test suite, documentation |

---

## 11. Success Metrics

| Metric | Current | Target | Verification |
|--------|---------|--------|--------------|
| Max module size | 1076 LOC | 200 LOC | `wc -l` on all `.py` files |
| Package count | 9 | 10 | `ls ehok/` |
| Interface files | 5 | 0 | No `interfaces/` directory |
| Factory files | 1 | 0 | No `factories.py` |
| Deprecated modules | 1 | 0 | No deprecation warnings |
| Total non-test LOC | ~13,500 | ~5,000 | `find ehok -name "*.py" \| xargs wc -l` |
| External config | None | 1 YAML | `config/ehok_config.yaml` exists |
| Test coverage | TBD | ≥80% | `pytest --cov` |

---

## 12. Risk Mitigation

### 12.1 Test Coverage Strategy
- Run full test suite after each phase (gate criterion: all tests pass)
- Create phase boundary contract tests before migration
- Maintain backwards compatibility during transition via re-exports
- Integration tests verify cross-package contracts

### 12.2 Import Stability
- Use `__init__.py` re-exports for stable public API
- Provide deprecation warnings for old import paths (1 release cycle)
- Document migration path for external consumers
- Semantic versioning: major bump on API break

### 12.3 Rollback Strategy
- Feature branches for each phase (e.g., `refactor/phase-1-types`)
- Git tags before major changes (e.g., `pre-refactor-v1`)
- Keep old structure until new is proven (parallel existence)
- CI/CD gates on test passage

### 12.4 Critical Path Dependencies

```
Phase 0 (Prep) ─────┬──▶ Phase 1 (Types) ──▶ All other phases
                    │
                    └──▶ Phase 2 (NSM) ────▶ Phase 3+ (Domain)
```

**Blocking Dependencies:**
- `types/` must complete before any domain package migration
- `nsm/` must complete before `sifting/`, `amplification/`
- `connection/` must complete before `protocol/`

---

## Appendix A: Line Count Targets by Module

```
ehok/
├── __init__.py                      (~40 LOC)   # Public API exports
├── types/                           (~480 LOC total)
│   ├── __init__.py                  (~30 LOC)
│   ├── keys.py                      (~100 LOC)
│   ├── records.py                   (~100 LOC)
│   ├── contracts.py                 (~150 LOC)
│   └── exceptions.py                (~100 LOC)
├── nsm/                             (~600 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── bounds.py                    (~200 LOC)
│   ├── feasibility.py               (~150 LOC)
│   ├── timing.py                    (~150 LOC)
│   └── parameters.py                (~80 LOC)
├── quantum/                         (~450 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── epr.py                       (~150 LOC)
│   ├── basis.py                     (~80 LOC)
│   ├── measurement.py               (~80 LOC)
│   └── batching.py                  (~120 LOC)
├── sifting/                         (~560 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── commitment.py                (~120 LOC)
│   ├── sifter.py                    (~150 LOC)
│   ├── qber.py                      (~120 LOC)
│   └── validation.py                (~150 LOC)
├── reconciliation/                  (~520 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── syndrome.py                  (~100 LOC)
│   ├── decoder.py                   (~180 LOC)
│   ├── verifier.py                  (~100 LOC)
│   └── leakage.py                   (~120 LOC)
├── amplification/                   (~440 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── toeplitz.py                  (~150 LOC)
│   ├── key_length.py                (~120 LOC)
│   └── formatter.py                 (~150 LOC)
├── ldpc/                            (~450 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── generator.py                 (~200 LOC)
│   ├── manager.py                   (~150 LOC)
│   └── distributions.py             (~80 LOC)
├── connection/                      (~420 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── socket.py                    (~150 LOC)
│   ├── messaging.py                 (~150 LOC)
│   └── ordering.py                  (~100 LOC)
├── protocol/                        (~730 LOC total)
│   ├── __init__.py                  (~20 LOC)
│   ├── config.py                    (~150 LOC)
│   ├── orchestrator.py              (~200 LOC)
│   ├── alice.py                     (~180 LOC)
│   └── bob.py                       (~180 LOC)
└── utils/                           (~260 LOC total)
    ├── __init__.py                  (~10 LOC)
    ├── logging.py                   (~150 LOC)
    ├── entropy.py                   (~50 LOC)
    └── random.py                    (~50 LOC)

PACKAGE TOTALS:
├── types/           ~480 LOC
├── nsm/             ~600 LOC
├── quantum/         ~450 LOC
├── sifting/         ~560 LOC
├── reconciliation/  ~520 LOC
├── amplification/   ~440 LOC
├── ldpc/            ~450 LOC
├── connection/      ~420 LOC
├── protocol/        ~730 LOC
└── utils/           ~260 LOC
───────────────────────────
GRAND TOTAL:        ~4,950 LOC (vs current ~13,500 LOC)
                    63% reduction
```

---

## Appendix B: Duplicate Code to Consolidate

### B.1 Binary Entropy Function (3 → 1)
**Current locations:**
- `analysis/nsm_bounds.py:_binary_entropy()` (private)
- `implementations/reconciliation/ldpc_reconciliator.py:_binary_entropy()` (private)
- `implementations/privacy_amplification/finite_key.py:binary_entropy()` (public)

**Target:** Single implementation in `utils/entropy.py`

### B.2 QBER Threshold Constants (3 → 1)
**Current locations:**
- `core/constants.py:QBER_THRESHOLD = 0.11`
- `analysis/nsm_bounds.py:QBER_WARNING_THRESHOLD = 0.11`
- `analysis/nsm_bounds.py:QBER_HARD_LIMIT = 0.22`

**Target:** Single source in YAML config `security.qber_warning_threshold`, `security.qber_abort_threshold`

### B.3 Storage Noise Default (3 → 1)
**Current locations:**
- `protocols/alice.py:storage_noise_r = 0.75` (hardcoded)
- `protocols/bob.py:storage_noise_r = 0.75` (hardcoded)
- `configs/protocol_config.py:DEFAULT_STORAGE_NOISE_R = 0.75`

**Target:** Single source in YAML config `nsm.storage_noise_r`

### B.4 Timing Constants (2 → 1)
**Current locations:**
- `configs/protocol_config.py:DEFAULT_DELTA_T_NS = 1_000_000_000`
- `core/config.py:delta_t_ns: int = 1_000_000_000`

**Target:** Single source in YAML config `nsm.delta_t_ns`

---

## Appendix C: Critical Integration Points (from Remediation Analysis)

The remediation documentation identifies several critical gaps that this simplification must address:

### C.1 Physical Model Adapter Gap

**Current State:** `quantum/noise_adapter.py` provides `physical_to_simulator()` but lacks:
- `PhysicalModelAdapter` class
- `estimate_storage_noise_from_netsquid()` function
- `to_squidasm_link_config()` method

**Resolution in New Architecture:**
```
quantum/
├── adapter.py  # NEW: PhysicalModelAdapter class
└── ...
```

The adapter bridges NSM physical parameters (μ, η, e_det) to SquidASM `DepolariseLinkConfig`. This is a **critical remediation item** that must be implemented during Phase 3 (Quantum package refactoring).

### C.2 Timing Enforcement Integration

**Current State:** `TimingEnforcer` class exists in `core/timing.py` but is **not wired** into protocol flow.

**Resolution in New Architecture:**
```python
# protocol/orchestrator.py
class ProtocolOrchestrator:
    def __init__(self, config: ProtocolConfig):
        self._timing = TimingEnforcer(config.nsm.delta_t_ns)
    
    def _before_basis_reveal(self, sim_time_ns: int) -> None:
        """Enforce NSM timing barrier before basis reveal."""
        self._timing.mark_basis_reveal_attempt(sim_time_ns)
```

### C.3 Ordered Socket Integration

**Current State:** `OrderedProtocolSocket` exists but protocols use raw `csocket` access.

**Resolution in New Architecture:**
```python
# connection/socket.py
class OrderedProtocolSocket:
    """Wraps raw socket with ordering enforcement."""
    
    def send_with_ack(self, envelope: MessageEnvelope) -> Generator: ...
    def recv_and_ack(self) -> Generator[..., MessageEnvelope]: ...

# protocol/alice.py (refactored)
class AliceEHOK(Program):
    def _phase2_commitment(self):
        # OLD: self.context.csockets[self.PEER_NAME].recv()
        # NEW:
        envelope = yield from self._socket.recv_and_ack()
```

### C.4 Leakage Manager Wiring

**Current State:** `LeakageSafetyManager` is implemented but not called from reconciliation loop.

**Resolution in New Architecture:**
```python
# reconciliation/leakage.py
class LeakageSafetyManager:
    def account_block(self, report: BlockLeakageReport) -> None: ...
    
# protocol/alice.py (wiring)
def _phase4_reconciliation(self):
    for block in blocks:
        result = self._reconciliator.reconcile(block)
        self._leakage_manager.account_block(BlockLeakageReport(
            block_index=block.index,
            syndrome_bits=result.syndrome_bits,
            hash_bits=64,
        ))
        if self._leakage_manager.is_cap_exceeded:
            raise LeakageCapExceededError()
```

---

## Appendix D: Configuration Parameter Reference

Complete parameter reference for `config/ehok_config.yaml`:

### D.1 Physical Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `mu_pair_per_coherence` | float | 3.145e-5 | (0, ∞) | Erven 2014 Table I |
| `eta_total_transmittance` | float | 0.0150 | (0, 1] | Erven 2014 Table I |
| `e_det` | float | 0.0093 | [0, 0.5] | Erven 2014 Table I |
| `p_dark` | float | 1.50e-8 | [0, 1] | Erven 2014 Table I |

### D.2 NSM Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `storage_noise_r` | float | 0.75 | [0, 1] | König 2012 |
| `storage_rate_nu` | float | 0.002 | [0, 1] | Erven 2014 |
| `delta_t_ns` | int | 1e9 | > 0 | Erven 2014 |
| `memory_T1_ns` | float | null | > 0 | NetSquid |
| `memory_T2_ns` | float | null | > 0, ≤ T1 | NetSquid |

### D.3 Security Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `epsilon_sec` | float | 1e-9 | (0, 1) | Composable security |
| `epsilon_cor` | float | 1e-15 | (0, 1) | Hash collision |
| `qber_warning_threshold` | float | 0.11 | (0, 0.22) | Schaffner 2009 |
| `qber_abort_threshold` | float | 0.22 | (0.11, 0.5) | Lupo 2023 |
| `test_set_fraction` | float | 0.1 | (0, 1] | Protocol design |
| `min_test_set_size` | int | 100 | > 0 | Finite-size |

### D.4 Quantum Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `total_pairs` | int | 10000 | > 0 | Protocol design |
| `batch_size` | int | 5 | > 0 | SquidASM constraint |
| `max_qubits` | int | 5 | > 0 | Node constraint |

### D.5 Reconciliation Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `ldpc_frame_size` | int | 4096 | {128, 256, ..., 4096} | LDPC design |
| `max_iterations` | int | 60 | > 0 | BP convergence |
| `bp_threshold` | float | 1e-6 | > 0 | BP convergence |
| `leakage_safety_margin` | float | 0.1 | [0, 1) | Security margin |
| `max_leakage_fraction` | float | 0.5 | (0, 1) | Abort threshold |

### D.6 Amplification Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `use_nsm_bounds` | bool | true | - | NSM vs QKD |
| `min_output_bits` | int | 64 | > 0 | Minimum key |

### D.7 Connection Parameters
| Parameter | Type | Default | Range | Reference |
|-----------|------|---------|-------|-----------|
| `ack_timeout_ns` | int | 1e9 | > 0 | Protocol design |
| `max_retries` | int | 3 | ≥ 0 | Robustness |

---

## Appendix E: Public API Surface

The simplified package exposes a minimal public API:

```python
# ehok/__init__.py

# Configuration
from ehok.protocol.config import ProtocolConfig

# Protocol execution
from ehok.protocol.alice import AliceEHOK
from ehok.protocol.bob import BobEHOK
from ehok.protocol.orchestrator import run_protocol

# Type contracts
from ehok.types.contracts import (
    QuantumPhaseOutput,
    SiftedKeyData,
    ReconciledKeyData,
    ObliviousTransferOutput,
)
from ehok.types.keys import AliceObliviousKey, BobObliviousKey
from ehok.types.exceptions import (
    EHOKProtocolError,
    FeasibilityError,
    TimingViolationError,
    ReconciliationError,
)

# NSM security analysis
from ehok.nsm.bounds import max_bound_entropy_rate, extractable_key_length
from ehok.nsm.feasibility import FeasibilityChecker, FeasibilityResult

__all__ = [
    # Config
    "ProtocolConfig",
    # Protocol
    "AliceEHOK",
    "BobEHOK", 
    "run_protocol",
    # Types
    "QuantumPhaseOutput",
    "SiftedKeyData",
    "ReconciledKeyData",
    "ObliviousTransferOutput",
    "AliceObliviousKey",
    "BobObliviousKey",
    # Exceptions
    "EHOKProtocolError",
    "FeasibilityError",
    "TimingViolationError",
    "ReconciliationError",
    # NSM
    "max_bound_entropy_rate",
    "extractable_key_length",
    "FeasibilityChecker",
    "FeasibilityResult",
]
```

**Usage Example:**
```python
from pathlib import Path
from ehok import ProtocolConfig, run_protocol

# Load external configuration
config = ProtocolConfig.from_yaml(Path("config/ehok_config.yaml"))

# Run protocol (returns ObliviousTransferOutput)
result = run_protocol(config, network_config_path=config.paths.network_config)

if result.protocol_succeeded:
    print(f"Alice keys: S_0={result.alice_keys.s0}, S_1={result.alice_keys.s1}")
    print(f"Bob key: S_C={result.bob_key.s_c}, C={result.bob_key.choice_bit}")
else:
    print(f"Protocol aborted: {result.abort_reason}")
```

---

*End of Simplification Plan*

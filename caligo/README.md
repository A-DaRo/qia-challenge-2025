# Caligo: OT Protocol (NSM) — Technical Implementation

Caligo is a Python implementation of a 1-out-of-2 Oblivious Transfer (OT) protocol in the Noisy Storage Model (NSM), with native integration into SquidASM for end-to-end simulation.

This README is a developer-oriented reference: package boundaries, primary entry points, and the high-level pipeline.

## Architecture Overview

Caligo is organized around three stratified execution contexts:

- **Exploration Layer (Meta-Analysis)**: Orchestrates large-scale campaigns to detect security cliffs using Bayesian optimization and surrogate modeling.
- **Phase D (Domain Logic)**: Cryptographic and protocol primitives that are independent of SquidASM (unit-testable in isolation).
- **Phase E (Execution Layer)**: SquidASM `Program` implementations and simulation utilities that run the full protocol.

Separation of concerns:

- **Typed phase boundaries**: Phases exchange immutable contracts in `caligo.types`.
- **Simulation-independent core**: Logic is decoupled from the simulator where possible, enabling faster unit testing.
- **Explicit infrastructure layer**: Messaging and timing constraints are implemented in dedicated modules.

## Repository Layout

Primary package layout:

```
caligo/
  caligo/
    protocol/         # Phase E: SquidASM programs + runner
    simulation/       # Network builder, timing barrier, noise models
    connection/       # Ordered classical messaging
    quantum/          # Phase D: EPR + basis selection + measurements
    sifting/          # Phase D: commit/reveal, sifting, QBER estimation
    reconciliation/   # Phase D: LDPC-based reconciliation
    amplification/    # Phase D: entropy + Toeplitz hashing + OT formatting
    security/         # Bounds, feasibility checks, finite-key helpers
    exploration/      # Exploration Suite: LHS, Surrogate, Active Learning
    types/            # Phase contracts, keys, exceptions
    utils/            # Logging and bitarray/math utilities
    tests/            # Unit/integration tests

  main_explor.py      # ENTRY POINT: Cliff detection campaign orchestrator
  nsm_configs/        # Example YAML parameter sets (simulation/scenarios)
  explor_configs/     # Exploration harness configs
```

## Core Packages

| Package | Responsibility | Typical types/functions |
|---|---|---|
| `caligo.exploration` | Security cliff detection (LHS, Surrogate, Active Learning) | `Phase1Executor`, `Phase3Executor`, `BatchedEPRConfig` |
| `caligo.protocol` | Phase E orchestration (SquidASM programs and runner) | `AliceProgram`, `BobProgram`, `run_protocol()`, `ProtocolParameters` |
| `caligo.simulation` | Simulation wiring and NSM enforcement | `CaligoNetworkBuilder`, `TimingBarrier`, `perfect_network_config()` |
| `caligo.connection` | Reliable ordered classical messaging | `OrderedSocket` |
| `caligo.quantum` | Quantum measurement utilities | `EPRGenerator`, `BasisSelector`, `MeasurementExecutor` |
| `caligo.sifting` | Basis sifting and parameter estimation | `SHA256Commitment`, `Sifter`, `QBEREstimator` |
| `caligo.reconciliation` | Information reconciliation (LDPC) | (orchestrators/encoders/decoders under `caligo.reconciliation`) |
| `caligo.amplification` | Privacy amplification and OT output formatting | `ToeplitzHasher`, `SecureKeyLengthCalculator`, `OTOutputFormatter` |
| `caligo.security` | NSM bounds and feasibility checks | `FeasibilityChecker`, `compute_finite_key_length()` |
| `caligo.types` | Contracts and exceptions | `QuantumPhaseResult`, `SiftingPhaseResult`, `ObliviousTransferOutput` |
| `caligo.utils` | Logging + helpers | `get_logger()`, bitarray conversions/utilities |

## Primary Entry Points

### 1. Run the Security Cliff Exploration Campaign

The primary entry point for large-scale analysis is `main_explor.py`. This script orchestrates a 3-phase pipeline (LHS Warmup -> Surrogate Training -> Active Learning) to map the security boundary of the protocol.

```bash
# Basic usage
python main_explor.py

# Custom configuration with worker override
python main_explor.py --config explor_configs/qia_challenge_config.yaml --workers 32

# Resume/Skip specific phases
python main_explor.py --skip-phase1 --skip-phase2
```

### 2. Run the full SquidASM-backed protocol (Phase E)

For single-shot execution, use `caligo.protocol.run_protocol()`.

```python
from caligo.protocol import ProtocolParameters, run_protocol
from caligo.simulation import NSMParameters

params = ProtocolParameters(
    num_pairs=1_000,
    num_qubits=10,
    nsm_params=NSMParameters(
        delta_t_ns=1_000_000,
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        detection_eff_eta=0.85,
    ),
)

ot, raw = run_protocol(params, bob_choice_bit=0)
```

**Notes:**
- You can pass a pre-built SquidASM network via the `network_config` argument.
- Hardware compatibility is validated before execution (see `caligo.simulation.validate_network_config`).

### 3. Build a network configuration

For most experiments, use `caligo.simulation.CaligoNetworkBuilder` or helper constructors such as `perfect_network_config()`.

### 4. Preflight feasibility checks (Phase D)

Use `caligo.security.FeasibilityChecker` to evaluate whether a parameter set is expected to yield a positive key length under the implemented bounds.

## Pipeline Workflow (High Level)

1. **Quantum data collection** (`caligo.quantum`)
   - Generate/receive EPR pairs and measure in random bases.
   - Record outcomes; enforce the NSM wait time via `TimingBarrier`.

2. **Sifting and QBER estimation** (`caligo.sifting`)
   - Commit/reveal of basis choices.
   - Discard mismatched bases; estimate QBER from test bits.
   - Abort if QBER violates the configured security thresholds.

3. **Reconciliation** (`caligo.reconciliation`)
   - LDPC syndrome exchange and iterative decoding to correct Bob’s key.
   - Track leakage (syndrome + verification) for downstream key-length computation.

4. **Amplification and OT formatting** (`caligo.amplification`)
   - Compute extractable length from NSM bounds and observed leakage.
   - Apply Toeplitz hashing and format OT outputs.

The phase boundary contracts are defined in `caligo.types` (e.g., `QuantumPhaseResult`, `SiftingPhaseResult`, `ObliviousTransferOutput`).

## Configuration

- **Exploration**: Main exploration configs live in `explor_configs/` (e.g. `qia_challenge_config.yaml`).
- **Simulation**: Example NSM/channel parameter sets live in `nsm_configs/`.
- **System**: LDPC degree distributions and artifacts live under `caligo/configs/`.

## Dependencies

- Base dependencies are declared in `pyproject.toml`.
- Simulation dependencies are fundamental and require SquidASM/NetQASM/NetSquid. Please follow the [SquidASM Installation Guide](https://squidasm.readthedocs.io/en/latest/installation.html).
- Exploration dependencies require Torch/GPyTorch with 'cuda' enabled (GPU-accellerated surrogate modeling). Please install nvidia drivers and CUDA toolkit first.

```bash
pip install -e .
pip install -e ".[exploration]"
```

## Testing

From `qia-challenge-2025/caligo/`:

```bash
pytest -v
```

Markers are defined in `pyproject.toml` (e.g., `integration`, `e2e`, `slow`).

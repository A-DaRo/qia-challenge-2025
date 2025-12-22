# Caligo: OT Protocol (NSM) — Technical Implementation

Caligo is a Python implementation of a 1-out-of-2 Oblivious Transfer (OT) protocol in the Noisy Storage Model (NSM), with native integration into SquidASM for end-to-end simulation.

This README is a developer-oriented reference: package boundaries, primary entry points, and the high-level pipeline.

## Architecture Overview

Caligo is organized around two execution contexts:

- **Phase D (domain logic)**: cryptographic and protocol primitives that are independent of SquidASM (unit-testable in isolation).
- **Phase E (execution layer)**: SquidASM `Program` implementations and simulation utilities that run the full protocol.

Separation of concerns:

- **Typed phase boundaries**: phases exchange immutable contracts in `caligo.types`.
- **Simulation-independent core**: most logic is designed to run without a simulator.
- **Explicit infrastructure layer**: messaging and timing constraints are implemented in dedicated modules.

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
    types/            # Phase contracts, keys, exceptions
    utils/            # Logging and bitarray/math utilities
    tests/            # Unit/integration tests

  nsm_configs/        # Example YAML parameter sets (simulation/scenarios)
  explor_configs/     # Exploration harness configs
```

## Core Packages

| Package | Responsibility | Typical types/functions |
|---|---|---|
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

### Run the full SquidASM-backed protocol (Phase E)

The main entry point is `caligo.protocol.run_protocol()`.

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

Notes:

- You can pass a pre-built SquidASM network via the `network_config` argument.
- Hardware compatibility is validated before execution (see `caligo.simulation.validate_network_config`).

### Build a network configuration

For most experiments, use `caligo.simulation.CaligoNetworkBuilder` or helper constructors such as `perfect_network_config()`.

### Preflight feasibility checks (Phase D)

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

- Example NSM/channel parameter sets live in `nsm_configs/` (YAML).
- LDPC degree distributions and related artifacts live under `caligo/configs/`.
- Exploration/sweep configurations live in `explor_configs/`.

## Dependencies

- Base dependencies are declared in `pyproject.toml`.
- Simulation dependencies are fundamental (extra `simulation`) and require SquidASM/NetQASM/NetSquid. Please follow [SquidASM Installation Guide](https://squidasm.readthedocs.io/en/latest/installation.html) to install the required simulation dependencies.

```bash
pip install -e .
pip install -e ".[simulation]"
```

## Testing

From `qia-challenge-2025/caligo/`:

```bash
pytest -v
```

Markers are defined in `pyproject.toml` (e.g., `integration`, `e2e`, `slow`).

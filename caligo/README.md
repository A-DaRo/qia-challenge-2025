# Caligo

**Caligo** (Latin: "fog/mist" — evoking the obscured nature of oblivious transfer) is a ground-up implementation of the E-HOK (Efficient High-dimensional Oblivious Key) protocol for 1-out-of-2 Oblivious Transfer using the Noisy Storage Model (NSM).

## Overview

Caligo implements secure oblivious transfer with native SquidASM/NetSquid integration for quantum network simulation. The protocol allows two parties (Alice and Bob) to exchange cryptographic keys such that:

- **Alice** generates two keys (S₀, S₁) but doesn't know which one Bob receives
- **Bob** receives exactly one key (Sᴄ) based on his choice bit C, but cannot learn the other key

## Installation

```bash
# Basic installation
pip install -e .

# With simulation support (SquidASM/NetSquid)
pip install -e ".[simulation]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

```python
from caligo import (
    ObliviousKey,
    AliceObliviousKey,
    BobObliviousKey,
    binary_entropy,
    get_logger,
)

# Get a logger for your module
logger = get_logger("my_protocol")

# Calculate entropy for QBER estimation
qber = 0.05
h = binary_entropy(qber)
logger.info(f"Binary entropy at QBER={qber}: {h:.4f}")
```

## Package Structure

```
caligo/
├── types/              # Domain primitives and phase contracts
│   ├── keys.py         # ObliviousKey, AliceObliviousKey, BobObliviousKey
│   ├── measurements.py # MeasurementRecord, RoundResult, DetectionEvent
│   ├── phase_contracts.py  # Phase I→IV boundary dataclasses
│   └── exceptions.py   # Exception hierarchy and enums
├── utils/              # Cross-cutting utilities
│   ├── logging.py      # SquidASM-compatible logging
│   ├── math.py         # Binary entropy, finite-size penalties
│   └── bitarray_utils.py  # Bitarray manipulation helpers
└── tests/              # Test suite
```

## Protocol Phases

The E-HOK protocol consists of four phases:

1. **Phase I (Quantum)**: EPR pair generation and measurement
2. **Phase II (Sifting)**: Basis sifting and QBER estimation
3. **Phase III (Reconciliation)**: Information reconciliation via LDPC codes
4. **Phase IV (Amplification)**: Privacy amplification to extract secure keys

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `QBER_HARD_LIMIT` | 22% | Maximum QBER for security possibility |
| `QBER_CONSERVATIVE_LIMIT` | 11% | Recommended operational threshold |
| `DEFAULT_EPSILON_SEC` | 10⁻¹⁰ | Default security parameter |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=caligo

# Run specific test module
pytest tests/test_types/test_keys.py -v
```

## Parallel EPR Generation

Caligo includes an optional parallel EPR generation layer intended to
accelerate large Monte Carlo-style simulations of the quantum phase.

- Sequential/parallel selection is controlled via `ParallelEPRConfig(enabled=...)`.
- The implementation preserves i.i.d.-compatible statistics (ordering is not
  physically meaningful for Phase I outcomes).

**Programmatic usage**

```python
from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory
from caligo.quantum.parallel import ParallelEPRConfig

cfg = CaligoConfig(
    num_epr_pairs=10000,
    parallel_config=ParallelEPRConfig(enabled=True, num_workers=4, pairs_per_batch=2500),
    network_config={"noise": 0.05},
)

strategy = EPRGenerationFactory(cfg).create_strategy()
try:
    alice_out, alice_bases, bob_out, bob_bases = strategy.generate(cfg.num_epr_pairs)
finally:
    # Parallel strategies hold worker pools and should be shut down.
    if hasattr(strategy, "shutdown"):
        strategy.shutdown()
```

**YAML config example**

See [configs/parallel.yaml](configs/parallel.yaml).

## References

- Erven et al. (2014): "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model"
- Schaffner et al. (2009): "Robust Cryptography in the Noisy-Quantum-Storage Model"
- König et al. (2012): "Unconditional Security from Noisy Quantum Storage"
- Lupo et al. (2020): "Performance of Practical Quantum Oblivious Key Distribution"

## License

MIT License

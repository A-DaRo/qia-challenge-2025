"""
E-HOK Baseline Protocol Constants

This module defines all constants and configuration parameters for the E-HOK
baseline implementation.

Based on:
- e-hok-baseline.md specifications
- Literature: Lemus et al. (arXiv:1909.11701, arXiv:2501.03973)

Notes
-----
These constants represent conservative, well-tested values for initial
implementation. They may be tuned based on empirical results and specific
deployment requirements.
"""

# ============================================================================
# Protocol Parameters
# ============================================================================

QBER_THRESHOLD = 0.11
"""
Abort if QBER > 11% (standard QKD threshold).

Notes
-----
QBER (Quantum Bit Error Rate) above this threshold indicates either excessive
channel noise or potential eavesdropping. The 11% threshold is a widely-accepted
conservative bound for BB84-style protocols.
"""

TARGET_EPSILON_SEC = 1e-9
"""
Target security parameter (ε = 10^-9).

Notes
-----
The security parameter ε bounds the probability that the key is distinguishable
from a uniform random key or that an adversary can gain information. A value of
10^-9 provides high security suitable for most applications.
"""

TEST_SET_FRACTION = 0.1
"""
Use 10% of sifted bits for error estimation.

Notes
-----
A larger test set provides more accurate QBER estimation but reduces final key
length. 10% is a standard compromise in QKD implementations.
"""

# ============================================================================
# Quantum Generation
# ============================================================================

TOTAL_EPR_PAIRS = 10_000
"""
Target number of raw EPR pairs to generate.

Notes
-----
This value balances execution time with sufficient statistics. In production,
this would be scaled up significantly (millions of pairs).
"""

BATCH_SIZE = 5
"""
EPR pairs generated per batch, limited by quantum memory (5 qubits).

Notes
-----
SquidASM simulations have limited quantum memory. Batching allows streaming
generation of large numbers of EPR pairs without exhausting memory.
"""

# Basis encoding: 0 = Z-basis (computational), 1 = X-basis (Hadamard)
BASIS_Z = 0
"""Z-basis (computational basis: |0⟩, |1⟩)."""

BASIS_X = 1
"""X-basis (Hadamard basis: |+⟩ = (|0⟩ + |1⟩)/√2, |-⟩ = (|0⟩ - |1⟩)/√2)."""

# ============================================================================
# Network Configuration
# ============================================================================

LINK_FIDELITY_MIN = 0.95
"""
Minimum acceptable link fidelity.

Notes
-----
Link fidelity measures the quality of entanglement. Below 95%, QBER will likely
exceed abort threshold. This parameter is used for network validation.
"""

CLASSICAL_TIMEOUT_SEC = 30.0
"""
Timeout for classical message exchanges (seconds).

Notes
-----
Prevents indefinite blocking on network failures. Should be tuned based on
expected network latency and message sizes.
"""

# ============================================================================
# LDPC Parameters
# ============================================================================

LDPC_CODE_RATE = 0.5
"""
Target code rate (k/n) for LDPC codes.

Notes
-----
Code rate of 0.5 provides good error correction capability while maintaining
reasonable efficiency. Lower rates correct more errors but leak more information.
"""

LDPC_MAX_ITERATIONS = 500
"""
Maximum iterations for belief propagation decoder.

Notes
-----
BP decoders are iterative. 50 iterations is typically sufficient for convergence
at moderate error rates. Higher values increase latency.
"""

LDPC_BP_THRESHOLD = 1e-6
"""
Convergence threshold for belief propagation.

Notes
-----
Decoder stops when message updates change by less than this threshold or when
max iterations is reached. Smaller values increase accuracy but may require
more iterations.
"""

# ============================================================================
# Privacy Amplification
# ============================================================================

PA_SECURITY_MARGIN = 100
"""
Additional bits to compress for security buffer.

Notes
-----
This margin accounts for estimation errors and provides additional security
cushion. It's subtracted from the theoretical maximum key length calculated
from the leftover hash lemma.
"""

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"
"""
Default logging level.

Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

LOG_TO_FILE = True
"""
Whether to log to file in addition to console.

Notes
-----
File logging enables post-execution analysis and debugging. Logs are written
to a timestamped file in the configured log directory.
"""

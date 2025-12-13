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

from pathlib import Path
import math
from typing import Dict, Any

import yaml

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
Use 20% of sifted bits for error estimation.

Notes
-----
A larger test set provides more accurate QBER estimation but reduces final key
length. We use 20% (increased from standard 10%) to ensure reliable QBER
estimation even with noisy channels and smaller key sizes. Statistical variance
in QBER estimation decreases with sqrt(test_set_size), so larger test sets are
critical for protocol reliability.
"""

MIN_TEST_SET_SIZE = 100
"""
Minimum test set size regardless of fraction.

Notes
-----
With small key sizes, even 20% might give unreliable QBER estimates due to
statistical variance. We enforce a minimum of 100 bits to ensure statistical
significance: expected error count = 100 * 0.02 = 2 errors (for 2% QBER),
which has manageable variance.
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

LDPC_FRAME_SIZE = 4096
"""
Fixed LDPC frame size ``n`` in bits.

Notes
-----
Matrices are constructed for a fixed frame and adapt to channel conditions via
rate selection and shortening, following the LDPC specification.
"""

LDPC_TEST_FRAME_SIZES = (64, 128, 256)
"""Available test frame sizes for accelerated system testing.

Notes
-----
Test matrices are stored separately and loaded only when:
1. ProtocolConfig.testing_mode = True
2. ProtocolConfig.ldpc_test_frame_size is set

Production code ALWAYS uses LDPC_FRAME_SIZE = 4096.
"""

LDPC_TEST_MATRIX_SUBDIR = "test_ldpc_matrices"
"""Directory name for test-specific LDPC matrices."""

LDPC_CODE_RATES = (
	0.50,
	0.55,
	0.60,
	0.65,
	0.70,
	0.75,
	0.80,
	0.85,
	0.90,
)
"""
Discrete LDPC code rates supported by the matrix pool.

Notes
-----
These rates are generated offline using PEG with optimized degree distributions
per the baseline LDPC specification.
"""

LDPC_DEFAULT_RATE = 0.50
"""
Default LDPC code rate used when a specific rate is not yet selected.
"""

LDPC_CODE_RATE = LDPC_DEFAULT_RATE
"""
Backward-compatible alias for default LDPC code rate.
"""

LDPC_AVAILABLE_RATES = LDPC_CODE_RATES
"""
Alias for supported LDPC rates used in test specifications.
"""

LDPC_CRITICAL_EFFICIENCY = 1.22
"""
Critical efficiency parameter ``f_crit`` used in rate selection criterion.
"""

LDPC_F_CRIT = LDPC_CRITICAL_EFFICIENCY
"""
Alias for critical efficiency to match specification naming.
"""

LDPC_MAX_ITERATIONS = 60
"""
Maximum iterations for belief propagation decoder per block.
"""

LDPC_BP_THRESHOLD = 1e-6
"""
Convergence threshold for belief propagation.
"""

LDPC_HASH_BITS = 50
"""
Number of bits used for polynomial hash verification of each block.
"""

LDPC_QBER_WINDOW_SIZE = 256
"""
Window size (in blocks) for integrated QBER estimation.
"""

LDPC_MATRIX_FILE_PATTERN = "ldpc_{frame_size}_rate{rate:.2f}.npz"
"""
Filename pattern for stored LDPC parity-check matrices.
"""

PEG_MAX_TREE_DEPTH = 10
"""
Maximum BFS depth used during PEG construction to maximize local girth.
"""

PEG_DEFAULT_SEED = 42
"""
Default seed for deterministic PEG matrix generation.
"""

LDPC_DEGREE_DISTRIBUTIONS_PATH = (
	Path(__file__).resolve().parent / ".." / "configs" / "ldpc_degree_distributions.yaml"
).resolve()
"""
Absolute path to edge-perspective degree distributions for PEG matrix
construction. See ldpc_specification.md section 2.1.1 for the optimized
polynomials.
"""


def _load_ldpc_degree_distributions(path: Path) -> Dict[float, Dict[str, Any]]:
	"""
	Load and validate LDPC degree distributions from YAML.

	Parameters
	----------
	path : Path
		Filesystem path to ``ldpc_degree_distributions.yaml``.

	Returns
	-------
	Dict[float, Dict[str, Any]]
		Mapping of code rate to degree distribution dictionaries.

	Raises
	------
	FileNotFoundError
		If the YAML file is missing.
	ValueError
		If any distribution is malformed or probabilities do not sum to 1.
	"""
	if not path.exists():
		raise FileNotFoundError(f"LDPC degree distribution file not found: {path}")

	with path.open("r", encoding="utf-8") as handle:
		raw_data = yaml.safe_load(handle) or {}

	distributions: Dict[float, Dict[str, Any]] = {}
	for rate_key, payload in raw_data.items():
		try:
			rate = float(rate_key)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Invalid rate key in LDPC distribution: {rate_key}") from exc

		for node_type in ("lambda", "rho"):
			if node_type not in payload:
				raise ValueError(f"Missing '{node_type}' section for rate {rate}")
			section = payload[node_type]
			degrees = section.get("degrees")
			probabilities = section.get("probabilities")
			if not isinstance(degrees, list) or not isinstance(probabilities, list):
				raise ValueError(f"Degrees/probabilities must be lists for rate {rate}")
			if len(degrees) != len(probabilities):
				raise ValueError(f"Mismatched lengths in {node_type} distribution for rate {rate}")
			if any(d < 1 for d in degrees):
				raise ValueError(f"Degrees must be >=1 in {node_type} distribution for rate {rate}")
			sum_probs = float(sum(probabilities))
			if sum_probs <= 0.0:
				raise ValueError(f"{node_type} probabilities must sum to positive value for rate {rate}")
			# Normalize if the sum deviates from 1.0 (robustness to typos in external sources)
			if not math.isclose(sum_probs, 1.0, rel_tol=1e-6, abs_tol=1e-6):
				normalized_probs = [float(p) / sum_probs for p in probabilities]
				payload[node_type]["probabilities"] = normalized_probs

		distributions[rate] = payload

	return distributions


LDPC_DEGREE_DISTRIBUTIONS = _load_ldpc_degree_distributions(LDPC_DEGREE_DISTRIBUTIONS_PATH)
"""
Edge-perspective degree distributions keyed by code rate, loaded from
``ldpc_degree_distributions.yaml``.
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

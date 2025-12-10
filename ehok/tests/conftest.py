"""
Pytest configuration for E-HOK tests.

This module provides:
1. CLI flag `--run-long` to conditionally run tests marked as `long`.
2. Test LDPC matrix fixtures with frame_size=128 for accelerated testing.
3. Protocol configuration fixtures for test isolation.

Notes
-----
Test matrices are generated with frame_size=128 to reduce simulation time
while preserving LDPC decoder behavior. Production code uses frame_size=4096.
See docs/ldpc_matrix_tests.md for theoretical justification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import scipy.sparse as sp

from ehok.core import constants
from ehok.core.config import ProtocolConfig
from ehok.core.data_structures import LDPCMatrixPool
from ehok.implementations.reconciliation.ldpc_matrix_manager import LDPCMatrixManager
from ehok.implementations.reconciliation.ldpc_reconciliator import LDPCReconciliator
from ehok.implementations.reconciliation.peg_generator import (
    PEGMatrixGenerator,
    DegreeDistribution,
)
from ehok.utils.logging import get_logger


logger = get_logger("tests.conftest")


# =============================================================================
# Constants: Test Matrix Configuration
# =============================================================================

TEST_FRAME_SIZE = 128
"""
Frame size for test LDPC matrices.

This value provides:
- Sufficient graph structure for BP decoder convergence
- ~32× speedup over production frame_size=4096
- Valid degree distribution realization for all rates
"""

TEST_LDPC_RATES = constants.LDPC_CODE_RATES
"""All production rates are available in test matrices."""


# =============================================================================
# CLI Options
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options for test execution."""
    parser.addoption(
        "--force-regen-matrices",
        action="store_true",
        default=False,
        help="Force regeneration of test LDPC matrices even if they exist.",
    )


# =============================================================================
# Test Matrix Generation Helpers
# =============================================================================


def _get_test_degree_distributions(rate: float) -> tuple[DegreeDistribution, DegreeDistribution]:
    """
    Get degree distributions for test matrix generation.

    Parameters
    ----------
    rate : float
        Target code rate.

    Returns
    -------
    tuple of DegreeDistribution
        (lambda_dist, rho_dist) for variable and check nodes.

    Notes
    -----
    Uses simplified distributions for small frame sizes. For n=128, the full
    Richardson optimized distributions may not be realizable (high-degree nodes
    cannot be satisfied). We use reduced-complexity distributions that still
    achieve good decoding thresholds.
    """
    # Simplified distributions suitable for small frame sizes
    # Based on Martinez-Mateo et al., "Blind Reconciliation" (2013)
    if rate <= 0.55:
        lambda_dist = DegreeDistribution(
            degrees=[2, 3, 6], probabilities=[0.4, 0.35, 0.25]
        )
        rho_dist = DegreeDistribution(degrees=[6, 7], probabilities=[0.5, 0.5])
    elif rate <= 0.70:
        lambda_dist = DegreeDistribution(
            degrees=[2, 3, 4], probabilities=[0.35, 0.4, 0.25]
        )
        rho_dist = DegreeDistribution(degrees=[5, 6], probabilities=[0.6, 0.4])
    else:  # High rates (0.75 - 0.90)
        lambda_dist = DegreeDistribution(
            degrees=[2, 3], probabilities=[0.5, 0.5]
        )
        rho_dist = DegreeDistribution(degrees=[3, 4], probabilities=[0.5, 0.5])

    return lambda_dist, rho_dist


def _generate_test_matrix(frame_size: int, rate: float, seed: int = 42) -> sp.csr_matrix:
    """
    Generate a single test LDPC matrix using PEG.

    Parameters
    ----------
    frame_size : int
        Number of columns (codeword length).
    rate : float
        Code rate R = k/n.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    sp.csr_matrix
        Parity-check matrix H with shape (m, n) where m = floor(n * (1 - R)).
    """
    lambda_dist, rho_dist = _get_test_degree_distributions(rate)

    generator = PEGMatrixGenerator(
        n=frame_size,
        rate=rate,
        lambda_dist=lambda_dist,
        rho_dist=rho_dist,
        max_tree_depth=constants.PEG_MAX_TREE_DEPTH,
        seed=seed,
    )

    return generator.generate()


# =============================================================================
# Fixtures: Test LDPC Matrices
# =============================================================================


@pytest.fixture(scope="session")
def test_ldpc_matrix_pool(request: pytest.FixtureRequest) -> LDPCMatrixPool:
    """
    Generate and cache test LDPC matrices for the entire test session.

    This fixture manages persistent test matrices in ehok/configs/test_ldpc_matrices.
    It checks for existing matrices and generates them only if missing or if
    --force-regen-matrices is specified.

    Returns
    -------
    LDPCMatrixPool
        Pool containing test matrices for all rates.
    """
    # Resolve persistent directory: ehok/configs/test_ldpc_matrices
    # We assume conftest.py is in ehok/tests/
    base_dir = Path(__file__).parent.parent / "configs" / constants.LDPC_TEST_MATRIX_SUBDIR
    base_dir.mkdir(parents=True, exist_ok=True)

    force_regen = request.config.getoption("--force-regen-matrices")
    matrices: dict[float, sp.spmatrix] = {}
    
    # Check if we need to regenerate
    # We need all rates to be present
    missing_rates = []
    for rate in TEST_LDPC_RATES:
        filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
            frame_size=TEST_FRAME_SIZE, rate=rate
        )
        if not (base_dir / filename).exists():
            missing_rates.append(rate)

    should_generate = force_regen or len(missing_rates) > 0

    if should_generate:
        logger.info(
            "Generating test LDPC matrices (frame_size=%d) in %s...", 
            TEST_FRAME_SIZE, base_dir
        )
        if force_regen:
            logger.info("Forced regeneration requested.")
        elif missing_rates:
            logger.info("Missing rates: %s", missing_rates)

        for rate in TEST_LDPC_RATES:
            # Only generate if forced or missing (optimization: could skip existing if not forced)
            # But for consistency, if we regenerate, maybe we should regenerate all?
            # Let's regenerate only missing unless forced.
            filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
                frame_size=TEST_FRAME_SIZE, rate=rate
            )
            file_path = base_dir / filename
            
            if force_regen or not file_path.exists():
                logger.debug("Generating matrix for rate %.2f...", rate)
                matrix = _generate_test_matrix(TEST_FRAME_SIZE, rate)
                sp.save_npz(file_path, matrix)
            
    # Load all matrices
    logger.info("Loading test LDPC matrices from %s...", base_dir)
    for rate in TEST_LDPC_RATES:
        filename = constants.LDPC_MATRIX_FILE_PATTERN.format(
            frame_size=TEST_FRAME_SIZE, rate=rate
        )
        try:
            matrix = sp.load_npz(base_dir / filename)
            matrices[rate] = matrix.astype(np.uint8)
        except Exception as e:
            logger.error("Failed to load matrix for rate %.2f: %s", rate, e)
            raise

    # Compute checksum for synchronization
    checksum = LDPCMatrixManager._compute_checksum(matrices)

    pool = LDPCMatrixPool(
        frame_size=TEST_FRAME_SIZE,
        matrices=matrices,
        rates=np.array(sorted(TEST_LDPC_RATES)),
        checksum=checksum,
    )

    logger.info(
        "Test matrix pool ready: %d rates, checksum=%s",
        len(matrices), checksum[:16] + "..."
    )

    return pool


@pytest.fixture(scope="session")
def test_ldpc_matrix_dir(
    test_ldpc_matrix_pool: LDPCMatrixPool,
) -> Path:
    """
    Return path to directory containing test LDPC matrices.

    Returns
    -------
    Path
        Directory containing .npz matrix files.
    """
    return Path(__file__).parent.parent / "configs" / constants.LDPC_TEST_MATRIX_SUBDIR


@pytest.fixture(autouse=True)
def _require_ldpc_matrix(request: pytest.FixtureRequest) -> None:
    """
    Automatically ensure LDPC matrices are available for marked tests.
    
    If a test is marked with @pytest.mark.require_ldpc_matrix, this fixture
    will trigger the test_ldpc_matrix_pool fixture to ensure matrices exist.
    """
    if request.node.get_closest_marker("require_ldpc_matrix"):
        request.getfixturevalue("test_ldpc_matrix_pool")


@pytest.fixture
def test_matrix_manager(test_ldpc_matrix_pool: LDPCMatrixPool) -> LDPCMatrixManager:
    """
    Create an LDPCMatrixManager with test matrices.

    Returns
    -------
    LDPCMatrixManager
        Manager wrapping the test matrix pool.
    """
    return LDPCMatrixManager(matrix_pool=test_ldpc_matrix_pool)


@pytest.fixture
def test_reconciliator(test_matrix_manager: LDPCMatrixManager) -> LDPCReconciliator:
    """
    Create an LDPCReconciliator configured for testing.

    Returns
    -------
    LDPCReconciliator
        Reconciliator with test matrices and default parameters.
    """
    return LDPCReconciliator(
        matrix_manager=test_matrix_manager,
        initial_qber_est=0.05,
    )


# =============================================================================
# Fixtures: Protocol Configuration
# =============================================================================


@pytest.fixture
def baseline_config() -> ProtocolConfig:
    """
    Baseline protocol configuration for tests.

    Returns
    -------
    ProtocolConfig
        Default baseline configuration.
    """
    return ProtocolConfig.baseline()


@pytest.fixture
def fast_test_config() -> ProtocolConfig:
    """
    Fast test configuration with reduced parameters.

    This configuration uses minimal EPR pairs and relaxed thresholds
    for quick functional tests.

    Returns
    -------
    ProtocolConfig
        Configuration optimized for fast test execution.
    """
    config = ProtocolConfig.baseline()
    config.quantum.total_pairs = 500
    config.quantum.batch_size = 10
    return config


# =============================================================================
# Fixtures: Random Data Generators
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """
    Seeded random number generator for reproducible tests.

    Returns
    -------
    np.random.Generator
        NumPy random generator with fixed seed.
    """
    return np.random.default_rng(seed=12345)


@pytest.fixture
def sample_sifted_key(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a sample sifted key for testing.

    Returns
    -------
    np.ndarray
        Binary array of length TEST_FRAME_SIZE.
    """
    return rng.integers(0, 2, size=TEST_FRAME_SIZE, dtype=np.uint8)


@pytest.fixture
def sample_sifted_key_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a pair of sifted keys with controlled QBER.

    Returns
    -------
    tuple of (alice_key, bob_key, actual_qber)
        Alice's key, Bob's key (with errors), and the actual QBER.
    """
    target_qber = 0.05
    alice_key = rng.integers(0, 2, size=TEST_FRAME_SIZE, dtype=np.uint8)

    # Introduce errors
    error_mask = rng.random(size=TEST_FRAME_SIZE) < target_qber
    bob_key = alice_key.copy()
    bob_key[error_mask] = 1 - bob_key[error_mask]

    actual_qber = np.mean(alice_key != bob_key)
    return alice_key, bob_key, float(actual_qber)

"""
Pytest fixtures for exploration module tests.

Provides common fixtures for testing the exploration suite including
sample configurations, protocol results, and temporary directory management.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from caligo.exploration.types import (
    ExplorationSample,
    ProtocolResult,
    ProtocolOutcome,
    ReconciliationStrategy,
    Phase1State,
    Phase2State,
    Phase3State,
    ExplorationConfig,
)
from caligo.exploration.sampler import ParameterBounds


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="exploration_test_") as td:
        yield Path(td)


@pytest.fixture
def exploration_output_dir(temp_dir: Path) -> Path:
    """Create an output directory structure for exploration tests."""
    output_dir = temp_dir / "exploration_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_parameters() -> Dict[str, Any]:
    """
    Standard parameter configuration for testing.

    Returns a dict matching ExplorationSample fields exactly as defined
    in types.py.
    """
    return {
        "storage_noise_r": 0.1,
        "storage_rate_nu": 0.5,
        "wait_time_ns": 1e6,
        "channel_fidelity": 0.95,
        "detection_efficiency": 0.8,
        "detector_error": 0.01,
        "dark_count_prob": 1e-5,
        "num_pairs": 100000,
        "strategy": ReconciliationStrategy.BASELINE,
    }


@pytest.fixture
def sample_exploration_sample(sample_parameters: Dict[str, Any]) -> ExplorationSample:
    """Create a valid ExplorationSample for testing."""
    return ExplorationSample(**sample_parameters)


@pytest.fixture
def sample_protocol_result(
    sample_exploration_sample: ExplorationSample,
) -> ProtocolResult:
    """Create a valid ProtocolResult for testing."""
    return ProtocolResult(
        sample=sample_exploration_sample,
        outcome=ProtocolOutcome.SUCCESS,
        net_efficiency=0.85,
        raw_key_length=50000,
        final_key_length=42500,
        qber_measured=0.03,
        reconciliation_efficiency=0.95,
        leakage_bits=10000,
        execution_time_seconds=2.5,
        error_message=None,
        metadata={"ldpc_iterations": 15},
    )


@pytest.fixture
def sample_results_batch() -> List[ProtocolResult]:
    """Create a batch of protocol results with varying outcomes."""
    results = []
    rng = np.random.default_rng(42)

    for i in range(20):
        # Vary the parameters within valid bounds
        fidelity = 0.55 + rng.uniform(0, 0.40)  # Must be > 0.5
        storage_r = rng.uniform(0, 1)
        storage_nu = rng.uniform(0.001, 1.0)
        wait_ns = 10 ** rng.uniform(5, 9)
        detect_eff = rng.uniform(0.01, 1.0)
        det_error = rng.uniform(0, 0.1)
        dark_prob = 10 ** rng.uniform(-8, -3)
        n_pairs = int(10 ** rng.uniform(4, 6))

        sample = ExplorationSample(
            storage_noise_r=storage_r,
            storage_rate_nu=storage_nu,
            wait_time_ns=wait_ns,
            channel_fidelity=fidelity,
            detection_efficiency=detect_eff,
            detector_error=det_error,
            dark_count_prob=dark_prob,
            num_pairs=n_pairs,
            strategy=ReconciliationStrategy.BASELINE if i % 2 == 0 else ReconciliationStrategy.BLIND,
        )

        # Simulate outcome based on fidelity
        if fidelity > 0.85:
            outcome = ProtocolOutcome.SUCCESS
            efficiency = 0.7 + rng.uniform(0, 0.25)
            qber = 0.01 + rng.uniform(0, 0.03)
        elif fidelity > 0.70:
            outcome = ProtocolOutcome.SUCCESS
            efficiency = 0.4 + rng.uniform(0, 0.3)
            qber = 0.05 + rng.uniform(0, 0.05)
        else:
            outcome = rng.choice(
                [ProtocolOutcome.SUCCESS, ProtocolOutcome.FAILURE_QBER],
            )
            efficiency = 0.1 + rng.uniform(0, 0.3) if outcome == ProtocolOutcome.SUCCESS else 0.0
            qber = 0.10 + rng.uniform(0, 0.12)

        raw_key = int(n_pairs * 0.5)
        final_key = int(raw_key * efficiency)
        leakage = int(raw_key * (1 - efficiency) * 0.3)

        results.append(
            ProtocolResult(
                sample=sample,
                outcome=outcome,
                net_efficiency=efficiency,
                raw_key_length=raw_key,
                final_key_length=final_key,
                qber_measured=qber,
                reconciliation_efficiency=0.95 if outcome == ProtocolOutcome.SUCCESS else 0.0,
                leakage_bits=leakage,
                execution_time_seconds=1.0 + rng.uniform(0, 2),
                error_message=None if outcome == ProtocolOutcome.SUCCESS else "Simulated failure",
                metadata={"batch_idx": i},
            )
        )

    return results


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def exploration_config() -> ExplorationConfig:
    """Standard exploration configuration for testing."""
    return ExplorationConfig()


@pytest.fixture
def parameter_bounds() -> ParameterBounds:
    """Default parameter bounds for sampling."""
    return ParameterBounds()


# =============================================================================
# Training Data Fixtures
# =============================================================================


@pytest.fixture
def training_data_small() -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Small training dataset for GP testing.

    Returns
    -------
    Tuple[NDArray, NDArray]
        (X, y) where X is (30, 9) features in transformed space and y is (30,) targets.
    """
    rng = np.random.default_rng(42)
    n_samples = 30
    n_features = 9

    # Generate features in transformed space (matching to_array output)
    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.uniform(0.0, 1.0, n_samples)  # storage_noise_r [0, 1]
    X[:, 1] = rng.uniform(np.log10(0.001), 0.0, n_samples)  # storage_rate_nu (log10)
    X[:, 2] = rng.uniform(5.0, 9.0, n_samples)  # wait_time_ns (log10)
    X[:, 3] = rng.uniform(0.501, 1.0, n_samples)  # channel_fidelity
    X[:, 4] = rng.uniform(-3.0, 0.0, n_samples)  # detection_efficiency (log10)
    X[:, 5] = rng.uniform(0.0, 0.1, n_samples)  # detector_error
    X[:, 6] = rng.uniform(-8.0, -3.0, n_samples)  # dark_count_prob (log10)
    X[:, 7] = rng.uniform(4.0, 6.0, n_samples)  # num_pairs (log10)
    X[:, 8] = rng.choice([0.0, 1.0], n_samples)  # strategy (binary)

    # Generate targets as a function of features (efficiency-like)
    # Simulates efficiency dropping as fidelity decreases
    y = 0.9 * X[:, 3] - 0.3 * X[:, 0] + 0.1 * rng.standard_normal(n_samples)
    y = np.clip(y, 0, 1)

    return X, y


@pytest.fixture
def training_data_twin() -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Training data for twin GP testing.

    Returns baseline and blind strategy datasets.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        (X_baseline, y_baseline, X_blind, y_blind) training data for the twin GP.
    """
    rng = np.random.default_rng(42)
    n_samples = 30
    n_features = 9

    # Generate features in transformed space
    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.uniform(0.0, 1.0, n_samples)
    X[:, 1] = rng.uniform(np.log10(0.001), 0.0, n_samples)
    X[:, 2] = rng.uniform(5.0, 9.0, n_samples)
    X[:, 3] = rng.uniform(0.501, 1.0, n_samples)  # fidelity
    X[:, 4] = rng.uniform(-3.0, 0.0, n_samples)
    X[:, 5] = rng.uniform(0.0, 0.1, n_samples)
    X[:, 6] = rng.uniform(-8.0, -3.0, n_samples)
    X[:, 7] = rng.uniform(4.0, 6.0, n_samples)
    X[:, 8] = 0.0  # All baseline for this fixture

    # Baseline: efficiency mainly depends on fidelity
    y_baseline = 0.85 * X[:, 3] - 0.2 * X[:, 0] + 0.05 * rng.standard_normal(n_samples)
    y_baseline = np.clip(y_baseline, 0, 1)

    # Blind: slightly different response, diverges at low fidelity
    y_blind = 0.80 * X[:, 3] - 0.3 * X[:, 0] + 0.05 * rng.standard_normal(n_samples)
    # Add divergence at low fidelity
    low_fidelity_mask = X[:, 3] < 0.7
    y_blind[low_fidelity_mask] *= 0.5  # Blind performs worse at low fidelity
    y_blind = np.clip(y_blind, 0, 1)

    # Return X for both strategies (same feature space)
    return X, y_baseline, X.copy(), y_blind


# =============================================================================
# Phase State Fixtures
# =============================================================================


@pytest.fixture
def phase1_state_partial(exploration_output_dir: Path) -> Phase1State:
    """Phase 1 state with partial completion."""
    rng = np.random.default_rng(42)
    return Phase1State(
        target_feasible_samples=50,
        feasible_samples_collected=25,
        total_samples_processed=30,
        current_batch_start=20,
        rng_state={"state": rng.bit_generator.state},
        current_phase="LHS",
        hdf5_path=exploration_output_dir / "exploration_data.h5",
    )


@pytest.fixture
def phase2_state_initial(exploration_output_dir: Path) -> Phase2State:
    """Phase 2 state before training."""
    return Phase2State(
        training_samples_used=0,
        last_training_mse=float("inf"),
        model_version=0,
        divergence_detected=False,
        current_phase="SURROGATE",
    )


@pytest.fixture
def phase2_state_trained(exploration_output_dir: Path) -> Phase2State:
    """Phase 2 state after training."""
    return Phase2State(
        training_samples_used=1000,
        last_training_mse=0.05,
        model_version=1,
        divergence_detected=False,
        current_phase="SURROGATE",
    )


@pytest.fixture
def phase3_state_partial(exploration_output_dir: Path) -> Phase3State:
    """Phase 3 state with partial completion."""
    return Phase3State(
        iteration=10,
        total_active_samples=160,
        best_cliff_point=None,
        best_cliff_efficiency=0.15,
        acquisition_history=[0.1, 0.12, 0.13, 0.15],
        current_phase="ACTIVE",
    )


# =============================================================================
# Mock EPR Data Fixtures
# =============================================================================


@pytest.fixture
def mock_epr_pairs() -> Tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.float64]]:
    """
    Provide EPR measurement data for tests.

    Strategy:
    - For the first `MAX_REAL_EPR` calls, attempt to generate real EPR data via
      the Caligo quantum pipeline (parallel EPR generation) for higher fidelity
      tests (fast, limited-size generation).
    - If generation fails or limit is exceeded, fall back to a lightweight
      synthetic generator (previous behaviour).

    This makes tests more truthful while avoiding slowdowns for full test
    suites. Use environment variable `TEST_REAL_EPR` to control the limit.
    """
    # Local cache / counters so we only do a few real generations in CI
    import os
    MAX_REAL_EPR = int(os.environ.get("TEST_REAL_EPR", "5"))

    if not hasattr(mock_epr_pairs, "_real_cache"):
        mock_epr_pairs._real_cache = []  # type: ignore[attr-defined]
        mock_epr_pairs._calls = 0  # type: ignore[attr-defined]

    mock_epr_pairs._calls += 1  # type: ignore[attr-defined]

    # Try real generation while under the limit
    if mock_epr_pairs._calls <= MAX_REAL_EPR:
        try:
            from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
            from caligo.quantum.parallel import ParallelEPRConfig

            num_pairs = 1000  # keep small to be fast in tests
            parallel_cfg = ParallelEPRConfig(enabled=True, num_workers=min(4, (os.cpu_count() or 1)), prefetch_batches=1)
            cfg = CaligoConfig(num_epr_pairs=num_pairs, parallel_config=parallel_cfg, network_config={"noise": 0.01})

            factory = EPRGenerationFactory(cfg)
            strategy = factory.create_strategy()
            try:
                alice_out, alice_bases, bob_out, bob_bases = strategy.generate(num_pairs)
            finally:
                # Ensure strategy clean-up if it exposes a shutdown method
                if hasattr(strategy, "shutdown"):
                    try:
                        strategy.shutdown()
                    except Exception:
                        pass

            # Convert boolean/int arrays to expected int8 arrays
            alice_bits = (np.asarray(alice_out, dtype=np.int8) & 1).astype(np.int8)
            bob_bits = (np.asarray(bob_out, dtype=np.int8) & 1).astype(np.int8)
            # Approximate per-pair fidelity: start from 0.95 baseline and add small noise
            rng = np.random.default_rng(42 + mock_epr_pairs._calls)  # deterministic variation
            fidelities = 0.95 + 0.05 * rng.random(num_pairs)

            # Cache and return
            mock_epr_pairs._real_cache.append((alice_bits, bob_bits, fidelities))  # type: ignore[attr-defined]
            return alice_bits, bob_bits, fidelities

        except Exception as exc:  # pragma: no cover - robust fallback
            # If anything goes wrong, fall back to synthetic generator below
            # but log the failure for debugging.
            try:
                from caligo.utils.logging import get_logger
                get_logger(__name__).warning("Real EPR generation failed in fixture: %s", exc)
            except Exception:
                pass

    # Fallback synthetic generator (previous behaviour)
    rng = np.random.default_rng(42 + mock_epr_pairs._calls)
    n_pairs = 1000

    # Generate correlated bits with some noise
    alice_bits = rng.integers(0, 2, size=n_pairs, dtype=np.int8)

    # Bob's bits are correlated but with ~2% error
    error_mask = rng.random(n_pairs) < 0.02
    bob_bits = alice_bits.copy()
    bob_bits[error_mask] = 1 - bob_bits[error_mask]

    # Fidelity per pair (consistent synthetic values)
    fidelities = 0.95 + 0.05 * rng.random(n_pairs)

    return alice_bits, bob_bits, fidelities


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def rng_state_dict() -> Dict[str, Any]:
    """Capture current RNG state as a serializable dict."""
    rng = np.random.default_rng(12345)
    _ = rng.random()  # Advance state
    state = rng.bit_generator.state
    # Convert to JSON-serializable format
    return {
        "bit_generator": state["bit_generator"],
        "state": {
            "state": int(state["state"]["state"]),
            "inc": int(state["state"]["inc"]),
        },
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }

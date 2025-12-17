"""
Pytest fixtures and test infrastructure for Caligo.

This module provides shared fixtures for testing all Caligo modules,
including sample data generators for phase contracts and security parameters.
"""

from __future__ import annotations

import pytest
import numpy as np
from bitarray import bitarray

from caligo.types.keys import (
    ObliviousKey,
    AliceObliviousKey,
    BobObliviousKey,
)
from caligo.types.measurements import (
    MeasurementRecord,
    RoundResult,
    DetectionEvent,
)
from caligo.types.phase_contracts import (
    QuantumPhaseResult,
    SiftingPhaseResult,
    ReconciliationPhaseResult,
    AmplificationPhaseResult,
    ObliviousTransferOutput,
)


# =============================================================================
# Security Parameter Fixtures
# =============================================================================


@pytest.fixture
def security_params() -> dict:
    """Standard security parameters for testing."""
    return {
        "epsilon_sec": 1e-10,
        "qber_hard_limit": 0.22,
        "qber_conservative": 0.11,
        "storage_noise_r": 0.75,
    }


@pytest.fixture
def epsilon_sec() -> float:
    """Default security parameter ε_sec."""
    return 1e-10


# =============================================================================
# Key Fixtures
# =============================================================================


@pytest.fixture
def sample_bitarray_8() -> bitarray:
    """8-bit sample bitarray."""
    return bitarray("10101010")


@pytest.fixture
def sample_bitarray_64() -> bitarray:
    """64-bit sample bitarray."""
    return bitarray("1010101001010101101010100101010110101010010101011010101001010101")


@pytest.fixture
def sample_oblivious_key(sample_bitarray_8: bitarray) -> ObliviousKey:
    """Sample ObliviousKey for testing."""
    return ObliviousKey(
        bits=sample_bitarray_8,
        length=8,
        security_param=1e-10,
        creation_time=1000.0,
    )


@pytest.fixture
def sample_alice_key() -> AliceObliviousKey:
    """Sample AliceObliviousKey for testing."""
    s0 = bitarray("10101010")
    s1 = bitarray("01010101")
    return AliceObliviousKey(
        s0=s0,
        s1=s1,
        key_length=8,
        security_parameter=1e-10,
        entropy_consumed=4.0,
    )


@pytest.fixture
def sample_bob_key() -> BobObliviousKey:
    """Sample BobObliviousKey for testing (choice=0)."""
    sc = bitarray("10101010")
    return BobObliviousKey(
        sc=sc,
        choice_bit=0,
        key_length=8,
        security_parameter=1e-10,
    )


# =============================================================================
# Measurement Fixtures
# =============================================================================


@pytest.fixture
def sample_measurement_record() -> MeasurementRecord:
    """Sample MeasurementRecord for testing."""
    return MeasurementRecord(
        round_id=0,
        outcome=1,
        basis=0,
        timestamp_ns=1000.0,
        detected=True,
    )


@pytest.fixture
def sample_round_result() -> RoundResult:
    """Sample RoundResult for testing (matching bases, matching outcomes)."""
    return RoundResult(
        round_id=0,
        alice_outcome=1,
        bob_outcome=1,
        alice_basis=0,
        bob_basis=0,
        alice_detected=True,
        bob_detected=True,
    )


@pytest.fixture
def sample_detection_event() -> DetectionEvent:
    """Sample DetectionEvent for testing."""
    return DetectionEvent(
        round_id=0,
        detected=True,
        timestamp_ns=1000.0,
    )


# =============================================================================
# Phase Contract Fixtures
# =============================================================================


@pytest.fixture
def sample_quantum_phase_result() -> QuantumPhaseResult:
    """Generate a valid QuantumPhaseResult for testing."""
    n = 1000
    np.random.seed(42)  # Reproducible tests
    return QuantumPhaseResult(
        measurement_outcomes=np.random.randint(0, 2, n, dtype=np.uint8),
        basis_choices=np.random.randint(0, 2, n, dtype=np.uint8),
        round_ids=np.arange(n, dtype=np.int64),
        generation_timestamp=1_000_000.0,
        num_pairs_requested=n,
        num_pairs_generated=n,
        detection_events=[],
        timing_barrier_marked=True,
    )


@pytest.fixture
def sample_sifting_result() -> SiftingPhaseResult:
    """Generate a valid SiftingPhaseResult for testing."""
    n = 500
    np.random.seed(42)
    key_bits = np.random.randint(0, 2, n).tolist()
    key = bitarray(key_bits)
    qber_est = 0.05
    penalty = 0.01
    return SiftingPhaseResult(
        sifted_key_alice=key,
        sifted_key_bob=key.copy(),  # Perfect correlation for test
        matching_indices=np.arange(n, dtype=np.int64),
        i0_indices=np.arange(0, n // 2, dtype=np.int64),
        i1_indices=np.arange(n // 2, n, dtype=np.int64),
        test_set_indices=np.arange(0, n // 10, dtype=np.int64),
        qber_estimate=qber_est,
        qber_adjusted=qber_est + penalty,
        finite_size_penalty=penalty,
        test_set_size=n // 10,
        timing_compliant=True,
    )


@pytest.fixture
def sample_reconciliation_result() -> ReconciliationPhaseResult:
    """Generate a valid ReconciliationPhaseResult for testing."""
    np.random.seed(42)
    key_bits = np.random.randint(0, 2, 400).tolist()
    key = bitarray(key_bits)
    return ReconciliationPhaseResult(
        reconciled_key=key,
        num_blocks=4,
        blocks_succeeded=4,
        blocks_failed=0,
        total_syndrome_bits=100,
        effective_rate=0.75,
        hash_verified=True,
        leakage_within_cap=True,
        leakage_cap=200,
    )


@pytest.fixture
def sample_amplification_result(sample_alice_key: AliceObliviousKey) -> AmplificationPhaseResult:
    """Generate a valid AmplificationPhaseResult for testing."""
    return AmplificationPhaseResult(
        oblivious_key=sample_alice_key,
        qber=0.06,
        key_length=8,
        entropy_consumed=16.0,
        entropy_rate=0.08,
        metrics={"timing_ms": 100.0},
    )


@pytest.fixture
def sample_ot_output(
    sample_alice_key: AliceObliviousKey,
    sample_bob_key: BobObliviousKey,
) -> ObliviousTransferOutput:
    """Generate a valid ObliviousTransferOutput for testing."""
    return ObliviousTransferOutput(
        alice_key=sample_alice_key,
        bob_key=sample_bob_key,
        protocol_succeeded=True,
        total_rounds=1000,
        final_key_length=8,
        security_parameter=1e-10,
        entropy_rate=0.008,
    )


# =============================================================================
# Numpy Array Fixtures
# =============================================================================


@pytest.fixture
def sample_outcomes_array() -> np.ndarray:
    """Sample array of measurement outcomes."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100, dtype=np.uint8)


@pytest.fixture
def sample_bases_array() -> np.ndarray:
    """Sample array of basis choices."""
    np.random.seed(43)
    return np.random.randint(0, 2, 100, dtype=np.uint8)


# =============================================================================
# Error Test Helpers
# =============================================================================


@pytest.fixture
def invalid_qber() -> float:
    """QBER value that should trigger abort (above hard limit)."""
    return 0.25  # Above 22% hard limit


@pytest.fixture
def borderline_qber() -> float:
    """QBER value at the hard limit boundary."""
    return 0.22

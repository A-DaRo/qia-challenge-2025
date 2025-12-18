"""Phase-boundary integration: Quantum tooling → Phase I DTO.

Covers REQ-P12-001/010/020 from the extended test spec.

This is intentionally simulator-free and deterministic: we use
`MeasurementExecutor.measure_batch_sync` with `simulated_outcomes`.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.quantum.basis import BasisSelector
from caligo.quantum.measurement import MeasurementExecutor
from caligo.types.exceptions import ContractViolation
from caligo.types.phase_contracts import QuantumPhaseResult


@pytest.mark.integration
def test_p12_001_measurement_and_bases_build_valid_quantum_phase_result() -> None:
    """REQ-P12-001: measurement+basis arrays should satisfy QuantumPhaseResult."""

    n = 128
    basis_selector = BasisSelector(seed=b"p12_seed")
    bases = basis_selector.select_batch(n)
    round_ids = np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(123)
    simulated_outcomes = rng.integers(0, 2, size=n, dtype=np.uint8)

    meas = MeasurementExecutor()
    outcomes = meas.measure_batch_sync(
        bases=bases,
        round_ids=round_ids,
        simulated_outcomes=simulated_outcomes,
    )

    dto = QuantumPhaseResult(
        measurement_outcomes=outcomes,
        basis_choices=bases,
        round_ids=round_ids,
        generation_timestamp=1234.5,  # opaque simulation-time value (ns)
        num_pairs_requested=n,
        num_pairs_generated=n,
        detection_events=[],
        timing_barrier_marked=True,
    )

    assert dto.num_pairs_generated == n
    assert len(dto.measurement_outcomes) == n
    assert len(dto.basis_choices) == n
    assert set(np.unique(dto.measurement_outcomes)).issubset({0, 1})
    assert set(np.unique(dto.basis_choices)).issubset({0, 1})


@pytest.mark.integration
def test_p12_010_quantum_phase_result_length_mismatch_rejected() -> None:
    """REQ-P12-010: mismatched lengths must raise ContractViolation."""

    n = 16
    outcomes = np.zeros(n, dtype=np.uint8)
    bases = np.zeros(n - 1, dtype=np.uint8)
    round_ids = np.arange(n, dtype=np.int64)

    with pytest.raises(ContractViolation, match=r"POST-Q-002"):
        QuantumPhaseResult(
            measurement_outcomes=outcomes,
            basis_choices=bases,
            round_ids=round_ids,
            generation_timestamp=0.0,
            num_pairs_requested=n,
            num_pairs_generated=n,
            detection_events=[],
            timing_barrier_marked=True,
        )


@pytest.mark.integration
def test_p12_020_quantum_phase_result_invalid_values_rejected() -> None:
    """REQ-P12-020: invalid basis/outcome values must raise ContractViolation."""

    n = 8
    outcomes = np.zeros(n, dtype=np.uint8)
    bases = np.zeros(n, dtype=np.uint8)
    round_ids = np.arange(n, dtype=np.int64)

    outcomes[0] = 2
    with pytest.raises(ContractViolation, match=r"POST-Q-003"):
        QuantumPhaseResult(
            measurement_outcomes=outcomes,
            basis_choices=bases,
            round_ids=round_ids,
            generation_timestamp=0.0,
            num_pairs_requested=n,
            num_pairs_generated=n,
            detection_events=[],
            timing_barrier_marked=True,
        )

    outcomes[0] = 0
    bases[0] = 2
    with pytest.raises(ContractViolation, match=r"POST-Q-004"):
        QuantumPhaseResult(
            measurement_outcomes=outcomes,
            basis_choices=bases,
            round_ids=round_ids,
            generation_timestamp=0.0,
            num_pairs_requested=n,
            num_pairs_generated=n,
            detection_events=[],
            timing_barrier_marked=True,
        )

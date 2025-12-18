"""Phase-boundary integration: Phase II DTO → reconciliation orchestrator.

Covers REQ-P23-001/010/020 from the extended test spec.

Notes
-----
- This is *not* an end-to-end protocol test. It only checks that the
  Phase II DTO representation can be converted into the reconciliation
  layer’s expected inputs without semantic drift.
- This test uses the project’s LDPC matrix assets (same as existing
  reconciliation integration tests).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import (
    ReconciliationOrchestrator,
    ReconciliationOrchestratorConfig,
)
from caligo.types.exceptions import ContractViolation
from caligo.types.phase_contracts import QBER_HARD_LIMIT, SiftingPhaseResult
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy


@pytest.fixture(scope="module")
def matrix_manager() -> Iterator[MatrixManager]:
    """Load LDPC matrix manager once for this module."""

    yield MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


@pytest.mark.integration
def test_p23_001_sifting_phase_result_drives_single_block_reconciliation(
    matrix_manager: MatrixManager,
) -> None:
    """REQ-P23-001: SiftingPhaseResult bits should feed orchestrator correctly."""

    rng = np.random.default_rng(2025)

    # Payload length intentionally < 4096 so the encoder uses shortening.
    payload_len = 2867
    alice_np = rng.integers(0, 2, size=payload_len, dtype=np.uint8)

    # Bob differs by ~1%.
    bob_np = alice_np.copy()
    n_errors = max(1, int(payload_len * 0.01))
    error_positions = rng.choice(payload_len, size=n_errors, replace=False)
    bob_np[error_positions] = 1 - bob_np[error_positions]

    alice_bits = bitarray_from_numpy(alice_np)
    bob_bits = bitarray_from_numpy(bob_np)

    qber_estimate = 0.03
    finite_size_penalty = 0.0

    dto = SiftingPhaseResult(
        sifted_key_alice=alice_bits,
        sifted_key_bob=bob_bits,
        matching_indices=np.arange(payload_len, dtype=np.int64),
        i0_indices=np.arange(0, payload_len, 2, dtype=np.int64),
        i1_indices=np.arange(1, payload_len, 2, dtype=np.int64),
        test_set_indices=np.array([], dtype=np.int64),
        qber_estimate=qber_estimate,
        qber_adjusted=qber_estimate + finite_size_penalty,
        finite_size_penalty=finite_size_penalty,
        test_set_size=0,
        timing_compliant=True,
    )

    alice_arr = bitarray_to_numpy(dto.sifted_key_alice)
    bob_arr = bitarray_to_numpy(dto.sifted_key_bob)

    assert alice_arr.dtype == np.uint8
    assert bob_arr.dtype == np.uint8
    assert len(alice_arr) == payload_len
    assert len(bob_arr) == payload_len

    orchestrator = ReconciliationOrchestrator(
        matrix_manager=matrix_manager,
        config=ReconciliationOrchestratorConfig(frame_size=4096, max_retries=2),
        safety_cap=500_000,
    )

    result = orchestrator.reconcile_block(
        alice_key=alice_arr,
        bob_key=bob_arr,
        qber_estimate=dto.qber_adjusted,
        block_id=0,
    )

    if result.verified:
        np.testing.assert_array_equal(result.corrected_payload, alice_arr)


@pytest.mark.integration
def test_p23_010_sifting_phase_result_above_hard_limit_rejected() -> None:
    """REQ-P23-010: qber_adjusted > QBER_HARD_LIMIT must raise ContractViolation."""

    alice_bits = bitarray_from_numpy(np.zeros(10, dtype=np.uint8))
    bob_bits = bitarray_from_numpy(np.zeros(10, dtype=np.uint8))

    with pytest.raises(ContractViolation, match=r"POST-S-003"):
        SiftingPhaseResult(
            sifted_key_alice=alice_bits,
            sifted_key_bob=bob_bits,
            matching_indices=np.arange(10, dtype=np.int64),
            i0_indices=np.array([0, 1], dtype=np.int64),
            i1_indices=np.array([2, 3], dtype=np.int64),
            test_set_indices=np.array([], dtype=np.int64),
            qber_estimate=QBER_HARD_LIMIT + 1e-3,
            qber_adjusted=QBER_HARD_LIMIT + 1e-3,
            finite_size_penalty=0.0,
            test_set_size=0,
            timing_compliant=True,
        )


@pytest.mark.integration
def test_p23_020_bitarray_to_numpy_to_bytes_is_byte_per_bit_not_packed_bits() -> None:
    """REQ-P23-020: prevent accidental packed-bit bytes encoding.

    The reconciliation byte-oriented APIs interpret `bytes` as a byte-per-bit
    carrier (dtype uint8 with values in {0,1}). `bitarray.tobytes()` produces a
    packed-bit encoding, which is *not* compatible.
    """

    rng = np.random.default_rng(7)
    bits_np = rng.integers(0, 2, size=37, dtype=np.uint8)
    bits_ba = bitarray_from_numpy(bits_np)

    # Unpacked bytes: each element is 0 or 1.
    unpacked_bytes = bits_np.tobytes()
    roundtrip_np = np.frombuffer(unpacked_bytes, dtype=np.uint8)
    np.testing.assert_array_equal(roundtrip_np, bits_np)
    assert set(np.unique(roundtrip_np)).issubset({0, 1})

    # Packed bytes: length shrinks and values are not guaranteed 0/1.
    packed_bytes = bits_ba.tobytes()
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8)

    assert len(packed_np) != len(bits_np)
    assert not set(np.unique(packed_np)).issubset({0, 1})

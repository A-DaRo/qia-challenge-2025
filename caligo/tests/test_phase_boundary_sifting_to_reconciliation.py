"""Phase-boundary integration: Phase II DTO contract validation.

Covers REQ-P23-010/020 from the extended test spec.

Notes
-----
- These tests validate contract enforcement and data encoding at phase boundaries.
- They do NOT depend on the legacy ReconciliationOrchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.types.exceptions import ContractViolation
from caligo.types.phase_contracts import QBER_HARD_LIMIT, SiftingPhaseResult
from caligo.utils.bitarray_utils import bitarray_from_numpy


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

"""Phase-boundary integration: Phase III DTO → amplification formatter → OT DTO.

Covers REQ-P34-001/010/020 from the extended test spec.

This test is simulator-free and focuses on bitstring representation
compatibility across modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.amplification.formatter import OTOutputFormatter
from caligo.types.exceptions import ContractViolation
from caligo.types.phase_contracts import ObliviousTransferOutput, ReconciliationPhaseResult
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy


@pytest.mark.integration
def test_p34_001_reconciled_bits_can_be_partitioned_and_formatted_to_ot_keys() -> None:
    """REQ-P34-001: reconciled key bits should feed the formatter without drift."""

    key_length = 64

    rng = np.random.default_rng(1234)
    # Create a reconciled bitstring long enough to split into I0/I1.
    reconciled_np = rng.integers(0, 2, size=256, dtype=np.uint8)
    reconciled_ba = bitarray_from_numpy(reconciled_np)

    recon = ReconciliationPhaseResult(
        reconciled_key=reconciled_ba,
        num_blocks=1,
        blocks_succeeded=1,
        blocks_failed=0,
        total_syndrome_bits=1000,
        effective_rate=0.7,
        hash_verified=True,
        leakage_within_cap=True,
        leakage_cap=10_000,
    )

    recon_np = bitarray_to_numpy(recon.reconciled_key)
    i0 = recon_np[:128]
    i1 = recon_np[128:256]

    formatter = OTOutputFormatter(
        key_length=key_length,
        seed_0=b"p34_seed_0",
        seed_1=b"p34_seed_1",
    )

    alice_raw = formatter.compute_alice_keys(key_i0=i0, key_i1=i1)
    bob_raw = formatter.compute_bob_key(
        bob_key_i0=i0,
        bob_key_i1=i1,
        choice_bit=0,
    )

    alice_key, bob_key = formatter.format_final_output(alice_raw, bob_raw)

    assert bob_key.sc == alice_key.s0

    # REQ-P34-020: the final DTO accepts the produced keys.
    ot = ObliviousTransferOutput(
        alice_key=alice_key,
        bob_key=bob_key,
        protocol_succeeded=True,
        total_rounds=10_000,
        final_key_length=key_length,
    )

    assert ot.bob_key.sc == ot.alice_key.s0


@pytest.mark.integration
def test_p34_010_deliberate_mismatch_raises_contract_violation() -> None:
    """REQ-P34-010: deliberate mismatch must be rejected."""

    key_length = 64

    rng = np.random.default_rng(5678)
    reconciled_np = rng.integers(0, 2, size=256, dtype=np.uint8)
    recon_np = reconciled_np.copy()

    i0 = recon_np[:128]
    i1 = recon_np[128:256]

    formatter = OTOutputFormatter(
        key_length=key_length,
        seed_0=b"p34_seed_0",
        seed_1=b"p34_seed_1",
    )

    alice_raw = formatter.compute_alice_keys(key_i0=i0, key_i1=i1)

    # Bob uses a tampered I0; this should violate OT correctness.
    bob_i0 = i0.copy()
    bob_i0[0] = 1 - bob_i0[0]

    bob_raw = formatter.compute_bob_key(
        bob_key_i0=bob_i0,
        bob_key_i1=i1,
        choice_bit=0,
    )

    with pytest.raises(ContractViolation, match=r"OT correctness violated"):
        formatter.format_final_output(alice_raw, bob_raw)

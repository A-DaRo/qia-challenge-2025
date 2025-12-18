"""Additional negative tests for phase boundary contracts.

These focus on post-conditions that were not covered by the baseline
`test_phase_contracts.py` suite.
"""

import pytest
from bitarray import bitarray

from caligo.types.exceptions import ContractViolation
from caligo.types.keys import AliceObliviousKey, BobObliviousKey
from caligo.types.phase_contracts import ObliviousTransferOutput, ReconciliationPhaseResult


def test_reconciliation_post_r_001_leakage_cap_exceeded_raises() -> None:
    with pytest.raises(ContractViolation, match="POST-R-001"):
        ReconciliationPhaseResult(
            reconciled_key=bitarray("10101010"),
            num_blocks=1,
            blocks_succeeded=1,
            blocks_failed=0,
            total_syndrome_bits=11,
            effective_rate=0.5,
            hash_verified=True,
            leakage_within_cap=False,
            leakage_cap=10,
        )


def test_reconciliation_leakage_within_cap_no_raise() -> None:
    # If within cap, contract should not raise.
    result = ReconciliationPhaseResult(
        reconciled_key=bitarray("10101010"),
        num_blocks=1,
        blocks_succeeded=1,
        blocks_failed=0,
        total_syndrome_bits=10,
        effective_rate=0.5,
        hash_verified=True,
        leakage_within_cap=False,
        leakage_cap=10,
    )
    assert result.total_syndrome_bits == 10


def test_ot_post_ot_002_bob_key_length_mismatch_raises() -> None:
    alice = AliceObliviousKey(
        s0=bitarray("1010"),
        s1=bitarray("0101"),
        key_length=4,
    )
    bob = BobObliviousKey(
        sc=bitarray("10101010"),
        choice_bit=0,
        key_length=8,
    )

    with pytest.raises(ContractViolation, match="POST-OT-002"):
        ObliviousTransferOutput(
            alice_key=alice,
            bob_key=bob,
            protocol_succeeded=True,
            total_rounds=100,
            final_key_length=4,
        )

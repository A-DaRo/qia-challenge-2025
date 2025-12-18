"""Integration tests: amplification formatter ↔ phase contracts.

These tests intentionally cross module boundaries:
- `caligo.amplification.formatter` (OTOutputFormatter)
- `caligo.types.keys` (AliceObliviousKey/BobObliviousKey invariants)
- `caligo.types.phase_contracts` (ObliviousTransferOutput postconditions)

They are designed to be simulator-free and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.amplification.formatter import OTOutputFormatter
from caligo.types.exceptions import ContractViolation
from caligo.types.phase_contracts import ObliviousTransferOutput
from caligo.types.keys import BobObliviousKey


def test_formatter_outputs_satisfy_oblivious_transfer_output_contract_choice_0() -> None:
    """Formatter output should satisfy POST-OT-00x when not tampered."""

    key_length = 64
    security_param = 1e-12
    entropy_consumed = 256.0

    formatter = OTOutputFormatter(
        key_length=key_length,
        seed_0=b"seed_0_deterministic",
        seed_1=b"seed_1_deterministic",
    )

    rng = np.random.default_rng(123)
    key_i0 = rng.integers(0, 2, size=256, dtype=np.uint8)
    key_i1 = rng.integers(0, 2, size=256, dtype=np.uint8)

    alice_raw = formatter.compute_alice_keys(key_i0=key_i0, key_i1=key_i1)
    bob_raw = formatter.compute_bob_key(
        bob_key_i0=key_i0,
        bob_key_i1=key_i1,
        choice_bit=0,
    )

    alice_key, bob_key = formatter.format_final_output(
        alice_output=alice_raw,
        bob_output=bob_raw,
        security_param=security_param,
        entropy_consumed=entropy_consumed,
    )

    assert alice_key.security_parameter == security_param
    assert alice_key.entropy_consumed == entropy_consumed
    assert bob_key.security_parameter == security_param

    # Phase-contract boundary: final DTO asserts Sᴄ correctness and lengths.
    ot = ObliviousTransferOutput(
        alice_key=alice_key,
        bob_key=bob_key,
        protocol_succeeded=True,
        total_rounds=10_000,
        final_key_length=key_length,
        security_parameter=security_param,
        entropy_rate=key_length / 10_000,
    )

    assert ot.bob_key.choice_bit == 0
    assert ot.bob_key.sc == ot.alice_key.s0


def test_oblivious_transfer_output_detects_tampering_after_formatting() -> None:
    """Phase-contract boundary must detect a corrupted Sᴄ even if lengths match."""

    key_length = 64
    formatter = OTOutputFormatter(
        key_length=key_length,
        seed_0=b"seed_0_deterministic",
        seed_1=b"seed_1_deterministic",
    )

    rng = np.random.default_rng(456)
    key_i0 = rng.integers(0, 2, size=256, dtype=np.uint8)
    key_i1 = rng.integers(0, 2, size=256, dtype=np.uint8)

    alice_raw = formatter.compute_alice_keys(key_i0=key_i0, key_i1=key_i1)
    bob_raw = formatter.compute_bob_key(
        bob_key_i0=key_i0,
        bob_key_i1=key_i1,
        choice_bit=1,
    )

    alice_key, bob_key = formatter.format_final_output(alice_raw, bob_raw)

    tampered_sc = bob_key.sc.copy()
    tampered_sc[0] = not tampered_sc[0]

    tampered_bob_key = BobObliviousKey(
        sc=tampered_sc,
        choice_bit=bob_key.choice_bit,
        key_length=bob_key.key_length,
        security_parameter=bob_key.security_parameter,
    )

    with pytest.raises(ContractViolation, match=r"POST-OT-003"):
        ObliviousTransferOutput(
            alice_key=alice_key,
            bob_key=tampered_bob_key,
            protocol_succeeded=True,
            total_rounds=10_000,
            final_key_length=key_length,
        )

"""E2E tests for Phase E (SquidASM execution).

These tests are conditional on SquidASM/NetSquid being available.
They validate end-to-end OT correctness:
- Bob receives exactly one of Alice's keys (S_c == S0 or S1)
- Output key length is positive (no "Death Valley" output contract violation)

Notes
-----
This is intentionally a small E2E to keep runtime manageable while still
exercising the ordered classical messaging, timing barrier, and the phase
pipeline (quantum → sifting → reconciliation → amplification).
"""

from __future__ import annotations

import pytest

from caligo.protocol import ProtocolParameters, run_protocol
from caligo.simulation.physical_model import NSMParameters


def _default_params(*, session_id: str, num_pairs: int) -> ProtocolParameters:
    return ProtocolParameters(
        session_id=session_id,
        nsm_params=NSMParameters(
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.99,
        ),
        num_pairs=num_pairs,
        num_qubits=200,
    )


@pytest.mark.parametrize("choice_bit", [0, 1])
def test_phase_e_end_to_end_ot_agreement(choice_bit: int) -> None:
    """Bob's output must match exactly Alice's chosen key."""

#    pytest.importorskip("squidasm")
#    pytest.importorskip("netsquid")

#    # The current environment may ship NetQASM 2.x, whose instruction set is
#    # not supported by SquidASM 0.13.x stack runner (e.g. it errors on
#    # instructions like "mov Q0 M0").
#    import netqasm
#
#    if str(getattr(netqasm, "__version__", "")).startswith("2"):
#        pytest.skip(
#            "NetQASM 2.x detected; Phase E SquidASM E2E requires NetQASM < 2.0 "
#            "(or a newer SquidASM stack runner that supports NetQASM 2.x)."
#        )

    # Needs to be large enough to overcome finite-size penalties in
    # `SecureKeyLengthCalculator` (epsilon=1e-10 => ~64 bits penalty).
    params = _default_params(session_id=f"e2e-{choice_bit}", num_pairs=2000)

    ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)

    assert ot.protocol_succeeded is True
    assert ot.final_key_length > 0

    assert ot.bob_key.choice_bit == choice_bit
    assert ot.bob_key.key_length == ot.final_key_length

    if choice_bit == 0:
        assert ot.bob_key.sc == ot.alice_key.s0
        assert ot.bob_key.sc != ot.alice_key.s1
    else:
        assert ot.bob_key.sc == ot.alice_key.s1
        assert ot.bob_key.sc != ot.alice_key.s0

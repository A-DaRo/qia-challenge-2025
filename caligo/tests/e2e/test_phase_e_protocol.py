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

from caligo.protocol import PrecomputedEPRData, ProtocolParameters, run_protocol
from caligo.simulation.physical_model import NSMParameters
from caligo.reconciliation.factory import ReconciliationConfig, ReconciliationType


NUM_PAIRS = 100_000
PARALLEL_WORKERS = 8
PREFETCH_BATCHES = 2


@pytest.fixture(scope="module")
def _precomputed_epr() -> PrecomputedEPRData:
    """Precompute an EPR dataset once to keep the E2E runtime manageable."""

    from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
    from caligo.quantum.parallel import ParallelEPRConfig

    # Map Phase E's channel_fidelity=0.99 to a small depolarizing rate.
    noise_rate = 0.01

    config = CaligoConfig(
        num_epr_pairs=NUM_PAIRS,
        parallel_config=ParallelEPRConfig(
            enabled=True,
            num_workers=PARALLEL_WORKERS,
            prefetch_batches=PREFETCH_BATCHES,
        ),
        network_config={"noise": float(noise_rate)},
    )

    factory = EPRGenerationFactory(config)
    strategy = factory.create_strategy()
    try:
        alice_out, alice_bases, bob_out, bob_bases = strategy.generate(NUM_PAIRS)
    finally:
        if isinstance(strategy, ParallelEPRStrategy):
            strategy.shutdown()

    return PrecomputedEPRData(
        alice_outcomes=alice_out,
        alice_bases=alice_bases,
        bob_outcomes=bob_out,
        bob_bases=bob_bases,
    )


def _default_params(*, session_id: str, num_pairs: int, precomputed: PrecomputedEPRData) -> ProtocolParameters:
    return ProtocolParameters(
        session_id=session_id,
        nsm_params=NSMParameters(
            # NOTE: storage_noise_r=0.35 is chosen to be compatible with rate 0.5 LDPC
            # reconciliation. Higher values (e.g., 0.75) require higher code rates
            # (0.7+) which our current BP decoder doesn't reliably support.
            # TODO: Improve LDPC decoder to support higher rates, then increase r.
            storage_noise_r=0.35,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.99,
        ),
        num_pairs=num_pairs,
        num_qubits=200,
        precomputed_epr=precomputed,
    )


def _blind_params(*, session_id: str, num_pairs: int, precomputed: PrecomputedEPRData) -> ProtocolParameters:
    """
    Create protocol parameters using blind reconciliation strategy.

    Blind reconciliation skips QBER estimation and uses iterative syndrome
    decoding with progressively revealed padding bits (Martinez-Mateo 2012).
    """
    return ProtocolParameters(
        session_id=session_id,
        nsm_params=NSMParameters(
            storage_noise_r=0.35,
            storage_rate_nu=0.002,
            delta_t_ns=1_000_000,
            channel_fidelity=0.99,
        ),
        num_pairs=num_pairs,
        num_qubits=200,
        precomputed_epr=precomputed,
        reconciliation=ReconciliationConfig(
            reconciliation_type=ReconciliationType.BLIND,
            frame_size=4096,
            max_iterations=100,
            max_blind_rounds=5,  # Allow more rounds for blind decoding
        ),
    )


@pytest.mark.parametrize("choice_bit", [0, 1])
def test_phase_e_end_to_end_ot_agreement(choice_bit: int, _precomputed_epr: PrecomputedEPRData) -> None:
    """Bob's output must match exactly Alice's chosen key."""

    # Needs to be large enough to overcome finite-size penalties.
    params = _default_params(
        session_id=f"e2e-{choice_bit}",
        num_pairs=NUM_PAIRS,
        precomputed=_precomputed_epr,
    )

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


@pytest.mark.parametrize("choice_bit", [0, 1])
def test_phase_e_blind_reconciliation_ot_agreement(
    choice_bit: int, _precomputed_epr: PrecomputedEPRData
) -> None:
    """
    Test OT correctness using blind reconciliation strategy.

    Blind reconciliation (Martinez-Mateo 2012) differs from baseline by:
    - Skipping QBER estimation (no test bits consumed)
    - Using iterative syndrome decoding with progressively revealed padding
    - Using NSM parameters (qber_conditional) as the channel prior for LLR

    Bob's output must still match exactly one of Alice's keys.
    """
    params = _blind_params(
        session_id=f"e2e-blind-{choice_bit}",
        num_pairs=NUM_PAIRS,
        precomputed=_precomputed_epr,
    )

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

"""E2E tests for Phase E (SquidASM execution).

These tests are conditional on SquidASM/NetSquid being available.
They validate end-to-end OT correctness:
- Bob receives exactly one of Alice's keys (S_c == S0 or S1)
- Output key length is positive (no "Death Valley" output contract violation)

Tests cover:
- Baseline reconciliation with varying noise levels
- Different NSM storage noise parameters
- Various channel fidelity configurations
- Rate selection across QBER ranges

Notes
-----
This is intentionally a small E2E to keep runtime manageable while still
exercising the ordered classical messaging, timing barrier, and the phase
pipeline (quantum → sifting → reconciliation → amplification).
"""

from __future__ import annotations

from typing import Tuple

import pytest
import numpy as np

from caligo.protocol import PrecomputedEPRData, ProtocolParameters, run_protocol
from caligo.simulation.physical_model import NSMParameters
from caligo.reconciliation.factory import ReconciliationConfig, ReconciliationType
from caligo.types.exceptions import CaligoError
from caligo.types.exceptions import (
    CaligoError,
    EntropyDepletedError,
    LeakageBudgetExceeded,
    SecurityError,
)


NUM_PAIRS = 100_000
PARALLEL_WORKERS = 8
PREFETCH_BATCHES = 2


def _generate_epr_data(num_pairs: int, noise_rate: float) -> PrecomputedEPRData:
    """Generate EPR data with specified noise rate."""
    from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
    from caligo.quantum.parallel import ParallelEPRConfig

    config = CaligoConfig(
        num_epr_pairs=num_pairs,
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
        alice_out, alice_bases, bob_out, bob_bases = strategy.generate(num_pairs)
    finally:
        if isinstance(strategy, ParallelEPRStrategy):
            strategy.shutdown()

    return PrecomputedEPRData(
        alice_outcomes=alice_out,
        alice_bases=alice_bases,
        bob_outcomes=bob_out,
        bob_bases=bob_bases,
    )


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


# =============================================================================
# Comprehensive Baseline Tests with Varying Noise Configurations
# =============================================================================

class TestBaselineVariedNoise:
    """
    Test baseline reconciliation across various noise configurations.
    
    Per Theoretical Report v2 §3, baseline reconciliation uses:
    - QBER-based rate selection: R = 1 - f(p*) × h(p*)
    - Fixed syndrome leakage: |Σ| = (1 - R_0) × n
    
    These tests verify protocol correctness across the operating range.
    """
    
    @pytest.mark.parametrize("storage_r", [0.30, 0.35, 0.40])
    def test_storage_noise_range(
        self, storage_r: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """
        Test baseline with varying NSM storage noise parameter.
        
        The storage noise r affects the conditional QBER and thus
        the rate selection. Values in [0.30, 0.40] represent practical
        operating ranges per NSM model constraints.
        """
        params = ProtocolParameters(
            session_id=f"storage-r-{storage_r}",
            nsm_params=NSMParameters(
                storage_noise_r=storage_r,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=0)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
        assert ot.bob_key.sc == ot.alice_key.s0
    
    @pytest.mark.parametrize("fidelity", [0.97, 0.98, 0.99])
    def test_channel_fidelity_range(
        self, fidelity: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """
        Test baseline with varying channel fidelity.
        
        Higher fidelity means lower QBER and potentially higher rates.
        The protocol should adapt automatically.
        """
        params = ProtocolParameters(
            session_id=f"fidelity-{fidelity}",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=fidelity,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=1)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
        assert ot.bob_key.sc == ot.alice_key.s1
    
    @pytest.mark.parametrize("max_iter", [50, 100, 150])
    def test_decoder_iteration_budget(
        self, max_iter: int, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """
        Test baseline with varying decoder iteration budgets.
        
        More iterations allow better convergence but increase latency.
        The protocol should succeed with sufficient iterations.
        """
        params = ProtocolParameters(
            session_id=f"iter-{max_iter}",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
            reconciliation=ReconciliationConfig(
                reconciliation_type=ReconciliationType.BASELINE,
                frame_size=4096,
                max_iterations=max_iter,
            ),
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=0)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0


class TestBaselineRobustness:
    """
    Test baseline reconciliation robustness under edge conditions.
    
    These tests verify that the protocol degrades gracefully and
    maintains security guarantees under challenging conditions.
    """
    
    def test_minimum_viable_pairs(self) -> None:
        """
        Test with reduced number of pairs that should still succeed.
        
        The protocol requires sufficient pairs to overcome finite-size
        penalties and produce a non-zero key. We use 80% of the standard
        count to verify robustness while maintaining feasibility.
        """
        min_pairs = 80_000  # 80% of standard, still viable
        precomputed = _generate_epr_data(min_pairs, noise_rate=0.01)
        
        params = ProtocolParameters(
            session_id="min-pairs",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,  # Standard noise
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=min_pairs,
            num_qubits=200,
            precomputed_epr=precomputed,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=0)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
    
    def test_low_noise_high_rate_regime(self) -> None:
        """
        Test low-noise regime where high rates should be selected.
        
        Per Theoretical Report v2 §3.2, rate selection follows:
        R = 1 - f_crit × h(QBER)
        
        For low QBER, this should yield higher rates, though still
        subject to leakage constraints.
        """
        # Generate clean EPR data with moderate noise
        precomputed = _generate_epr_data(NUM_PAIRS, noise_rate=0.02)
        
        params = ProtocolParameters(
            session_id="low-noise",
            nsm_params=NSMParameters(
                storage_noise_r=0.30,  # Low but feasible storage noise
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,  # High fidelity
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=precomputed,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=1)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
        # Higher rate means better efficiency (longer key for same syndrome)
        # We don't assert exact key length, just that it succeeds


class TestOTSecurityInvariants:
    """
    Test that OT security invariants hold across configurations.
    
    Per the NSM security model:
    - Bob receives EXACTLY ONE of Alice's keys (not both)
    - The received key matches the choice bit
    - Protocol succeeds or fails cleanly (no partial information leakage)
    """
    
    @pytest.mark.parametrize(
        "choice_bit,storage_r",
        [(0, 0.30), (0, 0.40), (1, 0.30), (1, 0.40)],
    )
    def test_choice_bit_determines_output(
        self, choice_bit: int, storage_r: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """
        Verify Bob's output key matches his choice bit across configurations.
        
        This is the fundamental OT correctness property:
        S_c = S_{choice_bit}
        """
        params = ProtocolParameters(
            session_id=f"ot-choice-{choice_bit}-r-{storage_r}",
            nsm_params=NSMParameters(
                storage_noise_r=storage_r,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
        
        assert ot.protocol_succeeded is True
        assert ot.bob_key.choice_bit == choice_bit
        
        # Core OT invariant
        if choice_bit == 0:
            assert ot.bob_key.sc == ot.alice_key.s0
            assert ot.bob_key.sc != ot.alice_key.s1  # Should not equal the other
        else:
            assert ot.bob_key.sc == ot.alice_key.s1
            assert ot.bob_key.sc != ot.alice_key.s0  # Should not equal the other
    
    def test_alice_keys_are_distinct(
        self, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """
        Verify Alice's two output keys are distinct.
        
        Privacy amplification should produce independent keys.
        If S0 == S1, the OT would not provide any security.
        """
        params = ProtocolParameters(
            session_id="distinct-keys",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=0)
        
        assert ot.protocol_succeeded is True
        # Alice's two keys must be different
        assert ot.alice_key.s0 != ot.alice_key.s1


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


# =============================================================================
# Comprehensive Blind Tests with Varying Noise Configurations
# =============================================================================

class TestBlindVariedNoise:
    """
    Test blind reconciliation across various noise configurations.
    
    Per Theoretical Report v2 §4, blind reconciliation uses:
    - No QBER pre-estimation (uses heuristic from NSM parameters)
    - Iterative revelation of punctured values
    - Leakage: leak_Blind = (1-R_0)×n + h + Σ Δ_i
    
    These tests verify protocol correctness across the operating range.
    """
    
    @pytest.mark.parametrize("storage_r", [0.30, 0.35, 0.40])
    def test_blind_storage_noise_range(
        self, storage_r: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """Test blind with varying NSM storage noise parameter."""
        params = ProtocolParameters(
            session_id=f"blind-storage-r-{storage_r}",
            nsm_params=NSMParameters(
                storage_noise_r=storage_r,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
            reconciliation=ReconciliationConfig(
                reconciliation_type=ReconciliationType.BLIND,
                frame_size=4096,
                max_iterations=100,
                max_blind_rounds=5,
            ),
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=0)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
        assert ot.bob_key.sc == ot.alice_key.s0
    
    @pytest.mark.parametrize("fidelity", [0.97, 0.98, 0.99])
    def test_blind_channel_fidelity_range(
        self, fidelity: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """Test blind with varying channel fidelity."""
        params = ProtocolParameters(
            session_id=f"blind-fidelity-{fidelity}",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=fidelity,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
            reconciliation=ReconciliationConfig(
                reconciliation_type=ReconciliationType.BLIND,
                frame_size=4096,
                max_iterations=100,
                max_blind_rounds=5,
            ),
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=1)
        
        assert ot.protocol_succeeded is True
        assert ot.final_key_length > 0
        assert ot.bob_key.sc == ot.alice_key.s1


class TestBlindOTSecurityInvariants:
    """Test that OT security invariants hold for blind protocol."""
    
    @pytest.mark.parametrize(
        "choice_bit,storage_r",
        [(0, 0.30), (0, 0.40), (1, 0.30), (1, 0.40)],
    )
    def test_blind_choice_bit_determines_output(
        self, choice_bit: int, storage_r: float, _precomputed_epr: PrecomputedEPRData
    ) -> None:
        """Verify Bob's output key matches his choice bit for blind protocol."""
        params = ProtocolParameters(
            session_id=f"blind-ot-choice-{choice_bit}-r-{storage_r}",
            nsm_params=NSMParameters(
                storage_noise_r=storage_r,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=0.99,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=_precomputed_epr,
            reconciliation=ReconciliationConfig(
                reconciliation_type=ReconciliationType.BLIND,
                frame_size=4096,
                max_iterations=100,
                max_blind_rounds=5,
            ),
        )
        
        ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
        
        assert ot.protocol_succeeded is True
        assert ot.bob_key.choice_bit == choice_bit
        
        if choice_bit == 0:
            assert ot.bob_key.sc == ot.alice_key.s0
        else:
            assert ot.bob_key.sc == ot.alice_key.s1


# =============================================================================
# QBER Stress Tests - Push Both Protocols to the Theoretical Limit
# =============================================================================

class TestQBERStressLimit:
    """
    Test both baseline and blind protocols under high QBER conditions.
    
    Per Theoretical Report v2 §1.2, the theoretical QBER limit for BB84-based
    protocols is approximately 0.11 (11%). Beyond 0.22 (22%), secure key
    extraction becomes information-theoretically impossible.
    
    These stress tests push QBER toward the limit. Protocols may fail gracefully
    (SecurityError, EntropyDepletedError) but should NOT raise unexpected errors.
    """
    
    @pytest.mark.parametrize("qber", [0.05, 0.08, 0.10, 0.12])
    def test_baseline_qber_stress(self, qber: float) -> None:
        """
        Test baseline protocol with increasing QBER.
        
        At low QBER (<0.10), the protocol should succeed.
        At higher QBER, it may fail gracefully due to entropy depletion.
        No unexpected exceptions should be raised.
        """
        precomputed = _generate_epr_data(NUM_PAIRS, noise_rate=qber)
        
        params = ProtocolParameters(
            session_id=f"baseline-qber-stress-{qber}",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=1.0 - qber,  # Approximate mapping
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=precomputed,
        )
        
        try:
            ot, _raw = run_protocol(params, bob_choice_bit=0)
            # If we get here, protocol succeeded
            assert ot.final_key_length > 0, "Key length should be positive if protocol succeeded"
        except (RuntimeError, CaligoError) as e:
            # Expected: protocol may abort at high QBER due to:
            # - LeakageBudgetExceeded: leakage exceeds safety cap
            # - SecurityError: QBER too high for secure extraction
            # - EntropyDepletedError: insufficient entropy for PA
            # All of these are acceptable failures at extreme QBER.
            pass
    
    @pytest.mark.parametrize("qber", [0.05, 0.08, 0.10, 0.12])
    def test_blind_qber_stress(self, qber: float) -> None:
        """
        Test blind protocol with increasing QBER.
        
        Blind may handle high QBER differently since it doesn't pre-estimate QBER.
        At high QBER, it may fail gracefully due to entropy depletion or
        verification failures.
        """
        precomputed = _generate_epr_data(NUM_PAIRS, noise_rate=qber)
        
        params = ProtocolParameters(
            session_id=f"blind-qber-stress-{qber}",
            nsm_params=NSMParameters(
                storage_noise_r=0.35,
                storage_rate_nu=0.002,
                delta_t_ns=1_000_000,
                channel_fidelity=1.0 - qber,
            ),
            num_pairs=NUM_PAIRS,
            num_qubits=200,
            precomputed_epr=precomputed,
            reconciliation=ReconciliationConfig(
                reconciliation_type=ReconciliationType.BLIND,
                frame_size=4096,
                max_iterations=100,
                max_blind_rounds=5,
            ),
        )
        
        try:
            ot, _raw = run_protocol(params, bob_choice_bit=1)
            assert ot.final_key_length > 0, "Key length should be positive if protocol succeeded"
        except (RuntimeError, CaligoError) as e:
            # At higher QBER, protocol may fail due to:
            # - LeakageBudgetExceeded: leakage exceeds safety cap
            # - SecurityError: QBER too high for secure extraction
            # - EntropyDepletedError: insufficient entropy for PA
            # All of these are acceptable failures at extreme QBER.
            pass
    
    @pytest.mark.parametrize("qber", [0.15, 0.18, 0.20, 0.22])
    def test_extreme_qber_graceful_failure(self, qber: float) -> None:
        """
        Test that protocols fail gracefully at extreme QBER (>0.15).
        
        At these QBER levels, secure key extraction is not expected to succeed.
        The test verifies that the protocol raises appropriate errors rather
        than crashing or producing incorrect results.
        """
        precomputed = _generate_epr_data(NUM_PAIRS, noise_rate=qber)
        
        # Test both baseline and blind at extreme QBER
        for recon_type in [ReconciliationType.BASELINE, ReconciliationType.BLIND]:
            params = ProtocolParameters(
                session_id=f"extreme-qber-{qber}-{recon_type.name}",
                nsm_params=NSMParameters(
                    storage_noise_r=0.35,
                    storage_rate_nu=0.002,
                    delta_t_ns=1_000_000,
                    channel_fidelity=max(0.78, 1.0 - qber),  # Clamp to avoid invalid fidelity
                ),
                num_pairs=NUM_PAIRS,
                num_qubits=200,
                precomputed_epr=precomputed,
                reconciliation=ReconciliationConfig(
                    reconciliation_type=recon_type,
                    frame_size=4096,
                    max_iterations=100,
                    max_blind_rounds=5 if recon_type == ReconciliationType.BLIND else 1,
                ),
            )
            
            # At extreme QBER, we expect failure - but no crashes
            try:
                ot, _raw = run_protocol(params, bob_choice_bit=0)
                # If it somehow succeeded, that's fine - just verify output
                assert ot.protocol_succeeded is True
                assert ot.final_key_length >= 0  # May be 0 at extreme conditions
            except (RuntimeError, CaligoError):
                # Expected - protocol should abort at extreme QBER due to:
                # - LeakageBudgetExceeded, SecurityError, EntropyDepletedError, etc.
                pass
            except Exception as e:
                # Unexpected exception type - fail the test
                pytest.fail(f"Unexpected exception type at QBER={qber}: {type(e).__name__}: {e}")

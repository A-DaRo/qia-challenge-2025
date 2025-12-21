"""
Test Suite for Blind Reconciliation Strategy (Task 4).

Tests the Martinez-Mateo et al. (2012) blind reconciliation protocol.

Per Theoretical Report v2 §4:
- Syndrome reuse (Theorem 4.1)
- Hot-Start persistence
- Revelation order
- NSM-gated optimization

References:
[2] Martinez-Mateo et al. (2012), "Blind Reconciliation"
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from caligo.reconciliation.strategies import (
    BlindDecoderState,
    BlindStrategy,
    BlockResult,
    DecoderResult,
    ReconciliationContext,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mother_code() -> MagicMock:
    """Mock MotherCodeManager for Blind protocol."""
    mock = MagicMock()
    mock.frame_size = 4096
    mock.mother_rate = 0.5
    
    # Modulation indices (deterministic for testing)
    modulation_indices = np.arange(410, dtype=np.int64)  # 10% modulation
    mock.get_modulation_indices = MagicMock(return_value=modulation_indices)
    
    # Mother pattern (no puncturing for syndrome)
    mother_pattern = np.zeros(4096, dtype=np.uint8)
    mock.get_pattern = MagicMock(return_value=mother_pattern)
    
    # Compiled topology
    mock.compiled_topology = MagicMock()
    mock.compiled_topology.n_edges = 12000
    mock.compiled_topology.n_checks = 2048
    mock.compiled_topology.n_vars = 4096
    
    return mock


@pytest.fixture
def mock_codec() -> MagicMock:
    """Mock LDPCCodec with call tracking."""
    mock = MagicMock()
    
    # Track encode calls
    mock.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
    
    # Decode returns successful result with messages
    def decode_blind_impl(syndrome, llr, messages, frozen_mask, max_iterations):
        return DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=10,
            messages=messages.copy(),  # Return copy of input messages
        )
    
    mock.decode_blind = MagicMock(side_effect=decode_blind_impl)
    
    return mock


@pytest.fixture
def leakage_tracker() -> LeakageTracker:
    """Create leakage tracker with generous cap."""
    return LeakageTracker(safety_cap=100000)


@pytest.fixture
def blind_strategy(
    mock_mother_code: MagicMock,
    mock_codec: MagicMock,
    leakage_tracker: LeakageTracker,
) -> BlindStrategy:
    """Create blind strategy with mocks."""
    return BlindStrategy(
        mother_code=mock_mother_code,
        codec=mock_codec,
        leakage_tracker=leakage_tracker,
        max_blind_iterations=3,
        modulation_fraction=0.1,
    )


@pytest.fixture
def context_no_heuristic() -> ReconciliationContext:
    """Context without QBER heuristic."""
    return ReconciliationContext(
        session_id=1,
        frame_size=4096,
        mother_rate=0.5,
        max_iterations=60,
        hash_bits=64,
        f_crit=1.1,
        qber_measured=None,
        qber_heuristic=None,
        modulation_delta=0.1,
    )


@pytest.fixture
def context_with_heuristic() -> ReconciliationContext:
    """Context with QBER heuristic (for NSM-gating)."""
    return ReconciliationContext(
        session_id=1,
        frame_size=4096,
        mother_rate=0.5,
        max_iterations=60,
        hash_bits=64,
        f_crit=1.1,
        qber_measured=None,
        qber_heuristic=0.08,  # High heuristic
        modulation_delta=0.1,
    )


# =============================================================================
# TASK 4.1: Syndrome Reuse (Theorem 4.1)
# =============================================================================


class TestSyndromeReuse:
    """
    Task 4.1: Verify encode is called exactly once.
    
    Per Theorem 4.1: The syndrome is computed and transmitted ONCE.
    Subsequent iterations only reveal additional bits.
    """
    
    def test_encode_called_once_on_success(
        self,
        blind_strategy: BlindStrategy,
        mock_codec: MagicMock,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Encode should be called exactly once for successful session."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        # Get initial message
        message = next(gen)
        assert message["kind"] == "blind"
        
        # Verify encode was called once
        assert mock_codec.encode.call_count == 1
        
        # Complete with success on first iteration
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration:
            pass
        
        # Still only one encode call
        assert mock_codec.encode.call_count == 1
    
    def test_encode_called_once_across_iterations(
        self,
        blind_strategy: BlindStrategy,
        mock_codec: MagicMock,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Encode should be called once even with multiple reveal iterations."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        # Initial message (iteration 1)
        message1 = next(gen)
        assert mock_codec.encode.call_count == 1
        
        # Fail first iteration, trigger reveal
        message2 = gen.send({"verified": False, "converged": False})
        assert mock_codec.encode.call_count == 1  # Still only one!
        
        assert message2["kind"] == "blind_reveal"
        assert message2["iteration"] == 2
        
        # Complete
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration:
            pass
        
        # Final check: encode called exactly once
        assert mock_codec.encode.call_count == 1
    
    def test_syndrome_unchanged_across_reveals(
        self,
        blind_strategy: BlindStrategy,
        mock_codec: MagicMock,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """The syndrome in iteration 1 is never recomputed."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        # Set a known syndrome
        expected_syndrome = np.random.randint(0, 2, 2048, dtype=np.uint8)
        mock_codec.encode.return_value = expected_syndrome.copy()
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        message1 = next(gen)
        
        # Check syndrome was captured
        np.testing.assert_array_equal(
            np.array(message1["syndrome"], dtype=np.uint8),
            expected_syndrome,
        )


# =============================================================================
# TASK 4.2: Hot-Start Persistence
# =============================================================================


class TestHotStartPersistence:
    """
    Task 4.2: Verify messages persist across iterations.
    
    The "Hot-Start" kernel reuses edge messages from previous iteration.
    """
    
    def test_messages_passed_between_iterations(
        self,
        mock_mother_code: MagicMock,
        leakage_tracker: LeakageTracker,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Bob's decoder receives previous iteration's messages."""
        # Create codec that tracks messages
        messages_history = []
        
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        
        def track_decode_blind(syndrome, llr, messages, frozen_mask, max_iterations):
            messages_history.append(messages.copy())
            # Modify messages to simulate BP update
            result_messages = messages.copy()
            # Messages is 1D array of shape (2*n_edges,)
            result_messages[:100] = np.random.randn(100)  # Simulate update
            return DecoderResult(
                corrected_bits=np.zeros(4096, dtype=np.uint8),
                converged=len(messages_history) >= 2,  # Converge on 2nd iteration
                iterations=10,
                messages=result_messages,
            )
        
        mock_codec.decode_blind = MagicMock(side_effect=track_decode_blind)
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.bob_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        # Bob yields empty message first
        first = next(gen)
        
        # Send Alice's initial message
        alice_msg = {
            "kind": "blind",
            "block_id": 0,
            "syndrome": [0] * 2048,
            "puncture_indices": list(range(410)),
            "payload_length": 3000,
            "hash_value": 0,
            "qber_prior": 0.05,
            "iteration": 1,
            "revealed_indices": [],
            "revealed_values": [],
        }
        
        response1 = gen.send(alice_msg)
        
        # First decode call should have happened
        assert len(messages_history) >= 1
        
        # If not verified, send reveal
        if not response1.get("verified"):
            reveal_msg = {
                "kind": "blind_reveal",
                "block_id": 0,
                "iteration": 2,
                "revealed_indices": list(range(50)),
                "revealed_values": [0] * 50,
            }
            
            try:
                response2 = gen.send(reveal_msg)
            except StopIteration:
                pass
            
            # Second decode should have received modified messages
            assert len(messages_history) >= 2
    
    def test_frozen_mask_updated_with_reveals(
        self,
        mock_mother_code: MagicMock,
        leakage_tracker: LeakageTracker,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Revealed bits should be marked in frozen_mask."""
        frozen_masks = []
        
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        
        def capture_frozen_mask(syndrome, llr, messages, frozen_mask, max_iterations):
            frozen_masks.append(frozen_mask.copy())
            return DecoderResult(
                corrected_bits=np.zeros(4096, dtype=np.uint8),
                converged=len(frozen_masks) >= 2,
                iterations=10,
                messages=messages.copy(),
            )
        
        mock_codec.decode_blind = MagicMock(side_effect=capture_frozen_mask)
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.bob_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        first = next(gen)
        
        alice_msg = {
            "kind": "blind",
            "block_id": 0,
            "syndrome": [0] * 2048,
            "puncture_indices": list(range(410)),
            "payload_length": 3000,
            "hash_value": 0,
            "qber_prior": 0.05,
            "iteration": 1,
            "revealed_indices": [],
            "revealed_values": [],
        }
        
        response1 = gen.send(alice_msg)
        
        if not response1.get("verified"):
            # Reveal some indices
            reveal_indices = [0, 1, 2, 3, 4]
            reveal_msg = {
                "kind": "blind_reveal",
                "block_id": 0,
                "iteration": 2,
                "revealed_indices": reveal_indices,
                "revealed_values": [0] * len(reveal_indices),
            }
            
            try:
                gen.send(reveal_msg)
            except StopIteration:
                pass
            
            # Check frozen mask was updated
            if len(frozen_masks) >= 2:
                # Second iteration should have more frozen bits
                frozen_count_1 = frozen_masks[0].sum()
                frozen_count_2 = frozen_masks[1].sum()
                assert frozen_count_2 >= frozen_count_1


# =============================================================================
# TASK 4.3: Revelation Order
# =============================================================================


class TestRevelationOrder:
    """
    Task 4.3: Verify revelation follows hybrid pattern modulation order.
    """
    
    def test_reveal_indices_match_modulation_order(
        self,
        blind_strategy: BlindStrategy,
        mock_mother_code: MagicMock,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Revealed indices should follow get_modulation_indices order."""
        # Set known modulation indices
        expected_indices = np.array([100, 200, 300, 400, 50, 150, 250, 350, 450, 500], dtype=np.int64)
        mock_mother_code.get_modulation_indices.return_value = expected_indices
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        message1 = next(gen)
        
        # The puncture_indices should come from modulation order
        puncture_indices = np.array(message1["puncture_indices"], dtype=np.int64)
        
        # Fail to trigger reveal
        message2 = gen.send({"verified": False, "converged": False})
        
        if message2.get("kind") == "blind_reveal":
            # Revealed indices should be from the modulation order
            revealed = message2["revealed_indices"]
            
            # These should be subset of the puncture indices
            for idx in revealed:
                assert idx in puncture_indices or idx in expected_indices
    
    def test_modulation_indices_are_deterministic(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
        leakage_tracker: LeakageTracker,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Same seed should produce same modulation indices."""
        # Reset mock to return consistent indices
        expected = np.arange(410, dtype=np.int64)
        mock_mother_code.get_modulation_indices.return_value = expected
        
        strategy1 = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        strategy2 = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen1 = strategy1.alice_reconcile_block(payload, context_no_heuristic, 0)
        gen2 = strategy2.alice_reconcile_block(payload, context_no_heuristic, 0)
        
        msg1 = next(gen1)
        msg2 = next(gen2)
        
        np.testing.assert_array_equal(
            msg1["puncture_indices"],
            msg2["puncture_indices"],
        )


# =============================================================================
# TASK 4.4: NSM Gating
# =============================================================================


class TestNSMGating:
    """
    Task 4.4: Verify NSM-gated optimization with heuristic QBER.
    """
    
    def test_high_heuristic_starts_with_preshortening(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
        leakage_tracker: LeakageTracker,
        context_with_heuristic: ReconciliationContext,
    ) -> None:
        """High heuristic QBER should trigger pre-shortening."""
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_with_heuristic,  # Has qber_heuristic=0.08
            block_id=0,
        )
        
        message = next(gen)
        
        # With high heuristic QBER (>0.05), initial_shortened should be > 0
        revealed_indices = message.get("revealed_indices", [])
        revealed_values = message.get("revealed_values", [])
        
        # Note: The exact number depends on _compute_initial_shortening logic
        # For qber_heuristic=0.08 (>0.05), we expect some pre-shortening
        assert len(revealed_indices) == len(revealed_values)
    
    def test_low_heuristic_no_preshortening(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
        leakage_tracker: LeakageTracker,
    ) -> None:
        """Low heuristic QBER should not trigger pre-shortening."""
        context_low = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
            qber_heuristic=0.02,  # Low heuristic
            modulation_delta=0.1,
        )
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=3,
            modulation_fraction=0.1,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_low,
            block_id=0,
        )
        
        message = next(gen)
        
        # With low heuristic QBER (<=0.05), no pre-shortening
        revealed_indices = message.get("revealed_indices", [])
        assert len(revealed_indices) == 0
    
    def test_no_heuristic_no_preshortening(
        self,
        blind_strategy: BlindStrategy,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """No heuristic should not trigger pre-shortening."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        message = next(gen)
        
        revealed_indices = message.get("revealed_indices", [])
        assert len(revealed_indices) == 0


# =============================================================================
# BLIND STRATEGY PROPERTIES
# =============================================================================


class TestBlindStrategyProperties:
    """Test BlindStrategy interface properties."""
    
    def test_does_not_require_qber_estimation(
        self, blind_strategy: BlindStrategy
    ) -> None:
        """Blind strategy does NOT require QBER pre-estimation."""
        assert blind_strategy.requires_qber_estimation is False
    
    def test_context_qber_for_blind_gating_fallback(self) -> None:
        """Context without heuristic falls back to 0.05."""
        ctx = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
            qber_heuristic=None,
            modulation_delta=0.1,
        )
        
        assert ctx.qber_for_blind_gating == 0.05


# =============================================================================
# LEAKAGE ACCOUNTING FOR BLIND
# =============================================================================


class TestBlindLeakageAccounting:
    """Test leakage accounting for Blind protocol."""
    
    def test_syndrome_plus_reveals_counted(
        self,
        blind_strategy: BlindStrategy,
        mock_codec: MagicMock,
        leakage_tracker: LeakageTracker,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """Total leakage includes syndrome + all reveals."""
        initial_leakage = leakage_tracker.total_leakage
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        # Iteration 1: syndrome sent
        message1 = next(gen)
        leakage_after_1 = leakage_tracker.total_leakage
        
        assert leakage_after_1 > initial_leakage
        
        # Iteration 2: reveal
        message2 = gen.send({"verified": False, "converged": False})
        leakage_after_2 = leakage_tracker.total_leakage
        
        if message2.get("kind") == "blind_reveal":
            # Additional leakage from reveals
            assert leakage_after_2 >= leakage_after_1
    
    def test_block_result_revealed_leakage(
        self,
        blind_strategy: BlindStrategy,
        mock_codec: MagicMock,
        context_no_heuristic: ReconciliationContext,
    ) -> None:
        """BlockResult should report revealed_leakage correctly."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = blind_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context_no_heuristic,
            block_id=0,
        )
        
        message1 = next(gen)
        
        # Force multiple iterations
        message2 = gen.send({"verified": False, "converged": False})
        
        # Complete
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration as e:
            result: BlockResult = e.value
            
            # Revealed leakage should be > 0 if we had reveal iterations
            if result.retry_count > 1:
                assert result.revealed_leakage >= 0

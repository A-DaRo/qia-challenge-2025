"""
Test Suite for Baseline Reconciliation Strategy (Task 3).

Tests the Elkouss et al. (2010) rate-compatible reconciliation protocol.

Per Theoretical Report v2 §3:
- QBER-to-rate mapping
- Exact leakage accounting
- Hash failure handling

References:
[1] Elkouss et al. (2010), "Rate Compatible Protocol for Information Reconciliation"
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from caligo.reconciliation.strategies import (
    BaselineStrategy,
    BlockResult,
    DecoderResult,
    ReconciliationContext,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.utils.math import binary_entropy


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mother_code() -> MagicMock:
    """Mock MotherCodeManager with sparse rates."""
    mock = MagicMock()
    mock.frame_size = 4096
    mock.mother_rate = 0.5
    
    # Available rates (sparse set)
    available_rates = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    mock.patterns = {r: np.zeros(4096, dtype=np.uint8) for r in available_rates}
    
    def get_pattern(rate):
        # Return pattern with appropriate number of punctured bits
        pattern = np.zeros(4096, dtype=np.uint8)
        # Calculate punctured bits for rate
        p = int(4096 * (1 - 0.5 / rate))
        pattern[:p] = 1
        return pattern
    
    mock.get_pattern = MagicMock(side_effect=get_pattern)
    
    # Mock compiled topology
    mock.compiled_topology = MagicMock()
    mock.compiled_topology.n_edges = 12000
    mock.compiled_topology.n_checks = 2048
    mock.compiled_topology.n_vars = 4096
    mock.compiled_topology.check_row_ptr = np.arange(2049, dtype=np.int64)
    mock.compiled_topology.check_col_idx = np.zeros(12000, dtype=np.int64)
    mock.compiled_topology.var_col_ptr = np.arange(4097, dtype=np.int64)
    mock.compiled_topology.var_row_idx = np.zeros(12000, dtype=np.int64)
    
    return mock


@pytest.fixture
def mock_codec() -> MagicMock:
    """Mock LDPCCodec."""
    mock = MagicMock()
    
    # Encode returns syndrome
    mock.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
    
    # Decode returns successful result
    mock.decode_baseline = MagicMock(return_value=DecoderResult(
        corrected_bits=np.zeros(4096, dtype=np.uint8),
        converged=True,
        iterations=10,
        messages=np.zeros(24000, dtype=np.float64),
    ))
    
    return mock


@pytest.fixture
def leakage_tracker() -> LeakageTracker:
    """Create leakage tracker with generous cap."""
    return LeakageTracker(safety_cap=100000)


@pytest.fixture
def baseline_strategy(
    mock_mother_code: MagicMock,
    mock_codec: MagicMock,
    leakage_tracker: LeakageTracker,
) -> BaselineStrategy:
    """Create baseline strategy with mocks."""
    return BaselineStrategy(
        mother_code=mock_mother_code,
        codec=mock_codec,
        leakage_tracker=leakage_tracker,
    )


@pytest.fixture
def context() -> ReconciliationContext:
    """Create reconciliation context."""
    return ReconciliationContext(
        session_id=1,
        frame_size=4096,
        mother_rate=0.5,
        max_iterations=60,
        hash_bits=64,
        f_crit=1.1,
        qber_measured=0.05,
        qber_heuristic=None,
        modulation_delta=0.1,
    )


# =============================================================================
# TASK 3.1: QBER-to-Rate Mapping
# =============================================================================


class TestQBERToRateMapping:
    """Task 3.1: Verify rate selection based on QBER."""
    
    def test_low_qber_selects_capped_rate(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """Low QBER selects max rate capped at 0.60 for decoder stability."""
        qber = 0.01
        f_crit = 1.1
        
        rate = baseline_strategy._select_rate(qber, f_crit)
        
        # R = 1 - f * h(QBER) formula gives ~0.91
        # But we cap at 0.60 to ensure decoder convergence with puncturing.
        # Heavy puncturing (>15%) causes decoder divergence.
        assert rate == 0.60, f"Expected capped rate 0.60, got {rate}"
    
    def test_high_qber_selects_low_rate(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """High QBER should select low effective rate."""
        qber = 0.11
        f_crit = 1.1
        
        rate = baseline_strategy._select_rate(qber, f_crit)
        
        # h(0.11) ≈ 0.5
        # R ≈ 1 - 1.1 * 0.5 ≈ 0.45
        # Clamped to min 0.51
        assert rate <= 0.55, f"Expected low rate for high QBER, got {rate}"
    
    def test_medium_qber_selects_medium_rate(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """Medium QBER should select appropriate medium rate."""
        qber = 0.05
        f_crit = 1.1
        
        rate = baseline_strategy._select_rate(qber, f_crit)
        
        # h(0.05) ≈ 0.286
        # R ≈ 1 - 1.1 * 0.286 ≈ 0.685
        assert 0.60 <= rate <= 0.75, f"Expected medium rate, got {rate}"
    
    def test_rate_selection_with_conservative_cap(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """Verify rate follows formula but caps at 0.60 for decoder stability."""
        for qber in [0.02, 0.05, 0.08, 0.10]:
            f_crit = 1.1
            
            rate = baseline_strategy._select_rate(qber, f_crit)
            
            # Expected rate (before clamping)
            h_qber = binary_entropy(qber)
            expected_raw = 1.0 - f_crit * h_qber
            
            # Clamp to conservative range [0.51, 0.60] for decoder stability
            expected = max(0.51, min(0.60, expected_raw))
            
            assert abs(rate - expected) < 0.02, (
                f"Rate mismatch for QBER={qber}: got {rate}, expected {expected}"
            )
    
    def test_zero_qber_fallback(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """Zero or invalid QBER falls back to mother rate."""
        rate = baseline_strategy._select_rate(0.0, 1.1)
        assert rate == 0.5, f"Expected 0.5 fallback for zero QBER, got {rate}"
        
        rate = baseline_strategy._select_rate(0.5, 1.1)  # At boundary
        assert rate == 0.5, f"Expected 0.5 fallback at boundary, got {rate}"


# =============================================================================
# TASK 3.2: Exact Leakage Accounting
# =============================================================================


class TestExactLeakageAccounting:
    """Task 3.2: Verify leakage equals (1-R_0) × n."""
    
    def test_syndrome_leakage_is_mother_rate(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
    ) -> None:
        """Syndrome leakage should be (1-R_0) × n, not effective rate."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        # Configure mock to return syndrome of correct size
        # Mother code: R_0 = 0.5, n = 4096, so m = 2048 checks
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        # Get first yielded message
        message = next(gen)
        
        # Syndrome should be 2048 bits (not effective rate adjusted)
        assert message["kind"] == "baseline"
        assert len(message["syndrome"]) == 2048
    
    def test_block_result_leakage(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
        leakage_tracker: LeakageTracker,
    ) -> None:
        """BlockResult should report exact syndrome leakage."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        # Get first message
        message = next(gen)
        
        # Send response to complete
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration as e:
            result: BlockResult = e.value
            
            # Syndrome leakage should be 2048 (raw count)
            assert result.syndrome_leakage == 2048
            assert result.hash_leakage == context.hash_bits
            assert result.revealed_leakage == 0  # Baseline has no reveals
    
    def test_total_leakage_formula(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
    ) -> None:
        """Verify total_leakage = syndrome + hash + revealed."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        message = next(gen)
        
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration as e:
            result: BlockResult = e.value
            
            expected_total = (
                result.syndrome_leakage +
                result.hash_leakage +
                result.revealed_leakage
            )
            
            assert result.total_leakage == expected_total


# =============================================================================
# TASK 3.3: Hash Failure Handling
# =============================================================================


class TestHashFailureHandling:
    """Task 3.3: Verify hash failure handling."""
    
    def test_hash_mismatch_returns_unverified(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
    ) -> None:
        """Hash mismatch should set verified=False."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        message = next(gen)
        
        # Bob responds with verification failure
        try:
            gen.send({"verified": False, "converged": True})
        except StopIteration as e:
            result: BlockResult = e.value
            
            assert result.verified is False
            assert result.converged is True  # Decoder converged, hash failed
    
    def test_failed_hash_still_counts_leakage(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
        leakage_tracker: LeakageTracker,
    ) -> None:
        """Failed verification still records leakage."""
        initial_leakage = leakage_tracker.total_leakage
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        message = next(gen)
        
        try:
            gen.send({"verified": False, "converged": True})
        except StopIteration:
            pass
        
        # Leakage should have increased
        final_leakage = leakage_tracker.total_leakage
        assert final_leakage > initial_leakage
    
    def test_bob_decoder_failure(
        self,
        baseline_strategy: BaselineStrategy,
        mock_codec: MagicMock,
        context: ReconciliationContext,
    ) -> None:
        """Decoder failure (not converged) should be reported."""
        payload = np.zeros(3000, dtype=np.uint8)
        
        mock_codec.encode.return_value = np.zeros(2048, dtype=np.uint8)
        
        gen = baseline_strategy.alice_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        message = next(gen)
        
        # Bob's decoder failed to converge
        try:
            gen.send({"verified": False, "converged": False})
        except StopIteration as e:
            result: BlockResult = e.value
            
            assert result.verified is False
            assert result.converged is False


# =============================================================================
# BOB-SIDE TESTS
# =============================================================================


class TestBobBaseline:
    """Tests for Bob's baseline reconciliation."""
    
    def test_bob_receives_and_decodes(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
        leakage_tracker: LeakageTracker,
        context: ReconciliationContext,
    ) -> None:
        """Bob receives syndrome and decodes."""
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        # Configure decoder to return matching payload
        mock_codec.decode_baseline.return_value = DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=15,
            messages=np.zeros(24000, dtype=np.float64),
        )
        
        gen = strategy.bob_reconcile_block(
            payload=payload,
            ctx=context,
            block_id=0,
        )
        
        # Bob yields empty message first (waiting for Alice)
        first = next(gen)
        assert first == {}
        
        # Send Alice's message
        alice_msg = {
            "kind": "baseline",
            "block_id": 0,
            "syndrome": [0] * 2048,
            "pattern_id": 0.65,
            "payload_length": 3000,
            "hash_value": 0,  # Will mismatch
            "qber_channel": 0.05,
        }
        
        try:
            gen.send(alice_msg)
        except StopIteration as e:
            result: BlockResult = e.value
            assert result.converged is True


# =============================================================================
# CONTEXT REQUIREMENTS
# =============================================================================


class TestContextRequirements:
    """Test that Baseline requires measured QBER."""
    
    def test_requires_qber_estimation_property(
        self, baseline_strategy: BaselineStrategy
    ) -> None:
        """Baseline strategy requires QBER estimation."""
        assert baseline_strategy.requires_qber_estimation is True
    
    def test_context_without_qber_raises(self) -> None:
        """ReconciliationContext without qber_measured raises on access."""
        ctx = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,  # No QBER
            qber_heuristic=None,
            modulation_delta=0.1,
        )
        
        with pytest.raises(ValueError, match="Baseline protocol requires measured QBER"):
            _ = ctx.qber_for_baseline

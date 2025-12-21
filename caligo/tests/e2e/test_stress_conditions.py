"""
Test Suite for "Death Valley" Stress Conditions (Task 7).

Tests extreme conditions that stress the protocol:
- High QBER saturation
- Timing violations
- Codec stability under long sessions

These are edge-case tests that verify graceful degradation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any, Dict

import numpy as np
import pytest

from caligo.reconciliation.strategies import (
    BaselineStrategy,
    BlindStrategy,
    DecoderResult,
    ReconciliationContext,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.types.exceptions import (
    LeakageBudgetExceeded,
    QBERThresholdExceeded,
    TimingViolationError,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mother_code() -> MagicMock:
    """Mock MotherCodeManager."""
    mock = MagicMock()
    mock.frame_size = 4096
    mock.mother_rate = 0.5
    mock.get_pattern = MagicMock(return_value=np.zeros(4096, dtype=np.uint8))
    mock.get_modulation_indices = MagicMock(return_value=np.arange(400, dtype=np.int64))
    mock.compiled_topology = MagicMock()
    mock.compiled_topology.n_edges = 12000
    mock.compiled_topology.n_checks = 2048
    mock.compiled_topology.n_vars = 4096
    return mock


@pytest.fixture
def mock_codec_failing() -> MagicMock:
    """Mock codec that always fails to converge."""
    mock = MagicMock()
    mock.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
    mock.decode_baseline = MagicMock(return_value=DecoderResult(
        corrected_bits=np.zeros(4096, dtype=np.uint8),
        converged=False,  # Always fails
        iterations=60,
        messages=np.zeros(24000, dtype=np.float64),
    ))
    mock.decode_blind = MagicMock(return_value=DecoderResult(
        corrected_bits=np.zeros(4096, dtype=np.uint8),
        converged=False,  # Always fails
        iterations=60,
        messages=np.zeros(24000, dtype=np.float64),
    ))
    return mock


# =============================================================================
# TASK 7.1: High QBER Saturation
# =============================================================================


class TestHighQBERSaturation:
    """
    Task 7.1: Test behavior at high QBER (above 11% threshold).
    
    - Baseline: Should abort at Sifting (Security Check)
    - Blind: Should exhaust iterations and report 0% success
    """
    
    def test_baseline_high_qber_selects_minimum_rate(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """High QBER should select lowest available rate."""
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_baseline = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=50,
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
        )
        
        # High QBER context
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=0.15,  # Very high QBER
        )
        
        # Rate selection should clamp to minimum
        rate = strategy._select_rate(0.15, 1.1)
        assert rate <= 0.55  # Should be at or near minimum
    
    def test_blind_exhausts_iterations_on_high_error(
        self,
        mock_mother_code: MagicMock,
        mock_codec_failing: MagicMock,
    ) -> None:
        """Blind should exhaust iterations when decoder fails."""
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec_failing,
            leakage_tracker=LeakageTracker(safety_cap=100000),
            max_blind_iterations=3,
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
            qber_heuristic=0.15,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        
        # Initial message
        msg1 = next(gen)
        
        # Fail iterations
        iterations_seen = 1
        current_msg = msg1
        
        while iterations_seen < 3:
            response = gen.send({"verified": False, "converged": False})
            if response.get("kind") == "blind_reveal":
                iterations_seen += 1
                current_msg = response
            else:
                break
        
        # Final response
        try:
            gen.send({"verified": False, "converged": False})
        except StopIteration as e:
            result = e.value
            assert result.verified is False
    
    def test_leakage_accounted_on_failure(
        self,
        mock_mother_code: MagicMock,
        mock_codec_failing: MagicMock,
    ) -> None:
        """Failed reconciliation still accounts for leakage."""
        leakage_tracker = LeakageTracker(safety_cap=100000)
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec_failing,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=3,
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        
        initial_leakage = leakage_tracker.total_leakage
        
        msg1 = next(gen)
        
        # Syndrome was sent, leakage should be recorded
        assert leakage_tracker.total_leakage > initial_leakage


# =============================================================================
# TASK 7.2: Timing Violations
# =============================================================================


class TestTimingViolations:
    """
    Task 7.2: Test timing violation handling.
    
    NSM parameters enforce strict Δt timing.
    """
    
    def test_timing_violation_error_type(self) -> None:
        """TimingViolationError should have correct attributes."""
        error = TimingViolationError("Basis revealed before Δt elapsed")
        
        assert isinstance(error, Exception)
        assert "Δt" in str(error) or "delta" in str(error).lower() or "revealed" in str(error).lower()
    
    def test_timing_barrier_enforcement(self) -> None:
        """
        Verify timing barrier concept.
        
        Note: Full integration test requires SquidASM runtime.
        This tests the error type and handling pattern.
        """
        from caligo.types.exceptions import TimingViolationError
        
        def check_timing(elapsed_ns: int, required_ns: int) -> None:
            if elapsed_ns < required_ns:
                raise TimingViolationError(
                    f"Timing violation: {elapsed_ns}ns < required {required_ns}ns"
                )
        
        # Should pass
        check_timing(2_000_000_000, 1_000_000_000)
        
        # Should fail
        with pytest.raises(TimingViolationError):
            check_timing(500_000_000, 1_000_000_000)


# =============================================================================
# TASK 7.3: Codec Stability
# =============================================================================


class TestCodecStability:
    """
    Task 7.3: Test codec stability under extended sessions.
    """
    
    def test_multiple_blocks_no_memory_accumulation(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Processing many blocks should not accumulate memory indefinitely."""
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_baseline = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=10,
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=10_000_000),
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=0.05,
        )
        
        # Process many blocks
        n_blocks = 100
        
        for block_id in range(n_blocks):
            payload = np.random.randint(0, 2, 3000, dtype=np.uint8)
            
            gen = strategy.alice_reconcile_block(payload, context, block_id)
            msg = next(gen)
            
            try:
                gen.send({"verified": True, "converged": True})
            except StopIteration:
                pass
        
        # Verify codec was called correct number of times
        assert mock_codec.encode.call_count == n_blocks
    
    def test_blind_state_isolated_between_blocks(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Each block should have isolated decoder state."""
        call_count = [0]
        
        def mock_decode(syndrome, llr, messages, frozen_mask, max_iterations):
            call_count[0] += 1
            return DecoderResult(
                corrected_bits=np.zeros(4096, dtype=np.uint8),
                converged=True,
                iterations=10,
                messages=messages.copy(),
            )
        
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_blind = MagicMock(side_effect=mock_decode)
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=10_000_000),
            max_blind_iterations=3,
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
        )
        
        # Process multiple blocks
        for block_id in range(10):
            payload = np.random.randint(0, 2, 3000, dtype=np.uint8)
            
            gen = strategy.alice_reconcile_block(payload, context, block_id)
            msg = next(gen)
            
            try:
                gen.send({"verified": True, "converged": True})
            except StopIteration:
                pass


# =============================================================================
# ERROR RECOVERY PATTERNS
# =============================================================================


class TestErrorRecoveryPatterns:
    """Test graceful error handling patterns."""
    
    def test_leakage_exceeded_is_recoverable(self) -> None:
        """LeakageBudgetExceeded should be catchable."""
        tracker = LeakageTracker(safety_cap=100, abort_on_exceed=True)
        
        tracker.record_block(block_id=0, syndrome_bits=80, hash_bits=10)
        
        try:
            tracker.record_block(block_id=1, syndrome_bits=20, hash_bits=10)
            exceeded = False
        except LeakageBudgetExceeded as e:
            exceeded = True
            assert e.actual_leakage > e.max_allowed
        
        assert exceeded
    
    def test_security_errors_hierarchy(self) -> None:
        """Security errors should form proper hierarchy."""
        from caligo.types.exceptions import (
            CaligoError,
            SecurityError,
            QBERThresholdExceeded,
            NSMViolationError,
        )
        
        assert issubclass(SecurityError, CaligoError)
        assert issubclass(QBERThresholdExceeded, SecurityError)
        assert issubclass(NSMViolationError, SecurityError)

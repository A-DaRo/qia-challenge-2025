"""
Unit tests for leakage tracker module.

Tests leakage accumulation, safety cap computation, and abort conditions.
"""

from __future__ import annotations

import pytest

from caligo.reconciliation.leakage_tracker import (
    LeakageRecord,
    LeakageTracker,
    compute_safety_cap,
)
from caligo.reconciliation import constants
from caligo.types.exceptions import LeakageBudgetExceeded


class TestComputeSafetyCap:
    """Tests for safety cap computation."""

    def test_safety_cap_formula(self) -> None:
        """Safety cap follows König et al. formula."""
        # l_max = n * (1 - h(QBER) - ε)
        n_sifted = 10000
        qber = 0.05
        epsilon = 0.01
        
        cap = compute_safety_cap(n_sifted, qber, epsilon)
        
        # Binary entropy of 0.05 ≈ 0.286
        # So cap ≈ 10000 * (1 - 0.286 - 0.01) ≈ 7040
        assert 6800 < cap < 7200

    def test_cap_decreases_with_qber(self) -> None:
        """Higher QBER means lower safety cap."""
        n_sifted = 10000
        
        cap_low = compute_safety_cap(n_sifted, 0.03)
        cap_high = compute_safety_cap(n_sifted, 0.10)
        
        assert cap_high < cap_low

    def test_cap_scales_with_sifted_bits(self) -> None:
        """Safety cap scales linearly with sifted key size."""
        qber = 0.05
        
        cap_small = compute_safety_cap(5000, qber)
        cap_large = compute_safety_cap(10000, qber)
        
        assert abs(cap_large - 2 * cap_small) < 10

    def test_cap_with_epsilon(self) -> None:
        """Epsilon reduces safety cap."""
        n_sifted = 10000
        qber = 0.05
        
        cap_no_eps = compute_safety_cap(n_sifted, qber, epsilon=0.0)
        cap_with_eps = compute_safety_cap(n_sifted, qber, epsilon=0.05)
        
        assert cap_with_eps < cap_no_eps


class TestLeakageRecord:
    """Tests for LeakageRecord dataclass."""

    def test_record_fields(self) -> None:
        """LeakageRecord holds expected fields."""
        record = LeakageRecord(
            block_id=0,
            syndrome_bits=1228,
            hash_bits=50,
            iteration=1,
        )
        
        assert record.block_id == 0
        assert record.syndrome_bits == 1228
        assert record.hash_bits == 50
        assert record.iteration == 1

    def test_total_leakage(self) -> None:
        """Total leakage is syndrome + hash bits."""
        record = LeakageRecord(
            block_id=0,
            syndrome_bits=1228,
            hash_bits=50,
            iteration=1,
        )
        
        assert record.total_leakage == 1278


class TestLeakageTracker:
    """Tests for LeakageTracker class."""

    def test_initial_leakage_zero(self) -> None:
        """New tracker has zero leakage."""
        tracker = LeakageTracker(safety_cap=5000)
        assert tracker.total_leakage == 0

    def test_record_block_accumulates(self) -> None:
        """Recording blocks accumulates leakage."""
        tracker = LeakageTracker(safety_cap=10000)
        
        tracker.record_block(
            block_id=0,
            syndrome_bits=1228,
            hash_bits=50,
        )
        
        assert tracker.total_leakage == 1278
        
        tracker.record_block(
            block_id=1,
            syndrome_bits=1228,
            hash_bits=50,
        )
        
        assert tracker.total_leakage == 2556

    def test_check_safety_under_cap(self) -> None:
        """Safety check passes when under cap."""
        tracker = LeakageTracker(safety_cap=10000)
        
        tracker.record_block(block_id=0, syndrome_bits=1000, hash_bits=50)
        
        assert tracker.check_safety() is True

    def test_check_safety_over_cap(self) -> None:
        """Circuit breaker raises exception when over cap."""
        # Phase 1: Circuit breaker pattern now immediately raises
        tracker = LeakageTracker(safety_cap=1000, abort_on_exceed=True)
        
        tracker.record_block(block_id=0, syndrome_bits=800, hash_bits=50)
        
        # Second block should trigger circuit breaker
        with pytest.raises(LeakageBudgetExceeded) as exc_info:
            tracker.record_block(block_id=1, syndrome_bits=800, hash_bits=50)
        
        assert exc_info.value.actual_leakage == 1700
        assert exc_info.value.max_allowed == 1000

    def test_should_abort_when_exceeded(self) -> None:
        """Circuit breaker raises exception when cap exceeded."""
        # Phase 1: Circuit breaker pattern now immediately raises
        tracker = LeakageTracker(safety_cap=500, abort_on_exceed=True)
        
        # First block is fine
        assert tracker.should_abort() is False
        
        tracker.record_block(block_id=0, syndrome_bits=400, hash_bits=50)
        assert tracker.should_abort() is False
        
        # Second block triggers circuit breaker
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_block(block_id=1, syndrome_bits=400, hash_bits=50)

    def test_remaining_budget(self) -> None:
        """Remaining budget computed correctly."""
        tracker = LeakageTracker(safety_cap=5000)
        
        assert tracker.remaining_budget == 5000
        
        tracker.record_block(block_id=0, syndrome_bits=1000, hash_bits=50)
        
        assert tracker.remaining_budget == 3950

    def test_records_maintained(self) -> None:
        """All records maintained for audit."""
        tracker = LeakageTracker(safety_cap=10000)
        
        for i in range(5):
            tracker.record_block(block_id=i, syndrome_bits=1000, hash_bits=50)
        
        assert len(tracker.records) == 5
        assert tracker.records[3].block_id == 3


class TestLeakageTrackerEdgeCases:
    """Edge case tests for LeakageTracker."""

    def test_zero_cap_always_aborts(self) -> None:
        """Zero safety cap triggers circuit breaker on any leakage."""
        # Phase 1: Circuit breaker immediately raises
        tracker = LeakageTracker(safety_cap=0, abort_on_exceed=True)
        
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_block(block_id=0, syndrome_bits=1, hash_bits=0)

    def test_large_single_block(self) -> None:
        """Large single block triggers circuit breaker."""
        # Phase 1: Circuit breaker immediately raises
        tracker = LeakageTracker(safety_cap=1000, abort_on_exceed=True)
        
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_block(block_id=0, syndrome_bits=1500, hash_bits=50)

    def test_blind_iteration_tracking(self) -> None:
        """Blind iterations add to total leakage."""
        tracker = LeakageTracker(safety_cap=5000)
        
        # First iteration
        tracker.record_block(block_id=0, syndrome_bits=500, hash_bits=50, iteration=1)
        assert tracker.total_leakage == 550
        
        # Second iteration (same block)
        tracker.record_block(block_id=0, syndrome_bits=500, hash_bits=50, iteration=2)
        assert tracker.total_leakage == 1100


# =============================================================================
# TASK 5: Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Task 5: Circuit Breaker pattern tests."""
    
    def test_one_bit_too_many(self) -> None:
        """
        Task 5.1: The "One Bit Too Many" Test.
        
        Set safety_cap=100, record 99 bits, then 2 bits.
        Circuit breaker should raise immediately.
        """
        tracker = LeakageTracker(safety_cap=100, abort_on_exceed=True)
        
        # Record 99 bits - should succeed
        tracker.record_block(block_id=0, syndrome_bits=99, hash_bits=0)
        assert tracker.total_leakage == 99
        assert tracker.remaining_budget == 1
        
        # Record 2 more bits - total=101, exceeds cap=100
        with pytest.raises(LeakageBudgetExceeded) as exc_info:
            tracker.record_block(block_id=1, syndrome_bits=2, hash_bits=0)
        
        assert exc_info.value.actual_leakage == 101
        assert exc_info.value.max_allowed == 100
    
    def test_exactly_at_cap(self) -> None:
        """Recording exactly at cap should NOT raise."""
        tracker = LeakageTracker(safety_cap=100, abort_on_exceed=True)
        
        # Record exactly 100 bits - should succeed
        tracker.record_block(block_id=0, syndrome_bits=100, hash_bits=0)
        assert tracker.total_leakage == 100
        assert tracker.check_safety() is True
        
        # Next record triggers breach
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_block(block_id=1, syndrome_bits=1, hash_bits=0)
    
    def test_blind_iteration_accumulation(self) -> None:
        """
        Task 5.2: Blind Iteration Accumulation.
        
        Record a Blind block across multiple iterations:
        - Iter 1: syndrome
        - Iter 2: 50 reveal bits
        - Iter 3: 50 reveal bits
        Assert total leakage sums correctly.
        """
        tracker = LeakageTracker(safety_cap=10000, abort_on_exceed=True)
        
        # Iteration 1: syndrome + hash
        syndrome_bits = 2048
        hash_bits = 64
        tracker.record_block(
            block_id=0,
            syndrome_bits=syndrome_bits,
            hash_bits=hash_bits,
            iteration=1,
        )
        
        expected_after_1 = syndrome_bits + hash_bits
        assert tracker.total_leakage == expected_after_1
        
        # Iteration 2: reveal 50 bits
        tracker.record_reveal(block_id=0, iteration=2, revealed_bits=50)
        expected_after_2 = expected_after_1 + 50
        assert tracker.total_leakage == expected_after_2
        
        # Iteration 3: reveal 50 more bits
        tracker.record_reveal(block_id=0, iteration=3, revealed_bits=50)
        expected_after_3 = expected_after_2 + 50
        assert tracker.total_leakage == expected_after_3
        
        # Verify total
        assert tracker.total_leakage == syndrome_bits + hash_bits + 100
    
    def test_reveal_triggers_circuit_breaker(self) -> None:
        """Reveal bits can trigger circuit breaker."""
        tracker = LeakageTracker(safety_cap=2200, abort_on_exceed=True)
        
        # Initial syndrome + hash
        tracker.record_block(block_id=0, syndrome_bits=2048, hash_bits=64)
        assert tracker.total_leakage == 2112
        
        # First reveal ok
        tracker.record_reveal(block_id=0, iteration=2, revealed_bits=50)
        assert tracker.total_leakage == 2162
        
        # Second reveal exceeds cap
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_reveal(block_id=0, iteration=3, revealed_bits=50)
    
    def test_abort_on_exceed_false_logs_only(self) -> None:
        """With abort_on_exceed=False, no exception is raised."""
        tracker = LeakageTracker(safety_cap=100, abort_on_exceed=False)
        
        # Exceed cap - should not raise
        tracker.record_block(block_id=0, syndrome_bits=200, hash_bits=0)
        
        assert tracker.total_leakage == 200
        assert tracker.check_safety() is False
        assert tracker.should_abort() is True
    
    def test_multiple_blocks_accumulate(self) -> None:
        """Multiple blocks accumulate correctly toward cap."""
        tracker = LeakageTracker(safety_cap=5000, abort_on_exceed=True)
        
        for i in range(4):
            tracker.record_block(block_id=i, syndrome_bits=1000, hash_bits=50)
        
        assert tracker.total_leakage == 4200  # 4 * (1000 + 50)
        assert tracker.num_blocks == 4
        
        # 5th block exceeds
        with pytest.raises(LeakageBudgetExceeded):
            tracker.record_block(block_id=4, syndrome_bits=1000, hash_bits=50)
    
    def test_shortening_bits_counted(self) -> None:
        """Shortening leakage is included in total."""
        tracker = LeakageTracker(safety_cap=5000, abort_on_exceed=True)
        
        # Record with shortening
        tracker.record_block(
            block_id=0,
            syndrome_bits=1000,
            hash_bits=50,
            n_shortened=100,
            frame_size=4096,
        )
        
        # Total should be > 1050 (includes shortening bits)
        assert tracker.total_leakage > 1050
    
    def test_remaining_budget_negative_when_exceeded(self) -> None:
        """Remaining budget goes negative when exceeded."""
        tracker = LeakageTracker(safety_cap=100, abort_on_exceed=False)
        
        tracker.record_block(block_id=0, syndrome_bits=150, hash_bits=0)
        
        assert tracker.remaining_budget == -50

"""
Unit tests for Sprint 2 Leakage Management.

Tests LeakageSafetyManager, BlockReconciliationReport, and leakage budget
computation per sprint_2_specification.md Section 4.
"""

import math
import pytest

from ehok.protocols.leakage_manager import (
    # Abort codes
    ABORT_CODE_LEAKAGE_CAP_EXCEEDED,
    # Dataclasses
    BlockReconciliationReport,
    LeakageState,
    # Classes
    LeakageSafetyManager,
    # Functions
    compute_max_leakage_budget,
)


# =============================================================================
# BlockReconciliationReport Tests
# =============================================================================


class TestBlockReconciliationReport:
    """Tests for reconciliation block reports."""

    def test_create_valid_report(self) -> None:
        """Valid report should be created successfully."""
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=128,
            hash_bits=32,
            decode_converged=True,
            hash_verified=True,
            iterations=5,
        )

        assert report.block_index == 0
        assert report.syndrome_bits == 128
        assert report.hash_bits == 32
        assert report.iterations == 5
        assert report.decode_converged is True
        assert report.hash_verified is True

    def test_total_leaked_bits(self) -> None:
        """Total leaked should be syndrome + hash."""
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=100,
            hash_bits=20,
            decode_converged=True,
            hash_verified=True,
            iterations=3,
        )

        assert report.total_leakage_bits == 120

    def test_zero_values_allowed(self) -> None:
        """Zero syndrome/hash bits should be allowed."""
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=0,
            hash_bits=0,
            decode_converged=False,
            hash_verified=False,
            iterations=0,
        )

        assert report.total_leakage_bits == 0


# =============================================================================
# compute_max_leakage_budget Tests
# =============================================================================


class TestComputeMaxLeakageBudget:
    """Tests for maximum leakage budget computation."""

    def test_basic_computation(self) -> None:
        """Budget should be computed correctly."""
        result = compute_max_leakage_budget(
            n_sifted_bits=100000,
            min_entropy_rate=0.5,
            epsilon_sec=1e-10,
        )

        # Budget should be positive and reasonable
        assert result > 0
        assert result < 100000

    def test_higher_entropy_increases_budget(self) -> None:
        """Higher min-entropy should increase available budget."""
        budget_low = compute_max_leakage_budget(
            n_sifted_bits=100000,
            min_entropy_rate=0.3,
            epsilon_sec=1e-10,
        )

        budget_high = compute_max_leakage_budget(
            n_sifted_bits=100000,
            min_entropy_rate=0.6,
            epsilon_sec=1e-10,
        )

        assert budget_high > budget_low

    def test_larger_key_increases_budget(self) -> None:
        """Larger sifted key should increase budget."""
        budget_small = compute_max_leakage_budget(
            n_sifted_bits=50000,
            min_entropy_rate=0.5,
            epsilon_sec=1e-10,
        )

        budget_large = compute_max_leakage_budget(
            n_sifted_bits=100000,
            min_entropy_rate=0.5,
            epsilon_sec=1e-10,
        )

        assert budget_large > budget_small

    def test_invalid_n_raises(self) -> None:
        """n <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_sifted_bits"):
            compute_max_leakage_budget(
                n_sifted_bits=0,
                min_entropy_rate=0.5,
                epsilon_sec=1e-10,
            )

    def test_invalid_entropy_rate_raises(self) -> None:
        """min_entropy_rate outside (0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="min_entropy_rate"):
            compute_max_leakage_budget(
                n_sifted_bits=100000,
                min_entropy_rate=0.0,
                epsilon_sec=1e-10,
            )

    def test_invalid_epsilon_raises(self) -> None:
        """Epsilon outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon_sec"):
            compute_max_leakage_budget(
                n_sifted_bits=100000,
                min_entropy_rate=0.5,
                epsilon_sec=0.0,
            )


# =============================================================================
# LeakageSafetyManager Tests (TASK-LEAKAGE-MGR-001)
# =============================================================================


class TestLeakageSafetyManager:
    """Tests for leakage tracking and budget enforcement."""

    @pytest.fixture
    def manager(self) -> LeakageSafetyManager:
        """Create manager with typical budget."""
        return LeakageSafetyManager(max_leakage_bits=10000)

    def test_init_valid_budget(self) -> None:
        """Manager initializes with valid budget."""
        manager = LeakageSafetyManager(max_leakage_bits=50000)

        assert manager.max_leakage_bits == 50000
        assert manager.total_syndrome_bits == 0
        assert manager.total_hash_bits == 0
        assert manager.wiretap_cost_bits == 0
        assert manager.num_blocks_processed == 0

    def test_init_invalid_budget(self) -> None:
        """Non-positive budget should raise."""
        with pytest.raises(ValueError, match="max_leakage_bits"):
            LeakageSafetyManager(max_leakage_bits=0)

    def test_get_summary_initial(self, manager: LeakageSafetyManager) -> None:
        """Initial summary should show zero usage."""
        summary = manager.get_summary()

        assert summary["total_syndrome_bits"] == 0
        assert summary["total_hash_bits"] == 0
        assert summary["wiretap_cost_bits"] == 0
        assert summary["max_leakage_bits"] == 10000
        assert summary["remaining_budget_bits"] == 10000
        assert summary["cap_exceeded"] is False
        assert summary["num_blocks_processed"] == 0

    def test_account_block_updates_tracking(self, manager: LeakageSafetyManager) -> None:
        """Recording block should update tracking."""
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=500,
            hash_bits=100,
            decode_converged=True,
            hash_verified=True,
            iterations=3,
        )

        within_budget = manager.account_block(report)

        assert within_budget is True

        assert manager.total_syndrome_bits == 500
        assert manager.total_hash_bits == 100
        assert manager.wiretap_cost_bits == 600
        assert manager.num_blocks_processed == 1

    def test_account_multiple_blocks(self, manager: LeakageSafetyManager) -> None:
        """Multiple blocks should accumulate."""
        for i in range(5):
            report = BlockReconciliationReport(
                block_index=i,
                syndrome_bits=100,
                hash_bits=20,
                decode_converged=True,
                hash_verified=True,
                iterations=2,
            )
            manager.account_block(report)

        assert manager.total_syndrome_bits == 500
        assert manager.total_hash_bits == 100
        assert manager.wiretap_cost_bits == 600
        assert manager.num_blocks_processed == 5

    def test_wiretap_cost_formula(self, manager: LeakageSafetyManager) -> None:
        """Wiretap cost should be |Σ| = Σ|Sᵢ| + Σ|hᵢ|."""
        blocks = [
            BlockReconciliationReport(0, 100, 10, True, True, 2),
            BlockReconciliationReport(1, 150, 15, True, True, 3),
            BlockReconciliationReport(2, 80, 8, True, True, 2),
        ]

        for block in blocks:
            manager.account_block(block)

        # Manual calculation
        expected_syndrome = 100 + 150 + 80
        expected_hash = 10 + 15 + 8
        expected_wiretap = expected_syndrome + expected_hash

        assert manager.total_syndrome_bits == expected_syndrome
        assert manager.total_hash_bits == expected_hash
        assert manager.wiretap_cost_bits == expected_wiretap

    def test_cap_exceeded_detection(self, manager: LeakageSafetyManager) -> None:
        """Should detect when cap is exceeded."""
        # Budget is 10000, create block that exceeds it
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=9000,
            hash_bits=2000,  # Total 11000 > 10000
            decode_converged=True,
            hash_verified=True,
            iterations=10,
        )

        within_budget = manager.account_block(report)

        assert within_budget is False
        assert manager.is_cap_exceeded is True
        assert manager.check_abort() == ABORT_CODE_LEAKAGE_CAP_EXCEEDED

    def test_cap_exceeded_cumulative(self, manager: LeakageSafetyManager) -> None:
        """Cap should be checked cumulatively."""
        # First block: 6000 bits (under cap)
        report1 = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=5000,
            hash_bits=1000,
            decode_converged=True,
            hash_verified=True,
            iterations=5,
        )
        within1 = manager.account_block(report1)
        assert within1 is True

        # Second block: 5000 more bits (total 11000 > 10000)
        report2 = BlockReconciliationReport(
            block_index=1,
            syndrome_bits=4000,
            hash_bits=1000,
            decode_converged=True,
            hash_verified=True,
            iterations=5,
        )
        within2 = manager.account_block(report2)
        assert within2 is False

    def test_remaining_budget_calculation(self, manager: LeakageSafetyManager) -> None:
        """Remaining budget should be correctly calculated."""
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=3000,
            hash_bits=500,
            decode_converged=True,
            hash_verified=True,
            iterations=3,
        )
        manager.account_block(report)

        assert manager.remaining_budget_bits == 10000 - 3500

    def test_remaining_budget_floor_when_exceeded(self) -> None:
        """Remaining budget floors at 0 when exceeded."""
        manager = LeakageSafetyManager(max_leakage_bits=1000)

        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=1500,
            hash_bits=200,
            decode_converged=True,
            hash_verified=True,
            iterations=5,
        )
        manager.account_block(report)

        # remaining_budget_bits uses max(0, remaining)
        assert manager.remaining_budget_bits == 0
        assert manager.is_cap_exceeded is True

    def test_account_syndrome_only(self, manager: LeakageSafetyManager) -> None:
        """Should allow accounting syndrome separately."""
        within = manager.account_syndrome(block_index=0, syndrome_bits=5000)

        assert within is True
        assert manager.total_syndrome_bits == 5000
        assert manager.total_hash_bits == 0

    def test_account_hash_only(self, manager: LeakageSafetyManager) -> None:
        """Should allow accounting hash separately."""
        within = manager.account_hash(block_index=0, hash_bits=500)

        assert within is True
        assert manager.total_hash_bits == 500
        assert manager.total_syndrome_bits == 0

    def test_reset_clears_state(self, manager: LeakageSafetyManager) -> None:
        """Reset should clear all tracking."""
        # Add some leakage
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=5000,
            hash_bits=1000,
            decode_converged=True,
            hash_verified=True,
            iterations=5,
        )
        manager.account_block(report)

        # Reset
        manager.reset()

        assert manager.total_syndrome_bits == 0
        assert manager.total_hash_bits == 0
        assert manager.wiretap_cost_bits == 0
        assert manager.num_blocks_processed == 0
        assert manager.remaining_budget_bits == 10000

    def test_get_block_reports_history(self, manager: LeakageSafetyManager) -> None:
        """Should track all recorded reports."""
        reports = [
            BlockReconciliationReport(0, 100, 10, True, True, 2),
            BlockReconciliationReport(1, 200, 20, True, True, 3),
            BlockReconciliationReport(2, 150, 15, False, False, 2),
        ]

        for report in reports:
            manager.account_block(report)

        history = manager.block_reports

        assert len(history) == 3
        assert history[0].block_index == 0
        assert history[1].block_index == 1
        assert history[2].decode_converged is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestLeakageManagementIntegration:
    """Integration tests for leakage management in reconciliation."""

    def test_realistic_reconciliation_scenario(self) -> None:
        """Simulate realistic LDPC reconciliation leakage."""
        # Compute budget based on parameters
        budget = compute_max_leakage_budget(
            n_sifted_bits=100000,
            min_entropy_rate=0.5,
            epsilon_sec=1e-10,
        )

        # Create manager with computed budget
        manager = LeakageSafetyManager(max_leakage_bits=budget)

        # Simulate 10 reconciliation blocks
        # Each block: 1000 bit block, syndrome ~100 bits, hash ~32 bits
        for i in range(10):
            report = BlockReconciliationReport(
                block_index=i,
                syndrome_bits=100,
                hash_bits=32,
                decode_converged=True,
                hash_verified=True,
                iterations=5,
            )
            within_budget = manager.account_block(report)

            # Should not exceed budget with reasonable parameters
            assert within_budget is True

    def test_degraded_channel_exceeds_budget(self) -> None:
        """Small budget should lead to budget exhaustion."""
        # Use a small budget to trigger exhaustion quickly
        manager = LeakageSafetyManager(max_leakage_bits=5000)

        # Try to reconcile many blocks with high syndrome cost
        exceeded = False
        for i in range(100):
            report = BlockReconciliationReport(
                block_index=i,
                syndrome_bits=500,  # High syndrome cost
                hash_bits=64,
                decode_converged=True,
                hash_verified=True,
                iterations=20,
            )
            within_budget = manager.account_block(report)
            if not within_budget:
                exceeded = True
                break

        # Should eventually exceed budget
        assert exceeded is True

    def test_precheck_via_budget_tracking(self) -> None:
        """Budget tracking helps avoid mid-protocol abort."""
        manager = LeakageSafetyManager(max_leakage_bits=5000)

        # Estimate leakage for next block
        estimated_syndrome = 800
        estimated_hash = 64
        estimated_total = estimated_syndrome + estimated_hash

        # Check if budget allows
        if manager.remaining_budget_bits >= estimated_total:
            # Safe to proceed
            report = BlockReconciliationReport(
                block_index=0,
                syndrome_bits=estimated_syndrome,
                hash_bits=estimated_hash,
                decode_converged=True,
                hash_verified=True,
                iterations=5,
            )
            within_budget = manager.account_block(report)
            assert within_budget is True
        else:
            # Would exceed - don't even try
            pass

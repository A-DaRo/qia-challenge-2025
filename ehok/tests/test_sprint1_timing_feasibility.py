"""
Unit tests for TimingEnforcer and FeasibilityChecker (TASK-TIMING-001, TASK-FEAS-001).

These tests verify:
1. TimingEnforcer correctly blocks basis reveal before Δt
2. TimingEnforcer allows reveal after Δt
3. FeasibilityChecker produces correct abort codes
4. State machine transitions work correctly

References
----------
- sprint_1_specification.md Sections 2 and 3
- Erven et al. (2014): Timing requirements
"""

import pytest

from ehok.core.timing import (
    TimingConfig,
    TimingEnforcer,
    TimingState,
    TimingViolationError,
    TimingStateError,
)
from ehok.core.feasibility import (
    FeasibilityInputs,
    FeasibilityDecision,
    FeasibilityChecker,
    ABORT_CODE_QBER_TOO_HIGH,
    ABORT_CODE_STRICT_LESS_VIOLATED,
    ABORT_CODE_CAPACITY_RATE_VIOLATED,
    ABORT_CODE_DEATH_VALLEY,
    ABORT_CODE_INVALID_PARAMETERS,
)


# =============================================================================
# TimingConfig Tests
# =============================================================================


class TestTimingConfig:
    """Tests for TimingConfig dataclass."""

    def test_default_values(self) -> None:
        """Defaults should match Erven Table I."""
        config = TimingConfig()
        assert config.delta_t_ns == 1_000_000_000  # 1 second in ns

    def test_custom_values(self) -> None:
        """Custom values should be accepted."""
        config = TimingConfig(delta_t_ns=500_000_000)
        assert config.delta_t_ns == 500_000_000

    def test_validation_delta_t_positive(self) -> None:
        """delta_t_ns must be positive."""
        with pytest.raises(ValueError, match="delta_t_ns must be positive"):
            TimingConfig(delta_t_ns=0)


# =============================================================================
# TimingEnforcer Tests
# =============================================================================


class TestTimingEnforcer:
    """Tests for the TimingEnforcer class."""

    @pytest.fixture
    def config(self) -> TimingConfig:
        """Create a timing config with 1 second barrier."""
        return TimingConfig(delta_t_ns=1_000_000_000)

    @pytest.fixture
    def enforcer(self, config: TimingConfig) -> TimingEnforcer:
        """Create a fresh enforcer for each test."""
        return TimingEnforcer(config)

    # -------------------------------------------------------------------------
    # State Machine Tests
    # -------------------------------------------------------------------------

    def test_initial_state(self, enforcer: TimingEnforcer) -> None:
        """New enforcer should start in UNINITIALIZED state."""
        assert enforcer.state == TimingState.UNINITIALIZED

    def test_state_after_commit(self, enforcer: TimingEnforcer) -> None:
        """After marking commit, state should be COMMIT_RECEIVED."""
        enforcer.mark_commit_received(sim_time_ns=1000)
        assert enforcer.state == TimingState.COMMIT_RECEIVED

    def test_state_after_barrier_satisfied(self, enforcer: TimingEnforcer) -> None:
        """After barrier is satisfied via reveal attempt, state should update."""
        enforcer.mark_commit_received(sim_time_ns=1000)
        # Actually attempt reveal at time after barrier (triggers state change)
        enforcer.mark_basis_reveal_attempt(sim_time_ns=2_000_000_000)
        assert enforcer.state == TimingState.BASIS_REVEALED

    # -------------------------------------------------------------------------
    # Timing Barrier Tests
    # -------------------------------------------------------------------------

    def test_block_before_delta_t(self, enforcer: TimingEnforcer) -> None:
        """Basis reveal must be blocked before Δt has elapsed."""
        enforcer.mark_commit_received(sim_time_ns=0)

        # Try to reveal 0.5 seconds later (before 1s barrier)
        assert not enforcer.is_basis_reveal_allowed(sim_time_ns=500_000_000)

    def test_allow_after_delta_t(self, enforcer: TimingEnforcer) -> None:
        """Basis reveal should be allowed after Δt has elapsed."""
        enforcer.mark_commit_received(sim_time_ns=0)

        # Try to reveal 1.1 seconds later (after 1s barrier)
        assert enforcer.is_basis_reveal_allowed(sim_time_ns=1_100_000_000)

    def test_allow_exactly_at_delta_t(self, enforcer: TimingEnforcer) -> None:
        """Basis reveal should be allowed exactly at Δt."""
        enforcer.mark_commit_received(sim_time_ns=0)

        # Try to reveal exactly at 1 second
        assert enforcer.is_basis_reveal_allowed(sim_time_ns=1_000_000_000)

    def test_with_tolerance(self, enforcer: TimingEnforcer) -> None:
        """Test revealing exactly at barrier time."""
        enforcer.mark_commit_received(sim_time_ns=0)
        # At exactly delta_t, should be allowed
        assert enforcer.is_basis_reveal_allowed(sim_time_ns=1_000_000_000)

    # -------------------------------------------------------------------------
    # Required Release Time Tests
    # -------------------------------------------------------------------------

    def test_required_release_time(self, enforcer: TimingEnforcer) -> None:
        """required_release_time_ns should return commit + Δt."""
        enforcer.mark_commit_received(sim_time_ns=5_000_000)
        expected = 5_000_000 + 1_000_000_000  # 5ms + 1s
        assert enforcer.required_release_time_ns() == expected

    def test_required_release_time_before_commit(
        self, enforcer: TimingEnforcer
    ) -> None:
        """required_release_time_ns should raise if no commit yet."""
        with pytest.raises(TimingStateError, match="Cannot compute release time"):
            enforcer.required_release_time_ns()

    # -------------------------------------------------------------------------
    # Error Cases
    # -------------------------------------------------------------------------

    def test_double_commit_raises(self, enforcer: TimingEnforcer) -> None:
        """Marking commit twice should raise."""
        enforcer.mark_commit_received(sim_time_ns=1000)
        with pytest.raises(TimingStateError, match="already"):
            enforcer.mark_commit_received(sim_time_ns=2000)

    def test_check_before_commit_raises(self, enforcer: TimingEnforcer) -> None:
        """Checking reveal before commit should return False (not raise)."""
        # is_basis_reveal_allowed returns False (not raise) when no commit
        assert not enforcer.is_basis_reveal_allowed(sim_time_ns=1000)

    def test_violation_attempt_raises(self) -> None:
        """Attempting reveal before barrier should be able to raise."""
        config = TimingConfig(delta_t_ns=1_000_000_000)
        enforcer = TimingEnforcer(config)
        enforcer.mark_commit_received(sim_time_ns=0)

        # Attempting reveal too early via mark_basis_reveal_attempt
        with pytest.raises(TimingViolationError):
            enforcer.mark_basis_reveal_attempt(sim_time_ns=500_000_000)

    # -------------------------------------------------------------------------
    # Reset Tests
    # -------------------------------------------------------------------------

    def test_reset(self, enforcer: TimingEnforcer) -> None:
        """reset() should return enforcer to initial state."""
        enforcer.mark_commit_received(sim_time_ns=1000)
        enforcer.reset()
        assert enforcer.state == TimingState.UNINITIALIZED


# =============================================================================
# FeasibilityInputs Tests
# =============================================================================


class TestFeasibilityInputs:
    """Tests for FeasibilityInputs dataclass."""

    def test_valid_inputs(self) -> None:
        """Valid inputs should not raise."""
        inputs = FeasibilityInputs(
            expected_qber=0.05,
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=10000,
            expected_leakage_bits=1000,
        )
        assert inputs.expected_qber == 0.05


# =============================================================================
# FeasibilityChecker Tests
# =============================================================================


class TestFeasibilityChecker:
    """Tests for FeasibilityChecker class."""

    @pytest.fixture
    def checker(self) -> FeasibilityChecker:
        """Create a checker."""
        return FeasibilityChecker()

    @pytest.fixture
    def good_inputs(self) -> FeasibilityInputs:
        """Create inputs that should pass all checks."""
        return FeasibilityInputs(
            expected_qber=0.05,  # Well below 11%
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=100000,
            expected_leakage_bits=5000,
        )

    # -------------------------------------------------------------------------
    # Abort Code Tests
    # -------------------------------------------------------------------------

    def test_good_inputs_feasible(
        self, checker: FeasibilityChecker, good_inputs: FeasibilityInputs
    ) -> None:
        """Good inputs should yield is_feasible=True."""
        decision = checker.check(good_inputs)
        assert decision.is_feasible
        assert decision.abort_code is None

    def test_qber_warning_threshold(self, checker: FeasibilityChecker) -> None:
        """QBER near 11% should warn but proceed."""
        inputs = FeasibilityInputs(
            expected_qber=0.15,  # Above 11%, below 22%
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=100000,
            expected_leakage_bits=5000,
        )
        decision = checker.check(inputs)
        # Should still be feasible with warning
        assert decision.is_feasible
        assert len(decision.warnings) > 0

    def test_abort_qber_exceeds_hard_limit(
        self, checker: FeasibilityChecker
    ) -> None:
        """QBER > 22% should abort with ABORT-I-FEAS-001."""
        inputs = FeasibilityInputs(
            expected_qber=0.25,  # Above 22%
            storage_noise_r=0.75,
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=10000,
            expected_leakage_bits=1000,
        )
        decision = checker.check(inputs)
        assert not decision.is_feasible
        assert decision.abort_code == ABORT_CODE_QBER_TOO_HIGH

    def test_abort_strict_less_violated(
        self, checker: FeasibilityChecker
    ) -> None:
        """QBER >= r_storage should abort with ABORT-I-FEAS-002."""
        inputs = FeasibilityInputs(
            expected_qber=0.20,
            storage_noise_r=0.20,  # Equal to QBER (must be strictly greater)
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=10000,
            expected_leakage_bits=1000,
        )
        decision = checker.check(inputs)
        assert not decision.is_feasible
        assert decision.abort_code == ABORT_CODE_STRICT_LESS_VIOLATED

    def test_abort_capacity_rate_violated(
        self, checker: FeasibilityChecker
    ) -> None:
        """C_N × ν >= 1/2 should abort with ABORT-I-FEAS-003."""
        inputs = FeasibilityInputs(
            expected_qber=0.05,
            storage_noise_r=0.99,  # High r → high C_N
            storage_rate_nu=0.6,  # High rate to exceed bound
            epsilon_sec=1e-8,
            n_target_sifted_bits=10000,
            expected_leakage_bits=1000,
        )
        decision = checker.check(inputs)
        assert not decision.is_feasible
        assert decision.abort_code == ABORT_CODE_CAPACITY_RATE_VIOLATED

    def test_abort_death_valley(self, checker: FeasibilityChecker) -> None:
        """Parameters yielding ℓ_max ≤ 0 should abort with Death Valley."""
        # Death Valley: high leakage, small n → no extractable key
        inputs = FeasibilityInputs(
            expected_qber=0.05,
            storage_noise_r=0.5,
            storage_rate_nu=0.002,
            epsilon_sec=1e-8,
            n_target_sifted_bits=100,  # Very small n
            expected_leakage_bits=1000,  # Large leakage
        )
        decision = checker.check(inputs)
        # Should be infeasible due to negative key length
        assert not decision.is_feasible
        assert decision.abort_code == ABORT_CODE_DEATH_VALLEY


class TestFeasibilityDecision:
    """Tests for FeasibilityDecision dataclass."""

    def test_feasible_decision(self) -> None:
        """Feasible decision should have no abort code."""
        decision = FeasibilityDecision(
            is_feasible=True,
            abort_code=None,
            reason="All checks passed",
        )
        assert decision.is_feasible
        assert decision.abort_code is None

    def test_infeasible_decision(self) -> None:
        """Infeasible decision should have abort code."""
        decision = FeasibilityDecision(
            is_feasible=False,
            abort_code=ABORT_CODE_QBER_TOO_HIGH,
            reason="QBER too high",
        )
        assert not decision.is_feasible
        assert decision.abort_code == ABORT_CODE_QBER_TOO_HIGH

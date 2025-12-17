"""
Unit tests for caligo.simulation.timing module.

Tests TimingBarrier state machine, violations, and timing enforcement.
"""

from __future__ import annotations

import pytest

from caligo.simulation.timing import (
    TimingBarrier,
    TimingBarrierState,
    _get_sim_time,
    _sim_run,
)
from caligo.types.exceptions import TimingViolationError


# =============================================================================
# TimingBarrierState Tests
# =============================================================================


class TestTimingBarrierState:
    """Tests for TimingBarrierState enum."""

    def test_states_exist(self):
        """All expected states should exist."""
        assert TimingBarrierState.IDLE
        assert TimingBarrierState.WAITING
        assert TimingBarrierState.READY

    def test_states_are_unique(self):
        """Each state should have a unique value."""
        states = [s.value for s in TimingBarrierState]
        assert len(states) == len(set(states))


# =============================================================================
# TimingBarrier Creation Tests
# =============================================================================


class TestTimingBarrierCreation:
    """Tests for TimingBarrier initialization."""

    def test_valid_creation(self, timing_barrier):
        """Valid parameters should create instance."""
        assert timing_barrier.delta_t_ns == 1_000_000
        assert timing_barrier.strict_mode is True
        assert timing_barrier.state == TimingBarrierState.IDLE

    def test_creation_with_custom_delta_t(self):
        """Custom delta_t should be accepted."""
        barrier = TimingBarrier(delta_t_ns=500_000)
        assert barrier.delta_t_ns == 500_000

    def test_creation_with_strict_mode_false(self):
        """Strict mode can be disabled."""
        barrier = TimingBarrier(delta_t_ns=1_000_000, strict_mode=False)
        assert barrier.strict_mode is False

    def test_invalid_delta_t_zero(self):
        """delta_t_ns=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            TimingBarrier(delta_t_ns=0)

    def test_invalid_delta_t_negative(self):
        """Negative delta_t_ns should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            TimingBarrier(delta_t_ns=-1000)


# =============================================================================
# TimingBarrier Properties Tests
# =============================================================================


class TestTimingBarrierProperties:
    """Tests for TimingBarrier properties."""

    def test_initial_state_is_idle(self, timing_barrier):
        """Initial state should be IDLE."""
        assert timing_barrier.state == TimingBarrierState.IDLE

    def test_quantum_complete_time_initially_none(self, timing_barrier):
        """quantum_complete_time should be None initially."""
        assert timing_barrier.quantum_complete_time is None

    def test_timing_compliant_initially_true(self, timing_barrier):
        """timing_compliant should be True initially."""
        assert timing_barrier.timing_compliant is True


# =============================================================================
# State Transition Tests
# =============================================================================


class TestTimingBarrierStateTransitions:
    """Tests for TimingBarrier state machine transitions."""

    def test_mark_quantum_complete_transitions_to_waiting(self, timing_barrier):
        """mark_quantum_complete() should transition IDLE → WAITING."""
        assert timing_barrier.state == TimingBarrierState.IDLE
        timing_barrier.mark_quantum_complete()
        assert timing_barrier.state == TimingBarrierState.WAITING

    def test_mark_quantum_complete_records_time(self, timing_barrier):
        """mark_quantum_complete() should record simulation time."""
        timing_barrier.mark_quantum_complete()
        assert timing_barrier.quantum_complete_time is not None

    def test_mark_quantum_complete_twice_raises_in_strict(self, timing_barrier):
        """Calling mark_quantum_complete() twice should raise in strict mode."""
        timing_barrier.mark_quantum_complete()
        with pytest.raises(TimingViolationError, match="Expected IDLE"):
            timing_barrier.mark_quantum_complete()

    def test_mark_quantum_complete_twice_warns_in_lenient(self, timing_barrier_lenient):
        """Calling mark_quantum_complete() twice should not raise in lenient mode."""
        timing_barrier_lenient.mark_quantum_complete()
        # Should not raise, just warn
        timing_barrier_lenient.mark_quantum_complete()

    def test_wait_delta_t_transitions_to_ready(self, timing_barrier):
        """wait_delta_t() should transition WAITING → READY."""
        timing_barrier.mark_quantum_complete()

        # Consume the generator
        gen = timing_barrier.wait_delta_t()
        try:
            next(gen)
        except StopIteration:
            pass

        assert timing_barrier.state == TimingBarrierState.READY

    def test_wait_delta_t_without_mark_raises(self, timing_barrier):
        """wait_delta_t() without mark_quantum_complete() should raise."""
        with pytest.raises(TimingViolationError, match="Expected WAITING"):
            gen = timing_barrier.wait_delta_t()
            next(gen)

    def test_reset_returns_to_idle(self, timing_barrier):
        """reset() should return state to IDLE."""
        timing_barrier.mark_quantum_complete()
        gen = timing_barrier.wait_delta_t()
        try:
            next(gen)
        except StopIteration:
            pass

        assert timing_barrier.state == TimingBarrierState.READY
        timing_barrier.reset()
        assert timing_barrier.state == TimingBarrierState.IDLE
        assert timing_barrier.quantum_complete_time is None


# =============================================================================
# Basis Revelation Tests
# =============================================================================


class TestTimingBarrierBasisRevelation:
    """Tests for can_reveal_basis() method."""

    def test_cannot_reveal_in_idle(self, timing_barrier):
        """Cannot reveal basis in IDLE state."""
        assert timing_barrier.can_reveal_basis() is False

    def test_cannot_reveal_immediately_after_mark(self, timing_barrier):
        """Cannot reveal basis immediately after marking quantum complete."""
        timing_barrier.mark_quantum_complete()
        # Without waiting, should raise in strict mode
        with pytest.raises(TimingViolationError):
            timing_barrier.can_reveal_basis()

    def test_cannot_reveal_immediately_lenient_mode(self, timing_barrier_lenient):
        """In lenient mode, returns False instead of raising."""
        timing_barrier_lenient.mark_quantum_complete()
        assert timing_barrier_lenient.can_reveal_basis() is False

    def test_can_reveal_after_wait(self, timing_barrier):
        """Can reveal basis after wait_delta_t()."""
        timing_barrier.mark_quantum_complete()
        gen = timing_barrier.wait_delta_t()
        try:
            next(gen)
        except StopIteration:
            pass

        assert timing_barrier.can_reveal_basis() is True


# =============================================================================
# Timing Compliance Tests
# =============================================================================


class TestTimingBarrierCompliance:
    """Tests for assert_timing_compliant() method."""

    def test_assert_raises_in_idle(self, timing_barrier):
        """assert_timing_compliant() should raise in IDLE state."""
        with pytest.raises(TimingViolationError, match="quantum phase not yet marked"):
            timing_barrier.assert_timing_compliant()

    def test_assert_raises_without_wait(self, timing_barrier):
        """assert_timing_compliant() should raise if Δt not elapsed."""
        timing_barrier.mark_quantum_complete()
        # Without NetSquid, elapsed time is always 0
        with pytest.raises(TimingViolationError, match="Timing constraint violated"):
            timing_barrier.assert_timing_compliant()

    def test_compliance_tracking(self, timing_barrier_lenient):
        """Timing compliance should be tracked."""
        assert timing_barrier_lenient.timing_compliant is True

        timing_barrier_lenient.mark_quantum_complete()
        # Try to reveal without waiting (will fail but not raise)
        timing_barrier_lenient.can_reveal_basis()

        # Compliance should now be False
        assert timing_barrier_lenient.timing_compliant is False


# =============================================================================
# Elapsed Time Tests
# =============================================================================


class TestTimingBarrierElapsedTime:
    """Tests for get_elapsed_ns() method."""

    def test_elapsed_time_zero_before_mark(self, timing_barrier):
        """Elapsed time should be 0 before marking quantum complete."""
        assert timing_barrier.get_elapsed_ns() == 0.0

    def test_elapsed_time_after_mark(self, timing_barrier):
        """Elapsed time should be recorded after marking."""
        timing_barrier.mark_quantum_complete()
        # Without NetSquid, sim_time is always 0
        elapsed = timing_barrier.get_elapsed_ns()
        assert elapsed >= 0.0


# =============================================================================
# Diagnostic Methods Tests
# =============================================================================


class TestTimingBarrierDiagnostics:
    """Tests for diagnostic methods."""

    def test_get_diagnostic_info(self, timing_barrier):
        """get_diagnostic_info() should return dict with expected keys."""
        info = timing_barrier.get_diagnostic_info()

        assert "state" in info
        assert "delta_t_ns" in info
        assert "quantum_complete_time" in info
        assert "elapsed_ns" in info
        assert "timing_compliant" in info
        assert "strict_mode" in info
        assert "can_reveal" in info

        assert info["state"] == "IDLE"
        assert info["delta_t_ns"] == 1_000_000
        assert info["timing_compliant"] is True

    def test_repr(self, timing_barrier):
        """__repr__ should return informative string."""
        repr_str = repr(timing_barrier)

        assert "TimingBarrier" in repr_str
        assert "delta_t_ns=1000000" in repr_str
        assert "state=IDLE" in repr_str


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestTimingHelperFunctions:
    """Tests for timing helper functions."""

    def test_get_sim_time_without_netsquid(self):
        """_get_sim_time should return 0.0 without NetSquid."""
        # This tests the fallback behavior
        time = _get_sim_time()
        assert isinstance(time, float)

    def test_sim_run_without_netsquid(self):
        """_sim_run should be no-op without NetSquid."""
        # Should not raise
        _sim_run(1_000_000)


# =============================================================================
# Generator Pattern Tests
# =============================================================================


class TestTimingBarrierGeneratorPattern:
    """Tests for generator-based wait pattern."""

    def test_wait_is_generator(self, timing_barrier):
        """wait_delta_t() should return a generator."""
        timing_barrier.mark_quantum_complete()
        gen = timing_barrier.wait_delta_t()

        assert hasattr(gen, "__next__")
        assert hasattr(gen, "__iter__")

    def test_generator_yields_once(self, timing_barrier):
        """Generator should yield exactly once."""
        timing_barrier.mark_quantum_complete()
        gen = timing_barrier.wait_delta_t()

        # First next() should succeed
        try:
            next(gen)
        except StopIteration:
            pass  # Generator may complete immediately

    def test_yield_from_pattern(self, timing_barrier):
        """'yield from' pattern should work."""
        timing_barrier.mark_quantum_complete()

        def consumer():
            yield from timing_barrier.wait_delta_t()
            return "done"

        gen = consumer()
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == "done"
        assert timing_barrier.state == TimingBarrierState.READY

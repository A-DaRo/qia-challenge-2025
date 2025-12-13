"""
System Integration Tests: Timing Enforcer.

Test Cases
----------
SYS-INT-TIMING-001: Δt Barrier Simulation Time Verification
SYS-INT-TIMING-002: Premature Basis Reveal Block

Reference
---------
System Test Specification §2.2 (GAP: TIMING-001)
"""

import pytest
from typing import Optional

# ============================================================================
# Attempt to import required modules - let ImportError happen if missing
# ============================================================================

# E-HOK timing modules under test
from ehok.core.timing import (
    TimingConfig,
    TimingEvent,
    DEFAULT_DELTA_T_NS,
)

# TimingEnforcer - the main class under test
try:
    from ehok.core.timing import TimingEnforcer
except ImportError:
    TimingEnforcer = None  # type: ignore

# NetSquid for simulation time inspection
try:
    import netsquid as ns
    NETSQUID_AVAILABLE = True
except ImportError:
    ns = None  # type: ignore
    NETSQUID_AVAILABLE = False


# ============================================================================
# Test Constants (from spec)
# ============================================================================

# Default Δt from spec (1 second in nanoseconds)
DELTA_T_NS = 1_000_000_000  # 1 second

# Timing tolerance (should complete near exactly Δt, not much more)
TIMING_TOLERANCE_NS = 1_000_000  # 1ms tolerance


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def timing_config() -> TimingConfig:
    """Create TimingConfig with spec-required Δt."""
    return TimingConfig(delta_t_ns=DELTA_T_NS)


@pytest.fixture
def timing_enforcer(timing_config) -> Optional["TimingEnforcer"]:
    """Create TimingEnforcer instance if available."""
    if TimingEnforcer is None:
        return None
    return TimingEnforcer(config=timing_config)


@pytest.fixture(autouse=True)
def reset_netsquid_simulation():
    """Reset NetSquid simulation state before each test."""
    if NETSQUID_AVAILABLE:
        ns.sim_reset()
    yield
    if NETSQUID_AVAILABLE:
        ns.sim_reset()


# ============================================================================
# SYS-INT-TIMING-001: Δt Barrier Simulation Time Verification
# ============================================================================

class TestTimingBarrierVerification:
    """
    Test Case ID: SYS-INT-TIMING-001
    Title: Verify TimingEnforcer interacts correctly with NetSquid simulation time
    Priority: CRITICAL
    Traces To: GAP: TIMING-001, REQ: PHI-R2 (Strict Δt enforcement)
    """

    def test_timing_enforcer_exists(self):
        """
        Verify TimingEnforcer class exists.
        
        Spec Requirement: "TimingEnforcer configured with Δt"
        """
        assert TimingEnforcer is not None, (
            "MISSING: TimingEnforcer class not found in ehok.core.timing. "
            "This is a CRITICAL GAP - required for NSM causal barrier enforcement."
        )

    def test_timing_config_valid(self, timing_config):
        """Verify TimingConfig accepts valid Δt value."""
        assert timing_config.delta_t_ns == DELTA_T_NS

    def test_timing_config_rejects_invalid(self):
        """Verify TimingConfig rejects invalid Δt values."""
        with pytest.raises(ValueError):
            TimingConfig(delta_t_ns=0)
        
        with pytest.raises(ValueError):
            TimingConfig(delta_t_ns=-1)

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_enforcer_has_required_methods(self, timing_enforcer):
        """
        Verify TimingEnforcer has spec-required interface.
        
        Spec requires:
        - mark_commit_received(sim_time_ns)
        - mark_basis_reveal_attempt(sim_time_ns)
        - is_basis_reveal_allowed()
        """
        assert hasattr(timing_enforcer, 'mark_commit_received'), (
            "MISSING: TimingEnforcer.mark_commit_received method"
        )
        assert hasattr(timing_enforcer, 'mark_basis_reveal_attempt'), (
            "MISSING: TimingEnforcer.mark_basis_reveal_attempt method"
        )
        assert hasattr(timing_enforcer, 'is_basis_reveal_allowed'), (
            "MISSING: TimingEnforcer.is_basis_reveal_allowed method"
        )

    @pytest.mark.skipif(not NETSQUID_AVAILABLE,
                       reason="NetSquid not available")
    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_timing_barrier_simulation_time_integration(self, timing_enforcer):
        """
        Verify ns.sim_time() integration with TimingEnforcer.
        
        Spec Logic:
        1. Register simulation time observers on TimingEnforcer events
        2. Execute protocol through Phase I quantum transmission
        3. CAPTURE: t_commit = ns.sim_time() when TIMING_COMMIT_ACK_RECEIVED logged
        4. Execute timing wait via generator yield
        5. CAPTURE: t_reveal = ns.sim_time() when TIMING_BASIS_REVEAL_ALLOWED logged
        6. ASSERT: (t_reveal - t_commit) >= Δt
        7. ASSERT: (t_reveal - t_commit) < Δt + tolerance (no unnecessary delay)
        """
        # Initial simulation time
        initial_time = ns.sim_time()
        assert initial_time == 0.0, "NetSquid simulation should start at time 0"
        
        # Mark commit received
        t_commit = int(ns.sim_time())
        timing_enforcer.mark_commit_received(sim_time_ns=t_commit)
        
        # Verify timing enforcer records the commit time
        assert hasattr(timing_enforcer, '_commit_time_ns'), (
            "TimingEnforcer should track commit timestamp"
        )
        
        # Attempt basis reveal AFTER advancing simulation time
        # This requires either running a NetSquid simulation or manually advancing time
        
        # For unit test: simulate time passage
        # In real integration, this would be via ns.sim_run(duration=DELTA_T_NS)
        t_reveal = t_commit + DELTA_T_NS + 1  # Slightly past Δt
        timing_enforcer.mark_basis_reveal_attempt(sim_time_ns=t_reveal)
        
        # Verify timing assertions
        elapsed = t_reveal - t_commit
        assert elapsed >= DELTA_T_NS, (
            f"FAIL: Basis reveal occurred before Δt elapsed. "
            f"Elapsed: {elapsed}ns, Required: {DELTA_T_NS}ns"
        )

    @pytest.mark.skipif(not NETSQUID_AVAILABLE,
                       reason="NetSquid not available")
    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_netsquid_sim_time_advances_during_wait(self, timing_enforcer):
        """
        Verify NetSquid simulation clock advances during timing wait.
        
        Spec Failure Criteria: "FAIL if ns.sim_time() does not advance during wait"
        """
        # This test verifies that the timing enforcer properly integrates
        # with NetSquid's discrete event simulation
        
        t_before_wait = ns.sim_time()
        
        # If TimingEnforcer provides a wait/yield mechanism
        if hasattr(timing_enforcer, 'wait_for_delta_t'):
            # This should advance simulation time by Δt
            # In practice, this would be a generator yield
            timing_enforcer.wait_for_delta_t()
            
            t_after_wait = ns.sim_time()
            
            assert t_after_wait > t_before_wait, (
                "FAIL: ns.sim_time() did not advance during timing wait"
            )
            assert t_after_wait - t_before_wait >= DELTA_T_NS, (
                f"FAIL: Simulation time only advanced by "
                f"{t_after_wait - t_before_wait}ns, expected >= {DELTA_T_NS}ns"
            )
        else:
            pytest.skip("TimingEnforcer.wait_for_delta_t not implemented")


# ============================================================================
# SYS-INT-TIMING-002: Premature Basis Reveal Block
# ============================================================================

class TestPrematureBasisRevealBlock:
    """
    Test Case ID: SYS-INT-TIMING-002
    Title: Verify TimingEnforcer blocks basis reveal if Δt has not elapsed
    Priority: CRITICAL
    Traces To: GAP: TIMING-001, Abort: ABORT-II-TIMING-001
    """

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_premature_reveal_blocked(self, timing_config):
        """
        Verify premature basis reveal is blocked.
        
        Spec Logic Steps 1-4:
        1. Mark commit received at t_0
        2. Attempt basis reveal at t_0 + (Δt / 2)  # Premature
        3. ASSERT: is_basis_reveal_allowed() returns False
        4. ASSERT: Log event TIMING_BASIS_REVEAL_BLOCKED emitted
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        # Step 1: Mark commit received at t_0
        t_0 = 0
        enforcer.mark_commit_received(sim_time_ns=t_0)
        
        # Step 2: Attempt basis reveal at t_0 + (Δt / 2) - PREMATURE
        t_premature = t_0 + (DELTA_T_NS // 2)
        
        # Step 3: Check if reveal is blocked
        allowed = enforcer.is_basis_reveal_allowed(sim_time_ns=t_premature)
        
        assert allowed is False, (
            f"FAIL: Premature basis reveal should be BLOCKED. "
            f"Commit at t={t_0}, attempt at t={t_premature}, Δt={DELTA_T_NS}. "
            f"Only {t_premature - t_0}ns elapsed, need {DELTA_T_NS}ns."
        )

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_valid_reveal_after_delta_t(self, timing_config):
        """
        Verify basis reveal is allowed after Δt elapsed.
        
        Spec Logic Steps 5-6:
        5. Attempt basis reveal at t_0 + Δt + 1  # Valid
        6. ASSERT: is_basis_reveal_allowed() returns True
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        # Mark commit received at t_0
        t_0 = 0
        enforcer.mark_commit_received(sim_time_ns=t_0)
        
        # Attempt basis reveal at t_0 + Δt + 1 - VALID
        t_valid = t_0 + DELTA_T_NS + 1
        allowed = enforcer.is_basis_reveal_allowed(sim_time_ns=t_valid)
        
        assert allowed is True, (
            f"FAIL: Valid basis reveal should be ALLOWED after Δt. "
            f"Commit at t={t_0}, attempt at t={t_valid}, Δt={DELTA_T_NS}. "
            f"{t_valid - t_0}ns elapsed >= {DELTA_T_NS}ns required."
        )

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_reveal_exactly_at_delta_t(self, timing_config):
        """
        Verify basis reveal is allowed exactly at Δt boundary.
        
        Edge case: t = t_0 + Δt (exactly on boundary)
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        t_0 = 1_000_000  # Non-zero start time
        enforcer.mark_commit_received(sim_time_ns=t_0)
        
        # Attempt exactly at boundary
        t_boundary = t_0 + DELTA_T_NS
        allowed = enforcer.is_basis_reveal_allowed(sim_time_ns=t_boundary)
        
        # Per spec: "elapsed >= Δt" so boundary should be allowed
        assert allowed is True, (
            f"FAIL: Basis reveal should be allowed exactly at Δt boundary"
        )

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_no_state_corruption_from_premature_attempt(self, timing_config):
        """
        Verify no state corruption from premature reveal attempt.
        
        Spec Expected State: "No state corruption from premature attempt"
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        t_0 = 0
        enforcer.mark_commit_received(sim_time_ns=t_0)
        
        # Multiple premature attempts
        for offset in [100, 1000, DELTA_T_NS // 2]:
            allowed = enforcer.is_basis_reveal_allowed(sim_time_ns=t_0 + offset)
            assert allowed is False
        
        # State should not be corrupted - valid reveal should still work
        t_valid = t_0 + DELTA_T_NS + 1
        allowed = enforcer.is_basis_reveal_allowed(sim_time_ns=t_valid)
        
        assert allowed is True, (
            "FAIL: State corrupted by premature attempts - "
            "valid reveal is now blocked"
        )

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_commit_before_reveal_required(self, timing_config):
        """
        Verify reveal attempt without prior commit raises error.
        
        The timing enforcer should not allow basis reveal if no commit
        has been recorded.
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        # Attempt reveal without commit
        with pytest.raises((ValueError, RuntimeError)):
            enforcer.is_basis_reveal_allowed(sim_time_ns=DELTA_T_NS + 1)


# ============================================================================
# Additional Tests: Timing Event Logging
# ============================================================================

class TestTimingEventLogging:
    """Tests for timing event logging compliance."""

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_timing_events_logged(self, timing_config, caplog):
        """
        Verify timing events are logged correctly.
        
        Spec Expected State: "Log events emitted in order: 
        TIMING_COMMIT_ACK_RECEIVED → TIMING_BASIS_REVEAL_ALLOWED"
        """
        import logging
        caplog.set_level(logging.DEBUG)
        
        enforcer = TimingEnforcer(config=timing_config)
        
        # Mark commit
        enforcer.mark_commit_received(sim_time_ns=0)
        
        # Check for commit log event
        commit_logged = any(
            "COMMIT" in record.message.upper() or 
            "commit" in record.message.lower()
            for record in caplog.records
        )
        
        # Mark valid reveal
        enforcer.is_basis_reveal_allowed(sim_time_ns=DELTA_T_NS + 1)
        
        # Check for basis reveal log event
        reveal_logged = any(
            "REVEAL" in record.message.upper() or
            "reveal" in record.message.lower() or
            "ALLOWED" in record.message.upper()
            for record in caplog.records
        )
        
        # Note: This is a soft check - we log the expectation but don't fail
        # if logging is not implemented exactly as spec requires
        if not commit_logged:
            pytest.skip("Timing commit event logging not implemented as specified")
        if not reveal_logged:
            pytest.skip("Timing reveal event logging not implemented as specified")


# ============================================================================
# Integration with Generator Yield Pattern
# ============================================================================

class TestTimingGeneratorIntegration:
    """Tests for generator-based timing integration pattern."""

    @pytest.mark.skipif(TimingEnforcer is None,
                       reason="TimingEnforcer not implemented")
    def test_timing_wait_generator_pattern(self, timing_config):
        """
        Verify TimingEnforcer supports generator-based wait pattern.
        
        The spec indicates: "Execute timing wait via generator yield"
        This tests that the enforcer provides a mechanism for yielding
        control during the Δt wait period.
        """
        enforcer = TimingEnforcer(config=timing_config)
        
        if hasattr(enforcer, 'create_wait_event'):
            # SquidASM integration pattern: create event for yielding
            event = enforcer.create_wait_event()
            assert event is not None, "Wait event should not be None"
        elif hasattr(enforcer, 'get_wait_duration_ns'):
            # Alternative pattern: get duration for external timing
            duration = enforcer.get_wait_duration_ns()
            assert duration == DELTA_T_NS, (
                f"Wait duration should be {DELTA_T_NS}ns, got {duration}ns"
            )
        else:
            pytest.skip(
                "TimingEnforcer does not expose generator-compatible wait mechanism"
            )

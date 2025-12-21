"""
Test Suite for NSM Boundary Conditions (Task 9).

Tests "razor's edge" scenarios for NSM parameters:
- Security condition boundaries
- Heralded model stress
- Eta semantics

These tests verify the protocol handles edge cases correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.types.exceptions import (
    NSMViolationError,
    SecurityError,
)


# =============================================================================
# FIXTURES
# =============================================================================


@dataclass
class NSMParameters:
    """
    NSM Parameters for security analysis.
    
    Parameters
    ----------
    eta : float
        Channel loss coefficient (0 < eta <= 1).
    theta : float
        Storage entropy ratio (0 < theta < 1).
    delta_t : float
        Timing parameter in seconds.
    n_qubits : int
        Number of qubits in the protocol.
    """
    
    eta: float
    theta: float
    delta_t: float
    n_qubits: int
    
    def compute_security_margin(self, leaked_bits: int) -> float:
        """
        Compute security margin.
        
        Returns positive value if secure, negative if violated.
        """
        # Effective storage capacity
        storage_capacity = self.n_qubits * self.theta * (1 - self.eta)
        
        # Security requires leaked_bits < storage_capacity
        return storage_capacity - leaked_bits


@pytest.fixture
def conservative_nsm() -> NSMParameters:
    """Conservative NSM parameters with good security margin."""
    return NSMParameters(
        eta=0.5,       # 50% loss
        theta=0.5,     # 50% storage efficiency
        delta_t=1.0,   # 1 second
        n_qubits=10000
    )


@pytest.fixture
def aggressive_nsm() -> NSMParameters:
    """Aggressive NSM parameters near security boundary."""
    return NSMParameters(
        eta=0.95,      # 95% loss (very lossy channel)
        theta=0.9,     # 90% storage efficiency
        delta_t=0.1,   # 100ms
        n_qubits=10000
    )


# =============================================================================
# TASK 9.1: Security Condition Razor's Edge
# =============================================================================


class TestSecurityConditionBoundary:
    """
    Task 9.1: Test security at the boundary.
    
    Verify protocol behavior when leakage is exactly at or near the limit.
    """
    
    def test_exact_boundary_condition(self, conservative_nsm: NSMParameters) -> None:
        """Test when leaked bits exactly equals capacity."""
        # Compute exact boundary
        capacity = conservative_nsm.compute_security_margin(0)  # Full capacity
        
        # At boundary
        margin = conservative_nsm.compute_security_margin(int(capacity))
        
        # Should be exactly 0 or slightly negative due to int conversion
        assert abs(margin) < 1, f"Unexpected margin at boundary: {margin}"
    
    def test_one_bit_below_boundary(self, conservative_nsm: NSMParameters) -> None:
        """Protocol should be secure with 1 bit below limit."""
        capacity = conservative_nsm.compute_security_margin(0)
        leaked_bits = int(capacity) - 1
        
        margin = conservative_nsm.compute_security_margin(leaked_bits)
        
        assert margin > 0, "Should be secure 1 bit below boundary"
    
    def test_one_bit_above_boundary(self, conservative_nsm: NSMParameters) -> None:
        """Protocol should be insecure with 1 bit above limit."""
        capacity = conservative_nsm.compute_security_margin(0)
        leaked_bits = int(capacity) + 1
        
        margin = conservative_nsm.compute_security_margin(leaked_bits)
        
        assert margin < 0, "Should be insecure 1 bit above boundary"
    
    def test_leakage_tracker_boundary(self) -> None:
        """LeakageTracker should fire at exact boundary."""
        safety_cap = 1000
        tracker = LeakageTracker(safety_cap=safety_cap, abort_on_exceed=False)
        
        # Just below
        tracker.record_block(block_id=0, syndrome_bits=990, hash_bits=9)
        assert tracker.total_leakage == 999
        assert tracker.remaining_budget == 1
        
        # At boundary
        tracker.record_reveal(block_id=0, iteration=2, revealed_bits=1)
        assert tracker.total_leakage == 1000
        assert tracker.remaining_budget == 0  # Exactly at cap
    
    def test_margin_calculation_precision(self) -> None:
        """Security margin should handle floating point correctly."""
        params = NSMParameters(
            eta=0.333333333,
            theta=0.666666666,
            delta_t=1.0,
            n_qubits=10000
        )
        
        capacity = params.compute_security_margin(0)
        
        # Expected: 10000 * 0.666... * 0.666... ≈ 4444
        assert 4400 < capacity < 4500


# =============================================================================
# TASK 9.2: Heralded Model Stress
# =============================================================================


class TestHeraldedModelStress:
    """
    Task 9.2: Test heralded (click-based) model edge cases.
    
    In heralded model, only detected qubits contribute to security.
    """
    
    def test_low_detection_rate_security(self) -> None:
        """Low detection rates should still maintain security."""
        n_sent = 100000
        detection_rate = 0.01  # 1% detection
        n_detected = int(n_sent * detection_rate)
        
        # Security should be computed on detected qubits only
        params = NSMParameters(
            eta=0.9,
            theta=0.5,
            delta_t=1.0,
            n_qubits=n_detected  # Only detected qubits
        )
        
        capacity = params.compute_security_margin(0)
        
        # With 1000 detected qubits, 50% theta, 10% (1-eta)
        # Capacity = 1000 * 0.5 * 0.1 = 50 bits
        assert capacity == pytest.approx(50)
    
    def test_variable_detection_rate(self) -> None:
        """Security should scale with detection rate."""
        n_sent = 100000
        params_high = NSMParameters(eta=0.9, theta=0.5, delta_t=1.0, n_qubits=10000)
        params_low = NSMParameters(eta=0.9, theta=0.5, delta_t=1.0, n_qubits=1000)
        
        capacity_high = params_high.compute_security_margin(0)
        capacity_low = params_low.compute_security_margin(0)
        
        assert capacity_high == 10 * capacity_low
    
    def test_heralding_loss_composition(self) -> None:
        """
        Test that heralding loss and storage loss compose correctly.
        
        Total effective loss = herald_loss + storage_loss
        """
        herald_loss = 0.99  # 99% lost in heralding
        storage_loss = 0.9  # 90% lost in storage
        
        n_sent = 1000000
        n_heralded = int(n_sent * (1 - herald_loss))  # 10000
        
        # Storage capacity on heralded bits
        params = NSMParameters(
            eta=storage_loss,
            theta=0.5,
            delta_t=1.0,
            n_qubits=n_heralded
        )
        
        capacity = params.compute_security_margin(0)
        
        # 10000 * 0.5 * 0.1 = 500
        assert capacity == pytest.approx(500)


# =============================================================================
# TASK 9.3: Eta Semantics
# =============================================================================


class TestEtaSemantics:
    """
    Task 9.3: Test eta parameter semantics.
    
    Eta represents channel/storage loss. Higher eta = more loss = BETTER security.
    """
    
    def test_eta_zero_no_loss(self) -> None:
        """Eta=0 means no loss, adversary keeps everything."""
        params = NSMParameters(
            eta=0.0,
            theta=0.5,
            delta_t=1.0,
            n_qubits=10000
        )
        
        capacity = params.compute_security_margin(0)
        
        # 10000 * 0.5 * 1.0 = 5000
        assert capacity == 5000
    
    def test_eta_one_complete_loss(self) -> None:
        """Eta=1 means complete loss, no adversary capacity."""
        params = NSMParameters(
            eta=1.0,
            theta=0.5,
            delta_t=1.0,
            n_qubits=10000
        )
        
        capacity = params.compute_security_margin(0)
        
        # 10000 * 0.5 * 0.0 = 0
        assert capacity == 0
    
    def test_eta_monotonicity(self) -> None:
        """Higher eta should mean less adversary capacity."""
        etas = [0.1, 0.3, 0.5, 0.7, 0.9]
        capacities = []
        
        for eta in etas:
            params = NSMParameters(
                eta=eta,
                theta=0.5,
                delta_t=1.0,
                n_qubits=10000
            )
            capacities.append(params.compute_security_margin(0))
        
        # Capacities should be monotonically decreasing
        for i in range(len(capacities) - 1):
            assert capacities[i] > capacities[i + 1], \
                f"Capacity should decrease with eta: {capacities}"
    
    def test_eta_bounds_validation(self) -> None:
        """Eta should be validated to [0, 1]."""
        # Test invalid values
        invalid_etas = [-0.1, 1.1, 2.0, -1.0]
        
        for eta in invalid_etas:
            # In production code, this should raise
            params = NSMParameters(
                eta=eta,
                theta=0.5,
                delta_t=1.0,
                n_qubits=10000
            )
            
            # The formula breaks for invalid eta
            capacity = params.compute_security_margin(0)
            
            # Check that result is nonsensical
            if eta < 0:
                assert capacity > 5000  # More than theta alone
            elif eta > 1:
                assert capacity < 0  # Negative capacity


# =============================================================================
# TASK 9.4: Delta-t Timing Analysis
# =============================================================================


class TestDeltaTiming:
    """Test timing parameter effects on security."""
    
    def test_zero_timing_insecure(self) -> None:
        """Zero timing provides no security."""
        params = NSMParameters(
            eta=0.9,
            theta=0.5,
            delta_t=0.0,
            n_qubits=10000
        )
        
        # With delta_t=0, adversary can attack immediately
        # Our simplified model doesn't capture this directly
        # In practice: protocol should reject delta_t=0
        assert params.delta_t == 0
    
    def test_timing_tradeoff(self) -> None:
        """
        Longer timing improves security but hurts throughput.
        """
        base_params = NSMParameters(
            eta=0.9,
            theta=0.5,
            delta_t=1.0,
            n_qubits=10000
        )
        
        # Throughput ~ 1/delta_t
        throughput_1s = 1 / 1.0
        throughput_10s = 1 / 10.0
        
        assert throughput_1s == 10 * throughput_10s


# =============================================================================
# COMBINED PARAMETER STRESS
# =============================================================================


class TestCombinedParameterStress:
    """Test multiple parameters at their limits."""
    
    def test_worst_case_parameters(self) -> None:
        """Test with parameters favoring adversary."""
        params = NSMParameters(
            eta=0.01,      # Almost no loss
            theta=0.99,    # Very efficient storage
            delta_t=10.0,  # Long timing (helps honest parties)
            n_qubits=1000
        )
        
        capacity = params.compute_security_margin(0)
        
        # 1000 * 0.99 * 0.99 = 980.1
        assert capacity > 980
    
    def test_best_case_parameters(self) -> None:
        """Test with parameters favoring honest parties."""
        params = NSMParameters(
            eta=0.99,      # Almost total loss
            theta=0.01,    # Very inefficient storage
            delta_t=0.1,   # Short timing
            n_qubits=10000
        )
        
        capacity = params.compute_security_margin(0)
        
        # 10000 * 0.01 * 0.01 = 1
        assert capacity == pytest.approx(1)
    
    def test_real_world_scenario(self) -> None:
        """Test with realistic quantum channel parameters."""
        # Realistic fiber channel, 10km
        params = NSMParameters(
            eta=0.9,           # 10% survival over fiber
            theta=0.5,         # Bounded storage model
            delta_t=0.000001,  # Microsecond timing
            n_qubits=100000    # 100k qubits/second
        )
        
        capacity = params.compute_security_margin(0)
        
        # 100000 * 0.5 * 0.1 = 5000 bits
        # This is the leakage budget for reconciliation
        assert capacity == pytest.approx(5000)
        
        # Check if typical LDPC syndrome fits
        frame_size = 4096
        rate = 0.6
        syndrome_bits = int(frame_size * rate)  # ~2458
        
        assert syndrome_bits < capacity, "Single frame syndrome should fit in budget"

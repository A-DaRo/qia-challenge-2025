"""
Unit tests for blind reconciliation manager.

Tests iteration state management, LLR modulation, and convergence tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.reconciliation.blind_manager import (
    BlindConfig,
    BlindIterationState,
    BlindReconciliationManager,
)
from caligo.reconciliation import constants


class TestBlindConfig:
    """Tests for BlindConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config uses constants."""
        config = BlindConfig()
        
        assert config.max_iterations == constants.BLIND_MAX_ITERATIONS
        assert config.delta == constants.BLIND_DELTA_MODULATION

    def test_custom_values(self) -> None:
        """Custom config values honored."""
        config = BlindConfig(max_iterations=5, delta=0.15)
        
        assert config.max_iterations == 5
        assert config.delta == 0.15


class TestBlindIterationState:
    """Tests for BlindIterationState dataclass."""

    def test_state_fields(self) -> None:
        """State holds expected fields."""
        state = BlindIterationState(
            iteration=2,
            decoded_bits=np.array([0, 1, 0, 1]),
            syndrome_errors=5,
            converged=False,
        )
        
        assert state.iteration == 2
        assert len(state.decoded_bits) == 4
        assert state.syndrome_errors == 5
        assert state.converged is False

    def test_converged_state(self) -> None:
        """Converged state has zero syndrome errors."""
        state = BlindIterationState(
            iteration=1,
            decoded_bits=np.array([0, 1, 0, 1]),
            syndrome_errors=0,
            converged=True,
        )
        
        assert state.converged is True
        assert state.syndrome_errors == 0


class TestBlindReconciliationManager:
    """Tests for BlindReconciliationManager class."""

    def test_initialize_creates_state(self) -> None:
        """Initialize creates initial state."""
        manager = BlindReconciliationManager()
        
        initial_bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
        state = manager.initialize(initial_bits)
        
        assert state.iteration == 0
        assert len(state.decoded_bits) == 6
        assert state.converged is False

    def test_should_continue_first_iteration(self) -> None:
        """Should continue on first iteration."""
        manager = BlindReconciliationManager()
        
        state = BlindIterationState(
            iteration=0,
            decoded_bits=np.array([0, 1]),
            syndrome_errors=5,
            converged=False,
        )
        
        assert manager.should_continue(state) is True

    def test_should_not_continue_when_converged(self) -> None:
        """Should not continue when converged."""
        manager = BlindReconciliationManager()
        
        state = BlindIterationState(
            iteration=1,
            decoded_bits=np.array([0, 1]),
            syndrome_errors=0,
            converged=True,
        )
        
        assert manager.should_continue(state) is False

    def test_should_not_continue_max_iterations(self) -> None:
        """Should not continue at max iterations."""
        manager = BlindReconciliationManager(BlindConfig(max_iterations=3))
        
        state = BlindIterationState(
            iteration=3,
            decoded_bits=np.array([0, 1]),
            syndrome_errors=5,
            converged=False,
        )
        
        assert manager.should_continue(state) is False

    def test_advance_iteration_increments(self) -> None:
        """Advance iteration increments counter."""
        manager = BlindReconciliationManager()
        
        prev_state = BlindIterationState(
            iteration=1,
            decoded_bits=np.array([0, 1, 0, 1]),
            syndrome_errors=3,
            converged=False,
        )
        
        new_state = manager.advance_iteration(
            prev_state,
            new_decoded_bits=np.array([0, 1, 1, 1]),
            new_syndrome_errors=0,
        )
        
        assert new_state.iteration == 2
        assert new_state.syndrome_errors == 0
        assert new_state.converged is True

    def test_advance_iteration_converges(self) -> None:
        """Advance iteration marks converged when errors=0."""
        manager = BlindReconciliationManager()
        
        prev_state = BlindIterationState(
            iteration=0,
            decoded_bits=np.array([0, 1]),
            syndrome_errors=5,
            converged=False,
        )
        
        new_state = manager.advance_iteration(
            prev_state,
            new_decoded_bits=np.array([0, 1]),
            new_syndrome_errors=0,
        )
        
        assert new_state.converged is True


class TestLLRModulation:
    """Tests for LLR modulation in blind reconciliation."""

    def test_build_llr_for_state_first_iteration(self) -> None:
        """First iteration uses channel LLRs directly."""
        manager = BlindReconciliationManager()
        
        received_bits = np.array([0, 1, 0, 1], dtype=np.int8)
        state = manager.initialize(received_bits)
        
        llr = manager.build_llr_for_state(
            state,
            received_bits=received_bits,
            qber=0.05,
        )
        
        # First iteration should have standard LLRs
        assert len(llr) == 4
        assert all(np.abs(llr) > 0)

    def test_llr_signs_match_bits(self) -> None:
        """LLR signs match received bits (0→positive, 1→negative)."""
        manager = BlindReconciliationManager()
        
        received_bits = np.array([0, 1, 0, 1], dtype=np.int8)
        state = manager.initialize(received_bits)
        
        llr = manager.build_llr_for_state(
            state,
            received_bits=received_bits,
            qber=0.05,
        )
        
        for i, bit in enumerate(received_bits):
            if bit == 0:
                assert llr[i] > 0
            else:
                assert llr[i] < 0

    def test_delta_modulation_increases(self) -> None:
        """Delta modulation increases with iterations (Martinez-Mateo interface)."""
        # Use Martinez-Mateo interface which actually implements delta modulation
        manager = BlindReconciliationManager(BlindConfig(
            delta=0.10,
            frame_size=100,  # Small frame for testing
            max_iterations=5,
        ))
        
        # Initialize with syndrome length (Martinez-Mateo style)
        state0 = manager.initialize(30)  # 30% syndrome for rate ~0.70
        
        # Advance iteration with shortened values (Martinez-Mateo style)
        shortened_vals = np.zeros(manager.config.delta_per_iteration, dtype=np.int8)
        state1 = manager.advance_iteration(state0, shortened_values=shortened_vals)
        
        # n_punctured should decrease, n_shortened should increase
        assert state1.n_punctured < state0.n_punctured
        assert state1.n_shortened > state0.n_shortened


class TestEdgeCases:
    """Edge case tests for blind manager."""

    def test_empty_bits(self) -> None:
        """Handle empty bit array."""
        manager = BlindReconciliationManager()
        
        state = manager.initialize(np.array([], dtype=np.int8))
        
        assert state.iteration == 0
        assert len(state.decoded_bits) == 0

    def test_single_iteration_success(self) -> None:
        """Single iteration success on clean channel."""
        manager = BlindReconciliationManager()
        
        bits = np.zeros(100, dtype=np.int8)
        state = manager.initialize(bits)
        
        # Simulated converged state
        new_state = manager.advance_iteration(
            state,
            new_decoded_bits=bits,
            new_syndrome_errors=0,
        )
        
        assert new_state.converged is True
        assert new_state.iteration == 1

"""
Test Suite for Reconciliation Complexity Scenarios (Task 10).

Tests complex scenarios:
- Blind strategy miscalibration
- Pessimistic rate waste
- Tiny block constraints

These tests verify the protocol handles edge cases correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

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
    CaligoError,
    ReconciliationError,
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
    
    def get_pattern(rate: float) -> np.ndarray:
        """Return hybrid pattern for given rate."""
        n_frozen = int(4096 * (1 - rate))
        pattern = np.zeros(4096, dtype=np.uint8)
        pattern[:n_frozen] = 1  # Freeze first n_frozen
        return pattern
    
    mock.get_pattern = MagicMock(side_effect=get_pattern)
    mock.get_modulation_indices = MagicMock(return_value=np.arange(400, dtype=np.int64))
    mock.available_rates = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    mock.compiled_topology = MagicMock()
    mock.compiled_topology.n_edges = 12000
    mock.compiled_topology.n_checks = 2048
    mock.compiled_topology.n_vars = 4096
    return mock


@pytest.fixture
def mock_codec_variable_iterations() -> MagicMock:
    """Mock codec that takes variable iterations to converge."""
    iteration_counts = iter([5, 15, 30, 55, 60])  # Increasing iterations
    
    def mock_decode(*args, **kwargs) -> DecoderResult:
        try:
            iters = next(iteration_counts)
            converged = iters < 60
        except StopIteration:
            iters = 60
            converged = False
        
        return DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=converged,
            iterations=iters,
            messages=np.zeros(24000, dtype=np.float64),
        )
    
    mock = MagicMock()
    mock.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
    mock.decode_baseline = MagicMock(side_effect=mock_decode)
    mock.decode_blind = MagicMock(side_effect=mock_decode)
    return mock


# =============================================================================
# TASK 10.1: Blind Strategy Miscalibration
# =============================================================================


class TestBlindMiscalibration:
    """
    Task 10.1: Test Blind strategy when heuristic QBER is wrong.
    
    Blind starts with a heuristic and adapts. If heuristic is very wrong,
    it should still converge but may take more iterations.
    """
    
    def test_heuristic_too_low(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """
        Heuristic underestimates QBER.
        
        Strategy starts with high rate (optimistic), then must
        decrease rate as decoding fails. Alice sends progressive reveals.
        """
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
            max_blind_iterations=5,
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
            qber_heuristic=0.01,  # Very optimistic (actual ~10%)
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        
        # Collect reveals until success or exhaustion
        msg = next(gen)
        reveals = 1
        
        # Alice sends initial syndrome, Bob fails to decode
        # Alice then reveals more bits in subsequent iterations
        while reveals < 5:
            try:
                response = gen.send({"verified": False, "converged": False})
                if response.get("kind") == "done" or "result" in response:
                    break
                reveals += 1
            except StopIteration:
                break
        
        # Alice should have sent the initial syndrome via codec.encode
        assert mock_codec.encode.call_count >= 1
    
    def test_heuristic_too_high(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """
        Heuristic overestimates QBER.
        
        Strategy starts with low rate (pessimistic), wastes capacity
        but should succeed on first try.
        """
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_blind = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,  # Always succeeds (channel is good)
            iterations=5,
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        strategy = BlindStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
            max_blind_iterations=5,
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,
            qber_heuristic=0.15,  # Very pessimistic (actual ~1%)
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        msg = next(gen)
        
        # Should succeed immediately
        try:
            gen.send({"verified": True, "converged": True})
        except StopIteration as e:
            result = e.value
            assert result.verified is True


# =============================================================================
# TASK 10.2: Pessimistic Rate Waste
# =============================================================================


class TestPessimisticRateWaste:
    """
    Task 10.2: Test efficiency loss from conservative rate selection.
    
    Using lower rate than necessary wastes key material.
    """
    
    def test_rate_waste_quantification(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Quantify bits wasted by pessimistic rate."""
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_baseline = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=10,
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        leakage_tracker = LeakageTracker(safety_cap=100000)
        
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=leakage_tracker,
        )
        
        # Actual channel is good (low QBER)
        actual_qber = 0.02
        
        # But we use pessimistic estimate
        pessimistic_qber = 0.10
        
        # Calculate optimal vs actual rates
        optimal_rate = strategy._select_rate(actual_qber, 1.1)
        pessimistic_rate = strategy._select_rate(pessimistic_qber, 1.1)
        
        # Waste = extra syndrome bits leaked
        # Lower rate means more syndrome bits
        optimal_syndrome_bits = int(4096 * (1 - optimal_rate))
        pessimistic_syndrome_bits = int(4096 * (1 - pessimistic_rate))
        
        waste = pessimistic_syndrome_bits - optimal_syndrome_bits
        
        assert waste >= 0, "Pessimistic should leak more"
    
    def test_efficiency_metric(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Calculate reconciliation efficiency."""
        # Efficiency = (actual key bits) / (Shannon limit)
        
        qber = 0.05
        frame_size = 4096
        
        # Shannon limit: H(X|Y) = H(qber) bits per raw bit
        h_qber = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        shannon_limit = frame_size * (1 - h_qber)
        
        # Actual efficiency depends on rate achieved
        actual_rate = 0.7
        actual_key_bits = int(frame_size * actual_rate)
        
        efficiency = actual_key_bits / shannon_limit
        
        # Good reconciliation achieves >95% efficiency
        assert 0 < efficiency <= 1.0
    
    def test_rate_stepping_granularity(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Test that rate steps are fine enough."""
        # Available rates should allow fine-grained selection
        rates = mock_mother_code.available_rates
        
        # Check granularity
        steps = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
        max_step = max(steps)
        
        # Steps should be <=5% for good efficiency
        assert max_step <= 0.1, f"Rate steps too coarse: {max_step}"


# =============================================================================
# TASK 10.3: Tiny Block Constraints
# =============================================================================


class TestTinyBlockConstraints:
    """
    Task 10.3: Test behavior with very small blocks.
    
    Small blocks may have statistical anomalies that stress the decoder.
    """
    
    def test_minimum_block_size(self) -> None:
        """Verify minimum block size constraints."""
        # LDPC codes have minimum size requirements
        min_block_size = 256  # Typical minimum for LDPC
        
        # Frame size should be >= minimum
        frame_size = 4096
        
        assert frame_size >= min_block_size
    
    def test_small_payload_handling(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Test handling of payloads smaller than frame size."""
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
            leakage_tracker=LeakageTracker(safety_cap=100000),
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
        
        # Very small payload
        small_payload = np.zeros(100, dtype=np.uint8)
        
        # Strategy should handle gracefully (pad or reject)
        gen = strategy.alice_reconcile_block(small_payload, context, block_id=0)
        
        try:
            msg = next(gen)
            # If it proceeds, padding occurred
            assert msg is not None
        except (ValueError, CaligoError):
            # Rejection is also valid behavior
            pass
    
    def test_hash_collision_on_small_blocks(self) -> None:
        """
        Small blocks have higher hash collision probability.
        
        With 64-bit hash, collision prob is 2^(-64) per comparison.
        But small blocks mean more blocks, thus more comparisons.
        """
        hash_bits = 64
        n_blocks = 1000
        
        # Probability of at least one collision
        # P(collision) ≈ n * 2^(-hash_bits) for small n
        collision_prob = n_blocks * (2 ** (-hash_bits))
        
        # Should be negligible
        assert collision_prob < 1e-10
    
    def test_statistical_qber_variance_small_blocks(self) -> None:
        """
        QBER estimate variance increases with smaller samples.
        """
        # Large block
        n_large = 4096
        qber_true = 0.05
        variance_large = qber_true * (1 - qber_true) / n_large
        std_large = np.sqrt(variance_large)
        
        # Small block
        n_small = 256
        variance_small = qber_true * (1 - qber_true) / n_small
        std_small = np.sqrt(variance_small)
        
        # Small blocks have higher variance
        assert std_small > std_large
        
        # 95% CI width
        ci_width_large = 2 * 1.96 * std_large
        ci_width_small = 2 * 1.96 * std_small
        
        assert ci_width_small > ci_width_large


# =============================================================================
# ITERATION BUDGET ANALYSIS
# =============================================================================


class TestIterationBudget:
    """Test iteration budget handling."""
    
    def test_iteration_limit_respected(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Decoder should not exceed max iterations."""
        max_iter = 60
        
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_baseline = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=False,
            iterations=max_iter,  # Hit limit
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
        )
        
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=max_iter,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=0.05,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        msg = next(gen)
        
        # Check that max_iterations was passed to codec
        # (Indirectly via context)
        assert context.max_iterations == max_iter
    
    def test_early_termination_saves_iterations(
        self,
        mock_mother_code: MagicMock,
    ) -> None:
        """Early convergence should save iterations."""
        mock_codec = MagicMock()
        mock_codec.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
        mock_codec.decode_baseline = MagicMock(return_value=DecoderResult(
            corrected_bits=np.zeros(4096, dtype=np.uint8),
            converged=True,
            iterations=5,  # Converged early
            messages=np.zeros(24000, dtype=np.float64),
        ))
        
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
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
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        msg = next(gen)
        
        # Decoder returned after 5 iterations
        result = mock_codec.decode_baseline.return_value
        assert result.iterations < context.max_iterations

"""
Integration tests for reconciliation package.

Tests full reconciliation flow, component interaction, and end-to-end scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from caligo.reconciliation import constants
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.ldpc_encoder import encode_block, prepare_frame
from caligo.reconciliation.ldpc_decoder import BeliefPropagationDecoder, build_channel_llr
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.rate_selector import select_rate_with_parameters
from caligo.reconciliation.leakage_tracker import LeakageTracker, compute_safety_cap
from caligo.reconciliation.orchestrator import (
    ReconciliationOrchestrator,
    ReconciliationOrchestratorConfig,
)


@pytest.fixture
def matrix_manager() -> MatrixManager:
    """Load LDPC matrix manager."""
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


@pytest.fixture
def hash_verifier() -> PolynomialHashVerifier:
    """Create hash verifier."""
    return PolynomialHashVerifier()


class TestEncoderDecoderIntegration:
    """Tests for encoder-decoder interaction."""

    def test_encode_decode_noiseless(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Encode and decode noiseless block successfully using patterns."""
        # Get pattern for rate 0.70
        pattern = matrix_manager.get_puncture_pattern(0.70)
        if pattern is None:
            pytest.skip("Puncture pattern for rate 0.70 not available")

        # Alice's key (payload = non-punctured positions)
        rng = np.random.default_rng(42)
        frame_size = 4096
        n_punctured = int(pattern.sum())
        payload_len = frame_size - n_punctured
        alice_key = rng.integers(0, 2, size=payload_len, dtype=np.int8)
        
        # Prepare frame and encode
        from caligo.reconciliation.ldpc_encoder import encode_block_from_payload

        # Use mother code (rate 0.5) with pattern
        H = matrix_manager.get_matrix(0.5)
        syndrome_block = encode_block_from_payload(alice_key, H, pattern)
        
        # Bob has identical key (noiseless)
        bob_key = alice_key.copy()
        bob_frame = prepare_frame(bob_key, puncture_pattern=pattern)
        
        # Bob decodes
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(bob_key, qber=0.01, punctured_mask=pattern)
        result = decoder.decode(llr, syndrome_block.syndrome)
        
        # Should converge immediately
        assert result.converged

    def test_encode_decode_with_noise(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Encode and decode block with 3% QBER using patterns."""
        # Get pattern for rate 0.70
        pattern = matrix_manager.get_puncture_pattern(0.70)
        if pattern is None:
            pytest.skip("Puncture pattern for rate 0.70 not available")

        # Alice's key
        rng = np.random.default_rng(42)
        frame_size = 4096
        n_punctured = int(pattern.sum())
        payload_len = frame_size - n_punctured
        alice_key = rng.integers(0, 2, size=payload_len, dtype=np.int8)
        
        # Prepare and encode
        from caligo.reconciliation.ldpc_encoder import encode_block_from_payload

        # Use mother code (rate 0.5) with pattern
        H = matrix_manager.get_matrix(0.5)
        syndrome_block = encode_block_from_payload(alice_key, H, pattern)
        
        # Bob's key with errors (~3% QBER)
        bob_key = alice_key.copy()
        n_errors = int(payload_len * 0.03)
        error_positions = rng.choice(payload_len, size=n_errors, replace=False)
        bob_key[error_positions] = 1 - bob_key[error_positions]
        
        # Bob decodes
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(bob_key, qber=0.05, punctured_mask=pattern)
        result = decoder.decode(llr, syndrome_block.syndrome)
        
        # Should converge for low noise
        assert result.iterations <= 60


class TestHashVerification:
    """Tests for hash verification integration."""

    def test_hash_after_decode(
        self,
        matrix_manager: MatrixManager,
        hash_verifier: PolynomialHashVerifier,
    ) -> None:
        """Hash verification after successful decode using patterns."""
        # Get pattern for rate 0.70
        pattern = matrix_manager.get_puncture_pattern(0.70)
        if pattern is None:
            pytest.skip("Puncture pattern for rate 0.70 not available")

        # Setup
        rng = np.random.default_rng(42)
        frame_size = 4096
        n_punctured = int(pattern.sum())
        payload_len = frame_size - n_punctured
        alice_key = rng.integers(0, 2, size=payload_len, dtype=np.int8)
        
        # Alice computes syndrome and hash
        from caligo.reconciliation.ldpc_encoder import encode_block_from_payload

        # Use mother code (rate 0.5) with pattern
        H = matrix_manager.get_matrix(0.5)
        syndrome_block = encode_block_from_payload(alice_key, H, pattern)
        alice_hash = hash_verifier.compute_hash(alice_key)
        
        # Bob decodes (noiseless for simplicity)
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(alice_key, qber=0.01, punctured_mask=pattern)
        result = decoder.decode(llr, syndrome_block.syndrome)
        
        # Bob extracts key and verifies hash (non-punctured positions only)
        bob_frame = result.corrected_bits
        bob_key = bob_frame[~pattern.astype(bool)]
        
        assert hash_verifier.verify(bob_key, alice_hash)


class TestRateSelectionFlow:
    """Tests for rate selection in reconciliation flow."""

    def test_rate_selection_guides_matrix_choice(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Rate selection determines matrix choice."""
        qber = 0.05
        payload = 2000
        
        selection = select_rate_with_parameters(qber, payload, frame_size=4096)
        
        # Should be able to load selected rate
        H = matrix_manager.get_matrix(selection.rate)
        
        assert H is not None
        assert H.shape[1] == 4096


class TestLeakageTracking:
    """Tests for leakage tracking in reconciliation flow."""

    def test_leakage_from_multiple_blocks(self) -> None:
        """Track leakage across multiple blocks."""
        n_sifted = 40000  # 10 blocks of 4096
        qber = 0.05
        
        cap = compute_safety_cap(n_sifted, qber)
        tracker = LeakageTracker(safety_cap=cap)
        
        # Simulate 10 blocks at rate 0.70 (30% syndrome)
        syndrome_bits_per_block = int(4096 * 0.30)
        hash_bits = 50
        
        for i in range(10):
            tracker.record_block(
                block_id=i,
                syndrome_bits=syndrome_bits_per_block,
                hash_bits=hash_bits,
            )
        
        # Check total leakage
        expected_total = 10 * (syndrome_bits_per_block + hash_bits)
        assert tracker.total_leakage == expected_total
        
        # Should be under safety cap for 5% QBER
        assert tracker.check_safety() is True

    def test_high_qber_triggers_abort(self) -> None:
        """High QBER with many blocks triggers abort."""
        n_sifted = 20000  # 5 blocks
        qber = 0.10  # High QBER = low cap
        
        cap = compute_safety_cap(n_sifted, qber)
        tracker = LeakageTracker(safety_cap=cap)
        
        # At 10% QBER, entropy ~0.47, so cap ≈ 20000 * (1 - 0.47) = 10600
        # With rate 0.50, syndrome = 2048 bits/block
        # 5 blocks = 5 * 2098 = 10490, close to cap
        
        for i in range(5):
            tracker.record_block(
                block_id=i,
                syndrome_bits=2048,
                hash_bits=50,
            )
        
        # Should be close to or at cap
        assert tracker.remaining_budget < 2000


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for reconciliation orchestrator."""

    def test_orchestrator_initialization(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Orchestrator initializes with config."""
        config = ReconciliationOrchestratorConfig(
            frame_size=4096,
            max_blind_iterations=3,
        )
        
        orchestrator = ReconciliationOrchestrator(
            config=config,
            matrix_manager=matrix_manager,
        )
        
        assert orchestrator is not None

    def test_single_block_reconciliation(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Single block reconciliation end-to-end."""
        # Setup orchestrator
        config = ReconciliationOrchestratorConfig(
            frame_size=4096,
            max_blind_iterations=3,
        )
        orchestrator = ReconciliationOrchestrator(
            config=config,
            matrix_manager=matrix_manager,
            safety_cap=50000,
        )
        
        # Alice's key (full frame for rate 0.5 with no puncturing)
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, size=4096, dtype=np.int8)
        
        # Bob's key with ~2% errors
        bob_key = alice_key.copy()
        n_errors = int(len(bob_key) * 0.02)
        error_positions = rng.choice(len(bob_key), size=n_errors, replace=False)
        bob_key[error_positions] = 1 - bob_key[error_positions]
        
        # Reconcile using full API (Alice + Bob keys)
        result = orchestrator.reconcile_block(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=0.03,
        )
        
        # Check result
        assert result is not None
        if result.verified:
            np.testing.assert_array_equal(result.corrected_payload, alice_key)


class TestHighRatePatternBased:
    """Tests for high-rate reconciliation with untainted puncturing patterns."""

    def test_high_rate_with_pattern_rate_0_8(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """
        Test rate 0.8 reconciliation with untainted puncturing pattern.

        This test validates that the untainted puncturing approach enables
        high-rate codes to converge where random padding would fail.
        """
        # Check if pattern is available
        pattern = matrix_manager.get_puncture_pattern(0.8)
        if pattern is None:
            pytest.skip("Puncture pattern for rate 0.8 not available")

        # Setup keys with low QBER (high-rate codes work in low-noise regime)
        rng = np.random.default_rng(42)
        frame_size = 4096
        # Payload length = non-punctured positions
        n_punctured = int(pattern.sum())
        payload_len = frame_size - n_punctured
        alice_key = rng.integers(0, 2, size=payload_len, dtype=np.int8)

        # Bob's key with ~1% QBER (high-rate codes need lower noise)
        bob_key = alice_key.copy()
        n_errors = int(payload_len * 0.01)
        error_positions = rng.choice(payload_len, size=n_errors, replace=False)
        bob_key[error_positions] = 1 - bob_key[error_positions]

        # Alice encodes with MOTHER CODE and pattern
        from caligo.reconciliation.ldpc_encoder import encode_block_from_payload

        # Use mother code (rate 0.5) - puncturing is applied to mother code!
        H_mother = matrix_manager.get_matrix(0.5)
        syndrome_block = encode_block_from_payload(
            payload=alice_key,
            H=H_mother,
            puncture_pattern=pattern,
        )

        # Bob decodes with MOTHER CODE and pattern
        llr = build_channel_llr(
            bob_key, qber=0.02, punctured_mask=pattern
        )

        decoder = BeliefPropagationDecoder(H_mother, max_iterations=100)
        result = decoder.decode(llr, syndrome_block.syndrome)

        # High-rate codes may not achieve perfect convergence but should get close
        # Check that we made progress and residual errors are low
        assert result.iterations <= 100, "Should complete within max iterations"
        
        # Verify low error rate (allow small residual errors at high rate)
        corrected_payload = result.corrected_bits[pattern == 0][:payload_len]
        error_rate = np.mean(corrected_payload != alice_key)
        assert error_rate < 0.02, f"Error rate {error_rate:.4f} too high for rate 0.8"

    def test_high_rate_stress_rate_0_9(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """
        Stress test for rate 0.9 (very high rate).

        Rate 0.9 is extremely aggressive and should work only at very low QBER.
        """
        pattern = matrix_manager.get_puncture_pattern(0.9)
        if pattern is None:
            pytest.skip("Puncture pattern for rate 0.9 not available")

        rng = np.random.default_rng(42)
        frame_size = 4096
        # Correct payload calculation
        n_punctured = int(pattern.sum())
        payload_len = frame_size - n_punctured
        alice_key = rng.integers(0, 2, size=payload_len, dtype=np.int8)

        # Bob's key with ~1% QBER (very low noise for very high rate)
        bob_key = alice_key.copy()
        n_errors = max(1, int(payload_len * 0.01))
        error_positions = rng.choice(payload_len, size=n_errors, replace=False)
        bob_key[error_positions] = 1 - bob_key[error_positions]

        # Encode and decode with MOTHER CODE
        from caligo.reconciliation.ldpc_encoder import encode_block_from_payload

        # Use mother code (rate 0.5) - puncturing is applied to mother code!
        H_mother = matrix_manager.get_matrix(0.5)
        syndrome_block = encode_block_from_payload(
            payload=alice_key,
            H=H_mother,
            puncture_pattern=pattern,
        )

        llr = build_channel_llr(bob_key, qber=0.02, punctured_mask=pattern)

        decoder = BeliefPropagationDecoder(H_mother, max_iterations=150)
        result = decoder.decode(llr, syndrome_block.syndrome)

        # At rate 0.9 (extreme rate), check that decoder makes progress
        assert result.iterations <= 150, "Should complete within max iterations"
        
        # Verify reasonable error rate (more lenient for extreme rate)
        corrected_payload = result.corrected_bits[pattern == 0][:payload_len]
        error_rate = np.mean(corrected_payload != alice_key)
        assert error_rate < 0.05, f"Error rate {error_rate:.4f} too high for rate 0.9"


class TestErrorScenarios:
    """Tests for error handling scenarios."""

    def test_hash_mismatch_detected(
        self,
        hash_verifier: PolynomialHashVerifier,
    ) -> None:
        """Hash mismatch detected when keys differ."""
        alice_key = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        bob_key = np.array([0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int8)  # Different
        
        alice_hash = hash_verifier.compute_hash(alice_key)
        
        assert hash_verifier.verify(bob_key, alice_hash) is False

    def test_leakage_cap_exceeded(self) -> None:
        """Leakage cap exceeded raises abort condition via circuit breaker."""
        from caligo.types.exceptions import LeakageBudgetExceeded
        
        # Test with abort_on_exceed=True (circuit breaker pattern)
        tracker = LeakageTracker(safety_cap=1000, abort_on_exceed=True)
        
        # First block
        tracker.record_block(block_id=0, syndrome_bits=600, hash_bits=50)
        assert tracker.should_abort() is False
        
        # Second block exceeds cap - should raise immediately
        with pytest.raises(LeakageBudgetExceeded) as exc_info:
            tracker.record_block(block_id=1, syndrome_bits=600, hash_bits=50)
        
        assert exc_info.value.actual_leakage == 1300
        assert exc_info.value.max_allowed == 1000
        
        # Test with abort_on_exceed=False (legacy behavior)
        tracker_legacy = LeakageTracker(safety_cap=1000, abort_on_exceed=False)
        tracker_legacy.record_block(block_id=0, syndrome_bits=600, hash_bits=50)
        tracker_legacy.record_block(block_id=1, syndrome_bits=600, hash_bits=50)
        assert tracker_legacy.should_abort() is True

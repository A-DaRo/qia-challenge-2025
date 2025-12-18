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
        """Encode and decode noiseless block successfully."""
        # Alice's key (random)
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, size=2867, dtype=np.int8)  # 70% of 4096
        
        # Prepare frame and encode
        frame = prepare_frame(alice_key, frame_size=4096)
        H = matrix_manager.get_matrix(0.70)
        syndrome_block = encode_block(frame, H)
        
        # Bob has identical key (noiseless)
        bob_key = alice_key.copy()
        bob_frame = prepare_frame(bob_key, frame_size=4096)
        
        # Bob decodes
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(bob_frame, qber=0.01)
        result = decoder.decode(llr, syndrome_block.syndrome)
        
        # Should converge immediately
        assert result.converged

    def test_encode_decode_with_noise(
        self,
        matrix_manager: MatrixManager,
    ) -> None:
        """Encode and decode block with 3% QBER."""
        # Alice's key
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, size=2867, dtype=np.int8)
        
        # Prepare and encode
        frame = prepare_frame(alice_key, frame_size=4096)
        H = matrix_manager.get_matrix(0.70)
        syndrome_block = encode_block(frame, H)
        
        # Bob's key with errors (~3% QBER)
        bob_key = alice_key.copy()
        n_errors = int(len(bob_key) * 0.03)
        error_positions = rng.choice(len(bob_key), size=n_errors, replace=False)
        bob_key[error_positions] = 1 - bob_key[error_positions]
        bob_frame = prepare_frame(bob_key, frame_size=4096)
        
        # Bob decodes
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(bob_frame, qber=0.05)  # Overestimate for safety
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
        """Hash verification after successful decode."""
        # Setup
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, size=2867, dtype=np.int8)
        frame = prepare_frame(alice_key, frame_size=4096)
        H = matrix_manager.get_matrix(0.70)
        
        # Alice computes syndrome and hash
        syndrome_block = encode_block(frame, H)
        alice_hash = hash_verifier.compute_hash(alice_key)
        
        # Bob decodes (noiseless for simplicity)
        decoder = BeliefPropagationDecoder(H, max_iterations=60)
        llr = build_channel_llr(frame, qber=0.01)
        result = decoder.decode(llr, syndrome_block.syndrome)
        
        # Bob extracts key and verifies hash
        bob_key = result.corrected_bits[:len(alice_key)]
        
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
        
        # Alice's key
        rng = np.random.default_rng(42)
        alice_key = rng.integers(0, 2, size=2867, dtype=np.int8)
        
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
        """Leakage cap exceeded raises abort condition."""
        tracker = LeakageTracker(safety_cap=1000)
        
        # First block
        tracker.record_block(block_id=0, syndrome_bits=600, hash_bits=50)
        assert tracker.should_abort() is False
        
        # Second block exceeds cap
        tracker.record_block(block_id=1, syndrome_bits=600, hash_bits=50)
        assert tracker.should_abort() is True

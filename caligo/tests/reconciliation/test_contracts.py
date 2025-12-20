"""Contract tests for reconciliation components.

These tests focus on invariants and cross-component contracts that keep
Alice–Bob synchronization stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.ldpc_decoder import BeliefPropagationDecoder, build_channel_llr
from caligo.reconciliation.ldpc_encoder import compute_syndrome, prepare_frame
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import (
    ReconciliationOrchestrator,
    ReconciliationOrchestratorConfig,
)
from caligo.types.exceptions import LeakageBudgetExceeded


@pytest.fixture(scope="module")
def matrix_manager() -> MatrixManager:
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


@pytest.mark.integration
def test_syndrome_linearity(matrix_manager: MatrixManager) -> None:
    H = matrix_manager.get_matrix(0.70)
    n = int(H.shape[1])
    rng = np.random.default_rng(123)
    x = rng.integers(0, 2, size=n, dtype=np.uint8)
    y = rng.integers(0, 2, size=n, dtype=np.uint8)

    sx = compute_syndrome(x, H)
    sy = compute_syndrome(y, H)
    sxy = compute_syndrome(np.bitwise_xor(x, y), H)

    np.testing.assert_array_equal(sxy, np.bitwise_xor(sx, sy))


@pytest.mark.integration
def test_decoder_converged_implies_syndrome_match(matrix_manager: MatrixManager) -> None:
    """Test that successful decoder convergence implies syndrome match."""
    # Use mother code (rate 0.5) with pattern for rate 0.70
    H = matrix_manager.get_matrix(0.5)
    compiled = matrix_manager.get_compiled(0.5)
    decoder = BeliefPropagationDecoder(parity_check_matrix=H, max_iterations=40)
    
    # Get pattern (skip if not available)
    pattern = matrix_manager.get_puncture_pattern(0.70)
    if pattern is None:
        pytest.skip("Puncture pattern for rate 0.70 not available")

    n = int(H.shape[1])
    m = int(H.shape[0])
    # Payload is non-punctured positions
    n_payload = int((pattern == 0).sum())
    payload = np.zeros(n_payload, dtype=np.uint8)
    target = np.zeros(m, dtype=np.uint8)
    llr = build_channel_llr(payload, qber=0.01, punctured_mask=pattern)

    res = decoder.decode(llr, target, H=compiled)
    if res.converged:
        assert compiled.count_syndrome_errors(res.corrected_bits, target) == 0
    # Note: With punctured patterns, perfect convergence may not always occur


def test_edge_qber_does_not_crash(matrix_manager: MatrixManager) -> None:
    """Test that edge QBER values don't crash LLR computation."""
    # Get pattern (use rate 0.70 as reference)
    pattern = matrix_manager.get_puncture_pattern(0.70)
    if pattern is None:
        pytest.skip("Puncture pattern for rate 0.70 not available")
    
    # Use frame size matching the pattern
    frame_size = len(pattern)
    n_punctured = int(pattern.sum())
    payload_len = frame_size - n_punctured
    bits = np.zeros(payload_len, dtype=np.uint8)

    llr0 = build_channel_llr(bits, qber=0.0, punctured_mask=pattern)
    assert llr0.shape == (frame_size,)

    llr11 = build_channel_llr(bits, qber=constants.QBER_RECONCILIATION_LIMIT, punctured_mask=pattern)
    assert llr11.shape == (frame_size,)


@pytest.mark.integration
def test_leakage_budget_exceeded_aborts(matrix_manager: MatrixManager) -> None:
    """High leakage cap constraint triggers abort."""
    rng = np.random.default_rng(7)
    # Use full frame for rate 0.5 (no puncturing)
    payload_len = 4096
    alice = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
    bob = alice.copy()

    # Use a tiny leakage budget that is guaranteed to be exceeded by the first block.
    tracker = LeakageTracker(safety_cap=0)
    orch = ReconciliationOrchestrator(
        matrix_manager=matrix_manager,
        leakage_tracker=tracker,
        config=ReconciliationOrchestratorConfig(frame_size=4096, max_iterations=10, max_retries=0),
    )

    with pytest.raises(LeakageBudgetExceeded):
        orch.reconcile_block(alice, bob, qber_estimate=0.01, block_id=0)

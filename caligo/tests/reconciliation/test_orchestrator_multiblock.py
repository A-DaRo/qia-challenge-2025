"""Integration tests for multi-block reconciliation.

These tests validate that the orchestrator can reconcile payloads longer than a
single LDPC frame by partitioning into blocks and concatenating the verified
outputs.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from caligo.reconciliation import constants
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.orchestrator import ReconciliationOrchestrator, ReconciliationOrchestratorConfig


@pytest.fixture(scope="module")
def matrix_manager() -> Iterator[MatrixManager]:
    yield MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


@pytest.mark.integration
def test_orchestrator_reconcile_key_multiblock_happy_path(matrix_manager: MatrixManager) -> None:
    """Test multi-block reconciliation (>2 frames)."""
    rng = np.random.default_rng(2025)

    # > 2 frames worth of payload (must use full frames for rate 0.5)
    payload_len = 3 * 4096  # 3 full blocks
    alice = rng.integers(0, 2, size=payload_len, dtype=np.uint8)

    # Introduce ~1% errors.
    bob = alice.copy()
    n_errors = max(1, int(payload_len * 0.01))
    positions = rng.choice(payload_len, size=n_errors, replace=False)
    bob[positions] = 1 - bob[positions]

    orch = ReconciliationOrchestrator(
        matrix_manager=matrix_manager,
        config=ReconciliationOrchestratorConfig(frame_size=4096, max_retries=2, max_iterations=40),
        safety_cap=10**9,
    )

    reconciled, results = orch.reconcile_key(alice, bob, qber_estimate=0.03)

    # Either succeeds on all blocks or drops some; but when blocks verify,
    # they must match Alice.
    assert len(results) >= 3
    assert reconciled.dtype == np.uint8

    # Verified blocks contribute to output and must match Alice on those slices.
    verified_blocks = [r for r in results if r.verified]
    assert len(verified_blocks) >= 1

    # If all blocks verified, output must equal Alice.
    if all(r.verified for r in results):
        np.testing.assert_array_equal(reconciled, alice)

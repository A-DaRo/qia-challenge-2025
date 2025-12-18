"""Integration tests for reconciliation factory wiring.

Implements REQ-FINT-001 from docs/caligo/extended_test_spec.md.

These tests are intentionally non-E2E, but they *do* use the existing
LDPC assets (same as other reconciliation integration tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from caligo.reconciliation.factory import ReconciliationConfig, ReconciliationType, create_reconciler


@pytest.mark.integration
def test_fint_001_blind_factory_reconciler_runs_one_block_when_assets_exist() -> None:
    """REQ-FINT-001: create_reconciler(BLIND) can reconcile one block with assets."""

    cfg = ReconciliationConfig(
        reconciliation_type=ReconciliationType.BLIND,
        frame_size=4096,
        max_iterations=20,
        max_blind_rounds=2,
    )

    reconciler = create_reconciler(cfg)

    rng = np.random.default_rng(4242)
    payload_len = 2867
    alice = rng.integers(0, 2, size=payload_len, dtype=np.uint8)

    bob = alice.copy()
    n_errors = max(1, int(payload_len * 0.02))
    positions = rng.choice(payload_len, size=n_errors, replace=False)
    bob[positions] = 1 - bob[positions]

    corrected_bytes, meta = reconciler.reconcile(alice.tobytes(), bob.tobytes())

    assert isinstance(corrected_bytes, (bytes, bytearray))
    assert meta["reconciliation_type"] == "blind"
    assert meta["qber_estimation_required"] is False
    assert meta["status"] in {"success", "failed"}

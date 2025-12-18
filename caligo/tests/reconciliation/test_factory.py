"""Unit tests for reconciliation factory.

Implements the requirements in docs/caligo/extended_test_spec.md (Section 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from caligo.reconciliation.factory import (
    BaselineReconciler,
    BlindReconciler,
    ReconciliationConfig,
    ReconciliationType,
    create_reconciler,
    create_reconciler_from_yaml,
)
from caligo.simulation.noise_models import ChannelNoiseProfile


class TestReconciliationType:
    def test_req_fac_001_from_string_accepts_variants(self) -> None:
        assert ReconciliationType.from_string("baseline") is ReconciliationType.BASELINE
        assert ReconciliationType.from_string(" BASELINE ") is ReconciliationType.BASELINE
        assert ReconciliationType.from_string("Blind") is ReconciliationType.BLIND
        assert ReconciliationType.from_string("interactive") is ReconciliationType.INTERACTIVE

        assert ReconciliationType.BASELINE.requires_qber_estimation is True
        assert ReconciliationType.INTERACTIVE.requires_qber_estimation is True
        assert ReconciliationType.BLIND.requires_qber_estimation is False

    def test_req_fac_002_from_string_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match=r"Valid types"):
            ReconciliationType.from_string("foo")


class TestReconciliationConfig:
    def test_req_fac_010_post_init_rejects_invalid_bounds(self) -> None:
        with pytest.raises(ValueError, match=r"frame_size"):
            ReconciliationConfig(frame_size=128)

        with pytest.raises(ValueError, match=r"max_iterations"):
            ReconciliationConfig(max_iterations=0)

        with pytest.raises(ValueError, match=r"target_rate"):
            ReconciliationConfig(target_rate=0.05)

        with pytest.raises(ValueError, match=r"safety_margin"):
            ReconciliationConfig(safety_margin=-0.1)

        with pytest.raises(ValueError, match=r"max_blind_rounds"):
            ReconciliationConfig(max_blind_rounds=0)

    def test_req_fac_011_from_dict_maps_type_and_preserves_extras(self) -> None:
        cfg = ReconciliationConfig.from_dict(
            {
                "type": "blind",
                "frame_size": 4096,
                "max_blind_rounds": 3,
                "unknown_param": 123,
            }
        )

        assert cfg.reconciliation_type is ReconciliationType.BLIND

        as_dict = cfg.to_dict()
        assert as_dict["type"] == "blind"
        assert as_dict["unknown_param"] == 123

    def test_req_fac_012_requires_and_skips_match_type(self) -> None:
        cfg_blind = ReconciliationConfig(reconciliation_type=ReconciliationType.BLIND)
        assert cfg_blind.requires_qber_estimation is False
        assert cfg_blind.skips_qber_estimation is True

        cfg_base = ReconciliationConfig(reconciliation_type=ReconciliationType.BASELINE)
        assert cfg_base.requires_qber_estimation is True
        assert cfg_base.skips_qber_estimation is False


class TestCreateReconciler:
    def test_req_fac_020_baseline_selected_and_requires_qber(self) -> None:
        cfg = ReconciliationConfig(reconciliation_type=ReconciliationType.BASELINE)
        reconciler = create_reconciler(cfg)
        assert isinstance(reconciler, BaselineReconciler)

        alice_bits = np.zeros(16, dtype=np.uint8).tobytes()
        bob_bits = np.zeros(16, dtype=np.uint8).tobytes()

        with pytest.raises(ValueError, match=r"requires measured QBER"):
            reconciler.reconcile(alice_bits, bob_bits)

    def test_req_fac_021_interactive_raises_not_implemented(self) -> None:
        cfg = ReconciliationConfig(reconciliation_type=ReconciliationType.INTERACTIVE)
        with pytest.raises(NotImplementedError):
            create_reconciler(cfg)

    def test_req_fac_022_blind_selected_and_metadata_says_no_qber_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = ReconciliationConfig(reconciliation_type=ReconciliationType.BLIND)
        reconciler = create_reconciler(cfg)
        assert isinstance(reconciler, BlindReconciler)

        @dataclass
        class _FakeBlockResult:
            corrected_payload: np.ndarray
            verified: bool
            converged: bool
            error_count: int
            syndrome_length: int

        class _FakeOrchestrator:
            def reconcile_block(self, alice_key, bob_key, qber_estimate, block_id=0):
                return _FakeBlockResult(
                    corrected_payload=alice_key.copy(),
                    verified=True,
                    converged=True,
                    error_count=0,
                    syndrome_length=123,
                )

        monkeypatch.setattr(reconciler, "_get_orchestrator", lambda: _FakeOrchestrator())

        alice_bits = np.zeros(64, dtype=np.uint8).tobytes()
        bob_bits = np.zeros(64, dtype=np.uint8).tobytes()

        corrected, meta = reconciler.reconcile(alice_bits, bob_bits)
        assert corrected == alice_bits
        assert meta["qber_estimation_required"] is False
        assert meta["status"] == "success"

    def test_req_fac_023_channel_profile_argument_is_accepted(self) -> None:
        cfg = ReconciliationConfig(reconciliation_type=ReconciliationType.BLIND)
        profile = ChannelNoiseProfile.perfect()
        reconciler = create_reconciler(cfg, channel_profile=profile)
        assert isinstance(reconciler, BlindReconciler)


class TestYamlHelpers:
    def test_req_fac_040_from_yaml_file_loads_reconciliation_section(self, tmp_path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            """
reconciliation:
  type: blind
  frame_size: 4096
  max_blind_rounds: 3
""".lstrip()
        )

        cfg = ReconciliationConfig.from_yaml_file(str(yaml_path))
        assert cfg.reconciliation_type is ReconciliationType.BLIND
        assert cfg.frame_size == 4096
        assert cfg.max_blind_rounds == 3

    def test_req_fac_041_create_reconciler_from_yaml_matches_manual(self, tmp_path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            """
reconciliation:
  type: baseline
  frame_size: 4096
""".lstrip()
        )

        reconciler_from_yaml = create_reconciler_from_yaml(str(yaml_path))
        manual_cfg = ReconciliationConfig.from_yaml_file(str(yaml_path))
        reconciler_manual = create_reconciler(manual_cfg)

        assert type(reconciler_from_yaml) is type(reconciler_manual)


class TestBlindGetOrchestratorWiring:
    def test_req_fac_030_get_orchestrator_wiring_uses_matrix_path_and_config_fields(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        cfg = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BLIND,
            frame_size=1024,
            max_iterations=7,
            max_blind_rounds=9,
            ldpc_matrix_path=str(tmp_path / "ldpc"),
        )
        reconciler = BlindReconciler(cfg)

        captured: dict = {}

        class _FakeMatrixManager:
            @classmethod
            def from_directory(cls, directory):
                captured["matrix_dir"] = str(directory)
                return "FAKE_MM"

        class _FakeLeakageTracker:
            def __init__(self, safety_cap: int):
                captured["safety_cap"] = safety_cap

        class _FakeOrchestrator:
            def __init__(self, matrix_manager=None, leakage_tracker=None, config=None, safety_cap=None):
                captured["matrix_manager"] = matrix_manager
                captured["leakage_tracker"] = leakage_tracker
                captured["orch_config"] = config
                captured["orch_safety_cap"] = safety_cap

        monkeypatch.setattr(
            "caligo.reconciliation.matrix_manager.MatrixManager",
            _FakeMatrixManager,
        )
        monkeypatch.setattr(
            "caligo.reconciliation.leakage_tracker.LeakageTracker",
            _FakeLeakageTracker,
        )
        monkeypatch.setattr(
            "caligo.reconciliation.orchestrator.ReconciliationOrchestrator",
            _FakeOrchestrator,
        )

        orch = reconciler._get_orchestrator()
        assert orch is not None

        assert captured["matrix_dir"].endswith("/ldpc")
        assert captured["matrix_manager"] == "FAKE_MM"
        assert captured["safety_cap"] == 0

        orch_cfg = captured["orch_config"]
        assert orch_cfg.frame_size == 1024
        assert orch_cfg.max_iterations == 7
        assert orch_cfg.max_retries == 9

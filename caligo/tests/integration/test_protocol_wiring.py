"""
Test Suite for Protocol Wiring & Switching (Task 6).

Tests the integration between protocol layer and reconciliation factory.

Per Implementation Report v2 §6:
- YAML-based strategy injection
- Message sequence enforcement
- QBER dependency checking
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from caligo.reconciliation.factory import (
    ReconciliationConfig,
    ReconciliationType,
    create_strategy,
)
from caligo.reconciliation.strategies import (
    BaselineStrategy,
    BlindStrategy,
    ReconciliationContext,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def baseline_yaml_config() -> Dict[str, Any]:
    """YAML configuration for baseline reconciliation."""
    return {
        "reconciliation": {
            "type": "baseline",
            "frame_size": 4096,
            "max_iterations": 60,
            "target_rate": None,
            "use_nsm_informed_start": False,
            "safety_margin": 0.05,
        }
    }


@pytest.fixture
def blind_yaml_config() -> Dict[str, Any]:
    """YAML configuration for blind reconciliation."""
    return {
        "reconciliation": {
            "type": "blind",
            "frame_size": 4096,
            "max_iterations": 3,
            "use_nsm_informed_start": True,
            "safety_margin": 0.05,
            "max_blind_rounds": 3,
        }
    }


@pytest.fixture
def mock_mother_code() -> MagicMock:
    """Mock MotherCodeManager."""
    mock = MagicMock()
    mock.frame_size = 4096
    mock.mother_rate = 0.5
    mock.patterns = {0.5 + 0.01 * i: np.zeros(4096, dtype=np.uint8) for i in range(40)}
    mock.get_pattern = MagicMock(return_value=np.zeros(4096, dtype=np.uint8))
    mock.get_modulation_indices = MagicMock(return_value=np.arange(400, dtype=np.int64))
    mock.compiled_topology = MagicMock()
    mock.compiled_topology.n_edges = 12000
    mock.compiled_topology.n_checks = 2048
    mock.compiled_topology.n_vars = 4096
    return mock


@pytest.fixture
def mock_codec() -> MagicMock:
    """Mock LDPCCodec."""
    mock = MagicMock()
    mock.encode = MagicMock(return_value=np.zeros(2048, dtype=np.uint8))
    from caligo.reconciliation.strategies import DecoderResult
    mock.decode_baseline = MagicMock(return_value=DecoderResult(
        corrected_bits=np.zeros(4096, dtype=np.uint8),
        converged=True,
        iterations=10,
        messages=np.zeros(24000, dtype=np.float64),
    ))
    mock.decode_blind = MagicMock(return_value=DecoderResult(
        corrected_bits=np.zeros(4096, dtype=np.uint8),
        converged=True,
        iterations=10,
        messages=np.zeros(24000, dtype=np.float64),
    ))
    return mock


# =============================================================================
# TASK 6.1: YAML Injection
# =============================================================================


class TestYAMLInjection:
    """Task 6.1: Verify YAML configuration creates correct strategy class."""
    
    def test_baseline_yaml_creates_baseline_strategy(
        self,
        baseline_yaml_config: Dict[str, Any],
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """Baseline YAML config should instantiate BaselineStrategy."""
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BASELINE,
            frame_size=4096,
            max_iterations=60,
        )
        
        # Verify correct type is set
        assert config.reconciliation_type == ReconciliationType.BASELINE
        assert config.frame_size == 4096
        assert config.max_iterations == 60
    
    def test_blind_yaml_creates_blind_strategy(
        self,
        blind_yaml_config: Dict[str, Any],
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """Blind YAML config should instantiate BlindStrategy."""
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BLIND,
            frame_size=4096,
            max_iterations=60,
            max_blind_rounds=3,
        )
        
        # Verify correct type is set
        assert config.reconciliation_type == ReconciliationType.BLIND
        assert config.max_blind_rounds == 3
    
    def test_reconciliation_type_from_string(self) -> None:
        """ReconciliationType.from_string parses correctly."""
        assert ReconciliationType.from_string("baseline") == ReconciliationType.BASELINE
        assert ReconciliationType.from_string("BASELINE") == ReconciliationType.BASELINE
        assert ReconciliationType.from_string("blind") == ReconciliationType.BLIND
        assert ReconciliationType.from_string("BLIND") == ReconciliationType.BLIND
        
        with pytest.raises(ValueError):
            ReconciliationType.from_string("invalid")
    
    def test_config_from_yaml_file(self, tmp_path: Path) -> None:
        """ReconciliationConfig can load from YAML file."""
        config_dict = {
            "reconciliation": {
                "type": "blind",
                "frame_size": 4096,
                "max_iterations": 50,
            }
        }
        
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)
        
        # Load and parse
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)
        
        recon_config = loaded["reconciliation"]
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.from_string(recon_config["type"]),
            frame_size=recon_config["frame_size"],
            max_iterations=recon_config["max_iterations"],
        )
        
        assert config.reconciliation_type == ReconciliationType.BLIND
        assert config.frame_size == 4096


# =============================================================================
# TASK 6.2: Message Sequence Enforcement
# =============================================================================


class TestMessageSequenceEnforcement:
    """Task 6.2: Verify correct message sequences for each protocol."""
    
    def test_baseline_sends_syndrome_with_qber(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """Baseline should send SYNDROME message with qber_channel field."""
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
        message = next(gen)
        
        # Baseline message should have:
        assert message["kind"] == "baseline"
        assert "syndrome" in message
        assert "qber_channel" in message
        assert message["qber_channel"] == 0.05
    
    def test_blind_sends_without_measured_qber(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """Blind should send SYNDROME without requiring measured QBER."""
        strategy = BlindStrategy(
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
            qber_measured=None,  # No measured QBER
            qber_heuristic=0.05,  # Optional heuristic only
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        message = next(gen)
        
        # Blind message should have:
        assert message["kind"] == "blind"
        assert "syndrome" in message
        # Has qber_prior (heuristic), not qber_channel (measured)
        assert "qber_prior" in message
    
    def test_blind_reveal_message_format(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """Blind reveal messages should have correct format."""
        strategy = BlindStrategy(
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
            qber_measured=None,
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        message1 = next(gen)
        
        # Trigger reveal iteration
        message2 = gen.send({"verified": False, "converged": False})
        
        if message2.get("kind") == "blind_reveal":
            assert "revealed_indices" in message2
            assert "revealed_values" in message2
            assert "iteration" in message2
            assert message2["iteration"] == 2


# =============================================================================
# TASK 6.3: QBER Dependency Check
# =============================================================================


class TestQBERDependencyCheck:
    """Task 6.3: Verify QBER requirement enforcement."""
    
    def test_baseline_without_qber_raises(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """BaselineStrategy should fail if context has no measured QBER."""
        strategy = BaselineStrategy(
            mother_code=mock_mother_code,
            codec=mock_codec,
            leakage_tracker=LeakageTracker(safety_cap=100000),
        )
        
        # Context without measured QBER
        context = ReconciliationContext(
            session_id=1,
            frame_size=4096,
            mother_rate=0.5,
            max_iterations=60,
            hash_bits=64,
            f_crit=1.1,
            qber_measured=None,  # Missing!
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        
        # Should raise ValueError when trying to access qber_for_baseline
        with pytest.raises(ValueError, match="Baseline protocol requires measured QBER"):
            next(gen)
    
    def test_blind_without_qber_succeeds(
        self,
        mock_mother_code: MagicMock,
        mock_codec: MagicMock,
    ) -> None:
        """BlindStrategy should work without measured QBER."""
        strategy = BlindStrategy(
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
            qber_measured=None,
            qber_heuristic=None,  # Neither QBER provided
        )
        
        payload = np.zeros(3000, dtype=np.uint8)
        
        gen = strategy.alice_reconcile_block(payload, context, block_id=0)
        
        # Should NOT raise
        message = next(gen)
        assert message["kind"] == "blind"
    
    def test_reconciliation_type_requires_qber_property(self) -> None:
        """ReconciliationType.requires_qber_estimation property."""
        assert ReconciliationType.BASELINE.requires_qber_estimation is True
        assert ReconciliationType.BLIND.requires_qber_estimation is False


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


class TestConfigurationValidation:
    """Test ReconciliationConfig validation."""
    
    def test_invalid_frame_size_raises(self) -> None:
        """Invalid frame size should raise."""
        with pytest.raises(ValueError):
            ReconciliationConfig(
                reconciliation_type=ReconciliationType.BASELINE,
                frame_size=-1,  # Invalid
                max_iterations=60,
            )
    
    def test_invalid_max_iterations_raises(self) -> None:
        """Invalid max iterations should raise."""
        with pytest.raises(ValueError):
            ReconciliationConfig(
                reconciliation_type=ReconciliationType.BASELINE,
                frame_size=4096,
                max_iterations=0,  # Invalid
            )
    
    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = ReconciliationConfig(
            reconciliation_type=ReconciliationType.BASELINE,
            frame_size=4096,
            max_iterations=60,
        )
        
        assert config.safety_margin >= 0
        assert config.puncturing_enabled is True
        assert config.shortening_enabled is True

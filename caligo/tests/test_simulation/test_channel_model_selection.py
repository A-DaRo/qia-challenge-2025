"""
Unit tests for ChannelModelSelection dataclass in physical_model.py.

Tests link model resolution logic and configuration generation.
"""

from __future__ import annotations

import pytest

from caligo.simulation.physical_model import (
    ChannelModelSelection,
    NSMParameters,
    ChannelParameters,
)
from caligo.types.exceptions import InvalidParameterError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nsm_params_basic() -> NSMParameters:
    """Basic NSM parameters for testing."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.95,
    )


@pytest.fixture
def nsm_params_full() -> NSMParameters:
    """NSM parameters with full physical model fields."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.95,
        detection_eff_eta=0.1,
        detector_error=0.005,
        dark_count_prob=1e-7,
    )


@pytest.fixture
def nsm_params_minimal() -> NSMParameters:
    """NSM parameters with perfect detection (ideal model)."""
    return NSMParameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
        channel_fidelity=0.90,
        detection_eff_eta=1.0,
        detector_error=0.0,
        dark_count_prob=0.0,
    )


@pytest.fixture
def channel_params() -> ChannelParameters:
    """Standard channel parameters."""
    return ChannelParameters(
        length_km=0.0,
        attenuation_db_per_km=0.2,
        t1_ns=20_000_000,
        t2_ns=2_000_000,
    )


# =============================================================================
# ChannelModelSelection Instantiation Tests
# =============================================================================


class TestChannelModelSelectionInstantiation:
    """Tests for ChannelModelSelection dataclass instantiation."""

    def test_default_values(self) -> None:
        """Default values should be 'auto' and 'detector_only'."""
        cms = ChannelModelSelection()
        assert cms.link_model == "auto"
        assert cms.eta_semantics == "detector_only"

    def test_explicit_model_values(self) -> None:
        """Explicit model values should be accepted."""
        cms = ChannelModelSelection(
            link_model="heralded-double-click",
            eta_semantics="end_to_end",  # Correct value
        )
        assert cms.link_model == "heralded-double-click"
        assert cms.eta_semantics == "end_to_end"

    def test_invalid_link_model_rejected(self) -> None:
        """Invalid link_model values should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match="link_model"):
            ChannelModelSelection(link_model="invalid_model")

    def test_invalid_eta_semantics_rejected(self) -> None:
        """Invalid eta_semantics values should raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match="eta_semantics"):
            ChannelModelSelection(eta_semantics="invalid_semantics")

    def test_all_valid_link_models(self) -> None:
        """All valid link_model values should be accepted."""
        valid_models = ["auto", "perfect", "depolarise", "heralded-double-click"]
        for model in valid_models:
            cms = ChannelModelSelection(link_model=model)
            assert cms.link_model == model

    def test_all_valid_eta_semantics(self) -> None:
        """All valid eta_semantics values should be accepted."""
        valid_semantics = ["detector_only", "end_to_end"]
        for semantic in valid_semantics:
            cms = ChannelModelSelection(eta_semantics=semantic)
            assert cms.eta_semantics == semantic


# =============================================================================
# resolve_link_model Tests
# =============================================================================


class TestResolveLinkModel:
    """Tests for resolve_link_model method."""

    def test_explicit_perfect_returns_perfect(
        self, nsm_params_full: NSMParameters
    ) -> None:
        """Explicit 'perfect' should return 'perfect'."""
        cms = ChannelModelSelection(link_model="perfect")
        result = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )
        assert result == "perfect"

    def test_explicit_depolarise_returns_depolarise(
        self, nsm_params_full: NSMParameters
    ) -> None:
        """Explicit 'depolarise' should return 'depolarise'."""
        cms = ChannelModelSelection(link_model="depolarise")
        result = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )
        assert result == "depolarise"

    def test_explicit_heralded_returns_heralded(
        self, nsm_params_full: NSMParameters
    ) -> None:
        """Explicit 'heralded-double-click' should return 'heralded-double-click'."""
        cms = ChannelModelSelection(link_model="heralded-double-click")
        result = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )
        assert result == "heralded-double-click"

    def test_auto_with_full_params_returns_heralded(
        self, nsm_params_full: NSMParameters
    ) -> None:
        """Auto resolution with η<1.0 should return 'heralded-double-click'."""
        cms = ChannelModelSelection(link_model="auto")
        result = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,  # 0.1 < 1.0
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )
        assert result == "heralded-double-click"

    def test_auto_without_eta_returns_depolarise(
        self, nsm_params_minimal: NSMParameters
    ) -> None:
        """Auto resolution with η=1.0 and no dark counts should return 'depolarise'."""
        cms = ChannelModelSelection(link_model="auto")
        result = cms.resolve_link_model(
            channel_fidelity=nsm_params_minimal.channel_fidelity,  # 0.90
            detection_eff_eta=nsm_params_minimal.detection_eff_eta,  # 1.0
            dark_count_prob=nsm_params_minimal.dark_count_prob,  # 0.0
            detector_error=nsm_params_minimal.detector_error,  # 0.0
        )
        assert result == "depolarise"

    def test_auto_with_perfect_fidelity_returns_perfect(self) -> None:
        """Auto resolution with all-perfect params should return 'perfect'."""
        cms = ChannelModelSelection(link_model="auto")
        result = cms.resolve_link_model(
            channel_fidelity=1.0,
            detection_eff_eta=1.0,
            dark_count_prob=0.0,
            detector_error=0.0,
        )
        assert result == "perfect"


# =============================================================================
# Integration Tests
# =============================================================================


class TestChannelModelSelectionIntegration:
    """Integration tests for ChannelModelSelection with network building."""

    def test_model_selection_round_trip(self) -> None:
        """ChannelModelSelection should be serializable and reconstructable."""
        original = ChannelModelSelection(
            link_model="heralded-double-click",
            eta_semantics="end_to_end",
        )

        # Simulate YAML round-trip
        data = {
            "link_model": original.link_model,
            "eta_semantics": original.eta_semantics,
        }
        reconstructed = ChannelModelSelection(**data)

        assert reconstructed.link_model == original.link_model
        assert reconstructed.eta_semantics == original.eta_semantics

    def test_model_resolution_consistency(
        self, nsm_params_full: NSMParameters
    ) -> None:
        """Multiple calls to resolve_link_model should return consistent results."""
        cms = ChannelModelSelection(link_model="auto")

        result1 = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )
        result2 = cms.resolve_link_model(
            channel_fidelity=nsm_params_full.channel_fidelity,
            detection_eff_eta=nsm_params_full.detection_eff_eta,
            dark_count_prob=nsm_params_full.dark_count_prob,
            detector_error=nsm_params_full.detector_error,
        )

        assert result1 == result2

    def test_frozen_dataclass(self) -> None:
        """ChannelModelSelection should be frozen (immutable)."""
        cms = ChannelModelSelection()
        with pytest.raises(AttributeError):
            cms.link_model = "depolarise"  # type: ignore

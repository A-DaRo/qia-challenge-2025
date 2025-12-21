"""Phase-boundary integration: Simulation noise models → reconciliation rate selection.

Covers REQ-P0R-001/010 from the extended test spec.

This test validates *oracle consistency* between:
- `caligo.simulation.noise_models.ChannelNoiseProfile`
- `caligo.simulation.physical_model.NSMParameters`
- reconciliation's discrete LDPC rate constants

No SquidASM/NetSquid simulation is involved. No legacy MatrixManager used.
"""

from __future__ import annotations

import pytest

from caligo.reconciliation import constants as recon_constants
from caligo.simulation.noise_models import ChannelNoiseProfile
from caligo.simulation.constants import QBER_HARD_LIMIT


@pytest.mark.integration
def test_p0r_001_profile_to_nsm_qber_and_rate_are_consistent() -> None:
    """REQ-P0R-001: total_qber should match NSM qber_channel and map to a known rate."""

    profile = ChannelNoiseProfile(
        source_fidelity=0.98,
        detector_efficiency=0.90,
        detector_error=0.01,
        dark_count_rate=1e-6,
        transmission_loss=0.0,
    )

    nsm = profile.to_nsm_parameters(
        storage_noise_r=0.75,
        storage_rate_nu=0.002,
        delta_t_ns=1_000_000,
    )

    assert profile.total_qber == pytest.approx(nsm.qber_channel, abs=1e-15)

    suggested = profile.suggested_ldpc_rate(safety_margin=0.0)
    # Check against the constant set of LDPC rates
    assert suggested in recon_constants.LDPC_CODE_RATES


@pytest.mark.integration
def test_p0r_010_infeasible_profiles_are_flagged() -> None:
    """REQ-P0R-010: total_qber >= QBER_HARD_LIMIT must be treated as infeasible."""

    # Low fidelity implies Q_source ~= (1-F)/2 near 0.25, above the hard limit.
    profile = ChannelNoiseProfile(
        source_fidelity=0.5001,
        detector_efficiency=1.0,
        detector_error=0.0,
        dark_count_rate=0.0,
        transmission_loss=0.0,
    )

    assert profile.total_qber >= QBER_HARD_LIMIT
    assert profile.is_feasible is False

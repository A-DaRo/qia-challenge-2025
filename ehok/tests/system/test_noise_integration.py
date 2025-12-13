"""
System Integration Tests: Noise Adapter.

Test Cases
----------
SYS-INT-NOISE-001: PhysicalModelAdapter Fidelity Mapping
SYS-INT-NOISE-002: Storage Noise r Derivation from T1/T2

Reference
---------
System Test Specification §2.1 (GAP: NOISE-PARAMS-001)
"""

import math
import pytest

# Standard library
from typing import Any, Optional

# ============================================================================
# Attempt to import required modules - let ImportError happen if missing
# ============================================================================

# E-HOK modules under test
from ehok.quantum.noise_adapter import (
    SimulatorNoiseParams,
    physical_to_simulator,
)

# PhysicalModelAdapter - this is the spec-required class that may not exist
try:
    from ehok.quantum.noise_adapter import PhysicalModelAdapter
except ImportError:
    PhysicalModelAdapter = None  # type: ignore

# T1/T2 storage noise derivation function - spec-required
try:
    from ehok.quantum.noise_adapter import estimate_storage_noise_from_netsquid
except ImportError:
    estimate_storage_noise_from_netsquid = None  # type: ignore

# Physical parameters configuration
from ehok.configs.protocol_config import PhysicalParameters

# NetSquid/SquidASM imports for white-box inspection
try:
    import netsquid as ns
    from netsquid.components.qchannel import QuantumChannel
    NETSQUID_AVAILABLE = True
except ImportError:
    ns = None  # type: ignore
    QuantumChannel = None  # type: ignore
    NETSQUID_AVAILABLE = False

try:
    from squidasm.sim.stack.config import StackNetworkConfig, Link, NoiseType
    from squidasm.sim.stack.program import ProgramContext
    from squidasm.run.stack.run import run
    SQUIDASM_AVAILABLE = True
except ImportError:
    StackNetworkConfig = None  # type: ignore
    Link = None  # type: ignore
    NoiseType = None  # type: ignore
    SQUIDASM_AVAILABLE = False


# ============================================================================
# Test Constants (from spec)
# ============================================================================

# SYS-INT-NOISE-001 parameters (from spec pre-conditions)
NSM_MU = 3.145e-5  # Source quality
NSM_ETA = 0.0150   # Detection efficiency
NSM_E_DET = 0.0093  # Intrinsic error rate

# SYS-INT-NOISE-002 parameters (from spec pre-conditions)
T1_NS = 1e9   # Amplitude damping time (1 second)
T2_NS = 5e8   # Dephasing time (0.5 seconds)
DELTA_T_NS = 1e9  # Wait time (1 second)

# Tolerance for fidelity comparison
FIDELITY_TOLERANCE = 1e-6
STORAGE_NOISE_TOLERANCE = 1e-4


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def physical_params_nsm() -> PhysicalParameters:
    """Create PhysicalParameters with NSM-specified values."""
    return PhysicalParameters(
        mu_pair_per_coherence=NSM_MU,
        eta_total_transmittance=NSM_ETA,
        e_det=NSM_E_DET,
    )


@pytest.fixture
def expected_link_fidelity() -> float:
    """
    Calculate expected link fidelity from NSM formula.
    
    F_expected = 1 - (μ + (1-η)·0.5 + e_det)
    
    Per spec: This is the theoretical fidelity derived from NSM parameters.
    """
    # Per spec formula:
    F_expected = 1.0 - (NSM_MU + (1 - NSM_ETA) * 0.5 + NSM_E_DET)
    return F_expected


# ============================================================================
# SYS-INT-NOISE-001: PhysicalModelAdapter Fidelity Mapping
# ============================================================================

class TestNoiseAdapterFidelityMapping:
    """
    Test Case ID: SYS-INT-NOISE-001
    Title: Verify PhysicalModelAdapter correctly translates NSM parameters to NetSquid Link fidelity
    Priority: CRITICAL
    Traces To: GAP: NOISE-PARAMS-001, REQ: PHI-R1 (Pre-flight feasibility)
    """

    def test_physical_model_adapter_exists(self):
        """
        Verify PhysicalModelAdapter class exists.
        
        Spec Requirement:
            "ehok/quantum/noise_adapter.py is configured with NSM parameters"
        """
        assert PhysicalModelAdapter is not None, (
            "MISSING: PhysicalModelAdapter class not found in ehok.quantum.noise_adapter. "
            "This is a CRITICAL GAP - the spec requires this class for NSM parameter translation."
        )

    @pytest.mark.skipif(PhysicalModelAdapter is None, 
                       reason="PhysicalModelAdapter not implemented")
    def test_adapter_accepts_nsm_parameters(self, physical_params_nsm):
        """
        Verify adapter can be configured with NSM parameters (μ, η, e_det).
        
        Spec Logic Step 1: "Configure PhysicalModelAdapter with NSM parameters"
        """
        # Attempt to instantiate the adapter
        adapter = PhysicalModelAdapter(
            mu=NSM_MU,
            eta=NSM_ETA,
            e_det=NSM_E_DET,
        )
        
        assert adapter is not None, "PhysicalModelAdapter instantiation failed"

    @pytest.mark.skipif(not SQUIDASM_AVAILABLE, 
                       reason="SquidASM not available")
    @pytest.mark.skipif(PhysicalModelAdapter is None,
                       reason="PhysicalModelAdapter not implemented")
    def test_adapter_creates_squidasm_network_config(self, physical_params_nsm):
        """
        Verify adapter output can create a SquidASM network.
        
        Spec Logic Step 2: "Create SquidASM network using adapter's output configuration"
        """
        adapter = PhysicalModelAdapter(
            mu=NSM_MU,
            eta=NSM_ETA,
            e_det=NSM_E_DET,
        )
        
        # Adapter should produce configuration suitable for SquidASM
        network_config = adapter.to_squidasm_config()
        
        assert isinstance(network_config, StackNetworkConfig), (
            f"Expected StackNetworkConfig, got {type(network_config)}"
        )

    @pytest.mark.skipif(not NETSQUID_AVAILABLE,
                       reason="NetSquid not available")
    @pytest.mark.skipif(not SQUIDASM_AVAILABLE,
                       reason="SquidASM not available")
    @pytest.mark.skipif(PhysicalModelAdapter is None,
                       reason="PhysicalModelAdapter not implemented")
    def test_quantum_channel_noise_model_inspection(
        self, physical_params_nsm, expected_link_fidelity
    ):
        """
        Verify QuantumChannel noise model matches expected fidelity.
        
        Spec Logic Steps 3-6:
        3. INSPECT: Retrieve underlying netsquid.components.qchannel.QuantumChannel
        4. INSPECT: Query QuantumChannel.models to obtain noise model parameters
        5. COMPUTE: Calculate expected fidelity from NSM formula
        6. ASSERT: Configured noise parameters match expected fidelity
        
        This is the CRITICAL white-box inspection test.
        """
        adapter = PhysicalModelAdapter(
            mu=NSM_MU,
            eta=NSM_ETA,
            e_det=NSM_E_DET,
        )
        
        network_config = adapter.to_squidasm_config()
        
        # Build network and inspect internals
        # NOTE: This requires knowledge of SquidASM network internal structure
        network = adapter.build_network(network_config)
        
        # Spec inspection point:
        # "channel = network._get_quantum_channel("Alice", "Bob")"
        channel = network._get_quantum_channel("Alice", "Bob")
        assert isinstance(channel, QuantumChannel), (
            f"Expected QuantumChannel, got {type(channel)}"
        )
        
        # Spec inspection point:
        # "noise_model = channel.models.get("noise_model")"
        noise_model = channel.models.get("quantum_noise_model")
        assert noise_model is not None, (
            "FAIL: No noise model attached to quantum channel. "
            "Spec requires DepolarNoiseModel or compatible."
        )
        
        # Spec inspection point:
        # "actual_depolar_rate = noise_model.properties["prob_max_mixed"]"
        actual_depolar_rate = None
        if hasattr(noise_model, 'properties'):
            actual_depolar_rate = noise_model.properties.get("prob_max_mixed")
        elif hasattr(noise_model, 'depolar_rate'):
            actual_depolar_rate = noise_model.depolar_rate
        
        assert actual_depolar_rate is not None, (
            "FAIL: Could not extract depolarization rate from noise model"
        )
        
        # Spec assertion:
        # "assert abs(actual_depolar_rate - (1 - F_expected)) < 1e-6"
        expected_depolar_rate = 1.0 - expected_link_fidelity
        
        assert abs(actual_depolar_rate - expected_depolar_rate) < FIDELITY_TOLERANCE, (
            f"FAIL: Depolar rate mismatch. "
            f"Expected: {expected_depolar_rate:.6f}, Got: {actual_depolar_rate:.6f}. "
            f"Difference: {abs(actual_depolar_rate - expected_depolar_rate):.2e}"
        )

    def test_physical_to_simulator_translation_exists(self, physical_params_nsm):
        """
        Verify basic physical_to_simulator function works.
        
        This tests the existing translation function as baseline.
        """
        sim_params = physical_to_simulator(physical_params_nsm)
        
        assert isinstance(sim_params, SimulatorNoiseParams)
        assert 0 <= sim_params.link_fidelity <= 1
        assert 0 <= sim_params.measurement_bitflip_prob <= 0.5
        assert 0 <= sim_params.expected_detection_prob <= 1


# ============================================================================
# SYS-INT-NOISE-002: Storage Noise r Derivation from T1/T2
# ============================================================================

class TestStorageNoiseDerivation:
    """
    Test Case ID: SYS-INT-NOISE-002
    Title: Verify adversary storage noise r is correctly derived from NetSquid T1/T2 memory parameters
    Priority: HIGH
    Traces To: GAP: STORAGE-LINK-001, phase_IV_analysis.md §3.2
    """

    def test_storage_noise_function_exists(self):
        """
        Verify estimate_storage_noise_from_netsquid function exists.
        
        Spec Logic Step 2: "Call noise_adapter.estimate_storage_noise_from_netsquid(T1, T2, delta_t)"
        """
        assert estimate_storage_noise_from_netsquid is not None, (
            "MISSING: estimate_storage_noise_from_netsquid function not found. "
            "This is a HIGH priority GAP - required for T1/T2 to storage noise mapping."
        )

    @pytest.mark.skipif(estimate_storage_noise_from_netsquid is None,
                       reason="estimate_storage_noise_from_netsquid not implemented")
    def test_storage_noise_calculation(self):
        """
        Verify storage noise r is correctly computed from T1/T2/Δt.
        
        Spec Logic Steps 3-4:
        3. COMPUTE: Expected storage noise from decay model:
           decay_amp = exp(-Δt / T1)
           decay_phase = exp(-Δt / T2)
           F_storage = 0.5 * (1 + decay_amp * decay_phase)
           r_expected = 1 - F_storage
        4. ASSERT: Returned r equals r_expected within tolerance 1e-4
        
        Expected: r ≈ 0.565 for specified parameters
        """
        # Compute expected storage noise per spec formula
        decay_amp = math.exp(-DELTA_T_NS / T1_NS)  # exp(-1) ≈ 0.368
        decay_phase = math.exp(-DELTA_T_NS / T2_NS)  # exp(-2) ≈ 0.135
        
        F_storage = 0.5 * (1 + decay_amp * decay_phase)
        r_expected = 1 - F_storage
        
        # Verify expected is approximately 0.565 as per spec
        assert abs(r_expected - 0.565) < 0.01, (
            f"Test setup verification: Expected r ≈ 0.565, computed {r_expected:.4f}"
        )
        
        # Call the function under test
        r_actual = estimate_storage_noise_from_netsquid(T1_NS, T2_NS, DELTA_T_NS)
        
        # Spec assertion
        assert abs(r_actual - r_expected) < STORAGE_NOISE_TOLERANCE, (
            f"FAIL: Storage noise mismatch. "
            f"Expected r: {r_expected:.4f}, Got: {r_actual:.4f}. "
            f"Difference: {abs(r_actual - r_expected):.2e}"
        )

    @pytest.mark.skipif(estimate_storage_noise_from_netsquid is None,
                       reason="estimate_storage_noise_from_netsquid not implemented")
    def test_storage_noise_valid_range(self):
        """
        Verify storage noise r is in valid range [0, 1].
        
        Spec Expected State: "r is in valid range [0, 1]"
        """
        r = estimate_storage_noise_from_netsquid(T1_NS, T2_NS, DELTA_T_NS)
        
        assert 0 <= r <= 1, (
            f"FAIL: Storage noise r={r:.4f} outside valid range [0, 1]"
        )

    @pytest.mark.skipif(estimate_storage_noise_from_netsquid is None,
                       reason="estimate_storage_noise_from_netsquid not implemented")
    def test_storage_noise_extreme_cases(self):
        """
        Verify storage noise behavior at extreme T1/T2 values.
        """
        # Perfect storage (very long T1, T2) -> r should be close to 0
        r_perfect = estimate_storage_noise_from_netsquid(
            T1=1e15, T2=1e15, delta_t=DELTA_T_NS
        )
        assert r_perfect < 0.01, (
            f"FAIL: Perfect storage should give r ≈ 0, got r={r_perfect:.4f}"
        )
        
        # Complete decoherence (very short T1, T2) -> r should be close to 0.5
        # (depolarizing to maximally mixed state)
        r_complete = estimate_storage_noise_from_netsquid(
            T1=1, T2=1, delta_t=DELTA_T_NS
        )
        assert r_complete >= 0.4, (
            f"FAIL: Complete decoherence should give r ≈ 0.5, got r={r_complete:.4f}"
        )


# ============================================================================
# Additional Tests: SimulatorNoiseParams Validation
# ============================================================================

class TestSimulatorNoiseParamsValidation:
    """Additional validation tests for SimulatorNoiseParams dataclass."""

    def test_valid_params_accepted(self):
        """Verify valid parameters are accepted."""
        params = SimulatorNoiseParams(
            link_fidelity=0.95,
            measurement_bitflip_prob=0.01,
            expected_detection_prob=0.70,
        )
        assert params.link_fidelity == 0.95

    def test_invalid_fidelity_rejected(self):
        """Verify fidelity outside [0,1] is rejected."""
        with pytest.raises(ValueError, match="link_fidelity must be in"):
            SimulatorNoiseParams(
                link_fidelity=1.5,
                measurement_bitflip_prob=0.01,
                expected_detection_prob=0.70,
            )

    def test_invalid_bitflip_rejected(self):
        """Verify bitflip probability outside [0, 0.5] is rejected."""
        with pytest.raises(ValueError, match="measurement_bitflip_prob must be in"):
            SimulatorNoiseParams(
                link_fidelity=0.95,
                measurement_bitflip_prob=0.6,
                expected_detection_prob=0.70,
            )

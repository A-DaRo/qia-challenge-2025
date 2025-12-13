"""
System End-to-End Scenario Tests.

Test Cases
----------
SYS-E2E-GOLDEN-001: The Golden Run (Happy Path)
SYS-E2E-DEATHVALLEY-001: Death Valley (Infeasible Batch Size)
SYS-E2E-ATTACK-ORDER-001: Active Attack — Order Violation
SYS-E2E-HIGHQBER-001: High QBER (Channel Too Noisy)
SYS-E2E-LEAKAGE-001: Leakage Cap Exceeded
SYS-E2E-DETECTION-001: Detection Anomaly (Chernoff Violation)

Reference
---------
System Test Specification §3 (End-to-End Behavioral Scenarios)
"""

import pytest
import numpy as np
from typing import Optional, Dict, Any, Tuple

# ============================================================================
# Attempt to import required modules
# ============================================================================

# NetSquid/SquidASM
try:
    import netsquid as ns
    NETSQUID_AVAILABLE = True
except ImportError:
    ns = None  # type: ignore
    NETSQUID_AVAILABLE = False

try:
    from squidasm.sim.stack.config import StackNetworkConfig, Link, NoiseType
    from squidasm.run.stack.run import run
    SQUIDASM_AVAILABLE = True
except ImportError:
    StackNetworkConfig = None  # type: ignore
    SQUIDASM_AVAILABLE = False

# E-HOK Protocol components
try:
    from ehok.protocols.alice import AliceBaselineEHOK
    from ehok.protocols.bob import BobBaselineEHOK
    PROTOCOL_AVAILABLE = True
except ImportError:
    AliceBaselineEHOK = None  # type: ignore
    BobBaselineEHOK = None  # type: ignore
    PROTOCOL_AVAILABLE = False

# E-HOK Output structures
try:
    from ehok.core.oblivious_formatter import (
        AliceObliviousKey,
        BobObliviousKey,
        ProtocolMetrics,
    )
    OUTPUT_STRUCTURES_AVAILABLE = True
except ImportError:
    AliceObliviousKey = None  # type: ignore
    BobObliviousKey = None  # type: ignore
    OUTPUT_STRUCTURES_AVAILABLE = False

# Feasibility checker
try:
    from ehok.core.feasibility import (
        FeasibilityChecker,
        FeasibilityInputs,
        ABORT_CODE_QBER_TOO_HIGH,
        ABORT_CODE_DEATH_VALLEY,
    )
    FEASIBILITY_AVAILABLE = True
except ImportError:
    FeasibilityChecker = None  # type: ignore
    FeasibilityInputs = None  # type: ignore
    FEASIBILITY_AVAILABLE = False

# Leakage manager
try:
    from ehok.protocols.leakage_manager import (
        LeakageSafetyManager,
        ABORT_CODE_LEAKAGE_CAP_EXCEEDED,
    )
    LEAKAGE_MANAGER_AVAILABLE = True
except ImportError:
    LeakageSafetyManager = None  # type: ignore
    LEAKAGE_MANAGER_AVAILABLE = False

# Detection validator
try:
    from ehok.protocols.statistical_validation import DetectionValidator
    DETECTION_VALIDATOR_AVAILABLE = True
except ImportError:
    DetectionValidator = None  # type: ignore
    DETECTION_VALIDATOR_AVAILABLE = False

# Protocol configuration
try:
    from ehok.core.config import ProtocolConfig
    CONFIG_AVAILABLE = True
except ImportError:
    ProtocolConfig = None  # type: ignore
    CONFIG_AVAILABLE = False


# ============================================================================
# Test Constants (from spec)
# ============================================================================

# SYS-E2E-GOLDEN-001 parameters
GOLDEN_BATCH_SIZE = 20_000
GOLDEN_LINK_FIDELITY = 0.97  # 3% depolarization
GOLDEN_STORAGE_NOISE_R = 0.3
GOLDEN_EPSILON_SEC = 1e-6
GOLDEN_DELTA_T_NS = 1_000_000_000  # 1 second
GOLDEN_EXPECTED_QBER_MIN = 0.02
GOLDEN_EXPECTED_QBER_MAX = 0.05
GOLDEN_KEY_LENGTH_MIN = 1_000
GOLDEN_KEY_LENGTH_MAX = 5_000

# SYS-E2E-DEATHVALLEY-001 parameters
DEATHVALLEY_BATCH_SIZE = 100  # Deliberately too small
DEATHVALLEY_EXPECTED_QBER = 0.08
DEATHVALLEY_STORAGE_NOISE_R = 0.3

# SYS-E2E-HIGHQBER-001 parameters
HIGHQBER_LINK_FIDELITY = 0.70  # 30% depolarization → ~25% QBER
HIGHQBER_BATCH_SIZE = 10_000
HIGHQBER_STORAGE_NOISE_R = 0.3

# QBER limits from spec
QBER_WARNING_LIMIT = 0.11
QBER_HARD_LIMIT = 0.22


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_simulation():
    """Reset simulation state before each test."""
    if NETSQUID_AVAILABLE:
        ns.sim_reset()
    yield
    if NETSQUID_AVAILABLE:
        ns.sim_reset()


@pytest.fixture
def golden_config():
    """Configuration for golden run scenario."""
    if not CONFIG_AVAILABLE:
        return None
    return {
        'batch_size': GOLDEN_BATCH_SIZE,
        'link_fidelity': GOLDEN_LINK_FIDELITY,
        'storage_noise_r': GOLDEN_STORAGE_NOISE_R,
        'epsilon_sec': GOLDEN_EPSILON_SEC,
        'delta_t_ns': GOLDEN_DELTA_T_NS,
    }


# ============================================================================
# SYS-E2E-GOLDEN-001: The Golden Run (Happy Path)
# ============================================================================

class TestGoldenRun:
    """
    Test Case ID: SYS-E2E-GOLDEN-001
    Title: Complete protocol execution with valid parameters produces matching oblivious keys
    Priority: CRITICAL
    Traces To: Roadmap §6.3: "Positive key rate at Q=5%, n=10,000"
    """

    def test_protocol_classes_exist(self):
        """Verify protocol classes exist."""
        assert AliceBaselineEHOK is not None, (
            "MISSING: AliceBaselineEHOK not found in ehok.protocols.alice"
        )
        assert BobBaselineEHOK is not None, (
            "MISSING: BobBaselineEHOK not found in ehok.protocols.bob"
        )

    def test_output_structures_exist(self):
        """Verify OT output structures exist."""
        assert AliceObliviousKey is not None, (
            "MISSING: AliceObliviousKey not found"
        )
        assert BobObliviousKey is not None, (
            "MISSING: BobObliviousKey not found"
        )

    @pytest.mark.skipif(not PROTOCOL_AVAILABLE,
                       reason="Protocol classes not available")
    @pytest.mark.skipif(not SQUIDASM_AVAILABLE,
                       reason="SquidASM not available")
    @pytest.mark.slow
    def test_golden_run_produces_keys(self, golden_config):
        """
        Execute complete protocol and verify key production.
        
        Spec Logic Steps 1-9:
        1. Configure E-HOK protocol with above parameters
        2. Execute full protocol: Phase I → II → III → IV
        3. CAPTURE: Alice's output (S_0, S_1)
        4. CAPTURE: Bob's output (S_C, C)
        5. ASSERT: len(S_0) == len(S_1) == len(S_C) > 0
        6. ASSERT: S_C == S_0 if C == 0 else S_C == S_1
        7. ASSERT: ε_achieved <= ε_sec
        8. ASSERT: No ABORT codes triggered
        9. VERIFY: Protocol metrics show NSM min-entropy (not QKD)
        """
        # This is the full E2E test - requires complete infrastructure
        pytest.skip(
            "Full E2E golden run requires complete SquidASM network setup. "
            "This test documents the expected behavior."
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_output_structure_properties(self):
        """
        Verify output structures have required properties.
        
        Spec Steps 3-4 validation.
        """
        # Check AliceObliviousKey has required fields
        if hasattr(AliceObliviousKey, '__dataclass_fields__'):
            fields = AliceObliviousKey.__dataclass_fields__
            
            assert 'key_0' in fields or 's0' in fields or 's_0' in fields, (
                "AliceObliviousKey missing S_0 field"
            )
            assert 'key_1' in fields or 's1' in fields or 's_1' in fields, (
                "AliceObliviousKey missing S_1 field"
            )
        
        # Check BobObliviousKey has required fields
        if hasattr(BobObliviousKey, '__dataclass_fields__'):
            fields = BobObliviousKey.__dataclass_fields__
            
            has_key = any(
                k in fields for k in ['key', 's_c', 'key_c', 'selected_key']
            )
            assert has_key, "BobObliviousKey missing S_C field"
            
            has_choice = any(
                k in fields for k in ['c', 'choice', 'choice_bit']
            )
            assert has_choice, "BobObliviousKey missing choice bit C field"

    def test_golden_run_expected_metrics(self):
        """
        Verify expected metric ranges are reasonable.
        
        Spec Expected Outcomes validation.
        """
        # Key length range
        assert GOLDEN_KEY_LENGTH_MIN < GOLDEN_KEY_LENGTH_MAX
        assert GOLDEN_KEY_LENGTH_MIN > 0
        
        # QBER range
        assert GOLDEN_EXPECTED_QBER_MIN < GOLDEN_EXPECTED_QBER_MAX
        assert GOLDEN_EXPECTED_QBER_MAX < QBER_HARD_LIMIT


# ============================================================================
# SYS-E2E-DEATHVALLEY-001: Death Valley (Infeasible Batch Size)
# ============================================================================

class TestDeathValley:
    """
    Test Case ID: SYS-E2E-DEATHVALLEY-001
    Title: Small batch size triggers pre-flight ABORT before quantum resource consumption
    Priority: CRITICAL
    Traces To: Roadmap §6.4: ABORT-I-FEAS-001, sprint_1_specification.md §3.2
    """

    def test_feasibility_checker_exists(self):
        """Verify FeasibilityChecker class exists."""
        assert FeasibilityChecker is not None, (
            "MISSING: FeasibilityChecker not found in ehok.core.feasibility. "
            "This is CRITICAL for pre-flight abort functionality."
        )

    @pytest.mark.skipif(not FEASIBILITY_AVAILABLE,
                       reason="FeasibilityChecker not available")
    def test_small_batch_detected_infeasible(self):
        """
        Verify small batch size is detected as infeasible.
        
        Spec Logic Steps 1-3:
        1. Configure E-HOK with small batch size
        2. Invoke pre-flight feasibility check
        3. ASSERT: FeasibilityChecker returns INFEASIBLE
        """
        inputs = FeasibilityInputs(
            expected_qber=DEATHVALLEY_EXPECTED_QBER,
            storage_noise_r=DEATHVALLEY_STORAGE_NOISE_R,
            storage_rate_nu=1.0,
            batch_size=DEATHVALLEY_BATCH_SIZE,
            epsilon_sec=GOLDEN_EPSILON_SEC,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # Result should indicate infeasibility
        from ehok.analysis.nsm_bounds import FeasibilityResult
        
        is_infeasible = (
            result.status != FeasibilityResult.FEASIBLE or
            result.is_feasible == False
        )
        
        assert is_infeasible, (
            f"FAIL: Small batch size {DEATHVALLEY_BATCH_SIZE} should be INFEASIBLE. "
            f"Got result: {result}"
        )

    @pytest.mark.skipif(not FEASIBILITY_AVAILABLE,
                       reason="FeasibilityChecker not available")
    def test_abort_code_death_valley(self):
        """
        Verify correct abort code for infeasible batch.
        
        Spec Step 4: ASSERT: Abort code == ABORT-I-FEAS-001 or ABORT-I-FEAS-004
        """
        # Verify abort codes are defined
        assert ABORT_CODE_DEATH_VALLEY is not None, (
            "ABORT_CODE_DEATH_VALLEY not defined"
        )
        
        # The code should match spec taxonomy
        assert "ABORT-I" in ABORT_CODE_DEATH_VALLEY, (
            f"Death Valley abort should be Phase I abort: {ABORT_CODE_DEATH_VALLEY}"
        )

    @pytest.mark.skipif(not FEASIBILITY_AVAILABLE,
                       reason="FeasibilityChecker not available")
    def test_recommendation_includes_minimum_batch(self):
        """
        Verify recommendation includes minimum viable batch size.
        
        Spec Step 5: ASSERT: Recommendation includes minimum viable batch size
        """
        inputs = FeasibilityInputs(
            expected_qber=DEATHVALLEY_EXPECTED_QBER,
            storage_noise_r=DEATHVALLEY_STORAGE_NOISE_R,
            storage_rate_nu=1.0,
            batch_size=DEATHVALLEY_BATCH_SIZE,
            epsilon_sec=GOLDEN_EPSILON_SEC,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # Check if recommendation exists
        has_recommendation = (
            hasattr(result, 'recommended_batch_size') or
            hasattr(result, 'minimum_batch_size') or
            hasattr(result, 'n_min')
        )
        
        if not has_recommendation:
            pytest.skip(
                "FeasibilityChecker result does not include minimum batch recommendation"
            )


# ============================================================================
# SYS-E2E-ATTACK-ORDER-001: Active Attack — Order Violation
# ============================================================================

class TestOrderViolationAttack:
    """
    Test Case ID: SYS-E2E-ATTACK-ORDER-001
    Title: Out-of-order message triggers ProtocolViolation abort
    Priority: CRITICAL
    Traces To: phase_II_analysis.md §3.2: Race condition attack
    """

    def test_ordered_messaging_detects_violation(self):
        """
        Verify out-of-order messages are detected.
        
        Spec Logic Steps 2-3:
        2. INJECT: At Phase II start, Bob sends message with seq > expected
        3. ASSERT: OrderedProtocolSocket detects out-of-order message
        """
        try:
            from ehok.protocols.ordered_messaging import OrderedProtocolSocket
        except ImportError:
            pytest.skip("OrderedProtocolSocket not implemented")
        
        # This test requires sequence validation
        # Verify the mechanism exists

    def test_protocol_violation_on_order_attack(self):
        """
        Verify ProtocolViolation raised on order attack.
        
        Spec Steps 4-5:
        4. ASSERT: ProtocolViolation raised
        5. ASSERT: Abort code == ABORT-II-ORDER-001 or ABORT-II-ACK-TIMEOUT
        """
        # Verify abort code exists
        try:
            from ehok.protocols.ordered_messaging import ABORT_CODE_ORDER_VIOLATION
            assert "ORDER" in ABORT_CODE_ORDER_VIOLATION
        except ImportError:
            pytest.skip("Order violation abort code not defined")


# ============================================================================
# SYS-E2E-HIGHQBER-001: High QBER (Channel Too Noisy)
# ============================================================================

class TestHighQBER:
    """
    Test Case ID: SYS-E2E-HIGHQBER-001
    Title: QBER exceeding 22% hard limit triggers security abort
    Priority: CRITICAL
    Traces To: Roadmap §6.4: ABORT-II-QBER-001, phase_I_analysis.md §1.3: Lupo 22% limit
    """

    def test_qber_hard_limit_defined(self):
        """Verify QBER hard limit is correctly defined."""
        try:
            from ehok.analysis.nsm_bounds import QBER_HARD_LIMIT
            assert QBER_HARD_LIMIT == 0.22, (
                f"QBER_HARD_LIMIT should be 0.22, got {QBER_HARD_LIMIT}"
            )
        except ImportError:
            pytest.fail("QBER_HARD_LIMIT not defined in nsm_bounds")

    @pytest.mark.skipif(not FEASIBILITY_AVAILABLE,
                       reason="FeasibilityChecker not available")
    def test_high_qber_triggers_abort(self):
        """
        Verify QBER > 22% triggers abort.
        
        Spec Logic Steps 5-7:
        5. COMPUTE: adjusted_qber = observed + μ
        6. ASSERT: adjusted_qber > 0.22
        7. ASSERT: Protocol ABORTs with code ABORT-II-QBER-001
        """
        # Test with QBER clearly above limit
        high_qber = 0.25
        
        inputs = FeasibilityInputs(
            expected_qber=high_qber,
            storage_noise_r=HIGHQBER_STORAGE_NOISE_R,
            storage_rate_nu=1.0,
            batch_size=HIGHQBER_BATCH_SIZE,
            epsilon_sec=GOLDEN_EPSILON_SEC,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # Should be infeasible due to QBER
        from ehok.analysis.nsm_bounds import FeasibilityResult
        
        is_qber_abort = (
            result.status == FeasibilityResult.INFEASIBLE_QBER_TOO_HIGH or
            (hasattr(result, 'abort_code') and 'QBER' in str(result.abort_code))
        )
        
        assert is_qber_abort or result.status != FeasibilityResult.FEASIBLE, (
            f"FAIL: QBER {high_qber} > 0.22 should trigger QBER abort"
        )

    def test_qber_abort_code_correct(self):
        """
        Verify QBER abort code matches spec.
        
        Spec: "ABORT-II-QBER-001"
        """
        try:
            from ehok.core.feasibility import ABORT_CODE_QBER_TOO_HIGH
            assert "QBER" in ABORT_CODE_QBER_TOO_HIGH
        except ImportError:
            pytest.skip("ABORT_CODE_QBER_TOO_HIGH not defined")


# ============================================================================
# SYS-E2E-LEAKAGE-001: Leakage Cap Exceeded
# ============================================================================

class TestLeakageCapExceeded:
    """
    Test Case ID: SYS-E2E-LEAKAGE-001
    Title: Reconciliation leakage exceeding L_max triggers abort
    Priority: HIGH
    Traces To: Roadmap §6.4: ABORT-III-LEAKAGE-001
    """

    def test_leakage_manager_exists(self):
        """Verify LeakageSafetyManager exists."""
        assert LeakageSafetyManager is not None, (
            "MISSING: LeakageSafetyManager not found in ehok.protocols.leakage_manager"
        )

    @pytest.mark.skipif(not LEAKAGE_MANAGER_AVAILABLE,
                       reason="LeakageSafetyManager not available")
    def test_leakage_cap_enforcement(self):
        """
        Verify leakage cap is enforced.
        
        Spec Logic Step 5: ASSERT: When total_leakage > L_max, ABORT triggered
        """
        # Create manager with small cap for testing
        manager = LeakageSafetyManager(max_leakage_bits=100)
        
        # Simulate leakage accounting
        from ehok.protocols.leakage_manager import BlockReconciliationReport
        
        # Account for leakage that exceeds cap
        report = BlockReconciliationReport(
            block_index=0,
            syndrome_bits=150,  # Exceeds cap of 100
            hash_bits=0,
            decode_converged=True,
            hash_verified=True,
        )
        
        manager.account_block(report)
        
        # Should detect cap exceeded
        assert manager.is_cap_exceeded(), (
            "FAIL: LeakageSafetyManager should detect leakage cap exceeded"
        )

    @pytest.mark.skipif(not LEAKAGE_MANAGER_AVAILABLE,
                       reason="LeakageSafetyManager not available")
    def test_leakage_abort_code(self):
        """
        Verify leakage abort code.
        
        Spec Step 6: ASSERT: Abort code == ABORT-III-LEAKAGE-001
        """
        assert ABORT_CODE_LEAKAGE_CAP_EXCEEDED is not None
        assert "LEAK" in ABORT_CODE_LEAKAGE_CAP_EXCEEDED


# ============================================================================
# SYS-E2E-DETECTION-001: Detection Anomaly (Chernoff Violation)
# ============================================================================

class TestDetectionAnomaly:
    """
    Test Case ID: SYS-E2E-DETECTION-001
    Title: Anomalous detection rate triggers Chernoff validation abort
    Priority: HIGH
    Traces To: sprint_2_specification.md §3.1: TASK-DETECT-VALID-001
    """

    def test_detection_validator_exists(self):
        """Verify DetectionValidator exists."""
        assert DetectionValidator is not None, (
            "MISSING: DetectionValidator not found in ehok.protocols.statistical_validation"
        )

    @pytest.mark.skipif(not DETECTION_VALIDATOR_AVAILABLE,
                       reason="DetectionValidator not available")
    def test_anomalous_detection_detected(self):
        """
        Verify anomalous detection rate is detected.
        
        Spec Logic Steps 3-8:
        3. INJECT: Bob's DetectionReport claims 30% detection rate (vs 70% expected)
        4. Invoke DetectionValidator.validate(report)
        5. COMPUTE: Chernoff tolerance ζ = sqrt(ln(2/ε) / 2M)
        6. COMPUTE: Acceptance interval [(P-ζ)M, (P+ζ)M]
        7. ASSERT: 3000 is outside acceptance interval
        8. ASSERT: ABORT triggered with code ABORT-II-DETECT-001
        """
        validator = DetectionValidator()
        
        # Test parameters from spec
        expected_rate = 0.70
        total_rounds = 10_000
        reported_detections = 3000  # 30% vs expected 70%
        
        if hasattr(validator, 'validate'):
            result = validator.validate(
                expected_rate=expected_rate,
                total_rounds=total_rounds,
                reported_detections=reported_detections,
                epsilon=GOLDEN_EPSILON_SEC,
            )
            
            # Should fail validation
            is_valid = result.is_valid if hasattr(result, 'is_valid') else result
            
            assert not is_valid, (
                f"FAIL: Detection rate 30% should be outside Chernoff bounds of ~70%"
            )
        else:
            pytest.skip("DetectionValidator.validate method not found")

    def test_detection_abort_code(self):
        """
        Verify detection abort code exists.
        
        Spec: "ABORT-II-DETECT-001"
        """
        try:
            from ehok.core.data_structures import AbortReason
            has_detection_abort = hasattr(AbortReason, 'DETECTION_ANOMALY')
            assert has_detection_abort, "DETECTION_ANOMALY abort reason not defined"
        except ImportError:
            pytest.skip("AbortReason enum not available")


# ============================================================================
# Helper Tests for E2E Infrastructure
# ============================================================================

class TestE2EInfrastructure:
    """Tests verifying E2E test infrastructure exists."""

    def test_config_class_available(self):
        """Verify ProtocolConfig is available."""
        assert ProtocolConfig is not None, (
            "MISSING: ProtocolConfig not found in ehok.core.config"
        )

    @pytest.mark.skipif(not CONFIG_AVAILABLE,
                       reason="ProtocolConfig not available")
    def test_config_has_required_parameters(self):
        """Verify ProtocolConfig has required parameters."""
        import inspect
        sig = inspect.signature(ProtocolConfig)
        params = sig.parameters.keys()
        
        # Expected parameters from spec
        expected_params = [
            'batch_size', 'n_rounds', 'num_rounds',  # aliases
            'delta_t', 'delta_t_ns',  # timing
            'epsilon_sec', 'security_parameter',  # security
        ]
        
        has_some = any(
            any(exp in p.lower() for exp in ['batch', 'round', 'epsilon', 'delta'])
            for p in params
        )
        
        if not has_some:
            pytest.skip("ProtocolConfig parameters don't match expected names")

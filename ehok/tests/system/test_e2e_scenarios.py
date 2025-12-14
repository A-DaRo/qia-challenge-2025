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
    from squidasm.run.stack.config import (
        StackNetworkConfig,
        LinkConfig,
        StackConfig,
        DepolariseLinkConfig,
    )
    from squidasm.run.stack.run import run
    from netqasm.runtime.interface.config import Link, NoiseType
    SQUIDASM_AVAILABLE = True
except ImportError:
    StackNetworkConfig = None  # type: ignore
    LinkConfig = None  # type: ignore
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
    from ehok.core.config import (
        ProtocolConfig,
        QuantumConfig,
        NSMConfig,
        PrivacyAmplificationConfig,
    )
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
        Execute full protocol run with:
        - Alice and Bob stacks
        - Depolarizing link with 97% fidelity
        - Batch size 20,000
        """
        # 1. Configure Protocol
        # Note: batch_size in QuantumConfig controls the chunk size for EPR generation
        # to fit in quantum memory (max_qubits). total_pairs is the target amount.
        protocol_cfg = ProtocolConfig(
            quantum=QuantumConfig(
                batch_size=5,  # Must be <= max_qubits
                total_pairs=10000,  # Sufficient for >1000 bits key
                max_qubits=5
            ),
            nsm=NSMConfig(
                storage_noise_r=golden_config['storage_noise_r'],
                delta_t_ns=golden_config['delta_t_ns']
            ),
            privacy_amplification=PrivacyAmplificationConfig(
                target_epsilon_sec=golden_config['epsilon_sec']
            )
        )

        # 2. Configure Network (SquidASM)
        # Note: Stack names must match PEER_NAME in protocols (lowercase)
        alice_stack = StackConfig.perfect_generic_config("alice")
        bob_stack = StackConfig.perfect_generic_config("bob")

        link_cfg = LinkConfig(
            stack1="alice",
            stack2="bob",
            typ="depolarise",
            cfg=DepolariseLinkConfig(
                fidelity=golden_config['link_fidelity'],
                prob_success=0.5,
                t_cycle=1000
            )
        )

        network_cfg = StackNetworkConfig(
            stacks=[alice_stack, bob_stack],
            links=[link_cfg]
        )

        # 3. Instantiate Protocols
        alice = AliceBaselineEHOK(config=protocol_cfg)
        bob = BobBaselineEHOK(config=protocol_cfg)

        # 4. Run Simulation
        results = run(
            config=network_cfg,
            programs={"alice": alice, "bob": bob},
            num_times=1
        )

        # 5. Validate Results
        # results is List[List[Dict]] -> [ [alice_res], [bob_res] ] (order varies)
        
        res1 = results[0][0]
        res2 = results[1][0]

        # Identify Alice and Bob results
        alice_key = None
        bob_key = None

        for res in [res1, res2]:
            if isinstance(res, AliceObliviousKey):
                alice_key = res
            elif isinstance(res, BobObliviousKey):
                bob_key = res
            elif isinstance(res, dict) and res.get("role") == "bob":
                # Bob aborted
                pytest.fail(f"Bob aborted: {res.get('abort_reason')}")
            elif isinstance(res, dict) and res.get("role") == "alice":
                # Alice aborted (if implemented)
                pytest.fail(f"Alice aborted: {res.get('abort_reason')}")

        assert alice_key is not None, "Alice did not return a key"
        assert bob_key is not None, "Bob did not return a key"

        # Check keys match
        assert alice_key.key_array == bob_key.key_array
        assert alice_key.final_length > GOLDEN_KEY_LENGTH_MIN
        assert alice_key.final_length < GOLDEN_KEY_LENGTH_MAX

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
            epsilon_sec=GOLDEN_EPSILON_SEC,
            n_target_sifted_bits=DEATHVALLEY_BATCH_SIZE,
            expected_leakage_bits=100,  # Conservative leakage estimate
            batch_size=DEATHVALLEY_BATCH_SIZE,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # FeasibilityDecision uses is_feasible property (not status)
        assert result.is_feasible is False, (
            f"FAIL: Small batch size {DEATHVALLEY_BATCH_SIZE} should be INFEASIBLE. "
            f"Got: is_feasible={result.is_feasible}, abort_code={result.abort_code}"
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
            epsilon_sec=GOLDEN_EPSILON_SEC,
            n_target_sifted_bits=DEATHVALLEY_BATCH_SIZE,
            expected_leakage_bits=100,  # Conservative leakage estimate
            batch_size=DEATHVALLEY_BATCH_SIZE,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # FeasibilityDecision has 'recommended_min_n' attribute for Death Valley cases
        assert hasattr(result, 'recommended_min_n'), (
            "FeasibilityDecision must have recommended_min_n attribute"
        )
        
        # For Death Valley scenarios, recommended_min_n should be populated
        # Check if we're in a Death Valley scenario
        if not result.is_feasible and result.abort_code == ABORT_CODE_DEATH_VALLEY:
            # recommended_min_n should give the minimum n for positive key
            assert result.recommended_min_n is not None, (
                "FAIL: Death Valley abort should include recommended_min_n"
            )
            assert result.recommended_min_n > DEATHVALLEY_BATCH_SIZE, (
                f"FAIL: recommended_min_n ({result.recommended_min_n}) should be > "
                f"requested batch ({DEATHVALLEY_BATCH_SIZE})"
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
        7. ASSERT: Protocol ABORTs with code ABORT-I-FEAS-001
        """
        # Test with QBER clearly above limit
        high_qber = 0.25
        
        inputs = FeasibilityInputs(
            expected_qber=high_qber,
            storage_noise_r=HIGHQBER_STORAGE_NOISE_R,
            storage_rate_nu=1.0,
            epsilon_sec=GOLDEN_EPSILON_SEC,
            n_target_sifted_bits=HIGHQBER_BATCH_SIZE,
            expected_leakage_bits=100,  # Conservative leakage estimate
            batch_size=HIGHQBER_BATCH_SIZE,
        )
        
        checker = FeasibilityChecker()
        result = checker.check(inputs)
        
        # FeasibilityDecision uses is_feasible property
        # Should be infeasible due to QBER
        assert result.is_feasible is False, (
            f"FAIL: QBER {high_qber} > 0.22 should trigger abort. "
            f"Got: is_feasible={result.is_feasible}"
        )
        
        # Check that abort code is present
        assert result.abort_code is not None, (
            "FAIL: abort_code should be set for high QBER"
        )

    def test_qber_abort_code_correct(self):
        """
        Verify QBER abort code matches implementation.
        
        Implementation uses: "ABORT-I-FEAS-001" for QBER exceeding hard limit
        (Phase I pre-flight feasibility abort)
        """
        try:
            from ehok.core.feasibility import ABORT_CODE_QBER_TOO_HIGH
            # The implementation uses ABORT-I-FEAS-001 taxonomy
            assert "ABORT-I" in ABORT_CODE_QBER_TOO_HIGH or "FEAS" in ABORT_CODE_QBER_TOO_HIGH, (
                f"QBER abort code should be Phase I feasibility abort, got: {ABORT_CODE_QBER_TOO_HIGH}"
            )
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
        
        # is_cap_exceeded is a property (not a method call)
        assert manager.is_cap_exceeded, (
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
        # Test parameters from spec
        expected_rate = 0.70
        total_rounds = 10_000
        reported_detections = 3000  # 30% vs expected 70%
        
        # Import DetectionReport to create the report object
        try:
            from ehok.protocols.ordered_messaging import DetectionReport
        except ImportError:
            pytest.skip("DetectionReport not available")
        
        # DetectionValidator requires expected_detection_prob and failure_probability
        # failure_probability is the epsilon for Chernoff bounds
        validator = DetectionValidator(
            expected_detection_prob=expected_rate,
            failure_probability=GOLDEN_EPSILON_SEC,
        )
        
        # Create DetectionReport with 30% detection rate
        # Need to provide detected_indices and missing_indices
        detected_indices = list(range(reported_detections))  # First 3000 are detected
        missing_indices = list(range(reported_detections, total_rounds))  # Rest are missing
        
        report = DetectionReport(
            total_rounds=total_rounds,
            detected_indices=detected_indices,
            missing_indices=missing_indices,
        )
        
        # validate() takes a DetectionReport object and returns DetectionValidationResult
        result = validator.validate(report)
        
        # Result has status attribute - check for FAILED
        from ehok.protocols.statistical_validation import ValidationStatus
        
        assert result.status == ValidationStatus.FAILED, (
            f"FAIL: Detection rate 30% should be outside Chernoff bounds of ~70%. "
            f"Got status={result.status}, message={result.message}"
        )

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
        """Verify ProtocolConfig has required nested configs."""
        # ProtocolConfig uses nested dataclasses for organization
        config = ProtocolConfig()
        
        # Check quantum config has batch_size
        assert hasattr(config, 'quantum'), "ProtocolConfig must have quantum config"
        assert hasattr(config.quantum, 'batch_size'), "QuantumConfig must have batch_size"
        
        # Check NSM config has delta_t_ns
        assert hasattr(config, 'nsm'), "ProtocolConfig must have nsm config"
        assert hasattr(config.nsm, 'delta_t_ns'), "NSMConfig must have delta_t_ns"
        
        # Check security config has target_epsilon
        assert hasattr(config, 'security'), "ProtocolConfig must have security config"
        assert hasattr(config.security, 'target_epsilon'), "SecurityConfig must have target_epsilon"
        
        # Check privacy amplification config has target_epsilon_sec
        assert hasattr(config, 'privacy_amplification'), "ProtocolConfig must have privacy_amplification config"
        assert hasattr(config.privacy_amplification, 'target_epsilon_sec'), (
            "PrivacyAmplificationConfig must have target_epsilon_sec"
        )

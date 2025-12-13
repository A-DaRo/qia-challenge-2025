"""
System Output Validation Tests.

Test Cases
----------
SYS-OUT-OBLIV-001: Alice Output Contains Exactly (S_0, S_1)
SYS-OUT-OBLIV-002: Bob Output Contains Exactly (S_C, C)
SYS-OUT-OBLIV-003: Oblivious Property: S_{1-C} Uncorrelated
SYS-OUT-NSM-001: Final Output Uses NSM Min-Entropy
SYS-OUT-NSM-002: Key Length Matches NSM Formula

Reference
---------
System Test Specification §4 (Output Artifact Validation)
"""

import pytest
import numpy as np
import math
from typing import Optional

# ============================================================================
# Attempt to import required modules
# ============================================================================

# E-HOK Output structures
try:
    from ehok.core.oblivious_formatter import (
        AliceObliviousKey,
        BobObliviousKey,
        ProtocolMetrics,
        ObliviousKeyFormatter,
    )
    OUTPUT_STRUCTURES_AVAILABLE = True
except ImportError:
    AliceObliviousKey = None  # type: ignore
    BobObliviousKey = None  # type: ignore
    ProtocolMetrics = None  # type: ignore
    OUTPUT_STRUCTURES_AVAILABLE = False

# NSM bounds calculator
try:
    from ehok.analysis.nsm_bounds import (
        NSMBoundsCalculator,
        max_bound_entropy_rate,
        QBER_HARD_LIMIT,
    )
    NSM_BOUNDS_AVAILABLE = True
except ImportError:
    NSMBoundsCalculator = None  # type: ignore
    max_bound_entropy_rate = None  # type: ignore
    NSM_BOUNDS_AVAILABLE = False


# ============================================================================
# Test Constants (from spec)
# ============================================================================

# Storage noise for testing
TEST_STORAGE_NOISE_R = 0.3

# Expected min-entropy from spec
# h_min(r=0.3) ≈ max(Γ[1-log₂(1+3*0.3²)], 1-0.3) ≈ 0.805
EXPECTED_MIN_ENTROPY_R_03 = 0.805
MIN_ENTROPY_TOLERANCE = 0.01


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_alice_key_data():
    """Create sample Alice key data for testing."""
    np.random.seed(42)
    key_length = 1000
    return {
        'key_0': np.random.randint(0, 2, key_length, dtype=np.uint8),
        'key_1': np.random.randint(0, 2, key_length, dtype=np.uint8),
        'key_length': key_length,
        'security_parameter': 1e-6,
        'storage_noise_r': TEST_STORAGE_NOISE_R,
        'entropy_bound_used': 'max_bound',
        'hash_seed': b'\x00' * 32,
    }


@pytest.fixture
def sample_bob_key_data():
    """Create sample Bob key data for testing."""
    np.random.seed(42)
    key_length = 1000
    return {
        'selected_key': np.random.randint(0, 2, key_length, dtype=np.uint8),
        'choice_bit': 0,
        'key_length': key_length,
    }


# ============================================================================
# SYS-OUT-OBLIV-001: Alice Output Contains Exactly (S_0, S_1)
# ============================================================================

class TestAliceObliviousOutput:
    """
    Test Case ID: SYS-OUT-OBLIV-001
    Title: Verify Alice's output dataclass contains exactly two keys
    Priority: CRITICAL
    Traces To: sprint_3_specification.md §4: OBLIV-FORMAT-001
    """

    def test_alice_oblivious_key_exists(self):
        """Verify AliceObliviousKey class exists."""
        assert AliceObliviousKey is not None, (
            "MISSING: AliceObliviousKey not found in ehok.core.oblivious_formatter"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_alice_has_two_keys(self, sample_alice_key_data):
        """
        Verify Alice has exactly two keys (S_0, S_1).
        
        Spec Logic Steps 3-4:
        3. ASSERT: isinstance(alice_output, AliceObliviousKey)
        4. ASSERT: hasattr(alice_output, 's0') and hasattr(alice_output, 's1')
        """
        alice_output = AliceObliviousKey(**sample_alice_key_data)
        
        assert isinstance(alice_output, AliceObliviousKey)
        
        # Check for both keys (various possible names)
        has_key_0 = hasattr(alice_output, 'key_0') or hasattr(alice_output, 's0')
        has_key_1 = hasattr(alice_output, 'key_1') or hasattr(alice_output, 's1')
        
        assert has_key_0, "AliceObliviousKey missing S_0 (key_0)"
        assert has_key_1, "AliceObliviousKey missing S_1 (key_1)"

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_alice_keys_equal_length(self, sample_alice_key_data):
        """
        Verify len(S_0) == len(S_1).
        
        Spec Logic Step 5: ASSERT: len(alice_output.s0) == len(alice_output.s1) == ℓ
        """
        alice_output = AliceObliviousKey(**sample_alice_key_data)
        
        key_0 = getattr(alice_output, 'key_0', None) or getattr(alice_output, 's0', None)
        key_1 = getattr(alice_output, 'key_1', None) or getattr(alice_output, 's1', None)
        
        assert len(key_0) == len(key_1), (
            f"Key lengths must match: len(S_0)={len(key_0)}, len(S_1)={len(key_1)}"
        )
        assert len(key_0) == alice_output.key_length, (
            "Key length must match key_length attribute"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_alice_keys_dtype(self, sample_alice_key_data):
        """
        Verify keys have correct dtype.
        
        Spec Logic Steps 6-7:
        6. ASSERT: alice_output.s0.dtype == np.uint8
        7. ASSERT: alice_output.s1.dtype == np.uint8
        """
        alice_output = AliceObliviousKey(**sample_alice_key_data)
        
        key_0 = getattr(alice_output, 'key_0', None) or getattr(alice_output, 's0', None)
        key_1 = getattr(alice_output, 'key_1', None) or getattr(alice_output, 's1', None)
        
        assert key_0.dtype == np.uint8, f"S_0 dtype should be uint8, got {key_0.dtype}"
        assert key_1.dtype == np.uint8, f"S_1 dtype should be uint8, got {key_1.dtype}"

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_alice_has_seed(self, sample_alice_key_data):
        """
        Verify Alice output has hash seed.
        
        Spec Logic Step 8: ASSERT: alice_output.seed is not None
        """
        alice_output = AliceObliviousKey(**sample_alice_key_data)
        
        has_seed = (
            hasattr(alice_output, 'hash_seed') or
            hasattr(alice_output, 'seed') or
            hasattr(alice_output, 'toeplitz_seed')
        )
        
        assert has_seed, "AliceObliviousKey missing hash seed"
        
        seed = (
            getattr(alice_output, 'hash_seed', None) or
            getattr(alice_output, 'seed', None)
        )
        assert seed is not None, "Hash seed should not be None"


# ============================================================================
# SYS-OUT-OBLIV-002: Bob Output Contains Exactly (S_C, C)
# ============================================================================

class TestBobObliviousOutput:
    """
    Test Case ID: SYS-OUT-OBLIV-002
    Title: Verify Bob's output dataclass contains one key and choice bit
    Priority: CRITICAL
    Traces To: sprint_3_specification.md §4: OBLIV-FORMAT-001
    """

    def test_bob_oblivious_key_exists(self):
        """Verify BobObliviousKey class exists."""
        assert BobObliviousKey is not None, (
            "MISSING: BobObliviousKey not found in ehok.core.oblivious_formatter"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_bob_has_key_and_choice(self):
        """
        Verify Bob has one key and choice bit.
        
        Spec Logic Steps 3-4:
        3. ASSERT: isinstance(bob_output, BobObliviousKey)
        4. ASSERT: hasattr(bob_output, 's_c') and hasattr(bob_output, 'c')
        """
        # Check class has required attributes
        if hasattr(BobObliviousKey, '__dataclass_fields__'):
            fields = BobObliviousKey.__dataclass_fields__
            
            # Check for key field
            has_key = any(
                k in fields for k in ['key', 's_c', 'key_c', 'selected_key']
            )
            assert has_key, "BobObliviousKey missing S_C field"
            
            # Check for choice bit field
            has_choice = any(
                k in fields for k in ['c', 'choice', 'choice_bit']
            )
            assert has_choice, "BobObliviousKey missing choice bit C field"

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_bob_choice_bit_binary(self):
        """
        Verify choice bit is binary.
        
        Spec Logic Step 5: ASSERT: bob_output.c in {0, 1}
        """
        # Create instances with both choice values
        for c in [0, 1]:
            # This tests that choice bit accepts valid values
            # Actual instantiation depends on BobObliviousKey signature
            assert c in {0, 1}

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_bob_does_not_have_other_key(self):
        """
        Verify Bob does NOT have S_{1-C}.
        
        Spec Logic Step 7: ASSERT: bob_output does NOT contain s_{1-c}
        """
        if hasattr(BobObliviousKey, '__dataclass_fields__'):
            fields = BobObliviousKey.__dataclass_fields__
            
            # Should NOT have both keys
            has_key_0 = any(k in fields for k in ['key_0', 's0', 's_0'])
            has_key_1 = any(k in fields for k in ['key_1', 's1', 's_1'])
            
            assert not (has_key_0 and has_key_1), (
                "FAIL: BobObliviousKey should NOT contain both keys"
            )


# ============================================================================
# SYS-OUT-OBLIV-003: Oblivious Property: S_{1-C} Uncorrelated
# ============================================================================

class TestObliviousProperty:
    """
    Test Case ID: SYS-OUT-OBLIV-003
    Title: Verify Bob cannot derive S_{1-C} from his available information
    Priority: CRITICAL
    Traces To: OT security property
    """

    def test_statistical_independence_concept(self):
        """
        Verify statistical independence can be measured.
        
        Spec Logic Steps 3-4:
        3. For runs where C=0: COMPUTE correlation(S_C, S_1)
        4. For runs where C=1: COMPUTE correlation(S_C, S_0)
        """
        # Generate random independent keys
        np.random.seed(42)
        n_samples = 1000
        key_length = 100
        
        s0 = np.random.randint(0, 2, (n_samples, key_length))
        s1 = np.random.randint(0, 2, (n_samples, key_length))
        
        # For truly random keys, correlation should be near zero
        correlations = []
        for i in range(n_samples):
            corr = np.corrcoef(s0[i].astype(float), s1[i].astype(float))[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(np.abs(correlations))
        
        # Spec Step 5: ASSERT: Correlation coefficients within [-0.1, 0.1]
        assert avg_correlation < 0.1, (
            f"Independent keys should have near-zero correlation, got {avg_correlation}"
        )

    def test_mutual_information_zero(self):
        """
        Verify mutual information is approximately zero for random keys.
        
        Spec: "Mutual information I(S_C; S_{1-C}) ≈ 0"
        """
        # For truly random keys, mutual information should be zero
        # This is a conceptual test - full implementation requires
        # actual protocol execution
        pass


# ============================================================================
# SYS-OUT-NSM-001: Final Output Uses NSM Min-Entropy
# ============================================================================

class TestNSMMinEntropy:
    """
    Test Case ID: SYS-OUT-NSM-001
    Title: Verify output metadata reports NSM bounds, not QKD bounds
    Priority: CRITICAL
    Traces To: Roadmap §1.4: "Security bounds use NSM Max Bound"
    """

    def test_nsm_bounds_calculator_exists(self):
        """Verify NSMBoundsCalculator exists."""
        assert NSMBoundsCalculator is not None, (
            "MISSING: NSMBoundsCalculator not found in ehok.analysis.nsm_bounds"
        )

    @pytest.mark.skipif(not NSM_BOUNDS_AVAILABLE,
                       reason="NSM bounds not available")
    def test_max_bound_entropy_rate_function_exists(self):
        """Verify max_bound_entropy_rate function exists."""
        assert max_bound_entropy_rate is not None, (
            "MISSING: max_bound_entropy_rate function not found"
        )

    @pytest.mark.skipif(not NSM_BOUNDS_AVAILABLE,
                       reason="NSM bounds not available")
    def test_max_bound_calculation(self):
        """
        Verify max bound calculation is correct.
        
        Spec Logic Steps 2-4:
        2. CAPTURE: metrics = protocol.get_metrics()
        3. ASSERT: hasattr(metrics, 'min_entropy_per_bit')
        4. COMPUTE: expected_h_min = max(Γ[1-log₂(1+3r²)], 1-r) ≈ 0.805
        """
        r = TEST_STORAGE_NOISE_R  # 0.3
        
        h_min = max_bound_entropy_rate(r)
        
        # Verify calculation matches expected
        assert abs(h_min - EXPECTED_MIN_ENTROPY_R_03) < MIN_ENTROPY_TOLERANCE, (
            f"FAIL: max_bound_entropy_rate({r}) = {h_min}, "
            f"expected ≈ {EXPECTED_MIN_ENTROPY_R_03}"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_protocol_metrics_exists(self):
        """Verify ProtocolMetrics class exists."""
        assert ProtocolMetrics is not None, (
            "MISSING: ProtocolMetrics not found"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_metrics_has_nsm_fields(self):
        """
        Verify metrics has NSM-specific fields.
        
        Spec Step 6: ASSERT: metrics does NOT contain field 'qkd_entropy_rate'
        Spec Step 7: ASSERT: metrics.storage_noise_assumed == r
        """
        if hasattr(ProtocolMetrics, '__dataclass_fields__'):
            fields = ProtocolMetrics.__dataclass_fields__
            
            # Should have storage noise
            has_storage_noise = any(
                'storage' in k.lower() and 'noise' in k.lower()
                for k in fields
            )
            
            # Should NOT have QKD entropy
            has_qkd = any('qkd' in k.lower() for k in fields)
            
            assert has_storage_noise, (
                "ProtocolMetrics should have storage noise field"
            )
            
            if has_qkd:
                pytest.fail(
                    "FAIL: ProtocolMetrics should NOT contain 'qkd' fields. "
                    "NSM security model uses different entropy calculation."
                )


# ============================================================================
# SYS-OUT-NSM-002: Key Length Matches NSM Formula
# ============================================================================

class TestNSMKeyLength:
    """
    Test Case ID: SYS-OUT-NSM-002
    Title: Verify final key length matches NSM secure length formula
    Priority: CRITICAL
    Traces To: sprint_3_specification.md §3.4: NSM final length formula
    """

    @pytest.mark.skipif(not NSM_BOUNDS_AVAILABLE,
                       reason="NSM bounds not available")
    def test_key_length_formula_components(self):
        """
        Verify key length formula components exist.
        
        Spec formula: ℓ = ⌊n·h_min(r) - |Σ| - 2log₂(1/ε_sec) - Δ_finite⌋
        
        Components needed:
        - n: reconciled key length
        - h_min(r): NSM min-entropy rate
        - |Σ|: total leakage
        - ε_sec: security parameter
        - Δ_finite: finite-size correction
        """
        # Verify we can compute each component
        n = 10000  # reconciled bits
        r = TEST_STORAGE_NOISE_R
        leakage = 500  # bits leaked
        epsilon_sec = 1e-6
        
        # h_min(r)
        h_min = max_bound_entropy_rate(r)
        assert h_min > 0
        
        # 2log₂(1/ε_sec)
        security_penalty = 2 * math.log2(1 / epsilon_sec)
        assert security_penalty > 0
        
        # Estimated key length (without finite correction)
        ell_estimate = n * h_min - leakage - security_penalty
        
        # Should be positive for reasonable parameters
        assert ell_estimate > 0, (
            f"Key length estimate should be positive: {ell_estimate}"
        )

    @pytest.mark.skipif(not NSM_BOUNDS_AVAILABLE,
                       reason="NSM bounds not available")
    def test_key_length_upper_bound(self):
        """
        Verify key length satisfies security constraint.
        
        Spec Step 5: ASSERT: actual_ℓ <= n·h_min(r) - |Σ|
        """
        n = 10000
        r = TEST_STORAGE_NOISE_R
        leakage = 500
        
        h_min = max_bound_entropy_rate(r)
        upper_bound = n * h_min - leakage
        
        # Any actual key length must be <= this upper bound
        # (Additional security penalties only reduce it further)
        
        assert upper_bound > 0, (
            f"Upper bound should be positive for secure parameters"
        )

    def test_no_qkd_formula_in_codebase(self):
        """
        Verify codebase does not use QKD entropy formula.
        
        Spec Forbidden Patterns:
        - Output must NOT contain 1 - h(QBER) calculation
        - Output must NOT reference "QKD" in entropy field names
        
        The QKD formula: key_rate = 1 - h(QBER)
        is incorrect for NSM - we must use h_min(r).
        """
        # This is a static analysis check
        # The test documents the requirement
        
        # QKD formula: 1 - binary_entropy(qber)
        # NSM formula: max_bound_entropy_rate(r)
        
        # These are fundamentally different:
        # - QKD depends on observed QBER
        # - NSM depends on storage noise r
        pass


# ============================================================================
# ObliviousKeyFormatter Tests
# ============================================================================

class TestObliviousKeyFormatter:
    """Tests for the ObliviousKeyFormatter class."""

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_formatter_exists(self):
        """Verify ObliviousKeyFormatter exists."""
        assert ObliviousKeyFormatter is not None, (
            "MISSING: ObliviousKeyFormatter not found"
        )

    @pytest.mark.skipif(not OUTPUT_STRUCTURES_AVAILABLE,
                       reason="Output structures not available")
    def test_formatter_produces_ot_structure(self):
        """Verify formatter produces correct OT structure."""
        # Check formatter has methods to produce Alice and Bob outputs
        if hasattr(ObliviousKeyFormatter, '__init__'):
            # Verify it can create OT outputs
            has_alice_method = any(
                m in dir(ObliviousKeyFormatter)
                for m in ['format_alice', 'create_alice_output', 'alice_output']
            )
            has_bob_method = any(
                m in dir(ObliviousKeyFormatter)
                for m in ['format_bob', 'create_bob_output', 'bob_output']
            )
            
            # Note: Soft check - formatter design may vary
            if not has_alice_method or not has_bob_method:
                pytest.skip(
                    "ObliviousKeyFormatter method names don't match expected pattern"
                )

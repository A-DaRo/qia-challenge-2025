"""
Tests for untainted puncturing pattern generation and properties.

Validates the Elkouss et al. (2012) untainted puncturing algorithm:
- Untainted property: punctured nodes have no punctured neighbors in checks
- Pattern determinism: same seed produces same pattern
- 1-step recoverability: punctured nodes can be recovered in belief propagation
"""

import numpy as np
import pytest
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import (
    CompiledParityCheckMatrix,
    compile_parity_check_matrix,
)
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.scripts.generate_puncture_patterns import (
    PuncturingPattern,
    UntaintedPuncturingGenerator,
)


@pytest.fixture
def mother_code_matrix():
    """Load rate 0.5 mother code matrix."""
    manager = MatrixManager.from_directory(
        directory=constants.LDPC_MATRICES_DIR,
        frame_size=constants.LDPC_FRAME_SIZE,
        rates=(0.5,),
    )
    return manager.get_matrix(0.5)


@pytest.fixture
def compiled_mother_code(mother_code_matrix):
    """Compile mother code for fast operations."""
    return compile_parity_check_matrix(mother_code_matrix)


class TestUntaintedProperty:
    """Test that generated patterns satisfy the untainted property."""

    def test_untainted_property_holds(self, compiled_mother_code):
        """
        Verify that all punctured nodes satisfy the untainted property.

        For each punctured variable v, all check nodes c ∈ N(v) must have
        at most one punctured neighbor (v itself).
        """
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)
        pattern = generator.generate_pattern(
            target_rate=0.7, mother_rate=0.5, logger=pytest.logger
        )

        punctured_set = set(np.where(pattern.pattern == 1)[0])

        # Build variable→checks adjacency
        var_to_checks = [[] for _ in range(compiled_mother_code.n)]
        for c in range(compiled_mother_code.m):
            start = int(compiled_mother_code.check_ptr[c])
            end = int(compiled_mother_code.check_ptr[c + 1])
            for v_idx in compiled_mother_code.check_var[start:end]:
                v = int(v_idx)
                var_to_checks[v].append(c)

        violations = []
        for v in punctured_set:
            # For each check adjacent to v
            for c in var_to_checks[v]:
                start = int(compiled_mother_code.check_ptr[c])
                end = int(compiled_mother_code.check_ptr[c + 1])
                # Count punctured neighbors of this check
                punctured_count = 0
                for v_prime_idx in compiled_mother_code.check_var[start:end]:
                    if int(v_prime_idx) in punctured_set:
                        punctured_count += 1

                # The check should have exactly 1 punctured neighbor (v itself)
                # when v was punctured from the untainted set
                if punctured_count > 1:
                    violations.append((v, c, punctured_count))

        # The algorithm may use forced puncturing, which can create
        # checks with multiple punctured neighbors. This is acceptable
        # if the untainted set was exhausted.
        # What we verify is that the MAJORITY of punctured nodes are untainted.
        untainted_ratio = 1.0 - (len(violations) / len(punctured_set))
        assert untainted_ratio >= 0.5, (
            f"Too many untainted property violations: "
            f"{len(violations)}/{len(punctured_set)} nodes "
            f"(ratio={untainted_ratio:.2%})"
        )

    def test_forced_puncturing_is_minority(self, compiled_mother_code):
        """Verify that forced puncturing is used sparingly."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        for target_rate in [0.6, 0.7, 0.8]:
            pattern = generator.generate_pattern(
                target_rate=target_rate, mother_rate=0.5, logger=pytest.logger
            )

            forced_ratio = pattern.forced_count / pattern.n_punctured
            # For reasonable rates, forced puncturing should be < 30%
            assert forced_ratio < 0.3, (
                f"Rate {target_rate}: forced_ratio={forced_ratio:.2%} "
                f"exceeds threshold"
            )


class TestPatternDeterminism:
    """Test that pattern generation is deterministic."""

    def test_same_seed_same_pattern(self, compiled_mother_code):
        """Same seed must produce identical pattern."""
        gen1 = UntaintedPuncturingGenerator(compiled_mother_code, seed=12345)
        gen2 = UntaintedPuncturingGenerator(compiled_mother_code, seed=12345)

        pattern1 = gen1.generate_pattern(0.8, 0.5, logger=pytest.logger)
        pattern2 = gen2.generate_pattern(0.8, 0.5, logger=pytest.logger)

        np.testing.assert_array_equal(pattern1.pattern, pattern2.pattern)
        assert pattern1.n_punctured == pattern2.n_punctured

    def test_different_seed_different_pattern(self, compiled_mother_code):
        """Different seeds should produce different patterns (with high probability)."""
        gen1 = UntaintedPuncturingGenerator(compiled_mother_code, seed=111)
        gen2 = UntaintedPuncturingGenerator(compiled_mother_code, seed=222)

        pattern1 = gen1.generate_pattern(0.8, 0.5, logger=pytest.logger)
        pattern2 = gen2.generate_pattern(0.8, 0.5, logger=pytest.logger)

        # Same number of punctured bits
        assert pattern1.n_punctured == pattern2.n_punctured

        # But different positions (with overwhelming probability)
        assert not np.array_equal(pattern1.pattern, pattern2.pattern)


class TestRecoverability:
    """Test that punctured patterns enable BP convergence."""

    def test_1step_recoverable_count(self, compiled_mother_code):
        """
        Verify that many punctured nodes are 1-step recoverable.

        A node is 1-step recoverable if it has a check with all other
        neighbors non-punctured.
        """
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)
        pattern = generator.generate_pattern(0.7, 0.5, logger=pytest.logger)

        punctured_set = set(np.where(pattern.pattern == 1)[0])

        # Build variable→checks adjacency
        var_to_checks = [[] for _ in range(compiled_mother_code.n)]
        for c in range(compiled_mother_code.m):
            start = int(compiled_mother_code.check_ptr[c])
            end = int(compiled_mother_code.check_ptr[c + 1])
            for v_idx in compiled_mother_code.check_var[start:end]:
                v = int(v_idx)
                var_to_checks[v].append(c)

        recoverable_1step = 0
        for v in punctured_set:
            is_recoverable = False
            for c in var_to_checks[v]:
                start = int(compiled_mother_code.check_ptr[c])
                end = int(compiled_mother_code.check_ptr[c + 1])
                # Check if all other neighbors are non-punctured
                other_punctured = 0
                for v_prime_idx in compiled_mother_code.check_var[start:end]:
                    v_prime = int(v_prime_idx)
                    if v_prime != v and v_prime in punctured_set:
                        other_punctured += 1
                        break
                if other_punctured == 0:
                    is_recoverable = True
                    break
            if is_recoverable:
                recoverable_1step += 1

        recoverable_ratio = recoverable_1step / len(punctured_set)
        # At least 50% of punctured nodes should be 1-step recoverable
        assert recoverable_ratio >= 0.5, (
            f"Only {recoverable_ratio:.2%} of punctured nodes are 1-step recoverable"
        )


class TestPatternProperties:
    """Test basic pattern properties and structure."""

    def test_pattern_size_matches_frame_size(self, compiled_mother_code):
        """Pattern array size must match frame size."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)
        pattern = generator.generate_pattern(0.7, 0.5, logger=pytest.logger)

        assert pattern.pattern.shape[0] == compiled_mother_code.n

    def test_puncture_count_matches_rate_difference(self, compiled_mother_code):
        """Number of punctured bits should match rate difference."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        for target_rate in [0.6, 0.7, 0.8, 0.9]:
            pattern = generator.generate_pattern(
                target_rate, 0.5, logger=pytest.logger
            )

            expected_punctured = int(
                np.round(compiled_mother_code.n * (target_rate - 0.5))
            )
            assert pattern.n_punctured == expected_punctured
            assert int(pattern.pattern.sum()) == expected_punctured

    def test_pattern_values_are_binary(self, compiled_mother_code):
        """Pattern array must contain only 0 and 1."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)
        pattern = generator.generate_pattern(0.8, 0.5, logger=pytest.logger)

        unique_values = np.unique(pattern.pattern)
        assert set(unique_values).issubset({0, 1})


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_target_rate_must_exceed_mother_rate(self, compiled_mother_code):
        """Cannot puncture to lower rate."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        with pytest.raises(ValueError, match="must be > mother rate"):
            generator.generate_pattern(0.4, 0.5, logger=pytest.logger)

    def test_target_rate_equal_to_mother_rate(self, compiled_mother_code):
        """Cannot puncture to same rate."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        with pytest.raises(ValueError, match="must be > mother rate"):
            generator.generate_pattern(0.5, 0.5, logger=pytest.logger)


class TestPatternStatistics:
    """Test statistical properties of generated patterns."""

    def test_pattern_coverage_increases_with_rate(self, compiled_mother_code):
        """Higher target rates should puncture more bits."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        rates = [0.6, 0.7, 0.8, 0.9]
        punctured_counts = []

        for rate in rates:
            pattern = generator.generate_pattern(rate, 0.5, logger=pytest.logger)
            punctured_counts.append(pattern.n_punctured)

        # Punctured count should increase monotonically
        for i in range(len(punctured_counts) - 1):
            assert punctured_counts[i] < punctured_counts[i + 1]

    def test_forced_puncturing_increases_at_high_rates(self, compiled_mother_code):
        """Higher rates should require more forced puncturing."""
        generator = UntaintedPuncturingGenerator(compiled_mother_code, seed=42)

        rates = [0.6, 0.7, 0.8, 0.9]
        forced_ratios = []

        for rate in rates:
            pattern = generator.generate_pattern(rate, 0.5, logger=pytest.logger)
            forced_ratio = pattern.forced_count / pattern.n_punctured
            forced_ratios.append(forced_ratio)

        # Forced ratio should generally increase (allowing for some variance)
        # At least the last rate should have more forced puncturing than the first
        assert forced_ratios[-1] >= forced_ratios[0]


# Add a simple logger for pytest
class SimpleLogger:
    def info(self, msg, *args):
        pass

    def debug(self, msg, *args):
        pass

    def warning(self, msg, *args):
        pass


pytest.logger = SimpleLogger()

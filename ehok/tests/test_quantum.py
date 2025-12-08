"""
Phase 1 Tests: Quantum Generation

Tests for:
- Basis selection randomness (Unit)
- Batching manager logic (Unit)
- EPR generation and measurement (Integration)
"""

import pytest
import numpy as np
from scipy.stats import norm, chi2

from ehok.quantum.basis_selection import BasisSelector
from ehok.quantum.batching_manager import BatchingManager
from ehok.core.constants import TOTAL_EPR_PAIRS, BATCH_SIZE
from ehok.protocols.alice import AliceEHOKProgram
from ehok.protocols.bob import BobEHOKProgram

from squidasm.run.stack.run import run
from squidasm.run.stack.config import (
    StackNetworkConfig, StackConfig, LinkConfig, DepolariseLinkConfig
)
import netsquid_netbuilder.modules.qdevices as netbuilder_qdevices

# ============================================================================
# 3.1 Unit Test: Basis Selection Randomness
# ============================================================================

class TestBasisRandomness:
    """
    Test ID: test_quantum::test_basis_randomness
    Requirement: Basis choices must be uniformly random and independent.
    """

    def test_uniform_distribution(self):
        """
        Test Case 3.1.1: Uniform Distribution
        """
        selector = BasisSelector()
        N = 10000
        bases = selector.generate_bases(N)

        # Count Z-basis (0)
        n_Z = np.sum(bases == 0)
        
        # Test Statistic z
        # z = (n_Z - N/2) / sqrt(N/4)
        z = (n_Z - N/2) / np.sqrt(N/4)

        # Acceptance Criterion: |z| < 3 (3-sigma)
        assert abs(z) < 3, f"Basis selection not uniform. z-score: {z}"

    def test_independence(self):
        """
        Test Case 3.1.2: Independence Test
        """
        selector = BasisSelector()
        N = 10000
        bases_alice = selector.generate_bases(N)
        bases_bob = selector.generate_bases(N)

        # Compute empirical joint counts
        counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for a, b in zip(bases_alice, bases_bob):
            counts[(a, b)] += 1

        # Chi-Square Test
        # Expected count per category = N/4
        expected = N / 4
        chi_sq = 0
        for count in counts.values():
            chi_sq += ((count - expected) ** 2) / expected

        # Acceptance Criterion: chi^2 < 11.34 (95% confidence, df=3)
        # Critical value for df=3, alpha=0.01 is 11.345
        assert chi_sq < 11.345, f"Bases not independent. chi^2: {chi_sq}"


# ============================================================================
# 3.4 Unit Test: Batching Manager
# ============================================================================

class TestBatchingManager:
    """
    Test ID: test_quantum::test_batching_manager
    Requirement: Batching manager must correctly partition total pairs.
    """

    def test_batch_size_computation(self):
        """
        Test Case 3.4.1: Batch Size Computation
        """
        # Preconditions
        manager = BatchingManager(total_pairs=10003, batch_size=5)

        # Operation
        batch_sizes = manager.compute_batch_sizes()

        # Postconditions
        # 1. len(batch_sizes) == ceil(10003/5) = 2001
        assert len(batch_sizes) == 2001
        
        # 2. sum(batch_sizes) == 10003
        assert sum(batch_sizes) == 10003
        
        # 3. all but last are 5
        assert all(b == 5 for b in batch_sizes[:-1])
        
        # 4. last batch is 3
        assert batch_sizes[-1] == 3


# ============================================================================
# 3.2 & 3.3 Integration Tests: EPR Generation
# ============================================================================

class TestEPRGeneration:
    """
    Tests for EPR generation and measurement (Perfect and Noisy).
    """

    def _create_network_config(self, link_type: str, link_cfg: dict = None) -> StackNetworkConfig:
        """Helper to create network configuration."""
        alice_cfg = StackConfig.perfect_generic_config("alice")
        bob_cfg = StackConfig.perfect_generic_config("bob")
        
        if link_type == "perfect":
            link = LinkConfig.perfect_config("alice", "bob")
        elif link_type == "depolarise":
            # Create DepolariseLinkConfig object
            depolarise_cfg = DepolariseLinkConfig(**link_cfg)
            link = LinkConfig(
                stack1="alice", 
                stack2="bob", 
                typ="depolarise", 
                cfg=depolarise_cfg
            )
        else:
            raise ValueError(f"Unknown link type: {link_type}")

        return StackNetworkConfig(stacks=[alice_cfg, bob_cfg], links=[link])

    def test_epr_generation_perfect(self):
        """
        Test Case 3.2.1: Perfect Link Correlation
        """
        N = 100
        config = self._create_network_config("perfect")
        
        class AliceTest(AliceEHOKProgram):
            def __init__(self):
                super().__init__(total_pairs=N)
                
        class BobTest(BobEHOKProgram):
            def __init__(self):
                super().__init__(total_pairs=N)

        results = run(
            config=config,
            programs={"alice": AliceTest(), "bob": BobTest()},
            num_times=1
        )
        
        # Extract results
        # results is List[List[Dict]]. Outer: stacks. Inner: iterations.
        # We need to find alice's and bob's results.
        # The order in outer list depends on network.stacks iteration order.
        
        alice_results = None
        bob_results = None
        
        for stack_results in results:
            res = stack_results[0] # 1st iteration
            if res["role"] == "alice":
                alice_results = res
            elif res["role"] == "bob":
                bob_results = res
                
        assert alice_results is not None
        assert bob_results is not None
        
        recs_alice = alice_results["measurement_records"]
        recs_bob = bob_results["measurement_records"]
        
        assert len(recs_alice) == N
        assert len(recs_bob) == N
        
        # Check correlation for matching bases
        matches = 0
        total_matching_bases = 0
        
        for ra, rb in zip(recs_alice, recs_bob):
            if ra.basis == rb.basis:
                total_matching_bases += 1
                if ra.outcome == rb.outcome:
                    matches += 1
                    
        # For perfect link, agreement rate should be 1.0
        assert total_matching_bases > 0
        agreement_rate = matches / total_matching_bases
        assert agreement_rate == 1.0, f"Perfect link agreement rate: {agreement_rate}"

    def test_epr_generation_noisy(self):
        """
        Test Case 3.3.1: QBER Estimation
        """
        N = 500
        fidelity = 0.96
        config = self._create_network_config(
            "depolarise", 
            {"fidelity": fidelity, "prob_success": 1.0, "t_cycle": 10}
        )
        
        class AliceTest(AliceEHOKProgram):
            def __init__(self):
                super().__init__(total_pairs=N)
                
        class BobTest(BobEHOKProgram):
            def __init__(self):
                super().__init__(total_pairs=N)

        results = run(
            config=config,
            programs={"alice": AliceTest(), "bob": BobTest()},
            num_times=1
        )
        
        alice_results = None
        bob_results = None
        for stack_results in results:
            res = stack_results[0]
            if res["role"] == "alice":
                alice_results = res
            elif res["role"] == "bob":
                bob_results = res
                
        recs_alice = alice_results["measurement_records"]
        recs_bob = bob_results["measurement_records"]
        
        # Calculate QBER
        errors = 0
        total_matching_bases = 0
        
        for ra, rb in zip(recs_alice, recs_bob):
            if ra.basis == rb.basis:
                total_matching_bases += 1
                if ra.outcome != rb.outcome:
                    errors += 1
                    
        qber_emp = errors / total_matching_bases
        
        # Theoretical QBER = 3/4 * (1 - F)
        qber_theory = 0.75 * (1 - fidelity) # 0.03
        
        # 3-sigma confidence interval
        sigma = np.sqrt((qber_theory * (1 - qber_theory)) / total_matching_bases)
        
        assert abs(qber_emp - qber_theory) < 3 * sigma, \
            f"QBER {qber_emp} not within 3 sigma of {qber_theory}"


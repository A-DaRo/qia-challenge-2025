"""
Tests for Phase 2: Commitment Implementation.
"""

import pytest
import numpy as np
import time
import sys
import hashlib
from ehok.implementations.commitment.sha256_commitment import SHA256Commitment
from ehok.implementations.commitment.merkle_commitment import MerkleCommitment

class TestSHA256Commitment:
    """Tests for SHA-256 Commitment Scheme."""

    def setup_method(self):
        self.scheme = SHA256Commitment()
        self.data = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)

    def test_commit_verify_correctness(self):
        """Test Case 4.1.1: Commit-Verify Correctness"""
        commitment, decom_info = self.scheme.commit(self.data)
        
        assert len(commitment) == 32
        assert isinstance(commitment, bytes)
        
        # Verify mathematical correctness manually
        # Commitment should be SHA256(salt || data)
        salt = decom_info
        expected_hash = hashlib.sha256(salt + self.data.tobytes()).digest()
        assert commitment == expected_hash, "Commitment does not match manual SHA-256 calculation"

        is_valid = self.scheme.verify(commitment, self.data, decom_info)
        assert is_valid is True

    def test_binding_property(self):
        """Test Case 4.1.2: Binding Property (Negative Test)"""
        commitment, decom_info = self.scheme.commit(self.data)
        
        fake_data = self.data.copy()
        fake_data[0] = 1 - fake_data[0]  # Flip first bit
        
        is_valid_fake = self.scheme.verify(commitment, fake_data, decom_info)
        assert is_valid_fake is False

    def test_subset_opening(self):
        """Test Case 4.2.1: Correct Subset Opening"""
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        # Verify subset data matches
        assert len(subset_data) == len(test_indices)
        assert np.array_equal(subset_data, full_data[test_indices])
        
        # Verify proof
        is_valid = self.scheme.verify(commitment, subset_data, proof)
        assert is_valid is True

    def test_tampered_subset_rejected(self):
        """Test Case 4.2.2: Tampered Subset Rejected"""
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        tampered_subset = subset_data.copy()
        tampered_subset[0] = 1 - tampered_subset[0]
        
        is_valid = self.scheme.verify(commitment, tampered_subset, proof)
        assert is_valid is False


class TestMerkleCommitment:
    """Tests for Merkle Tree Commitment Scheme."""

    def setup_method(self):
        self.scheme = MerkleCommitment()
        self.data = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)

    def test_commit_verify_correctness(self):
        commitment, decom_info = self.scheme.commit(self.data)
        
        assert len(commitment) == 32
        assert isinstance(commitment, bytes)
        
        # Verify mathematical correctness manually for a small tree
        # Data: [0, 1, 1, 0, 1, 0, 1, 1] (8 leaves)
        tree, salts = decom_info
        leaves = []
        for i in range(len(self.data)):
            leaf = hashlib.sha256(salts[i] + self.data[i].tobytes()).digest()
            leaves.append(leaf)
        
        # Build tree manually
        level = leaves
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i+1] if i+1 < len(level) else left
                parent = hashlib.sha256(left + right).digest()
                next_level.append(parent)
            level = next_level
        
        expected_root = level[0]
        assert commitment == expected_root, "Merkle root does not match manual calculation"

        # Full verification
        is_valid = self.scheme.verify(commitment, self.data, salts)
        assert is_valid is True

    def test_binding_property(self):
        commitment, decom_info = self.scheme.commit(self.data)
        tree, salts = decom_info
        
        fake_data = self.data.copy()
        fake_data[0] = 1 - fake_data[0]
        
        is_valid_fake = self.scheme.verify(commitment, fake_data, salts)
        assert is_valid_fake is False

    def test_subset_opening(self):
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        assert len(subset_data) == len(test_indices)
        assert np.array_equal(subset_data, full_data[test_indices])
        
        is_valid = self.scheme.verify(commitment, subset_data, proof)
        assert is_valid is True

    def test_tampered_subset_rejected(self):
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        tampered_subset = subset_data.copy()
        tampered_subset[0] = 1 - tampered_subset[0]
        
        is_valid = self.scheme.verify(commitment, tampered_subset, proof)
        assert is_valid is False


def test_benchmark_commitments():
    """Benchmark SHA-256 vs Merkle Commitment."""
    N = 10000
    T_size = 500
    data = np.random.randint(0, 2, N, dtype=np.uint8)
    indices = np.random.choice(N, T_size, replace=False)
    indices.sort() # Sort for consistency
    
    sha_scheme = SHA256Commitment()
    merkle_scheme = MerkleCommitment()
    
    print(f"\nBenchmark (N={N}, T={T_size}):")
    
    # --- SHA-256 ---
    start = time.time()
    sha_com, sha_decom = sha_scheme.commit(data)
    sha_commit_time = time.time() - start
    
    start = time.time()
    sha_sub, sha_proof = sha_scheme.open_subset(indices, data, sha_decom)
    sha_open_time = time.time() - start
    
    start = time.time()
    sha_valid = sha_scheme.verify(sha_com, sha_sub, sha_proof)
    sha_verify_time = time.time() - start
    
    # Proof size: indices + full_data + salt
    # indices: T * 8 bytes (int64)
    # full_data: N bytes
    # salt: 32 bytes
    sha_proof_size = indices.nbytes + data.nbytes + 32
    
    assert sha_valid
    
    # --- Merkle ---
    start = time.time()
    merkle_com, merkle_decom = merkle_scheme.commit(data)
    merkle_commit_time = time.time() - start
    
    start = time.time()
    merkle_sub, merkle_proof = merkle_scheme.open_subset(indices, data, merkle_decom)
    merkle_open_time = time.time() - start
    
    start = time.time()
    merkle_valid = merkle_scheme.verify(merkle_com, merkle_sub, merkle_proof)
    merkle_verify_time = time.time() - start
    
    # Proof size: indices + salts + proofs
    # indices: T * 8 bytes
    # salts: T * 32 bytes
    # proofs: T * log2(N) * 32 bytes
    merkle_proof_bytes = 0
    merkle_proof_bytes += indices.nbytes
    merkle_proof_bytes += sum(len(s) for s in merkle_proof[1])
    for p in merkle_proof[2]:
        merkle_proof_bytes += sum(len(h) for h in p)
        
    assert merkle_valid
    
    print(f"{'Metric':<20} | {'SHA-256':<15} | {'Merkle':<15}")
    print("-" * 56)
    print(f"{'Commit Time (s)':<20} | {sha_commit_time:<15.5f} | {merkle_commit_time:<15.5f}")
    print(f"{'Open Time (s)':<20} | {sha_open_time:<15.5f} | {merkle_open_time:<15.5f}")
    print(f"{'Verify Time (s)':<20} | {sha_verify_time:<15.5f} | {merkle_verify_time:<15.5f}")
    print(f"{'Proof Size (bytes)':<20} | {sha_proof_size:<15} | {merkle_proof_bytes:<15}")

    # --- Theoretical Crossover Analysis ---
    # SHA-256 Proof Size ≈ N * item_size
    # Merkle Proof Size ≈ T * (32 + log2(N) * 32)
    # Crossover when N * item_size > T * 32 * (1 + log2(N))
    # item_size > (T/N) * 32 * (1 + log2(N))
    
    log2_N = np.log2(N)
    merkle_overhead_per_item = 32 * (1 + log2_N)
    crossover_item_size = (T_size / N) * merkle_overhead_per_item
    
    print(f"\nTheoretical Crossover Analysis (N={N}, T={T_size}):")
    print(f"Merkle overhead per opened item: {merkle_overhead_per_item:.2f} bytes")
    print(f"SHA-256 becomes inefficient when item size > {crossover_item_size:.2f} bytes")
    
    # Verify this with a simulated large item test
    # We simulate large items by just assuming the data array represents larger chunks
    # But our implementation hashes row-by-row.
    # If we had data shape (N, 1000), then item size is 1000 bytes.
    
    large_item_size = int(crossover_item_size * 2) # Should favor Merkle
    print(f"\nSimulating Large Items (Size={large_item_size} bytes):")
    
    # Create data with shape (N, large_item_size)
    # To save memory/time in test, we reduce N for this specific check
    N_sim = 1000
    T_sim = 50
    data_large = np.random.randint(0, 2, (N_sim, large_item_size), dtype=np.uint8)
    indices_sim = np.random.choice(N_sim, T_sim, replace=False)
    
    # SHA-256
    sha_com, sha_decom = sha_scheme.commit(data_large)
    sha_sub, sha_proof = sha_scheme.open_subset(indices_sim, data_large, sha_decom)
    sha_size = data_large.nbytes + indices_sim.nbytes + 32
    
    # Merkle
    merkle_com, merkle_decom = merkle_scheme.commit(data_large)
    merkle_sub, merkle_proof = merkle_scheme.open_subset(indices_sim, data_large, merkle_decom)
    
    merkle_size = indices_sim.nbytes
    merkle_size += sum(len(s) for s in merkle_proof[1])
    for p in merkle_proof[2]:
        merkle_size += sum(len(h) for h in p)
        
    print(f"{'Metric':<20} | {'SHA-256':<15} | {'Merkle':<15}")
    print("-" * 56)
    print(f"{'Proof Size (bytes)':<20} | {sha_size:<15} | {merkle_size:<15}")
    
    if merkle_size < sha_size:
        print(">> Merkle Tree is more bandwidth-efficient for these parameters.")
    else:
        print(">> SHA-256 is still more bandwidth-efficient.")

    def test_subset_opening(self):
        """Test Case 4.2.1: Correct Subset Opening"""
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        # Verify subset data matches
        assert len(subset_data) == len(test_indices)
        assert np.array_equal(subset_data, full_data[test_indices])
        
        # Verify proof
        is_valid = self.scheme.verify(commitment, subset_data, proof)
        assert is_valid is True

    def test_tampered_subset_rejected(self):
        """Test Case 4.2.2: Tampered Subset Rejected"""
        full_data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        commitment, decom_info = self.scheme.commit(full_data)
        test_indices = np.array([1, 3, 7, 9], dtype=np.int64)
        
        subset_data, proof = self.scheme.open_subset(test_indices, full_data, decom_info)
        
        tampered_subset = subset_data.copy()
        tampered_subset[0] = 1 - tampered_subset[0]
        
        is_valid = self.scheme.verify(commitment, tampered_subset, proof)
        assert is_valid is False
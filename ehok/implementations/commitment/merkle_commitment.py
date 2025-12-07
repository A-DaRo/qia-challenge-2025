"""
Merkle Tree based commitment scheme implementation.
"""

import hashlib
import secrets
from typing import Tuple, Any, List, Optional
import numpy as np
import math

from ehok.interfaces.commitment import ICommitmentScheme
from ehok.core.exceptions import CommitmentVerificationError


class MerkleTree:
    """
    Simple Merkle Tree implementation for commitment.
    """
    def __init__(self, leaves: List[bytes]):
        self.leaves = leaves
        self.levels = [leaves]
        self._build_tree()

    def _build_tree(self):
        current_level = self.leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Duplicate last element if odd number
                    right = left
                
                # Hash(left || right)
                parent = hashlib.sha256(left + right).digest()
                next_level.append(parent)
            
            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> bytes:
        if not self.levels:
            return b''
        return self.levels[-1][0]

    def get_proof(self, index: int) -> List[bytes]:
        """Get authentication path for a leaf index."""
        proof = []
        num_levels = len(self.levels)
        
        # We don't need the root in the proof
        for level_idx in range(num_levels - 1):
            level = self.levels[level_idx]
            
            # Find sibling index
            if index % 2 == 0:
                sibling_idx = index + 1
            else:
                sibling_idx = index - 1
            
            # Handle odd number of nodes (duplicate case)
            if sibling_idx >= len(level):
                sibling_idx = index # Sibling is self (duplicated)
            
            proof.append(level[sibling_idx])
            
            # Move to parent index
            index //= 2
            
        return proof

    @staticmethod
    def verify_proof(leaf: bytes, proof: List[bytes], root: bytes, index: int) -> bool:
        """Verify a Merkle proof."""
        current_hash = leaf
        
        for sibling in proof:
            if index % 2 == 0:
                # We are left, sibling is right
                current_hash = hashlib.sha256(current_hash + sibling).digest()
            else:
                # We are right, sibling is left
                current_hash = hashlib.sha256(sibling + current_hash).digest()
            
            index //= 2
            
        return current_hash == root


class MerkleCommitment(ICommitmentScheme):
    """
    Merkle Tree based commitment scheme.

    This implementation builds a Merkle tree over the salted hashes of the data rows.
    The commitment is the root of the tree (32 bytes).
    Opening a subset requires only O(log N) overhead per item.

    Attributes
    ----------
    salt_length : int
        Length of the random salt in bytes (default: 32).
    """

    def __init__(self, salt_length: int = 32):
        self.salt_length = salt_length

    def commit(self, data: np.ndarray) -> Tuple[bytes, Any]:
        """
        Generate a Merkle root commitment.

        Parameters
        ----------
        data : np.ndarray
            Data to commit.

        Returns
        -------
        commitment : bytes
            The Merkle Root (32 bytes).
        decommitment_info : Any
            (MerkleTree object, List[bytes] salts).
            We keep the tree object to generate proofs later.
        """
        n_items = data.shape[0]
        salts = []
        leaves = []

        for i in range(n_items):
            salt = secrets.token_bytes(self.salt_length)
            salts.append(salt)
            
            row_bytes = data[i].tobytes()
            # Leaf = Hash(salt || data)
            leaf = hashlib.sha256(salt + row_bytes).digest()
            leaves.append(leaf)

        tree = MerkleTree(leaves)
        
        # We return the tree and salts as decommitment info
        # In a real distributed system, Bob keeps this locally.
        return tree.root, (tree, salts)

    def verify(self, commitment: bytes, data: np.ndarray, 
               decommitment_info: Any) -> bool:
        """
        Verify Merkle proof or full tree.

        Parameters
        ----------
        commitment : bytes
            The Merkle Root.
        data : np.ndarray
            Data to verify.
        decommitment_info : Any
            If verifying subset: Tuple(indices, salts, proofs).
            If verifying full tree: List[bytes] (salts).

        Returns
        -------
        valid : bool
        """
        # Case 1: Full verification (rebuild tree)
        if isinstance(decommitment_info, list):
            salts = decommitment_info
            if len(salts) != len(data):
                return False
            
            leaves = []
            for i in range(len(data)):
                row_bytes = data[i].tobytes()
                leaf = hashlib.sha256(salts[i] + row_bytes).digest()
                leaves.append(leaf)
            
            # Rebuild tree to check root
            tree = MerkleTree(leaves)
            return tree.root == commitment

        # Case 2: Subset verification with proofs
        if not isinstance(decommitment_info, tuple) or len(decommitment_info) != 3:
            return False

        indices, salts, proofs = decommitment_info
        
        if len(data) != len(indices) or len(data) != len(salts) or len(data) != len(proofs):
            return False

        for i, idx in enumerate(indices):
            salt = salts[i]
            proof = proofs[i]
            row_bytes = data[i].tobytes()
            
            # Recompute leaf
            leaf = hashlib.sha256(salt + row_bytes).digest()
            
            # Verify proof
            if not MerkleTree.verify_proof(leaf, proof, commitment, int(idx)):
                return False
                
        return True

    def open_subset(self, indices: np.ndarray, data: np.ndarray,
                    decommitment_info: Any) -> Tuple[np.ndarray, Any]:
        """
        Open commitment for a subset.

        Parameters
        ----------
        indices : np.ndarray
            Indices to open.
        data : np.ndarray
            Full data.
        decommitment_info : Any
            (tree, salts) from commit().

        Returns
        -------
        subset_data : np.ndarray
        subset_proof : Any
            (indices, subset_salts, proofs)
        """
        tree, full_salts = decommitment_info
        
        subset_data = data[indices]
        subset_salts = [full_salts[i] for i in indices]
        proofs = [tree.get_proof(int(i)) for i in indices]
        
        return subset_data, (indices, subset_salts, proofs)

"""
SHA-256 based commitment scheme implementation.
"""

import hashlib
import secrets
from typing import Tuple, Any, List, Union, Optional
import numpy as np

from ehok.interfaces.commitment import ICommitmentScheme
from ehok.core.exceptions import CommitmentVerificationError


class SHA256Commitment(ICommitmentScheme):
    """
    SHA-256 hash-based commitment scheme.

    This implementation treats the commitment as a SINGLE SHA-256 hash of the
    entire data array (concatenated). This provides computational binding.
    
    Trade-off:
    - Compact commitment (32 bytes).
    - Inefficient opening: Must reveal ENTIRE data to verify even a subset.
    
    Attributes
    ----------
    salt_length : int
        Length of the random salt in bytes (default: 32).
    """

    def __init__(self, salt_length: int = 32):
        """
        Initialize the SHA-256 commitment scheme.

        Parameters
        ----------
        salt_length : int, optional
            Length of the random salt in bytes, by default 32.
        """
        self.salt_length = salt_length

    def commit(self, data: np.ndarray) -> Tuple[bytes, Any]:
        """
        Generate a commitment to data.

        Parameters
        ----------
        data : np.ndarray
            Data to commit.

        Returns
        -------
        commitment : bytes
            SHA-256 hash of (salt || data).
        decommitment_info : bytes
            The salt used.
        """
        salt = secrets.token_bytes(self.salt_length)
        data_bytes = data.tobytes()
        
        # H = SHA256(salt || data)
        commitment = hashlib.sha256(salt + data_bytes).digest()
        
        return commitment, salt

    def verify(self, commitment: bytes, data: np.ndarray, 
               decommitment_info: Any) -> bool:
        """
        Verify that data matches commitment.

        Parameters
        ----------
        commitment : bytes
            The commitment hash.
        data : np.ndarray
            Data to verify. 
            NOTE: For SHA256, this MUST be the FULL data.
            If called from open_subset context, the 'proof' contains the full data,
            and this 'data' argument might be just the subset.
            However, the interface implies 'data' is what we are verifying.
            
            If decommitment_info is a tuple (full_data, salt), we verify full_data
            against commitment, and optionally check if 'data' is consistent with full_data.
            
        decommitment_info : Any
            If full verification: bytes (salt).
            If subset verification: Tuple[np.ndarray, bytes] (full_data, salt).

        Returns
        -------
        valid : bool
        """
        full_data = None
        salt = None

        if isinstance(decommitment_info, bytes):
            # Full verification context where 'data' is full data
            full_data = data
            salt = decommitment_info
        elif isinstance(decommitment_info, tuple) and len(decommitment_info) == 3:
            # Subset verification context
            # decommitment_info contains (indices, full_data, salt)
            indices, full_data_in_proof, salt = decommitment_info
            
            # Verify that 'data' (subset) matches 'full_data' at 'indices'
            try:
                if not np.array_equal(data, full_data_in_proof[indices]):
                    return False
            except Exception:
                return False

            full_data = full_data_in_proof
        else:
            # logger.error("Invalid decommitment_info format")
            return False

        # Recompute hash
        try:
            data_bytes = full_data.tobytes()
            expected_hash = hashlib.sha256(salt + data_bytes).digest()
            
            if secrets.compare_digest(expected_hash, commitment):
                return True
            else:
                # logger.warning("SHA256 verification failed: Hash mismatch")
                return False
        except Exception as e:
            # logger.error(f"SHA256 verification error: {e}")
            return False

    def open_subset(self, indices: np.ndarray, data: np.ndarray,
                    decommitment_info: Any) -> Tuple[np.ndarray, Any]:
        """
        Open commitment for a subset of positions.

        For SHA-256, this requires revealing the ENTIRE data to allow verification.

        Parameters
        ----------
        indices : np.ndarray
            Indices to open.
        data : np.ndarray
            Full data array.
        decommitment_info : Any
            The salt.

        Returns
        -------
        subset_data : np.ndarray
            Data at specified indices.
        subset_proof : Tuple[np.ndarray, np.ndarray, bytes]
            (indices, full_data, salt) - The proof requires the full data.
        """
        subset_data = data[indices]
        salt = decommitment_info
        
        # Proof is the indices, full data and the salt
        subset_proof = (indices, data, salt)
        
        return subset_data, subset_proof

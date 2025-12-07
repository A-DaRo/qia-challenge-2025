"""
Abstract interface for commitment schemes.

This module defines the abstract base class for commitment schemes used in the
E-HOK protocol to ensure that Bob commits to his measurement outcomes and bases
before Alice reveals her bases.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np


class ICommitmentScheme(ABC):
    """
    Abstract interface for commitment schemes.
    
    A commitment scheme is a cryptographic primitive that allows one party (Bob)
    to commit to a value while keeping it hidden, with the ability to reveal it
    later. It must satisfy two properties:
    1. Binding: Bob cannot change the committed value after commitment
    2. Hiding: Alice cannot learn the committed value before opening
    
    Security Requirement
    --------------------
    Computationally binding commitment. Bob commits to (outcomes, bases) before
    Alice reveals her bases, preventing selective basis disclosure attacks.
    
    Notes
    -----
    The commitment phase is critical for E-HOK security. Without it, Bob could
    adaptively choose bases after learning Alice's choices, breaking the protocol.
    """
    
    @abstractmethod
    def commit(self, data: np.ndarray) -> Tuple[bytes, Any]:
        """
        Generate a commitment to data.
        
        This method creates a cryptographic commitment that binds Bob to specific
        measurement outcomes and bases while hiding the actual values from Alice.
        
        Parameters
        ----------
        data : np.ndarray
            Data to commit (concatenated outcomes || bases).
        
        Returns
        -------
        commitment : bytes
            The commitment value (e.g., hash digest).
        decommitment_info : Any
            Information needed to open commitment (e.g., salt/nonce).
        
        Notes
        -----
        For SHA-256: decommitment_info is the original data.
        For Merkle: decommitment_info includes tree structure.
        
        The commitment must be efficiently computable and the decommitment info
        must be kept secret until the opening phase.
        """
        pass
    
    @abstractmethod
    def verify(self, commitment: bytes, data: np.ndarray, 
               decommitment_info: Any) -> bool:
        """
        Verify that data matches commitment.
        
        This method checks whether the opened data is consistent with the
        original commitment, detecting any attempt to modify the committed values.
        
        Parameters
        ----------
        commitment : bytes
            The commitment to verify against.
        data : np.ndarray
            Data to verify.
        decommitment_info : Any
            Decommitment information from Bob.
        
        Returns
        -------
        valid : bool
            True if commitment is valid, False otherwise.
        
        Notes
        -----
        Verification failure indicates either protocol violation or data
        corruption, and should result in protocol abort.
        """
        pass
    
    @abstractmethod
    def open_subset(self, indices: np.ndarray, data: np.ndarray,
                    decommitment_info: Any) -> Tuple[np.ndarray, Any]:
        """
        Open commitment for a subset of positions.
        
        This method allows revealing only part of the committed data, which is
        useful for the test set verification phase where only selected positions
        need to be disclosed.
        
        Parameters
        ----------
        indices : np.ndarray
            Indices to open (test set T).
        data : np.ndarray
            Full data array.
        decommitment_info : Any
            Decommitment information.
        
        Returns
        -------
        subset_data : np.ndarray
            Data at specified indices.
        subset_proof : Any
            Proof for subset (for Merkle: authentication paths).
        
        Notes
        -----
        For SHA-256: Opens entire data (subset opening not optimized).
        For Merkle: Returns only authentication paths for indices, providing
        succinct proofs without revealing the full committed data.
        
        Subset opening enables efficiency gains in advanced implementations
        (e.g., Merkle trees) while maintaining security.
        """
        pass

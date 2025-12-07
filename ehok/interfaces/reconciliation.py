"""
Abstract interface for information reconciliation (error correction).

This module defines the abstract base class for reconciliation algorithms used
to correct errors in the sifted key caused by quantum noise.
"""

from abc import ABC, abstractmethod
import numpy as np


class IReconciliator(ABC):
    """
    Abstract interface for information reconciliation (error correction).
    
    Information reconciliation is the process of correcting errors between Alice's
    and Bob's sifted keys caused by quantum channel noise. The reconciliation
    must be efficient and minimize information leakage to potential eavesdroppers.
    
    Goal
    ----
    Correct errors in sifted key using syndrome-based methods (e.g., LDPC codes).
    
    Security
    --------
    Leakage must be accounted for in privacy amplification. The syndrome
    transmitted from Alice to Bob reveals information about the key that must be
    compressed out in the final privacy amplification step.
    
    Notes
    -----
    Modern QKD systems typically use LDPC codes or Cascade for reconciliation.
    LDPC codes provide predictable performance and are preferred for baseline
    implementations.
    """
    
    @abstractmethod
    def compute_syndrome(self, key: np.ndarray) -> np.ndarray:
        """
        Compute syndrome from key (Alice's side).
        
        The syndrome is computed by multiplying the key by a parity check matrix.
        It captures information about potential errors without revealing the key
        itself (though it does leak some information to eavesdroppers).
        
        Parameters
        ----------
        key : np.ndarray
            Sifted key bits (after removing test set).
        
        Returns
        -------
        syndrome : np.ndarray
            Syndrome vector S = H @ key (mod 2).
        
        Mathematical Definition
        -----------------------
        Given parity check matrix H ∈ GF(2)^{m×n}:
            S = H · key (mod 2)
        
        Where m is the number of parity checks and n is the key length.
        """
        pass
    
    @abstractmethod
    def reconcile(self, key: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        """
        Correct errors using received syndrome (Bob's side).
        
        Using the syndrome received from Alice and his own noisy key, Bob attempts
        to correct errors and recover Alice's key. This typically involves iterative
        decoding algorithms (e.g., belief propagation for LDPC codes).
        
        Parameters
        ----------
        key : np.ndarray
            Bob's noisy sifted key.
        syndrome : np.ndarray
            Syndrome received from Alice.
        
        Returns
        -------
        corrected_key : np.ndarray
            Error-corrected key matching Alice's.
        
        Mathematical Definition
        -----------------------
        Find error vector e such that:
            H · (key ⊕ e) = syndrome (mod 2)
        Return: key ⊕ e
        
        Notes
        -----
        If decoding fails (e.g., too many errors), this method should raise
        ReconciliationFailedError. The protocol must then abort.
        """
        pass
    
    @abstractmethod
    def estimate_leakage(self, syndrome_length: int, qber: float) -> float:
        """
        Estimate information leakage from reconciliation.
        
        The syndrome transmitted during reconciliation reveals information to
        potential eavesdroppers. This information must be accounted for when
        computing the secure final key length in privacy amplification.
        
        Parameters
        ----------
        syndrome_length : int
            Length of syndrome (number of parity checks).
        qber : float
            Measured quantum bit error rate.
        
        Returns
        -------
        leakage : float
            Information leaked to Eve (in bits).
        
        Notes
        -----
        Conservative estimate: leakage ≈ syndrome_length + safety_margin.
        
        More precise estimates can use Shannon entropy:
            leakage = n · h(qber)
        where h(x) = -x·log₂(x) - (1-x)·log₂(1-x) is binary entropy.
        
        For baseline implementation, a conservative estimate is preferred to
        ensure security at the cost of some efficiency.
        """
        pass

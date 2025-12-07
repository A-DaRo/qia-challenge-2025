"""
Core data structures for the E-HOK protocol.

This module defines the fundamental data structures used throughout the E-HOK
baseline implementation, including the oblivious key representation, measurement
records, and protocol execution results.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class ObliviousKey:
    """
    Represents the output of the E-HOK protocol.
    
    An oblivious key is a shared secret where one party (Bob) has partial
    knowledge and the other party (Alice) has complete knowledge. This is the
    fundamental output of the E-HOK protocol used for secure multiparty
    computation.
    
    Attributes
    ----------
    key_value : np.ndarray
        Final key bits as uint8 array (values 0 or 1).
    knowledge_mask : np.ndarray
        Mask indicating knowledge: 0 = known, 1 = unknown (oblivious).
        For Alice: all zeros. For Bob: 1s at positions corresponding to I_1.
    security_param : float
        Estimated epsilon security parameter (εsec).
    qber : float
        Measured quantum bit error rate on the test set.
    final_length : int
        Length of the final key after privacy amplification.
    
    Notes
    -----
    The knowledge mask enables the oblivious key property: Alice knows all bits,
    while Bob knows only a subset. This asymmetric knowledge is crucial for
    certain secure computation protocols.
    """
    key_value: np.ndarray
    knowledge_mask: np.ndarray
    security_param: float
    qber: float
    final_length: int
    
    def __post_init__(self):
        """
        Validate data structure consistency.
        
        Raises
        ------
        AssertionError
            If validation checks fail.
        """
        assert self.key_value.shape == self.knowledge_mask.shape, \
            "Key and mask must have same shape"
        assert self.key_value.dtype == np.uint8, "Key must be uint8"
        assert self.knowledge_mask.dtype == np.uint8, "Mask must be uint8"
        assert np.all((self.key_value == 0) | (self.key_value == 1)), \
            "Key values must be 0 or 1"
        assert np.all((self.knowledge_mask == 0) | (self.knowledge_mask == 1)), \
            "Mask values must be 0 or 1"


@dataclass
class MeasurementRecord:
    """
    Record of a single EPR measurement.
    
    This structure captures all relevant information about a quantum measurement
    performed during the E-HOK protocol's quantum phase.
    
    Attributes
    ----------
    outcome : int
        Measurement outcome (0 or 1).
    basis : int
        Measurement basis (0 = Z, 1 = X).
    timestamp : float
        Simulation time when measurement occurred (ns).
    
    Notes
    -----
    Basis encoding follows standard quantum information conventions:
    - 0 = Z-basis (computational basis: |0⟩, |1⟩)
    - 1 = X-basis (Hadamard basis: |+⟩, |-⟩)
    """
    outcome: int
    basis: int
    timestamp: float
    
    def __post_init__(self):
        """
        Validate measurement record.
        
        Raises
        ------
        AssertionError
            If outcome or basis values are invalid.
        """
        assert self.outcome in [0, 1], "Outcome must be 0 or 1"
        assert self.basis in [0, 1], "Basis must be 0 (Z) or 1 (X)"


@dataclass
class ProtocolResult:
    """
    Complete protocol execution result with statistics.
    
    This structure encapsulates all outcomes and metrics from a complete E-HOK
    protocol execution, enabling detailed analysis and debugging.
    
    Attributes
    ----------
    oblivious_key : Optional[ObliviousKey]
        The final oblivious key (None if protocol aborted).
    success : bool
        Whether protocol completed successfully.
    abort_reason : Optional[str]
        Reason for abort (if success=False).
    raw_count : int
        Number of raw EPR pairs generated.
    sifted_count : int
        Number of sifted bits (matching bases, |I_0|).
    test_count : int
        Number of bits used for error estimation (|T|).
    final_count : int
        Number of bits after privacy amplification.
    qber : float
        Quantum bit error rate measured on test set.
    execution_time_ms : float
        Total protocol execution time (simulation time).
    
    Notes
    -----
    The counts provide insight into protocol efficiency:
    - raw_count: Total quantum resources consumed
    - sifted_count: Effective bits after basis reconciliation
    - test_count: Bits sacrificed for security verification
    - final_count: Secure key length after all post-processing
    """
    oblivious_key: Optional[ObliviousKey]
    success: bool
    abort_reason: Optional[str]
    raw_count: int
    sifted_count: int
    test_count: int
    final_count: int
    qber: float
    execution_time_ms: float

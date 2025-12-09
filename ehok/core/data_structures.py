"""
Core data structures for the E-HOK protocol.

This module defines the fundamental data structures used throughout the E-HOK
baseline implementation, including the oblivious key representation, measurement
records, and protocol execution results.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from .exceptions import EHOKException


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
    certain secure computation protocols. Baseline simulations may set
    ``ProtocolConfig.approximate_knowledge_mask`` to ``True``, in which case the
    mask encodes an approximation rather than an exact mapping through the
    privacy amplification matrix.
    """
    key_value: np.ndarray
    knowledge_mask: np.ndarray
    security_param: float
    qber: float
    final_length: int
    
    def __post_init__(self) -> None:
        """Validate data structure consistency."""
        if self.key_value.shape != self.knowledge_mask.shape:
            raise ValueError("Key and mask must have same shape")
        if self.key_value.dtype != np.uint8:
            raise ValueError("Key must be uint8")
        if self.knowledge_mask.dtype != np.uint8:
            raise ValueError("Mask must be uint8")
        if not np.all((self.key_value == 0) | (self.key_value == 1)):
            raise ValueError("Key values must be 0 or 1")
        if not np.all((self.knowledge_mask == 0) | (self.knowledge_mask == 1)):
            raise ValueError("Mask values must be 0 or 1")
        if len(self.key_value) != self.final_length:
            raise ValueError("final_length must equal key length")
        if len(self.knowledge_mask) != self.final_length:
            raise ValueError("knowledge_mask length must equal final_length")
        if not 0.0 <= float(self.qber) <= 1.0:
            raise ValueError("qber must be in [0, 1]")
        if float(self.security_param) <= 0:
            raise ValueError("security_param must be positive")


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
    
    def __post_init__(self) -> None:
        """Validate measurement record."""
        if self.outcome not in (0, 1):
            raise ValueError("Outcome must be 0 or 1")
        if self.basis not in (0, 1):
            raise ValueError("Basis must be 0 (Z) or 1 (X)")


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

    def __post_init__(self) -> None:
        """Validate protocol result invariants."""
        if self.raw_count < 0:
            raise ValueError("raw_count must be non-negative")
        if self.sifted_count < 0 or self.sifted_count > self.raw_count:
            raise ValueError("sifted_count must satisfy 0 <= sifted_count <= raw_count")
        if self.test_count < 0:
            raise ValueError("test_count must be non-negative")
        if self.final_count < 0:
            raise ValueError("final_count must be non-negative")
        if self.sifted_count < self.test_count + self.final_count:
            raise ValueError("sifted_count must be >= test_count + final_count")
        if not 0.0 <= float(self.qber) <= 1.0:
            raise ValueError("qber must be in [0, 1]")
        if self.success and self.oblivious_key is None:
            raise ValueError("oblivious_key must be set when success is True")
        if self.oblivious_key is not None and self.final_count != self.oblivious_key.final_length:
            raise ValueError("final_count must equal oblivious_key.final_length when key is present")


@dataclass
class ExecutionMetrics(ProtocolResult):
    """Extended protocol metrics derived from :class:`ProtocolResult`.

    This dataclass enriches the base result with commonly used derived
    quantities so later analysis layers do not need to recompute them.
    """

    raw_to_sifted_ratio: Optional[float] = None
    sifted_to_final_ratio: Optional[float] = None
    leakage_bits: Optional[float] = None

    def __post_init__(self) -> None:
        # Validate base fields first
        super().__post_init__()

        if self.raw_count > 0:
            self.raw_to_sifted_ratio = self.sifted_count / self.raw_count
        if self.sifted_count > 0:
            self.sifted_to_final_ratio = (
                self.final_count / self.sifted_count
            )

        # leakage_bits is left to upstream calculators; ensure non-negative if set
        if self.leakage_bits is not None and self.leakage_bits < 0:
            raise ValueError("leakage_bits must be non-negative when provided")


@dataclass
class LDPCBlockResult:
    """
    Result of processing a single LDPC block during reconciliation.

    Attributes
    ----------
    verified : bool
        True if hash verification succeeded for the block.
    error_count : int
        Number of errors corrected (Hamming weight of error vector).
    block_length : int
        Length of the payload portion (excludes padding) in bits.
    syndrome_length : int
        Number of syndrome bits transmitted for this block.
    hash_bits : int
        Number of verification hash bits transmitted (default 50).
    """

    verified: bool
    error_count: int
    block_length: int
    syndrome_length: int
    hash_bits: int = 50

    def __post_init__(self) -> None:
        if self.block_length < 0:
            raise ValueError("block_length must be non-negative")
        if self.error_count < 0:
            raise ValueError("error_count must be non-negative")
        if self.error_count > self.block_length:
            raise ValueError("error_count cannot exceed block_length")
        if self.syndrome_length < 0:
            raise ValueError("syndrome_length must be non-negative")
        if self.hash_bits <= 0:
            raise ValueError("hash_bits must be positive")
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be a boolean")


@dataclass
class LDPCMatrixPool:
    """
    Pool of pre-generated LDPC matrices at different code rates.

    Attributes
    ----------
    frame_size : int
        Fixed frame size ``n`` for all matrices.
    matrices : Dict[float, sp.spmatrix]
        Mapping from code rate to parity-check matrix in CSR format.
    rates : np.ndarray
        Sorted array of available code rates for selection.
    checksum : str
        SHA-256 checksum over the matrix pool used for synchronization.
    """

    frame_size: int
    matrices: Dict[float, sp.spmatrix]
    rates: np.ndarray
    checksum: str

    def __post_init__(self) -> None:
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if self.rates.size == 0:
            raise ValueError("rates must not be empty")
        if not np.all(np.diff(self.rates) > 0):
            raise ValueError("rates must be strictly increasing")
        for rate, matrix in self.matrices.items():
            if matrix.shape[1] != self.frame_size:
                raise ValueError(
                    f"Matrix for rate {rate} has incompatible frame size {matrix.shape[1]}"
                )
        if not isinstance(self.checksum, str) or not self.checksum:
            raise ValueError("checksum must be a non-empty string")


@dataclass
class LDPCReconciliationResult:
    """
    Aggregate result of LDPC reconciliation across all processed blocks.

    Attributes
    ----------
    corrected_key : np.ndarray
        Concatenated verified payload bits after reconciliation.
    qber_estimate : float
        Integrated QBER estimate computed from block outcomes.
    total_leakage : int
        Total information leakage (syndrome plus hash bits).
    blocks_processed : int
        Number of LDPC blocks processed.
    blocks_verified : int
        Number of blocks that passed verification.
    blocks_discarded : int
        Number of blocks discarded due to decoder failure or hash mismatch.
    """

    corrected_key: np.ndarray
    qber_estimate: float
    total_leakage: int
    blocks_processed: int
    blocks_verified: int
    blocks_discarded: int

    def __post_init__(self) -> None:
        if self.total_leakage < 0:
            raise ValueError("total_leakage must be non-negative")
        if self.blocks_processed < 0:
            raise ValueError("blocks_processed must be non-negative")
        if self.blocks_verified < 0:
            raise ValueError("blocks_verified must be non-negative")
        if self.blocks_discarded < 0:
            raise ValueError("blocks_discarded must be non-negative")
        if self.blocks_verified + self.blocks_discarded > self.blocks_processed:
            raise ValueError("verified + discarded cannot exceed processed blocks")
        if not 0.0 <= float(self.qber_estimate) <= 1.0:
            raise ValueError("qber_estimate must be in [0, 1]")
        if self.corrected_key.dtype != np.uint8:
            raise ValueError("corrected_key must be uint8")

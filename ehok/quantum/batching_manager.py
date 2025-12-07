"""
EPR Batching Manager for streaming quantum operations.

Handles generation of large numbers of EPR pairs while respecting
quantum memory constraints.
"""

import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass
from math import ceil

from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.epr_socket import EprMeasureResult, RandomBasis

from ..core.constants import BATCH_SIZE, TOTAL_EPR_PAIRS
from ..utils.logging import get_logger
from .basis_selection import BasisSelector

logger = get_logger("quantum.batching_manager")


@dataclass
class BatchResult:
    """
    Result of a single batch of EPR measurements.
    
    Attributes
    ----------
    outcomes : np.ndarray
        Measurement outcomes (0 or 1), shape (batch_size,).
    bases : np.ndarray
        Measurement bases (0=Z, 1=X), shape (batch_size,).
    timestamps : np.ndarray
        Simulation time of each measurement (ns), shape (batch_size,).
    batch_index : int
        Index of this batch in the sequence.
    """
    outcomes: np.ndarray
    bases: np.ndarray
    timestamps: np.ndarray
    batch_index: int


class BatchingManager:
    """
    Manage streaming EPR generation and measurement.
    
    Coordinates batch-by-batch EPR pair generation to overcome
    quantum memory limitations.
    """
    
    def __init__(
        self,
        total_pairs: int = TOTAL_EPR_PAIRS,
        batch_size: int = BATCH_SIZE
    ):
        self.total_pairs = total_pairs
        self.batch_size = batch_size
        self.num_batches = ceil(total_pairs / batch_size)
        self.actual_pairs = self.num_batches * batch_size
        
        logger.info(
            f"BatchingManager: {self.actual_pairs} pairs "
            f"({self.num_batches} batches of {self.batch_size})"
        )
    
    def compute_batch_sizes(self) -> List[int]:
        """Compute size of each batch."""
        sizes = [self.batch_size] * self.num_batches
        remainder = self.total_pairs % self.batch_size
        if remainder != 0:
            sizes[-1] = remainder
        return sizes


class EPRGenerator:
    """
    Generate EPR pairs using SquidASM EPRSocket with batching.
    
    Uses create_keep/recv_keep followed by manual measurement to ensure
    independent random basis selection and correct result extraction.
    """
    
    def __init__(self, epr_socket: EPRSocket, role: str):
        self.epr_socket = epr_socket
        self.role = role.lower()
        assert self.role in ["alice", "bob"], "Role must be 'alice' or 'bob'"
        
        self.basis_selector = BasisSelector()
        logger.info(f"EPRGenerator initialized for role={self.role}")
    
    def generate_batch_alice(
        self, 
        batch_size: int,
        sim_time_ns: float
    ) -> List[Qubit]:
        """
        Generate EPR batch (Alice's side - creator).
        
        Returns list of Qubits (futures) to be measured.
        """
        logger.debug(f"Alice generating batch: size={batch_size}")
        
        # Use create_keep to get qubits in memory
        qubits: List[Qubit] = self.epr_socket.create_keep(
            number=batch_size
        )
        return qubits
    
    def generate_batch_bob(
        self,
        batch_size: int,
        sim_time_ns: float
    ) -> List[Qubit]:
        """
        Receive EPR batch (Bob's side - receiver).
        
        Returns list of Qubits (futures) to be measured.
        """
        logger.debug(f"Bob receiving batch: size={batch_size}")
        
        # Use recv_keep to get qubits in memory
        qubits: List[Qubit] = self.epr_socket.recv_keep(
            number=batch_size
        )
        return qubits

    def measure_batch(
        self,
        qubits: List[Qubit],
        sim_time_ns: float
    ) -> tuple[List[Any], np.ndarray]:
        """
        Measure a batch of qubits in random bases.
        
        Parameters
        ----------
        qubits : List[Qubit]
            List of qubits to measure.
        sim_time_ns : float
            Current simulation time.
            
        Returns
        -------
        outcome_futures : List[Future]
            List of measurement outcome futures.
        bases : np.ndarray
            Array of chosen bases (0=Z, 1=X).
        """
        batch_size = len(qubits)
        
        # Generate random bases locally
        bases = self.basis_selector.generate_bases(batch_size)
        
        outcome_futures = []
        for i, qubit in enumerate(qubits):
            basis_enum = bases[i] # 0 or 1
            # Map 0->Z, 1->X. NetQASM measure() takes string or enum?
            # Usually measure() takes no args (Z) or we apply rotation then measure.
            # Or qubit.measure(future=...)
            
            # NetQASM Qubit.measure() usually measures in Z.
            # To measure in X, we apply H then measure.
            
            if basis_enum == 1: # X-basis
                qubit.H()
            
            # Measure in Z-basis (standard)
            outcome = qubit.measure()
            outcome_futures.append(outcome)
            
        return outcome_futures, bases

    @staticmethod
    def extract_batch_results(
        outcome_futures: List[Any],
        bases: np.ndarray,
        sim_time_ns: float
    ) -> BatchResult:
        """
        Extract measurement outcomes from futures after flush().
        """
        batch_size = len(outcome_futures)
        
        # Extract outcomes (now available after flush)
        # Note: outcome_futures are NetQASM Futures, cast to int
        outcomes = np.array(
            [int(f) for f in outcome_futures],
            dtype=np.uint8
        )
        
        timestamps = np.full(batch_size, sim_time_ns, dtype=np.float64)
        
        return BatchResult(
            outcomes=outcomes,
            bases=bases,
            timestamps=timestamps,
            batch_index=0  # Will be set by caller
        )

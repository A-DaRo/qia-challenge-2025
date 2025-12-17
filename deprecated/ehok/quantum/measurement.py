"""
Measurement buffering and processing logic.

Handles storage and processing of quantum measurement results.
"""

from typing import List
import numpy as np

from ..core.data_structures import MeasurementRecord
from ..utils.logging import get_logger

logger = get_logger("quantum.measurement")

class MeasurementBuffer:
    """
    Buffer for storing and managing measurement records.
    """
    
    def __init__(self):
        self.records: List[MeasurementRecord] = []
        
    def add_batch(self, outcomes: np.ndarray, bases: np.ndarray, timestamps: np.ndarray):
        """
        Add a batch of measurement results to the buffer.
        
        Parameters
        ----------
        outcomes : np.ndarray
            Measurement outcomes (0 or 1).
        bases : np.ndarray
            Measurement bases (0=Z, 1=X).
        timestamps : np.ndarray
            Simulation timestamps.
        """
        count = len(outcomes)
        if len(bases) != count or len(timestamps) != count:
            raise ValueError("Input arrays must have same length")
            
        for i in range(count):
            record = MeasurementRecord(
                outcome=int(outcomes[i]),
                basis=int(bases[i]),
                timestamp=float(timestamps[i])
            )
            self.records.append(record)
            
        logger.debug(f"Added {count} records to buffer. Total: {len(self.records)}")
        
    def get_outcomes(self) -> np.ndarray:
        """Get all outcomes as numpy array."""
        return np.array([r.outcome for r in self.records], dtype=np.uint8)
        
    def get_bases(self) -> np.ndarray:
        """Get all bases as numpy array."""
        return np.array([r.basis for r in self.records], dtype=np.uint8)
        
    def clear(self):
        """Clear the buffer."""
        self.records.clear()
        
    def __len__(self):
        return len(self.records)

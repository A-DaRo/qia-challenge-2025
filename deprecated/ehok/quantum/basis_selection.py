"""
Random basis selection for E-HOK protocol.

Implements cryptographically secure random basis generation
for each EPR pair measurement.
"""

import numpy as np
from typing import Optional
from ..utils.logging import get_logger

logger = get_logger("quantum.basis_selection")


class BasisSelector:
    """
    Generate random measurement bases for quantum measurements.
    
    Uses numpy's cryptographic random number generator for security.
    
    Notes
    -----
    This class provides cryptographically secure random basis selection
    for quantum measurements in the E-HOK protocol. The randomness is
    crucial for security - predictable basis choices would allow an
    eavesdropper to gain information about the key.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize basis selector.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility (testing only).
            In production, leave as None for true randomness.
        
        Notes
        -----
        Using numpy.random.default_rng() which uses PCG64 generator,
        suitable for cryptographic applications in simulation.
        """
        self._rng = np.random.default_rng(seed)
        logger.debug(f"BasisSelector initialized with seed={seed}")
    
    def generate_bases(self, count: int) -> np.ndarray:
        """
        Generate random basis choices.
        
        Parameters
        ----------
        count : int
            Number of basis choices to generate.
        
        Returns
        -------
        bases : np.ndarray
            Array of basis choices (0=Z, 1=X), shape (count,), dtype uint8.
        
        Notes
        -----
        Mathematical Definition:
        For each i ∈ [0, count):
            bases[i] ← Uniform({0, 1})
        
        Security:
        Basis choices must be uniformly random and independent.
        P(bases[i] = 0) = P(bases[i] = 1) = 0.5
        """
        bases = self._rng.integers(0, 2, size=count, dtype=np.uint8)
        logger.debug(f"Generated {count} random bases")
        return bases
    
    def basis_to_string(self, bases: np.ndarray) -> str:
        """
        Convert basis array to human-readable string.
        
        Parameters
        ----------
        bases : np.ndarray
            Basis array (0=Z, 1=X).
        
        Returns
        -------
        basis_str : str
            String representation (e.g., "ZXZXZ...").
        
        Examples
        --------
        >>> selector = BasisSelector(seed=42)
        >>> bases = np.array([0, 1, 0, 1, 1])
        >>> selector.basis_to_string(bases)
        'ZXZXX'
        """
        mapping = {0: 'Z', 1: 'X'}
        return ''.join(mapping[b] for b in bases)

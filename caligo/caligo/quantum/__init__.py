"""
Quantum operations package for E-HOK protocol Phase I.

This package provides the quantum measurement primitives for EPR pair
generation, basis selection, and measurement execution in the Noisy
Storage Model.

Public API
----------
EPRGenerator
    Generates EPR pairs using SquidASM's EPRSocket.
BasisSelector
    Uniform random basis selection for BB84-style measurements.
MeasurementExecutor
    Executes quantum measurements and records outcomes.
BatchingManager
    Memory-constrained batch processing for large key generation.

References
----------
- Erven et al. (2014): Experimental implementation
- Schaffner et al. (2009): Protocol definition
"""

from caligo.quantum.basis import BasisSelector
from caligo.quantum.batching import BatchingManager, BatchConfig, BatchResult
from caligo.quantum.epr import EPRGenerator, EPRGenerationConfig
from caligo.quantum.measurement import MeasurementExecutor, MeasurementBuffer

__all__ = [
    # EPR generation
    "EPRGenerator",
    "EPRGenerationConfig",
    # Basis selection
    "BasisSelector",
    # Measurement
    "MeasurementExecutor",
    "MeasurementBuffer",
    # Batching
    "BatchingManager",
    "BatchConfig",
    "BatchResult",
]

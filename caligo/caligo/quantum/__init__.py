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

Parallel Generation
-------------------
ParallelEPRConfig
    Configuration for parallel EPR generation.
ParallelEPROrchestrator
    Orchestrates parallel EPR generation across worker processes.
EPRGenerationFactory
    Factory for creating EPR generation strategies.
SequentialEPRStrategy
    Sequential EPR generation (original behavior).
ParallelEPRStrategy
    Parallel EPR generation using multiprocessing.

References
----------
- Erven et al. (2014): Experimental implementation
- Schaffner et al. (2009): Protocol definition
- parallel_generation.md: Parallel EPR generation design document
"""

from caligo.quantum.basis import BasisSelector
from caligo.quantum.batching import BatchingManager, BatchConfig, BatchResult
from caligo.quantum.epr import EPRGenerator, EPRGenerationConfig
from caligo.quantum.measurement import MeasurementExecutor, MeasurementBuffer

# Parallel generation modules
from caligo.quantum.parallel import (
    ParallelEPRConfig,
    ParallelEPROrchestrator,
    EPRWorkerResult,
)
from caligo.quantum.factory import (
    EPRGenerationStrategy,
    SequentialEPRStrategy,
    ParallelEPRStrategy,
    EPRGenerationFactory,
    CaligoConfig,
)
from caligo.quantum.workers import (
    MinimalAliceWorkerProgram,
    MinimalBobWorkerProgram,
    EPRWorkerTask,
    generate_epr_batch_standalone,
)

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
    # Parallel generation - Config
    "ParallelEPRConfig",
    "CaligoConfig",
    # Parallel generation - Orchestration
    "ParallelEPROrchestrator",
    "EPRWorkerResult",
    # Parallel generation - Factory & Strategies
    "EPRGenerationStrategy",
    "EPRGenerationFactory",
    "SequentialEPRStrategy",
    "ParallelEPRStrategy",
    # Parallel generation - Workers
    "MinimalAliceWorkerProgram",
    "MinimalBobWorkerProgram",
    "EPRWorkerTask",
    "generate_epr_batch_standalone",
]

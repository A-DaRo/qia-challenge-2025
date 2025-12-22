"""
Caligo Exploration Suite

This package provides an autonomous adversarial stress-testing framework
for the Caligo OT protocol. It empirically verifies security boundaries
by executing the full protocol stack across a 9-dimensional parameter space.

Architecture
------------
The suite uses a "Pre-computation + Injection" pattern:

1. **Quantum Layer**: Parallel EPR generation via BatchedEPROrchestrator
2. **Classical Layer**: Protocol execution via ProtocolHarness
3. **Exploration Layer**: LHS warmup → GP surrogate → Bayesian active learning

Modules
-------
types
    Data structures: ExplorationSample, ProtocolResult, Phase states
persistence
    HDF5 storage and checkpoint management via StateManager
sampler
    Latin Hypercube Sampling for initial exploration
epr_batcher
    Batch orchestration for parallel EPR generation
harness
    Protocol execution sandbox with injection support
lhs_executor
    Phase 1: LHS warmup data generation
surrogate
    Phase 2: Twin Gaussian Process efficiency landscape
surrogate_trainer
    Phase 2 executor for GP training
active
    Phase 3: Bayesian optimizer with Numba-accelerated acquisition
active_executor
    Phase 3 executor for active stress testing

References
----------
- parameter_explor.md: Design specification
- König et al. (2012): NSM security model
- Erven et al. (2014): Experimental parameters

Example
-------
>>> from caligo.exploration import Phase1Executor, Phase2Executor, Phase3Executor
>>> from pathlib import Path
>>>
>>> # Phase 1: LHS warmup
>>> p1 = Phase1Executor(output_dir=Path("./exploration_results"))
>>> p1.run(num_samples=2000, batch_size=50)
>>>
>>> # Phase 2: Train surrogate model
>>> p2 = Phase2Executor(data_path=Path("./exploration_results/exploration_data.h5"))
>>> p2.run()
>>>
>>> # Phase 3: Active stress testing
>>> p3 = Phase3Executor(
...     data_path=Path("./exploration_results/exploration_data.h5"),
...     surrogate_path=Path("./exploration_results/surrogate.pkl"),
... )
>>> p3.run(num_iterations=100, batch_size=16)
"""

from caligo.exploration.types import (
    ExplorationSample,
    ProtocolResult,
    Phase1State,
    Phase2State,
    Phase3State,
    ExplorationConfig,
)
from caligo.exploration.persistence import StateManager, HDF5Writer
from caligo.exploration.sampler import LHSSampler, ParameterBounds
from caligo.exploration.epr_batcher import BatchedEPROrchestrator
from caligo.exploration.harness import ProtocolHarness, HarnessConfig
from caligo.exploration.lhs_executor import Phase1Executor
from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.exploration.surrogate_trainer import Phase2Executor
from caligo.exploration.active import BayesianOptimizer, AcquisitionConfig
from caligo.exploration.active_executor import Phase3Executor

__all__ = [
    # Data structures
    "ExplorationSample",
    "ProtocolResult",
    "Phase1State",
    "Phase2State",
    "Phase3State",
    "ExplorationConfig",
    # Persistence
    "StateManager",
    "HDF5Writer",
    # Sampling
    "LHSSampler",
    "ParameterBounds",
    # EPR generation
    "BatchedEPROrchestrator",
    # Protocol execution
    "ProtocolHarness",
    "HarnessConfig",
    # Executors
    "Phase1Executor",
    "Phase2Executor",
    "Phase3Executor",
    # Surrogate modeling
    "EfficiencyLandscape",
    # Bayesian optimization
    "BayesianOptimizer",
    "AcquisitionConfig",
]

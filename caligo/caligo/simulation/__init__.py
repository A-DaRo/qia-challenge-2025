"""
Caligo simulation package: SquidASM/NetSquid integration layer.

This package bridges the gap between NSM theoretical parameters and
NetSquid's discrete-event simulation engine. It implements the temporal
semantics required for NSM security.

Core Components
---------------
NSMParameters : dataclass
    Noisy Storage Model parameters with NetSquid noise model mappings.
ChannelParameters : dataclass
    Physical quantum channel parameters (fiber, detectors, memory).
TimingBarrier : class
    Enforces NSM wait time Δt as a causal barrier in simulation.
CaligoNetworkBuilder : class
    Factory for creating SquidASM network configurations.

Noise Models
------------
NSMStorageNoiseModel : class
    Wrapper for adversary storage noise modeling.
ChannelNoiseProfile : dataclass
    Aggregate noise profile for the trusted quantum channel.

Factory Functions
-----------------
create_depolar_noise_model(params) -> DepolarNoiseModel
    Create NetSquid depolarizing model from NSM parameters.
create_t1t2_noise_model(params) -> T1T2NoiseModel
    Create NetSquid T1T2 model for memory decoherence.

Configuration Presets
---------------------
perfect_network_config() -> StackNetworkConfig
    Noiseless network for unit testing.
realistic_network_config() -> StackNetworkConfig
    Realistic noise parameters for development.
erven_experimental_config() -> StackNetworkConfig
    Parameters matching Erven et al. (2014) experiment.

References
----------
- König et al. (2012): NSM definition, Markovian noise
- Erven et al. (2014): Experimental parameters, Δt semantics
- Schaffner et al. (2009): 11% QBER threshold, security conditions
"""

from caligo.simulation.physical_model import (
    NSMParameters,
    ChannelParameters,
    create_depolar_noise_model,
    create_t1t2_noise_model,
)

from caligo.simulation.timing import (
    TimingBarrier,
    TimingBarrierState,
)

from caligo.simulation.noise_models import (
    NSMStorageNoiseModel,
    ChannelNoiseProfile,
)

from caligo.simulation.network_builder import (
    CaligoNetworkBuilder,
    perfect_network_config,
    realistic_network_config,
    erven_experimental_config,
)

__all__ = [
    # Physical model
    "NSMParameters",
    "ChannelParameters",
    "create_depolar_noise_model",
    "create_t1t2_noise_model",
    # Timing
    "TimingBarrier",
    "TimingBarrierState",
    # Noise models
    "NSMStorageNoiseModel",
    "ChannelNoiseProfile",
    # Network builder
    "CaligoNetworkBuilder",
    "perfect_network_config",
    "realistic_network_config",
    "erven_experimental_config",
]

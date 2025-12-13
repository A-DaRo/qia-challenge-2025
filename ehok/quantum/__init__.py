"""
Quantum operations for E-HOK protocol.

This subpackage handles quantum resource management, including EPR pair
generation, batching, basis selection, and measurements.
"""

from ehok.quantum.noise_adapter import (
    SimulatorNoiseParams,
    physical_to_simulator,
    estimate_qber_from_physical,
    estimate_sifted_rate,
    validate_physical_params_for_simulation,
)

__all__ = [
    "SimulatorNoiseParams",
    "physical_to_simulator",
    "estimate_qber_from_physical",
    "estimate_sifted_rate",
    "validate_physical_params_for_simulation",
]

"""Analysis and verification utilities for E-HOK."""

from .nsm_bounds import (
    NSMBoundsCalculator,
    NSMBoundsInputs,
    NSMBoundsResult,
    FeasibilityResult,
    binary_entropy,
    gamma_function,
    collision_entropy_rate,
    max_bound_entropy_rate,
    channel_capacity,
)

__all__ = [
    "NSMBoundsCalculator",
    "NSMBoundsInputs",
    "NSMBoundsResult",
    "FeasibilityResult",
    "binary_entropy",
    "gamma_function",
    "collision_entropy_rate",
    "max_bound_entropy_rate",
    "channel_capacity",
]

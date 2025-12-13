"""
Configuration schemas for E-HOK protocol.

This subpackage provides typed configuration dataclasses for physical,
security, and protocol parameters based on literature values.

Primary Reference
-----------------
- Erven et al. (2014): Table I experimental parameters
"""

from ehok.configs.protocol_config import (
    PhysicalParameters,
    NSMSecurityParameters,
    ProtocolParameters,
    ProtocolConfig,
)

__all__ = [
    "PhysicalParameters",
    "NSMSecurityParameters",
    "ProtocolParameters",
    "ProtocolConfig",
]

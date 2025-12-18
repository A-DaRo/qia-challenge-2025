"""SquidASM program orchestration for Caligo.

This package provides the Phase E orchestration layer: SquidASM programs for
Alice and Bob plus a simple runner.
"""

from caligo.protocol.base import ProtocolParameters
from caligo.protocol.alice import AliceProgram
from caligo.protocol.bob import BobProgram
from caligo.protocol.orchestrator import run_protocol

__all__ = [
    "ProtocolParameters",
    "AliceProgram",
    "BobProgram",
    "run_protocol",
]

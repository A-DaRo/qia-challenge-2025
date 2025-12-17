"""
Utility functions for E-HOK protocol.

This subpackage provides helper utilities including logging infrastructure
and classical communication wrappers.
"""

from ehok.utils.logging import setup_ehok_logging, get_logger

__all__ = [
    "setup_ehok_logging",
    "get_logger",
]

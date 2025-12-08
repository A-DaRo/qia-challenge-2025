"""
Privacy amplification implementations.

This subpackage contains concrete implementations of the IPrivacyAmplifier
interface, including Toeplitz hashing for 2-universal hashing.
"""

from .toeplitz_amplifier import ToeplitzAmplifier

__all__ = ["ToeplitzAmplifier"]

"""
Factory helpers to wire protocol strategies from configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import scipy.sparse as sp

from ehok.core.config import ProtocolConfig
from ehok.interfaces import (
    ICommitmentScheme,
    IReconciliator,
    IPrivacyAmplifier,
    ISamplingStrategy,
    INoiseEstimator,
)
from ehok.implementations.commitment.sha256_commitment import SHA256Commitment
from ehok.implementations.reconciliation import LDPCReconciliator, LDPCMatrixManager
from ehok.implementations.privacy_amplification.toeplitz_amplifier import (
    ToeplitzAmplifier,
)
from ehok.implementations.sampling.random_sampling import RandomSamplingStrategy
from ehok.implementations.noise.simple_noise_estimator import SimpleNoiseEstimator


def build_commitment_scheme(config: ProtocolConfig) -> ICommitmentScheme:
    """Return the commitment scheme for the given configuration."""
    return SHA256Commitment()


def build_reconciliator(
    config: ProtocolConfig,
    parity_check_matrix: Optional[sp.spmatrix] = None,
) -> IReconciliator:
    """Return reconciliator, loading matrix pool as needed."""
    if parity_check_matrix is not None:
        raise NotImplementedError(
            "Direct matrix injection is not supported; use matrix_path directory with pool files."
        )

    matrix_dir = config.reconciliation.matrix_path
    if matrix_dir is None:
        matrix_dir = Path(__file__).resolve().parents[1] / "configs" / "ldpc_matrices"
    manager = LDPCMatrixManager.from_directory(Path(matrix_dir))
    return LDPCReconciliator(manager)


def build_privacy_amplifier(config: ProtocolConfig) -> IPrivacyAmplifier:
    """Return privacy amplifier instance for configuration."""
    return ToeplitzAmplifier()


def build_sampling_strategy(config: ProtocolConfig) -> ISamplingStrategy:
    """Return sampling strategy used during sifting."""
    return RandomSamplingStrategy()


def build_noise_estimator(config: ProtocolConfig) -> INoiseEstimator:
    """Return noise estimator instance."""
    return SimpleNoiseEstimator()

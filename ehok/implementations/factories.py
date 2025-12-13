"""
Factory helpers to wire protocol strategies from configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import scipy.sparse as sp

from ehok.core import constants
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
from ehok.implementations.reconciliation.ldpc_bp_decoder import LDPCBeliefPropagation
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
    frame_size = constants.LDPC_FRAME_SIZE

    if config.reconciliation.testing_mode:
        # Override directory and frame size for testing
        if matrix_dir is None:
            matrix_dir = Path(__file__).resolve().parents[1] / "configs" / constants.LDPC_TEST_MATRIX_SUBDIR
        
        if config.reconciliation.ldpc_test_frame_size is not None:
            frame_size = config.reconciliation.ldpc_test_frame_size
    elif matrix_dir is None:
        matrix_dir = Path(__file__).resolve().parents[1] / "configs" / "ldpc_matrices"

    # Do not autogenerate matrices at runtime: enforce explicit pre-generated files
    manager = LDPCMatrixManager.from_directory(
        Path(matrix_dir), 
        frame_size=frame_size, 
        autogenerate_if_missing=False
    )
    # Configure BP decoder from protocol config thresholds
    decoder = LDPCBeliefPropagation(max_iterations=config.reconciliation.max_iterations, threshold=config.reconciliation.bp_threshold)
    return LDPCReconciliator(manager, bp_decoder=decoder)


def build_privacy_amplifier(config: ProtocolConfig) -> IPrivacyAmplifier:
    """
    Return privacy amplifier instance for configuration.
    
    Uses NSM-compliant finite-key parameters.
    """
    return ToeplitzAmplifier(
        epsilon_sec=config.privacy_amplification.target_epsilon_sec,
        epsilon_cor=config.privacy_amplification.target_epsilon_cor,
        use_fft=config.privacy_amplification.use_fft_compression,
        fft_threshold=config.privacy_amplification.fft_threshold,
    )


def build_sampling_strategy(config: ProtocolConfig) -> ISamplingStrategy:
    """Return sampling strategy used during sifting."""
    return RandomSamplingStrategy()


def build_noise_estimator(config: ProtocolConfig) -> INoiseEstimator:
    """Return noise estimator instance."""
    return SimpleNoiseEstimator()

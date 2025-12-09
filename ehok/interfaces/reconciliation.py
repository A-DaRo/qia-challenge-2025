"""
Abstract interface for LDPC-based information reconciliation.

This module defines the block-oriented reconciliation interface that supports
rate adaptation, shortening, integrated QBER estimation, and hash verification
as specified in the LDPC reconciliation plan.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class IReconciliator(ABC):
    """
    Abstract interface for LDPC-based information reconciliation.

    Implementations operate on fixed-size LDPC frames with optional shortening
    and provide integrated QBER estimation inputs.
    """

    @abstractmethod
    def select_rate(self, qber_est: float) -> float:
        """
        Select an LDPC code rate based on the current QBER estimate.

        Parameters
        ----------
        qber_est : float
            Current estimated QBER.

        Returns
        -------
        float
            Selected code rate from the available pool.
        """
        pass

    @abstractmethod
    def compute_shortening(
        self, rate: float, qber_est: float, target_payload: int
    ) -> int:
        """
        Compute the number of shortened bits needed for a target payload size.

        Parameters
        ----------
        rate : float
            Selected LDPC code rate.
        qber_est : float
            Current estimated QBER.
        target_payload : int
            Desired payload length in bits.

        Returns
        -------
        int
            Number of shortened bits to append as padding.
        """
        pass

    @abstractmethod
    def reconcile_block(
        self,
        key_block: np.ndarray,
        syndrome: np.ndarray,
        rate: float,
        n_shortened: int,
        prng_seed: int,
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Decode a single LDPC block and report convergence.

        Parameters
        ----------
        key_block : np.ndarray
            Alice's noisy payload bits for the block.
        syndrome : np.ndarray
            Difference syndrome received from Bob.
        rate : float
            LDPC rate used for the block.
        n_shortened : int
            Number of shortened bits appended via the shared PRNG.
        prng_seed : int
            Seed for generating the deterministic padding bits.

        Returns
        -------
        Tuple[np.ndarray, bool, int]
            Corrected payload bits, convergence flag, and corrected error count.
        """
        pass

    @abstractmethod
    def compute_syndrome_block(
        self, key_block: np.ndarray, rate: float, n_shortened: int, prng_seed: int
    ) -> np.ndarray:
        """
        Compute the syndrome for Bob's reference block (payload only input).

        Parameters
        ----------
        key_block : np.ndarray
            Bob's payload bits for the block.
        rate : float
            Selected LDPC rate.
        n_shortened : int
            Number of shortened bits appended.
        prng_seed : int
            Seed for generating padding bits.

        Returns
        -------
        np.ndarray
            Syndrome vector for the block.
        """
        pass

    @abstractmethod
    def verify_block(
        self, block_alice: np.ndarray, block_bob: np.ndarray
    ) -> Tuple[bool, bytes]:
        """
        Verify that Alice's corrected block matches Bob's reference.

        Parameters
        ----------
        block_alice : np.ndarray
            Alice's corrected payload bits.
        block_bob : np.ndarray
            Bob's reference payload bits.

        Returns
        -------
        Tuple[bool, bytes]
            Verification flag and the transmitted hash value.
        """
        pass

    @abstractmethod
    def estimate_leakage_block(self, syndrome_length: int, hash_bits: int = 50) -> int:
        """
        Estimate information leakage for a single reconciled block.

        Parameters
        ----------
        syndrome_length : int
            Length of transmitted syndrome in bits.
        hash_bits : int, optional
            Number of hash bits used for verification, by default 50.

        Returns
        -------
        int
            Total leakage in bits for the block.
        """
        pass

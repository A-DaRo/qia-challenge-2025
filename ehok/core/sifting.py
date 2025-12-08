"""
Sifting and sampling logic for E-HOK.

This module implements the classical post-processing steps for Phase 3 of the
E-HOK protocol: basis sifting, test set sampling, and QBER estimation.
"""

import numpy as np
from typing import Tuple
from .constants import TEST_SET_FRACTION, MIN_TEST_SET_SIZE, QBER_THRESHOLD
from .exceptions import QBERTooHighError
from ..utils.logging import get_logger

logger = get_logger("sifting")


class SiftingManager:
    """
    Manage basis sifting and error estimation.

    This class provides static methods to handle the comparison of measurement
    bases, selection of a random subset for error estimation, and calculation
    of the Quantum Bit Error Rate (QBER).
    """

    @staticmethod
    def identify_matching_bases(
        bases_alice: np.ndarray,
        bases_bob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify matching and mismatched basis indices.

        Compares the basis choices of Alice and Bob to determine which bits
        should be kept (sifted key) and which should be discarded.

        Parameters
        ----------
        bases_alice : np.ndarray
            Alice's basis choices (e.g., 0 for Z, 1 for X).
        bases_bob : np.ndarray
            Bob's basis choices.

        Returns
        -------
        I_0 : np.ndarray
            Indices where bases match (sifted key candidates).
        I_1 : np.ndarray
            Indices where bases mismatch (to be discarded or used for oblivious key mask).
        """
        matches = (bases_alice == bases_bob)
        I_0 = np.where(matches)[0]
        I_1 = np.where(~matches)[0]

        logger.info(
            f"Sifting: |I_0|={len(I_0)}, |I_1|={len(I_1)} "
            f"({len(I_0)/(len(I_0)+len(I_1))*100:.1f}% matched)"
        )
        return I_0, I_1

    @staticmethod
    def select_test_set(
        I_0: np.ndarray,
        fraction: float = TEST_SET_FRACTION,
        seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select random test set from I_0.

        Randomly chooses a subset of the sifted key indices to be revealed
        for error estimation. The remaining indices form the raw key.

        IMPORTANT: Both parties must use the same seed to select the same test set.
        If seed is None, a deterministic seed is derived from I_0 itself.

        Parameters
        ----------
        I_0 : np.ndarray
            Indices where bases match.
        fraction : float, optional
            Fraction of I_0 to use for testing, by default TEST_SET_FRACTION.
        seed : int, optional
            Random seed for reproducibility. If None, derives seed from I_0.

        Returns
        -------
        test_set : np.ndarray
            Indices selected for testing (T).
        key_set : np.ndarray
            Remaining indices for key (I_0 \\ T).
        """
        # If no seed provided, derive deterministic seed from I_0
        # This ensures both parties select the same test set
        if seed is None:
            seed = int(np.sum(I_0) % (2**31))
        
        rng = np.random.default_rng(seed)
        
        # Calculate test size: max of (fraction * |I_0|, MIN_TEST_SET_SIZE, 1)
        # but capped at |I_0| to avoid selecting more than available
        test_size_fraction = int(len(I_0) * fraction)
        test_size = max(1, min(len(I_0), max(test_size_fraction, MIN_TEST_SET_SIZE)))

        test_set = rng.choice(I_0, size=test_size, replace=False)
        test_set.sort()

        key_set = np.setdiff1d(I_0, test_set)

        logger.info(
            f"Test set: {len(test_set)} bits "
            f"({fraction*100:.1f}% of sifted)"
        )
        return test_set, key_set

    @staticmethod
    def estimate_qber(
        outcomes_alice: np.ndarray,
        outcomes_bob: np.ndarray,
        test_indices: np.ndarray
    ) -> float:
        """
        Estimate QBER on test set.

        Calculates the error rate by comparing Alice's and Bob's outcomes
        on the revealed test set indices.

        Parameters
        ----------
        outcomes_alice : np.ndarray
            Alice's measurement outcomes.
        outcomes_bob : np.ndarray
            Bob's measurement outcomes.
        test_indices : np.ndarray
            Indices to test (T).

        Returns
        -------
        qber : float
            Quantum bit error rate.
        """
        alice_test = outcomes_alice[test_indices]
        bob_test = outcomes_bob[test_indices]

        errors = np.sum(alice_test != bob_test)
        qber = errors / len(test_indices)

        logger.info(
            f"QBER estimation: {errors}/{len(test_indices)} "
            f"errors = {qber*100:.2f}%"
        )
        return qber

    @staticmethod
    def check_qber_abort(qber: float, threshold: float = QBER_THRESHOLD) -> None:
        """
        Raise exception if QBER exceeds threshold.

        Parameters
        ----------
        qber : float
            Calculated Quantum Bit Error Rate.
        threshold : float, optional
            Maximum allowed QBER, by default QBER_THRESHOLD.

        Raises
        ------
        QBERTooHighError
            If qber > threshold.
        """
        if qber > threshold:
            raise QBERTooHighError(qber, threshold)
        logger.info(f"QBER {qber*100:.2f}% < threshold {threshold*100:.0f}%: OK")

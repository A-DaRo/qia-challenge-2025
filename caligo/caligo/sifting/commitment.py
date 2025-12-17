"""
SHA-256 based cryptographic commitment scheme.

Implements a commit-reveal protocol for basis choices, ensuring
binding and hiding properties required for secure sifting.

References
----------
- NIST FIPS 180-4: SHA-256 specification
- Schaffner et al. (2009): Commitment in OT protocol
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import (
    CommitmentVerificationError,
    InvalidParameterError,
)

logger = get_logger(__name__)


# SHA-256 produces 32-byte (256-bit) hashes
HASH_LENGTH_BYTES = 32

# Default nonce length for commitment randomness
DEFAULT_NONCE_LENGTH = 32


@dataclass
class CommitmentResult:
    """
    Result of a commitment operation.

    Parameters
    ----------
    commitment : bytes
        The commitment hash (32 bytes for SHA-256).
    nonce : bytes
        The random nonce used (must be kept secret until reveal).
    data_hash : bytes
        Hash of the committed data (for verification).
    """

    commitment: bytes
    nonce: bytes
    data_hash: bytes


class SHA256Commitment:
    """
    Cryptographic commitment using SHA-256.

    Provides binding and hiding properties:
    - Binding: Cannot find different message with same commitment
    - Hiding: Commitment reveals nothing about the message

    Construction: commit(m) = SHA256(nonce || m)

    Parameters
    ----------
    nonce_length : int
        Length of random nonce in bytes (default 32).

    Notes
    -----
    Security Properties:
    - Collision resistance: Finding m' ≠ m with same commitment
      requires ~2^128 operations (birthday bound)
    - Pre-image resistance: Finding m from commit(m) requires
      ~2^256 operations

    References
    ----------
    - NIST FIPS 180-4: SHA-256
    - Schaffner et al. (2009): "Alice commits to her basis choices"
    """

    def __init__(self, nonce_length: int = DEFAULT_NONCE_LENGTH) -> None:
        """
        Initialize commitment scheme.

        Parameters
        ----------
        nonce_length : int
            Nonce length in bytes.
        """
        if nonce_length < 16:
            raise InvalidParameterError(
                f"nonce_length={nonce_length} too short, minimum 16 bytes"
            )
        self._nonce_length = nonce_length

    def commit(self, data: bytes) -> CommitmentResult:
        """
        Create a commitment to data.

        Parameters
        ----------
        data : bytes
            Data to commit to.

        Returns
        -------
        CommitmentResult
            Commitment with nonce (keep nonce secret!).

        Notes
        -----
        The nonce must be kept secret until the reveal phase.
        Revealing the nonce before measurement completion
        violates the protocol security.
        """
        # Generate cryptographically secure random nonce
        nonce = secrets.token_bytes(self._nonce_length)

        # Compute commitment: H(nonce || data)
        hasher = hashlib.sha256()
        hasher.update(nonce)
        hasher.update(data)
        commitment = hasher.digest()

        # Also compute hash of data alone (for verification)
        data_hash = hashlib.sha256(data).digest()

        logger.debug(
            f"Created commitment: {len(data)} bytes data, "
            f"commitment={commitment[:8].hex()}..."
        )

        return CommitmentResult(
            commitment=commitment,
            nonce=nonce,
            data_hash=data_hash,
        )

    def commit_bases(self, bases: np.ndarray) -> CommitmentResult:
        """
        Create a commitment to basis choices.

        Parameters
        ----------
        bases : np.ndarray
            Array of basis choices (0/1).

        Returns
        -------
        CommitmentResult
            Commitment to the basis array.
        """
        # Pack bases into bytes efficiently
        bases_bytes = np.packbits(bases.astype(np.uint8)).tobytes()
        return self.commit(bases_bytes)

    def verify(
        self, commitment: bytes, nonce: bytes, data: bytes
    ) -> bool:
        """
        Verify a commitment against revealed data.

        Parameters
        ----------
        commitment : bytes
            Original commitment hash.
        nonce : bytes
            Revealed nonce.
        data : bytes
            Revealed data.

        Returns
        -------
        bool
            True if commitment is valid.

        Raises
        ------
        CommitmentVerificationError
            If verification fails (potential attack).
        """
        # Recompute commitment
        hasher = hashlib.sha256()
        hasher.update(nonce)
        hasher.update(data)
        computed = hasher.digest()

        is_valid = secrets.compare_digest(commitment, computed)

        if not is_valid:
            logger.warning(
                "Commitment verification FAILED - potential attack!"
            )
            raise CommitmentVerificationError(
                "Commitment verification failed: "
                f"expected {commitment[:8].hex()}..., "
                f"got {computed[:8].hex()}..."
            )

        logger.debug("Commitment verified successfully")
        return True

    def verify_bases(
        self,
        commitment: bytes,
        nonce: bytes,
        bases: np.ndarray,
    ) -> bool:
        """
        Verify a commitment against revealed basis choices.

        Parameters
        ----------
        commitment : bytes
            Original commitment hash.
        nonce : bytes
            Revealed nonce.
        bases : np.ndarray
            Revealed basis choices.

        Returns
        -------
        bool
            True if commitment is valid.
        """
        bases_bytes = np.packbits(bases.astype(np.uint8)).tobytes()
        return self.verify(commitment, nonce, bases_bytes)

    @staticmethod
    def hash_data(data: bytes) -> bytes:
        """
        Compute SHA-256 hash of data.

        Parameters
        ----------
        data : bytes
            Data to hash.

        Returns
        -------
        bytes
            32-byte SHA-256 hash.
        """
        return hashlib.sha256(data).digest()

    @staticmethod
    def generate_nonce(length: int = DEFAULT_NONCE_LENGTH) -> bytes:
        """
        Generate a cryptographically secure random nonce.

        Parameters
        ----------
        length : int
            Nonce length in bytes.

        Returns
        -------
        bytes
            Random nonce.
        """
        return secrets.token_bytes(length)

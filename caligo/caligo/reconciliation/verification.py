"""
Secure Block Verification for Reconciliation.

Provides IT-secure verification using HMAC-SHA256 with session-specific
salts, replacing the insecure polynomial hash and local SHA-256 implementations.

Per Martinez-Mateo §4.1 and Audit Report P1-2:
- Hash must cover ONLY payload (not padding)
- Session salt provides IT-security guarantees
- Constant-time comparison prevents timing attacks

References
----------
- Carter & Wegman (1979): Universal hash families
- NIST SP 800-107: HMAC specifications
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Optional

import numpy as np

from caligo.reconciliation import constants


@dataclass(frozen=True)
class VerificationSalt:
    """
    Session-specific verification salt for IT-secure hashing.
    
    Generated once per session and exchanged during Phase II sifting.
    The salt ensures that hash collisions cannot be pre-computed by
    an adversary, providing information-theoretic security guarantees.
    
    Attributes
    ----------
    value : bytes
        Random 32-byte salt value.
    """
    value: bytes
    
    @classmethod
    def generate(cls) -> "VerificationSalt":
        """
        Generate a cryptographically secure random salt.
        
        Returns
        -------
        VerificationSalt
            Fresh salt for this session.
        """
        return cls(value=secrets.token_bytes(32))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "VerificationSalt":
        """
        Reconstruct salt from received bytes.
        
        Parameters
        ----------
        data : bytes
            Salt bytes received from peer.
            
        Returns
        -------
        VerificationSalt
            Reconstructed salt.
            
        Raises
        ------
        ValueError
            If data length is not 32 bytes.
        """
        if len(data) != 32:
            raise ValueError(f"Salt must be 32 bytes, got {len(data)}")
        return cls(value=data)


class SecureBlockVerifier:
    """
    HMAC-SHA256 based block verification with session salt.
    
    Replaces both PolynomialHashVerifier and local SHA-256 compute_hash
    with a unified, IT-secure implementation.
    
    Per Audit Report P1-2:
    - Uses HMAC-SHA256 for collision resistance
    - Session salt prevents offline dictionary attacks
    - Constant-time comparison prevents timing side-channels
    - Hash covers ONLY payload (per Martinez-Mateo §4.1)
    
    Parameters
    ----------
    session_salt : VerificationSalt
        Session-specific salt (must be exchanged securely).
    output_bits : int, optional
        Output hash length in bits (default 64).
        
    Attributes
    ----------
    session_salt : VerificationSalt
        The session salt in use.
    output_bits : int
        Configured output length.
    """
    
    def __init__(
        self,
        session_salt: VerificationSalt,
        output_bits: int = constants.LDPC_HASH_BITS,
    ) -> None:
        if output_bits <= 0 or output_bits > 256:
            raise ValueError("output_bits must be in range (0, 256]")
        self._salt = session_salt
        self._output_bits = output_bits
        self._output_bytes = (output_bits + 7) // 8
    
    @property
    def session_salt(self) -> VerificationSalt:
        """Return the session salt."""
        return self._salt
    
    @property
    def output_bits(self) -> int:
        """Return configured output hash length in bits."""
        return self._output_bits
    
    def compute_tag(
        self,
        payload: np.ndarray,
        block_id: int,
    ) -> bytes:
        """
        Compute verification tag for payload.
        
        IMPORTANT: This hashes ONLY the payload, not frame padding.
        Per Martinez-Mateo §4.1, revealed padding bits are independent
        of the secret key and should not affect verification.
        
        Parameters
        ----------
        payload : np.ndarray
            Payload bits (uint8) - the secret key material only.
        block_id : int
            Block identifier for domain separation.
            
        Returns
        -------
        bytes
            Truncated HMAC-SHA256 tag (exactly output_bits of entropy).
        """
        # Construct HMAC key: salt || block_id
        key = self._salt.value + block_id.to_bytes(8, 'little')
        
        # Compute HMAC-SHA256
        tag = hmac.new(key, payload.tobytes(), hashlib.sha256).digest()
        
        # Truncate to output_bytes, then mask to exactly output_bits
        # This ensures bytes->int->bytes round-trips correctly
        truncated = tag[:self._output_bytes]
        tag_int = int.from_bytes(truncated, 'little')
        masked_int = tag_int & ((1 << self._output_bits) - 1)
        return masked_int.to_bytes(self._output_bytes, 'little')
    
    def compute_tag_int(
        self,
        payload: np.ndarray,
        block_id: int,
    ) -> int:
        """
        Compute verification tag as integer (for protocol compatibility).
        
        Parameters
        ----------
        payload : np.ndarray
            Payload bits (uint8).
        block_id : int
            Block identifier.
            
        Returns
        -------
        int
            Tag as integer (little-endian), within output_bits.
        """
        tag_bytes = self.compute_tag(payload, block_id)
        return int.from_bytes(tag_bytes, 'little')
    
    def verify(
        self,
        payload: np.ndarray,
        expected_tag: bytes,
        block_id: int,
    ) -> bool:
        """
        Verify payload matches expected tag using constant-time comparison.
        
        Parameters
        ----------
        payload : np.ndarray
            Payload bits to verify.
        expected_tag : bytes
            Expected tag from peer.
        block_id : int
            Block identifier.
            
        Returns
        -------
        bool
            True if tags match.
        """
        computed = self.compute_tag(payload, block_id)
        return hmac.compare_digest(computed, expected_tag[:self._output_bytes])
    
    def verify_int(
        self,
        payload: np.ndarray,
        expected_tag: int,
        block_id: int,
    ) -> bool:
        """
        Verify payload matches expected integer tag.
        
        Parameters
        ----------
        payload : np.ndarray
            Payload bits to verify.
        expected_tag : int
            Expected tag as integer (should be from compute_tag_int).
        block_id : int
            Block identifier.
            
        Returns
        -------
        bool
            True if tags match.
        """
        # Convert int to bytes - no masking needed since compute_tag_int
        # returns already-masked values
        expected_bytes = expected_tag.to_bytes(self._output_bytes, 'little')
        return self.verify(payload, expected_bytes, block_id)


def create_deterministic_verifier(seed: int = 0) -> SecureBlockVerifier:
    """
    Create a verifier with deterministic salt for testing.
    
    WARNING: Only use in tests! Production code must use random salts.
    
    Parameters
    ----------
    seed : int
        Seed for deterministic salt generation.
        
    Returns
    -------
    SecureBlockVerifier
        Verifier with deterministic salt.
    """
    # Generate deterministic salt from seed
    salt_bytes = hashlib.sha256(seed.to_bytes(8, 'little')).digest()
    salt = VerificationSalt(value=salt_bytes)
    return SecureBlockVerifier(session_salt=salt)

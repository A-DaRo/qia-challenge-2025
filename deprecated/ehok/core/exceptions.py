"""
Exception hierarchy for the E-HOK protocol.

This module defines all custom exceptions used throughout the E-HOK
implementation, providing structured error handling and clear error reporting.
"""


class EHOKException(Exception):
    """
    Base exception for E-HOK protocol.
    
    All protocol-specific exceptions inherit from this base class, enabling
    catch-all error handling for protocol-related errors while distinguishing
    them from other system exceptions.
    """
    pass


class SecurityException(EHOKException):
    """
    Raised when security conditions are violated.
    
    This exception indicates that a security-critical condition has been
    violated, such as QBER exceeding thresholds or commitment verification
    failures. These exceptions typically result in protocol abort.
    """
    pass


class ProtocolError(EHOKException):
    """
    Raised when protocol execution encounters an error.
    
    This exception covers non-security-related protocol failures, such as
    reconciliation failures or communication errors. These may be recoverable
    or indicate implementation bugs.
    """
    pass


class MatrixSynchronizationError(ProtocolError):
    """
    Raised when LDPC matrix pools differ between parties.

    This signals a fatal protocol mismatch detected during initialization when
    matrix checksums do not align. Execution must halt to avoid desynchronised
    reconciliation and potential security issues.

    Attributes
    ----------
    local_checksum : str
        SHA-256 checksum computed locally.
    remote_checksum : str
        SHA-256 checksum received from the peer.
    """

    def __init__(self, local_checksum: str, remote_checksum: str) -> None:
        self.local_checksum = local_checksum
        self.remote_checksum = remote_checksum
        super().__init__(
            "LDPC matrix pool mismatch: local checksum "
            f"{local_checksum} != remote checksum {remote_checksum}"
        )


class QBERTooHighError(SecurityException):
    """
    Raised when QBER exceeds abort threshold.
    
    High QBER indicates either excessive noise or potential eavesdropping.
    The protocol must abort to ensure security.
    
    Attributes
    ----------
    measured_qber : float
        The measured QBER value.
    threshold : float
        The maximum acceptable QBER threshold.
    """
    def __init__(self, measured_qber: float, threshold: float):
        """
        Initialize QBERTooHighError.
        
        Parameters
        ----------
        measured_qber : float
            The measured QBER value.
        threshold : float
            The maximum acceptable QBER threshold.
        """
        self.measured_qber = measured_qber
        self.threshold = threshold
        super().__init__(
            f"QBER {measured_qber:.4f} exceeds threshold {threshold:.4f}"
        )


class ReconciliationFailedError(ProtocolError):
    """
    Raised when error correction fails.
    
    This exception indicates that the reconciliation (error correction) process
    failed to converge or produced an invalid result. This may occur with
    extremely high error rates or decoder failures.
    """
    pass


class CommitmentVerificationError(SecurityException):
    """
    Raised when commitment verification fails.
    
    This exception indicates that a commitment could not be verified, suggesting
    either protocol violation by a party or transmission errors. This is a
    security-critical failure requiring protocol abort.
    """
    pass

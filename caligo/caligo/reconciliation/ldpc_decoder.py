"""
Belief-Propagation Decoder for LDPC Codes.

This module implements the sum-product belief-propagation algorithm in the
log-likelihood ratio (LLR) domain for efficient syndrome-based decoding.

The decoder is used by Bob to correct errors in his received key using
Alice's syndrome, without revealing error positions back to Alice.

References
----------
- MacKay (2003): Information Theory, Inference, and Learning Algorithms
- Kschischang et al. (2001): Factor graphs and the sum-product algorithm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import (
    CompiledParityCheckMatrix,
    compile_parity_check_matrix,
)
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class DecodeResult:
    """
    Result of a single BP decoding attempt.

    Attributes
    ----------
    corrected_bits : np.ndarray
        Corrected bit vector (full frame including padding).
    converged : bool
        True if syndrome matched target (successful decode).
    iterations : int
        Number of iterations executed.
    syndrome_errors : int
        Number of unsatisfied parity checks at termination.
    """

    corrected_bits: np.ndarray
    converged: bool
    iterations: int
    syndrome_errors: int

    @property
    def corrected(self) -> np.ndarray:
        """Alias for corrected_bits (backward compatibility)."""
        return self.corrected_bits

    @property
    def error_count(self) -> int:
        """Alias for syndrome_errors (backward compatibility)."""
        return self.syndrome_errors


# =============================================================================
# Belief Propagation Decoder
# =============================================================================


class BeliefPropagationDecoder:
    """
    Sum-product belief-propagation decoder in log-likelihood domain.

    Implements iterative message-passing on the Tanner graph to decode
    syndrome-based error correction for LDPC codes.

    Parameters
    ----------
    parity_check_matrix : sp.csr_matrix, optional
        Parity-check matrix (m × n) in CSR format. If provided, will be
        used as default for decode() calls.
    max_iterations : int, optional
        Maximum BP iterations, by default from constants.
    threshold : float, optional
        Message stability threshold for early stopping.

    Attributes
    ----------
    parity_check_matrix : sp.csr_matrix or None
        Default parity-check matrix for decoding.
    max_iterations : int
        Configured iteration limit.
    threshold : float
        Convergence threshold.

    Notes
    -----
    Uses tanh-domain update for check-to-variable messages:

    .. math::
        μ_{c→v} = 2·arctanh(∏_{v'∈N(c)\\v} tanh(μ_{v'→c}/2))

    Variable-to-check messages accumulate channel LLR plus incoming
    check messages (excluding the target check).
    """

    def __init__(
        self,
        parity_check_matrix: sp.csr_matrix = None,
        max_iterations: int = constants.LDPC_MAX_ITERATIONS,
        threshold: float = constants.LDPC_BP_THRESHOLD,
    ) -> None:
        self.parity_check_matrix = parity_check_matrix
        self.max_iterations = max_iterations
        self.threshold = threshold
        self._compiled_cache: Dict[int, CompiledParityCheckMatrix] = {}

    def _get_compiled(
        self,
        H: Union[sp.csr_matrix, CompiledParityCheckMatrix],
    ) -> CompiledParityCheckMatrix:
        """Get a compiled representation, caching by matrix object identity."""

        if isinstance(H, CompiledParityCheckMatrix):
            return H

        H_csr = H.tocsr()
        key = id(H_csr)
        cached = self._compiled_cache.get(key)
        if cached is not None and cached.m == H_csr.shape[0] and cached.n == H_csr.shape[1]:
            return cached

        compiled = compile_parity_check_matrix(H_csr, checksum="")
        self._compiled_cache[key] = compiled
        return compiled

    def decode(
        self,
        llr: np.ndarray,
        target_syndrome: np.ndarray,
        H: Union[sp.csr_matrix, CompiledParityCheckMatrix, None] = None,
        max_iterations: Optional[int] = None,
    ) -> DecodeResult:
        """
        Decode using belief propagation with target syndrome.

        Parameters
        ----------
        llr : np.ndarray
            Initial log-likelihood ratios (length n).
            Positive → high confidence bit=0, negative → bit=1.
        target_syndrome : np.ndarray
            Target syndrome vector in {0,1}^m.
        H : sp.csr_matrix, optional
            Parity-check matrix (m × n) in CSR format. Uses constructor
            matrix if not provided.

        Returns
        -------
        DecodeResult
            Decoding result with corrected bits and convergence status.

        Raises
        ------
        ValueError
            If dimensions mismatch or matrix has no edges.
        """
        # Use constructor matrix if H not provided
        if H is None:
            H = self.parity_check_matrix
        if H is None:
            raise ValueError("No parity-check matrix provided")

        compiled = self._get_compiled(H)
        m, n = compiled.m, compiled.n
        llr = llr.astype(np.float64)

        max_iters = int(self.max_iterations if max_iterations is None else max_iterations)
        if max_iters < 1:
            raise ValueError("max_iterations must be >= 1")

        if llr.shape[0] != n:
            raise ValueError(f"LLR length {llr.shape[0]} != H columns {n}")
        if target_syndrome.shape[0] != m:
            raise ValueError(f"Syndrome length {target_syndrome.shape[0]} != H rows {m}")

        edge_count = compiled.edge_count
        if edge_count <= 0:
            raise ValueError("Parity-check matrix has no edges")

        # Message arrays: r = check→var, q = var→check
        r = np.zeros(edge_count, dtype=np.float64)
        q = np.zeros(edge_count, dtype=np.float64)

        # Initialize var→check messages with channel LLR
        for v in range(n):
            start = int(compiled.var_ptr[v])
            end = int(compiled.var_ptr[v + 1])
            if start == end:
                continue
            edges = compiled.var_edges[start:end]
            q[edges] = llr[v]

        converged = False
        decoded = np.zeros(n, dtype=np.uint8)
        iteration = 0

        for iteration in range(1, max_iters + 1):
            # Check node update (horizontal step)
            target_u8 = target_syndrome.astype(np.uint8, copy=False)
            for c in range(m):
                start = int(compiled.check_ptr[c])
                end = int(compiled.check_ptr[c + 1])
                if start == end:
                    continue

                tanh_vals = np.tanh(q[start:end] / 2.0)
                zero_mask = tanh_vals == 0.0
                zero_count = int(np.count_nonzero(zero_mask))

                if zero_count == 0:
                    total_prod = float(np.prod(tanh_vals))
                    prod_excl = total_prod / tanh_vals
                elif zero_count == 1:
                    prod_nonzero = float(np.prod(tanh_vals[~zero_mask]))
                    prod_excl = np.zeros_like(tanh_vals)
                    prod_excl[zero_mask] = prod_nonzero
                else:
                    prod_excl = np.zeros_like(tanh_vals)

                # Flip sign if syndrome bit is 1
                sign = -1.0 if int(target_u8[c]) == 1 else 1.0
                prod_excl = np.clip(sign * prod_excl, -0.999999, 0.999999)
                r[start:end] = 2.0 * np.arctanh(prod_excl)

            # Variable node update (vertical step) and hard decision
            for v in range(n):
                start = int(compiled.var_ptr[v])
                end = int(compiled.var_ptr[v + 1])
                if start == end:
                    total = llr[v]
                else:
                    edges = compiled.var_edges[start:end]
                    total = llr[v] + float(np.sum(r[edges]))
                    q[edges] = total - r[edges]
                decoded[v] = 1 if total < 0 else 0

            # Check convergence: syndrome matches target
            syndrome_errors = compiled.count_syndrome_errors(decoded, target_u8)
            if syndrome_errors == 0:
                converged = True
                break

            # Early stopping if messages stabilize
            if np.max(np.abs(r)) < self.threshold:
                logger.debug("Early stopping: messages stabilized at iter %d", iteration)
                break
        else:
            # Compute final syndrome errors if didn't break
            target_u8 = target_syndrome.astype(np.uint8, copy=False)
            syndrome_errors = compiled.count_syndrome_errors(decoded, target_u8)

        return DecodeResult(
            corrected_bits=decoded,
            converged=converged,
            iterations=iteration,
            syndrome_errors=int(syndrome_errors if not converged else 0),
        )


# =============================================================================
# LLR Construction Utilities
# =============================================================================


def build_channel_llr(
    bob_bits: np.ndarray,
    qber: float,
    n_shortened: int = 0,
    known_bits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct initial LLRs from BSC channel model.

    Parameters
    ----------
    bob_bits : np.ndarray
        Bob's received bits (payload only), dtype int8 or uint8.
    qber : float
        Estimated quantum bit error rate.
    n_shortened : int, optional
        Number of shortened (padding) bits. Default 0.

    Returns
    -------
    np.ndarray
        LLR array for full frame [payload | shortened].

    Notes
    -----
    For BSC with crossover probability p (QBER):
        LLR = log((1-p)/p) × (-1)^bit

    Sign convention: bit=0 → positive LLR, bit=1 → negative LLR.
    Shortened bits have LLR = ±100.0 (infinite confidence).
    """
    if known_bits is not None:
        # known_bits takes precedence over n_shortened.
        n_shortened = int(len(known_bits))

    n = len(bob_bits) + int(n_shortened)
    qber_clamped = np.clip(qber, 1e-6, 0.5 - 1e-6)
    channel_llr = np.log((1 - qber_clamped) / qber_clamped)

    llr = np.zeros(n, dtype=np.float64)
    # Payload: sign based on received bit (0→positive, 1→negative)
    llr[:len(bob_bits)] = channel_llr * (1 - 2 * bob_bits.astype(np.float64))
    # Shortened/known bits: high confidence with correct sign.
    if n_shortened > 0:
        if known_bits is None:
            # Backward-compatible default: assume known bits are zeros.
            llr[len(bob_bits):] = constants.LDPC_LLR_SHORTENED
        else:
            known_u8 = known_bits.astype(np.uint8, copy=False)
            llr[len(bob_bits):] = constants.LDPC_LLR_SHORTENED * (1 - 2 * known_u8.astype(np.float64))

    return llr


def syndrome_guided_refinement(
    llr: np.ndarray,
    local_syndrome: np.ndarray,
    received_syndrome: np.ndarray,
    H: Union[sp.csr_matrix, CompiledParityCheckMatrix, None] = None,
    payload_len: int = None,
) -> np.ndarray:
    """
    Refine LLRs based on syndrome mismatch.

    Bits participating in mismatched parity checks receive reduced
    confidence, improving decoder performance near capacity.

    Parameters
    ----------
    llr : np.ndarray
        Initial LLR array (will be copied).
    local_syndrome : np.ndarray
        Locally computed syndrome.
    received_syndrome : np.ndarray
        Syndrome received from Alice.
    H : sp.csr_matrix, optional
        Parity-check matrix. If not provided, uses simple heuristic.
    payload_len : int, optional
        Payload length (to preserve padding LLRs).

    Returns
    -------
    np.ndarray
        Refined LLR array.

    Notes
    -----
    When syndromes match, returns original LLRs unchanged.
    When syndromes differ, reduces confidence in participating bits.
    """
    # If syndromes match, no refinement needed
    if np.array_equal(local_syndrome, received_syndrome):
        return llr.copy()

    llr = llr.copy()
    
    # If H is provided, use detailed refinement
    if H is not None:
        if isinstance(H, CompiledParityCheckMatrix):
            compiled = H
        else:
            compiled = compile_parity_check_matrix(H.tocsr(), checksum="")

        n_vars = len(llr)
        actual_payload_len = payload_len if payload_len is not None else n_vars

        # Compute error syndrome (where local != received)
        error_syndrome = (local_syndrome != received_syndrome).astype(np.int8)

        # Count mismatched checks per variable
        mismatch_count = np.zeros(n_vars, dtype=np.float64)
        for check_idx, error_bit in enumerate(error_syndrome):
            if error_bit == 1:
                start = int(compiled.check_ptr[check_idx])
                end = int(compiled.check_ptr[check_idx + 1])
                participating_vars = compiled.check_var[start:end]
                mismatch_count[participating_vars] += 1

        # Compute reliability factor
        max_degree = float(max(compiled.max_variable_degree, 1))
        if max_degree > 0:
            reliability_factor = 1.0 - (mismatch_count / max_degree)
            # Apply only to payload region (preserve padding)
            llr[:actual_payload_len] *= reliability_factor[:actual_payload_len]
    else:
        # Simple heuristic: reduce all magnitudes slightly
        reduction_factor = 0.9
        llr *= reduction_factor

    return llr

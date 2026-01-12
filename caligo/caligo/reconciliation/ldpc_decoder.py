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
        self._r_buf: Optional[np.ndarray] = None
        self._q_buf: Optional[np.ndarray] = None
        self._decoded_buf: Optional[np.ndarray] = None
        self._var_edge_var_idx_cache: Dict[int, np.ndarray] = {}
        self._r_by_var_edge_buf: Optional[np.ndarray] = None

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
        if llr.dtype != np.float64:
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
        if self._r_buf is None or int(self._r_buf.shape[0]) != int(edge_count):
            self._r_buf = np.zeros(edge_count, dtype=np.float64)
            self._q_buf = np.zeros(edge_count, dtype=np.float64)
        assert self._r_buf is not None
        assert self._q_buf is not None
        r = self._r_buf
        q = self._q_buf
        r.fill(0.0)
        q.fill(0.0)

        if self._decoded_buf is None or int(self._decoded_buf.shape[0]) != int(n):
            self._decoded_buf = np.zeros(n, dtype=np.uint8)
        assert self._decoded_buf is not None
        decoded = self._decoded_buf

        # Precompute var-index per entry in var_edges (cached per compiled matrix)
        compiled_id = id(compiled)
        var_idx_for_var_edge = self._var_edge_var_idx_cache.get(compiled_id)
        if var_idx_for_var_edge is None or int(var_idx_for_var_edge.shape[0]) != int(edge_count):
            degrees = np.diff(compiled.var_ptr).astype(np.int64, copy=False)
            var_idx_for_var_edge = np.repeat(np.arange(n, dtype=np.int64), degrees)
            self._var_edge_var_idx_cache[compiled_id] = var_idx_for_var_edge

        # Initialize var→check messages with channel LLR (vectorized)
        q[compiled.var_edges] = llr[var_idx_for_var_edge]

        converged = False
        iteration = 0

        target_u8 = target_syndrome.astype(np.uint8, copy=False)

        for iteration in range(1, max_iters + 1):
            # Check node update (horizontal step)
            for c in range(m):
                start = int(compiled.check_ptr[c])
                end = int(compiled.check_ptr[c + 1])
                if start == end:
                    continue

                tanh_vals = np.tanh(q[start:end] * 0.5)
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

                # CORRECTED: Apply syndrome sign AFTER arctanh, not before
                # Per MacKay (2003) and Kschischang (2001):
                # μ_{c→v} = (-1)^{s_c} × 2 × arctanh(∏ tanh(·))
                # The sign flips the MESSAGE direction, not the arctanh INPUT
                prod_excl_clipped = np.clip(prod_excl, -0.999999, 0.999999)
                base_msg = 2.0 * np.arctanh(prod_excl_clipped)
                
                # Apply syndrome sign to the result
                if int(target_u8[c]) == 1:
                    base_msg = -base_msg
                
                r[start:end] = base_msg

            # Variable node update (vertical step) and hard decision (vectorized)
            if self._r_by_var_edge_buf is None or int(self._r_by_var_edge_buf.shape[0]) != int(edge_count):
                self._r_by_var_edge_buf = np.empty(edge_count, dtype=np.float64)
            assert self._r_by_var_edge_buf is not None
            # Gather r in var-edge order into a reusable buffer
            np.take(r, compiled.var_edges, out=self._r_by_var_edge_buf)
            # Sum of incoming check messages per variable in var order
            sum_r_per_var = np.add.reduceat(
                self._r_by_var_edge_buf,
                compiled.var_ptr[:-1].astype(np.int64, copy=False),
            )
            total_llr = llr + sum_r_per_var
            decoded[:] = (total_llr < 0.0).astype(np.uint8)
            # Scatter updated q back to edge order
            q[compiled.var_edges] = total_llr[var_idx_for_var_edge] - self._r_by_var_edge_buf

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
            syndrome_errors = compiled.count_syndrome_errors(decoded, target_u8)

        return DecodeResult(
            corrected_bits=decoded.copy(),
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
    punctured_mask: np.ndarray,
) -> np.ndarray:
    """
    Construct initial LLRs from BSC channel model with untainted puncturing.

    Parameters
    ----------
    bob_bits : np.ndarray
        Bob's received bits (payload only), dtype int8 or uint8.
    qber : float
        Estimated quantum bit error rate.
    punctured_mask : np.ndarray
        Binary mask where punctured_mask[i]=1 indicates punctured position.
        Punctured bits get LLR=0 (neutral, infinite uncertainty).

    Returns
    -------
    np.ndarray
        LLR array for full frame.

    Notes
    -----
    For BSC with crossover probability p (QBER):
        LLR = log((1-p)/p) × (-1)^bit

    Sign convention: bit=0 → positive LLR, bit=1 → negative LLR.

    **Punctured bits receive neutral LLR=0** (infinite uncertainty).
    This aligns with Elkouss et al. definition where punctured variable nodes
    contribute no initial information to belief propagation.
    """
    n = int(punctured_mask.shape[0])
    qber_clamped = np.clip(qber, 1e-6, 0.5 - 1e-6)
    channel_llr = np.log((1 - qber_clamped) / qber_clamped)

    llr = np.empty(n, dtype=np.float64)

    # Extract non-punctured positions and fill with Bob's bits
    non_punctured_indices = np.where(punctured_mask == 0)[0]
    if len(non_punctured_indices) != len(bob_bits):
        raise ValueError(
            f"Number of non-punctured positions ({len(non_punctured_indices)}) "
            f"!= bob_bits length ({len(bob_bits)})"
        )

    bob_full = np.zeros(n, dtype=np.uint8)
    bob_full[non_punctured_indices] = bob_bits.astype(np.uint8, copy=False)

    # Compute LLRs for all positions
    llr[:] = bob_full.astype(np.float64)
    llr *= -2.0
    llr += 1.0
    llr *= float(channel_llr)

    # Set punctured positions to neutral (LLR=0, infinite uncertainty)
    punctured_indices = np.where(punctured_mask == 1)[0]
    llr[punctured_indices] = 0.0

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

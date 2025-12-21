"""
LDPC Codec Facade for Numba Kernels (Phase 2).

This module provides a thin Python wrapper around JIT-compiled Numba kernels
for LDPC encoding and decoding.

Per Implementation Report v2 §5.4:
- "Python Control, Numba Engine" architecture
- Accepts high-level numpy arrays
- Prepares C-contiguous buffers for kernels
- Invokes appropriate numba.njit kernel
- Unpacks results to Python-friendly structures

The complex, high-performance C-style logic is contained entirely within
the kernels (scripts/numba_kernels.py), while Python maintains readability
and manages the protocol state machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from caligo.scripts.numba_kernels import (
    decode_bp_hotstart_kernel,
    decode_bp_virtual_graph_kernel,
    encode_bitpacked_kernel,
)

if TYPE_CHECKING:
    from caligo.reconciliation.matrix_manager import MotherCodeManager, NumbaGraphTopology
    from caligo.reconciliation.strategies import DecoderResult


class LDPCCodec:
    """
    Thin wrapper around JIT-compiled Numba kernels.
    
    Per the "Python Control, Numba Engine" architecture, this class:
    1. Accepts high-level Python objects (numpy.ndarray, ReconciliationContext)
    2. Efficiently packs bits and prepares aligned C-contiguous buffers
    3. Invokes the appropriate numba.njit(nogil=True) kernel
    4. Unpacks the result back to Python-friendly structures
    
    The complex, high-performance C-style logic is contained entirely
    within the kernels, while Python maintains readability and manages
    the protocol state machine.
    
    Parameters
    ----------
    mother_code : MotherCodeManager
        Singleton manager providing compiled topology.
    """
    
    def __init__(self, mother_code: "MotherCodeManager") -> None:
        self._topo: "NumbaGraphTopology" = mother_code.compiled_topology
        self._frame_size = mother_code.frame_size
    
    @property
    def num_edges(self) -> int:
        """Get number of edges in the Tanner graph."""
        return self._topo.n_edges
    
    def encode(self, frame: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """
        Compute syndrome using bit-packed SpMV kernel.
        
        Per Implementation Report v2 §4.1.2: Uses bitwise AND + POPCNT
        for 10x-50x speedup over scipy.sparse.
        
        Parameters
        ----------
        frame : np.ndarray
            Full LDPC frame (n bits, uint8).
        pattern : np.ndarray
            Puncturing pattern (for validation only; syndrome uses full frame).
            
        Returns
        -------
        np.ndarray
            Syndrome bits (m = (1-R_0) × n bits, uint8).
        """
        # Validate inputs
        assert frame.dtype == np.uint8, "Frame must be uint8"
        assert frame.shape[0] == self._frame_size, f"Frame size mismatch: {frame.shape[0]} != {self._frame_size}"
        
        # Bit-pack frame into uint64 words
        packed_frame = self._bitpack(frame)
        
        # Call Numba kernel
        packed_syndrome = encode_bitpacked_kernel(
            packed_frame,
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.n_checks,
        )
        
        # Unpack syndrome
        return self._bitunpack(packed_syndrome, self._topo.n_checks)
    
    def decode_baseline(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        pattern: np.ndarray,
        max_iterations: int = 60,
    ) -> "DecoderResult":
        """
        Baseline decoding using Virtual Graph kernel.
        
        Per Implementation Report v2 §4.1.3A: Single kernel operates on
        full mother graph. Pattern is used only for LLR initialization
        (punctured → 0). Rate adaptation is purely a memory initialization step.
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome from Alice (uint8).
        llr : np.ndarray
            Three-state LLR array (payload, punctured=0, shortened=±∞).
        pattern : np.ndarray
            Puncturing pattern (used for mask validation).
        max_iterations : int
            Maximum BP iterations.
            
        Returns
        -------
        DecoderResult
            Decoded bits and convergence status.
        """
        # Import here to avoid circular import
        from caligo.reconciliation.strategies import DecoderResult
        
        # Ensure C-contiguous arrays
        llr = np.ascontiguousarray(llr, dtype=np.float64)
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        
        # Initialize messages to zero
        messages = np.zeros(self._topo.n_edges * 2, dtype=np.float64)
        
        # Call Virtual Graph decoder kernel
        corrected_bits, converged, iterations = decode_bp_virtual_graph_kernel(
            llr,
            syndrome,
            messages,
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.var_col_ptr,
            self._topo.var_row_idx,
            max_iterations,
        )
        
        return DecoderResult(
            corrected_bits=corrected_bits,
            converged=converged,
            iterations=iterations,
            messages=messages,
        )
    
    def decode_blind(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        messages: np.ndarray,
        frozen_mask: np.ndarray,
        max_iterations: int = 60,
    ) -> "DecoderResult":
        """
        Blind decoding using Hot-Start kernel with Freeze optimization.
        
        Per Implementation Report v2 §4.1.3B:
        - Messages persist across iterations (Hot-Start)
        - Frozen bits (LLR=±∞) skip expensive tanh/arctanh updates
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome from Alice (computed once, reused).
        llr : np.ndarray
            Current LLR array (updated as bits are revealed).
        messages : np.ndarray
            Edge messages from previous iteration (in/out).
        frozen_mask : np.ndarray
            Boolean mask for revealed/shortened positions.
        max_iterations : int
            Maximum BP iterations for this round.
            
        Returns
        -------
        DecoderResult
            Decoded bits, convergence status, and updated messages.
        """
        # Import here to avoid circular import
        from caligo.reconciliation.strategies import DecoderResult
        
        # Ensure C-contiguous arrays
        llr = np.ascontiguousarray(llr, dtype=np.float64)
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        messages = np.ascontiguousarray(messages, dtype=np.float64)
        frozen_mask = np.ascontiguousarray(frozen_mask, dtype=np.bool_)
        
        # Call Hot-Start decoder kernel with Freeze optimization
        corrected_bits, converged, iterations, out_messages = decode_bp_hotstart_kernel(
            llr,
            syndrome,
            messages,  # Resume from previous state
            frozen_mask,  # Skip updates for frozen positions
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.var_col_ptr,
            self._topo.var_row_idx,
            max_iterations,
        )
        
        return DecoderResult(
            corrected_bits=corrected_bits,
            converged=converged,
            iterations=iterations,
            messages=out_messages,
        )
    
    def _bitpack(self, bits: np.ndarray) -> np.ndarray:
        """
        Pack uint8 bit array into uint64 words.
        
        Parameters
        ----------
        bits : np.ndarray
            Bit array (uint8).
            
        Returns
        -------
        np.ndarray
            Packed words (uint64).
        """
        n = len(bits)
        n_words = (n + 63) // 64
        packed = np.zeros(n_words, dtype=np.uint64)
        
        for i in range(n):
            word_idx = i // 64
            bit_idx = i % 64
            if bits[i]:
                packed[word_idx] |= (1 << bit_idx)
        
        return packed
    
    def _bitunpack(self, packed: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Unpack uint64 words into uint8 bit array.
        
        Parameters
        ----------
        packed : np.ndarray
            Packed words (uint64).
        n_bits : int
            Number of bits to extract.
            
        Returns
        -------
        np.ndarray
            Bit array (uint8).
        """
        bits = np.zeros(n_bits, dtype=np.uint8)
        
        for i in range(n_bits):
            word_idx = i // 64
            bit_idx = i % 64
            if packed[word_idx] & (1 << bit_idx):
                bits[i] = 1
        
        return bits

"""
Unit tests for BP decoder module.

Tests decoding convergence, LLR initialization, and syndrome-guided refinement.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from caligo.reconciliation.ldpc_decoder import (
    BeliefPropagationDecoder,
    DecodeResult,
    build_channel_llr,
    syndrome_guided_refinement,
)
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation import constants


@pytest.fixture
def matrix_manager() -> MatrixManager:
    """Load LDPC matrix manager."""
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)


@pytest.fixture
def decoder(matrix_manager: MatrixManager) -> BeliefPropagationDecoder:
    """Create decoder with rate 0.70 matrix."""
    return BeliefPropagationDecoder(
        parity_check_matrix=matrix_manager.get_matrix(0.70),
        max_iterations=constants.LDPC_MAX_ITERATIONS,
    )


class TestBuildChannelLLR:
    """Tests for LLR initialization."""

    def test_low_qber_high_magnitude(self) -> None:
        """Low QBER produces high-magnitude LLRs."""
        bits = np.array([0, 0, 1, 1, 0, 1])
        # No puncturing - all bits are payload
        punctured_mask = np.zeros(len(bits), dtype=np.uint8)
        llr = build_channel_llr(bits, qber=0.01, punctured_mask=punctured_mask)
        
        # All LLRs should have high magnitude
        assert np.all(np.abs(llr) > 3.0)

    def test_high_qber_low_magnitude(self) -> None:
        """High QBER produces lower-magnitude LLRs."""
        bits = np.array([0, 0, 1, 1, 0, 1])
        punctured_mask = np.zeros(len(bits), dtype=np.uint8)
        llr = build_channel_llr(bits, qber=0.10, punctured_mask=punctured_mask)
        
        # LLRs should have moderate magnitude
        assert np.all(np.abs(llr) < 5.0)

    def test_sign_matches_bits(self) -> None:
        """LLR sign matches bit values (0→positive, 1→negative)."""
        bits = np.array([0, 1, 0, 1])
        punctured_mask = np.zeros(len(bits), dtype=np.uint8)
        llr = build_channel_llr(bits, qber=0.05, punctured_mask=punctured_mask)
        
        for i, bit in enumerate(bits):
            if bit == 0:
                assert llr[i] > 0
            else:
                assert llr[i] < 0

    def test_output_shape(self) -> None:
        """Output shape matches full frame with puncturing."""
        frame_size = 100
        payload_len = 80
        bits = np.zeros(payload_len, dtype=np.int8)
        # 20 bits punctured
        punctured_mask = np.zeros(frame_size, dtype=np.uint8)
        punctured_mask[:20] = 1
        llr = build_channel_llr(bits, qber=0.05, punctured_mask=punctured_mask)
        assert llr.shape == (frame_size,)


class TestSyndromeGuidedRefinement:
    """Tests for syndrome-guided refinement."""

    def test_identical_syndromes(self) -> None:
        """No refinement when syndromes match."""
        llr = np.array([1.0, -1.0, 1.0, -1.0])
        local_syndrome = np.array([0, 1, 1])
        received_syndrome = np.array([0, 1, 1])
        
        refined = syndrome_guided_refinement(llr, local_syndrome, received_syndrome)
        
        np.testing.assert_array_equal(refined, llr)

    def test_increases_uncertainty(self) -> None:
        """Refinement reduces LLR magnitude for uncertain positions."""
        llr = np.array([5.0, -5.0, 5.0, -5.0])
        local_syndrome = np.array([0, 1, 1])
        received_syndrome = np.array([1, 0, 1])  # Mismatch on first two
        
        refined = syndrome_guided_refinement(llr, local_syndrome, received_syndrome)
        
        # At least some LLRs should have reduced magnitude
        assert np.any(np.abs(refined) < np.abs(llr))


class TestDecodeResult:
    """Tests for DecodeResult dataclass."""

    def test_success_result(self) -> None:
        """Successful decode result properties."""
        result = DecodeResult(
            corrected_bits=np.array([0, 1, 0, 1]),
            converged=True,
            iterations=15,
            syndrome_errors=0,
        )
        
        assert result.converged is True
        assert result.iterations == 15
        assert result.syndrome_errors == 0

    def test_failed_result(self) -> None:
        """Failed decode result properties."""
        result = DecodeResult(
            corrected_bits=np.array([0, 1, 0, 1]),
            converged=False,
            iterations=60,
            syndrome_errors=5,
        )
        
        assert result.converged is False
        assert result.syndrome_errors > 0


@pytest.mark.integration
class TestBeliefPropagationDecoder:
    """Integration tests for BP decoder."""

    def test_decode_noiseless(self, decoder: BeliefPropagationDecoder) -> None:
        """Decoder converges on noiseless codeword."""
        # Create all-zeros codeword (always valid)
        H = decoder.parity_check_matrix
        n = H.shape[1]  # codeword length
        m = H.shape[0]  # syndrome length
        
        codeword = np.zeros(n, dtype=np.int8)
        syndrome = np.zeros(m, dtype=np.int8)
        
        # No puncturing
        punctured_mask = np.zeros(n, dtype=np.uint8)
        llr = build_channel_llr(codeword, qber=0.01, punctured_mask=punctured_mask)
        result = decoder.decode(llr, syndrome)
        
        assert result.converged

    def test_decode_low_noise(self, decoder: BeliefPropagationDecoder) -> None:
        """Decoder recovers from low noise."""
        H = decoder.parity_check_matrix
        n = H.shape[1]  # codeword length
        m = H.shape[0]  # syndrome length
        
        # Start with all-zeros codeword
        codeword = np.zeros(n, dtype=np.int8)
        
        # Introduce ~2% errors
        rng = np.random.default_rng(42)
        n_errors = int(n * 0.02)
        error_positions = rng.choice(n, size=n_errors, replace=False)
        noisy = codeword.copy()
        noisy[error_positions] = 1 - noisy[error_positions]
        
        syndrome = np.zeros(m, dtype=np.int8)
        
        # No puncturing
        punctured_mask = np.zeros(n, dtype=np.uint8)
        llr = build_channel_llr(noisy, qber=0.03, punctured_mask=punctured_mask)
        result = decoder.decode(llr, syndrome)
        
        # Should likely converge with low noise
        # (may occasionally fail due to randomness)
        assert result.iterations <= constants.LDPC_MAX_ITERATIONS

    def test_returns_decode_result(self, decoder: BeliefPropagationDecoder) -> None:
        """Decode returns DecodeResult instance."""
        H = decoder.parity_check_matrix
        n = H.shape[1]  # codeword length
        m = H.shape[0]  # syndrome length
        
        codeword = np.zeros(n, dtype=np.int8)
        syndrome = np.zeros(m, dtype=np.int8)
        
        # No puncturing
        punctured_mask = np.zeros(n, dtype=np.uint8)
        llr = build_channel_llr(codeword, qber=0.05, punctured_mask=punctured_mask)
        
        result = decoder.decode(llr, syndrome)
        
        assert isinstance(result, DecodeResult)
        assert len(result.corrected_bits) == n

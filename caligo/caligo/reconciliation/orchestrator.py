"""
Phase III Reconciliation Orchestrator.

Coordinates the complete reconciliation flow between Alice and Bob,
integrating rate selection, encoding, decoding, verification, and
leakage tracking.

This is the main entry point for Phase III execution.

References
----------
- recon_phase_spec.md: Complete specification
- caligo_architecture.md: Phase III integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from bitarray import bitarray

from caligo.reconciliation import constants
from caligo.reconciliation.ldpc_decoder import (
    BeliefPropagationDecoder,
    DecodeResult,
    build_channel_llr,
    syndrome_guided_refinement,
)
from caligo.reconciliation.ldpc_encoder import (
    SyndromeBlock,
    encode_block,
    generate_padding,
    prepare_frame,
)
from caligo.reconciliation.compiled_matrix import CompiledParityCheckMatrix
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.rate_selector import select_rate_with_parameters
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ReconciliationOrchestratorConfig:
    """
    Orchestrator configuration.

    Attributes
    ----------
    frame_size : int
        LDPC frame size.
    max_iterations : int
        BP decoder iterations.
    max_blind_iterations : int
        Blind reconciliation iterations (alias for max_retries).
    max_retries : int
        Retry attempts per block.
    hash_bits : int
        Verification hash length.
    f_crit : float
        Efficiency threshold.
    """

    frame_size: int = constants.LDPC_FRAME_SIZE
    max_iterations: int = constants.LDPC_MAX_ITERATIONS
    max_blind_iterations: int = constants.LDPC_MAX_RETRIES  # Alias
    max_retries: int = constants.LDPC_MAX_RETRIES
    hash_bits: int = constants.LDPC_HASH_BITS
    f_crit: float = constants.LDPC_F_CRIT


# =============================================================================
# Block Result
# =============================================================================


@dataclass
class BlockResult:
    """
    Result of single block reconciliation.

    Attributes
    ----------
    corrected_payload : np.ndarray
        Corrected bits (payload only).
    verified : bool
        Hash verification passed.
    converged : bool
        BP decoder converged.
    error_count : int
        Detected errors in payload.
    syndrome_length : int
        Leakage from this block.
    """

    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    error_count: int
    syndrome_length: int


# =============================================================================
# Orchestrator
# =============================================================================


class ReconciliationOrchestrator:
    """
    Coordinate Alice/Bob reconciliation flow.

    Manages the complete Phase III execution including:
    - Block partitioning
    - Rate selection
    - Syndrome encoding (Alice)
    - BP decoding (Bob)
    - Hash verification
    - Leakage tracking

    Parameters
    ----------
    matrix_manager : MatrixManager
        LDPC matrix pool.
    leakage_tracker : LeakageTracker, optional
        Leakage accounting. Created internally if not provided.
    config : ReconciliationOrchestratorConfig, optional
        Orchestrator settings.
    safety_cap : int, optional
        Leakage safety cap (alternative to leakage_tracker).
    """

    def __init__(
        self,
        matrix_manager: MatrixManager = None,
        leakage_tracker: LeakageTracker = None,
        config: Optional[ReconciliationOrchestratorConfig] = None,
        safety_cap: int = None,
    ) -> None:
        self.config = config or ReconciliationOrchestratorConfig()
        self.matrix_manager = matrix_manager
        
        # Create leakage tracker if not provided
        if leakage_tracker is not None:
            self.leakage_tracker = leakage_tracker
        elif safety_cap is not None:
            self.leakage_tracker = LeakageTracker(safety_cap=safety_cap)
        else:
            self.leakage_tracker = LeakageTracker(safety_cap=0)  # No cap
            
        self.decoder = BeliefPropagationDecoder(
            max_iterations=self.config.max_iterations
        )
        self.hash_verifier = PolynomialHashVerifier(
            hash_bits=self.config.hash_bits
        )

    def reconcile_block(
        self,
        alice_key: np.ndarray,
        bob_key: np.ndarray,
        qber_estimate: float,
        block_id: int = 0,
    ) -> BlockResult:
        """
        Reconcile a single key block.

        Parameters
        ----------
        alice_key : np.ndarray
            Alice's payload bits (uint8).
        bob_key : np.ndarray
            Bob's payload bits (uint8).
        qber_estimate : float
            QBER for rate selection.
        block_id : int
            Block identifier for logging.

        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        payload_len = len(alice_key)
        frame_size = self.config.frame_size

        # Rate selection
        rate_params = select_rate_with_parameters(
            qber_estimate=qber_estimate,
            payload_length=payload_len,
            frame_size=frame_size,
            available_rates=self.matrix_manager.rates,
            f_crit=self.config.f_crit,
        )

        rate = rate_params.rate
        # In baseline coset decoding, shortened bits are the padding needed
        # to embed the payload into a fixed-length LDPC frame.
        n_shortened = max(0, frame_size - payload_len)
        prng_seed = block_id + 12345  # Deterministic seed

        # Get parity-check matrix
        H = self.matrix_manager.get_matrix(rate)
        compiled_H = self.matrix_manager.get_compiled(rate)

        # Alice: encode syndrome
        syndrome_block = encode_block(
            alice_key,  # positional to avoid param name mismatch
            H,
            rate,
            n_shortened,
            prng_seed,
        )

        # Bob: decode
        decode_result = self._decode_with_retry(
            bob_key=bob_key,
            syndrome=syndrome_block.syndrome,
            H=compiled_H,
            n_shortened=n_shortened,
            prng_seed=prng_seed,
            qber_estimate=qber_estimate,
        )

        # Extract corrected payload
        corrected_payload = decode_result.corrected_bits[:payload_len]
        error_count = int(np.sum(bob_key != corrected_payload))

        # Verify with hash
        hash_seed = block_id
        alice_hash = self.hash_verifier.compute_hash(alice_key, hash_seed)
        # verify(bits, expected_hash, seed)
        verified = self.hash_verifier.verify(corrected_payload, alice_hash, hash_seed)

        # Record leakage
        self.leakage_tracker.record_block(
            syndrome_length=len(syndrome_block.syndrome),
            n_shortened=n_shortened,
            frame_size=frame_size,
            block_id=block_id,
        )

        logger.debug(
            "Block %d: rate=%.2f, converged=%s, verified=%s, errors=%d",
            block_id, rate, decode_result.converged, verified, error_count
        )

        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=decode_result.converged,
            error_count=error_count,
            syndrome_length=len(syndrome_block.syndrome),
        )

    def reconcile_key(
        self,
        alice_key: np.ndarray,
        bob_key: np.ndarray,
        qber_estimate: float,
    ) -> Tuple[np.ndarray, List[BlockResult]]:
        """
        Reconcile an arbitrarily long key by partitioning into LDPC blocks.

        Parameters
        ----------
        alice_key : np.ndarray
            Alice's payload bits.
        bob_key : np.ndarray
            Bob's payload bits.
        qber_estimate : float
            QBER for rate selection / LLR construction.

        Returns
        -------
        Tuple[np.ndarray, List[BlockResult]]
            (reconciled_payload, per_block_results). Failed blocks are dropped
            from the returned reconciled_payload.
        """
        if len(alice_key) != len(bob_key):
            raise ValueError("alice_key and bob_key must have same length")

        block_size = int(self.config.frame_size)
        alice_blocks = partition_key(alice_key, block_size)
        bob_blocks = partition_key(bob_key, block_size)

        corrected_parts: List[np.ndarray] = []
        results: List[BlockResult] = []

        for block_id, (a_blk, b_blk) in enumerate(zip(alice_blocks, bob_blocks)):
            res = self.reconcile_block(
                alice_key=a_blk,
                bob_key=b_blk,
                qber_estimate=qber_estimate,
                block_id=block_id,
            )
            results.append(res)
            if res.verified:
                corrected_parts.append(res.corrected_payload)

        if len(corrected_parts) == 0:
            return np.array([], dtype=np.uint8), results

        return np.concatenate(corrected_parts).astype(np.uint8), results

    def _decode_with_retry(
        self,
        bob_key: np.ndarray,
        syndrome: np.ndarray,
        H: CompiledParityCheckMatrix,
        n_shortened: int,
        prng_seed: int,
        qber_estimate: float,
    ) -> DecodeResult:
        """
        Decode with retry and LLR damping.

        Parameters
        ----------
        bob_key : np.ndarray
            Bob's received bits.
        syndrome : np.ndarray
            Alice's syndrome.
        H : CompiledParityCheckMatrix
            Compiled parity-check representation (fast adjacency + syndrome ops).
        n_shortened : int
            Shortened bits.
        prng_seed : int
            Padding seed.
        qber_estimate : float
            QBER for LLR.

        Returns
        -------
        DecodeResult
            Best decoding result after retries.
        """
        payload_len = len(bob_key)

        # Construct Bob's full frame [payload | padding].
        bob_frame = prepare_frame(
            bob_key,
            n_shortened=n_shortened,
            prng_seed=prng_seed,
        )

        # Coset decoding target is Alice's syndrome H·x_A.
        target_syndrome = syndrome.astype(np.uint8, copy=False)
        local_syndrome = H.compute_syndrome(bob_frame)

        known_padding = generate_padding(n_shortened, prng_seed)

        best_result = None

        for attempt in range(self.config.max_retries + 1):
            # LLR damping for retries
            llr_damping = 1.0 - attempt * 0.15

            # Build channel LLRs for Alice's bits, using Bob's received bits as
            # the channel observation and treating padding as known bits.
            llr = build_channel_llr(bob_key, qber_estimate, known_bits=known_padding)
            llr[:payload_len] *= llr_damping

            # Apply syndrome-guided refinement
            # Pass local and received syndromes for proper refinement
            llr = syndrome_guided_refinement(
                llr, local_syndrome, syndrome, H, payload_len
            )

            iter_scale = 1.0 + attempt * 0.5
            result = self.decoder.decode(
                llr,
                target_syndrome,
                H=H,
                max_iterations=int(self.config.max_iterations * iter_scale),
            )

            # For coset decoding, `result.corrected_bits` is the decoded estimate
            # of Alice's full frame.
            error_count = int(np.sum(bob_key != result.corrected_bits[:payload_len]))

            if result.converged:
                return result

            best_result = result

        return best_result or result

    def should_abort(self) -> bool:
        """Check if leakage cap exceeded."""
        return self.leakage_tracker.should_abort()


def partition_key(
    key: np.ndarray,
    block_size: int,
) -> List[np.ndarray]:
    """
    Partition key into blocks.

    Parameters
    ----------
    key : np.ndarray
        Full key array.
    block_size : int
        Target block size.

    Returns
    -------
    List[np.ndarray]
        List of key blocks.
    """
    blocks = []
    for i in range(0, len(key), block_size):
        blocks.append(key[i:i + block_size])
    return blocks

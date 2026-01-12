"""
Phase III Reconciliation Orchestrator.

.. deprecated:: 2.0
    This module is DEPRECATED and will be removed in a future version.
    Use the Strategy Pattern (caligo.reconciliation.strategies) instead.
    
    The ReconciliationOrchestrator class has been superseded by:
    - BlindStrategy for blind reconciliation
    - BaselineStrategy for baseline reconciliation
    
    The partition_key utility function remains available.

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

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from bitarray import bitarray

from caligo.reconciliation import constants
from caligo.reconciliation.ldpc_decoder import (
    BeliefPropagationDecoder,
    DecodeResult,
)
from caligo.reconciliation.block_reconciler import (
    BlockReconciler,
    BlockReconcilerConfig,
    BlockResult,
)
from caligo.reconciliation.compiled_matrix import CompiledParityCheckMatrix
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.utils.logging import get_logger

logger = get_logger(__name__)

# Emit deprecation warning on module import
warnings.warn(
    "caligo.reconciliation.orchestrator is deprecated. "
    "Use caligo.reconciliation.strategies.BlindStrategy or BaselineStrategy instead.",
    DeprecationWarning,
    stacklevel=2,
)


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
    retry_damping_step : float
        Damping step per retry attempt (multiplies payload LLRs).
    retry_iteration_scale : float
        Additional iteration scaling per retry attempt.
    verify : bool
        Whether to run verification hash and account its leakage.
    enforce_leakage_cap : bool
        If True, exceeding the leakage cap raises a fatal error.
    """

    frame_size: int = constants.LDPC_FRAME_SIZE
    max_iterations: int = constants.LDPC_MAX_ITERATIONS
    max_blind_iterations: int = constants.LDPC_MAX_RETRIES  # Alias
    max_retries: int = constants.LDPC_MAX_RETRIES
    hash_bits: int = constants.LDPC_HASH_BITS
    f_crit: float = constants.LDPC_F_CRIT
    retry_damping_step: float = 0.15
    retry_iteration_scale: float = 0.5
    verify: bool = True
    enforce_leakage_cap: bool = True


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
        if matrix_manager is None:
            raise ValueError("matrix_manager must be provided")
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

        self._block_reconciler = BlockReconciler(
            matrix_manager=self.matrix_manager,
            decoder=self.decoder,
            hash_verifier=self.hash_verifier,
            leakage_tracker=self.leakage_tracker,
            config=BlockReconcilerConfig(
                frame_size=int(self.config.frame_size),
                max_iterations=int(self.config.max_iterations),
                max_retries=int(self.config.max_retries),
                hash_bits=int(self.config.hash_bits),
                f_crit=float(self.config.f_crit),
                retry_damping_step=float(self.config.retry_damping_step),
                retry_iteration_scale=float(self.config.retry_iteration_scale),
                verify=bool(self.config.verify),
                enforce_leakage_cap=bool(self.config.enforce_leakage_cap),
            ),
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

        Notes
        -----
        The returned `BlockResult` includes observability fields:

        - `retry_report`: retry attempts, damping factors, and iteration limits
        - `leakage`: per-block leakage by source
        """
        result = self._block_reconciler.reconcile_baseline(
            alice_key=alice_key,
            bob_key=bob_key,
            qber_estimate=float(qber_estimate),
            block_id=int(block_id),
        )
        logger.debug(
            "Block %d: converged=%s, verified=%s, errors=%d",
            int(block_id), bool(result.converged), bool(result.verified), int(result.error_count)
        )
        return result

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
        """Backward-compatible alias for `decode_with_retry`.

        Notes
        -----
        This method is intentionally kept for backward compatibility.
        New code (including the Phase III on-wire Bob handler) should call
        `decode_with_retry`.
        """
        return self.decode_with_retry(
            bob_key=bob_key,
            syndrome=syndrome,
            H=H,
            n_shortened=n_shortened,
            prng_seed=prng_seed,
            qber_estimate=qber_estimate,
        )

    def decode_with_retry(
        self,
        bob_key: np.ndarray,
        syndrome: np.ndarray,
        H: CompiledParityCheckMatrix,
        n_shortened: int,
        prng_seed: int,
        qber_estimate: float,
    ) -> DecodeResult:
        """Decode one baseline block with retry and LLR damping.

        This is an effectively-public Phase III entrypoint: Bob consumes Alice's
        on-wire metadata (syndrome, rate-derived matrix, shortening params,
        deterministic seed) and must reproduce the exact same frame embedding.

        Parameters
        ----------
        bob_key : np.ndarray
            Bob's received payload bits.
        syndrome : np.ndarray
            Alice's target syndrome.
        H : CompiledParityCheckMatrix
            Compiled parity-check representation.
        n_shortened : int
            Number of shortened bits appended to payload.
        prng_seed : int
            Deterministic padding seed (must match Alice).
        qber_estimate : float
            QBER used for channel LLR construction.

        Returns
        -------
        DecodeResult
            Best decoding result after retries.

        Notes
        -----
        Recoverable vs. fatal protocol decisions should be handled at a higher
        layer:

        - `DecodeResult.converged` reports algorithm status.
        - Leakage cap enforcement is handled via `LeakageTracker` during full
          block reconciliation.
        """
        result, _report = self._block_reconciler._decode_with_retry(
            bob_key=bob_key,
            syndrome=syndrome,
            H=H,
            n_shortened=int(n_shortened),
            prng_seed=int(prng_seed),
            qber_estimate=float(qber_estimate),
            raise_on_failure=False,
        )
        return result

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

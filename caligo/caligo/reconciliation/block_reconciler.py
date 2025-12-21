"""Single-block reconciliation.

This module provides a narrow, testable unit that reconciles exactly one
baseline (coset-decoding) LDPC block.

The intent is to keep `ReconciliationOrchestrator` focused on multi-block
iteration and aggregation, while making the single-block contract explicit and
removing hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from caligo.reconciliation import constants
from caligo.reconciliation.compiled_matrix import CompiledParityCheckMatrix
from caligo.reconciliation.hash_verifier import PolynomialHashVerifier
from caligo.reconciliation.ldpc_decoder import (
    BeliefPropagationDecoder,
    DecodeResult,
    build_channel_llr,
    syndrome_guided_refinement,
)
from caligo.reconciliation.ldpc_encoder import (
    encode_block_from_payload,
    prepare_frame,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker
from caligo.reconciliation.matrix_manager import MatrixManager
from caligo.reconciliation.rate_selector import select_rate_with_parameters
from caligo.types.exceptions import DecodingFailure, LeakageBudgetExceeded
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetryReport:
    """Observability report for block retry behavior."""

    attempts: int
    damping_factors: List[float]
    iteration_limits: List[int]
    best_syndrome_errors: int


@dataclass(frozen=True)
class LeakageBreakdown:
    """Leakage accounting for a block, by source."""

    syndrome_bits: int
    hash_bits: int
    retry_penalty_bits: int


@dataclass
class BlockResult:
    """
    Result of single block reconciliation.
    
    Per Theoretical Report v2 §1.2, leakage accounting must be exact:
    leak_EC = syndrome_leakage + hash_leakage + revealed_leakage
    """
    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    iterations_used: int
    syndrome_leakage: int
    revealed_leakage: int
    hash_leakage: int
    retry_count: int
    effective_rate: float = 0.5
    
    # Legacy compatibility
    error_count: int = 0
    syndrome_length: int = 0
    retry_report: Optional[RetryReport] = None
    leakage: Optional[LeakageBreakdown] = None

    @property
    def total_leakage(self) -> int:
        """
        Total leakage for this block.
        
        Per Theoretical Report v2 Eq. (leak_EC):
        leak = |Σ| + |Hash| + |Revealed|
        """
        return self.syndrome_leakage + self.revealed_leakage + self.hash_leakage


@dataclass(frozen=True)
class BlockReconcilerConfig:
    """Configuration for single-block reconciliation."""

    frame_size: int
    max_iterations: int
    max_retries: int
    hash_bits: int
    f_crit: float
    retry_damping_step: float = 0.15
    retry_iteration_scale: float = 0.5
    verify: bool = True
    enforce_leakage_cap: bool = True


class BlockReconciler:
    """Reconcile exactly one baseline LDPC block."""

    def __init__(
        self,
        matrix_manager: MatrixManager,
        decoder: BeliefPropagationDecoder,
        hash_verifier: PolynomialHashVerifier,
        leakage_tracker: LeakageTracker,
        config: BlockReconcilerConfig,
    ) -> None:
        self._matrix_manager = matrix_manager
        self._decoder = decoder
        self._hash_verifier = hash_verifier
        self._leakage_tracker = leakage_tracker
        self._config = config

    def reconcile_baseline(
        self,
        alice_key: np.ndarray,
        bob_key: np.ndarray,
        qber_estimate: float,
        block_id: int,
    ) -> BlockResult:
        """Reconcile one block and record leakage.

        Parameters
        ----------
        block_id : int
            Block identifier (used for deterministic seed derivation).

        Returns
        -------
        BlockResult
            Block reconciliation result.
        """
        if bool(self._config.enforce_leakage_cap) and self._leakage_tracker.should_abort():
            raise LeakageBudgetExceeded(
                "Reconciliation leakage budget exceeded",
                actual_leakage=int(self._leakage_tracker.total_leakage),
                max_allowed=int(self._leakage_tracker.safety_cap),
            )
        if int(alice_key.shape[0]) != int(bob_key.shape[0]):
            raise ValueError("alice_key and bob_key must have same length")

        payload_len = int(alice_key.shape[0])
        frame_size = int(self._config.frame_size)

        rate_params = select_rate_with_parameters(
            qber_estimate=float(qber_estimate),
            payload_length=payload_len,
            frame_size=frame_size,
            available_rates=self._matrix_manager.rates,
            f_crit=float(self._config.f_crit),
        )
        rate = float(rate_params.rate)

        # Get puncture pattern for this rate
        # Special case: rate 0.5 is the mother code rate (no puncturing)
        puncture_pattern = self._matrix_manager.get_puncture_pattern(rate)
        if puncture_pattern is None:
            if abs(rate - 0.5) < 0.01:
                # Rate 0.5 = mother code, no puncturing needed
                puncture_pattern = np.zeros(frame_size, dtype=np.uint8)
            else:
                raise ValueError(
                    f"No puncture pattern available for rate {rate}. "
                    f"Available rates: {self._matrix_manager.available_pattern_rates}"
                )

        # Always use mother code (rate 0.5) with puncturing patterns
        mother_rate = 0.5
        H = self._matrix_manager.get_matrix(mother_rate)
        compiled_H = self._matrix_manager.get_compiled(mother_rate)

        syndrome_block = encode_block_from_payload(
            payload=alice_key,
            H=H,
            puncture_pattern=puncture_pattern,
        )

        decode_result, retry_report = self._decode_with_retry(
            bob_key=bob_key,
            syndrome=syndrome_block.syndrome,
            H=compiled_H,
            qber_estimate=float(qber_estimate),
            puncture_pattern=puncture_pattern,
            raise_on_failure=False,
        )

        corrected_payload = decode_result.corrected_bits[:payload_len].astype(
            np.uint8, copy=False
        )
        error_count = int(np.sum(bob_key != corrected_payload))

        verified = True
        hash_bits = 0
        if self._config.verify:
            hash_seed = int(block_id)
            alice_hash = int(self._hash_verifier.compute_hash(alice_key, hash_seed))
            verified = bool(
                self._hash_verifier.verify(corrected_payload, alice_hash, seed=hash_seed)
            )
            hash_bits = int(self._config.hash_bits)

        # Conservative retry penalty: baseline protocol does not send retry
        # counts on-wire; keep explicit and default to 0.
        retry_penalty_bits = 0

        self._leakage_tracker.record_block(
            block_id=int(block_id),
            syndrome_bits=int(syndrome_block.syndrome.shape[0]),
            hash_bits=int(hash_bits),
            n_shortened=int(puncture_pattern.sum()),  # Number of punctured bits
            frame_size=int(frame_size),
        )
        if bool(self._config.enforce_leakage_cap) and self._leakage_tracker.should_abort():
            raise LeakageBudgetExceeded(
                "Reconciliation leakage budget exceeded",
                actual_leakage=int(self._leakage_tracker.total_leakage),
                max_allowed=int(self._leakage_tracker.safety_cap),
            )

        leakage = LeakageBreakdown(
            syndrome_bits=int(syndrome_block.syndrome.shape[0]),
            hash_bits=int(hash_bits),
            retry_penalty_bits=int(retry_penalty_bits),
        )

        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=bool(decode_result.converged),
            iterations_used=int(decode_result.iterations),
            syndrome_leakage=int(syndrome_block.syndrome.shape[0]),
            revealed_leakage=0,  # Baseline protocol doesn't reveal bits
            hash_leakage=int(hash_bits),
            retry_count=int(retry_report.attempts - 1),
            effective_rate=float(rate),
            error_count=error_count,
            syndrome_length=int(syndrome_block.syndrome.shape[0]),
            retry_report=retry_report,
            leakage=leakage,
        )

    def _decode_with_retry(
        self,
        bob_key: np.ndarray,
        syndrome: np.ndarray,
        H: CompiledParityCheckMatrix,
        qber_estimate: float,
        puncture_pattern: np.ndarray,
        *,
        raise_on_failure: bool = False,
    ) -> tuple[DecodeResult, RetryReport]:
        """Decode with retry and return a retry report.

        Parameters
        ----------
        puncture_pattern : np.ndarray
            Untainted puncturing pattern.
        raise_on_failure : bool
            If True, raise `DecodingFailure` when no attempt converges.
            If False, return the best attempt (backward-compatible behavior).
        """
        payload_len = int(bob_key.shape[0])

        # Pattern-based mode: construct frame using puncture pattern
        bob_frame = prepare_frame(
            bob_key,
            puncture_pattern=puncture_pattern,
        )
        # Construct LLR with punctured mask
        base_llr = build_channel_llr(
            bob_key,
            float(qber_estimate),
            punctured_mask=puncture_pattern,
        )

        target_syndrome = syndrome.astype(np.uint8, copy=False)
        local_syndrome = H.compute_syndrome(bob_frame)

        best: DecodeResult | None = None
        best_syndrome_errors: int | None = None
        damping_factors: List[float] = []
        iteration_limits: List[int] = []

        for attempt in range(int(self._config.max_retries) + 1):
            damping = float(1.0 - float(attempt) * float(self._config.retry_damping_step))
            damping_factors.append(damping)

            iter_scale = float(1.0 + float(attempt) * float(self._config.retry_iteration_scale))
            iteration_limit = int(float(self._config.max_iterations) * iter_scale)
            iteration_limits.append(iteration_limit)

            llr = base_llr.copy()
            llr[:payload_len] *= damping

            llr = syndrome_guided_refinement(
                llr,
                local_syndrome,
                target_syndrome,
                H,
                payload_len,
            )

            res = self._decoder.decode(
                llr,
                target_syndrome,
                H=H,
                max_iterations=iteration_limit,
            )

            if best is None or int(res.syndrome_errors) < int(best.syndrome_errors):
                best = res
                best_syndrome_errors = int(res.syndrome_errors)

            if bool(res.converged):
                report = RetryReport(
                    attempts=int(attempt) + 1,
                    damping_factors=damping_factors,
                    iteration_limits=iteration_limits,
                    best_syndrome_errors=int(res.syndrome_errors),
                )
                return res, report

        assert best is not None
        report = RetryReport(
            attempts=int(self._config.max_retries) + 1,
            damping_factors=damping_factors,
            iteration_limits=iteration_limits,
            best_syndrome_errors=int(
                best_syndrome_errors if best_syndrome_errors is not None else best.syndrome_errors
            ),
        )
        if raise_on_failure:
            raise DecodingFailure(
                f"BP decoder failed to converge after {report.attempts} attempts"
            )
        return best, report

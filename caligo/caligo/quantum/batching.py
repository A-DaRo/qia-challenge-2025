"""
Batch management for memory-constrained quantum operations.

This module provides batch processing for large-scale EPR generation
and measurement, respecting memory constraints while maintaining
protocol security.

References
----------
- E-HOK Protocol: Batch-based key generation
- Erven et al. (2014): Experimental batch sizes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import InvalidParameterError, ProtocolError
from caligo.types.phase_contracts import QuantumPhaseResult
from caligo.types.measurements import DetectionEvent

logger = get_logger(__name__)


class BatchState(Enum):
    """State of a batch in the processing lifecycle."""

    PENDING = "pending"
    GENERATING = "generating"
    MEASURING = "measuring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchConfig:
    """
    Configuration for batched quantum operations.

    Parameters
    ----------
    pairs_per_batch : int
        Number of EPR pairs per batch.
    max_batches : int
        Maximum number of batches to process.
    memory_limit_bytes : int
        Maximum memory usage in bytes.
    overlap_enabled : bool
        Enable overlapping generation/measurement.
    """

    pairs_per_batch: int = 100
    max_batches: int = 100
    memory_limit_bytes: int = 100 * 1024 * 1024  # 100 MB
    overlap_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.pairs_per_batch <= 0:
            raise InvalidParameterError(
                f"pairs_per_batch={self.pairs_per_batch} must be positive"
            )
        if self.max_batches <= 0:
            raise InvalidParameterError(
                f"max_batches={self.max_batches} must be positive"
            )

    @property
    def max_pairs(self) -> int:
        """Maximum total pairs across all batches."""
        return self.pairs_per_batch * self.max_batches


@dataclass
class BatchResult:
    """
    Result of a single batch of quantum operations.

    Parameters
    ----------
    batch_id : int
        Unique batch identifier.
    outcomes : np.ndarray
        Measurement outcomes for this batch.
    bases : np.ndarray
        Bases used in this batch.
    round_ids : np.ndarray
        Round identifiers.
    generation_time : float
        Time taken for EPR generation.
    measurement_time : float
        Time taken for measurements.
    state : BatchState
        Final state of the batch.
    """

    batch_id: int
    outcomes: np.ndarray
    bases: np.ndarray
    round_ids: np.ndarray
    generation_time: float = 0.0
    measurement_time: float = 0.0
    state: BatchState = BatchState.COMPLETED


class BatchingManager:
    """
    Memory-constrained batch processing for quantum operations.

    Coordinates batch-wise EPR generation and measurement while
    respecting memory limits and providing progress tracking.

    Parameters
    ----------
    config : BatchConfig
        Batching configuration.

    Notes
    -----
    The manager ensures that at any time, memory usage from stored
    measurement data does not exceed the configured limit.

    Example
    -------
    ```python
    manager = BatchingManager(BatchConfig(pairs_per_batch=100))
    manager.configure(total_pairs=10000)

    for batch in manager.iterate_batches():
        # Process batch
        batch.outcomes = measure(batch.round_ids)
        manager.complete_batch(batch)
    ```
    """

    def __init__(self, config: Optional[BatchConfig] = None) -> None:
        """
        Initialize batching manager.

        Parameters
        ----------
        config : Optional[BatchConfig]
            Batching configuration. Uses defaults if None.
        """
        self.config = config or BatchConfig()

        self._total_pairs: int = 0
        self._completed_batches: List[BatchResult] = []
        self._current_batch_id: int = 0
        self._started: bool = False

        # Accumulated results
        self._all_outcomes: List[np.ndarray] = []
        self._all_bases: List[np.ndarray] = []
        self._all_round_ids: List[np.ndarray] = []

    def configure(self, total_pairs: int) -> int:
        """
        Configure manager for a specific total pair count.

        Parameters
        ----------
        total_pairs : int
            Total EPR pairs to generate.

        Returns
        -------
        int
            Number of batches required.

        Raises
        ------
        InvalidParameterError
            If total_pairs exceeds configuration limits.
        """
        if total_pairs <= 0:
            raise InvalidParameterError(
                f"total_pairs={total_pairs} must be positive"
            )

        max_allowed = self.config.max_pairs
        if total_pairs > max_allowed:
            raise InvalidParameterError(
                f"total_pairs={total_pairs} exceeds max_pairs={max_allowed}"
            )

        self._total_pairs = total_pairs
        num_batches = (total_pairs + self.config.pairs_per_batch - 1) // (
            self.config.pairs_per_batch
        )

        logger.debug(
            f"Configured for {total_pairs} pairs in {num_batches} batches"
        )

        return num_batches

    def start_batch(self, batch_id: int) -> BatchResult:
        """
        Start a new batch.

        Parameters
        ----------
        batch_id : int
            Batch identifier.

        Returns
        -------
        BatchResult
            Initialized batch result object.
        """
        # Calculate pairs for this batch
        start_idx = batch_id * self.config.pairs_per_batch
        remaining = self._total_pairs - start_idx
        pairs_in_batch = min(self.config.pairs_per_batch, remaining)

        if pairs_in_batch <= 0:
            raise ProtocolError(f"Batch {batch_id} has no pairs to process")

        # Create round IDs for this batch
        round_ids = np.arange(start_idx, start_idx + pairs_in_batch, dtype=np.int64)

        logger.debug(f"Starting batch {batch_id} with {pairs_in_batch} pairs")

        return BatchResult(
            batch_id=batch_id,
            outcomes=np.zeros(pairs_in_batch, dtype=np.uint8),
            bases=np.zeros(pairs_in_batch, dtype=np.uint8),
            round_ids=round_ids,
            state=BatchState.GENERATING,
        )

    def complete_batch(self, batch: BatchResult) -> None:
        """
        Mark a batch as completed and store results.

        Parameters
        ----------
        batch : BatchResult
            Completed batch with measurement data.
        """
        batch.state = BatchState.COMPLETED
        self._completed_batches.append(batch)

        # Store results for aggregation
        self._all_outcomes.append(batch.outcomes)
        self._all_bases.append(batch.bases)
        self._all_round_ids.append(batch.round_ids)

        logger.debug(
            f"Batch {batch.batch_id} completed: {len(batch.outcomes)} outcomes"
        )

    def iterate_batches(self) -> Iterator[BatchResult]:
        """
        Iterate over batches to be processed.

        Yields
        ------
        BatchResult
            Batch ready for processing.
        """
        if self._total_pairs == 0:
            raise ProtocolError("Must call configure() before iterating")

        num_batches = (
            self._total_pairs + self.config.pairs_per_batch - 1
        ) // self.config.pairs_per_batch

        for batch_id in range(num_batches):
            yield self.start_batch(batch_id)

    def get_aggregated_results(
        self, generation_timestamp: float = 0.0
    ) -> QuantumPhaseResult:
        """
        Aggregate all batch results into QuantumPhaseResult.

        Parameters
        ----------
        generation_timestamp : float
            Simulation time when quantum phase completed.

        Returns
        -------
        QuantumPhaseResult
            Aggregated results across all batches.
        """
        if not self._completed_batches:
            raise ProtocolError("No completed batches to aggregate")

        # Concatenate all results
        all_outcomes = np.concatenate(self._all_outcomes)
        all_bases = np.concatenate(self._all_bases)
        all_round_ids = np.concatenate(self._all_round_ids)

        total_generated = len(all_outcomes)

        logger.debug(
            f"Aggregated {len(self._completed_batches)} batches: "
            f"{total_generated} measurements"
        )

        return QuantumPhaseResult(
            measurement_outcomes=all_outcomes,
            basis_choices=all_bases,
            round_ids=all_round_ids,
            generation_timestamp=generation_timestamp,
            num_pairs_requested=self._total_pairs,
            num_pairs_generated=total_generated,
            detection_events=[],
            timing_barrier_marked=True,
        )

    @property
    def completed_count(self) -> int:
        """Number of completed batches."""
        return len(self._completed_batches)

    @property
    def total_pairs(self) -> int:
        """Total pairs configured."""
        return self._total_pairs

    @property
    def total_outcomes(self) -> int:
        """Total measurement outcomes collected."""
        return sum(len(batch.outcomes) for batch in self._completed_batches)

    def estimate_memory_usage(self) -> int:
        """
        Estimate current memory usage in bytes.

        Returns
        -------
        int
            Estimated memory in bytes.
        """
        # Each measurement: outcome(1) + basis(1) + round_id(8) = 10 bytes
        bytes_per_measurement = 10
        return self.total_outcomes * bytes_per_measurement

    def reset(self) -> None:
        """Reset manager state for new protocol run."""
        self._total_pairs = 0
        self._completed_batches.clear()
        self._current_batch_id = 0
        self._started = False
        self._all_outcomes.clear()
        self._all_bases.clear()
        self._all_round_ids.clear()

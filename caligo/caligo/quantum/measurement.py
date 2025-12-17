"""
Quantum measurement execution and result buffering.

This module provides measurement execution using NetQASM's
QubitMeasureBasis and efficient buffering of measurement outcomes.

References
----------
- NetQASM: QubitMeasureBasis enum
- BB84 Protocol: Z and X basis measurements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from caligo.utils.logging import get_logger
from caligo.quantum.basis import BASIS_X, BASIS_Z

if TYPE_CHECKING:
    from netqasm.sdk.qubit import Qubit

logger = get_logger(__name__)


@dataclass
class MeasurementResult:
    """
    Single measurement result.

    Parameters
    ----------
    outcome : int
        Measurement outcome (0 or 1).
    basis : int
        Basis used (0=Z, 1=X).
    round_id : int
        Round/pair identifier.
    measurement_time : float
        Simulation time of measurement.
    """

    outcome: int
    basis: int
    round_id: int
    measurement_time: float = 0.0


class MeasurementBuffer:
    """
    Efficient buffer for accumulating measurement outcomes.

    Pre-allocates memory for batch sizes and provides efficient
    append and retrieval operations.

    Parameters
    ----------
    capacity : int
        Initial buffer capacity.
    growth_factor : float
        Factor to grow buffer when full (default 2.0).
    """

    def __init__(self, capacity: int = 1000, growth_factor: float = 2.0) -> None:
        """
        Initialize measurement buffer.

        Parameters
        ----------
        capacity : int
            Initial buffer capacity.
        growth_factor : float
            Growth factor when resizing.
        """
        self._capacity = capacity
        self._growth_factor = growth_factor

        # Pre-allocate arrays
        self._outcomes = np.zeros(capacity, dtype=np.uint8)
        self._bases = np.zeros(capacity, dtype=np.uint8)
        self._round_ids = np.zeros(capacity, dtype=np.int64)
        self._times = np.zeros(capacity, dtype=np.float64)

        self._count = 0

    def add_outcome(
        self,
        outcome: int,
        basis: int,
        round_id: int,
        measurement_time: float = 0.0,
    ) -> None:
        """
        Add a single measurement outcome.

        Parameters
        ----------
        outcome : int
            Measurement result (0 or 1).
        basis : int
            Measurement basis (0=Z, 1=X).
        round_id : int
            Round identifier.
        measurement_time : float
            Simulation time.
        """
        if self._count >= self._capacity:
            self._grow()

        self._outcomes[self._count] = outcome
        self._bases[self._count] = basis
        self._round_ids[self._count] = round_id
        self._times[self._count] = measurement_time
        self._count += 1

    def add_batch(
        self,
        outcomes: np.ndarray,
        bases: np.ndarray,
        round_ids: np.ndarray,
        measurement_times: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a batch of measurement outcomes efficiently.

        Parameters
        ----------
        outcomes : np.ndarray
            Array of measurement outcomes.
        bases : np.ndarray
            Array of bases used.
        round_ids : np.ndarray
            Array of round identifiers.
        measurement_times : Optional[np.ndarray]
            Array of measurement times.
        """
        batch_size = len(outcomes)

        # Ensure capacity
        while self._count + batch_size > self._capacity:
            self._grow()

        end_idx = self._count + batch_size
        self._outcomes[self._count : end_idx] = outcomes
        self._bases[self._count : end_idx] = bases
        self._round_ids[self._count : end_idx] = round_ids

        if measurement_times is not None:
            self._times[self._count : end_idx] = measurement_times
        else:
            self._times[self._count : end_idx] = 0.0

        self._count += batch_size

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve all buffered measurements.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (outcomes, bases, round_ids, times) - copies of buffered data.
        """
        return (
            self._outcomes[: self._count].copy(),
            self._bases[: self._count].copy(),
            self._round_ids[: self._count].copy(),
            self._times[: self._count].copy(),
        )

    def clear(self) -> None:
        """Clear the buffer without deallocating memory."""
        self._count = 0

    def _grow(self) -> None:
        """Grow buffer capacity."""
        new_capacity = int(self._capacity * self._growth_factor)
        logger.debug(f"Growing buffer: {self._capacity} -> {new_capacity}")

        new_outcomes = np.zeros(new_capacity, dtype=np.uint8)
        new_bases = np.zeros(new_capacity, dtype=np.uint8)
        new_round_ids = np.zeros(new_capacity, dtype=np.int64)
        new_times = np.zeros(new_capacity, dtype=np.float64)

        new_outcomes[: self._count] = self._outcomes[: self._count]
        new_bases[: self._count] = self._bases[: self._count]
        new_round_ids[: self._count] = self._round_ids[: self._count]
        new_times[: self._count] = self._times[: self._count]

        self._outcomes = new_outcomes
        self._bases = new_bases
        self._round_ids = new_round_ids
        self._times = new_times
        self._capacity = new_capacity

    @property
    def count(self) -> int:
        """Number of measurements in buffer."""
        return self._count

    @property
    def capacity(self) -> int:
        """Current buffer capacity."""
        return self._capacity

    def __len__(self) -> int:
        """Number of measurements in buffer."""
        return self._count


class MeasurementExecutor:
    """
    Executes quantum measurements using NetQASM basis specification.

    Parameters
    ----------
    buffer : Optional[MeasurementBuffer]
        Buffer to store results. Creates new if None.
    """

    def __init__(self, buffer: Optional[MeasurementBuffer] = None) -> None:
        """
        Initialize measurement executor.

        Parameters
        ----------
        buffer : Optional[MeasurementBuffer]
            Optional measurement buffer.
        """
        self._buffer = buffer or MeasurementBuffer()
        self._measurement_count = 0

    def measure_qubit(
        self,
        qubit: "Qubit",
        basis: int,
        round_id: int,
        context=None,
    ):
        """
        Measure a qubit in the specified basis.

        This is a generator function for use in SquidASM programs.

        Parameters
        ----------
        qubit : Qubit
            NetQASM qubit reference.
        basis : int
            Measurement basis (0=Z, 1=X).
        round_id : int
            Round identifier.
        context
            Optional SquidASM context.

        Yields
        ------
        int
            Measurement outcome (0 or 1).
        """
        try:
            from netqasm.sdk.classical_communication.message import StructuredMessage
            from netqasm.sdk.qubit import QubitMeasureBasis

            # Select measurement basis
            measure_basis = (
                QubitMeasureBasis.Z if basis == BASIS_Z else QubitMeasureBasis.X
            )

            # Perform measurement
            result = qubit.measure(basis=measure_basis)

            # Flush to get result
            if context is not None:
                yield from context.connection.flush()

            outcome = int(result)
        except ImportError:
            # Fallback for testing without NetQASM
            outcome = np.random.randint(0, 2)

        self._measurement_count += 1
        measurement_time = 0.0

        self._buffer.add_outcome(
            outcome=outcome,
            basis=basis,
            round_id=round_id,
            measurement_time=measurement_time,
        )

        return outcome

    def measure_qubit_sync(
        self,
        basis: int,
        round_id: int,
        simulated_outcome: Optional[int] = None,
    ) -> int:
        """
        Simulate measurement without quantum hardware.

        Parameters
        ----------
        basis : int
            Measurement basis.
        round_id : int
            Round identifier.
        simulated_outcome : Optional[int]
            Predetermined outcome (for testing).

        Returns
        -------
        int
            Measurement outcome (0 or 1).
        """
        if simulated_outcome is not None:
            outcome = simulated_outcome
        else:
            outcome = np.random.randint(0, 2)

        self._measurement_count += 1

        self._buffer.add_outcome(
            outcome=outcome,
            basis=basis,
            round_id=round_id,
            measurement_time=0.0,
        )

        return outcome

    def measure_batch_sync(
        self,
        bases: np.ndarray,
        round_ids: np.ndarray,
        simulated_outcomes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate batch measurement without quantum hardware.

        Parameters
        ----------
        bases : np.ndarray
            Array of measurement bases.
        round_ids : np.ndarray
            Array of round identifiers.
        simulated_outcomes : Optional[np.ndarray]
            Predetermined outcomes (for testing).

        Returns
        -------
        np.ndarray
            Array of measurement outcomes.
        """
        n = len(bases)

        if simulated_outcomes is not None:
            outcomes = simulated_outcomes
        else:
            outcomes = np.random.randint(0, 2, size=n, dtype=np.uint8)

        self._buffer.add_batch(
            outcomes=outcomes,
            bases=bases,
            round_ids=round_ids,
        )

        self._measurement_count += n
        return outcomes

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all measurement results.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (outcomes, bases, round_ids).
        """
        outcomes, bases, round_ids, _ = self._buffer.get_batch()
        return outcomes, bases, round_ids

    @property
    def buffer(self) -> MeasurementBuffer:
        """Access the measurement buffer."""
        return self._buffer

    @property
    def measurement_count(self) -> int:
        """Total measurements performed."""
        return self._measurement_count

    def clear(self) -> None:
        """Clear buffer and reset count."""
        self._buffer.clear()
        self._measurement_count = 0

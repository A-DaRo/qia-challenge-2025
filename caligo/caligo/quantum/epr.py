"""
EPR pair generation using SquidASM's EPRSocket.

This module provides the interface for generating entangled pairs
in SquidASM simulations, abstracting the underlying NetQASM operations.

References
----------
- SquidASM Documentation: EPRSocket interface
- NetQASM Tutorial: Entanglement generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List

import numpy as np

from caligo.utils.logging import get_logger
from caligo.types.exceptions import EPRGenerationError

if TYPE_CHECKING:
    from squidasm.sim.stack.program import ProgramContext

logger = get_logger(__name__)


@dataclass
class EPRGenerationConfig:
    """
    Configuration for EPR pair generation.

    Parameters
    ----------
    pairs_per_batch : int
        Number of EPR pairs per generation batch.
    timeout_ns : float
        Timeout for EPR generation in nanoseconds.
    fidelity_threshold : float
        Minimum acceptable fidelity (0-1).
    retry_attempts : int
        Number of retry attempts on failure.
    """

    pairs_per_batch: int = 100
    timeout_ns: float = 1e9  # 1 second
    fidelity_threshold: float = 0.9
    retry_attempts: int = 3


@dataclass
class EPRBatch:
    """
    Result of EPR pair generation.

    Parameters
    ----------
    qubit_refs : List
        References to local qubits from EPR pairs.
    generation_time : float
        Simulation time when generation completed.
    num_pairs : int
        Number of pairs successfully generated.
    batch_id : int
        Unique identifier for this batch.
    """

    qubit_refs: List
    generation_time: float
    num_pairs: int
    batch_id: int = 0


class EPRGenerator:
    """
    Generates EPR pairs using SquidASM's EPRSocket.

    This class wraps the SquidASM EPRSocket for entanglement generation,
    providing a clean interface for the E-HOK protocol's quantum phase.

    Parameters
    ----------
    config : EPRGenerationConfig
        Generation configuration.

    Notes
    -----
    Usage in SquidASM program:

    ```python
    generator = EPRGenerator()
    # Within program context:
    batch = yield from generator.generate_batch(
        epr_socket=epr_socket,
        num_pairs=100,
        context=context
    )
    ```

    References
    ----------
    - SquidASM EPRSocket: create_keep() for local storage
    - NetQASM: QubitMeasureBasis for measurement basis
    """

    def __init__(self, config: Optional[EPRGenerationConfig] = None) -> None:
        """
        Initialize EPR generator.

        Parameters
        ----------
        config : Optional[EPRGenerationConfig]
            Generation configuration. Uses defaults if None.
        """
        self.config = config or EPRGenerationConfig()
        self._batch_counter = 0
        self._total_generated = 0

    def generate_batch(
        self,
        epr_socket,
        num_pairs: int,
        context: Optional["ProgramContext"] = None,
    ):
        """
        Generate a batch of EPR pairs.

        This is a generator function that yields to SquidASM's event loop.
        Must be called with `yield from` in a SquidASM program.

        Parameters
        ----------
        epr_socket
            SquidASM EPRSocket for entanglement generation.
        num_pairs : int
            Number of EPR pairs to generate.
        context : Optional[ProgramContext]
            SquidASM program context (for timing).

        Yields
        ------
        EPRBatch
            Generated EPR pair batch with qubit references.

        Raises
        ------
        EPRGenerationError
            If generation fails after retry attempts.

        Notes
        -----
        The qubits returned are local qubits - the remote party has
        the entangled partner qubits from their EPRSocket.
        """
        batch_id = self._batch_counter
        self._batch_counter += 1

        logger.debug(f"Generating batch {batch_id}: {num_pairs} EPR pairs")

        for attempt in range(self.config.retry_attempts):
            try:
                # Use EPRSocket's create_keep for local storage
                qubit_refs = epr_socket.create_keep(number=num_pairs)
                yield from context.connection.flush() if context else []

                generation_time = 0.0
                if context is not None:
                    try:
                        generation_time = float(
                            context.csocket.msg_from_peer()
                            if hasattr(context, "csocket")
                            else 0.0
                        )
                    except Exception:
                        generation_time = 0.0

                self._total_generated += num_pairs
                logger.debug(
                    f"Batch {batch_id}: Generated {num_pairs} pairs "
                    f"(total: {self._total_generated})"
                )

                return EPRBatch(
                    qubit_refs=list(qubit_refs)
                    if hasattr(qubit_refs, "__iter__")
                    else [qubit_refs],
                    generation_time=generation_time,
                    num_pairs=num_pairs,
                    batch_id=batch_id,
                )

            except Exception as e:
                logger.warning(
                    f"Batch {batch_id} attempt {attempt + 1} failed: {e}"
                )
                if attempt == self.config.retry_attempts - 1:
                    raise EPRGenerationError(
                        f"EPR generation failed after {self.config.retry_attempts} "
                        f"attempts: {e}"
                    ) from e

        # Should not reach here
        raise EPRGenerationError("EPR generation failed unexpectedly")

    def generate_batch_sync(self, num_pairs: int) -> EPRBatch:
        """
        Generate simulated EPR pairs without SquidASM context.

        This method is for unit testing and non-simulation scenarios.
        It creates placeholder qubit references.

        Parameters
        ----------
        num_pairs : int
            Number of pairs to simulate.

        Returns
        -------
        EPRBatch
            Batch with simulated qubit references.
        """
        batch_id = self._batch_counter
        self._batch_counter += 1

        self._total_generated += num_pairs

        # Create placeholder references
        qubit_refs = list(range(num_pairs))

        logger.debug(f"Sync batch {batch_id}: {num_pairs} simulated pairs")

        return EPRBatch(
            qubit_refs=qubit_refs,
            generation_time=0.0,
            num_pairs=num_pairs,
            batch_id=batch_id,
        )

    @property
    def total_generated(self) -> int:
        """Total EPR pairs generated across all batches."""
        return self._total_generated

    def reset_counters(self) -> None:
        """Reset batch and total counters."""
        self._batch_counter = 0
        self._total_generated = 0

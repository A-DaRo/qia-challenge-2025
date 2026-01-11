"""
SquidASM Program classes for quantum protocol execution.

This module provides Program implementations that run within the SquidASM
simulation framework, enabling proper NetSquid-based quantum operations
with realistic noise models.

The programs defined here are minimal building blocks for QKD protocols,
designed to be composed and extended as needed.

References
----------
- SquidASM Program interface: squidasm.sim.stack.program.Program
- NetQASM EPRSocket: netqasm.sdk.epr_socket.EPRSocket
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List

if TYPE_CHECKING:
    from squidasm.sim.stack.program import ProgramContext, ProgramMeta


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class EPRMeasurementConfig:
    """
    Configuration for EPR pair generation and measurement.

    Parameters
    ----------
    peer_name : str
        Name of the peer node for EPR generation.
    num_pairs : int
        Number of EPR pairs to generate and measure.
    is_creator : bool
        True if this node initiates EPR creation (create_keep),
        False if receiving (recv_keep).
    random_seed : int, optional
        Seed for basis selection. If None, uses system entropy.

    Attributes
    ----------
    peer_name : str
    num_pairs : int
    is_creator : bool
    random_seed : int | None
    """

    peer_name: str
    num_pairs: int
    is_creator: bool
    random_seed: int | None = None


# =============================================================================
# EPR Generation Program
# =============================================================================


class EPRGeneratorProgram:
    """
    Minimal SquidASM Program for EPR pair generation and measurement.

    This program generates EPR pairs using the EPRSocket interface,
    measures each qubit in a randomly selected basis (Z or X), and
    returns the outcomes and bases used.

    The program is designed to be stateless and composable, suitable
    for parallel execution across multiple processes.

    Parameters
    ----------
    config : EPRMeasurementConfig
        Configuration specifying peer, num_pairs, role, and seed.

    Attributes
    ----------
    config : EPRMeasurementConfig
        The configuration used for this program instance.

    Notes
    -----
    The measurement basis encoding:
    - 0: Z basis (computational basis {|0⟩, |1⟩})
    - 1: X basis (Hadamard basis {|+⟩, |-⟩})

    The measurement outcome encoding:
    - 0: First eigenstate (|0⟩ for Z, |+⟩ for X)
    - 1: Second eigenstate (|1⟩ for Z, |-⟩ for X)

    Examples
    --------
    >>> config = EPRMeasurementConfig(
    ...     peer_name="Bob",
    ...     num_pairs=100,
    ...     is_creator=True,
    ...     random_seed=42,
    ... )
    >>> program = EPRGeneratorProgram(config)
    >>> # Run via squidasm.run.stack.run.run()
    """

    def __init__(self, config: EPRMeasurementConfig) -> None:
        """Initialize with measurement configuration."""
        self._config = config
        self._rng = random.Random(config.random_seed)

    @property
    def config(self) -> EPRMeasurementConfig:
        """Get the program configuration."""
        return self._config

    @property
    def meta(self) -> "ProgramMeta":
        """
        Request program meta information.

        Returns
        -------
        ProgramMeta
            Metadata specifying sockets and qubit requirements.
        """
        # Late import to avoid circular dependencies and allow
        # this module to be imported without SquidASM
        from squidasm.sim.stack.program import ProgramMeta

        return ProgramMeta(
            name=f"epr_generator_{self._config.peer_name}",
            csockets=[self._config.peer_name],
            epr_sockets=[self._config.peer_name],
            max_qubits=min(self._config.num_pairs, 100),  # Limit per batch
        )

    def run(self, context: "ProgramContext") -> Generator[Any, Any, Dict[str, Any]]:
        """
        Run the EPR generation and measurement program.

        Parameters
        ----------
        context : ProgramContext
            SquidASM context providing connection and sockets.

        Yields
        ------
        Generator
            NetQASM flush operations (handled by SquidASM runtime).

        Returns
        -------
        Dict[str, Any]
            Results containing:
            - outcomes: List[int] - Measurement outcomes
            - bases: List[int] - Measurement bases used
            - num_pairs: int - Number of pairs processed
        """
        epr_socket = context.epr_sockets[self._config.peer_name]
        connection = context.connection

        outcomes: List[int] = []
        bases: List[int] = []

        # Process in batches to manage qubit memory
        batch_size = min(self._config.num_pairs, 50)
        remaining = self._config.num_pairs

        while remaining > 0:
            current_batch = min(batch_size, remaining)

            # Generate EPR pairs
            if self._config.is_creator:
                qubits = epr_socket.create_keep(number=current_batch)
            else:
                qubits = epr_socket.recv_keep(number=current_batch)

            # Measure each qubit in random basis
            batch_bases = [self._rng.randint(0, 1) for _ in range(current_batch)]
            batch_outcomes = []

            for qubit, basis in zip(qubits, batch_bases):
                if basis == 1:  # X basis
                    qubit.H()  # Rotate to computational basis for measurement
                result = qubit.measure()
                batch_outcomes.append(result)

            # Flush to execute quantum operations
            yield from connection.flush()

            # Collect results (futures resolved after flush)
            outcomes.extend([int(m) for m in batch_outcomes])
            bases.extend(batch_bases)

            remaining -= current_batch

        return {
            "outcomes": outcomes,
            "bases": bases,
            "num_pairs": self._config.num_pairs,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_epr_program_pair(
    alice_name: str,
    bob_name: str,
    num_pairs: int,
    seed: int | None = None,
) -> tuple["EPRGeneratorProgram", "EPRGeneratorProgram"]:
    """
    Create a matched pair of EPR generator programs for Alice and Bob.

    Parameters
    ----------
    alice_name : str
        Name of Alice's node.
    bob_name : str
        Name of Bob's node.
    num_pairs : int
        Number of EPR pairs to generate.
    seed : int, optional
        Random seed for basis selection. Alice and Bob get different
        derived seeds to ensure independent basis choices.

    Returns
    -------
    tuple[EPRGeneratorProgram, EPRGeneratorProgram]
        (alice_program, bob_program) pair ready for simulation.

    Examples
    --------
    >>> alice_prog, bob_prog = create_epr_program_pair(
    ...     alice_name="Alice",
    ...     bob_name="Bob",
    ...     num_pairs=1000,
    ...     seed=42,
    ... )
    >>> results = run(
    ...     config=network_config,
    ...     programs={"Alice": alice_prog, "Bob": bob_prog},
    ...     num_times=1,
    ... )
    """
    # Derive different seeds for Alice and Bob to ensure independence
    alice_seed = None if seed is None else seed * 2 + 1
    bob_seed = None if seed is None else seed * 2 + 2

    alice_config = EPRMeasurementConfig(
        peer_name=bob_name,
        num_pairs=num_pairs,
        is_creator=True,
        random_seed=alice_seed,
    )

    bob_config = EPRMeasurementConfig(
        peer_name=alice_name,
        num_pairs=num_pairs,
        is_creator=False,
        random_seed=bob_seed,
    )

    return EPRGeneratorProgram(alice_config), EPRGeneratorProgram(bob_config)

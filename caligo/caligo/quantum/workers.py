"""
Worker programs and functions for parallel EPR generation.

This module provides the worker-side implementations for parallel EPR
generation, including minimal NetQASM programs optimized for isolated
execution in worker processes.

The key design principle is that workers must be fully self-contained:
- No shared state with main process (except serialized config)
- Fresh NetSquid simulator instance per worker
- Independent RNG state for each batch

Architecture
------------
Worker processes execute `_worker_generate_epr()`, which:
1. Resets NetSquid simulator state
2. Constructs minimal network topology
3. Runs Alice/Bob programs for EPR generation
4. Returns serializable results

References
----------
- NetQASM Tutorial: Minimal program structure
- SquidASM EPRSocket: create_keep() / recv_keep() patterns
- Python multiprocessing: Pickling requirements
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from squidasm.sim.stack.program import Program, ProgramContext

logger = get_logger(__name__)


# =============================================================================
# Serializable Task Specification
# =============================================================================


@dataclass
class EPRWorkerTask:
    """
    Task specification for a single worker batch.

    This dataclass is fully serializable (picklable) for safe transfer
    to worker processes via multiprocessing.

    Parameters
    ----------
    batch_id : int
        Unique identifier for this batch.
    num_pairs : int
        Number of EPR pairs to generate.
    start_index : int
        Global index of first pair in this batch (for ordering).
    noise_config : Dict[str, float]
        Noise model parameters (serialized from ChannelParameters).
    rng_seed : Optional[int]
        Seed for worker's random number generator.
        If None, derived from batch_id and system time.

    Attributes
    ----------
    batch_id : int
    num_pairs : int
    start_index : int
    noise_config : Dict[str, float]
    rng_seed : Optional[int]

    Examples
    --------
    >>> task = EPRWorkerTask(
    ...     batch_id=0,
    ...     num_pairs=1000,
    ...     start_index=0,
    ...     noise_config={"depolarize_rate": 0.05},
    ... )
    """

    batch_id: int
    num_pairs: int
    start_index: int
    noise_config: Dict[str, float] = field(default_factory=dict)
    rng_seed: Optional[int] = None


# =============================================================================
# Minimal Worker Programs (NetQASM-based)
# =============================================================================


class MinimalAliceWorkerProgram:
    """
    Lightweight Alice program for worker processes.

    This program strips unnecessary overhead from the full protocol
    implementation, focusing solely on EPR generation and measurement.
    It is designed for use in parallel worker processes where only
    the quantum phase is executed.

    Unlike the full `AliceProgram`, this class:
    - Does not implement sifting, reconciliation, or amplification
    - Does not use OrderedSocket for classical communication
    - Does not enforce timing barriers (handled by main process)
    - Stores results in memory for batch return

    Parameters
    ----------
    num_pairs : int
        Number of EPR pairs to generate.
    basis_seed : Optional[int]
        Seed for basis selection RNG. If None, uses system random.

    Attributes
    ----------
    _num_pairs : int
        Number of pairs to generate.
    _outcomes : List[int]
        Measurement outcomes (populated after run()).
    _bases : List[int]
        Measurement bases (populated after run()).

    Examples
    --------
    >>> program = MinimalAliceWorkerProgram(num_pairs=100)
    >>> # In SquidASM context:
    >>> results = yield from program.run(context)
    >>> program.get_results()
    {"outcomes": [...], "bases": [...]}

    Notes
    -----
    This class follows SquidASM's Program interface but is simplified
    for worker context. It can be used with `run_programs()` in a
    minimal network configuration.
    """

    PEER_NAME = "Bob"

    def __init__(
        self,
        num_pairs: int,
        basis_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize minimal Alice worker program.

        Parameters
        ----------
        num_pairs : int
            Number of EPR pairs to generate.
        basis_seed : Optional[int]
            Seed for basis selection RNG.
        """
        self._num_pairs = num_pairs
        self._basis_seed = basis_seed
        self._outcomes: List[int] = []
        self._bases: List[int] = []
        self._rng = random.Random(basis_seed)

    @property
    def meta(self) -> Any:
        """
        Program metadata for SquidASM.

        Returns
        -------
        ProgramMeta
            Metadata specifying EPR sockets and qubit requirements.
        """
        try:
            from squidasm.sim.stack.program import ProgramMeta
            return ProgramMeta(
                name="minimal_alice_worker",
                csockets=[self.PEER_NAME],
                epr_sockets=[self.PEER_NAME],
                max_qubits=2,  # Minimal qubit requirement
            )
        except ImportError:
            # Return dict for standalone testing
            return {
                "name": "minimal_alice_worker",
                "csockets": [self.PEER_NAME],
                "epr_sockets": [self.PEER_NAME],
                "max_qubits": 2,
            }

    def run(self, context: Any) -> Generator[Any, None, Dict[str, Any]]:
        """
        Execute EPR generation loop.

        This generator function yields to SquidASM's event loop for
        each EPR pair creation and measurement operation.

        Parameters
        ----------
        context : ProgramContext
            SquidASM program execution context.

        Yields
        ------
        EventExpression
            NetQASM events for EPR operations.

        Returns
        -------
        Dict[str, Any]
            Results dictionary with "outcomes" and "bases" lists.
        """
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        self._outcomes = []
        self._bases = []

        for pair_idx in range(self._num_pairs):
            # Create EPR pair (Alice initiates)
            qubit = epr_socket.create_keep(1)[0]
            yield from connection.flush()

            # Random basis choice (0=Z, 1=X)
            basis = self._rng.randint(0, 1)
            self._bases.append(basis)

            # Apply Hadamard for X-basis measurement
            if basis == 1:
                qubit.H()
            yield from connection.flush()

            # Measure qubit
            result = qubit.measure()
            yield from connection.flush()

            self._outcomes.append(int(result))

        return self.get_results()

    def get_results(self) -> Dict[str, List[int]]:
        """
        Return measurement outcomes and bases.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary with keys:
            - "outcomes": List of measurement results (0 or 1)
            - "bases": List of measurement bases (0=Z, 1=X)
        """
        return {
            "outcomes": self._outcomes.copy(),
            "bases": self._bases.copy(),
        }


class MinimalBobWorkerProgram:
    """
    Lightweight Bob program for worker processes.

    Mirrors `MinimalAliceWorkerProgram` but uses `recv_keep()` instead
    of `create_keep()` for EPR socket operations.

    Parameters
    ----------
    num_pairs : int
        Number of EPR pairs to receive.
    basis_seed : Optional[int]
        Seed for basis selection RNG. If None, uses system random.

    Attributes
    ----------
    _num_pairs : int
        Number of pairs to receive.
    _outcomes : List[int]
        Measurement outcomes (populated after run()).
    _bases : List[int]
        Measurement bases (populated after run()).

    Examples
    --------
    >>> program = MinimalBobWorkerProgram(num_pairs=100)
    >>> # In SquidASM context:
    >>> results = yield from program.run(context)
    >>> program.get_results()
    {"outcomes": [...], "bases": [...]}
    """

    PEER_NAME = "Alice"

    def __init__(
        self,
        num_pairs: int,
        basis_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize minimal Bob worker program.

        Parameters
        ----------
        num_pairs : int
            Number of EPR pairs to receive.
        basis_seed : Optional[int]
            Seed for basis selection RNG.
        """
        self._num_pairs = num_pairs
        self._basis_seed = basis_seed
        self._outcomes: List[int] = []
        self._bases: List[int] = []
        self._rng = random.Random(basis_seed)

    @property
    def meta(self) -> Any:
        """
        Program metadata for SquidASM.

        Returns
        -------
        ProgramMeta
            Metadata specifying EPR sockets and qubit requirements.
        """
        try:
            from squidasm.sim.stack.program import ProgramMeta
            return ProgramMeta(
                name="minimal_bob_worker",
                csockets=[self.PEER_NAME],
                epr_sockets=[self.PEER_NAME],
                max_qubits=2,
            )
        except ImportError:
            return {
                "name": "minimal_bob_worker",
                "csockets": [self.PEER_NAME],
                "epr_sockets": [self.PEER_NAME],
                "max_qubits": 2,
            }

    def run(self, context: Any) -> Generator[Any, None, Dict[str, Any]]:
        """
        Execute EPR reception loop.

        Parameters
        ----------
        context : ProgramContext
            SquidASM program execution context.

        Yields
        ------
        EventExpression
            NetQASM events for EPR operations.

        Returns
        -------
        Dict[str, Any]
            Results dictionary with "outcomes" and "bases" lists.
        """
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        self._outcomes = []
        self._bases = []

        for pair_idx in range(self._num_pairs):
            # Receive EPR pair (Bob receives)
            qubit = epr_socket.recv_keep(1)[0]
            yield from connection.flush()

            # Random basis choice (0=Z, 1=X)
            basis = self._rng.randint(0, 1)
            self._bases.append(basis)

            # Apply Hadamard for X-basis measurement
            if basis == 1:
                qubit.H()
            yield from connection.flush()

            # Measure qubit
            result = qubit.measure()
            yield from connection.flush()

            self._outcomes.append(int(result))

        return self.get_results()

    def get_results(self) -> Dict[str, List[int]]:
        """
        Return measurement outcomes and bases.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary with keys:
            - "outcomes": List of measurement results (0 or 1)
            - "bases": List of measurement bases (0=Z, 1=X)
        """
        return {
            "outcomes": self._outcomes.copy(),
            "bases": self._bases.copy(),
        }


# =============================================================================
# Standalone Worker Function (No SquidASM Dependency)
# =============================================================================


def generate_epr_batch_standalone(
    num_pairs: int,
    noise_rate: float = 0.0,
    alice_seed: Optional[int] = None,
    bob_seed: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Generate EPR batch without SquidASM (standalone mode).

    This function provides a fast, standalone implementation of EPR
    generation for cases where full NetSquid simulation is not required.
    It uses a simplified noise model that produces statistically
    equivalent results.

    Parameters
    ----------
    num_pairs : int
        Number of EPR pairs to generate.
    noise_rate : float, optional
        Depolarizing noise rate (0.0 = ideal, 1.0 = fully depolarized).
        Default is 0.0.
    alice_seed : Optional[int]
        Seed for Alice's RNG. If None, uses system random.
    bob_seed : Optional[int]
        Seed for Bob's RNG. If None, uses system random.

    Returns
    -------
    alice_outcomes : List[int]
        Alice's measurement outcomes (0 or 1).
    alice_bases : List[int]
        Alice's measurement bases (0=Z, 1=X).
    bob_outcomes : List[int]
        Bob's measurement outcomes (0 or 1).
    bob_bases : List[int]
        Bob's measurement bases (0=Z, 1=X).

    Notes
    -----
    The noise model assumes depolarizing channel:
    - With probability (1-noise_rate): Perfect Bell state correlation
    - With probability noise_rate: Maximally mixed state

    When bases match:
    - Ideal: Perfect correlation (Alice outcome = Bob outcome)
    - Noisy: Correlation broken with probability noise_rate

    When bases differ:
    - Always: Random, uncorrelated outcomes (standard BB84)

    Examples
    --------
    >>> alice_out, alice_bases, bob_out, bob_bases = \\
    ...     generate_epr_batch_standalone(1000, noise_rate=0.05)
    >>> len(alice_out)
    1000
    >>> # Check QBER on matching bases
    >>> matching = [i for i in range(1000) if alice_bases[i] == bob_bases[i]]
    >>> errors = sum(alice_out[i] != bob_out[i] for i in matching)
    >>> qber = errors / len(matching)
    >>> 0.03 < qber < 0.07  # ~5% expected
    True
    """
    alice_rng = random.Random(alice_seed)
    bob_rng = random.Random(bob_seed)
    noise_rng = random.Random()  # Separate RNG for noise

    alice_outcomes: List[int] = []
    alice_bases: List[int] = []
    bob_outcomes: List[int] = []
    bob_bases: List[int] = []

    for _ in range(num_pairs):
        # Independent basis selection
        alice_basis = alice_rng.randint(0, 1)
        bob_basis = bob_rng.randint(0, 1)

        # Generate outcomes based on BB84 correlations
        if alice_basis == bob_basis:
            # Same basis: correlated outcomes (up to noise)
            alice_outcome = alice_rng.randint(0, 1)
            if noise_rng.random() < noise_rate:
                # Noise breaks correlation
                bob_outcome = bob_rng.randint(0, 1)
            else:
                # Perfect correlation
                bob_outcome = alice_outcome
        else:
            # Different bases: no correlation
            alice_outcome = alice_rng.randint(0, 1)
            bob_outcome = bob_rng.randint(0, 1)

        alice_outcomes.append(alice_outcome)
        alice_bases.append(alice_basis)
        bob_outcomes.append(bob_outcome)
        bob_bases.append(bob_basis)

    return alice_outcomes, alice_bases, bob_outcomes, bob_bases


# =============================================================================
# SquidASM-Based Worker Function
# =============================================================================


def run_epr_worker_squidasm(
    task: EPRWorkerTask,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Run EPR generation worker using full SquidASM simulation.

    This function creates an isolated SquidASM simulation environment
    and runs minimal Alice/Bob programs to generate EPR pairs with
    realistic quantum noise.

    Parameters
    ----------
    task : EPRWorkerTask
        Worker task specification.

    Returns
    -------
    alice_outcomes : List[int]
        Alice's measurement outcomes.
    alice_bases : List[int]
        Alice's measurement bases.
    bob_outcomes : List[int]
        Bob's measurement outcomes.
    bob_bases : List[int]
        Bob's measurement bases.

    Raises
    ------
    ImportError
        If SquidASM is not available.
    RuntimeError
        If simulation fails.

    Notes
    -----
    This function resets NetSquid state via `ns.sim_reset()` to ensure
    complete isolation from other workers and the main process.
    """
    try:
        import netsquid as ns
        from squidasm.run.stack.run import run
        from squidasm.run.stack.config import (
            StackNetworkConfig,
            StackConfig,
            LinkConfig,
        )
    except ImportError as e:
        raise ImportError(
            "SquidASM is required for full simulation mode. "
            "Use generate_epr_batch_standalone() for standalone mode."
        ) from e

    # Reset NetSquid for clean state
    ns.sim_reset()

    # Set RNG seed
    seed = task.rng_seed
    if seed is None:
        import time
        seed = int(time.time() * 1000) + task.batch_id * 12345
    random.seed(seed)
    np.random.seed(seed % (2**31))

    # Extract noise rate
    noise_rate = task.noise_config.get("depolarize_rate", 0.0)

    # Build minimal network configuration
    alice_cfg = StackConfig(
        name="Alice",
        qdevice_typ="generic",
        qdevice_cfg={"num_qubits": 2},
    )
    bob_cfg = StackConfig(
        name="Bob",
        qdevice_typ="generic",
        qdevice_cfg={"num_qubits": 2},
    )
    link_cfg = LinkConfig(
        stack1="Alice",
        stack2="Bob",
        typ="depolarise",
        cfg={"fidelity": 1.0 - noise_rate},
    )

    network_cfg = StackNetworkConfig(
        stacks=[alice_cfg, bob_cfg],
        links=[link_cfg],
    )

    # Create programs
    alice_seed = seed
    bob_seed = seed + 1000000  # Different seed for Bob
    alice_program = MinimalAliceWorkerProgram(task.num_pairs, alice_seed)
    bob_program = MinimalBobWorkerProgram(task.num_pairs, bob_seed)

    # Run simulation
    results = run(
        config=network_cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=1,
    )

    # Extract results
    alice_results = alice_program.get_results()
    bob_results = bob_program.get_results()

    return (
        alice_results["outcomes"],
        alice_results["bases"],
        bob_results["outcomes"],
        bob_results["bases"],
    )

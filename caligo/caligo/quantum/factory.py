"""
Factory pattern for EPR generation strategies.

This module implements the Strategy pattern to enable seamless switching
between sequential and parallel EPR generation modes based on runtime
configuration. The factory centralizes strategy selection logic and
ensures consistent interface across both modes.

Design Patterns
---------------
- **Strategy Pattern**: `EPRGenerationStrategy` protocol defines common
  interface; `SequentialEPRStrategy` and `ParallelEPRStrategy` implement it.
- **Factory Pattern**: `EPRGenerationFactory` encapsulates strategy creation
  based on configuration.

Usage
-----
```python
from caligo.quantum.factory import EPRGenerationFactory
from caligo.quantum.parallel import ParallelEPRConfig

config = CaligoConfig(
    num_epr_pairs=10000,
    parallel_config=ParallelEPRConfig(enabled=True, num_workers=4),
)

factory = EPRGenerationFactory(config, network_config)
strategy = factory.create_strategy()

# Strategy interface is identical regardless of implementation
results = strategy.generate(total_pairs=10000)
alice_outcomes, alice_bases, bob_outcomes, bob_bases = results
```

References
----------
- Gamma et al. (1994): Design Patterns, Strategy Pattern
- Python typing.Protocol: Structural subtyping for interfaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    runtime_checkable,
)

from caligo.quantum.parallel import ParallelEPRConfig, ParallelEPROrchestrator
from caligo.quantum.workers import generate_epr_batch_standalone
from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from caligo.simulation.physical_model import NSMParameters

logger = get_logger(__name__)


# =============================================================================
# Strategy Protocol (Interface)
# =============================================================================


@runtime_checkable
class EPRGenerationStrategy(Protocol):
    """
    Protocol (interface) for EPR generation strategies.

    This protocol defines the common interface that all EPR generation
    strategies must implement, enabling polymorphic switching between
    sequential and parallel generation without code changes.

    Methods
    -------
    generate(total_pairs: int) -> Tuple[List[int], List[int], List[int], List[int]]
        Generate EPR pairs and return measurement results.

    Notes
    -----
    This uses Python's `typing.Protocol` for structural subtyping, meaning
    any class implementing the `generate()` method with the correct signature
    is automatically considered an `EPRGenerationStrategy`.

    Examples
    --------
    >>> def use_strategy(strategy: EPRGenerationStrategy, n: int):
    ...     results = strategy.generate(n)
    ...     return results[0]  # alice_outcomes

    >>> # Both strategies work identically
    >>> seq = SequentialEPRStrategy(network_config)
    >>> par = ParallelEPRStrategy(config, network_config)
    >>> use_strategy(seq, 1000)
    >>> use_strategy(par, 1000)
    """

    def generate(
        self, total_pairs: int
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Generate EPR pairs using strategy-specific method.

        Parameters
        ----------
        total_pairs : int
            Number of EPR pairs to generate.

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
        """
        ...


# =============================================================================
# Sequential Strategy (Original Behavior)
# =============================================================================


class SequentialEPRStrategy:
    """
    Sequential EPR generation strategy (original implementation).

    This strategy generates EPR pairs one at a time in a single process,
    matching the original Caligo behavior. Use this when:
    - Running small simulations (< 1000 pairs)
    - Debugging or testing
    - Parallel overhead would exceed benefits
    - Strict reproducibility is required

    Parameters
    ----------
    network_config : Dict[str, Any]
        Network configuration dictionary containing noise parameters.
    use_squidasm : bool, optional
        Whether to use full SquidASM simulation. Default False uses
        standalone mode for faster execution.

    Attributes
    ----------
    _network_config : Dict[str, Any]
        Stored network configuration.
    _use_squidasm : bool
        Whether to use SquidASM simulation.

    Examples
    --------
    >>> network_config = {"noise": 0.05, "distance_km": 10}
    >>> strategy = SequentialEPRStrategy(network_config)
    >>> results = strategy.generate(1000)
    >>> len(results[0])  # alice_outcomes
    1000

    Notes
    -----
    The standalone mode uses a simplified noise model that produces
    statistically equivalent results to full SquidASM simulation but
    runs significantly faster.
    """

    def __init__(
        self,
        network_config: Dict[str, Any],
        use_squidasm: bool = False,
    ) -> None:
        """
        Initialize sequential EPR strategy.

        Parameters
        ----------
        network_config : Dict[str, Any]
            Network configuration with noise parameters.
        use_squidasm : bool, optional
            Use full SquidASM simulation. Default False.
        """
        self._network_config = network_config
        self._use_squidasm = use_squidasm
        self._logger = get_logger(__name__)

    def generate(
        self, total_pairs: int
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Generate EPR pairs sequentially.

        Parameters
        ----------
        total_pairs : int
            Number of EPR pairs to generate.

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

        Raises
        ------
        ValueError
            If total_pairs <= 0.
        """
        if total_pairs <= 0:
            raise ValueError(f"total_pairs must be > 0, got {total_pairs}")

        self._logger.debug(f"Generating {total_pairs} EPR pairs sequentially")

        # Extract noise rate from config
        noise_rate = self._network_config.get("noise", 0.0)

        if self._use_squidasm:
            # Full SquidASM simulation (slower but more realistic)
            from caligo.quantum.workers import run_epr_worker_squidasm, EPRWorkerTask

            task = EPRWorkerTask(
                batch_id=0,
                num_pairs=total_pairs,
                start_index=0,
                noise_config={"depolarize_rate": noise_rate},
            )
            return run_epr_worker_squidasm(task)
        else:
            # Standalone mode (fast, statistically equivalent)
            return generate_epr_batch_standalone(
                num_pairs=total_pairs,
                noise_rate=noise_rate,
            )


# =============================================================================
# Parallel Strategy
# =============================================================================


class ParallelEPRStrategy:
    """
    Parallel EPR generation strategy using multiprocessing.

    This strategy distributes EPR generation across multiple worker
    processes, significantly reducing wall-clock time for large
    simulations. Use this when:
    - Running large simulations (> 10000 pairs)
    - Wall-clock time is critical
    - System has multiple CPU cores available

    Parameters
    ----------
    config : ParallelEPRConfig
        Parallel execution configuration.
    network_config : Dict[str, Any]
        Network configuration dictionary.

    Attributes
    ----------
    _orchestrator : ParallelEPROrchestrator
        The underlying parallel orchestrator.

    Examples
    --------
    >>> config = ParallelEPRConfig(enabled=True, num_workers=4)
    >>> network_config = {"noise": 0.05}
    >>> strategy = ParallelEPRStrategy(config, network_config)
    >>> results = strategy.generate(50000)
    >>> len(results[0])
    50000
    >>> strategy.shutdown()  # Clean up workers

    Notes
    -----
    The strategy should be explicitly shut down via `shutdown()` to
    release worker resources, or used within a context manager.

    **Performance Tip**: For optimal performance, set `pairs_per_batch`
    to approximately `total_pairs / num_workers`.
    """

    def __init__(
        self,
        config: ParallelEPRConfig,
        network_config: Dict[str, Any],
    ) -> None:
        """
        Initialize parallel EPR strategy.

        Parameters
        ----------
        config : ParallelEPRConfig
            Parallel execution configuration.
        network_config : Dict[str, Any]
            Network configuration dictionary.
        """
        self._config = config
        self._network_config = network_config
        self._orchestrator = ParallelEPROrchestrator(config, network_config)
        self._logger = get_logger(__name__)

    def generate(
        self, total_pairs: int
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Generate EPR pairs in parallel using worker pool.

        Parameters
        ----------
        total_pairs : int
            Number of EPR pairs to generate.

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

        Raises
        ------
        SimulationError
            If worker processes fail.
        """
        self._logger.info(
            f"Generating {total_pairs} EPR pairs in parallel "
            f"({self._config.num_workers} workers)"
        )
        return self._orchestrator.generate_parallel(total_pairs)

    def shutdown(self) -> None:
        """Shutdown worker pool and release resources."""
        self._orchestrator.shutdown()

    def __enter__(self) -> "ParallelEPRStrategy":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class CaligoConfig:
    """
    Master configuration for Caligo QKD protocol.

    This dataclass aggregates all configuration for a Caligo protocol run,
    including EPR generation, network parameters, and security settings.

    Parameters
    ----------
    num_epr_pairs : int
        Total number of EPR pairs to generate.
    parallel_config : ParallelEPRConfig
        Parallel generation settings. Set `enabled=True` to use parallel mode.
    network_config : Dict[str, Any]
        Quantum network parameters (noise, distance, etc.).
    security_epsilon : float
        Security parameter ε for finite-key analysis.

    Attributes
    ----------
    num_epr_pairs : int
    parallel_config : ParallelEPRConfig
    network_config : Dict[str, Any]
    security_epsilon : float

    Examples
    --------
    >>> # Sequential mode (default)
    >>> config = CaligoConfig(num_epr_pairs=10000)
    >>> config.parallel_config.enabled
    False

    >>> # Parallel mode
    >>> config = CaligoConfig(
    ...     num_epr_pairs=100000,
    ...     parallel_config=ParallelEPRConfig(enabled=True, num_workers=8),
    ... )
    >>> config.parallel_config.enabled
    True
    """

    num_epr_pairs: int
    parallel_config: ParallelEPRConfig = field(default_factory=ParallelEPRConfig)
    network_config: Dict[str, Any] = field(default_factory=dict)
    security_epsilon: float = 1e-10


# =============================================================================
# Factory
# =============================================================================


class EPRGenerationFactory:
    """
    Factory for creating EPR generation strategies.

    This factory encapsulates the logic for selecting and instantiating
    the appropriate EPR generation strategy based on configuration.
    It provides a clean interface for strategy creation without exposing
    implementation details to callers.

    Parameters
    ----------
    config : CaligoConfig
        Global Caligo configuration.

    Attributes
    ----------
    _config : CaligoConfig
        Stored configuration.

    Examples
    --------
    >>> # Automatic strategy selection based on config
    >>> config = CaligoConfig(
    ...     num_epr_pairs=50000,
    ...     parallel_config=ParallelEPRConfig(enabled=True, num_workers=4),
    ... )
    >>> factory = EPRGenerationFactory(config)
    >>> strategy = factory.create_strategy()
    >>> isinstance(strategy, ParallelEPRStrategy)
    True

    >>> # Sequential mode
    >>> config = CaligoConfig(num_epr_pairs=1000)
    >>> factory = EPRGenerationFactory(config)
    >>> strategy = factory.create_strategy()
    >>> isinstance(strategy, SequentialEPRStrategy)
    True

    Notes
    -----
    The factory checks `config.parallel_config.enabled` to determine
    which strategy to create. When `enabled=False`, it always creates
    a `SequentialEPRStrategy`.

    **Extensibility**: To add new strategies, implement the
    `EPRGenerationStrategy` protocol and add a new creation branch
    to `create_strategy()`.
    """

    def __init__(self, config: CaligoConfig) -> None:
        """
        Initialize EPR generation factory.

        Parameters
        ----------
        config : CaligoConfig
            Global Caligo configuration.
        """
        self._config = config
        self._logger = get_logger(__name__)

    def create_strategy(self) -> EPRGenerationStrategy:
        """
        Create appropriate EPR generation strategy.

        Returns
        -------
        EPRGenerationStrategy
            Strategy instance based on configuration:
            - `ParallelEPRStrategy` if `parallel_config.enabled=True`
            - `SequentialEPRStrategy` otherwise

        Examples
        --------
        >>> factory = EPRGenerationFactory(config)
        >>> strategy = factory.create_strategy()
        >>> results = strategy.generate(config.num_epr_pairs)
        """
        if self._config.parallel_config.enabled:
            self._logger.info(
                f"Creating ParallelEPRStrategy with "
                f"{self._config.parallel_config.num_workers} workers"
            )
            return ParallelEPRStrategy(
                config=self._config.parallel_config,
                network_config=self._config.network_config,
            )
        else:
            self._logger.info("Creating SequentialEPRStrategy")
            return SequentialEPRStrategy(
                network_config=self._config.network_config,
            )

    def create_sequential(self) -> SequentialEPRStrategy:
        """
        Explicitly create a sequential strategy.

        Useful when you want to force sequential mode regardless of
        configuration.

        Returns
        -------
        SequentialEPRStrategy
            Sequential generation strategy.
        """
        return SequentialEPRStrategy(
            network_config=self._config.network_config,
        )

    def create_parallel(self) -> ParallelEPRStrategy:
        """
        Explicitly create a parallel strategy.

        Useful when you want to force parallel mode regardless of
        configuration. Uses the parallel_config settings even if
        `enabled=False`.

        Returns
        -------
        ParallelEPRStrategy
            Parallel generation strategy.

        Notes
        -----
        Even if `parallel_config.enabled=False`, this method will
        create a parallel strategy using the other settings from
        the config.
        """
        return ParallelEPRStrategy(
            config=self._config.parallel_config,
            network_config=self._config.network_config,
        )

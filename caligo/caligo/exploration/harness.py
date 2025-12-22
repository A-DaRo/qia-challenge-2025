"""
Protocol execution harness for exploration with EPR injection.

This module provides a sandbox environment for running the Caligo protocol
with pre-computed EPR data, bypassing the expensive quantum simulation
layer while preserving full classical protocol execution.

Architecture
------------
The harness operates in two modes:

1. **Full Simulation**: Uses SquidASM for quantum + classical layers
2. **Injection Mode**: Injects pre-computed EPR data, runs classical only

Injection mode provides 10-100x speedup for exploration campaigns while
maintaining identical classical protocol behavior.

```
┌──────────────────────────────────────────────────────────┐
│                    ProtocolHarness                       │
│  ┌─────────────────────┐  ┌─────────────────────────┐    │
│  │   ExplorationSample │  │   PrecomputedEPRData    │    │
│  └──────────┬──────────┘  └───────────┬─────────────┘    │
│             │                         │                  │
│             ▼                         ▼                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Protocol Execution Layer                 │    │
│  │   (Alice + Bob with EPR bypass enabled)          │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │              ProtocolResult                      │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from caligo.exploration.epr_batcher import (
    BatchedEPROrchestrator,
    BatchedEPRConfig,
    BatchedEPRResult,
    build_nsm_parameters_from_sample,
)
from caligo.exploration.types import (
    ExplorationSample,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)
from caligo.protocol.base import PrecomputedEPRData, ProtocolParameters
from caligo.reconciliation.factory import ReconciliationConfig, ReconciliationType
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Harness Configuration
# =============================================================================


@dataclass
class HarnessConfig:
    """
    Configuration for the protocol execution harness.

    Parameters
    ----------
    use_injection : bool
        If True, use pre-computed EPR injection mode.
        If False, run full SquidASM simulation.
    timeout_seconds : float
        Timeout for single protocol execution.
    generate_epr_parallel : bool
        Whether to generate EPR data in parallel.
    epr_workers : int
        Number of workers for parallel EPR generation.
    session_id_prefix : str
        Prefix for session IDs.
    capture_metadata : bool
        Whether to capture detailed execution metadata.

    Attributes
    ----------
    use_injection : bool
    timeout_seconds : float
    generate_epr_parallel : bool
    epr_workers : int
    session_id_prefix : str
    capture_metadata : bool
    """

    use_injection: bool = True
    timeout_seconds: float = 300.0
    generate_epr_parallel: bool = False
    epr_workers: int = 4
    session_id_prefix: str = "exploration"
    capture_metadata: bool = True


# =============================================================================
# Result Classification
# =============================================================================


def classify_outcome(
    result_dict: Dict[str, Any],
    error: Optional[Exception] = None,
) -> ProtocolOutcome:
    """
    Classify protocol execution outcome.

    Parameters
    ----------
    result_dict : Dict[str, Any]
        Raw protocol result dictionary.
    error : Optional[Exception]
        Exception if execution failed.

    Returns
    -------
    ProtocolOutcome
        Classified outcome.
    """
    if error is not None:
        error_str = str(error).lower()
        if "timeout" in error_str:
            return ProtocolOutcome.FAILURE_TIMEOUT
        if "qber" in error_str:
            return ProtocolOutcome.FAILURE_QBER
        if "reconciliation" in error_str or "ldpc" in error_str:
            return ProtocolOutcome.FAILURE_RECONCILIATION
        if "security" in error_str:
            return ProtocolOutcome.FAILURE_SECURITY
        return ProtocolOutcome.FAILURE_ERROR

    if result_dict.get("aborted", False):
        reason = str(result_dict.get("reason", "")).lower()
        if "qber" in reason:
            return ProtocolOutcome.FAILURE_QBER
        if "reconciliation" in reason:
            return ProtocolOutcome.FAILURE_RECONCILIATION
        if "security" in reason:
            return ProtocolOutcome.FAILURE_SECURITY
        return ProtocolOutcome.FAILURE_ERROR

    key_length = result_dict.get("key_length", 0)
    if key_length <= 0:
        return ProtocolOutcome.FAILURE_SECURITY

    return ProtocolOutcome.SUCCESS


# =============================================================================
# Protocol Harness
# =============================================================================


class ProtocolHarness:
    """
    Sandbox for executing Caligo protocol with exploration samples.

    This class provides a unified interface for running the protocol
    in either full simulation mode or injection mode (with pre-computed
    EPR data).

    Parameters
    ----------
    config : HarnessConfig
        Harness configuration.
    epr_orchestrator : Optional[BatchedEPROrchestrator]
        EPR generation orchestrator. If None, created automatically.

    Attributes
    ----------
    config : HarnessConfig
        Stored configuration.
    _epr_orchestrator : BatchedEPROrchestrator
        EPR generation orchestrator.
    _session_counter : int
        Counter for unique session IDs.

    Examples
    --------
    >>> harness = ProtocolHarness(HarnessConfig(use_injection=True))
    >>> sample = ExplorationSample(...)
    >>> epr_result = harness.generate_epr(sample)
    >>> result = harness.execute(sample, epr_data=epr_result.epr_data)
    >>> print(result.outcome)
    ProtocolOutcome.SUCCESS

    Notes
    -----
    The harness should be shut down via `shutdown()` to release resources.
    Use as a context manager for automatic cleanup:

    >>> with ProtocolHarness(config) as harness:
    ...     result = harness.execute(sample)
    """

    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        epr_orchestrator: Optional[BatchedEPROrchestrator] = None,
    ) -> None:
        """
        Initialize the protocol harness.

        Parameters
        ----------
        config : Optional[HarnessConfig]
            Configuration. Uses defaults if None.
        epr_orchestrator : Optional[BatchedEPROrchestrator]
            EPR orchestrator. Created if None.
        """
        self.config = config or HarnessConfig()

        if epr_orchestrator is not None:
            self._epr_orchestrator = epr_orchestrator
            self._owns_orchestrator = False
        else:
            self._epr_orchestrator = BatchedEPROrchestrator(
                BatchedEPRConfig(max_workers=self.config.epr_workers)
            )
            self._owns_orchestrator = True

        self._session_counter = 0

        logger.info(
            "Initialized ProtocolHarness (injection=%s, timeout=%.1fs)",
            self.config.use_injection,
            self.config.timeout_seconds,
        )

    def __enter__(self) -> "ProtocolHarness":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shut down the harness and release resources."""
        if self._owns_orchestrator:
            self._epr_orchestrator.shutdown()
        logger.debug("Shut down ProtocolHarness")

    def _next_session_id(self) -> str:
        """Generate a unique session ID."""
        self._session_counter += 1
        return f"{self.config.session_id_prefix}_{self._session_counter:06d}"

    def generate_epr(self, sample: ExplorationSample) -> BatchedEPRResult:
        """
        Generate EPR data for a sample.

        Parameters
        ----------
        sample : ExplorationSample
            Parameter configuration.

        Returns
        -------
        BatchedEPRResult
            EPR generation result.
        """
        return self._epr_orchestrator.generate_single(sample)

    def generate_epr_batch(
        self,
        samples: List[ExplorationSample],
    ) -> List[BatchedEPRResult]:
        """
        Generate EPR data for multiple samples.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Parameter configurations.

        Returns
        -------
        List[BatchedEPRResult]
            EPR generation results.
        """
        return self._epr_orchestrator.generate_batch(samples)

    def _build_protocol_params(
        self,
        sample: ExplorationSample,
        epr_data: Optional[PrecomputedEPRData] = None,
    ) -> ProtocolParameters:
        """
        Build protocol parameters from a sample.

        Parameters
        ----------
        sample : ExplorationSample
            Parameter configuration.
        epr_data : Optional[PrecomputedEPRData]
            Pre-computed EPR data (for injection mode).

        Returns
        -------
        ProtocolParameters
            Protocol parameters.
        """
        nsm_params = build_nsm_parameters_from_sample(sample)

        # Map strategy to reconciliation config
        recon_type = (
            ReconciliationType.BLIND
            if sample.strategy == ReconciliationStrategy.BLIND
            else ReconciliationType.BASELINE
        )
        # Construct reconciliation configuration using the current API
        recon_config = ReconciliationConfig(reconciliation_type=recon_type)

        return ProtocolParameters(
            session_id=self._next_session_id(),
            nsm_params=nsm_params,
            num_pairs=sample.num_pairs,
            precomputed_epr=epr_data,
            reconciliation=recon_config,
        )

    def execute(
        self,
        sample: ExplorationSample,
        epr_data: Optional[PrecomputedEPRData] = None,
        bob_choice_bit: int = 0,
    ) -> ProtocolResult:
        """
        Execute the protocol for a sample.

        Parameters
        ----------
        sample : ExplorationSample
            Parameter configuration.
        epr_data : Optional[PrecomputedEPRData]
            Pre-computed EPR data. If None and use_injection=True,
            EPR data will be generated.
        bob_choice_bit : int
            Bob's choice bit (0 or 1).

        Returns
        -------
        ProtocolResult
            Protocol execution result.

        Notes
        -----
        If `config.use_injection=True` and no EPR data is provided,
        EPR data will be generated first.
        """
        start_time = time.perf_counter()
        metadata: Dict[str, Any] = {}

        try:
            # Generate EPR data if needed
            if self.config.use_injection and epr_data is None:
                epr_result = self.generate_epr(sample)
                if not epr_result.is_success():
                    return self._make_failure_result(
                        sample=sample,
                        outcome=ProtocolOutcome.FAILURE_ERROR,
                        error_message=f"EPR generation failed: {epr_result.error}",
                        execution_time=time.perf_counter() - start_time,
                    )
                epr_data = epr_result.epr_data
                metadata["epr_generation_time"] = epr_result.generation_time_seconds

            # Build protocol parameters
            params = self._build_protocol_params(sample, epr_data)

            # Run the protocol
            from caligo.protocol.orchestrator import run_protocol

            ot_result, raw_results = run_protocol(
                params=params,
                bob_choice_bit=bob_choice_bit,
            )

            execution_time = time.perf_counter() - start_time

            # Extract metrics
            alice_result = raw_results.get("Alice", {})
            bob_result = raw_results.get("Bob", {})

            if self.config.capture_metadata:
                metadata.update({
                    "alice_rounds": alice_result.get("total_rounds", 0),
                    "bob_rounds": bob_result.get("total_rounds", 0),
                    "ldpc_iterations": alice_result.get("ldpc_iterations", 0),
                    "syndrome_exchanges": alice_result.get("syndrome_exchanges", 0),
                })

            # Determine outcome
            outcome = classify_outcome(alice_result)

            return ProtocolResult(
                sample=sample,
                outcome=outcome,
                net_efficiency=(
                    ot_result.final_key_length / sample.num_pairs
                    if ot_result.final_key_length > 0
                    else 0.0
                ),
                raw_key_length=alice_result.get("sifted_key_length", 0),
                final_key_length=ot_result.final_key_length,
                qber_measured=alice_result.get("qber", float("nan")),
                reconciliation_efficiency=alice_result.get(
                    "reconciliation_efficiency", 0.0
                ),
                leakage_bits=alice_result.get("leakage_bits", 0),
                execution_time_seconds=execution_time,
                error_message=None,
                metadata=metadata,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_message = f"{type(e).__name__}: {e}"

            if self.config.capture_metadata:
                metadata["traceback"] = traceback.format_exc()

            outcome = classify_outcome({}, error=e)

            logger.warning(
                "Protocol execution failed for sample: %s",
                error_message,
            )

            return self._make_failure_result(
                sample=sample,
                outcome=outcome,
                error_message=error_message,
                execution_time=execution_time,
                metadata=metadata,
            )

    def execute_batch(
        self,
        samples: List[ExplorationSample],
        epr_data_list: Optional[List[PrecomputedEPRData]] = None,
    ) -> List[ProtocolResult]:
        """
        Execute the protocol for multiple samples.

        Parameters
        ----------
        samples : List[ExplorationSample]
            Parameter configurations.
        epr_data_list : Optional[List[PrecomputedEPRData]]
            Pre-computed EPR data for each sample.

        Returns
        -------
        List[ProtocolResult]
            Protocol execution results (same order as input).

        Notes
        -----
        If EPR data is not provided and use_injection=True, EPR data
        will be generated for all samples first.
        """
        # Generate EPR data if needed
        if self.config.use_injection and epr_data_list is None:
            epr_results = self.generate_epr_batch(samples)
            epr_data_list = [r.epr_data for r in epr_results]

        # Execute protocols
        results = []
        for i, sample in enumerate(samples):
            epr_data = epr_data_list[i] if epr_data_list else None
            result = self.execute(sample, epr_data=epr_data)
            results.append(result)

        return results

    def _make_failure_result(
        self,
        sample: ExplorationSample,
        outcome: ProtocolOutcome,
        error_message: str,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProtocolResult:
        """Create a failure result object."""
        return ProtocolResult(
            sample=sample,
            outcome=outcome,
            net_efficiency=0.0,
            raw_key_length=0,
            final_key_length=0,
            qber_measured=float("nan"),
            reconciliation_efficiency=0.0,
            leakage_bits=0,
            execution_time_seconds=execution_time,
            error_message=error_message,
            metadata=metadata or {},
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def execute_sample_with_precomputed_epr(
    sample: ExplorationSample,
    epr_data: PrecomputedEPRData,
    timeout: float = 300.0,
) -> ProtocolResult:
    """
    Execute protocol for a single sample with pre-computed EPR data.

    This is a convenience function for simple use cases. For batch
    execution, use ProtocolHarness directly.

    Parameters
    ----------
    sample : ExplorationSample
        Parameter configuration.
    epr_data : PrecomputedEPRData
        Pre-computed EPR data.
    timeout : float
        Execution timeout in seconds.

    Returns
    -------
    ProtocolResult
        Protocol execution result.
    """
    config = HarnessConfig(use_injection=True, timeout_seconds=timeout)
    with ProtocolHarness(config) as harness:
        return harness.execute(sample, epr_data=epr_data)


def execute_sample_full_simulation(
    sample: ExplorationSample,
    timeout: float = 300.0,
) -> ProtocolResult:
    """
    Execute protocol for a single sample with full simulation.

    Parameters
    ----------
    sample : ExplorationSample
        Parameter configuration.
    timeout : float
        Execution timeout in seconds.

    Returns
    -------
    ProtocolResult
        Protocol execution result.
    """
    config = HarnessConfig(use_injection=False, timeout_seconds=timeout)
    with ProtocolHarness(config) as harness:
        return harness.execute(sample)

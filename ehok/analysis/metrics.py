"""Derived metrics helpers for protocol executions."""

from __future__ import annotations

from typing import Optional

from ehok.core.config import ProtocolConfig
from ehok.core.data_structures import ProtocolResult, ExecutionMetrics


def compute_execution_metrics(
    result: ProtocolResult, config: ProtocolConfig, leakage_bits: Optional[float] = None
) -> ExecutionMetrics:
    """Lift a :class:`ProtocolResult` into :class:`ExecutionMetrics`.

    Parameters
    ----------
    result : ProtocolResult
        Base protocol result to augment.
    config : ProtocolConfig
        Configuration snapshot used during the run.
    leakage_bits : float, optional
        Optional explicit leakage estimate to attach.
    """

    metrics = ExecutionMetrics(
        oblivious_key=result.oblivious_key,
        success=result.success,
        abort_reason=result.abort_reason,
        raw_count=result.raw_count,
        sifted_count=result.sifted_count,
        test_count=result.test_count,
        final_count=result.final_count,
        qber=result.qber,
        execution_time_ms=result.execution_time_ms,
        leakage_bits=leakage_bits,
    )
    return metrics

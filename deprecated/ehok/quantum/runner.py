"""Quantum phase runner encapsulating NetSquid-specific batching logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List

import numpy as np
import netsquid as ns
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.data_structures import MeasurementRecord
from ehok.quantum.batching_manager import BatchingManager, EPRGenerator
from ehok.quantum.measurement import MeasurementBuffer
from ehok.utils.logging import get_logger

logger = get_logger("quantum.runner")


@dataclass
class QuantumPhaseResult:
    """Collected measurement data for a quantum phase."""

    outcomes: np.ndarray
    bases: np.ndarray
    timestamps: np.ndarray
    measurement_records: List[MeasurementRecord]


class QuantumPhaseRunner:
    """Run Phase I (quantum generation) independent of protocol orchestration."""

    def __init__(
        self,
        context,
        peer_name: str,
        role: str,
        config: ProtocolConfig,
    ) -> None:
        self.context = context
        self.peer_name = peer_name
        self.role = role.lower()
        self.config = config

        self.epr_socket = context.epr_sockets[self.peer_name]
        self.batching_manager = BatchingManager(
            total_pairs=self.config.quantum.total_pairs,
            batch_size=self.config.quantum.batch_size,
        )
        self.generator = EPRGenerator(self.epr_socket, self.role)
        self.buffer = MeasurementBuffer()

    @property
    def csocket(self):
        return self.context.csockets[self.peer_name]

    @property
    def connection(self):
        return self.context.connection

    def run(self) -> Generator[EventExpression, None, QuantumPhaseResult]:
        logger.info("=== PHASE 1: Quantum Generation (%s) ===", self.role)
        batch_sizes = self.batching_manager.compute_batch_sizes()

        for i, batch_size in enumerate(batch_sizes):
            logger.debug("Batch %d/%d size=%d", i + 1, len(batch_sizes), batch_size)
            sim_time = ns.sim_time()

            if self.role == "alice":
                qubits = self.generator.generate_batch_alice(batch_size, sim_time)
            else:
                qubits = self.generator.generate_batch_bob(batch_size, sim_time)

            outcome_futures, bases = self.generator.measure_batch(qubits, sim_time)
            yield from self.connection.flush()

            batch_result = self.generator.extract_batch_results(
                outcome_futures, bases, sim_time
            )
            batch_result.batch_index = i
            self.buffer.add_batch(
                batch_result.outcomes, batch_result.bases, batch_result.timestamps
            )

        logger.info("Generated %d EPR pairs", len(self.buffer))
        outcomes = self.buffer.get_outcomes()
        bases = self.buffer.get_bases()
        timestamps = np.array([r.timestamp for r in self.buffer.records])
        return QuantumPhaseResult(
            outcomes=outcomes,
            bases=bases,
            timestamps=timestamps,
            measurement_records=list(self.buffer.records),
        )

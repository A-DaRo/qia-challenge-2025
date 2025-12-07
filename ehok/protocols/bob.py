"""
Bob's protocol for E-HOK Baseline.

Phase 1: Quantum Generation
"""

import numpy as np
from typing import Generator, List, Dict, Any

from pydynaa import EventExpression
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.sim.stack.common import LogManager
from netqasm.sdk import EPRSocket
import netsquid as ns

from ..core.constants import TOTAL_EPR_PAIRS, BATCH_SIZE
from ..core.data_structures import MeasurementRecord
from ..quantum.batching_manager import BatchingManager, EPRGenerator, BatchResult
from ..quantum.measurement import MeasurementBuffer
from ..utils.classical_sockets import StructuredSocket

logger = LogManager.get_stack_logger("ehok.protocols.bob")

class BobEHOKProgram(Program):
    """
    Bob's program for E-HOK.
    """
    PEER_NAME = "alice"

    def __init__(self, total_pairs: int = TOTAL_EPR_PAIRS):
        self.total_pairs = total_pairs
        self.measurement_buffer = MeasurementBuffer()

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_ehok",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=5,
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        """
        Execute Bob's protocol.
        """
        logger.info("Bob started E-HOK protocol")
        self.context = context
        
        # Initialize components
        self.epr_socket = context.epr_sockets[self.PEER_NAME]
        self.csocket = StructuredSocket(context.csockets[self.PEER_NAME])
        
        self.batching_manager = BatchingManager(self.total_pairs, BATCH_SIZE)
        self.epr_generator = EPRGenerator(self.epr_socket, "bob")

        # Phase 1: Quantum Generation
        yield from self._phase1_quantum_generation()
        
        
        logger.info(f"Bob finished. Generated {len(self.measurement_buffer)} records.")
        
        return {
            "role": "bob",
            "measurement_records": self.measurement_buffer.records
        }

    def _phase1_quantum_generation(self) -> Generator[EventExpression, None, None]:
        """
        Execute Phase 1: Receive and measure EPR pairs in batches.
        """
        logger.info("Starting Phase 1: Quantum Generation")
        
        batch_sizes = self.batching_manager.compute_batch_sizes()
        
        for i, batch_size in enumerate(batch_sizes):
            logger.debug(f"Processing batch {i+1}/{len(batch_sizes)} (size={batch_size})")
            
            # 1. Receive EPR pairs (returns futures)
            sim_time = ns.sim_time()
            qubits = self.epr_generator.generate_batch_bob(batch_size, sim_time)
            
            # 2. Measure locally (returns futures)
            outcome_futures, bases = self.epr_generator.measure_batch(qubits, sim_time)
            
            # 3. Flush to execute on quantum hardware
            yield from self.context.connection.flush()
            
            # 4. Extract results
            batch_result = self.epr_generator.extract_batch_results(
                outcome_futures, bases, sim_time
            )
            batch_result.batch_index = i
            
            # 5. Store in measurement buffer
            self.measurement_buffer.add_batch(
                batch_result.outcomes,
                batch_result.bases,
                batch_result.timestamps
            )
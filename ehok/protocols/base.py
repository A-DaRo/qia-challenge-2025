"""Protocol orchestration base class for extensible E-HOK roles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Dict, Any

from pydynaa import EventExpression
from squidasm.sim.stack.program import Program, ProgramMeta # type: ignore

from ehok.core.config import ProtocolConfig
from ehok.core.sifting import SiftingManager
from ehok.core.data_structures import ProtocolResult, ObliviousKey
from ehok.implementations import factories
from ehok.quantum.runner import QuantumPhaseRunner, QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.base")


class EHOKRole(Program, ABC):
    """Template base class implementing the five protocol phases."""

    PEER_NAME: str
    ROLE: str

    def __init__(self, config: ProtocolConfig | None = None, total_pairs: int | None = None, **_: Any):
        self.config = config or ProtocolConfig.baseline()
        if total_pairs is not None:
            self.config.quantum.total_pairs = total_pairs
        self.sifting_manager = SiftingManager()

        # Strategy placeholders; built lazily in run()
        self.commitment_scheme = None
        self.reconciliator = None
        self.privacy_amplifier = None
        self.sampling_strategy = None
        self.noise_estimator = None

    @property
    def meta(self) -> ProgramMeta:  # type: ignore[override]
        return ProgramMeta(
            name=f"{self.ROLE}_ehok",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=self.config.quantum.max_qubits,
        )

    def _build_strategies(self) -> None:
        self.commitment_scheme = factories.build_commitment_scheme(self.config)
        self.sampling_strategy = factories.build_sampling_strategy(self.config)
        self.noise_estimator = factories.build_noise_estimator(self.config)

    def _build_privacy_amplifier(self) -> None:
        self.privacy_amplifier = factories.build_privacy_amplifier(self.config)

    def _build_reconciliator(self, parity_check_matrix=None) -> None:
        self.reconciliator = factories.build_reconciliator(self.config, parity_check_matrix)

    def _build_quantum_runner(self, context) -> QuantumPhaseRunner:
        return QuantumPhaseRunner(context, self.PEER_NAME, self.ROLE, self.config)

    def run(
        self, context
    ) -> Generator[EventExpression, None, Dict[str, Any]]:  # type: ignore[override]
        """Main template method executed by SquidASM."""
        logger.info("Starting %s role", self.ROLE)
        self.context = context
        self._build_strategies()

        quantum_runner = self._build_quantum_runner(context)
        quantum_result: QuantumPhaseResult = yield from self._phase1_quantum(
            quantum_runner
        )

        result = yield from self._execute_remaining_phases(quantum_result)
        return result

    @abstractmethod
    def _execute_remaining_phases(
        self, quantum_result: QuantumPhaseResult
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        """Roles implement classical phases using provided strategies."""
        raise NotImplementedError

    def _phase1_quantum(
        self, quantum_runner: QuantumPhaseRunner
    ) -> Generator[EventExpression, None, QuantumPhaseResult]:
        return (yield from quantum_runner.run())

    @staticmethod
    def _result_success(
        oblivious_key: ObliviousKey,
        qber: float,
        raw_count: int,
        sifted_count: int,
        test_count: int,
        final_count: int,
        role: str,
        measurement_records,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "oblivious_key": oblivious_key,
            "qber": qber,
            "raw_count": raw_count,
            "sifted_count": sifted_count,
            "test_count": test_count,
            "final_count": final_count,
            "role": role,
            "measurement_records": measurement_records,
        }

    @staticmethod
    def _result_abort(
        abort_reason: str,
        qber: float,
        raw_count: int,
        sifted_count: int,
        test_count: int,
        role: str,
        measurement_records,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "abort_reason": abort_reason,
            "oblivious_key": None,
            "qber": qber,
            "raw_count": raw_count,
            "sifted_count": sifted_count,
            "test_count": test_count,
            "final_count": 0,
            "role": role,
            "measurement_records": measurement_records,
        }

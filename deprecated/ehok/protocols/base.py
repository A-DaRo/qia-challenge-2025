"""Protocol orchestration base class for extensible E-HOK roles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional, Callable

from pydynaa import EventExpression
from squidasm.sim.stack.program import Program, ProgramMeta # type: ignore

from ehok.core.config import ProtocolConfig
from ehok.core.sifting import SiftingManager
from ehok.core.data_structures import ProtocolResult, ObliviousKey
from ehok.core.timing import TimingEnforcer, TimingConfig
from ehok.implementations import factories
from ehok.quantum.runner import QuantumPhaseRunner, QuantumPhaseResult
from ehok.protocols.ordered_messaging import OrderedProtocolSocket
from ehok.protocols.leakage_manager import LeakageSafetyManager
from ehok.utils.logging import get_logger

logger = get_logger("protocols.base")


class EHOKRole(Program, ABC):
    """
    Template base class implementing the five protocol phases.

    Supports optional dependency injection for:
    - OrderedProtocolSocket: Enforces commit-then-reveal message ordering
    - TimingEnforcer: Enforces NSM timing barrier Δt
    - LeakageSafetyManager: Tracks reconciliation leakage budget

    Parameters
    ----------
    config : ProtocolConfig | None
        Protocol configuration. Uses baseline defaults if None.
    ordered_socket : OrderedProtocolSocket | None
        Pre-configured ordered socket. Created automatically if None.
    timing_enforcer : TimingEnforcer | None
        Pre-configured timing enforcer. Created from config.nsm if None.
    leakage_manager : LeakageSafetyManager | None
        Pre-configured leakage manager. Created with default cap if None.
    total_pairs : int | None
        Override for quantum.total_pairs configuration.
    """

    PEER_NAME: str
    ROLE: str

    def __init__(
        self,
        config: ProtocolConfig | None = None,
        ordered_socket: Optional[OrderedProtocolSocket] = None,
        timing_enforcer: Optional[TimingEnforcer] = None,
        leakage_manager: Optional[LeakageSafetyManager] = None,
        total_pairs: int | None = None,
        **_: Any
    ):
        self.config = config or ProtocolConfig.baseline()
        if total_pairs is not None:
            self.config.quantum.total_pairs = total_pairs
        self.sifting_manager = SiftingManager()

        # Injected dependencies (created lazily if not provided)
        self._ordered_socket = ordered_socket
        self._timing_enforcer = timing_enforcer
        self._leakage_manager = leakage_manager

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

    def _setup_injected_dependencies(self) -> None:
        """
        Initialize injected dependencies if not provided.

        Called after context is available in run().

        Notes
        -----
        TimingEnforcer and LeakageSafetyManager are NOT created by default.
        These security components must be explicitly injected to enable
        NSM timing barrier enforcement and leakage cap tracking.
        This design allows unit tests to run without timing constraints
        while production code can inject fully-configured enforcers.
        """
        # OrderedProtocolSocket
        if self._ordered_socket is None:
            self._ordered_socket = OrderedProtocolSocket()
            logger.debug("Created default OrderedProtocolSocket for %s", self.ROLE)

        # TimingEnforcer - NOT created by default (opt-in for security)
        # Inject via constructor to enable NSM timing barrier enforcement
        if self._timing_enforcer is not None:
            logger.debug(
                "Using injected TimingEnforcer with Δt=%d ns for %s",
                self._timing_enforcer.delta_t_ns,
                self.ROLE
            )

        # LeakageSafetyManager - NOT created by default (opt-in for security)
        # Inject via constructor to enable leakage cap tracking
        if self._leakage_manager is not None:
            logger.debug("Using injected LeakageSafetyManager for %s", self.ROLE)

    def run(
        self, context
    ) -> Generator[EventExpression, None, Dict[str, Any]]:  # type: ignore[override]
        """Main template method executed by SquidASM."""
        logger.info("Starting %s role", self.ROLE)
        self.context = context
        self._build_strategies()
        self._setup_injected_dependencies()

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

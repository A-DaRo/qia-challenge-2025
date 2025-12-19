"""Protocol program base classes.

The Caligo Phase E layer is implemented as SquidASM Programs with generator-
based `run()` methods.

This module intentionally keeps configuration lightweight and aligned with the
existing Caligo primitives (quantum/sifting/reconciliation/amplification).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from caligo.connection import OrderedSocket
from caligo.simulation.timing import TimingBarrier
from caligo.simulation.physical_model import NSMParameters
from caligo.types.exceptions import SecurityError
from caligo.utils.logging import get_logger

from caligo.reconciliation.factory import ReconciliationConfig

logger = get_logger(__name__)


try:
    from pydynaa import EventExpression
    from squidasm.sim.stack.program import Program  # type: ignore
except Exception:  # pragma: no cover
    EventExpression = Any

    class Program:  # type: ignore
        pass




@dataclass(frozen=True)
class PrecomputedEPRData:
    """Precomputed EPR measurement data for both parties.

    This is an optional acceleration hook for tests and benchmarking.
    When provided via `ProtocolParameters.precomputed_epr`, Phase E programs
    will skip interacting with SquidASM EPR sockets and instead consume this
    dataset.

    Parameters
    ----------
    alice_outcomes : List[int]
        Alice measurement outcomes (0/1).
    alice_bases : List[int]
        Alice measurement bases (0=Z, 1=X).
    bob_outcomes : List[int]
        Bob measurement outcomes (0/1).
    bob_bases : List[int]
        Bob measurement bases (0=Z, 1=X).
    """

    alice_outcomes: List[int]
    alice_bases: List[int]
    bob_outcomes: List[int]
    bob_bases: List[int]


@dataclass(frozen=True)
class ProtocolParameters:
    """Parameters for a Caligo protocol run.

    Parameters
    ----------
    session_id : str
        Identifier for ordered messaging session.
    nsm_params : NSMParameters
        NSM parameters including Δt and storage noise.
    num_pairs : int
        Number of EPR pairs to generate/measure.
    num_qubits : int
        Maximum number of qubits used on the stack.
    precomputed_epr : Optional[PrecomputedEPRData]
        Optional precomputed EPR measurement data. When set, the Phase E
        quantum phase will use this dataset instead of EPR socket operations.
    reconciliation : ReconciliationConfig
        Reconciliation configuration (baseline vs blind) and LDPC parameters.
    """

    session_id: str
    nsm_params: NSMParameters
    num_pairs: int
    num_qubits: int = 10
    precomputed_epr: Optional[PrecomputedEPRData] = None
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)


class CaligoProgram(Program, ABC):
    """Template base class for Caligo protocol programs."""

    PEER: str
    ROLE: str

    def __init__(self, params: ProtocolParameters) -> None:
        self._params = params
        self._timing_barrier = TimingBarrier(delta_t_ns=params.nsm_params.delta_t_ns)
        self._ordered_socket: Optional[OrderedSocket] = None

    @property
    def params(self) -> ProtocolParameters:
        return self._params

    @property
    def meta(self) -> Any:
        from squidasm.sim.stack.program import ProgramMeta  # type: ignore[import-not-found]

        return ProgramMeta(
            name=f"caligo_{self.ROLE}",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._params.num_qubits,
        )

    def run(self, context: Any) -> Generator[Any, None, Dict[str, Any]]:
        """Run the protocol program.

        Returns a Python dictionary of results. This method is a generator in
        SquidASM.
        """

        self._ordered_socket = OrderedSocket(
            socket=context.csockets[self.PEER],
            session_id=self._params.session_id,
        )

        try:
            result = yield from self._run_protocol(context)
            return result
        except SecurityError as exc:
            logger.error("Protocol aborted (%s): %s", self.ROLE, exc)
            return {"role": self.ROLE, "aborted": True, "reason": str(exc)}

    @abstractmethod
    def _run_protocol(
        self, context: Any
    ) -> Generator[Any, None, Dict[str, Any]]:
        """Role-specific protocol implementation."""

        raise NotImplementedError

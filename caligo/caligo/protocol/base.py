"""Protocol program base classes.

The Caligo Phase E layer is implemented as SquidASM Programs with generator-
based `run()` methods.

This module intentionally keeps configuration lightweight and aligned with the
existing Caligo primitives (quantum/sifting/reconciliation/amplification).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from caligo.connection import OrderedSocket
from caligo.simulation.timing import TimingBarrier
from caligo.simulation.physical_model import NSMParameters
from caligo.types.exceptions import SecurityError
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


try:
    from pydynaa import EventExpression
    from squidasm.sim.stack.program import Program  # type: ignore
except Exception:  # pragma: no cover
    EventExpression = Any

    class Program:  # type: ignore
        pass




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
    """

    session_id: str
    nsm_params: NSMParameters
    num_pairs: int
    num_qubits: int = 10


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

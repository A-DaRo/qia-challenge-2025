"""
Classical socket wrappers for E-HOK protocol.

Provides typed, structured communication over SquidASM's ClassicalSocket.
"""

import json
import pickle
from typing import Any, Optional, Type, TypeVar, Union, Generator
from dataclasses import asdict, is_dataclass

from pydynaa import EventExpression
from ehok.utils.logging import get_logger
from squidasm.sim.stack.csocket import ClassicalSocket # type: ignore

logger = get_logger("ehok.utils.classical_sockets")

T = TypeVar("T")

class StructuredSocket:
    """
    Wrapper around ClassicalSocket for sending/receiving structured data.
    
    Handles serialization (JSON/Pickle) and type checking.
    """
    
    def __init__(self, socket: ClassicalSocket, use_pickle: bool = False):
        """
        Initialize structured socket.
        
        Parameters
        ----------
        socket : ClassicalSocket
            Underlying SquidASM classical socket.
        use_pickle : bool
            If True, use pickle for serialization (allows numpy arrays).
            If False, use JSON (safer, but limited types).
        """
        self.socket = socket
        self.use_pickle = use_pickle
    
    def send_structured(self, data: Any) -> None:
        """
        Send structured data.
        
        Parameters
        ----------
        data : Any
            Data to send. Must be serializable.
        """
        if self.use_pickle:
            payload = pickle.dumps(data).decode('latin1') # Encode to string for socket
        else:
            if is_dataclass(data):
                payload = json.dumps(asdict(data))
            else:
                payload = json.dumps(data)
        
        self.socket.send(payload)
    
    def recv_structured(self, cls: Optional[Type[T]] = None) -> Generator[EventExpression, None, Any]:
        """
        Receive structured data.
        
        Parameters
        ----------
        cls : Type[T], optional
            Expected class type. If provided, attempts to cast/validate.
        
        Returns
        -------
        data : Any
            Deserialized data.
        """
        payload = yield from self.socket.recv()
        
        if self.use_pickle:
            data = pickle.loads(payload.encode('latin1'))
        else:
            data = json.loads(payload)
            if cls is not None and is_dataclass(cls):
                data = cls(**data)
        
        return data

    def send_str(self, msg: str) -> None:
        """Send raw string."""
        self.socket.send(msg)

    def recv_str(self) -> str:
        """Receive raw string."""
        return self.socket.recv()

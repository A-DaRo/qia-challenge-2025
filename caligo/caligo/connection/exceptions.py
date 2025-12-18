"""Connection-layer exceptions.

Phase E specifies a dedicated connection exception hierarchy. Caligo already
defines these exception types in :mod:`caligo.types.exceptions` so this module
re-exports them to keep a single canonical hierarchy.
"""

from caligo.types.exceptions import (
    AckTimeoutError,
    ConnectionError,
    OrderingViolationError,
    OutOfOrderError,
    SessionMismatchError,
)

__all__ = [
    "ConnectionError",
    "OrderingViolationError",
    "AckTimeoutError",
    "SessionMismatchError",
    "OutOfOrderError",
]

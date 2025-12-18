"""Tests for EPRGenerator.generate_batch generator-mode behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pytest

from caligo.quantum.epr import EPRGenerationConfig, EPRGenerator
from caligo.types.exceptions import EPRGenerationError


def _run_generator_to_return(gen) -> Any:
    """Run a generator to completion and return its StopIteration value."""
    while True:
        try:
            next(gen)
        except StopIteration as exc:
            return exc.value


class _FlakyEPRSocket:
    def __init__(self, fail_times: int, value_factory) -> None:
        self._fail_times = fail_times
        self._calls = 0
        self._value_factory = value_factory

    @property
    def calls(self) -> int:
        return self._calls

    def create_keep(self, number: int) -> Any:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("boom")
        return self._value_factory(number)


@dataclass
class _DummyConnection:
    def flush(self) -> Iterable[None]:
        return []


@dataclass
class _DummyCSocket:
    value: float

    def msg_from_peer(self) -> float:
        return self.value


@dataclass
class _DummyContext:
    connection: _DummyConnection
    csocket: Optional[_DummyCSocket] = None


def test_generate_batch_retries_then_succeeds_without_context() -> None:
    config = EPRGenerationConfig(retry_attempts=3)
    generator = EPRGenerator(config=config)

    socket = _FlakyEPRSocket(fail_times=2, value_factory=lambda n: list(range(n)))

    batch = _run_generator_to_return(
        generator.generate_batch(epr_socket=socket, num_pairs=5, context=None)
    )

    assert socket.calls == 3
    assert batch.num_pairs == 5
    assert batch.qubit_refs == [0, 1, 2, 3, 4]
    assert generator.total_generated == 5


def test_generate_batch_raises_after_all_retries() -> None:
    config = EPRGenerationConfig(retry_attempts=2)
    generator = EPRGenerator(config=config)

    socket = _FlakyEPRSocket(fail_times=999, value_factory=lambda n: list(range(n)))

    gen = generator.generate_batch(epr_socket=socket, num_pairs=5, context=None)
    with pytest.raises(EPRGenerationError, match="failed after 2 attempts"):
        _run_generator_to_return(gen)


def test_generate_batch_uses_context_time_when_available() -> None:
    config = EPRGenerationConfig(retry_attempts=1)
    generator = EPRGenerator(config=config)

    socket = _FlakyEPRSocket(fail_times=0, value_factory=lambda n: ["q"] * n)
    context = _DummyContext(connection=_DummyConnection(), csocket=_DummyCSocket(1234.0))

    batch = _run_generator_to_return(
        generator.generate_batch(epr_socket=socket, num_pairs=3, context=context)
    )

    assert batch.generation_time == 1234.0
    assert batch.qubit_refs == ["q", "q", "q"]


def test_generate_batch_wraps_noniterable_qubit_ref() -> None:
    config = EPRGenerationConfig(retry_attempts=1)
    generator = EPRGenerator(config=config)

    socket = _FlakyEPRSocket(fail_times=0, value_factory=lambda n: 42)

    batch = _run_generator_to_return(
        generator.generate_batch(epr_socket=socket, num_pairs=1, context=None)
    )

    assert batch.qubit_refs == [42]

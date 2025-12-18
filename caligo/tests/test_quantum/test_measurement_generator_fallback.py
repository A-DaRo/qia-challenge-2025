"""Tests for MeasurementExecutor.measure_qubit generator fallback path."""

from __future__ import annotations

import builtins

import pytest

from caligo.quantum.measurement import MeasurementExecutor


def _run_generator_to_return(gen):
    while True:
        try:
            next(gen)
        except StopIteration as exc:
            return exc.value


class _DummyQubit:
    def measure(self, basis=None):  # pragma: no cover
        raise AssertionError("Should not be called in fallback path")


def test_measure_qubit_falls_back_when_netqasm_unavailable(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("netqasm"):
            raise ImportError("netqasm blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    executor = MeasurementExecutor()

    outcome = _run_generator_to_return(
        executor.measure_qubit(qubit=_DummyQubit(), basis=0, round_id=7, context=None)
    )

    assert outcome in (0, 1)
    assert executor.measurement_count == 1

    outcomes, bases, round_ids = executor.get_results()
    assert outcomes.tolist() == [outcome]
    assert bases.tolist() == [0]
    assert round_ids.tolist() == [7]

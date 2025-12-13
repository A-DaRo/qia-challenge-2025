"""
Top-level pytest configuration for E-HOK project.

This module provides project-wide pytest configuration:
- `--run-long` CLI flag for long-running tests
- `--seed` CLI flag for deterministic test execution
- Test markers registration

INFRA-004: Deterministic Test Policy
------------------------------------
Deterministic tests must be repeatable: same seed → identical outputs.

Markers:
- @pytest.mark.unit: Unit tests (no simulation required)
- @pytest.mark.integration: Integration tests (require SquidASM simulation)
- @pytest.mark.deterministic: Tests that must be reproducible
- @pytest.mark.long: Long-running tests (>2s)

Simulation Randomness Policy:
- NetSquid/SquidASM simulation can be seeded with:
    ```python
    import netsquid as ns
    ns.set_random_state(ns.util.RandomState(seed=42))
    ```
- Tests requiring simulation reproducibility must explicitly seed NetSquid
- Non-deterministic simulation tests should be marked and excluded from
  parity gating

References
----------
- sprint_0_specification.md (INFRA-004)
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add global CLI options."""
    parser.addoption(
        "--run-long",
        action="store_true",
        default=False,
        help="Run long (>2s) integration tests marked with @pytest.mark.long",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection based on markers and CLI options."""
    run_long = config.getoption("--run-long")
    if run_long:
        return
    skip_long = pytest.mark.skip(
        reason="need --run-long to run long integration tests"
    )
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)


"""
Pytest configuration for E-HOK tests.

Defines the CLI flag `--run-long` to conditionally run tests marked as `long`.
By default, long tests are skipped unless `--run-long` is specified.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    # Attempt to read the --run-long option if it exists (top-level conftest defines it).
    # If the option is not present (older pytest configs), default to skipping long tests.
    try:
        run_long = config.getoption("--run-long")
    except Exception:
        run_long = False
    if run_long:
        # Nothing to tune; user asked to run long tests
        return

    skip_long = pytest.mark.skip(reason="need --run-long to run long integration tests")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)

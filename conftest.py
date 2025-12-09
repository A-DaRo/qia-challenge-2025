"""
Top-level pytest configuration.

We add the `--run-long` CLI flag at the project root so it is available
to pytest's option parser before conftest files are processed. This allows
running tests marked with `@pytest.mark.long` via `--run-long`.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-long",
        action="store_true",
        default=False,
        help="Run long (>2s) integration tests marked with @pytest.mark.long",
    )


def pytest_collection_modifyitems(config, items):
    run_long = config.getoption("--run-long")
    if run_long:
        return
    skip_long = pytest.mark.skip(reason="need --run-long to run long integration tests")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)

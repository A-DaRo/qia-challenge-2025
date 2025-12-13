# Local CI Checks

This document describes how to run the same quality checks locally that are
enforced in the CI pipeline, as defined in [sprint_0_specification.md](../implementation%20plan/sprint_0_specification.md)
(INFRA-001).

## Prerequisites

Ensure you have the development dependencies installed:

```bash
pip install -e ".[dev]"
```

## Stage 1: Quality Gate

The CI pipeline enforces the following checks on every push. Run these locally
before committing to ensure your changes will pass CI.

### 1. Type Checking (mypy)

```bash
mypy --strict ehok/ --ignore-missing-imports
```

Expected: No errors. All code must have complete type annotations.

### 2. Linting (flake8)

```bash
# Critical errors (always enforced)
flake8 ehok/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Full lint check (informational)
flake8 ehok/ --count --max-complexity=10 --max-line-length=100 --statistics
```

Expected: Zero critical errors (E9, F63, F7, F82).

### 3. Unit Tests (pytest)

```bash
pytest -m "not integration" --tb=short -q
```

Expected: All unit tests pass. Integration tests are skipped unless explicitly requested.

## Running All Quality Checks

Run all Stage 1 checks in sequence:

```bash
# Quick validation script
echo "=== mypy ===" && mypy --strict ehok/ --ignore-missing-imports && \
echo "=== flake8 ===" && flake8 ehok/ --count --select=E9,F63,F7,F82 --show-source --statistics && \
echo "=== pytest ===" && pytest -m "not integration" --tb=short -q && \
echo "=== All checks passed! ==="
```

## Integration Tests

Integration tests require SquidASM simulation and are not run by default:

```bash
# Run integration tests (requires SquidASM)
pytest -m "integration" --tb=short

# Run all tests including long-running ones
pytest --run-long
```

## Deterministic Testing

For reproducible tests, use the `--seed` option:

```bash
pytest -m "deterministic" --seed=42
```

## Test Coverage

Generate coverage reports:

```bash
pytest --cov=ehok --cov-report=html
open htmlcov/index.html
```

## Pre-commit Hook (Recommended)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running pre-commit checks..."

# Quick type check
mypy --strict ehok/ --ignore-missing-imports || exit 1

# Critical lint errors only
flake8 ehok/ --count --select=E9,F63,F7,F82 --show-source --statistics || exit 1

# Fast unit tests
pytest -m "not integration" --tb=short -q -x || exit 1

echo "Pre-commit checks passed!"
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

## CI Environment Variables

The CI pipeline uses GitHub Secrets for NetSquid credentials:

- `NETSQUID_USER`: NetSquid PyPI username
- `NETSQUID_PASSWORD`: NetSquid PyPI password

For local development, configure pip:

```bash
pip config set global.extra-index-url "https://<user>:<pass>@pypi.netsquid.org"
```

## Troubleshooting

### "Module not found" in mypy

Ensure ehok is installed in editable mode:

```bash
pip install -e .
```

### SquidASM import errors

For unit tests, SquidASM is optional. The logging module falls back gracefully:

```python
from ehok.utils.logging import is_squidasm_available
print(f"SquidASM available: {is_squidasm_available()}")
```

### Test isolation issues

Reset logging state between tests:

```python
from ehok.utils.logging import reset_logging_state
reset_logging_state()
```

## References

- [CI Pipeline](.github/workflows/ci.yml) — GitHub Actions workflow
- [sprint_0_specification.md](../implementation%20plan/sprint_0_specification.md) — INFRA-001 requirements
- [master_roadmap.md](../implementation%20plan/master_roadmap.md) — CI/CD gate structure

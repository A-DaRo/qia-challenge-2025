"""
Logging infrastructure for the E-HOK protocol.

This module provides structured logging utilities compatible with SquidASM's
LogManager, enabling consistent logging across all E-HOK components.

All standalone runnable scripts automatically log to files named after the
module (e.g., `generate_ldpc.log`, `run_baseline.log`) and optionally to
the terminal when the `--log-show` flag is provided.

INFRA-003 Requirements (sprint_0_specification.md)
--------------------------------------------------
1. All E-HOK modules obtain loggers through a single utility surface.
2. Script logging supports:
   - Always-on file logging to a configured directory.
   - Optional terminal logging (disabled by default for CI cleanliness).
3. Logging does NOT duplicate handlers on repeated setup.
4. Logging is compatible with non-simulation unit tests.

References
----------
- sprint_0_specification.md (INFRA-003)
- master_roadmap.md (Section 2.3 CI/CD Gate Structure)
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# Thread-safe tracking of configured loggers to prevent handler duplication
_configured_loggers: set[str] = set()
_logger_lock = threading.Lock()

# Fallback flag for non-simulation environments
_squidasm_available: bool = True

try:
    from squidasm.sim.stack.common import LogManager  # type: ignore
except ImportError:
    _squidasm_available = False


def _get_base_logger(name: str) -> logging.Logger:
    """
    Get a logger, with fallback for non-simulation environments.

    Parameters
    ----------
    name : str
        Logger name (hierarchical, e.g., "ehok.quantum.measurement").

    Returns
    -------
    logging.Logger
        Logger instance.

    Notes
    -----
    When SquidASM is not available (e.g., in unit tests), falls back to
    standard logging.getLogger(). This ensures compatibility with
    non-simulation test environments per INFRA-003 requirement 4.
    """
    if _squidasm_available:
        return LogManager.get_stack_logger(name)
    else:
        return logging.getLogger(name)


def setup_script_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    show_terminal: bool = False,
) -> logging.Logger:
    """
    Configure structured logging for standalone scripts.

    This function sets up dual-output logging:
    - File output: ALWAYS enabled, named after the script module
    - Terminal output: Controlled by `show_terminal` flag (typically via `--log-show`)

    This function is idempotent: calling it multiple times with the same
    script_name will NOT create duplicate handlers.

    Parameters
    ----------
    script_name : str
        Name of the script/module (e.g., "generate_ldpc", "run_baseline").
        Used to name the log file: `<script_name>.log`.
    log_dir : Optional[Path]
        Directory for log files. If None, defaults to `./logs`.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    show_terminal : bool
        If True, logs are displayed in terminal. If False, only written to file.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance with file and optional terminal handlers.

    Notes
    -----
    Uses SquidASM's LogManager for compatibility with NetSquid logging when
    available. Falls back to standard logging for non-simulation environments.

    The file handler is ALWAYS enabled to ensure complete audit trails.
    Terminal output is optional to avoid cluttering output during batch runs.

    Examples
    --------
    >>> from pathlib import Path
    >>> logger = setup_script_logging("generate_ldpc", show_terminal=True)
    >>> logger.info("Matrix generation started")
    # Logs to both ./logs/generate_ldpc.log and terminal

    >>> logger = setup_script_logging("run_baseline", show_terminal=False)
    >>> logger.info("Protocol execution started")
    # Logs ONLY to ./logs/run_baseline.log (silent terminal)
    """
    logger_name = f"ehok.{script_name}"

    with _logger_lock:
        # Check if already configured (prevents handler duplication per INFRA-003 requirement 3)
        if logger_name in _configured_loggers:
            return _get_base_logger(logger_name)

        # Determine log directory
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        log_dir = Path(log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get hierarchical logger
        logger = _get_base_logger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove any existing handlers to avoid duplication
        logger.handlers.clear()

        # IMPORTANT: Prevent propagation to parent loggers to avoid SquidASM's default handlers
        logger.propagate = False

        # Structured format
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler: ALWAYS enabled
        log_file = log_dir / f"{script_name}.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        # Terminal handler: Controlled by show_terminal flag
        if show_terminal:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(log_format)
            logger.addHandler(console_handler)

        # Mark as configured
        _configured_loggers.add(logger_name)

        # Log initialization message (goes to file always, terminal if enabled)
        logger.info("=" * 70)
        logger.info("Logging initialized for script: %s", script_name)
        logger.info("Log file: %s", log_file)
        logger.info("Log level: %s", log_level)
        logger.info("Terminal output: %s", "ENABLED" if show_terminal else "DISABLED")
        logger.info("SquidASM available: %s", _squidasm_available)
        logger.info("=" * 70)

    return logger


def setup_ehok_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    show_terminal: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for E-HOK protocol (legacy interface).

    This function sets up a logger with both console and optional file output,
    using formatting compatible with SquidASM's logging infrastructure.

    **NOTE:** For standalone scripts, prefer `setup_script_logging()` which
    provides better control over file/terminal output separation.

    This function is idempotent: calling it multiple times will NOT create
    duplicate handlers.

    Parameters
    ----------
    log_dir : Optional[Path]
        Directory for log files. If None, logs to console only.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    show_terminal : bool
        If True, logs are displayed in terminal. Default True for backward
        compatibility.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.

    Notes
    -----
    Uses SquidASM's LogManager for compatibility with NetSquid logging when
    available. Falls back to standard logging for non-simulation environments.

    Examples
    --------
    >>> from pathlib import Path
    >>> logger = setup_ehok_logging(Path("./logs"), "DEBUG")
    >>> logger.info("Protocol started")
    2025-12-07 10:30:45 - ehok - INFO - Protocol started
    """
    logger_name = "ehok"

    with _logger_lock:
        # Check if already configured (prevents handler duplication)
        if logger_name in _configured_loggers:
            return _get_base_logger(logger_name)

        logger = _get_base_logger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove any existing handlers
        logger.handlers.clear()
        logger.propagate = False

        # Structured format
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (controlled by show_terminal)
        if show_terminal:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)

        # File handler if directory specified
        if log_dir is not None:
            log_dir = Path(log_dir).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / "ehok_protocol.log", mode="a", encoding="utf-8"
            )
            file_handler.setFormatter(console_format)
            logger.addHandler(file_handler)

        _configured_loggers.add(logger_name)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This function retrieves a logger with a hierarchical name under the "ehok"
    namespace, enabling fine-grained control over logging levels per module.

    This is the primary interface for obtaining loggers within E-HOK modules,
    satisfying INFRA-003 requirement 1.

    Parameters
    ----------
    name : str
        Module name (e.g., "quantum.measurement", "protocols.alice").

    Returns
    -------
    logger : logging.Logger
        Logger instance for the specified module.

    Notes
    -----
    The logger name will be "ehok.<name>", enabling hierarchical configuration.
    For example, setting the level for "ehok.quantum" affects all quantum
    submodules.

    This function is compatible with non-simulation environments. When SquidASM
    is unavailable, it falls back to standard Python logging.

    Examples
    --------
    >>> logger = get_logger("quantum.measurement")
    >>> logger.debug("Measured qubit in Z basis: outcome=0")
    2025-12-07 10:30:45 - ehok.quantum.measurement - DEBUG - Measured qubit in Z basis: outcome=0
    """
    return _get_base_logger(f"ehok.{name}")


def reset_logging_state() -> None:
    """
    Reset logging configuration state for testing.

    This function clears the set of configured loggers, allowing handlers to be
    reconfigured. Primarily used in test fixtures to ensure test isolation.

    **Warning:** This should only be used in test environments.
    """
    global _configured_loggers
    with _logger_lock:
        _configured_loggers = set()


def is_squidasm_available() -> bool:
    """
    Check if SquidASM is available for logging.

    Returns
    -------
    bool
        True if SquidASM's LogManager is available, False otherwise.
    """
    return _squidasm_available


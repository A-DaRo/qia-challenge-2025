"""
Structured logging with SquidASM compatibility.

This module provides a unified logging interface that works both
with SquidASM's LogManager and in standalone unit test mode.

Requirements (UTIL-LOG-*):
- UTIL-LOG-001: Single utility function for all logger access
- UTIL-LOG-002: Graceful fallback when SquidASM unavailable
- UTIL-LOG-003: Idempotent setup (no handler duplication)
- UTIL-LOG-004: Configurable file and terminal output
- UTIL-LOG-005: Thread-safe logger configuration
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

# Thread lock for configuration safety (UTIL-LOG-005)
_config_lock = threading.Lock()

# Track configured loggers to prevent handler duplication (UTIL-LOG-003)
_configured_loggers: set[str] = set()

# Default log format
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _try_import_squidasm_logmanager() -> Optional[type]:
    """
    Attempt to import SquidASM's LogManager.

    Returns
    -------
    Optional[type]
        LogManager class if available, None otherwise.
    """
    try:
        from squidasm.util.log_manager import LogManager

        return LogManager
    except ImportError:
        return None


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with SquidASM compatibility.

    Parameters
    ----------
    name : str
        Hierarchical logger name (e.g., "caligo.quantum.epr").

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Falls back to standard logging.getLogger() when SquidASM is
    not available, enabling use in non-simulation unit tests.

    Examples
    --------
    >>> logger = get_logger("caligo.sifting.commitment")
    >>> logger.info("Commitment generated successfully")
    """
    log_manager = _try_import_squidasm_logmanager()

    if log_manager is not None:
        # Use SquidASM's LogManager for simulation context
        try:
            return log_manager.get_stack_logger(name)
        except Exception:
            # Fall back if LogManager fails (e.g., not initialized)
            pass

    # Standard library fallback
    return logging.getLogger(name)


def setup_script_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    show_terminal: bool = False,
) -> logging.Logger:
    """
    Configure logging for standalone scripts.

    Sets up dual-output logging:
    - File output: ALWAYS enabled, named <script_name>.log
    - Terminal output: Controlled by show_terminal flag

    This function is idempotent — repeated calls do not duplicate handlers.

    Parameters
    ----------
    script_name : str
        Name of the script (used for log filename and logger name).
    log_dir : Optional[Path]
        Directory for log files. Defaults to current directory.
    log_level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    show_terminal : bool
        If True, also log to terminal/console.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_script_logging("my_simulation", log_level="DEBUG")
    >>> logger.info("Simulation started")
    """
    with _config_lock:
        logger_name = f"caligo.scripts.{script_name}"

        # Check if already configured (UTIL-LOG-003)
        if logger_name in _configured_loggers:
            return logging.getLogger(logger_name)

        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        # File handler (always enabled)
        if log_dir is None:
            log_dir = Path.cwd()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{script_name}.log"

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Terminal handler (optional)
        if show_terminal:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        _configured_loggers.add(logger_name)

        return logger


def reset_logging_state() -> None:
    """
    Reset logging configuration state (for testing purposes).

    Clears the set of configured loggers, allowing fresh configuration.
    Use with caution — primarily intended for test teardown.
    """
    with _config_lock:
        _configured_loggers.clear()

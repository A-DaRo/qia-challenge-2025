"""
Logging infrastructure for the E-HOK protocol.

This module provides structured logging utilities compatible with SquidASM's
LogManager, enabling consistent logging across all E-HOK components.

All standalone runnable scripts automatically log to files named after the
module (e.g., `generate_ldpc.log`, `run_baseline.log`) and optionally to
the terminal when the `--log-show` flag is provided.
"""

from squidasm.sim.stack.common import LogManager
import logging
from pathlib import Path
from typing import Optional
import sys


def setup_script_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    show_terminal: bool = False
) -> logging.Logger:
    """
    Configure structured logging for standalone scripts.
    
    This function sets up dual-output logging:
    - File output: ALWAYS enabled, named after the script module
    - Terminal output: Controlled by `show_terminal` flag (typically via `--log-show`)
    
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
    Uses SquidASM's LogManager for compatibility with NetSquid logging.
    This ensures that E-HOK logs integrate seamlessly with SquidASM's
    simulation event logs.
    
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
    # Determine log directory
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get hierarchical logger
    logger = LogManager.get_stack_logger(f"ehok.{script_name}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # IMPORTANT: Prevent propagation to parent loggers to avoid SquidASM's default handlers
    logger.propagate = False
    
    # Structured format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler: ALWAYS enabled
    log_file = log_dir / f"{script_name}.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Terminal handler: Controlled by show_terminal flag
    if show_terminal:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    # Log initialization message (goes to file always, terminal if enabled)
    logger.info("=" * 70)
    logger.info("Logging initialized for script: %s", script_name)
    logger.info("Log file: %s", log_file)
    logger.info("Log level: %s", log_level)
    logger.info("Terminal output: %s", "ENABLED" if show_terminal else "DISABLED")
    logger.info("=" * 70)
    
    return logger


def setup_ehok_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure structured logging for E-HOK protocol (legacy interface).
    
    This function sets up a logger with both console and optional file output,
    using formatting compatible with SquidASM's logging infrastructure.
    
    **NOTE:** For standalone scripts, prefer `setup_script_logging()` which
    provides better control over file/terminal output separation.
    
    Parameters
    ----------
    log_dir : Optional[Path]
        Directory for log files. If None, logs to console only.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    
    Notes
    -----
    Uses SquidASM's LogManager for compatibility with NetSquid logging.
    This ensures that E-HOK logs integrate seamlessly with SquidASM's
    simulation event logs.
    
    Examples
    --------
    >>> from pathlib import Path
    >>> logger = setup_ehok_logging(Path("./logs"), "DEBUG")
    >>> logger.info("Protocol started")
    2025-12-07 10:30:45 - ehok - INFO - Protocol started
    """
    logger = LogManager.get_stack_logger("ehok")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with structured format
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if directory specified
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "ehok_protocol.log")
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    This function retrieves a logger with a hierarchical name under the "ehok"
    namespace, enabling fine-grained control over logging levels per module.
    
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
    
    Examples
    --------
    >>> logger = get_logger("quantum.measurement")
    >>> logger.debug("Measured qubit in Z basis: outcome=0")
    2025-12-07 10:30:45 - ehok.quantum.measurement - DEBUG - Measured qubit in Z basis: outcome=0
    """
    return LogManager.get_stack_logger(f"ehok.{name}")

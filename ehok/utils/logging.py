"""
Logging infrastructure for the E-HOK protocol.

This module provides structured logging utilities compatible with SquidASM's
LogManager, enabling consistent logging across all E-HOK components.
"""

from squidasm.sim.stack.common import LogManager
import logging
from pathlib import Path
from typing import Optional


def setup_ehok_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure structured logging for E-HOK protocol.
    
    This function sets up a logger with both console and optional file output,
    using formatting compatible with SquidASM's logging infrastructure.
    
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

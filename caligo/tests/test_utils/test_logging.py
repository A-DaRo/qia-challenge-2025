"""Unit tests for caligo.utils.logging module."""

import logging
import tempfile
from pathlib import Path

import pytest

from caligo.utils.logging import (
    get_logger,
    setup_script_logging,
    reset_logging_state,
)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Test that get_logger returns a logging.Logger instance."""
        logger = get_logger("caligo.test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_preserved(self):
        """Test that logger name is preserved."""
        logger = get_logger("caligo.quantum.epr")
        assert logger.name == "caligo.quantum.epr"

    def test_hierarchical_logger_names(self):
        """Test hierarchical logger naming."""
        parent = get_logger("caligo")
        child = get_logger("caligo.test")
        assert child.name.startswith(parent.name)


class TestSetupScriptLogging:
    """Tests for setup_script_logging function."""

    def setup_method(self):
        """Reset logging state before each test."""
        reset_logging_state()

    def teardown_method(self):
        """Clean up after each test."""
        reset_logging_state()

    def test_returns_logger(self):
        """Test that setup_script_logging returns a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_script_logging("test_script", log_dir=Path(tmpdir))
            assert isinstance(logger, logging.Logger)

    def test_creates_log_file(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_script_logging("test_script", log_dir=log_dir)
            log_file = log_dir / "test_script.log"
            assert log_file.exists()

    def test_writes_to_log_file(self):
        """Test that messages are written to log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = setup_script_logging("test_script", log_dir=log_dir)
            logger.info("Test message")
            
            # Force flush
            for handler in logger.handlers:
                handler.flush()
            
            log_file = log_dir / "test_script.log"
            content = log_file.read_text()
            assert "Test message" in content

    def test_idempotent_setup(self):
        """Test that repeated calls don't duplicate handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger1 = setup_script_logging("test_script", log_dir=log_dir)
            handler_count1 = len(logger1.handlers)
            
            logger2 = setup_script_logging("test_script", log_dir=log_dir)
            handler_count2 = len(logger2.handlers)
            
            assert logger1 is logger2
            assert handler_count1 == handler_count2

    def test_log_level_configuration(self):
        """Test that log level is properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_script_logging(
                "test_debug", log_dir=Path(tmpdir), log_level="DEBUG"
            )
            assert logger.level == logging.DEBUG

    def test_terminal_output_disabled_by_default(self):
        """Test that terminal output is disabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_script_logging("test_script", log_dir=Path(tmpdir))
            # Check no StreamHandler (console handler)
            stream_handlers = [
                h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            assert len(stream_handlers) == 0

    def test_terminal_output_enabled(self):
        """Test that terminal output can be enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_script_logging(
                "test_terminal", log_dir=Path(tmpdir), show_terminal=True
            )
            stream_handlers = [
                h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            assert len(stream_handlers) == 1


class TestResetLoggingState:
    """Tests for reset_logging_state function."""

    def test_reset_allows_reconfiguration(self):
        """Test that reset allows fresh configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            logger1 = setup_script_logging("test_reset", log_dir=log_dir)
            reset_logging_state()
            
            # After reset, we should be able to configure again
            # (though logger object is reused by Python's logging module)
            logger2 = setup_script_logging(
                "test_reset", log_dir=log_dir, show_terminal=True
            )
            
            # The logger should now have the terminal handler
            stream_handlers = [
                h for h in logger2.handlers if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            # Note: Due to Python's logger caching, handlers accumulate
            # This test verifies reset_logging_state clears our tracking set
            assert len(logger2.handlers) > len(logger1.handlers) or len(stream_handlers) >= 1

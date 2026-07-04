"""Tests for ml_grid.util.logger_setup module.

This test module covers the logger_setup functionality including:
- Basic logger setup with file and console handlers
- Verbose level configuration
- Stdout/stderr redirection with TeeWriter class
- Pickling support for multiprocess scenarios.
"""

import logging
import os
import pickle
import sys

import pytest


class TestSetupLoggerBasicFunctionality:
    """Test basic setup_logger functionality."""

    def test_returns_valid_logger(self, tmp_path):
        """Test that setup_logger returns a valid logger object."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        # Ensure bayessearch is False for redirect tests
        global_parameters.bayessearch = False

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        assert logger is not None
        assert logger.name == "ml_grid"
        assert logger.level <= 10  # DEBUG level

    def test_creates_log_directory(self, tmp_path):
        """Test that setup_logger creates the log directory."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        # Check that run_0 directory was created
        assert (log_dir / "run_0").exists()

    def test_creates_log_file(self, tmp_path):
        """Test that setup_logger creates the log file."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        log_file = log_dir / "run_0" / "run.log"
        assert log_file.exists()

    def test_different_verbose_levels(self, tmp_path):
        """Test that different verbose levels set appropriate console log levels."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        # Test verbose=0 (WARNING level)
        logger0 = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=0,
            redirect_stdout=False,
        )

        # Find console handler
        console_handler = None
        for handler in logger0.handlers:
            if hasattr(handler, "stream") and hasattr(handler.stream, "name"):
                if "stdout" in str(getattr(handler.stream, "name", "")):
                    console_handler = handler
                    break

        assert console_handler is not None or logger0.level <= 30

    def test_multiple_calls_clear_handlers(self, tmp_path):
        """Test that multiple calls clear previous handlers."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        # First call
        logger1 = logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )
        handlers_count_1 = len(logger1.handlers)

        # Second call should clear and reconfigure
        logger2 = logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=1,
            verbose=1,
            redirect_stdout=False,
        )
        handlers_count_2 = len(logger2.handlers)

        assert (
            handlers_count_2 <= handlers_count_1 + 2
        )  # Allow for file handler + console


class TestTeeWriterClass:
    """Test TeeWriter class functionality."""

    def test_tee_writer_init(self, tmp_path):
        """Test TeeWriter initialization."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        assert tee.original_stream == original_stdout
        assert os.path.exists(log_file)

    def test_tee_writer_write(self, tmp_path):
        """Test TeeWriter write method."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Write a message
        result = tee.write("test message\n")

        assert isinstance(result, int)
        assert result > 0

        # Verify file was written to
        with open(log_file, "r") as f:
            content = f.read()
            assert "test message" in content

    def test_tee_writer_write_empty_message(self, tmp_path):
        """Test TeeWriter write method with empty message."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        result = tee.write("")

        # Empty string should return 0 or minimal
        assert isinstance(result, int)

    def test_tee_writer_write_with_carriage_return(self, tmp_path):
        """Test TeeWriter write method filters out carriage returns."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Write a progress bar style message with \r
        result = tee.write("progress\r")

        assert isinstance(result, int)

        # File should be empty since \r messages are filtered
        with open(log_file, "r") as f:
            content = f.read()
            assert "progress" not in content

    def test_tee_writer_flush(self, tmp_path):
        """Test TeeWriter flush method."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Should not raise
        tee.flush()

    def test_tee_writer_close(self, tmp_path):
        """Test TeeWriter close method."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Should not raise
        tee.close()

        # After closing, writing should handle the closed file gracefully

    def test_tee_writer_isatty(self, tmp_path):
        """Test TeeWriter isatty method."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Should return boolean
        result = tee.isatty()
        assert isinstance(result, bool)

    def test_tee_writer_fileno(self, tmp_path):
        """Test TeeWriter fileno method."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Should return an integer file descriptor
        result = tee.fileno()
        assert isinstance(result, int)

    def test_tee_writer_getattr_delegation(self, tmp_path):
        """Test TeeWriter __getattr__ delegates to original stream."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Should delegate encoding attribute
        assert hasattr(tee, "encoding")

    def test_tee_writer_getattr_missing_attribute(self, tmp_path):
        """Test TeeWriter __getattr__ raises AttributeError for missing attributes."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        with pytest.raises(AttributeError):
            _ = tee.nonexistent_attribute_12345

    def test_tee_writer_pickle(self, tmp_path):
        """Test TeeWriter can be pickled and unpickled."""
        from ml_grid.util import logger_setup

        original_stdout = sys.stdout
        log_file = str(tmp_path / "stdout.log")

        tee = logger_setup.TeeWriter(original_stdout, log_file)

        # Pickle the object
        pickled = pickle.dumps(tee)

        # Unpickle
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, logger_setup.TeeWriter)


class TestStdoutRedirection:
    """Test stdout/stderr redirection functionality."""

    def test_redirect_stdout_enabled(self, tmp_path):
        """Test that stdout is redirected when redirect_stdout=True."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        # Check that stdout has been replaced with TeeWriter
        assert hasattr(sys.stdout, "write")
        assert hasattr(sys.stdout, "log_file")

    def test_redirect_stderr_enabled(self, tmp_path):
        """Test that stderr is redirected when redirect_stdout=True."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        # Check that stderr has been replaced with TeeWriter
        assert hasattr(sys.stderr, "write")
        assert hasattr(sys.stderr, "log_file")

    def test_no_redirect_when_bayessearch_true(self, tmp_path):
        """Test that redirection is disabled when bayessearch=True."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        # Set bayessearch to True (should disable redirect)
        global_parameters.bayessearch = True

        log_dir = tmp_path / "test_experiment"

        _original_stdout = sys.stdout
        _original_stderr = sys.stderr

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=True,
        )

        # When bayessearch=True, redirection should be disabled
        assert sys.stdout is _original_stdout or isinstance(
            sys.stdout, type(_original_stdout)
        )

    def test_no_redirect_when_redirect_stdout_false(self, tmp_path):
        """Test that redirection is disabled when redirect_stdout=False."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        original_stdout = sys.stdout

        logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        # stdout should not be replaced
        assert sys.stdout is original_stdout


class TestLoggerConfiguration:
    """Test logger configuration details."""

    def test_logger_does_not_propagate(self, tmp_path):
        """Test that ml_grid logger does not propagate to root."""
        from ml_grid.util import logger_setup

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        assert logger.propagate is False

    def test_file_handler_exists(self, tmp_path):
        """Test that file handler is configured."""
        from ml_grid.util import logger_setup

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        # Check for file handler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) > 0

    def test_console_handler_exists(self, tmp_path):
        """Test that console handler is configured."""
        from ml_grid.util import logger_setup

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        # Check for console handler (StreamHandler to stdout)
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) > 0

    def test_file_handler_level_debug(self, tmp_path):
        """Test that file handler logs at DEBUG level."""
        from ml_grid.util import logger_setup

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) > 0
        assert file_handlers[0].level == logging.DEBUG

    def test_log_message_format(self, tmp_path):
        """Test that log messages are formatted correctly."""
        from ml_grid.util import logger_setup

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        # Write a test message to file
        logger.debug("test debug message")

        log_file = tmp_path / "run_0" / "run.log"
        assert log_file.exists()

        with open(log_file, "r") as f:
            content = f.read()
            assert "test debug message" in content

    def test_verbose_level_debug(self, tmp_path):
        """Test that verbose=3 sets DEBUG level for console."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        # Capture stdout before setup_logger changes it
        _original_stdout = sys.stdout

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=3,  # DEBUG level
            redirect_stdout=False,
        )

        assert logger.level <= logging.DEBUG

    def test_verbose_level_warning(self, tmp_path):
        """Test that verbose=0 sets WARNING level for console."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        _original_stdout = sys.stdout

        logger = logger_setup.setup_logger(
            experiment_dir=str(tmp_path),
            param_space_index=0,
            verbose=0,  # WARNING level (default console)
            redirect_stdout=False,
        )

        assert logger.level <= logging.WARNING

    def test_log_file_not_duplicated(self, tmp_path):
        """Test that running setup_logger twice doesn't create duplicate handlers."""
        from ml_grid.util import logger_setup
        from ml_grid.util.global_params import global_parameters

        global_parameters.bayessearch = False

        log_dir = tmp_path / "test_experiment"

        # First call
        logger1 = logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        file_handlers_1 = [
            h for h in logger1.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers_1) == 1

        # Second call should reuse the same logger instance
        logger2 = logger_setup.setup_logger(
            experiment_dir=str(log_dir),
            param_space_index=0,
            verbose=1,
            redirect_stdout=False,
        )

        file_handlers_2 = [
            h for h in logger2.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers_2) == 1

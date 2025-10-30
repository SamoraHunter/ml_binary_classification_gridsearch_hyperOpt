import logging
import os
import sys
from datetime import datetime
from typing import Optional, Any, Dict
from ml_grid.util.global_params import global_parameters


def setup_logger(
    experiment_dir: str,
    param_space_index: int,
    verbose: int = 1,
    redirect_stdout: bool = True,
) -> logging.Logger:
    """Sets up a logger that writes to a file and the console.

    This function creates a timestamped main experiment folder and a sub-folder
    for the specific run (e.g., 'run_0'). It configures a logger to save all
    messages to 'run.log' inside the sub-folder and also redirects stdout/stderr
    to this logger.

    Args:
        experiment_dir (str): The path to the parent directory for this group
            of experimental runs.
        param_space_index (int): The index of the parameter space run.
        verbose (int): The verbosity level. Higher numbers mean more logs to
            the console.
        redirect_stdout (bool): If True, redirects stdout and stderr to the
            logger. Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Avoid reconfiguring the root logger if it's already set up.
    if logging.getLogger().handlers:
        logging.getLogger().handlers.clear()

    # Get a specific logger instead of configuring the root
    logger = logging.getLogger("ml_grid")
    logger.setLevel(logging.DEBUG)

    # Prevent messages from being passed to the root logger
    logger.propagate = False

    # Clear existing handlers from this specific logger to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create the specific sub-folder for this run (e.g., "run_0")
    log_folder_path = os.path.join(experiment_dir, f"run_{param_space_index}")
    os.makedirs(log_folder_path, exist_ok=True)
    log_file = os.path.join(log_folder_path, "run.log")

    # File handler - always logs at DEBUG level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler verbosity is controlled by the 'verbose' parameter
    # CRITICAL: Use sys.__stdout__ to prevent recursion loops
    console_log_level = logging.WARNING
    if verbose >= 3:
        console_log_level = logging.DEBUG
    elif verbose >= 1:
        console_log_level = logging.INFO

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Only disable redirection when using bayessearch (hyperopt manages its own stdout)
    should_redirect = redirect_stdout and not global_parameters.bayessearch

    if should_redirect:
        # Capture the current stdout/stderr (which might be Jupyter's custom streams)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_log = os.path.join(log_folder_path, "stdout.log")
        stderr_log = os.path.join(log_folder_path, "stderr.log")

        class TeeWriter:
            """Writes to both original stream and a file, preserving all stream behaviors."""

            def __init__(self, original_stream, log_file_path):
                self.original_stream = original_stream
                self.log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")
                # Preserve attributes from original stream
                self.encoding = getattr(original_stream, "encoding", "utf-8")
                self.errors = getattr(original_stream, "errors", "strict")
                self.mode = getattr(original_stream, "mode", "w")

            def write(self, message: str) -> int:
                """Write to both streams, return bytes written."""
                written = 0
                # Always write to original stream first (for Jupyter/console visibility)
                if message:
                    try:
                        result = self.original_stream.write(message)
                        written = result if result is not None else len(message)
                    except Exception:
                        written = len(message)

                    # Then append to log file
                    try:
                        self.log_file.write(message)
                    except Exception:
                        pass

                return written

            def flush(self) -> None:
                """Flush both streams."""
                try:
                    self.original_stream.flush()
                except Exception:
                    pass
                try:
                    self.log_file.flush()
                except Exception:
                    pass

            def close(self) -> None:
                """Close only the log file, not the original stream."""
                try:
                    self.log_file.close()
                except Exception:
                    pass

            def isatty(self) -> bool:
                """Check if original stream is a TTY."""
                return getattr(self.original_stream, "isatty", lambda: False)()

            def fileno(self):
                """Return file descriptor of original stream if available."""
                if hasattr(self.original_stream, "fileno"):
                    return self.original_stream.fileno()
                raise OSError("fileno not available")

            def __getattr__(self, name):
                """Delegate any other attributes to the original stream."""
                return getattr(self.original_stream, name)

        sys.stdout = TeeWriter(original_stdout, stdout_log)
        sys.stderr = TeeWriter(original_stderr, stderr_log)

    return logger

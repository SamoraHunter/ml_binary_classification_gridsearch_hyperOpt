import logging
import os
import sys
from datetime import datetime
from typing import Optional, Any, Dict


def setup_logger(
    experiment_dir: str,
    param_space_index: int,
    verbose: int = 1,
) -> logging.Logger:
    """Sets up a logger that writes to a file and the console.

    This function creates a timestamped main experiment folder and a sub-folder
    for the specific run (e.g., 'run_0'). It configures a logger to save all
    messages to 'run.log' inside the sub-folder and also redirects stdout/stderr
    to this logger.

    Args:
        experiment_dir (str): The path to the parent directory for this group
            of experimental runs.
        verbose (int): The verbosity level. Higher numbers mean more logs to
            the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Avoid reconfiguring the root logger if it's already set up.
    if logging.getLogger().handlers:
        logging.getLogger().handlers.clear()

    # Create the specific sub-folder for this run (e.g., "run_0")
    log_folder_path = os.path.join(experiment_dir, f"run_{param_space_index}")

    os.makedirs(log_folder_path, exist_ok=True)
    log_file = os.path.join(log_folder_path, "run.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Console handler verbosity is controlled by the 'verbose' parameter
    console_log_level = logging.WARNING
    if verbose >= 3:
        console_log_level = logging.DEBUG
    elif verbose >= 1:
        console_log_level = logging.INFO

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Redirect stdout to the logger
    class LoggerWriter:
        """A file-like object to redirect stdout to a logger."""

        def __init__(self, logger: logging.Logger, level: int):
            """Initializes the LoggerWriter.

            Args:
                logger (logging.Logger): The logger instance to write to.
                level (int): The logging level to use (e.g., logging.INFO).
            """
            self.logger = logger
            self.level = level
            self.buffer = ""

        def write(self, message: str) -> None:
            """Writes a message to the logger.

            Empty messages are ignored.

            Args:
                message (str): The message to log.
            """
            self.buffer += message
            if '\n' in self.buffer:
                lines = self.buffer.split('\n')
                for line in lines[:-1]:
                    if line.strip():
                        self.logger.log(self.level, line.strip())
                self.buffer = lines[-1]

        def flush(self) -> None:
            """A no-op flush method to comply with the file-like object interface."""
            pass

    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger

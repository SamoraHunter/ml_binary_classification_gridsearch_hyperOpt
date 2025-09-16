import logging
import os
import sys
from datetime import datetime
from types import FrameType
from typing import Any, Optional


def setup_logger(log_folder_path: str = ".") -> logging.Logger:
    """Sets up a logger that writes to a file and the console.

    This function configures a logger to save debug-level messages to a
    timestamped log file and info-level messages to the console. It also
    sets up a system trace to log file and line number for debugging purposes,
    and redirects stdout to the logger.

    Args:
        log_folder_path (str, optional): The directory where log files will be
            stored. Defaults to ".".

    Returns:
        logging.Logger: The configured logger instance.
    """

    # Get the root directory of the notebook
    notebook_dir = ""
    try:
        from IPython import get_ipython

        ipython_instance = get_ipython()
        if ipython_instance:
            notebook_dir = os.path.dirname(
                ipython_instance.config["IPKernelApp"]["connection_file"]
            )
    except (ImportError, NameError, KeyError, AttributeError):
        # Fallback if IPython is not available or configured as expected
        notebook_dir = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv else __file__))
    print("notebook_dir", notebook_dir)

    # Navigate up from the notebook directory to get the logs directory
    folder_name = log_folder_path
    current_dir = os.getcwd()
    # Combine the current directory and folder name to get the target directory
    logs_dir = os.path.abspath(os.path.join(current_dir, folder_name))
    print("logs_dir", logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    # Get the current date and time
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up logging with the current date and time in the log file name
    log_file = os.path.join(logs_dir, f"{current_date_time}_ml_grid.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create a logger
    logger = logging.getLogger(__name__)

    # logger.addFilter(ExcludeMatplotlibFontManagerFilter())

    # Define a handler to print log messages to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Define a trace function for logging
    def tracefunc(frame: FrameType, event: str, arg: Any) -> Optional[callable]:
        """A trace function to log execution line by line for debugging.

        This function is registered with `sys.settrace` and logs the file
        and line number for each line event that occurs within the project's
        directory, ignoring certain library paths.

        Args:
            frame (FrameType): The current stack frame.
            event (str): The type of event (e.g., 'line', 'call').
            arg (Any): Event-specific argument.

        Returns:
            Optional[callable]: The trace function itself to continue tracing, or
            None to stop tracing in the current scope.
        """
        substrings_to_ignore = [
            "matplotlib.font_manager",
            "matplotlib",
        ]

        # Only log events from files within the notebook directory
        if notebook_dir in frame.f_code.co_filename:
            filename = frame.f_code.co_filename

            if any(substring in filename for substring in substrings_to_ignore):
                return None
            if event == "line":
                logger.debug(
                    f"{event}: {frame.f_code.co_filename} - Line {frame.f_lineno}"
                )

        return tracefunc

    # Register the trace function globally for all events
    sys.settrace(tracefunc)

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

        def write(self, message: str) -> None:
            """Writes a message to the logger.

            Empty messages are ignored.

            Args:
                message (str): The message to log.
            """
            if message.strip():
                self.logger.log(self.level, message.strip())

        def flush(self) -> None:
            """A no-op flush method to comply with the file-like object interface."""
            pass

    sys.stdout = LoggerWriter(logger, logging.INFO)

    return logger

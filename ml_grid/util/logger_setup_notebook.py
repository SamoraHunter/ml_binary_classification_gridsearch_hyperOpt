import logging
import os
import sys
from datetime import datetime


def setup_logger(notebook_verbose=False):
    # Get the directory path of the current module
    module_dir = os.path.dirname(os.path.realpath(__file__))

    ml_grid_root_dir = os.path.dirname(os.path.abspath(__file__)).split("ml_grid")[0]

    # Navigate up two directories from the module directory to get the logs directory
    logs_dir = os.path.abspath(
        os.path.join(module_dir, "..", "..", "HFE_ML_experiments", "logs")
    )
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

    # Define a handler to print log messages to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        logging.INFO
    )  # Set the level to INFO or any level you desire
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Define a trace function for logging
    def tracefunc(frame, event, arg):

        ignore_log_paths_list = [
            "ml_grid_env",
            "grid_param_space.py",
            "ml_grid",
            "matplotlib",
        ]
        file_path = frame.f_code.co_filename

        if any(path in file_path for path in ignore_log_paths_list):
            return tracefunc
        # Only log events from files within your project directory
        # if "ml_grid_env" in frame.f_code.co_filename:
        #     return tracefunc

        if ml_grid_root_dir in frame.f_code.co_filename:
            if event == "line":
                logger.debug(
                    f"{event}: {frame.f_code.co_filename} - Line {frame.f_lineno}"
                )

        return tracefunc

    # Register the trace function globally for all events
    sys.settrace(tracefunc)

    # Custom Stream Handler to capture both stdout and stderr
    class NotebookStreamHandler(logging.StreamHandler):
        def __init__(self, stream=None):
            super().__init__(stream)
            self.stream = stream

        def emit(self, record):
            try:
                # Write the LogRecord to the original stream
                if self.stream is not None:
                    msg = self.format(record)
                    self.stream.write("{}\n".format(msg))
                    self.flush()

                # Emit the LogRecord to the logger's handlers
                logging.StreamHandler.emit(self, record)
            except Exception:
                self.handleError(record)

    if notebook_verbose:
        # Add the custom stream handler to the logger
        logger.addHandler(NotebookStreamHandler(sys.stdout))

    return logger

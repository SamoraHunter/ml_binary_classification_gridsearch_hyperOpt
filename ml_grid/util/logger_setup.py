import logging
import os
from datetime import datetime
import sys
from IPython.core.getipython import get_ipython


def setup_logger(log_folder_path="."):
    # Get the directory path of the current module
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # Get the root directory of the notebook
    try:
        from IPython import get_ipython
        notebook_dir = os.path.dirname(get_ipython().config["IPKernelApp"]["connection_file"])
    except (AttributeError, NameError):
        notebook_dir = os.path.dirname(os.path.abspath(__file__))
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
    def tracefunc(frame, event, arg):

        substrings_to_ignore = [
            "matplotlib.font_manager",
            "matplotlib",
        ]

        # Only log events from files within the notebook directory
        if notebook_dir in frame.f_code.co_filename:
            filename = frame.f_code.co_filename

            if any(substring in filename for substring in substrings_to_ignore):
                return
            if event == "line":
                logger.debug(
                    f"{event}: {frame.f_code.co_filename} - Line {frame.f_lineno}"
                )

        return tracefunc

    # Register the trace function globally for all events
    sys.settrace(tracefunc)

    # Redirect stdout to the logger
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            if message.strip():
                self.logger.log(self.level, message.strip())

        def flush(self):
            pass

    sys.stdout = LoggerWriter(logger, logging.INFO)

    return logger

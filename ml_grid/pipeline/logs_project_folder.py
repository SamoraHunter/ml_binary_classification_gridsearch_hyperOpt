import logging
import pathlib
import os
from typing import Any, Dict, Optional


class log_folder:
    """Creates a unique log folder for each experimental run based on its parameters."""

    def __init__(
        self,
        local_param_dict: Dict[str, Any],
        additional_naming: Optional[str],
        base_project_dir: str,
    ) -> None:
        """Initializes the log folder and sets up basic logging.

        This constructor generates a unique folder name by concatenating the
        values from the `local_param_dict`. It then creates this folder and
        configures a basic logger to write to a 'log.log' file inside it.

        Note:
            This class re-configures the root logger on each instantiation,
            which may have unintended side effects in a larger application.

        Args:
            local_param_dict (Dict[str, Any]): A dictionary of parameters for the
                current pipeline run.
            additional_naming (Optional[str]): An additional string to append to
                the folder name.
            base_project_dir (str): The root directory for the project.
        """

        str_b = ""
        for key in local_param_dict.keys():
            if key != "data":
                str_b = str_b + "_" + str(local_param_dict.get(key))
            else:
                for data_key in local_param_dict.get("data", {}):
                    str_b = str_b + str(int(local_param_dict.get("data", {}).get(data_key)))

        global_param_str = str_b

        print(global_param_str)

        log_folder_name = f"{global_param_str}{additional_naming or ''}"
        log_folder_path = os.path.join(base_project_dir, log_folder_name, "logs")

        pathlib.Path(log_folder_path).mkdir(parents=True, exist_ok=True)

        full_log_path = os.path.join(log_folder_path, "log.log")

        logging.basicConfig(filename=full_log_path)
        stderrLogger = logging.StreamHandler()
        stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger().addHandler(stderrLogger)

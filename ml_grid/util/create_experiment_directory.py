import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def create_experiment_directory(
    base_dir: str, additional_naming: Optional[str] = None
) -> str:
    """Creates a single, timestamped directory for a group of experiment runs.

    This function should be called once at the beginning of an experiment script
    to create a unique parent folder for all the runs in that batch.

    Args:
        base_dir (str): The base directory where experiment folders will be stored
                        (e.g., 'notebooks/HFE_ML_experiments').
        additional_naming (Optional[str], optional): A descriptive name to append
                                                     to the timestamp. Defaults to None.

    Returns:
        str: The full path to the created experiment directory.
    """
    logger = logging.getLogger("ml_grid")

    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = (
        f"{current_date_time}_{additional_naming}"
        if additional_naming
        else current_date_time
    )
    experiment_dir = Path(base_dir) / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory created: {experiment_dir}")
    return str(experiment_dir)

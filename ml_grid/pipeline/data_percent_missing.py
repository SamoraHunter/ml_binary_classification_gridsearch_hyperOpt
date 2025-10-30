import logging
import os
import pickle
from typing import Any, Dict, List


def handle_percent_missing(
    local_param_dict: Dict[str, Any],
    all_df_columns: List[str],
    file_name: str,
    drop_list: List[str],
) -> List[str]:
    """Handles the removal of columns with a high percentage of missing data.

    This function reads a pre-calculated dictionary of missing data percentages
    from a pickle file. It then identifies columns that exceed the missing
    data threshold specified in `local_param_dict` and adds them to the
    `drop_list`.

    Args:
        local_param_dict (Dict[str, Any]): Dictionary of parameters for the
            current pipeline, must contain a 'percent_missing' threshold.
        all_df_columns (List[str]): All column names in the DataFrame to be
            processed.
        file_name (str): The base name of the input data file, used to find
            the corresponding percent missing pickle file.
        drop_list (List[str]): List of columns to be dropped from the
            DataFrame.

    Returns:
        List[str]: Updated list of columns to be dropped from the dataframe.
    """
    logger = logging.getLogger("ml_grid")
    # Check for null pointer references
    assert local_param_dict is not None
    assert all_df_columns is not None
    assert drop_list is not None

    percent_missing_drop_list = []

    filename = file_name.replace(".csv", "")

    # Check if the file with .pkl extension exists, otherwise use .pickle
    if os.path.exists(f"{filename}_percent_missing.pkl"):
        percent_missing_filename = f"{filename}_percent_missing.pkl"
    else:
        percent_missing_filename = f"{filename}_percent_missing.pickle"

    # Check if the file exists
    if os.path.exists(percent_missing_filename):
        with open(percent_missing_filename, "rb") as handle:
            try:
                percent_missing_dict = pickle.load(handle)
            except Exception as e:
                logger.error(f"Error loading pickle file: {e}")
                percent_missing_dict = {}
    else:
        logger.warning(
            f"File {percent_missing_filename} not found. Returning empty dict."
        )
        percent_missing_dict = {}

    percent_missing_threshold = local_param_dict.get("percent_missing")
    if percent_missing_threshold is not None and percent_missing_dict != {}:
        # print(
        #     f"Identifying columns with > {percent_missing_threshold} percent missing data..."
        # )

        # Iterate through columns
        for col in all_df_columns:
            # Try to get the value from the dictionary
            try:
                if (
                    col in percent_missing_dict
                    and percent_missing_dict.get(col) > percent_missing_threshold
                ):
                    percent_missing_drop_list.append(col)
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}")
                pass

        logger.info(
            f"Identified {len(percent_missing_drop_list)} columns with > {percent_missing_threshold} percent missing data."
        )

        # Extend the drop list with identified columns
        drop_list.extend(percent_missing_drop_list)

    else:
        logger.info(
            "percent_missing_threshold is None or percent_missing_dict is empty. Skipping percent missing data check."
        )

    return drop_list

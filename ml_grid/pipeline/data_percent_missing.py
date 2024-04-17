import pickle
import os
from typing import List


def handle_percent_missing(
    local_param_dict: dict, all_df_columns: List[str], drop_list: List[str]
) -> List[str]:
    """
    Handles the removal of columns with a high percentage of missing data.

    Args:
        local_param_dict (dict): Dictionary of parameters for the current pipeline.
        all_df_columns (List[str]): All the column names in the dataframe to be processed.
        drop_list (List[str]): List of columns to be dropped from the dataframe.

    Returns:
        List[str]: Updated list of columns to be dropped from the dataframe.
    """
    # Check for null pointer references
    assert local_param_dict is not None
    assert all_df_columns is not None
    assert drop_list is not None

    percent_missing_drop_list = []

    # Check if the file exists
    if os.path.exists("percent_missing_dict.pickle"):
        with open("percent_missing_dict.pickle", "rb") as handle:
            try:
                percent_missing_dict = pickle.load(handle)
            except Exception as e:
                print(f"Error loading pickle file: {e}")
                percent_missing_dict = {}
    else:
        print("File 'percent_missing_dict.pickle' not found. Returning empty dict.")
        percent_missing_dict = {}

    percent_missing_threshold = local_param_dict.get("percent_missing")
    if percent_missing_threshold is not None and percent_missing_dict is not {}:
        # print(
        #     f"Identifying columns with > {percent_missing_threshold} percent missing data..."
        # )

        # Iterate through columns
        for col in all_df_columns:
            # Try to get the value from the dictionary
            try:
                if percent_missing_dict.get(col) > percent_missing_threshold:
                    percent_missing_drop_list.append(col)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                pass

        print(
            f"Identified {len(percent_missing_drop_list)} columns with > {percent_missing_threshold} percent missing data."
        )

        # Extend the drop list with identified columns
        drop_list.extend(percent_missing_drop_list)

    else:
        print(
            "percent_missing_threshold is None or percent_missing_dict is empty. Skipping percent missing data check."
        )

    return drop_list

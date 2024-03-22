import pickle
import os


def handle_percent_missing(local_param_dict, all_df_columns, drop_list):
    percent_missing_drop_list = []

    # Check if the file exists
    if os.path.exists("percent_missing_dict.pickle"):
        with open("percent_missing_dict.pickle", "rb") as handle:
            percent_missing_dict = pickle.load(handle)

        percent_missing_threshold = local_param_dict.get("percent_missing")

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
    else:
        print("File 'percent_missing_dict.pickle' not found. Returning empty list.")

    # Extend the drop list with identified columns
    drop_list.extend(percent_missing_drop_list)

    return drop_list

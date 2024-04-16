import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import sequence
from IPython.display import display


def add_date_order_sequence_column(df):
    """
    Add a sequence column based on the timestamp within each client_idcode group.

    Args:
    df (DataFrame): DataFrame with 'timestamp' and 'client_idcode' columns.

    Returns:
    DataFrame: DataFrame with added 'date_order_sequence' column.
    """
    # Convert 'timestamp' column to datetime if it's not already
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort dataframe by 'client_idcode' and 'timestamp'
    df.sort_values(by=["client_idcode", "timestamp"], inplace=True)

    # Group by 'client_idcode' and assign a sequential order based on timestamp
    df["date_order_sequence"] = df.groupby("client_idcode").cumcount() + 1

    return df


def max_client_idcode_sequence_length(df):
    """
    Calculate the maximum sequence length for client_idcode.

    Args:
    df (DataFrame): DataFrame with 'client_idcode' column.

    Returns:
    int: Maximum sequence length.
    """
    # Count occurrences of each client_idcode
    idcode_counts = df["client_idcode"].value_counts()

    # Find the maximum count
    max_length = idcode_counts.max()

    return max_length


def convert_Xy_to_time_series(X, y, max_seq_length):
    """
    Convert DataFrame into time series format suitable for training.

    Args:
    X (DataFrame): Features DataFrame.
    y (Series): Target variable.

    Returns:
    tuple: Tuple containing X and y in the format suitable for time series training.
    """
    feature_list = X.columns

    X_list = []
    y_list = []

    for pat in tqdm(X["client_idcode"].unique()):
        pat_data = (
            X[X["client_idcode"] == pat][feature_list]
            .drop("client_idcode", axis=1)
            .values
        )
        pat_multi_vector = sequence.pad_sequences(
            np.transpose(pat_data), maxlen=max_seq_length
        )
        X_list.append(pat_multi_vector)
        y_list.append(y[X["client_idcode"] == pat].iloc[0])

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    return X_array, y_array


# Example usage:
# X, y = convert_df_to_time_series(pre_ts_df, pre_ts_df['outcome_var_1'])

from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def correlation_coefficient(col1: pd.Series, col2: pd.Series) -> float:
    """Calculates the Pearson correlation coefficient between two pandas Series.

    Args:
        col1 (pd.Series): The first series.
        col2 (pd.Series): The second series.

    Returns:
        float: The correlation coefficient.
    """
    return col1.corr(col2)


def handle_correlation_matrix(
    local_param_dict: Dict[str, Any],
    drop_list: List[Any],
    df: pd.DataFrame,
    chunk_size: int = 50,
) -> List[Any]:
    """Identifies highly correlated columns and adds them to a drop list.

    This function calculates the correlation coefficient between numeric columns
    in the input DataFrame in chunks to manage memory usage. Pairs of columns
    with a correlation greater than the specified threshold are added to the
    `drop_list` as tuples.

    Note:
        This function currently adds tuples of `(column, correlated_column)` to
        the `drop_list`. Downstream processing might expect a flat list of
        column names to drop, which could cause these correlated columns to not
        be dropped as intended.

    Calculates the correlation coefficient between each column in the input DataFrame
    using chunks to avoid memory issues. The correlation threshold is defined by
    the 'corr' key in the local_param_dict dictionary.

    Args:
        local_param_dict (Dict[str, Any]): Dictionary containing local parameters,
            including the 'corr' threshold.
        drop_list (List[Any]): A list to which pairs of correlated columns will
            be appended.
        df (pd.DataFrame): The input DataFrame.
        chunk_size (int, optional): The size of each chunk for correlation
            calculation. Defaults to 50.

    Returns:
        List[Any]: The updated list containing unique pairs of correlated columns.
    """

    if chunk_size >= len(df):
        chunk_size = len(df) - 1
    # Define the correlation threshold
    threshold = local_param_dict.get("corr", 0.25)

    # Remove non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df_numeric = df[numeric_columns]

    # Split columns into chunks
    column_chunks = [
        df_numeric.columns[i : i + chunk_size]
        for i in range(0, len(df_numeric.columns), chunk_size)
    ]

    # Iterate through each column chunk
    for chunk in tqdm(column_chunks, desc="Calculating Correlations"):
        # Calculate the correlation coefficients for the current chunk
        try:
            correlations = df_numeric[chunk].corr()
        except Exception as e:
            print(
                "Encountered exception while calculating correlations for chunk", chunk
            )
            print(e)
            continue

        # Iterate through each column in the chunk
        for col in chunk:
            # Filter columns with correlation coefficient greater than the threshold
            try:
                correlated_cols = correlations[col][
                    (correlations[col] > threshold) & (correlations[col] != 1)
                ].index.tolist()
            except KeyError:
                print(
                    "Encountered KeyError while calculating correlations for column",
                    col,
                )
                print("Continuing with an empty list of correlated columns")
                correlated_cols = []

            # Add the correlated columns to the list
            drop_list.extend(
                [(col, corr_col) for corr_col in correlated_cols]
            )

    # Remove duplicates from the list
    drop_list = list(set(drop_list))

    return drop_list


# def handle_correlation_matrix(local_param_dict, drop_list, df, chunk_size=50):
#     """
#     Calculate correlated columns in chunks.

#     Calculates the correlation coefficient between each column in the input DataFrame
#     using chunks to avoid memory issues. The correlation threshold is defined by
#     the 'corr' key in the local_param_dict dictionary.

#     Args:
#         local_param_dict (dict): Dictionary containing local parameters, including the correlation threshold.
#         drop_list (list): List to which correlated columns will be appended.
#         df (pandas.DataFrame): Input DataFrame.
#         chunk_size (int, optional): Size of each chunk for correlation calculation. Default is 50.

#     Returns:
#         list: List of correlated columns.
#     """

#     if chunk_size >= len(df):
#         chunk_size = len(df) - 1
#     # Define the correlation threshold
#     threshold = local_param_dict.get("corr", 0.25)

#     # Remove non-numeric columns
#     numeric_columns = df.select_dtypes(include=["number"]).columns
#     df_numeric = df[numeric_columns]

#     # Split columns into chunks
#     column_chunks = [
#         df_numeric.columns[i : i + chunk_size]
#         for i in range(0, len(df_numeric.columns), chunk_size)
#     ]

#     # Iterate through each column chunk
#     for chunk in tqdm(column_chunks, desc="Calculating Correlations"):
#         # Calculate the correlation coefficients for the current chunk
#         try:
#             correlations = df_numeric[chunk].corr()
#         except:
#             print(
#                 "Encountered exception while calculating correlations for chunk", chunk
#             )
#             print(traceback.format_exc())
#             continue

#         # Iterate through each column in the chunk
#         for col in chunk:
#             # Filter columns with correlation coefficient greater than the threshold
#             try:
#                 correlated_cols = correlations[col][
#                     correlations[col].abs() > threshold
#                 ].index.tolist()
#             except KeyError:
#                 print(
#                     "Encountered KeyError while calculating correlations for column",
#                     col,
#                 )
#                 print("Continuing with an empty list of correlated columns")
#                 correlated_cols = []
#             except AttributeError:
#                 print(
#                     "Encountered AttributeError while calculating correlations for column",
#                     col,
#                 )
#                 print("Continuing with an empty list of correlated columns")
#                 correlated_cols = []

#             # Exclude the current column from the correlated columns list if it's in the list
#             if col in correlated_cols:
#                 correlated_cols.remove(col)

#             # Add the correlated columns to the list
#             drop_list.extend([(col, corr_col) for corr_col in correlated_cols])

#     # Remove duplicates from the list
#     drop_list = list(set(drop_list))

#     return drop_list


# Example usage:
# input_csv_path = '../concatenated_data_concatenated_output_imputed_f_b_m_collapsed_mean.csv'
# df = pd.read_csv(input_csv_path)
# local_param_dict = {'corr': 0.25}  # Example threshold value

# correlated_columns = handle_correlation_matrix(df, local_param_dict)
# print("Columns with correlation greater than", local_param_dict['corr'])
# print(correlated_columns)


# def handle_correlation_matrix(local_param_dict, drop_list, df):
#     print("Handling correlation matrix")
#     temp_col_list = list(df.select_dtypes(include=[float, int]).columns)

#     # Calculate absolute correlation matrix
#     corr_matrix = df.select_dtypes(include=[float, int]).corr().abs()

#     # Create a True/False mask and apply it
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#     tri_df = corr_matrix.mask(mask)

#     # List column names of highly correlated features (r > local_param_dict['corr'])
#     corr_to_drop = [
#         c for c in tri_df.columns if any(tri_df[c] > local_param_dict.get("corr"))
#     ]

#     print(
#         f"Identified {len(corr_to_drop)} correlated features to drop at >{local_param_dict.get('corr')}"
#     )
#     drop_list.extend(corr_to_drop)

#     return drop_list
# import pandas as pd
# import numpy as np


# def correlation_coefficient(x, y):
#     """
#     Calculate the correlation coefficient between two lists of values.

#     Parameters:
#         x (list): First list of values.
#         y (list): Second list of values.

#     Returns:
#         float: Correlation coefficient between x and y.
#     """
#     n = len(x)
#     sum_x = sum(x)
#     sum_y = sum(y)
#     sum_x_sq = sum(xi**2 for xi in x)
#     sum_y_sq = sum(yi**2 for yi in y)
#     sum_xy = sum(xi * yi for xi, yi in zip(x, y))

#     numerator = n * sum_xy - sum_x * sum_y
#     denominator = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2)) ** 0.5

#     if denominator == 0:
#         return 0
#     else:
#         return numerator / denominator


# def handle_correlation_matrix(local_param_dict, drop_list, df):
#     print("Handling correlation matrix")
#     temp_col_list = list(df.select_dtypes(include=[float, int]).columns)

#     # Initialize an empty DataFrame to store correlation coefficients
#     corr_matrix = pd.DataFrame(index=temp_col_list, columns=temp_col_list)

#     # Calculate correlation coefficients for each pair of columns
#     for i, col1 in enumerate(temp_col_list):
#         for j, col2 in enumerate(temp_col_list):
#             if i != j:
#                 corr_matrix.loc[col1, col2] = correlation_coefficient(
#                     df[col1], df[col2]
#                 )

#     # Convert the DataFrame to absolute values
#     corr_matrix = corr_matrix.abs()

#     # Create a True/False mask and apply it
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#     tri_df = corr_matrix.mask(mask)

#     # List column names of highly correlated features (r > local_param_dict['corr'])
#     corr_to_drop = [
#         c for c in tri_df.columns if any(tri_df[c] > local_param_dict.get("corr"))
#     ]

#     print(
#         f"Identified {len(corr_to_drop)} correlated features to drop at >{local_param_dict.get('corr')}"
#     )
#     drop_list.extend(corr_to_drop)

#     return drop_list


# Example usage:
# local_param_dict = {'corr': 0.25}  # Example threshold value
# drop_list = []

# # Assuming df is your DataFrame
# # Replace df with your actual DataFrame
# # Call the function to update the drop_list
# updated_drop_list = handle_correlation_matrix(local_param_dict, drop_list, df)
# print("Updated drop list:", updated_drop_list)

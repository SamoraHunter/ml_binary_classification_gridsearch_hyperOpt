import sys
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
    for chunk in tqdm(column_chunks, desc="Calculating Correlations", file=sys.stdout):
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
                    correlations[col].abs() > threshold
                ].index.tolist()
            except KeyError:
                print(
                    "Encountered KeyError while calculating correlations for column",
                    col,
                )
                print("Continuing with an empty list of correlated columns")
                correlated_cols = []

            # Exclude the column itself from the list of correlated columns
            if col in correlated_cols:
                correlated_cols.remove(col)

            # Add the correlated columns to the list
            for corr_col in correlated_cols:
                # Add only the second column of the pair to avoid dropping both
                if col != corr_col:
                    drop_list.append(corr_col)

    # Remove duplicates from the list
    drop_list = list(set(drop_list))

    return drop_list

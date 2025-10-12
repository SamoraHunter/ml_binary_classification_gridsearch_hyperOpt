import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
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


def calculate_correlation_chunk(
    df: pd.DataFrame,
    cols_chunk1: List[str],
    cols_chunk2: List[str],
) -> pd.DataFrame:
    """Calculate correlation between two chunks of columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_chunk1 (List[str]): First set of column names.
        cols_chunk2 (List[str]): Second set of column names.
    
    Returns:
        pd.DataFrame: Correlation matrix for the chunk pairs.
    """
    # Extract data for both chunks
    data1 = df[cols_chunk1].values
    data2 = df[cols_chunk2].values
    
    # Calculate means and standard deviations
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    std1 = np.std(data1, axis=0, ddof=1)
    std2 = np.std(data2, axis=0, ddof=1)
    
    # Center the data
    centered1 = data1 - mean1
    centered2 = data2 - mean2
    
    # Calculate correlation matrix chunk
    n = len(df)
    corr_chunk = np.dot(centered1.T, centered2) / (n - 1)
    
    # Normalize by standard deviations
    corr_chunk = corr_chunk / np.outer(std1, std2)
    
    return pd.DataFrame(corr_chunk, index=cols_chunk1, columns=cols_chunk2)


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
    
    To avoid dropping both columns in a highly correlated pair, this function
    only adds one of the columns from each pair to the `drop_list`.

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

    # Define the correlation threshold
    threshold = local_param_dict.get("corr") if local_param_dict.get("corr") is not None else 0.98

    # Remove non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numeric_columns) == 0:
        return drop_list
    
    df_numeric = df[numeric_columns]
    n_cols = len(numeric_columns)
    
    # Adjust chunk size if necessary
    if chunk_size >= n_cols:
        chunk_size = n_cols
    
    # Track which columns to drop
    to_drop = set()
    
    # Calculate number of chunks
    n_chunks = int(np.ceil(n_cols / chunk_size))
    
    print(f"Processing {n_cols} columns in {n_chunks} chunks of size {chunk_size}...")
    
    # Process correlation matrix in chunks (upper triangle only)
    for i in tqdm(range(n_chunks), desc="Processing column chunks"):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, n_cols)
        cols_i = numeric_columns[start_i:end_i]
        
        # Process only the upper triangle (j <= i)
        for j in range(i + 1):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n_cols)
            cols_j = numeric_columns[start_j:end_j]
            
            # Calculate correlation for this chunk pair
            try:
                if i == j:
                    # Diagonal chunk - calculate self-correlation
                    corr_chunk = df_numeric[cols_i].corr().abs()
                    
                    # Only look at upper triangle within this chunk
                    for idx_i, col_i in enumerate(cols_i):
                        if col_i in to_drop:
                            continue
                        for idx_j in range(idx_i):
                            col_j = cols_j[idx_j]
                            if col_j in to_drop:
                                continue
                            if corr_chunk.iloc[idx_j, idx_i] > threshold:
                                to_drop.add(col_i)
                                break
                        if col_i in to_drop:
                            break
                else:
                    # Off-diagonal chunk
                    corr_chunk = calculate_correlation_chunk(df_numeric, cols_i, cols_j).abs()
                    
                    # Check all pairs in this chunk
                    for idx_i, col_i in enumerate(cols_i):
                        if col_i in to_drop:
                            continue
                        for idx_j, col_j in enumerate(cols_j):
                            if col_j in to_drop:
                                continue
                            if corr_chunk.iloc[idx_i, idx_j] > threshold:
                                # Drop the column that appears later in the original list
                                col_to_drop = col_i if start_i + idx_i > start_j + idx_j else col_j
                                to_drop.add(col_to_drop)
                                if col_to_drop == col_i:
                                    break
                        if col_i in to_drop:
                            break
                            
            except Exception as e:
                print(f"Warning: Error processing chunk ({i}, {j}): {e}", file=sys.stderr)
                continue
    
    # Add the identified columns to the initial drop_list
    drop_list.extend(list(to_drop))
    
    print(f"Identified {len(to_drop)} columns to drop due to high correlation.")
    
    # Return a list of unique columns to drop
    return list(set(drop_list))
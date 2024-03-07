import pandas as pd
import numpy as np


def handle_correlation_matrix(local_param_dict, drop_list, df):
    print("Handling correlation matrix")
    temp_col_list = list(df.select_dtypes(include=[float, int]).columns)

    # Calculate absolute correlation matrix
    corr_matrix = df.select_dtypes(include=[float, int]).corr().abs()

    # Create a True/False mask and apply it
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # List column names of highly correlated features (r > local_param_dict['corr'])
    corr_to_drop = [
        c for c in tri_df.columns if any(tri_df[c] > local_param_dict.get("corr"))
    ]

    print(
        f"Identified {len(corr_to_drop)} correlated features to drop at >{local_param_dict.get('corr')}"
    )
    drop_list.extend(corr_to_drop)

    return drop_list

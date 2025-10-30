from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler


class data_scale_methods:
    """A class for applying scaling methods to data."""

    def __init__(self) -> None:
        """Initializes the data_scale_methods class."""
        pass

    def standard_scale_method(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies StandardScaler to numeric columns of a DataFrame.

        This method identifies numeric columns and applies standard scaling.
        Non-numeric columns are passed through without modification.

        Args:
            X (pd.DataFrame): The input DataFrame to scale.

        Returns:
            pd.DataFrame: The scaled DataFrame, with numeric columns scaled and
            non-numeric columns preserved. Note: The column order may change.
        """
        # Separate numeric and non-numeric columns
        numeric_cols: List[str] = selector(dtype_exclude=object)(X)
        non_numeric_cols: List[str] = [
            col for col in X.columns if col not in numeric_cols
        ]

        # Define transformers
        transformers = [("scaler", StandardScaler(), numeric_cols)]

        # Include non-numeric columns to be passed through
        if non_numeric_cols:
            transformers.append(("passthrough", "passthrough", non_numeric_cols))

        # Apply transformations
        preprocessor = ColumnTransformer(transformers)
        X_scaled = preprocessor.fit_transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols + non_numeric_cols)

        return X_scaled

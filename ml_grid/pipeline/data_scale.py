import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector


class data_scale_methods:

    def __init__(self):
        """ "data scaling methods"""

    def standard_scale_method(self, X):
        # Separate numeric and non-numeric columns
        numeric_cols = selector(dtype_exclude=object)(X)
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]

        # Define transformers
        transformers = [("somename", StandardScaler(), numeric_cols)]

        # Include non-numeric columns to be passed through
        if non_numeric_cols:
            transformers.append(("passthrough", "passthrough", non_numeric_cols))

        # Apply transformations
        scaler = ColumnTransformer(transformers)
        X_scaled = scaler.fit_transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols + non_numeric_cols)

        return X_scaled

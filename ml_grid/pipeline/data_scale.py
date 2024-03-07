import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler


class data_scale_methods:

    def __init__(self):
        """ "data scaling methods"""

    def standard_scale_method(self, X):

        # can add param dict method for split

        col_names = X.columns
        scaler = ColumnTransformer(
            [
                (
                    "somename",
                    StandardScaler(),
                    list(X.columns),
                )
            ],
            remainder="passthrough",
        )
        X = scaler.fit_transform(X)

        X = pd.DataFrame(X, columns=col_names)

        return X

import numpy as np
import pandas as pd
import sklearn


class feature_methods:

    def __init__(self):
        """set 100% for all, if not 100 then pass to function, always % of n input features. Calculate dynamically."""

    def getNfeaturesANOVAF(self, n, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
            X_train = X_train.values
        elif isinstance(X_train, np.ndarray):
            feature_names = np.arange(X_train.shape[1])
        else:
            raise ValueError("X_train must be a pandas DataFrame or numpy array")

        res = []
        for i, col in enumerate(X_train.T):
            res.append(
                (
                    feature_names[i],
                    sklearn.feature_selection.f_classif(col.reshape(-1, 1), y_train)[0],
                )
            )

        sortedList = sorted(res, key=lambda X: X[1], reverse=True)
        nFeatures = sortedList[:n]
        finalColNames = [elem[0] for elem in nFeatures]
        return finalColNames

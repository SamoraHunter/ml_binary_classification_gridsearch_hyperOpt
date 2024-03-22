import numpy as np
import sklearn


class feature_methods:

    def __init__(self):
        """set 100% for all, if not 100 then pass to function, always % of n input features. Calculate dynamically."""

    def getNfeaturesANOVAF(self, n, X_train, y_train):
        res = []
        for colName in X_train.columns:
            if colName != "intercept":
                res.append(
                    (
                        colName,
                        sklearn.feature_selection.f_classif(
                            np.array(X_train[colName]).reshape(-1, 1), y_train
                        )[0],
                    )
                )
        sortedList = sorted(res, key=lambda X: X[1])
        sortedList.reverse()
        nFeatures = sortedList[:n]
        finalColNames = []
        for elem in nFeatures:
            finalColNames.append(elem[0])
        return finalColNames

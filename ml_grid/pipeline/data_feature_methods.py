import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_selection


class feature_methods:

    def __init__(self):
        """set 100% for all, if not 100 then pass to function, always % of n input features. Calculate dynamically."""

    def getNfeaturesANOVAF(self, n, X_train, y_train):
        """
        Get the top n features based on the ANOVA F-value
        for classification problem.

        The ANOVA F-value is calculated for each feature in X_train
        and the resulting F-values are sorted in descending order.
        The top n features with the highest F-values are returned.

        Parameters
        ----------
        n : int
            Number of top features to return.
        X_train : array-like of shape (n_samples, n_features)
            Training data. Can be a pandas DataFrame or a numpy array.
        y_train : array-like of shape (n_samples,)
            Target variable.

        Returns
        -------
        finalColNames : list
            List of column names of top n features.
            If X_train is a pandas DataFrame, the column names
            are used, otherwise the column indices are used.
        """

        # check if input is a pandas DataFrame or numpy array
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns  # get column names
            X_train = X_train.values  # convert to numpy array
        elif isinstance(X_train, np.ndarray):
            feature_names = np.arange(X_train.shape[1])  # use indices as column names
        else:
            raise ValueError("X_train must be a pandas DataFrame or numpy array")

        # calculate F-value for each column in X_train
        # F-value is calculated by sklearn.feature_selection.f_classif
        # input is a 2D numpy array and target variable y_train
        # output is a 1D numpy array of F-values
        res = []
        for i, col in enumerate(X_train.T):
            res.append(
                (
                    feature_names[i],  # add column name or index to tuple
                    sklearn.feature_selection.f_classif(col.reshape(-1, 1), y_train)[0],
                )
            )

        # sort the list based on F-value in descending order
        sortedList = sorted(res, key=lambda X: X[1], reverse=True)
        print(sortedList)
        # return column names of top n features
        nFeatures = sortedList[:n]  # get top n features
        finalColNames = [elem[0] for elem in nFeatures]  # get column names
        return finalColNames

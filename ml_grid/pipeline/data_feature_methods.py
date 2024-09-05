import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_selection
from PyImpetus import PPIMBC
from sklearn.svm import SVC
import pandas as pd

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


    def getNFeaturesMarkovBlanket(self, n, X_train, y_train):

        """
        Get the names of the top n features from the Markov Blanket (MB) using PyImpetus.

        Parameters:
        - n (int): The number of top features to retrieve.
        - X_train (array-like): The training input samples.
        - y_train (array-like): The target values.

        Returns:
        - list: A list containing the names of the top n features from the Markov Blanket.

        Example:
        ```
        # Import necessary modules
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        # Generate synthetic data for binary classification
        X, y = make_classification(n_samples=1500, n_features=20, n_informative=8, n_classes=2, random_state=42)

        # Split the data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42)

        # Get the top 5 features from the Markov Blanket
        top_features = getNFeaturesMarkovBlanket(5, X_train, y_train)
        ```
        """
        
        # Initialize the PyImpetus object with desired parameters
        model = PPIMBC(model=SVC(random_state=27, class_weight="balanced"), 
                    p_val_thresh=0.05, 
                    num_simul=30, 
                    simul_size=0.2, 
                    simul_type=0, 
                    sig_test_type="non-parametric", 
                    cv=5, 
                    random_state=27, 
                    n_jobs=-1, 
                    verbose=2)
        
        # Fit and transform the training data
        df_train_transformed = model.fit_transform(X_train, y_train)
        
        # Get the feature names from the Markov blanket (MB) and truncate by n elements
        feature_names = model.MB[:n]
        
        return feature_names

    




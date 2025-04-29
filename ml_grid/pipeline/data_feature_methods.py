import numpy as np
import pandas as pd
from PyImpetus import PPIMBC
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif

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

        # Check if input is a pandas DataFrame or numpy array
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns  # Get column names
            X_train = X_train.values  # Convert to numpy array
        elif isinstance(X_train, np.ndarray):
            feature_names = np.arange(X_train.shape[1])  # Use indices as column names
        else:
            raise ValueError("X_train must be a pandas DataFrame or numpy array")

        # Calculate F-value for each column in X_train
        res = []
        for i, col in enumerate(X_train.T):
            # Get the F-values from f_classif
            f_values = f_classif(col.reshape(-1, 1), y_train)[0]

            # If the F-value is not NaN, add it to the results
            if not np.isnan(f_values[0]):
                res.append((feature_names[i], f_values[0]))

        # Sort the list based on F-value in descending order
        sortedList = sorted(res, key=lambda x: x[1], reverse=True)

        # Return column names of top n features
        nFeatures = sortedList[:n]  # Get top n features
        finalColNames = [elem[0] for elem in nFeatures]  # Get column names

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

    




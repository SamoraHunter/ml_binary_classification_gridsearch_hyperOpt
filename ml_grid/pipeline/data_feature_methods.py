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

        # Calculate F-values for all features at once
        f_values, _ = f_classif(X_train, y_train)

        # Create a list of (feature_name, f_value) tuples, ignoring NaNs
        res = [
            (feature_names[i], f_values[i])
            for i in range(len(feature_names))
            if not np.isnan(f_values[i])
        ]

        # Sort the list based on F-value in descending order
        sortedList = sorted(res, key=lambda x: x[1], reverse=True)

        # Return column names of top n features
        nFeatures = sortedList[:n]  # Get top n features
        finalColNames = [elem[0] for elem in nFeatures]

        # Add a check to ensure that at least one feature is returned.
        # If not, it means all features were filtered out (e.g., all had NaN F-values),
        # which would lead to an empty X_train and cause pipeline failure.
        if not finalColNames:
            # Fallback: if all features were filtered, return the single best one that is not NaN.
            # This can happen if n is too small or all f-values are NaN.
            if sortedList:
                return [sortedList[0][0]]
            else:
                raise ValueError("getNfeaturesANOVAF returned no features. All features might have NaN F-values.")

        return finalColNames



    def getNFeaturesMarkovBlanket(
        self,
        n,
        X_train,
        y_train,
        num_simul: int = 30,
        cv: int = 5,
        svc_kernel: str = "rbf",
    ):

        """
        Get the names of the top n features from the Markov Blanket (MB) using PyImpetus.

        Parameters:
        - n (int): The number of top features to retrieve.
        - X_train (array-like): The training input samples.
        - y_train (array-like): The target values.
        - num_simul (int): Number of simulations for stability selection in PyImpetus.
        - cv (int): Number of cross-validation folds.
        - svc_kernel (str): The kernel to be used by the SVC model.

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
        # Ensure input is a pandas DataFrame to access column names
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "X_train must be a pandas DataFrame for getNFeaturesMarkovBlanket."
            )
        original_columns = X_train.columns
        
        # Initialize the PyImpetus object with desired parameters
        model = PPIMBC(model=SVC(random_state=27, class_weight="balanced", kernel=svc_kernel),
                    p_val_thresh=0.05, 
                    num_simul=num_simul,
                    simul_size=0.2, 
                    simul_type=0, 
                    sig_test_type="non-parametric", 
                    cv=cv,
                    random_state=27, 
                    n_jobs=-1, 
                    verbose=2)
        
        # Fit and transform the training data
        # PyImpetus works with numpy arrays and returns feature indices in model.MB
        model.fit(X_train.values, y_train)
        
        # Get the feature indices from the Markov blanket (MB)
        feature_indices = model.MB

        # Map indices back to original column names and truncate by n
        feature_names = [original_columns[i] for i in feature_indices][:n]

        # Fallback: If feature selection returns an empty list, but the model found features,
        # return the single most important one. This prevents pipeline failure.
        if not feature_names and feature_indices:
            feature_names = [original_columns[feature_indices[0]]]
        
        return feature_names

    

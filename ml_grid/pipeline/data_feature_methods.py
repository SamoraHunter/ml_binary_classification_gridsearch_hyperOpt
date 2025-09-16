from typing import List, Union

import numpy as np
import pandas as pd
from PyImpetus import PPIMBC
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC


class feature_methods:
    def __init__(self) -> None:
        """Initializes the feature_methods class."""
        pass

    def getNfeaturesANOVAF(
        self, n: int, X_train: Union[pd.DataFrame, np.ndarray], y_train: pd.Series
    ) -> List[str]:
        """Gets the top n features based on the ANOVA F-value.

        This method is for classification problems. The ANOVA F-value is
        calculated for each feature in X_train, and the resulting F-values are
        sorted in descending order. The top n features with the highest F-values
        are returned.

        Args:
            n (int): The number of top features to return.
            X_train (Union[pd.DataFrame, np.ndarray]): Training data.
            y_train (pd.Series): Target variable.

        Raises:
            ValueError: If X_train is not a pandas DataFrame or numpy array, or
                if no features can be returned (e.g., all have NaN F-values).

        Returns
            List[str]: A list of column names for the top n features.
        """

        # Check if input is a pandas DataFrame or numpy array
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns  # Get column names
            X_train = X_train.values  # Convert to numpy array
        elif isinstance(X_train, np.ndarray):
            feature_names = np.arange(
                X_train.shape[1]
            )  # Use indices as column names
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
                raise ValueError(
                    "getNfeaturesANOVAF returned no features. All features might have NaN F-values."
                )

        return finalColNames



    def getNFeaturesMarkovBlanket(
        self,
        n: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        num_simul: int = 30,
        cv: int = 5,
        svc_kernel: str = "rbf",
    ) -> List[str]:
        """Gets the top n features from the Markov Blanket (MB) using PyImpetus.

        Args:
            n (int): The number of top features to retrieve.
            X_train (pd.DataFrame): The training input samples.
            y_train (pd.Series): The target values.
            num_simul (int): Number of simulations for stability selection in
                PyImpetus. Defaults to 30.
            cv (int): Number of cross-validation folds. Defaults to 5.
            svc_kernel (str): The kernel to be used by the SVC model.
                Defaults to "rbf".

        Raises:
            TypeError: If X_train is not a pandas DataFrame.

        Returns:
            List[str]: A list containing the names of the top n features from
            the Markov Blanket.
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

    

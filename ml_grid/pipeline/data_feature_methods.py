import logging
from typing import List, Union

import numpy as np
import pandas as pd
from PyImpetus import PPIMBC
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
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
            feature_names = np.arange(X_train.shape[1])  # Use indices as column names
        else:
            raise ValueError("X_train must be a pandas DataFrame or numpy array")

        # Ensure at least one feature is requested
        n = max(1, n)

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
        classifier=None,
        num_simul: int = 30,
        cv: int = 5,
        svc_kernel: str = "rbf",
        suppress_print: bool = True,
    ) -> List[str]:
        """Gets the top n features from the Markov Blanket (MB) using PyImpetus.

        Args:
            n (int): The number of top features to retrieve.
            X_train (pd.DataFrame): The training input samples.
            y_train (pd.Series): The target values.
            classifier: The classifier to use for feature selection. If None,
                defaults to SVC.
            num_simul (int): Number of simulations for stability selection in
                PyImpetus. Defaults to 30.
            cv (int): Number of cross-validation folds. Defaults to 5.
            svc_kernel (str): The kernel to be used by the SVC model.
                Defaults to "rbf".
            suppress_print (bool): If True, suppresses stdout from the fit method.
                Defaults to True.

        Raises:
            TypeError: If X_train is not a pandas DataFrame.

        Returns:
            List[str]: A list containing the names of the top n features from
            the Markov Blanket.
        """
        import os

        # Ensure input is a pandas DataFrame to access column names
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "X_train must be a pandas DataFrame for getNFeaturesMarkovBlanket."
            )
        original_columns = X_train.columns

        # Use StratifiedKFold to ensure class distribution is maintained in each fold,
        # preventing errors when a fold contains only one class.
        stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=27)

        # Use the provided classifier, or default to SVC if none is given.
        # CRITICAL: Set verbose=False for SVC to prevent LibSVM output
        model_to_use = (
            classifier
            if classifier is not None
            else SVC(
                random_state=27,
                class_weight="balanced",
                kernel=svc_kernel,
                verbose=False,  # This is the key parameter for LibSVM
            )
        )

        # Ensure verbose is set to False at multiple levels
        if hasattr(model_to_use, "set_params"):
            try:
                model_to_use.set_params(verbose=False)
            except:
                pass
        if hasattr(model_to_use, "verbose"):
            model_to_use.verbose = False

        # Suppress output at the OS level BEFORE creating any model objects
        devnull_fd = None
        old_stdout_fd = None
        old_stderr_fd = None

        if suppress_print:
            try:
                # Save original file descriptors
                old_stdout_fd = os.dup(1)
                old_stderr_fd = os.dup(2)

                # Open devnull and redirect stdout/stderr to it
                devnull_fd = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
            except Exception as e:
                # If suppression fails, just continue without it
                logging.getLogger("ml_grid").warning(f"Could not suppress output: {e}")
                suppress_print = False

        try:
            # Initialize the PyImpetus object with desired parameters
            model = PPIMBC(
                model=model_to_use,
                p_val_thresh=0.05,
                num_simul=num_simul,
                simul_size=0.2,
                simul_type=0,
                sig_test_type="non-parametric",
                cv=stratified_kfold,
                random_state=27,
                n_jobs=-1,
                verbose=0,
            )

            # Fit the model (this is where LibSVM prints)
            model.fit(X_train.values, y_train)

        except ValueError as e:
            # Restore output before logging
            if suppress_print and old_stdout_fd is not None:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                if devnull_fd is not None:
                    os.close(devnull_fd)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)

            # This handles cases where PyImpetus fails due to numerical precision
            # issues (e.g., y_prob > 1). We'll log the error and fall back to
            # using all original features for this trial.
            logging.getLogger("ml_grid").error(
                f"PyImpetus failed during fit: {e}. Using all features as a fallback."
            )
            return list(original_columns)
        finally:
            # Always restore stdout/stderr
            if suppress_print and old_stdout_fd is not None:
                try:
                    os.dup2(old_stdout_fd, 1)
                    os.dup2(old_stderr_fd, 2)
                    if devnull_fd is not None:
                        os.close(devnull_fd)
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                except:
                    pass  # Silently fail if restoration doesn't work

        # Get the feature indices from the Markov blanket (MB)
        selected_features = model.MB

        # PyImpetus can return column names (str) or indices (int).
        # We need to handle both cases to get the final list of feature names.
        if all(isinstance(f, int) for f in selected_features):
            # It returned indices, so map them to names
            feature_names = [original_columns[i] for i in selected_features][:n]
        else:
            # It returned names directly
            feature_names = list(selected_features)[:n]

        # Fallback: If feature selection returns an empty list, but the model found features,
        # return the single most important one. This prevents pipeline failure.
        if not feature_names and selected_features:
            # Re-evaluate the first selected feature to ensure it's a valid name
            first_feature = selected_features[0]
            feature_names = [
                (
                    original_columns[first_feature]
                    if isinstance(first_feature, int)
                    else first_feature
                )
            ]
        return feature_names

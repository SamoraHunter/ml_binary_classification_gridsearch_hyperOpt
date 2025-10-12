from typing import Any, Tuple

import pandas as pd
from ml_grid.pipeline.data_feature_methods import feature_methods

# rename this class


class feature_importance_methods:
    """A class to handle feature selection using different importance methods."""

    def __init__(self) -> None:
        """Initializes the feature_importance_methods class."""
        pass

    def handle_feature_importance_methods(
        self,
        target_n_features: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_train: pd.Series,
        X_test_orig: pd.DataFrame,
        ml_grid_object: Any,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        """Applies a feature selection method to reduce the number of features.

        This method selects features based on the method specified in the
        `ml_grid_object`'s parameters (e.g., 'anova' or 'markov_blanket') and
        reduces the datasets (X_train, X_test, X_test_orig) to the top
        `target_n_features`.

        Args:
            target_n_features (int): The target number of features to select.
            X_train (pd.DataFrame): The training feature data.
            X_test (pd.DataFrame): The testing feature data.
            y_test (pd.Series): The testing target data.
            y_train (pd.Series): The training target data.
            X_test_orig (pd.DataFrame): The original (unsplit) testing feature
                data.
            ml_grid_object (Any): The main pipeline object containing parameters
                and other data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]: 
            A tuple containing the modified X_train, aligned y_train, X_test, 
            aligned y_test, and X_test_orig DataFrames with selected features.
        """

        # Work with copies to avoid modifying the original DataFrames in the calling scope
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        X_test_orig_copy = X_test_orig.copy()

        self.feature_method = ml_grid_object.local_param_dict.get("feature_selection_method")

        if self.feature_method == "anova" or self.feature_method is None:
            print("feature_method ANOVA") 
            fm = feature_methods()
            # The data pipeline now guarantees a clean index, so no reset is needed here.
            features = fm.getNfeaturesANOVAF(n=target_n_features, X_train=X_train_copy, y_train=y_train)

        elif self.feature_method == "markov_blanket":
            print("feature method Markov") 
            fm = feature_methods()
            # The data pipeline now guarantees a clean index, so no reset is needed here.
            features = fm.getNFeaturesMarkovBlanket(n=target_n_features, X_train=X_train_copy, y_train=y_train)

        print(f"target_n_features: {target_n_features}")
        print(f"Selected features: {features}")

        # CRITICAL FIX: Apply feature selection to the X_train that was passed in,
        # which has already been cleaned of post-split constant columns.
        X_train_out = X_train_copy[features]

        # Apply the same feature selection to the test sets
        X_test_out = X_test.copy()[features]
        X_test_orig_out = X_test_orig.copy()[features]

        # The y series do not need to be modified, as they are already aligned.
        return X_train_out, y_train, X_test_out, y_test, X_test_orig_out

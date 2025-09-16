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
        y_train: pd.Series,
        X_test_orig: pd.DataFrame,
        ml_grid_object: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Applies a feature selection method to reduce the number of features.

        This method selects features based on the method specified in the
        `ml_grid_object`'s parameters (e.g., 'anova' or 'markov_blanket') and
        reduces the datasets (X_train, X_test, X_test_orig) to the top
        `target_n_features`.

        Args:
            target_n_features (int): The target number of features to select.
            X_train (pd.DataFrame): The training feature data.
            X_test (pd.DataFrame): The testing feature data.
            y_train (pd.Series): The training target data.
            X_test_orig (pd.DataFrame): The original (unsplit) testing feature
                data.
            ml_grid_object (Any): The main pipeline object containing parameters
                and other data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
            the modified X_train, X_test, and X_test_orig DataFrames with
            the selected features.
        """

        feature_method = ml_grid_object.local_param_dict.get("feature_selection_method")

        if feature_method == "anova" or feature_method is None:
            print("feature_method ANOVA") 
            fm = feature_methods() 
            features = fm.getNfeaturesANOVAF(n=target_n_features, X_train=X_train, y_train=y_train)

        elif feature_method == "markov_blanket":
            print("feature method Markov") 
            fm = feature_methods() 
            features = fm.getNFeaturesMarkovBlanket(n=target_n_features, X_train=X_train, y_train=y_train)

        print(f"target_n_features: {target_n_features}")
        print(f"Selected features: {features}")

        X_train = X_train[features]

        X_test = X_test[features]

        X_test_orig = X_test_orig[features]

        return X_train, X_test, X_test_orig

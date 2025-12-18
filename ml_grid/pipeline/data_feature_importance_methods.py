import logging
from typing import Any, Tuple

import pandas as pd

from ml_grid.pipeline.data_feature_methods import feature_methods


class feature_importance_methods:
    """A class to handle feature selection using different importance methods."""

    def __init__(self) -> None:
        """Initializes the feature_importance_methods class."""
        self.feature_method = "None"

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

        logger = logging.getLogger("ml_grid")

        # Work with copies to avoid modifying the original DataFrames in the calling scope
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        X_test_orig_copy = X_test_orig.copy()

        self.feature_method = ml_grid_object.local_param_dict.get(
            "feature_selection_method"
        )

        # Default to all features initially
        features = list(X_train_copy.columns)

        if self.feature_method == "anova" or self.feature_method is None:
            logger.info("feature_method ANOVA")
            fm = feature_methods()
            features = fm.getNfeaturesANOVAF(
                n=target_n_features, X_train=X_train_copy, y_train=y_train
            )

        elif self.feature_method == "markov_blanket":
            logger.info("feature method Markov")
            fm = feature_methods()
            features = fm.getNFeaturesMarkovBlanket(
                n=target_n_features, X_train=X_train_copy, y_train=y_train
            )

        logger.info(f"target_n_features: {target_n_features}")

        # --- Column Validation ---
        # Filter the requested 'features' to ensure they actually exist in the DataFrame.
        # This handles cases where selectors return indices, 'ColumnX' names, or
        # names that were dropped/renamed in previous pipeline steps.

        valid_features = [f for f in features if f in X_train_copy.columns]

        if len(valid_features) == 0:
            logger.warning(
                f"Feature selection ({self.feature_method}) returned 0 valid features. "
                f"Requested examples: {features[:5] if features else 'None'}. "
                "Falling back to ALL original features to prevent crash."
            )
            valid_features = list(X_train_copy.columns)
        elif len(valid_features) < len(features):
            logger.warning(
                f"{len(features) - len(valid_features)} selected features were not found in X_train columns. Dropped invalid keys."
            )

        logger.info(
            f"Final selected features ({len(valid_features)}): {valid_features}"
        )

        # Apply the validated selection
        X_train_out = X_train_copy[valid_features]
        X_test_out = X_test_copy[valid_features]
        X_test_orig_out = X_test_orig_copy[valid_features]

        # The y series do not need to be modified, as they are already aligned.
        return X_train_out, y_train, X_test_out, y_test, X_test_orig_out

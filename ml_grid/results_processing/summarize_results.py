# summarize_results.py
"""
Module for creating tabular summaries from ML results data.
"""

import pandas as pd
from typing import Optional, List
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

from ml_grid.results_processing.core import get_clean_data


class ResultsSummarizer:
    """Provides methods to summarize and transform results data into concise DataFrames."""

    def __init__(self, data: pd.DataFrame):
        """Initializes the summarizer.

        Args:
            data (pd.DataFrame): Aggregated results DataFrame.

        Raises:
            ValueError: If the input data is not a non-empty pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
        self.data = data
        self.clean_data = get_clean_data(data)

    def get_best_model_per_outcome(self, metric: str = "auc") -> pd.DataFrame:
        """Finds the best model for each outcome and expands the feature list.

        This method identifies the single best-performing model run for each
        outcome variable based on the specified metric. It then transforms the
        'decoded_features' list into a set of boolean columns, where each new
        column represents a feature and its value indicates whether that feature
        was used in the best model run.

        Args:
            metric (str, optional): The performance metric to use for determining
                the "best" model. Defaults to 'auc'.

        Returns:
            pd.DataFrame: A DataFrame containing the best model run for each outcome, with
            additional boolean columns for each feature.
        """
        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("Data must contain an 'outcome_variable' column.")
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in the data.")
        if "decoded_features" not in self.clean_data.columns:
            raise ValueError(
                "Data must contain a 'decoded_features' column. "
                "Ensure ResultsAggregator was run with a feature names CSV."
            )

        # 1. Find the index of the maximum metric value for each outcome group
        best_indices = self.clean_data.groupby("outcome_variable")[metric].idxmax()
        best_models_df = self.clean_data.loc[best_indices].copy()

        # 2. Convert 'decoded_features' into boolean columns
        # Handle rows where 'decoded_features' might be NaN or not a list
        feature_lists = best_models_df["decoded_features"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        if feature_lists.empty or feature_lists.apply(len).sum() == 0:
            warnings.warn(
                "No features found in 'decoded_features' for the best models. Returning summary without feature columns.",
                stacklevel=2,
            )
            return best_models_df

        # Use MultiLabelBinarizer to create the feature indicator matrix
        mlb = MultiLabelBinarizer()

        feature_indicator_df = pd.DataFrame(
            mlb.fit_transform(feature_lists),
            columns=mlb.classes_,
            index=feature_lists.index,
        ).astype(bool)

        # 3. Combine the original best models data with the new feature columns
        # Use rsuffix to handle cases where a feature name (e.g., 'age') conflicts
        # with a feature category column name in the main dataframe.
        result_df = best_models_df.join(feature_indicator_df, rsuffix="_feature")

        # Drop original list-based columns for clarity and sort
        result_df = result_df.drop(
            columns=["f_list", "decoded_features"], errors="ignore"
        )

        return result_df.sort_values(by=metric, ascending=False)

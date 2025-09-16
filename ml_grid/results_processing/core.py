# core.py
"""
Core module for ML results aggregation and management.

Handles loading, aggregating, and basic processing of results data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
import re
import ast


class ResultsAggregator:
    """Aggregates ML results from a hierarchical folder structure.

    This class is designed to find and consolidate results from multiple
    experiment runs, where each run is stored in a timestamped subfolder
    containing a `final_grid_score_log.csv` file.

    The expected structure is: `root_folder/timestamp_folder/final_grid_score_log.csv`
    """

    def __init__(
        self, root_folder: str, feature_names_csv: Optional[str] = None
    ) -> None:
        """Initializes the ResultsAggregator.

        Args:
            root_folder (str): The path to the master root folder containing
                experiment run subfolders.
            feature_names_csv (Optional[str], optional): The path to a CSV file
                whose headers are the original feature names. This is required
                for decoding feature lists. Defaults to None.
        """
        self.root_folder = Path(root_folder)
        self.feature_names: Optional[List[str]] = None
        self.aggregated_data: Optional[pd.DataFrame] = None

        if feature_names_csv:
            self.load_feature_names(feature_names_csv)

    def load_feature_names(self, feature_names_csv: str) -> None:
        """Loads feature names from the column headers of a CSV file.

        Args:
            feature_names_csv (str): The path to the CSV file.
        """
        try:
            # Reading with nrows=0 is an efficient way to get only the headers.
            feature_df = pd.read_csv(feature_names_csv, nrows=0)
            self.feature_names = feature_df.columns.tolist()
            print(f"Loaded {len(self.feature_names)} feature names from CSV columns.")
        except Exception as e:
            warnings.warn(
                f"Could not load feature names from {feature_names_csv}: {e}",
                stacklevel=2,
            )

    def get_available_runs(self) -> List[str]:
        """Gets a list of available timestamped run folders.

        Returns:
            List[str]: A sorted list of valid run folder names.

        Raises:
            ValueError: If the root folder does not exist.
        """
        if not self.root_folder.exists():
            raise ValueError(f"Root folder {self.root_folder} does not exist")

        timestamp_folders = []
        for item in self.root_folder.iterdir():
            if item.is_dir():
                log_file = item / "final_grid_score_log.csv"
                if log_file.exists():
                    timestamp_folders.append(item.name)

        return sorted(timestamp_folders)

    def load_single_run(self, timestamp_folder: str) -> pd.DataFrame:
        """Loads results from a specific timestamped run folder.

        Args:
            timestamp_folder (str): The name of the run folder.

        Returns:
            pd.DataFrame: A DataFrame containing the results for that run.

        Raises:
            FileNotFoundError: If the log file does not exist in the folder.
        """
        log_path = self.root_folder / timestamp_folder / "final_grid_score_log.csv"

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        df = pd.read_csv(log_path)
        df["run_timestamp"] = timestamp_folder

        # Parse feature lists if they exist and feature names are available
        if "f_list" in df.columns and self.feature_names is not None:
            df["decoded_features"] = df["f_list"].apply(self._decode_features)

        return df

    def aggregate_all_runs(self) -> pd.DataFrame:
        """Aggregates results from all available runs in the root folder.

        Returns:
            pd.DataFrame: A single DataFrame containing all aggregated results.

        Raises:
            ValueError: If no valid runs are found.
        """
        available_runs = self.get_available_runs()
        if not available_runs:
            raise ValueError("No valid runs found in the root folder")
        return self.aggregate_specific_runs(available_runs)

    def aggregate_specific_runs(self, run_names: List[str]) -> pd.DataFrame:
        """Aggregates results from a specified list of run folders.

        Args:
            run_names (List[str]): A list of run folder names to aggregate.

        Returns:
            pd.DataFrame: A single DataFrame containing the aggregated results.

        Raises:
            ValueError: If no data could be loaded from the specified runs.
        """
        all_dataframes = []

        for run in run_names:
            try:
                df = self.load_single_run(run)
                all_dataframes.append(df)
                print(f"Loaded run: {run} ({len(df)} records)")
            except Exception as e:
                print(f"Warning: Could not load run {run}: {e}")

        if not all_dataframes:
            raise ValueError("No data could be loaded from the specified runs")

        self.aggregated_data = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total aggregated records: {len(self.aggregated_data)}")
        return self.aggregated_data

    def _decode_features(self, feature_string: str) -> List[str]:
        """Decodes a feature list string into a list of active feature names.

        This method follows a three-step process:
        1.  Parse the string representation (e.g., "[[0, 1, 0]]") into a
            simple binary list.
        2.  Validate that the length of the binary list matches the master list
            of feature names.
        3.  Map the binary list to the feature names, selecting names where the
            flag is 1.

        Args:
            feature_string (str): String representation of a feature list,
                like "[[0, 1, 0, 0, 1]]".

        Returns:
            List[str]: A list of feature names that were selected (where the
            value is 1).
        """
        if self.feature_names is None:
            warnings.warn(
                "Cannot decode features: feature_names are not loaded. "
                "Please ensure a feature_names_csv was provided during "
                "ResultsAggregator initialization.",
                stacklevel=2,
            )
            return []

        if not isinstance(feature_string, str) or pd.isna(feature_string):
            return []

        try:
            # Step 1: Convert the string to a simple list of 0s and 1s.
            # Make parsing robust by extracting list content from formats
            # like 'array([[0, 1]])'
            match = re.search(r"(\[.*\])", feature_string)
            parseable_string = match.group(1) if match else feature_string

            binary_map = ast.literal_eval(parseable_string)

            # Flatten nested lists like [[[0, 1, 0]]] down to [0, 1, 0]
            while (
                isinstance(binary_map, (list, tuple))
                and len(binary_map) > 0
                and isinstance(binary_map[0], (list, tuple))
            ):
                binary_map = binary_map[0]

            if not binary_map:
                return []

            # Step 2: Confirm the length of the binary list matches the
            # original feature names.
            if len(binary_map) != len(self.feature_names):
                warnings.warn(
                    f"Feature list length mismatch for string '{feature_string}'. "
                    f"Parsed list has length {len(binary_map)}, but loaded "
                    f"feature names have length {len(self.feature_names)}. "
                    "This will lead to incorrect feature mapping. Returning "
                    "empty list. Please ensure 'feature_names_csv' corresponds "
                    "to the feature set used during the experiment run.",
                    stacklevel=2,
                )
                return []

            # Step 3: Use the map to decode the feature names where the flag is
            # 1.
            # int(flag) handles both integer (1) and float (1.0) representations.
            return [
                name
                for name, flag in zip(self.feature_names, binary_map)
                if int(flag) == 1
            ]

        except (ValueError, SyntaxError, TypeError, IndexError) as e:
            warnings.warn(
                f"Could not decode feature string: '{feature_string}'. It may "
                f"be malformed or contain non-numeric values. Error: {e}",
                stacklevel=2,
            )
            return []

    def get_summary_stats(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Gets summary statistics for the aggregated results.

        Args:
            data (Optional[pd.DataFrame], optional): The DataFrame to summarize.
                If None, uses the internally stored aggregated data.
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing descriptive statistics.

        Raises:
            ValueError: If no data is available.
        """
        if data is None:
            data = self.aggregated_data

        if data is None:
            raise ValueError("No data available. Run aggregate_all_runs() first.")

        # Key metrics to summarize
        metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]
        available_metrics = [col for col in metrics if col in data.columns]

        summary = data[available_metrics].describe()

        # Add additional summary info
        summary.loc["count_runs"] = data["run_timestamp"].nunique()
        summary.loc["count_algorithms"] = data["method_name"].nunique()

        if "outcome_variable" in data.columns:
            summary.loc["count_outcomes"] = data["outcome_variable"].nunique()

        return summary

    def get_outcome_variables(self, data: Optional[pd.DataFrame] = None) -> List[str]:
        """Gets a list of unique outcome variables from the data.

        Args:
            data (Optional[pd.DataFrame], optional): The DataFrame to inspect.
                If None, uses the internally stored aggregated data.
                Defaults to None.

        Returns:
            List[str]: A sorted list of unique outcome variable names.

        Raises:
            ValueError: If no data is available or the 'outcome_variable'
                column is missing.
        """
        if data is None:
            data = self.aggregated_data

        if data is None:
            raise ValueError("No data available.")

        if "outcome_variable" not in data.columns:
            raise ValueError("outcome_variable column not found in data")

        return sorted(data["outcome_variable"].unique().tolist())

    def get_data_by_outcome(
        self, outcome_variable: str, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Filters the data for a specific outcome variable.

        Args:
            outcome_variable (str): The outcome variable to filter by.
            data (Optional[pd.DataFrame], optional): The DataFrame to filter.
                If None, uses the internally stored aggregated data.
                Defaults to None.

        Returns:
            pd.DataFrame: A new DataFrame containing only the data for the
            specified outcome.

        Raises:
            ValueError: If no data is available, the 'outcome_variable' column
                is missing, or no data is found for the specified outcome.
        """
        if data is None:
            data = self.aggregated_data

        if data is None:
            raise ValueError("No data available.")

        if "outcome_variable" not in data.columns:
            raise ValueError("outcome_variable column not found in data")

        filtered_data = data[data["outcome_variable"] == outcome_variable].copy()

        if len(filtered_data) == 0:
            raise ValueError(f"No data found for outcome variable: {outcome_variable}")

        return filtered_data

    def get_outcome_summary(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Gets summary statistics stratified by outcome variable.

        Args:
            data (Optional[pd.DataFrame], optional): The DataFrame to summarize.
                If None, uses the internally stored aggregated data.
                Defaults to None.

        Returns:
            pd.DataFrame: A multi-index DataFrame with summary statistics for
            each outcome variable.

        Raises:
            ValueError: If no data is available or the 'outcome_variable'
                column is missing.
        """
        if data is None:
            data = self.aggregated_data

        if data is None:
            raise ValueError("No data available.")

        if "outcome_variable" not in data.columns:
            raise ValueError("outcome_variable column not found in data")

        # Key metrics to summarize
        metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]
        available_metrics = [col for col in metrics if col in data.columns]

        # Clean data (remove failed runs)
        clean_data = data[data["failed"] == 0] if "failed" in data.columns else data

        # Group by outcome variable and calculate summary stats
        outcome_summary = (
            clean_data.groupby("outcome_variable")[available_metrics]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )

        return outcome_summary


class DataValidator:
    """A utility class for validating and checking the quality of results data."""

    @staticmethod
    def validate_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """Validates the structure and quality of a results DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            Dict[str, Any]: A dictionary containing the validation report.
        """
        validation_report = {
            "total_records": len(df),
            "columns_present": df.columns.tolist(),
            "missing_columns": [],
            "data_quality_issues": [],
            "outcome_variables": None,
            "algorithms": None,
            "runs": None,
        }

        # Expected key columns
        expected_columns = [
            "method_name",
            "auc",
            "f1",
            "precision",
            "recall",
            "accuracy",
        ]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        validation_report["missing_columns"] = missing_columns

        # Check for outcome variables
        if "outcome_variable" in df.columns:
            validation_report["outcome_variables"] = df[
                "outcome_variable"
            ].unique().tolist()

        # Check for algorithms
        if "method_name" in df.columns:
            validation_report["algorithms"] = df["method_name"].unique().tolist()

        # Check for runs
        if "run_timestamp" in df.columns:
            validation_report["runs"] = df["run_timestamp"].unique().tolist()

        # Check for data quality issues
        if "failed" in df.columns:
            failed_count = (df["failed"] == 1).sum()
            if failed_count > 0:
                validation_report["data_quality_issues"].append(
                    f"{failed_count} failed runs detected"
                )

        # Check for missing values in key metrics
        key_metrics = ["auc", "f1", "precision", "recall", "accuracy"]
        for metric in key_metrics:
            if metric in df.columns:
                missing_count = df[metric].isna().sum()
                if missing_count > 0:
                    validation_report["data_quality_issues"].append(
                        f"{missing_count} missing values in {metric}"
                    )

        return validation_report

    @staticmethod
    def print_validation_report(validation_report: Dict[str, Any]) -> None:
        """Prints a formatted validation report to the console.

        Args:
            validation_report (Dict[str, Any]): The validation report dictionary
                generated by `validate_data_structure`.
        """
        print("=" * 50)
        print("DATA VALIDATION REPORT")
        print("=" * 50)

        print(f"Total records: {validation_report['total_records']}")
        print(f"Columns present: {len(validation_report['columns_present'])}")

        if validation_report["missing_columns"]:
            print(f"Missing expected columns: {validation_report['missing_columns']}")

        if validation_report["outcome_variables"]:
            print(
                f"Outcome variables ({len(validation_report['outcome_variables'])}): "
                f"{validation_report['outcome_variables']}"
            )

        if validation_report["algorithms"]:
            print(
                f"Algorithms ({len(validation_report['algorithms'])}): "
                f"{validation_report['algorithms'][:5]}{'...' if len(validation_report['algorithms']) > 5 else ''}"
            )

        if validation_report["runs"]:
            print(
                f"Runs ({len(validation_report['runs'])}): "
                f"{validation_report['runs'][:3]}{'...' if len(validation_report['runs']) > 3 else ''}"
            )

        if validation_report["data_quality_issues"]:
            print("\nData Quality Issues:")
            for issue in validation_report["data_quality_issues"]:
                print(f"  - {issue}")
        else:
            print("\nNo data quality issues detected.")

        print("=" * 50)


def get_clean_data(df: pd.DataFrame, remove_failed: bool = True) -> pd.DataFrame:
    """A utility function to get clean data for analysis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        remove_failed (bool, optional): If True, removes rows where the 'failed'
            column is 1. Defaults to True.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if remove_failed and "failed" in df.columns:
        return df[df["failed"] == 0].copy()
    return df.copy()


def stratify_by_outcome(
    df: pd.DataFrame, func: callable, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Applies a function to data stratified by outcome variable.

    Args:
        df (pd.DataFrame): DataFrame with an 'outcome_variable' column.
        func (callable): The function to apply to each outcome's data subset.
        *args (Any): Positional arguments to pass to the function.
        **kwargs (Any): Keyword arguments to pass to the function.

    Returns:
        Dict[str, Any]: A dictionary with outcome variables as keys and the
        results of the function as values.

    Raises:
        ValueError: If the 'outcome_variable' column is not found.
    """
    if "outcome_variable" not in df.columns:
        raise ValueError("outcome_variable column not found in data")

    results = {}

    for outcome in df["outcome_variable"].unique():
        outcome_data = df[df["outcome_variable"] == outcome]
        try:
            results[outcome] = func(outcome_data, *args, **kwargs)
        except Exception as e:
            print(f"Warning: Could not process outcome {outcome}: {e}")
            results[outcome] = None
    
    return results
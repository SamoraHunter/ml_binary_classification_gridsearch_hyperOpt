"""
Module for generating synthetic datasets that mimic the structure of real-world
data used in the ml-grid pipeline.
"""

import logging
import random
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class SyntheticDataGenerator:
    """
    Generates a synthetic DataFrame for testing the ml-grid pipeline.

    This class creates a dataset with realistic column names and structures,
    including various feature types (blood tests, annotations, etc.), multiple
    configurable outcome variables, and a controllable signal-to-noise ratio.

    Attributes:
        n_rows (int): The number of rows (samples) in the dataset.
        n_features (int): The total number of feature columns to generate.
        n_outcome_vars (int): The number of outcome variable columns to create.
        feature_strength (float): A factor to control the influence of
            "important" features on the outcome variables. Higher values
            create a stronger signal.
        percent_important_features (float): The percentage of total features
            that will be correlated with the outcome variables.
        percent_binary_features (float): The percentage of features that should
            be binary (0 or 1).
        percent_int_features (float): The percentage of features that should be
            integer-based (e.g., counts).
    """

    def __init__(
        self,
        n_rows: int = 1000,
        n_features: int = 150,
        n_outcome_vars: int = 3,
        feature_strength: float = 0.8,
        percent_important_features: float = 0.1,
        percent_binary_features: float = 0.15,
        percent_int_features: float = 0.2,
        verbose: bool = True,
    ):
        """
        Initializes the SyntheticDataGenerator with specified parameters.

        Args:
            n_rows (int): Number of rows for the synthetic dataset.
            n_features (int): Number of feature columns to generate.
            n_outcome_vars (int): Number of outcome variables to generate.
            feature_strength (float): Strength of the signal from important
                features. Must be between 0 and 1.
            percent_important_features (float): Percentage of features that
                should be predictive of the outcome.
            percent_binary_features (float): Percentage of features to be binary.
            percent_int_features (float): Percentage of features to be integer-based.
            verbose (bool): If True, prints generation status messages.
        """
        if not 0 <= feature_strength <= 1:
            raise ValueError("feature_strength must be between 0 and 1.")
        if not 0 <= percent_binary_features + percent_int_features <= 1:
            raise ValueError(
                "The sum of binary and int feature percentages must be <= 1."
            )

        self.n_rows = n_rows
        self.n_features = n_features
        self.n_outcome_vars = n_outcome_vars
        self.feature_strength = feature_strength
        self.percent_important_features = percent_important_features
        self.percent_binary_features = percent_binary_features
        self.percent_int_features = percent_int_features
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Based on ml_grid/pipeline/column_names.py and sample data
        self._feature_prefixes = [
            "Alkaline Phosphatase",
            "RBC",
            "PLT",
            "Sodium",
            "Potassium",
            "C-reactive Protein",
            "Glycated Hb",
            "White Cell Count",
            "Monocytes",
            "MCHC.",
            "Calcium measurement",
            "Neutrophils",
            "Insertion - action (qualifier value)",
            "Routine (qualifier value)",
            "General treatment (procedure)",
            "Research fellow (occupation)",
            "Phlebotomy (procedure)",
            "Date of birth (observable entity)",
            "Antibiotic (product)",
            "Hypercholesterolemia (disorder)",
            "Sinus rhythm (finding)",
            "Capsule (basic dose form)",
            "Transplantation of liver (procedure)",
            "History of clinical finding",
        ]
        self._feature_suffixes = [
            "_mean",
            "_median",
            "_mode",
            "_std",
            "_num-tests",
            "_days-since-last-test",
            "_max",
            "_min",
            "_most-recent",
            "_earliest-test",
            "_days-between-first-last",
            "_contains-extreme-low",
            "_contains-extreme-high",
            "_num-diagnostic-order",
            "_count",
            "_count_subject_present",
            "_count_subject_not_present",
            "_count_relative_present",
            "_count_relative_not_present",
            "_count_subject_present_mrc_cs",
        ]
        # Suffixes that imply integer or binary types
        self._int_suffixes = ["_count", "_num-tests", "_num-diagnostic-order"]
        self._binary_suffixes = ["_contains-extreme-low", "_contains-extreme-high"]

        self._special_features = [
            "age",
            "male",
            "bmi_value",
            "census_ethnicity_white",
            "core_02_val",
            "bed_type_A",
            "vte_status_1",
            "hosp_site_X",
            "core_resus_status",
            "news_score",
            "client_idcode",
        ]

    def _generate_column_names(self) -> List[str]:
        """Generates a list of realistic, structured feature names."""
        self.logger.info(f"Generating {self.n_features} column names...")

        generated_names = []

        # Add some special features first
        num_special = min(len(self._special_features), 5)
        generated_names.extend(random.sample(self._special_features, num_special))

        # Calculate how many more names we need
        remaining = self.n_features - len(generated_names)

        if remaining <= 0:
            return generated_names[: self.n_features]

        # 1. Generate all possible clean combinations (Prefix + Suffix)
        # This avoids _r1 suffixes unless we run out of unique clean names
        clean_combinations = [
            f"{prefix}{suffix}"
            for prefix in self._feature_prefixes
            for suffix in self._feature_suffixes
        ]
        random.shuffle(clean_combinations)

        # Take as many as we need from clean combinations
        num_from_clean = min(len(clean_combinations), remaining)
        generated_names.extend(clean_combinations[:num_from_clean])
        remaining -= num_from_clean

        # 2. If we still need more, use rounds (_r1, _r2, etc.)
        round_num = 1
        max_rounds = 10
        while remaining > 0 and round_num < max_rounds:
            round_candidates = [
                f"{prefix}_r{round_num}{suffix}"
                for prefix in self._feature_prefixes
                for suffix in self._feature_suffixes
            ]
            random.shuffle(round_candidates)

            take_round = min(remaining, len(round_candidates))
            generated_names.extend(round_candidates[:take_round])
            remaining -= take_round
            round_num += 1

        # Final shuffle
        random.shuffle(generated_names)
        return generated_names[: self.n_features]

    def _generate_typed_data(self, feature_names: List[str]) -> np.ndarray:
        """
        Generates data with appropriate types based on column names.

        This is significantly faster than modifying DataFrame columns after creation.
        """
        # Pre-categorize columns to avoid repeated string matching
        age_cols = []
        binary_cols = []
        bmi_cols = []
        int_cols = []
        binary_suffix_cols = []
        normal_cols = []

        self.logger.info("Categorizing columns...")
        for idx, col in enumerate(feature_names):
            if col == "age":
                age_cols.append(idx)
            elif col == "male" or "vte_status" in col or "bed_type" in col:
                binary_cols.append(idx)
            elif col == "bmi_value":
                bmi_cols.append(idx)
            elif any(s in col for s in self._int_suffixes):
                int_cols.append(idx)
            elif any(s in col for s in self._binary_suffixes):
                binary_suffix_cols.append(idx)
            else:
                normal_cols.append(idx)

        # Pre-allocate array
        data = np.empty((self.n_rows, len(feature_names)), dtype=np.float32)

        # Generate data in bulk for each category
        self.logger.info("Generating typed data...")
        if age_cols:
            for idx in age_cols:
                data[:, idx] = np.random.randint(20, 90, size=self.n_rows)

        if binary_cols:
            for idx in binary_cols:
                data[:, idx] = np.random.randint(0, 2, size=self.n_rows)

        if bmi_cols:
            for idx in bmi_cols:
                data[:, idx] = np.random.uniform(18, 45, size=self.n_rows)

        if int_cols:
            for idx in int_cols:
                data[:, idx] = np.random.poisson(5, size=self.n_rows) * random.randint(
                    1, 5
                )

        if binary_suffix_cols:
            for idx in binary_suffix_cols:
                data[:, idx] = np.random.randint(0, 2, size=self.n_rows)

        # Generate all normal columns at once
        if normal_cols:
            self.logger.info(
                f"Generating {len(normal_cols)} normal distribution columns..."
            )
            normal_data = np.random.randn(self.n_rows, len(normal_cols)).astype(
                np.float32
            )
            data[:, normal_cols] = normal_data

        return data

    def generate(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Generates and returns the synthetic DataFrame and a map of important features.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]:
                - The fully generated synthetic dataset.
                - A dictionary mapping each outcome variable to its list of important features.
        """
        self.logger.info(
            f"Starting generation: {self.n_rows} rows × {self.n_features} features"
        )

        # 1. Generate feature names
        self.logger.info("Generating column names...")
        feature_names = self._generate_column_names()

        # 2. Generate typed data directly (much faster than modifying after)
        data = self._generate_typed_data(feature_names)

        self.logger.info("Creating DataFrame...")
        df = pd.DataFrame(data, columns=feature_names)

        # Dictionary to hold outcome variables and metadata
        new_cols_dict = {}
        outcome_to_features_map = {}

        # 3. Determine number of important features
        n_important = max(1, int(self.n_features * self.percent_important_features))

        self.logger.info(f"Generating {self.n_outcome_vars} outcome variables.")

        # 4. Generate outcome variables with progress bar
        for i in tqdm(
            range(1, self.n_outcome_vars + 1),
            desc="Creating outcomes",
            disable=not self.logger.isEnabledFor(logging.INFO),
        ):
            outcome_col_name = f"outcome_var_{i}"

            # Select a unique set of important features for *this* outcome
            important_features = (
                df.columns.to_series()
                .sample(
                    n=n_important, random_state=42 + i  # Use index `i` to vary the seed
                )
                .tolist()
            )
            outcome_to_features_map[outcome_col_name] = important_features

            if i <= 3 or i == self.n_outcome_vars:  # Only log first 3 and last
                self.logger.info(
                    f"  For '{outcome_col_name}', selected {len(important_features)} important features"
                )

            # Create signal from important features (vectorized)
            signal = df[important_features].values.sum(axis=1) * self.feature_strength

            # Create noise
            noise_strength = 1 - self.feature_strength
            noise = (
                np.random.randn(self.n_rows).astype(np.float32)
                * noise_strength
                * signal.std()
            )

            # Combine signal and noise, then create binary outcome
            combined_signal = signal + noise
            threshold = np.median(combined_signal)
            outcome = (combined_signal > threshold).astype(np.int8)

            # Randomly flip 10% of outcomes
            flip_mask = np.random.rand(self.n_rows) < 0.1
            outcome[flip_mask] = 1 - outcome[flip_mask]

            new_cols_dict[outcome_col_name] = outcome

        # 5. Add metadata columns
        self.logger.info("Adding metadata columns...")
        if "client_idcode" not in df.columns:
            new_cols_dict["client_idcode"] = [f"id_{j}" for j in range(self.n_rows)]

        new_cols_dict["Unnamed: 0"] = np.arange(self.n_rows, dtype=np.int32)

        # 6. Concatenate all at once
        self.logger.info("Concatenating final DataFrame...")
        new_cols_df = pd.DataFrame(new_cols_dict, index=df.index)
        df = pd.concat(
            [
                new_cols_df[["Unnamed: 0"]],
                df,
                new_cols_df.drop(columns=["Unnamed: 0"]),
            ],
            axis=1,
        )

        # 7. Introduce missing values (vectorized per column)
        self.logger.info("Introducing missing values...")
        feature_cols = [
            col
            for col in df.columns
            if not (
                col.startswith("outcome_var")
                or col == "client_idcode"
                or col == "Unnamed: 0"
            )
        ]

        cols_with_nans = random.sample(feature_cols, int(len(feature_cols) * 0.15))

        for col in tqdm(
            cols_with_nans,
            desc="Adding NaNs",
            disable=not self.logger.isEnabledFor(logging.INFO),
        ):
            frac = random.uniform(0.01, 0.2)
            n_nans = int(self.n_rows * frac)
            nan_indices = np.random.choice(self.n_rows, size=n_nans, replace=False)
            df.loc[nan_indices, col] = np.nan

        self.logger.info(f"Generation complete! Shape: {df.shape}")
        return df, outcome_to_features_map


class SyntheticTSDataGenerator:
    """
    Generates a synthetic longitudinal (time-series) dataset for testing the
    ml-grid pipeline.

    The output is a long-format 2D DataFrame where each row represents one
    observation for one patient at one timestamp — mirroring the real data
    format exactly. Multiple rows share the same ``client_idcode``, forming
    that patient's time series. Feature columns and outcome generation follow
    the same signal/noise approach as ``SyntheticDataGenerator``.

    Attributes:
        n_instances (int): Number of unique patients (client_idcodes).
        n_timepoints (int): Number of timestamped rows per patient.
        n_features (int): Number of clinical feature columns to generate.
        n_outcome_vars (int): Number of binary outcome columns to generate.
        feature_strength (float): Controls signal strength (0 = pure noise,
            1 = pure signal).
        percent_important_features (float): Fraction of features that influence
            each outcome.
        percent_missing (float): Approximate fraction of feature values to set
            to NaN.
        start_date (str): ISO date string for the first timestamp.
    """

    def __init__(
        self,
        n_instances: int = 200,
        n_timepoints: int = 50,
        n_features: int = 100,
        n_outcome_vars: int = 1,
        feature_strength: float = 0.8,
        percent_important_features: float = 0.1,
        percent_missing: float = 0.1,
        start_date: str = "2022-01-01",
        verbose: bool = True,
    ):
        """
        Initializes the SyntheticTSDataGenerator.

        Args:
            n_instances (int): Number of unique patients.
            n_timepoints (int): Number of daily timestamped rows per patient.
            n_features (int): Number of feature columns to generate.
            n_outcome_vars (int): Number of binary outcome columns to generate.
            feature_strength (float): Strength of the signal from important
                features. Must be between 0 and 1.
            percent_important_features (float): Fraction of features that
                should be predictive of each outcome.
            percent_missing (float): Approximate percentage of feature values
                to set to NaN.
            start_date (str): ISO date string for the first timestamp
                (e.g. ``"2022-01-01"``).
            verbose (bool): If True, enables logging of generation status.
        """
        if not 0 <= feature_strength <= 1:
            raise ValueError("feature_strength must be between 0 and 1.")

        self.n_instances = n_instances
        self.n_timepoints = n_timepoints
        self.n_features = n_features
        self.n_outcome_vars = n_outcome_vars
        self.feature_strength = feature_strength
        self.percent_important_features = percent_important_features
        self.percent_missing = percent_missing
        self.start_date = start_date
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Reuse the same prefixes/suffixes as SyntheticDataGenerator for
        # consistent, realistic column names across both generators.
        self._feature_prefixes = [
            "Alkaline Phosphatase",
            "RBC",
            "PLT",
            "Sodium",
            "Potassium",
            "C-reactive Protein",
            "Glycated Hb",
            "White Cell Count",
            "Monocytes",
            "MCHC.",
            "Calcium measurement",
            "Neutrophils",
            "Insertion - action (qualifier value)",
            "Routine (qualifier value)",
            "General treatment (procedure)",
            "Research fellow (occupation)",
            "Phlebotomy (procedure)",
            "Date of birth (observable entity)",
            "Antibiotic (product)",
            "Hypercholesterolemia (disorder)",
            "Sinus rhythm (finding)",
            "Capsule (basic dose form)",
            "Transplantation of liver (procedure)",
            "History of clinical finding",
        ]
        self._feature_suffixes = [
            "_mean",
            "_median",
            "_mode",
            "_std",
            "_num-tests",
            "_days-since-last-test",
            "_max",
            "_min",
            "_most-recent",
            "_earliest-test",
            "_days-between-first-last",
            "_contains-extreme-low",
            "_contains-extreme-high",
            "_num-diagnostic-order",
            "_count",
            "_count_subject_present",
            "_count_subject_not_present",
            "_count_relative_present",
            "_count_relative_not_present",
            "_count_subject_present_mrc_cs",
        ]
        self._int_suffixes = ["_count", "_num-tests", "_num-diagnostic-order"]
        self._binary_suffixes = ["_contains-extreme-low", "_contains-extreme-high"]
        # Note: client_idcode is excluded — it is added as a structural column,
        # not a randomly sampled feature.
        self._special_features = [
            "age",
            "male",
            "bmi_value",
            "census_ethnicity_white",
            "core_02_val",
            "bed_type_A",
            "vte_status_1",
            "hosp_site_X",
            "core_resus_status",
            "news_score",
        ]

    def _generate_column_names(self) -> List[str]:
        """Generates a list of realistic, structured feature names."""
        self.logger.info(f"Generating {self.n_features} feature column names...")

        generated_names = []
        num_special = min(len(self._special_features), 5)
        generated_names.extend(random.sample(self._special_features, num_special))
        remaining = self.n_features - len(generated_names)

        if remaining <= 0:
            return generated_names[: self.n_features]

        clean_combinations = [
            f"{prefix}{suffix}"
            for prefix in self._feature_prefixes
            for suffix in self._feature_suffixes
        ]
        random.shuffle(clean_combinations)
        num_from_clean = min(len(clean_combinations), remaining)
        generated_names.extend(clean_combinations[:num_from_clean])
        remaining -= num_from_clean

        round_num = 1
        max_rounds = 10
        while remaining > 0 and round_num < max_rounds:
            round_candidates = [
                f"{prefix}_r{round_num}{suffix}"
                for prefix in self._feature_prefixes
                for suffix in self._feature_suffixes
            ]
            random.shuffle(round_candidates)
            take_round = min(remaining, len(round_candidates))
            generated_names.extend(round_candidates[:take_round])
            remaining -= take_round
            round_num += 1

        random.shuffle(generated_names)
        return generated_names[: self.n_features]

    def _generate_typed_data(self, feature_names: List[str], n_rows: int) -> np.ndarray:
        """
        Generates typed feature data for ``n_rows`` rows based on column semantics.

        Mirrors ``SyntheticDataGenerator._generate_typed_data()`` exactly,
        operating on the full flattened row count (``n_instances × n_timepoints``).
        """
        age_cols, binary_cols, bmi_cols = [], [], []
        int_cols, binary_suffix_cols, normal_cols = [], [], []

        for idx, col in enumerate(feature_names):
            if col == "age":
                age_cols.append(idx)
            elif col == "male" or "vte_status" in col or "bed_type" in col:
                binary_cols.append(idx)
            elif col == "bmi_value":
                bmi_cols.append(idx)
            elif any(s in col for s in self._int_suffixes):
                int_cols.append(idx)
            elif any(s in col for s in self._binary_suffixes):
                binary_suffix_cols.append(idx)
            else:
                normal_cols.append(idx)

        data = np.empty((n_rows, len(feature_names)), dtype=np.float32)

        if age_cols:
            for idx in age_cols:
                data[:, idx] = np.random.randint(20, 90, size=n_rows)
        if binary_cols:
            for idx in binary_cols:
                data[:, idx] = np.random.randint(0, 2, size=n_rows)
        if bmi_cols:
            for idx in bmi_cols:
                data[:, idx] = np.random.uniform(18, 45, size=n_rows)
        if int_cols:
            for idx in int_cols:
                data[:, idx] = np.random.poisson(5, size=n_rows) * random.randint(1, 5)
        if binary_suffix_cols:
            for idx in binary_suffix_cols:
                data[:, idx] = np.random.randint(0, 2, size=n_rows)
        if normal_cols:
            data[:, normal_cols] = np.random.randn(n_rows, len(normal_cols)).astype(
                np.float32
            )

        return data

    def generate(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Generates and returns the synthetic longitudinal DataFrame.

        The output is a long-format 2D DataFrame with one row per
        ``(client_idcode, timestamp)`` pair — matching the structure of the
        real ml-grid time-series data exactly. Each patient has exactly
        ``n_timepoints`` consecutive daily rows. Outcome labels are generated
        per-row using the same signal/noise + median-threshold approach as
        ``SyntheticDataGenerator``.

        Column order: ``client_idcode | timestamp | <features> | <outcome_vars>``

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]:
                - The fully generated longitudinal dataset.
                - A dictionary mapping each outcome variable name to its list
                  of important feature names used to construct it.
        """
        from datetime import datetime, timedelta

        total_rows = self.n_instances * self.n_timepoints
        self.logger.info(
            f"Starting TS generation: {self.n_instances} patients × "
            f"{self.n_timepoints} timepoints = {total_rows} total rows, "
            f"{self.n_features} features"
        )

        # 1. Generate feature column names
        feature_names = self._generate_column_names()

        # 2. Generate typed feature data for all rows at once
        self.logger.info("Generating typed feature data...")
        data = self._generate_typed_data(feature_names, total_rows)
        df = pd.DataFrame(data, columns=feature_names)

        # 3. Build client_idcode and timestamp columns.
        #    Each patient gets n_timepoints consecutive daily timestamps.
        self.logger.info("Building client_idcode and timestamp columns...")
        start = datetime.fromisoformat(self.start_date)
        client_ids = np.repeat(np.arange(1, self.n_instances + 1), self.n_timepoints)
        timestamps = np.tile(
            [
                (start + timedelta(days=d)).strftime("%Y-%m-%d")
                for d in range(self.n_timepoints)
            ],
            self.n_instances,
        )

        # 4. Generate outcome variables — per-row, same approach as tabular version
        outcome_to_features_map: dict[str, list[str]] = {}
        new_cols_dict: dict[str, np.ndarray] = {}
        n_important = max(1, int(self.n_features * self.percent_important_features))

        self.logger.info(f"Generating {self.n_outcome_vars} outcome variable(s)...")
        for i in tqdm(
            range(1, self.n_outcome_vars + 1),
            desc="Creating outcomes",
            disable=not self.logger.isEnabledFor(logging.INFO),
        ):
            outcome_col_name = f"outcome_var_{i}"

            # Randomly sample important features for this outcome
            important_features = (
                df.columns.to_series()
                .sample(n=n_important, random_state=42 + i)
                .tolist()
            )
            outcome_to_features_map[outcome_col_name] = important_features
            self.logger.info(
                f"  For '{outcome_col_name}', selected {len(important_features)} important features"
            )

            # Signal: weighted sum of important features across all rows
            signal = df[important_features].values.sum(axis=1) * self.feature_strength

            # Noise: scaled by (1 - feature_strength) and signal std
            noise_strength = 1 - self.feature_strength
            noise = (
                np.random.randn(total_rows).astype(np.float32)
                * noise_strength
                * signal.std()
            )

            # Threshold at median → binary outcome per row
            combined_signal = signal + noise
            threshold = np.median(combined_signal)
            outcome = (combined_signal > threshold).astype(np.int8)

            # Randomly flip 10% of labels
            flip_mask = np.random.rand(total_rows) < 0.1
            outcome[flip_mask] = 1 - outcome[flip_mask]

            new_cols_dict[outcome_col_name] = outcome

        # 5. Introduce missing values into feature columns
        self.logger.info("Introducing missing values...")
        cols_with_nans = random.sample(feature_names, int(len(feature_names) * 0.15))
        for col in tqdm(
            cols_with_nans,
            desc="Adding NaNs",
            disable=not self.logger.isEnabledFor(logging.INFO),
        ):
            frac = random.uniform(0.01, 0.2)
            n_nans = int(total_rows * frac)
            nan_indices = np.random.choice(total_rows, size=n_nans, replace=False)
            df.loc[nan_indices, col] = np.nan

        # 6. Assemble final DataFrame: client_idcode | timestamp | features | outcomes
        self.logger.info("Assembling final DataFrame...")
        outcome_df = pd.DataFrame(new_cols_dict, index=df.index)
        df.insert(0, "timestamp", timestamps)
        df.insert(0, "client_idcode", client_ids)
        df = pd.concat([df, outcome_df], axis=1)

        self.logger.info(f"Generation complete! Shape: {df.shape}")
        return df, outcome_to_features_map


def generate_synthetic_ts_data(
    n_instances: int = 200,
    n_timepoints: int = 50,
    n_features: int = 100,
    n_outcome_vars: int = 1,
    feature_strength: float = 0.8,
    percent_important_features: float = 0.1,
    percent_missing: float = 0.1,
    start_date: str = "2022-01-01",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    A convenience function to generate a synthetic longitudinal dataset.

    The returned DataFrame has one row per ``(client_idcode, timestamp)`` pair,
    matching the structure of real ml-grid time-series data exactly.

    Args:
        n_instances (int): Number of unique patients.
        n_timepoints (int): Number of daily timestamped rows per patient.
        n_features (int): Number of feature columns to generate.
        n_outcome_vars (int): Number of binary outcome columns to generate.
        feature_strength (float): Strength of the signal from important
            features. Must be between 0 and 1.
        percent_important_features (float): Fraction of features that should
            be predictive of each outcome.
        percent_missing (float): Approximate percentage of feature values to
            set to NaN.
        start_date (str): ISO date string for the first timestamp.
        verbose (bool): If True, enables logging of generation status.

    Returns:
        tuple[pd.DataFrame, dict[str, list[str]]]:
            - The generated longitudinal dataset.
            - A dictionary mapping each outcome variable to its important features.
    """
    generator = SyntheticTSDataGenerator(
        n_instances=n_instances,
        n_timepoints=n_timepoints,
        n_features=n_features,
        n_outcome_vars=n_outcome_vars,
        feature_strength=feature_strength,
        percent_important_features=percent_important_features,
        percent_missing=percent_missing,
        start_date=start_date,
        verbose=verbose,
    )
    return generator.generate()


def generate_synthetic_data(
    n_rows: int = 1000,
    n_features: int = 150,
    n_outcome_vars: int = 3,
    feature_strength: float = 0.8,
    percent_important_features: float = 0.1,
    percent_binary_features: float = 0.15,
    percent_int_features: float = 0.2,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    A convenience function to generate a synthetic dataset.

    This function instantiates the SyntheticDataGenerator, calls its generate
    method, and returns the resulting DataFrame.

    Args:
        n_rows (int): Number of rows for the synthetic dataset.
        n_features (int): Number of feature columns to generate.
        n_outcome_vars (int): Number of outcome variables to generate.
        feature_strength (float): Strength of the signal from important
            features. Must be between 0 and 1.
        percent_important_features (float): Percentage of features that
            should be predictive of the outcome.
        percent_binary_features (float): Percentage of features to be binary.
        percent_int_features (float): Percentage of features to be integer-based.
        verbose (bool): If True, enables logging of generation status.


    Returns:
        tuple[pd.DataFrame, dict[str, list[str]]]:
            - The generated synthetic dataset.
            - A dictionary mapping each outcome variable to its list of important features.
    """
    generator = SyntheticDataGenerator(
        n_rows=n_rows,
        n_features=n_features,
        n_outcome_vars=n_outcome_vars,
        feature_strength=feature_strength,
        percent_important_features=percent_important_features,
        percent_binary_features=percent_binary_features,
        percent_int_features=percent_int_features,
        verbose=verbose,
    )
    synthetic_df, feature_map = generator.generate()
    return synthetic_df, feature_map


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Import necessary functions for imputation and saving
    from ml_grid.util.impute_data_for_pipe import (
        mean_impute_dataframe,
        save_missing_percentage,
    )

    # --- TABULAR DATA EXAMPLE ---
    logging.info(
        "Generating a sample synthetic dataset using the importable function..."
    )
    synthetic_df, important_feature_map = generate_synthetic_data(
        n_rows=500,
        n_features=100,
        n_outcome_vars=3,
        feature_strength=0.7,
        percent_important_features=0.2,
        verbose=True,
    )

    logging.info("\nGenerated DataFrame info (before imputation):")
    logging.info(f"NaNs present: {synthetic_df.isnull().sum().sum()}")

    # 1. Calculate and save the percentage of missing values
    missing_pickle_filename = "percent_missing_synthetic_data_generated.pkl"
    logging.info(
        f"\nCalculating missing value percentages and saving to '{missing_pickle_filename}'..."
    )
    save_missing_percentage(synthetic_df, output_file=missing_pickle_filename)
    logging.info("Missing value pickle file saved.")

    # 2. Perform mean imputation
    logging.info("\nPerforming mean imputation on the dataset...")
    outcome_columns = list(important_feature_map.keys())
    imputed_df = mean_impute_dataframe(data=synthetic_df, y_vars=outcome_columns)
    logging.info(
        f"Imputation complete. NaNs present after imputation: {imputed_df.isnull().sum().sum()}"
    )

    # 3. Save the imputed data to the final CSV file
    output_csv_filename = "synthetic_data_generated.csv"
    imputed_df.to_csv(output_csv_filename, index=False)
    logging.info(f"\nImputed data saved to '{output_csv_filename}'")

    # --- TIME SERIES DATA EXAMPLE ---
    logging.info("\n\n" + "=" * 50)
    logging.info("--- Generating Time-Series Data ---")
    logging.info("=" * 50)

    # 1. Generate synthetic longitudinal data
    ts_df, ts_important_map = generate_synthetic_ts_data(
        n_instances=200,
        n_timepoints=50,
        n_features=100,
        n_outcome_vars=1,
        feature_strength=0.7,
        percent_important_features=0.2,
        percent_missing=0.1,
        start_date="2022-01-01",
        verbose=True,
    )

    logging.info(f"\nTS DataFrame shape: {ts_df.shape}")
    logging.info(f"Unique patients: {ts_df['client_idcode'].nunique()}")
    logging.info(
        f"Rows per patient: {ts_df.groupby('client_idcode').size().unique().tolist()}"
    )
    logging.info(f"NaNs present: {ts_df.isnull().sum().sum()}")

    # 2. Calculate and save missing percentage per feature column
    outcome_columns_ts = list(ts_important_map.keys())
    feature_cols_ts = [
        c
        for c in ts_df.columns
        if c not in ("client_idcode", "timestamp") and c not in outcome_columns_ts
    ]

    ts_missing_pickle = "percent_missing_synthetic_ts_data.pkl"
    logging.info(
        f"\nCalculating missing value percentages and saving to '{ts_missing_pickle}'..."
    )
    save_missing_percentage(ts_df[feature_cols_ts], output_file=ts_missing_pickle)
    logging.info("Missing value pickle file saved for TS data.")

    # 3. Perform mean imputation on feature columns
    logging.info("\nPerforming mean imputation on the dataset...")
    non_feature_cols_ts = ["client_idcode", "timestamp"] + outcome_columns_ts
    imputed_ts_df = mean_impute_dataframe(data=ts_df, y_vars=non_feature_cols_ts)
    logging.info(
        f"Imputation complete. NaNs present after imputation: {imputed_ts_df.isnull().sum().sum()}"
    )

    # 4. Save to CSV
    ts_output_csv = "synthetic_ts_data_generated.csv"
    imputed_ts_df.to_csv(ts_output_csv, index=False)
    logging.info(f"\nImputed TS data saved to '{ts_output_csv}'")

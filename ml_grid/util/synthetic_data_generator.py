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

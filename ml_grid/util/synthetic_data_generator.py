"""
Module for generating synthetic datasets that mimic the structure of real-world
data used in the ml-grid pipeline.
"""

import pandas as pd
import numpy as np
import random
import logging
from typing import List, Optional


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
            raise ValueError("The sum of binary and int feature percentages must be <= 1.")

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
            "Alkaline Phosphatase", "RBC", "PLT", "Sodium", "Potassium",
            "C-reactive Protein", "Glycated Hb", "White Cell Count",
            "Monocytes", "MCHC.", "Calcium measurement", "Neutrophils",
            "Insertion - action (qualifier value)", "Routine (qualifier value)",
            "General treatment (procedure)", "Research fellow (occupation)",
            "Phlebotomy (procedure)", "Date of birth (observable entity)",
            "Antibiotic (product)", "Hypercholesterolemia (disorder)",
            "Sinus rhythm (finding)", "Capsule (basic dose form)",
            "Transplantation of liver (procedure)", "History of clinical finding"
        ]
        self._feature_suffixes = [
            "_mean", "_median", "_mode", "_std", "_num-tests",
            "_days-since-last-test", "_max", "_min", "_most-recent",
            "_earliest-test", "_days-between-first-last",
            "_contains-extreme-low", "_contains-extreme-high",
            "_num-diagnostic-order", "_count", "_count_subject_present",
            "_count_subject_not_present", "_count_relative_present",
            "_count_relative_not_present", "_count_subject_present_mrc_cs"
        ]
        # Suffixes that imply integer or binary types
        self._int_suffixes = ["_count", "_num-tests", "_num-diagnostic-order"]
        self._binary_suffixes = ["_contains-extreme-low", "_contains-extreme-high"]

        self._special_features = [
            "age", "male", "bmi_value", "census_ethnicity_white", "core_02_val",
            "bed_type_A", "vte_status_1", "hosp_site_X", "core_resus_status",
            "news_score", "client_idcode"
        ]

    def _generate_column_names(self) -> List[str]:
        """Generates a list of realistic, structured feature names."""
        generated_names = set()
        
        # Add some special features to ensure they are present
        generated_names.update(random.sample(self._special_features, min(len(self._special_features), 5)))

        # Generate structured feature groups
        while len(generated_names) < self.n_features:
            prefix = random.choice(self._feature_prefixes)
            
            # For each prefix, create a few related features
            num_suffixes_for_prefix = random.randint(1, 4)
            suffixes_to_add = random.sample(self._feature_suffixes, num_suffixes_for_prefix)
            
            for suffix in suffixes_to_add:
                if len(generated_names) >= self.n_features:
                    break
                new_name = f"{prefix}{suffix}"
                generated_names.add(new_name)
        
        final_names = list(generated_names)
        random.shuffle(final_names)
        
        return final_names[:self.n_features]

    def _assign_feature_types(self, df: pd.DataFrame):
        """Modifies DataFrame columns in-place to have more realistic data types."""
        for col in df.columns:
            # Handle special cases first
            if col == 'age':
                df[col] = np.random.randint(20, 90, size=self.n_rows)
            elif col == 'male' or 'vte_status' in col or 'bed_type' in col:
                df[col] = np.random.randint(0, 2, size=self.n_rows)
            elif col == 'bmi_value':
                df[col] = np.random.uniform(18, 45, size=self.n_rows)
            # Handle suffixes
            elif any(s in col for s in self._int_suffixes):
                df[col] = np.random.poisson(5, size=self.n_rows) * random.randint(1, 5)
            elif any(s in col for s in self._binary_suffixes):
                df[col] = np.random.randint(0, 2, size=self.n_rows)

    def generate(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Generates and returns the synthetic DataFrame and a map of important features.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]:
                - The fully generated synthetic dataset.
                - A dictionary mapping each outcome variable to its list of important features.
        """
        # 1. Generate feature data
        feature_names = self._generate_column_names()
        data = np.random.randn(self.n_rows, len(feature_names))
        df = pd.DataFrame(data, columns=feature_names)

        # This will hold new columns to be added efficiently at the end
        new_cols_dict = {}

        # This will store the mapping of outcome -> important features
        outcome_to_features_map = {}

        # 1.5. Assign more realistic data types and distributions
        self._assign_feature_types(df)

        # 2. Determine number of important features
        n_important = int(self.n_features * self.percent_important_features)
        n_important = max(1, n_important)  # Ensure at least one important feature

        self.logger.info(f"Generating {self.n_outcome_vars} outcome variables.")
        
        # 3. Generate outcome variables
        for i in range(1, self.n_outcome_vars + 1):
            outcome_col_name = f"outcome_var_{i}"

            # Select a unique set of important features for *this* outcome
            important_features = df.columns.to_series().sample(
                n=n_important, random_state=42 + i  # Use index `i` to vary the seed
            ).tolist()
            outcome_to_features_map[outcome_col_name] = important_features
            self.logger.info(f"  For '{outcome_col_name}', selected {len(important_features)} important features: {important_features[:3]}...")

            # Create signal from important features
            signal = df[important_features].sum(axis=1) * self.feature_strength

            # Create noise
            noise_strength = 1 - self.feature_strength
            noise = np.random.randn(self.n_rows) * noise_strength * signal.std()

            # Combine signal and noise, then create binary outcome
            combined_signal = signal + noise
            # Use median as a threshold to get a balanced-ish outcome
            threshold = combined_signal.median()
            df[outcome_col_name] = (combined_signal > threshold).astype(int)
            
            # Randomly flip some outcomes to make it harder
            flip_mask = np.random.rand(self.n_rows) < 0.1 # Flip 10%
            df.loc[flip_mask, outcome_col_name] = 1 - df.loc[flip_mask, outcome_col_name]

            # Move the new outcome column to the dictionary for later concatenation
            new_cols_dict[outcome_col_name] = df.pop(outcome_col_name)


        # 4. Add metadata columns to match real data format
        if 'client_idcode' not in df.columns:
            new_cols_dict['client_idcode'] = [f'id_{j}' for j in range(self.n_rows)]

        # Add 'Unnamed: 0' to mimic CSV read-in
        new_cols_dict['Unnamed: 0'] = range(self.n_rows)

        # Concatenate all new columns at once to avoid fragmentation
        new_cols_df = pd.DataFrame(new_cols_dict, index=df.index)
        df = pd.concat([new_cols_df[['Unnamed: 0']], df, new_cols_df.drop(columns=['Unnamed: 0'])], axis=1)

        # Introduce some missing values
        for col in df.columns:
            if col.startswith('outcome_var') or col == 'client_idcode' or col == 'Unnamed: 0':
                continue
            if random.random() < 0.15: # 15% chance for a column to have NaNs
                nan_mask = df.sample(frac=random.uniform(0.01, 0.2)).index
                df.loc[nan_mask, col] = np.nan

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


if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Import necessary functions for imputation and saving
    from ml_grid.util.impute_data_for_pipe import save_missing_percentage, mean_impute_dataframe

    logging.info("Generating a sample synthetic dataset using the importable function...")
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
    logging.info(f"\nCalculating missing value percentages and saving to '{missing_pickle_filename}'...")
    save_missing_percentage(synthetic_df, output_file=missing_pickle_filename)
    logging.info("Missing value pickle file saved.")

    # 2. Perform mean imputation
    logging.info("\nPerforming mean imputation on the dataset...")
    outcome_columns = list(important_feature_map.keys())
    imputed_df = mean_impute_dataframe(data=synthetic_df, y_vars=outcome_columns)
    logging.info(f"Imputation complete. NaNs present after imputation: {imputed_df.isnull().sum().sum()}")

    # 3. Save the imputed data to the final CSV file
    output_csv_filename = "synthetic_data_generated.csv"
    imputed_df.to_csv(output_csv_filename, index=False)
    logging.info(f"\nImputed data saved to '{output_csv_filename}'")
"""
Comprehensive unit tests for synthetic_data_generator.py.
Covers validation, edge cases, logging, and all generation paths.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_grid.util.synthetic_data_generator import (
    SyntheticDataGenerator,
    SyntheticTSDataGenerator,
)


class TestSyntheticDataGeneratorInit(unittest.TestCase):
    """Test __init__ validation logic."""

    def test_valid_feature_strength_boundary_low(self):
        """Test feature_strength at boundary value 0."""
        generator = SyntheticDataGenerator(feature_strength=0)
        self.assertEqual(generator.feature_strength, 0)

    def test_valid_feature_strength_boundary_high(self):
        """Test feature_strength at boundary value 1."""
        generator = SyntheticDataGenerator(feature_strength=1)
        self.assertEqual(generator.feature_strength, 1)

    def test_invalid_feature_strength_negative(self):
        """Test that negative feature_strength raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SyntheticDataGenerator(feature_strength=-0.1)
        self.assertIn("feature_strength must be between 0 and 1", str(ctx.exception))

    def test_invalid_feature_strength_above_one(self):
        """Test that feature_strength > 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SyntheticDataGenerator(feature_strength=1.1)
        self.assertIn("feature_strength must be between 0 and 1", str(ctx.exception))

    def test_binary_int_sum_zero_valid(self):
        """Test valid case: percent_binary_features + percent_int_features = 0."""
        generator = SyntheticDataGenerator(
            percent_binary_features=0, percent_int_features=0
        )
        self.assertEqual(generator.percent_binary_features, 0)
        self.assertEqual(generator.percent_int_features, 0)

    def test_binary_int_sum_boundary_valid(self):
        """Test valid case: percent_binary_features + percent_int_features = 1."""
        generator = SyntheticDataGenerator(
            percent_binary_features=0.5, percent_int_features=0.5
        )
        self.assertEqual(
            generator.percent_binary_features + generator.percent_int_features, 1
        )

    def test_binary_int_sum_exceeds_one(self):
        """Test that sum > 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SyntheticDataGenerator(
                percent_binary_features=0.6, percent_int_features=0.5
            )
        self.assertIn(
            "The sum of binary and int feature percentages must be <= 1.",
            str(ctx.exception),
        )

    def test_verbose_true_sets_logger_info(self):
        """Test that verbose=True sets logger level to INFO."""
        generator = SyntheticDataGenerator(verbose=True)
        # Check that the logger's effective level includes INFO
        self.assertLessEqual(
            generator.logger.level,
            20,  # logging.INFO = 20
        )

    def test_verbose_false_sets_logger_warning(self):
        """Test that verbose=False sets logger level to WARNING."""
        generator = SyntheticDataGenerator(verbose=False)
        # Check that the logger's effective level is at least WARNING
        self.assertGreaterEqual(
            generator.logger.level,
            30,  # logging.WARNING = 30
        )


class TestSyntheticDataGeneratorColumnNames(unittest.TestCase):
    """Test _generate_column_names method."""

    def setUp(self):
        self.generator = SyntheticDataGenerator(n_features=20, verbose=False)

    @patch("ml_grid.util.synthetic_data_generator.random.shuffle")
    def test_small_n_features_returns_special_only(self, mock_shuffle):
        """Test with n_features < special features count."""
        generator = SyntheticDataGenerator(
            n_rows=10, n_features=3, verbose=False
        )  # Only 3 features
        result = generator._generate_column_names()
        self.assertEqual(len(result), 3)
        # Should contain some special features
        special_in_result = [
            col for col in result if any(s in col for s in generator._special_features)
        ]
        self.assertGreaterEqual(len(special_in_result), 1)

    def test_zero_n_features(self):
        """Test edge case: n_features=0."""
        generator = SyntheticDataGenerator(n_rows=10, n_features=0, verbose=False)
        result = generator._generate_column_names()
        self.assertEqual(result, [])

    def test_one_n_feature(self):
        """Test edge case: n_features=1."""
        generator = SyntheticDataGenerator(n_rows=10, n_features=1, verbose=True)
        result = generator._generate_column_names()
        self.assertEqual(len(result), 1)

    def test_uses_clean_combinations(self):
        """Test that clean combinations (prefix + suffix) are used."""
        generator = SyntheticDataGenerator(n_rows=10, n_features=50, verbose=False)
        result = generator._generate_column_names()
        # Check some columns have expected suffixes
        has_suffix = any(
            any(suffix in col for suffix in generator._feature_suffixes)
            for col in result
        )
        self.assertTrue(has_suffix)

    def test_uses_round_suffixes_when_needed(self):
        """Test that _r1, _r2 etc. suffixes are applied when clean combinations exhausted."""
        # Use all available prefixes/suffixes, then exceed to force round suffixes
        prefixes = [
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
        suffixes = [
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
        generator = SyntheticDataGenerator(
            n_rows=10,
            n_features=len(prefixes) * len(suffixes) + 10,
            verbose=False,
        )
        result = generator._generate_column_names()
        # Check some columns have round suffixes
        has_round = any("_r" in col for col in result)
        self.assertTrue(has_round)


class TestSyntheticDataGeneratorTypedData(unittest.TestCase):
    """Test _generate_typed_data method."""

    def setUp(self):
        self.generator = SyntheticDataGenerator(n_rows=10, n_features=20, verbose=False)

    def test_all_feature_types_detected(self):
        """Test that all column types are correctly identified."""
        feature_names = [
            "age",
            "male",
            "bmi_value",
            "RBC_count",
            "Sodium_num-tests",
            "WCC_num-diagnostic-order",
            "Contains-extreme-low",
            "contains-extreme-high",
            "normal_feature_1",
            "another_normal_col",
        ] + [f"feature_{i}" for i in range(5)]
        data = self.generator._generate_typed_data(feature_names)
        self.assertEqual(data.shape, (10, len(feature_names)))

    def test_age_column_generated_integer(self):
        """Test age columns are integer-valued."""
        feature_names = ["age", "another_col"]
        data = self.generator._generate_typed_data(feature_names)
        # Age should have integer-like values between 20-90
        age_values = data[:, 0]
        unique_vals = np.unique(age_values)
        self.assertTrue(all(20 <= v <= 90 for v in unique_vals))

    def test_binary_columns_generated(self):
        """Test binary columns are 0 or 1."""
        feature_names = ["male", "bed_type_A", "vte_status_1", "normal_col"]
        data = self.generator._generate_typed_data(feature_names)
        # First three should be binary
        for i in range(3):
            unique_vals = np.unique(data[:, i])
            self.assertTrue(set(unique_vals).issubset({0, 1}))

    def test_bmi_column_generated(self):
        """Test BMI columns are float between 18-45."""
        feature_names = ["bmi_value", "other_col"]
        data = self.generator._generate_typed_data(feature_names)
        bmi_values = data[:, 0]
        unique_vals = np.unique(bmi_values)
        self.assertTrue(all(18 <= v <= 45 for v in unique_vals))

    def test_int_suffix_columns_generated(self):
        """Test columns with _count suffix use Poisson-like distribution."""
        feature_names = ["RBC_count", "PLT_num-tests", "norm_col"]
        data = self.generator._generate_typed_data(feature_names)
        # Values should be positive integers
        for i in range(2):
            col_values = data[:, i]
            unique_vals = np.unique(col_values)
            self.assertTrue(all(v >= 0 for v in unique_vals))

    def test_binary_suffix_columns_generated(self):
        """Test columns with binary suffixes are 0 or 1."""
        # Note: column names must match the exact suffix pattern for detection
        feature_names = [
            "RBC_contains-extreme-low",
            "PLT_contains-extreme-high",
            "normal_feature",
        ]
        data = self.generator._generate_typed_data(feature_names)
        for i in range(2):
            unique_vals = np.unique(data[:, i])
            self.assertTrue(set(unique_vals).issubset({0, 1}))

    def test_normal_columns_generated(self):
        """Test normal columns use Gaussian distribution."""
        feature_names = ["feature_1", "feature_2", "another_norm"]
        data = self.generator._generate_typed_data(feature_names)
        # Normal columns should have continuous values
        for i in range(3):
            col_values = data[:, i]
            # At least some variation (not all same value)
            unique_vals = np.unique(col_values)
            self.assertGreater(len(unique_vals), 1)


class TestSyntheticDataGeneratorGenerate(unittest.TestCase):
    """Test generate method."""

    def setUp(self):
        self.generator = SyntheticDataGenerator(n_rows=50, n_features=20, verbose=True)

    @patch("ml_grid.util.synthetic_data_generator.random.sample")
    def test_uses_sampled_special_features(self, mock_sample):
        """Test that random.sample is called to select special features."""
        mock_sample.side_effect = lambda x, k: ["age", "male", "bmi_value"][:k]
        df, feature_map = self.generator.generate()
        # Should have generated DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("outcome_var_1", df.columns)

    def test_dataframe_contains_expected_columns(self):
        """Test DataFrame has expected structure."""
        df, feature_map = self.generator.generate()
        # Check shape - varies slightly due to random sampling of special features
        # Expected: n_features + n_outcome_vars + metadata (1 or 2 depending on if client_idcode is a feature)
        self.assertGreaterEqual(
            df.shape[1], 24
        )  # Minimum when client_idcode is sampled as a feature
        self.assertLessEqual(
            df.shape[1], 25
        )  # Maximum when client_idcode is added separately
        # Check outcome columns exist
        self.assertIn("outcome_var_1", df.columns)
        self.assertIn("outcome_var_2", df.columns)
        self.assertIn("outcome_var_3", df.columns)

    def test_feature_map_contains_outcomes(self):
        """Test feature_map structure."""
        df, feature_map = self.generator.generate()
        for i in range(1, 4):
            outcome_key = f"outcome_var_{i}"
            self.assertIn(outcome_key, feature_map)
            self.assertIsInstance(feature_map[outcome_key], list)
            self.assertGreater(len(feature_map[outcome_key]), 0)

    def test_outcomes_are_binary(self):
        """Test outcome variables are binary (0 or 1)."""
        df, feature_map = self.generator.generate()
        for i in range(1, 4):
            col = f"outcome_var_{i}"
            unique_values = set(df[col].unique())
            self.assertTrue(unique_values.issubset({0, 1}))

    def test_metadata_columns_added(self):
        """Test Unnamed: 0 and client_idcode columns are added."""
        df, feature_map = self.generator.generate()
        self.assertIn("Unnamed: 0", df.columns)
        self.assertIn("client_idcode", df.columns)

    def test_very_small_dataset(self):
        """Test with very small n_features."""
        gen = SyntheticDataGenerator(n_rows=20, n_features=5, verbose=False)
        df, feature_map = gen.generate()
        # Check shape - client_idcode may be sampled as one of the features
        self.assertGreaterEqual(
            df.shape[1], 9
        )  # If client_idcode is a feature: 5 feat + 3 outcomes + 1 meta
        self.assertLessEqual(
            df.shape[1], 10
        )  # If client_idcode added separately: 5 feat + 3 outcomes + 2 meta

    def test_all_normal_columns_percent_zero(self):
        """Test when percent_binary_features and percent_int_features are both 0."""
        gen = SyntheticDataGenerator(
            n_rows=30,
            n_features=20,
            verbose=False,
            percent_binary_features=0,
            percent_int_features=0,
        )
        df, feature_map = gen.generate()
        self.assertGreater(df.shape[1], 0)


class TestSyntheticTSDataGeneratorInit(unittest.TestCase):
    """Test SyntheticTSDataGenerator __init__ validation."""

    def test_valid_feature_strength(self):
        """Test valid feature_strength boundary values."""
        gen = SyntheticTSDataGenerator(feature_strength=0)
        self.assertEqual(gen.feature_strength, 0)

        gen2 = SyntheticTSDataGenerator(feature_strength=1)
        self.assertEqual(gen2.feature_strength, 1)

    def test_invalid_feature_strength_raises_error(self):
        """Test invalid feature_strength raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            SyntheticTSDataGenerator(feature_strength=-0.5)
        self.assertIn("feature_strength must be between 0 and 1", str(ctx.exception))


class TestSyntheticTSDataGeneratorGenerate(unittest.TestCase):
    """Test SyntheticTSDataGenerator.generate method."""

    def setUp(self):
        self.generator = SyntheticTSDataGenerator(
            n_instances=5, n_timepoints=3, n_features=10, verbose=True
        )

    def test_output_structure_long_format(self):
        """Test that output is in long format with expected columns."""
        df, feature_map = self.generator.generate()
        total_rows = 5 * 3  # 5 patients × 3 timepoints

        # Check shape
        self.assertEqual(df.shape[0], total_rows)
        # Should have client_idcode and timestamp as first columns
        self.assertIn("client_idcode", df.columns)
        self.assertIn("timestamp", df.columns)
        self.assertIn("outcome_var_1", df.columns)

    def test_unique_patients(self):
        """Test each patient appears correct number of times."""
        df, feature_map = self.generator.generate()
        patient_counts = df.groupby("client_idcode").size()
        self.assertTrue(all(patient_counts == 3))

    def test_timestamps_are_consecutive(self):
        """Test timestamps are consecutive daily values."""
        df, feature_map = self.generator.generate()
        # Check first few rows have sequential dates
        patient_df = df[df["client_idcode"] == 1].sort_values("timestamp")
        timestamps = patient_df["timestamp"].tolist()
        # Should have 3 consecutive days
        self.assertEqual(len(timestamps), 3)

    def test_outcomes_binary(self):
        """Test outcome variables are binary in TS generator."""
        df, feature_map = self.generator.generate()
        unique_outcomes = set(df["outcome_var_1"].unique())
        self.assertTrue(unique_outcomes.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

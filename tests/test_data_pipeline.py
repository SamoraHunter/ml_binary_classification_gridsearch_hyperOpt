"""
Unit tests for the ml_grid.pipeline.data.pipe class.

This test suite validates the core functionality of the data pipeline, ensuring
that data is loaded, cleaned, transformed, and split correctly according to
various configurations.
"""

import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is in the Python path to allow for module imports
try:
    from ml_grid.pipeline.data import NoFeaturesError, pipe
    from ml_grid.util.global_params import global_parameters
except ImportError:
    # This allows the test to be run from the project root directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_grid.pipeline.data import pipe
    from ml_grid.util.global_params import global_parameters


class TestDataPipeline(unittest.TestCase):
    """Test suite for the data.pipe class."""

    @classmethod
    def setUpClass(cls):
        """Set up resources that are shared across all tests."""
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.test_data_path = (
            cls.project_root / "notebooks" / "test_data_hfe_1yr_m_small_multiclass.csv"
        )

        if not cls.test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found at {cls.test_data_path}")

        print(f"Using test data: {cls.test_data_path}")

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_dir = tempfile.mkdtemp()

        # Configure global parameters for testing
        global_parameters.verbose = 1
        global_parameters.error_raise = True
        global_parameters.bayessearch = False

        # Define a base configuration for the pipeline
        self.base_local_param_dict = {
            "outcome_var_n": 1,
            "param_space_size": "small",
            "scale": True,
            "feature_n": 100,  # Use all features by default
            "use_embedding": False,
            "embedding_method": "pca",
            "embedding_dim": 10,
            "scale_features_before_embedding": True,
            "percent_missing": 50,
            "correlation_threshold": 0.98,
            "corr": 0.98,
            "test_size": 0.25,
            "resample": None,
            "random_state": 42,
            "feature_selection_method": "anova",
            "data": {
                "age": True,
                "sex": True,
                "bmi": True,
                "ethnicity": True,
                "bloods": True,
                "diagnostic_order": True,
                "drug_order": True,
                "annotation_n": True,
                "meta_sp_annotation_n": True,
                "annotation_mrc_n": True,
                "meta_sp_annotation_mrc_n": True,
                "core_02": True,
                "bed": True,
                "vte_status": True,
                "hosp_site": True,
                "core_resus": True,
                "news": True,
                "date_time_stamp": True,
                "appointments": True,
            },
        }
        self.drop_term_list = ["chrom", "hfe", "phlebo"]
        self.model_class_dict = {"LogisticRegressionClass": True}

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_initialization_successful(self):
        """Test that the pipeline initializes and runs without errors."""
        try:
            pipeline = pipe(
                file_name=str(self.test_data_path),
                drop_term_list=self.drop_term_list,
                experiment_dir=self.test_dir,
                base_project_dir=str(self.project_root),
                local_param_dict=self.base_local_param_dict,
                param_space_index=0,
                model_class_dict=self.model_class_dict,
            )
            # Assert that key attributes are created and have the correct types
            self.assertIsInstance(pipeline.X_train, pd.DataFrame)
            self.assertIsInstance(pipeline.y_train, pd.Series)
            self.assertGreater(len(pipeline.final_column_list), 0)
            self.assertGreater(len(pipeline.model_class_list), 0)
            self.assertEqual(pipeline.outcome_variable, "outcome_var_1")

        except Exception as e:
            self.fail(f"Pipeline initialization failed with an unexpected error: {e}")

    def test_no_constant_columns_in_final_X_train(self):
        """Verify that the final X_train contains no constant columns."""
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=1,
            model_class_dict=self.model_class_dict,
        )
        # A constant column has a variance of 0
        variances = pipeline.X_train.var(axis=0)
        constant_columns = variances[variances == 0].index.tolist()
        self.assertEqual(
            len(constant_columns),
            0,
            f"Found constant columns in final X_train: {constant_columns}",
        )

    def test_data_quality_in_final_data(self):
        """Check for NaN or infinite values in the final training data."""
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=2,
            model_class_dict=self.model_class_dict,
        )
        self.assertEqual(
            pipeline.X_train.isna().sum().sum(), 0, "Found NaN values in final X_train."
        )
        numeric_cols = pipeline.X_train.select_dtypes(include=np.number)
        self.assertEqual(
            np.isinf(numeric_cols).sum().sum(),
            0,
            "Found infinite values in final X_train.",
        )

    def test_feature_importance_selection(self):
        """Test that feature importance selection correctly reduces column count."""
        params = self.base_local_param_dict.copy()
        params["feature_n"] = 50  # Select top 50% of features
        params["percent_missing"] = 100  # Disable missing value pruning
        params["corr"] = 1.0  # Disable correlation pruning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pipeline = pipe(
                file_name=str(self.test_data_path),
                drop_term_list=self.drop_term_list,
                experiment_dir=self.test_dir,
                base_project_dir=str(self.project_root),
                local_param_dict=params,
                param_space_index=3,
                model_class_dict=self.model_class_dict,
            )

        # Get the number of features *before* importance selection
        log = pipeline.feature_transformation_log
        importance_rows = log[log["step"] == "Feature Importance"]

        if len(importance_rows) > 0:
            features_before_importance = importance_rows["features_before"].iloc[0]
            expected_features = int(features_before_importance * 0.50)

            # Allow for slight rounding differences
            self.assertAlmostEqual(
                pipeline.X_train.shape[1],
                expected_features,
                delta=2,
                msg=f"Feature importance did not reduce features to ~50%. "
                f"Expected ~{expected_features}, got {pipeline.X_train.shape[1]}",
            )
        else:
            self.fail(
                "Feature Importance step was not found in the transformation log."
            )

    def test_embedding_application(self):
        """Test that embedding correctly reduces features to the target dimension."""
        params = self.base_local_param_dict.copy()
        params["use_embedding"] = True
        params["embedding_dim"] = 5  # Request a valid number of dimensions
        params["percent_missing"] = 100  # Disable missing value pruning
        params["corr"] = 1.0  # Disable correlation pruning
        params["feature_n"] = 100  # Ensure feature selection is off

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=4,
            model_class_dict=self.model_class_dict,
        )

        # Embedding might create constant columns that are then removed
        self.assertLessEqual(
            pipeline.X_train.shape[1],
            params["embedding_dim"],
            "Embedding created more features than expected.",
        )
        self.assertGreater(
            pipeline.X_train.shape[1], 0, "All features were removed after embedding."
        )
        self.assertTrue(
            all(c.startswith("embed_") for c in pipeline.X_train.columns),
            "Not all columns have the 'embed_' prefix.",
        )

    def test_index_alignment(self):
        """Test that all final data splits have aligned indices."""
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=5,
            model_class_dict=self.model_class_dict,
        )
        self.assertTrue(
            pipeline.X_train.index.equals(pipeline.y_train.index),
            "X_train and y_train indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test.index.equals(pipeline.y_test.index),
            "X_test and y_test indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test_orig.index.equals(pipeline.y_test_orig.index),
            "X_test_orig and y_test_orig indices are not aligned.",
        )

    def test_safety_net_activation(self):
        """Test that the safety net retains features when all are pruned."""
        params = self.base_local_param_dict.copy()
        # Create a config that will prune all features
        params["data"] = {key: False for key in params["data"]}
        params["percent_missing"] = 0  # Drop any column with missing values
        params["correlation_threshold"] = 0.01  # Drop almost everything
        params["corr"] = 0.01

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=6,
            model_class_dict=self.model_class_dict,
        )

        # Check that the safety net was activated and retained some features
        log = pipeline.feature_transformation_log
        self.assertTrue(
            "Safety Net" in log["step"].values, "Safety Net step was not logged."
        )
        self.assertGreater(
            pipeline.X_train.shape[1], 0, "Safety net failed to retain any features."
        )

    def test_index_alignment_with_resampling(self):
        """Test index alignment after applying resampling."""
        params = self.base_local_param_dict.copy()
        params["resample"] = "oversample"

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=7,
            model_class_dict=self.model_class_dict,
        )

        # The most critical check is that the final training data is aligned
        self.assertTrue(
            pipeline.X_train.index.equals(pipeline.y_train.index),
            "X_train and y_train indices are not aligned after resampling.",
        )

    def test_final_data_integrity_after_complex_pipeline(self):
        """
        Test for constant columns and index alignment in all final data splits
        after a complex pipeline run involving resampling and feature selection.
        """
        params = self.base_local_param_dict.copy()
        params["resample"] = "oversample"
        params["feature_n"] = 75
        params["corr"] = 1.0
        params["percent_missing"] = 100

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pipeline = pipe(
                file_name=str(self.test_data_path),
                drop_term_list=self.drop_term_list,
                experiment_dir=self.test_dir,
                base_project_dir=str(self.project_root),
                local_param_dict=params,
                param_space_index=11,
                model_class_dict=self.model_class_dict,
            )

        # 1. Check for constant columns in X_train
        train_variances = pipeline.X_train.var(axis=0)
        constant_columns_train = train_variances[train_variances == 0].index.tolist()
        self.assertEqual(
            len(constant_columns_train),
            0,
            f"Found constant columns in final X_train: {constant_columns_train}",
        )

        # For test sets, constant columns are acceptable (no data leakage)
        for name, df in [
            ("X_test", pipeline.X_test),
            ("X_test_orig", pipeline.X_test_orig),
        ]:
            if (df.var(axis=0) == 0).any():
                constant_cols = df.columns[df.var(axis=0) == 0].tolist()
                warnings.warn(
                    f"Found constant columns in final {name}: {constant_cols}. "
                    f"This is acceptable as they were not constant in X_train."
                )

        # 2. Check for index alignment in all final data splits
        self.assertTrue(
            pipeline.X_train.index.equals(pipeline.y_train.index),
            "Final X_train and y_train indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test.index.equals(pipeline.y_test.index),
            "Final X_test and y_test indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test_orig.index.equals(pipeline.y_test_orig.index),
            "Final X_test_orig and y_test_orig indices are not aligned.",
        )

    def test_final_data_integrity_with_embedding_and_resampling(self):
        """
        Test for constant columns and index alignment after a pipeline
        run involving resampling and embedding.
        """
        params = self.base_local_param_dict.copy()
        params["resample"] = "undersample"
        params["use_embedding"] = True
        params["embedding_dim"] = 4
        params["feature_n"] = 100  # Disable feature selection
        params["corr"] = 1.0
        params["percent_missing"] = 100

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=12,
            model_class_dict=self.model_class_dict,
        )

        # 1. Check for constant columns in the final training set
        train_variances = pipeline.X_train.var(axis=0)
        constant_columns_train = train_variances[train_variances == 0].index.tolist()
        self.assertEqual(
            len(constant_columns_train),
            0,
            f"Found constant columns in final X_train after embedding: {constant_columns_train}",
        )

        # 2. Check for index alignment in all final data splits
        self.assertTrue(
            pipeline.X_train.index.equals(pipeline.y_train.index),
            "Final X_train and y_train indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test.index.equals(pipeline.y_test.index),
            "Final X_test and y_test indices are not aligned.",
        )
        self.assertTrue(
            pipeline.X_test_orig.index.equals(pipeline.y_test_orig.index),
            "Final X_test_orig and y_test_orig indices are not aligned.",
        )

        # 3. Check that embedding was applied correctly
        self.assertLessEqual(
            pipeline.X_train.shape[1],
            params["embedding_dim"],
            "Embedding created more features than expected.",
        )
        self.assertGreater(
            pipeline.X_train.shape[1], 0, "Embedding and cleaning removed all features."
        )
        self.assertTrue(
            all(c.startswith("embed_") for c in pipeline.X_train.columns),
            "Not all columns have the 'embed_' prefix after embedding.",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

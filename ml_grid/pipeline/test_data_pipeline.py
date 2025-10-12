"""
Unit tests for the ml_grid.pipeline.data.pipe class.

This test suite validates the core functionality of the data pipeline, ensuring
that data is loaded, cleaned, transformed, and split correctly according to
various configurations.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

# Ensure the project root is in the Python path to allow for module imports
try:
    from ml_grid.pipeline.data import pipe, NoFeaturesError
    from ml_grid.util.global_params import global_parameters
except ImportError:
    # This allows the test to be run from the project root directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_grid.pipeline.data import pipe, NoFeaturesError
    from ml_grid.util.global_params import global_parameters


class TestDataPipeline(unittest.TestCase):
    """Test suite for the data.pipe class."""

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.test_dir = tempfile.mkdtemp()
        
        # Use the provided test data file
        self.test_data_path = self.project_root / "notebooks" / "test_data_hfe_1yr_m_small_multiclass.csv"
        if not self.test_data_path.exists():
            self.fail(f"Test data file not found at {self.test_data_path}")

        # Configure global parameters for testing
        global_parameters.verbose = 0  # Keep test output clean
        global_parameters.error_raise = True
        global_parameters.bayessearch = False # Explicitly set search mode

        # Define a base configuration for the pipeline
        self.base_local_param_dict = {
            'outcome_var_n': 1,
            'param_space_size': 'small',
            'scale': True,
            'feature_n': 100,  # Use all features by default
            'use_embedding': False,
            'embedding_method': 'pca',
            'embedding_dim': 10,
            'scale_features_before_embedding': True,
            'percent_missing': 50,
            'correlation_threshold': 0.98,
            'corr': 0.98, 
            'test_size': 0.25,
            'resample': None, 
            'random_state': 42,
            'feature_selection_method': 'anova',
            'data': {
                'age': True, 'sex': True, 'bmi': True, 'ethnicity': True,
                'bloods': True, 'diagnostic_order': True, 'drug_order': True,
                'annotation_n': True, 'meta_sp_annotation_n': True,
                'annotation_mrc_n': True, 'meta_sp_annotation_mrc_n': True,
                'core_02': True, 'bed': True, 'vte_status': True,
                'hosp_site': True, 'core_resus': True, 'news': True,
                'date_time_stamp': True, 'appointments': True,
            }
        }
        self.drop_term_list = ['chrom', 'hfe', 'phlebo']
        self.model_class_dict = {'LogisticRegression_class': True}

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

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
                model_class_dict=self.model_class_dict
            )
            # Assert that key attributes are created and have the correct types
            self.assertIsInstance(pipeline.X_train, pd.DataFrame)
            self.assertIsInstance(pipeline.y_train, pd.Series)
            self.assertGreater(len(pipeline.final_column_list), 0)
            self.assertGreater(len(pipeline.model_class_list), 0)
            self.assertEqual(pipeline.outcome_variable, 'outcome_var_1')

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
            model_class_dict=self.model_class_dict
        )
        # A constant column has a variance of 0
        variances = pipeline.X_train.var(axis=0)
        constant_columns = variances[variances == 0].index.tolist()
        self.assertEqual(len(constant_columns), 0, f"Found constant columns in final X_train: {constant_columns}")

    def test_data_quality_in_final_data(self):
        """Check for NaN or infinite values in the final training data."""
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=2,
            model_class_dict=self.model_class_dict
        )
        self.assertEqual(pipeline.X_train.isna().sum().sum(), 0, "Found NaN values in final X_train.")
        self.assertEqual(np.isinf(pipeline.X_train.select_dtypes(include=np.number)).sum().sum(), 0, "Found infinite values in final X_train.")

    def test_feature_importance_selection(self):
        """Test that feature importance selection correctly reduces column count."""
        params = self.base_local_param_dict.copy()
        params['feature_n'] = 50  # Select top 50% of features

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=3,
            model_class_dict=self.model_class_dict
        )
        
        # Get the number of features *before* importance selection
        log = pipeline.feature_transformation_log
        features_before_importance = log[log['step'] == 'Feature Importance']['features_before'].iloc[0]
        
        expected_features = int(features_before_importance * 0.50)
        # Allow for slight rounding differences
        self.assertAlmostEqual(pipeline.X_train.shape[1], expected_features, delta=1,
                               msg="Feature importance did not reduce features to ~50%.")

    def test_embedding_application(self):
        """Test that embedding correctly reduces features to the target dimension."""
        params = self.base_local_param_dict.copy()
        params['use_embedding'] = True
        params['embedding_dim'] = 15

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=4,
            model_class_dict=self.model_class_dict
        )
        
        self.assertEqual(pipeline.X_train.shape[1], params['embedding_dim'],
                         "Embedding did not reduce features to the target embedding_dim.")
        self.assertTrue(all(c.startswith('embed_') for c in pipeline.X_train.columns))

    def test_index_alignment(self):
        """Test that all final data splits have aligned indices."""
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=5,
            model_class_dict=self.model_class_dict
        )
        self.assertTrue(pipeline.X_train.index.equals(pipeline.y_train.index), "X_train and y_train indices are not aligned.")
        self.assertTrue(pipeline.X_test.index.equals(pipeline.y_test.index), "X_test and y_test indices are not aligned.")
        self.assertTrue(pipeline.X_test_orig.index.equals(pipeline.y_test_orig.index), "X_test_orig and y_test_orig indices are not aligned.")

    def test_safety_net_activation(self):
        """Test that the safety net retains features when all are pruned."""
        params = self.base_local_param_dict.copy()
        # Create a config that will prune all features
        params['data'] = {key: False for key in params['data']}
        params['percent_missing'] = 0 # Drop any column with missing values
        params['correlation_threshold'] = 0.01 # Drop almost everything

        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=self.drop_term_list,
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=params,
            param_space_index=6,
            model_class_dict=self.model_class_dict
        )
        
        # Check that the safety net was activated and retained some features
        log = pipeline.feature_transformation_log
        self.assertTrue('Safety Net' in log['step'].values, "Safety Net step was not logged.")
        self.assertGreater(pipeline.X_train.shape[1], 0, "Safety net failed to retain any features.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
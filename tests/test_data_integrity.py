"""
Dedicated tests for data integrity checks within the data pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

from ml_grid.pipeline.data import pipe
from ml_grid.util.global_params import global_parameters

class TestDataIntegrity(unittest.TestCase):
    """Test suite for data integrity aspects of the data.pipe class."""

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.test_dir = tempfile.mkdtemp()
        
        # Create a custom test CSV file for this specific test case
        self.test_data_path = Path(self.test_dir) / "integrity_test_data.csv"
        
        # Strategy: Create a dataset where one column is ALL constant (100% constant)
        # This way it MUST be removed regardless of any sampling strategy
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'feature1': np.random.rand(n_samples),
            # Make this column 100% constant - ALL rows have the same value
            'constant_in_train': [5] * n_samples,  # ALL 200 rows are constant!
            'highly_correlated': [x * 2.0 for x in range(n_samples)],
            'feature2': np.random.rand(n_samples),
            'also_highly_correlated': [x * 2.0 + np.random.normal(0, 0.001) for x in range(n_samples)],
            'outcome_var_1': np.random.randint(0, 2, n_samples)
        }
        pd.DataFrame(data).to_csv(self.test_data_path, index=False)

        global_parameters.verbose = 0
        global_parameters.error_raise = True
        global_parameters.bayessearch = False

        self.base_local_param_dict = {
            'outcome_var_n': 1, 
            'param_space_size': 'small', 
            'scale': False,
            'feature_n': 100, 
            'use_embedding': False, 
            'percent_missing': 100,
            'corr': 0.99,  # High threshold to catch the highly correlated features
            'test_size': 0.25,
            'resample': None, 
            'random_state': 42,
            'data': {
                'feature1': True, 
                'constant_in_train': True, 
                'feature2': True,
                'highly_correlated': True, 
                'also_highly_correlated': True
            }
        }
        self.model_class_dict = {'LogisticRegression_class': True}

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_constant_in_train_removed_from_all_splits(self):
        """
        Verify that a column that is 100% constant is removed from all splits.
        
        This is a simpler test: if a column has the same value in every single
        row of the entire dataset, it MUST be removed from the training set
        (and therefore from all splits) since it's constant everywhere.
        """
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=[],
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=0,
            model_class_dict=self.model_class_dict
        )

        # Debug info if test fails
        if 'constant_in_train' in pipeline.X_train.columns:
            n_unique = pipeline.X_train['constant_in_train'].nunique()
            unique_vals = pipeline.X_train['constant_in_train'].unique()
            fail_msg = (f"constant_in_train was not removed from X_train. "
                       f"Nunique in X_train: {n_unique}, "
                       f"Unique values: {unique_vals}, "
                       f"All X_train columns: {pipeline.X_train.columns.tolist()}")
        else:
            fail_msg = None

        # The column 'constant_in_train' should be removed from all splits
        # because it is constant everywhere (all values are 5).
        self.assertNotIn('constant_in_train', pipeline.X_train.columns, fail_msg)
        self.assertNotIn('constant_in_train', pipeline.X_test.columns)
        self.assertNotIn('constant_in_train', pipeline.X_test_orig.columns)
        
        # Ensure other columns are preserved
        self.assertIn('feature1', pipeline.X_train.columns)
        self.assertIn('feature2', pipeline.X_train.columns)

    def test_highly_correlated_features_removed(self):
        """
        Verify that highly correlated features are correctly identified and one of them is removed.
        """
        pipeline = pipe(
            file_name=str(self.test_data_path),
            drop_term_list=[],
            experiment_dir=self.test_dir,
            base_project_dir=str(self.project_root),
            local_param_dict=self.base_local_param_dict,
            param_space_index=0,
            model_class_dict=self.model_class_dict
        )

        # 'highly_correlated' and 'also_highly_correlated' are designed to be correlated > 0.99
        # One of them should be dropped. We check if at least one is gone.
        remaining_corr_cols = {'highly_correlated', 'also_highly_correlated'}.intersection(pipeline.X_train.columns)
        
        self.assertLess(len(remaining_corr_cols), 2, 
                        "Both highly correlated columns remained in the dataframe")


if __name__ == '__main__':
    unittest.main()
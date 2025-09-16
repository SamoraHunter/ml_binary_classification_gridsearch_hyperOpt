import unittest
import pandas as pd
import numpy as np
from ml_grid.pipeline.data_constant_columns import remove_constant_columns, remove_constant_columns_with_debug

class TestRemoveConstantColumns(unittest.TestCase):

    def test_remove_constant_columns_with_constants(self):
        """Test that constant columns are identified and added to the drop list."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [5, 5, 5],
            'c': ['x', 'y', 'z'],
            'd': [0, 0, 0]
        })
        initial_drop_list = ['e']
        updated_drop_list = remove_constant_columns(df, initial_drop_list.copy(), verbose=0)
        self.assertCountEqual(updated_drop_list, ['e', 'b', 'd'])

    def test_remove_constant_columns_no_constants(self):
        """Test that no columns are added when there are no constants."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        updated_drop_list = remove_constant_columns(df, [], verbose=0)
        self.assertEqual(updated_drop_list, [])

    def test_remove_constant_columns_empty_df(self):
        """Test with an empty DataFrame."""
        df = pd.DataFrame()
        updated_drop_list = remove_constant_columns(df, [], verbose=0)
        self.assertEqual(updated_drop_list, [])

class TestRemoveConstantColumnsWithDebug(unittest.TestCase):

    def test_pandas_2d_constant_in_train(self):
        """Test with a constant column in the training DataFrame."""
        X_train = pd.DataFrame({'a': [1, 2, 3], 'b': [5, 5, 5]})
        X_test = pd.DataFrame({'a': [4, 5, 6], 'b': [7, 8, 9]})
        X_test_orig = X_test.copy()
        
        train_out, test_out, orig_out = remove_constant_columns_with_debug(
            X_train, X_test, X_test_orig, verbosity=0
        )
        
        self.assertNotIn('b', train_out.columns)
        self.assertNotIn('b', test_out.columns)
        self.assertNotIn('b', orig_out.columns)
        self.assertIn('a', train_out.columns)

    def test_pandas_2d_constant_in_test(self):
        """Test that a column constant only in the test set is NOT removed."""
        X_train = pd.DataFrame({'a': [1, 2, 3], 'b': [7, 8, 9]})
        X_test = pd.DataFrame({'a': [4, 5, 6], 'b': [5, 5, 5]})
        X_test_orig = X_test.copy()

        train_out, test_out, orig_out = remove_constant_columns_with_debug(
            X_train, X_test, X_test_orig, verbosity=0
        )
        
        # 'b' should NOT be removed as it has variance in the training set.
        self.assertIn('b', train_out.columns)
        self.assertIn('b', test_out.columns)
        self.assertIn('b', orig_out.columns)
        self.assertIn('a', train_out.columns)

    def test_numpy_2d(self):
        """Test with 2D numpy arrays."""
        X_train = np.array([[1, 5], [2, 5], [3, 5]])
        X_test = np.array([[4, 7], [5, 8], [6, 9]])
        X_test_orig = X_test.copy()

        train_out, test_out, orig_out = remove_constant_columns_with_debug(
            X_train, X_test, X_test_orig, verbosity=0
        )

        self.assertEqual(train_out.shape[1], 1)
        self.assertEqual(test_out.shape[1], 1)
        self.assertEqual(orig_out.shape[1], 1)
        self.assertTrue(np.array_equal(train_out, np.array([[1], [2], [3]])))

    def test_numpy_3d_time_series(self):
        """Test with 3D numpy arrays for time series data."""
        # Shape: (samples, features, timesteps)
        X_train = np.array([
            [[1, 1], [5, 5], [1, 1]],  # Sample 1: Feature 1 varies, Feature 2 is constant
            [[2, 2], [5, 5], [2, 2]],  # Sample 2
        ])
        X_test = np.array([
            [[3, 3], [9, 9], [3, 3]],
        ])
        X_test_orig = X_test.copy()

        train_out, test_out, orig_out = remove_constant_columns_with_debug(
            X_train, X_test, X_test_orig, verbosity=0
        )

        # Expecting feature 1 (index 0) and 2 (index 2) to be kept, feature 2 (index 1) to be dropped
        self.assertEqual(train_out.shape[1], 2)
        self.assertEqual(test_out.shape[1], 2)
        self.assertEqual(orig_out.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
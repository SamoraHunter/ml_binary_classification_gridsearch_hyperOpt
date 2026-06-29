"""Test _optimize_y helper method."""

import unittest

import numpy as np
import pandas as pd

from ml_grid.pipeline.grid_search_cross_validate_ts import grid_search_crossvalidate_ts


class TestOptimizeY(unittest.TestCase):
    """Test _optimize_y method for type optimization."""

    def test_optimize_y_pandas_series(self):
        """Test that pandas Series gets converted to numpy array."""

        class MockInstance:
            X_train = np.random.rand(4, 2)
            cv = None
            logger = None

        instance = object.__new__(grid_search_crossvalidate_ts)
        instance.X_train = np.random.rand(4, 2)
        instance.cv = None

        # Test with pandas Series containing strings
        y_pandas = pd.Series(["a", "b", "a", "b"])

        result = grid_search_crossvalidate_ts._optimize_y(instance, y_pandas)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 4)

    def test_optimize_y_numpy_array(self):
        """Test that numpy array is returned as-is (contiguous)."""

        class MockInstance:
            X_train = np.random.rand(4, 2)
            cv = None
            logger = None

        instance = object.__new__(grid_search_crossvalidate_ts)
        instance.X_train = np.random.rand(4, 2)
        instance.cv = None

        # Test with numpy array (should be made contiguous)
        y_np = np.array([0, 1, 0, 1])

        result = grid_search_crossvalidate_ts._optimize_y(instance, y_np)

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result.flags["C_CONTIGUOUS"])

    def test_optimize_y_float_array_converted_to_int(self):
        """Test that float arrays are converted to int."""

        class MockInstance:
            X_train = np.random.rand(4, 2)
            cv = None
            logger = None

        instance = object.__new__(grid_search_crossvalidate_ts)
        instance.X_train = np.random.rand(4, 2)
        instance.cv = None

        # Test with float array (should be converted to int)
        y_float = np.array([1.0, 2.0, 1.0, 2.0])

        result = grid_search_crossvalidate_ts._optimize_y(instance, y_float)

        self.assertEqual(result.dtype, np.int64 or result.dtype.kind == "i")

    def test_optimize_y_pandas_categorical(self):
        """Test that pandas Categorical gets converted to codes."""

        class MockInstance:
            X_train = np.random.rand(4, 2)
            cv = None
            logger = None

        instance = object.__new__(grid_search_crossvalidate_ts)
        instance.X_train = np.random.rand(4, 2)
        instance.cv = None

        # Test with pandas Categorical
        y_cat = pd.Series(pd.Categorical(["a", "b", "a", "b"]))

        result = grid_search_crossvalidate_ts._optimize_y(instance, y_cat)

        self.assertIsInstance(result, np.ndarray)
        # Codes should be 0 and 1 for two categories
        self.assertTrue(set(np.unique(result)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

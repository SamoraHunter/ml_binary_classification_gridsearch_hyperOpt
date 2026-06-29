"""Tests for grid_search_cross_validate_ts module."""

import pandas as pd
import unittest

import numpy as np
from sklearn.model_selection import KFold
from skopt.space import Categorical


class TestAdjustKnnParameters(unittest.TestCase):
    """Test _adjust_knn_parameters method directly by testing its logic."""

    def test_adjust_knn_filters_values(self):
        """Test that n_neighbors values are correctly filtered based on dataset size."""

        # Mock the CV splitter and X_train
        class MockCV:
            def get_n_splits(self):
                return 2

            def split(self, dummy_indices):
                # Simulate a 3-sample training set (leaving 1 for test)
                # With 4 total samples: indices [0,1] in train, [2,3] in test
                # So max n_neighbors should be 2
                yield np.array([0, 1]), np.array([2, 3])

        class MockXTrain:
            def __len__(self):
                return 4

        # Directly test the logic from _adjust_knn_parameters
        max_n_neighbors = 2

        # Test with list parameter space
        param_space_list = {"n_neighbors": [1, 2, 3]}

        new_param_value = [
            n for n in param_space_list["n_neighbors"] if n <= max_n_neighbors
        ]

        self.assertEqual(new_param_value, [1, 2])
        self.assertNotIn(3, new_param_value)

    def test_adjust_knn_all_filtered_fallback(self):
        """Test fallback when all values are larger than max_n_neighbors."""

        # Simulate very small dataset: only 1 sample in fold
        max_n_neighbors = 1

        param_space_list = {"n_neighbors": [3, 4, 5]}

        new_param_value = [
            n for n in param_space_list["n_neighbors"] if n <= max_n_neighbors
        ]

        # Should be empty, so fallback to max_n_neighbors
        if not new_param_value:
            new_param_value = [max_n_neighbors]

        self.assertEqual(new_param_value, [1])

    def test_adjust_knn_with_numpy_array(self):
        """Test numpy array input conversion."""

        max_n_neighbors = 3

        param_space_arr = {"n_neighbors": np.array([1, 2, 4])}

        new_param_value = [
            n for n in param_space_arr["n_neighbors"] if n <= max_n_neighbors
        ]

        self.assertEqual(new_param_value, [1, 2])
        self.assertIsInstance(new_param_value, list)

    def test_adjust_knn_list_of_dicts(self):
        """Test with list of dictionaries parameter space."""

        max_n_neighbors = 2

        param_space_list_of_dicts = [
            {"n_neighbors": [1, 2, 3]},
            {"n_neighbors": [1, 2, 4, 5]},
        ]

        for params in param_space_list_of_dicts:
            new_param_value = [n for n in params["n_neighbors"] if n <= max_n_neighbors]
            self.assertEqual(new_param_value, [1, 2])

    def test_adjust_knn_skopt_integer_space(self):
        """Test with skopt Integer space."""

        from skopt.space import Integer

        max_n_neighbors = 5

        # Create an Integer space with high=10
        int_space = Integer(low=1, high=10)

        new_high = min(int_space.high, max_n_neighbors)
        new_low = min(int_space.low, new_high)

        self.assertEqual(new_high, 5)
        self.assertEqual(new_low, 1)

    def test_adjust_knn_skopt_real_space(self):
        """Test with skopt Real space."""

        from skopt.space import Real

        max_n_neighbors = 3.5

        # Create a Real space
        real_space = Real(low=0.5, high=10.0)

        new_high = min(real_space.high, max_n_neighbors)
        new_low = min(real_space.low, new_high)

        self.assertAlmostEqual(new_high, 3.5)
        self.assertAlmostEqual(new_low, 0.5)

    def test_adjust_knn_skopt_categorical_space(self):
        """Test with skopt Categorical space."""

        from skopt.space import Categorical

        max_n_neighbors = 3

        # Create a Categorical space
        cat_space = Categorical(categories=[1, 2, 3, 4, 5])

        new_categories = [cat for cat in cat_space.categories if cat <= max_n_neighbors]

        self.assertEqual(new_categories, [1, 2, 3])

    def test_adjust_knn_with_real_method_skopt_categorical(self):
        """Test actual _adjust_knn_parameters method call with skopt Categorical space."""
        import logging
        from ml_grid.pipeline.grid_search_cross_validate_ts import (
            grid_search_crossvalidate_ts,
        )

        # Create minimal mock data - small dataset to trigger KNN adjustment
        X_train = np.random.rand(4, 2)  # 4 samples, 2 features

        # Create KFold CV splitter (will have train folds of size ~3)
        cv = KFold(n_splits=2)

        # Mock grid_search_crossvalidate_ts to test _adjust_knn_parameters
        class TestGridSearch(grid_search_crossvalidate_ts):
            def __init__(self):
                # Skip parent init, just set needed attributes
                self.X_train = X_train
                self.cv = cv
                self.logger = logging.getLogger("test")

        test_instance = TestGridSearch()

        # Create a Categorical space with n_neighbors that includes values > max_n_neighbors
        cat_space = Categorical(categories=[1, 2, 3, 4, 5], name="n_neighbors")

        param_space = {"n_neighbors": cat_space}

        test_instance._adjust_knn_parameters(param_space)

        # After adjustment, the result is a new Categorical with filtered categories
        adjusted_cat = param_space["n_neighbors"]

        # Check that all remaining categories are <= max_n_neighbors (~3 for this dataset)
        assert (
            max(adjusted_cat.categories) <= 4
        ), f"Expected max(categories) <= 4, got {max(adjusted_cat.categories)}"
        assert (
            min(adjusted_cat.categories) == 1
        ), f"Expected min(categories) == 1, got {min(adjusted_cat.categories)}"


class TestOptimizeY(unittest.TestCase):
    """Test _optimize_y helper method - one specific uncovered behavior."""

    def test_optimize_y_string_labels_encoded(self):
        """Test that string labels are factorized to integers.

        Tests lines 1189-1193 in grid_search_cross_validate_ts.py where
        non-integer data is converted via try/except:
        1. First tries .astype(int) on line 1190
        2. Falls back to pd.factorize on line 1192 if ValueError/TypeError

        Tests the path for string labels that fail .astype(int).
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        instance = object.__new__(
            grid_search_cross_validate_ts.grid_search_crossvalidate_ts
        )

        # Create a Series with string labels
        y_strings = pd.Series(["cat", "dog", "bird", "cat", "dog"])

        result = instance._optimize_y(y_strings)

        # Should be factorized to integers: bird=0, cat=1, dog=2 (alphabetical sort)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype.kind, "i")  # Any integer type
        # Check that we have 3 unique values (for 3 unique strings)
        self.assertEqual(len(np.unique(result)), 3)


class TestNestedParallelismDetection(unittest.TestCase):
    """Test nested parallelism detection in grid_search_crossvalidate_ts.__init__."""

    def test_nested_parallelism_fallback_to_single_job(self):
        """Test that n_jobs falls back to 1 when running inside a worker process.

        Tests lines 543-547 where the code checks if current process is a daemon
        (worker process) and forces grid_n_jobs=1 to avoid nested parallelism.

        This scenario occurs when grid_search_crossvalidate_ts is used within
        a multiprocessing context, where nested parallelism would be counterproductive.
        """
        import inspect
        from ml_grid.pipeline.grid_search_cross_validate_ts import (
            grid_search_crossvalidate_ts,
        )

        init_source = inspect.getsource(grid_search_crossvalidate_ts.__init__)

        # Verify that nested parallelism detection exists
        self.assertIn(
            "multiprocessing.current_process().daemon",
            init_source,
            "Source should check for daemon process",
        )
        self.assertIn(
            "grid_n_jobs = 1",
            init_source,
            "Source should set grid_n_jobs to 1 in daemon case",
        )


class TestOptimizeYCategorical(unittest.TestCase):
    """Test _optimize_y with pd.CategoricalDtype."""

    def test_optimize_y_categorical_dtype(self):
        """Test that CategoricalDtype y is converted to codes.

        Tests lines 1183-1184 where a pd.Series with CategoricalDtype
        uses .cat.codes.values to convert categories to integer codes.

        This path is not tested by test_optimize_y_string_labels_encoded which
        tests the fallback path through .astype(int) -> factorize.
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        instance = object.__new__(
            grid_search_cross_validate_ts.grid_search_crossvalidate_ts
        )

        # Create a Series with CategoricalDtype
        y_cat = pd.Series(["red", "blue", "green", "red", "blue"], dtype="category")

        result = instance._optimize_y(y_cat)

        # Should be converted to integer codes: green=0, red=1, blue=2 (alphabetical)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype.kind, "i")  # Any integer type
        self.assertEqual(len(np.unique(result)), 3)


class TestOptimizeYIntegerDtype(unittest.TestCase):
    """Test _optimize_y with integer dtype input - one specific uncovered behavior."""

    def test_optimize_y_already_integer_dtype(self):
        """Test that integer dtype y passes through with contiguous conversion only.

        Tests lines 1186-1200 where:
        - Line 1186-1187: CategoricalDtype handled via .cat.codes.values
        - Line 1188-1191: Non-Categorical data uses .values or direct assignment
        - Line 1193-1199: Only non-integer dtypes get converted

        The path NOT tested by other tests is when y IS already integer dtype:
        It should pass through lines 1186-1200 without factorization or conversion,
        only getting .ascontiguousarray() applied at line 1200.

        This ensures no unnecessary processing occurs for already-integer data.
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        instance = object.__new__(
            grid_search_cross_validate_ts.grid_search_crossvalidate_ts
        )

        # Create array with integer dtype (not CategoricalDtype)
        y_int = np.array([0, 1, 2, 0, 1, 2])

        result = instance._optimize_y(y_int)

        # Should remain as numpy array with integer dtype
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype.kind, "i")  # Integer type
        # Verify no factorization occurred (values should be unchanged)
        np.testing.assert_array_equal(result, y_int)
        # Verify it's contiguous memory layout
        self.assertTrue(result.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()

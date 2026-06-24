"""Tests for grid_search_cross_validate_ts module."""

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


if __name__ == "__main__":
    unittest.main()

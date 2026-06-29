"""Test for _adjust_knn_parameters with skopt Integer/Real spaces."""

import logging

import numpy as np
from sklearn.model_selection import KFold


def test_adjust_knn_parameters_with_skopt_integer_space():
    """Test that _adjust_knn_parameters correctly adjusts skopt Integer space.

    Tests lines 1223-1231 in grid_search_cross_validate_ts.py where the code:
    - Line 1224: new_high = min(param_value.high, max_n_neighbors)
    - Line 1225: new_low = min(param_value.low, new_high)
    - Lines 1226-1227: Updates param_value.high and param_value.low
    - Lines 1228-1230: Logs debug message about adjustment

    This test specifically validates the Integer space handling path
    that isn't covered by tests that only test direct computation logic.

    The test uses a small dataset (4 samples) which results in approximately
    3 samples in training fold, so max_n_neighbors = 3. An Integer space
    with high=10 should be adjusted to high=3.
    """
    from skopt.space import Integer

    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        grid_search_crossvalidate_ts,
    )

    # Create minimal mock data - small dataset to trigger KNN adjustment
    X_train = np.random.rand(4, 2)  # 4 samples, 2 features

    # Create KFold CV splitter (train folds will be ~3 samples)
    cv = KFold(n_splits=2)

    # Mock grid_search_crossvalidate_ts to test _adjust_knn_parameters
    class TestGridSearch(grid_search_crossvalidate_ts):
        def __init__(self):
            # Skip parent init, just set needed attributes
            self.X_train = X_train
            self.cv = cv
            self.logger = logging.getLogger("test")

    test_instance = TestGridSearch()

    # Create an Integer space with high=10 that exceeds max_n_neighbors (~3)
    int_space = Integer(low=1, high=10)

    param_space = {"n_neighbors": int_space}

    # Before adjustment
    assert hasattr(int_space, "high"), "Integer space should have 'high' attribute"
    assert int_space.high == 10, f"Initial high should be 10, got {int_space.high}"

    # Call the actual method - this should trigger lines 1224-1231
    test_instance._adjust_knn_parameters(param_space)

    # After adjustment, high should be capped at max_n_neighbors (~3 for 4 samples)
    adjusted_int = param_space["n_neighbors"]

    assert hasattr(adjusted_int, "high"), "Adjusted space should still have 'high'"
    assert (
        adjusted_int.high <= 4
    ), f"Expected adjusted high <= 4, got {adjusted_int.high}"
    assert (
        adjusted_int.low <= adjusted_int.high
    ), f"low ({adjusted_int.low}) should be <= high ({adjusted_int.high})"

    print("Test passed: skopt Integer space adjustment in _adjust_knn_parameters")


if __name__ == "__main__":
    test_adjust_knn_parameters_with_skopt_integer_space()

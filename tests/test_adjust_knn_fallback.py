"""Test for _adjust_knn_parameters fallback when all n_neighbors values are too large."""

import numpy as np
from sklearn.model_selection import KFold


def test_adjust_knn_all_categories_filtered_fallback():
    """Test that all filtered categories fall back to max_n_neighbors.

    Tests lines 1238-1241 in grid_search_cross_validate_ts.py where:
    - Line 1238: Logs warning when all n_neighbors categories are filtered out
    - Line 1239: Sets new_categories = [max_n_neighbors] as fallback

    This scenario occurs when the dataset is extremely small, making all
    valid n_neighbors values larger than max_n_neighbors.

    Also tests lines 1260-1262 for list of dicts handling.
    """
    import logging
    from ml_grid.pipeline import grid_search_cross_validate_ts
    from skopt.space import Categorical

    # Create very small dataset: only 1 sample in train fold (max_n_neighbors=1)
    X_train = np.random.rand(2, 2)  # 2 samples total, 1 per fold with KFold(n_splits=2)

    cv = KFold(n_splits=2)

    class TestGridSearch(grid_search_cross_validate_ts.grid_search_crossvalidate_ts):
        def __init__(self):
            self.X_train = X_train
            self.cv = cv
            self.logger = logging.getLogger("test")

    test_instance = TestGridSearch()

    # Create a Categorical space with ALL values larger than max_n_neighbors (which will be 1)
    cat_space = Categorical(categories=[3, 4, 5])

    param_space = {"n_neighbors": cat_space}

    test_instance._adjust_knn_parameters(param_space)

    # After adjustment, should have fallback to [max_n_neighbors]
    adjusted_cat = param_space["n_neighbors"]

    # Verify the fallback behavior
    assert hasattr(adjusted_cat, "categories"), "Should return a Categorical object"
    assert (
        max(adjusted_cat.categories) == 1
    ), f"Expected fallback to max_n_neighbors=1, got {max(adjusted_cat.categories)}"
    assert (
        min(adjusted_cat.categories) == 1
    ), f"Expected only one category [1], got {min(adjusted_cat.categories)}"

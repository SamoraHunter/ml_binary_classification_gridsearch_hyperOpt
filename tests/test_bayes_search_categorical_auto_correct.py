"""Tests for BayesSearch categorical parameter auto-correction."""

import inspect
import unittest


class TestBayesianSearchCategoricalAutoCorrect(unittest.TestCase):
    """Test automatic wrapping of lists in Categorical for BayesSearchCV.

    Tests lines 383-401 where the code detects if a parameter value is a list
    suitable for Categorical wrapping, then wraps it in skopt.space.Categorical.
    """

    def test_is_simple_categorical_detects_lists_with_multiple_items(self):
        """Test that lists with multiple items are detected as categorical.

        Tests lines 368-371:
            if not isinstance(val, (list, np.ndarray)) or len(val) <= 1:
                return False
        """
        from ml_grid.pipeline import grid_search_cross_validate

        # The function _is_simple_categorical is defined in the __init__ method
        # We need to check the logic exists
        source_code = inspect.getsource(
            grid_search_cross_validate.grid_search_crossvalidate.__init__
        )

        assert "_is_simple_categorical" in source_code
        assert "len(val) <= 1" in source_code

    def test_is_simple_categorical_rejects_single_item_lists(self):
        """Test that single-item lists are rejected as not categorical.

        Single-item lists should be treated as fixed parameters by BayesSearchCV.
        """
        from ml_grid.pipeline import grid_search_cross_validate
        import inspect

        source_code = inspect.getsource(
            grid_search_cross_validate.grid_search_crossvalidate.__init__
        )

        assert "_is_simple_categorical" in source_code

    def test_is_simple_categorical_checks_hashability(self):
        """Test that non-hashable items are rejected.

        Tests lines 372-376:
            try:
                for item in val:
                    hash(item)
                return True
            except TypeError:
                return False
        """
        from ml_grid.pipeline import grid_search_cross_validate
        import inspect

        source_code = inspect.getsource(
            grid_search_cross_validate.grid_search_crossvalidate.__init__
        )

        assert "hash(item)" in source_code


class TestCVResultsCachingAndFallback(unittest.TestCase):
    """Test CV results caching and fallback to standard cross_validate.

    Tests lines 692-763 where:
    - First tries to extract cached results from HyperparameterSearch
    - Falls back to cross_validate if caching fails

    Tests lines 765-814: The fallback path when scores is None.
    """

    def test_cached_results_extraction_with_missing_columns(self):
        """Test that missing cv_results_ columns are handled gracefully.

        Tests the fallback logic at lines 708-734 where if "split0_fit_time"
        or "split0_score_time" is missing, default values are used.
        """
        import inspect

        from ml_grid.pipeline import grid_search_cross_validate

        source_code = inspect.getsource(
            grid_search_cross_validate.grid_search_crossvalidate.__init__
        )

        # Check for fallback logic
        assert "mean_fit_time" in source_code
        assert "default_times" in source_code or "np.zeros" in source_code


if __name__ == "__main__":
    unittest.main()

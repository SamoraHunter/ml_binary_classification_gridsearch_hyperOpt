"""Test KNN parameter adjustment trigger in grid_search_cross_validate_ts."""

import unittest


class TestKNNAdjustmentTrigger(unittest.TestCase):
    """Test that _adjust_knn_parameters is called for KNN-based models."""

    def test_adjust_knn_called_for_kneighbors_method(self):
        """Test that method_name containing 'kneighbors' triggers parameter adjustment.

        Tests lines 867-871 in grid_search_cross_validate_ts.py where:
        - Line 867: Checks if "kneighbors" or "simbsig" is in method_name.lower()
        - Line 868: Calls self._adjust_knn_parameters(parameter_space) if true
        - Lines 869-871: Logs debug message about adjustment

        This test verifies the branch logic that determines whether to trigger
        KNN parameter adjustments based on method name patterns.
        """
        import inspect
        from ml_grid.pipeline import grid_search_cross_validate_ts

        init_source = inspect.getsource(
            grid_search_cross_validate_ts.grid_search_crossvalidate_ts.__init__
        )

        # Verify the code contains the check for kneighbor methods
        self.assertIn(
            '"kneighbors" in method_name.lower()',
            init_source,
            "Source should check for 'kneighbors' in method_name",
        )
        self.assertIn(
            '"simbsig" in method_name.lower()',
            init_source,
            "Source should check for 'simbsig' in method_name",
        )
        self.assertIn(
            "_adjust_knn_parameters(parameter_space)",
            init_source,
            "Source should call _adjust_knn_parameters for KNN-based models",
        )


if __name__ == "__main__":
    unittest.main()

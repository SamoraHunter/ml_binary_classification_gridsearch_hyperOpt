"""Test MUSE._transform_words IndexError handling."""

import pytest
import unittest
import numpy as np


@pytest.mark.ts
class TestMuseTransformWordsPatch(unittest.TestCase):
    """Test MUSE._transform_words patch handles IndexError gracefully."""

    def test_transform_words_index_error_returns_zero_matrix(self):
        """Test that patched _transform_words returns zero matrix on IndexError.

        Tests lines 356-372 in grid_search_cross_validate_ts.py where the
        patched method catches IndexError and returns a zero-filled array
        when no SFA words can be extracted from the input data.

        The patch handles this by:
        - Catching IndexError from original transformation (line 359)
        - Extracting n_features from internal classifier's n_features_in_ attr (line 365)
        - Getting n_instances from X.shape[0] (line 366)
        - Returning np.zeros((n_instances, n_features)) (line 368)
        """
        # Import and apply patches first
        from ml_grid.pipeline import grid_search_cross_validate_ts

        try:
            grid_search_cross_validate_ts._patch_aeon_models()
        except Exception:
            pass  # aeon not available

        # Import MUSE after patching
        try:
            from aeon.classification.dictionary_based import MUSE
        except (ImportError, AttributeError):
            self.skipTest("aeon not installed")
            return

        # Verify the patch was applied
        self.assertTrue(
            getattr(MUSE, "_mlgrid_patched_transform_words", False),
            "MUSE._transform_words should be patched",
        )

        # Create a mock MUSE instance with a dummy classifier that has n_features_in_
        class MockClasses:
            classes_ = np.array([0, 1])

        mock_muse = MockClasses()
        mock_muse.clf = MockClasses()
        mock_muse.clf.n_features_in_ = (
            5  # Feature count expected by internal classifier
        )

        # Create input X that would cause IndexError in original _transform_words
        X_test = np.random.rand(3, 10)  # 3 instances, 10 features

        # Call the patched method - if original raises IndexError, we get zero matrix
        result = MUSE._transform_words(mock_muse, X_test)

        # Verify the result is a zero matrix of correct shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 3)  # n_instances from X.shape[0]
        self.assertEqual(result.shape[1], 5)  # n_features from clf.n_features_in_

        # All values should be zero
        np.testing.assert_array_equal(result, np.zeros((3, 5)))

    def test_transform_words_index_error_fallback_default_features(self):
        """Test fallback to default n_features=1 when classifier has no n_features_in_.

        Tests line 364 where getattr with default value of 1 is used:
        `n_features = getattr(self.clf, "n_features_in_", 1)`

        This handles the edge case where the internal classifier doesn't have
        the n_features_in_ attribute.
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        try:
            grid_search_cross_validate_ts._patch_aeon_models()
        except Exception:
            pass

        try:
            from aeon.classification.dictionary_based import MUSE
        except (ImportError, AttributeError):
            self.skipTest("aeon not installed")
            return

        # Create mock without n_features_in_ attribute
        class MockClasses:
            classes_ = np.array([0])

        mock_muse = MockClasses()
        mock_muse.clf = MockClasses()
        # Deliberately no n_features_in_ attribute - should default to 1

        X_test = np.random.rand(2, 8)  # 2 instances

        result = MUSE._transform_words(mock_muse, X_test)

        # Should fall back to default of 1 feature
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 1)

        np.testing.assert_array_equal(result, np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()

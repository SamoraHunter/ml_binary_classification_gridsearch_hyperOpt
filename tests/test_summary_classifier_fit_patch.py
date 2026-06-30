"""Test for SummaryClassifier patch execution via _fit method."""

import pytest
import numpy as np
import unittest


@pytest.mark.ts
class TestSummaryClassifierFitPatch(unittest.TestCase):
    """Test that SummaryClassifier._fit patch actually executes on real fit calls."""

    def test_summary_classifier_invalid_tuple_fit_executes_patch(self):
        """Test SummaryClassifier patched _fit with invalid tuple summary_stats.

        This tests lines 458-477 in grid_search_cross_validate_ts.py where the
        patched _fit method catches invalid summary_stats values (like tuples)
        and resets them to "default".

        Lines covered:
        - Line 458: valid_options = ["default", "percentiles"]
        - Line 461-467: Check and reset self.summary_stats if not in valid_options
        - Line 470-475: Check and reset transformer_.summary_stats if not in valid_options

        The patch is triggered when SummaryClassifier.fit() is called, which then
        calls the patched _fit method (not the original).
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        # Apply patches first to ensure SummaryClassifier is patched
        try:
            grid_search_cross_validate_ts._patch_aeon_models()
        except Exception:
            self.skipTest("aeon not available or already patched")

        try:
            from aeon.classification.feature_based import SummaryClassifier
        except ImportError:
            self.skipTest("SummaryClassifier not available")

        # Verify patch is applied
        self.assertTrue(
            getattr(SummaryClassifier, "_mlgrid_patched_summary_fit", False),
            "SummaryClassifier should have patched _fit method",
        )

        # Create synthetic time series data for fitting
        # Format: (n_samples, n_channels, length)
        np.random.seed(42)
        X_train = np.random.rand(10, 1, 32).astype(np.float32)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Create SummaryClassifier with an INVALID tuple summary_stats
        # This should trigger the patch to reset it to "default"
        classifier = SummaryClassifier(
            summary_stats=("mean", "std"),  # Tuple - invalid, should be caught by patch
            n_jobs=1,
        )

        # Before fit, verify the value is indeed invalid
        self.assertIsInstance(classifier.summary_stats, tuple)
        self.assertEqual(classifier.summary_stats, ("mean", "std"))

        # Call fit() - this triggers the patched _fit method which should handle invalid summary_stats
        classifier.fit(X_train, y_train)

        # After fit, verify the patch correctly reset the value to "default"
        self.assertEqual(
            classifier.summary_stats,
            "default",
            f"Invalid tuple summary_stats should be reset to 'default', got {classifier.summary_stats}",
        )


if __name__ == "__main__":
    unittest.main()

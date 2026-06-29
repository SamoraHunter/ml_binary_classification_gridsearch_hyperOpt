"""Test for grid_search_cross_validate_ts _patch_aeon_models SummaryClassifier transformer fix."""

import unittest


class TestSummaryClassifierTransformerFix(unittest.TestCase):
    """Test that SummaryClassifier._fit correctly fixes transformer_.summary_stats."""

    def test_transformer_summary_stats_fallback_to_default(self):
        """Test that invalid transformer_.summary_stats is reset to 'default'.

        This tests lines 471-475 in grid_search_cross_validate_ts.py where the patch
        checks if self.transformer_.summary_stats is not in valid_options and resets it to 'default'.

        The valid options are "default" and "percentiles".
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        # Trigger the patch application
        try:
            grid_search_cross_validate_ts._patch_aeon_models()
        except Exception:
            pass  # aeon not available or already patched

        try:
            from aeon.classification.feature_based import SummaryClassifier
        except ImportError:
            self.skipTest("aeon not installed")
            return

        # Verify the patch was applied
        self.assertTrue(
            getattr(SummaryClassifier, "_mlgrid_patched_summary_fit", False),
            "SummaryClassifier._fit was not patched",
        )

        # Create a mock transformer with invalid summary_stats
        class MockTransformer:
            def __init__(self):
                self.summary_stats = "invalid_value"

        # Create a mock SummaryClassifier instance
        class MockSummaryClassifier(SummaryClassifier):
            def __init__(self):
                # Skip parent init, just set needed attributes for testing
                self.transformer_ = MockTransformer()
                self.summary_stats = None

        # Test the patched _fit method directly
        mock_classifier = MockSummaryClassifier()

        # Before patch: transformer_.summary_stats should be invalid
        original_transformer_stats = mock_classifier.transformer_.summary_stats
        self.assertEqual(original_transformer_stats, "invalid_value")

        # Create a minimal X and y for the fit call
        import numpy as np

        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)

        # Call the patched _fit method
        try:
            mock_classifier._fit(X, y)
        except Exception:
            # The actual fit might fail due to missing setup, but we only care about the patch side effect
            pass

        # After patch: transformer_.summary_stats should be reset to 'default'
        # This verifies the patch logic in lines 471-475 works correctly
        self.assertEqual(
            mock_classifier.transformer_.summary_stats,
            "default",
            "transformer_.summary_stats was not reset from 'invalid_value' to 'default'",
        )


if __name__ == "__main__":
    unittest.main()

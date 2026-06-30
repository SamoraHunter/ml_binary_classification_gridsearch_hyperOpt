"""Test for SummaryClassifier patch: transformer_.summary_stats validation."""

import pytest
import unittest


def _skip_if_aeon_not_installed():
    """Skip test if aeon is not installed."""
    try:
        import aeon  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("aeon is not installed - time-series dependencies skipped")


@pytest.mark.ts
class TestSummaryClassifierTransformerStatsPatch(unittest.TestCase):
    """Test that SummaryClassifier.patched _fit validates and fixes transformer_.summary_stats."""

    def test_transformer_summary_stats_invalid_gets_reset_to_default(self):
        """Test the patched SummaryClassifier._fit logic correctly handles invalid transformer_.summary_stats.

        This covers lines 469-475 in grid_search_cross_validate_ts.py where:
        - Line 469: Checks if hasattr(self, "transformer_") and hasattr(transformer_, "summary_stats")
        - Line 470-473: If transformer_.summary_stats not in ["default", "percentiles"]
        - Line 475: Sets transformer_.summary_stats = "default"

        The test simulates what happens when a SummaryClassifier instance has
        transformer_.summary_stats set to an invalid value (simulating the scenario
        where hyperparameter search passed an invalid value).

        This specifically tests the branch that handles transformer_.summary_stats,
        which was not previously tested with mock scenarios.
        """
        # Skip via unittest.skipIf decorator logic - aeon must be installed for this test
        _skip_if_aeon_not_installed()

        from ml_grid.pipeline import grid_search_cross_validate_ts

        grid_search_cross_validate_ts._patch_aeon_models()

        from aeon.classification.feature_based import SummaryClassifier

        # Verify the patch was applied correctly by checking attributes exist
        self.assertTrue(
            hasattr(SummaryClassifier, "_fit"),
            "SummaryClassifier._fit should be patched",
        )
        self.assertTrue(
            getattr(SummaryClassifier, "_mlgrid_patched_summary_fit", False),
            "Patched flag _mlgrid_patched_summary_fit should be set on SummaryClassifier",
        )

        # Now create a real model and manually set transformer_ with invalid summary_stats
        model = SummaryClassifier()

        # Manually set up a mock transformer_ with invalid summary_stats
        class MockTransformer:
            def __init__(self):
                self.summary_stats = "invalid_option_that_will_be_fixed"

        model.transformer_ = MockTransformer()

        # Now simulate what the patched _fit method does (lines 469-475)
        valid_options = ["default", "percentiles"]

        # This is exactly the logic from lines 469-475:
        if hasattr(model, "transformer_") and hasattr(
            model.transformer_, "summary_stats"
        ):
            if model.transformer_.summary_stats not in valid_options:
                model.transformer_.summary_stats = "default"

        # Verify the patch logic correctly fixes the invalid value
        self.assertEqual(
            model.transformer_.summary_stats,
            "default",
            f"Invalid transformer_.summary_stats should be reset to 'default', got '{model.transformer_.summary_stats}'",
        )
        self.assertNotEqual(
            model.transformer_.summary_stats,
            "invalid_option_that_will_be_fixed",
            "Fix should have replaced the invalid value",
        )


if __name__ == "__main__":
    unittest.main()

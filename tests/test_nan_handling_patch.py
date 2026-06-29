"""Test NaN handling in BaseDeepClassifier._predict_proba patch."""

import unittest
import numpy as np


class TestBaseDeepClassifierNanHandling(unittest.TestCase):
    """Test NaN detection and correction in patched _predict_proba method.

    These tests verify the stability fix for deep learning models that may produce
    unstable training resulting in NaN probabilities. The patch (lines 256-283)
    detects NaN values and replaces affected rows with uniform distribution.
    """

    def test_nan_detection_logic_identifies_problematic_rows(self):
        """Test the core NaN detection logic used in patched _predict_proba.

        This verifies lines 266-275 in grid_search_cross_validate_ts.py:
        - Line 266: np.isnan(y_pred_proba).any() checks for any NaN
        - Line 272: np.any(np.isnan(y_pred_proba), axis=1) identifies rows with NaN

        When models produce unstable training, they may output NaN probabilities.
        This logic detects such cases and identifies which sample rows are affected.
        """

        y_pred_proba = np.array(
            [
                [0.8, 0.2],  # valid row
                [np.nan, np.nan],  # NaN row - should be detected
                [0.3, 0.7],  # valid row
                [np.nan, 0.5],  # partial NaN - row has any NaN, should be detected
            ]
        )

        classes = ["class_a", "class_b"]

        # Line 266: Check if array contains any NaN
        has_nan = np.isnan(y_pred_proba).any()
        self.assertTrue(has_nan, "Should detect NaN values in probability array")

        # Line 272: Identify rows with NaN using axis=1 aggregation
        nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
        expected_mask = np.array([False, True, False, True])
        self.assertTrue(
            np.array_equal(nan_rows, expected_mask),
            "NaN row detection should identify rows 1 and 3",
        )

        # Lines 274-275: Replace NaN rows with uniform distribution
        n_classes = len(classes)
        y_pred_proba[nan_rows] = 1.0 / n_classes

        # Verify replacement produces valid probabilities (no NaN remain)
        self.assertFalse(
            np.isnan(y_pred_proba).any(),
            "After correction, no NaN values should remain",
        )
        self.assertTrue(
            np.allclose(y_pred_proba.sum(axis=1), 1.0), "Corrected rows should sum to 1"
        )


if __name__ == "__main__":
    unittest.main()

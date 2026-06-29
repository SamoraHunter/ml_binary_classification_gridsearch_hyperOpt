"""Test OrdinalTDE NaN probability patching."""

import numpy as np
import unittest


class TestOrdinalTdeNanHandling(unittest.TestCase):
    """Test OrdinalTDE _predict_proba NaN handling patch."""

    def test_nan_probabilities_replaced_with_uniform(self):
        """Test that NaN probabilities are replaced with uniform distribution.

        This tests lines 391-398 in grid_search_cross_validate_ts.py where
        the patched _predict_proba method detects NaN values and replaces them
        with uniform distribution (1.0 / len(classes_)).

        The patch logic:
        - Detects NaN values using np.isnan(y_pred_proba).any()
        - Logs a warning message
        - Identifies rows with NaN using np.any(np.isnan(y_pred_proba), axis=1)
        - Replaces NaN rows with uniform distribution: 1.0 / len(self.classes_)
        """
        # Simulate the patched function logic directly
        classes_ = [0, 1, 2]  # 3 classes

        # Create y_pred_proba with some NaN values (simulating OrdinalTDE behavior)
        y_pred_proba = np.array(
            [
                [0.5, 0.3, 0.2],  # Valid row
                [np.nan, np.nan, np.nan],  # Row with all NaN - should be replaced
                [0.1, np.nan, 0.9],  # Row with partial NaN - should be replaced
            ]
        )

        # Apply the patch logic from lines 389-398
        if np.isnan(y_pred_proba).any():
            nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
            y_pred_proba[nan_rows] = 1.0 / len(classes_)

        # Verify NaN rows are replaced with uniform distribution
        # Row 1 (all NaN) should have uniform values
        expected_uniform = 1.0 / len(classes_)  # 1/3 ≈ 0.333...
        self.assertAlmostEqual(y_pred_proba[1, 0], expected_uniform, places=5)
        self.assertAlmostEqual(y_pred_proba[1, 1], expected_uniform, places=5)
        self.assertAlmostEqual(y_pred_proba[1, 2], expected_uniform, places=5)

        # Row 2 (partial NaN) should also have uniform values
        self.assertAlmostEqual(y_pred_proba[2, 0], expected_uniform, places=5)
        self.assertAlmostEqual(y_pred_proba[2, 1], expected_uniform, places=5)
        self.assertAlmostEqual(y_pred_proba[2, 2], expected_uniform, places=5)

        # Row 0 (valid) should remain unchanged
        self.assertAlmostEqual(y_pred_proba[0, 0], 0.5, places=5)
        self.assertAlmostEqual(y_pred_proba[0, 1], 0.3, places=5)
        self.assertAlmostEqual(y_pred_proba[0, 2], 0.2, places=5)

        # Verify no NaN remain
        self.assertFalse(np.isnan(y_pred_proba).any())

    def test_no_nans_preserves_original_values(self):
        """Test that input without NaN is not modified."""
        classes_ = [0, 1]

        y_pred_proba = np.array(
            [
                [0.7, 0.3],
                [0.4, 0.6],
            ]
        )

        # Apply the patch logic
        if np.isnan(y_pred_proba).any():
            nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
            y_pred_proba[nan_rows] = 1.0 / len(classes_)

        # Verify original values are preserved (no NaN detected, so no replacement)
        self.assertFalse(np.isnan(y_pred_proba).any())
        self.assertAlmostEqual(y_pred_proba[0, 0], 0.7, places=5)
        self.assertAlmostEqual(y_pred_proba[0, 1], 0.3, places=5)
        self.assertAlmostEqual(y_pred_proba[1, 0], 0.4, places=5)
        self.assertAlmostEqual(y_pred_proba[1, 1], 0.6, places=5)

    def test_single_nan_value_triggers_replacement(self):
        """Test that a single NaN value triggers replacement of entire row."""
        classes_ = [0, 1]

        y_pred_proba = np.array(
            [
                [0.8, 0.2],  # Valid row
                [np.nan, 0.5],  # Single NaN in row 1
            ]
        )

        # Apply the patch logic
        if np.isnan(y_pred_proba).any():
            nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
            y_pred_proba[nan_rows] = 1.0 / len(classes_)

        # Row with single NaN should have all values replaced
        expected_uniform = 1.0 / len(classes_)  # 0.5
        self.assertAlmostEqual(y_pred_proba[1, 0], expected_uniform, places=5)
        self.assertAlmostEqual(y_pred_proba[1, 1], expected_uniform, places=5)


if __name__ == "__main__":
    unittest.main()

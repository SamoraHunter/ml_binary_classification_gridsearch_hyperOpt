"""
Unit tests for ml_grid.pipeline.data_scale module.

This test suite validates the data scaling functionality, ensuring that
numeric columns are properly standardized while non-numeric columns are
preserved as expected.
"""

import unittest

import numpy as np
import pandas as pd

try:
    from ml_grid.pipeline.data_scale import data_scale_methods
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_grid.pipeline.data_scale import data_scale_methods


class TestDataScaleMethods(unittest.TestCase):
    """Test suite for the data_scale_methods class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.scaler = data_scale_methods()

        # Create a DataFrame with mixed numeric and non-numeric columns
        self.mixed_data = pd.DataFrame(
            {
                "age": [25, 30, 35, 40],
                "income": [50000, 60000, 70000, 80000],
                "category": ["A", "B", "A", "C"],
            }
        )

    def test_standard_scale_method_with_mixed_columns(self):
        """Test that numeric columns are scaled and non-numeric are preserved."""
        result = self.scaler.standard_scale_method(self.mixed_data)

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that all expected columns are present
        self.assertEqual(set(result.columns), set(self.mixed_data.columns))

        # Check that numeric columns have been scaled (mean ~0, std ~1)
        # Note: sklearn uses population std (ddof=0), so pandas ddof=1 gives ~1.12
        numeric_cols = ["age", "income"]
        for col in numeric_cols:
            self.assertAlmostEqual(result[col].mean(), 0, places=5)
            # Use numpy.std with ddof=0 to match sklearn's behavior
            self.assertAlmostEqual(np.std(result[col], ddof=0), 1, places=5)

        # Check that non-numeric column is preserved
        self.assertTrue(result["category"].equals(self.mixed_data["category"]))

    def test_standard_scale_method_all_numeric_columns(self):
        """Test scaling when all columns are numeric (edge case)."""
        all_numeric = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "feature3": [100, 200, 300, 400, 500],
            }
        )

        result = self.scaler.standard_scale_method(all_numeric)

        # Verify result is a DataFrame with correct columns
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(set(result.columns), set(all_numeric.columns))

        # Verify all numeric columns are standardized using sklearn's ddof=0
        for col in result.columns:
            self.assertAlmostEqual(result[col].mean(), 0, places=5)
            self.assertAlmostEqual(np.std(result[col], ddof=0), 1, places=5)

    def test_standard_scale_method_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()

        # Empty DataFrame should be handled gracefully (returns empty result)
        result = self.scaler.standard_scale_method(empty_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
        self.assertEqual(len(result.columns), 0)

    def test_standard_scale_method_single_column(self):
        """Test scaling with a single numeric column."""
        single_col = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        result = self.scaler.standard_scale_method(single_col)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["value"])
        # Note: sklearn uses population std (ddof=0)
        self.assertAlmostEqual(result["value"].mean(), 0, places=5)
        self.assertAlmostEqual(np.std(result["value"], ddof=0), 1, places=5)


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from ml_grid.util.synthetic_data_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a generator and generate a test DataFrame for all test methods."""
        self.n_rows = 100
        self.n_features = 20
        self.n_outcome_vars = 2
        self.generator = SyntheticDataGenerator(
            n_rows=self.n_rows,
            n_features=self.n_features,
            n_outcome_vars=self.n_outcome_vars,
            verbose=False,
        )
        self.df, self.feature_map = self.generator.generate()

    def test_output_is_dataframe(self):
        """Test that the output is a pandas DataFrame."""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_dataframe_shape(self):
        """Test the shape of the generated DataFrame. It should have features + outcomes + metadata columns."""
        expected_rows = self.n_rows
        # n_features + n_outcome_vars + 'Unnamed: 0' + 'client_idcode'
        expected_cols = self.n_features + self.n_outcome_vars + 2
        self.assertEqual(self.df.shape, (expected_rows, expected_cols))

    def test_number_of_outcome_vars(self):
        """Test that the correct number of outcome variables are generated."""
        outcome_cols = [
            col for col in self.df.columns if col.startswith("outcome_var_")
        ]
        self.assertEqual(len(outcome_cols), self.n_outcome_vars)
        self.assertEqual(len(self.feature_map), self.n_outcome_vars)

    def test_outcome_variables_are_binary(self):
        """Test that the outcome variables are binary (0 or 1)."""
        for i in range(1, self.n_outcome_vars + 1):
            outcome_col = f"outcome_var_{i}"
            self.assertIn(outcome_col, self.df.columns)
            unique_outcomes = self.df[outcome_col].unique()
            # Using a set for comparison is robust to the order and presence of NaNs
            self.assertTrue(set(unique_outcomes).issubset({0, 1}))

    def test_feature_map_correctness(self):
        """Test that the feature map contains correct keys and non-empty lists of features."""
        for i in range(1, self.n_outcome_vars + 1):
            outcome_col = f"outcome_var_{i}"
            self.assertIn(outcome_col, self.feature_map)
            self.assertIsInstance(self.feature_map[outcome_col], list)
            self.assertGreater(len(self.feature_map[outcome_col]), 0)


if __name__ == "__main__":
    unittest.main()

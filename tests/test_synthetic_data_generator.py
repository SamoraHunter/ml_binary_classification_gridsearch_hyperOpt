import unittest
import pandas as pd
from ml_grid.util.synthetic_data_generator import generate_time_series, columns


class TestGenerateTimeSeries(unittest.TestCase):

    def setUp(self):
        """Set up a test DataFrame for all test methods."""
        self.num_clients = 5
        self.num_rows_per_client = 10
        self.df = generate_time_series(self.num_clients, self.num_rows_per_client)

    def test_output_is_dataframe(self):
        """Test that the output is a pandas DataFrame."""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_dataframe_shape(self):
        """Test the shape of the generated DataFrame."""
        expected_rows = self.num_clients * self.num_rows_per_client
        expected_cols = len(columns)
        self.assertEqual(self.df.shape, (expected_rows, expected_cols))

    def test_number_of_unique_clients(self):
        """Test that the number of unique clients is correct."""
        self.assertEqual(self.df['client_idcode'].nunique(), self.num_clients)

    def test_timestamp_column_type(self):
        """Test that the timestamp column has the correct data type."""
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df['timestamp']))

    def test_sorting(self):
        """Test that the DataFrame is sorted by client_idcode and timestamp."""
        # Check if each client's timestamps are sorted
        for client_id in self.df['client_idcode'].unique():
            client_df = self.df[self.df['client_idcode'] == client_id]
            self.assertTrue(client_df['timestamp'].is_monotonic_increasing)

    def test_outcome_variable_is_binary(self):
        """Test that the outcome variable is binary (0 or 1)."""
        outcome_col = 'outcome_var_1'
        self.assertIn(outcome_col, self.df.columns)
        unique_outcomes = self.df[outcome_col].unique()
        self.assertTrue(all(item in [0, 1] for item in unique_outcomes))


if __name__ == '__main__':
    unittest.main()
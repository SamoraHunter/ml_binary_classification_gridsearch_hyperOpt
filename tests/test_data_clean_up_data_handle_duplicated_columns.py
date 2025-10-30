import unittest

import pandas as pd

from ml_grid.pipeline.data_clean_up import clean_up_class


class TestHandleDuplicatedColumns(unittest.TestCase):

    def test_handle_duplicated_columns_normal_case(self):
        # Prepare input DataFrame with duplicated columns
        # Create a sample DataFrame with duplicate columns
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "A": [7, 8, 9]}  # 'A' is duplicated
        df = pd.DataFrame(data)

        # Apply the operation to remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Create an instance of YourClass
        your_instance = clean_up_class()

        # Call the function under test
        result = your_instance.handle_duplicated_columns(df)

        # Check if only unique columns remain
        self.assertListEqual(list(result.columns), ["A", "B"])

    def test_handle_duplicated_columns_empty_dataframe(self):
        # Prepare an empty DataFrame
        X = pd.DataFrame()

        # Create an instance of YourClass
        your_instance = clean_up_class()

        # Call the function under test
        result = your_instance.handle_duplicated_columns(X)

        # Assert that the result is an empty DataFrame
        self.assertTrue(result.empty)

    def test_handle_duplicated_columns_no_duplicates(self):
        # Prepare input DataFrame with no duplicated columns
        X = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
                "D": [10, 11, 12],
                "E": [13, 14, 15],
            }
        )

        # Create an instance of YourClass
        your_instance = clean_up_class()

        # Call the function under test
        result = your_instance.handle_duplicated_columns(X)

        # Assert that the result DataFrame is the same as the input DataFrame
        self.assertTrue(result.equals(X))


if __name__ == "__main__":
    unittest.main()

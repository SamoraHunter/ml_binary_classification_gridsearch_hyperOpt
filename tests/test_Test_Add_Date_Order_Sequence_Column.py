import pandas as pd
import numpy as np
import unittest

# Assuming the function is defined in a module named 'time_series_helper'
from ml_grid.util.time_series_helper import add_date_order_sequence_column


class TestAddDateOrderSequenceColumn(unittest.TestCase):

    def test_convert_timestamp_to_datetime(self):
        # Test converting 'timestamp' column to datetime
        df = pd.DataFrame(
            {"timestamp": ["2022-01-01", "2022-01-02"], "client_idcode": [1, 2]}
        )
        result_df = add_date_order_sequence_column(df)

        self.assertTrue(result_df["timestamp"].dtype == "datetime64[ns, UTC]")

    def test_group_by_and_assign_sequence(self):
        # Test grouping by 'client_idcode' and assigning a sequential order based on timestamp
        df = pd.DataFrame(
            {
                "timestamp": ["2022-01-01", "2022-01-01", "2022-01-02"],
                "client_idcode": [1, 1, 2],
            }
        )
        result_df = add_date_order_sequence_column(df)
        self.assertTrue(result_df["date_order_sequence"].tolist() == [1, 2, 1])


if __name__ == "__main__":
    unittest.main()

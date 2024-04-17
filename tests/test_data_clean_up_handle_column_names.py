import pandas as pd
import re
import unittest

from ml_grid.pipeline.data_clean_up import clean_up_class


class TestHandleColumnNames(unittest.TestCase):

    def test_no_column_rename(self):
        df = pd.DataFrame({"[A": [1, 2], "B": [3, 4]})
        clean_up = clean_up_class()
        clean_up.rename_cols = False
        result = clean_up.handle_column_names(df)
        self.assertTrue(
            all(col in result.columns for col in df.columns),
            "Columns should not be renamed when rename_cols is False.",
        )

    def test_column_rename(self):
        df = pd.DataFrame({"[A": [1, 2], "B": [3, 4]})
        clean_up = clean_up_class()
        clean_up.rename_cols = True
        result = clean_up.handle_column_names(df)
        self.assertTrue(
            all(
                "_" in col
                for col in result.columns
                if any(X in str(col) for X in set(("[", "]", "<")))
            ),
            "Columns with bad characters should be renamed.",
        )

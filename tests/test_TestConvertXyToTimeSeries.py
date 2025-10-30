import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml_grid.util.time_series_helper import convert_Xy_to_time_series


class TestConvertXyToTimeSeries(unittest.TestCase):

    def setUp(self):
        # Sample data generation
        X, y = make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        X["client_idcode"] = np.random.choice(range(100), size=1000)
        y = pd.Series(y)

        # Splitting the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Assuming max_seq_length is defined
        self.max_seq_length = 10

        # Get the actual number of unique clients in the training set
        self.num_unique_clients_train = self.X_train["client_idcode"].nunique()

        # Converting train data into time series format
        self.X_train_ts, self.y_train_ts = convert_Xy_to_time_series(
            self.X_train, self.y_train, self.max_seq_length
        )

    def test_X_train_ts_shape(self):
        self.assertEqual(self.X_train_ts.shape, (self.num_unique_clients_train, 10, 10))

    def test_y_train_ts_shape(self):
        self.assertEqual(self.y_train_ts.shape, (self.num_unique_clients_train,))

    def test_returns_tuple(self):
        result = convert_Xy_to_time_series(
            self.X_train, self.y_train, self.max_seq_length
        )
        self.assertIsInstance(result, tuple)

    def test_returns_correct_format(self):
        X, y = convert_Xy_to_time_series(
            self.X_train, self.y_train, self.max_seq_length
        )
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)


if __name__ == "__main__":
    unittest.main()

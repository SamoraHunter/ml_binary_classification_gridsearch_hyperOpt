import unittest

import numpy as np
import pandas as pd

from ml_grid.pipeline.data_feature_methods import feature_methods


class TestFeatureMethods(unittest.TestCase):

    def setUp(self):
        self.feature_methods = feature_methods()

    def test_getNfeaturesANOVAF_return_correct_number_of_features(self):
        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "feature4": [10, 11, 12],
                "feature5": [13, 14, 15],
            }
        )
        y_train = np.array([1, 0, 1])
        n_features = 1
        result = self.feature_methods.getNfeaturesANOVAF(n_features, X_train, y_train)
        self.assertEqual(len(result), n_features)

    def test_getNfeaturesANOVAF_raise_value_error_for_invalid_input(self):
        X_train = "not_a_dataframe"
        y_train = np.array([1, 0, 1])
        n_features = 1
        with self.assertRaises(ValueError):
            self.feature_methods.getNfeaturesANOVAF(n_features, X_train, y_train)

    def test_getNfeaturesANOVAF_return_expected_top_features(self):
        X_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "feature4": [10, 11, 12],
                "feature5": [13, 14, 15],
            }
        )
        y_train = np.array([1, 0, 1])
        n_features = 1
        result = self.feature_methods.getNfeaturesANOVAF(n_features, X_train, y_train)
        self.assertEqual(result, ["feature1"])


if __name__ == "__main__":
    unittest.main()

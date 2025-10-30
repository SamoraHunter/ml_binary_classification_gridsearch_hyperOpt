import unittest

import numpy as np
import pandas as pd

from ml_grid.pipeline.data_train_test_split import get_data_split, is_valid_shape


class TestDataTrainTestSplit(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame and Series for testing."""
        # Create an imbalanced dataset
        data = {f"feature_{i}": np.random.rand(100) for i in range(5)}
        data["target"] = [0] * 80 + [1] * 20
        self.X = pd.DataFrame(data)
        self.y = self.X.pop("target")

    def test_is_valid_shape(self):
        """Test the is_valid_shape function."""
        self.assertTrue(is_valid_shape(self.X))
        self.assertTrue(is_valid_shape(self.X.values))

        # Test with a 3D numpy array
        invalid_shape_np = np.random.rand(10, 5, 2)
        self.assertFalse(is_valid_shape(invalid_shape_np))

        # Test with a list
        self.assertFalse(is_valid_shape([1, 2, 3]))

    def test_get_data_split_no_resample(self):
        """Test data splitting without any resampling."""
        local_param_dict = {"resample": None}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # Check original test set size (25% of total)
        self.assertEqual(len(X_test_orig), 25)
        self.assertEqual(len(y_test_orig), 25)

        # Check final train/test sizes (split from the initial 75%)
        # 75% of 75 = 56.25 -> 56, 25% of 75 = 18.75 -> 19
        self.assertEqual(len(X_train), 56)
        self.assertEqual(len(y_train), 56)
        self.assertEqual(len(X_test), 19)
        self.assertEqual(len(y_test), 19)

        # Total samples should be conserved
        self.assertEqual(len(X_train) + len(X_test) + len(X_test_orig), len(self.X))

    def test_get_data_split_undersample(self):
        """Test data splitting with undersampling."""
        local_param_dict = {"resample": "undersample"}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # Original data is split 75/25 -> 75/25.
        # X_test_orig should have 25% of 100 samples = 25.
        self.assertEqual(len(X_test_orig), 25)

        # The initial training set of 75 (60 class 0, 15 class 1) is undersampled
        # to have 15 of each class, totaling 30.
        # This is then split 75/25 -> 22/8
        self.assertEqual(len(X_train), 22)
        self.assertEqual(len(y_train), 22)
        self.assertEqual(len(X_test), 8)
        self.assertEqual(len(y_test), 8)

        # Check if the training set is balanced after the full process
        # The final y_train comes from a split of a balanced set, so it should be roughly balanced
        # With 22 samples, it should be 11 of each.
        self.assertEqual(y_train.value_counts()[0], 11)
        self.assertEqual(y_train.value_counts()[1], 11)

    def test_get_data_split_oversample(self):
        """Test data splitting with oversampling."""
        local_param_dict = {"resample": "oversample"}
        X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = get_data_split(
            self.X, self.y, local_param_dict
        )

        # Original data is split 75/25 -> 75/25.
        # With stratification, y_test_orig should have 25% of each class:
        # 25% of 80 (class 0) = 20
        # 25% of 20 (class 1) = 5
        self.assertEqual(len(X_test_orig), 25)

        # The initial training set of 75 (60 class 0, 15 class 1) is oversampled
        # to have 60 of each class, totaling 120.
        # This is then split 75/25 -> 90/30
        self.assertEqual(len(X_train), 90)
        self.assertEqual(len(y_train), 90)
        self.assertEqual(len(X_test), 30)

        # The final y_train should be as balanced as possible.
        # With 90 samples from a balanced set, the split should be perfectly balanced.
        self.assertEqual(y_train.value_counts()[0], 45)
        self.assertEqual(y_train.value_counts()[1], 45)

    def test_invalid_shape_overrides_resample(self):
        """Test that resampling is disabled for invalid (e.g., 3D) data shapes."""
        X_3d = np.random.rand(100, 5, 2)
        y_3d = pd.Series(self.y.values)  # y can remain 1D

        local_param_dict = {"resample": "oversample"}
        # This should run without error and default to 'resample': None
        X_train, _, _, _, _, _ = get_data_split(X_3d, y_3d, local_param_dict)
        # Check that the dictionary was modified in-place
        self.assertIsNone(local_param_dict["resample"])
        self.assertEqual(X_train.shape[0], 56)  # Should match 'no resample' case


if __name__ == "__main__":
    unittest.main()

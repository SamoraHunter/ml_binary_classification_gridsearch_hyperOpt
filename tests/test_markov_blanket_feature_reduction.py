import unittest

import lightgbm as lgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml_grid.pipeline.data_feature_methods import (
    feature_methods,
)


class TestGetNFeaturesMarkovBlanket(unittest.TestCase):
    def test_number_of_features(self):
        # Generate synthetic data for binary classification
        X, y = make_classification(
            n_samples=150, n_features=10, n_informative=3, n_classes=2, random_state=42
        )

        # Split the data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42)

        # Scale features to be in the range [0, 1] to avoid issues with log_loss in PyImpetus
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        # Convert numpy array to pandas DataFrame as the method expects it
        X_train = pd.DataFrame(
            X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])]
        )

        # Create an instance of the correct class
        my_instance = feature_methods()

        # Define the classifier to be used by PyImpetus
        classifier = lgb.LGBMClassifier(random_state=42, verbosity=-1)

        # Call the function to get the top 5 features
        top_features = my_instance.getNFeaturesMarkovBlanket(
            5, X_train, y_train, classifier=classifier
        )

        # Assert that the number of features returned is less than or equal to
        # the number requested, and that some features are returned.
        self.assertLessEqual(len(top_features), 5)
        self.assertGreater(len(top_features), 0)


if __name__ == "__main__":
    unittest.main()

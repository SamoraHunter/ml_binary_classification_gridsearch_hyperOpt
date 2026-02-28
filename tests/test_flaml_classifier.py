import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from ml_grid.model_classes.FLAMLClassifierWrapper import FLAMLClassifierWrapper
from ml_grid.model_classes.flaml_classifier_class import FLAMLClassifierClass


class TestFLAMLClassifier(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {"feature_0": [1.0, 2.0, 3.0, 4.0], "feature_1": [4.0, 3.0, 2.0, 1.0]}
        )
        self.y = pd.Series([0, 1, 0, 1], name="target")

    def test_init(self):
        clf = FLAMLClassifierWrapper(time_budget=120, metric="roc_auc")
        self.assertEqual(clf.time_budget, 120)
        self.assertEqual(clf.metric, "roc_auc")
        self.assertIsNone(clf.model_)

    @patch("ml_grid.model_classes.FLAMLClassifierWrapper.AutoML")
    def test_fit(self, mock_automl_cls):
        # Setup mocks
        mock_automl_instance = MagicMock()
        mock_automl_cls.return_value = mock_automl_instance

        clf = FLAMLClassifierWrapper(time_budget=60)

        # Test fit
        clf.fit(self.X, self.y)

        # Verify AutoML init
        mock_automl_cls.assert_called_once()

        # Verify fit call
        mock_automl_instance.fit.assert_called_once()
        _, kwargs = mock_automl_instance.fit.call_args
        self.assertEqual(kwargs["time_budget"], 60)
        self.assertEqual(kwargs["task"], "classification")

        # Verify attributes set
        self.assertIsNotNone(clf.model_)

    @patch("ml_grid.model_classes.FLAMLClassifierWrapper.AutoML")
    def test_predict(self, mock_automl_cls):
        # Setup mock
        mock_automl_instance = MagicMock()
        mock_automl_cls.return_value = mock_automl_instance

        # Mock predict return
        mock_automl_instance.predict.return_value = np.array([0, 1, 0, 1])

        clf = FLAMLClassifierWrapper()
        clf.fit(self.X, self.y)

        preds = clf.predict(self.X)

        self.assertIsInstance(preds, np.ndarray)
        np.testing.assert_array_equal(preds, np.array([0, 1, 0, 1]))

        # Verify predict called on internal model
        mock_automl_instance.predict.assert_called_once_with(self.X)

    def test_missing_flaml(self):
        # Simulate missing flaml by patching AutoML to None
        with patch("ml_grid.model_classes.FLAMLClassifierWrapper.AutoML", None):
            clf = FLAMLClassifierWrapper()
            with self.assertRaises(ImportError):
                clf.fit(self.X, self.y)


class TestFLAMLClassifierClass(unittest.TestCase):
    def test_structure(self):
        # Mock global_parameters to control bayessearch flag
        with patch(
            "ml_grid.model_classes.flaml_classifier_class.global_parameters"
        ) as mock_globals:
            # Case 1: Grid Search (bayessearch = False)
            mock_globals.bayessearch = False

            config = FLAMLClassifierClass()
            self.assertEqual(config.method_name, "FLAMLClassifier")
            self.assertIsInstance(
                config.algorithm_implementation, FLAMLClassifierWrapper
            )
            self.assertIsInstance(config.parameter_space, list)

            # Case 2: Bayes Search (bayessearch = True)
            mock_globals.bayessearch = True

            config = FLAMLClassifierClass()
            self.assertIsInstance(config.parameter_space, dict)
            self.assertIn("time_budget", config.parameter_space)


if __name__ == "__main__":
    unittest.main()

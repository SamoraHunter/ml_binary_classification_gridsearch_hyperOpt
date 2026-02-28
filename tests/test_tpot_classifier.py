import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from ml_grid.model_classes.TPOTClassifierWrapper import TPOTClassifierWrapper
from ml_grid.model_classes.tpot_classifier_class import TPOTClassifierClass


class TestTPOTClassifier(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {"feature_0": [1.0, 2.0, 3.0, 4.0], "feature_1": [4.0, 3.0, 2.0, 1.0]}
        )
        self.y = pd.Series([0, 1, 0, 1], name="target")

    def test_init(self):
        clf = TPOTClassifierWrapper(generations=10, population_size=50)
        self.assertEqual(clf.generations, 10)
        self.assertEqual(clf.population_size, 50)
        self.assertIsNone(clf.model_)

    @patch("ml_grid.model_classes.TPOTClassifierWrapper.TPOTClassifier")
    def test_fit(self, mock_tpot_cls):
        # Setup mocks
        mock_tpot_instance = MagicMock()
        mock_tpot_cls.return_value = mock_tpot_instance

        clf = TPOTClassifierWrapper(generations=5, population_size=20)

        # Test fit
        clf.fit(self.X, self.y)

        # Verify TPOTClassifier init
        mock_tpot_cls.assert_called_once()
        _, kwargs = mock_tpot_cls.call_args
        self.assertEqual(kwargs["generations"], 5)
        self.assertEqual(kwargs["population_size"], 20)
        self.assertEqual(kwargs["disable_update_check"], True)

        # Verify fit call
        mock_tpot_instance.fit.assert_called_once_with(self.X, self.y)

        # Verify attributes set
        self.assertIsNotNone(clf.model_)

    @patch("ml_grid.model_classes.TPOTClassifierWrapper.TPOTClassifier")
    def test_predict(self, mock_tpot_cls):
        # Setup mock
        mock_tpot_instance = MagicMock()
        mock_tpot_cls.return_value = mock_tpot_instance

        # Mock predict return
        mock_tpot_instance.predict.return_value = np.array([0, 1, 0, 1])

        clf = TPOTClassifierWrapper()
        clf.fit(self.X, self.y)

        preds = clf.predict(self.X)

        self.assertIsInstance(preds, np.ndarray)
        np.testing.assert_array_equal(preds, np.array([0, 1, 0, 1]))

        # Verify predict called on internal model
        mock_tpot_instance.predict.assert_called_once_with(self.X)

    def test_missing_tpot(self):
        # Simulate missing tpot by patching TPOTClassifier to None
        with patch("ml_grid.model_classes.TPOTClassifierWrapper.TPOTClassifier", None):
            clf = TPOTClassifierWrapper()
            with self.assertRaises(ImportError):
                clf.fit(self.X, self.y)


class TestTPOTClassifierClass(unittest.TestCase):
    def test_structure(self):
        # Mock global_parameters to control bayessearch flag
        with patch(
            "ml_grid.model_classes.tpot_classifier_class.global_parameters"
        ) as mock_globals:
            # Case 1: Grid Search (bayessearch = False)
            mock_globals.bayessearch = False

            config = TPOTClassifierClass()
            self.assertEqual(config.method_name, "TPOTClassifier")
            self.assertIsInstance(
                config.algorithm_implementation, TPOTClassifierWrapper
            )
            self.assertIsInstance(config.parameter_space, list)

            # Case 2: Bayes Search (bayessearch = True)
            mock_globals.bayessearch = True

            config = TPOTClassifierClass()
            self.assertIsInstance(config.parameter_space, dict)
            self.assertIn("generations", config.parameter_space)
            self.assertIn("population_size", config.parameter_space)


if __name__ == "__main__":
    unittest.main()

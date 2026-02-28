import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from ml_grid.model_classes.AutoKerasClassifierWrapper import AutoKerasClassifierWrapper
from ml_grid.model_classes.auto_keras_classifier_class import AutoKerasClassifierClass


class TestAutoKerasClassifier(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {"feature_0": [1.0, 2.0, 3.0, 4.0], "feature_1": [4.0, 3.0, 2.0, 1.0]}
        )
        self.y = pd.Series([0, 1, 0, 1], name="target")

    def test_init(self):
        clf = AutoKerasClassifierWrapper(max_trials=5, epochs=20)
        self.assertEqual(clf.max_trials, 5)
        self.assertEqual(clf.epochs, 20)
        self.assertIsNone(clf.model_)

    @patch(
        "ml_grid.model_classes.AutoKerasClassifierWrapper.ak.StructuredDataClassifier"
    )
    @patch("ml_grid.model_classes.AutoKerasClassifierWrapper.tempfile.mkdtemp")
    @patch("ml_grid.model_classes.AutoKerasClassifierWrapper.shutil.rmtree")
    def test_fit(self, mock_rmtree, mock_mkdtemp, mock_ak_cls):
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/mock_autokeras_dir"
        mock_ak_instance = MagicMock()
        mock_ak_cls.return_value = mock_ak_instance

        clf = AutoKerasClassifierWrapper(max_trials=2)

        # Test fit
        clf.fit(self.X, self.y)

        # Verify StructuredDataClassifier init
        mock_ak_cls.assert_called_once()
        _, kwargs = mock_ak_cls.call_args
        self.assertEqual(kwargs["max_trials"], 2)
        self.assertEqual(kwargs["directory"], "/tmp/mock_autokeras_dir")

        # Verify fit call
        mock_ak_instance.fit.assert_called_once()
        _, fit_kwargs = mock_ak_instance.fit.call_args
        self.assertEqual(fit_kwargs["epochs"], 10)
        np.testing.assert_array_equal(fit_kwargs["x"], self.X.values)
        np.testing.assert_array_equal(fit_kwargs["y"], self.y.values)

        # Verify attributes set
        self.assertIsNotNone(clf.model_)

    @patch(
        "ml_grid.model_classes.AutoKerasClassifierWrapper.ak.StructuredDataClassifier"
    )
    def test_predict(self, mock_ak_cls):
        # Setup mock
        mock_ak_instance = MagicMock()
        mock_ak_cls.return_value = mock_ak_instance

        # Mock predict return (AutoKeras returns array of shape (N, 1))
        mock_ak_instance.predict.return_value = np.array([[0], [1], [0], [1]])

        clf = AutoKerasClassifierWrapper()
        clf.fit(self.X, self.y)

        preds = clf.predict(self.X)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (4,))
        np.testing.assert_array_equal(preds, np.array([0, 1, 0, 1]))

    @patch(
        "ml_grid.model_classes.AutoKerasClassifierWrapper.ak.StructuredDataClassifier"
    )
    def test_predict_proba(self, mock_ak_cls):
        # Setup mock
        mock_ak_instance = MagicMock()
        mock_ak_cls.return_value = mock_ak_instance

        # Mock export_model and its predict method
        mock_keras_model = MagicMock()
        mock_ak_instance.export_model.return_value = mock_keras_model
        # Return (N, 1) probabilities for binary classification
        mock_keras_model.predict.return_value = np.array([[0.1], [0.9], [0.2], [0.8]])

        clf = AutoKerasClassifierWrapper()
        clf.fit(self.X, self.y)

        probas = clf.predict_proba(self.X)

        self.assertIsInstance(probas, np.ndarray)
        self.assertEqual(probas.shape, (4, 2))  # Should be converted to (N, 2)
        self.assertAlmostEqual(probas[0, 0], 0.9)  # 1 - 0.1
        self.assertAlmostEqual(probas[0, 1], 0.1)

        # Verify predict called on internal keras model with numpy array
        mock_keras_model.predict.assert_called_once()
        np.testing.assert_array_equal(
            mock_keras_model.predict.call_args[0][0], self.X.values
        )


class TestAutoKerasClassifierClass(unittest.TestCase):
    def test_structure(self):
        with patch(
            "ml_grid.model_classes.auto_keras_classifier_class.global_parameters"
        ) as mock_globals:
            # Case 1: Grid Search
            mock_globals.test_mode = False
            mock_globals.bayessearch = False
            config = AutoKerasClassifierClass()
            self.assertEqual(config.method_name, "AutoKerasClassifier")
            self.assertIsInstance(config.parameter_space, list)

            # Case 2: Bayes Search
            mock_globals.bayessearch = True
            config = AutoKerasClassifierClass()
            self.assertIsInstance(config.parameter_space, dict)
            self.assertIn("max_trials", config.parameter_space)


if __name__ == "__main__":
    unittest.main()

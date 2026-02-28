import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from ml_grid.model_classes.AutoGluonClassifier import AutoGluonClassifier
from ml_grid.model_classes.auto_gluon_classifier_class import AutoGluonClassifierClass


class TestAutoGluonClassifier(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {"feature_0": [1.0, 2.0, 3.0, 4.0], "feature_1": [4.0, 3.0, 2.0, 1.0]}
        )
        self.y = pd.Series([0, 1, 0, 1], name="target")

    def test_init(self):
        clf = AutoGluonClassifier(time_limit=120, presets="high_quality")
        self.assertEqual(clf.time_limit, 120)
        self.assertEqual(clf.presets, "high_quality")
        self.assertIsNone(clf.predictor_)

    @patch("ml_grid.model_classes.AutoGluonClassifier.TabularPredictor")
    @patch("ml_grid.model_classes.AutoGluonClassifier.tempfile.mkdtemp")
    @patch("ml_grid.model_classes.AutoGluonClassifier.shutil.rmtree")
    def test_fit(self, mock_rmtree, mock_mkdtemp, mock_predictor_cls):
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/mock_autogluon_dir"

        mock_predictor_instance = MagicMock()
        mock_predictor_cls.return_value = mock_predictor_instance
        mock_predictor_instance.class_labels = [0, 1]

        clf = AutoGluonClassifier(time_limit=60)

        # Test fit
        clf.fit(self.X, self.y)

        # Verify TabularPredictor init
        mock_predictor_cls.assert_called_once()
        _, kwargs = mock_predictor_cls.call_args
        self.assertEqual(kwargs["label"], "target")
        self.assertEqual(kwargs["path"], "/tmp/mock_autogluon_dir")

        # Verify fit call
        mock_predictor_instance.fit.assert_called_once()
        _, fit_kwargs = mock_predictor_instance.fit.call_args
        self.assertEqual(fit_kwargs["time_limit"], 45)

        # Verify attributes set
        self.assertTrue(hasattr(clf, "classes_"))
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
        self.assertIsNotNone(clf.model_id)

    @patch("ml_grid.model_classes.AutoGluonClassifier.TabularPredictor")
    def test_predict(self, mock_predictor_cls):
        # Setup mock
        mock_predictor_instance = MagicMock()
        mock_predictor_cls.return_value = mock_predictor_instance
        mock_predictor_instance.class_labels = [0, 1]

        # Mock predict return
        mock_predictor_instance.predict.return_value = pd.Series([0, 1, 0, 1])

        clf = AutoGluonClassifier()
        clf.fit(self.X, self.y)

        preds = clf.predict(self.X)

        self.assertIsInstance(preds, np.ndarray)
        np.testing.assert_array_equal(preds, np.array([0, 1, 0, 1]))

        # Verify predict called on predictor
        mock_predictor_instance.predict.assert_called_once()

    @patch("ml_grid.model_classes.AutoGluonClassifier.TabularPredictor")
    def test_predict_proba(self, mock_predictor_cls):
        # Setup mock
        mock_predictor_instance = MagicMock()
        mock_predictor_cls.return_value = mock_predictor_instance
        mock_predictor_instance.class_labels = [0, 1]

        # Mock predict_proba return (DataFrame)
        proba_df = pd.DataFrame({0: [0.9, 0.1, 0.8, 0.2], 1: [0.1, 0.9, 0.2, 0.8]})
        mock_predictor_instance.predict_proba.return_value = proba_df

        clf = AutoGluonClassifier()
        clf.fit(self.X, self.y)

        probas = clf.predict_proba(self.X)

        self.assertIsInstance(probas, np.ndarray)
        self.assertEqual(probas.shape, (4, 2))
        np.testing.assert_array_almost_equal(probas, proba_df.values)

    def test_missing_autogluon(self):
        # Simulate missing autogluon by patching TabularPredictor to None
        with patch("ml_grid.model_classes.AutoGluonClassifier.TabularPredictor", None):
            clf = AutoGluonClassifier()
            with self.assertRaises(ImportError):
                clf.fit(self.X, self.y)


class TestAutoGluonClassifierClass(unittest.TestCase):
    def test_structure(self):
        # Mock global_parameters to control bayessearch flag
        with patch(
            "ml_grid.model_classes.auto_gluon_classifier_class.global_parameters"
        ) as mock_globals:
            mock_globals.test_mode = False
            # Case 1: Grid Search (bayessearch = False)
            mock_globals.bayessearch = False

            config = AutoGluonClassifierClass()
            self.assertEqual(config.method_name, "AutoGluonClassifier")
            self.assertIsInstance(config.algorithm_implementation, AutoGluonClassifier)
            self.assertIsInstance(config.parameter_space, list)

            # Case 2: Bayes Search (bayessearch = True)
            mock_globals.bayessearch = True

            config = AutoGluonClassifierClass()
            self.assertIsInstance(config.parameter_space, dict)
            self.assertIn("time_limit", config.parameter_space)

            # Case 3: Test Mode
            mock_globals.test_mode = True
            config = AutoGluonClassifierClass()
            self.assertIsInstance(config.parameter_space, list)
            self.assertEqual(config.parameter_space[0]["time_limit"], [5])


if __name__ == "__main__":
    unittest.main()

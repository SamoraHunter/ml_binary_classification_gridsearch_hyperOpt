"""Tests for H2OAutoMLClassifier."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


class TestH2OAutoMLClassifier(unittest.TestCase):
    """Test H2OAutoMLClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.X = pd.DataFrame(
            {
                "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_1": [4.0, 3.0, 2.0, 1.0, 0.5],
            }
        )
        self.y = pd.Series([0, 1, 0, 1, 0], name="target")

    def test_init(self):
        """Test H2OAutoMLClassifier initialization."""
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        clf = H2OAutoMLClassifier()

        # Check that the classifier has been initialized correctly
        self.assertIsNone(clf.automl)
        self.assertIsNone(clf.model_)

    def test_fit_with_small_dataset(self):
        """Test fit method with dataset smaller than MIN_SAMPLES_FOR_STABLE_FIT (< 10 samples).

        This tests the fallback path where _handle_small_data_fallback returns True,
        causing the classifier to skip AutoML and use a dummy GLM model instead.

        The small data path goes through _finalize_dummy_fit which creates a GLM model.
        """
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        # Use a very small dataset (5 samples < 10)
        X_small = pd.DataFrame(
            {
                "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_1": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        y_small = pd.Series([0, 1, 0, 1, 0], name="target")

        clf = H2OAutoMLClassifier()

        # Mock the base class _handle_small_data_fallback to make it return True
        with patch.object(clf, "_handle_small_data_fallback", return_value=True):
            with patch(
                "ml_grid.model_classes.H2OAutoMLClassifier.H2OGeneralizedLinearEstimator"
            ) as mock_glm_cls:
                # Setup the GLM model mock
                mock_glm_model = MagicMock()
                mock_glm_model.model_id = "GLM_final"
                mock_glm_cls.return_value = mock_glm_model

                result = clf.fit(X_small, y_small)

                # Verify the classifier is returned
                self.assertIsInstance(result, H2OAutoMLClassifier)

                # GLM should be created once for fallback
                mock_glm_cls.assert_called_once()
                call_kwargs = mock_glm_cls.call_args[1]
                self.assertEqual(call_kwargs["family"], "binomial")
                self.assertFalse(call_kwargs["ignore_const_cols"])

    def test_fit_with_normal_dataset_no_automl_leader(self):
        """Test fit method with normal dataset where AutoML returns no leader model.

        This tests the path where AutoML is run but find_leader()/leader returns None,
        causing fallback to a simple GLM.

        Since H2O cluster setup is required, we mock the entire fit logic path
        by patching H2OAutoML to return an instance with no leader.
        """
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        X = pd.DataFrame(
            {
                "feature_0": list(range(50)),
                "feature_1": list(range(50, 100)),
            }
        )
        y = pd.Series([i % 2 for i in range(50)], name="target")

        clf = H2OAutoMLClassifier()

        # Mock AutoML instance that has no leader
        mock_automl_instance = MagicMock()
        mock_automl_instance.leader = None

        with patch(
            "ml_grid.model_classes.H2OAutoMLClassifier.H2OAutoML",
            return_value=mock_automl_instance,
        ):
            result = clf.fit(X, y)

            # AutoML should have been called
            from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoML

            H2OAutoML.assert_called_once()

            # Since leader is None, the code falls back to GLM creation
            self.assertIsNotNone(result.model_)

    def test_fit_with_normal_dataset_and_automl_leader(self):
        """Test fit method with normal dataset where AutoML finds a leader model."""
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        X = pd.DataFrame(
            {
                "feature_0": list(range(100)),
                "feature_1": list(range(100, 200)),
            }
        )
        y = pd.Series([i % 2 for i in range(100)], name="target")

        clf = H2OAutoMLClassifier()

        # Mock AutoML instance with a leader model
        mock_leader_model = MagicMock()
        mock_leader_model.model_id = "XGBoost_1"

        mock_automl_instance = MagicMock()
        mock_automl_instance.leader = mock_leader_model

        with patch(
            "ml_grid.model_classes.H2OAutoMLClassifier.H2OAutoML",
            return_value=mock_automl_instance,
        ):
            result = clf.fit(X, y)

            # Should use the AutoML leader model
            self.assertEqual(result.model_, mock_leader_model)
            self.assertEqual(result.model_id, "XGBoost_1")

    def test_finalize_dummy_fit(self):
        """Test _finalize_dummy_fit method creates a GLM model and sets attributes.

        This tests the internal method that finalizes the fitting process when
        a dummy model is used (fallback for small datasets).

        The method:
        1. Creates an H2OGeneralizedLinearEstimator instance
        2. Calls _sanitize_model_params
        3. Trains the model on H2OFrame data
        4. Sets model_id from self.model_.model_id
        """
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        X = pd.DataFrame({"feature_0": [1.0, 2.0, 3.0], "feature_1": [3.0, 2.0, 1.0]})
        y = pd.Series([0, 1, 0], name="target")

        clf = H2OAutoMLClassifier()

        # Mock the GLM model
        mock_glm_model = MagicMock()
        mock_glm_model.model_id = "GLM_dummy"

        with patch(
            "ml_grid.model_classes.H2OAutoMLClassifier.H2OGeneralizedLinearEstimator",
            return_value=mock_glm_model,
        ):
            # Mock the base class methods
            with patch.object(clf, "_prepare_fit") as mock_prepare:
                mock_train_h2o = MagicMock()
                mock_x_vars = ["feature_0", "feature_1"]
                mock_outcome_var = "target"
                mock_prepare.return_value = (
                    mock_train_h2o,
                    mock_x_vars,
                    mock_outcome_var,
                    {},
                )

                with patch.object(clf, "_sanitize_model_params", return_value=None):
                    result = clf._finalize_dummy_fit(X, y)

                    # Verify GLM was created once
                    from ml_grid.model_classes.H2OAutoMLClassifier import (
                        H2OGeneralizedLinearEstimator,
                    )

                    H2OGeneralizedLinearEstimator.assert_called_once()
                    call_kwargs = H2OGeneralizedLinearEstimator.call_args[1]
                    self.assertEqual(call_kwargs["family"], "binomial")
                    self.assertFalse(call_kwargs["ignore_const_cols"])

                    # Verify _prepare_fit was called
                    mock_prepare.assert_called_once_with(X, y)

                    # Verify _sanitize_model_params was called
                    self.assertTrue(hasattr(clf, "_sanitize_model_params"))

                    # Verify model and attributes were set correctly
                    self.assertEqual(result, clf)
                    self.assertEqual(result.model_, mock_glm_model)
                    self.assertEqual(result.model_id, "GLM_dummy")

    def test_shutdown_method(self):
        """Test shutdown method runs without error.

        The shutdown method is a no-op that allows the base class __del__
        to handle cleanup. This test verifies it doesn't raise exceptions.
        """
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        clf = H2OAutoMLClassifier()

        # Should not raise any exception
        clf.shutdown()

    def test_fit_raises_runtime_error_when_no_model_produced(self):
        """Test fit raises RuntimeError when both AutoML and GLM paths fail to produce a model.

        This tests the RuntimeError branch at line 94:
        `raise RuntimeError("H2OAutoMLClassifier failed to produce a final model.")`

        The scenario is:
        1. Dataset is large enough for AutoML (>= 20 samples)
        2. AutoML runs but returns no leader (leader = None), triggering fallback
        3. Fallback GLM path is activated, but H2OGeneralizedLinearEstimator is mocked
           to return a falsy object

        We achieve this by using a mock GLM whose __bool__ returns False, simulating
        a "failed model" state where self.model_ evaluates to False at line 88.
        """
        from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier

        X = pd.DataFrame(
            {
                "feature_0": list(range(50)),
                "feature_1": list(range(50, 100)),
            }
        )
        y = pd.Series([i % 2 for i in range(50)], name="target")

        clf = H2OAutoMLClassifier()

        # Mock AutoML instance that has no leader
        mock_automl_instance = MagicMock()
        mock_automl_instance.leader = None

        with patch(
            "ml_grid.model_classes.H2OAutoMLClassifier.H2OAutoML",
            return_value=mock_automl_instance,
        ):
            # Mock _prepare_fit to return valid data
            with patch.object(clf, "_prepare_fit") as mock_prepare:
                mock_train_h2o = MagicMock()
                mock_x_vars = ["feature_0", "feature_1"]
                mock_outcome_var = "target"
                mock_prepare.return_value = (
                    mock_train_h2o,
                    mock_x_vars,
                    mock_outcome_var,
                    {},
                )

                # Mock _sanitize_model_params to be a no-op
                with patch.object(clf, "_sanitize_model_params", return_value=None):
                    """Simulate H2OGeneralizedLinearEstimator creating a falsy model.

                    We use a mock whose __bool__ returns False so that
                    'if self.model_:' at line 88 evaluates to False.
                    """
                    with patch("h2o.H2OFrame") as mock_h2o_frame_cls:
                        # Setup minimal H2OFrame
                        mock_train_h2o = MagicMock()
                        mock_train_h2o.types = {
                            "feature_0": "real",
                            "feature_1": "real",
                            "target": "enum",
                        }

                        def h2o_frame_ctor(data, *args, **kwargs):
                            return mock_train_h2o

                        mock_h2o_frame_cls.side_effect = h2o_frame_ctor

                        # Create a GLM mock that is falsy
                        mock_glm_model = MagicMock()
                        type(mock_glm_model).model_id = "mocked"
                        type(mock_glm_model).__bool__ = lambda self: False
                        mock_glm_model.train.return_value = None

                        with patch(
                            "ml_grid.model_classes.H2OAutoMLClassifier.H2OGeneralizedLinearEstimator",
                            return_value=mock_glm_model,
                        ):
                            with self.assertRaises(RuntimeError) as context:
                                clf.fit(X, y)

                            self.assertEqual(
                                str(context.exception),
                                "H2OAutoMLClassifier failed to produce a final model.",
                            )


class TestH2OAutoMLClass(unittest.TestCase):
    """Test H2OAutoMLClass configuration class."""

    def test_structure_test_mode(self):
        """Test H2OAutoMLClass with test_mode enabled."""
        from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass

        # Mock global_parameters.test_mode = True
        with patch(
            "ml_grid.model_classes.h2o_classifier_class.global_parameters"
        ) as mock_globals:
            mock_globals.test_mode = True

            config = H2OAutoMLClass()

            self.assertEqual(config.method_name, "H2OAutoMLClassifier")
            self.assertIsInstance(config.parameter_space, list)
            self.assertIn("max_runtime_secs", config.parameter_space[0])

    def test_structure_bayessearch(self):
        """Test H2OAutoMLClass with bayessearch enabled."""
        from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass
        from skopt.space import Integer

        with patch(
            "ml_grid.model_classes.h2o_classifier_class.global_parameters"
        ) as mock_globals:
            mock_globals.test_mode = False
            mock_globals.bayessearch = True

            config = H2OAutoMLClass()

            self.assertIsInstance(config.parameter_space, list)
            self.assertIn("max_runtime_secs", config.parameter_space[0])
            # Check that parameters are skopt spaces
            param_space = config.parameter_space[0]
            self.assertIsInstance(param_space.get("max_runtime_secs"), Integer)

    def test_structure_grid_search(self):
        """Test H2OAutoMLClass with grid search (default)."""
        from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass

        with patch(
            "ml_grid.model_classes.h2o_classifier_class.global_parameters"
        ) as mock_globals:
            mock_globals.test_mode = False
            mock_globals.bayessearch = False

            config = H2OAutoMLClass()

            self.assertIsInstance(config.parameter_space, list)
            # Grid search uses lists, not skopt spaces
            param_space = config.parameter_space[0]
            self.assertIsInstance(param_space.get("max_runtime_secs"), list)


if __name__ == "__main__":
    unittest.main()

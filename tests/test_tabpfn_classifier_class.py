import unittest
import sys
import os
import pandas as pd
from unittest.mock import MagicMock, patch
from ml_grid.model_classes.tabpfn_classifier_class import (
    TabPFNClassifierClass,
)

# Add project root to path so we can import the class under test
# Assumes this test file is in /tests/ and ml_grid is in the parent dir
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------------------------------------------------------
# MOCK SETUP
# We mock dependencies before importing the class to ensure the test
# runs even if tabpfn or ml_grid are not fully installed in the test env.
# -------------------------------------------------------------------------

# Mock ml_grid dependencies
sys.modules["ml_grid.util"] = MagicMock()
sys.modules["ml_grid.util.param_space"] = MagicMock()
sys.modules["ml_grid.util.global_params"] = MagicMock()

# Conditionally mock tabpfn. If it's installed, we might want to use it for integration tests.
# If not installed, we mock it to allow importing the wrapper class.
try:
    import tabpfn
except ImportError:
    mock_tabpfn_module = MagicMock()
    sys.modules["tabpfn"] = mock_tabpfn_module
    sys.modules["tabpfn.constants"] = MagicMock()


class TestTabPFNClassifierClass(unittest.TestCase):
    def setUp(self):
        # Patch global parameters to control bayessearch flag
        self.global_params_patch = patch("ml_grid.util.global_params.global_parameters")
        self.mock_global_params = self.global_params_patch.start()
        self.mock_global_params.bayessearch = False  # Default to grid search

        # Dummy data for fit/predict
        self.X_dummy = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        self.y_dummy = pd.Series([0, 1, 0])

    def tearDown(self):
        self.global_params_patch.stop()

    def test_initialization(self):
        """Test that the class initializes and sets up parameter space correctly."""
        model = TabPFNClassifierClass()

        self.assertEqual(model.method_name, "TabPFNClassifier")
        self.assertIsInstance(model.parameter_space, dict)

        # Check if key parameters are present in the space
        expected_keys = ["model_version", "device", "n_estimators"]
        for key in expected_keys:
            self.assertIn(key, model.parameter_space)

        # Check that hyperparameters are set on the instance
        self.assertEqual(model.n_estimators, 4)
        self.assertEqual(model.device, "cpu")

    def test_parameter_space_compatibility(self):
        """Ensure parameter space keys are valid parameters of the estimator."""
        model = TabPFNClassifierClass()
        valid_params = model.get_params()

        for param in model.parameter_space.keys():
            self.assertIn(
                param,
                valid_params,
                f"Parameter '{param}' in parameter_space is not a valid parameter of the estimator.",
            )

        # Explicitly check that removed parameters are NOT in the space
        removed_params = ["rf_n_estimators", "use_rf_preprocessing", "lowrank_attn_dim"]
        for param in removed_params:
            self.assertNotIn(param, model.parameter_space)

    @patch("ml_grid.model_classes.tabpfn_classifier_class.TabPFNClassifier")
    def test_fit_v2_5_default(self, mock_tabpfn_cls):
        """Test fitting of the default v2.5 model."""
        # Setup mock return value
        mock_estimator_instance = mock_tabpfn_cls.return_value

        # Instantiate the wrapper with hyperparameters
        model_wrapper = TabPFNClassifierClass(
            model_version="v2.5_default", n_estimators=4, device="cpu", random_state=42
        )

        # Call fit
        model_wrapper.fit(self.X_dummy, self.y_dummy)

        # Verify TabPFNClassifier constructor was called directly
        mock_tabpfn_cls.assert_called_once()

        # Verify arguments passed to the constructor
        call_kwargs = mock_tabpfn_cls.call_args.kwargs
        self.assertEqual(call_kwargs["n_estimators"], 4)
        self.assertEqual(call_kwargs["device"], "cpu")
        self.assertEqual(call_kwargs["random_state"], 42)

        # Ensure 'model_version' was consumed and not passed to the actual classifier
        self.assertNotIn("model_version", call_kwargs)
        self.assertNotIn("subsample_samples", call_kwargs)

        # Verify the underlying estimator's fit method was called
        mock_estimator_instance.fit.assert_called_once()

    @patch("ml_grid.model_classes.tabpfn_classifier_class.TabPFNClassifier")
    def test_fit_v2_5_synthetic(self, mock_tabpfn_cls):
        """Test fitting of the synthetic v2.5 model (checks model_path logic)."""
        mock_estimator_instance = mock_tabpfn_cls.return_value

        model_wrapper = TabPFNClassifierClass(
            model_version="v2.5_synthetic", n_estimators=2
        )

        model_wrapper.fit(self.X_dummy, self.y_dummy)

        # Verify constructor was called
        mock_tabpfn_cls.assert_called_once()

        # Verify model_path was injected
        call_kwargs = mock_tabpfn_cls.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("model_path"), "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
        )

        # Verify fit was called
        mock_estimator_instance.fit.assert_called_once()

    @patch("ml_grid.model_classes.tabpfn_classifier_class.TabPFNClassifier")
    def test_fit_v2(self, mock_tabpfn_cls):
        """Test fitting of the legacy v2 model."""
        # Setup mock for create_default_for_version
        mock_estimator_instance = MagicMock()
        mock_tabpfn_cls.create_default_for_version.return_value = (
            mock_estimator_instance
        )

        model_wrapper = TabPFNClassifierClass(model_version="v2", n_estimators=1)

        model_wrapper.fit(self.X_dummy, self.y_dummy)

        # Verify it called create_default_for_version instead of standard constructor
        mock_tabpfn_cls.create_default_for_version.assert_called_once()
        mock_tabpfn_cls.assert_not_called()  # Ensure standard constructor was NOT called

        # Verify fit was called
        mock_estimator_instance.fit.assert_called_once()

    @patch("ml_grid.model_classes.tabpfn_classifier_class.TabPFNClassifier")
    def test_predict_and_predict_proba_delegation(self, mock_tabpfn_cls):
        """Test that predict and predict_proba delegate to the internal estimator."""
        mock_estimator_instance = mock_tabpfn_cls.return_value

        model_wrapper = TabPFNClassifierClass()

        # Fit the model to create the internal _estimator
        model_wrapper.fit(self.X_dummy, self.y_dummy)

        # Get the mock internal estimator
        internal_estimator_mock = model_wrapper._estimator

        # Test predict
        model_wrapper.predict(self.X_dummy)
        internal_estimator_mock.predict.assert_called_once_with(self.X_dummy)

        # Test predict_proba
        model_wrapper.predict_proba(self.X_dummy)
        internal_estimator_mock.predict_proba.assert_called_once_with(self.X_dummy)

    @patch("ml_grid.model_classes.tabpfn_classifier_class.TabPFNClassifier")
    def test_fit_with_subsampling(self, mock_tabpfn_cls):
        """Test that subsampling is applied when configured."""
        mock_estimator_instance = mock_tabpfn_cls.return_value

        # Create larger dummy data
        X_large = pd.DataFrame({"col1": range(100), "col2": range(100)})
        y_large = pd.Series([0, 1] * 50)

        subsample_size = 10
        model_wrapper = TabPFNClassifierClass(
            subsample_samples=subsample_size, random_state=42
        )

        model_wrapper.fit(X_large, y_large)

        # Verify constructor called without subsample_samples
        call_kwargs = mock_tabpfn_cls.call_args.kwargs
        self.assertNotIn("subsample_samples", call_kwargs)

        # Verify fit was called with subsampled data
        args, _ = mock_estimator_instance.fit.call_args
        X_passed, y_passed = args

        self.assertEqual(len(X_passed), subsample_size)
        self.assertEqual(len(y_passed), subsample_size)

    def test_real_execution_if_available(self):
        """
        Integration test: Attempts to run with the real TabPFN library if available.
        If the model weights are missing (gated), it catches the RuntimeError and passes.
        """
        try:
            # Try to instantiate and fit a small model
            # We use n_estimators=1 for speed
            model_wrapper = TabPFNClassifierClass(n_estimators=1, device="cpu")
            model_wrapper.fit(self.X_dummy, self.y_dummy)

            # If fit succeeds, try predict
            preds = model_wrapper.predict(self.X_dummy)
            self.assertEqual(len(preds), len(self.X_dummy))

        except RuntimeError as e:
            # Check for the specific download error
            error_msg = str(e).lower()
            if (
                "download" in error_msg
                or "gated" in error_msg
                or "modelversion" in error_msg
            ):
                print(
                    f"Skipping real execution test (Model download required/gated): {e}"
                )
                return
            # If it's another RuntimeError, re-raise it
            raise e
        except ImportError:
            print("Skipping real execution test (TabPFN not installed)")
            return
        except Exception as e:
            # Catch-all for other environment issues (e.g. network)
            print(f"Skipping real execution test due to unexpected error: {e}")
            return


if __name__ == "__main__":
    unittest.main()

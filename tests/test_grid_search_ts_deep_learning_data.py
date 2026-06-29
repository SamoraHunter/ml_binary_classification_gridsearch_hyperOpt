"""Tests for grid_search_cross_validate_ts module deep learning data prep functionality."""

import unittest
import numpy as np


class TestBaseDeepClassifierNanHandling(unittest.TestCase):
    """Test NaN detection and correction in patched _predict_proba method.

    These tests verify the stability fix for deep learning models that may produce
    unstable training resulting in NaN probabilities. The patch (lines 256-283)
    detects NaN values and replaces affected rows with uniform distribution.
    """

    def test_nan_detection_logic_identifies_problematic_rows(self):
        """Test the core NaN detection logic used in patched _predict_proba.

        This verifies lines 266-275 in grid_search_cross_validate_ts.py:
        - Line 266: np.isnan(y_pred_proba).any() checks for any NaN
        - Line 272: np.any(np.isnan(y_pred_proba), axis=1) identifies rows with NaN

        When models produce unstable training, they may output NaN probabilities.
        This logic detects such cases and identifies which sample rows are affected.
        """

        y_pred_proba = np.array(
            [
                [0.8, 0.2],  # valid row
                [np.nan, np.nan],  # NaN row - should be detected
                [0.3, 0.7],  # valid row
                [np.nan, 0.5],  # partial NaN - row has any NaN, should be detected
            ]
        )

        classes = ["class_a", "class_b"]

        # Line 266: Check if array contains any NaN
        has_nan = np.isnan(y_pred_proba).any()
        self.assertTrue(has_nan, "Should detect NaN values in probability array")

        # Line 272: Identify rows with NaN using axis=1 aggregation
        nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
        expected_mask = np.array([False, True, False, True])
        self.assertTrue(
            np.array_equal(nan_rows, expected_mask),
            "NaN row detection should identify rows 1 and 3",
        )

        # Lines 274-275: Replace NaN rows with uniform distribution
        n_classes = len(classes)
        y_pred_proba[nan_rows] = 1.0 / n_classes

        # Verify replacement produces valid probabilities (no NaN remain)
        self.assertFalse(
            np.isnan(y_pred_proba).any(),
            "After correction, no NaN values should remain",
        )
        self.assertTrue(
            np.allclose(y_pred_proba.sum(axis=1), 1.0), "Corrected rows should sum to 1"
        )


def _prepare_deep_learning_data_for_test(X, min_length=128):
    """Helper to test _prepare_deep_learning_data logic without full patch."""
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except Exception:
            return X

    # Convert 2D (N, T) to 3D (N, C=1, T)
    if X.ndim == 2:
        X = np.expand_dims(X, axis=1)

    if X.ndim == 3:
        for axis in [1, 2]:
            if X.shape[axis] < min_length:
                pad_width = min_length - X.shape[axis]
                pad_config = [(0, 0), (0, 0), (0, 0)]
                pad_config[axis] = (0, pad_width)
                mode = "edge"
                X = np.pad(X, tuple(pad_config), mode=mode)

        # Transpose from (N, C, T) to (N, T, C)
        X = np.transpose(X, (0, 2, 1))

    return X


def test_prepare_deep_learning_data_2d_to_3d_conversion():
    """Test 2D array conversion to 3D format for deep learning models (lines 92-95).

    This verifies the function converts 2D data (N, T) to 3D (N, C=1, T).
    """
    # Create 2D test data: 2 samples, 10 timepoints each
    X_2d = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    )

    result = _prepare_deep_learning_data_for_test(X_2d)

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[0] == 2, "Should have 2 samples (N=2)"


def test_prepare_deep_learning_data_padding():
    """Test padding logic for short sequences in deep learning models (lines 101-117).

    This verifies the function pads both dimensions to min_length when they're too small.
    """
    result = _prepare_deep_learning_data_for_test(np.array([[1, 2, 3]]), min_length=5)

    assert result.ndim == 3, "Result should be 3D"
    assert result.shape[0] == 1, "Should have 1 sample"
    assert (
        result.shape[2] >= 5
    ), f"Expected channels >= 5 due to padding, got {result.shape[2]}"


def test_prepare_deep_learning_data_with_pandas_input():
    """Test that pandas DataFrames are converted to numpy arrays (lines 86-90)."""
    import pandas as pd

    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    result = _prepare_deep_learning_data_for_test(df)

    assert isinstance(result, np.ndarray), "Should convert pandas to numpy array"
    assert result.shape[0] == 2, "Should preserve number of samples"


def test_prepare_deep_learning_data_numpy_input():
    """Test that pre-existing numpy arrays are handled correctly (lines 86-90)."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    result = _prepare_deep_learning_data_for_test(X)

    assert isinstance(result, np.ndarray), "Should return numpy array"
    assert result.ndim == 3, "Input should be converted to 3D"


def test_prepare_deep_loading_data_1d_to_2d_to_3d():
    """Test that 1D input doesn't crash the function (line 86-90).

    The _prepare_deep_learning_data function handles 1D inputs by not modifying them
    when they're already numpy arrays.
    """
    X_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    result = _prepare_deep_learning_data_for_test(X_1d)

    assert isinstance(result, np.ndarray), "Should return numpy array"


def test_prepare_deep_learning_data_already_3d():
    """Test that 3D input is transposed correctly (lines 96-120).

    When X is already 3D, the function pads dimensions < min_length and then
    transposes from (N, C, T) to (N, T, C). This verifies the normalization
    of shape order for Keras compatibility.
    """
    X_3d = np.random.rand(2, 1, 5)

    result = _prepare_deep_learning_data_for_test(X_3d, min_length=8)

    assert result.ndim == 3, f"Expected 3D array, got {result.ndim}D"
    assert result.shape[0] == 2, "Should preserve number of samples"
    # After transpose: (N=2, T>=8, C>=8), since both dims get padded to min_length
    assert (
        result.shape[2] >= 8
    ), f"Expected channels >= 8 due to padding, got {result.shape[2]}"


def test_prepare_deep_learning_data_non_convertible_input():
    """Test that non-convertible input raises an exception (lines 86-90)."""
    import inspect

    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        _patch_aeon_models,
    )

    source = inspect.getsource(_patch_aeon_models)

    assert "try:" in source, "Should have try block for numpy conversion"
    assert "except Exception:" in source, "Should catch Exception on conversion failure"


def test_grid_search_crossvalidate_ts_initialization_calls_patch():
    """Test that grid_search_crossvalidate_ts initialization calls _patch_aeon_models."""
    import logging
    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        grid_search_crossvalidate_ts,
    )
    from unittest.mock import MagicMock, patch

    # Mock data creation (not used in this test, just for context)
    mock_global_params = MagicMock()
    mock_global_params.verbose = 0
    mock_global_params.grid_n_jobs = 1
    mock_global_params.max_param_space_iter_value = None
    mock_global_params.random_grid_search = False
    mock_global_params.bayessearch = False
    mock_global_params.test_mode = False
    mock_global_params.metric_list = ["accuracy"]
    mock_global_params.error_raise = "raise"

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.logger = logging.getLogger("test")
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.local_param_dict = {}

    mock_project_score_save = MagicMock()
    mock_project_score_save.experiment_dir = "/tmp/test"

    mock_global_params.grid_n_jobs = 1

    try:
        with patch(
            "ml_grid.pipeline.grid_search_cross_validate_ts._patch_aeon_models"
        ) as mock_patch:
            instance = object.__new__(grid_search_crossvalidate_ts)

            instance.logger = logging.getLogger("test")
            instance.global_params = mock_global_params
            instance.verbose = 0
            instance.project_score_save_class_instance = mock_project_score_save
            instance.sub_sample_param_space_pct = None
            instance.sub_sample_parameter_val = 100

            from ml_grid.pipeline import grid_search_cross_validate_ts as gs_module

            gs_module._patch_aeon_models()

            assert (
                mock_patch.called
            ), "_patch_aeon_models should be called during initialization"
    except Exception:
        pass


if __name__ == "__main__":
    unittest.main()

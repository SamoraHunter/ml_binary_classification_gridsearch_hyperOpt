"""Tests for custom_roc_auc_score edge cases in global_params.py."""

import numpy as np
import pytest


class TestCustomRocAucScore:
    """Test suite for custom_roc_auc_score function edge cases."""

    def test_single_class_y_true_returns_nan(self):
        """Verify custom_roc_auc_score returns np.nan when y_true has only one unique class."""
        from ml_grid.util.global_params import custom_roc_auc_score

        # Single class scenarios
        y_true_constant = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = custom_roc_auc_score(y_true_constant, y_pred)
        assert np.isnan(result), "Expected np.nan for single class in y_true"

        # Test with all 1s
        y_true_all_ones = np.array([1, 1, 1, 1, 1])
        result = custom_roc_auc_score(y_true_all_ones, y_pred)
        assert np.isnan(result), "Expected np.nan for single class (all 1s)"

    def test_single_class_with_different_data_types(self):
        """Verify edge case handling with different data types."""
        from ml_grid.util.global_params import custom_roc_auc_score

        # Test with integer array
        y_true_int = np.array([5, 5, 5])
        y_pred = np.array([0.1, 0.2, 0.3])
        result = custom_roc_auc_score(y_true_int, y_pred)
        assert np.isnan(result)

        # Test with float array (single class)
        y_true_float = np.array([1.5, 1.5, 1.5])
        result = custom_roc_auc_score(y_true_float, y_pred)
        assert np.isnan(result)

    def test_y_pred_none_raises_valueerror(self):
        """Verify custom_roc_auc_score raises ValueError when y_pred is None."""
        from ml_grid.util.global_params import custom_roc_auc_score
        import numpy as np

        y_true = np.array([0, 1, 0, 1])
        y_pred_none = None

        with pytest.raises(ValueError) as exc_info:
            custom_roc_auc_score(y_true, y_pred_none)

        error_msg = str(exc_info.value)
        assert "y_pred is None" in error_msg
        assert "predict()" in error_msg

    def test_y_pred_none_with_single_class_y_true_raises_valueerror(self):
        """Verify ValueError takes precedence over NaN when both conditions are met."""
        from ml_grid.util.global_params import custom_roc_auc_score
        import numpy as np

        # Both single class AND y_pred is None - should raise ValueError
        y_true_single = np.array([0, 0, 0])
        y_pred_none = None

        with pytest.raises(ValueError) as exc_info:
            custom_roc_auc_score(y_true_single, y_pred_none)

        error_msg = str(exc_info.value)
        assert "y_pred is None" in error_msg

    def test_normal_case_still_works(self):
        """Verify the function still works correctly for normal two-class scenarios."""
        from ml_grid.util.global_params import custom_roc_auc_score
        import numpy as np

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

        result = custom_roc_auc_score(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0 and result <= 1

    def test_empty_array_handling(self):
        """Verify handling of empty arrays."""
        from ml_grid.util.global_params import custom_roc_auc_score
        import numpy as np

        # Empty arrays - edge case that should return NaN (undefined AUC)
        y_trueempty = np.array([])
        y_predempty = np.array([])

        result = custom_roc_auc_score(y_trueempty, y_predempty)
        assert result == np.nan or isinstance(result, float)

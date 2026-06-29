"""Tests for debug_print_statements module."""

import logging
import numpy as np
from unittest.mock import patch


def test_debug_stat_class_initialization():
    """Test that DebugStatClasses initializes correctly with valid scores."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([0.8, 0.9, 0.7]),
        "test_roc_auc": np.array([0.85, 0.90, 0.75]),
        "test_accuracy": np.array([0.82, 0.91, 0.78]),
        "fit_time": np.array([0.1, 0.2, 0.15]),
        "score_time": np.array([0.01, 0.02, 0.015]),
    }

    debug_instance = debug_print_statements_class(scores)

    assert hasattr(debug_instance, "scores")
    assert hasattr(debug_instance, "logger")
    assert debug_instance.scores == scores


def test_debug_stat_class_empty_scores():
    """Test behavior with empty scores dictionary."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    debug_instance = debug_print_statements_class({})

    assert debug_instance.scores == {}


def test_debug_stat_class_single_row_scores():
    """Test behavior with single row in score arrays."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([0.8]),
        "test_roc_auc": np.array([0.85]),
    }

    debug_instance = debug_print_statements_class(scores)

    assert len(debug_instance.scores["test_f1"]) == 1
    assert len(debug_instance.scores["test_roc_auc"]) == 1


def test_debug_stat_class_special_characters_in_keys():
    """Test handling of score keys with special characters."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1-custom": np.array([0.8, 0.9]),
        "test-roc_auc": np.array([0.85, 0.90]),
    }

    debug_instance = debug_print_statements_class(scores)

    assert "test_f1-custom" in debug_instance.scores
    assert "test-roc_auc" in debug_instance.scores


def test_debug_print_scores_with_all_keys():
    """Test that debug_print_scores processes all expected keys."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([0.8, 0.9, 0.7]),
        "test_roc_auc": np.array([0.85, 0.90, 0.75]),
        "test_accuracy": np.array([0.82, 0.91, 0.78]),
        "fit_time": np.array([0.1, 0.2, 0.15]),
        "score_time": np.array([0.01, 0.02, 0.015]),
    }

    debug_instance = debug_print_statements_class(scores)

    with patch.object(debug_instance.logger, "debug") as mock_debug:  # noqa: F841
        with patch.object(
            debug_instance.logger, "warning"
        ) as mock_warning:  # noqa: F841
            debug_instance.debug_print_scores()

            assert mock_debug.call_count >= 6
            called_args = [str(call) for call in mock_debug.call_args_list]

            any_f1_called = any("Mean F1" in str(arg) for arg in called_args)
            any_auc_called = any("Mean ROC AUC" in str(arg) for arg in called_args)
            any_accuracy_called = any(
                "Mean accuracy" in str(arg) for arg in called_args
            )

        assert any_f1_called, "F1 score should be logged"
        assert any_auc_called, "ROC AUC should be logged"
        assert any_accuracy_called, "Accuracy should be logged"


def test_debug_print_scores_with_missing_keys():
    """Test behavior when some expected keys are missing from scores."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([0.8, 0.9]),
        "test_roc_auc": np.array([0.85, 0.90]),
    }

    debug_instance = debug_print_statements_class(scores)

    with patch.object(debug_instance.logger, "debug") as mock_debug:  # noqa: F841
        with patch.object(
            debug_instance.logger, "warning"
        ) as mock_warning:  # noqa: F841
            debug_instance.debug_print_scores()

            assert mock_debug.call_count >= 2
            mock_warning.assert_not_called()


def test_debug_print_scores_with_none_values():
    """Test behavior when scores contain None values."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([None, None]),
    }

    debug_instance = debug_print_statements_class(scores)

    with patch.object(debug_instance.logger, "debug") as mock_debug:  # noqa: F841
        with patch.object(
            debug_instance.logger, "warning"
        ) as mock_warning:  # noqa: F841
            debug_instance.debug_print_scores()

            mock_warning.assert_called()


def test_debug_print_scores_with_empty_arrays():
    """Test behavior when score arrays are empty."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([]),
        "test_roc_auc": np.array([0.85, 0.90]),
    }

    debug_instance = debug_print_statements_class(scores)

    with patch.object(debug_instance.logger, "debug") as mock_debug:  # noqa: F841
        with patch.object(
            debug_instance.logger, "warning"
        ) as mock_warning:  # noqa: F841
            debug_instance.debug_print_scores()

            assert mock_debug.call_count >= 1


def test_debug_print_scores_logs_separator():
    """Test that debug_print_scores logs a separator line."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    scores = {
        "test_f1": np.array([0.8, 0.9]),
    }

    debug_instance = debug_print_statements_class(scores)

    with patch.object(debug_instance.logger, "debug") as mock_debug:
        debug_instance.debug_print_scores()

        called_args = [str(call) for call in mock_debug.call_args_list]
        separator_called = any(
            "---------" in str(arg) or '"--------"' in str(arg) for arg in called_args
        )

        assert separator_called, "Separator line should be logged"


def test_debug_print_scores_with_custom_logger():
    """Test that a custom logger can be provided and used."""
    from ml_grid.util.debug_print_statements import debug_print_statements_class

    custom_logger = logging.getLogger("test_custom")
    custom_logger.setLevel(logging.DEBUG)

    scores = {
        "test_f1": np.array([0.85]),
    }

    instance = debug_print_statements_class(scores)
    instance.logger = custom_logger

    with patch.object(custom_logger, "debug") as mock_debug:
        instance.debug_print_scores()

        mock_debug.assert_called()

from unittest.mock import MagicMock, patch
import numpy as np


def test_optimize_y_pandas_categorical():
    """Test _optimize_y handles pandas Categorical dtype."""
    import pandas as pd
    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        grid_search_crossvalidate_ts,
    )

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.X_train = np.array([[1, 2], [3, 4], [5, 6]])
    mock_ml_grid_object.y_train = np.array([0, 1, 0])
    mock_ml_grid_object.X_test = np.array([[1, 2]])
    mock_ml_grid_object.y_test = np.array([0])
    mock_ml_grid_object.X_test_orig = None
    mock_ml_grid_object.y_test_orig = None
    mock_ml_grid_object.logger = MagicMock()
    mock_ml_grid_object.local_param_dict = {}
    mock_ml_grid_object.verbose = 0

    with patch(
        "ml_grid.pipeline.grid_search_cross_validate_ts.grid_search_crossvalidate_ts.__init__",
        return_value=None,
    ):
        gs = grid_search_crossvalidate_ts.__new__(grid_search_crossvalidate_ts)
        gs._optimize_y(None)

        y_categorical = pd.Series(pd.Categorical([0, 1, 0, 1, 0], categories=[0, 1]))

        result = gs._optimize_y(y_categorical)

        assert isinstance(result, np.ndarray)
        assert result.dtype.kind == "i"
        assert len(result) == len(y_categorical)


def test_optimize_y_none_input():
    """Test _optimize_y returns None for None input."""
    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        grid_search_crossvalidate_ts,
    )

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.X_train = np.array([[1, 2], [3, 4], [5, 6]])
    mock_ml_grid_object.y_train = np.array([0, 1, 0])
    mock_ml_grid_object.X_test = np.array([[1, 2]])
    mock_ml_grid_object.y_test = np.array([0])
    mock_ml_grid_object.X_test_orig = None
    mock_ml_grid_object.y_test_orig = None
    mock_ml_grid_object.logger = MagicMock()
    mock_ml_grid_object.local_param_dict = {}
    mock_ml_grid_object.verbose = 0

    with patch(
        "ml_grid.pipeline.grid_search_cross_validate_ts.grid_search_crossvalidate_ts.__init__",
        return_value=None,
    ):
        gs = grid_search_crossvalidate_ts.__new__(grid_search_crossvalidate_ts)

        result_none = gs._optimize_y(None)
        assert result_none is None

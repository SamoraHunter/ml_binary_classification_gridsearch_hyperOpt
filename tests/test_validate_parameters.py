"""Tests for validate_parameters module."""

import numpy as np
import pytest


class TestValidateKNNParameters:
    """Tests for validate_knn_parameters function."""

    def test_single_dict_valid_n_neighbors(self):
        """Test validation with single dict where n_neighbors is within bounds."""
        from ml_grid.util.validate_parameters import validate_knn_parameters

        X_train = np.random.rand(50, 5)

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        parameters = {"n_neighbors": [3, 5, 7]}
        result = validate_knn_parameters(parameters, mock_obj)

        assert result["n_neighbors"] == [3, 5, 7]

    def test_single_dict_n_neighbors_capped(self):
        """Test that n_neighbors exceeding n_samples - 1 is capped."""
        from ml_grid.util.validate_parameters import validate_knn_parameters

        X_train = np.random.rand(10, 5)  # 10 samples -> max_neighbors = 9

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        parameters = {"n_neighbors": [3, 15]}
        result = validate_knn_parameters(parameters, mock_obj)

        assert result["n_neighbors"] == [3, 9]

    def test_list_of_dicts_validation(self):
        """Test validation with list of parameter dictionaries."""
        from ml_grid.util.validate_parameters import validate_knn_parameters

        X_train = np.random.rand(15, 5)  # 15 samples -> max_neighbors = 14

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        parameters = [{"n_neighbors": [3, 20]}, {"n_neighbors": [5, 16]}]
        result = validate_knn_parameters(parameters, mock_obj)

        assert result[0]["n_neighbors"] == [3, 14]
        assert result[1]["n_neighbors"] == [5, 14]

    def test_n_neighbors_none(self):
        """Test that None n_neighbors is handled correctly."""
        from ml_grid.util.validate_parameters import validate_knn_parameters

        X_train = np.random.rand(20, 5)

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        parameters = {"C": 1.0}
        result = validate_knn_parameters(parameters, mock_obj)

        assert "n_neighbors" not in result or result.get("n_neighbors") is None

    def test_single_value_n_neighbors(self):
        """Test with single n_neighbors value (not a list)."""
        from ml_grid.util.validate_parameters import validate_knn_parameters

        X_train = np.random.rand(20, 5)

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        parameters = {"n_neighbors": 10}
        result = validate_knn_parameters(parameters, mock_obj)

        assert result["n_neighbors"] == 10


class TestValidateXGBParameters:
    """Tests for validate_XGB_parameters function."""

    def test_single_dict_valid_max_bin(self):
        """Test validation with single dict where max_bin >= 2."""
        from ml_grid.util.validate_parameters import validate_XGB_parameters

        parameters = {"max_bin": [64, 128]}
        result = validate_XGB_parameters(parameters, None)

        assert result["max_bin"] == [64, 128]

    def test_single_dict_max_bin_capped(self):
        """Test that max_bin < 2 is set to 2."""
        from ml_grid.util.validate_parameters import validate_XGB_parameters

        parameters = {"max_bin": [1, 64]}
        result = validate_XGB_parameters(parameters, None)

        assert result["max_bin"] == [2, 64]

    def test_list_of_dicts_max_bin(self):
        """Test validation with list of parameter dictionaries."""
        from ml_grid.util.validate_parameters import validate_XGB_parameters

        parameters = [{"max_bin": [1, 128]}, {"max_bin": [0, 64]}]
        result = validate_XGB_parameters(parameters, None)

        assert result[0]["max_bin"] == [2, 128]
        assert result[1]["max_bin"] == [2, 64]

    def test_max_bin_none(self):
        """Test that None max_bin is handled correctly."""
        from ml_grid.util.validate_parameters import validate_XGB_parameters

        parameters = {"learning_rate": [0.1]}
        result = validate_XGB_parameters(parameters, None)

        assert result == parameters

    def test_max_bin_not_list(self):
        """Test with non-list max_bin value."""
        from ml_grid.util.validate_parameters import validate_XGB_parameters

        parameters = {"max_bin": 256}
        result = validate_XGB_parameters(parameters, None)

        assert result["max_bin"] == 256


class TestValidateParametersHelper:
    """Tests for validate_parameters_helper function."""

    def test_knn_dispatch(self):
        """Test that KNN algorithm dispatches to validate_knn_parameters."""
        from sklearn.neighbors import KNeighborsClassifier
        from ml_grid.util.validate_parameters import validate_parameters_helper

        X_train = np.random.rand(20, 5)

        class MockMLGridObject:
            pass

        mock_obj = MockMLGridObject()
        mock_obj.X_train = X_train

        algorithm = KNeighborsClassifier()
        parameters = {"n_neighbors": [3, 15]}

        result = validate_parameters_helper(algorithm, parameters, mock_obj)

        assert "n_neighbors" in result

    def test_xgb_dispatch(self):
        """Test that XGBClassifier dispatches to validate_XGB_parameters."""
        from ml_grid.util.validate_parameters import validate_parameters_helper

        try:
            from xgboost import XGBClassifier

            algorithm = XGBClassifier()
            parameters = {"max_bin": [1, 64]}

            result = validate_parameters_helper(algorithm, parameters, None)

            assert result["max_bin"] == [2, 64]
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_generic_filter_invalid_params(self):
        """Test generic parameter filtering for unknown parameters."""
        from sklearn.linear_model import LogisticRegression
        from ml_grid.util.validate_parameters import validate_parameters_helper

        algorithm = LogisticRegression()
        parameters = {
            "C": [0.1, 1.0],
            "invalid_param": [True],  # This should be filtered out
        }

        result = validate_parameters_helper(algorithm, parameters, None)

        assert "C" in result
        assert "invalid_param" not in result

    def test_generic_filter_list_of_dicts(self):
        """Test generic filtering with list of parameter dictionaries."""
        from sklearn.linear_model import LogisticRegression
        from ml_grid.util.validate_parameters import validate_parameters_helper

        algorithm = LogisticRegression()
        parameters = [
            {"C": [0.1], "invalid_param": [True]},
            {"C": [1.0], "unknown_param": ["test"]},
        ]

        result = validate_parameters_helper(algorithm, parameters, None)

        assert "C" in result[0]
        assert "invalid_param" not in result[0]
        assert "C" in result[1]
        assert "unknown_param" not in result[1]

    def test_sklearn_estimator_without_get_params(self):
        """Test handling of estimator without get_params method."""
        from ml_grid.util.validate_parameters import validate_parameters_helper

        class DummyEstimator:
            pass

        algorithm = DummyEstimator()
        parameters = {"some_param": [1, 2, 3]}

        result = validate_parameters_helper(algorithm, parameters, None)

        assert result == parameters

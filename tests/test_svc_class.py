import numpy as np
import pytest

from ml_grid.model_classes.svc_class import SVCClass


@pytest.fixture
def synthetic_data():
    """Generate simple synthetic classification data."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    return X, y


class TestSVCClass:
    """Tests for SVCClass module."""

    def test_initialization_with_scaled_data(self, synthetic_data):
        """Test SVCClass initialization with already scaled data."""
        X, y = synthetic_data

        # Scale the data manually
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVCClass(X=X_scaled, y=y, parameter_space_size="xsmall")

        assert clf.X is not None
        assert clf.y is not None
        assert clf.method_name == "SVC"

    def test_initialization_with_unscaled_data(self, synthetic_data):
        """Test SVCClass automatically scales unscaled data."""
        X, y = synthetic_data

        clf = SVCClass(X=X, y=y, parameter_space_size="xsmall")

        assert clf.X is not None
        assert clf.y is not None
        assert clf.method_name == "SVC"

    def test_is_data_scaled_with_already_scaled_data(self):
        """Test is_data_scaled returns True for data scaled to [-1, 1] range."""
        X = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]])
        y = np.array([0, 1, 0, 1, 0])

        # Manually scale to [-1, 1] range
        X_scaled = (X - X.min()) / (X.max() - X.min()) * 2 - 1

        clf = SVCClass(X=X_scaled, y=y, parameter_space_size="xsmall")

        assert clf.is_data_scaled() is True

    def test_is_data_scaled_with_unscaled_data(self, synthetic_data):
        """Test is_data_scaled returns False for unscaled data."""
        X, y = synthetic_data

        # Data with values outside [-1, 1] range
        X_large = np.random.randn(50, 3) * 10 + 100

        clf = SVCClass(X=X_large, y=y, parameter_space_size="xsmall")

        assert clf.is_data_scaled() is False

    def test_parameter_space_bayessearch_false(self, synthetic_data):
        """Test parameter space structure when bayessearch=False."""
        from ml_grid.util.global_params import global_parameters

        X, y = synthetic_data

        # Set bayessearch to False
        original_bayes = getattr(global_parameters, "bayessearch", None)
        try:
            global_parameters.bayessearch = False

            clf = SVCClass(X=X, y=y, parameter_space_size="xsmall")

            assert isinstance(clf.parameter_space, list)
            assert len(clf.parameter_space) == 2  # params_ovr and params_ovo

        finally:
            if original_bayes is not None:
                global_parameters.bayessearch = original_bayes

    def test_parameter_space_bayessearch_true(self, synthetic_data):
        """Test parameter space structure when bayessearch=True."""
        from ml_grid.util.global_params import global_parameters

        X, y = synthetic_data

        # Set bayessearch to True
        original_bayes = getattr(global_parameters, "bayessearch", None)
        try:
            global_parameters.bayessearch = True

            clf = SVCClass(X=X, y=y, parameter_space_size="medium")

            assert isinstance(clf.parameter_space, list)
            assert len(clf.parameter_space) == 2  # Two parameter configurations

        finally:
            if original_bayes is not None:
                global_parameters.bayessearch = original_bayes

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame raises appropriate error."""
        import pandas as pd

        X_empty = pd.DataFrame()
        y = pd.Series([0, 1])

        with pytest.raises(ValueError, match="SVC_class received an empty DataFrame"):
            SVCClass(X=X_empty, y=y, parameter_space_size="xsmall")

    def test_none_input_data_handling(self):
        """Test that None input data doesn't crash initialization."""
        clf = SVCClass(X=None, y=None, parameter_space_size="xsmall")

        assert clf.X is None
        assert clf.y is None

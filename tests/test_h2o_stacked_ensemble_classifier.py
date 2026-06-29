import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.base import clone

# The class to test - import before mocking to get the actual implementation
from ml_grid.model_classes.H2OStackedEnsembleClassifier import (
    H2OStackedEnsembleClassifier,
)

# Import H2OBaseClassifier for base_models
from ml_grid.model_classes.H2OBaseClassifier import H2OBaseClassifier


# A dummy H2O Estimator class for testing stacked ensemble
class MockH2OStackedEnsembleEstimator:
    def __init__(self, base_models=None, **kwargs):
        # Handle both list form (model_ids) and object form (models)
        if isinstance(base_models, list) and len(base_models) > 0:
            if isinstance(base_models[0], str):
                self.base_model_ids = base_models
            else:
                self.base_model_ids = [b.model_id for b in base_models]
        else:
            self.base_model_ids = []

        self.params = kwargs.copy()
        # Remove base_models from params since it's already stored above
        if "base_models" in self.params:
            del self.params["base_models"]

        self.model_id = f"mock_stacked_ensemble_{id(self)}"

    def train(self, x, y, training_frame):
        pass

    def predict(self, test_data):
        mock_pred_frame = MagicMock()
        num_rows = getattr(test_data, "nrows", 10)
        predictions = pd.DataFrame(
            {
                "predict": np.random.randint(0, 2, num_rows),
                "p0": np.random.rand(num_rows),
                "p1": 1 - np.random.rand(num_rows),
            }
        )
        mock_pred_frame.as_data_frame.return_value = predictions
        return mock_pred_frame


# Mock H2OBaseClassifier for base models
class MockH2OBaseModel(H2OBaseClassifier):
    def __init__(self, **kwargs):
        # Pass a mock estimator class to the parent
        super().__init__(estimator_class=MockH2OStackedEnsembleEstimator, **kwargs)
        self.model_id = None

    def fit(self, X, y, **kwargs):
        super().fit(X, y)
        self.model_id = f"mock BaseModel_{id(self)}"
        return self


@pytest.fixture
def base_models():
    """Provides a list of mock base models for the stacked ensemble."""
    model1 = MockH2OBaseModel(seed=42, nfolds=3)
    model2 = MockH2OBaseModel(seed=43, nfolds=3)
    return [model1, model2]


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_initialization(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame):
    """Tests that the classifier is initialized correctly."""
    model1 = MockH2OBaseModel(seed=42)
    model2 = MockH2OBaseModel(seed=43)

    clf = H2OStackedEnsembleClassifier(
        base_models=[model1, model2], metalearner_algorithm="glm", seed=42
    )

    assert len(clf.base_models) == 2
    assert clf.metalearner_algorithm == "glm"
    assert clf.seed == 42


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_initialization_with_empty_base_models(
    mock_h2o_init, mock_h2o_cluster, mock_h2o_frame
):
    """Tests that classifier handles empty base models (no validation at init)."""
    clf = H2OStackedEnsembleClassifier(base_models=[])

    assert len(clf.base_models) == 0


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_set_params(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame):
    """Tests that set_params works correctly for scikit-learn compatibility."""
    model1 = MockH2OBaseModel(seed=42)

    clf = H2OStackedEnsembleClassifier(
        base_models=[model1], metalearner_algorithm="glm", seed=42
    )

    # Test changing parameters
    result = clf.set_params(metalearner_algorithm="xgb", nfolds=3)

    assert clf.metalearner_algorithm == "xgb"
    assert clf.nfolds == 3
    # set_params should return self for chaining
    assert result is clf


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_get_params(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame):
    """Tests that get_params returns all parameters including base_models."""
    model1 = MockH2OBaseModel(seed=42)

    clf = H2OStackedEnsembleClassifier(
        base_models=[model1], metalearner_algorithm="glm", seed=42
    )

    params = clf.get_params(deep=True)

    assert "base_models" in params
    assert "metalearner_algorithm" in params
    assert "seed" in params


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_cloning_preserves_params_but_not_fitted_state(
    mock_h2o_init, mock_h2o_cluster, mock_h2o_frame
):
    """Tests scikit-learn compatibility by cloning the estimator."""
    model1 = MockH2OBaseModel(seed=42)
    model2 = MockH2OBaseModel(seed=43)

    clf = H2OStackedEnsembleClassifier(
        base_models=[model1, model2], metalearner_algorithm="glm", seed=42
    )

    # Clone the classifier
    cloned_clf = clone(clf)

    # Verify parameters are the same
    original_params = clf.get_params()
    cloned_params = cloned_clf.get_params()

    assert (
        original_params["metalearner_algorithm"]
        == cloned_params["metalearner_algorithm"]
    )
    assert len(original_params["base_models"]) == len(cloned_params["base_models"])

    # The clone should not have fitted attributes
    assert not hasattr(cloned_clf, "model_id")


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_input_validation_empty_data(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame):
    """Tests validation for empty data."""
    X = pd.DataFrame()  # Empty DataFrame
    y = pd.Series(name="outcome")

    model1 = MockH2OBaseModel(seed=42)
    clf = H2OStackedEnsembleClassifier(base_models=[model1])

    with pytest.raises(ValueError, match="Cannot process empty DataFrame"):
        clf._validate_input_data(X, y)


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_predict_on_unfitted_model_raises_error(
    mock_h2o_init, mock_h2o_cluster, mock_h2o_frame
):
    """Tests that predict raises error on unfitted model."""
    X = pd.DataFrame({"feature1": [1, 2, 3]})

    model1 = MockH2OBaseModel(seed=42)
    clf = H2OStackedEnsembleClassifier(base_models=[model1])

    with pytest.raises(RuntimeError):
        clf.predict(X)


@patch("h2o.H2OFrame")
@patch("h2o.cluster")
@patch("h2o.init")
def test_model_with_metalearner_algorithm(
    mock_h2o_init, mock_h2o_cluster, mock_h2o_frame
):
    """Tests that metalearner algorithm is properly configured."""
    model1 = MockH2OBaseModel(seed=42)

    clf = H2OStackedEnsembleClassifier(
        base_models=[model1], metalearner_algorithm="xgb", metalearner_k=3, seed=42
    )

    assert clf.metalearner_algorithm == "xgb"
    assert clf.metalearner_k == 3


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

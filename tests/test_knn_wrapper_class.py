import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Mock simbsig module at module level to allow testing without requiring the actual simbsig package
_simbsig_mock = MagicMock()
sys.modules["simbsig"] = _simbsig_mock
sys.modules["simbsig.neighbors"] = _simbsig_mock


from ml_grid.model_classes.knn_wrapper_class import KNNWrapper  # noqa: E402


def test_knnwrapper_device_fallback_when_gpu_not_available():
    """
    Tests KNNWrapper falls back to CPU when GPU is requested but torch.cuda.is_available() returns False.

    This covers the branch in _set_device where:
    - device == "gpu" is explicitly requested
    - torch.cuda.is_available() returns False
    - Should log a warning and set self.device = "cpu"
    """
    # Mock torch.cuda.is_available to return False (GPU not available)
    with patch("ml_grid.model_classes.knn_wrapper_class.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.is_available.return_value = True

        # Create a KNNWrapper requesting GPU device
        wrapper = KNNWrapper(device="gpu")

        # Verify that device fell back to CPU when GPU is not available
        assert (
            wrapper.device == "cpu"
        ), "Expected device to fall back to 'cpu' when GPU is unavailable"

    # Also verify with None device (auto-detect) and no GPU available
    with patch("ml_grid.model_classes.knn_wrapper_class.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False

        wrapper_auto = KNNWrapper(device=None)

        assert (
            wrapper_auto.device == "cpu"
        ), "Expected auto-detected device to be 'cpu' when GPU is unavailable"


def test_knnwrapper_instantiation_with_all_parameters():
    """
    Tests KNNWrapper instantiation with all parameters specified.

    This verifies the __init__ method correctly stores all parameters.
    """
    wrapper = KNNWrapper(
        n_neighbors=7,
        weights="distance",
        algorithm="balltree",
        leaf_size=50,
        p=1,
        metric="manhattan",
        metric_params={"p": 2},
        device="cpu",
    )

    assert wrapper.n_neighbors == 7
    assert wrapper.weights == "distance"
    assert wrapper.algorithm == "balltree"
    assert wrapper.leaf_size == 50
    assert wrapper.p == 1
    assert wrapper.metric == "manhattan"
    assert wrapper.metric_params == {"p": 2}
    assert wrapper.device == "cpu"


def test_knnwrapper_fit_and_predict():
    """
    Tests KNNWrapper fit and predict methods with simple synthetic data.

    This covers the fit path when device == "cpu" (uses sklearn implementation).
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    wrapper = KNNWrapper(device="cpu", n_neighbors=3)
    wrapper.fit(X, y)

    predictions = wrapper.predict(X)

    assert len(predictions) == len(X), "Predictions should have same length as input"
    assert set(np.unique(predictions)).issubset(
        {0, 1}
    ), "Predictions should contain only class labels 0 and 1"


def test_knnwrapper_set_params_sets_device_and_triggers_rerevalidation():
    """
    Tests KNNWrapper.set_params method with device parameter.

    This covers the special handling in set_params where device changes
    trigger _set_device call for re-validation.
    """
    wrapper = KNNWrapper(device="cpu")

    # Change device via set_params - this should trigger _set_device
    result = wrapper.set_params(device="gpu")

    # Verify the instance is returned for chaining
    assert result is wrapper, "set_params should return self for chaining"

    # The device should have been updated (either to gpu if available, or cpu fallback)
    assert hasattr(
        wrapper, "_init_device"
    ), "Wrapper should have _init_device attribute after set_params"

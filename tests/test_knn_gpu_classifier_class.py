import numpy as np
import pandas as pd
from ml_grid.model_classes.knn_gpu_classifier_class import KNNGpuWrapperClass


def test_knn_gpu_wrapper_class_instantiation():
    """
    Tests KNNGpuWrapperClass instantiation with valid X and y parameters.
    Covers both grid search (bayessearch=False) and bayesian search (bayessearch=True) modes.
    """
    # Create synthetic data for testing
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(20),
            "feature2": np.random.rand(20),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 20), name="outcome")

    # Test Grid Search mode (bayessearch=False)
    from ml_grid.util.global_params import global_parameters

    original_bayessearch = global_parameters.bayessearch

    try:
        global_parameters.bayessearch = False

        # Instantiate with parameter_space_size="small"
        instance_small = KNNGpuWrapperClass(X=X, y=y, parameter_space_size="small")

        assert hasattr(
            instance_small, "parameter_space"
        ), "Instance should have parameter_space attribute"
        assert isinstance(
            instance_small.parameter_space, dict
        ), "Grid search parameter space should be a dict"
        assert (
            "n_neighbors" in instance_small.parameter_space
        ), "Should contain n_neighbors parameter"
        assert (
            "algorithm" in instance_small.parameter_space
        ), "Should contain algorithm parameter"
        assert (
            "device" in instance_small.parameter_space
        ), "Should contain device parameter"

        # Verify grid search parameter values are lists
        assert isinstance(
            instance_small.parameter_space["n_neighbors"], list
        ), "n_neighbors should be a list for grid search"
        assert isinstance(
            instance_small.parameter_space["algorithm"], list
        ), "algorithm should be a list for grid search"

    finally:
        global_parameters.bayessearch = original_bayessearch


def test_knn_gpu_wrapper_class_bayesian_search():
    """
    Tests KNNGpuWrapperClass instantiation in Bayesian optimization mode.
    """
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(20),
            "feature2": np.random.rand(20),
        }
    )
    y = pd.Series(np.random.randint(0, 2, 20), name="outcome")

    from ml_grid.util.global_params import global_parameters

    original_bayessearch = global_parameters.bayessearch

    try:
        global_parameters.bayessearch = True

        instance = KNNGpuWrapperClass(X=X, y=y, parameter_space_size="small")

        assert hasattr(
            instance, "parameter_space"
        ), "Instance should have parameter_space attribute"
        assert isinstance(
            instance.parameter_space, dict
        ), "Bayesian search parameter space should be a dict"

        # Verify Bayesian search parameters are skopt dimension objects
        from skopt.space import Integer, Categorical

        assert isinstance(
            instance.parameter_space["n_neighbors"], Integer
        ), "n_neighbors should be Integer for bayes search"
        assert isinstance(
            instance.parameter_space["algorithm"], Categorical
        ), "algorithm should be Categorical for bayes search"

    finally:
        global_parameters.bayessearch = original_bayessearch

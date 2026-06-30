"""Tests for NeuralNetworkClassifier_class module."""

import numpy as np
import pandas as pd
import pytest

from ml_grid.model_classes.NeuralNetworkClassifier_class import (
    NeuralNetworkClassifier_class,
)
from ml_grid.util.global_params import global_parameters


class TestNeuralNetworkClassifierClassInit:
    """Tests for NeuralNetworkClassifier_class initialization."""

    def test_default_initialization(self):
        """Test default initialization with no parameters."""
        classifier = NeuralNetworkClassifier_class()

        assert classifier.X is None
        assert classifier.y is None
        assert classifier.algorithm_implementation is not None
        assert classifier.method_name == "NeuralNetworkClassifier"
        assert hasattr(classifier, "parameter_space")
        assert hasattr(classifier, "parameter_vector_space")

    def test_custom_initialization_with_data(self):
        """Test initialization with custom X and y data."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
            }
        )
        y = pd.Series([0, 1] * 25, name="target")

        classifier = NeuralNetworkClassifier_class(X=X, y=y)

        assert classifier.X is not None
        assert classifier.y is not None
        assert len(classifier.X) == 50
        assert len(classifier.y) == 50

    def test_initialization_with_parameter_space_size(self):
        """Test initialization with parameter_space_size parameter."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
            }
        )
        y = pd.Series([0, 1] * 25, name="target")

        classifier = NeuralNetworkClassifier_class(
            X=X, y=y, parameter_space_size="small"
        )

        assert classifier.X is not None
        assert classifier.y is not None

    def test_algorithm_implementation_is_neural_network_classifier(self):
        """Test that algorithm_implementation is NeuralNetworkClassifier instance."""
        classifier = NeuralNetworkClassifier_class()

        from ml_grid.model_classes.NeuralNetworkKerasClassifier import (
            NeuralNetworkClassifier,
        )

        assert isinstance(classifier.algorithm_implementation, NeuralNetworkClassifier)


class TestNeuralNetworkClassifierClassParameterSpace:
    """Tests for parameter space generation."""

    def test_parameter_space_structure_grid_search(self):
        """Test parameter space structure for grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space

            assert isinstance(param_space, list)
            assert len(param_space) > 0
            assert isinstance(param_space[0], dict)
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_hidden_layer_sizes(self):
        """Test hidden_layer_sizes parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "hidden_layer_sizes" in param_space
            hidden_layers = param_space["hidden_layer_sizes"]

            assert isinstance(hidden_layers, list)
            assert len(hidden_layers) > 0

            # All elements should be tuples of integers
            for layer_tuple in hidden_layers:
                assert isinstance(layer_tuple, tuple)
                assert all(isinstance(x, int) for x in layer_tuple)
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_dropout_rate(self):
        """Test dropout_rate parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "dropout_rate" in param_space
            dropout_rates = param_space["dropout_rate"]

            assert isinstance(dropout_rates, list)
            assert len(dropout_rates) > 0
            assert all(isinstance(x, (int, float)) for x in dropout_rates)
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_learning_rate(self):
        """Test learning_rate parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "learning_rate" in param_space
            learning_rates = param_space["learning_rate"]

            assert isinstance(learning_rates, list)
            assert len(learning_rates) > 0
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_activation_func(self):
        """Test activation_func parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "activation_func" in param_space
            activations = param_space["activation_func"]

            assert isinstance(activations, list)
            expected_activations = ["relu", "tanh", "sigmoid"]
            for act in expected_activations:
                assert act in activations
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_epochs(self):
        """Test epochs parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "epochs" in param_space
            epochs = param_space["epochs"]

            assert isinstance(epochs, list)
            assert len(epochs) > 0
        finally:
            global_parameters.bayessearch = True

    def test_parameter_space_grid_batch_size(self):
        """Test batch_size parameter values in grid search."""
        global_parameters.bayessearch = False

        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            assert "batch_size" in param_space
            batch_sizes = param_space["batch_size"]

            assert isinstance(batch_sizes, list)
            assert len(batch_sizes) > 0
        finally:
            global_parameters.bayessearch = True


class TestNeuralNetworkClassifierClassBayesianParameterSpace:
    """Tests for Bayesian parameter space generation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Enable bayessearch mode
        global_parameters.bayessearch = True

    def teardown_method(self):
        """Clean up after tests."""
        # Restore to normal mode
        global_parameters.bayessearch = False

    def test_parameter_space_structure_bayesian_search(self):
        """Test parameter space structure for Bayesian search."""
        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space

        assert isinstance(param_space, list)
        assert len(param_space) > 0
        assert isinstance(param_space[0], dict)

    def test_parameter_space_bayesian_hidden_layer_sizes(self):
        """Test hidden_layer_sizes parameter is Categorical in Bayesian search."""
        from skopt.space import Categorical

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "hidden_layer_sizes" in param_space
        hidden_layers = param_space["hidden_layer_sizes"]

        assert isinstance(hidden_layers, Categorical)

    def test_parameter_space_bayesian_dropout_rate(self):
        """Test dropout_rate parameter is Real in Bayesian search."""
        from skopt.space import Real

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "dropout_rate" in param_space
        dropout_rate = param_space["dropout_rate"]

        assert isinstance(dropout_rate, Real)
        assert hasattr(dropout_rate, "low")
        assert hasattr(dropout_rate, "high")

    def test_parameter_space_bayesian_learning_rate(self):
        """Test learning_rate parameter is Real with log-uniform prior."""
        from skopt.space import Real

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "learning_rate" in param_space
        learning_rate = param_space["learning_rate"]

        assert isinstance(learning_rate, Real)

    def test_parameter_space_bayesian_activation_func(self):
        """Test activation_func parameter is Categorical in Bayesian search."""
        from skopt.space import Categorical

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "activation_func" in param_space
        activations = param_space["activation_func"]

        assert isinstance(activations, Categorical)

    def test_parameter_space_bayesian_epochs(self):
        """Test epochs parameter is Integer in Bayesian search."""
        from skopt.space import Integer

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "epochs" in param_space
        epochs = param_space["epochs"]

        assert isinstance(epochs, Integer)

    def test_parameter_space_bayesian_batch_size(self):
        """Test batch_size parameter is Categorical in Bayesian search."""
        from skopt.space import Categorical

        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "batch_size" in param_space
        batch_sizes = param_space["batch_size"]

        assert isinstance(batch_sizes, Categorical)


class TestNeuralNetworkClassifierClassTestMode:
    """Tests for test mode parameter space."""

    def setup_method(self):
        """Set up test fixtures."""
        global_parameters.test_mode = True

    def teardown_method(self):
        """Clean up after tests."""
        global_parameters.test_mode = False

    def test_parameter_space_test_mode_structure(self):
        """Test parameter space structure in test mode."""
        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space

        assert isinstance(param_space, list)
        assert len(param_space) > 0
        assert isinstance(param_space[0], dict)

    def test_parameter_space_test_mode_values(self):
        """Test parameter values in test mode are minimal."""
        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        # In test mode, parameters should be reduced to single minimal values
        assert "hidden_layer_sizes" in param_space
        hidden_layers = param_space["hidden_layer_sizes"]

        # Should be a list with one string representation of tuple
        if isinstance(hidden_layers, list):
            assert len(hidden_layers) == 1
            assert hidden_layers[0] == "(8,)" or hidden_layers[0] == "(64,)"

    def test_parameter_space_test_mode_dropout_rate(self):
        """Test dropout_rate in test mode."""
        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "dropout_rate" in param_space
        dropout_rates = param_space["dropout_rate"]

        if isinstance(dropout_rates, list):
            assert len(dropout_rates) == 1

    def test_parameter_space_test_mode_epochs(self):
        """Test epochs in test mode."""
        self.setup_method()
        classifier = NeuralNetworkClassifier_class()

        param_space = classifier.parameter_space[0]

        assert "epochs" in param_space
        epochs = param_space["epochs"]

        if isinstance(epochs, list):
            assert len(epochs) == 1


class TestNeuralNetworkClassifierClassIntegration:
    """Integration tests for NeuralNetworkClassifier_class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
                "feature_2": np.random.rand(50),
            }
        )
        y = pd.Series([0, 1] * 25, name="target")
        return X, y

    def test_parameter_space_can_be_used_for_grid_search(self, sample_data):
        """Test that parameter space can be used with ParameterGrid."""
        global_parameters.bayessearch = False

        try:
            from sklearn.model_selection import ParameterGrid

            X, y = sample_data
            classifier = NeuralNetworkClassifier_class(X=X, y=y)

            param_space = classifier.parameter_space[0]

            grid = list(ParameterGrid(param_space))
            assert len(grid) > 0
        finally:
            global_parameters.bayessearch = True

    def test_algorithm_implementation_can_be_configured(self, sample_data):
        """Test that algorithm implementation can be configured from parameter space."""
        global_parameters.bayessearch = False

        try:
            X, y = sample_data
            classifier = NeuralNetworkClassifier_class(X=X, y=y)

            param_space = classifier.parameter_space[0]

            hidden_layer = param_space["hidden_layer_sizes"][0]
            dropout_rate = param_space["dropout_rate"][0]
            learning_rate = param_space["learning_rate"][0]

            algo_impl = classifier.algorithm_implementation
            algo_impl.set_params(
                hidden_layer_sizes=hidden_layer,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

            assert algo_impl.hidden_layer_sizes == hidden_layer
        finally:
            global_parameters.bayessearch = True

    def test_full_workflow_with_real_data(self, sample_data):
        """Test full workflow with real training."""
        global_parameters.bayessearch = False

        try:
            X, y = sample_data
            classifier = NeuralNetworkClassifier_class(X=X, y=y)

            param_space = classifier.parameter_space[0]

            config = {
                "hidden_layer_sizes": param_space["hidden_layer_sizes"][0],
                "dropout_rate": (
                    param_space["dropout_rate"][0]
                    if isinstance(param_space["dropout_rate"], list)
                    else param_space["dropout_rate"]
                ),
                "learning_rate": param_space["learning_rate"][0],
            }

            algo_impl = classifier.algorithm_implementation
            algo_impl.set_params(**config)

            X_np = X.to_numpy()
            y_np = y.to_numpy()

            algo_impl.fit(X_np, y_np)

            assert algo_impl.model is not None

            predictions = algo_impl.predict(X_np[:10])
            assert len(predictions) == 10
        finally:
            global_parameters.bayessearch = True


class TestNeuralNetworkClassifierClassEdgeCases:
    """Tests for edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        X = pd.DataFrame()
        y = pd.Series([], name="target")

        classifier = NeuralNetworkClassifier_class(X=X, y=y)

        assert classifier.X is not None
        assert classifier.y is not None

    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame({"feature_0": np.random.rand(30)})
        y = pd.Series([0, 1] * 15, name="target")

        classifier = NeuralNetworkClassifier_class(X=X, y=y)

        assert classifier.X.shape[1] == 1

    def test_many_features(self):
        """Test with many features."""
        n_features = 50
        X = pd.DataFrame(
            {f"feature_{i}": np.random.rand(30) for i in range(n_features)}
        )
        y = pd.Series([0, 1] * 15, name="target")

        classifier = NeuralNetworkClassifier_class(X=X, y=y)

        assert classifier.X.shape[1] == n_features

    def test_large_dataset(self):
        """Test with large dataset."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(1000),
                "feature_1": np.random.rand(1000),
            }
        )
        y = pd.Series([0, 1] * 500, name="target")

        classifier = NeuralNetworkClassifier_class(X=X, y=y)

        assert len(classifier.X) == 1000

    def test_parameter_space_with_string_hidden_layers(self):
        """Test parameter space with string representation of tuples."""
        global_parameters.bayessearch = True
        try:
            classifier = NeuralNetworkClassifier_class()

            param_space = classifier.parameter_space[0]

            # In Bayesian mode, hidden_layer_sizes should be strings
            assert "hidden_layer_sizes" in param_space

            # skopt stores them as Categorical with string values
            from skopt.space import Categorical

            assert isinstance(param_space["hidden_layer_sizes"], Categorical)
        finally:
            global_parameters.bayessearch = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

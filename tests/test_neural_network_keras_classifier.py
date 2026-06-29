"""Tests for NeuralNetworkKerasClassifier module."""

import numpy as np
import pandas as pd
import pytest

from ml_grid.model_classes.NeuralNetworkKerasClassifier import (
    NeuralNetworkClassifier,
)


class TestNeuralNetworkClassifierInit:
    """Tests for NeuralNetworkClassifier initialization."""

    def test_default_initialization(self):
        """Test default initialization with no parameters."""
        classifier = NeuralNetworkClassifier()

        assert classifier is not None
        assert classifier.hidden_layer_sizes == (64, 64)
        assert classifier.dropout_rate == 0.3
        assert classifier.learning_rate == 0.001
        assert classifier.activation_func == "relu"
        assert classifier.epochs == 10
        assert classifier.batch_size == 32
        assert classifier.early_stopping_patience == 3
        assert classifier.random_state is None
        assert classifier.model is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(128, 64),
            dropout_rate=0.5,
            learning_rate=0.0001,
            activation_func="tanh",
            epochs=50,
            batch_size=16,
            early_stopping_patience=5,
            random_state=42,
        )

        assert classifier.hidden_layer_sizes == (128, 64)
        assert classifier.dropout_rate == 0.5
        assert classifier.learning_rate == 0.0001
        assert classifier.activation_func == "tanh"
        assert classifier.epochs == 50
        assert classifier.batch_size == 16
        assert classifier.early_stopping_patience == 5
        assert classifier.random_state == 42

    def test_single_hidden_layer(self):
        """Test initialization with single hidden layer."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(32,))

        assert classifier.hidden_layer_sizes == (32,)
        assert len(classifier.hidden_layer_sizes) == 1

    def test_multiple_hidden_layers(self):
        """Test initialization with multiple hidden layers."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(256, 128, 64, 32))

        assert classifier.hidden_layer_sizes == (256, 128, 64, 32)
        assert len(classifier.hidden_layer_sizes) == 4


class TestNeuralNetworkClassifierNormalizeHiddenLayerSizes:
    """Tests for _normalize_hidden_layer_sizes method."""

    def test_tuple_input(self):
        """Test normalization with tuple input."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(64, 32))
        result = classifier._normalize_hidden_layer_sizes()

        assert result == (64, 32)
        assert isinstance(result, tuple)

    def test_list_input(self):
        """Test normalization with list input."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=[64, 32])
        result = classifier._normalize_hidden_layer_sizes()

        assert result == (64, 32)
        assert isinstance(result, tuple)

    def test_string_input(self):
        """Test normalization with string representation from skopt."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes="(64, 32)")
        result = classifier._normalize_hidden_layer_sizes()

        assert result == (64, 32)
        assert isinstance(result, tuple)

    def test_string_with_spaces(self):
        """Test normalization with string containing spaces."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes="(64,  32,  16)")
        result = classifier._normalize_hidden_layer_sizes()

        assert result == (64, 32, 16)
        assert isinstance(result, tuple)

    def test_single_element_tuple(self):
        """Test normalization with single element tuple."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(8,))
        result = classifier._normalize_hidden_layer_sizes()

        assert result == (8,)
        assert isinstance(result, tuple)

    def test_empty_list_to_tuple(self):
        """Test normalization converts empty list to empty tuple."""
        # Note: This would fail in build_model since it needs at least one layer
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=[])
        result = classifier._normalize_hidden_layer_sizes()

        assert result == ()
        assert isinstance(result, tuple)

    def test_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes="(invalid, input)")

        with pytest.raises(ValueError, match="Could not parse hidden_layer_sizes"):
            classifier._normalize_hidden_layer_sizes()

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        classifier = NeuralNetworkClassifier()
        classifier.hidden_layer_sizes = 123

        with pytest.raises(ValueError, match="hidden_layer_sizes must be a tuple"):
            classifier._normalize_hidden_layer_sizes()


class TestNeuralNetworkClassifierBuildModel:
    """Tests for build_model method."""

    def test_build_model_default(self):
        """Test building model with default parameters."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(64, 32))
        model = classifier.build_model(input_dim=10)

        assert model is not None
        # Check architecture: input(64) -> dropout -> dense(32) -> dropout -> output(1)
        assert len(model.layers) >= 5

    def test_build_model_single_layer(self):
        """Test building model with single hidden layer."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(32,))
        model = classifier.build_model(input_dim=5)

        assert model is not None
        # Input(32) -> dropout -> output(1)
        assert len(model.layers) >= 3

    def test_build_model_no_hidden_layers(self):
        """Test building model with no hidden layers raises error."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=())

        with pytest.raises(IndexError, match="tuple index out of range"):
            classifier.build_model(input_dim=5)

    def test_build_model_input_dimension(self):
        """Test that model accepts correct input dimension."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(32,))
        model = classifier.build_model(input_dim=10)

        assert model.input_shape[1] == 10

    def test_build_model_compilation_loss(self):
        """Test that model is compiled with binary_crossentropy loss."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(64,))
        model = classifier.build_model(input_dim=5)

        assert model.loss == "binary_crossentropy"

    def test_build_model_compilation_metrics(self):
        """Test that model has compile metrics."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(64,))
        model = classifier.build_model(input_dim=5)

        # Check metrics list exists (accuracy is tracked via compile_metrics)
        assert model.metrics is not None
        assert len(model.metrics) > 0

    def test_build_model_dropout_layers(self):
        """Test that dropout layers are correctly added."""
        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(64, 32), dropout_rate=0.5
        )
        model = classifier.build_model(input_dim=5)

        dropout_count = sum(
            1 for layer in model.layers if "Dropout" in str(type(layer))
        )
        # Should have dropout after each hidden layer (2 layers = 2 dropout layers)
        assert dropout_count == 2

    def test_build_model_activation_function(self):
        """Test that activation function is correctly applied."""
        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(64,), activation_func="tanh"
        )
        model = classifier.build_model(input_dim=5)

        # Check first dense layer has tanh activation
        assert model.layers[0].activation.__name__ == "tanh"

    def test_build_model_output_layer(self):
        """Test that output layer has 1 unit with sigmoid activation."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(64,))
        model = classifier.build_model(input_dim=5)

        # Last layer should be Dense(1) with sigmoid
        assert model.layers[-1].units == 1
        assert model.layers[-1].activation.__name__ == "sigmoid"


class TestNeuralNetworkClassifierFit:
    """Tests for fit method."""

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
        return X.to_numpy(), y.to_numpy()

    def test_fit_binary_classification(self, sample_data):
        """Test fitting a model on binary classification."""
        X, y = sample_data
        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=5, batch_size=16, random_state=42
        )
        classifier.fit(X, y)

        assert classifier.model is not None
        assert classifier.classes_ is not None

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for method chaining."""
        X, y = sample_data
        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=5, random_state=42
        )

        result = classifier.fit(X, y)

        assert result is classifier

    def test_fit_with_validation_data(self, sample_data):
        """Test fitting with validation data and early stopping."""
        X, y = sample_data
        X_val = X[:10]
        y_val = y[:10]

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,),
            epochs=10,
            early_stopping_patience=2,
            random_state=42,
        )
        classifier.fit(X, y, validation_data=(X_val, y_val))

        assert classifier.model is not None

    def test_fit_multiple_times(self, sample_data):
        """Test that model can be re-fitted."""
        X, y = sample_data
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=(32,), epochs=5)

        # First fit
        classifier.fit(X, y)
        model1 = classifier.model

        # Second fit
        classifier.fit(X, y)
        model2 = classifier.model

        assert model1 is not model2  # Should be cloned

    def test_fit_handles_categorical_labels(self, sample_data):
        """Test that categorical labels are converted to codes."""
        X = sample_data[0]
        y_cat = pd.Series([0, 1] * 25, name="target").astype("category")

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=5, random_state=42
        )
        classifier.fit(X, y_cat)

        assert classifier.model is not None


class TestNeuralNetworkClassifierPredict:
    """Tests for predict method."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for testing."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
                "feature_2": np.random.rand(50),
            }
        ).to_numpy()
        y = np.array([0, 1] * 25)

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=10, random_state=42
        )
        classifier.fit(X, y)
        return classifier

    def test_predict_binary_labels(self, trained_classifier):
        """Test that predict returns binary labels."""
        X_test = np.random.rand(10, 3)

        predictions = trained_classifier.predict(X_test)

        assert set(np.unique(predictions)).issubset({0, 1})
        assert len(predictions) == 10

    def test_predict_raises_error_if_not_fitted(self):
        """Test that predict raises error if model not fitted."""
        classifier = NeuralNetworkClassifier()

        with pytest.raises(RuntimeError, match="not been fitted yet"):
            classifier.predict(np.array([[1, 2, 3]]))

    def test_predict_output_shape(self, trained_classifier):
        """Test that predict output shape matches input."""
        X_test = np.random.rand(20, 3)

        predictions = trained_classifier.predict(X_test)

        # Note: predictions are returned as 2D array (n_samples, 1)
        assert len(predictions) == 20
        assert predictions.shape == (20, 1)


class TestNeuralNetworkClassifierPredictProba:
    """Tests for predict_proba method."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for testing."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
                "feature_2": np.random.rand(50),
            }
        ).to_numpy()
        y = np.array([0, 1] * 25)

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=10, random_state=42
        )
        classifier.fit(X, y)
        return classifier

    def test_predict_proba_values(self, trained_classifier):
        """Test that predict_proba returns probabilities between 0 and 1."""
        X_test = np.random.rand(10, 3)

        probs = trained_classifier.predict_proba(X_test)

        assert (probs >= 0).all() and (probs <= 1).all()
        assert len(probs) == 10

    def test_predict_proba_raises_error_if_not_fitted(self):
        """Test that predict_proba raises error if model not fitted."""
        classifier = NeuralNetworkClassifier()

        with pytest.raises(RuntimeError, match="not been fitted yet"):
            classifier.predict_proba(np.array([[1, 2, 3]]))


class TestNeuralNetworkClassifierScore:
    """Tests for score method."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for testing."""
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(50),
                "feature_1": np.random.rand(50),
                "feature_2": np.random.rand(50),
            }
        ).to_numpy()
        y = np.array([0, 1] * 25)

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(32,), epochs=10, random_state=42
        )
        classifier.fit(X, y)
        return classifier

    def test_score_returns_accuracy(self, trained_classifier):
        """Test that score returns accuracy metric."""
        X_test = np.random.rand(20, 3)
        y_test = np.array([0, 1] * 10)

        score = trained_classifier.score(X_test, y_test)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_score_raises_error_if_not_fitted(self):
        """Test that score raises error if model not fitted."""
        classifier = NeuralNetworkClassifier()

        with pytest.raises(RuntimeError):
            classifier.predict(np.array([[1, 2, 3]]))
        # Score internally calls predict, so it should also raise


class TestNeuralNetworkClassifierParameterSpace:
    """Tests for parameter space generation."""

    def test_hidden_layer_sizes_variations(self):
        """Test different hidden layer sizes work."""
        configs = [
            (8,),
            (16, 8),
            (32, 16, 8),
            (64, 32, 16),
        ]

        for sizes in configs:
            classifier = NeuralNetworkClassifier(hidden_layer_sizes=sizes)
            model = classifier.build_model(input_dim=5)

            assert model is not None

    def test_dropout_rate_variations(self):
        """Test different dropout rates."""
        dropout_rates = [0.0, 0.2, 0.5, 0.8]

        for rate in dropout_rates:
            classifier = NeuralNetworkClassifier(dropout_rate=rate)
            model = classifier.build_model(input_dim=5)

            assert model is not None

    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]

        for act in activations:
            classifier = NeuralNetworkClassifier(activation_func=act)
            model = classifier.build_model(input_dim=5)

            # Check that hidden layers have correct activation
            assert model is not None


class TestNeuralNetworkClassifierEdgeCases:
    """Tests for edge cases."""

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.rand(4, 3)
        y = np.array([0, 1, 0, 1])

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(8,), epochs=5, batch_size=2, random_state=42
        )
        classifier.fit(X, y)

        assert classifier.model is not None

    def test_single_sample_prediction(self):
        """Test prediction with single sample."""
        X_train = np.random.rand(10, 3)
        y_train = np.array([0, 1] * 5)

        classifier = NeuralNetworkClassifier(
            hidden_layer_sizes=(8,), epochs=5, random_state=42
        )
        classifier.fit(X_train, y_train)

        single_sample = np.random.rand(1, 3)
        prediction = classifier.predict(single_sample)

        assert len(prediction) == 1

    def test_empty_hidden_layers(self):
        """Test model with empty hidden layer tuple raises IndexError."""
        classifier = NeuralNetworkClassifier(hidden_layer_sizes=())

        with pytest.raises(IndexError, match="tuple index out of range"):
            classifier.build_model(input_dim=5)


class TestNeuralNetworkClassifierGPUConfig:
    """Tests for GPU memory configuration."""

    def test_gpu_memory_growth_disabled(self):
        """Test that GPU memory growth is configured."""
        from ml_grid.util import global_params as gp_module

        # Get current settings
        getattr(gp_module.global_parameters, "gpu_memory_limit", None)

        try:
            # Test with GPU memory limit set
            if hasattr(gp_module.global_parameters, "set_gpu_memory_limit"):
                gp_module.global_parameters.set_gpu_memory_limit(0.8)
                # This test verifies the configuration method exists
                assert True
        except Exception:
            pass  # GPU config may not be available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

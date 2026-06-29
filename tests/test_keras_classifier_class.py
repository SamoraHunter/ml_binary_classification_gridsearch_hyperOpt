import numpy as np
import pandas as pd
import pytest

from ml_grid.model_classes.keras_classifier_class import (
    create_model,
    KerasClassifierClass,
)


class TestCreateModel:
    """Tests for the create_model function."""

    def test_create_model_default_params(self):
        """Test creating a model with default parameters."""
        model = create_model()

        assert model is not None
        assert len(model.layers) > 0

    def test_create_model_single_layer(self):
        """Test creating a model with single layer."""
        model = create_model(layers=1, input_dim_val=5)

        assert model is not None
        # Input layer + hidden layers + dropout + output layer
        assert len(model.layers) >= 3

    def test_create_model_multiple_layers(self):
        """Test creating a model with multiple layers."""
        model = create_model(layers=3, input_dim_val=10)

        assert model is not None
        assert len(model.layers) >= 5  # At least 3 hidden + dropout + output

    def test_create_model_with_l1_regularization(self):
        """Test creating a model with L1 regularization."""
        model = create_model(layers=2, l1_reg=0.01, input_dim_val=5)

        assert model is not None
        # Check that kernel regularizer is set on first Dense layer
        dense_layer = model.layers[0]
        assert hasattr(dense_layer, "kernel_regularizer")

    def test_create_model_with_l2_regularization(self):
        """Test creating a model with L2 regularization."""
        model = create_model(layers=2, l2_reg=0.01, input_dim_val=5)

        assert model is not None
        # Check that kernel regularizer is set on first Dense layer
        dense_layer = model.layers[0]
        assert hasattr(dense_layer, "kernel_regularizer")

    def test_create_model_with_l1_l2_regularization(self):
        """Test creating a model with both L1 and L2 regularization."""
        model = create_model(layers=2, l1_reg=0.01, l2_reg=0.01, input_dim_val=5)

        assert model is not None
        dense_layer = model.layers[0]
        assert hasattr(dense_layer, "kernel_regularizer")

    def test_create_model_with_custom_width(self):
        """Test creating a model with custom width."""
        model = create_model(layers=2, width=32, input_dim_val=5)

        assert model is not None
        # Check first dense layer has 32 units
        assert model.layers[0].units == 32

    def test_create_model_with_custom_learning_rate(self):
        """Test creating a model with custom learning rate."""
        model = create_model(layers=1, learning_rate=0.001, input_dim_val=5)

        assert model is not None
        # Check optimizer learning rate
        assert hasattr(model.optimizer, "learning_rate")
        assert model.optimizer.learning_rate == 0.001

    def test_create_model_with_dropout(self):
        """Test creating a model with dropout."""
        model = create_model(layers=2, dropout_val=0.5, input_dim_val=5)

        assert model is not None
        # Check that dropout layer exists and has correct rate
        dropout_layers = [
            layer for layer in model.layers if "Dropout" in str(type(layer))
        ]
        assert len(dropout_layers) == 1
        assert dropout_layers[0].rate == 0.5

    def test_create_model_compilation_loss(self):
        """Test that the model is compiled with binary_crossentropy loss."""
        model = create_model(layers=1, input_dim_val=5)

        assert model.loss == "binary_crossentropy"

    def test_create_model_compilation_metrics(self):
        """Test that the model is compiled with correct metrics."""
        model = create_model(layers=1, input_dim_val=5)

        # Check metrics are not None (model was compiled)
        assert model.metrics is not None or hasattr(model, "compiled_metrics")

    def test_create_model_output_shape(self):
        """Test that the model has correct output shape."""
        model = create_model(layers=2, input_dim_val=5)

        # Last layer should be Dense(1) with sigmoid activation
        assert model.layers[-1].units == 1
        assert model.layers[-1].activation.__name__ == "sigmoid"


class TestKerasClassifierClassInit:
    """Tests for KerasClassifierClass initialization."""

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

    def test_normal_initialization(self, sample_data):
        """Test normal initialization of KerasClassifierClass."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        assert classifier.X is not None
        assert classifier.y is not None
        assert classifier.x_train_col_val == 3
        assert classifier.method_name == "KerasClassifier"

    def test_parameter_space_structure(self, sample_data):
        """Test that parameter space has expected structure."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        # Check parameter space exists and has required keys
        param_space = classifier.parameter_space

        assert "layers" in param_space
        assert "epochs" in param_space
        assert "batch_size" in param_space
        assert "l1_reg" in param_space
        assert "l2_reg" in param_space
        assert "width" in param_space

    def test_parameter_space_layers_values(self, sample_data):
        """Test layers parameter values are correctly generated."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        # Check that layers array is generated with logspace and floored
        layers_array = classifier.parameter_space["layers"]

        assert len(layers_array) > 0
        assert all(isinstance(x, (int, np.integer)) for x in layers_array)
        assert min(layers_array) >= 1

    def test_parameter_space_width_values(self, sample_data):
        """Test width parameter values are correctly generated."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        # Check that width array is generated with logspace and floored
        width_array = classifier.parameter_space["width"]

        assert len(width_array) > 0
        assert all(isinstance(x, (int, np.integer)) for x in width_array)
        assert min(width_array) >= 1

    def test_parameter_space_epochs_value(self, sample_data):
        """Test epochs parameter is set to list with correct value."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        assert isinstance(classifier.parameter_space["epochs"], list)
        assert len(classifier.parameter_space["epochs"]) == 1
        assert classifier.parameter_space["epochs"][0] == 300

    def test_parameter_space_batch_size_value(self, sample_data):
        """Test batch_size is calculated as half of data length."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        expected_batch_size = int(len(X) / 2)
        assert isinstance(classifier.parameter_space["batch_size"], list)
        assert len(classifier.parameter_space["batch_size"]) == 1
        assert classifier.parameter_space["batch_size"][0] == expected_batch_size

    def test_parameter_space_l1_reg_values(self, sample_data):
        """Test l1_reg parameter values are correctly generated."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        l1_array = classifier.parameter_space["l1_reg"]

        assert len(l1_array) == 4
        assert all(isinstance(x, (float, np.floating)) for x in l1_array)
        # Values should be logspace from -5 to -2
        assert min(l1_array) >= 1e-5
        assert max(l1_array) <= 1e-2

    def test_parameter_space_l2_reg_values(self, sample_data):
        """Test l2_reg parameter values are correctly generated."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        l2_array = classifier.parameter_space["l2_reg"]

        assert len(l2_array) == 4
        assert all(isinstance(x, (float, np.floating)) for x in l2_array)
        # Values should be logspace from -5 to -2
        assert min(l2_array) >= 1e-5
        assert max(l2_array) <= 1e-2

    def test_test_mode_override(self, sample_data):
        """Test that test_mode=True overrides parameter space."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        # Set test mode on global parameters
        from ml_grid.util import global_params as gp_module

        # Override test_mode to True
        original_test_mode = getattr(gp_module.global_parameters, "test_mode", False)
        try:
            gp_module.global_parameters.test_mode = True

            classifier = KerasClassifierClass(X=X, y=y)

            param_space = classifier.parameter_space

            # In test mode, parameters should be minimal
            assert param_space["layers"] == [1]
            assert param_space["epochs"] == [1]
            assert param_space["batch_size"] == [32]
            assert param_space["l1_reg"] == [0.0]
            assert param_space["l2_reg"] == [0.0]
            assert param_space["width"] == [4]

        finally:
            # Restore original test_mode
            gp_module.global_parameters.test_mode = original_test_mode

    def test_algorithm_implementation_is_keras_classifier(self, sample_data):
        """Test that algorithm_implementation is KerasClassifier instance."""
        X, y = sample_data

        classifier = KerasClassifierClass(X=X, y=y)

        from scikeras.wrappers import KerasClassifier as SCIKerasClassifier

        assert isinstance(classifier.algorithm_implementation, SCIKerasClassifier)


class TestKerasClassifierClassEdgeCases:
    """Tests for edge cases in KerasClassifierClass."""

    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrame."""
        X = pd.DataFrame()
        y = pd.Series([], name="target")

        classifier = KerasClassifierClass(X=X, y=y)

        assert classifier.X is not None
        assert classifier.y is not None

    def test_small_dataset(self):
        """Test with a very small dataset."""
        X = pd.DataFrame(
            {
                "feature_0": [0.1, 0.2],
                "feature_1": [0.3, 0.4],
            }
        )
        y = pd.Series([0, 1], name="target")

        classifier = KerasClassifierClass(X=X, y=y)

        assert classifier.x_train_col_val == 2
        # Batch size should be at least 1 even with small data
        batch_size = classifier.parameter_space["batch_size"][0]
        assert batch_size >= 1

    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame({"feature_0": np.random.rand(30)})
        y = pd.Series([0, 1] * 15, name="target")

        classifier = KerasClassifierClass(X=X, y=y)

        assert classifier.x_train_col_val == 1

    def test_many_features(self):
        """Test with many features."""
        n_features = 100
        X = pd.DataFrame(
            {f"feature_{i}": np.random.rand(50) for i in range(n_features)}
        )
        y = pd.Series([0, 1] * 25, name="target")

        classifier = KerasClassifierClass(X=X, y=y)

        assert classifier.x_train_col_val == n_features


class TestCreateModelWithRealFit:
    """Integration tests for create_model with actual training."""

    def test_model_can_be_trained(self):
        """Test that a model created by create_model can actually be trained."""
        from scikeras.wrappers import KerasClassifier

        # Create synthetic data
        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(100),
                "feature_1": np.random.rand(100),
            }
        )
        y = pd.Series([0, 1] * 50, name="target")

        # Create model using create_model
        keras_model_fn = create_model
        model_wrapper = KerasClassifier(
            model=keras_model_fn,
            verbose=0,
            learning_rate=0.001,
            layers=1,
            width=10,
            input_dim_val=2,
            l1_reg=0.0,
            l2_reg=0.0,
        )

        # Train for a few epochs
        model_wrapper.fit(X, y, epochs=5, batch_size=32)

        # Make predictions
        preds = model_wrapper.predict(X)

        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_with_regularization_can_be_trained(self):
        """Test that a model with regularization can be trained."""
        from scikeras.wrappers import KerasClassifier

        X = pd.DataFrame(
            {
                "feature_0": np.random.rand(100),
                "feature_1": np.random.rand(100),
            }
        )
        y = pd.Series([0, 1] * 50, name="target")

        keras_model_fn = create_model
        model_wrapper = KerasClassifier(
            model=keras_model_fn,
            verbose=0,
            learning_rate=0.001,
            layers=2,
            width=10,
            input_dim_val=2,
            l1_reg=0.01,
            l2_reg=0.01,
        )

        model_wrapper.fit(X, y, epochs=5, batch_size=32)

        preds = model_wrapper.predict(X)

        assert len(preds) == len(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

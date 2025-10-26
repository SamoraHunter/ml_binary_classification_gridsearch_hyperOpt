from typing import Optional
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for a Keras Sequential neural network.

    This class builds a simple feed-forward neural network for binary
    classification and wraps it to be compatible with scikit-learn's API,
    allowing it to be used in pipelines and hyperparameter tuning tools like
    GridSearchCV.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        activation_func: str = "relu",
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping_patience: int = 3,
        random_state: Optional[int] = None,
    ):
        """Initializes the NeuralNetworkClassifier.

        Args:
            hidden_layer_sizes (tuple[int, ...]): The number of units per hidden layer.
            dropout_rate (float): Dropout rate for the dropout layers.
            learning_rate (float): Learning rate for the Adam optimizer.
            activation_func (str): Activation function for the hidden layers.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per gradient update.
            early_stopping_patience (int): Number of epochs with no improvement
                on validation loss after which training will be stopped.
            random_state (Optional[int]): Seed for reproducibility. Defaults to None.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        print(f"Before conversion: hidden_layer_sizes: {self.hidden_layer_sizes}, type: {type(self.hidden_layer_sizes)}")
        if isinstance(self.hidden_layer_sizes, str):
            import ast
            self.hidden_layer_sizes = ast.literal_eval(self.hidden_layer_sizes)
        print(f"After conversion: hidden_layer_sizes: {self.hidden_layer_sizes}, type: {type(self.hidden_layer_sizes)}")
        print(f"hidden_layer_sizes: {self.hidden_layer_sizes}, type: {type(self.hidden_layer_sizes)}")
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model: Optional[Sequential] = None
        self.classes_: Optional[np.ndarray] = None

        if isinstance(self.hidden_layer_sizes, str):
            import ast
            self.hidden_layer_sizes = ast.literal_eval(self.hidden_layer_sizes)

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)

    def build_model(self, input_dim: int) -> Sequential:
        """Builds and compiles the Keras Sequential model.

        Args:
            input_dim (int): The number of input features.

        Returns:
            Sequential: The compiled Keras model.
        """
        model = Sequential()
        # Add input layer
        model.add(Dense(units=self.hidden_layer_sizes[0], activation=self.activation_func, input_dim=input_dim))
        model.add(Dropout(rate=self.dropout_rate))
        # Add subsequent hidden layers
        for units in self.hidden_layer_sizes[1:]:
            model.add(Dense(units=units, activation=self.activation_func))
            model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "NeuralNetworkClassifier":
        """Fits the neural network model to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.

        Returns:
            NeuralNetworkClassifier: The fitted estimator.
        """
        # Clear previous session to avoid layer name conflicts
        tf.keras.backend.clear_session()

        # --- FIX for 'Invalid dtype: category' ---
        # Keras expects numerical labels, not pandas categoricals.
        # If y is a categorical Series, convert it to its numerical codes.
        if hasattr(y, 'dtype') and str(y.dtype) == 'category':
            y = y.cat.codes.to_numpy()

        # Store class labels
        self.classes_ = np.unique(y)

        # Build and train the model
        if self.model is None:
            self.model = self.build_model(input_dim=X.shape[1])
        else:
            # Re-compile the model if it's being re-fitted (e.g., in a pipeline)
            self.model = clone_model(self.model)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        callbacks = []
        if "validation_data" in kwargs and self.early_stopping_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True
            ))

        self.model = self.build_model(input_dim=X.shape[1])
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            **kwargs
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (np.ndarray): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict().")
        # Predict class probabilities
        y_pred = self.model.predict(X, verbose=0)
        # Convert probabilities to class labels (0 or 1)
        return np.round(y_pred).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The class probabilities of the input samples.
        """
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict_proba().")
        # Return class probabilities
        return self.model.predict(X, verbose=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X (np.ndarray): Test samples.
            y (np.ndarray): True labels for X.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        # Calculate accuracy
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
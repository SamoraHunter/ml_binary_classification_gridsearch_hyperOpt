from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
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
        hidden_units_1: int = 64,
        hidden_units_2: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        activation_func: str = "relu",
        epochs: int = 10,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ):
        """Initializes the NeuralNetworkClassifier.

        Args:
            hidden_units_1 (int): Number of units in the first hidden layer.
            hidden_units_2 (int): Number of units in the second hidden layer.
            dropout_rate (float): Dropout rate for the dropout layers.
            learning_rate (float): Learning rate for the Adam optimizer.
            activation_func (str): Activation function for the hidden layers.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per gradient update.
            random_state (Optional[int]): Seed for reproducibility. Defaults to None.
        """
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model: Optional[Sequential] = None
        self.classes_: Optional[np.ndarray] = None

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
        model.add(
            Dense(
                units=self.hidden_units_1,
                activation=self.activation_func,
                input_dim=input_dim,
            )
        )
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.hidden_units_2, activation=self.activation_func))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkClassifier":
        """Fits the neural network model to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.

        Returns:
            NeuralNetworkClassifier: The fitted estimator.
        """
        # --- FIX for 'Invalid dtype: category' ---
        # Keras expects numerical labels, not pandas categoricals.
        # If y is a categorical Series, convert it to its numerical codes.
        if hasattr(y, 'dtype') and str(y.dtype) == 'category':
            y = y.cat.codes.to_numpy()

        # Store class labels
        self.classes_ = np.unique(y)

        # Build and train the model
        self.model = self.build_model(input_dim=X.shape[1])
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (np.ndarray): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
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
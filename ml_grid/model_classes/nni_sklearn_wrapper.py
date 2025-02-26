import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_units_1=64,
        hidden_units_2=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        activation_func="relu",
        epochs=10,
        batch_size=32,
        random_state=None,
    ):
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = None

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)

    def build_model(self, input_dim):
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

    def fit(self, X, y):
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

    def predict(self, X):
        # Predict class probabilities
        y_pred = self.model.predict(X, verbose=0)
        # Convert probabilities to class labels (0 or 1)
        return np.round(y_pred).astype(int)

    def predict_proba(self, X):
        # Return class probabilities
        return self.model.predict(X, verbose=0)

    def score(self, X, y):
        # Calculate accuracy
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
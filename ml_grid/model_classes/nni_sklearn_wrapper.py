import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
    ):
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(
            Dense(
                units=self.hidden_units_1,
                activation=self.activation_func,
                input_dim=self.X_train.shape[1],
            )
        )
        model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.hidden_units_2, activation=self.activation_func))
        model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        model.add(Dense(units=1, activation="sigmoid"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model = self.build_model()
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self

    def predict(self, X):
        # y_pred = self.model.predict_classes(X)
        y_pred = self.model.predict(X)
        return np.round(y_pred)

    def score(self, X, y):
        y_pred = np.round(self.predict(X))
        accuracy = accuracy_score(y, y_pred)
        return accuracy

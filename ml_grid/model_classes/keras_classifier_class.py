import math
from typing import Any, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.constraints import max_norm
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from ml_grid.util import param_space
from scikeras.wrappers import KerasClassifier

def create_model(
    layers: int = 1,
    kernel_reg: tf.keras.regularizers.Regularizer = tf.keras.regularizers.l1_l2(
        l1=0, l2=0
    ),
    width: int = 15,
    learning_rate: float = 0.01,
    dropout_val: float = 0.2,
    input_dim_val: int = 0,
) -> Sequential:
    """Builds and compiles a Keras Sequential model.

    Args:
        layers (int): The number of dense layers in the model.
        kernel_reg (tf.keras.regularizers.Regularizer): Kernel regularizer.
        width (int): The number of units in each dense layer.
        learning_rate (float): The learning rate for the Adam optimizer.
        dropout_val (float): The dropout rate.
        input_dim_val (int): The input dimension for the first layer.

    Returns:
        Sequential: The compiled Keras model.
    """
    model = Sequential()
    for i in range(0, layers):
        model.add(
            Dense(
                math.floor(width),
                input_dim=input_dim_val,
                kernel_initializer="uniform", activation="linear",
                kernel_constraint=max_norm(4), 
                kernel_regularizer=kernel_reg,
            )
        )

    model.add(Dropout(dropout_val))
    model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
    # Compile model
    optimizer = Adam(learning_rate=learning_rate) 
    metric = tf.keras.metrics.AUC()

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[metric, "accuracy"],
    )

    return model

class kerasClassifier_class:
    """Keras Sequential model classifier wrapped for use with scikit-learn."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the kerasClassifier_class.

        This configures a Keras Sequential model for binary classification,
        wrapped in a KerasClassifier to be compatible with scikit-learn's
        hyperparameter tuning utilities.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        self.X = X
        self.y = y

        self.x_train_col_val = len(X.columns)

        self.algorithm_implementation = KerasClassifier(
            model=create_model,
            verbose=0,
            learning_rate=0.0001,
            layers=1,
            width=1,
            input_dim_val=self.x_train_col_val,
        )
        self.method_name = "kerasClassifier_class"
        X_data = self.X
        y_data = self.y

        # vals = np.linspace(2, 750, 6)
        vals = np.logspace(1, 2.0, 3)

        floorer = lambda t: math.floor(t)
        floored_width = np.array([floorer(xi) for xi in vals])
        floored_width = np.insert(floored_width, 0, 1, axis=None)
        floored_width

        vals = np.logspace(1, 2.0, 3)

        floorer = lambda t: math.floor(t)
        floored_depth = np.array([floorer(xi) for xi in vals])
        floored_depth = np.insert(floored_depth, 0, 1, axis=None)
        floored_depth

        length_x_data = len(self.X)

        length_x_data = length_x_data

        self.parameter_space = {
            "layers": floored_depth,
            #'epochs':log_large_long,
            "epochs": [300],
            "batch_size": [int(length_x_data / 2)],
            # kernel_reg=reg_reg_list_comb,
            "width": floored_width,
            #'learning_rate' : np.logspace(-4, -6, 2)
            # dropout_val = np.logspace(-1, -3, 2)
        }

    # The duplicate create_model method has been removed. The module-level
    # function will be used instead.

import keras

from aeon.classification.deep_learning.mlp import MLPClassifier


class MLPClassifier_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        log_epoch = ml_grid_object.local_param_dict.get("log_epoch")

        self.algorithm_implementation = MLPClassifier

        self.method_name = "MLPClassifier"

        self.parameter_space = {
            "n_epochs": [log_epoch],  # Number of epochs to train the model
            "batch_size": [8, 16, 32],  # Number of samples per gradient update
            "random_state": [random_state_val],  # Seed for random number generation
            "verbose": [verbose_param],  # Whether to output extra information
            "loss": [
                "binary_crossentropy"
            ],  # Fit parameter for the Keras model #must be binary? # 'mean_squared_error',
            #'file_path': ['./', '/models'],                         # File path when saving ModelCheckpoint callback
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'top_model'],          # The name of the file of the best model
            #'last_file_name': ['last_model', 'final_model'],        # The name of the file of the last model
            "optimizer": [
                keras.optimizers.Adadelta(),
                keras.optimizers.Adam(),
            ],  # Keras optimizer
            "metrics": [
                ["accuracy"],
                ["accuracy", "mae"],
            ],  # List of strings for metrics
            "activation": [
                "sigmoid",
                "relu",
            ],  # Activation function used in the output linear layer
            "use_bias": [True, False],  # Whether the layer uses a bias vector
        }

from aeon.classification.deep_learning.inception_time import (
    IndividualInceptionClassifier,
)


class IndividualInceptionClassifier_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        log_epoch = ml_grid_object.local_param_dict.get("log_epoch")

        self.algorithm_implementation = IndividualInceptionClassifier

        self.method_name = "IndividualInceptionClassifier"

        self.parameter_space = {
            "depth": [6, 8, 10],
            "nb_filters": [32, 64, 128],
            "nb_conv_per_layer": [3, 4, 5],
            "kernel_size": [30, 40, 50],
            "use_max_pooling": [True, False],
            "max_pool_size": [2, 3, 4],
            "strides": [1, 2],
            "dilation_rate": [1, 2],
            "padding": ["same", "valid"],
            "activation": ["relu", "elu"],
            "use_bias": [True, False],
            "use_residual": [True, False],
            "use_bottleneck": [True, False],
            "bottleneck_size": [16, 32, 64],
            "use_custom_filters": [True, False],
            "batch_size": [32, 64, 128],
            "use_mini_batch_size": [True, False],
            "n_epochs": [log_epoch],
            #'callbacks': [None, [ReduceOnPlateau(), ModelCheckpoint()]],
            #'file_path': ['./', '/path/to/save'],
            "save_best_model": [False],  # Whether or not to save the best model
            "save_last_model": [False],  # Whether or not to save the last model
            #'best_file_name': ['best_model', 'model_best'],
            #'last_file_name': ['last_model', 'model_last'],
            "random_state": [random_state_val],
            "verbose": [verbose_param],
            #'optimizer': [Adam(), RMSprop(), SGD()],
            "loss": ["categorical_crossentropy", "binary_crossentropy"],
            "metrics": ["accuracy"],
        }

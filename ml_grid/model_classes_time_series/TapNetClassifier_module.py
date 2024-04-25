from aeon.classification.deep_learning import TapNetClassifier

import keras

from ml_grid.util.param_space import ParamSpace


class TapNetClassifier_class:

    def __init__(self, ml_grid_object):

        verbose_param = ml_grid_object.verbose
        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")
        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = TapNetClassifier()

        self.method_name = "TapNetClassifier"

        self.parameter_space = {
            "filter_sizes": [(256, 256, 128), (128, 128, 64)],
            "kernel_size": [(8, 5, 3), (4, 3, 2)],
            "layers": [(500, 300), (400, 200)],
            "n_epochs": log_epoch,
            "batch_size": [16, 32],
            "dropout": [0.5, 0.3, 0.2],
            "dilation": [1, 2],
            "activation": ["sigmoid", "relu"],
            "loss": ["binary_crossentropy", "categorical_crossentropy"],
            "optimizer": [keras.optimizers.Adam(0.01), keras.optimizers.SGD(0.01)],
            "use_bias": [True, False],
            "use_rp": [True, False],
            "use_att": [True, False],
            "use_lstm": [True, False],
            "use_cnn": [True, False],
            "verbose": [verbose_param],
            "random_state": [random_state_val],
        }

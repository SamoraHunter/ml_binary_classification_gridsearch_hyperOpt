from aeon.classification.deep_learning.fcn import FCNClassifier
import keras


class FCNClassifier_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        log_epoch = ml_grid_object.local_param_dict.get("log_epoch")

        self.algorithm_implementation = FCNClassifier

        self.method_name = "FCNClassifier"

        self.parameter_space = {
            "n_layers": [3],
            "n_filters": [128, 256, 128],
            "kernel_size": [8, 5, 3],
            "dilation_rate": [1],
            "strides": [1],
            "padding": ["same"],
            "activation": ["relu"],
            "use_bias": [True],
            "n_epochs": [log_epoch],
            "batch_size": [16],
            "use_mini_batch_size": [True],
            "random_state": [random_state_val],
            "verbose": [verbose_param],
            "loss": ["categorical_crossentropy"],
            "metrics": [None],
            "optimizer": [keras.optimizers.Adam(0.01), keras.optimizers.SGD(0.01)],
            #'n_jobs':[1] #not a param
            #'file_path': ['./'],
            #'save_best_model': [False],
            #'save_last_model': [False],
            #'best_file_name': ['best_model'],
            #'last_file_name': ['last_model'],
            #'callbacks': [None]
        }

from aeon.classification.deep_learning.cnn import CNNClassifier

from ml_grid.util.param_space import ParamSpace


class CNNClassifier_class:

    def __init__(self, ml_grid_object):

        time_limit_param = ml_grid_object.global_params.time_limit_param

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        param_space = ParamSpace(
            ml_grid_object.local_param_dict.get("param_space_size")
        )

        log_epoch = param_space.param_dict.get("log_epoch")

        self.algorithm_implementation = CNNClassifier()

        self.method_name = "CNNClassifier"

        self.parameter_space = {
            #'n_layers': [2, 3, 4],
            #'kernel_size': [3, 5, 7],
            #'n_filters': [[6, 12], [8, 16], [10, 20]],
            #'avg_pool_size': [2, 3, 4],
            "activation": ["sigmoid", "relu"],
            "padding": ["valid"],
            #'strides': [1, 2],
            "dilation_rate": [1, 2],
            "use_bias": [True],
            "random_state": [random_state_val],
            "n_epochs": [log_epoch],
            "batch_size": [16, 32, 64],
            "verbose": [verbose_param],
            "loss": ["binary_crossentropy"],
            "metrics": ["accuracy"],
            #'save_best_model': [True, False],
            #'save_last_model': [True, False],
            #'best_file_name': ['best_model', 'top_model'],
            #'last_file_name': ['last_model', 'final_model'],
        }

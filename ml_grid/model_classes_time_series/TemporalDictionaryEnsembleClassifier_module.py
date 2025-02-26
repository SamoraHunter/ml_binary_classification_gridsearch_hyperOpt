from aeon.classification.dictionary_based._tde import TemporalDictionaryEnsemble

from ml_grid.util.global_params import global_parameters
from ml_grid.util.param_space import ParamSpace


class TemporalDictionaryEnsemble_class:

    def __init__(self, ml_grid_object):

        verbose_param = ml_grid_object.verbose

        random_state_val = ml_grid_object.global_params.random_state_val

        time_limit_param = global_parameters.time_limit_param

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = TemporalDictionaryEnsemble()

        self.method_name = "TemporalDictionaryEnsemble"

        self.parameter_space = {
            "n_parameter_samples": [100, 250, 500],
            "max_ensemble_size": [25, 50, 100],
            "max_win_len_prop": [0.5, 1.0],
            "min_window": [5, 10, 15],
            "randomly_selected_params": [25, 50, 75],
            "bigrams": [True, False, None],
            "dim_threshold": [0.7, 0.85, 0.95],
            "max_dims": [10, 20, 30],
            "time_limit_in_minutes": time_limit_param,
            "contract_max_n_parameter_samples": [100, 250, 500],
            "typed_dict": [True, False],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }

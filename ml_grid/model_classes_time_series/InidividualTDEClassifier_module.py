from aeon.classification.dictionary_based._tde import IndividualTDE


class IndividualTDE_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = IndividualTDE()

        self.method_name = "IndividualTDE"

        self.parameter_space = {
            "window_size": [5, 10, 15],
            "word_length": [4, 8, 12],
            "norm": [True, False],
            "levels": [1, 2, 3],
            "igb": [True, False],
            "alphabet_size": [3, 4, 5],
            "bigrams": [True, False],
            "dim_threshold": [0.8, 0.85, 0.9],
            "max_dims": [15, 20, 25],
            "typed_dict": [True, False],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }

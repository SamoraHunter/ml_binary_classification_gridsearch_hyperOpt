from aeon.classification.hybrid._hivecote_v1 import HIVECOTEV1


class HIVECOTEV1_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = HIVECOTEV1

        self.method_name = "HIVECOTEV1"

        self.parameter_space = {
            "stc_params": [None],
            "tsf_params": [None],
            "rise_params": [None],
            "cboss_params": [None],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }

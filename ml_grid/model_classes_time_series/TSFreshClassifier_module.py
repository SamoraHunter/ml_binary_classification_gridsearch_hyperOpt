from aeon.classification.feature_based import TSFreshClassifier


class TSFreshClassifier_class:
    def __init__(self, ml_grid_object):
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val
        self.ml_grid_object = ml_grid_object
        self.method_name = "TSFreshClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "default_fc_parameters": ["minimal"],
                "n_jobs": [1],
            }
            self.algorithm_implementation = TSFreshClassifier()
            return

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "default_fc_parameters": ["minimal"],
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
        else:
            self.parameter_space = {
                "default_fc_parameters": ["minimal"],
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
        self.algorithm_implementation = TSFreshClassifier()

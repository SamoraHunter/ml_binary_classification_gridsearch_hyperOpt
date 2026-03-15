from aeon.classification.interval_based import TimeSeriesForestClassifier
from skopt.space import Categorical


class TimeSeriesForestClassifier_class:
    def __init__(self, ml_grid_object):
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val
        self.ml_grid_object = ml_grid_object
        self.method_name = "TimeSeriesForestClassifier"

        gp = ml_grid_object.global_params
        test_mode = getattr(gp, "test_mode", False)
        if not test_mode and hasattr(gp, "__dict__"):
            test_mode = gp.__dict__.get("test_mode", False)

        if test_mode:
            self.parameter_space = {
                "n_estimators": [10],
                "min_interval_length": [3],
                "n_jobs": [1],
                "random_state": [random_state_val],
            }
        elif ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "n_estimators": Categorical([50, 100, 200]),
                "min_interval_length": Categorical([3, 5]),
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
        else:
            self.parameter_space = {
                "n_estimators": [50, 100, 200],
                "min_interval_length": [3, 5],
                "n_jobs": [n_jobs_model_val],
                "random_state": [random_state_val],
            }
        self.algorithm_implementation = TimeSeriesForestClassifier()

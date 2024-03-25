from aeon.classification.feature_based._tsfresh_classifier import TSFreshClassifier
from sklearn.ensemble import RandomForestClassifier


class TSFreshClassifier_class:

    def __init__(self, ml_grid_object):

        verbose_param = ml_grid_object.verbose
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = TSFreshClassifier

        self.method_name = "TSFreshClassifier"

        self.parameter_space = {
            "default_fc_parameters": ["minimal", "efficient", "comprehensive"],
            "relevant_feature_extractor": [True, False],
            "estimator": [None, RandomForestClassifier(n_estimators=200)],
            "verbose": [verbose_param],
            "n_jobs": [n_jobs_model_val],
            "chunksize": [None, 10, 100],
            "random_state": [random_state_val],
        }

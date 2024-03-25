from aeon.classification.feature_based._summary_classifier import SummaryClassifier
from sklearn.ensemble import RandomForestClassifier


class SummaryClassifier_class:

    def __init__(self, ml_grid_object):

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val
        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = SummaryClassifier

        self.method_name = "SummaryClassifier"

        self.parameter_space = {
            "summary_functions": [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "sum",
                "skew",
                "kurt",
                "var",
                "mad",
                "sem",
                "nunique",
                "count",
            ],
            "summary_quantiles": [None, [0.25, 0.5, 0.75]],
            "estimator": [None, RandomForestClassifier(n_estimators=200)],
            "n_jobs": [n_jobs_model_val],
            "random_state": [random_state_val],
        }

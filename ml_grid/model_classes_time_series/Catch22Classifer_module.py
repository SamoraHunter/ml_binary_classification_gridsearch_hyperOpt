from aeon.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Catch22Classifier_class:

    def __init__(self, ml_grid_object):

        verbose_param = ml_grid_object.verbose
        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = Catch22Classifier()

        self.method_name = "Catch22Classifier"

        self.parameter_space = {
            "features": ["all", ["DN_HistogramMode_5", "DN_HistogramMode_10"]],
            "catch24": [True, False],
            "outlier_norm": [True, False],
            "replace_nans": [True, False],
            "use_pycatch22": [True, False],
            "estimator": [
                RandomForestClassifier(n_estimators=200),
                DecisionTreeClassifier(),
            ],
            "random_state": [random_state_val],
            "n_jobs": [n_jobs_model_val],
        }

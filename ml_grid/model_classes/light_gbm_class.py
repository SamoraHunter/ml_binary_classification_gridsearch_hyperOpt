import lightgbm as lgb
from ml_grid.util import param_space

# from sklearn.base import BaseEstimator, ClassifierMixin

from ml_grid.model_classes.lightgbm_class import LightGBMClassifier


class LightGBMClassifierWrapper:

    def __init__(self, X=None, y=None, parameter_space_size=None):
        self.X = X
        self.y = y

        self.algorithm_implementation = (
            LightGBMClassifier()
        )  # lgb.LGBMClassifier() #custom skelarn wrapper
        self.method_name = "LightGBMClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        self.parameter_space = {
            "boosting_type": ["gbdt", "dart", "goss"],
            "num_leaves": self.parameter_vector_space.param_dict.get("log_large_long"),
            "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),
            "n_estimators": self.parameter_vector_space.param_dict.get(
                "log_large_long"
            ),
            "objective": ["binary"],
            "num_class": [1],
            "metric": ["logloss"],
            "feature_fraction": [0.8, 0.9, 1.0],
            "early_stopping_rounds": [None, 10, 20],
        }


print("Imported LightGBM classifier wrapper class")
# light_gbm_class LightGBMClassifierWrapper

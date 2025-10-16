from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ORuleFitClassifier import H2ORuleFitClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_rulefit_classifier_class")

class H2O_RuleFit_class:
    """H2ORuleFitClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ORuleFitClassifier()
        self.method_name = "H2ORuleFitClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "max_rule_length": Integer(2, 10),
                    "model_type": Categorical(["rules_and_linear", "rules", "linear"]),
                    "rule_generation_ntrees": Integer(20, 100),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "max_rule_length": [3, 5, 10],
                    "model_type": ["rules_and_linear", "rules"],
                    "rule_generation_ntrees": [50],
                    "seed": [1, 42],
                }
            ]
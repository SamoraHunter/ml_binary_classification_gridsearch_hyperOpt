from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ODeepLearningClassifier import H2ODeepLearningClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_deeplearning_classifier_class")

class H2O_DeepLearning_class:
    """H2ODeepLearningClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ODeepLearningClassifier()
        self.method_name = "H2ODeepLearningClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            # Bayesian search requires scalar parameters for BayesSearchCV
            # We encode the 'hidden' architecture as a categorical string identifier
            # which will be decoded in the classifier wrapper
            self.parameter_space = [
                {
                    "epochs": Integer(5, 20),
                    # Custom encoding: Use string identifiers instead of tuple/list architectures
                    # These will be mapped to actual layer configurations in the classifier
                    "hidden_config": Categorical(['small', 'medium', 'large']),
                    "activation": Categorical(['Rectifier', 'Tanh']),
                    "l1": Real(1e-5, 1e-3, "log-uniform"),  # Changed lower bound from 0
                    "l2": Real(1e-5, 1e-3, "log-uniform"),  # Changed lower bound from 0
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            # Grid search can handle list/tuple parameters directly
            self.parameter_space = [
                {
                    "epochs": [5, 10],
                    "hidden": [[20, 20], [50, 50]],
                    "activation": ['Rectifier', 'Tanh'],
                    "l1": [0, 1e-4],
                    "l2": [0, 1e-4],
                    "seed": [1, 42],
                }
            ]
from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2ONaiveBayesClassifier import H2ONaiveBayesClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_naive_bayes_classifier_class")

class H2O_NaiveBayes_class:
    """H2ONaiveBayesClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = H2ONaiveBayesClassifier()
        self.method_name = "H2ONaiveBayesClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "laplace": Real(1e-5, 10.0, "log-uniform"),
                    "min_sdev": Real(0.001, 1.0, "log-uniform"),
                    "eps_sdev": Real(1e-5, 1.0, "log-uniform"),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "laplace": [0, 1, 5],
                    "min_sdev": [0.001, 0.1],
                    "eps_sdev": [0, 0.001, 0.1],
                    "seed": [1, 42],
                }
            ]
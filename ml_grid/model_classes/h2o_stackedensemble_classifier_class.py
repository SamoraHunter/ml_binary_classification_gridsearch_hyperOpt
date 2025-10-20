from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OStackedEnsembleClassifier import H2OStackedEnsembleClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer
import logging
from h2o.estimators import (
    H2OGradientBoostingEstimator,
    H2ORandomForestEstimator,
    H2OGeneralizedLinearEstimator,
)

logging.getLogger('ml_grid').debug("Imported h2o_stackedensemble_classifier_class")

class H2O_StackedEnsemble_class:
    """H2OStackedEnsembleClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y

        # --- FIX: Define and pass base models to the Stacked Ensemble ---
        # A stacked ensemble requires a list of base models to train.
        # We define a default set of H2O models here.
        base_models = [
            H2OGradientBoostingEstimator(model_id="gbm_base_se", seed=1),
            H2ORandomForestEstimator(model_id="drf_base_se", seed=1),
            # --- FIX: Use 'lambda_' instead of 'lambda' ---
            # 'lambda' is a reserved keyword in Python. H2O's API uses 'lambda_'.
            # Also, explicitly set family='binomial' for classification.
            H2OGeneralizedLinearEstimator(model_id="glm_base_se", seed=1, family='binomial', lambda_=0.0),
        ]
        self.algorithm_implementation = H2OStackedEnsembleClassifier(base_models=base_models)
        self.method_name = "H2OStackedEnsembleClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        # The parameter space is minimal as the base models are fixed.
        # We can expose metalearner params here in the future.
        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "metalearner_algorithm": Categorical(["AUTO", "glm"]),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "metalearner_algorithm": ["AUTO", "glm"],
                    "seed": [1, 42],
                }
            ]
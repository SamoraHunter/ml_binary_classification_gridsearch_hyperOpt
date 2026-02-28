"""TPOT Classifier Configuration.

This module contains the TPOTClassifierClass, which is a configuration
class for the TPOTClassifierWrapper. It provides parameter spaces for
grid search and Bayesian optimization, with a focus on providing a fast
default for unit testing.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Categorical, Integer

from ml_grid.model_classes.TPOTClassifierWrapper import TPOTClassifierWrapper
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)


class TPOTClassifierClass:
    """Configuration class for TPOTClassifierWrapper."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = TPOTClassifierWrapper()
        self.method_name = "TPOTClassifier"

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        if getattr(global_parameters, "test_mode", False):
            if global_parameters.bayessearch:
                self.parameter_space = {
                    "generations": Integer(2, 5),
                    "population_size": Integer(5, 10),
                    "max_time_mins": Integer(1, 2),
                }
            else:
                self.parameter_space = [
                    {"generations": [2], "population_size": [5], "max_time_mins": [1]}
                ]
        elif global_parameters.bayessearch:
            # A slightly larger space for Bayesian search, but still constrained
            self.parameter_space = {
                "generations": Integer(5, 100),
                "population_size": Integer(20, 100),
                "scoring": Categorical(
                    ["accuracy", "f1", "roc_auc", "precision", "recall"]
                ),
                "max_time_mins": Integer(10, 120),  # Time limit is crucial
            }
        else:
            # Expanded parameter space for grid search
            self.parameter_space = [
                {
                    "generations": [5, 10, 20],
                    "population_size": [20, 50, 100],
                    "max_time_mins": [10, 30, 60],
                    "scoring": ["accuracy", "f1", "roc_auc"],
                }
            ]

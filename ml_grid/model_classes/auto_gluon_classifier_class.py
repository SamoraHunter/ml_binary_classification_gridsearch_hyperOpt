"""AutoGluon Classifier Configuration.

This module contains the AutoGluonClassifierClass, which is a configuration
class for the AutoGluonClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Categorical, Integer

from ml_grid.model_classes.AutoGluonClassifier import AutoGluonClassifier
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)


class AutoGluonClassifierClass:
    """Configuration class for AutoGluonClassifier."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = AutoGluonClassifier()
        self.method_name = "AutoGluonClassifier"

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        if getattr(global_parameters, "test_mode", False):
            self.parameter_space = [
                {
                    "time_limit": [5],
                    "presets": ["medium_quality"],
                }
            ]
        elif global_parameters.bayessearch:
            self.parameter_space = {
                "time_limit": Integer(120, 240),
                "presets": Categorical(["medium_quality"]),
            }
        else:
            self.parameter_space = [
                {
                    "time_limit": [120, 180],
                    "presets": ["medium_quality"],
                }
            ]

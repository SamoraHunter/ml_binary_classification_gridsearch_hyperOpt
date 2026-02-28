"""AutoKeras Classifier Configuration.

This module contains the AutoKerasClassifierClass, which is a configuration
class for the AutoKerasClassifierWrapper.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Integer

from ml_grid.model_classes.AutoKerasClassifierWrapper import AutoKerasClassifierWrapper
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)


class AutoKerasClassifierClass:
    """Configuration class for AutoKerasClassifierWrapper."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = AutoKerasClassifierWrapper()
        self.method_name = "AutoKerasClassifier"

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        if getattr(global_parameters, "test_mode", False):
            # Extremely small parameter space for fast unit testing
            logger.info("Using test_mode parameter space for AutoKerasClassifier")
            self.parameter_space = [{"max_trials": [2], "epochs": [3]}]
        elif global_parameters.bayessearch:
            self.parameter_space = {
                "max_trials": Integer(3, 10),
                "epochs": Integer(10, 30),
            }
        else:
            self.parameter_space = [
                {
                    "max_trials": [3, 5],
                    "epochs": [10, 20],
                }
            ]

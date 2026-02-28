"""FLAML Classifier Configuration.

This module contains the FLAMLClassifierClass, which is a configuration
class for the FLAMLClassifierWrapper.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Integer

from ml_grid.model_classes.FLAMLClassifierWrapper import FLAMLClassifierWrapper
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)


class FLAMLClassifierClass:
    """Configuration class for FLAMLClassifierWrapper."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        self.algorithm_implementation = FLAMLClassifierWrapper()
        self.method_name = "FLAMLClassifier"

        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        if global_parameters.bayessearch:
            self.parameter_space = {
                "time_budget": Integer(1, 5),
            }
        else:
            self.parameter_space = [
                {
                    "time_budget": [1, 2],
                }
            ]

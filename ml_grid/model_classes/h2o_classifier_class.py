"""H2O AutoML Classifier.

This module contains the H2OAutoMLConfig, which is a configuration
class for the H2OAutoMLClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional

from skopt.space import Categorical, Integer

from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_classifier_class")


class H2OAutoMLClass:
    """Configuration class for H2OAutoMLClassifier.

    This class holds an instance of the H2OAutoMLClassifier and defines its
    hyperparameter search space for both grid search and Bayesian optimization.

    Attributes:
        algorithm_implementation (H2OAutoMLClassifier): An instance of the
            H2OAutoMLClassifier.
        method_name (str): The name of the algorithm, "H2OAutoMLClassifier".
        parameter_space (List[Dict[str, Any]]): A list of dictionaries defining
            the hyperparameter search space.
    """

    def __init__(
        self,
        parameter_space_size: Optional[str] = None,
    ) -> None:
        """Initializes the H2OAutoMLClass.

        Args:
            parameter_space_size (Optional[str]): A string indicating the size of
                the parameter space to use (e.g., 'small', 'medium'). This
                argument is currently unused but kept for API consistency.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        global_params = global_parameters
        logger.debug("Initializing H2OAutoMLClass")

        self.algorithm_implementation = H2OAutoMLClassifier()
        self.method_name: str = "H2OAutoMLClassifier"
        self.parameter_space: List[Dict[str, Any]]

        if getattr(global_parameters, "test_mode", False):
            self.parameter_space = [
                {
                    "max_runtime_secs": [5],
                    "max_models": [1],
                    "nfolds": [2],
                }
            ]
        elif global_parameters.bayessearch:
            # Define the parameter space for Bayesian optimization
            self.parameter_space = [
                {
                    "max_runtime_secs": Integer(60, 360),  # 1 to 6 minutes
                    "nfolds": Integer(2, 10),  # Number of folds in cross-validation
                    "seed": Integer(1, 1000),  # Random seed for reproducibility
                    "max_models": Integer(5, 20),  # Number of models to build
                    "balance_classes": Categorical(
                        [True, False]
                    ),  # Whether to balance classes
                }
            ]
        else:
            # Define the parameter space for traditional grid search
            self.parameter_space = [
                {
                    "max_runtime_secs": [120, 240],  # 2 or 4 minutes
                    "nfolds": [2, 5, 10],  # Different fold numbers for cross-validation
                    "seed": [1, 42, 123],  # Different random seeds
                    "max_models": [10, 20],  # Number of models to try
                    "balance_classes": [True, False],  # Balance classes option
                }
            ]

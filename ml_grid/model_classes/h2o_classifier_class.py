from typing import Optional
from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_classifier_class")

class H2OAutoMLConfig:
    """
    Configuration class for H2OAutoMLClassifier.

    This class holds an instance of the H2OAutoMLClassifier and defines its
    hyperparameter search space for both grid search and Bayesian optimization.
    """

    def __init__(
        self,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the H2OAutoMLConfig.

        Args:
            parameter_space_size (Optional[str]): A string indicating the size of
                the parameter space to use (e.g., 'small', 'medium'). This argument
                is currently unused but kept for API consistency.
        """
        global_params = global_parameters
        logger.debug("Initializing H2OAutoMLConfig")

        self.algorithm_implementation = H2OAutoMLClassifier()
        self.method_name = "H2OAutoMLClassifier"

        if global_params.bayessearch:
            # Define the parameter space for Bayesian optimization
            self.parameter_space = [
                {"max_runtime_secs": Integer(60, 360),  # 1 to 6 minutes
                 "nfolds": Integer(2, 10),  # Number of folds in cross-validation
                 "seed": Integer(1, 1000),  # Random seed for reproducibility
                 "max_models": Integer(5, 20),  # Number of models to build
                 "balance_classes": Categorical([True, False]),  # Whether to balance classes
                }
            ]
        else:
            # Define the parameter space for traditional grid search
            self.parameter_space = [
                {"max_runtime_secs": [120, 240],  # 2 or 4 minutes
                 "nfolds": [2, 5, 10],  # Different fold numbers for cross-validation
                 "seed": [1, 42, 123],  # Different random seeds
                 "max_models": [10, 20],  # Number of models to try
                 "balance_classes": [True, False],  # Balance classes option
                }
            ]

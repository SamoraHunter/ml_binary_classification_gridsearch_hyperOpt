from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
# from h2o.sklearn import H2OAutoMLClassifier

print("Imported h2o_classifier_class")

class h2o_classifier_class:
    """H2OAutoMLClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize the H2OAutoMLClassifier_class.

        Args:
            X (_type_): Feature matrix for training (optional).
            y (_type_): Target vector for training (optional).
            parameter_space_size (_type_): Size of the parameter space for optimization.
        """
        global_params = global_parameters()
        print("init h2o_classifier_class")
        self.X = X
        self.y = y

        self.algorithm_implementation = H2OAutoMLClassifier()
        self.method_name = "H2OAutoMLClassifier"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_params.bayessearch:
            # Define the parameter space for Bayesian optimization
            self.parameter_space = [
                {"max_runtime_secs": Integer(60, 900),  # Uniform between 1 minute and 15 mins
                 "nfolds": Integer(2, 10),  # Number of folds in cross-validation
                 "seed": Integer(1, 1000),  # Random seed for reproducibility
                 "max_models": Integer(2, 10),  # Number of models to build
                 "balance_classes": Categorical([True, False]),  # Whether to balance classes
                 "project_name": Categorical([None])  # Project name (None means auto-generated)
                }
            ]
        else:
            # Define the parameter space for traditional grid search
            self.parameter_space = [
                {"max_runtime_secs": [360],  # 1 hour runtime
                 "nfolds": [2, 5, 10],  # Different fold numbers for cross-validation
                 "seed": [1, 42, 123],  # Different random seeds
                 "max_models": [10, 20, 50],  # Number of models to try
                 "balance_classes": [True, False],  # Balance classes option
                 "project_name": [None],  # Project name for H2O
                }
            ]

        return None

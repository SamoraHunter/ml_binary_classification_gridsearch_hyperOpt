from sklearn.ensemble import AdaBoostClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer

print("Imported AdaBoostClassifier class")

class adaboost_class:
    """AdaBoostClassifier with support for both Bayesian and non-Bayesian parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize the adaboost_class.

        Args:
            X (_type_): Feature matrix for training (optional).
            y (_type_): Target vector for training (optional).
            parameter_space_size (_type_): Size of the parameter space for optimization.
        """
        global_params = global_parameters()
        self.X = X
        self.y = y

        # Use the standard AdaBoostClassifier directly
        self.algorithm_implementation = AdaBoostClassifier()
        self.method_name = "AdaBoostClassifier"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_params.bayessearch:
            # Define the parameter space for Bayesian optimization
            self.parameter_space = {
                "algorithm": Categorical(["SAMME.R", "SAMME"]),
                "learning_rate": Real(0.01, 1, prior="log-uniform"),
                "n_estimators": Integer(10, 500),
            }

            # Log parameter space for verification
            #print(f"Bayesian Parameter Space: {self.parameter_space}")

        else:
            # Define the parameter space for traditional grid search
            self.parameter_space = {
                "algorithm": ["SAMME.R", "SAMME"],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
                "n_estimators": [50, 100, 200, 500],
                "random_state": [None],
            }

            # Log parameter space for verification
            #print(f"Traditional Parameter Space: {self.parameter_space}")

        return None

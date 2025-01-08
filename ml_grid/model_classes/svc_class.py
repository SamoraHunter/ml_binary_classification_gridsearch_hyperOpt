from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skopt.space import Real, Categorical

print("Imported SVC class")


class SVC_class:
    """SVC with support for Bayesian and traditional grid search parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """Initialize SVC_class.

        Args:
            X (pd.DataFrame): DataFrame containing input features.
            y (pd.Series): Series containing target labels.
            parameter_space_size (int): Size of the parameter space.
        """
        self.X = X
        self.y = y

        # Enforce scaling for SVM method
        if not self.is_data_scaled():
            self.scale_data()

        self.algorithm_implementation = SVC()
        self.method_name = "SVC"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters().bayessearch:
            # Bayesian Optimization: Define parameter space using pre-defined schemes
            self.parameter_space = [{
                
                "C": self.parameter_vector_space.param_dict.get("log_small"),
                "break_ties": Categorical([False]),
                # 'cache_size': self.parameter_vector_space.param_dict.get("log_large"),  # Uncomment if needed
                # 'class_weight': self.parameter_vector_space.param_dict.get("enum_class_weights"),  # Example for enumerating class weights
                "coef0": self.parameter_vector_space.param_dict.get("log_small"),
                "decision_function_shape": Categorical(["ovo"]),
                "degree": self.parameter_vector_space.param_dict.get("log_med"),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["rbf", "linear", "poly", "sigmoid"]),
                "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                # 'probability': Categorical([True, False]),  # Uncomment if needed
                # 'random_state': Categorical([None]),  # Example for random state
                "shrinking": self.parameter_vector_space.param_dict.get("bool_param"),
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
                "verbose": Categorical([True, False]),
            },{
                
                "C": self.parameter_vector_space.param_dict.get("log_small"),
                "break_ties": Categorical([True, False]),
                # 'cache_size': self.parameter_vector_space.param_dict.get("log_large"),  # Uncomment if needed
                # 'class_weight': self.parameter_vector_space.param_dict.get("enum_class_weights"),  # Example for enumerating class weights
                "coef0": self.parameter_vector_space.param_dict.get("log_small"),
                "decision_function_shape": Categorical(["ovr"]),
                "degree": self.parameter_vector_space.param_dict.get("log_med"),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["rbf", "linear", "poly", "sigmoid"]),
                "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                # 'probability': Categorical([True, False]),  # Uncomment if needed
                # 'random_state': Categorical([None]),  # Example for random state
                "shrinking": self.parameter_vector_space.param_dict.get("bool_param"),
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
                "verbose": Categorical([True, False]),
            }
                                    
                                    ]
            
        else:
            # Traditional Grid Search: Define parameter space using lists
            self.parameter_space = {
                "C": list(self.parameter_vector_space.param_dict.get("log_small")),
                "break_ties": list(
                    self.parameter_vector_space.param_dict.get("bool_param")
                ),
                # 'cache_size': [200],  # Uncomment if needed
                # 'class_weight': [None, "balanced"]
                # + [{0: w} for w in [1, 2, 4, 6, 10]],  # Enumerate class weights
                "coef0": list(self.parameter_vector_space.param_dict.get("log_small")),
                "decision_function_shape": ["ovr", "ovo"],
                "degree": list(self.parameter_vector_space.param_dict.get("log_med")),
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
                "max_iter": list(
                    self.parameter_vector_space.param_dict.get("log_large_long")
                ),
                # 'probability': [False],  # Uncomment if needed
                # 'random_state': [None],  # Example for random state
                "shrinking": list(
                    self.parameter_vector_space.param_dict.get("bool_param")
                ),
                "tol": list(self.parameter_vector_space.param_dict.get("log_small")),
                "verbose": [False],
            }

        return None

    def is_data_scaled(self):
        """
        Check if data has been scaled to [0, 1] or [-1, 1] range.

        Returns:
            bool: True if data has been scaled, False if not.
        """
        # Calculate the range of values for each feature
        min_val = self.X.min().min()
        max_val = self.X.max().max()

        # Check if data is scaled to [0, 1] or [-1, 1] range
        if (min_val >= 0 and max_val <= 1) or (min_val >= -1 and max_val <= 1):
            return True
        else:
            return False

    def scale_data(self):
        """Scale the data to [0, 1] range using MinMaxScaler."""
        # Initialize MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform the data
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

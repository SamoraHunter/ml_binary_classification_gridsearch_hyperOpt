from scipy import sparse
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
            try:
                # Data validation checks before scaling
                if self.X is None:
                    raise ValueError("Input data X is None - data not loaded properly")
                    
                if isinstance(self.X, pd.DataFrame) and self.X.empty:
                    #raise ValueError("Input data X is an empty DataFrame")
                    print("warn: SVC data scaling, X data is empty")

                if( self.X.empty == False):
                    if not hasattr(self, 'scaler'):
                        self.scaler = StandardScaler()  # or whichever scaler you're using
                        
                    # Convert sparse matrices if needed
                    if sparse.issparse(self.X):
                        self.X = self.X.toarray()
                        
                    # Ensure numeric data
                    if isinstance(self.X, pd.DataFrame):
                        non_numeric = self.X.select_dtypes(exclude=['number']).columns
                        if len(non_numeric) > 0:
                            raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
                            
                    # Debug logging 
                    #print(f"Scaling data with shape: {self.X.shape}")
                    #print(f"Sample values before scaling:\n{self.X.iloc[:3,:3] if isinstance(self.X, pd.DataFrame) else self.X[:3,:3]}")
                    
                    # Perform scaling
                    self.X = pd.DataFrame(
                        self.scaler.fit_transform(self.X), 
                        columns=self.X.columns if hasattr(self.X, 'columns') else None,
                        index=self.X.index if hasattr(self.X, 'index') else None
                    )
                    
                    print("Data scaling completed successfully")
                
            except Exception as e:
                error_msg = f"Data scaling failed: {str(e)}"
                print(error_msg)
                
                # Additional debug info
                if hasattr(self, 'X'):
                    print(f"Data type: {type(self.X)}")
                    if hasattr(self.X, 'shape'):
                        print(f"Shape: {self.X.shape}")
                    if isinstance(self.X, pd.DataFrame):
                        print(f"Columns: {self.X.columns.tolist()}")
                        print(f"Data types:\n{self.X.dtypes}")
                        
                raise RuntimeError(error_msg) from e

        self.algorithm_implementation = SVC()
        self.method_name = "SVC"

        # Define the parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
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

from scipy import sparse
from typing import Optional
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from skopt.space import Real, Categorical

logging.getLogger('ml_grid').debug("Imported SVC class")


class SVC_class:
    """SVC with support for Bayesian and traditional grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the SVC_class.

        This class requires scaled data. If the input data `X` is not detected
        as scaled, it will be automatically scaled using `StandardScaler`.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training. Defaults to None.
            y (Optional[pd.Series]): Target vector for training. Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        self.X = X
        self.y = y

        # Enforce scaling for SVM method
        if not self.is_data_scaled():
            try:
                # Data validation checks before scaling
                if self.X is None:
                    raise ValueError("Input data X is None - data not loaded properly")
                    
                # If the dataframe is empty, there's nothing to scale.
                # The pipeline will likely fail later, but we avoid a scaling error here.
                if isinstance(self.X, pd.DataFrame) and self.X.empty:
                    raise ValueError("SVC_class received an empty DataFrame. Halting execution.")

                elif not self.X.empty:
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
                    
                    logging.getLogger('ml_grid').info("Data scaling completed successfully for SVC")
                
            except Exception as e:
                error_msg = f"Data scaling failed: {str(e)}"
                logging.getLogger('ml_grid').error(error_msg)
                
                # Additional debug info
                if hasattr(self, 'X'):
                    logging.getLogger('ml_grid').debug(f"Data type: {type(self.X)}")
                    if hasattr(self.X, 'shape'):
                        logging.getLogger('ml_grid').debug(f"Shape: {self.X.shape}")
                    if isinstance(self.X, pd.DataFrame):
                        logging.getLogger('ml_grid').debug(f"Columns: {self.X.columns.tolist()}")
                        logging.getLogger('ml_grid').debug(f"Data types:\n{self.X.dtypes}")
                        
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
            # Split into two dictionaries to handle the 'ovo' and 'break_ties' constraint.
            base_params = {
                "C": self.parameter_vector_space.param_dict.get("log_small"),
                "coef0": self.parameter_vector_space.param_dict.get("log_small"),
                "degree": self.parameter_vector_space.param_dict.get("log_med"),
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
                "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
                "shrinking": self.parameter_vector_space.param_dict.get("bool_param"),
                "tol": self.parameter_vector_space.param_dict.get("log_small"),
                "verbose": [False],
            }

            # Dictionary 1: For 'ovr', break_ties can be True or False
            params_ovr = base_params.copy()
            params_ovr.update({
                "decision_function_shape": ["ovr"],
                "break_ties": self.parameter_vector_space.param_dict.get("bool_param"),
            })

            # Dictionary 2: For 'ovo', break_ties MUST be False
            params_ovo = base_params.copy()
            params_ovo.update({
                "decision_function_shape": ["ovo"],
                "break_ties": [False],
            })

            # Convert all skopt spaces to lists for GridSearchCV
            for p in [params_ovr, params_ovo]:
                for k, v in p.items():
                    if not isinstance(v, list):
                        p[k] = list(v)

            self.parameter_space = [params_ovr, params_ovo]

        return None

    def is_data_scaled(self) -> bool:
        """Checks if the feature matrix `X` is scaled.

        This method determines if the data appears to be scaled by checking if all
        feature values fall within the [0, 1] or [-1, 1] range.

        Returns:
            bool: True if data appears to be scaled, False otherwise.
        """
        if self.X is None or self.X.empty:
            return False

        # Select only numeric columns for min/max checks
        numeric_X = self.X.select_dtypes(include="number")
        if numeric_X.empty:
            return False

        # Calculate the range of values for each feature
        min_val = numeric_X.min().min()
        max_val = numeric_X.max().max()

        # Check if data is scaled to [0, 1] or [-1, 1] range
        if (min_val >= 0 and max_val <= 1) or (min_val >= -1 and max_val <= 1):
            return True

        return False

    def scale_data(self) -> None:
        """Scales the feature matrix `X` using MinMaxScaler.
        """
        # Initialize MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform the data
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

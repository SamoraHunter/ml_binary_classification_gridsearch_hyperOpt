import numpy as np
import xgboost as xgb
from ml_grid.util import param_space
from skopt.space import Real, Categorical, Integer
from ml_grid.util.global_params import global_parameters

print("Imported XGB class")

class XGB_class_class:
    """xgb with support for Bayesian and Grid Search parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """Initialize XGBClassifier.

        Args:
            X (pd.DataFrame): DataFrame containing input features.
            y (pd.Series): Series containing target labels.
            parameter_space_size (int): Size of the parameter space.
        
        The XGB_class_class wraps the XGBoost classifier algorithm. The class
        allows for easy configuration and use within a grid search or Bayesian 
        optimization framework by setting up a parameter space that can be customized.
        """
        self.X = X
        self.y = y

        # Initialize the algorithm implementation using XGBClassifier
        self.algorithm_implementation = xgb.XGBClassifier()
        self.method_name = "XGBClassifier"

        # Initialize the parameter space handler
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        
        # Patch max_bin dynamically for compatibility
        def patch_max_bin(param_value):
            """
            Ensure max_bin parameter is >= 2.

            Args:
                param_value: The original parameter value.

            Returns:
                Patched parameter value.
            """
            if isinstance(param_value, int):
                return max(2, param_value)
            elif hasattr(param_value, "rvs"):  # For sampled values (e.g., skopt spaces)
                return Real(2, param_value.high, prior=param_value.prior) if isinstance(param_value, Real) else Integer(2, param_value.high)
            else:
                return param_value

        # Set up the parameter space based on the selected optimization method
        if global_parameters.bayessearch:
            # Bayesian Optimization: Define parameter space using Real and Categorical
            self.parameter_space = [{
                "objective": Categorical(["binary:logistic"]),  # Objective function for binary classification
                "booster": Categorical(["gbtree", "gblinear", "dart"]),  # Type of boosting model
                "gamma": self.parameter_vector_space.param_dict.get("log_small"),  # Regularization parameter
                "grow_policy": Categorical(["depthwise", "lossguide"]),  # Tree growth policy
                "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),  # Learning rate
                "max_bin": patch_max_bin(self.parameter_vector_space.param_dict.get("log_large_long")),  # Max bins for discretization
                "max_depth": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max depth of tree
                "max_leaves": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max number of leaves
                "min_child_weight": self.parameter_vector_space.param_dict.get("log_small"),  # Minimum sum of instance weight in a child
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),  # Number of boosting rounds
                "n_jobs": Categorical([-1]),  # Number of parallel threads to use for training
                "random_state": Categorical([None]),  # Random state for reproducibility
                "reg_alpha": self.parameter_vector_space.param_dict.get("log_small"),  # L1 regularization term
                "reg_lambda": self.parameter_vector_space.param_dict.get("log_small"),  # L2 regularization term
                "sampling_method": Categorical(["uniform"]),  # Sampling method during training
                "verbosity": Categorical([0]),  # Verbosity level during training
                "tree_method": Categorical(["auto"])
                

                },{
                    "objective": Categorical(["binary:logistic"]),  # Objective function for binary classification
                "booster": Categorical(["gbtree", "gblinear", "dart"]),  # Type of boosting model
                "gamma": self.parameter_vector_space.param_dict.get("log_small"),  # Regularization parameter
                "grow_policy": Categorical(["depthwise", "lossguide"]),  # Tree growth policy
                "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),  # Learning rate
                "max_bin": patch_max_bin(self.parameter_vector_space.param_dict.get("log_large_long")),  # Max bins for discretization
                "max_depth": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max depth of tree
                "max_leaves": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max number of leaves
                "min_child_weight": self.parameter_vector_space.param_dict.get("log_small"),  # Minimum sum of instance weight in a child
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),  # Number of boosting rounds
                "n_jobs": Categorical([-1]),  # Number of parallel threads to use for training
                "random_state": Categorical([None]),  # Random state for reproducibility
                "reg_alpha": self.parameter_vector_space.param_dict.get("log_small"),  # L1 regularization term
                "reg_lambda": self.parameter_vector_space.param_dict.get("log_small"),  # L2 regularization term
                "sampling_method": Categorical(["uniform"]),  # Sampling method during training
                "verbosity": Categorical([0]),  # Verbosity level during training
                "tree_method": Categorical(["auto"])

                #value 1 for Parameter max_bin should be greater equal to 2
                #max_bin: if using histogram-based algorithm, maximum number of bins per feature
                    
                    
                }]


                # Future use parameters for Bayesian optimization
                # "use_label_encoder": Categorical([True, False]),  # Use label encoder
                # "base_score": Real(0.0, 1.0, "uniform"),  # Base score for predictions
                # "callbacks": [None],  # Custom callbacks for training
                # "colsample_bylevel": Real(0.5, 1, "uniform"),  # Column sampling by level
                # "colsample_bynode": Real(0.5, 1, "uniform"),  # Column sampling by node
                # "colsample_bytree": Real(0.5, 1, "uniform"),  # Column sampling by tree
                # "early_stopping_rounds": Categorical([None]),  # Early stopping for boosting rounds
                # "enable_categorical": Categorical([True, False]),  # Enable categorical variables (if needed)
                # "eval_metric": Categorical([None]),  # Evaluation metric (optional)
                # "gpu_id": Categorical([None]),  # GPU id to use for training
                # "importance_type": Categorical(["weight", "gain", "cover"]),  # Type of feature importance calculation
                # "interaction_constraints": Categorical([None]),  # Constraints for feature interaction
                # "max_cat_to_onehot": Real(1, 100, "uniform"),  # Max categories for one-hot encoding
                # "max_delta_step": Real(0, 10, "uniform"),  # Max delta step for optimization
                # "monotone_constraints": Categorical([None]),  # Constraints for monotonicity in predictions
                # "num_parallel_tree": Real(1, 10, "uniform"),  # Number of parallel trees in boosting
                # "predictor": Categorical(["cpu_predictor", "gpu_predictor"]),  # Type of predictor (e.g., 'gpu_predictor')
                # "scale_pos_weight": Real(1, 10, "uniform"),  # Scale weight for positive class
                # "subsample": Real(0.5, 1, "uniform"),  # Subsampling ratio for training
                # "tree_method": Categorical(["auto", "gpu_hist", "hist"]),  # Tree method for GPU (optional)
                # "validate_parameters": Categorical([None]),  # Validate parameters during training
            
        else:
            # Traditional Grid Search: Define parameter space using lists
            self.parameter_space = {
                "objective": ["binary:logistic"],  # Objective function for binary classification
                "booster": ["gbtree", "gblinear", "dart"],  # Type of boosting model
                "gamma": self.parameter_vector_space.param_dict.get("log_small"),  # Regularization parameter
                "grow_policy": ["depthwise", "lossguide"],  # Tree growth policy
                "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),  # Learning rate
                "max_bin": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max bins for discretization
                "max_depth": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max depth of tree
                "max_leaves": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max number of leaves
                "min_child_weight": [None],  # Minimum sum of instance weight in a child
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),  # Number of boosting rounds
                "n_jobs": [-1],  # Number of parallel threads for training
                "random_state": [None],  # Random state for reproducibility
                "reg_alpha": self.parameter_vector_space.param_dict.get("log_small"),  # L1 regularization term
                "reg_lambda": self.parameter_vector_space.param_dict.get("log_small"),  # L2 regularization term
                "sampling_method": ["uniform"],  # Sampling method during training
                "verbosity": [0],  # Verbosity level during training

                # Future use parameters for Grid Search (currently commented out)
                # "use_label_encoder": [True, False],  # Use label encoder
                # "base_score": [0.0, 1.0],  # Base score for predictions
                # "callbacks": [None],  # Custom callbacks for training
                # "colsample_bylevel": [0.5, 1],  # Column sampling by level
                # "colsample_bynode": [0.5, 1],  # Column sampling by node
                # "colsample_bytree": [0.5, 1],  # Column sampling by tree
                # "early_stopping_rounds": [None],  # Early stopping for boosting rounds
                # "enable_categorical": [True, False],  # Enable categorical variables (if needed)
                # "eval_metric": [None],  # Evaluation metric (optional)
                # "gpu_id": [None],  # GPU id to use for training
                # "importance_type": ["weight", "gain", "cover"],  # Type of feature importance calculation
                # "interaction_constraints": [None],  # Constraints for feature interaction
                # "max_cat_to_onehot": [1, 100],  # Max categories for one-hot encoding
                # "max_delta_step": [0, 10],  # Max delta step for optimization
                # "monotone_constraints": [None],  # Constraints for monotonicity in predictions
                # "num_parallel_tree": [1, 10],  # Number of parallel trees in boosting
                # "predictor": ["cpu_predictor", "gpu_predictor"],  # Type of predictor (e.g., 'gpu_predictor')
                # "scale_pos_weight": [1, 10],  # Scale weight for positive class
                # "subsample": [0.5, 1],  # Subsampling ratio for training
                # "tree_method": ["auto", "gpu_hist", "hist"],  # Tree method for GPU (optional)
                # "validate_parameters": [None],  # Validate parameters during training
            }

        return None

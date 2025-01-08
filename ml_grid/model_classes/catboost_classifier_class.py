import numpy as np
from catboost import CatBoostClassifier
from skopt.space import Categorical, Real, Integer
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters

class CatBoost_class:
    """CatBoost Classifier with hyperparameter tuning."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize the CatBoost_class.

        Args:
            X (_type_): Feature matrix for training (optional).
            y (_type_): Target vector for training (optional).
            parameter_space_size (_type_): Size of the parameter space for optimization.
        """
        global_params = global_parameters()  # Fetch global parameters
        self.X = X
        self.y = y

        # Use CatBoostClassifier directly
        self.algorithm_implementation = CatBoostClassifier()
        self.method_name = "CatBoostClassifier"

        # Initialize parameter vector space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        # Define parameter space for Bayesian search or traditional grid search
        if global_params.bayessearch:
            self.parameter_space = {
                "iterations": Integer(100, 1000),
                "learning_rate": Real(0.01, 0.3, prior="uniform"),
                "depth": Integer(4, 10),
                "l2_leaf_reg": Real(1e-5, 1, prior="log-uniform"),
                "random_strength": Real(1e-5, 1, prior="log-uniform"),
                "rsm": Real(0.8, 1, prior="uniform"),
                "loss_function": Categorical(["Logloss", "CrossEntropy"]),
                "eval_metric": Categorical(["Accuracy", "AUC"]),
                "bootstrap_type": Categorical(["Bernoulli", "MVS"]),
                "subsample": Real(0.8, 1, prior="uniform"),
                "max_bin": Integer(32, 128),
                "grow_policy": Categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
                "min_data_in_leaf": Integer(1, 7),
                "one_hot_max_size": Integer(2, 10),
                "leaf_estimation_method": Categorical(["Newton", "Gradient"]),
                "fold_permutation_block": Integer(1, 5),
                "od_pval": Real(1e-9, 0.1, prior="log-uniform"),
                "od_wait": Integer(10, 30),
                "verbose": Categorical([0]),
                "allow_const_label": Categorical([True]),
            }
            print(f"Bayesian Parameter Space: {self.parameter_space}")
        else:
            self.parameter_space = {
                "iterations": [100, 200, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.3],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1e-5, 1e-3, 0.1, 1],
                "random_strength": [1e-5, 1e-3, 0.1, 1],
                "rsm": [0.8, 1],
                "loss_function": ["Logloss", "CrossEntropy"],
                "eval_metric": ["Accuracy", "AUC"],
                "bootstrap_type": ["Bernoulli", "MVS"],
                "subsample": [0.8, 1],
                "max_bin": [32, 64, 128],
                "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
                "min_data_in_leaf": [1, 3, 5, 7],
                "one_hot_max_size": [2, 5, 10],
                "leaf_estimation_method": ["Newton", "Gradient"],
                "fold_permutation_block": [1, 3, 5],
                "od_pval": [1e-9, 1e-7, 1e-5, 1e-3],
                "od_wait": [10, 20, 30],
                "verbose": [0],
                "allow_const_label": [True],
            }
            print(f"Traditional Parameter Space: {self.parameter_space}")

        return None

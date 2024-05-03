from catboost import CatBoostClassifier

from ml_grid.util import param_space

print("Imported CatBoostClassifier class")


class CatBoost_class:
    """CatBoost classifier."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """
        Initialize CatBoost classifier.

        Args:
            X (_type_): Description of input features.
            y (_type_): Description of target variable.
            parameter_space_size (_type_): Size of parameter space.
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = CatBoostClassifier()
        self.method_name = "CatBoostClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = [
            {
                # General Parameters
                "iterations": [100, 500, 1000],  # Number of trees to grow
                "learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.3,
                ],  # Shrinkage rate for tree updates
                "depth": [4, 6, 8, 10],  # Depth of the tree
                # Tree-Boosting Parameters
                # "bagging_temperature": [
                #     0,
                #     1,
                #     5,
                #     10,
                # ],  # Controls intensity of random selection of data used for training
                # "border_count": [32, 64, 128],  # Number of splits for numerical features
                "l2_leaf_reg": [3, 5, 7, 9],  # L2 regularization coefficient
                "random_strength": [
                    1,
                    2,
                    3,
                ],  # Magnitude of randomness in data sampling
                "rsm": [0.8, 1],  # Random selection of features for each tree
                # Learning Task Parameters
                "loss_function": ["Logloss", "CrossEntropy"],  # Loss function
                "eval_metric": ["Accuracy", "AUC"],  # Metric to use for validation
                #           "custom_metric": [],  # User-defined custom metric
                # "auto_class_weights": ["None", "Balanced"],  # Handling class imbalance
                # Optimization Parameters
                "bootstrap_type": [
                    "Bernoulli",
                    "MVS",
                    #    "Poisson",
                ],  # Sampling strategy
                "subsample": [0.8, 1],  # Sample rate for bootstrap type
                "max_bin": [
                    32,
                    64,
                    128,
                ],  # Maximum number of bins to use for numerical features
                "grow_policy": [
                    "SymmetricTree",
                    "Depthwise",
                    "Lossguide",
                ],  # Tree growing strategy
                "min_data_in_leaf": [1, 3, 5, 7],  # Minimum number of samples in a leaf
                # "max_leaves": [31, 63, 127],  # Maximum number of leaves in a tree
                # "num_boost_round": [10, 50, 100],  # Number of boosting rounds
                # Others
                #            "ignored_features": [[]],  # Features to ignore
                "one_hot_max_size": [2, 5, 10],  # Maximum size of one-hot encoding
                "leaf_estimation_method": [
                    "Newton",
                    "Gradient",
                ],  # Method for leaf value calculation
                "feature_border_type": [
                    "MinEntropy",
                    "MaxLogSum",
                ],  # Method to use for splitting numerical features
                # CatBoost Extensions
                "bayesian_matrix_reg": [
                    0.1,
                    0.5,
                    1,
                ],  # Bayesian regularization coefficient
                "fold_permutation_block": [1, 3, 5],  # Fold permutation block size
                "od_pval": [0.01, 0.05, 0.1],  # P-value threshold for early stopping
                "od_wait": [
                    10,
                    20,
                    30,
                ],  # Number of iterations to wait before early stopping
                "verbose": [0],
            },
            {
                # General Parameters
                # "iterations": [100, 500, 1000],  # Number of trees to grow
                "learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.3,
                ],  # Shrinkage rate for tree updates
                "depth": [4, 6, 8, 10],  # Depth of the tree
                # Tree-Boosting Parameters
                # "border_count": [32, 64, 128],  # Number of splits for numerical features
                "l2_leaf_reg": [3, 5, 7, 9],  # L2 regularization coefficient
                "random_strength": [
                    1,
                    2,
                    3,
                ],  # Magnitude of randomness in data sampling
                "rsm": [0.8, 1],  # Random selection of features for each tree
                # Learning Task Parameters
                "loss_function": ["Logloss", "CrossEntropy"],  # Loss function
                "eval_metric": ["Accuracy", "AUC"],  # Metric to use for validation
                #           "custom_metric": [],  # User-defined custom metric
                # "auto_class_weights": ["None", "Balanced"],  # Handling class imbalance
                # Optimization Parameters
                "bootstrap_type": [
                    "Bernoulli",
                    "MVS",
                    #    "Poisson",
                ],  # Sampling strategy
                "subsample": [0.8, 1],  # Sample rate for bootstrap type
                "max_bin": [
                    32,
                    64,
                    128,
                ],  # Maximum number of bins to use for numerical features
                "grow_policy": [
                    "SymmetricTree",
                    "Depthwise",
                    "Lossguide",
                ],  # Tree growing strategy
                "min_data_in_leaf": [1, 3, 5, 7],  # Minimum number of samples in a leaf
                #  "max_leaves": [31, 63, 127],  # Maximum number of leaves in a tree
                # "num_boost_round": [10, 50, 100],  # Number of boosting rounds
                # Others
                #            "ignored_features": [[]],  # Features to ignore
                "one_hot_max_size": [2, 5, 10],  # Maximum size of one-hot encoding
                "leaf_estimation_method": [
                    "Newton",
                    "Gradient",
                ],  # Method for leaf value calculation
                "feature_border_type": [
                    "MinEntropy",
                    "MaxLogSum",
                ],  # Method to use for splitting numerical features
                # CatBoost Extensions
                "bayesian_matrix_reg": [
                    0.1,
                    0.5,
                    1,
                ],  # Bayesian regularization coefficient
                "fold_permutation_block": [1, 3, 5],  # Fold permutation block size
                "od_pval": [0.01, 0.05, 0.1],  # P-value threshold for early stopping
                "od_wait": [
                    10,
                    20,
                    30,
                ],  # Number of iterations to wait before early stopping
                "verbose": [0],
            },
            {
                # General Parameters
                # "iterations": [100, 500, 1000],  # Number of trees to grow
                "learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.3,
                ],  # Shrinkage rate for tree updates
                "depth": [4, 6, 8, 10],  # Depth of the tree
                # Tree-Boosting Parameters
                # "border_count": [32, 64, 128],  # Number of splits for numerical features
                "bagging_temperature": [
                    0,
                    1,
                    5,
                    10,
                ],  # Controls intensity of random selection of data used for training
                "l2_leaf_reg": [3, 5, 7, 9],  # L2 regularization coefficient
                "random_strength": [
                    1,
                    2,
                    3,
                ],  # Magnitude of randomness in data sampling
                "rsm": [0.8, 1],  # Random selection of features for each tree
                # Learning Task Parameters
                "loss_function": ["Logloss", "CrossEntropy"],  # Loss function
                "eval_metric": ["Accuracy", "AUC"],  # Metric to use for validation
                #           "custom_metric": [],  # User-defined custom metric
                # "auto_class_weights": ["None", "Balanced"],  # Handling class imbalance
                # Optimization Parameters
                "bootstrap_type": [
                    "Bayesian",
                ],  # Sampling strategy
                # "subsample": [0.8, 1],  # Sample rate for bootstrap type
                "max_bin": [
                    32,
                    64,
                    128,
                ],  # Maximum number of bins to use for numerical features
                "grow_policy": [
                    "SymmetricTree",
                    "Depthwise",
                    "Lossguide",
                ],  # Tree growing strategy
                "min_data_in_leaf": [1, 3, 5, 7],  # Minimum number of samples in a leaf
                # "max_leaves": [31, 63, 127],  # Maximum number of leaves in a tree
                # "num_boost_round": [10, 50, 100],  # Number of boosting rounds
                # Others
                #            "ignored_features": [[]],  # Features to ignore
                "one_hot_max_size": [2, 5, 10],  # Maximum size of one-hot encoding
                "leaf_estimation_method": [
                    "Newton",
                    "Gradient",
                ],  # Method for leaf value calculation
                "feature_border_type": [
                    "MinEntropy",
                    "MaxLogSum",
                ],  # Method to use for splitting numerical features
                # CatBoost Extensions
                "bayesian_matrix_reg": [
                    0.1,
                    0.5,
                    1,
                ],  # Bayesian regularization coefficient
                "fold_permutation_block": [1, 3, 5],  # Fold permutation block size
                "od_pval": [0.01, 0.05, 0.1],  # P-value threshold for early stopping
                "od_wait": [
                    10,
                    20,
                    30,
                ],  # Number of iterations to wait before early stopping
                "verbose": [0],
            },
        ]

        return None

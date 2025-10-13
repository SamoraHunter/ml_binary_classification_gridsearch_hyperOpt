from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from ml_grid.model_classes.tabtransformerClassifier import TabTransformerClassifier
from ml_grid.util import param_space
from skopt.space import Real, Categorical
import logging

from ml_grid.util.global_params import global_parameters

logging.getLogger('ml_grid').debug("Imported TabTransformerClassifier class")

class TabTransformerWrapper(TabTransformerClassifier):
    """A wrapper for TabTransformerClassifier to handle tuple-based parameters.

    This is necessary for hyperparameter search libraries like BayesSearchCV
    that may not handle tuple parameters directly. It maps integer indices
    to predefined tuple values.
    """

    def set_params(self, **params: Any) -> "TabTransformerWrapper":
        """Sets the parameters of the estimator.

        This method intercepts integer-indexed parameters like 'categories' and
        'mlp_hidden_mults' and maps them to their corresponding tuple values
        before passing them to the parent's set_params method.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            TabTransformerWrapper: The instance with updated parameters.
        """
        tuple_mapping = {
            "categories": [
                (10, 5, 6, 5, 8),
                (15, 7, 8, 6, 10),
                (12, 6, 7, 5, 9),
                # Add more tuples as needed
            ],
            "mlp_hidden_mults": [
                (4, 2),
                (3, 1),
                # Add more tuples as needed
            ],
        }

        # Replace indices with corresponding tuples
        for key, mapping in tuple_mapping.items():
            if key in params:
                index = params.pop(key)
                if isinstance(index, int) and 0 <= index < len(mapping):
                    params[key] = mapping[index]

        return super().set_params(**params)


class TabTransformer_class:
    """TabTransformerClassifier with support for Bayesian and Grid Search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the TabTransformer_class.

        This class wraps the TabTransformerClassifier for use in hyperparameter
        search frameworks. It automatically determines the number of categories
        for categorical features and the number of continuous features from the
        input data `X`. It then sets up parameter spaces for both Bayesian search
        and grid search.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training. Defaults to None.
            y (Optional[pd.Series]): Target vector for training. Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.
        """
        self.X = X
        self.y = y

        # Split the DataFrame into categorical and continuous parts
        df_categ = X.select_dtypes(
            include=["object", "category"]
        )  # Select categorical columns
        df_cont = X.select_dtypes(
            include=["float64", "int64"]
        )  # Select continuous columns

        # Calculate the number of unique values within each category
        categories = tuple(df_categ[col].nunique() for col in df_categ.columns)

        # Calculate the number of continuous columns
        num_continuous = df_cont.shape[1]

        # Print the results
        logging.getLogger('ml_grid').info(f"TabTransformer: Number of unique values within each category: {categories}")
        logging.getLogger('ml_grid').info(f"TabTransformer: Number of continuous columns: {num_continuous}")

        self.method_name = "TabTransformerClassifier"

        # Algorithm Implementation
        # self.algorithm_implementation = TabTransformerClassifier(categories, num_continuous)

        if global_parameters.bayessearch is False:
            self.algorithm_implementation = TabTransformerClassifier(categories, num_continuous)
        else:
            self.algorithm_implementation = TabTransformerWrapper(
                categories, num_continuous
            )  # Wrapper necessary for passing priors to bayescv

        # Parameter Space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters.bayessearch:
            # Bayesian Optimization: Define parameter space using Real and Categorical
            self.parameter_space = {
                "categories": Categorical([0, 1, 2]),  # Indices for the tuple mapping
                "num_continuous": Real(1, 10),  # Number of continuous features
                "dim": Real(1, 32),  # Dimensionality of token embeddings
                "dim_out": Real(0, 1),  # Output dimensionality
                "depth": Real(2, 6),  # Number of transformer layers
                "heads": Real(2, 8),  # Number of attention heads
                "attn_dropout": Real(0.0, 0.5),  # Dropout rate for attention layers
                "ff_dropout": Real(0.0, 0.5),  # Dropout rate for feedforward layers
                "mlp_hidden_mults": Categorical(
                    [0, 1]
                ),  # Indices for the tuple mapping  # Multipliers for hidden layer dimensions
                "mlp_act": Categorical(["ReLU"]),  # Activation function as string
                "continuous_mean_std": Categorical([None]),  # Handle tensor creation later
            }

        else:
            # Traditional Grid Search: Define parameter space using lists
            self.parameter_space = {
                "categories": [(10, 5, 6, 5, 8)],  # Example categories: Tuple of category counts
                "num_continuous": [10],  # Number of continuous features
                "dim": [32],  # Dimensionality of token embeddings
                "dim_out": [1],  # Output dimensionality
                "depth": [6],  # Number of transformer layers
                "heads": [8],  # Number of attention heads
                "attn_dropout": [0.1],  # Dropout rate for attention layers
                "ff_dropout": [0.1],  # Dropout rate for feedforward layers
                "mlp_hidden_mults": [(4, 2)],  # Multipliers for hidden layer dimensions in the MLP
                "mlp_act": [nn.ReLU()],  # Activation function for the MLP
                "continuous_mean_std": [torch.randn(10, 2)],  # Mean and std of continuous features
            }

        return None

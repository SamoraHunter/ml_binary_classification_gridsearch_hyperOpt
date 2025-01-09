import torch
import torch.nn as nn
from ml_grid.model_classes.tabtransformerClassifier import TabTransformerClassifier
from ml_grid.util import param_space
from skopt.space import Real, Categorical

from ml_grid.util.global_params import global_parameters

print("Imported TabTransformerClassifier class")

class TabTransformer_class:
    """TabTransformerClassifier with support for Bayesian and Grid Search parameter spaces."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """Initialize TabTransformerClassifier.

        Args:
            X (pd.DataFrame): DataFrame containing input features.
            y (pd.Series): Series containing target labels.
            parameter_space_size (int): Size of the parameter space.

        The TabTransformerClassifier class is a wrapper for the TabTransformerClassifier
        algorithm found in the tab_transformer_pytorch library. It is designed to be
        used with the ml_grid search functions and takes in pandas DataFrames and
        Series objects as input.

        This function initializes the class by setting instance variables for the input
        data (X) and target labels (y), and initializing the TabTransformerClassifier
        algorithm implementation. It also sets up a parameter space for the class by
        creating a ml_grid.util.param_space.ParamSpace object with the specified size
        and populating it with valid parameter combinations.

        The parameter space for the TabTransformerClassifier is set up as follows:

        - categories (tuple of ints): Tuple of category counts for each categorical
          feature. For example, tuple(10, 5, 6, 5, 8) would indicate that there are
          10 categories in the 1st categorical feature, 5 categories in the 2nd
          categorical feature, etc.

        - num_continuous (int): Number of continuous features in the input data.

        - dim (int): Dimensionality of token embeddings.

        - dim_out (int): Output dimensionality.

        - depth (int): Number of transformer layers.

        - heads (int): Number of attention heads.

        - attn_dropout (float): Dropout rate for attention layers.

        - ff_dropout (float): Dropout rate for feedforward layers.

        - mlp_hidden_mults (tuple of ints): Multipliers for hidden layer dimensions
          in the MLP. For example, tuple(4, 2) would indicate that the first hidden
          layer should have 4 times the number of units as the input and the second
          hidden layer should have 2 times the number of units as the first hidden
          layer.

        - mlp_act (torch.nn.Module): Activation function for the MLP.

        - continuous_mean_std (torch.Tensor): Mean and standard deviation of
          continuous features. This is a tensor with shape (num_continuous, 2), where
          the first column is the mean and the second column is the standard deviation.
        """
        self.X = X
        self.y = y

        # Split the DataFrame into categorical and continuous parts
        df_categ = X.select_dtypes(include=['object', 'category'])  # Select categorical columns
        df_cont = X.select_dtypes(include=['float64', 'int64'])     # Select continuous columns

        # Calculate the number of unique values within each category
        categories = tuple(df_categ[col].nunique() for col in df_categ.columns)

        # Calculate the number of continuous columns
        num_continuous = df_cont.shape[1]

        # Print the results
        print("Number of unique values within each category:", categories)
        print("Number of continuous columns:", num_continuous)

        self.method_name = "TabTransformerClassifier"

        # Algorithm Implementation
        self.algorithm_implementation = TabTransformerClassifier(categories, num_continuous)

        # Parameter Space
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)

        if global_parameters().bayessearch:
            # Bayesian Optimization: Define parameter space using Real and Categorical
            self.parameter_space = {
                "categories": Categorical([(10, 5, 6, 5, 8)]),  # Example categories: Tuple of category counts
                "num_continuous": Real(1, 10),  # Number of continuous features
                "dim": Real(1, 32),  # Dimensionality of token embeddings
                "dim_out": Real(0, 1),  # Output dimensionality
                "depth": Real(2, 6),  # Number of transformer layers
                "heads": Real(2, 8),  # Number of attention heads
                "attn_dropout": Real(0.0, 0.5),  # Dropout rate for attention layers
                "ff_dropout": Real(0.0, 0.5),  # Dropout rate for feedforward layers
                "mlp_hidden_mults": Categorical([(4, 2)]),  # Multipliers for hidden layer dimensions in the MLP
                "mlp_act": Categorical([nn.ReLU()]),  # Activation function for the MLP
                "continuous_mean_std": Categorical([torch.randn(10, 2)]),  # Mean and std of continuous features
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

import torch
import torch.nn as nn
from ml_grid.model_classes.tabtransformerClassifier import TabTransformerClassifier
from ml_grid.util import param_space

print("Imported TabTransformerClassifier class")

class TabTransformer_class:
    """TabTransformerClassifier."""

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
        
        self.parameter_space = {
            "categories": [
                # Example categories: Tuple of category counts for each categorical feature
                (10, 5, 6, 5, 8)
            ],
            "num_continuous": [
                # Example num_continuous: Number of continuous features
                10
            ],
            "dim": [
                # Example dim: Dimensionality of token embeddings
                32
            ],
            "dim_out": [
                # Example dim_out: Output dimensionality
                1
            ],
            "depth": [
                # Example depth: Number of transformer layers
                6
            ],
            "heads": [
                # Example heads: Number of attention heads
                8
            ],
            "attn_dropout": [
                # Example attn_dropout: Dropout rate for attention layers
                0.1
            ],
            "ff_dropout": [
                # Example ff_dropout: Dropout rate for feedforward layers
                0.1
            ],
            "mlp_hidden_mults": [
                # Example mlp_hidden_mults: Multipliers for hidden layer dimensions in the MLP
                (4, 2)
            ],
            "mlp_act": [
                # Example mlp_act: Activation function for the MLP
                nn.ReLU()
            ],
            "continuous_mean_std": [
                # Example continuous_mean_std: Mean and standard deviation of continuous features
                torch.randn(10, 2)
            ]
        }


        return None


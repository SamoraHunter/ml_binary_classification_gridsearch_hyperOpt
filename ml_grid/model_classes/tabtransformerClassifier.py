"""TabTransformer Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for the TabTransformer model.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from sklearn.base import BaseEstimator, ClassifierMixin


class TabTransformerClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for the TabTransformer model.

    This class wraps the `TabTransformer` from the `tab-transformer-pytorch`
    library to make it compatible with the scikit-learn API.

    Note:
        This wrapper's `fit` method is a no-op. The model is intended to be
        trained in a standard PyTorch training loop. This wrapper is primarily
        for inference and integration with scikit-learn's evaluation tools.
    """

    def __init__(
        self,
        categories: Tuple[int, ...],
        num_continuous: int,
        dim: int = 32,
        dim_out: int = 1,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_mults: Tuple[int, ...] = (4, 2),
        mlp_act: Optional[nn.Module] = None,
        continuous_mean_std: Optional[torch.Tensor] = None,
    ):
        """Initializes the TabTransformerClassifier.

        Args:
            categories (Tuple[int, ...]): A tuple containing the number of unique
                categories for each categorical feature.
            num_continuous (int): The number of continuous features.
            dim (int): The dimension of embeddings.
            dim_out (int): The output dimension of the model.
            depth (int): The number of transformer layers.
            heads (int): The number of attention heads.
            attn_dropout (float): Dropout rate for attention layers.
            ff_dropout (float): Dropout rate for the feed-forward network.
            mlp_hidden_mults (Tuple[int, ...]): A tuple defining the multipliers
                for the hidden layers of the MLP.
            mlp_act (Optional[nn.Module]): The activation function for the MLP.
                Defaults to nn.ReLU().
            continuous_mean_std (Optional[torch.Tensor]): A tensor of shape
                (num_continuous, 2) for normalizing continuous features.
                Defaults to None.
        """
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        self.dim_out = dim_out
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_act = mlp_act if mlp_act is not None else nn.ReLU()
        self.continuous_mean_std = continuous_mean_std

        self.model = TabTransformer(
            categories=self.categories,
            num_continuous=self.num_continuous,
            dim=self.dim,
            dim_out=self.dim_out,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            mlp_hidden_mults=self.mlp_hidden_mults,
            mlp_act=self.mlp_act,
            continuous_mean_std=self.continuous_mean_std,
        )

    def fit(self, X: Any, y: Any) -> "TabTransformerClassifier":
        """A no-op fit method to comply with the scikit-learn API.

        Args:
            X (Any): Ignored.
            y (Any): Ignored.

        Returns:
            TabTransformerClassifier: The instance itself.
        """
        return self

    def predict_proba(self, X: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        Args:
            X (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the
                categorical features tensor and the continuous features tensor.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        self.model.eval()  # type: ignore
        with torch.no_grad():
            x_categ, x_cont = X
            pred = self.model(x_categ, x_cont)
            return torch.sigmoid(pred).numpy()

    def predict(self, X: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the
                categorical features tensor and the continuous features tensor.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

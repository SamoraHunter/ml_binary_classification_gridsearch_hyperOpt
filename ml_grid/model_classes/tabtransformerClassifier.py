import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from sklearn.base import BaseEstimator, ClassifierMixin

class TabTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, categories, num_continuous, dim=32, dim_out=1, depth=6, heads=8,
                 attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_mults=(4, 2),
                 mlp_act=nn.ReLU(), continuous_mean_std=None):
        self.categories = categories
        self.num_continuous = num_continuous
        self.dim = dim
        self.dim_out = dim_out
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_act = mlp_act
        self.continuous_mean_std = continuous_mean_std
        
        self.model = TabTransformer(categories=categories, num_continuous=num_continuous, dim=dim, dim_out=dim_out,
                                    depth=depth, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                    mlp_hidden_mults=mlp_hidden_mults, mlp_act=mlp_act,
                                    continuous_mean_std=continuous_mean_std)

    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            x_categ, x_cont = X
            pred = self.model(x_categ, x_cont)
            return torch.sigmoid(pred).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

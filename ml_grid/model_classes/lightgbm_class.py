import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import re
import pandas


class LightGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        feature_fraction=0.9,
        early_stopping_rounds=None,
    ):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.num_class = num_class
        self.metric = metric
        self.feature_fraction = feature_fraction
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective=self.objective,
            num_class=self.num_class,
            metric=self.metric,
            feature_fraction=self.feature_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        # X.columns = X.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
        new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X.columns}
        new_n_list = list(new_names.values())
        # [LightGBM] Feature appears more than one time.
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }
        X = X.rename(columns=new_names)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' before 'predict'."
            )

        # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
        new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in X.columns}
        new_n_list = list(new_names.values())
        # [LightGBM] Feature appears more than one time.
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }
        X = X.rename(columns=new_names)

        return self.model.predict(X)

    def score(self, X, y):
        if self.model is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' before 'score'."
            )
        return self.model.score(X, y)

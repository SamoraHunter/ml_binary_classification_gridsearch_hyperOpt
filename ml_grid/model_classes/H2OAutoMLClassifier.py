import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class H2OAutoMLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_runtime_secs=360, nfolds=2, seed=1):
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed
        self.automl = None

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # outcome_var = y.columns[0]
        try:
            outcome_var = y.columns[0]
        except:

            outcome_var = y.name

        x = list(X.columns)
        y_n = outcome_var
        try:
            x.remove(y_n)
        except:
            pass

        h2o.init()
        # train_df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)

        train_df = pd.concat([X, y], axis=1)

        train_h2o = h2o.H2OFrame(train_df)

        train_h2o[y_n] = train_h2o[y_n].asfactor()

        self.automl = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            max_models=5,
            nfolds=self.nfolds,
            seed=self.seed,
        )

        self.automl.train(y=y_n, x=x, training_frame=train_h2o)
        return self

    def predict(self, X):
        check_is_fitted(self)
        # X = check_array(X)
        # test_h2o = h2o.H2OFrame(pd.DataFrame(X))
        test_h2o = h2o.H2OFrame(X)
        predictions = self.automl.leader.predict(test_h2o)

        # return predictions[:,0]
        return predictions["predict"].as_data_frame().values

    def predict_proba(self, X):
        raise NotImplementedError("H2O AutoML does not support predict_proba.")

    def get_params(self, deep=True):

        return {
            "max_runtime_secs": self.max_runtime_secs,
            "nfolds": self.nfolds,
            "seed": self.seed,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_leader_params(
        self,
    ):
        return self.automl.leader.params
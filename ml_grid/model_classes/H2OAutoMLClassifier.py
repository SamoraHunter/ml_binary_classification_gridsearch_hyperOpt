from typing import Any, Dict, Optional

import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class H2OAutoMLClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for H2O's AutoML.

    This class allows H2O's AutoML to be used as a standard scikit-learn
    classifier, making it compatible with tools like GridSearchCV and
    BayesSearchCV.
    """

    def __init__(
        self, max_runtime_secs: int = 360, nfolds: int = 2, seed: int = 1
    ):
        """Initializes the H2OAutoMLClassifier.

        Args:
            max_runtime_secs (int): Maximum time in seconds to run the AutoML process.
            nfolds (int): Number of folds for cross-validation.
            seed (int): Random seed for reproducibility.
        """
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed
        self.automl: Optional[H2OAutoML] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OAutoMLClassifier":
        """Fits the H2O AutoML model.

        This method initializes an H2O cluster, converts the pandas DataFrame
        and Series to H2O Frames, and then trains the AutoML model.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values.

        Returns:
            H2OAutoMLClassifier: The fitted estimator.
        """
        self.classes_ = np.unique(y)

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (pd.DataFrame): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels.
        """
        check_is_fitted(self)
        test_h2o = h2o.H2OFrame(X)
        predictions = self.automl.leader.predict(test_h2o)

        return predictions["predict"].as_data_frame().values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        Note:
            This method is not implemented for H2O AutoML.

        Args:
            X (pd.DataFrame): The input samples.

        Raises:
            NotImplementedError: H2O AutoML does not support predict_proba.
        """
        raise NotImplementedError("H2O AutoML does not support predict_proba.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        return {
            "max_runtime_secs": self.max_runtime_secs,
            "nfolds": self.nfolds,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> "H2OAutoMLClassifier":
        """Sets the parameters of this estimator.

        Args:
            **params (Any): Estimator parameters.

        Returns:
            H2OAutoMLClassifier: The instance with updated parameters.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_leader_params(self) -> Dict[str, Any]:
        """Gets the parameters of the best model found by AutoML.

        Returns:
            Dict[str, Any]: A dictionary of the leader model's parameters.
        """
        check_is_fitted(self)
        return self.automl.leader.params
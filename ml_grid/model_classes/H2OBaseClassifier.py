from typing import Any, Dict, Optional
import inspect
import h2o
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class H2OBaseClassifier(BaseEstimator, ClassifierMixin):
    """A base class for scikit-learn compatible H2O classifier wrappers.

    This class provides common functionality for H2O model wrappers, including:
    - H2O cluster management (initialization and shutdown).
    - scikit-learn API compatibility (`get_params`, `set_params`).
    - Common `predict` and `predict_proba` implementations.
    - Robust handling of small datasets in the `fit` method.
    """

    def __init__(self, estimator_class, **kwargs):
        """Initializes the H2OBaseClassifier.

        Args:
            estimator_class: The H2O estimator class (e.g., H2OGradientBoostingEstimator).
            **kwargs: Keyword arguments passed to the H2O estimator.
        """
        self.estimator_class = estimator_class
        self.model_params = kwargs
        self.model: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self._is_cluster_owner = False
        self._was_fit_on_constant_feature = False
        self._using_dummy_model = False

    def _ensure_h2o_is_running(self):
        """Safely checks for and initializes an H2O cluster if not running."""
        cluster = h2o.cluster()
        if not (cluster and cluster.is_running()):
            h2o.init()
            self._is_cluster_owner = True

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """Prepares data and parameters for fitting."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")

        self.classes_ = np.unique(y)
        outcome_var = y.name
        x_vars = list(X.columns)

        self._ensure_h2o_is_running()

        train_df = pd.concat([X, y], axis=1)
        train_h2o = h2o.H2OFrame(train_df)
        train_h2o[outcome_var] = train_h2o[outcome_var].asfactor()

        model_params = self.model_params.copy()

        # If there's only one feature, prevent H2O from dropping it if it's constant.
        # Check if the estimator supports 'ignore_const_cols' before setting it.
        if len(x_vars) == 1 and self.estimator_class:
            # Track if we are in this problematic state
            if X[x_vars[0]].nunique() <= 1:
                self._was_fit_on_constant_feature = True

            estimator_params = inspect.signature(self.estimator_class).parameters
            if 'ignore_const_cols' in estimator_params:
                model_params.setdefault('ignore_const_cols', False)

        return train_h2o, x_vars, outcome_var, model_params

    def _handle_small_data_fallback(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Checks for small data and fits a dummy model if needed. Returns True if fallback was used."""
        # --- UNIVERSAL FALLBACK for very small data ---
        # Many H2O models become unstable or crash on predict with fewer than ~10 samples.
        min_samples_for_stable_fit = 10
        if len(X) < min_samples_for_stable_fit:
            print(
                f"Warning: Dataset is too small ({len(X)} rows). "
                f"Using a robust dummy model to prevent H2O server errors."
            )
            self._using_dummy_model = True
            # Use _prepare_fit to get the necessary H2OFrame and variables
            train_h2o, x_vars, outcome_var, _ = self._prepare_fit(X, y)
            from h2o.estimators import H2OGeneralizedLinearEstimator
            dummy_model = H2OGeneralizedLinearEstimator(family='binomial', ignore_const_cols=False)
            dummy_model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self.model = dummy_model
            return True
        return False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OBaseClassifier":
        """Fits the H2O model."""
        if self.estimator_class is not None and self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)
        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for samples in X."""
        check_is_fitted(self)

        # If the model was trained on a single constant feature, its predictions
        # are meaningless and can crash the server. Return a dummy prediction.
        if self._was_fit_on_constant_feature or self._using_dummy_model:
            # Predict the first class for all samples
            return np.full(len(X), self.classes_[0])

        # Final safety check: if model is None for any reason, return dummy prediction
        if self.model is None:
            print("Warning: self.model is None in predict(). Returning dummy prediction.")
            return np.full(len(X), self.classes_[0])

        self._ensure_h2o_is_running()
        test_h2o = h2o.H2OFrame(X)
        predictions = self.model.predict(test_h2o)
        return predictions["predict"].as_data_frame().values.ravel()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for samples in X."""
        check_is_fitted(self)

        # If the model was trained on a single constant feature, return a
        # dummy probability distribution (e.g., 100% for the first class).
        if self._was_fit_on_constant_feature or self._using_dummy_model:
            n_samples = len(X)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            proba[:, 0] = 1.0  # 100% probability for the first class
            return proba

        # Final safety check: if model is None for any reason, return dummy prediction
        if self.model is None:
            print("Warning: self.model is None in predict_proba(). Returning dummy prediction.")
            n_samples = len(X)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            proba[:, 0] = 1.0
            return proba
        self._ensure_h2o_is_running()
        test_h2o = h2o.H2OFrame(X)
        predictions = self.model.predict(test_h2o)
        prob_df = predictions.drop("predict").as_data_frame()

        return prob_df.values

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator."""
        return self.model_params

    def set_params(self, **params: Any) -> "H2OBaseClassifier":
        """Sets the parameters of this estimator."""
        self.model_params.update(params)
        return self

    def shutdown(self):
        """Shuts down the H2O cluster if this instance started it."""
        try:
            if self._is_cluster_owner:
                cluster = h2o.cluster()
                # Only shut down if this is the last client connected.
                # This is safer in environments where multiple processes might share a cluster.
                if cluster and cluster.is_running() and len(cluster.nodes) > 0 and cluster.nodes[0].healthy and cluster.nodes[0].num_cpus > 0 and cluster.nodes[0].proxy_connections == 1:
                    self.ml_grid_object.logger.info("This is the last client. Shutting down H2O cluster.")
                    cluster.shutdown()
                    self._is_cluster_owner = False
                else:
                    self.ml_grid_object.logger.info("Not shutting down H2O cluster as other clients may be connected.")
        except Exception:
            pass
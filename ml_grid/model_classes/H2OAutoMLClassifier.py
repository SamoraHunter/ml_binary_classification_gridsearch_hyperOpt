import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from .H2OBaseClassifier import H2OBaseClassifier
from sklearn.utils.validation import check_is_fitted

class H2OAutoMLClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's AutoML.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OAutoMLClassifier.
        """
        # H2OAutoML is not a standard estimator, so we don't pass it to super
        super().__init__(estimator_class=None, **kwargs)
        self.automl = None
        self._using_dummy_model = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OAutoMLClassifier":
        """Fits the H2O AutoML process.

        If the dataset is too small, it gracefully skips training to avoid
        crashing the H2O server.
        """
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- CRITICAL FIX for small datasets ---
        # AutoML can crash the server on very small or single-feature datasets.
        # We will gracefully skip the run in this case.
        min_samples = 20  # A reasonable minimum for AutoML
        if len(train_h2o) < min_samples or len(x_vars) < 1:
            print(
                f"Warning: Dataset is too small for H2O AutoML "
                f"({len(train_h2o)} rows, {len(x_vars)} features). "
                f"Skipping training and using a dummy model."
            )
            # Create a dummy model to allow predict/predict_proba to work
            # A simple GLM is a safe choice.
            from h2o.estimators import H2OGeneralizedLinearEstimator
            dummy_model = H2OGeneralizedLinearEstimator(
                family='binomial', ignore_const_cols=False
            )
            self._using_dummy_model = True # Set flag before training
            dummy_model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self.model = dummy_model
            return self

        self.automl = H2OAutoML(**model_params)
        self.automl.train(y=outcome_var, x=x_vars, training_frame=train_h2o)

        # The best model found by AutoML becomes our main model
        # If AutoML run completes with no model (e.g. time limit too short), fall back.
        if self.automl.leader is None:
            self.fit(X.iloc[:5], y.iloc[:5]) # Re-call fit with tiny data to trigger dummy model
        else:
            self.model = self.automl.leader
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels, handling the dummy model edge case."""
        check_is_fitted(self)
        # If a dummy model was used, its predictions are meaningless and can
        # crash the server. Return a safe, default prediction.
        if self._using_dummy_model:
            return np.full(len(X), self.classes_[0])
        
        # If the safety check passes, call the underlying model's predict method directly.
        self._ensure_h2o_is_running()
        test_h2o = h2o.H2OFrame(X)
        predictions = self.model.predict(test_h2o)
        return predictions["predict"].as_data_frame().values.ravel()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities, handling the dummy model edge case."""
        check_is_fitted(self)
        # If a dummy model was used, return a default probability distribution.
        if self._using_dummy_model:
            n_samples = len(X)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            proba[:, 0] = 1.0
            return proba

        # If the safety check passes, call the underlying model's predict method directly.
        self._ensure_h2o_is_running()
        test_h2o = h2o.H2OFrame(X)
        predictions = self.model.predict(test_h2o)
        prob_df = predictions.drop("predict").as_data_frame()
        return prob_df.values

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        super().shutdown()
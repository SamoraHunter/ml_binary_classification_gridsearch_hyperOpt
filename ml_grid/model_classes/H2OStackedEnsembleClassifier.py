import pandas as pd
from h2o.estimators import H2OStackedEnsembleEstimator, H2OGradientBoostingEstimator, H2ORandomForestEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import numpy as np
import h2o
from sklearn.utils.validation import check_is_fitted

class H2OStackedEnsembleClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Stacked Ensemble.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OStackedEnsembleClassifier.
        """
        super().__init__(H2OStackedEnsembleEstimator, **kwargs)
        self._using_dummy_model = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OStackedEnsembleClassifier":
        """Fits the H2O Stacked Ensemble model.

        This involves training default base models and then the metalearner.
        """
        # --- CRITICAL FIX for single-feature data ---
        # StackedEnsemble is inherently unstable on single-feature data, as the
        # metalearner receives only one feature (the predictions of base models),
        # which can be constant. We fall back to a dummy model in this case.
        if len(X.columns) < 2:
            print(
                f"Warning: Dataset has only {len(X.columns)} feature(s). "
                f"H2OStackedEnsemble is unstable in this scenario. "
                f"Skipping stacking and using a dummy model."
            )
            self._using_dummy_model = True
            self._handle_small_data_fallback(X, y)
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # StackedEnsemble requires base models to be trained with CV.
        nfolds = 5 if len(train_h2o) >= 10 else 2

        # Define robust base models that can handle small/constant data folds
        base_gbm = H2OGradientBoostingEstimator(
            model_id="base_gbm_se", seed=1, nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True, min_rows=1, ignore_const_cols=False
        )
        base_drf = H2ORandomForestEstimator(
            model_id="base_drf_se", seed=1, nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True, min_rows=1, ignore_const_cols=False
        )

        # Train base models
        base_gbm.train(x=x_vars, y=outcome_var, training_frame=train_h2o)
        base_drf.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

        # Set the base models for the stacked ensemble
        model_params['base_models'] = [base_gbm.model_id, base_drf.model_id]

        # H2OStackedEnsembleEstimator does not accept 'ignore_const_cols'.
        # The base class may have added it, so we remove it here.
        model_params.pop('ignore_const_cols', None)

        # Ensure the metalearner is also robust to constant features.
        # This can happen if all base models make the same prediction.
        if 'metalearner_params' not in model_params:
            model_params['metalearner_params'] = {}
        model_params['metalearner_params'].setdefault('ignore_const_cols', False)

        # Train the stacked ensemble
        self.model = self.estimator_class(**model_params)
        self.model.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels, handling the dummy model edge case."""
        check_is_fitted(self)
        if self._using_dummy_model:
            return np.full(len(X), self.classes_[0])
        
        return super().predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities, handling the dummy model edge case."""
        check_is_fitted(self)
        if self._using_dummy_model:
            n_samples = len(X)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            proba[:, 0] = 1.0
            return proba

        return super().predict_proba(X)

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        super().shutdown()
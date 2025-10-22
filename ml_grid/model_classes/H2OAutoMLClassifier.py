import h2o
import numpy as np
import pandas as pd
import logging
from h2o.automl import H2OAutoML
from .H2OBaseClassifier import H2OBaseClassifier
from h2o.estimators import H2OGeneralizedLinearEstimator

logger = logging.getLogger(__name__)

class H2OAutoMLClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's AutoML.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OAutoMLClassifier.

        Note: H2OAutoML is not a standard estimator, so we use a placeholder
        in the base class and manage the AutoML process within the `fit` method.
        """
        # Use a placeholder estimator. The actual model will be the AutoML leader.
        # This allows us to use the base class infrastructure.
        super().__init__(estimator_class=H2OGeneralizedLinearEstimator, **kwargs)
        self.automl = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OAutoMLClassifier":
        """Fits the H2O AutoML process.

        If the dataset is too small or AutoML fails to find a leader model,
        it gracefully falls back to a simple GLM model.
        """
        # --- 1. Standard Validation and Preparation ---
        # Use base class methods for validation. This also handles small data errors.
        X, y = self._validate_input_data(X, y)
        self._validate_min_samples_for_fit(X, y)

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- 2. AutoML Specific Checks and Execution ---
        min_samples = 20  # A reasonable minimum for AutoML
        run_automl = len(train_h2o) >= min_samples and len(x_vars) >= 1

        if run_automl:
            self.logger.info("Dataset is large enough. Running H2O AutoML...")
            self.automl = H2OAutoML(**model_params)
            self.automl.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            
            # The best model found by AutoML becomes our main model
            if self.automl.leader:
                self.logger.info(f"AutoML found a leader model: {self.automl.leader.model_id}")
                self.model_ = self.automl.leader
            else:
                self.logger.warning("H2O AutoML finished but found no leader model. Falling back to a simple GLM.")
                run_automl = False # Trigger the fallback logic

        if not run_automl:
            self.logger.warning(
                f"Dataset too small for H2O AutoML or AutoML failed. "
                f"({len(train_h2o)} rows, {len(x_vars)} features). "
                f"Fitting a simple GLM as a fallback."
            )
            # Use a simple, robust GLM as a fallback model
            self.model_ = H2OGeneralizedLinearEstimator(
                family='binomial', ignore_const_cols=False
            )
            self.model_.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self._using_dummy_model = True # Set flag for reference

        # --- 3. Finalize Fit using Base Class Standards ---
        # CRITICAL: Store the model_id for persistence and retrieval.
        # This allows the base class predict/predict_proba to work correctly.
        if self.model_:
            self.model_id = self.model_.model_id
            self.logger.info(f"✓✓✓ SUCCESS: H2OAutoMLClassifier is fitted. Final model_id: {self.model_id}")
        else:
            raise RuntimeError("H2OAutoMLClassifier failed to produce a final model.")

        return self

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        # The base class __del__ handles cleanup. A specific shutdown method
        # is better handled at the pipeline level.
        pass

    # predict() and predict_proba() are now inherited from H2OBaseClassifier
    # and will work correctly because we set self.model_ and self.model_id.
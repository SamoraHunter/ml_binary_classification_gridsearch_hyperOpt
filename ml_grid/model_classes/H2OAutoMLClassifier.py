import logging

import pandas as pd
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGeneralizedLinearEstimator

from .H2OBaseClassifier import H2OBaseClassifier

logger = logging.getLogger(__name__)


class H2OAutoMLClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's AutoML."""

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

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            **kwargs: Additional keyword arguments (not used by this implementation).

        Returns:
            H2OAutoMLClassifier: The fitted classifier instance.
        """
        # --- 1. Standard Validation and Preparation ---
        # Use base class methods for validation. This also handles small data errors.
        X, y = self._validate_input_data(X, y)
        # Handle small data fallback before preparing H2O frames
        if self._handle_small_data_fallback(X, y):
            # If a dummy model was fit, we can finalize and return
            return self._finalize_dummy_fit(X, y)

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
                self.logger.info(
                    f"AutoML found a leader model: {self.automl.leader.model_id}"
                )
                self.model_ = self.automl.leader
            else:
                self.logger.warning(
                    "H2O AutoML finished but found no leader model. Falling back to a simple GLM."
                )
                run_automl = False  # Trigger the fallback logic

        if not run_automl:
            self.logger.warning(
                f"Dataset too small for H2O AutoML or AutoML failed. "
                f"({len(train_h2o)} rows, {len(x_vars)} features). "
                f"Fitting a simple GLM as a fallback."
            )
            # Use a simple, robust GLM as a fallback model
            self.model_ = H2OGeneralizedLinearEstimator(
                family="binomial", ignore_const_cols=False
            )
            self._sanitize_model_params()
            self.model_.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self._using_dummy_model = True  # Set flag for reference

        # --- 3. Finalize Fit using Base Class Standards ---
        # CRITICAL: Store the model_id for persistence and retrieval.
        # This allows the base class predict/predict_proba to work correctly.
        if self.model_:
            self.model_id = self.model_.model_id
            self.logger.info(
                f"✓✓✓ SUCCESS: H2OAutoMLClassifier is fitted. Final model_id: {self.model_id}"
            )
        else:
            raise RuntimeError("H2OAutoMLClassifier failed to produce a final model.")

        return self

    def _finalize_dummy_fit(self, X, y):
        """Finalizes the fitting process when a dummy model is used."""
        self.logger.info("Finalizing fit for dummy GLM model.")
        # Use a simple, robust GLM as a fallback model
        self.model_ = H2OGeneralizedLinearEstimator(
            family="binomial", ignore_const_cols=False
        )
        self._sanitize_model_params()
        # We need to create a minimal H2OFrame to train on
        train_h2o, x_vars, outcome_var, _ = self._prepare_fit(X, y)
        self.model_.train(y=outcome_var, x=x_vars, training_frame=train_h2o)

        # Set the model_id for predict() to work
        self.model_id = self.model_.model_id
        self.logger.info(
            f"✓✓✓ SUCCESS: H2OAutoMLClassifier is fitted with a fallback model. Final model_id: {self.model_id}"
        )
        return self

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        # The base class __del__ handles cleanup. A specific shutdown method
        # is better handled at the pipeline level.
        pass

    # predict() and predict_proba() are now inherited from H2OBaseClassifier
    # and will work correctly because we set self.model_ and self.model_id.

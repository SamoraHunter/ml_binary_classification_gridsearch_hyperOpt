import logging
import inspect
import pandas as pd
from typing import List

# Import H2O estimators
try:
    from h2o.estimators import H2OStackedEnsembleEstimator
except ImportError:
    logging.getLogger(__name__).warning(
        "H2OStackedEnsembleEstimator could not be imported. "
        "H2OStackedEnsembleClassifier will not be available."
    )
    class H2OStackedEnsembleEstimator: ...

# Import the base class
from .H2OBaseClassifier import H2OBaseClassifier

# Configure logging
logger = logging.getLogger(__name__)


class H2OStackedEnsembleClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for the H2O Stacked Ensemble classifier.
    
    This class adheres to the scikit-learn API (fit, predict, predict_proba)
    by inheriting from H2OBaseClassifier and uses H2OStackedEnsembleEstimator
    as its underlying model.
    
    This wrapper is designed to be used within the ml_grid pipeline, but has
    known limitations with scikit-learn's cross-validation (like GridSearchCV)
    due to H2O's management of base model CV predictions.
    
    The `fit` method is overridden to handle the specific requirements of
    a stacked ensemble, namely the `base_models` parameter.
    
    The `predict` and `predict_proba` methods are inherited from H2OBaseClassifier.
    """

    def __init__(self, base_models: List[H2OBaseClassifier] = None, **kwargs):
        """Initializes the H2OStackedEnsembleClassifier.
        
        Args:
            base_models (List[H2OBaseClassifier]): A list of *fitted*
                H2OBaseClassifier (or compatible) instances. These models
                *must* have been trained with `nfolds > 1` and 
                `keep_cross_validation_predictions=True`.
            **kwargs: Keyword arguments passed to the H2OStackedEnsembleEstimator.
                Common arguments include `metalearner_algorithm`, `seed`, etc.
        """
        # Pass base_models along with other kwargs to the parent constructor.
        # This ensures it's treated as a standard sklearn parameter.
        kwargs['base_models'] = base_models if base_models is not None else []
        kwargs['estimator_class'] = H2OStackedEnsembleEstimator
        super().__init__(**kwargs)


    def set_params(self, **kwargs):
        """
        Overrides set_params to correctly handle the `base_models` list,
        which is critical for scikit-learn's `clone` function.
        """
        super().set_params(**kwargs)
        return self

    def get_params(self, deep: bool = True) -> dict:
        """
        Overrides get_params to ensure `base_models` is included,
        allowing scikit-learn's `clone` to work correctly.
        """
        # Rely on the parent's get_params, which will correctly handle all attributes
        # set in __init__, including `base_models`.
        return super().get_params(deep=deep)

    def score(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        This method is required for scikit-learn compatibility, especially
        for use with tools like GridSearchCV when no `scoring` is specified.

        Args:
            X: Test samples.
            y: True labels for X.
            sample_weight: Sample weights (not used by H2O models).

        Returns:
            The mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OStackedEnsembleClassifier":
        """Fits the H2O Stacked Ensemble model, making it compatible with scikit-learn's CV tools.

        This method encapsulates the entire two-stage fitting process:
        1. It first fits each of the base models on the provided training data, ensuring
           they are trained with cross-validation to generate predictions for the metalearner.
        2. It then collects the model IDs of the fitted base models.
        3. Finally, it trains the metalearner (the stacked ensemble model) using these
           base models.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self

        Raises:
            ValueError: If `base_models` is empty or not provided.
            Exception: Re-raises any exception from H2O's `train` method.
        """
        try:
            # 1. Initial validation
            X, y = self._validate_input_data(X, y)
            if self._handle_small_data_fallback(X, y):
                return self

            if not self.base_models:
                raise ValueError("`base_models` parameter is empty. "
                                 "H2OStackedEnsembleClassifier requires a "
                                 "list of base model estimators.")

            # 2. Fit each base model
            self.logger.info(f"Fitting {len(self.base_models)} base models for StackedEnsemble...")
            base_models_list = []
            for i, model_wrapper in enumerate(self.base_models):
                self.logger.debug(f"Fitting base model {i+1}: {type(model_wrapper).__name__}")
                model_wrapper.set_params(
                    nfolds=5,  # A reasonable default for base model CV
                    keep_cross_validation_predictions=True,
                    fold_assignment="Modulo"
                )
                # CRITICAL FIX: Explicitly call the fit method from H2OBaseClassifier
                # to ensure correct data handling (pandas -> H2OFrame) and avoid the
                # 'cbind' error from the native H2O fit method.
                H2OBaseClassifier.fit(model_wrapper, X, y)
                base_models_list.append(model_wrapper.model_id)
            self.logger.info("All base models fitted.")

            # 3. Fit the metalearner (the ensemble itself)
            # The parent _prepare_fit handles data conversion and parameter extraction
            train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

            self.model_ = H2OStackedEnsembleEstimator(
                base_models=base_models_list,
                **model_params
            )
            self.model_.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

            # 4. Store fitted attributes for sklearn compatibility
            # These are inherited from the parent's _prepare_fit call
            self.model_id = self.model_.model_id

            self.logger.debug(f"Successfully fitted {self.estimator_class.__name__}")

        except Exception as e:
            self.logger.critical(
                f"A critical, unrecoverable error occurred during H2OStackedEnsemble fit: {e}",
                exc_info=True
            )
            raise e

        return self
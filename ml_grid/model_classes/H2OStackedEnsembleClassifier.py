import os
import tempfile
from typing import Any, Dict, List, Optional

import h2o
import pandas as pd
from h2o.estimators import H2OStackedEnsembleEstimator
from sklearn.utils.validation import check_is_fitted

from .H2OBaseClassifier import H2OBaseClassifier


class H2OStackedEnsembleClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Stacked Ensemble.

    This class handles the training of base models and a metalearner to create
    a stacked ensemble. It is designed to be robust within scikit-learn's
    cross-validation framework by managing model state and persistence.
    """

    def __init__(
        self,
        base_models: Optional[List[Any]] = None,
        metalearner_algorithm: str = "AUTO",
        **kwargs,
    ):
        """Initializes the H2OStackedEnsembleClassifier.

        Args:
            base_models (Optional[List[Any]]): A list of un-trained H2O estimator
                instances to be used as base models.
            metalearner_algorithm (str): The algorithm to use for the metalearner.
                Defaults to "AUTO".
            **kwargs: Additional keyword arguments passed to the H2OStackedEnsembleEstimator.
        """
        kwargs.pop("estimator_class", None)
        super().__init__(estimator_class=H2OStackedEnsembleEstimator, **kwargs)

        if base_models is None:
            raise ValueError("H2OStackedEnsembleClassifier requires a list of 'base_models'.")

        self.base_models = base_models
        self.metalearner_algorithm = metalearner_algorithm

        # Internal state for managing trained models
        self._trained_base_model_ids: List[str] = []
        self._checkpoint_dir = tempfile.mkdtemp()

    def __sklearn_clone__(self):
        """Custom clone implementation to handle raw H2O estimators in base_models.
        
        This bypasses sklearn's default cloning for base_models to avoid the
        lambda/lambda_ issue in raw H2O estimators.
        """
        # Get all parameters except base_models
        params = self.get_params(deep=False)
        base_models = params.pop('base_models', None)
        
        # Clone base_models manually, handling raw H2O estimators
        cloned_base_models = []
        if base_models:
            for model in base_models:
                cloned_model = self._clone_h2o_estimator(model)
                cloned_base_models.append(cloned_model)
        
        # Create new instance with cloned parameters
        params['base_models'] = cloned_base_models
        cloned = type(self)(**params)
        
        return cloned
    
    def _clone_h2o_estimator(self, estimator):
        """Clone an H2O estimator, handling the lambda/lambda_ issue and model_id conflicts.
        
        Args:
            estimator: An H2O estimator instance or our wrapper
            
        Returns:
            A cloned estimator with a unique model_id
        """
        from sklearn.base import clone as sklearn_clone
        import uuid
        
        # Check if it's a raw H2O estimator
        if hasattr(estimator, '__module__') and 'h2o.estimators' in str(estimator.__module__):
            # It's a raw H2O estimator - handle lambda_ specially
            model_class = type(estimator)
            
            # Get parameters using H2O's _parms property if available
            if hasattr(estimator, '_parms') and estimator._parms:
                params = dict(estimator._parms)
            else:
                # Fall back to inspecting attributes
                params = {}
                import inspect
                sig = inspect.signature(model_class)
                for param_name in sig.parameters:
                    if param_name == 'self':
                        continue
                    if hasattr(estimator, param_name):
                        value = getattr(estimator, param_name)
                        if value is not None:
                            params[param_name] = value
            
            # Critical fix: Ensure lambda_ is used, not lambda
            if 'lambda' in params:
                params['lambda_'] = params.pop('lambda')
            
            # CRITICAL FIX: Generate unique model_id for each clone to prevent conflicts in CV
            # During cross-validation, sklearn clones the estimator for each fold
            # If all clones have the same model_id, H2O will have conflicts
            if 'model_id' in params and params['model_id']:
                original_id = params['model_id']
                # Append a unique suffix to prevent collisions
                unique_suffix = str(uuid.uuid4())[:8]
                params['model_id'] = f"{original_id}_{unique_suffix}"
                self.logger.debug(f"Cloning H2O estimator: {original_id} -> {params['model_id']}")
            
            # Create new instance
            try:
                return model_class(**params)
            except TypeError as e:
                if 'lambda' in str(e):
                    # Still having lambda issues - try one more time with cleaned params
                    params.pop('lambda', None)
                    if hasattr(estimator, 'lambda_'):
                        params['lambda_'] = estimator.lambda_
                    elif hasattr(estimator, '_lambda'):
                        params['lambda_'] = estimator._lambda
                    return model_class(**params)
                else:
                    raise
        else:
            # It's our wrapper or another sklearn estimator - use normal cloning
            return sklearn_clone(estimator)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator.
        
        Overrides parent to handle base_models - we return it but don't deep clone it
        here because __sklearn_clone__ handles it specially.
        """
        # Get params from parent
        params = super().get_params(deep=False)  # Always use deep=False for parent
        
        # base_models is already in params from parent's implementation
        # We don't deep clone it here - let __sklearn_clone__ handle it
        
        return params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OStackedEnsembleClassifier":
        """Fits the stacked ensemble model.

        This involves training each base model on the data and then training a
        metalearner on the predictions of the base models.

        Args:
            X (pd.DataFrame): The training feature data.
            y (pd.Series): The training target data.

        Returns:
            self: The fitted estimator.
        """
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- ROBUSTNESS FIX: Train or reload base models ---
        self._train_or_reload_base_models(train_h2o, x_vars, outcome_var)

        if not self._trained_base_model_ids:
            raise RuntimeError("Base models failed to train or load.")

        # Configure and train the stacked ensemble
        model_params["base_models"] = self._trained_base_model_ids
        model_params.setdefault("metalearner_algorithm", self.metalearner_algorithm)

        self.model = self.estimator_class(**model_params)
        self.model.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

        self.logger.info(f"Successfully fitted StackedEnsemble with metalearner {self.model.metalearner().algo}")
        return self

    def _train_or_reload_base_models(self, train_h2o, x_vars, outcome_var):
        """
        Trains the base models, saving checkpoints to prevent garbage collection.
        If a model ID already exists, it attempts to reload it.
        
        NOTE: H2O StackedEnsemble requires base models to use cross-validation.
        We set nfolds=5 by default if not already set.
        """
        self.logger.info(f"Training {len(self.base_models)} base models...")
        self._trained_base_model_ids = []

        for model_estimator in self.base_models:
            model_id = model_estimator.model_id

            # Check if model already exists in the cluster
            # h2o.get_model() raises an exception if the model doesn't exist
            try:
                existing_model = h2o.get_model(model_id)
                if existing_model:
                    self.logger.debug(f"Base model {model_id} already exists in H2O cluster.")
                    self._trained_base_model_ids.append(model_id)
                    continue
            except Exception:
                # Model doesn't exist - we'll train it below
                pass

            # Check if model exists in checkpoint directory
            checkpoint_path = os.path.join(self._checkpoint_dir, model_id)
            if os.path.exists(checkpoint_path):
                try:
                    self.logger.debug(f"Reloading base model {model_id} from {checkpoint_path}")
                    reloaded_model = h2o.load_model(checkpoint_path)
                    self._trained_base_model_ids.append(reloaded_model.model_id)
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to reload model {model_id}: {e}. Retraining.")

            # --- CRITICAL FIX: H2O StackedEnsemble requires base models to use CV ---
            # Set nfolds if not already set - this is required for stacking
            if not hasattr(model_estimator, 'nfolds') or model_estimator.nfolds is None or model_estimator.nfolds == 0:
                model_estimator.nfolds = 5
                self.logger.debug(f"Set nfolds=5 for base model {model_id} (required for StackedEnsemble)")
            
            # Also ensure keep_cross_validation_predictions is enabled
            if hasattr(model_estimator, 'keep_cross_validation_predictions'):
                model_estimator.keep_cross_validation_predictions = True

            # --- ROBUSTNESS FIX: Save checkpoints to prevent GC ---
            # This is critical for preventing base models from being deleted
            # before the stack can use them, especially in CV loops.
            model_estimator.export_checkpoints_dir = self._checkpoint_dir

            self.logger.debug(f"Training base model: {model_estimator.algo} with id {model_id}")
            model_estimator.train(x=x_vars, y=outcome_var, training_frame=train_h2o)
            self._trained_base_model_ids.append(model_estimator.model_id)

        self.logger.info(f"Finished training base models. IDs: {self._trained_base_model_ids}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts using the fitted stacked ensemble model.
        Adds a check to ensure base models are available before prediction.
        """
        check_is_fitted(self)
        self._ensure_h2o_is_running()

        # --- ROBUSTNESS FIX: Verify base models exist before predict ---
        try:
            # If trained IDs are missing (e.g., after cloning), try to get them from the fitted model object
            if not self._trained_base_model_ids and hasattr(self.model, 'base_models'):
                self.logger.debug("Trained base model IDs not found on wrapper, retrieving from H2O model object.")
                self._trained_base_model_ids = [m['name'] for m in self.model.base_models]

            for model_id in self._trained_base_model_ids:
                # Check if model exists in cluster
                try:
                    model = h2o.get_model(model_id)
                    if model is None:
                        raise RuntimeError(f"Model {model_id} returned None")
                except Exception:
                    # Model not in cluster - try to reload from checkpoint
                    checkpoint_path = os.path.join(self._checkpoint_dir, model_id)
                    if os.path.exists(checkpoint_path):
                        self.logger.info(f"Base model {model_id} not in cluster, reloading from checkpoint.")
                        h2o.load_model(checkpoint_path)
                    else:
                        raise RuntimeError(
                            f"Base model {model_id} not found in H2O cluster or checkpoint directory. "
                            "Models may have been garbage collected."
                        )

            # Call parent predict method
            return super().predict(X)

        except RuntimeError:
            # Re-raise RuntimeError as-is (already has good context)
            raise
        except Exception as e:
            # Wrap other exceptions for clarity
            raise RuntimeError(f"An unexpected error occurred during StackedEnsemble predict: {e}")

    def set_params(self, **params: Any) -> "H2OStackedEnsembleClassifier":
        """Sets the parameters of this estimator.

        This method is overridden to handle the special 'base_models' parameter.
        """
        # This is called by sklearn.clone()
        if 'base_models' in params:
            self.base_models = params.pop('base_models')
            # After cloning, the list of trained model IDs will be empty.
            self._trained_base_model_ids = []

        # Set remaining parameters on the superclass
        super().set_params(**params)
        return self

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Overrides the base method to correctly handle stacked ensemble parameters.
        """
        # Get common parameters from the parent class
        params = super()._get_model_params()
        
        # Add the specific parameters for the Stacked Ensemble
        params['metalearner_algorithm'] = self.metalearner_algorithm
        # Note: 'base_models' is handled separately in the fit method
        
        return params
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from h2o.exceptions import H2OResponseError
from h2o.estimators import (
    H2OGeneralizedAdditiveEstimator,
    H2OGeneralizedLinearEstimator,
)

from .H2OBaseClassifier import H2OBaseClassifier

logger = logging.getLogger(__name__)


class H2OGAMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Generalized Additive Models."""

    def __init__(self, _suppress_low_cardinality_error=True, **kwargs):
        """Initializes the H2OGAMClassifier."""
        kwargs.pop("estimator_class", None)
        # GAM prefers Coordinate Descent, but we handle failures gracefully
        kwargs.setdefault("solver", "COORDINATE_DESCENT")

        super().__init__(estimator_class=H2OGeneralizedAdditiveEstimator, **kwargs)
        self._suppress_low_cardinality_error = _suppress_low_cardinality_error
        self._fallback_to_glm = False

    def _prepare_fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Any, List[str], str, Dict[str, Any]]:
        """
        Validates gam_columns and determines if fallback to GLM is needed.
        """
        train_h2o, x_vars, outcome_var, initial_model_params = super()._prepare_fit(
            X, y
        )
        model_params = initial_model_params.copy()
        self._fallback_to_glm = False

        # --- 1. Parameter Normalization ---
        if "gam_columns" in model_params:
            gc = model_params["gam_columns"]
            if isinstance(gc, str):
                model_params["gam_columns"] = [gc]
            elif isinstance(gc, int):
                model_params["gam_columns"] = [str(gc)]
            elif isinstance(gc, tuple):
                model_params["gam_columns"] = list(gc)
            elif isinstance(gc, list) and gc and isinstance(gc[0], list):
                model_params["gam_columns"] = [
                    item for sublist in gc for item in sublist
                ]

        # --- 2. Auto-Selection of GAM Columns ---
        if "gam_columns" not in model_params or not model_params["gam_columns"]:
            potential_gam_cols = []
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    if X[col].nunique() > 10:
                        potential_gam_cols.append(col)

            if potential_gam_cols:
                model_params["gam_columns"] = potential_gam_cols
            else:
                pass

        gam_columns = model_params.get("gam_columns", [])

        # Normalize params
        if gam_columns:
            for param in ["bs", "num_knots", "scale"]:
                val = model_params.get(param)
                if param == "bs" and isinstance(val, str):
                    bs_map = {"cs": 0, "tp": 1}
                    val = bs_map.get(val, 0)
                if isinstance(val, (int, float)) or (
                    param == "bs" and isinstance(val, int)
                ):
                    model_params[param] = [val] * len(gam_columns)

        # --- 3. Cardinality Check & Filtering ---
        needs_fallback = False
        if gam_columns:
            suitable_gam_cols = []
            suitable_knots = []
            suitable_bs = []
            suitable_scale = []

            num_knots_list = model_params.get("num_knots", [5] * len(gam_columns))
            bs_list = model_params.get("bs", [0] * len(gam_columns))
            scale_list = model_params.get("scale", [0.1] * len(gam_columns))

            for i, col in enumerate(gam_columns):
                if col not in X.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(X[col]):
                    continue

                n_unique = X[col].nunique()
                required_knots = num_knots_list[i] if i < len(num_knots_list) else 5

                if n_unique < (required_knots * 2):
                    if not self._suppress_low_cardinality_error:
                        raise ValueError(
                            f"Skipping GAM col '{col}': {n_unique} unique < 2 * {required_knots} knots."
                        )
                    continue

                # --- FIX: Check for unique quantiles to prevent H2O knot generation failure ---
                try:
                    # H2O uses quantiles for knots. If quantiles are not unique, it fails.
                    # We check if we can generate 'required_knots' unique bins.
                    pd.qcut(X[col], q=required_knots, duplicates="raise")
                except ValueError:
                    if not self._suppress_low_cardinality_error:
                        raise ValueError(
                            f"Skipping GAM col '{col}': Cannot generate {required_knots} unique quantiles (distribution too skewed)."
                        )
                    continue

                suitable_gam_cols.append(col)
                suitable_knots.append(required_knots)
                if i < len(bs_list):
                    suitable_bs.append(bs_list[i])
                if i < len(scale_list):
                    suitable_scale.append(scale_list[i])

            if not suitable_gam_cols:
                self.logger.warning(
                    "No suitable GAM columns remaining. Falling back to GLM."
                )
                needs_fallback = True
            else:
                model_params["gam_columns"] = suitable_gam_cols
                model_params["num_knots"] = suitable_knots
                model_params["bs"] = suitable_bs
                model_params["scale"] = suitable_scale
        else:
            self.logger.warning("No GAM columns found. Falling back to GLM.")
            needs_fallback = True

        # --- 4. Setup Fallback ---
        if needs_fallback:
            self._fallback_to_glm = True
            import inspect

            glm_keys = set(
                inspect.signature(H2OGeneralizedLinearEstimator).parameters.keys()
            )
            model_params = {k: v for k, v in model_params.items() if k in glm_keys}

        return train_h2o, x_vars, outcome_var, model_params

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OGAMClassifier":
        """Fits the model, using GLM fallback if validation failed."""

        X, y = self._validate_input_data(X, y)
        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        if self._fallback_to_glm:
            self.logger.info(
                "Using H2OGeneralizedLinearEstimator (Fallback) with L_BFGS"
            )
            estimator_cls = H2OGeneralizedLinearEstimator
            # Force L_BFGS and disable lambda search on fallback
            model_params["solver"] = "L_BFGS"
            model_params["remove_collinear_columns"] = False
            model_params["lambda_search"] = False
        else:
            estimator_cls = self.estimator_class

        self.model_ = estimator_cls(**model_params)

        # --- RUNTIME TRAIN WITH FALLBACK ---
        try:
            self.model_.train(x=x_vars, y=outcome_var, training_frame=train_h2o)
        except (H2OResponseError, Exception) as e:
            error_msg = str(e).lower()
            if (
                "gam_columns" in error_msg
                or "knots" in error_msg
                or "illegal argument" in error_msg
                or "nullpointerexception" in error_msg
            ):
                self.logger.warning(
                    f"H2O GAM Training failed ({e}). Attempting Fallback to GLM (L_BFGS)."
                )

                import inspect

                glm_keys = set(
                    inspect.signature(H2OGeneralizedLinearEstimator).parameters.keys()
                )
                glm_params = {k: v for k, v in model_params.items() if k in glm_keys}

                # FORCE SAFE SETTINGS
                glm_params["solver"] = "L_BFGS"
                glm_params["remove_collinear_columns"] = False
                glm_params["lambda_search"] = False

                self.model_ = H2OGeneralizedLinearEstimator(**glm_params)
                self.model_.train(x=x_vars, y=outcome_var, training_frame=train_h2o)
            else:
                raise e

        self.model_id = self.model_.model_id

        return self

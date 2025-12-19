"""H2O Generalized Additive Model (GAM) Classifier.

This module contains the H2OGAMClass, which is a configuration class for
the H2OGAMClassifier. It provides parameter spaces for grid search and
Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from skopt.space import Categorical, Integer, Real

from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger(__name__)


class H2OGAMClass:
    """A configuration class for the H2OGAMClassifier.

    Provides parameter spaces for grid search and Bayesian optimization.
    The parameter space is dynamically generated to include columns from the
    input data `X` for the `gam_columns` parameter, filtering out columns
    unsuitable for smoothing (e.g., low cardinality).
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: str = "small",
    ) -> None:
        """Initializes the H2OGAMClass."""
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        # Instantiate the Estimator
        self.algorithm_implementation: H2OGAMClassifier = H2OGAMClassifier()

        self.method_name: str = "H2OGAMClassifier"
        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

        # --- SMART GAM COLUMN SELECTION ---
        # Filter X to find only numeric columns with sufficient cardinality (>10).
        gam_cols = []
        if X is not None:
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Check cardinality (>10 unique values)
                    if X[col].nunique() > 10:
                        # Check distribution for at least 5 knots (default minimum)
                        try:
                            pd.qcut(X[col], q=5, duplicates="raise")
                            gam_cols.append(col)
                        except ValueError:
                            pass

        if not gam_cols and X is not None:
            logger.warning(
                "No high-cardinality numeric columns found for GAM splines. Search will likely fallback to GLM."
            )

        # Define Parameter Space
        if global_parameters.bayessearch:
            # Bayesian Search Space
            param_space = {
                "num_knots": Integer(5, 10),
                "bs": Categorical(["cs", "tp"]),
                "scale": Real(0.01, 1.0, "log-uniform"),
                "seed": Integer(1, 1000),
                "solver": Categorical(["COORDINATE_DESCENT"]),
            }
            if gam_cols:
                param_space["gam_columns"] = Categorical(gam_cols)

            self.parameter_space = param_space
        else:
            # Grid/Random Search Space
            param_space = {
                "num_knots": [5, 8, 10],
                "bs": ["cs", "tp"],
                "scale": [0.01, 0.1, 0.5, 1.0],
                "seed": [1, 42, 123],
                "solver": ["COORDINATE_DESCENT"],
            }
            if gam_cols:
                param_space["gam_columns"] = gam_cols

            self.parameter_space = [param_space]

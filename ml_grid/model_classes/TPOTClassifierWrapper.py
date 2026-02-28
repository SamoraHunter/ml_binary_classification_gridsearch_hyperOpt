"""TPOT Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for TPOTClassifier.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Attempt to import TPOT
try:
    from tpot import TPOTClassifier
except ImportError:
    TPOTClassifier = None

logger = logging.getLogger(__name__)


class TPOTClassifierWrapper(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for TPOTClassifier."""

    def __init__(
        self,
        generations: int = 5,
        population_size: int = 20,
        offspring_size: Optional[int] = None,
        mutation_rate: float = 0.9,
        crossover_rate: float = 0.1,
        scoring: str = "accuracy",
        cv: int = 5,
        subsample: float = 1.0,
        n_jobs: int = -1,
        max_time_mins: Optional[int] = None,
        max_eval_time_mins: float = 5,
        random_state: int = 42,
        verbosity: int = 2,
        early_stop: Optional[int] = None,
    ):
        self.generations = generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scoring = scoring
        self.cv = cv
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.random_state = random_state
        self.verbosity = verbosity
        self.early_stop = early_stop

        self.model_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "TPOTClassifierWrapper":
        if TPOTClassifier is None:
            raise ImportError(
                "TPOT is not installed. Please install it to use TPOTClassifierWrapper."
            )

        self.model_ = TPOTClassifier(
            generations=self.generations,
            population_size=self.population_size,
            offspring_size=self.offspring_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            scoring=self.scoring,
            cv=self.cv,
            subsample=self.subsample,
            n_jobs=self.n_jobs,
            max_time_mins=self.max_time_mins,
            max_eval_time_mins=self.max_eval_time_mins,
            random_state=self.random_state,
            verbosity=self.verbosity,
            early_stop=self.early_stop,
            disable_update_check=True,
        )

        # TPOT can be slow. For quick checks, it's useful to see it has started.
        logger.info(
            f"Starting TPOT fit with generations={self.generations}, population_size={self.population_size}..."
        )

        self.model_.fit(X, y, **kwargs)

        # After fitting, TPOT stores the best pipeline in the `fitted_pipeline_` attribute.
        # We must set `classes_` for scikit-learn compatibility (e.g., for check_is_fitted).
        # While TPOT exposes `self.model_.classes_`, inferring from `y` is a more robust
        # fallback, consistent with other wrappers in this project.
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        else:
            self.classes_ = np.unique(y)

        logger.info("TPOT fit completed.")
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_"])
        return self.model_.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_"])
        return self.model_.predict_proba(X)

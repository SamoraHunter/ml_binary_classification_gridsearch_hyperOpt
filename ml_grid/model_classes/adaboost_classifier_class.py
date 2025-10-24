from typing import Optional
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
from hyperopt import hp

import logging

class adaboost_class:
    """AdaBoostClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
        estimator: Optional[object] = None,
    ):
        self.X = X
        self.y = y
        
        self.algorithm_implementation = AdaBoostClassifier(estimator=estimator)
        self.method_name = "AdaBoostClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        
        is_bayes_search = global_parameters.bayessearch

        if is_bayes_search:
            # For Bayesian search, we need to sample the complete estimator with its parameters
            # The key issue is that we can't set estimator__ parameters without first having an estimator
            def sample_dt_estimator(max_depth):
                """Helper to create DecisionTree estimator with sampled parameters"""
                return DecisionTreeClassifier(max_depth=int(max_depth), random_state=1)
            
            def sample_svc_estimator(C, kernel):
                """Helper to create SVC estimator with sampled parameters"""
                return SVC(C=C, kernel=kernel, random_state=1, probability=True)
            
            # Note: We use scope.int() wrapper to convert quniform results to integers
            # This is necessary because sklearn expects integer types for n_estimators
            from hyperopt.pyll import scope
            
            self.parameter_space = hp.choice('estimator_type', [
                # Choice 1: DecisionTreeClassifier with SAMME
                {
                    "estimator": hp.choice('dt_estimator', [
                        DecisionTreeClassifier(max_depth=1, random_state=1),
                        DecisionTreeClassifier(max_depth=2, random_state=1),
                        DecisionTreeClassifier(max_depth=3, random_state=1),
                        DecisionTreeClassifier(max_depth=4, random_state=1),
                        DecisionTreeClassifier(max_depth=5, random_state=1),
                    ]),
                    "n_estimators": scope.int(hp.quniform('dt_n_estimators', 50, 500, 10)),
                    "learning_rate": hp.loguniform('dt_learning_rate', -4.6, 0.0), # ~0.01 to 1.0
                    "algorithm": 'SAMME',
                },
                # Choice 2: SVC with SAMME
                {
                    "estimator": hp.choice('svc_estimator', [
                        SVC(C=0.1, kernel='rbf', random_state=1, probability=True),
                        SVC(C=0.1, kernel='poly', random_state=1, probability=True),
                        SVC(C=1.0, kernel='rbf', random_state=1, probability=True),
                        SVC(C=1.0, kernel='poly', random_state=1, probability=True),
                        SVC(C=10.0, kernel='rbf', random_state=1, probability=True),
                        SVC(C=10.0, kernel='poly', random_state=1, probability=True),
                    ]),
                    "n_estimators": scope.int(hp.quniform('svc_n_estimators', 50, 500, 10)),
                    "learning_rate": hp.loguniform('svc_learning_rate', -4.6, 0.0),
                    "algorithm": 'SAMME',
                }
            ])
        else:
            # For Grid/Random search, define parameter spaces for each valid combination
            self.parameter_space = [
                {
                    # SAMME with DecisionTree
                    "estimator": [DecisionTreeClassifier(random_state=1)],
                    "estimator__max_depth": [1, 2, 3],
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "algorithm": ['SAMME'],
                },
                {
                    # SAMME with SVC (requires probability=True)
                    "estimator": [SVC(random_state=1, probability=True)],
                    "estimator__C": [0.1, 1, 10],
                    "estimator__kernel": ['rbf', 'poly'],
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "algorithm": ['SAMME'],
                },
            ]
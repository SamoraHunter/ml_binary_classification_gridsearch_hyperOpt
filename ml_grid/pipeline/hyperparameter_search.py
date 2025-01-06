from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
from skopt import BayesSearchCV
from sklearn.base import is_classifier
from ml_grid.util.validate_parameters import validate_parameters_helper
from ml_grid.util.global_params import global_parameters

class HyperparameterSearch:
    def __init__(
        self,
        algorithm,
        parameter_space,
        method_name,
        global_params,
        sub_sample_pct=100,
        max_iter=100,
        ml_grid_object=None
    ):
        """
        Initialize the HyperparameterSearchGridRandom class.

        Parameters
        ----------
        algorithm: Estimator
            The algorithm to use in the grid search.
        parameter_space: dict
            The parameter space to search.
        method_name: str
            The name of the algorithm.
        global_params: GlobalParameters
            The global parameters object.
        sub_sample_pct: int, optional
            The percentage of the parameter space to sample.
            Defaults to 100.
        max_iter: int, optional
            The maximum number of iterations to run the grid search for.
            Defaults to 100.
        """
        self.algorithm = algorithm
        self.parameter_space = parameter_space
        self.method_name = method_name
        self.global_params = global_params
        self.sub_sample_pct = sub_sample_pct
        self.max_iter = max_iter
        self.ml_grid_object = ml_grid_object
        global_params = global_parameters()
        
        if self.ml_grid_object is None:
            raise ValueError("ml_grid_object is required.")

        assert is_classifier(self.algorithm), f"The provided algorithm is not a valid scikit-learn classifier. : {type(algorithm)}"
        # permit knn gpu model
        
        # Configure warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Configure GPUs if applicable
        if "keras" in method_name.lower() or "xgbclassifier" in method_name.lower() or "catboostclassifier" in method_name.lower():
            self._configure_gpu()

    def _configure_gpu(self):
        import tensorflow as tf
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def run_search(self, X_train, y_train):
        """
        Executes the hyperparameter search using GridSearchCV or RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            model: Best estimator after hyperparameter search.
        """
        random_search = self.global_params.random_grid_search
        grid_n_jobs = self.global_params.grid_n_jobs
        bayessearch = self.global_params.bayessearch

        if(bayessearch is False):
            # Validate parameters
            parameters = validate_parameters_helper(
                algorithm_implementation=self.algorithm,
                parameters=self.parameter_space,
                ml_grid_object=self.ml_grid_object
            )
        else:
            parameters = self.parameter_space
            
        if bayessearch:
            # print("Running bayessearch hyperparam...")
            # print("parameters: ", parameters)
            # print("max iter: ", self.max_iter)
            # print("algorithm: ", self.algorithm)
            # Bayesian Optimization
            grid = BayesSearchCV(
                estimator=self.algorithm,
                search_spaces=parameters,
                n_iter=self.max_iter,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                verbose=1,
                error_score="raise"
            )

        elif random_search:
            n_iter = min(
                self.max_iter,
                max(2, int(len(ParameterGrid(parameters)) * self.sub_sample_pct / 100))
            )

            grid = RandomizedSearchCV(
                self.algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                n_iter=n_iter,
                error_score="raise"
            )
        else:
            grid = GridSearchCV(
                self.algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                error_score=np.nan
            )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        return best_model

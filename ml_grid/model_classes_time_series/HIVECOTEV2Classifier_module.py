from typing import Any, Dict, List

from aeon.classification.hybrid._hivecote_v2 import HIVECOTEV2
from ml_grid.pipeline.data import pipe


class HIVECOTEV2_class:
    """A wrapper for the aeon HIVECOTEV2 time-series classifier.

    This class provides a consistent interface for the HIVECOTEV2 classifier,
    including defining a hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon HIVECOTEV2 classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: HIVECOTEV2
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the HIVECOTEV2_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        time_limit_param = ml_grid_object.global_params.time_limit_param

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = HIVECOTEV2()
        self.method_name = "HIVECOTEV2"

        self.parameter_space = {
            "stc_params": [
                None
            ],  # Parameters for the ShapeletTransformClassifier module. If None, uses the default parameters with a 2-hour transform contract.
            "drcif_params": [
                None
            ],  # Parameters for the DrCIF module. If None, uses the default parameters with n_estimators set to 500.
            "arsenal_params": [
                None
            ],  # Parameters for the Arsenal module. If None, uses the default parameters.
            "tde_params": [
                None
            ],  # Parameters for the TemporalDictionaryEnsemble module. If None, uses the default parameters.
            "time_limit_in_minutes": time_limit_param,  # Time contract to limit build time in minutes, overriding n_estimators/n_parameter_samples for each component. Default of 0 means n_estimators/n_parameter_samples for each component is used.
            "save_component_probas": [
                False
            ],  # When predict/predict_proba is called, save each HIVE-COTEV2 component probability predictions in component_probas.
            "verbose": [
                verbose_param
            ],  # Level of output printed to the console (for information only).
            "random_state": [
                random_state_val
            ],  # If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
            "n_jobs": [
                n_jobs_model_val
            ],  # The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
            #'parallel_backend': [ 'multiprocessing'], #None, 'loky',  , 'threading'# Specify the parallelization backend implementation in joblib for Catch22, if None a ‘prefer’ value of “threads” is used by default. Valid options are “loky”, “multiprocessing”, “threading” or a custom backend. See the joblib Parallel documentation for more details.
        }

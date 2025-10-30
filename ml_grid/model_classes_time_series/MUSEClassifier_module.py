from typing import Any, Dict, List

from aeon.classification.dictionary_based._muse import MUSE

from ml_grid.pipeline.data import pipe


class MUSE_class:
    """A wrapper for the aeon MUSE time-series classifier.

    MUSE stands for MUltivariate Symbolic Extension. This class provides a
    consistent interface for the classifier, including defining a
    hyperparameter search space.

    Attributes:
        algorithm_implementation: An instance of the aeon MUSE classifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: MUSE
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the MUSE_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """

        random_state_val = ml_grid_object.global_params.random_state_val
        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = MUSE()
        self.method_name = "MUSE"
        self.parameter_space = {
            "anova": [
                True,
                False,
            ],  # If True, Fourier coefficient selection is done via a one-way ANOVA test
            "variance": [
                True,
                False,
            ],  # If True, Fourier coefficient selection is done via the largest variance
            "bigrams": [True, False],  # Whether to create bigrams of SFA words
            "window_inc": [
                2,
                4,
            ],  # Increment used to determine the next window size for BoP model
            "alphabet_size": [
                4,
                6,
                8,
            ],  # Number of possible letters (values) for each word
            "use_first_order_differences": [
                True,
                False,
            ],  # If True, adds the first order differences of each dimension to the data
            "feature_selection": [
                "chi2",
                "none",
                "random",
            ],  # Sets the feature selection strategy to be used
            "p_threshold": [
                0.01,
                0.05,
                0.1,
            ],  # P-value threshold for chi-squared test on bag-of-words
            "support_probabilities": [
                True,
                False,
            ],  # If True, trains a LogisticRegression to support predict_proba()
            "n_jobs": [
                n_jobs_model_val
            ],  # Number of jobs to run in parallel for fit and predict
            "random_state": [random_state_val],  # Seed for random number generation
        }

from typing import Any, Dict, List

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from skopt.space import Categorical, Integer

from ml_grid.pipeline.data import pipe


class KNeighborsTimeSeriesClassifier_class:
    """A wrapper for the aeon KNeighborsTimeSeriesClassifier.

    This class provides a consistent interface for the
    KNeighborsTimeSeriesClassifier, including defining a hyperparameter
    search space.

    Attributes:
        algorithm_implementation: An instance of the aeon
            KNeighborsTimeSeriesClassifier.
        method_name (str): The name of the classifier method.
        parameter_space (Dict[str, List[Any]]): The hyperparameter search space
            for the classifier.
    """

    algorithm_implementation: KNeighborsTimeSeriesClassifier
    method_name: str
    parameter_space: Dict[str, List[Any]]

    def __init__(self, ml_grid_object: pipe):
        """Initializes the KNeighborsTimeSeriesClassifier_class.

        Args:
            ml_grid_object (pipe): An instance of the main data pipeline object.
        """
        self.algorithm_implementation = KNeighborsTimeSeriesClassifier()
        self.method_name = "KNeighborsTimeSeriesClassifier"

        if getattr(ml_grid_object.global_params, "test_mode", False):
            self.parameter_space = {
                "n_neighbors": [1],
                "distance": ["euclidean"],
                "n_jobs": [1],
            }
            return

        if ml_grid_object.global_params.bayessearch:
            self.parameter_space = {
                "distance": Categorical(
                    [
                        "dtw",
                        "euclidean",
                    ]
                ),
                "n_neighbors": Integer(2, 5),
                "n_jobs": [ml_grid_object.global_params.knn_n_jobs],
            }
        else:
            self.parameter_space = {
                "distance": [
                    "dtw",
                    "euclidean",
                ],  # , 'cityblock' 'ctw', 'sqeuclidean','sax' 'softdtw'
                "n_neighbors": [2, 3, 5],  # [log_med_long]
                "n_jobs": [ml_grid_object.global_params.knn_n_jobs],
            }

        # nb consider probability scoring on binary class eval: CalibratedClassifierCV

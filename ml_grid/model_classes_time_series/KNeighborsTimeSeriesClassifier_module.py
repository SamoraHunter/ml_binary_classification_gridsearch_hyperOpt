from typing import Any, Dict, List

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from ml_grid.pipeline.data import pipe


class KNeighborsTimeSeriesClassifier_class:
    """A wrapper for the aeon KNeighborsTimeSeriesClassifier."""

    def __init__(self, ml_grid_object: pipe):
        """Initializes the KNeighborsTimeSeriesClassifier_class.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                data and global parameters.
        """
        self.algorithm_implementation: KNeighborsTimeSeriesClassifier = (
            KNeighborsTimeSeriesClassifier()
        )

        self.method_name: str = "KNeighborsTimeSeriesClassifier"

        self.parameter_space: Dict[str, List[Any]] = {
            "distance": [
                "dtw",
                "euclidean",
            ],  # , 'cityblock' 'ctw', 'sqeuclidean','sax' 'softdtw'
            "n_neighbors": [2, 3, 5],  # [log_med_long]
            "n_jobs": [ml_grid_object.global_params.knn_n_jobs],
        }

        # nb consider probability scoring on binary class eval: CalibratedClassifierCV
